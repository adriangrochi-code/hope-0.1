"""Refuerzo con recompensas verificables (RLVR, estilo GRPO; Shao et al. 2024, DeepSeek-R1 2025).

Objetivo del usuario: **más capacidad de razonamiento con la mínima cantidad de parámetros**,
para que cuando la red crezca ese «saber pensar» se amplifique.

Por qué esto y no solo más datos: el preentrenamiento enseña a imitar texto; el refuerzo
premia **llegar bien** a la respuesta. Hope tiene lo que hace falta: problemas con respuesta
exacta (``hope/data/generators.py``), herramientas que verifican y el crítico CRE/HCRI. Va
**después** del SFT y solo sobre problemas al borde de lo que el modelo ya resuelve (informe
«Razonamiento por parámetro» §1.4; a 135M, un GRPO directo sobre el modelo base empeoró).

Un paso:

1. Para cada problema, Hope genera un **grupo** de ``group`` soluciones con temperatura, en
   **un solo lote** cuando hay generación en lote (``sampler``; ver ``default_sampler``).
2. Cada una recibe una recompensa: correcta y con razonamiento que pasa la crítica = 1;
   correcta con pasos rotos («acierto por casualidad») = 0,2; incorrecta = 0.
3. **Ventaja relativa al grupo sin dividir por el desvío** (Dr. GRPO; Liu et al. 2025): dividir
   por el desvío sesga hacia las preguntas muy fáciles o muy difíciles.
4. Pérdida = −ventaja · Σ log p(byte) / ``max_new`` (constante, no el largo de cada muestra:
   normalizar por el largo alarga las respuestas incorrectas) + ``kl_weight`` · KL por byte
   (estimador k3, también sumado y dividido por ``max_new``). ``kl_weight`` ~1·10⁻³ (CMU).
5. **Grupos sin varianza** (todas correctas o todas incorrectas): no enseñan nada. Se
   descartan y, si hay de dónde, se **rellena** el lote con otros problemas (muestreo dinámico
   de DAPO; Yu et al. 2025). ``ProblemPool`` además retira los problemas que salen siempre
   bien y posterga los que salen siempre mal (descarte inteligente: quedarse en el borde).
6. **Currículo de largo**: ``max_new`` empieza en 512 bytes y sube a 1.024 y 2.048 cuando
   muchas respuestas se cortan por largo (o cada ``curriculum_every`` pasos), como DeepScaleR.

La recompensa con costo (``efficient_reward``, ``grpo_step_tools``) queda como **fase final
opcional y corta**, sobre problemas que el modelo ya resuelve (Fatemi et al. 2025).
"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch
import torch.nn.functional as F

from hope.constraints.critic import critique


@dataclass
class RLConfig:
    group: int = 8
    temperature: float = 0.8
    max_new: int = 512                        # largo inicial de las respuestas (bytes)
    curriculum: tuple[int, ...] = (1024, 2048)    # largos siguientes del currículo
    curriculum_truncation: float = 0.2        # fracción de respuestas cortadas que hace subir el largo
    curriculum_every: int = 0                 # además, subir cada N pasos (0 = solo por cortes)
    lr: float = 1e-5
    kl_weight: float = 1e-3
    clip: float = 1.0
    stop: bytes = b"\x00"
    # Normalizaciones. Los valores por defecto son los de Dr. GRPO; "largo" y "desvio" reproducen
    # el GRPO original (para comparar).
    loss_norm: str = "constante"              # "constante" (÷ max_new) | "largo" (÷ largo de cada muestra)
    adv_norm: str = "centrada"                # "centrada" (r − media) | "desvio" ((r − media) / desvío)
    zero_variance: str = "rellenar"           # "rellenar" | "descartar"
    max_refill: int = 4                       # intentos de relleno por grupo descartado


@dataclass
class RLReport:
    steps: int = 0
    mean_reward: list[float] = field(default_factory=list)
    skipped_groups: int = 0
    refilled_groups: int = 0
    max_new_history: list[int] = field(default_factory=list)


class Sampler(Protocol):
    def __call__(self, model, prompt: bytes, n: int, *, max_new: int, temperature: float,
                 stop: bytes) -> list[bytes]: ...


def serial_sampler(model, prompt: bytes, n: int, *, max_new: int, temperature: float, stop: bytes) -> list[bytes]:
    """Una muestra por vez (siempre disponible)."""
    return [model.generate(prompt, max_new, temperature=temperature, stop=stop) for _ in range(n)]


def default_sampler(model, prompt: bytes, n: int, *, max_new: int, temperature: float, stop: bytes) -> list[bytes]:
    """El grupo **en un solo lote** si existe ``hope.core.batched.generate_batch`` (rama de
    generación en lote): leer los pesos una vez para las ``n`` muestras cuesta casi lo mismo que
    una en la A770, limitada por el ancho de banda. Si no está, genera en serie."""
    try:
        from hope.core.batched import generate_batch
    except ImportError:
        return serial_sampler(model, prompt, n, max_new=max_new, temperature=temperature, stop=stop)
    return generate_batch(model, [prompt] * n, max_new, temperature=temperature, stop=stop)


def token_logprobs(model, prompt: bytes, completions: list[bytes]) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probabilidad de cada byte de cada respuesta, en **una pasada** para todo el grupo.

    Devuelve ``(logp [G, L], máscara [G, L])``. Se rellena a la derecha: el modelo es causal
    (también el parcheo), así que el relleno no cambia la probabilidad de los bytes reales."""
    dev = model.head.weight.device
    lens = [len(c) for c in completions]
    width = len(prompt) + max(lens)
    seq = torch.zeros(len(completions), width, dtype=torch.long, device=dev)
    mask = torch.zeros(len(completions), width - 1, device=dev)
    for i, c in enumerate(completions):
        row = list(prompt + c)
        seq[i, :len(row)] = torch.tensor(row, device=dev)
        mask[i, max(len(prompt) - 1, 0):len(row) - 1] = 1
    logp = F.log_softmax(model(seq[:, :-1])["logits"].float(), dim=-1)
    lp = logp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
    start = max(len(prompt) - 1, 0)          # con prompt vacío, el primer byte no tiene predicción
    return lp[:, start:], mask[:, start:]


def completion_logprob(model, prompt: bytes, completion: bytes) -> tuple[torch.Tensor, int]:
    """Suma de log-probabilidades de ``completion`` dado ``prompt`` (con gradiente)."""
    lp, m = token_logprobs(model, prompt, [completion])
    return (lp * m).sum(), int(m.sum())


def verified_reward(problem: dict, text: str, extract: Callable[[str], str | None],
                    match: Callable[[str | None, str], bool]) -> float:
    ans = extract(text)
    if not match(ans, problem["respuesta"]):
        return 0.0
    verdict = critique(problem["problema"], text, ans).verdict
    return 1.0 if verdict in ("ok", "incierto") else 0.2


def default_extract(text: str) -> str | None:
    m = re.search(r"Resultado:\s*(.+)", text)
    return m.group(1).strip().splitlines()[0] if m else None


def group_advantage(r: torch.Tensor, adv_norm: str = "centrada") -> torch.Tensor:
    adv = r - r.mean()
    if adv_norm == "desvio":
        adv = adv / (r.std() + 1e-6)
    elif adv_norm != "centrada":
        raise ValueError(f"adv_norm desconocido: {adv_norm}")
    return adv


def policy_loss(lp: torch.Tensor, ref_lp: torch.Tensor, mask: torch.Tensor, adv: torch.Tensor,
                cfg: RLConfig, max_new: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pérdida del grupo [G] y KL por byte (métrica). ``lp``, ``ref_lp``, ``mask``: [G, L]."""
    n = mask.sum(-1).clamp_min(1)
    denom = torch.full_like(n, float(max_new)) if cfg.loss_norm == "constante" else n
    if cfg.loss_norm not in ("constante", "largo"):
        raise ValueError(f"loss_norm desconocido: {cfg.loss_norm}")
    diff = (ref_lp - lp) * mask
    kl_tok = (torch.exp(diff) - diff - 1) * mask           # k3 por byte, siempre ≥ 0
    loss = (-adv * (lp * mask).sum(-1) + cfg.kl_weight * kl_tok.sum(-1)) / denom
    return loss, (kl_tok.sum(-1) / n).detach()


class ProblemPool:
    """Problemas de práctica con su tasa de acierto, para quedarse en el borde de la competencia.

    * ``draw()`` da el próximo problema activo como ``(prompt, recompensa(texto))``.
    * ``record(clave, recompensas)`` actualiza su tasa. Si el grupo sale todo bien, el problema
      se **retira** por fácil; si sale todo mal, se **posterga** ``cooldown`` sorteos (quizás
      más adelante el modelo llegue). Así el relleno no gasta generación en lo que no enseña.
    * ``edge(lo, hi)`` lista los problemas con tasa entre ``lo`` y ``hi`` (franja borde)."""

    def __init__(self, problems: list[dict], prompt_fn: Callable[[dict], str],
                 reward_fn: Callable[[dict, str], float], *, seed: int = 0, cooldown: int = 50):
        self.problems, self.prompt_fn, self.reward_fn = problems, prompt_fn, reward_fn
        self.rng = random.Random(seed)
        self.cooldown = cooldown
        self.stats = [{"vistos": 0, "aciertos": 0.0, "estado": "activo", "vuelve": 0} for _ in problems]
        self.draws = 0

    def _active(self) -> list[int]:
        return [i for i, s in enumerate(self.stats)
                if s["estado"] == "activo" or (s["estado"] == "postergado" and s["vuelve"] <= self.draws)]

    def draw(self):
        act = self._active()
        if not act:
            return None
        self.draws += 1
        i = self.rng.choice(act)
        p = self.problems[i]
        return i, self.prompt_fn(p), (lambda text, p=p: self.reward_fn(p, text))

    def record(self, key: int, rewards: list[float]) -> None:
        s = self.stats[key]
        s["vistos"] += len(rewards)
        s["aciertos"] += sum(r >= 1.0 for r in rewards)
        if min(rewards) >= 1.0:
            s["estado"] = "retirado"
        elif max(rewards) <= 0.0:
            s["estado"], s["vuelve"] = "postergado", self.draws + self.cooldown
        else:
            s["estado"] = "activo"

    def pass_rate(self, key: int) -> float | None:
        s = self.stats[key]
        return s["aciertos"] / s["vistos"] if s["vistos"] else None

    def edge(self, lo: float = 0.1, hi: float = 0.8) -> list[int]:
        return [i for i in range(len(self.problems))
                if (r := self.pass_rate(i)) is not None and lo <= r <= hi]

    def summary(self) -> dict:
        from collections import Counter
        return dict(Counter(s["estado"] for s in self.stats))


class GRPO:
    def __init__(self, model, cfg: RLConfig = RLConfig(), *, reference=None, optimizer=None,
                 sampler: Sampler | None = None):
        self.model, self.cfg = model, cfg
        self.ref = reference if reference is not None else copy.deepcopy(model).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.opt = optimizer or torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
        self.sampler = sampler or default_sampler
        self.report = RLReport()
        self.max_new = cfg.max_new
        self._stage = 0

    # --- currículo de largo ------------------------------------------------------------
    def _advance_curriculum(self, truncated: float) -> None:
        cfg = self.cfg
        if self._stage >= len(cfg.curriculum):
            return
        by_steps = cfg.curriculum_every and self.report.steps and self.report.steps % cfg.curriculum_every == 0
        if truncated >= cfg.curriculum_truncation or by_steps:
            self.max_new = max(self.max_new, cfg.curriculum[self._stage])
            self._stage += 1

    def _sample_group(self, pb: bytes) -> list[bytes]:
        m = self.model
        was = m.training
        m.eval()
        with torch.no_grad():
            out = self.sampler(m, pb, self.cfg.group, max_new=self.max_new,
                               temperature=self.cfg.temperature, stop=self.cfg.stop)
        m.train(was)
        return out

    def step(self, prompts: list[str], reward_fn: Callable[[int, str], float], *,
             refill: Callable[[], tuple | None] | None = None,
             on_group: Callable[[object, list[float]], None] | None = None) -> dict:
        """``prompts[i]`` → ``group`` muestras; ``reward_fn(i, texto)`` da la recompensa.

        ``refill()``: otra fuente de problemas para reemplazar los grupos sin varianza; devuelve
        ``(clave, prompt, recompensa(texto))`` o ``None``. ``on_group(clave, recompensas)`` se
        llama con cada grupo evaluado (p. ej. ``ProblemPool.record``)."""
        cfg = self.cfg
        groups, rewards_all, truncated, total = [], [], 0, 0
        pending = [(i, p, (lambda t, i=i: reward_fn(i, t))) for i, p in enumerate(prompts)]
        refills_left = cfg.max_refill * len(prompts) if (refill and cfg.zero_variance == "rellenar") else 0
        while pending:
            key, prompt, rew = pending.pop(0)
            pb = prompt.encode("utf-8")
            samples = self._sample_group(pb)
            texts = [s.decode("utf-8", "replace") for s in samples]
            r = torch.tensor([float(rew(t)) for t in texts])
            rewards_all += r.tolist()
            total += len(samples)
            truncated += sum(len(s) >= self.max_new for s in samples)
            if on_group is not None:
                on_group(key, r.tolist())
            if float(r.max() - r.min()) < 1e-6:
                self.report.skipped_groups += 1
                if refills_left > 0:
                    nxt = refill()
                    if nxt is not None:
                        refills_left -= 1
                        self.report.refilled_groups += 1
                        pending.append(nxt)
                continue
            keep = [j for j, s in enumerate(samples) if s]
            if keep:
                groups.append((pb, [samples[j] for j in keep], group_advantage(r, cfg.adv_norm)[keep]))
        trunc = truncated / max(total, 1)
        stats = {"recompensa_media": sum(rewards_all) / max(len(rewards_all), 1),
                 "grupos": len(groups), "terminos": sum(len(g[1]) for g in groups),
                 "cortadas": trunc, "max_new": self.max_new}
        self.report.mean_reward.append(stats["recompensa_media"])
        self.report.max_new_history.append(self.max_new)
        if groups:
            stats["kl"] = self._update(groups)
            self.report.steps += 1
        self._advance_curriculum(trunc)
        return stats

    def step_pool(self, pool: ProblemPool, n: int) -> dict:
        """Un paso con ``n`` problemas del pool; los grupos sin varianza se reemplazan del mismo pool."""
        drawn = [d for d in (pool.draw() for _ in range(n)) if d is not None]
        if not drawn:
            return {"recompensa_media": 0.0, "grupos": 0, "terminos": 0, "pool": pool.summary()}
        keys = [k for k, _, _ in drawn]
        rewards = [r for _, _, r in drawn]

        def refill():                     # claves de relleno: ("pool", índice), para no confundirlas
            d = pool.draw()
            return None if d is None else (("pool", d[0]), d[1], d[2])

        def record(k, r):
            pool.record(k[1] if isinstance(k, tuple) else keys[k], r)

        stats = self.step([p for _, p, _ in drawn], lambda i, t: rewards[i](t), refill=refill, on_group=record)
        stats["pool"] = pool.summary()
        return stats

    def _update(self, groups) -> float:
        m, cfg = self.model, self.cfg
        # En modo evaluación: la política que se optimiza es la que generó las muestras (sin
        # profundidad al azar del núcleo, que solo existe en modo entrenamiento).
        was = m.training
        m.eval()
        self.opt.zero_grad(set_to_none=True)
        n_terms = sum(len(s) for _, s, _ in groups)
        kls = []
        for pb, samples, adv in groups:
            lp, mask = token_logprobs(m, pb, samples)
            with torch.no_grad():
                ref_lp, _ = token_logprobs(self.ref, pb, samples)
            loss, kl = policy_loss(lp, ref_lp, mask, adv.to(lp.device), cfg, self.max_new)
            (loss.sum() / n_terms).backward()
            kls += kl.tolist()
        torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.clip)
        self.opt.step()
        m.train(was)
        return sum(kls) / max(len(kls), 1)


def problems_from_generators(families: list[str], n: int, seed: int = 0, level: int = 1) -> list[dict]:
    """Problemas de práctica con respuesta exacta para el refuerzo (nunca los de evaluación)."""
    from hope.data.generators import generate
    out = []
    for fam in families:
        for ep in generate(fam, n, seed=seed, level=level):
            d = ep.data
            out.append({"tema": d.get("tema", fam), "problema": d["problema"], "respuesta": str(d["resultado"])})
    return out


# ---------------------------------------------------------------------------
# Rapidez además de calidad (pedido del usuario): «si la calculadora es más rápida y exacta
# que predecir 6·7 byte a byte, que aprenda a usarla».
# ---------------------------------------------------------------------------
@dataclass
class Rollout:
    prompt: bytes
    segments: list[tuple[bytes, bool]]         # (bytes, generado_por_hope); False = resultado de herramienta
    seconds: float
    tool_seconds: float
    tool_calls: int

    @property
    def text(self) -> str:
        return b"".join(s for s, _ in self.segments).decode("utf-8", "replace")

    @property
    def model_bytes(self) -> int:
        return sum(len(s) for s, gen in self.segments if gen)


def rollout_with_tools(model, prompt: str, runner, *, temperature: float = 0.8, max_new: int = 200,
                       max_calls: int = 4) -> Rollout:
    """Como ``generate_with_tools`` pero registra qué bytes generó Hope y cuánto tardó cada parte."""
    import time

    from hope.runtime.toolcalls import CLOSE
    t0 = time.perf_counter()
    pb = prompt.encode("utf-8")
    text, segs, tool_s, budget = pb, [], 0.0, max_new
    for _ in range(max_calls + 1):
        out = model.generate(text, max_new=budget, stop_after_line=b"Resultado:", temperature=temperature,
                             stop_seq=CLOSE.encode())
        segs.append((out, True))
        text += out
        budget -= len(out)
        if not out.decode("utf-8", "replace").endswith(CLOSE) or budget <= 0 or len(runner.calls) >= max_calls:
            break
        t1 = time.perf_counter()
        res = runner.run_text(b"".join(s for s, _ in segs).decode("utf-8", "replace")).encode("utf-8")
        tool_s += time.perf_counter() - t1
        segs.append((res, False))
        text += res
    return Rollout(pb, segs, time.perf_counter() - t0, tool_s, len(runner.calls))


def masked_logprob(model, r: Rollout, per_byte: bool = False):
    """Log-probabilidad solo de los bytes que generó Hope (no de los resultados de herramientas).
    Con ``per_byte`` devuelve ``(logp [L], máscara [L])`` en vez de ``(suma, cantidad)``."""
    full = r.prompt + b"".join(s for s, _ in r.segments)
    mask = [0] * len(r.prompt)
    for s, gen in r.segments:
        mask += [1 if gen else 0] * len(s)
    seq = torch.tensor([list(full)], device=model.head.weight.device)
    logp = F.log_softmax(model(seq[:, :-1])["logits"].float(), dim=-1)
    lp = logp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
    m = torch.tensor(mask[1:], device=lp.device, dtype=lp.dtype)
    if per_byte:
        return lp, m
    return (lp * m).sum(), int(m.sum())


@dataclass
class CostWeights:
    """La calidad manda: el castigo por costo nunca supera ``max_penalty`` (< 1), así que una
    respuesta correcta y lenta siempre gana a una rápida e incorrecta. Entre dos correctas, gana
    la más barata: menos bytes generados por Hope y menos tiempo total."""
    per_model_byte: float = 0.002            # 100 bytes generados = −0,2
    per_second: float = 0.1                  # relativo a ``target_seconds``
    target_seconds: float = 1.0
    max_penalty: float = 0.5


def efficient_reward(quality: float, r: Rollout, w: CostWeights = CostWeights()) -> float:
    penalty = w.per_model_byte * r.model_bytes + w.per_second * r.seconds / w.target_seconds
    return quality - min(w.max_penalty, penalty)


def grpo_step_tools(rl: "GRPO", prompts: list[str], quality_fn: Callable[[int, str], float], runner_factory,
                    w: CostWeights | None = CostWeights()) -> dict:
    """Paso GRPO con herramientas: recompensa = calidad verificada − costo (bytes propios + tiempo).

    Es la **fase final opcional**: corta y sobre problemas que el modelo ya resuelve (Fatemi et al.
    2025). Con ``w=None`` la recompensa es solo la calidad. Misma pérdida que ``GRPO.step``
    (Dr. GRPO: ventaja centrada, suma ÷ ``max_new``, KL k3 por byte)."""
    cfg, m = rl.cfg, rl.model
    was_training = m.training                     # el paso no cambia el modo del modelo
    groups, rewards, uses = [], [], 0
    for i, prompt in enumerate(prompts):
        m.eval()
        with torch.no_grad():
            rolls = [rollout_with_tools(m, prompt, runner_factory(), temperature=cfg.temperature,
                                        max_new=rl.max_new) for _ in range(cfg.group)]
        r = torch.tensor([quality_fn(i, x.text) if w is None else efficient_reward(quality_fn(i, x.text), x, w)
                          for x in rolls])
        rewards += r.tolist()
        uses += sum(x.tool_calls > 0 for x in rolls)
        if float(r.max() - r.min()) < 1e-6:
            rl.report.skipped_groups += 1
            continue
        adv = group_advantage(r, cfg.adv_norm)
        groups += [(x, a) for x, a in zip(rolls, adv.tolist()) if x.model_bytes]
    stats = {"recompensa_media": sum(rewards) / max(len(rewards), 1),
             "uso_herramientas": uses / max(len(rewards), 1), "terminos": len(groups)}
    if not groups:
        m.train(was_training)
        return stats
    rl.opt.zero_grad(set_to_none=True)
    kls = []
    for x, a in groups:
        lp, mask = masked_logprob(m, x, per_byte=True)
        with torch.no_grad():
            ref, _ = masked_logprob(rl.ref, x, per_byte=True)
        loss, kl = policy_loss(lp[None], ref[None], mask[None], torch.tensor([a], device=lp.device), cfg, rl.max_new)
        (loss.sum() / len(groups)).backward()
        kls += kl.tolist()
    torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.clip)
    rl.opt.step()
    rl.report.steps += 1
    stats["kl"] = sum(kls) / len(kls)
    m.train(was_training)
    return stats
