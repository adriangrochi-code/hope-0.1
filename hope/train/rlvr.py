"""Refuerzo con recompensas verificables (RLVR, estilo GRPO; Shao et al. 2024, DeepSeek-R1 2025).

Objetivo del usuario: **más capacidad de razonamiento con la mínima cantidad de parámetros**,
para que cuando la red crezca ese «saber pensar» se amplifique.

Por qué esto y no solo más datos: el preentrenamiento enseña a imitar texto; el refuerzo
premia **llegar bien** a la respuesta. En modelos chicos es lo que más rindió en
razonamiento (DeepSeek-R1-Distill-1.5B, DeepScaleR-1.5B superan a modelos muchas veces
más grandes en matemática). Hope tiene lo que hace falta: problemas con respuesta exacta
(``hope/data/generators.py``), herramientas que verifican y el crítico CRE/HCRI.

Un paso:

1. Para cada problema, Hope genera un **grupo** de ``group`` soluciones con temperatura.
2. Cada una recibe una recompensa: correcta y con razonamiento que pasa la crítica = 1;
   correcta con pasos rotos («acierto por casualidad») = 0,2; incorrecta = 0.
3. **Ventaja relativa al grupo**: (r − media) / desvío. No hace falta un modelo crítico
   aparte (eso es GRPO), lo que ahorra memoria en la A770.
4. Se sube la probabilidad de las soluciones mejores que el promedio y se baja la de las
   peores, con una penalización KL que no deja alejarse demasiado del modelo de referencia.

Los grupos donde todas las soluciones tienen la misma recompensa no enseñan nada y se saltean.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from hope.constraints.critic import critique


@dataclass
class RLConfig:
    group: int = 8
    temperature: float = 0.8
    max_new: int = 200
    lr: float = 1e-5
    kl_weight: float = 0.02
    clip: float = 1.0
    stop: bytes = b"\x00"


@dataclass
class RLReport:
    steps: int = 0
    mean_reward: list[float] = field(default_factory=list)
    skipped_groups: int = 0


def completion_logprob(model, prompt: bytes, completion: bytes) -> tuple[torch.Tensor, int]:
    """Suma de log-probabilidades de ``completion`` dado ``prompt`` (con gradiente)."""
    seq = torch.tensor([list(prompt + completion)], device=model.head.weight.device)
    out = model(seq[:, :-1])
    logp = F.log_softmax(out["logits"].float(), dim=-1)
    tgt = seq[:, 1:]
    lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[:, len(prompt) - 1:]
    return lp.sum(), lp.numel()


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


class GRPO:
    def __init__(self, model, cfg: RLConfig = RLConfig(), *, reference=None, optimizer=None):
        self.model, self.cfg = model, cfg
        self.ref = reference if reference is not None else copy.deepcopy(model).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.opt = optimizer or torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
        self.report = RLReport()

    def step(self, prompts: list[str], reward_fn: Callable[[int, str], float]) -> dict:
        """``prompts[i]`` → ``group`` muestras; ``reward_fn(i, texto)`` da la recompensa."""
        cfg, m = self.cfg, self.model
        terms, rewards_all = [], []
        for i, prompt in enumerate(prompts):
            pb = prompt.encode("utf-8")
            m.eval()
            with torch.no_grad():
                samples = [m.generate(pb, cfg.max_new, temperature=cfg.temperature, stop=cfg.stop)
                           for _ in range(cfg.group)]
            texts = [s.decode("utf-8", "replace") for s in samples]
            r = torch.tensor([float(reward_fn(i, t)) for t in texts])
            rewards_all += r.tolist()
            if float(r.std()) < 1e-6:
                self.report.skipped_groups += 1
                continue
            adv = (r - r.mean()) / (r.std() + 1e-6)
            for s, a in zip(samples, adv.tolist()):
                if not s:
                    continue
                terms.append((pb, s, a))
        stats = {"recompensa_media": sum(rewards_all) / max(len(rewards_all), 1), "terminos": len(terms)}
        self.report.mean_reward.append(stats["recompensa_media"])
        if not terms:
            return stats
        self.opt.zero_grad(set_to_none=True)
        total_kl = 0.0
        for pb, s, a in terms:
            lp, n = completion_logprob(m, pb, s)
            with torch.no_grad():
                ref_lp, _ = completion_logprob(self.ref, pb, s)
            # KL aproximado por byte (estimador k3: e^{ref−lp} − (ref−lp) − 1, siempre ≥ 0)
            diff = (ref_lp - lp) / n
            kl = torch.exp(diff) - diff - 1
            loss = (-a * lp / n + cfg.kl_weight * kl) / len(terms)
            loss.backward()
            total_kl += float(kl)
        torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.clip)
        self.opt.step()
        self.report.steps += 1
        stats["kl"] = total_kl / len(terms)
        return stats


def problems_from_generators(families: list[str], n: int, seed: int = 0, level: int = 1) -> list[dict]:
    """Problemas de práctica con respuesta exacta para el refuerzo (nunca los de evaluación)."""
    from hope.data.generators import generate
    out = []
    for fam in families:
        for ep in generate(fam, n, seed=seed, level=level):
            d = ep.data
            out.append({"tema": d.get("tema", fam), "problema": d["problema"], "respuesta": str(d["resultado"])})
    return out
