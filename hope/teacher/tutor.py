"""Tutoría on-policy: un Qwen grande como **tutor completo** de Hope, no solo destilador
(ARQUITECTURA_HN1 §10.1.5; pedido del usuario).

El ciclo maestro–alumno (``cycle.py``) y la cosecha (``harvest.py``) usan al maestro para
**escribir** explicaciones. El tutor, en cambio, trabaja sobre lo que **Hope** produce
(destilación on-policy; «Razonamiento por parámetro para Hope», §on-policy): revisa los
intentos de Hope, encuentra el error, da pistas, propone problemas a medida y califica el
razonamiento. Cada consulta al tutor rinde más porque ataca un error real de Hope y no un
tema genérico.

Una sesión de tutoría:

1. **Diagnóstico** (``diagnose``): el tutor prueba a Hope con problemas de cada tema
   (generadores verificables + problemas propios del tutor, verificados con herramientas;
   e imágenes procedurales si el modelo tiene visión) y arma un **perfil** de fortalezas y
   debilidades: exactitud, tipos de error, uso de herramientas y notas del tutor.
2. **Intento → crítica → pista → reintento** (``tutor_problem``): Hope resuelve; el tutor revisa
   **el intento de Hope** (no su propia solución), identifica el error conceptual o de
   procedimiento y da una **pista sin la respuesta** (si la pista filtra la respuesta, se
   descarta y se usa una genérica). Hope reintenta. Si vuelve a fallar, el tutor explica
   completo (episodio verificado) y Hope anota en su cuaderno (``Notebook.write``) la idea
   clave y el error a evitar; la lección queda también en la base (``learn_lesson``).
3. **Variaciones dirigidas** (``targeted_variations``): el tutor genera problemas nuevos que
   atacan exactamente el error encontrado; los verificados entran al **repaso espaciado**
   (intervalos 1, 2, 4… días; fallar vuelve a 1).
4. **Evaluación del razonamiento y de las herramientas**: el tutor puntúa pasos, elección de
   herramienta y eficiencia (0–1). Esas notas modulan la recompensa del refuerzo
   (``combined_reward``), pero **la verificación automática manda**: una respuesta incorrecta
   vale 0 diga lo que diga el tutor, y el tutor solo puede mover una respuesta correcta
   dentro de [1 − w, 1] (el tutor puede equivocarse).
5. **Preguntas de Hope** (``maybe_ask``): si la cabeza de incertidumbre de Hope marca duda
   alta, Hope le pregunta al tutor por la línea más dudosa; queda registro de preguntas y
   respuestas (``preguntas.jsonl``).
6. **Currículo adaptativo** (``next_topics``): el próximo tema se elige por el perfil
   (debilidad), la tasa de mejora (estancado = más atención) y el repaso pendiente; un tema
   está **dominado** con ≥ ``mastery_accuracy`` en las últimas ``mastery_window`` respuestas
   (y al menos ``mastery_min_items``) y sin repasos vencidos.
7. **Datos para entrenar** (``TutorData``): intentos corregidos con máscara de pérdida (los
   pasos equivocados de Hope quedan como contexto con peso 0; la revisión y el reintento
   correcto, con peso 1), episodios verificados del tutor y lecciones del cuaderno. Se
   escriben como ``train.bin``/``train.mask`` (fuente de ``hope.train.run``) y se consolidan
   en la sesión. **SFT antes que RL** (MobileLLM-R1): el refuerzo con la recompensa
   combinada (``reinforce``) solo corre sobre temas que ya superan ``rl_min_accuracy``.
8. **Guardián y verificación**: todo el tráfico pasa por ``Teacher.ask`` (tutor local = nivel 1;
   remoto = aprobación humana). Las respuestas del tutor se verifican con herramientas cuando
   se puede (problemas, variaciones, episodios, respuestas a preguntas); lo que falla la
   verificación se registra como error del tutor y **no entra al entrenamiento**.

Uso::

    python -m hope.teacher.tutor --resume runs/hope_a770/4_vision_procedural \\
        --teacher configs/tutor_a770.yaml --out runs/tutoria --rounds 20

o como etapa ``tutoria`` de ``hope.train.run`` (``configs/plan_a770_continuacion.yaml``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


from guardian.core import Action, Guardian
from hope.constraints.critic import STEP_RE, critique
from hope.epistemic import EpistemicStatus
from hope.memory.knowledge import DAY, KnowledgeBase, Notebook
from hope.memory.pesam import ExperienceLog
from hope.memory.store import MemoryStore
from hope.teacher.client import MockTeacher, Teacher, TeacherConfig, TeacherDenied
from hope.teacher.cycle import Topic, answers_match, extract_answer, load_curriculum
from hope.teacher.episode import (FORMAT_HINT, RULES_HINT, Episode, EpisodeParseError, parse_episode)
from hope.tools.builtin import arith
from hope.tools.catalog import Catalog
from hope.tools.verify import check_step, verify_episode
from hope.train.continual import Mixture, Source, Trainer

TUTOR_SYSTEM = (
    "Sos el tutor de Hope, una red neuronal pequeña que está aprendiendo a razonar. No resolvés por "
    "ella: revisás SUS intentos, encontrás el error concreto (conceptual, de procedimiento, de cálculo, "
    "de herramienta o de lectura del enunciado), das pistas que NO contienen la respuesta, proponés "
    "problemas que atacan ese error y calificás el razonamiento. Cuando des un valor numérico, incluí "
    "'verificacion' para comprobarlo con herramientas. Respondé solo con un objeto JSON."
)
VERIF_HINT = ('"verificacion": {"tipo": "aritmetica", "expr": "...", "esperado": "..."} '
              '(o {"tipo": "unidades", "expr": "2 A * 5 ohm", "esperado": "10 V"} o '
              '{"tipo": "codigo", "codigo": "print(...)", "salida_esperada": "..."})')
ERROR_KINDS = ("conceptual", "procedimiento", "calculo", "herramienta", "enunciado", "formato", "ninguno")


def _msgs(task: str, body: str) -> list[dict]:
    return [{"role": "system", "content": TUTOR_SYSTEM},
            {"role": "user", "content": f"TUTOR_TAREA: {task}\n{body}"}]


def problems_request(topic: Topic, n: int, *, focus: str = "", avoid: list[str] | None = None) -> list[dict]:
    extra = f"Atacá exactamente este error de Hope: {focus}\n" if focus else ""
    if avoid:
        extra += "No repitas:\n" + "\n".join(f"- {a[:160]}" for a in avoid[-6:]) + "\n"
    return _msgs("problemas", (
        f"TEMA: {topic.name}\nNIVEL: {topic.level}\nIDIOMA: {topic.language}\nCANTIDAD: {n}\n{extra}"
        f"Proponé {n} problemas nuevos con su respuesta breve (valor con unidad o frase corta). Formato: "
        f'{{"problemas": [{{"problema": "...", "respuesta": "...", {VERIF_HINT}}}]}}'))


def review_request(topic: str, problem: str, attempt: str, answer: str | None, expected: str | None,
                   critic_notes: str = "", tool_calls: list[str] | None = None) -> list[dict]:
    tools = "\n".join(f"- {c}" for c in (tool_calls or [])) or "- (ninguna)"
    return _msgs("revision", (
        f"TEMA: {topic}\nPROBLEMA: {problem}\nINTENTO DE HOPE:\n{attempt.strip()}\n"
        f"RESPUESTA DE HOPE: {answer!r}\n"
        + (f"RESPUESTA VERIFICADA: {expected}\n" if expected is not None else "")
        + (f"REVISIÓN AUTOMÁTICA (CRE):\n{critic_notes}\n" if critic_notes else "")
        + f"HERRAMIENTAS QUE USÓ HOPE:\n{tools}\n"
        "Revisá el intento de Hope (no tu propia solución). Si hay error, decí cuál es, de qué tipo "
        f"({', '.join(ERROR_KINDS)}), en qué paso, y una pista que NO contenga la respuesta final. "
        "Puntuá de 0 a 1 los pasos, la elección de herramienta y la eficiencia. Formato: "
        '{"correcto": true|false, "error": "...", "tipo": "...", "paso": 1, "pista": "...", '
        '"idea_clave": "...", "error_a_evitar": "...", '
        '"puntaje": {"pasos": 0.0, "herramienta": 0.0, "eficiencia": 0.0}}'))


def explanation_request(topic: Topic, problem: str, attempts: list[str], error: str) -> list[dict]:
    tries = "\n---\n".join(a.strip()[-400:] for a in attempts)
    return _msgs("explicacion", (
        f"TEMA: {topic.name}\nNIVEL: {topic.level}\nIDIOMA: {topic.language}\n"
        f"Hope falló dos veces este problema: {problem}\nSus intentos:\n{tries}\nError detectado: {error}\n"
        "Explicá completo el procedimiento correcto de ESTE problema, paso a paso, señalando dónde se "
        f"equivocó Hope, con variaciones con respuesta. {RULES_HINT}Formato:\n{FORMAT_HINT}"))


def question_to_tutor(topic: str, problem: str, partial: str, doubt: str) -> list[dict]:
    return _msgs("pregunta", (
        f"TEMA: {topic}\nHope está resolviendo: {problem}\nLo que lleva:\n{partial.strip()[-400:]}\n"
        f"Hope no está segura de esta parte: {doubt.strip()[:200]!r}\n"
        "Respondé la duda concreta sin dar la respuesta final del problema. Si das un valor, incluí "
        f'verificacion. Formato: {{"respuesta": "...", {VERIF_HINT}}}'))


def _parse_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    cand = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    data, _ = json.JSONDecoder().raw_decode(cand)
    if not isinstance(data, dict):
        raise ValueError("se esperaba un objeto JSON")
    return data


# ---------------------------------------------------------------------------
# Pistas sin la respuesta
# ---------------------------------------------------------------------------
def _num_variants(answer: str) -> list[str]:
    first = str(answer).strip().split()[0] if str(answer).strip() else ""
    try:
        v = arith(first.replace(",", "."))
    except Exception:
        return []
    s = format(v.normalize(), "f") if hasattr(v, "normalize") else str(v)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return list({s, s.replace(".", ",")})


def leaks_answer(hint: str, answer: str | None) -> bool:
    """¿La pista contiene la respuesta final? (número exacto o, si no es numérica, la frase)."""
    if not hint or answer is None:
        return False
    nums = _num_variants(answer)
    if nums:
        return any(re.search(rf"(?<![\d.,]){re.escape(n)}(?![\d]|[.,]\d)", hint) for n in nums)
    a = re.sub(r"\W+", " ", str(answer).lower()).strip()
    return len(a) >= 3 and a in re.sub(r"\W+", " ", hint.lower())


def safe_hint(hint: str | None, answer: str | None, *, step: int | None = None, kind: str = "") -> str:
    """La pista del tutor si no filtra la respuesta; si la filtra, una genérica a partir del error."""
    if hint and not leaks_answer(hint, answer):
        return hint.strip()
    generic = {"calculo": "revisá la cuenta", "procedimiento": "revisá el orden de los pasos",
               "conceptual": "revisá qué regla o fórmula corresponde", "herramienta":
               "pensá qué herramienta conviene usar y con qué datos", "enunciado": "releé el enunciado"}
    where = f"en el paso {step}" if step else "paso por paso"
    fb = f"Revisá {where}: {generic.get(kind, 'comprobá cada resultado intermedio')}."
    return fb if not leaks_answer(fb, answer) else "Revisá tu procedimiento paso a paso."


# ---------------------------------------------------------------------------
# Recompensa: la verificación automática manda
# ---------------------------------------------------------------------------
def combined_reward(verified: float, scores: dict | None, tutor_weight: float = 0.3) -> float:
    """``verified`` ∈ {0, 0,2, 1} (``hope.train.rlvr.verified_reward``). Si es 0, el tutor no puede
    rescatarla. Si no, el promedio de las notas del tutor (0–1) la mueve dentro de
    [verified·(1 − w), verified]: el tutor afina entre respuestas correctas, nunca las invierte."""
    if verified <= 0:
        return 0.0
    vals = [min(max(float(v), 0.0), 1.0) for v in (scores or {}).values() if isinstance(v, (int, float))]
    if not vals:
        return verified
    return verified * (1 - tutor_weight + tutor_weight * sum(vals) / len(vals))


# ---------------------------------------------------------------------------
# Estructuras
# ---------------------------------------------------------------------------
@dataclass
class Problem:
    tema: str
    problema: str
    respuesta: str
    origen: str = "generador"          # generador | tutor | variacion | imagen
    verificado: bool = True
    foco: str = ""                     # error que ataca (variaciones dirigidas)
    image: object | None = None        # tensor uint8 [3, S, S] para problemas con imagen

    def key(self) -> str:
        return " ".join(self.problema.lower().split())


@dataclass
class Attempt:
    prompt: str
    text: str
    answer: str | None
    correct: bool
    verdict: str
    issues: list[dict]
    tool_calls: list[dict]
    uncertainty: float | None = None


@dataclass
class TopicProfile:
    attempts: int = 0
    correct: int = 0
    recent: list[int] = field(default_factory=list)            # 1/0 de las últimas respuestas (primer intento)
    sessions: list[float] = field(default_factory=list)        # exactitud por ronda
    errors: dict = field(default_factory=dict)                 # tipo de error → cantidad
    tool_calls: int = 0
    tool_ok: int = 0
    scores: dict = field(default_factory=dict)                 # criterio → [suma, n]
    hints_helped: int = 0
    explanations: int = 0
    mastered: bool = False

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.attempts if self.attempts else None

    def recent_accuracy(self, window: int) -> float | None:
        r = self.recent[-window:]
        return sum(r) / len(r) if r else None

    @property
    def improvement(self) -> float:
        """Tasa de mejora: exactitud de la última ronda menos la del promedio de las anteriores."""
        if len(self.sessions) < 2:
            return 0.0
        prev = self.sessions[:-1][-3:]
        return self.sessions[-1] - sum(prev) / len(prev)

    def mean_score(self, key: str) -> float | None:
        s = self.scores.get(key)
        return s[0] / s[1] if s and s[1] else None


class Profile:
    """Fortalezas y debilidades de Hope por tema (incluye herramientas e imágenes)."""

    def __init__(self):
        self.topics: dict[str, TopicProfile] = {}

    def get(self, topic: str) -> TopicProfile:
        return self.topics.setdefault(topic, TopicProfile())

    def strengths(self, min_items: int = 3, thr: float = 0.7) -> list[str]:
        return sorted(t for t, p in self.topics.items()
                      if p.attempts >= min_items and (p.recent_accuracy(10) or 0) >= thr)

    def weaknesses(self, min_items: int = 1, thr: float = 0.5) -> list[str]:
        return sorted((t for t, p in self.topics.items()
                       if p.attempts >= min_items and (p.recent_accuracy(10) or 0) < thr),
                      key=lambda t: self.topics[t].recent_accuracy(10) or 0)

    def to_dict(self) -> dict:
        return {t: {**asdict(p), "accuracy": p.accuracy, "improvement": p.improvement}
                for t, p in self.topics.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        pr = cls()
        names = TopicProfile.__dataclass_fields__
        for t, v in d.items():
            pr.topics[t] = TopicProfile(**{k: v[k] for k in names if k in v})
        return pr


@dataclass
class ReviewItem:
    problem: dict
    due: float
    interval_days: float = 1.0
    streak: int = 0


class ReviewQueue:
    """Repaso espaciado de los problemas que atacan errores de Hope (Leitner: 1, 2, 4… días)."""

    def __init__(self, clock=time.time):
        self.items: list[ReviewItem] = []
        self.clock = clock

    def add(self, p: Problem, *, delay_days: float = 0.0) -> bool:
        if any(i.problem["problema"] == p.problema for i in self.items):
            return False
        d = {k: getattr(p, k) for k in Problem.__dataclass_fields__ if k != "image"}
        self.items.append(ReviewItem(d, self.clock() + delay_days * DAY))
        return True

    def due(self, topic: str | None = None, limit: int = 100) -> list[ReviewItem]:
        now = self.clock()
        out = [i for i in self.items if i.due <= now and (topic is None or i.problem["tema"] == topic)]
        return sorted(out, key=lambda i: i.due)[:limit]

    def pending(self, topic: str) -> int:
        return len(self.due(topic))

    def record(self, item: ReviewItem, ok: bool) -> None:
        item.streak, item.interval_days = (item.streak + 1, item.interval_days * 2) if ok else (0, 1.0)
        item.due = self.clock() + item.interval_days * DAY

    def to_list(self) -> list[dict]:
        return [asdict(i) for i in self.items]

    def load(self, rows: list[dict]) -> None:
        self.items = [ReviewItem(**r) for r in rows]


class TutorData:
    """Lo que la tutoría deja para entrenar: cada registro es una lista de (texto, se_entrena)."""

    def __init__(self):
        self.records: list[dict] = []

    def add(self, kind: str, segments: list[tuple[str, bool]], *, weight: float = 1.0, **meta) -> None:
        segs = [(t, bool(m)) for t, m in segments if t]
        if not any(m for _, m in segs):
            return
        self.records.append({"tipo": kind, "segmentos": segs, "peso": weight, **meta})

    def kinds(self) -> Counter:
        return Counter(r["tipo"] for r in self.records)

    def items(self) -> list[list[tuple[str, bool]]]:
        """Segmentos repetidos según el peso (1,0 → 3 copias; 0,3 → 1), como en el ciclo."""
        return [r["segmentos"] for r in self.records for _ in range(max(1, round(r["peso"] * 3)))]

    def source(self, name: str = "tutor", weight: float = 1.0) -> Source:
        return Source.from_segments(name, self.items(), weight)

    def write(self, out: str | Path, val_percent: float = 2.0) -> dict:
        """``train.bin``/``train.mask`` y ``val.bin``/``val.mask`` (fuente de ``hope.train.run``)
        + ``registros.jsonl``. Se reescribe entero en cada llamada (es reanudable)."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        files = {s: (open(out / f"{s}.bin", "wb"), open(out / f"{s}.mask", "wb")) for s in ("train", "val")}
        stats = Counter()
        with open(out / "registros.jsonl", "w", encoding="utf-8") as js:
            for r in self.records:
                text = "".join(t for t, _ in r["segmentos"])
                h = hashlib.sha256(text.encode()).hexdigest()
                split = "val" if int(h[:8], 16) % 10_000 < val_percent * 100 else "train"
                fb, fm = files[split]
                for _ in range(max(1, round(r["peso"] * 3))):
                    for t, m in r["segmentos"]:
                        b = t.encode("utf-8")
                        fb.write(b)
                        fm.write(bytes([1 if m else 0]) * len(b))
                    fb.write(b"\x00")
                    fm.write(b"\x01")
                stats[split] += 1
                stats[r["tipo"]] += 1
                js.write(json.dumps({"split": split, **r}, ensure_ascii=False) + "\n")
        for fb, fm in files.values():
            fb.close()
            fm.close()
        return dict(stats)


def corrected_segments(prompt: str, first: str, wrong_step: int | None, revision: str,
                       fixed: str) -> list[tuple[str, bool]]:
    """Intento corregido con máscara: el enunciado y lo equivocado son contexto (0); los pasos
    correctos antes del error, la revisión y el reintento correcto se entrenan (1)."""
    segs: list[tuple[str, bool]] = [(prompt, False)]
    lines = first.splitlines(keepends=True)
    ok = wrong_step is not None
    n = 0
    for line in lines:
        if STEP_RE.search(line):
            n += 1
        if ok and n >= wrong_step or line.lstrip().startswith("Resultado:"):
            ok = False
        segs.append((line, ok))
    if segs and not segs[-1][0].endswith("\n"):
        segs.append(("\n", False))
    segs.append((revision.strip() + "\n", True))
    segs.append((fixed.strip() + "\n", True))
    return segs


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
@dataclass
class TutorConfig:
    diagnosis_items: int = 6            # problemas por tema en el diagnóstico
    tutor_problems: int = 2             # de esos, cuántos propone el tutor (el resto, generadores)
    problems_per_topic: int = 4         # problemas por tema y ronda en la tutoría
    topics_per_round: int = 3
    variations_per_error: int = 2
    max_variations_per_topic: int = 6
    max_tutor_calls: int = 1000
    answer_max_bytes: int = 400
    tool_calls: bool = True
    use_notebook_context: bool = True
    images: bool = True                 # problemas con imagen si el modelo tiene visión
    # Preguntas de Hope
    ask_threshold: float = 2.5          # incertidumbre media (nats/byte predichos) para preguntar
    max_questions_per_round: int = 6
    # Currículo adaptativo
    mastery_accuracy: float = 0.9
    mastery_window: int = 10
    mastery_min_items: int = 8
    review_weight: float = 0.5
    stagnation_weight: float = 0.3
    level_gate: float = 0.5             # penalización por cada nivel por encima del más bajo sin dominar
    # Datos y consolidación (SFT)
    consolidation_steps: int = 200
    replay_weight: float = 0.3
    train_unverified: bool = False      # problemas sin verificación: solo diagnóstico, no entrenamiento
    # Refuerzo con la recompensa combinada (después del SFT)
    rl: bool = False
    rl_min_accuracy: float = 0.3        # SFT antes que RL: solo temas que ya aciertan algo
    rl_steps: int = 1
    rl_problems: int = 4
    rl_group: int = 4
    rl_max_new: int = 200
    rl_tutor_scores: bool = True        # pedir notas al tutor para las muestras correctas
    tutor_weight: float = 0.3


# ---------------------------------------------------------------------------
class TutorSession:
    def __init__(self, trainer: Trainer, tutor: Teacher, memory: MemoryStore, catalog: Catalog,
                 topics: list[Topic], *, cfg: TutorConfig = TutorConfig(), guardian: Guardian | None = None,
                 pesam: ExperienceLog | None = None, out_dir: str | Path | None = None,
                 replay: list[Source] | None = None, clock=time.time, seed: int = 0,
                 uncertainty_fn: Callable[[str, str], tuple[float, str]] | None = None, log=print):
        self.trainer, self.tutor, self.memory, self.catalog = trainer, tutor, memory, catalog
        self.topics = {t.name: t for t in sorted(topics, key=lambda t: t.level)}
        self.cfg, self.guardian, self.pesam, self.log, self.clock = cfg, guardian, pesam, log, clock
        self.out_dir = Path(out_dir) if out_dir else None
        self.replay = replay or []
        self.rng = random.Random(seed)
        self.kb = KnowledgeBase(memory)
        self.notebook = Notebook(self.kb, clock=clock)
        self.profile = Profile()
        self.reviews = ReviewQueue(clock)
        self.data = TutorData()
        self.questions: list[dict] = []
        self.rewards: list[dict] = []
        self.tutor_errors: list[dict] = []
        self.seen: set[str] = set()
        self.rounds = 0
        self.variations_added: Counter = Counter()
        self.uncertainty_fn = uncertainty_fn or self._uncertainty
        self._questions_this_round = 0
        from hope.runtime.toolcalls import ToolRunner
        self.tool_runner = ToolRunner(catalog, memory)

    # --- tutor --------------------------------------------------------------------
    def _ask_json(self, messages: list[dict], purpose: str) -> dict | None:
        if self.tutor.calls >= self.cfg.max_tutor_calls:
            return None
        try:
            reply = self.tutor.ask(messages, purpose=f"tutor:{purpose}")
        except TeacherDenied as e:
            self._record("tutor_denegado", EpistemicStatus.OBSERVADO, purpose=purpose, error=str(e))
            self.log(f"  tutor denegado: {e}")
            return None
        try:
            return {"_texto": reply.text, "_modelo": reply.model, **_parse_json(reply.text)}
        except Exception as e:
            self._tutor_error("respuesta_invalida", purpose=purpose, error=str(e))
            return None

    def _tutor_error(self, kind: str, **data) -> None:
        self.tutor_errors.append({"tipo": kind, **data})
        self._record("error_del_tutor", EpistemicStatus.REFUTADO, tipo=kind, **data)

    def verify_problem(self, raw: dict, topic: Topic, origin: str, focus: str = "") -> Problem | None:
        """Problema del tutor → se verifica su respuesta. Falla → se descarta (error del tutor)."""
        if not raw.get("problema") or raw.get("respuesta") in (None, ""):
            return None
        ans = str(raw["respuesta"]).strip()
        chk = raw.get("verificacion") if isinstance(raw.get("verificacion"), dict) else None
        status = check_step(self.catalog, chk) if chk else "no_verificable"
        if status == "ok" and chk.get("esperado") is not None and chk.get("tipo") in ("aritmetica", "unidades"):
            if not answers_match(ans, str(chk["esperado"])):    # verificó otra cosa que la respuesta dada
                status = "falla"
        if status == "falla":
            self._tutor_error("problema_refutado", tema=topic.name, problema=raw["problema"], respuesta=ans,
                              verificacion=chk, origen=origin)
            return None
        return Problem(topic.name, str(raw["problema"]).strip(), ans, origin, status == "ok", focus)

    def tutor_problems(self, topic: Topic, n: int, *, focus: str = "", origin: str = "tutor") -> list[Problem]:
        if n <= 0:
            return []
        rep = self._ask_json(problems_request(topic, n, focus=focus, avoid=list(self.seen)[-6:]),
                             f"problemas:{topic.name}")
        out = []
        for raw in (rep or {}).get("problemas", [])[:n]:
            if isinstance(raw, dict):
                p = self.verify_problem(raw, topic, origin, focus)
                if p is not None and p.key() not in self.seen:
                    out.append(p)
        return out

    def generated_problems(self, topic: Topic, n: int) -> list[Problem]:
        from hope.data.generators import family_for, generate
        fam = family_for(topic.name)
        if fam is None or n <= 0:
            return []
        eps = generate(fam, n * 3, seed=1000 * self.rounds + self.rng.randrange(10_000), level=topic.level,
                       lang=topic.language)
        out = [Problem(topic.name, e.data["problema"], str(e.data["resultado"]), "generador")
               for e in eps if " ".join(e.data["problema"].lower().split()) not in self.seen]
        return out[:n]

    def image_problems(self, n: int) -> list[Problem]:
        """Preguntas sobre imágenes procedurales (respuesta exacta por construcción)."""
        import torch

        from hope.data.vision import IMG_TAG, procedural
        if n <= 0 or not getattr(self.trainer.model.cfg, "vision", False):
            return []
        out = []
        size = self.trainer.model.cfg.vis_image
        for arr, ann in procedural(n, seed=self.rng.randrange(1 << 30), size=size):
            q = (ann.get("preguntas") or [{}])[0]
            if not q.get("pregunta"):
                continue
            img = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()
            out.append(Problem("imagen", f"{IMG_TAG} {q['pregunta']}", str(q["respuesta"]), "imagen", True,
                               image=img))
        return out

    # --- Hope ---------------------------------------------------------------------
    def _context(self, topic: str, problem: str) -> str:
        if not self.cfg.use_notebook_context:
            return ""
        ctx = self.kb.context(topic, problem, k=2)
        return ctx.text + "\n\n" if ctx.text else ""

    def prompt_for(self, p: Problem, extra: str = "") -> str:
        base = f"Tema: {p.tema}\nProblema: {p.problema}\n"
        return self._context(p.tema, p.problema) + base + (extra.strip() + "\n" if extra else "")

    def _generate(self, prompt: str, temperature: float = 0.0, image=None) -> tuple[str, list[dict]]:
        model = self.trainer.model
        kw = {"image": image.unsqueeze(0).to(model.head.weight.device)} if image is not None else {}
        n0 = len(self.tool_runner.calls)
        if self.cfg.tool_calls:
            from hope.runtime.toolcalls import generate_with_tools
            text = generate_with_tools(model, prompt, self.tool_runner, max_new=self.cfg.answer_max_bytes,
                                       temperature=temperature, **kw)
        else:
            text = model.generate(prompt, max_new=self.cfg.answer_max_bytes, temperature=temperature,
                                  stop_after_line=b"Resultado:", **kw).decode("utf-8", errors="replace")
        calls = [{"herramienta": c.name, "entrada": c.input[:120], "ok": c.ok}
                 for c in self.tool_runner.calls[n0:]]
        return text, calls

    def _uncertainty(self, prompt: str, completion: str) -> tuple[float, str]:
        """Incertidumbre media que predice la cabeza de Hope sobre su propia respuesta y la línea más dudosa."""
        import torch
        model = self.trainer.model
        pb, cb = prompt.encode("utf-8")[-1024:], completion.encode("utf-8")[:1024]
        if not cb:
            return 0.0, ""
        seq = torch.tensor([list(pb + cb)], device=model.head.weight.device)
        was = model.training
        model.eval()
        with torch.no_grad():
            u = model(seq[:, :-1])["uncertainty"][0, max(len(pb) - 1, 0):].float().cpu().numpy()
        model.train(was)
        best, best_line, pos = -1.0, "", 0
        for line in completion.splitlines(keepends=True):
            n = len(line.encode("utf-8"))
            seg = u[pos:pos + n]
            if len(seg) and line.strip() and float(seg.mean()) > best:
                best, best_line = float(seg.mean()), line
            pos += n
        return float(u.mean()), best_line.strip()

    def attempt(self, p: Problem, extra: str = "", *, temperature: float = 0.0) -> Attempt:
        prompt = self.prompt_for(p, extra)
        text, calls = self._generate(prompt, temperature, p.image)
        ans = extract_answer(text)
        crit = critique(p.problema, text, ans)
        ok = answers_match(ans, p.respuesta) and crit.verdict in ("ok", "incierto")
        issues = [{"tipo": i.kind, "detalle": i.detail, "paso": i.step} for i in crit.issues]
        return Attempt(prompt, text, ans, ok, crit.verdict, issues, calls)

    # --- 5. preguntas de Hope ---------------------------------------------------------
    def maybe_ask(self, p: Problem, att: Attempt) -> str:
        """Si Hope duda, pregunta al tutor por la línea más dudosa. Devuelve la ayuda utilizable ('' si no)."""
        if self._questions_this_round >= self.cfg.max_questions_per_round:
            return ""
        unc, doubt = self.uncertainty_fn(att.prompt, att.text)
        att.uncertainty = unc
        if unc < self.cfg.ask_threshold:
            return ""
        self._questions_this_round += 1
        rep = self._ask_json(question_to_tutor(p.tema, p.problema, att.text, doubt or att.text[-120:]),
                             f"pregunta:{p.tema}")
        entry = {"tema": p.tema, "problema": p.problema, "duda": doubt, "incertidumbre": round(unc, 3),
                 "respuesta_tutor": None, "estado": "sin_respuesta"}
        if rep and rep.get("respuesta"):
            reply = str(rep["respuesta"])
            chk = rep.get("verificacion") if isinstance(rep.get("verificacion"), dict) else None
            status = check_step(self.catalog, chk) if chk else "no_verificable"
            entry["respuesta_tutor"] = reply
            if status == "falla":
                entry["estado"] = "refutada"
                self._tutor_error("respuesta_refutada", tema=p.tema, pregunta=doubt, respuesta=reply)
            elif leaks_answer(reply, p.respuesta):
                entry["estado"] = "filtra_la_respuesta"           # no se usa: Hope copiaría
            else:
                entry["estado"] = "verificada" if status == "ok" else "no_verificable"
        self.questions.append(entry)
        self._record("pregunta_de_hope", EpistemicStatus.OBSERVADO, **entry)
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with open(self.out_dir / "preguntas.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if entry["estado"] in ("verificada", "no_verificable"):
            return f"Respuesta del tutor a mi duda: {entry['respuesta_tutor']}"
        return ""

    # --- revisión del tutor + verificación automática -------------------------------
    def review(self, p: Problem, att: Attempt) -> dict:
        notes = "\n".join(f"- {('paso ' + str(i['paso']) + ': ') if i['paso'] else ''}{i['detalle']}"
                          for i in att.issues)
        calls = [f"{c['herramienta']}({c['entrada']}) {'ok' if c['ok'] else 'error'}" for c in att.tool_calls]
        rep = self._ask_json(review_request(p.tema, p.problema, att.text, att.answer,
                                            p.respuesta if p.verificado else None, notes, calls),
                             f"revision:{p.tema}") or {}
        kind = str(rep.get("tipo", "")).lower() or (att.issues[0]["tipo"] if att.issues else "")
        step = rep.get("paso") if isinstance(rep.get("paso"), int) else None
        if step is None:
            steps = [i["paso"] for i in att.issues if i.get("paso")]
            step = min(steps) if steps else None
        tutor_says = rep.get("correcto")
        if tutor_says is not None and bool(tutor_says) != att.correct:
            # La verificación automática manda: el tutor se equivocó al juzgar el intento.
            self._tutor_error("revision_contradice_verificacion", tema=p.tema, problema=p.problema,
                              tutor_dice=bool(tutor_says), verificado=att.correct)
        scores = rep.get("puntaje") if isinstance(rep.get("puntaje"), dict) else {}
        return {"correcto": att.correct, "tutor_correcto": tutor_says, "error": str(rep.get("error", "")),
                "tipo": kind or "procedimiento", "paso": step,
                "pista": safe_hint(rep.get("pista"), p.respuesta, step=step, kind=kind),
                "pista_filtrada": bool(rep.get("pista")) and leaks_answer(str(rep.get("pista")), p.respuesta),
                "idea_clave": str(rep.get("idea_clave", "")), "error_a_evitar": str(rep.get("error_a_evitar", "")),
                "puntaje": {k: v for k, v in scores.items() if isinstance(v, (int, float))}}

    def _reward(self, p: Problem, att: Attempt, scores: dict) -> float:
        if not answers_match(att.answer, p.respuesta):
            verified = 0.0
        else:
            verified = 1.0 if att.verdict in ("ok", "incierto") else 0.2
        r = combined_reward(verified, scores, self.cfg.tutor_weight)
        self.rewards.append({"tema": p.tema, "prompt": att.prompt, "texto": att.text, "verificada": verified,
                             "puntaje_tutor": scores, "recompensa": r})
        return r

    def _update_profile(self, p: Problem, att: Attempt, rev: dict | None) -> None:
        tp = self.profile.get(p.tema)
        tp.attempts += 1
        tp.correct += int(att.correct)
        tp.recent = (tp.recent + [int(att.correct)])[-50:]
        if not att.correct:
            k = (rev or {}).get("tipo") or (att.issues[0]["tipo"] if att.issues else "desconocido")
            tp.errors[k] = tp.errors.get(k, 0) + 1
        tp.tool_calls += len(att.tool_calls)
        tp.tool_ok += sum(c["ok"] for c in att.tool_calls)
        for k, v in ((rev or {}).get("puntaje") or {}).items():
            s = tp.scores.setdefault(k, [0.0, 0])
            s[0] += float(v)
            s[1] += 1

    # --- 2. intento → crítica → pista → reintento ------------------------------------
    def tutor_problem(self, p: Problem, *, review_item: ReviewItem | None = None) -> dict:
        self.seen.add(p.key())
        topic = self.topics.get(p.tema) or Topic(p.tema)
        att = self.attempt(p)
        help_ = self.maybe_ask(p, att)
        if help_:                                            # reintento con la ayuda pedida por Hope
            att = self.attempt(p, help_)
        rev = self.review(p, att)
        self._update_profile(p, att, rev)
        self._reward(p, att, rev["puntaje"])
        trainable = p.verificado or self.cfg.train_unverified
        out = {"tema": p.tema, "problema": p.problema, "origen": p.origen, "primer_intento": att.correct,
               "pista_filtrada": rev["pista_filtrada"], "pista": None,
               "reintento": None, "explicacion": None, "variaciones": 0}
        if review_item is not None:
            self.reviews.record(review_item, att.correct)
        if att.correct:
            if trainable:
                self.data.add("solucion_propia", [(att.prompt, False), (att.text, True)], tema=p.tema,
                              origen=p.origen)
            self._record("tutoria", EpistemicStatus.OBSERVADO, **out)
            return out
        out["pista"] = rev["pista"]
        att2 = self.attempt(p, f"Pista del tutor: {rev['pista']}")
        out["reintento"] = att2.correct
        self._reward(p, att2, {})
        revision = f"Revisión: {rev['error'] or rev['pista']}"
        if att2.correct:
            self.profile.get(p.tema).hints_helped += 1
            if trainable:
                self.data.add("intento_corregido",
                              corrected_segments(att.prompt, att.text, rev["paso"], revision, att2.text), tema=p.tema,
                              origen=p.origen)
        else:
            out["explicacion"] = self.explain(topic, p, [att.text, att2.text], rev)
            if out["explicacion"] and trainable and out["explicacion"]["texto"]:
                self.data.add("intento_corregido",
                              corrected_segments(att.prompt, att.text, rev["paso"], revision,
                                                 out["explicacion"]["texto"]),
                              weight=out["explicacion"]["peso"], tema=p.tema, origen=p.origen)
        out["variaciones"] = len(self.targeted_variations(topic, p, rev))
        self._record("tutoria", EpistemicStatus.OBSERVADO, **out)
        return out

    def explain(self, topic: Topic, p: Problem, attempts: list[str], rev: dict) -> dict | None:
        """Explicación completa (episodio) verificada; Hope anota la idea clave y el error a evitar."""
        rep = self._ask_json(explanation_request(topic, p.problema, attempts, rev["error"] or rev["tipo"]),
                             f"explicacion:{topic.name}")
        if rep is None:
            return None
        try:
            ep = verify_episode(self.catalog, Episode(parse_episode(rep["_texto"]), rep.get("_modelo", "")))
        except EpisodeParseError as e:
            self._tutor_error("episodio_invalido", tema=topic.name, error=str(e))
            return None
        status = ep.verification["status"]
        final_ok = answers_match(str(ep.data.get("resultado", "")), p.respuesta) if p.verificado else True
        if status == EpistemicStatus.REFUTADO.value or (p.verificado and not final_ok
                                                         and ep.data.get("problema", "").strip() == p.problema):
            self._tutor_error("explicacion_refutada", tema=topic.name, problema=p.problema,
                              verificacion=ep.verification)
            return {"estado": "REFUTADO", "texto": "", "peso": 0.0}
        self.profile.get(p.tema).explanations += 1
        text = ep.training_text(tool_calls=self.cfg.tool_calls)
        weight = float(ep.verification.get("weight", 1.0 if status == "RESPALDADO" else 0.3))
        verified = status == EpistemicStatus.RESPALDADO.value
        self.data.add("episodio_tutor", [(text, True)], weight=weight, tema=topic.name)
        # Hope anota en su cuaderno: la idea clave y el error a evitar (lección en la base).
        idea = rev["idea_clave"] or str(ep.data.get("plan", "")) or rev["pista"]
        avoid = rev["error_a_evitar"] or rev["error"] or "; ".join(map(str, ep.data.get("errores_frecuentes", [])))
        src = rep.get("_modelo") or "tutor"
        if idea:
            self.notebook.write(f"{topic.name}: idea clave", idea, source=src, verified=verified)
        if avoid:
            self.notebook.write(f"{topic.name}: error a evitar", avoid, source=src, verified=verified)
        self.kb.learn_lesson(topic.name, p.problema, None, [x for x in (rev["error"], avoid) if x], source="tutor")
        return {"estado": status, "texto": text, "peso": weight}

    # --- 3. variaciones dirigidas -----------------------------------------------------
    def targeted_variations(self, topic: Topic, p: Problem, rev: dict) -> list[Problem]:
        left = self.cfg.max_variations_per_topic - self.variations_added[topic.name]
        n = min(self.cfg.variations_per_error, left)
        if n <= 0:
            return []
        focus = rev["error"] or rev["error_a_evitar"] or rev["tipo"]
        probs = self.tutor_problems(topic, n, focus=f"{focus} (problema original: {p.problema})",
                                    origin="variacion")
        added = [v for v in probs if (v.verificado or self.cfg.train_unverified) and self.reviews.add(v)]
        self.variations_added[topic.name] += len(added)
        self._record("variaciones_dirigidas", EpistemicStatus.OBSERVADO, tema=topic.name, foco=focus,
                     pedidas=n, verificadas=len(added))
        return added

    # --- 1. diagnóstico -----------------------------------------------------------------
    def diagnose(self, topics: list[str] | None = None) -> dict:
        """Prueba a Hope en cada tema sin ayuda y arma el perfil. Los fallos los clasifica el tutor."""
        names = topics or list(self.topics)
        for name in names:
            t = self.topics[name]
            k = min(self.cfg.tutor_problems, self.cfg.diagnosis_items)
            probs = self.tutor_problems(t, k, origin="tutor")
            probs += self.generated_problems(t, self.cfg.diagnosis_items - len(probs))
            hits = 0
            for p in probs:
                self.seen.add(p.key())
                att = self.attempt(p)
                rev = None if att.correct else self.review(p, att)
                self._update_profile(p, att, rev)
                hits += att.correct
            if probs:
                self.profile.get(name).sessions.append(hits / len(probs))
        if self.cfg.images:
            imgs = self.image_problems(self.cfg.diagnosis_items)
            for p in imgs:
                att = self.attempt(p)
                self._update_profile(p, att, None)
            if imgs:
                tp = self.profile.get("imagen")
                tp.sessions.append(tp.recent_accuracy(len(imgs)) or 0.0)
        rep = {"perfil": self.profile.to_dict(), "fortalezas": self.profile.strengths(),
               "debilidades": self.profile.weaknesses()}
        self._record("diagnostico", EpistemicStatus.OBSERVADO, fortalezas=rep["fortalezas"],
                     debilidades=rep["debilidades"],
                     exactitud={t: p.accuracy for t, p in self.profile.topics.items()})
        return rep

    # --- 6. currículo adaptativo --------------------------------------------------------
    def update_mastery(self) -> None:
        c = self.cfg
        for name, tp in self.profile.topics.items():
            acc = tp.recent_accuracy(c.mastery_window)
            tp.mastered = (len(tp.recent) >= c.mastery_min_items and acc is not None
                           and acc >= c.mastery_accuracy and self.reviews.pending(name) == 0)

    def mastered(self, topic: str) -> bool:
        return self.profile.get(topic).mastered

    def priority(self, name: str) -> float:
        tp = self.profile.get(name)
        due = self.reviews.pending(name)
        if tp.mastered and not due:
            return 0.0
        acc = tp.recent_accuracy(self.cfg.mastery_window)
        weakness = 1.0 if acc is None else 1.0 - acc
        stagnant = max(0.0, -tp.improvement) + (0.1 if len(tp.sessions) >= 2 and tp.improvement == 0 else 0.0)
        level = self.topics[name].level if name in self.topics else 1
        open_levels = [self.topics[n].level for n in self.topics if not self.profile.get(n).mastered]
        gate = self.cfg.level_gate * max(0, level - min(open_levels)) if open_levels else 0.0
        return max(0.0, weakness + self.cfg.review_weight * min(due, 4) / 4
                   + self.cfg.stagnation_weight * stagnant - gate) + (0.05 if due else 0.0)

    def next_topics(self, k: int | None = None) -> list[str]:
        self.update_mastery()
        ranked = sorted(self.topics, key=lambda n: (-self.priority(n), self.topics[n].level, n))
        return [n for n in ranked if self.priority(n) > 0][: k or self.cfg.topics_per_round]

    # --- 7. consolidación (SFT) y refuerzo (RL) ---------------------------------------------
    def lesson_texts(self) -> list[str]:
        return self.notebook.practice_texts(self.notebook.due(50))

    def consolidate(self) -> dict:
        lessons = self.lesson_texts()
        for t in lessons:
            self.data.add("leccion", [(t, True)], weight=0.5)
        if not self.data.records or self.cfg.consolidation_steps <= 0:
            return {}
        if self.guardian is not None:
            d = self.guardian.decide(Action("experiment.train", {"steps": self.cfg.consolidation_steps,
                                                                 "kind": "consolidación de la tutoría"},
                                            requested_by="tutor"))
            if not d.allowed:
                return {"denied": list(d.reasons)}
        src = self.data.source("tutor", 1.0 - (self.cfg.replay_weight if self.replay else 0.0))
        sources = [src] + [Source(r.name, r.data, self.cfg.replay_weight / len(self.replay), r.mask)
                           for r in self.replay]
        probe = ["".join(t for t, m in r["segmentos"] if m) for r in self.data.records[-20:]]
        before = self.trainer.text_loss(probe)
        rep = self.trainer.train(Mixture(sources), self.cfg.consolidation_steps,
                                 eval_sources=[s for s in sources if s.name != "tutor"] or [src])
        out = {"pasos": rep.steps, "bpb_antes": before, "bpb_despues": self.trainer.text_loss(probe),
               "registros": len(self.data.records), "lecciones": len(lessons)}
        self._record("consolidacion_tutoria", EpistemicStatus.OBSERVADO, **out)
        return out

    def reinforce(self) -> dict:
        """GRPO con recompensa = verificación automática ⊗ notas del tutor (después del SFT)."""
        from hope.train.rlvr import GRPO, RLConfig, default_extract, verified_reward
        topics = [n for n in self.topics
                  if (self.profile.get(n).recent_accuracy(self.cfg.mastery_window) or 0) >= self.cfg.rl_min_accuracy]
        if not topics:
            return {"omitido": "ningún tema supera rl_min_accuracy (SFT antes que RL)"}
        probs = [p for n in topics for p in self.generated_problems(self.topics[n], self.cfg.rl_problems)]
        if not probs:
            return {"omitido": "sin problemas verificables"}
        if not hasattr(self, "_grpo"):
            self._grpo = GRPO(self.trainer.model, RLConfig(group=self.cfg.rl_group, max_new=self.cfg.rl_max_new))
        cache: dict[str, float] = {}

        def reward(i: int, text: str) -> float:
            p = probs[i]
            prob = {"problema": p.problema, "respuesta": p.respuesta}
            v = verified_reward(prob, text, default_extract, answers_match)
            key = hashlib.sha1(f"{i}\x00{text}".encode()).hexdigest()
            if key in cache:
                return cache[key]
            scores = {}
            if v > 0 and self.cfg.rl_tutor_scores:      # el tutor solo afina entre las correctas
                att = Attempt(prompts[i], text, default_extract(text), True, "ok", [], [])
                scores = self.review(p, att)["puntaje"]
            cache[key] = combined_reward(v, scores, self.cfg.tutor_weight)
            return cache[key]

        prompts = [self.prompt_for(p) for p in probs]
        stats = [self._grpo.step(prompts, reward) for _ in range(self.cfg.rl_steps)]
        out = {"temas": topics, "problemas": len(probs), "pasos": stats}
        self._record("refuerzo_tutoria", EpistemicStatus.OBSERVADO, **out)
        return out

    # --- ronda completa -----------------------------------------------------------------
    def review_round(self, limit: int = 20) -> int:
        n = 0
        for item in self.reviews.due(limit=limit):
            d = item.problem
            p = Problem(d["tema"], d["problema"], d["respuesta"], d.get("origen", "variacion"),
                        d.get("verificado", True), d.get("foco", ""))
            self.tutor_problem(p, review_item=item)
            n += 1
        return n

    def round(self) -> dict:
        t0 = time.time()
        self._questions_this_round = 0
        if not any(tp.attempts for tp in self.profile.topics.values()):
            self.diagnose()                              # primera ronda: diagnóstico sin ayuda
        chosen = self.next_topics()
        results = []
        for name in chosen:
            t = self.topics[name]
            due = self.reviews.due(name, limit=self.cfg.problems_per_topic)
            for item in due:
                d = item.problem
                results.append(self.tutor_problem(Problem(d["tema"], d["problema"], d["respuesta"],
                                                          d.get("origen", "variacion"), d.get("verificado", True),
                                                          d.get("foco", "")), review_item=item))
            k = self.cfg.problems_per_topic - len(due)
            probs = self.tutor_problems(t, min(self.cfg.tutor_problems, max(k, 0)))
            probs += self.generated_problems(t, k - len(probs))
            hits = 0
            for p in probs:
                r = self.tutor_problem(p)
                hits += r["primer_intento"]
                results.append(r)
            if probs:
                self.profile.get(name).sessions.append(hits / len(probs))
        cons = self.consolidate()
        rl = self.reinforce() if self.cfg.rl else {}
        self.update_mastery()
        self.rounds += 1
        summary = {"ronda": self.rounds, "temas": chosen, "problemas": len(results),
                   "primer_intento": sum(r["primer_intento"] for r in results),
                   "con_pista": sum(bool(r["reintento"]) for r in results),
                   "explicaciones": sum(r["explicacion"] is not None for r in results),
                   "variaciones": sum(r["variaciones"] for r in results),
                   "preguntas": self._questions_this_round, "consultas_tutor": self.tutor.calls,
                   "errores_tutor": len(self.tutor_errors),
                   "dominados": [n for n in self.topics if self.mastered(n)],
                   "consolidacion": cons, "refuerzo": rl, "segundos": round(time.time() - t0, 2)}
        self._record("ronda_tutoria", EpistemicStatus.OBSERVADO, **summary)
        self.save()
        return summary

    def run(self, rounds: int = 10, max_hours: float | None = None) -> list[dict]:
        t0, hist = time.time(), []
        for _ in range(rounds):
            if self.tutor.calls >= self.cfg.max_tutor_calls:
                self.log("presupuesto de consultas al tutor agotado")
                break
            if max_hours is not None and (time.time() - t0) / 3600 >= max_hours:
                break
            if not self.next_topics():
                self.log("todos los temas dominados y sin repasos vencidos")
                break
            s = self.round()
            hist.append(s)
            self.log(f"tutoría ronda {s['ronda']}: temas {s['temas']}, aciertos {s['primer_intento']}/"
                     f"{s['problemas']}, consultas {s['consultas_tutor']}, dominados {s['dominados']}")
        return hist

    # --- persistencia -----------------------------------------------------------------------
    def save(self) -> None:
        if not self.out_dir:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        st = {"rondas": self.rounds, "perfil": self.profile.to_dict(), "repaso": self.reviews.to_list(),
              "vistos": sorted(self.seen)[-5000:], "variaciones": dict(self.variations_added),
              "errores_tutor": self.tutor_errors[-500:]}
        (self.out_dir / "tutor_estado.json").write_text(json.dumps(st, indent=2, ensure_ascii=False, default=str),
                                                        encoding="utf-8")
        with open(self.out_dir / "recompensas.jsonl", "w", encoding="utf-8") as f:
            for r in self.rewards:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.data.write(self.out_dir / "datos")

    def load(self) -> bool:
        p = self.out_dir / "tutor_estado.json" if self.out_dir else None
        if not p or not p.exists():
            return False
        st = json.loads(p.read_text(encoding="utf-8"))
        self.rounds = st.get("rondas", 0)
        self.profile = Profile.from_dict(st.get("perfil", {}))
        self.reviews.load(st.get("repaso", []))
        self.seen = set(st.get("vistos", []))
        self.variations_added = Counter(st.get("variaciones", {}))
        self.tutor_errors = st.get("errores_tutor", [])
        reg = self.out_dir / "datos" / "registros.jsonl"
        if reg.exists():
            for line in reg.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                r.pop("split", None)
                r["segmentos"] = [tuple(s) for s in r["segmentos"]]
                self.data.records.append(r)
        return True

    def _record(self, kind: str, status: EpistemicStatus, **data) -> None:
        if self.pesam is not None:
            self.pesam.record(kind, status, **data)


# ---------------------------------------------------------------------------
# Tutor simulado (pruebas): extiende MockTeacher
# ---------------------------------------------------------------------------
class MockTutor(MockTeacher):
    """Tutor determinista que entiende las tareas de ``TUTOR_TAREA``. Para probar la verificación
    puede equivocarse a propósito:

    * ``wrong_problems``: fracción de problemas con la respuesta equivocada (la verificación los descarta);
    * ``leaky_hints``: la pista incluye la respuesta (se reemplaza por una genérica);
    * ``wrong_reviews``: dice «correcto» cuando no lo es y viceversa (la verificación manda);
    * ``error_rate`` (de MockTeacher): episodios con un paso equivocado (no se entrenan).
    """

    def __init__(self, config: TeacherConfig | None = None, guardian: Guardian | None = None, *,
                 wrong_problems: float = 0.0, leaky_hints: bool = False, wrong_reviews: bool = False,
                 error_rate: float = 0.0, seed: int = 0):
        super().__init__(config, guardian, error_rate=error_rate, seed=seed)
        self.wrong_problems, self.leaky_hints, self.wrong_reviews = wrong_problems, leaky_hints, wrong_reviews
        self.tasks: Counter = Counter()

    def _complete(self, messages: list[dict]) -> tuple[str, dict]:
        prompt = messages[-1]["content"]
        m = re.search(r"TUTOR_TAREA:\s*(\w+)", prompt)
        if not m:
            return super()._complete(messages)
        task = m.group(1)
        self.tasks[task] += 1
        if task == "explicacion":
            return super()._complete(messages)
        out = getattr(self, f"_t_{task}")(prompt)
        return "<think>borrador</think>\n```json\n" + json.dumps(out, ensure_ascii=False) + "\n```", {"mock": True}

    def _sum(self, carry: bool) -> tuple[int, int]:
        while True:
            a, b = self.rnd.randint(10, 89), self.rnd.randint(10, 89)
            if ((a % 10 + b % 10) >= 10) == carry:
                return a, b

    def _t_problemas(self, prompt: str) -> dict:
        n = int(re.search(r"CANTIDAD:\s*(\d+)", prompt).group(1))
        targeted = "Atacá exactamente" in prompt
        out = []
        for _ in range(n):
            a, b = self._sum(carry=targeted or self.rnd.random() < 0.5)
            ans = a + b + (1 if self.rnd.random() < self.wrong_problems else 0)
            out.append({"problema": f"¿Cuánto es {a} + {b}?", "respuesta": str(ans),
                        "verificacion": {"tipo": "aritmetica", "expr": f"{a} + {b}", "esperado": str(ans)}})
        return {"problemas": out}

    def _t_revision(self, prompt: str) -> dict:
        exp = re.search(r"RESPUESTA VERIFICADA:\s*(.+)", prompt)
        given = re.search(r"RESPUESTA DE HOPE:\s*(.+)", prompt).group(1).strip()
        given = None if given == "None" else given.strip("'\"")
        ok = bool(exp) and answers_match(given, exp.group(1).strip())
        nums = re.search(r"PROBLEMA:.*?(\d+)\s*\+\s*(\d+)", prompt)
        carry = bool(nums) and (int(nums.group(1)) % 10 + int(nums.group(2)) % 10) >= 10
        hint = "Sumá primero las unidades y fijate si pasan de 9: en ese caso se lleva 1 a las decenas."
        if self.leaky_hints and exp:
            hint += f" La respuesta es {exp.group(1).strip()}."
        return {"correcto": (not ok) if self.wrong_reviews else ok,
                "error": "" if ok else ("olvidó el acarreo al sumar las unidades" if carry
                                        else "no llegó al resultado con una suma correcta"),
                "tipo": "ninguno" if ok else ("procedimiento" if carry else "calculo"), "paso": 1,
                "pista": "" if ok else hint,
                "idea_clave": "Si las unidades suman 10 o más, se escribe la unidad y se lleva 1 a las decenas.",
                "error_a_evitar": "olvidar el acarreo",
                "puntaje": {"pasos": 0.9 if ok else 0.2, "herramienta": 0.8, "eficiencia": 0.7}}

    def _t_pregunta(self, prompt: str) -> dict:
        return {"respuesta": "Separá unidades y decenas; si las unidades pasan de 9, llevá 1.",
                "verificacion": {"tipo": "aritmetica", "expr": "7 + 5", "esperado": "12"}}


# ---------------------------------------------------------------------------
# Etapa de ``hope.train.run`` y línea de comandos
# ---------------------------------------------------------------------------
def run_stage(trainer: Trainer, spec: dict, out: Path, *, guardian: Guardian | None = None,
              max_hours: float | None = None, log=print) -> dict:
    """Etapa ``tutoria`` de un plan. ``spec``: maestro, curriculo, horas, datos, rondas_max,
    pasos_consolidacion, opcional, rl. Guarda el estado en ``out`` y los datos en ``spec['datos']``
    (fuente con máscara para las etapas siguientes)."""
    from hope.teacher.client import load_teacher
    from hope.tools.builtin import default_catalog
    out.mkdir(parents=True, exist_ok=True)
    tutor = load_teacher(spec.get("maestro", "configs/tutor_a770.yaml"), guardian)
    if not isinstance(tutor, MockTeacher) and spec.get("opcional"):
        import urllib.request
        try:                                           # sin servidor local la etapa opcional se omite
            urllib.request.urlopen(tutor.config.endpoint.rstrip("/") + "/models", timeout=5)
        except Exception as e:
            log(f"etapa de tutoría omitida (opcional; tutor no disponible: {e})")
            return {"omitida": str(e)}
    cfg = TutorConfig(**{k: v for k, v in spec.get("config", {}).items() if k in TutorConfig.__dataclass_fields__})
    if "pasos_consolidacion" in spec:
        cfg.consolidation_steps = int(spec["pasos_consolidacion"])
    cfg.rl = bool(spec.get("rl", cfg.rl))
    sess = TutorSession(trainer, tutor, MemoryStore(out / "memoria.sqlite"), default_catalog(guardian),
                        load_curriculum(spec.get("curriculo", "configs/curriculum.yaml")), cfg=cfg,
                        guardian=guardian, pesam=ExperienceLog(out / "pesam.jsonl"), out_dir=out, log=log)
    sess.load()
    hist = sess.run(int(spec.get("rondas_max", 10**6)), max_hours=max_hours)
    stats = sess.data.write(spec.get("datos", "data/tutor"))
    return {"rondas": len(hist), "datos": stats, "consultas": tutor.calls}


def main(argv: list[str] | None = None) -> None:
    from guardian.cli import load as load_guardian
    from hope.core.hn1 import HN1Model, preset
    from hope.data.corpus import ByteData
    from hope.teacher.client import load_teacher
    from hope.tools.builtin import default_catalog
    from hope.train.continual import TrainConfig

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resume", default=None, help="checkpoint de Hope (sin esto, modelo nuevo)")
    ap.add_argument("--preset", default="hn1-25m")
    ap.add_argument("--teacher", default="configs/tutor_a770.yaml")
    ap.add_argument("--curriculum", default="configs/curriculum.yaml")
    ap.add_argument("--guardian", default="configs/guardian.yaml")
    ap.add_argument("--replay", action="append", default=[], help="carpeta de corpus para repaso (repetible)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--max-tutor-calls", type=int, default=1000)
    ap.add_argument("--rl", action="store_true", help="refuerzo con la recompensa combinada después del SFT")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    guardian = load_guardian(args.guardian)
    tcfg = TrainConfig(batch=8, seq_bytes=512)
    trainer = (Trainer.resume(args.resume, tcfg, args.device) if args.resume
               else Trainer(HN1Model(preset(args.preset)), tcfg, args.device))
    out = Path(args.out)
    sess = TutorSession(trainer, load_teacher(args.teacher, guardian), MemoryStore(out / "memoria.sqlite"),
                        default_catalog(guardian), load_curriculum(args.curriculum),
                        cfg=TutorConfig(max_tutor_calls=args.max_tutor_calls, rl=args.rl), guardian=guardian,
                        pesam=ExperienceLog(out / "pesam.jsonl"), out_dir=out,
                        replay=[Source(Path(r).name, ByteData(Path(r) / "train.bin"), 1.0) for r in args.replay])
    sess.load()
    sess.run(args.rounds, args.max_hours)
    trainer.save(out, extra={"tutoria": {"rondas": sess.rounds}})


if __name__ == "__main__":
    main()
