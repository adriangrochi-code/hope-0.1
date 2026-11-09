"""Ciclo maestro–alumno: Qwen enseña, Hope aprende, pregunta y vuelve a intentar.

Dos fases (decisión del usuario: «al inicio Qwen le enseña todo lo que pueda
hasta que Hope pueda razonar de forma semi-funcional»):

**1. Arranque (currículo).** Qwen enseña por iniciativa propia, tema por tema
y nivel por nivel, siguiendo ``configs/curriculum.yaml``. Cuando Hope supera
los temas de un nivel, se habilita el siguiente.

**2. Preguntas (carencias).** Una vez que Hope cumple el criterio de
«razonamiento semi-funcional», el ciclo se invierte: Hope intenta, detecta qué
falla y le hace a Qwen una pregunta concreta (qué intentó, qué tiene en memoria,
qué salió mal), en lugar de recibir lecciones genéricas (HLA §11 y §14).

Cada vuelta, en las dos fases:

  1. Qwen explica (episodio con pasos, verificación inversa y variaciones).
  2. Las herramientas verifican los pasos. Lo refutado no se entrena (queda como
     ejemplo de error); lo no verificable pesa menos.
  3. Lo verificado entra en la **memoria** → Hope puede usarlo enseguida (aprendizaje rápido).
  4. Cada vuelta se **consolida** en los pesos: episodios + repaso del corpus
     (aprendizaje consolidado, sin olvido catastrófico).
  5. Hope resuelve las **variaciones** (problemas que nunca entrenó), razonando
     paso a paso, y se mide la exactitud por tema.

Todo queda registrado en PESAM y respeta los presupuestos (consultas al maestro,
pasos de entrenamiento) y las decisiones del guardián.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from guardian.core import Action, Guardian
from hope.epistemic import EpistemicStatus
from hope.memory.pesam import ExperienceLog
from hope.memory.retrieval import retrieve
from hope.memory.store import Level, MemoryStore
from hope.teacher.client import Teacher, TeacherDenied
from hope.teacher.episode import Episode, EpisodeParseError, curriculum_request, parse_episode, question_request
from hope.tools.builtin import arith, numbers_equal
from hope.tools.catalog import Catalog
from hope.constraints.critic import critique
from hope.constraints.solver import analyze
from hope.runtime.loop import solve_with_review
from hope.data.generators import family_for, generate as generate_episodes
from hope.tools.verify import verify_episode
from hope.train.star import StarConfig, self_improve
from hope.train.continual import Mixture, Source, Trainer


@dataclass
class Topic:
    name: str
    level: int = 1
    language: str = "es"


@dataclass
class SemiFunctional:
    """Criterio para pasar de la fase de arranque a la de preguntas.

    Valores por defecto **provisionales**: los fija el usuario.
    """

    min_topic_accuracy: float = 0.6     # exactitud mínima en variaciones no vistas de un tema
    min_topic_fraction: float = 0.8     # fracción de temas evaluados que deben superarla
    min_items_per_topic: int = 5        # sin suficientes problemas, el tema no cuenta como superado
    all_levels_taught: bool = True      # hay que haber pasado por todos los niveles del currículo


@dataclass
class CycleConfig:
    episodes_per_topic: int = 4
    consolidation_steps: int = 200
    replay_weight: float = 0.5
    max_teacher_calls: int = 1000
    max_rounds: int = 100
    max_followups_per_topic: int = 2
    eval_items_per_topic: int = 20
    answer_max_bytes: int = 400
    answer_with_memory: bool = True
    constraint_hints: bool = True       # HCRI guía el razonamiento desde el principio (HCRI §3)
    tool_calls: bool = True             # Hope llama herramientas mientras razona (⟦calc: …⟧ → …)
    review_iterations: int = 2          # intentos con crítica de CRE (1 = sin revisión)
    generated_per_topic: int = 20       # episodios de los generadores verificables por tema y ronda
    self_training: bool = True          # STaR: practicar y aprender de soluciones propias verificadas
    practice_per_topic: int = 8
    star: StarConfig = field(default_factory=StarConfig)
    criterion: SemiFunctional = field(default_factory=SemiFunctional)


# ---------------------------------------------------------------------------
def extract_answer(text: str) -> str | None:
    m = re.search(r"Resultado:\s*(.+)", text)
    return m.group(1).strip() if m else None


def answers_match(given: str | None, expected: str) -> bool:
    if given is None:
        return False
    g, e = given.strip().rstrip("."), str(expected).strip().rstrip(".")
    try:
        return numbers_equal(arith(g.split()[0]), arith(e.split()[0])) and (
            g.split()[1:2] == e.split()[1:2])          # número y unidad (si la hay) deben coincidir
    except Exception:
        return re.sub(r"\W+", " ", g.lower()).strip() == re.sub(r"\W+", " ", e.lower()).strip()


class TeachingCycle:
    def __init__(self, trainer: Trainer, teacher: Teacher, memory: MemoryStore, catalog: Catalog,
                 curriculum: list[Topic], *, replay: list[Source] | None = None, cfg: CycleConfig = CycleConfig(),
                 guardian: Guardian | None = None, pesam: ExperienceLog | None = None,
                 out_dir: str | Path | None = None, log=print):
        self.trainer, self.teacher, self.memory, self.catalog = trainer, teacher, memory, catalog
        self.curriculum = sorted(curriculum, key=lambda t: t.level)
        self.replay = replay or []
        self.cfg, self.guardian, self.pesam, self.log = cfg, guardian, pesam, log
        self.out_dir = Path(out_dir) if out_dir else None
        self.phase = "arranque"
        self.level = self.curriculum[0].level if self.curriculum else 1
        self.training_texts: list[tuple[str, float]] = []           # (texto, peso)
        self.trained_problems: set[str] = set()
        self.eval_pool: dict[str, list[dict]] = {}                  # tema → variaciones (nunca se entrenan)
        self.last_eval: dict[str, dict] = {}
        self.history: list[dict] = []

    # --- maestro ----------------------------------------------------------------
    def _ask(self, messages, purpose: str) -> Episode | None:
        if self.teacher.calls >= self.cfg.max_teacher_calls:
            return None
        try:
            reply = self.teacher.ask(messages, purpose=purpose)
        except TeacherDenied as e:
            self.log(f"  maestro denegado: {e}")
            return None
        try:
            ep = Episode(parse_episode(reply.text), reply.model, reply.raw_thinking)
        except EpisodeParseError as e:
            self._record("episodio_invalido", EpistemicStatus.REFUTADO, purpose=purpose, error=str(e))
            return None
        return verify_episode(self.catalog, ep)

    def _absorb(self, ep: Episode, topic: Topic, origin: str) -> None:
        """Memoria inmediata + buffer de entrenamiento + variaciones para evaluar."""
        status = EpistemicStatus(ep.verification["status"])
        weight = ep.verification["weight"]
        text = ep.training_text(tool_calls=self.cfg.tool_calls)
        self.memory.add_engram(kind="procedimiento" if status != EpistemicStatus.REFUTADO else "error_del_maestro",
                               level=Level.PROCEDIMENTAL, concept=topic.name, content=text,
                               description=f"episodio de {ep.source_model or 'maestro'} ({origin})",
                               status=status, confidence=weight, importance=0.6 if weight else 0.2)
        if weight > 0:
            self.training_texts.append((text, weight))
            self.trained_problems.add(ep.data["problema"].strip())
            pool = self.eval_pool.setdefault(topic.name, [])
            for v in ep.variations:
                if v["problema"].strip() not in self.trained_problems:
                    pool.append({"problema": v["problema"], "respuesta": str(v["respuesta"]),
                                 "language": topic.language})
            self.eval_pool[topic.name] = pool[-self.cfg.eval_items_per_topic:]
        self._record("episodio", status, topic=topic.name, origin=origin, verification=ep.verification,
                     problem=ep.data["problema"])

    # --- Hope responde ---------------------------------------------------------
    def answer(self, topic: str, problem: str) -> tuple[str, str | None]:
        text, ans, _ = self.answer_reviewed(topic, problem)
        return text, ans

    def answer_reviewed(self, topic: str, problem: str):
        """Hope razona con restricciones (HCRI), CRE critica y el Loop Controller decide si reintentar."""
        base = f"Tema: {topic}\nProblema: {problem}\n"
        if self.cfg.answer_with_memory:
            found = retrieve(self.memory, f"{topic} {problem}", k=1, min_score=0.3)
            if found.found and found.items[0].engram.kind == "procedimiento":
                base = f"Ejemplo resuelto:\n{found.items[0].engram.content}\n\n" + base
        an = analyze(problem)
        if self.cfg.constraint_hints and an.hint():
            base += an.hint() + "\n"

        def attempt(feedback: str):
            prompt = base + (feedback + "\n" if feedback else "")
            text = self._generate(prompt, 0.0)
            return text, extract_answer(text)

        res = solve_with_review(attempt, lambda t, a: critique(problem, t, a, an),
                                max_iterations=self.cfg.review_iterations)
        return res.text, res.answer, res

    def _generate(self, prompt: str, temperature: float) -> str:
        model = self.trainer.model
        if self.cfg.tool_calls:
            from hope.runtime.toolcalls import ToolRunner, generate_with_tools
            if not hasattr(self, "tool_runner"):
                self.tool_runner = ToolRunner(self.catalog, self.memory)
            return generate_with_tools(model, prompt, self.tool_runner, max_new=self.cfg.answer_max_bytes,
                                       temperature=temperature)
        return model.generate(prompt, max_new=self.cfg.answer_max_bytes, temperature=temperature,
                              stop_after_line=b"Resultado:").decode("utf-8", errors="replace")

    def evaluate(self, topics: list[Topic]) -> dict[str, dict]:
        res = {}
        for t in topics:
            items = self.eval_pool.get(t.name, [])
            failures, correct = [], 0
            loops = {}
            for it in items:
                text, given, loop = self.answer_reviewed(t.name, it["problema"])
                loops[loop.stop_reason] = loops.get(loop.stop_reason, 0) + 1
                if answers_match(given, it["respuesta"]):
                    correct += 1
                else:
                    failures.append({"problema": it["problema"], "esperado": it["respuesta"],
                                     "respuesta_hope": given, "razonamiento": text[-300:],
                                     "critica": loop.critique.feedback(), "tema": t.name,
                                     "tipos_error": [i.kind for i in loop.critique.issues]})
            res[t.name] = {"items": len(items), "accuracy": correct / len(items) if items else None,
                           "failures": failures, "loop_stops": loops}
        self.last_eval = res
        self._record("evaluacion", EpistemicStatus.OBSERVADO,
                     results={k: {"items": v["items"], "accuracy": v["accuracy"]} for k, v in res.items()},
                     failures=[{k: f[k] for k in ("problema", "esperado", "respuesta_hope", "tema", "tipos_error")}
                               for v in res.values() for f in v["failures"]][:50])
        return res

    def semi_functional(self) -> bool:
        c = self.cfg.criterion
        if c.all_levels_taught and self.level < max(t.level for t in self.curriculum):
            return False
        scored = [v for v in self.last_eval.values() if v["items"] >= c.min_items_per_topic]
        if not scored:
            return False
        passed = sum(v["accuracy"] >= c.min_topic_accuracy for v in scored)
        return passed / len(scored) >= c.min_topic_fraction

    # --- consolidación en los pesos -------------------------------------------------
    def consolidate(self) -> dict:
        if not self.training_texts:
            return {}
        if self.guardian is not None:
            d = self.guardian.decide(Action("experiment.train", {"steps": self.cfg.consolidation_steps,
                                                                 "kind": "consolidación de episodios"},
                                            requested_by="hla"))
            if not d.allowed:
                return {"denied": list(d.reasons)}
        # El peso de cada episodio se aplica repitiéndolo (1,0 → 3 copias; 0,3 → 1 copia).
        texts = [t for t, w in self.training_texts for _ in range(max(1, round(w * 3)))]
        sources = [Source.from_texts("episodios", texts, 1.0 - (self.cfg.replay_weight if self.replay else 0))]
        for r in self.replay:
            sources.append(Source(r.name, r.data, self.cfg.replay_weight / len(self.replay)))
        probe = [t for t, _ in self.training_texts[-20:]]
        before = self.trainer.text_loss(probe)
        rep = self.trainer.train(Mixture(sources), self.cfg.consolidation_steps,
                                 eval_sources=[s for s in sources if s.name != "episodios"])
        after = self.trainer.text_loss(probe)
        out = {"steps": rep.steps, "episode_bpb_before": before, "episode_bpb_after": after,
               "replay_forgetting": rep.forgetting}
        self._record("consolidacion", EpistemicStatus.OBSERVADO, **out)
        return out

    # --- rondas ------------------------------------------------------------------
    def _topics_now(self) -> list[Topic]:
        return [t for t in self.curriculum if t.level <= self.level]

    def add_generated(self, topics: list[Topic]) -> int:
        """Episodios de los generadores procedurales (gratis, verificados por herramientas)."""
        n = 0
        for t in topics:
            fam = family_for(t.name)
            if fam is None or self.cfg.generated_per_topic <= 0:
                continue
            for ep in generate_episodes(fam, self.cfg.generated_per_topic, seed=self._round_seed, level=t.level,
                                        lang=t.language):
                self._absorb(verify_episode(self.catalog, ep), t, "generador")
                n += 1
        return n

    def practice(self, topics: list[Topic]) -> dict:
        """STaR sobre problemas generados de práctica (nunca los de evaluación)."""
        problems = []
        eval_texts = {it["problema"] for pool in self.eval_pool.values() for it in pool}
        for t in topics:
            fam = family_for(t.name)
            if fam is None:
                continue
            for ep in generate_episodes(fam, self.cfg.practice_per_topic, seed=10_000 + self._round_seed,
                                        level=t.level, lang=t.language):
                if ep.data["problema"] not in eval_texts:
                    problems.append({"tema": t.name, "problema": ep.data["problema"],
                                     "respuesta": ep.data["resultado"]})
        rep = self_improve(lambda prompt, temp: self._generate(prompt, temp),
                           problems, extract=extract_answer, match=answers_match, cfg=self.cfg.star)
        for text in rep.texts:
            self.training_texts.append((text, 1.0))
        self._record("autoentrenamiento", EpistemicStatus.OBSERVADO, **rep.as_dict())
        return rep.as_dict()

    @property
    def _round_seed(self) -> int:
        return len(self.history)

    def bootstrap_round(self) -> dict:
        topics = self._topics_now()
        for t in topics:
            for _ in range(self.cfg.episodes_per_topic):
                ep = self._ask(curriculum_request(t.name, t.level, t.language), purpose=f"currículo:{t.name}")
                if ep is not None:
                    self._absorb(ep, t, "currículo")
        generated = self.add_generated(topics)
        star = self.practice(topics) if self.cfg.self_training else {}
        cons = self.consolidate()
        ev = self.evaluate(topics)
        c = self.cfg.criterion
        level_topics = [t for t in topics if t.level == self.level]
        if all((ev[t.name]["accuracy"] or 0) >= c.min_topic_accuracy and ev[t.name]["items"] >= c.min_items_per_topic
               for t in level_topics):
            higher = [t.level for t in self.curriculum if t.level > self.level]
            if higher:
                self.level = min(higher)
                self.log(f"  nivel {self.level} habilitado")
        return {"consolidation": cons, "evaluation": ev, "generated": generated, "self_training": star}

    def question_round(self) -> dict:
        topics = self._topics_now()
        ev = self.last_eval or self.evaluate(topics)
        asked = 0
        for t in topics:
            failures = ev.get(t.name, {}).get("failures", [])[: self.cfg.max_followups_per_topic]
            for f in failures:
                known = [r.engram.content[:200] for r in retrieve(self.memory, f"{t.name} {f['problema']}", k=3).items]
                ep = self._ask(question_request(t.name, attempt=f["respuesta_hope"] or "(sin respuesta)",
                                                expected=f["esperado"], known=known, problem=f["problema"],
                                                language=t.language, review=f.get("critica", "")),
                              purpose=f"pregunta:{t.name}")
                if ep is not None:
                    asked += 1
                    self._absorb(ep, t, "pregunta de Hope")
        star = self.practice(topics) if self.cfg.self_training else {}
        cons = self.consolidate()
        return {"questions": asked, "consolidation": cons, "evaluation": self.evaluate(topics), "self_training": star}

    def run(self, rounds: int | None = None) -> list[dict]:
        for r in range(rounds or self.cfg.max_rounds):
            if self.teacher.calls >= self.cfg.max_teacher_calls:
                self.log("presupuesto de consultas al maestro agotado")
                break
            t0 = time.time()
            report = self.bootstrap_round() if self.phase == "arranque" else self.question_round()
            if self.phase == "arranque" and self.semi_functional():
                self.phase = "preguntas"
                self.log("criterio de razonamiento semi-funcional alcanzado: Hope pasa a preguntar")
                self._record("cambio_de_fase", EpistemicStatus.OBSERVADO, to="preguntas", round=r)
            summary = {"round": r, "phase": self.phase, "level": self.level, "teacher_calls": self.teacher.calls,
                       "generated": report.get("generated", 0), "self_training": report.get("self_training", {}),
                       "seconds": time.time() - t0, "consolidation": report.get("consolidation"),
                       "accuracy": {k: v["accuracy"] for k, v in report["evaluation"].items()}}
            self.history.append(summary)
            self.log(f"ronda {r}: fase {self.phase}, nivel {self.level}, consultas {self.teacher.calls}, "
                     f"exactitud {summary['accuracy']}")
            if self.out_dir:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                (self.out_dir / "rounds.json").write_text(json.dumps(self.history, indent=2, ensure_ascii=False,
                                                                     default=str), encoding="utf-8")
                self.trainer.save(self.out_dir, extra={"teaching": {"phase": self.phase, "level": self.level}})
        return self.history

    def _record(self, kind: str, status: EpistemicStatus, **data) -> None:
        if self.pesam is not None:
            self.pesam.record(kind, status, phase=self.phase, **data)


def load_curriculum(path: str | Path) -> list[Topic]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [Topic(**t) for t in raw["topics"]]


def main(argv: list[str] | None = None) -> None:
    from hope.core.hn1 import HN1Model, preset
    from hope.data.corpus import ByteData
    from hope.teacher.client import load_teacher
    from hope.tools.builtin import default_catalog
    from hope.train.continual import TrainConfig
    from guardian.cli import load as load_guardian

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", default=None, help="checkpoint de Hope (sin esto, modelo nuevo)")
    ap.add_argument("--preset", default="hn1-25m")
    ap.add_argument("--teacher", default="configs/teacher.yaml")
    ap.add_argument("--curriculum", default="configs/curriculum.yaml")
    ap.add_argument("--guardian", default="configs/guardian.yaml")
    ap.add_argument("--replay", action="append", default=[], help="carpeta de corpus para repaso (repetible)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--episodes-per-topic", type=int, default=4)
    ap.add_argument("--consolidation-steps", type=int, default=200)
    ap.add_argument("--max-teacher-calls", type=int, default=1000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    guardian = load_guardian(args.guardian)
    tcfg = TrainConfig(batch=8, seq_bytes=512)
    trainer = (Trainer.resume(args.resume, tcfg, args.device) if args.resume
               else Trainer(HN1Model(preset(args.preset)), tcfg, args.device))
    out = Path(args.out)
    cycle = TeachingCycle(
        trainer, load_teacher(args.teacher, guardian), MemoryStore(out / "memoria.sqlite"),
        default_catalog(guardian), load_curriculum(args.curriculum),
        replay=[Source(Path(r).name, ByteData(Path(r) / "train.bin"), 1.0) for r in args.replay],
        cfg=CycleConfig(episodes_per_topic=args.episodes_per_topic, consolidation_steps=args.consolidation_steps,
                        max_teacher_calls=args.max_teacher_calls),
        guardian=guardian, pesam=ExperienceLog(out / "pesam.jsonl"), out_dir=out)
    cycle.run(args.rounds)


if __name__ == "__main__":
    main()
