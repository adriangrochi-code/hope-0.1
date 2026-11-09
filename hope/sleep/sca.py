"""SCA — sueño cognitivo artificial (documento SCA; Carta art. 10).

En los ratos sin tareas, Hope repasa su experiencia y prepara trabajo para la
vigilia. **Durante el sueño solo imagina y propone**: no modifica pesos ni
estructura, no ejecuta acciones externas y todo lo que produce queda marcado
como ``GENERADO_EN_SUENO`` (SCA §2, §7). Durante la vigilia, ``wake_review``
somete las ideas a pruebas técnicas y actualiza su estado con evidencia.

Modos (SCA §4):
  * **ligero** — consolidación de memoria (MDD §8) y poda de estadísticas;
  * **asociativo** — relaciones nuevas entre engramas parecidos de temas distintos
    (marcadas como hipotéticas, nunca como causales);
  * **analítico** — reexamina fallos registrados en PESAM, encuentra errores
    recurrentes y, si hay un «respondedor», reintenta los problemas;
  * **creativo** — combina procedimientos de temas distintos en problemas compuestos;
  * **evolutivo** — lee el diagnóstico de uso (LTH) y propone cambios a ASG/AANS
    (iteraciones inútiles, especialistas ociosos, neuronas sin uso), sin aplicarlos.

El ciclo es interrumpible: guarda en qué modo quedó y retoma desde ahí (SCA §11).
"""

from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from guardian.core import Action, Guardian
from hope.epistemic import EpistemicStatus
from hope.memory.pesam import ExperienceLog
from hope.memory.store import MemoryStore, Relation
from hope.sleep.ideas import Idea, IdeaStore

MODES = ("ligero", "asociativo", "analitico", "creativo", "evolutivo")


@dataclass
class SleepBudget:
    max_seconds: float = 600.0
    max_ideas: int = 50
    associative_pairs: int = 20
    min_similarity: float = 0.35
    reattempts: int = 10


class SleepCycle:
    def __init__(self, memory: MemoryStore, ideas: IdeaStore, pesam: ExperienceLog | None = None, *,
                 answerer: Callable[[str, str], tuple[str, str | None]] | None = None,
                 usage_report: dict | None = None, guardian: Guardian | None = None,
                 state_path: str | Path | None = None, budget: SleepBudget = SleepBudget()):
        self.memory, self.ideas, self.pesam = memory, ideas, pesam
        self.answerer, self.usage_report, self.guardian = answerer, usage_report, guardian
        self.state_path = Path(state_path) if state_path else None
        self.budget = budget
        self.dream_id = f"dream_{uuid.uuid4().hex[:8]}"
        self._created = 0

    # --- utilidades ------------------------------------------------------------
    def _allowed(self) -> bool:
        if self.guardian is None:
            return True
        return self.guardian.decide(Action("memory.write", {"source": "sca", "dream": self.dream_id},
                                           requested_by="sca")).allowed

    def _propose(self, mode: str, **kw) -> Idea | None:
        if self._created >= self.budget.max_ideas:
            return None
        idea = self.ideas.add(Idea(self.dream_id, mode, **kw))
        if idea:
            self._created += 1
            if self.pesam:
                self.pesam.record("idea_del_sueno", EpistemicStatus.GENERADO_EN_SUENO,
                                  idea=idea.idea_id, mode=mode, description=idea.description)
        return idea

    # --- modos -------------------------------------------------------------------
    def ligero(self) -> dict:
        rep = self.memory.consolidate()
        rep["ngram_stats_pruned"] = self.memory.prune_ngram_stats(200_000)
        return rep

    def asociativo(self) -> dict:
        rows = self.memory.db.execute(
            "SELECT id, concept, content FROM engrams WHERE archived = 0 AND kind != 'pasaje' "
            "AND status != 'REFUTADO' ORDER BY updated DESC LIMIT ?", (self.budget.associative_pairs,)).fetchall()
        made = 0
        for eid, concept, content in rows:
            for other, sim in self.memory.vector_search(f"{concept}\n{content}", k=5):
                if other == eid or sim < self.budget.min_similarity:
                    continue
                o = self.memory.get(other)
                if o.concept == concept or o.kind == "pasaje" or o.status == EpistemicStatus.REFUTADO:
                    continue
                if any(n == other for n, _, _ in self.memory.neighbors(eid)):
                    continue
                self.memory.relate(eid, other, Relation.SIMILARIDAD, confidence=round(sim, 3),
                                   status=EpistemicStatus.GENERADO_EN_SUENO, provenance=self.dream_id)
                self._propose("asociativo",
                              description=f"«{concept}» y «{o.concept}» podrían compartir un principio",
                              hypothesis=f"lo aprendido en «{concept}» ayuda a resolver «{o.concept}»",
                              generating_reasoning=f"similitud de representación {sim:.2f} entre engramas de temas distintos",
                              knowledge_used=[eid, other], initial_confidence=0.2,
                              priority=sim, possible_contradictions=["similitud superficial sin estructura común"],
                              proposed_tests=[{"tipo": "transferencia", "desde": concept, "hacia": o.concept}])
                made += 1
                break
        self.memory.commit()
        return {"relations_proposed": made}

    def analitico(self) -> dict:
        failures = []
        if self.pesam:
            for row in self.pesam.query("evaluacion"):
                failures += row["data"].get("failures", [])
        kinds = Counter(k for f in failures for k in f.get("tipos_error", []))
        topics = Counter(f.get("tema", "?") for f in failures)
        for kind, n in kinds.most_common(3):
            if n >= 2:
                self._propose("analitico", description=f"error recurrente: {kind} ({n} veces)",
                              hypothesis=f"Hope necesita práctica específica en «{kind}»",
                              generating_reasoning="conteo de tipos de error de la crítica (CRE) en evaluaciones",
                              related_experiences=[f.get("problema", "") for f in failures if kind in f.get("tipos_error", [])][:5],
                              initial_confidence=min(0.9, 0.3 + 0.1 * n), priority=n,
                              proposed_tests=[{"tipo": "reevaluar", "problemas": [
                                  f for f in failures if kind in f.get("tipos_error", [])][:5]}])
        reattempted = solved = 0
        if self.answerer:
            from hope.teacher.cycle import answers_match
            for f in failures[-self.budget.reattempts:]:
                _, given = self.answerer(f.get("tema", ""), f["problema"])
                reattempted += 1
                if answers_match(given, f["esperado"]):
                    solved += 1
        return {"failures_seen": len(failures), "recurrent_error_kinds": dict(kinds),
                "weak_topics": dict(topics.most_common(5)), "reattempted": reattempted, "now_solved": solved}

    def creativo(self) -> dict:
        rows = self.memory.db.execute(
            "SELECT DISTINCT concept FROM engrams WHERE kind = 'procedimiento' AND status != 'REFUTADO'").fetchall()
        concepts = [r[0] for r in rows]
        made = 0
        for i, a in enumerate(concepts):
            for b in concepts[i + 1:]:
                if self._propose("creativo", description=f"problema compuesto: «{a}» + «{b}»",
                                 hypothesis=f"combinar «{a}» y «{b}» resuelve problemas que ninguno resuelve solo",
                                 generating_reasoning="combinación de procedimientos de temas distintos (HLA §13)",
                                 initial_confidence=0.15, priority=0.5,
                                 proposed_tests=[{"tipo": "ejercicio_compuesto", "temas": [a, b]}]):
                    made += 1
        return {"compositions_proposed": made}

    def evolutivo(self) -> dict:
        rep = self.usage_report or {}
        made = 0
        it = rep.get("iterations", {})
        gains = it.get("gain_of_each_extra_iteration", [])
        weights = it.get("weight_per_iteration", [])
        for n, (g, w) in enumerate(zip(gains, weights)):
            if n > 0 and abs(g) < 1e-3:
                if self._propose("evolutivo", description=f"la iteración {n + 1} del núcleo no mejora la pérdida",
                                 hypothesis="reentrenar la parada o reducir max_iters ahorra cómputo sin perder calidad",
                                 generating_reasoning=f"diagnóstico LTH: ganancia {g:.2e}, peso {w:.3f}",
                                 initial_confidence=0.4, priority=1.0 - w,
                                 proposed_tests=[{"tipo": "experimento_asg", "cambio": "max_iters",
                                                  "valor": n}]):
                    made += 1
        usage = rep.get("expert_usage", [])
        if usage:
            mean = sum(usage) / len(usage)
            for e, u in enumerate(usage):
                if u < 0.25 * mean:
                    made += bool(self._propose(
                        "evolutivo", description=f"especialista {e} casi sin uso ({u:.1%})",
                        hypothesis="reinicializarlo o reemplazarlo por otro tipo mejora el uso de la capacidad",
                        generating_reasoning="uso del router muy por debajo del promedio",
                        initial_confidence=0.3, priority=mean - u,
                        proposed_tests=[{"tipo": "experimento_asg", "cambio": "reemplazar_especialista",
                                         "especialista": e}]))
        for layer, frac in rep.get("idle_unit_fraction (Taylor < 10% de la mediana)", {}).items():
            if frac > 0.2:
                made += bool(self._propose(
                    "evolutivo", description=f"{frac:.0%} de neuronas ociosas en {layer}",
                    hypothesis="reinicializarlas (ReinitUnits) recupera capacidad sin agregar parámetros",
                    generating_reasoning="importancia de Taylor por debajo del 10% de la mediana",
                    initial_confidence=0.3, priority=frac,
                    proposed_tests=[{"tipo": "experimento_asg", "cambio": "reinit_units", "capa": layer}]))
        return {"architecture_ideas": made}

    # --- ciclo ------------------------------------------------------------------
    def run(self, modes: tuple[str, ...] = MODES, should_stop: Callable[[], bool] = lambda: False) -> dict:
        """Ejecuta los modos en orden. Si se interrumpe, guarda el estado para retomar."""
        start = 0
        if self.state_path and self.state_path.exists():
            st = json.loads(self.state_path.read_text(encoding="utf-8"))
            start, self.dream_id = st.get("next_mode", 0), st.get("dream_id", self.dream_id)
        if not self._allowed():
            return {"denied": True}
        t0, report = time.time(), {}
        for i in range(start, len(modes)):
            if should_stop() or time.time() - t0 > self.budget.max_seconds:
                self._save_state(i)
                report["interrupted_at"] = modes[i]
                return report
            report[modes[i]] = getattr(self, modes[i])()
        if self.state_path and self.state_path.exists():
            self.state_path.unlink()
        if self.pesam:
            self.pesam.record("sueno", EpistemicStatus.OBSERVADO, dream=self.dream_id, report=report)
        return report

    def _save_state(self, next_mode: int) -> None:
        if self.state_path:
            self.state_path.write_text(json.dumps({"next_mode": next_mode, "dream_id": self.dream_id}),
                                       encoding="utf-8")


# --- vigilia -----------------------------------------------------------------------
Tester = Callable[[Idea, dict], tuple[str | None, str, str]]   # → (resultado, evidencia, detalle)


def wake_review(ideas: IdeaStore, testers: dict[str, Tester], max_ideas: int = 20) -> dict:
    """Prueba las ideas pendientes con las pruebas disponibles (SCA §7, estado B).

    La vigilia trata cada idea con el mismo rigor que una propuesta ajena: no
    sube de estado por haber sido imaginada, solo por el resultado de una prueba.
    """
    counts = Counter()
    for idea in ideas.pending()[:max_ideas]:
        for test in idea.proposed_tests:
            tester = testers.get(test.get("tipo"))
            if tester is None:
                continue
            outcome, ref, detail = tester(idea, test)
            idea.apply_result(outcome, ref, detail)
            counts[outcome or "sin_conclusion"] += 1
        ideas.update(idea)
    return dict(counts)


def main(argv: list[str] | None = None) -> dict:
    """python -m hope.sleep.sca --dir runs/ensenanza [--usage runs/hn1_25m/usage.json] [--minutos 10]"""
    import argparse
    ap = argparse.ArgumentParser(description="Sueño cognitivo de Hope (SCA)")
    ap.add_argument("--dir", required=True, help="carpeta del ciclo de enseñanza (memoria.sqlite, pesam.jsonl)")
    ap.add_argument("--usage", default=None, help="reporte de uso LTH (usage.json)")
    ap.add_argument("--minutos", type=float, default=10.0)
    args = ap.parse_args(argv)
    d = Path(args.dir)
    usage = json.loads(Path(args.usage).read_text(encoding="utf-8")) if args.usage else None
    sca = SleepCycle(MemoryStore(d / "memoria.sqlite"), IdeaStore(d / "ideas.jsonl"), ExperienceLog(d / "pesam.jsonl"),
                     usage_report=usage, state_path=d / "sueno_estado.json",
                     budget=SleepBudget(max_seconds=args.minutos * 60))
    rep = sca.run()
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    return rep


if __name__ == "__main__":
    main()
