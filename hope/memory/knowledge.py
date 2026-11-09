"""Base de conocimiento de trabajo: guardar lo verificado, aprender de los errores y reanalizar mejor.

Pedido del usuario: «una pequeña base de datos donde pueda almacenar conocimiento y mejorar la
calidad al reanalizar cosas», optimizada al máximo. Se apoya en ``MemoryStore`` (SQLite +
vectores + n-gramas + estados epistémicos) y agrega lo que hace falta para razonar:

* **Soluciones verificadas** (``solucion``): problema, razonamiento y resultado que pasaron la
  crítica (y, si hubo, el consenso del consejo). Estado RESPALDADO; sin verificar, INFERIDO.
* **Lecciones** (``leccion``): los errores que encontró la crítica en respuestas descartadas
  («6 * 7 no da 41»). Son lo que más ayuda al reanalizar: dicen qué **no** hacer.
* **Contexto para responder** (``context``): antes de resolver, se recuperan las soluciones y
  lecciones más parecidas y se anteponen al problema. Lo refutado nunca se usa.

Optimización (para que la base no crezca sin control ni se llene de ruido):

1. **Sin duplicados**: si ya hay una solución casi igual con la misma respuesta, se refuerza
   (uso + confianza) en lugar de agregar otra.
2. **Contradicciones**: si una solución nueva **verificada** da otra respuesta para el mismo
   problema, la vieja pasa a REFUTADO y se registra la relación de contradicción.
3. **Utilidad medida**: cada recuerdo usado en una respuesta que terminó verificada suma un
   uso; los que se usan y no ayudan pierden confianza.
4. ``optimize()``: consolida (lo usado y respaldado pasa a memoria semántica; lo poco usado,
   poco importante y viejo se archiva), poda estadísticas de n-gramas y devuelve un informe.

Todo es texto con procedencia: se puede auditar qué sabe Hope y de dónde lo sacó.
"""

from __future__ import annotations

from dataclasses import dataclass

from hope.epistemic import EpistemicStatus
from hope.memory.retrieval import retrieve
from hope.memory.store import Level, MemoryStore, Relation


def _norm(s: str | None) -> str:
    return " ".join((s or "").lower().split())


@dataclass
class Context:
    text: str
    used: list[str]                      # ids de los recuerdos usados


class KnowledgeBase:
    def __init__(self, store: MemoryStore | None = None, *, path: str = ":memory:", dup_score: float = 0.9,
                 max_ngram_entries: int = 2_000_000):
        self.store = store or MemoryStore(path)
        self.dup_score, self.max_ngram_entries = dup_score, max_ngram_entries

    # --- escribir -------------------------------------------------------------------
    def _similar(self, kind: str, query: str, k: int = 5):
        res = retrieve(self.store, query, k=k, min_score=0.0)
        return [r for r in res.items if r.engram.kind == kind]

    def learn_solution(self, topic: str, problem: str, reasoning: str, answer: str | None, *,
                       verified: bool, source: str = "hope") -> str:
        content = f"Problema: {problem}\n{reasoning.strip()}\nResultado: {answer}"
        for r in self._similar("solucion", f"{topic} {problem}"):
            e = r.engram
            same_problem = _norm(e.description) == _norm(problem)
            if same_problem and _norm(e.key) == _norm(answer):
                self.store.touch(e.id)                                     # 1. refuerzo, no duplicado
                if verified and e.status != EpistemicStatus.RESPALDADO:
                    self.store.update_status(e.id, EpistemicStatus.RESPALDADO, f"verificado otra vez ({source})")
                self.store.commit()
                return e.id
            if same_problem and verified and e.status != EpistemicStatus.REFUTADO:
                self.store.update_status(e.id, EpistemicStatus.REFUTADO,     # 2. contradicción
                                         f"otra respuesta verificada: {answer} ({source})")
        status = EpistemicStatus.RESPALDADO if verified else EpistemicStatus.INFERIDO
        eid = self.store.add_engram(kind="solucion", level=Level.EPISODICA, concept=topic, content=content,
                                    key=answer, description=problem, importance=0.7 if verified else 0.4,
                                    confidence=0.8 if verified else 0.4, status=status)
        for r in self._similar("solucion", f"{topic} {problem}"):
            if r.engram.id != eid and _norm(r.engram.description) == _norm(problem) \
                    and r.engram.status == EpistemicStatus.REFUTADO:
                self.store.relate(eid, r.engram.id, Relation.CONTRADICCION, confidence=0.9)
        self.store.commit()
        return eid

    def learn_lesson(self, topic: str, problem: str, wrong_answer: str | None, issues: list[str],
                     source: str = "crítica") -> str | None:
        if not issues:
            return None
        content = (f"En «{problem}» una respuesta dio {wrong_answer} y estaba mal: " + "; ".join(issues[:3]))
        for r in self._similar("leccion", f"{topic} {problem} {' '.join(issues)}"):
            if _norm(r.engram.content) == _norm(content):
                self.store.touch(r.engram.id)
                self.store.commit()
                return r.engram.id
        eid = self.store.add_engram(kind="leccion", level=Level.PROCEDIMENTAL, concept=topic, content=content,
                                    key=wrong_answer, description=problem, importance=0.8, confidence=0.7,
                                    status=EpistemicStatus.RESPALDADO)
        self.store.commit()
        return eid

    # --- leer -----------------------------------------------------------------------
    def context(self, topic: str, problem: str, k: int = 3, min_score: float = 0.3) -> Context:
        res = retrieve(self.store, f"{topic} {problem}", k=2 * k, min_score=min_score)
        sols = [r for r in res.items if r.engram.kind == "solucion"
                and r.engram.status != EpistemicStatus.REFUTADO][:k]
        lessons = [r for r in res.items if r.engram.kind == "leccion"][:k]
        parts = []
        if sols:
            parts.append("Conocimiento previo (verificado):\n" + "\n---\n".join(r.engram.content for r in sols))
        if lessons:
            parts.append("Errores a evitar:\n" + "\n".join(f"- {r.engram.content}" for r in lessons))
        return Context("\n\n".join(parts), [r.engram.id for r in sols + lessons])

    def feedback(self, used: list[str], helped: bool) -> None:
        """3. Utilidad: lo que se usó en una respuesta verificada suma un uso; si no ayudó, baja confianza."""
        for eid in used:
            if helped:
                self.store.touch(eid)
            else:
                self.store.db.execute("UPDATE engrams SET confidence = MAX(0.05, confidence - 0.1) WHERE id = ?",
                                      (eid,))
        self.store.commit()

    # --- mantener ---------------------------------------------------------------------
    def optimize(self, **consolidate_kw) -> dict:
        rep = self.store.consolidate(**consolidate_kw)
        rep["ngramas_podados"] = self.store.prune_ngram_stats(self.max_ngram_entries)
        rep.update(self.stats())
        return rep

    def stats(self) -> dict:
        db = self.store.db
        rows = db.execute("SELECT kind, status, COUNT(*) FROM engrams WHERE archived = 0 GROUP BY kind, status")
        by = {}
        for kind, status, n in rows:
            by.setdefault(kind, {})[status] = n
        return {"por_tipo_y_estado": by, **self.store.stats()}
