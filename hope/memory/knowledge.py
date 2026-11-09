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
        from hope.memory.graph import KnowledgeGraph
        self.store = store or MemoryStore(path)
        self.dup_score, self.max_ngram_entries = dup_score, max_ngram_entries
        self.graph = KnowledgeGraph(self.store)

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
        self.graph.link(eid)
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
        self.graph.link(eid)
        return eid

    # --- leer -----------------------------------------------------------------------
    def context(self, topic: str, problem: str, k: int = 3, min_score: float = 0.3) -> Context:
        res = retrieve(self.store, f"{topic} {problem}", k=2 * k, min_score=min_score)
        sols = [r for r in res.items if r.engram.kind == "solucion"
                and r.engram.status != EpistemicStatus.REFUTADO][:k]
        lessons = [r for r in res.items if r.engram.kind == "leccion"][:k]
        used = [r.engram.id for r in sols + lessons]
        related = []
        for nid, _a in self.graph.expand(used, k=k):          # lo relacionado, sin buscar de cero
            e = self.store.get(nid)
            if e.kind in ("nota", "solucion", "leccion"):
                related.append(e)
        parts = []
        if sols:
            parts.append("Conocimiento previo (verificado):\n" + "\n---\n".join(r.engram.content for r in sols))
        if lessons:
            parts.append("Errores a evitar:\n" + "\n".join(f"- {r.engram.content}" for r in lessons))
        if related:
            parts.append("Relacionado:\n" + "\n".join(f"- {e.concept}: {e.content[:300]}" for e in related))
        return Context("\n\n".join(parts), used + [e.id for e in related])

    def feedback(self, used: list[str], helped: bool) -> None:
        """3. Utilidad: lo que se usó en una respuesta verificada suma un uso; si no ayudó, baja confianza."""
        if helped and len(used) > 1:
            self.graph.co_used(used)
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


# ---------------------------------------------------------------------------
# Cuaderno de notas (pedido del usuario): «como una persona que va anotando lo que aprende,
# definiciones y detalles; si no se acuerda, lo busca; con la repetición se vuelve mecánico».
# ---------------------------------------------------------------------------
import time as _time  # noqa: E402

DAY = 86400.0


@dataclass
class Note:
    id: str
    concept: str
    text: str
    lookups: int
    interval_days: float
    due: float
    streak: int
    automated: bool


class Notebook:
    """Notas en la misma base SQLite, con un programa de repaso espaciado (Leitner / SM-2 simplificado).

    Ciclo de una nota:

    1. **Anotar** (``write``): concepto + texto, con procedencia. Las notas del maestro o de una
       fuente quedan OBSERVADO; las propias, INFERIDO. Mismo concepto y mismo texto = no se duplica;
       mismo concepto con texto nuevo = se actualiza y se guarda la versión anterior en el historial.
    2. **Consultar** (``lookup``): cuando Hope no se acuerda, busca por concepto (exacto o parecido).
       Cada consulta cuenta: lo que se consulta seguido es lo que conviene aprender de memoria.
    3. **Repasar** (``due`` + ``practice_texts``): las notas vencidas se convierten en preguntas de
       repaso («¿Qué es X?» → definición) que entran al entrenamiento: así pasan de la base a los pesos.
    4. **Comprobar sin mirar** (``record_recall``): si Hope responde bien sin consultar, el intervalo
       se duplica (1, 2, 4, 8… días); si falla, vuelve a 1 día. Tras ``automate_after`` aciertos
       seguidos la nota queda **automatizada**: ya no hace falta consultarla (sigue guardada).
    """

    def __init__(self, kb: KnowledgeBase, *, automate_after: int = 4, clock=_time.time):
        self.kb, self.store, self.clock = kb, kb.store, clock
        self.automate_after = automate_after
        self.store.db.execute(
            "CREATE TABLE IF NOT EXISTS notas_repaso (id TEXT PRIMARY KEY, concepto TEXT, consultas INTEGER, "
            "intervalo REAL, proximo REAL, racha INTEGER, automatizada INTEGER)")
        self.store.db.commit()

    @staticmethod
    def _key(concept: str) -> str:
        return " ".join(concept.lower().split())

    def _row(self, eid: str) -> Note:
        e = self.store.get(eid)
        r = self.store.db.execute("SELECT consultas, intervalo, proximo, racha, automatizada FROM notas_repaso "
                                  "WHERE id = ?", (eid,)).fetchone()
        return Note(eid, e.concept, e.content, r[0], r[1], r[2], r[3], bool(r[4]))

    def _find(self, concept: str) -> str | None:
        r = self.store.db.execute("SELECT id FROM notas_repaso WHERE concepto = ?", (self._key(concept),)).fetchone()
        return r[0] if r else None

    def write(self, concept: str, text: str, *, source: str = "hope", verified: bool = False) -> str:
        eid = self._find(concept)
        if eid is not None:
            old = self.store.get(eid)
            if " ".join(old.content.split()) == " ".join(text.split()):
                self.store.touch(eid)
            else:                                     # nueva versión: la anterior queda en el historial
                # El vector de búsqueda se recalcula con el texto nuevo (si no, la nota se seguiría
                # encontrando por lo que decía antes y no por lo que dice ahora).
                vec = self.store.embedder.embed([f"{old.concept}\n{text}"])[0]
                self.store.db.execute("UPDATE engrams SET content = ?, updated = ?, vector = ? WHERE id = ?",
                                      (text, self.clock(), vec.tobytes(), eid))
                if eid in self.store._ids:
                    self.store._matrix[self.store._ids.index(eid)] = vec
                self.store._history(eid, {"op": "version", "anterior": old.content, "fuente": source})
            self.store.commit()
            return eid
        status = EpistemicStatus.OBSERVADO if verified else EpistemicStatus.INFERIDO
        eid = self.store.add_engram(kind="nota", level=Level.SEMANTICA, concept=concept, content=text,
                                    key=self._key(concept), description=f"nota ({source})", importance=0.6,
                                    confidence=0.8 if verified else 0.5, status=status)
        self.store.db.execute("INSERT INTO notas_repaso VALUES (?,?,?,?,?,?,?)",
                              (eid, self._key(concept), 0, 1.0, self.clock() + DAY, 0, 0))
        self.store.commit()
        self.kb.graph.link(eid)
        return eid

    def lookup(self, concept: str) -> Note | None:
        eid = self._find(concept)
        if eid is None:                               # no exacto: la nota más parecida
            hits = [r for r in retrieve(self.store, concept, k=5, min_score=0.2).items if r.engram.kind == "nota"]
            if not hits:
                return None
            eid = hits[0].engram.id
        self.store.touch(eid)
        self.store.db.execute("UPDATE notas_repaso SET consultas = consultas + 1 WHERE id = ?", (eid,))
        self.store.commit()
        return self._row(eid)

    def due(self, limit: int = 100, *, include_automated: bool = False) -> list[Note]:
        q = ("SELECT id FROM notas_repaso WHERE proximo <= ?" + ("" if include_automated else " AND automatizada = 0")
             + " ORDER BY consultas DESC, proximo ASC LIMIT ?")
        return [self._row(r[0]) for r in self.store.db.execute(q, (self.clock(), limit))]

    @staticmethod
    def question(note: Note) -> str:
        return f"¿Qué es «{note.concept}»? Respondé sin consultar notas.\n"

    def practice_texts(self, notes: list[Note]) -> list[str]:
        """Textos de repaso para el entrenamiento (de la base a los pesos)."""
        return [self.question(n) + n.text for n in notes]

    def record_recall(self, note_id: str, recalled: bool) -> Note:
        n = self._row(note_id)
        if recalled:
            streak, interval = n.streak + 1, n.interval_days * 2
        else:
            streak, interval = 0, 1.0
        automated = streak >= self.automate_after
        self.store.db.execute("UPDATE notas_repaso SET racha = ?, intervalo = ?, proximo = ?, automatizada = ? "
                              "WHERE id = ?", (streak, interval, self.clock() + interval * DAY, int(automated),
                                               note_id))
        if automated:
            self.store.update_status(note_id, EpistemicStatus.VALIDADO_CONDICIONAL,
                                     f"recordado sin consultar {streak} veces seguidas")
        self.store.commit()
        return self._row(note_id)

    def stats(self) -> dict:
        one = lambda q: self.store.db.execute(q).fetchone()[0]  # noqa: E731
        return {"notas": one("SELECT COUNT(*) FROM notas_repaso"),
                "automatizadas": one("SELECT COUNT(*) FROM notas_repaso WHERE automatizada = 1"),
                "consultas": one("SELECT COALESCE(SUM(consultas), 0) FROM notas_repaso")}
