"""Almacén de memoria de Hope: AHMC + MDD v0 (glosario §2; MDD §2–§9, §14).

Un único almacén SQLite con:

* **documentos** — archivo de originales (nivel «archivo» de AHMC). Nunca se
  reemplazan por resúmenes: para números, fórmulas y unidades se recupera el
  texto exacto por documento y posición (ASG §12).
* **engramas** — unidades de memoria con el esquema de MDD §4.2: concepto,
  contenido, procedencia, importancia, uso, confianza, estado epistémico,
  condiciones, excepciones e historial. Cada engrama tiene un nivel AHMC
  (episódica, semántica, procedimental, archivo) y un vector para buscar por similitud.
* **relaciones** — índice asociativo compartido con tipos de MDD §5. La
  causalidad siempre se marca como hipotética o verificada.
* **n-gramas** — índice de trigramas de palabras sobre los documentos, para
  recuperación textual exacta (MDD §3), con estadísticas de frecuencia y
  utilidad para podar lo que no sirve.

La reorganización no borra originales (MDD §7): retirar = archivar.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from hope.epistemic import EpistemicStatus
from hope.memory.embed import HashingEmbedder


class Level(str, Enum):
    EPISODICA = "EPISODICA"
    SEMANTICA = "SEMANTICA"
    PROCEDIMENTAL = "PROCEDIMENTAL"
    ARCHIVO = "ARCHIVO"


class Relation(str, Enum):
    SIMILARIDAD = "SIMILARIDAD"
    EQUIVALENCIA = "EQUIVALENCIA"
    DEPENDENCIA = "DEPENDENCIA"
    CAUSALIDAD_HIPOTETICA = "CAUSALIDAD_HIPOTETICA"
    CAUSALIDAD_VERIFICADA = "CAUSALIDAD_VERIFICADA"
    SECUENCIA = "SECUENCIA"
    CONTRADICCION = "CONTRADICCION"
    COMPLEMENTARIEDAD = "COMPLEMENTARIEDAD"
    GENERALIZACION = "GENERALIZACION"
    ESPECIALIZACION = "ESPECIALIZACION"
    APLICACION = "APLICACION"
    EXCEPCION = "EXCEPCION"


# Estados que no deben devolverse como conocimiento salvo que se pida.
EXCLUDED_BY_DEFAULT = {EpistemicStatus.REFUTADO}

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, sha256 TEXT UNIQUE, source TEXT, text TEXT, license TEXT, created REAL);
CREATE TABLE IF NOT EXISTS engrams (
  id TEXT PRIMARY KEY, kind TEXT, level TEXT, concept TEXT, key TEXT, content TEXT, description TEXT,
  doc_id TEXT, offset INTEGER, length INTEGER, importance REAL, use_count INTEGER, confidence REAL,
  status TEXT, conditions TEXT, exceptions TEXT, created REAL, updated REAL, last_used REAL,
  archived INTEGER DEFAULT 0, vector BLOB);
CREATE INDEX IF NOT EXISTS engrams_key ON engrams(key);
CREATE TABLE IF NOT EXISTS engram_history (engram_id TEXT, ts REAL, change TEXT);
CREATE TABLE IF NOT EXISTS relations (
  id TEXT PRIMARY KEY, src TEXT, dst TEXT, type TEXT, confidence REAL, status TEXT, provenance TEXT, created REAL);
CREATE INDEX IF NOT EXISTS relations_src ON relations(src);
CREATE INDEX IF NOT EXISTS relations_dst ON relations(dst);
CREATE TABLE IF NOT EXISTS ngrams (h INTEGER, doc_id TEXT, offset INTEGER);
CREATE INDEX IF NOT EXISTS ngrams_h ON ngrams(h);
CREATE TABLE IF NOT EXISTS ngram_stats (h INTEGER PRIMARY KEY, gram TEXT, count INTEGER, hits INTEGER);
"""

WORD_RE = re.compile(r"\w+", re.UNICODE)


def _gram_hash(gram: str) -> int:
    return int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "little", signed=True)


def word_trigrams(text: str) -> list[tuple[str, int]]:
    """Trigramas de palabras en minúscula con la posición (en caracteres) del primero."""
    words = [(m.group().lower(), m.start()) for m in WORD_RE.finditer(text)]
    return [(" ".join(w for w, _ in words[i:i + 3]), words[i][1]) for i in range(len(words) - 2)]


@dataclass
class Engram:
    id: str
    kind: str
    level: Level
    concept: str
    content: str
    key: str | None = None
    description: str = ""
    doc_id: str | None = None
    offset: int | None = None
    length: int | None = None
    importance: float = 0.5
    use_count: int = 0
    confidence: float = 0.5
    status: EpistemicStatus = EpistemicStatus.OBSERVADO
    conditions: list = field(default_factory=list)
    exceptions: list = field(default_factory=list)
    archived: bool = False


@dataclass
class FactResult:
    engram: Engram
    contradicts: list[str]          # IDs de engramas con el mismo dato y otro valor


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:", embedder=None):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)
        self.embedder = embedder or HashingEmbedder()
        self._ids: list[str] = []
        self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._load_vectors()

    # --- vectores en RAM -----------------------------------------------------
    def _load_vectors(self) -> None:
        rows = self.db.execute("SELECT id, vector FROM engrams WHERE archived = 0 AND vector IS NOT NULL").fetchall()
        self._ids = [r[0] for r in rows]
        self._matrix = (np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
                        if rows else np.zeros((0, self.embedder.dim), dtype=np.float32))

    def _add_vector(self, eid: str, vec: np.ndarray) -> None:
        self._ids.append(eid)
        self._matrix = np.vstack([self._matrix, vec[None]])

    # --- documentos ----------------------------------------------------------
    def add_document(self, text: str, source: str, *, license: str | None = None,
                     passage_chars: int = 600, index_ngrams: bool = True) -> str:
        """Archiva un documento original, lo indexa por n-gramas y crea engramas de pasajes."""
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = self.db.execute("SELECT id FROM documents WHERE sha256 = ?", (sha,)).fetchone()
        if row:
            return row[0]                                   # deduplicación exacta
        doc_id = f"doc_{sha[:16]}"
        self.db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?)",
                        (doc_id, sha, source, text, license, time.time()))
        if index_ngrams:
            grams = word_trigrams(text)
            self.db.executemany("INSERT INTO ngrams VALUES (?,?,?)",
                                [(_gram_hash(g), doc_id, off) for g, off in grams])
            for g, _ in grams:
                self.db.execute("INSERT INTO ngram_stats VALUES (?,?,1,0) ON CONFLICT(h) DO UPDATE SET count = count + 1",
                                (_gram_hash(g), g))
        for start, end in self._passages(text, passage_chars):
            self.add_engram(kind="pasaje", level=Level.ARCHIVO, concept=source,
                            content=text[start:end], doc_id=doc_id, offset=start, length=end - start,
                            status=EpistemicStatus.OBSERVADO, importance=0.3)
        self.db.commit()
        return doc_id

    @staticmethod
    def _passages(text: str, size: int) -> list[tuple[int, int]]:
        """Cortes en límites de párrafo o de oración, sin superar ``size`` caracteres si se puede."""
        out, start = [], 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                cut = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
                if cut > start + size // 3:
                    end = cut + 1
            out.append((start, end))
            start = end
        return out

    def source_text(self, doc_id: str, offset: int = 0, length: int | None = None) -> str:
        """Texto **original exacto** (no un resumen)."""
        row = self.db.execute("SELECT text FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            raise KeyError(doc_id)
        return row[0][offset:offset + length if length is not None else None]

    # --- engramas --------------------------------------------------------------
    def add_engram(self, *, kind: str, level: Level, concept: str, content: str, key: str | None = None,
                   description: str = "", doc_id: str | None = None, offset: int | None = None,
                   length: int | None = None, importance: float = 0.5, confidence: float = 0.5,
                   status: EpistemicStatus = EpistemicStatus.OBSERVADO, conditions: list | None = None,
                   exceptions: list | None = None, embed: bool = True) -> str:
        eid = f"eng_{uuid.uuid4().hex[:12]}"
        now = time.time()
        vec = self.embedder.embed([f"{concept}\n{content}"])[0] if embed else None
        self.db.execute(
            "INSERT INTO engrams VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)",
            (eid, kind, Level(level).value, concept, key, content, description, doc_id, offset, length,
             importance, 0, confidence, EpistemicStatus(status).value, json.dumps(conditions or []),
             json.dumps(exceptions or []), now, now, None, vec.tobytes() if vec is not None else None))
        self._history(eid, {"op": "create", "status": EpistemicStatus(status).value})
        if vec is not None:
            self._add_vector(eid, vec)
        return eid

    def get(self, eid: str) -> Engram:
        r = self.db.execute("SELECT id, kind, level, concept, content, key, description, doc_id, offset, length, "
                            "importance, use_count, confidence, status, conditions, exceptions, archived "
                            "FROM engrams WHERE id = ?", (eid,)).fetchone()
        if r is None:
            raise KeyError(eid)
        return Engram(r[0], r[1], Level(r[2]), r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12],
                      EpistemicStatus(r[13]), json.loads(r[14]), json.loads(r[15]), bool(r[16]))

    def update_status(self, eid: str, status: EpistemicStatus, evidence: str) -> None:
        self.db.execute("UPDATE engrams SET status = ?, updated = ? WHERE id = ?",
                        (status.value, time.time(), eid))
        self._history(eid, {"op": "status", "status": status.value, "evidence": evidence})
        self.db.commit()

    def touch(self, eid: str) -> None:
        self.db.execute("UPDATE engrams SET use_count = use_count + 1, last_used = ? WHERE id = ?", (time.time(), eid))

    def history(self, eid: str) -> list[dict]:
        return [json.loads(r[0]) for r in
                self.db.execute("SELECT change FROM engram_history WHERE engram_id = ? ORDER BY ts", (eid,))]

    def _history(self, eid: str, change: dict) -> None:
        self.db.execute("INSERT INTO engram_history VALUES (?,?,?)", (eid, time.time(), json.dumps(change)))

    # --- hechos estructurados y contradicciones ---------------------------------
    def remember_fact(self, subject: str, attribute: str, value: str, *, source: str,
                      status: EpistemicStatus = EpistemicStatus.OBSERVADO, confidence: float = 0.7,
                      conditions: list | None = None, embed: bool = True, commit: bool = True) -> FactResult:
        """Guarda «sujeto · atributo = valor». Si ya hay otro valor para el mismo dato,
        **no lo sobrescribe**: guarda ambos y los relaciona como contradicción (MDD §14)."""
        key = f"{subject.strip().lower()}|{attribute.strip().lower()}"
        others = self.db.execute("SELECT id, content, status FROM engrams WHERE key = ? AND archived = 0",
                                 (key,)).fetchall()
        for oid, content, _ in others:
            if content == value:
                self.touch(oid)
                self.db.commit()
                return FactResult(self.get(oid), [])
        eid = self.add_engram(kind="hecho", level=Level.SEMANTICA, concept=f"{subject} · {attribute}",
                              content=value, key=key, description=f"fuente: {source}", status=status,
                              confidence=confidence, conditions=conditions, embed=embed)
        contradicts = [oid for oid, _, st in others if EpistemicStatus(st) not in EXCLUDED_BY_DEFAULT]
        for oid in contradicts:
            self.relate(eid, oid, Relation.CONTRADICCION, confidence=1.0, provenance="remember_fact")
        if commit:
            self.db.commit()
        return FactResult(self.get(eid), contradicts)

    def lookup(self, subject: str, attribute: str, include_refuted: bool = False) -> list[Engram]:
        key = f"{subject.strip().lower()}|{attribute.strip().lower()}"
        ids = [r[0] for r in self.db.execute("SELECT id FROM engrams WHERE key = ? AND archived = 0", (key,))]
        engs = [self.get(i) for i in ids]
        if not include_refuted:
            engs = [e for e in engs if e.status not in EXCLUDED_BY_DEFAULT]
        for e in engs:
            self.touch(e.id)
        self.db.commit()
        return sorted(engs, key=lambda e: e.confidence, reverse=True)

    def contradictions(self) -> list[tuple[str, str]]:
        return [(r[0], r[1]) for r in
                self.db.execute("SELECT src, dst FROM relations WHERE type = ?", (Relation.CONTRADICCION.value,))]

    # --- relaciones --------------------------------------------------------------
    def relate(self, src: str, dst: str, rtype: Relation, *, confidence: float = 0.5,
               status: EpistemicStatus = EpistemicStatus.INFERIDO, provenance: str = "") -> str:
        if rtype == Relation.CAUSALIDAD_VERIFICADA and status not in (EpistemicStatus.VALIDADO,
                                                                      EpistemicStatus.VALIDADO_CONDICIONAL):
            raise ValueError("una causalidad solo es «verificada» con estado VALIDADO (MDD §5)")
        rid = f"rel_{uuid.uuid4().hex[:12]}"
        self.db.execute("INSERT INTO relations VALUES (?,?,?,?,?,?,?,?)",
                        (rid, src, dst, Relation(rtype).value, confidence, status.value, provenance, time.time()))
        return rid

    def neighbors(self, eid: str, types: set[Relation] | None = None) -> list[tuple[str, Relation, float]]:
        rows = self.db.execute("SELECT dst, type, confidence FROM relations WHERE src = ? "
                               "UNION ALL SELECT src, type, confidence FROM relations WHERE dst = ?", (eid, eid))
        out = [(r[0], Relation(r[1]), r[2]) for r in rows]
        return [o for o in out if types is None or o[1] in types]

    # --- búsquedas básicas -------------------------------------------------------
    def vector_search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        q = self.embedder.embed([query])[0]
        sims = self._matrix @ q
        top = np.argsort(-sims)[:k]
        return [(self._ids[i], float(sims[i])) for i in top]

    def ngram_search(self, query: str, k: int = 10) -> list[tuple[str, int, int]]:
        """Documentos con más trigramas en común: (doc_id, coincidencias, posición de la primera)."""
        grams = [_gram_hash(g) for g, _ in word_trigrams(query)]
        if not grams:
            return []
        q = ",".join("?" * len(grams))
        rows = self.db.execute(f"SELECT doc_id, COUNT(DISTINCT h), MIN(offset) FROM ngrams WHERE h IN ({q}) "
                               f"GROUP BY doc_id ORDER BY 2 DESC LIMIT ?", (*grams, k)).fetchall()
        self.db.executemany("UPDATE ngram_stats SET hits = hits + 1 WHERE h = ?", [(g,) for g in grams])
        return [(r[0], r[1], r[2]) for r in rows]

    def passage_at(self, doc_id: str, offset: int) -> str | None:
        r = self.db.execute("SELECT id FROM engrams WHERE doc_id = ? AND offset <= ? AND offset + length > ? "
                            "AND kind = 'pasaje'", (doc_id, offset, offset)).fetchone()
        return r[0] if r else None

    # --- mantenimiento -----------------------------------------------------------
    def prune_ngram_stats(self, max_entries: int) -> int:
        """Poda estadísticas de n-gramas por utilidad (aciertos y frecuencia); el índice de documentos queda."""
        n = self.db.execute("SELECT COUNT(*) FROM ngram_stats").fetchone()[0]
        if n <= max_entries:
            return 0
        self.db.execute("DELETE FROM ngram_stats WHERE h IN (SELECT h FROM ngram_stats "
                        "ORDER BY hits * 4 + count ASC LIMIT ?)", (n - max_entries,))
        self.db.commit()
        return n - max_entries

    def consolidate(self, *, min_uses: int = 3, archive_after_s: float = 30 * 86400,
                    archive_below_importance: float = 0.2) -> dict:
        """MDD §8: episódico usado y respaldado → semántico; poco usado y poco importante → archivado.

        La frecuencia no es el único criterio: lo importante o verificado no se archiva aunque se use poco.
        """
        strong = (EpistemicStatus.RESPALDADO.value, EpistemicStatus.VALIDADO.value,
                  EpistemicStatus.VALIDADO_CONDICIONAL.value)
        q = ",".join("?" * len(strong))
        promoted = self.db.execute(
            f"UPDATE engrams SET level = ? WHERE level = ? AND use_count >= ? AND status IN ({q}) AND archived = 0",
            (Level.SEMANTICA.value, Level.EPISODICA.value, min_uses, *strong)).rowcount
        cutoff = time.time() - archive_after_s
        archived = self.db.execute(
            f"UPDATE engrams SET archived = 1 WHERE archived = 0 AND kind != 'pasaje' AND importance < ? "
            f"AND status NOT IN ({q}) AND COALESCE(last_used, created) < ?",
            (archive_below_importance, *strong, cutoff)).rowcount
        self.db.commit()
        if archived:
            self._load_vectors()
        return {"promoted_to_semantic": promoted, "archived": archived}

    def stats(self) -> dict:
        one = lambda q: self.db.execute(q).fetchone()[0]  # noqa: E731
        return {"documents": one("SELECT COUNT(*) FROM documents"),
                "engrams": one("SELECT COUNT(*) FROM engrams WHERE archived = 0"),
                "archived": one("SELECT COUNT(*) FROM engrams WHERE archived = 1"),
                "relations": one("SELECT COUNT(*) FROM relations"),
                "ngram_postings": one("SELECT COUNT(*) FROM ngrams")}

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()
