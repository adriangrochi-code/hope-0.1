"""Recuperación contextual híbrida (MDD §9; AHMC).

Combina varias señales en lugar de buscar solo coincidencias textuales:

* similitud vectorial (engramas y pasajes);
* coincidencias de trigramas de palabras (texto casi literal);
* expansión asociativa de un salto por el índice de relaciones;
* importancia y estado epistémico (lo validado pesa más, lo refutado se excluye);
* contradicciones conocidas: se devuelven **marcadas**, nunca se ocultan.

Cada resultado incluye por qué se recuperó y la referencia al original exacto.
Si nada supera el umbral, el resultado lo dice: Hope debe reconocer que no
encontró la información (UAC) en lugar de inventarla.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hope.epistemic import EpistemicStatus
from hope.memory.store import EXCLUDED_BY_DEFAULT, Engram, MemoryStore, Relation

STATUS_WEIGHT = {
    EpistemicStatus.VALIDADO: 1.0, EpistemicStatus.VALIDADO_CONDICIONAL: 0.9,
    EpistemicStatus.RESPALDADO: 0.8, EpistemicStatus.OBSERVADO: 0.8, EpistemicStatus.PROBADO: 0.6,
    EpistemicStatus.INFERIDO: 0.5, EpistemicStatus.SIMULADO: 0.4, EpistemicStatus.HIPOTESIS: 0.4,
    EpistemicStatus.GENERADO_EN_SUENO: 0.3, EpistemicStatus.INCONCLUSO: 0.3, EpistemicStatus.REFUTADO: 0.0,
}


@dataclass
class Retrieved:
    engram: Engram
    score: float
    reasons: list[str] = field(default_factory=list)
    contradicted_by: list[str] = field(default_factory=list)

    @property
    def source(self) -> dict | None:
        e = self.engram
        return None if e.doc_id is None else {"doc_id": e.doc_id, "offset": e.offset, "length": e.length}


@dataclass
class RetrievalResult:
    items: list[Retrieved]
    found: bool

    def texts(self) -> list[str]:
        return [r.engram.content for r in self.items]


@dataclass
class Weights:
    vector: float = 1.0
    ngram: float = 1.0
    association: float = 0.3
    importance: float = 0.2


def retrieve(store: MemoryStore, query: str, *, k: int = 5, min_score: float = 0.25,
             weights: Weights = Weights(), include_refuted: bool = False) -> RetrievalResult:
    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    def add(eid: str, s: float, why: str) -> None:
        scores[eid] = scores.get(eid, 0.0) + s
        reasons.setdefault(eid, []).append(why)

    for eid, sim in store.vector_search(query, k=4 * k):
        if sim > 0:
            add(eid, weights.vector * sim, f"similitud {sim:.2f}")

    n_q = max(len(query.split()) - 2, 1)
    for doc_id, hits, offset in store.ngram_search(query, k=k):
        pid = store.passage_at(doc_id, offset)
        if pid:
            add(pid, weights.ngram * min(hits / n_q, 1.0), f"{hits} trigramas en común")

    for eid in list(scores)[: 2 * k]:
        for nid, rtype, conf in store.neighbors(eid):
            if rtype != Relation.CONTRADICCION:
                add(nid, weights.association * conf * scores[eid], f"asociado ({rtype.value}) a {eid}")

    items = []
    for eid, s in scores.items():
        e = store.get(eid)
        if e.archived or (e.status in EXCLUDED_BY_DEFAULT and not include_refuted):
            continue
        s = s * STATUS_WEIGHT[e.status] + weights.importance * e.importance
        contra = [n for n, t, _ in store.neighbors(eid, {Relation.CONTRADICCION})]
        items.append(Retrieved(e, s, reasons[eid], contra))
    items.sort(key=lambda r: r.score, reverse=True)
    items = [r for r in items if r.score >= min_score][:k]
    for r in items:
        store.touch(r.engram.id)
    store.commit()
    return RetrievalResult(items, found=bool(items))
