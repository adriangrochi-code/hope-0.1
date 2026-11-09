"""Grafo de conocimiento: relaciones entre datos ya calculadas, para no buscar de cero.

Pedido del usuario: «que encuentre patrones y correlaciones entre datos; una red de grafos para
que muchos datos queden relacionados y se encuentren rápido, sin buscar de cero».

Usa la tabla de relaciones de ``MemoryStore`` (tipadas, con confianza y estado epistémico):

* **Enlaces automáticos al guardar** (``link``):
  - SIMILARIDAD con los recuerdos más parecidos (vectores + n-gramas), con su puntaje;
  - DEPENDENCIA cuando el texto **menciona** el concepto de otra nota («la potencia usa la
    *ley de Ohm*»): así una definición trae las definiciones que necesita.
* **Correlación por uso** (``co_used``, aprendizaje hebbiano): los recuerdos que se usan juntos
  en una respuesta verificada refuerzan su enlace COMPLEMENTARIEDAD («lo que se usa junto, se une»).
  Los enlaces que nunca se refuerzan se debilitan con ``decay`` y los muy débiles se podan.
* **Recuperación por vecindad** (``expand``): desde lo encontrado se propaga activación por el
  grafo (1–2 saltos, multiplicando confianzas) y se traen los vecinos más fuertes. Es mucho más
  barato que otra búsqueda y trae lo **relacionado**, no solo lo **parecido**.
* **Patrones** (``patterns``): conceptos centrales (los más conectados), grupos de conocimiento
  (componentes conexas por enlaces fuertes) y pares que siempre se usan juntos.

Para datos numéricos (tablas), la correlación estadística está en la herramienta
``⟦correlacion: csv⟧`` (``hope/tools/eng_tools.py``).
"""

from __future__ import annotations

import time
from collections import defaultdict

from hope.epistemic import EpistemicStatus
from hope.memory.retrieval import retrieve
from hope.memory.store import MemoryStore, Relation


class KnowledgeGraph:
    def __init__(self, store: MemoryStore, *, similar_k: int = 4, min_similarity: float = 0.35):
        self.store, self.similar_k, self.min_similarity = store, similar_k, min_similarity

    # --- construir ---------------------------------------------------------------------
    def _edge(self, a: str, b: str, rtype: Relation) -> tuple[str, float] | None:
        r = self.store.db.execute("SELECT id, confidence FROM relations WHERE type = ? AND "
                                  "((src = ? AND dst = ?) OR (src = ? AND dst = ?))",
                                  (rtype.value, a, b, b, a)).fetchone()
        return (r[0], r[1]) if r else None

    def _strengthen(self, a: str, b: str, rtype: Relation, amount: float, provenance: str) -> None:
        e = self._edge(a, b, rtype)
        if e is None:
            self.store.relate(a, b, rtype, confidence=min(1.0, amount), provenance=provenance)
        else:
            self.store.db.execute("UPDATE relations SET confidence = ? WHERE id = ?",
                                  (min(1.0, e[1] + amount * (1 - e[1])), e[0]))

    def link(self, eid: str) -> int:
        """Enlaza un recuerdo recién guardado con lo parecido y con los conceptos que menciona."""
        e = self.store.get(eid)
        n = 0
        for r in retrieve(self.store, f"{e.concept} {e.content}", k=self.similar_k + 1, min_score=0.0).items:
            if r.engram.id != eid and r.score >= self.min_similarity and self._edge(eid, r.engram.id,
                                                                                  Relation.SIMILARIDAD) is None:
                self.store.relate(eid, r.engram.id, Relation.SIMILARIDAD, confidence=min(1.0, r.score),
                                  provenance="similitud")
                n += 1
        text = e.content.lower()
        for oid, concept in self.store.db.execute(
                "SELECT id, concept FROM engrams WHERE archived = 0 AND id != ? AND length(concept) >= 4", (eid,)):
            if concept and concept.lower() in text and self._edge(eid, oid, Relation.DEPENDENCIA) is None:
                self.store.relate(eid, oid, Relation.DEPENDENCIA, confidence=0.7, provenance="mención")
                n += 1
        self.store.commit()
        return n

    def co_used(self, ids: list[str], amount: float = 0.2) -> None:
        """Refuerzo hebbiano entre los recuerdos usados juntos en una respuesta verificada."""
        ids = list(dict.fromkeys(ids))
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                self._strengthen(a, b, Relation.COMPLEMENTARIEDAD, amount, "uso conjunto")
        self.store.commit()

    def decay(self, factor: float = 0.95, prune_below: float = 0.05) -> dict:
        """Lo que no se refuerza se debilita; lo muy débil se poda (solo enlaces aprendidos por uso)."""
        db = self.store.db
        db.execute("UPDATE relations SET confidence = confidence * ? WHERE type = ?",
                   (factor, Relation.COMPLEMENTARIEDAD.value))
        pruned = db.execute("DELETE FROM relations WHERE type = ? AND confidence < ?",
                            (Relation.COMPLEMENTARIEDAD.value, prune_below)).rowcount
        self.store.commit()
        return {"podados": pruned}

    # --- usar ----------------------------------------------------------------------------
    def expand(self, seeds: list[str], *, hops: int = 2, k: int = 5, min_activation: float = 0.15) -> list[tuple[str, float]]:
        """Propagación de activación desde ``seeds``: devuelve los vecinos más activados (sin las semillas)."""
        act = {s: 1.0 for s in seeds}
        frontier = dict(act)
        for _ in range(hops):
            nxt: dict[str, float] = defaultdict(float)
            for node, a in frontier.items():
                for nb, rtype, conf in self.store.neighbors(node):
                    if rtype == Relation.CONTRADICCION:
                        continue
                    v = a * conf
                    if v >= min_activation and v > act.get(nb, 0.0):
                        nxt[nb] = max(nxt[nb], v)
            for nb, v in nxt.items():
                act[nb] = max(act.get(nb, 0.0), v)
            frontier = nxt
        alive = []
        for nid, v in sorted(act.items(), key=lambda kv: -kv[1]):
            if nid in seeds:
                continue
            try:
                if self.store.get(nid).status == EpistemicStatus.REFUTADO:
                    continue
            except KeyError:
                continue
            alive.append((nid, v))
        return alive[:k]

    def patterns(self, strong: float = 0.5, top: int = 10) -> dict:
        rows = list(self.store.db.execute("SELECT src, dst, type, confidence FROM relations"))
        degree: dict[str, float] = defaultdict(float)
        parent: dict[str, str] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        pairs = []
        for src, dst, rtype, conf in rows:
            if rtype == Relation.CONTRADICCION.value:
                continue
            degree[src] += conf
            degree[dst] += conf
            if conf >= strong:
                parent[find(src)] = find(dst)
            if rtype == Relation.COMPLEMENTARIEDAD.value:
                pairs.append((conf, src, dst))
        groups: dict[str, list[str]] = defaultdict(list)
        for node in parent:
            groups[find(node)].append(node)

        def name(eid):
            try:
                return self.store.get(eid).concept
            except KeyError:
                return eid
        return {
            "centrales": [(name(e), round(d, 2)) for e, d in sorted(degree.items(), key=lambda kv: -kv[1])[:top]],
            "grupos": sorted(([name(e) for e in g] for g in groups.values() if len(g) > 1), key=len, reverse=True),
            "usados_juntos": [(name(a), name(b), round(c, 2)) for c, a, b in sorted(pairs, reverse=True)[:top]],
            "enlaces": len(rows), "calculado": time.time(),
        }
