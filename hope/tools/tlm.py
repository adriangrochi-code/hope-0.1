"""TLM — memoria de aprendizaje sobre herramientas (Herramientas §3.4; docs/HERRAMIENTAS_HN1.md §3 nivel 7).

Aprende **qué herramientas le sirven a Hope para cada tema**, a partir de las
llamadas registradas: cuántas veces se usó cada una, cuántas funcionaron y
cuántas veces la respuesta final fue correcta. Con eso sugiere herramientas en
el prompt («Herramientas útiles para la ley de Ohm: unidades, calc»).

La utilidad se estima con un suavizado de Laplace para no sobrevalorar lo
visto pocas veces: (útiles + 1) / (usos + 2) × (funcionaron + 1) / (usos + 2).

También registra **composiciones** (HTO, ``hope/tools/compose.py``): por tema,
cada tubería «h1 > h2 > h3» acumula usos, éxitos, resultados exactos y tiempo,
para preferir las composiciones rápidas y exactas.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

COMPOSITIONS_KEY = "__composiciones__"


class ToolUsageMemory:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.stats: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        self.compositions: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.compositions.update(data.pop(COMPOSITIONS_KEY, {}))
            self.stats.update(data)

    def record(self, calls, topic: str, answer_correct: bool) -> None:
        for c in calls:
            s = self.stats[topic].setdefault(c.name, {"uses": 0, "ok": 0, "useful": 0, "seconds": 0.0})
            s["uses"] += 1
            s["ok"] += int(c.ok)
            s["useful"] += int(c.ok and answer_correct)
            s["seconds"] += c.seconds
        self.save()

    def utility(self, topic: str, tool: str) -> float:
        s = self.stats.get(topic, {}).get(tool)
        if not s:
            return 0.0
        n = s["uses"]
        return (s["useful"] + 1) / (n + 2) * (s["ok"] + 1) / (n + 2)

    def suggest(self, topic: str, k: int = 3, min_utility: float = 0.3) -> list[str]:
        ranked = sorted(self.stats.get(topic, {}), key=lambda t: -self.utility(topic, t))
        return [t for t in ranked if self.utility(topic, t) >= min_utility][:k]

    def hint(self, topic: str) -> str:
        tools = self.suggest(topic)
        return f"Herramientas útiles para «{topic}»: {', '.join(tools)}" if tools else ""

    # --- composiciones (HTO) -----------------------------------------------------
    def record_composition(self, chain: list[str], topic: str, *, ok: bool, seconds: float,
                           exact: bool | None = None, step_seconds: list[float] | None = None) -> None:
        key = " > ".join(chain)
        s = self.compositions[topic].setdefault(key, {"uses": 0, "ok": 0, "exact": 0, "checked": 0,
                                                     "seconds": 0.0, "step_seconds": [0.0] * len(chain)})
        s["uses"] += 1
        s["ok"] += int(ok)
        if exact is not None:
            s["checked"] += 1
            s["exact"] += int(bool(exact) and ok)
        s["seconds"] += seconds
        for i, t in enumerate(step_seconds or []):
            if i < len(s["step_seconds"]):
                s["step_seconds"][i] += t
        self.save()

    def composition_stats(self, topic: str, chain: list[str] | str) -> dict | None:
        key = chain if isinstance(chain, str) else " > ".join(chain)
        s = self.compositions.get(topic, {}).get(key)
        if not s:
            return None
        n = s["uses"]
        return {**s, "success_rate": s["ok"] / n, "mean_seconds": s["seconds"] / n,
                "exact_rate": s["exact"] / s["checked"] if s["checked"] else None}

    def composition_utility(self, topic: str, chain: list[str] | str, time_scale: float = 1.0) -> float:
        """Exactitud y éxito suavizados (Laplace), penalizados por el tiempo medio: rápida y exacta = alta."""
        s = self.composition_stats(topic, chain)
        if not s:
            return 0.0
        n = s["uses"]
        exact = (s["exact"] + 1) / (s["checked"] + 2) if s["checked"] else 0.5
        return exact * (s["ok"] + 1) / (n + 2) / (1.0 + s["mean_seconds"] / time_scale)

    def best_compositions(self, topic: str, k: int = 3) -> list[str]:
        return sorted(self.compositions.get(topic, {}), key=lambda c: -self.composition_utility(topic, c))[:k]

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {**self.stats, COMPOSITIONS_KEY: self.compositions} if self.compositions else self.stats
            self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
