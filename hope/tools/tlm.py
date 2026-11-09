"""TLM — memoria de aprendizaje sobre herramientas (Herramientas §3.4; docs/HERRAMIENTAS_HN1.md §3 nivel 7).

Aprende **qué herramientas le sirven a Hope para cada tema**, a partir de las
llamadas registradas: cuántas veces se usó cada una, cuántas funcionaron y
cuántas veces la respuesta final fue correcta. Con eso sugiere herramientas en
el prompt («Herramientas útiles para la ley de Ohm: unidades, calc»).

La utilidad se estima con un suavizado de Laplace para no sobrevalorar lo
visto pocas veces: (útiles + 1) / (usos + 2) × (funcionaron + 1) / (usos + 2).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


class ToolUsageMemory:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.stats: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        if self.path and self.path.exists():
            self.stats.update(json.loads(self.path.read_text(encoding="utf-8")))

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

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.stats, indent=2, ensure_ascii=False), encoding="utf-8")
