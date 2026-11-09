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

**Selección aprendida** (docs/HERRAMIENTAS_HN1.md §8.3): ``rank`` combina las estadísticas
del tema y de los **temas parecidos** (palabras en común con el tema y el enunciado), con el
éxito y el tiempo medio de cada herramienta; ``context_hint`` lo escribe en el contexto con un
formato fijo que Hope aprende a leer en el currículo (``hope/data/tool_curriculum.py``)::

    Herramientas sugeridas: unidades (éxito 97 %, 0,02 s), calc (éxito 80 %, 0,00 s)

``ninguna`` significa «responder directo» (lo registra ``record_direct``: en temas triviales
gana porque no cuesta tiempo de herramienta). ``parse_hint`` lee la línea.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

COMPOSITIONS_KEY = "__composiciones__"
HINT_PREFIX = "Herramientas sugeridas:"
DIRECT = "ninguna"                       # responder directo, sin herramienta
_STOP = {"de", "la", "el", "los", "las", "y", "en", "con", "un", "una", "por", "que", "del", "para", "es", "se",
         "al", "lo", "su", "como", "cual", "cuanto", "cuanta", "cuantos", "the", "of", "and", "what", "is"}
_HINT_RE = re.compile(r"([\wáéíóúñ]+) \(éxito (\d+) %, ([\d,]+) s\)")


def topic_words(text: str) -> set[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return {w for w in re.findall(r"[a-z]{3,}", t) if w not in _STOP}


def parse_hint(line: str) -> list[tuple[str, float, float]]:
    """«Herramientas sugeridas: calc (éxito 98 %, 0,01 s), …» → [(herramienta, éxito, segundos)]."""
    if HINT_PREFIX not in line:
        return []
    return [(m.group(1), int(m.group(2)) / 100, float(m.group(3).replace(",", ".")))
            for m in _HINT_RE.finditer(line.split(HINT_PREFIX, 1)[1])]


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

    def record_direct(self, topic: str, answer_correct: bool, seconds: float = 0.0) -> None:
        """Una respuesta sin herramienta: alimenta la opción «ninguna» (responder directo)."""
        s = self.stats[topic].setdefault(DIRECT, {"uses": 0, "ok": 0, "useful": 0, "seconds": 0.0})
        s["uses"] += 1
        s["ok"] += 1
        s["useful"] += int(answer_correct)
        s["seconds"] += seconds
        self.save()

    # --- selección aprendida: temas parecidos, éxito y tiempo ----------------------
    def similar_topics(self, topic: str, problem: str = "", k: int = 5, min_sim: float = 0.2) -> list[tuple[str, float]]:
        """Temas registrados parecidos a ``topic`` (+ palabras del enunciado): fracción de las
        palabras del tema guardado que aparecen en la consulta. El mismo tema vale 1."""
        q = topic_words(f"{topic} {problem}")
        out = []
        for t in self.stats:
            if t == topic:
                out.append((t, 1.0))
                continue
            tw = topic_words(t)
            if tw:
                sim = len(tw & q) / len(tw)
                if sim >= min_sim:
                    out.append((t, sim))
        return sorted(out, key=lambda x: -x[1])[:k]

    def rank(self, topic: str, problem: str = "", *, time_scale: float = 1.0) -> list[tuple[str, float, dict]]:
        """Herramientas ordenadas por lo que rindió en temas parecidos, castigando el tiempo medio.

        Por tema parecido ``t`` (similitud ``s_t``) y herramienta ``h``:
        ``parte = usos(h) / usos(t)`` (qué tan seguido se usa h en ese tema) y
        ``utilidad = (útiles+1)/(usos+2) · (funcionaron+1)/(usos+2)`` (Laplace).
        puntaje(h) = Σ s_t · parte · utilidad / Σ s_t, dividido por (1 + segundos_medios / time_scale).
        """
        sims = self.similar_topics(topic, problem)
        total_sim = sum(s for _, s in sims) or 1.0
        score: dict[str, float] = defaultdict(float)
        agg: dict[str, dict[str, float]] = defaultdict(lambda: {"uses": 0.0, "useful": 0.0, "seconds": 0.0})
        for t, sim in sims:
            tools = self.stats[t]
            n_topic = sum(s["uses"] for s in tools.values()) or 1
            for tool, s in tools.items():
                n = s["uses"]
                util = (s["useful"] + 1) / (n + 2) * (s["ok"] + 1) / (n + 2)
                score[tool] += sim * n / n_topic * util / total_sim
                for key in agg[tool]:
                    agg[tool][key] += sim * s[key]
        ranked = []
        for tool, sc in score.items():
            a = agg[tool]
            mean_s = a["seconds"] / max(a["uses"], 1e-9)
            ranked.append((tool, sc / (1.0 + mean_s / time_scale),
                           {"usos": a["uses"], "exito": a["useful"] / max(a["uses"], 1e-9), "segundos": mean_s}))
        return sorted(ranked, key=lambda x: -x[1])

    def context_hint(self, topic: str, problem: str = "", k: int = 3, min_score: float = 0.1) -> str:
        """Línea para el contexto (formato documentado arriba); vacía si no hay datos suficientes."""
        best = [r for r in self.rank(topic, problem) if r[1] >= min_score][:k]
        if not best:
            return ""
        return HINT_PREFIX + " " + ", ".join(
            f"{tool} (éxito {round(100 * st['exito'])} %, {st['segundos']:.2f} s)".replace(".", ",")
            for tool, _, st in best)

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
