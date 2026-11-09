"""Llamadas a herramientas mientras Hope razona (docs/HERRAMIENTAS_HN1.md §2).

Hope escribe ``⟦nombre: entrada⟧``; el runtime la ejecuta (a través del catálogo
y del guardián) y agrega ``→ resultado``; Hope sigue razonando con ese resultado.

    Paso 1: ⟦calc: 220 * 0,05⟧ → 11
    Paso 2: ⟦unidades: 11 V / (0,05 A) en ohm⟧ → 220 ohm

Cada llamada queda registrada (entrada, salida, éxito, duración): es la materia
prima de la TLM, la memoria de uso de herramientas (Herramientas §3.4).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from hope.tools.catalog import Catalog, ToolDenied

OPEN, CLOSE, ARROW = "⟦", "⟧", " → "
CALL_RE = re.compile(r"⟦\s*([\wáéíóúñ]+)\s*:\s*(.*?)\s*⟧", re.DOTALL)
MAX_OUT = 300

HELP = {
    "calc": "aritmética exacta: ⟦calc: 220 * 0,05⟧",
    "fraccion": "fracciones exactas: ⟦fraccion: 1/3 + 1/6⟧",
    "estad": "estadística: ⟦estad: 3 5 7 9⟧",
    "unidades": "unidades: ⟦unidades: 11 V / (0,05 A) en ohm⟧",
    "resolver": "ecuaciones: ⟦resolver: 2x + 3 = 7⟧ o ⟦resolver: x + y = 10; x - y = 2⟧",
    "derivar": "derivada: ⟦derivar: x^2 sin(x) | x⟧",
    "integrar": "integral: ⟦integrar: x^2 | x | 0 | 1⟧",
    "limite": "límite: ⟦limite: sin(x)/x | x | 0⟧",
    "matriz": "álgebra lineal: ⟦matriz: det [[1,2],[3,4]]⟧, inv, eig, solve [[..]] [..]",
    "logica": "lógica (z3): ⟦logica: x + y == 10; x - y == 2⟧",
    "python": "ejecutar Python en sandbox: ⟦python: print(sum(range(10)))⟧",
    "memoria": "buscar en la memoria propia: ⟦memoria: ley de Ohm⟧",
}


@dataclass
class ToolCall:
    name: str
    input: str
    output: str
    ok: bool
    seconds: float


@dataclass
class ToolRunner:
    catalog: Catalog
    memory: object | None = None                 # MemoryStore, para ⟦memoria: ...⟧
    calls: list[ToolCall] = field(default_factory=list)

    def _dispatch(self, name: str, arg: str) -> str:
        c = self.catalog.call
        parts = [p.strip() for p in arg.split("|")]
        if name == "calc":
            return str(c("hte_arith_v1", expr=arg).normalize())
        if name == "fraccion":
            return str(c("hte_fraction_v1", expr=arg))
        if name == "estad":
            return ", ".join(f"{k}={v:.6g}" for k, v in c("hte_stats_v1", numbers=arg).items())
        if name == "unidades":
            m = re.match(r"(.+?)\s+(?:en|a|to|in)\s+([^\s].*)$", arg)
            if m:
                c("hte_units_v1", expr=m.group(1))                # pasa por el guardián y valida la expresión
                from hope.tools import builtin
                q = builtin._UREG.parse_expression(normalize(m.group(1))).to(m.group(2))
                return f"{q.magnitude:.6g} {m.group(2)}"
            return c("hte_units_v1", expr=arg)["value"]
        if name == "resolver":
            sols = c("hte_solve_v1", equations=parts[0], variables=parts[1] if len(parts) > 1 else "")
            return "; ".join(", ".join(f"{k} = {v}" for k, v in s.items()) for s in sols) or "sin solución"
        if name == "derivar":
            return c("hte_derivative_v1", expr=parts[0], var=parts[1] if len(parts) > 1 else "x")
        if name == "integrar":
            return c("hte_integral_v1", expr=parts[0], var=parts[1] if len(parts) > 1 else "x",
                     lower=parts[2] if len(parts) > 3 else None, upper=parts[3] if len(parts) > 3 else None)
        if name == "limite":
            return c("hte_limit_v1", expr=parts[0], var=parts[1] if len(parts) > 1 else "x",
                     to=parts[2] if len(parts) > 2 else "0")
        if name == "matriz":
            op, _, rest = arg.partition(" ")
            m = re.match(r"\s*(\[\[.*?\]\])\s*(\[.*\])?\s*$", rest, re.DOTALL)
            if not m:
                raise ValueError("formato: ⟦matriz: det [[1,2],[3,4]]⟧")
            return str(c("hte_linalg_v1", op=op, matrix=m.group(1), vector=m.group(2)))
        if name == "logica":
            r = c("hte_logic_v1", constraints=parts[0], variables=parts[1] if len(parts) > 1 else "")
            return ("satisfacible: " + ", ".join(f"{k} = {v}" for k, v in r["modelo"].items())
                    if r["sat"] else "insatisfacible")
        if name == "python":
            r = c("hte_python_v1", code=arg)
            return r["stdout"].strip() if r["ok"] else "error: " + (r["stderr"].strip().splitlines() or ["?"])[-1]
        if name == "memoria":
            if self.memory is None:
                return "sin memoria disponible"
            from hope.memory.retrieval import retrieve
            res = retrieve(self.memory, arg, k=1)
            return res.items[0].engram.content[:MAX_OUT] if res.found else "no encontrado"
        raise ValueError(f"herramienta desconocida «{name}». Disponibles: {', '.join(HELP)}")

    def run(self, name: str, arg: str) -> str:
        t0 = time.perf_counter()
        try:
            out, ok = self._dispatch(name.lower(), arg), True
        except ToolDenied as e:
            out, ok = f"denegado: {e}", False
        except Exception as e:                                   # noqa: BLE001 — se informa a Hope, no se oculta
            out, ok = f"error: {str(e)[:150]}", False
        out = str(out)[:MAX_OUT]
        self.calls.append(ToolCall(name, arg, out, ok, time.perf_counter() - t0))
        return out

    def run_text(self, text: str) -> str:
        """Ejecuta la última llamada abierta al final de ``text`` y devuelve « → resultado»."""
        m = list(CALL_RE.finditer(text))
        if not m:
            return ""
        last = m[-1]
        return ARROW + self.run(last.group(1), last.group(2))


def normalize(s: str) -> str:
    from hope.tools.builtin import normalize_number_text
    return normalize_number_text(s)


def generate_with_tools(model, prompt: str, runner: ToolRunner, *, max_new: int = 400, max_calls: int = 8,
                        stop_after_line: bytes | None = b"Resultado:", temperature: float = 0.0,
                        generate: Callable | None = None) -> str:
    """Genera; cada vez que Hope cierra una llamada ``⟧``, se ejecuta y se inserta el resultado.

    Usa la generación con caché del modelo; después de cada resultado se retoma con
    el texto extendido (el prefill es mucho más barato que generar byte a byte).
    """
    gen = generate or model.generate
    text, produced = prompt, ""
    budget = max_new
    for _ in range(max_calls + 1):
        out = gen(text, max_new=budget, stop_after_line=stop_after_line, temperature=temperature,
                  stop_seq=CLOSE.encode()).decode("utf-8", errors="replace")
        text += out
        produced += out
        budget -= len(out.encode())
        if not out.endswith(CLOSE) or budget <= 0 or len(runner.calls) >= max_calls:
            break
        result = runner.run_text(produced)
        text += result
        produced += result
    return produced
