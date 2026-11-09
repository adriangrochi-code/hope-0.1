"""HTO — composición de herramientas en tuberías (Herramientas §3.6; docs/HERRAMIENTAS_HN1.md §3 nivel 7).

Encadena herramientas del catálogo: la salida de una es la entrada de la siguiente.

    ⟦componer: invertir > contar_vocales | murciélago⟧ → 5

* **Verificación de tipos antes de ejecutar**, con las fichas: el tipo de salida
  de cada paso tiene que caber en el tipo de entrada del siguiente
  (``int`` cabe en ``float``/``number``; nada se convierte en silencio salvo a
  ``str`` cuando la ficha declara texto). Con ``h.campo`` se elige un campo de una
  salida con varios.
* **Ejecución paso a paso** a través del catálogo (y del guardián): las
  herramientas creadas por TCG corren en el sandbox, en un proceso aparte. Se
  comprueba el tipo real de cada resultado; el primer error detiene la tubería
  y dice en qué paso falló.
* **Registro en la TLM** del tiempo de cada paso, del total y de si la
  composición funcionó y fue exacta (cuando se conoce la respuesta), para que
  Hope aprenda qué composiciones son rápidas y exactas para cada tema.
"""

from __future__ import annotations

import inspect
import json
import re
import time
from dataclasses import dataclass, field

from hope.tools.catalog import Catalog
from hope.tools.creation import type_alternatives, value_has_type

# Nombres cortos de las llamadas en el texto → herramientas del catálogo.
ALIASES = {
    "calc": "hte_arith_v1", "fraccion": "hte_fraction_v1", "estad": "hte_stats_v1", "unidades": "hte_units_v1",
    "resolver": "hte_solve_v1", "derivar": "hte_derivative_v1", "integrar": "hte_integral_v1",
    "limite": "hte_limit_v1", "matriz": "hte_linalg_v1", "logica": "hte_logic_v1", "python": "hte_python_v1",
    "lint": "hte_lint_v1", "diff": "hte_diff_v1", "circuito": "hte_circuit_v1", "filtro": "hte_filter_v1",
    "convertir": "hte_convert_v1", "correlacion": "hte_correlation_v1",
}
_NUMERIC = {"int", "float", "number"}


class CompositionError(ValueError):
    pass


@dataclass
class Step:
    tool_id: str
    field: str | None = None

    @property
    def label(self) -> str:
        return self.tool_id + (f".{self.field}" if self.field else "")


@dataclass
class StepResult:
    name: str                 # id de la herramienta (para la TLM)
    input: object
    output: object
    ok: bool
    seconds: float
    error: str = ""


@dataclass
class CompositionResult:
    chain: list[str]
    ok: bool
    value: object
    steps: list[StepResult] = field(default_factory=list)
    seconds: float = 0.0
    exact: bool | None = None
    error: str = ""


def type_fits(out_t: str, in_t: str) -> bool:
    """¿Un valor del tipo ``out_t`` es entrada válida para ``in_t``?"""
    outs, ins = type_alternatives(out_t), type_alternatives(in_t)
    if "any" in ins or "any" in outs:          # sin tipo declarado: se comprueba al ejecutar
        return True
    for o in outs:
        ok = o in ins or (o == "int" and ins & {"float", "number"}) or (o in {"float", "number"} and
                                                                           ins & {"float", "number"})
        if not ok:
            return False
    return True


class Composer:
    def __init__(self, catalog: Catalog, tlm=None, aliases: dict[str, str] | None = None):
        self.catalog = catalog
        self.tlm = tlm
        self.aliases = ALIASES if aliases is None else aliases

    # --- nombres ------------------------------------------------------------------
    def resolve(self, name: str) -> str:
        name = name.strip()
        if name in self.catalog:
            return name
        if self.aliases.get(name) in self.catalog:
            return self.aliases[name]
        created = sorted((tid for tid in self.catalog.ids() if re.fullmatch(rf"tcg_{re.escape(name)}_v(\d+)", tid)),
                         key=lambda t: int(t.rsplit("_v", 1)[1]))
        if created:
            return created[-1]
        raise CompositionError(f"herramienta desconocida en la composición: «{name}»")

    def parse(self, chain: str | list[str]) -> list[Step]:
        parts = chain if isinstance(chain, list) else [p for p in chain.split(">")]
        steps = []
        for p in parts:
            p = p.strip()
            if not p:
                raise CompositionError("paso vacío en la composición")
            name, _, fld = p.partition(".") if p not in self.catalog else (p, "", "")
            steps.append(Step(self.resolve(name), fld.strip() or None))
        if not steps:
            raise CompositionError("composición vacía")
        return steps

    # --- fichas -------------------------------------------------------------------
    def _main_input(self, tool_id: str) -> tuple[str, str]:
        """Parámetro que recibe la salida del paso anterior y su tipo declarado."""
        spec = self.catalog.spec(tool_id)
        params = [p for p in inspect.signature(self.catalog.function(tool_id)).parameters.values()
                  if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        declared = list(spec.inputs.items())
        if declared and (not params or declared[0][0] in {p.name for p in params}):
            return declared[0]
        if not params:                              # función con **kwargs y sin ficha de entrada
            raise CompositionError(f"{tool_id}: la ficha no declara entradas")
        return params[0].name, declared[0][1] if declared else "any"

    def _extra_required(self, tool_id: str) -> list[str]:
        spec = self.catalog.spec(tool_id)
        main = self._main_input(tool_id)[0]
        return [k for k, t in spec.inputs.items() if k != main and not str(t).endswith("?")
                and k in inspect.signature(self.catalog.function(tool_id)).parameters]

    def _output_type(self, step: Step) -> str:
        outs = self.catalog.spec(step.tool_id).outputs
        if step.field:
            if step.field not in outs:
                raise CompositionError(f"{step.tool_id} no tiene el campo de salida «{step.field}»")
            return outs[step.field]
        if len(outs) == 1:
            return next(iter(outs.values()))
        return "dict" if outs else "any"

    def check(self, steps: list[Step]) -> list[str]:
        """Problemas de tipos de la tubería según las fichas (vacío = compatible)."""
        problems = []
        for i, st in enumerate(steps):
            try:
                self._output_type(st)
            except CompositionError as e:
                problems.append(str(e))
                continue
            if i == 0:
                continue
            prev = steps[i - 1]
            out_t = self._output_type(prev)
            name, in_t = self._main_input(st.tool_id)
            if not type_fits(out_t, in_t):
                problems.append(f"paso {i + 1}: {prev.label} produce {out_t} pero {st.tool_id} espera "
                                f"{in_t} en «{name}»")
            extra = self._extra_required(st.tool_id)
            if extra:
                problems.append(f"paso {i + 1}: {st.tool_id} necesita además {', '.join(extra)}; "
                                "no puede ir en medio de una tubería")
        return problems

    # --- ejecución ----------------------------------------------------------------
    def _coerce_input(self, tool_id: str, value):
        """La entrada del texto llega como cadena; si la ficha pide otro tipo, se interpreta como JSON."""
        _, in_t = self._main_input(tool_id)
        if isinstance(value, str) and not ({"str", "any"} & type_alternatives(in_t)):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def run(self, chain: str | list[str], value, *, topic: str = "composición", expected=None) -> CompositionResult:
        steps = self.parse(chain)
        labels = [s.label for s in steps]
        problems = self.check(steps)
        if problems:
            raise CompositionError("tipos incompatibles: " + "; ".join(problems))
        t_start = time.perf_counter()
        results: list[StepResult] = []
        current = self._coerce_input(steps[0].tool_id, value)
        error = ""
        for i, st in enumerate(steps):
            t0 = time.perf_counter()
            try:
                if i == 0 and isinstance(current, dict) and len(self.catalog.spec(st.tool_id).inputs) > 1:
                    kwargs = current                                   # varias entradas en la primera herramienta
                else:
                    kwargs = {self._main_input(st.tool_id)[0]: current}
                out = self.catalog.call(st.tool_id, **kwargs)
                if st.field:
                    out = out[st.field]
                out_t = self._output_type(st)
                if "str" in type_alternatives(out_t) and not isinstance(out, (str, dict, list)):
                    out = str(out)                                     # la ficha declara la forma de texto
                if not value_has_type(out, out_t):
                    raise TypeError(f"devolvió {type(out).__name__}, la ficha dice {out_t}")
                results.append(StepResult(st.tool_id, current, out, True, time.perf_counter() - t0))
                current = out
            except Exception as e:                                     # noqa: BLE001 — se informa el paso
                error = f"paso {i + 1} ({st.label}): {str(e)[:200]}"
                results.append(StepResult(st.tool_id, current, None, False, time.perf_counter() - t0, str(e)[:200]))
                break
        ok = not error
        exact = None if expected is None else (ok and _same(current, expected))
        res = CompositionResult(labels, ok, current if ok else None, results, time.perf_counter() - t_start,
                                exact, error)
        if self.tlm is not None:
            self.tlm.record(results, topic, answer_correct=exact if exact is not None else ok)
            self.tlm.record_composition(labels, topic, ok=ok, seconds=res.seconds, exact=exact,
                                        step_seconds=[r.seconds for r in results])
        return res

    def run_text(self, arg: str, *, topic: str = "composición") -> str:
        """Formato de la llamada en el texto: ``h1 > h2 > h3 | entrada``."""
        chain, sep, value = arg.partition("|")
        if not sep:
            raise CompositionError("formato: ⟦componer: h1 > h2 | entrada⟧")
        r = self.run(chain, value.strip(), topic=topic)
        if not r.ok:
            return "error: " + r.error
        v = r.value
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)


def _same(a, b) -> bool:
    from hope.tools.builtin import numbers_equal
    if not isinstance(a, (str, bool)) and not isinstance(b, bool):
        try:
            return numbers_equal(a, b)
        except Exception:                                               # noqa: BLE001 — no son números
            return str(a).strip() == str(b).strip()
    return str(a).strip() == str(b).strip()
