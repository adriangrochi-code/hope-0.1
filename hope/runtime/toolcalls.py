"""Llamadas a herramientas mientras Hope razona (docs/HERRAMIENTAS_HN1.md §2).

Hope escribe ``⟦nombre: entrada⟧``; el runtime la ejecuta (a través del catálogo
y del guardián) y agrega ``→ resultado``; Hope sigue razonando con ese resultado.

    Paso 1: ⟦calc: 220 * 0,05⟧ → 11
    Paso 2: ⟦unidades: 11 V / (0,05 A) en ohm⟧ → 220 ohm

Cada llamada queda registrada (entrada, salida, éxito, duración): es la materia
prima de la TLM, la memoria de uso de herramientas (Herramientas §3.4).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from hope.tools.catalog import Catalog, ToolDenied

OPEN, CLOSE, ARROW = "⟦", "⟧", " → "
CALL_RE = re.compile(r"⟦\s*([\wáéíóúñ]+)\s*:\s*(.*?)\s*⟧", re.DOTALL)
MAX_OUT = 600

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
    "anotar": "anotar en el cuaderno: ⟦anotar: ley de Ohm | V = I · R; la tensión es corriente por resistencia⟧",
    "nota": "consultar el cuaderno: ⟦nota: ley de Ohm⟧",
    "pruebas": "pytest: ⟦pruebas: <código> ||| <pruebas>⟧",
    "lint": "análisis estático: ⟦lint: <código>⟧",
    "compilar": "C/C++: ⟦compilar: c ||| <código>⟧",
    "buscar": "buscar en código: ⟦buscar: patrón | carpeta⟧",
    "diff": "diferencias: ⟦diff: <antes> ||| <después>⟧",
    "definicion": "dónde está definido: ⟦definicion: nombre | carpeta⟧ o ⟦definicion: ruta.py:línea:col⟧",
    "referencias": "dónde se usa: ⟦referencias: nombre | carpeta⟧",
    "tipos": "tipos (pyright/mypy): ⟦tipos: ruta.py⟧ o ⟦tipos: ruta.py | línea | expresión⟧",
    "estructura": "símbolos de un archivo o carpeta: ⟦estructura: ruta⟧",
    "recuperar": "código parecido del proyecto (RepoCoder): ⟦recuperar: descripción | carpeta⟧",
    "circuito": "simular: ⟦circuito: V1 in 0 5; R1 in out 1k; R2 out 0 1k⟧ (agregar «.ac 1k» para alterna)",
    "filtro": "respuesta en frecuencia: ⟦filtro: rc_pasabajos 1k 100n⟧",
    "sql": "consultar CSV: ⟦sql: <csv> ||| SELECT ...⟧",
    "convertir": "formatos: ⟦convertir: csv json ||| <texto>⟧",
    "correlacion": "relaciones entre columnas de una tabla: ⟦correlacion: <csv>⟧",
    "crear_herramienta": "pedir al maestro una herramienta nueva (TCG/TVE): ⟦crear_herramienta: contar vocales de un texto⟧",
    "componer": "encadenar herramientas: ⟦componer: invertir > contar_vocales | murciélago⟧",
    "buscar_web": "buscar en internet (permiso del usuario): ⟦buscar_web: ley de Ohm⟧ o ⟦buscar_web: arxiv: mamba⟧",
    "enlaces_web": "enlaces de una página para seguir navegando: ⟦enlaces_web: https://docs.python.org/3/⟧",
    "buscar_wiki": "Wikipedia local sin red, con título, URL y licencia: ⟦buscar_wiki: ley de Ohm⟧",
    "inspeccionar": "ver e inspeccionar una página (solo lectura): ⟦inspeccionar: https://developer.mozilla.org/⟧",
    "leer_web": "leer una página (dato no confiable, con URL): ⟦leer_web: https://es.wikipedia.org/wiki/Ley_de_Ohm⟧",
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
    forge: object | None = None                  # ToolForge (TCG/TVE), para ⟦crear_herramienta: ...⟧
    tlm: object | None = None                    # ToolUsageMemory, registra las composiciones
    web: object | None = None                    # WebTools (hope/tools/web.py), para ⟦buscar_web⟧ y ⟦leer_web⟧
    wiki: object | None = None                   # WikiIndex (hope/memory/wiki_index.py), para ⟦buscar_wiki⟧
    browser: object | None = None                # ReadOnlyBrowser (hope/tools/web_visual.py), para ⟦inspeccionar⟧
    topic: str = "composición"
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
        if name in ("anotar", "nota"):
            if self.memory is None:
                return "sin cuaderno disponible"
            from hope.memory.knowledge import KnowledgeBase, Notebook
            nb = self.memory if isinstance(self.memory, Notebook) else Notebook(
                self.memory if isinstance(self.memory, KnowledgeBase) else KnowledgeBase(self.memory))
            if name == "anotar":
                concept, _, text = arg.partition("|")
                if not text.strip():
                    return "formato: ⟦anotar: concepto | texto⟧"
                nb.write(concept.strip(), text.strip(), source="hope")
                return "anotado"
            n = nb.lookup(arg.strip())
            return n.text if n else "no hay nota sobre eso"
        if name == "memoria":
            if self.memory is None:
                return "sin memoria disponible"
            from hope.memory.retrieval import retrieve
            res = retrieve(getattr(self.memory, "store", self.memory), arg, k=1)   # MemoryStore, base o cuaderno
            return res.items[0].engram.content[:MAX_OUT] if res.found else "no encontrado"
        if name == "crear_herramienta":
            if self.forge is None:
                return "sin creador de herramientas disponible"
            from hope.tools.creation import ToolCreationError
            try:
                rec = self.forge.create(arg.strip())
            except ToolCreationError as e:
                return f"rechazada: {str(e)[:300]}"
            spec = self.catalog.spec(rec.tool_id)
            ins = ", ".join(f"{k}: {t}" for k, t in spec.inputs.items())
            outs = ", ".join(f"{k}: {t}" for k, t in spec.outputs.items())
            return f"creada {rec.tool_id} ({spec.status.value}; {ins} → {outs})"
        if name == "buscar_wiki":
            if self.wiki is None:
                return "sin Wikipedia local (python -m hope.memory.wiki_index ingerir ...)"
            from hope.tools.web import format_untrusted
            return format_untrusted(self.wiki.search_text(arg), "wikipedia local", max_chars=520)
        if name == "inspeccionar":
            if self.browser is None:
                return "sin navegador de solo lectura configurado"
            return self.browser.capture(arg.strip(), inspect=True).summary()
        if name == "enlaces_web":
            if self.web is None:
                return "sin acceso a internet configurado"
            return self.web.links_text(arg)
        if name in ("buscar_web", "leer_web"):
            if self.web is None:
                return "sin acceso a internet configurado"
            from hope.tools import web as webmod
            if isinstance(self.web, webmod.WebTools):
                webmod.register(self.catalog, self.web)
                return c("hte_web_search_v1" if name == "buscar_web" else "hte_web_read_v1",
                         **({"query": arg} if name == "buscar_web" else {"url": arg}))
            return self.web.search_text(arg) if name == "buscar_web" else self.web.read_text(arg)   # grabado
        if name == "componer":
            from hope.tools.compose import Composer
            return Composer(self.catalog, tlm=self.tlm).run_text(arg, topic=self.topic)
        big = [p.strip() for p in arg.split("|||")]
        if name == "pruebas":
            r = c("hte_pytest_v1", code=big[0], tests=big[1] if len(big) > 1 else "")
            return f"{r['passed']} pasaron, {r['failed']} fallaron" + ("" if r["ok"] else f"\n{r['output'][-400:]}")
        if name == "lint":
            r = c("hte_lint_v1", code=arg)
            return "sin problemas" if r["ok"] else r.get("error") or "; ".join(
                f"línea {i['line']}: {i['code']} {i['message']}" for i in r["issues"][:8])
        if name == "compilar":
            r = c("hte_compile_v1", lang=big[0].strip() if len(big) > 1 else "c", code=big[-1])
            if r["stage"] == "compilación":
                return "error de compilación: " + r["errors"].strip()[-400:]
            return r["stdout"].strip() if r["ok"] else f"error de ejecución (código {r['returncode']})"
        if name == "buscar":
            root = parts[1] if len(parts) > 1 else "."
            g = self.catalog.guardian
            if g is not None:
                from guardian.core import Action
                d = g.decide(Action("file.read", {"path": root}, requested_by="hte"))
                if not d.allowed:
                    raise ToolDenied("; ".join(d.reasons))
            hits = c("hte_search_v1", pattern=parts[0], root=root)
            return "\n".join(f"{h['path']}:{h['line']}: {h['text']}" for h in hits) or "sin coincidencias"
        if name in EDITOR_TOOLS:
            return editor_dispatch(name, arg, self._editor_for)
        if name == "diff":
            return c("hte_diff_v1", a=big[0], b=big[1] if len(big) > 1 else "") or "sin diferencias"
        if name == "circuito":
            r = c("hte_circuit_v1", netlist=arg)
            return json.dumps(r, ensure_ascii=False)
        if name == "filtro":
            r = c("hte_filter_v1", spec=arg)
            fc = r["corte_3db_hz_aprox"]
            return (f"corte −3 dB ≈ {fc:.6g} Hz, " if fc else "") + f"polos {r['polos']}, estable={r['estable']}"
        if name == "sql":
            return json.dumps(c("hte_sql_v1", csv_text=big[0], query=big[1] if len(big) > 1 else ""),
                              ensure_ascii=False)
        if name == "convertir":
            src, dst = big[0].split()[:2]
            return c("hte_convert_v1", text=big[1] if len(big) > 1 else "", src=src, dst=dst)
        if name == "correlacion":
            r = c("hte_correlation_v1", csv_text=arg)
            return "; ".join(f"{p['a']}~{p['b']}: r={p['pearson']} ρ={p['spearman']} ({p['fuerza']})"
                             for p in r["pares"][:6]) or "sin columnas numéricas"
        raise ValueError(f"herramienta desconocida «{name}». Disponibles: {', '.join(HELP)}")

    def _editor_for(self, root: str):
        """Herramientas de editor sobre ``root`` (permiso ``tool.execute`` del catálogo y ``file.read``
        del guardián para cada ruta)."""
        from hope.tools.editor import EditorTools, guardian_check
        g = self.catalog.guardian
        if g is not None:
            from guardian.core import Action
            d = g.decide(Action("tool.execute", {"tool": "hte_editor_v1", "sandboxed": True}, requested_by="hte"))
            if not d.allowed:
                raise ToolDenied("; ".join(d.reasons))
        return EditorTools(root, guardian_check(g) if g is not None else (lambda p, op: None))

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


EDITOR_TOOLS = ("definicion", "referencias", "tipos", "estructura", "recuperar")


def editor_dispatch(name: str, arg: str, make) -> str:
    """Las herramientas de editor (``hope/tools/editor.py``) en formato de texto para ⟦…⟧.
    ``make(raíz) -> EditorTools``; la raíz es el último campo ``| carpeta`` (por defecto «.»)."""
    parts = [p.strip() for p in arg.split("|")]
    if name == "tipos":
        path = parts[0]
        ed = make(".")
        r = ed.tipos(path, int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else None,
                     parts[2] if len(parts) > 2 else None)
        head = f"tipo de {r['expresion']}: {r.get('tipo')}\n" if r.get("expresion") else ""
        errs = "; ".join(f"línea {e['linea']}: {e['mensaje']}" for e in r.get("errores", [])[:6])
        return head + (errs or ("sin errores de tipos" if r.get("herramienta") else r.get("error", "")))
    ed = make(parts[1] if len(parts) > 1 and parts[1] else ".")
    if name == "estructura":
        return ed.estructura_texto(parts[0] or ".")
    if name == "definicion":
        hits = ed.definicion(parts[0])
        return "\n".join(f"{h['ruta']}:{h['linea']} {h.get('firma') or h['nombre']}" for h in hits[:8]) or "no encontrada"
    if name == "referencias":
        hits = ed.referencias(parts[0])
        return "\n".join(f"{h['ruta']}:{h['linea']}: {h['texto']}" for h in hits[:20]) or "sin referencias"
    from hope.tools.editor import contexto
    return contexto(ed.recuperar(parts[0], k=3), max_bytes=1200) or "nada parecido"


def normalize(s: str) -> str:
    from hope.tools.builtin import normalize_number_text
    return normalize_number_text(s)


def generate_with_tools(model, prompt: str, runner: ToolRunner, *, max_new: int = 400, max_calls: int = 8,
                        stop_after_line: bytes | None = b"Resultado:", temperature: float = 0.0,
                        generate: Callable | None = None, **gen_kwargs) -> str:
    """Genera; cada vez que Hope cierra una llamada ``⟧``, se ejecuta y se inserta el resultado.

    Usa la generación con caché del modelo; después de cada resultado se retoma con
    el texto extendido (el prefill es mucho más barato que generar byte a byte).
    """
    gen = generate or model.generate
    text, produced = prompt, ""
    budget = max_new
    for _ in range(max_calls + 1):
        out = gen(text, max_new=budget, stop_after_line=stop_after_line, temperature=temperature,
                  stop_seq=CLOSE.encode(), **gen_kwargs).decode("utf-8", errors="replace")
        text += out
        produced += out
        budget -= len(out.encode())
        if not out.endswith(CLOSE) or budget <= 0 or len(runner.calls) >= max_calls:
            break
        result = runner.run_text(produced)
        text += result
        produced += result
    return produced
