"""Bucle de autodepuración (estilo *Self-Debugging*, Chen et al. 2023, y *AlphaCodium*, Ridnik et al. 2024).

Un modelo chico rara vez acierta un programa de una vez, pero puede **leer el error y corregir**
si alguien ejecuta el código por él. El bucle:

1. **Entender** el problema: reformularlo (entradas, salida, casos borde). Opcional.
2. **Generar pruebas** ``assert f(…) == …`` además de los ejemplos del enunciado (las
   «anclas»), y **validarlas con pruebas por mutación**: se generan mutantes de una solución
   candidata (``+``↔``-``, ``<``↔``<=``, constantes ±1, ``and``↔``or``, …) y se mide cuántos
   mata cada prueba. Una prueba que no distingue ningún mutante no aporta y se descarta; si el
   conjunto mata pocos mutantes se piden más pruebas.
3. **Escribir** el código (con la gramática de ``hope/core/gramatica.py`` si se pide, y con el
   código parecido del proyecto recuperado por ``EditorTools.recuperar`` si hay carpeta).
4. **Ejecutar** las pruebas en el sandbox, una por una (cada una informa lo que obtuvo).
5. **Leer el error**: las primeras fallas como «assert f(2) == 4 → obtuvo 5» o la excepción.
6. **Corregir** con ese informe y volver a 4, hasta ``rounds`` veces.

Anclaje de pruebas (AlphaCodium): las anclas son la verdad; nunca se acepta un arreglo que
rompa un ancla para pasar una prueba generada (la mejor solución se elige primero por anclas).

Integración:

* ``autodebug_agent`` → un ``Agent`` para ``BudgetedResponder`` (su veredicto sale de las
  pruebas, no de la crítica de pasos aritméticos);
* ``autodepurar_carpeta`` → en el harness: ediciones buscar/reemplazar, el comando de pruebas
  de la carpeta en el sandbox sobre una copia, y la mejor edición queda **propuesta** (no escrita).
"""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol

from hope.tools.sandbox import run_python

# ---------------------------------------------------------------------------
# Quién escribe: el modelo de Hope, el maestro o un guion (pruebas)
# ---------------------------------------------------------------------------


class Coder(Protocol):
    def text(self, prompt: str, *, temperature: float = 0.0, max_new: int = 300, stop: bytes | None = b"\n\n") -> str: ...

    def code(self, prompt: str, *, code_prefix: str = "", temperature: float = 0.0, max_new: int = 400) -> str: ...


def cut_function(code: str, name: str | None = None) -> str:
    """Se queda con las definiciones iniciales: corta en la primera línea a la izquierda que no sea
    ``def``/``class``/decorador/import (lo que el modelo siga escribiendo después)."""
    lines = code.split("\n")
    out, started = [], False
    for ln in lines:
        top = ln[:1] not in (" ", "\t", "") and not ln.startswith(("#",))
        if top and started and not ln.startswith(("def ", "class ", "@", "import ", "from ")):
            break
        if ln.startswith(("def ", "class ", "@")):
            started = True
        out.append(ln)
    return "\n".join(out).rstrip() + "\n"


@dataclass
class ModelCoder:
    """Hope (``HN1Model``) como programador. ``grammar=True`` usa la decodificación con gramática."""
    model: object
    grammar: bool = False
    n_iters: int | None = None
    stats: dict = field(default_factory=lambda: {"bytes": 0, "llamadas": 0})

    def _gen(self, prompt: str, max_new: int, temperature: float, **kw) -> str:
        out = self.model.generate(prompt, max_new, temperature=temperature, n_iters=self.n_iters, **kw)
        self.stats["bytes"] += len(out)
        self.stats["llamadas"] += 1
        return out.decode("utf-8", "replace")

    def text(self, prompt: str, *, temperature: float = 0.0, max_new: int = 300, stop: bytes | None = b"\n\n") -> str:
        return self._gen(prompt, max_new, temperature, stop_seq=stop)

    def code(self, prompt: str, *, code_prefix: str = "", temperature: float = 0.0, max_new: int = 400) -> str:
        if self.grammar:
            from hope.core.gramatica import generate_code
            g = generate_code(self.model, prompt + code_prefix, code_prefix=code_prefix, max_new=max_new,
                              temperature=temperature, n_iters=self.n_iters, stop_seq=b"\n\n\n")
            self.stats["bytes"] += len(g.raw.encode())
            self.stats["llamadas"] += 1
            return cut_function(g.text)
        body = self._gen(prompt + code_prefix, max_new, temperature, stop_seq=b"\n\n\n")
        return cut_function(code_prefix + body)


@dataclass
class TeacherCoder:
    """El maestro (cliente OpenAI-compatible de ``hope/teacher/client.py``) como programador."""
    teacher: object
    stats: dict = field(default_factory=lambda: {"bytes": 0, "llamadas": 0})

    def _ask(self, prompt: str) -> str:
        r = self.teacher.ask([{"role": "user", "content": prompt}], purpose="autodepurar")
        self.stats["bytes"] += len(r.text.encode())
        self.stats["llamadas"] += 1
        return r.text

    def text(self, prompt: str, *, temperature: float = 0.0, max_new: int = 300, stop: bytes | None = None) -> str:
        return self._ask(prompt)

    def code(self, prompt: str, *, code_prefix: str = "", temperature: float = 0.0, max_new: int = 400) -> str:
        t = self._ask(prompt + code_prefix + "\n\nRespondé solo con el código Python completo en un bloque ```python.")
        m = re.search(r"```(?:python)?\n(.*?)```", t, re.DOTALL)
        return (m.group(1) if m else t).rstrip() + "\n"


@dataclass
class ScriptedCoder:
    """Respuestas fijas en orden (para pruebas y demostraciones)."""
    codes: list[str] = field(default_factory=list)
    tests: str = ""
    reflection: str = ""
    replies: list[str] = field(default_factory=list)     # respuestas de ``text`` en orden (modo carpeta)
    i: int = 0
    prompts: list[str] = field(default_factory=list)

    def text(self, prompt: str, *, temperature: float = 0.0, max_new: int = 300, stop: bytes | None = None) -> str:
        self.prompts.append(prompt)
        if self.replies:
            return self.replies.pop(0)
        return self.tests if "assert" in prompt[-200:] else self.reflection

    def code(self, prompt: str, *, code_prefix: str = "", temperature: float = 0.0, max_new: int = 400) -> str:
        self.prompts.append(prompt)
        c = self.codes[min(self.i, len(self.codes) - 1)]
        self.i += 1
        return c


# ---------------------------------------------------------------------------
# Ejecutar pruebas una por una (en el sandbox)
# ---------------------------------------------------------------------------
RUNNER = r'''
import sys, json, traceback
data = json.loads(sys.stdin.read())
res = []
for code in data["codes"]:
    g = {"__name__": "solucion"}
    row = {"carga": None, "pruebas": []}
    try:
        exec(compile(code, "solucion.py", "exec"), g)
    except BaseException as e:
        row["carga"] = type(e).__name__ + ": " + str(e)[:200]
        res.append(row)
        continue
    for t in data["tests"]:
        r = {"prueba": t, "ok": False}
        try:
            node = compile(t, "<prueba>", "exec", 0x400)   # PyCF_ONLY_AST
            st = node.body[0]
            if type(st).__name__ == "Assert" and type(st.test).__name__ == "Compare" and len(st.test.ops) == 1 \
                    and type(st.test.ops[0]).__name__ == "Eq":
                import ast
                got = eval(compile(ast.Expression(st.test.left), "<prueba>", "eval"), g)
                want = eval(compile(ast.Expression(st.test.comparators[0]), "<prueba>", "eval"), g)
                r["obtenido"] = repr(got)[:200]
                r["ok"] = bool(got == want)
                if not r["ok"]:
                    r["error"] = "obtuvo " + repr(got)[:150] + ", se esperaba " + repr(want)[:150]
            else:
                exec(compile(t, "<prueba>", "exec"), g)
                r["ok"] = True
        except AssertionError:
            r["error"] = "AssertionError"
        except BaseException as e:
            tb = traceback.extract_tb(e.__traceback__)
            src = code.split("\n")
            where = next((f"línea {f.lineno}: {src[f.lineno - 1].strip() if 0 < f.lineno <= len(src) else ''}"
                          for f in reversed(tb) if f.filename == "solucion.py"), "")
            r["error"] = (type(e).__name__ + ": " + str(e)[:150] + (" (" + where + ")" if where else ""))
        row["pruebas"].append(r)
    res.append(row)
print(json.dumps(res))
'''


def run_tests_detailed(codes: list[str], tests: list[str], *, timeout_s: float = 10.0) -> list[dict]:
    """Corre ``tests`` contra cada código (un proceso del sandbox para todos). Por código:
    ``{"carga": error | None, "pruebas": [{prueba, ok, error?, obtenido?}], "pasan": n}``."""
    r = run_python(RUNNER, stdin=json.dumps({"codes": codes, "tests": tests}), timeout_s=timeout_s)
    try:
        rows = json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        if len(codes) > 1:                      # un código colgó o rompió el proceso: de a uno
            return [run_tests_detailed([c], tests, timeout_s=max(2.0, timeout_s / len(codes)))[0] for c in codes]
        why = "tiempo agotado" if r.timed_out else (r.stderr.strip().splitlines() or ["error del sandbox"])[-1][:200]
        rows = [{"carga": why, "pruebas": [{"prueba": t, "ok": False, "error": why} for t in tests]}]
    for row in rows:
        row["pasan"] = sum(p["ok"] for p in row["pruebas"])
    return rows


# ---------------------------------------------------------------------------
# Pruebas por mutación
# ---------------------------------------------------------------------------
_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult, ast.Div: ast.Mult,
         ast.Mod: ast.FloorDiv, ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
         ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.And: ast.Or, ast.Or: ast.And, ast.In: ast.NotIn,
         ast.NotIn: ast.In}


def mutants(code: str, max_mutants: int = 20) -> list[str]:
    """Mutantes de primer orden del código (un cambio chico cada uno), sin repetir."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.BoolOp, ast.AugAssign)) and type(node.op) in _SWAP:
            sites.append((node, "op", None))
        elif isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in _SWAP:
                    sites.append((node, "cmp", i))
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            sites.append((node, "bool", None))
        elif isinstance(node, ast.Constant) and isinstance(node.value, int):
            sites.append((node, "int", None))
        elif isinstance(node, ast.Return) and node.value is not None:
            sites.append((node, "ret", None))
    out, seen = [], {ast.unparse(tree)}
    for node, kind, i in sites:
        if len(out) >= max_mutants:
            break
        if kind == "op":
            old = node.op
            node.op = _SWAP[type(old)]()
            m = ast.unparse(tree)
            node.op = old
        elif kind == "cmp":
            old = node.ops[i]
            node.ops[i] = _SWAP[type(old)]()
            m = ast.unparse(tree)
            node.ops[i] = old
        elif kind == "bool":
            node.value = not node.value
            m = ast.unparse(tree)
            node.value = not node.value
        elif kind == "int":
            node.value += 1
            m = ast.unparse(tree)
            node.value -= 1
        else:
            old = node.value
            node.value = ast.Constant(None)
            m = ast.unparse(tree)
            node.value = old
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


@dataclass
class MutationReport:
    mutants: int
    killed: int
    per_test: dict[str, int]

    @property
    def score(self) -> float:
        return self.killed / self.mutants if self.mutants else 0.0


def mutation_score(code: str, tests: list[str], *, max_mutants: int = 20, timeout_s: float = 20.0) -> MutationReport:
    """Cuántos mutantes de ``code`` mata el conjunto de pruebas, y cuántos mata cada prueba."""
    ms = mutants(code, max_mutants)
    if not ms or not tests:
        return MutationReport(len(ms), 0, {t: 0 for t in tests})
    rows = run_tests_detailed(ms, tests, timeout_s=timeout_s)
    per = {t: 0 for t in tests}
    killed = 0
    for row in rows:
        fails = [p["prueba"] for p in row["pruebas"] if not p["ok"]] if row["carga"] is None else list(tests)
        killed += bool(fails)
        for t in fails:
            per[t] = per.get(t, 0) + 1
    return MutationReport(len(ms), killed, per)


# ---------------------------------------------------------------------------
# El bucle
# ---------------------------------------------------------------------------
@dataclass
class CodeTask:
    enunciado: str
    nombre: str                      # función a escribir
    firma: str = ""                  # "def nombre(a, b):" (si se conoce)
    ejemplos: list[str] = field(default_factory=list)      # asserts del enunciado (anclas)
    contexto: str = ""               # código del proyecto recuperado, etc.

    def head(self) -> str:
        """Firma + docstring con el enunciado y los ejemplos (estilo HumanEval)."""
        sig = self.firma or f"def {self.nombre}(*args):"
        doc = self.enunciado.strip().replace('"""', "'''")
        ex = "".join(f"\n    >>> {e.replace('assert ', '', 1).split(' == ')[0]}\n    {e.split(' == ', 1)[1]}"
                     for e in self.ejemplos[:3] if " == " in e)
        return f'{sig}\n    """{doc}\n{ex}\n    """\n'


@dataclass
class Round:
    n: int
    code: str
    anchors_ok: int
    anchors: int
    tests_ok: int
    tests: int
    feedback: str
    seconds: float


@dataclass
class DebugReport:
    code: str
    ok: bool                         # pasa todas las anclas y todas las pruebas que quedaron
    anchors_ok: int
    anchors: int
    tests_ok: int
    tests: int
    rounds: list[Round]
    generated_tests: list[str]
    dropped_tests: list[str]
    mutation: dict | None
    reflection: str
    seconds: float
    model_bytes: int = 0

    @property
    def verdict(self) -> str:
        if self.anchors and self.anchors_ok < self.anchors:
            return "errores"
        if self.ok and (self.anchors or (self.mutation or {}).get("puntaje", 0) >= 0.5):
            return "ok"
        return "incierto" if self.tests_ok else "errores"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["veredicto"] = self.verdict
        return d


def parse_tests(text: str, name: str) -> list[str]:
    """Los ``assert`` que llaman a ``name`` y compilan, sin repetir."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("assert ") or f"{name}(" not in line:
            continue
        try:
            ast.parse(line)
        except SyntaxError:
            continue
        if line not in out:
            out.append(line)
    return out


def feedback_text(row: dict, max_fail: int = 3) -> str:
    if row["carga"] is not None:
        return f"El código no carga: {row['carga']}"
    fails = [p for p in row["pruebas"] if not p["ok"]][:max_fail]
    return "\n".join(f"- {p['prueba']} → {p.get('error', 'falla')}" for p in fails)


@dataclass
class SelfDebugger:
    coder: Coder
    rounds: int = 4
    n_tests: int = 5
    reflect: bool = True
    mutation: bool = True
    min_mutation: float = 0.5
    temperature_fix: float = 0.6
    timeout_s: float = 10.0
    max_new: int = 400
    log: Callable[[str], None] | None = None

    def _say(self, msg: str) -> None:
        if self.log:
            self.log(msg)

    def _gen_tests(self, task: CodeTask, reflection: str, extra: str = "") -> list[str]:
        prompt = (f"{task.contexto}\n{task.head()}" + (f"# Análisis: {reflection.strip()[:300]}\n" if reflection else "")
                  + extra + f"\n# Pruebas de {task.nombre}\n" + "".join(e + "\n" for e in task.ejemplos)
                  + f"assert {task.nombre}(")
        out = self.coder.text(prompt, temperature=0.3, max_new=300)
        return [t for t in parse_tests(f"assert {task.nombre}(" + out, task.nombre) if t not in task.ejemplos]

    def solve(self, task: CodeTask) -> DebugReport:
        t0 = time.perf_counter()
        reflection = ""
        if self.reflect:
            reflection = self.coder.text(f"{task.enunciado.strip()}\nEntradas, salida y casos borde:", max_new=200)
        tests = self._gen_tests(task, reflection)[:self.n_tests]
        anchors = list(task.ejemplos)
        head = task.head()
        prompt = (task.contexto + "\n" if task.contexto else "")
        code = self.coder.code(prompt, code_prefix=head, max_new=self.max_new)
        rounds: list[Round] = []
        best: tuple | None = None
        dropped: list[str] = []
        mutation = None
        asked_more = False
        for n in range(self.rounds):
            t1 = time.perf_counter()
            row = run_tests_detailed([code], anchors + tests, timeout_s=self.timeout_s)[0]
            res = row["pruebas"]
            a_ok = sum(p["ok"] for p in res[:len(anchors)]) if row["carga"] is None else 0
            t_ok = sum(p["ok"] for p in res[len(anchors):]) if row["carga"] is None else 0
            fb = feedback_text(row)
            rounds.append(Round(n, code, a_ok, len(anchors), t_ok, len(tests), fb, time.perf_counter() - t1))
            key = (a_ok, t_ok, -n)
            if best is None or key > best[0]:
                best = (key, code)
            self._say(f"ronda {n}: anclas {a_ok}/{len(anchors)}, pruebas {t_ok}/{len(tests)}")
            if a_ok == len(anchors) and t_ok == len(tests):
                # Validar las pruebas: ¿distinguen a la solución de sus mutantes?
                if self.mutation and tests:
                    mr = mutation_score(code, anchors + tests, timeout_s=self.timeout_s * 2)
                    useless = [t for t in tests if mr.per_test.get(t, 0) == 0]
                    if useless:
                        dropped += useless
                        tests = [t for t in tests if t not in useless]
                    mutation = {"mutantes": mr.mutants, "muertos": mr.killed, "puntaje": round(mr.score, 3)}
                    if mr.score < self.min_mutation and n + 1 < self.rounds and not asked_more:
                        asked_more = True
                        more = self._gen_tests(task, reflection, extra="# Más pruebas, con casos borde:\n")
                        more = [t for t in more if t not in tests and t not in dropped][:self.n_tests]
                        if more:
                            tests += more
                            continue
                break
            # Anclaje: si una prueba generada contradice una solución que pasa todas las anclas en dos
            # rondas seguidas, se la considera sospechosa y se descarta.
            if a_ok == len(anchors) and anchors and len(rounds) >= 2 and rounds[-2].anchors_ok == len(anchors):
                prev = run_tests_detailed([rounds[-2].code], tests, timeout_s=self.timeout_s)[0]
                both = [t for t, p, q in zip(tests, res[len(anchors):], prev["pruebas"]) if not p["ok"] and not q["ok"]]
                if both:
                    dropped += both
                    tests = [t for t in tests if t not in both]
                    continue
            if n + 1 >= self.rounds:
                break
            fix_prompt = (f"{task.contexto}\n# Código con errores:\n{_comment(code)}\n# Fallas:\n{_comment(fb)}\n"
                          f"# Versión corregida:\n")
            code = self.coder.code(fix_prompt, code_prefix=head, temperature=self.temperature_fix, max_new=self.max_new)
        final = best[1] if best else code
        row = run_tests_detailed([final], anchors + tests, timeout_s=self.timeout_s)[0]
        res = row["pruebas"]
        a_ok = sum(p["ok"] for p in res[:len(anchors)]) if row["carga"] is None else 0
        t_ok = sum(p["ok"] for p in res[len(anchors):]) if row["carga"] is None else 0
        return DebugReport(final, a_ok == len(anchors) and t_ok == len(tests) and row["carga"] is None, a_ok,
                           len(anchors), t_ok, len(tests), rounds, tests, dropped, mutation, reflection,
                           time.perf_counter() - t0, getattr(self.coder, "stats", {}).get("bytes", 0))


def _comment(text: str) -> str:
    return "\n".join("# " + ln for ln in text.rstrip().split("\n"))


# ---------------------------------------------------------------------------
# Integración: agente para BudgetedResponder
# ---------------------------------------------------------------------------
SIG_RE = re.compile(r"^\s*(def\s+(\w+)\s*\(.*?\)\s*(?:->\s*[^:]+)?:)", re.M)
ASSERT_RE = re.compile(r"^\s*(assert\s+(\w+)\(.+)$", re.M)


def task_from_text(problem: str) -> CodeTask | None:
    """Extrae la función pedida del texto: una firma ``def f(...):`` o ejemplos ``assert f(...) == ...``."""
    sig = SIG_RE.search(problem)
    asserts = [m.group(1).strip() for m in ASSERT_RE.finditer(problem)]
    name = sig.group(2) if sig else (ASSERT_RE.search(problem).group(2) if asserts else None)
    if name is None:
        return None
    statement = ASSERT_RE.sub("", SIG_RE.sub("", problem)).strip()
    return CodeTask(statement, name, sig.group(1) if sig else "",
                    [a for a in asserts if f"{name}(" in a])


def autodebug_agent(debugger: SelfDebugger, name: str = "autodepurador"):
    """``Agent`` para ``BudgetedResponder``: si el problema pide una función, la escribe y depura.
    Su veredicto (``meta["veredicto"]``) sale de las pruebas en el sandbox."""
    from hope.council.agents import Agent, AgentProfile, Candidate

    class AutodebugAgent(Agent):
        def solve(self, topic: str, problem: str) -> Candidate:
            t0 = time.perf_counter()
            task = task_from_text(problem)
            if task is None:
                return Candidate(self.name, "no hay una función que escribir en el pedido", None,
                                 time.perf_counter() - t0, 0, {"veredicto": "incierto"})
            rep = debugger.solve(task)
            text = (f"```python\n{rep.code.rstrip()}\n```\nPruebas: anclas {rep.anchors_ok}/{rep.anchors}, "
                    f"generadas {rep.tests_ok}/{rep.tests}"
                    + (f", mutación {rep.mutation['puntaje']:.0%}" if rep.mutation else "")
                    + f"\nResultado: {'pasa las pruebas' if rep.ok else 'no pasa todas las pruebas'}")
            return Candidate(self.name, text, "pasa las pruebas" if rep.ok else "falla", time.perf_counter() - t0,
                             len(text.encode()), {"veredicto": rep.verdict, "autodepuracion": rep.as_dict()})

    prof = AgentProfile(name, "código que pasa sus pruebas", "Escribí, probá en el sandbox y corregí.",
                        use_tools=True, use_memory=False)
    return AutodebugAgent(prof, solver=lambda *a: ("", None))


# ---------------------------------------------------------------------------
# Integración: harness (carpeta de trabajo)
# ---------------------------------------------------------------------------
@dataclass
class FolderDebugReport:
    ok: bool
    edits: list                       # hope.harness.tools.Edit de la mejor ronda (propuestas, no escritas)
    rounds: list[dict]
    retrieved: list[dict]


def autodepurar_carpeta(folder, coder: Coder, request: str, path: str, test_command: str | list[str], *,
                        rounds: int = 3, k_context: int = 3) -> FolderDebugReport:
    """Bucle de autodepuración sobre un archivo de la carpeta del harness.

    En cada ronda el ``coder`` escribe bloques buscar/reemplazar; se aplican **en memoria**, se
    corre ``test_command`` en el sandbox sobre una copia con esas ediciones y, si falla, la salida
    de las pruebas vuelve al ``coder``. Nada se escribe en la carpeta: el resultado son ediciones
    propuestas (``FolderTools.apply_edit`` o ``apply_edits_checked`` las aplica con el guardián)."""
    from hope.tools.diff_edits import FORMAT_HELP, EditError
    from hope.tools.editor import EditorTools, contexto

    def check(p: str, op: str) -> None:
        folder._check("file.read", {"path": p, "op": op})

    editor = EditorTools(folder.root, check)
    retrieved = editor.recuperar(request + "\n" + path, k=k_context)
    current = folder.read(path)
    history, best = [], None
    fb = ""
    for n in range(rounds):
        prompt = (f"{contexto(retrieved)}\n# Archivo {path}:\n{current}\n# Pedido: {request}\n"
                  + (f"# Las pruebas fallaron:\n{_comment(fb[-1200:])}\n" if fb else "") + f"{FORMAT_HELP}\n")
        reply = coder.text(prompt, temperature=0.0 if n == 0 else 0.6, max_new=800, stop=None)
        try:
            edits = folder.propose_search_replace(reply, source="autodepurar")
        except EditError as e:
            fb = f"La edición no se pudo aplicar: {e}"
            history.append({"ronda": n, "ok": False, "error": str(e)})
            continue
        if not edits:
            fb = "No escribiste ningún bloque buscar/reemplazar."
            history.append({"ronda": n, "ok": False, "error": fb})
            continue
        r = folder.run(test_command, overlay={e.rel: e.new for e in edits})
        out = (r.stdout + r.stderr)[-2000:]
        history.append({"ronda": n, "ok": r.ok, "archivos": [e.rel for e in edits], "salida": out[-600:]})
        if r.ok:
            best = edits
            break
        fb = out
    return FolderDebugReport(best is not None, best or [], history, retrieved)
