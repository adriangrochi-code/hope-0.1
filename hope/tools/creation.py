"""TCG/TVE — creación y validación independiente de herramientas (Herramientas §4–§5, §18; ARQUITECTURA_HN1 §10.4).

Cuando Hope detecta una tarea que repite sin herramienta (o lo pide con
``⟦crear_herramienta: descripción⟧``), le pide al maestro (Qwen) una herramienta
completa: **especificación** (nombre, entradas y salidas tipadas, dominio),
**código Python** y **pruebas**. El código del maestro es **no confiable**.

TVE la valida antes de registrarla, sin creerle al maestro:

1. **especificación**: nombre válido, tipos conocidos, la función existe con esas entradas;
2. **revisión de seguridad** (AST): solo módulos de una lista permitida; sin red,
   archivos, procesos, ``eval``/``exec``/``__import__`` ni atributos ``__dunder__``;
3. **análisis estático** con ruff, si está instalado (errores graves la rechazan);
4. **pruebas propias** del maestro, en el sandbox, con límites de tiempo y memoria;
5. **pruebas independientes**: casos conocidos que no escribió el maestro
   (los aporta Hope o quien pide la herramienta) se convierten en pruebas
   generadas aquí. Una herramienta no se valida solo con sus propias pruebas (§18).

Si pasa, queda en el catálogo como **EXPERIMENTAL** con su ficha y su
procedencia (maestro, pedido, hash del código). Se ejecuta siempre en el
sandbox, en un proceso aparte, a través del guardián. Consolidarla (ESTABLE)
exige **aprobación humana** a través del guardián (acción ``tool.consolidate``,
nivel 3), atada al hash exacto del código.

Si falla, el motivo se guarda y se le muestra al maestro en el próximo intento.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from guardian.core import Action, Decision, Guardian
from hope.epistemic import LifecycleStatus
from hope.tools.catalog import Catalog, ToolSpec
from hope.tools.sandbox import run_command

# Módulos que una herramienta creada puede importar: cálculo puro, sin efectos.
ALLOWED_MODULES = {
    "math", "cmath", "fractions", "decimal", "statistics", "itertools", "functools", "collections",
    "re", "datetime", "json", "string", "operator", "heapq", "bisect", "typing", "dataclasses",
    "unicodedata", "numpy", "sympy",
}
# En las pruebas, además, el framework y el módulo bajo prueba.
ALLOWED_IN_TESTS = ALLOWED_MODULES | {"pytest", "solucion"}
FORBIDDEN_CALLS = {
    "open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "input",
    "breakpoint", "getattr", "setattr", "delattr", "memoryview", "help", "exit", "quit",
}
# Tipos admitidos en las fichas (JSON). «number» acepta int y float.
KNOWN_TYPES = {"str", "int", "float", "number", "bool", "list", "dict", "any", "none"}
NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{1,40}$")
SEVERE_LINT = ("E9", "F63", "F7", "F82")       # sintaxis, comparaciones inválidas, nombres indefinidos

TOOL_REQUEST = """Diseñá una herramienta de software para Hope.
HERRAMIENTA: {description}
{known}{failures}
Respondé SOLO con un bloque ```json con estos campos:
- "nombre": identificador en minúsculas con guiones bajos;
- "funcion": nombre de la función Python principal;
- "dominio", "objetivo", "descripcion": textos breves en español;
- "entradas": objeto {{parámetro: tipo}} y "salidas": objeto {{nombre: tipo}} con un solo campo;
  tipos admitidos: str, int, float, number, bool, list, dict;
- "codigo": módulo Python que define la función. Solo biblioteca estándar de cálculo
  (math, fractions, decimal, statistics, re, itertools, collections, ...), numpy o sympy.
  Sin red, sin archivos, sin procesos, sin eval/exec;
- "pruebas": pruebas pytest que importan la función con «from solucion import <funcion>»;
- "limitaciones": lista de textos.
"""


class ToolCreationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
@dataclass
class ToolProposal:
    nombre: str
    funcion: str
    dominio: str
    objetivo: str
    descripcion: str
    entradas: dict
    salidas: dict
    codigo: str
    pruebas: str
    limitaciones: list = field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> ToolProposal:
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        candidate = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            raise ToolCreationError(f"la respuesta del maestro no es JSON válido: {e}") from e
        missing = [k for k in ("nombre", "funcion", "entradas", "salidas", "codigo", "pruebas") if k not in data]
        if missing:
            raise ToolCreationError(f"faltan campos en la propuesta: {', '.join(missing)}")
        return cls(nombre=str(data["nombre"]), funcion=str(data["funcion"]),
                   dominio=str(data.get("dominio", "")), objetivo=str(data.get("objetivo", "")),
                   descripcion=str(data.get("descripcion", "")), entradas=dict(data["entradas"]),
                   salidas=dict(data["salidas"]), codigo=str(data["codigo"]), pruebas=str(data["pruebas"]),
                   limitaciones=list(data.get("limitaciones", [])))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.codigo.encode()).hexdigest()


@dataclass
class ValidationReport:
    ok: bool
    reasons: list[str]
    checks: dict                     # etapa → "ok" | "falla" | "omitido" (+ detalle)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tipos
def type_alternatives(t: str) -> set[str]:
    t = str(t).strip().lower().rstrip("?")
    alts = {a.strip() for a in t.split("|") if a.strip()}
    return {"float" if a == "decimal" else a for a in alts}


def _is_number(v) -> bool:
    from decimal import Decimal
    from fractions import Fraction
    return isinstance(v, (int, float, Decimal, Fraction)) and not isinstance(v, bool)


def value_has_type(value, t: str) -> bool:
    alts = type_alternatives(t)
    if "any" in alts:
        return True
    checks = {
        "str": lambda v: isinstance(v, str),
        "bool": lambda v: isinstance(v, bool),
        "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "float": _is_number,
        "number": _is_number,
        "list": lambda v: isinstance(v, list),
        "dict": lambda v: isinstance(v, dict),
        "none": lambda v: v is None,
    }
    return any(checks.get(a, lambda v: False)(value) for a in alts)


# ---------------------------------------------------------------------------
# Revisión de seguridad (estática, sobre el AST)
def security_review(code: str, *, allowed_modules: set[str] = ALLOWED_MODULES) -> list[str]:
    """Devuelve los problemas encontrados (vacío = sin problemas).

    Es una revisión **por lista permitida**: lo que no está explícitamente
    permitido se rechaza. No reemplaza al sandbox; se suma a él.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"error de sintaxis en la línea {e.lineno}: {e.msg}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in allowed_modules:
                    problems.append(f"importa un módulo no permitido: {a.name} (línea {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if node.level or mod not in allowed_modules:
                problems.append(f"importa un módulo no permitido: {node.module or '.'} (línea {node.lineno})")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            problems.append(f"llamada prohibida: {node.func.id}() (línea {node.lineno})")
        elif isinstance(node, ast.Name) and node.id in {"__builtins__", "__loader__", "__spec__"}:
            problems.append(f"acceso prohibido: {node.id} (línea {node.lineno})")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            problems.append(f"acceso a atributo interno: .{node.attr} (línea {node.lineno})")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"estado global mutable (línea {node.lineno})")
    return problems


def _function_signature_problems(p: ToolProposal) -> list[str]:
    problems = []
    if not NAME_RE.match(p.nombre):
        problems.append(f"nombre inválido: «{p.nombre}» (minúsculas, dígitos y guiones bajos)")
    if not NAME_RE.match(p.funcion):
        problems.append(f"nombre de función inválido: «{p.funcion}»")
    if not p.entradas:
        problems.append("la herramienta no declara entradas")
    if len(p.salidas) != 1:
        problems.append("la herramienta debe declarar exactamente una salida")
    for k, t in {**p.entradas, **p.salidas}.items():
        if not type_alternatives(t) <= KNOWN_TYPES:
            problems.append(f"tipo desconocido para «{k}»: {t}")
    try:
        tree = ast.parse(p.codigo)
    except SyntaxError:
        return problems                                 # lo informa la revisión de seguridad
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == p.funcion), None)
    if fn is None:
        problems.append(f"el código no define la función «{p.funcion}»")
    else:
        params = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        missing = [k for k in p.entradas if k not in params]
        if missing:
            problems.append(f"la función no acepta las entradas declaradas: {', '.join(missing)}")
    return problems


# ---------------------------------------------------------------------------
# Ejecución en el sandbox
_RUNNER = """import json, os, sys
sys.path.insert(0, os.getcwd())
from solucion import {fn}
args = json.loads(sys.stdin.read())
print("__RESULTADO__" + json.dumps({fn}(**args), ensure_ascii=False))
"""


def run_created_tool(p: ToolProposal, args: dict, *, timeout_s: float = 5.0, mem_mb: int = 1024):
    """Ejecuta la herramienta en un proceso aparte del sandbox y devuelve su resultado (JSON)."""
    r = run_command([sys.executable, "-I", "ejecutar.py"],
                    files={"solucion.py": p.codigo, "ejecutar.py": _RUNNER.format(fn=p.funcion)},
                    stdin=json.dumps(args), timeout_s=timeout_s, mem_mb=mem_mb)
    if r.timed_out:
        raise TimeoutError(f"la herramienta superó el límite de {timeout_s:g} s")
    if not r.ok or "__RESULTADO__" not in r.stdout:
        last = (r.stderr.strip().splitlines() or ["sin salida"])[-1]
        raise RuntimeError(f"la herramienta falló: {last[:200]}")
    return json.loads(r.stdout.split("__RESULTADO__", 1)[1])


def _pytest(code: str, tests: str, *, timeout_s: float, mem_mb: int) -> dict:
    r = run_command([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_solucion.py"],
                    files={"solucion.py": code, "test_solucion.py": tests}, timeout_s=timeout_s, mem_mb=mem_mb)
    out = (r.stdout + r.stderr).strip()
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    return {"ok": r.ok and passed > 0, "passed": passed, "failed": failed, "timed_out": r.timed_out,
            "memory": "MemoryError" in out, "output": out[-600:]}


def known_case_tests(p: ToolProposal, cases: list[dict]) -> str:
    """Pruebas generadas acá (no por el maestro) a partir de casos conocidos ``{"entrada": {...}, "salida": ...}``."""
    lines = [f"from solucion import {p.funcion}", "import math", "",
             "def _igual(a, b):",
             "    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):",
             "        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)",
             "    return a == b", ""]
    for i, c in enumerate(cases):
        args = json.dumps(c["entrada"], ensure_ascii=False)
        want = json.dumps(c["salida"], ensure_ascii=False)
        lines += [f"def test_caso_conocido_{i}():",
                  "    import json as _j",
                  f"    got = {p.funcion}(**_j.loads({args!r}))",
                  f"    assert _igual(got, _j.loads({want!r})), got", ""]
    return "\n".join(lines)


def validate(p: ToolProposal, known_cases: list[dict] | None = None, *, timeout_s: float = 30.0,
             mem_mb: int = 1024) -> ValidationReport:
    """TVE: todas las etapas; la herramienta se acepta solo si ninguna falla."""
    reasons, warnings, checks = [], [], {}

    spec_problems = _function_signature_problems(p)
    checks["especificacion"] = "falla" if spec_problems else "ok"
    reasons += spec_problems

    sec = security_review(p.codigo) + [f"pruebas: {x}" for x in
                                       security_review(p.pruebas, allowed_modules=ALLOWED_IN_TESTS)]
    checks["seguridad"] = "falla" if sec else "ok"
    reasons += [f"seguridad: {x}" for x in sec]
    if reasons:                       # código inseguro o mal especificado: no se ejecuta nada
        return ValidationReport(False, reasons, checks, warnings)

    from hope.tools.code_tools import lint
    lr = lint(p.codigo)
    if lr.get("error"):
        checks["estatico"] = "omitido"
        warnings.append(f"análisis estático omitido: {lr['error']}")
    else:
        severe = [i for i in lr["issues"] if i["code"].startswith(SEVERE_LINT)]
        checks["estatico"] = "falla" if severe else "ok"
        reasons += [f"estático: línea {i['line']}: {i['code']} {i['message']}" for i in severe]
        warnings += [f"estilo: línea {i['line']}: {i['code']} {i['message']}"
                     for i in lr["issues"] if i not in severe]

    def _stage(name: str, tests: str) -> None:
        r = _pytest(p.codigo, tests, timeout_s=timeout_s, mem_mb=mem_mb)
        checks[name] = "ok" if r["ok"] else "falla"
        if r["timed_out"]:
            reasons.append(f"{name}: superó el límite de tiempo ({timeout_s:g} s)")
        elif r["memory"]:
            reasons.append(f"{name}: superó el límite de memoria ({mem_mb} MB)")
        elif not r["ok"]:
            tail = [ln for ln in r["output"].splitlines() if ln.strip()][-3:]
            reasons.append(f"{name}: {r['passed']} pasaron, {r['failed']} fallaron — " + " | ".join(tail)[:300])

    _stage("pruebas_propias", p.pruebas)
    if known_cases:
        _stage("pruebas_independientes", known_case_tests(p, known_cases))
    else:
        checks["pruebas_independientes"] = "omitido"
        warnings.append("sin casos conocidos: falta validación independiente (queda solo EXPERIMENTAL)")
    return ValidationReport(not reasons, reasons, checks, warnings)


# ---------------------------------------------------------------------------
@dataclass
class ToolRecord:
    tool_id: str
    proposal: ToolProposal
    report: ValidationReport
    request: str
    teacher_model: str
    created: float
    status: str = LifecycleStatus.EXPERIMENTAL.value


class ToolForge:
    """Crea herramientas con el maestro, las valida (TVE) y las registra como EXPERIMENTAL.

    ``store_dir`` (opcional) guarda código, fichas y fallas, para retomarlas entre sesiones.
    """

    def __init__(self, catalog: Catalog, teacher, *, guardian: Guardian | None = None,
                 store_dir: str | Path | None = None, repeat_threshold: int = 3,
                 timeout_s: float = 30.0, mem_mb: int = 1024):
        self.catalog = catalog
        self.teacher = teacher
        self.guardian = guardian if guardian is not None else catalog.guardian
        self.store = Path(store_dir) if store_dir else None
        self.repeat_threshold = repeat_threshold
        self.timeout_s, self.mem_mb = timeout_s, mem_mb
        self.records: dict[str, ToolRecord] = {}
        self.failures: dict[str, list[str]] = {}       # pedido → motivos de los intentos fallidos
        self.task_counts: dict[str, int] = {}
        if self.store:
            self._load()

    # --- detección de tareas repetidas -----------------------------------------
    @staticmethod
    def _key(text: str) -> str:
        return " ".join(re.findall(r"[\wáéíóúñü]+", text.lower()))

    def has_tool_for(self, task: str, min_overlap: float = 0.5) -> bool:
        words = {w for w in self._key(task).split() if len(w) > 3}
        if not words:
            return True
        for s in self.catalog.search(task):
            text = f"{s.objective} {s.description} {s.domain} {s.name}".lower()
            if sum(w in text for w in words) / len(words) >= min_overlap:
                return True
        return False

    def observe(self, task: str) -> bool:
        """Registra que Hope hizo ``task`` a mano. True si conviene pedir una herramienta."""
        k = self._key(task)
        self.task_counts[k] = self.task_counts.get(k, 0) + 1
        return self.task_counts[k] >= self.repeat_threshold and not self.has_tool_for(task)

    # --- creación ---------------------------------------------------------------
    def request_messages(self, description: str, known_cases: list[dict] | None = None) -> list[dict]:
        known = ("CASOS CONOCIDOS (se usarán para validarla, no para copiarlos en el código):\n" +
                 "\n".join(json.dumps(c, ensure_ascii=False) for c in known_cases[:5]) + "\n"
                 if known_cases else "")
        prev = self.failures.get(self._key(description), [])
        failures = ("FALLAS ANTERIORES (corregilas):\n" + "\n".join(f"- {r}" for r in prev[-8:]) + "\n"
                    if prev else "")
        return [{"role": "system", "content": "Sos el maestro de Hope. Diseñás herramientas seguras y verificables."},
                {"role": "user", "content": TOOL_REQUEST.format(description=description, known=known,
                                                                failures=failures)}]

    def _new_id(self, nombre: str) -> str:
        v = 1
        while f"tcg_{nombre}_v{v}" in self.catalog:
            v += 1
        return f"tcg_{nombre}_v{v}"

    def create(self, description: str, known_cases: list[dict] | None = None) -> ToolRecord:
        """Pide la herramienta al maestro, la valida y, si pasa, la registra como EXPERIMENTAL.

        Si falla, lanza ``ToolCreationError`` con los motivos (y quedan guardados para el próximo intento).
        """
        key = self._key(description)
        reply = self.teacher.ask(self.request_messages(description, known_cases), purpose="tcg")
        try:
            proposal = ToolProposal.from_text(reply.text)
        except ToolCreationError as e:
            self._fail(key, [str(e)])
            raise
        report = validate(proposal, known_cases, timeout_s=self.timeout_s, mem_mb=self.mem_mb)
        if not report.ok:
            self._fail(key, report.reasons)
            raise ToolCreationError("herramienta rechazada: " + "; ".join(report.reasons))
        rec = ToolRecord(self._new_id(proposal.nombre), proposal, report, description,
                         getattr(reply, "model", "") or "", time.time())
        self._register(rec)
        self.records[rec.tool_id] = rec
        self.failures.pop(key, None)
        self._save()
        return rec

    def _fail(self, key: str, reasons: list[str]) -> None:
        self.failures.setdefault(key, []).extend(reasons)
        self._save()

    def _register(self, rec: ToolRecord) -> None:
        p = rec.proposal
        spec = ToolSpec(
            rec.tool_id, p.nombre.replace("_", " "), "1.0", p.dominio or "creada", p.objetivo or p.descripcion,
            p.descripcion or rec.request, dict(p.entradas), dict(p.salidas), status=LifecycleStatus(rec.status),
            execution="sandbox", cost="medio (proceso aparte)",
            limitations=list(p.limitaciones),
            risks=["código generado por el maestro: no confiable hasta la aprobación humana"],
            permissions=["sandbox"],
            tests=[f"pruebas del maestro ({rec.report.checks.get('pruebas_propias')})",
                   f"casos conocidos ({rec.report.checks.get('pruebas_independientes')})"],
            changelog=[(f"{time.strftime('%Y-%m-%d', time.localtime(rec.created))}: creada por TCG, "
                        f"validada por TVE ({', '.join(f'{k}={v}' for k, v in rec.report.checks.items())})")],
            provenance=(f"TCG: maestro «{rec.teacher_model}», pedido «{rec.request}», "
                        f"código sha256={p.sha256[:16]}"))
        timeout, mem = min(self.timeout_s, 10.0), self.mem_mb

        def fn(**kwargs):
            for k, t in p.entradas.items():
                if k not in kwargs:
                    raise TypeError(f"falta la entrada «{k}»")
                if not value_has_type(kwargs[k], t):
                    raise TypeError(f"la entrada «{k}» debe ser {t}")
            out = run_created_tool(p, kwargs, timeout_s=timeout, mem_mb=mem)
            t_out = next(iter(p.salidas.values()))
            if not value_has_type(out, t_out):
                raise TypeError(f"la salida no es {t_out}: {out!r}"[:200])
            return out
        self.catalog.register(spec, fn)

    # --- consolidación (aprobación humana) --------------------------------------
    def consolidation_action(self, tool_id: str) -> Action:
        rec = self.records[tool_id]
        return Action("tool.consolidate", {"tool": tool_id, "version": self.catalog.spec(tool_id).version,
                                           "code_sha256": rec.proposal.sha256},
                      requested_by="tcg", justification=f"consolidar herramienta creada: {rec.request}")

    def consolidate(self, tool_id: str, *, approval_id: str | None = None) -> Decision:
        """Pide pasar la herramienta a ESTABLE. Sin aprobación humana queda una solicitud pendiente."""
        if tool_id not in self.records:
            raise KeyError(f"no es una herramienta creada por TCG: {tool_id}")
        if self.guardian is None:
            raise PermissionError("consolidar una herramienta creada requiere el guardián y aprobación humana")
        rec = self.records[tool_id]
        if rec.status != LifecycleStatus.EXPERIMENTAL.value:
            raise ValueError(f"{tool_id} está {rec.status}, no EXPERIMENTAL")
        if rec.report.checks.get("pruebas_independientes") != "ok":
            raise ValueError(f"{tool_id} no pasó pruebas independientes: no se puede consolidar")
        d = self.guardian.decide(self.consolidation_action(tool_id), approval_id=approval_id)
        if d.allowed:
            rec.status = LifecycleStatus.ESTABLE.value
            spec = self.catalog.spec(tool_id)
            spec.status = LifecycleStatus.ESTABLE
            spec.changelog.append(f"{time.strftime('%Y-%m-%d')}: consolidada con aprobación humana ({approval_id})")
            self._save()
        return d

    # --- persistencia -----------------------------------------------------------
    def _save(self) -> None:
        if not self.store:
            return
        self.store.mkdir(parents=True, exist_ok=True)
        data = {"herramientas": {tid: {**asdict(r), "proposal": asdict(r.proposal), "report": asdict(r.report)}
                                 for tid, r in self.records.items()},
                "fallas": self.failures}
        (self.store / "registro.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        for tid, r in self.records.items():
            (self.store / f"{tid}.py").write_text(r.proposal.codigo, encoding="utf-8")

    def _load(self) -> None:
        f = self.store / "registro.json"
        if not f.exists():
            return
        data = json.loads(f.read_text(encoding="utf-8"))
        self.failures = data.get("fallas", {})
        for tid, d in data.get("herramientas", {}).items():
            p = ToolProposal(**d["proposal"])
            if p.sha256 != hashlib.sha256((self.store / f"{tid}.py").read_text(encoding="utf-8").encode()).hexdigest():
                continue                         # el código cambió en disco: no se carga sin revalidar
            rec = ToolRecord(tid, p, ValidationReport(**d["report"]), d["request"], d["teacher_model"],
                             d["created"], d["status"])
            if tid not in self.catalog:
                self._register(rec)
            self.records[tid] = rec
