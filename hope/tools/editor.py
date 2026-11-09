"""Herramientas de editor para Hope Code: definición, referencias, tipos, estructura y recuperación.

Lo que un programador mira **antes** de escribir: dónde está definida una función, quién la usa,
qué tipo tiene una expresión, qué símbolos hay en un archivo, y qué código parecido ya existe en
el proyecto (RepoCoder; Zhang et al. 2023: recuperar del propio repositorio, generar un
borrador y volver a recuperar con ese borrador).

Todo es de **solo lectura**. Cada operación pide permiso al guardián (``file.read``) a través de
``check(ruta, operación)``: en el harness es ``FolderTools._check`` (dentro de la carpeta se
permite; fuera, aprobación humana). El análisis de tipos corre en el sandbox sobre una copia.

* ``estructura(ruta)`` — clases, funciones, métodos y constantes (``ast``), de un archivo o de
  toda la carpeta.
* ``definicion(nombre | "ruta:línea:col")`` — con **jedi** si está instalado (sigue imports);
  si no, índice propio con ``ast`` (``Clase.metodo`` también sirve).
* ``referencias(...)`` — jedi o, si no está, recorrido del ``ast`` (no cuenta comentarios ni
  cadenas, a diferencia de un grep).
* ``tipos(ruta[, línea, expresión])`` — **pyright** o **mypy** si están: errores de tipos del
  archivo, o el tipo de una expresión (``reveal_type`` insertado en una copia).
* ``recuperar(consulta)`` — BM25 sobre funciones y clases del proyecto; ``recuperar_iterativo``
  repite con el borrador generado (RepoCoder).
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from hope.tools.sandbox import IGNORE_IN_COPY, run_command

MAX_FILES = 2000
MAX_FILE_BYTES = 400_000

Check = Callable[[str, str], None]          # (ruta absoluta, operación) → lanza si no hay permiso


def _no_check(path: str, op: str) -> None:
    return None


@dataclass
class Symbol:
    path: str
    kind: str                    # clase | funcion | metodo | variable
    name: str                    # nombre calificado (Clase.metodo)
    line: int
    end_line: int
    signature: str = ""
    doc: str = ""

    def as_dict(self) -> dict:
        return {"ruta": self.path, "tipo": self.kind, "nombre": self.name, "linea": self.line,
                "fin": self.end_line, "firma": self.signature, "doc": self.doc}


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
    except Exception:                                    # noqa: BLE001
        args = "…"
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"def {node.name}({args}){ret}"


def symbols_of(src: str, path: str = "") -> list[Symbol]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out: list[Symbol] = []

    def doc1(n) -> str:
        d = ast.get_docstring(n) or ""
        return d.strip().split("\n")[0][:120]

    def visit(body, prefix: str, in_class: bool):
        for n in body:
            if isinstance(n, ast.ClassDef):
                name = prefix + n.name
                bases = ", ".join(ast.unparse(b) for b in n.bases)
                out.append(Symbol(path, "clase", name, n.lineno, n.end_lineno or n.lineno,
                                  f"class {n.name}({bases})" if bases else f"class {n.name}", doc1(n)))
                visit(n.body, name + ".", True)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(Symbol(path, "metodo" if in_class else "funcion", prefix + n.name, n.lineno,
                                  n.end_lineno or n.lineno, _signature(n), doc1(n)))
            elif isinstance(n, (ast.Assign, ast.AnnAssign)) and not prefix:
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        out.append(Symbol(path, "variable", t.id, n.lineno, n.end_lineno or n.lineno,
                                          ast.unparse(n)[:120]))
    visit(tree.body, "", False)
    return out


def _tokens(text: str) -> list[str]:
    """Palabras e identificadores partidos (snake_case y camelCase), en minúsculas."""
    out = []
    for w in re.findall(r"[A-Za-zÁÉÍÓÚáéíóúñÑ_][\wáéíóúñÑ]*", text):
        for part in re.split(r"_|(?<=[a-z])(?=[A-Z])", w):
            if len(part) > 1:
                out.append(part.lower())
    return out


@dataclass
class Chunk:
    path: str
    name: str
    line: int
    code: str
    tokens: Counter = field(default_factory=Counter)


@dataclass
class EditorTools:
    root: Path
    check: Check = _no_check
    max_files: int = MAX_FILES
    _index: list[Chunk] | None = field(default=None, repr=False)

    def __post_init__(self):
        self.root = Path(self.root).resolve()

    # --- archivos -----------------------------------------------------------------
    def resolve(self, path: str | os.PathLike) -> Path:
        p = Path(os.path.expanduser(str(path)))
        return (p if p.is_absolute() else self.root / p).resolve()

    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    def _py_files(self, base: Path) -> list[Path]:
        if base.is_file():
            return [base]
        out = []
        for d, dirs, files in os.walk(base):
            dirs[:] = sorted(x for x in dirs if x not in IGNORE_IN_COPY and not x.startswith("."))
            for f in sorted(files):
                if f.endswith(".py"):
                    out.append(Path(d, f))
                    if len(out) >= self.max_files:
                        return out
        return out

    def _read(self, p: Path) -> str:
        if p.stat().st_size > MAX_FILE_BYTES:
            return ""
        return p.read_text(encoding="utf-8", errors="replace")

    # --- estructura ---------------------------------------------------------------
    def estructura(self, path: str = ".") -> list[dict]:
        p = self.resolve(path)
        self.check(str(p), "estructura")
        out = []
        for f in self._py_files(p):
            out += [s.as_dict() for s in symbols_of(self._read(f), self.rel(f))]
        return out

    def estructura_texto(self, path: str = ".", max_lines: int = 60) -> str:
        rows = self.estructura(path)
        lines = [f"{r['ruta']}:{r['linea']} {r['tipo']} {r['firma'] or r['nombre']}"
                 + (f"  # {r['doc']}" if r["doc"] else "") for r in rows[:max_lines]]
        if len(rows) > max_lines:
            lines.append(f"… ({len(rows) - max_lines} símbolos más)")
        return "\n".join(lines) or "sin símbolos"

    # --- definición y referencias --------------------------------------------------------
    @staticmethod
    def _position(query: str) -> tuple[str, int, int] | None:
        m = re.match(r"^(.+?\.py):(\d+)(?::(\d+))?$", query.strip())
        return (m.group(1), int(m.group(2)), int(m.group(3) or 0)) if m else None

    def _jedi_script(self, path: Path):
        try:
            import jedi
        except ImportError:
            return None
        return jedi.Script(self._read(path), path=str(path), project=jedi.Project(str(self.root)))

    def _snippet(self, path: Path, line: int, n: int = 6) -> str:
        try:
            lines = self._read(path).split("\n")
        except OSError:
            return ""
        return "\n".join(lines[line - 1:line - 1 + n])

    def definicion(self, query: str) -> list[dict]:
        """``query``: un nombre (``Clase.metodo``) o una posición ``ruta.py:línea[:columna]``."""
        pos = self._position(query)
        self.check(str(self.root if pos is None else self.resolve(pos[0])), "definicion")
        if pos is not None:
            path = self.resolve(pos[0])
            script = self._jedi_script(path)
            line, col = pos[1], pos[2]
            if script is not None:
                out = []
                for d in script.goto(line, col, follow_imports=True):
                    if d.module_path is None or d.line is None:
                        continue
                    mp = Path(d.module_path).resolve()
                    if self.root not in mp.parents:
                        out.append({"ruta": str(mp), "linea": d.line, "nombre": d.full_name or d.name,
                                    "firma": d.description, "fuera_de_la_carpeta": True})
                        continue
                    out.append({"ruta": self.rel(mp), "linea": d.line, "nombre": d.full_name or d.name,
                                "firma": d.description, "codigo": self._snippet(mp, d.line)})
                return out
            # sin jedi: el identificador en esa posición y búsqueda por nombre
            text = self._read(path).split("\n")
            row = text[line - 1] if 0 < line <= len(text) else ""
            words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[A-Za-z_]\w*", row)]
            hit = next((w for s, e, w in words if s <= col <= e), words[0][2] if words else "")
            query = hit
        name = query.strip()
        out = []
        for f in self._py_files(self.root):
            for s in symbols_of(self._read(f), self.rel(f)):
                if s.name == name or s.name.endswith("." + name) or (s.kind == "variable" and s.name == name):
                    out.append({"ruta": s.path, "linea": s.line, "nombre": s.name, "firma": s.signature,
                                "codigo": self._snippet(f, s.line, min(s.end_line - s.line + 1, 12))})
        return out

    def referencias(self, query: str, max_results: int = 50) -> list[dict]:
        pos = self._position(query)
        self.check(str(self.root if pos is None else self.resolve(pos[0])), "referencias")
        if pos is not None:
            path = self.resolve(pos[0])
            script = self._jedi_script(path)
            if script is not None:
                out = []
                for r in script.get_references(pos[1], pos[2]):
                    if r.module_path is None:
                        continue
                    mp = Path(r.module_path).resolve()
                    if mp != self.root and self.root not in mp.parents:
                        continue
                    out.append({"ruta": self.rel(mp), "linea": r.line, "texto": self._snippet(mp, r.line, 1).strip(),
                                "definicion": r.is_definition()})
                return out[:max_results]
            text = self._read(path).split("\n")
            row = text[pos[1] - 1] if 0 < pos[1] <= len(text) else ""
            words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[A-Za-z_]\w*", row)]
            query = next((w for s, e, w in words if s <= pos[2] <= e), words[0][2] if words else "")
        name = query.strip().split(".")[-1]
        out = []
        for f in self._py_files(self.root):
            src = self._read(f)
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            lines = src.split("\n")
            seen = set()
            for n in ast.walk(tree):
                hit = ((isinstance(n, ast.Name) and n.id == name) or (isinstance(n, ast.Attribute) and n.attr == name)
                       or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name)
                       or (isinstance(n, ast.alias) and (n.asname or n.name).split(".")[-1] == name)
                       or (isinstance(n, ast.arg) and n.arg == name))
                ln = getattr(n, "lineno", None)
                if hit and ln and ln not in seen:
                    seen.add(ln)
                    out.append({"ruta": self.rel(f), "linea": ln, "texto": lines[ln - 1].strip()[:160],
                                "definicion": isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))})
            if len(out) >= max_results:
                break
        return sorted(out, key=lambda r: (r["ruta"], r["linea"]))[:max_results]

    # --- tipos ----------------------------------------------------------------------
    @staticmethod
    def checker() -> str | None:
        for name in ("pyright", "mypy"):
            if shutil.which(name):
                return name
        return None

    def tipos(self, path: str, line: int | None = None, expr: str | None = None, *, timeout_s: float = 90.0) -> dict:
        """Errores de tipos del archivo o, con ``line`` y ``expr``, el tipo de la expresión en esa línea."""
        p = self.resolve(path)
        self.check(str(p), "tipos")
        tool = self.checker()
        if tool is None:
            return self._tipos_jedi(p, line, expr)
        rel = self.rel(p)
        src = self._read(p)
        overlay = {}
        probe_line = None
        if line is not None and expr:
            lines = src.split("\n")
            base = lines[line - 1] if 0 < line <= len(lines) else ""
            indent = re.match(r"\s*", base).group()
            if base.rstrip().endswith(":"):
                indent += "    "
            lines.insert(line, f"{indent}reveal_type({expr})")
            overlay[rel] = "\n".join(lines)
            probe_line = line + 1
        if tool == "pyright":
            argv = [shutil.which("pyright"), "--outputjson", rel]
        else:
            argv = [shutil.which("mypy"), "--no-incremental", "--cache-dir=/dev/null", "--show-column-numbers",
                    "--no-error-summary", "--ignore-missing-imports", "--follow-imports=silent", rel]
        # node (pyright) reserva mucha memoria virtual: con el límite de 1 GB del sandbox no arranca.
        r = run_command(argv, files=overlay, copy_tree=str(self.root), timeout_s=timeout_s,
                        mem_mb=4096 if tool == "pyright" else 1024)
        diags = self._parse_pyright(r.stdout) if tool == "pyright" else self._parse_mypy(r.stdout, rel)
        out = {"herramienta": tool, "ruta": rel, "tiempo_agotado": r.timed_out}
        if probe_line is not None:
            hit = next((d for d in diags if d["linea"] == probe_line and d["nivel"] in ("note", "information")), None)
            m = hit and re.search(r'(?:Revealed type is|Type of "[^"]*" is) "(.+)"', hit["mensaje"])
            out["tipo"] = m.group(1) if m else None
            out["expresion"] = expr
            diags = [d for d in diags if d["linea"] != probe_line]
            for d in diags:
                if d["linea"] > probe_line:
                    d["linea"] -= 1
        out["errores"] = [d for d in diags if d["nivel"] == "error"]
        out["ok"] = not out["errores"]
        return out

    @staticmethod
    def _parse_pyright(stdout: str) -> list[dict]:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return []
        return [{"linea": d["range"]["start"]["line"] + 1, "columna": d["range"]["start"]["character"] + 1,
                 "nivel": d["severity"], "mensaje": d["message"].split("\n")[0], "regla": d.get("rule", "")}
                for d in data.get("generalDiagnostics", [])]

    @staticmethod
    def _parse_mypy(stdout: str, rel: str) -> list[dict]:
        out = []
        for m in re.finditer(r"^(.+?):(\d+):(?:(\d+):)? (error|note|warning): (.+?)(?:  \[([\w-]+)\])?$", stdout, re.M):
            if Path(m.group(1)).name != Path(rel).name:
                continue
            out.append({"linea": int(m.group(2)), "columna": int(m.group(3) or 0), "nivel": m.group(4),
                        "mensaje": m.group(5), "regla": m.group(6) or ""})
        return out

    def _tipos_jedi(self, p: Path, line: int | None, expr: str | None) -> dict:
        script = self._jedi_script(p)
        if script is None or line is None or not expr:
            return {"herramienta": None, "ok": False, "error": "sin pyright, mypy ni jedi", "errores": []}
        row = self._read(p).split("\n")[line - 1]
        col = row.find(expr.split("(")[0].split(".")[-1])
        names = script.infer(line, max(col, 0))
        return {"herramienta": "jedi", "ruta": self.rel(p), "expresion": expr, "errores": [], "ok": True,
                "tipo": ", ".join(sorted({n.name for n in names})) or None}

    # --- recuperación por repositorio (RepoCoder) -------------------------------------
    def _build_index(self) -> list[Chunk]:
        if self._index is not None:
            return self._index
        chunks = []
        for f in self._py_files(self.root):
            src = self._read(f)
            lines = src.split("\n")
            for s in symbols_of(src, self.rel(f)):
                if s.kind not in ("funcion", "metodo", "clase") or (s.kind == "clase" and s.end_line - s.line > 60):
                    continue
                code = "\n".join(lines[s.line - 1:s.end_line])[:4000]
                chunks.append(Chunk(s.path, s.name, s.line, code, Counter(_tokens(code))))
        self._index = chunks
        return chunks

    def recuperar(self, consulta: str, k: int = 3, *, exclude: str | None = None) -> list[dict]:
        """BM25 sobre las funciones y clases del proyecto. ``exclude``: nombre a omitir (la propia función)."""
        self.check(str(self.root), "recuperar")
        chunks = self._build_index()
        if not chunks:
            return []
        q = _tokens(consulta)
        n = len(chunks)
        avg = sum(sum(c.tokens.values()) for c in chunks) / n
        df = Counter(t for c in chunks for t in set(c.tokens))
        scored = []
        for c in chunks:
            if exclude and (c.name == exclude or c.name.endswith("." + exclude)):
                continue
            dl = sum(c.tokens.values())
            s = 0.0
            for t in set(q):
                tf = c.tokens.get(t, 0)
                if tf:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / avg))
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda t: -t[0])
        return [{"ruta": c.path, "linea": c.line, "nombre": c.name, "codigo": c.code, "puntaje": round(s, 3)}
                for s, c in scored[:k]]

    def recuperar_iterativo(self, consulta: str, generate: Callable[[str], str] | None = None, *,
                            k: int = 3, rondas: int = 2) -> list[dict]:
        """RepoCoder: recuperar → generar un borrador con ese contexto → recuperar con el borrador."""
        found = self.recuperar(consulta, k)
        for _ in range(max(rondas - 1, 0)):
            if generate is None:
                break
            draft = generate(contexto(found) + "\n" + consulta)
            more = self.recuperar(consulta + "\n" + draft, k)
            seen = {(r["ruta"], r["linea"]) for r in found}
            found = found + [r for r in more if (r["ruta"], r["linea"]) not in seen]
            found = sorted(found, key=lambda r: -r["puntaje"])[:k]
        return found


def contexto(found: list[dict], max_bytes: int = 3000) -> str:
    """Texto con el código recuperado, para anteponer al pedido."""
    parts, used = [], 0
    for r in found:
        block = f"# De {r['ruta']}:{r['linea']}\n{r['codigo']}\n"
        if used + len(block.encode()) > max_bytes:
            break
        parts.append(block)
        used += len(block.encode())
    return "\n".join(parts)


def guardian_check(guardian, requested_by: str = "hte") -> Check:
    """``check`` que pide ``file.read`` al guardián para cada ruta (fuera del harness)."""
    from guardian.core import Action
    from hope.tools.catalog import ToolDenied

    def check(path: str, op: str) -> None:
        d = guardian.decide(Action("file.read", {"path": path, "op": op}, requested_by=requested_by))
        if not d.allowed:
            raise ToolDenied("; ".join(d.reasons) or "lectura denegada")
    return check
