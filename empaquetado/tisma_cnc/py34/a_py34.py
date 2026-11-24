"""Translates the TISMA CNC code to Python 3.4, the last official Python for Windows XP.

Run when building the package (needs Python 3.9+ for ``ast.unparse``); the source code stays
written in modern Python. What changes:

* f-strings → ``"...".format(...)``
* type annotations: removed (in ``@dataclass`` classes the field list is kept in ``__dc_fields__``
  for the dataclasses version in ``_tisma34``)
* ``{**a, **b}`` → ``_tc_merge(a, b)``; ``[*a, *b]`` → ``_tc_list(a, b)``
* calls with several ``*``/``**`` or arguments after ``*`` → a single list/dictionary
* ``from __future__ import annotations`` is dropped; each module imports ``_tisma34`` first

What it cannot translate (walrus, positional-only parameters, async…) is rejected with an error:
that way a change that would break XP shows up at build time and not on the customer's PC.

How it fits: used by ``construir.py`` (``armar_xp``: translates the copied app into the XP package
folder) and ``preparar_pruebas.py`` (translates the app and its tests to run them with a real Python
3.4); the tests ``tests/test_cnc_py34.py`` call ``traducir`` directly. The translated code needs
``_tisma34.py`` next to it (copied untranslated: it is already written for 3.4).

How it works: each ``.py`` is read with ``ast.parse`` (with the modern Python of the build PC), an
``ast.NodeTransformer`` (``A34``) rewrites the nodes 3.4 does not understand, ``ast.unparse`` writes
source code again and ``_revisar`` reads it back to check that nothing modern is left. Comments are
lost (``ast`` does not keep them), docstrings are not: the translated code is never edited, it is
regenerated.

Example:

    >>> print(traducir('x: int = 1\\nprint(f"{x:>3}")'))
    # -*- coding: utf-8 -*-
    # Traducido a Python 3.4 por empaquetado/tisma_cnc/py34/a_py34.py: no editar.
    import _tisma34
    x = 1
    print('{0:>3}'.format(x))

Uso: python a_py34.py ORIGEN DESTINO   (carpetas; copia todo y traduce los .py)
"""
# Note: main() prints the LAST line of this docstring as usage help: it must remain the «Uso:» (Usage) line,
# in Spanish because the user sees it.

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

# ==================================================================================================
# ======================================== CONFIGURATION =========================================
# ==================================================================================================

# Name of the compatibility module imported by each translated file (py34/_tisma34.py). If the file is
# renamed, change it here too and in construir.py / preparar_pruebas.py, which copy it.
SHIM = "_tisma34"


# Translation error: the code uses something Python 3.4 lacks and that this translator cannot rewrite
# (walrus, async, match, positional-only parameters, typing outside annotations…). The message gives
# file and line. Stopping the build is deliberate: it is better than failing on the XP PC.
class NoTraducible(Exception):
    pass


# ==================================================================================================
# ===================================== NODE-BUILDING HELPERS ======================================
# ==================================================================================================

def _name(n: str) -> ast.Name:
    """Load-context ``ast.Name`` node for the name ``n`` (e.g. ``_tc_merge``, ``tuple``)."""
    return ast.Name(id=n, ctx=ast.Load())


def _call(fn: str, args: list[ast.expr], keywords: list[ast.keyword] | None = None) -> ast.Call:
    """Call node ``fn(*args, **keywords)`` to a function by name (no attributes)."""
    return ast.Call(func=_name(fn), args=args, keywords=keywords or [])


def _is_dataclass(cls: ast.ClassDef) -> bool:
    """Does the class carry the ``@dataclass``, ``@dataclass(...)`` or ``@dataclasses.dataclass`` decorator?

    Only the name (``dataclass``) is checked, not where it was imported from: enough for TISMA CNC's code.
    """
    for d in cls.decorator_list:
        f = d.func if isinstance(d, ast.Call) else d
        if (isinstance(f, ast.Name) and f.id == "dataclass") or (isinstance(f, ast.Attribute) and f.attr == "dataclass"):
            return True
    return False


# ==================================================================================================
# ========================================== TRANSLATOR ==========================================
# ==================================================================================================

class A34(ast.NodeTransformer):
    """Walks the syntax tree and rewrites what Python 3.4 does not understand (one ``visit_X`` per node).

    Each ``visit_X`` method receives a node of type X and returns the node that replaces it (the same
    one, another, or ``None`` to delete it), as ``ast.NodeTransformer`` requires. Calling
    ``generic_visit`` first makes the children get translated before the parent (e.g. an f-string
    inside a ``{**a}``).

    Attribute ``usa``: names from ``_tisma34`` the module needs (``_tc_merge``, ``_tc_list``);
    ``traducir`` imports them at the top.
    """

    def __init__(self) -> None:
        """Starts with no ``_tisma34`` helper in use."""
        self.usa: set[str] = set()

    # --- syntax that is not translated ------------------------------------------------------------
    def visit_NamedExpr(self, node):
        """``x := …`` has no direct equivalent in a 3.4 expression: rejected."""
        raise NoTraducible(f"línea {node.lineno}: ':=' (Python 3.8) no existe en 3.4")

    def visit_AsyncFunctionDef(self, node):
        """``async def`` does not exist in 3.4 (it would be a different program): rejected."""
        raise NoTraducible(f"línea {node.lineno}: async def (Python 3.5) no existe en 3.4")

    def visit_MatchAs(self, node):  # pragma: no cover - any match
        """``match``/``case`` (3.10) does not exist in 3.4: rejected. It hooks into ``MatchAs``, the capture
        pattern (``case _``, ``case x``, ``… as y``), which appears in almost every ``match``."""
        raise NoTraducible(f"línea {node.lineno}: match (Python 3.10)")

    # --- annotations ------------------------------------------------------------------------------
    def _args(self, a: ast.arguments) -> None:
        """Removes the annotations of all parameters (they exist in 3.4, but may name types such as
        ``list[str]`` or ``X | None`` that fail in 3.4 when the function is defined). Rejects ``/``."""
        if a.posonlyargs:
            raise NoTraducible("parámetros solo posicionales '/' (Python 3.8)")
        for arg in a.args + a.kwonlyargs + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else []):
            arg.annotation = None

    def visit_FunctionDef(self, node):
        """Function: no annotations on the parameters or the return value; then, its body."""
        self._args(node.args)
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_Lambda(self, node):
        """Lambda: it has no annotations; only what is inside is translated."""
        self.generic_visit(node)
        return node

    def visit_If(self, node):
        """``if TYPE_CHECKING:`` is removed (its ``else`` remains, if any): those imports are only for the
        type checker and might not exist in 3.4 (``typing``)."""
        t = node.test
        if (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"):
            return node.orelse or None                     # only for the type checker
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        """Class: removes the attribute annotations and, if it is a ``@dataclass``, keeps the field list.

        Why: variable annotations do not exist in 3.4 (``x: int = 0`` is from 3.6), so the ``dataclass``
        in ``_tisma34`` has nowhere to get the fields from. They are added at the start of the class
        (after the docstring):

        * ``__dc_fields__ = ('a', 'b')``: the fields in order (without the ``ClassVar`` ones), for
          ``_tisma34``;
        * ``__annotations__ = {'a': object, …}``: for the real ``dataclasses`` (3.7+), so the translated
          code runs the same on a newer Python (the tests take advantage of that).

        Then each ``x: T = v`` becomes ``x = v`` and each ``x: T`` without a value is removed (if the class
        ends up empty, ``pass`` is added).
        """
        if _is_dataclass(node):
            names = [s.target.id for s in node.body
                     if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and s.simple
                     and not _is_classvar(s.annotation)]
            fields = ast.Assign(targets=[ast.Name(id="__dc_fields__", ctx=ast.Store())],
                                value=ast.Tuple(elts=[ast.Constant(n) for n in names], ctx=ast.Load()))
            # __annotations__ for the real dataclasses (3.7+): so the translated code runs on any Python
            anns = ast.Assign(targets=[ast.Name(id="__annotations__", ctx=ast.Store())],
                              value=ast.Dict(keys=[ast.Constant(n) for n in names],
                                             values=[_name("object") for _ in names]))
            pos = 1 if node.body and _is_docstring(node.body[0]) else 0
            node.body[pos:pos] = [fields, anns]
        self.generic_visit(node)
        node.body = [self._ann(s) for s in node.body]
        node.body = [s for s in node.body if s is not None] or [ast.Pass()]
        return node

    @staticmethod
    def _ann(s):
        """Annotated class attribute → plain assignment (``x: T = v`` → ``x = v``) or ``None`` if it has no
        value (``x: T``), so it gets removed. Everything else stays the same."""
        if isinstance(s, ast.AnnAssign):
            if s.value is None:
                return None
            return ast.Assign(targets=[s.target], value=s.value)
        return s

    def visit_AnnAssign(self, node):
        """Annotated variable outside a class (e.g. ``self.usa: set[str] = set()``): ``x = v``.

        Without a value (``x: int``) it does nothing at run time: it is replaced by ``pass`` (``None``
        cannot be returned because it could leave an empty block).
        """
        self.generic_visit(node)
        if node.value is None:
            return ast.Pass()
        return ast.Assign(targets=[node.target], value=node.value)

    # --- f-strings --------------------------------------------------------------------------------
    def visit_JoinedStr(self, node):
        """f-string → ``"template".format(arg0, arg1, …)``.

        Example: ``f"{x:>{n}} mm"`` → ``"{0:>{1}} mm".format(x, n)``. The fields are numbered
        (``{0}``, ``{1}``…) instead of left empty so that a nested format (``{n}`` inside the format
        spec) still matches its argument. The expressions are translated too (they may contain
        f-strings or ``**``).
        """
        args: list[ast.expr] = []
        tpl = self._fmt(node, args)
        self.generic_visit_list(args)
        return ast.Call(func=ast.Attribute(value=ast.Constant(tpl), attr="format", ctx=ast.Load()),
                        args=args, keywords=[])

    def generic_visit_list(self, nodes: list[ast.expr]) -> None:
        """Translates each node of the list ``nodes`` and replaces it in place."""
        for i, n in enumerate(nodes):
            nodes[i] = self.visit(n)

    def _fmt(self, node: ast.JoinedStr, args: list[ast.expr]) -> str:
        """Builds the ``str.format`` template for the f-string ``node`` and appends its expressions to ``args``.

        It is recursive because the format spec (what comes after ``:``) is itself an f-string. Braces
        in the literal text are doubled (``{`` → ``{{``) so that ``format`` leaves them as text.
        ``!r``/``!s``/``!a`` are kept (``conversion`` is the ASCII code of the letter; -1 if none).
        """
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value).replace("{", "{{").replace("}", "}}"))
                continue
            assert isinstance(v, ast.FormattedValue)
            idx = len(args)
            args.append(v.value)
            field = "{" + str(idx)
            if v.conversion != -1:
                field += "!" + chr(v.conversion)
            if v.format_spec is not None:
                field += ":" + self._fmt(v.format_spec, args)
            out.append(field + "}")
        return "".join(out)

    # --- unpacking (PEP 448, Python 3.5) ----------------------------------------------------------
    def visit_Dict(self, node):
        """``{**a, "k": v, **b}`` → ``_tc_merge(a, {"k": v}, b)``.

        Consecutive plain keys are grouped into a single dictionary and the order is kept: as in modern
        Python, if a key repeats the last one wins. In the AST, ``**a`` is a ``None`` key.
        """
        self.generic_visit(node)
        if all(k is not None for k in node.keys):
            return node
        parts: list[ast.expr] = []
        run_k: list[ast.expr] = []
        run_v: list[ast.expr] = []
        for k, v in zip(node.keys, node.values):
            if k is None:
                if run_k:
                    parts.append(ast.Dict(keys=run_k, values=run_v))
                    run_k, run_v = [], []
                parts.append(v)
            else:
                run_k.append(k)
                run_v.append(v)
        if run_k:
            parts.append(ast.Dict(keys=run_k, values=run_v))
        self.usa.add("_tc_merge")
        return _call("_tc_merge", parts)

    def _seq(self, elts: list[ast.expr]) -> ast.expr | None:
        """``[x, *a, y, *b]`` → ``_tc_list([x], a, [y], b)`` (a list). ``None`` if there is no ``*``.

        Used by lists, tuples, sets and calls; each one then converts the list to the right type.
        """
        if not any(isinstance(e, ast.Starred) for e in elts):
            return None
        parts: list[ast.expr] = []
        run: list[ast.expr] = []
        for e in elts:
            if isinstance(e, ast.Starred):
                if run:
                    parts.append(ast.List(elts=run, ctx=ast.Load()))
                    run = []
                parts.append(e.value)
            else:
                run.append(e)
        if run:
            parts.append(ast.List(elts=run, ctx=ast.Load()))
        self.usa.add("_tc_list")
        return _call("_tc_list", parts)

    def visit_List(self, node):
        """``[*a, *b]`` → ``_tc_list(a, b)``. On the left side of an assignment (``[a, *b] = x``) it is
        left as is: that does exist in 3.4 (PEP 3132, Python 3.0)."""
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Store):
            return node
        return self._seq(node.elts) or node

    def visit_Tuple(self, node):
        """``(*a, *b)`` → ``tuple(_tc_list(a, b))``; as with lists, the left side stays the same."""
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Store):
            return node
        seq = self._seq(node.elts)
        return _call("tuple", [seq]) if seq else node

    def visit_Set(self, node):
        """``{*a, *b}`` → ``set(_tc_list(a, b))``."""
        self.generic_visit(node)
        seq = self._seq(node.elts)
        return _call("set", [seq]) if seq else node

    def visit_Call(self, node):
        """Calls: in 3.4 only one ``*args`` at the end of the positionals and one ``**kwargs`` at the end
        of the keywords are allowed.

        ``f(a, *b, c, *d)`` → ``f(a, *_tc_list(b, [c], d))`` and
        ``f(x=1, **a, y=2, **b)`` → ``f(x=1, **_tc_merge(a, {'y': 2}, b))``. If it already follows the
        3.4 rule, the call stays the same (the most common case).
        """
        self.generic_visit(node)
        # positionals: in 3.4 only one *args, at the end
        stars = [i for i, a in enumerate(node.args) if isinstance(a, ast.Starred)]
        if len(stars) > 1 or (stars and stars[-1] != len(node.args) - 1):
            first = stars[0]
            seq = self._seq(node.args[first:])
            node.args = node.args[:first] + [ast.Starred(value=seq, ctx=ast.Load())]
        # keywords: in 3.4 a single **kwargs, at the end
        # (in the AST, "**x" is a keyword with arg None)
        dstars = [i for i, k in enumerate(node.keywords) if k.arg is None]
        if len(dstars) > 1 or (dstars and dstars[-1] != len(node.keywords) - 1):
            first = dstars[0]
            parts: list[ast.expr] = []
            for k in node.keywords[first:]:
                if k.arg is None:
                    parts.append(k.value)
                else:
                    parts.append(ast.Dict(keys=[ast.Constant(k.arg)], values=[k.value]))
            self.usa.add("_tc_merge")
            node.keywords = node.keywords[:first] + [ast.keyword(arg=None, value=_call("_tc_merge", parts))]
        return node

    def visit_Return(self, node):
        """``return``: nothing to change in itself; the returned expression is translated."""
        self.generic_visit(node)
        return node


def _is_classvar(ann: ast.expr) -> bool:
    """Is the annotation ``ClassVar`` or ``ClassVar[...]`` (with or without ``typing.``)? Those attributes
    belong to the class, they are not dataclass fields."""
    t = ann.value if isinstance(ann, ast.Subscript) else ann
    return (isinstance(t, ast.Name) and t.id == "ClassVar") or (isinstance(t, ast.Attribute) and t.attr == "ClassVar")


def _is_docstring(s: ast.stmt) -> bool:
    """Is the statement ``s`` a bare string (a docstring, if it is the first one in the body)?"""
    return isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)


# ==================================================================================================
# ===================================== TRANSLATING A MODULE =====================================
# ==================================================================================================

def traducir(src: str, nombre: str = "<código>") -> str:
    """Translates the source code ``src`` (modern Python) to Python 3.4 and returns the new text.

    Parameters: ``src``, the contents of a ``.py``; ``nombre``, for error messages.

    Steps: removes the ``from __future__`` imports (``annotations`` does not exist in 3.4), applies
    ``A34``, removes the ``typing`` imports no longer used (``_sin_typing``), adds ``import _tisma34``
    (and whatever is used from it) after the module docstring, writes the code back and checks it
    (``_revisar``). The result starts with the encoding declaration: in 3.4 the default is already
    UTF-8, but this makes it explicit and independent of anything.

    Raises: ``NoTraducible`` (with the name and line) if there is something it cannot translate;
    ``SyntaxError`` if ``src`` is not valid Python.
    """
    tree = ast.parse(src, nombre)
    tree.body = [s for s in tree.body
                 if not (isinstance(s, ast.ImportFrom) and s.module == "__future__")]
    t = A34()
    try:
        tree = t.visit(tree)
    except NoTraducible as e:
        raise NoTraducible(f"{nombre}: {e}") from None
    _sin_typing(tree, nombre)
    # The _tisma34 import goes as early as possible (after the docstring): it has to patch pathlib,
    # json, subprocess… before the module uses them.
    pos = 1 if tree.body and _is_docstring(tree.body[0]) else 0
    imp: list[ast.stmt] = [ast.Import(names=[ast.alias(name=SHIM)])]
    if t.usa:
        # sorted: fixed order (two translations of the same code give the same text)
        imp.append(ast.ImportFrom(module=SHIM, names=[ast.alias(name=n) for n in sorted(t.usa)], level=0))
    tree.body[pos:pos] = imp
    ast.fix_missing_locations(tree)   # new nodes have no line number; unparse/compile require one
    out = ast.unparse(tree)
    _revisar(out, nombre)
    return "# -*- coding: utf-8 -*-\n# Traducido a Python 3.4 por empaquetado/tisma_cnc/py34/a_py34.py: no editar.\n" + out + "\n"


def _sin_typing(tree: ast.Module, nombre: str) -> None:
    """Python 3.4 has no ``typing``: without annotations, those names are no longer used and the import goes.

    Walks every block (module, functions, ``if``, ``try``…) and in each ``from typing import …`` or
    ``from collections.abc import …`` keeps only the names that still appear in the code. If any
    ``typing`` name remains (e.g. ``cast(...)`` in an expression), it is rejected: it would fail on
    import in 3.4. ``collections.abc`` does exist in 3.4, so whatever is still used from it stays.
    ``import typing`` is removed if ``typing`` no longer appears. A block left empty gets ``pass``.

    Raises: ``NoTraducible`` if ``typing`` is used outside annotations.
    """
    # Names still in use: plain variables and the left part of "x.y" (e.g. typing.cast).
    usados = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    usados |= {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    for n in list(ast.walk(tree)):
        for campo in ("body", "orelse", "finalbody", "handlers"):
            cuerpo = getattr(n, campo, None)
            if not isinstance(cuerpo, list):
                continue
            nuevo = []
            for s in cuerpo:
                if isinstance(s, ast.ImportFrom) and s.module in ("typing", "collections.abc") and s.level == 0:
                    s.names = [a for a in s.names if (a.asname or a.name) in usados]
                    if not s.names:
                        continue
                    if s.module == "typing":
                        raise NoTraducible(f"{nombre}: usa typing fuera de anotaciones: "
                                           + ", ".join(a.name for a in s.names))
                elif isinstance(s, ast.Import) and any(a.name == "typing" for a in s.names) and "typing" not in usados:
                    continue
                nuevo.append(s)
            if len(nuevo) != len(cuerpo):
                setattr(n, campo, nuevo or [ast.Pass()])


def _revisar(out: str, nombre: str) -> None:
    """The translated code must contain nothing Python 3.4 does not understand (safety net).

    Reads the translated text back and looks for f-strings, annotations, walrus, async/await and
    ``{**}``/``[*]``/``(*)`` unpacking. If anything shows up, it is a bug in this translator (a case
    ``A34`` does not cover): better to stop at build time than to find out on the XP PC. It does not
    detect standard library functions or modules missing in 3.4; that is covered by ``_tisma34`` and the
    translated test suite (``preparar_pruebas.py``).

    Raises: ``NoTraducible`` with what was left and its line.
    """
    tree = ast.parse(out, nombre)
    for n in ast.walk(tree):
        bad = None
        if isinstance(n, (ast.JoinedStr, ast.AnnAssign, ast.NamedExpr, ast.AsyncFunctionDef, ast.Await)):
            bad = type(n).__name__
        elif isinstance(n, ast.Dict) and any(k is None for k in n.keys):
            bad = "{**...}"
        elif isinstance(n, (ast.List, ast.Set)) and any(isinstance(e, ast.Starred) for e in n.elts):
            bad = "[*...]"
        elif isinstance(n, ast.Tuple) and isinstance(n.ctx, ast.Load) and any(isinstance(e, ast.Starred) for e in n.elts):
            bad = "(*...)"
        elif isinstance(n, ast.arg) and n.annotation is not None:
            bad = "anotación"
        elif isinstance(n, (ast.FunctionDef,)) and n.returns is not None:
            bad = "anotación de retorno"
        if bad:
            raise NoTraducible(f"{nombre}: quedó {bad} (línea {getattr(n, 'lineno', '?')})")


# ==================================================================================================
# ================================= FOLDERS AND COMMAND LINE =====================================
# ==================================================================================================

def traducir_carpeta(origen: Path, destino: Path) -> int:
    """Copies ``origen`` to ``destino`` translating each ``.py`` (other files are copied as they are).

    Skips ``__pycache__``. The ``.py`` files are read and written in UTF-8. ``destino`` may exist (files
    are overwritten). Walks in order (``sorted``) so errors always come out in the same order.

    Returns: how many modules were translated. Raises: ``NoTraducible`` on the first file that fails.
    """
    n = 0
    for p in sorted(origen.rglob("*")):
        if "__pycache__" in p.parts:
            continue
        q = destino / p.relative_to(origen)
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        elif p.suffix == ".py":
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_text(traducir(p.read_text(encoding="utf-8"), str(p)), encoding="utf-8")
            n += 1
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
    return n


def main(argv: list[str]) -> int:
    """Command line: ``python a_py34.py ORIGEN DESTINO``.

    Parameters: ``argv`` without the program name. Returns 0 if it translated, 2 (and prints the
    «Uso:» line of the module docstring) if there are not exactly two arguments.
    """
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1])
        return 2
    n = traducir_carpeta(Path(argv[0]), Path(argv[1]))
    print(f"{n} módulos traducidos a Python 3.4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
