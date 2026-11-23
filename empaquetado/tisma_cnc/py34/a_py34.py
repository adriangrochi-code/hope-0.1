"""Traduce el código de TISMA CNC a Python 3.4, el último Python oficial para Windows XP.

Se corre al armar el paquete (necesita Python 3.9+ por ``ast.unparse``); el código fuente sigue
escrito en Python moderno. Lo que cambia:

* f-strings → ``"...".format(...)``
* anotaciones de tipos: se borran (en las clases ``@dataclass`` la lista de campos queda en
  ``__dc_fields__`` para la versión de dataclasses de ``_tisma34``)
* ``{**a, **b}`` → ``_tc_merge(a, b)``; ``[*a, *b]`` → ``_tc_list(a, b)``
* llamadas con varios ``*``/``**`` o argumentos después de ``*`` → una sola lista/diccionario
* ``from __future__ import annotations`` se va; cada módulo importa ``_tisma34`` primero

Lo que no sabe traducir (walrus, parámetros solo posicionales, async…) lo rechaza con error: así un
cambio que rompería XP se ve al armar y no en la PC del cliente.

Uso: python a_py34.py ORIGEN DESTINO   (carpetas; copia todo y traduce los .py)
"""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

SHIM = "_tisma34"


class NoTraducible(Exception):
    pass


def _name(n: str) -> ast.Name:
    return ast.Name(id=n, ctx=ast.Load())


def _call(fn: str, args: list[ast.expr], keywords: list[ast.keyword] | None = None) -> ast.Call:
    return ast.Call(func=_name(fn), args=args, keywords=keywords or [])


def _is_dataclass(cls: ast.ClassDef) -> bool:
    for d in cls.decorator_list:
        f = d.func if isinstance(d, ast.Call) else d
        if (isinstance(f, ast.Name) and f.id == "dataclass") or (isinstance(f, ast.Attribute) and f.attr == "dataclass"):
            return True
    return False


class A34(ast.NodeTransformer):
    def __init__(self) -> None:
        self.usa: set[str] = set()

    # --- sintaxis que no se traduce -------------------------------------------------------
    def visit_NamedExpr(self, node):
        raise NoTraducible(f"línea {node.lineno}: ':=' (Python 3.8) no existe en 3.4")

    def visit_AsyncFunctionDef(self, node):
        raise NoTraducible(f"línea {node.lineno}: async def (Python 3.5) no existe en 3.4")

    def visit_MatchAs(self, node):  # pragma: no cover - cualquier match
        raise NoTraducible(f"línea {node.lineno}: match (Python 3.10)")

    # --- anotaciones ---------------------------------------------------------------------
    def _args(self, a: ast.arguments) -> None:
        if a.posonlyargs:
            raise NoTraducible("parámetros solo posicionales '/' (Python 3.8)")
        for arg in a.args + a.kwonlyargs + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else []):
            arg.annotation = None

    def visit_FunctionDef(self, node):
        self._args(node.args)
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_Lambda(self, node):
        self.generic_visit(node)
        return node

    def visit_If(self, node):
        t = node.test
        if (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"):
            return node.orelse or None                     # solo para el verificador de tipos
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        if _is_dataclass(node):
            names = [s.target.id for s in node.body
                     if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and s.simple
                     and not _is_classvar(s.annotation)]
            fields = ast.Assign(targets=[ast.Name(id="__dc_fields__", ctx=ast.Store())],
                                value=ast.Tuple(elts=[ast.Constant(n) for n in names], ctx=ast.Load()))
            # __annotations__ para el dataclasses real (3.7+): así lo traducido corre en cualquier Python
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
        if isinstance(s, ast.AnnAssign):
            if s.value is None:
                return None
            return ast.Assign(targets=[s.target], value=s.value)
        return s

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if node.value is None:
            return ast.Pass()
        return ast.Assign(targets=[node.target], value=node.value)

    # --- f-strings -----------------------------------------------------------------------
    def visit_JoinedStr(self, node):
        args: list[ast.expr] = []
        tpl = self._fmt(node, args)
        self.generic_visit_list(args)
        return ast.Call(func=ast.Attribute(value=ast.Constant(tpl), attr="format", ctx=ast.Load()),
                        args=args, keywords=[])

    def generic_visit_list(self, nodes: list[ast.expr]) -> None:
        for i, n in enumerate(nodes):
            nodes[i] = self.visit(n)

    def _fmt(self, node: ast.JoinedStr, args: list[ast.expr]) -> str:
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

    # --- desempaquetado (PEP 448, Python 3.5) ----------------------------------------------
    def visit_Dict(self, node):
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
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Store):
            return node
        return self._seq(node.elts) or node

    def visit_Tuple(self, node):
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Store):
            return node
        seq = self._seq(node.elts)
        return _call("tuple", [seq]) if seq else node

    def visit_Set(self, node):
        self.generic_visit(node)
        seq = self._seq(node.elts)
        return _call("set", [seq]) if seq else node

    def visit_Call(self, node):
        self.generic_visit(node)
        # posicionales: en 3.4 solo un *args y al final
        stars = [i for i, a in enumerate(node.args) if isinstance(a, ast.Starred)]
        if len(stars) > 1 or (stars and stars[-1] != len(node.args) - 1):
            first = stars[0]
            seq = self._seq(node.args[first:])
            node.args = node.args[:first] + [ast.Starred(value=seq, ctx=ast.Load())]
        # nombrados: en 3.4 un solo **kwargs y al final
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
        self.generic_visit(node)
        return node


def _is_classvar(ann: ast.expr) -> bool:
    t = ann.value if isinstance(ann, ast.Subscript) else ann
    return (isinstance(t, ast.Name) and t.id == "ClassVar") or (isinstance(t, ast.Attribute) and t.attr == "ClassVar")


def _is_docstring(s: ast.stmt) -> bool:
    return isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)


def traducir(src: str, nombre: str = "<código>") -> str:
    tree = ast.parse(src, nombre)
    tree.body = [s for s in tree.body
                 if not (isinstance(s, ast.ImportFrom) and s.module == "__future__")]
    t = A34()
    try:
        tree = t.visit(tree)
    except NoTraducible as e:
        raise NoTraducible(f"{nombre}: {e}") from None
    _sin_typing(tree, nombre)
    pos = 1 if tree.body and _is_docstring(tree.body[0]) else 0
    imp: list[ast.stmt] = [ast.Import(names=[ast.alias(name=SHIM)])]
    if t.usa:
        imp.append(ast.ImportFrom(module=SHIM, names=[ast.alias(name=n) for n in sorted(t.usa)], level=0))
    tree.body[pos:pos] = imp
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)
    _revisar(out, nombre)
    return "# -*- coding: utf-8 -*-\n# Traducido a Python 3.4 por empaquetado/tisma_cnc/py34/a_py34.py: no editar.\n" + out + "\n"


def _sin_typing(tree: ast.Module, nombre: str) -> None:
    """Python 3.4 no trae ``typing``: sin anotaciones, esos nombres ya no se usan y el import se va."""
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
    """Lo traducido no debe tener nada que Python 3.4 no entienda (red de seguridad)."""
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


def traducir_carpeta(origen: Path, destino: Path) -> int:
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
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1])
        return 2
    n = traducir_carpeta(Path(argv[0]), Path(argv[1]))
    print(f"{n} módulos traducidos a Python 3.4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
