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

Cómo encaja: lo usan ``construir.py`` (``armar_xp``: traduce la app copiada a la carpeta del paquete
XP) y ``preparar_pruebas.py`` (traduce la app y sus pruebas para correrlas con un Python 3.4 real); las
pruebas ``tests/test_cnc_py34.py`` llaman a ``traducir`` directamente. Lo traducido necesita
``_tisma34.py`` al lado (se copia sin traducir: ya está escrito para 3.4).

Cómo funciona: se lee cada ``.py`` con ``ast.parse`` (con el Python moderno de la PC que arma), un
``ast.NodeTransformer`` (``A34``) reescribe los nodos que 3.4 no entiende, ``ast.unparse`` vuelve a
escribir código fuente y ``_revisar`` lo vuelve a leer para comprobar que no quedó nada moderno. Se
pierden los comentarios (``ast`` no los guarda), no los docstrings: el código traducido no se edita,
se vuelve a generar.

Ejemplo:

    >>> print(traducir('x: int = 1\\nprint(f"{x:>3}")'))
    # -*- coding: utf-8 -*-
    # Traducido a Python 3.4 por empaquetado/tisma_cnc/py34/a_py34.py: no editar.
    import _tisma34
    x = 1
    print('{0:>3}'.format(x))

Uso: python a_py34.py ORIGEN DESTINO   (carpetas; copia todo y traduce los .py)
"""
# Nota: main() muestra la ÚLTIMA línea de este docstring como ayuda de uso: tiene que seguir siendo la de «Uso:».

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

# ==================================================================================================
# ========================================= CONFIGURACIÓN ==========================================
# ==================================================================================================

# Nombre del módulo de compatibilidad que importa cada archivo traducido (py34/_tisma34.py). Si se
# renombra el archivo, cambiar también acá y en construir.py / preparar_pruebas.py, que lo copian.
SHIM = "_tisma34"


# Error de traducción: el código usa algo que Python 3.4 no tiene y que este traductor no sabe
# reescribir (walrus, async, match, parámetros solo posicionales, typing fuera de anotaciones…). El
# mensaje dice archivo y línea. Que corte el armado es a propósito: es mejor que fallar en la PC con XP.
class NoTraducible(Exception):
    pass


# ==================================================================================================
# ==================================== AYUDAS PARA ARMAR NODOS =====================================
# ==================================================================================================

def _name(n: str) -> ast.Name:
    """Nodo ``ast.Name`` de lectura para el nombre ``n`` (p. ej. ``_tc_merge``, ``tuple``)."""
    return ast.Name(id=n, ctx=ast.Load())


def _call(fn: str, args: list[ast.expr], keywords: list[ast.keyword] | None = None) -> ast.Call:
    """Nodo de llamada ``fn(*args, **keywords)`` a una función por nombre (sin atributos)."""
    return ast.Call(func=_name(fn), args=args, keywords=keywords or [])


def _is_dataclass(cls: ast.ClassDef) -> bool:
    """¿La clase lleva el decorador ``@dataclass``, ``@dataclass(...)`` o ``@dataclasses.dataclass``?

    Se mira solo el nombre (``dataclass``), no de dónde se importó: alcanza para el código de TISMA CNC.
    """
    for d in cls.decorator_list:
        f = d.func if isinstance(d, ast.Call) else d
        if (isinstance(f, ast.Name) and f.id == "dataclass") or (isinstance(f, ast.Attribute) and f.attr == "dataclass"):
            return True
    return False


# ==================================================================================================
# =========================================== TRADUCTOR ============================================
# ==================================================================================================

class A34(ast.NodeTransformer):
    """Recorre el árbol sintáctico y reescribe lo que Python 3.4 no entiende (un ``visit_X`` por nodo).

    Cada método ``visit_X`` recibe un nodo de tipo X y devuelve el nodo que lo reemplaza (el mismo,
    otro, o ``None`` para borrarlo), como pide ``ast.NodeTransformer``. Llamar a ``generic_visit``
    primero hace que se traduzcan los hijos antes que el padre (p. ej. una f-string dentro de un
    ``{**a}``).

    Atributo ``usa``: nombres de ``_tisma34`` que necesita el módulo (``_tc_merge``, ``_tc_list``);
    ``traducir`` los importa al principio.
    """

    def __init__(self) -> None:
        """Arranca sin ninguna ayuda de ``_tisma34`` en uso."""
        self.usa: set[str] = set()

    # --- sintaxis que no se traduce ---------------------------------------------------------------
    def visit_NamedExpr(self, node):
        """``x := …`` no tiene equivalente directo en una expresión de 3.4: se rechaza."""
        raise NoTraducible(f"línea {node.lineno}: ':=' (Python 3.8) no existe en 3.4")

    def visit_AsyncFunctionDef(self, node):
        """``async def`` no existe en 3.4 (sería otro programa): se rechaza."""
        raise NoTraducible(f"línea {node.lineno}: async def (Python 3.5) no existe en 3.4")

    def visit_MatchAs(self, node):  # pragma: no cover - cualquier match
        """``match``/``case`` (3.10) no existe en 3.4: se rechaza. Se engancha en ``MatchAs``, el patrón de
        captura (``case _``, ``case x``, ``… as y``), que aparece en casi cualquier ``match``."""
        raise NoTraducible(f"línea {node.lineno}: match (Python 3.10)")

    # --- anotaciones ------------------------------------------------------------------------------
    def _args(self, a: ast.arguments) -> None:
        """Borra las anotaciones de todos los parámetros (en 3.4 existen, pero pueden nombrar tipos como
        ``list[str]`` o ``X | None`` que en 3.4 fallan al definir la función). Rechaza ``/``."""
        if a.posonlyargs:
            raise NoTraducible("parámetros solo posicionales '/' (Python 3.8)")
        for arg in a.args + a.kwonlyargs + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else []):
            arg.annotation = None

    def visit_FunctionDef(self, node):
        """Función: sin anotaciones en los parámetros ni en el valor devuelto; después, su cuerpo."""
        self._args(node.args)
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_Lambda(self, node):
        """Lambda: no tiene anotaciones; solo se traduce lo de adentro."""
        self.generic_visit(node)
        return node

    def visit_If(self, node):
        """``if TYPE_CHECKING:`` se borra (queda su ``else`` si tiene): esos imports son solo para el
        verificador de tipos y en 3.4 podrían no existir (``typing``)."""
        t = node.test
        if (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"):
            return node.orelse or None                     # solo para el verificador de tipos
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        """Clase: borra las anotaciones de atributos y, si es ``@dataclass``, deja la lista de campos.

        Por qué: en 3.4 no existen las anotaciones de variables (``x: int = 0`` es de 3.6), así que el
        ``dataclass`` de ``_tisma34`` no tiene de dónde sacar los campos. Se agregan al principio de la
        clase (después del docstring):

        * ``__dc_fields__ = ('a', 'b')``: los campos en orden (sin los ``ClassVar``), para ``_tisma34``;
        * ``__annotations__ = {'a': object, …}``: para el ``dataclasses`` real (3.7+), así el código
          traducido corre igual en un Python nuevo (las pruebas lo aprovechan).

        Después cada ``x: T = v`` pasa a ``x = v`` y cada ``x: T`` sin valor se borra (si la clase queda
        vacía, se pone ``pass``).
        """
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
        """Atributo de clase anotado → asignación común (``x: T = v`` → ``x = v``) o ``None`` si no tiene
        valor (``x: T``), para borrarlo. Lo demás queda igual."""
        if isinstance(s, ast.AnnAssign):
            if s.value is None:
                return None
            return ast.Assign(targets=[s.target], value=s.value)
        return s

    def visit_AnnAssign(self, node):
        """Variable anotada fuera de una clase (p. ej. ``self.usa: set[str] = set()``): ``x = v``.

        Sin valor (``x: int``) no hace nada en ejecución: se cambia por ``pass`` (no se puede devolver
        ``None`` porque podría dejar un bloque vacío).
        """
        self.generic_visit(node)
        if node.value is None:
            return ast.Pass()
        return ast.Assign(targets=[node.target], value=node.value)

    # --- f-strings --------------------------------------------------------------------------------
    def visit_JoinedStr(self, node):
        """f-string → ``"plantilla".format(arg0, arg1, …)``.

        Ejemplo: ``f"{x:>{n}} mm"`` → ``"{0:>{1}} mm".format(x, n)``. Los campos se numeran
        (``{0}``, ``{1}``…) en vez de dejarlos vacíos para que un formato anidado (``{n}`` dentro del
        especificador) siga correspondiendo a su argumento. Las expresiones se traducen también (pueden
        tener f-strings o ``**`` adentro).
        """
        args: list[ast.expr] = []
        tpl = self._fmt(node, args)
        self.generic_visit_list(args)
        return ast.Call(func=ast.Attribute(value=ast.Constant(tpl), attr="format", ctx=ast.Load()),
                        args=args, keywords=[])

    def generic_visit_list(self, nodes: list[ast.expr]) -> None:
        """Traduce cada nodo de la lista ``nodes`` y lo reemplaza en el lugar."""
        for i, n in enumerate(nodes):
            nodes[i] = self.visit(n)

    def _fmt(self, node: ast.JoinedStr, args: list[ast.expr]) -> str:
        """Arma la plantilla de ``str.format`` para la f-string ``node`` y agrega sus expresiones a ``args``.

        Es recursiva porque el especificador de formato (lo que va después de ``:``) es a su vez una
        f-string. Las llaves del texto literal se duplican (``{`` → ``{{``) para que ``format`` las deje
        como texto. ``!r``/``!s``/``!a`` se conservan (``conversion`` es el código ASCII de la letra;
        -1 si no hay).
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

    # --- desempaquetado (PEP 448, Python 3.5) -----------------------------------------------------
    def visit_Dict(self, node):
        """``{**a, "k": v, **b}`` → ``_tc_merge(a, {"k": v}, b)``.

        Las claves sueltas consecutivas se agrupan en un solo diccionario y se respeta el orden: como
        en Python moderno, si una clave se repite gana la última. En el AST, ``**a`` es una clave ``None``.
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
        """``[x, *a, y, *b]`` → ``_tc_list([x], a, [y], b)`` (una lista). ``None`` si no hay ``*``.

        Lo usan listas, tuplas, conjuntos y llamadas; cada uno convierte después la lista al tipo que
        corresponde.
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
        """``[*a, *b]`` → ``_tc_list(a, b)``. Del lado izquierdo de una asignación (``[a, *b] = x``) se
        deja igual: eso sí existe en 3.4 (PEP 3132, Python 3.0)."""
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Store):
            return node
        return self._seq(node.elts) or node

    def visit_Tuple(self, node):
        """``(*a, *b)`` → ``tuple(_tc_list(a, b))``; como en las listas, el lado izquierdo queda igual."""
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
        """Llamadas: en 3.4 solo vale un ``*args`` al final de los posicionales y un ``**kwargs`` al final
        de los nombrados.

        ``f(a, *b, c, *d)`` → ``f(a, *_tc_list(b, [c], d))`` y
        ``f(x=1, **a, y=2, **b)`` → ``f(x=1, **_tc_merge(a, {'y': 2}, b))``. Si ya cumple la regla de
        3.4, la llamada queda igual (lo más común).
        """
        self.generic_visit(node)
        # posicionales: en 3.4 solo un *args y al final
        stars = [i for i, a in enumerate(node.args) if isinstance(a, ast.Starred)]
        if len(stars) > 1 or (stars and stars[-1] != len(node.args) - 1):
            first = stars[0]
            seq = self._seq(node.args[first:])
            node.args = node.args[:first] + [ast.Starred(value=seq, ctx=ast.Load())]
        # nombrados: en 3.4 un solo **kwargs y al final
        # (en el AST, «**x» es un keyword con arg None)
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
        """``return``: nada que cambiar en sí mismo; se traduce la expresión que devuelve."""
        self.generic_visit(node)
        return node


def _is_classvar(ann: ast.expr) -> bool:
    """¿La anotación es ``ClassVar`` o ``ClassVar[...]`` (con o sin ``typing.``)? Esos atributos son de
    la clase, no campos del dataclass."""
    t = ann.value if isinstance(ann, ast.Subscript) else ann
    return (isinstance(t, ast.Name) and t.id == "ClassVar") or (isinstance(t, ast.Attribute) and t.attr == "ClassVar")


def _is_docstring(s: ast.stmt) -> bool:
    """¿La sentencia ``s`` es una cadena suelta (un docstring, si es la primera del cuerpo)?"""
    return isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)


# ==================================================================================================
# ==================================== TRADUCCIÓN DE UN MÓDULO =====================================
# ==================================================================================================

def traducir(src: str, nombre: str = "<código>") -> str:
    """Traduce el código fuente ``src`` (Python moderno) a Python 3.4 y devuelve el texto nuevo.

    Parámetros: ``src``, el contenido de un ``.py``; ``nombre``, para los mensajes de error.

    Pasos: quita los ``from __future__`` (``annotations`` no existe en 3.4), aplica ``A34``, quita los
    imports de ``typing`` que ya no se usan (``_sin_typing``), agrega ``import _tisma34`` (y lo que use
    de él) después del docstring del módulo, vuelve a escribir el código y lo revisa (``_revisar``).
    El resultado empieza con la declaración de codificación: en 3.4 el valor por defecto ya es UTF-8,
    pero así queda claro y no depende de nada.

    Errores: ``NoTraducible`` (con el nombre y la línea) si hay algo que no se sabe traducir;
    ``SyntaxError`` si ``src`` no es Python válido.
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
    # El import de _tisma34 va lo antes posible (después del docstring): tiene que parchar pathlib,
    # json, subprocess… antes de que el módulo los use.
    pos = 1 if tree.body and _is_docstring(tree.body[0]) else 0
    imp: list[ast.stmt] = [ast.Import(names=[ast.alias(name=SHIM)])]
    if t.usa:
        # sorted: orden fijo (dos traducciones del mismo código dan el mismo texto)
        imp.append(ast.ImportFrom(module=SHIM, names=[ast.alias(name=n) for n in sorted(t.usa)], level=0))
    tree.body[pos:pos] = imp
    ast.fix_missing_locations(tree)   # los nodos nuevos no tienen número de línea; unparse/compile lo piden
    out = ast.unparse(tree)
    _revisar(out, nombre)
    return "# -*- coding: utf-8 -*-\n# Traducido a Python 3.4 por empaquetado/tisma_cnc/py34/a_py34.py: no editar.\n" + out + "\n"


def _sin_typing(tree: ast.Module, nombre: str) -> None:
    """Python 3.4 no trae ``typing``: sin anotaciones, esos nombres ya no se usan y el import se va.

    Recorre todos los bloques (módulo, funciones, ``if``, ``try``…) y en cada ``from typing import …`` o
    ``from collections.abc import …`` deja solo los nombres que todavía aparecen en el código. Si de
    ``typing`` queda alguno (p. ej. ``cast(...)`` en una expresión), se rechaza: en 3.4 fallaría al
    importar. ``collections.abc`` sí existe en 3.4, así que lo que se siga usando de ahí queda.
    ``import typing`` se borra si ``typing`` ya no aparece. Un bloque que queda vacío recibe ``pass``.

    Errores: ``NoTraducible`` si se usa ``typing`` fuera de anotaciones.
    """
    # Nombres que se siguen usando: variables sueltas y la parte izquierda de «x.y» (p. ej. typing.cast).
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
    """Lo traducido no debe tener nada que Python 3.4 no entienda (red de seguridad).

    Vuelve a leer el texto ya traducido y busca f-strings, anotaciones, walrus, async/await y
    desempaquetados ``{**}``/``[*]``/``(*)``. Si aparece algo, es un error de este traductor (un caso que
    ``A34`` no cubre): mejor cortar al armar que descubrirlo en la PC con XP. No detecta funciones o
    módulos de la biblioteca estándar que falten en 3.4; eso lo cubren ``_tisma34`` y la suite de pruebas
    traducida (``preparar_pruebas.py``).

    Errores: ``NoTraducible`` con lo que quedó y su línea.
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
# ================================== CARPETAS Y LÍNEA DE COMANDOS ==================================
# ==================================================================================================

def traducir_carpeta(origen: Path, destino: Path) -> int:
    """Copia ``origen`` a ``destino`` traduciendo cada ``.py`` (los demás archivos van tal cual).

    Saltea ``__pycache__``. Los ``.py`` se leen y escriben en UTF-8. ``destino`` puede existir (se
    sobrescriben los archivos). Recorre en orden (``sorted``) para que los errores salgan siempre en
    el mismo orden.

    Devuelve: cuántos módulos se tradujeron. Errores: ``NoTraducible`` en el primer archivo que falle.
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
    """Línea de comandos: ``python a_py34.py ORIGEN DESTINO``.

    Parámetros: ``argv`` sin el nombre del programa. Devuelve 0 si tradujo, 2 (y muestra la línea
    «Uso:» del docstring del módulo) si los argumentos no son dos.
    """
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1])
        return 2
    n = traducir_carpeta(Path(argv[0]), Path(argv[1]))
    print(f"{n} módulos traducidos a Python 3.4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
