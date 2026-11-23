# -*- coding: utf-8 -*-
# ruff: noqa  (escrito para Python 3.4 a propósito: %-format, object, sin f-strings)
"""Capa de compatibilidad para correr TISMA CNC en Python 3.4 (el último oficial para Windows XP).

La importa cada módulo traducido por empaquetado/tisma_cnc/py34/a_py34.py. En Python 3.7 o más nuevo no
hace nada. En 3.4–3.6 agrega lo que falta de la biblioteca estándar:

* ``dataclasses`` (3.7): dataclass, field, fields, asdict, replace, is_dataclass. El traductor deja
  en cada clase la lista de campos en ``__dc_fields__`` (3.4 no guarda las anotaciones de variables).
* ``secrets`` (3.6), ``math.inf``/``math.nan``/``math.isclose`` (3.5),
  ``http.server.ThreadingHTTPServer`` (3.7), ``subprocess.run`` (3.5).
* ``pathlib.Path``: read_text/write_text/read_bytes/write_bytes, mkdir(exist_ok=…), home() (3.5) y
  resolve() sin error si la ruta no existe (3.6).
* ``_tc_merge`` y ``_tc_list``: lo que en el código original es ``{**a, **b}`` y ``[*a, *b]`` (3.5).

Escrito en Python 3.4 a propósito: sin f-strings ni anotaciones.
"""
# ==================================================================================================
# ========================================== CÓMO ENCAJA ===========================================
# ==================================================================================================
#
# Qué es: un módulo que, al importarse, «parcha» (agrega atributos a) módulos de la biblioteca estándar
# de Python 3.4 para que se parezcan a los de 3.7, la versión con la que se escribe TISMA CNC.
#
# Quién lo usa: cada módulo traducido empieza con «import _tisma34» (lo agrega a_py34.traducir justo
# después del docstring), así los parches están puestos antes de que el módulo use pathlib, json, etc.
# Se importa una sola vez (Python guarda el módulo en sys.modules) y los parches valen para todos.
# construir.armar_xp lo copia a la carpeta app/ del paquete XP; preparar_pruebas.py, a la raíz de la
# carpeta de pruebas traducidas.
#
# Qué usa: solo la biblioteca estándar de 3.4. NO se traduce: tiene que estar escrito directamente en
# Python 3.4 (nada de f-strings, anotaciones, «{**a}», dataclasses…). Por eso usa %-format y
# «class X(object)». Para comprobarlo: python3.4 -m py_compile _tisma34.py.
#
# Convenciones:
#   * Solo se agrega lo que falta (hasattr / try-import): en un Python más nuevo no se toca nada y se
#     usa la versión original, que es más completa y más rápida.
#   * Se implementa solo lo que usan TISMA CNC y sus pruebas, no la API completa: si el código empieza
#     a usar algo más (otro parámetro de subprocess.run, dataclasses.astuple…), hay que agregarlo acá y
#     la suite traducida (preparar_pruebas.py + Python 3.4 real) es la que avisa.
#   * Nombres de variables globales con «_» adelante (_n, _s, _loads…): son internos del parche.
#
# Ejemplo (en Python 3.4):
#     >>> import _tisma34, dataclasses, pathlib
#     >>> pathlib.Path("x.txt").write_text("hola", encoding="utf-8")   # no existía en 3.4
#     4

import sys

# Versión de Python que está corriendo, como tupla comparable: (3, 4), (3, 7)…
PY = sys.version_info[:2]
# Referencias a los objetos de consola originales que se reemplazan (ver «consola» más abajo). Se
# guardan acá para que nunca los libere el recolector de basura.
_VIEJOS = []


# ==================================================================================================
# ================================= DESEMPAQUETADO (PEP 448, 3.5) ==================================
# ==================================================================================================
# Estas dos funciones existen siempre (también en 3.7+): el código traducido las llama aunque el
# Python sea nuevo, porque la traducción es la misma para cualquier versión.

# {**a, **b, "k": v} → _tc_merge(a, b, {"k": v}). Diccionario nuevo con todos los pares; si una clave
# se repite gana la última, igual que en Python moderno. No modifica los originales.
def _tc_merge(*dicts):
    out = {}
    for d in dicts:
        out.update(d)
    return out


# [*a, *b, x] → _tc_list(a, b, [x]). Lista nueva con los elementos de todas las partes, en orden.
# Cada parte puede ser cualquier iterable (lista, tupla, generador, conjunto…).
def _tc_list(*parts):
    out = []
    for p in parts:
        out.extend(p)
    return out


# ==================================================================================================
# ================================= PARCHES PARA PYTHON 3.4 A 3.6 ==================================
# ==================================================================================================
# Todo lo que sigue solo corre en Python < 3.7. En 3.7+ el bloque se saltea entero.

if PY < (3, 7):
    import collections
    import math
    import os
    import pathlib
    import types

    # --- math -------------------------------------------------------------------------------------
    # math.inf/math.nan y math.isclose aparecieron en 3.5.
    if not hasattr(math, "inf"):
        math.inf = float("inf")
        math.nan = float("nan")
    if not hasattr(math, "isclose"):
        # Copia de la definición de la PEP 485: «a y b están cerca» si la diferencia es menor que la
        # tolerancia relativa (respecto del mayor de los dos) o que la absoluta. Los infinitos solo son
        # «cercanos» a sí mismos (lo resuelve el a == b del principio).
        def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
            if a == b:
                return True
            if math.isinf(a) or math.isinf(b):
                return False
            diff = abs(b - a)
            return diff <= abs(rel_tol * b) or diff <= abs(rel_tol * a) or diff <= abs_tol
        math.isclose = isclose

    # --- consola: en XP es cp850/cp437 y 3.4 se cae con «…» o «▶» (3.6+ escribe en Unicode) -------
    # En 3.4/3.5, print() de un carácter que la página de códigos de la consola no tiene lanza
    # UnicodeEncodeError y corta el programa. Se cambian stdout/stderr por otros sobre el mismo buffer
    # con errors="replace" (el carácter sale como «?») y line_buffering=True (cada línea se ve en el
    # momento, como en la consola original).
    if PY < (3, 6):
        import io
        for _n in ("stdout", "stderr"):
            _s = getattr(sys, _n, None)
            # Solo si es la consola de verdad (sys.__stdout__) y sigue en modo «strict»: si pytest o
            # alguien ya la redirigió o la cambió, no se toca.
            if (_s is not None and _s is getattr(sys, "__%s__" % _n, None) and hasattr(_s, "buffer")
                    and getattr(_s, "errors", "") == "strict"):           # solo la consola original
                _VIEJOS.append(_s)              # si el original se libera, cierra la consola
                _s.flush()
                setattr(sys, _n, io.TextIOWrapper(_s.buffer, encoding=_s.encoding or "ascii", errors="replace",
                                                  line_buffering=True))

    # --- json.loads(bytes) (3.6) ------------------------------------------------------------------
    # Hasta 3.5 json.loads solo acepta str. Con bytes se decodifican como UTF-8; «utf-8-sig» además
    # descarta el BOM que dejan algunos editores de Windows (el Bloc de notas) al principio del archivo.
    import json
    if PY < (3, 6):
        _loads = json.loads

        def loads(s, **kw):
            if isinstance(s, (bytes, bytearray)):
                s = bytes(s).decode("utf-8-sig")
            return _loads(s, **kw)
        json.loads = loads

    # --- pathlib ----------------------------------------------------------------------------------
    # Se agregan como métodos de pathlib.Path (y por herencia de PosixPath/WindowsPath). La presencia de
    # read_text marca si hace falta todo el grupo: los cinco aparecieron juntos en 3.5.
    P = pathlib.Path
    if not hasattr(P, "read_text"):
        # Path.read_text/write_text/read_bytes/write_bytes (3.5): abren, leen o escriben todo y cierran.
        def read_text(self, encoding=None, errors=None):
            with self.open(mode="r", encoding=encoding, errors=errors) as f:
                return f.read()

        def write_text(self, data, encoding=None, errors=None):
            with self.open(mode="w", encoding=encoding, errors=errors) as f:
                return f.write(data)

        def read_bytes(self):
            with self.open(mode="rb") as f:
                return f.read()

        def write_bytes(self, data):
            # bytes(data): acepta también bytearray/memoryview, como el original.
            with self.open(mode="wb") as f:
                return f.write(bytes(data))

        P.read_text, P.write_text, P.read_bytes, P.write_bytes = read_text, write_text, read_bytes, write_bytes
        _mkdir = P.mkdir

        # Path.mkdir con exist_ok (3.5): si ya existe como carpeta y exist_ok=True no es error; si existe
        # pero es un archivo, el error se mantiene (igual que en Python moderno).
        def mkdir(self, mode=0o777, parents=False, exist_ok=False):
            try:
                _mkdir(self, mode, parents)
            except FileExistsError:
                if not exist_ok or not self.is_dir():
                    raise
        P.mkdir = mkdir
        # Path.home() (3.5): la carpeta del usuario, como os.path.expanduser("~").
        P.home = classmethod(lambda cls: cls(os.path.expanduser("~")))
    if PY < (3, 6):
        _resolve = P.resolve

        # Path.resolve(strict=False) (3.6): hasta 3.5 resolve() lanza FileNotFoundError si la ruta no
        # existe, y TISMA CNC resuelve carpetas de datos antes de crearlas. Con strict=False (lo normal
        # desde 3.6) se usa os.path.realpath, que no exige que exista.
        def resolve(self, strict=False):
            try:
                return _resolve(self)
            except (FileNotFoundError, OSError):
                if strict:
                    raise
                return self.__class__(os.path.realpath(str(self)))
        P.resolve = resolve

    # --- http.server ------------------------------------------------------------------------------
    # http.server.ThreadingHTTPServer (3.7): un HTTPServer que atiende cada pedido en su propio hilo
    # (el navegador abre varias conexiones a la vez). daemon_threads: los hilos no impiden cerrar el
    # programa. Es la misma definición que trae 3.7.
    import http.server
    import socketserver
    if not hasattr(http.server, "ThreadingHTTPServer"):
        class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
        http.server.ThreadingHTTPServer = ThreadingHTTPServer

    # --- subprocess.run (solo lo que usan TISMA CNC y sus pruebas) --------------------------------
    # subprocess.run y CompletedProcess aparecieron en 3.5. Esta versión soporta input, timeout, check,
    # capture_output (que en realidad es de 3.7), text/universal_newlines y el resto de los parámetros
    # de Popen (por **kw).
    import subprocess
    if not hasattr(subprocess, "run"):
        # Resultado de run(): los mismos atributos que el original (args, returncode, stdout, stderr).
        class CompletedProcess(object):
            def __init__(self, args, returncode, stdout=None, stderr=None):
                self.args, self.returncode, self.stdout, self.stderr = args, returncode, stdout, stderr

            # Lanza CalledProcessError si el programa terminó con código distinto de 0. (En 3.4
            # CalledProcessError no recibe stderr: solo código, comando y salida.)
            def check_returncode(self):
                if self.returncode:
                    raise subprocess.CalledProcessError(self.returncode, self.args, self.stdout)

        # Corre «args», espera que termine y devuelve un CompletedProcess. Si se pasa el tiempo límite,
        # mata el proceso, recoge lo que quedó (para no dejar un proceso zombi) y vuelve a lanzar
        # TimeoutExpired, como el original.
        def run(args, input=None, timeout=None, check=False, capture_output=False, text=None,
                universal_newlines=False, **kw):
            if capture_output:
                kw["stdout"] = subprocess.PIPE
                kw["stderr"] = subprocess.PIPE
            if input is not None:
                kw["stdin"] = subprocess.PIPE
            # text=True (3.7) es lo mismo que universal_newlines=True: salida como str en vez de bytes.
            proc = subprocess.Popen(args, universal_newlines=bool(text or universal_newlines), **kw)
            try:
                out, err = proc.communicate(input, timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise
            res = CompletedProcess(args, proc.returncode, out, err)
            if check:
                res.check_returncode()
            return res
        subprocess.run = run
        subprocess.CompletedProcess = CompletedProcess

    # --- secrets (3.6) ----------------------------------------------------------------------------
    # Módulo secrets (3.6): números al azar aptos para seguridad (la app los usa para el identificador
    # de sesión que queda en la auditoría del guardián). Se arma un módulo nuevo con lo mínimo, sobre
    # os.urandom (el generador del sistema operativo, el mismo que usa el secrets real), y se registra
    # en sys.modules para que «import secrets» lo encuentre. token_urlsafe: base64 apto para URL y sin el relleno «=».
    if "secrets" not in sys.modules:
        try:
            import secrets  # noqa: F401
        except ImportError:
            import base64
            import binascii
            import hmac
            import random
            _sr = random.SystemRandom()
            secrets = types.ModuleType("secrets")
            secrets.token_bytes = lambda n=32: os.urandom(n)
            secrets.token_hex = lambda n=32: binascii.hexlify(os.urandom(n)).decode("ascii")
            secrets.token_urlsafe = lambda n=32: base64.urlsafe_b64encode(os.urandom(n)).rstrip(b"=").decode("ascii")
            secrets.choice = _sr.choice
            # Comparación en tiempo constante (no revela cuántos caracteres coinciden): existe desde 3.3.
            secrets.compare_digest = hmac.compare_digest
            sys.modules["secrets"] = secrets

    # --- dataclasses (3.7) ------------------------------------------------------------------------
    # Una versión reducida del módulo dataclasses de 3.7, con lo que usa TISMA CNC: dataclass (con
    # init, repr, eq y frozen), field (default, default_factory, init, repr, compare), fields, asdict,
    # replace, is_dataclass, MISSING, Field y FrozenInstanceError. No implementa order, unsafe_hash,
    # metadata ni astuple (se aceptan algunos parámetros y se ignoran, para no fallar).
    #
    # De dónde salen los campos: en 3.4 no existe «x: int = 0» dentro de una clase, así que el traductor
    # (a_py34.A34.visit_ClassDef) deja la tupla __dc_fields__ con los nombres en orden y cambia cada
    # campo por «x = 0» (o lo borra si no tiene valor).
    #
    # Igual que el original, __init__ se genera como texto de Python y se compila con exec: así queda
    # una función común, con los parámetros con nombre, y es tan rápida como una escrita a mano.
    try:
        import dataclasses  # noqa: F401
    except ImportError:
        dc = types.ModuleType("dataclasses")
        # MISSING: marca «sin valor» (None puede ser un valor por defecto válido).
        MISSING = object()
        # _FACTORY: valor por defecto de los parámetros con default_factory; si llega este, se llama a la
        # fábrica (así cada objeto recibe su propia lista/diccionario y no uno compartido).
        _FACTORY = object()

        # Error al asignar o borrar un atributo de una dataclass congelada (frozen=True).
        class FrozenInstanceError(AttributeError):
            pass

        # Descripción de un campo. __slots__: liviano, y evita atributos mal escritos.
        class Field(object):
            __slots__ = ("name", "default", "default_factory", "init", "repr", "compare")

            def __init__(self, default=MISSING, default_factory=MISSING, init=True, repr=True, compare=True):
                self.name = None
                self.default, self.default_factory = default, default_factory
                self.init, self.repr, self.compare = init, repr, compare

        # field(...): como el original; hash y metadata se aceptan pero se ignoran.
        def field(default=MISSING, default_factory=MISSING, init=True, repr=True, compare=True, hash=None,
                  metadata=None):
            return Field(default, default_factory, init, repr, compare)

        # Convierte la clase «cls» en dataclass (le agrega __dataclass_fields__, __init__, __repr__,
        # __eq__, __hash__ y, si es frozen, __setattr__/__delattr__ que prohíben cambios). Devuelve la
        # misma clase.
        def _process(cls, frozen, eq, do_repr, init):
            # Campos heredados primero (de la base más lejana a la más cercana) y después los propios:
            # el mismo orden que el original. OrderedDict porque en 3.4 dict no conserva el orden y el
            # orden de los campos es el de los parámetros de __init__.
            fields_ = collections.OrderedDict()
            for base in reversed(cls.__mro__[1:]):
                for f in getattr(base, "__dataclass_fields__", {}).values():
                    fields_[f.name] = f
            for name in cls.__dict__.get("__dc_fields__", ()):
                v = cls.__dict__.get(name, MISSING)
                f = v if isinstance(v, Field) else Field(default=v)
                f.name = name
                if isinstance(v, Field):
                    # «x = field(default=3)»: el atributo de clase pasa a ser el valor por defecto (3);
                    # sin default (p. ej. solo default_factory) se borra, como en el original.
                    if f.default is not MISSING:
                        setattr(cls, name, f.default)
                    else:
                        delattr(cls, name)
                fields_[name] = f
            cls.__dataclass_fields__ = fields_
            # Espacio de nombres para el exec: los valores por defecto y fábricas se pasan como
            # variables (_def_x, _fac_x) y no como texto, así sirve cualquier objeto.
            ns = {"MISSING": MISSING, "_FACTORY": _FACTORY, "_set": object.__setattr__}
            params, body = [], []
            for f in fields_.values():
                if not f.init:
                    # Campo fuera de __init__: solo se inicializa si tiene fábrica.
                    if f.default_factory is not MISSING:
                        body.append("_set(self, %r, _fac_%s())" % (f.name, f.name))
                        ns["_fac_" + f.name] = f.default_factory
                    continue
                if f.default_factory is not MISSING:
                    params.append("%s=_FACTORY" % f.name)
                    ns["_fac_" + f.name] = f.default_factory
                    body.append("_set(self, %r, _fac_%s() if %s is _FACTORY else %s)" % (f.name, f.name, f.name,
                                                                                         f.name))
                elif f.default is not MISSING:
                    params.append("%s=_def_%s" % (f.name, f.name))
                    ns["_def_" + f.name] = f.default
                    body.append("_set(self, %r, %s)" % (f.name, f.name))
                else:
                    params.append(f.name)
                    body.append("_set(self, %r, %s)" % (f.name, f.name))
            if hasattr(cls, "__post_init__"):
                body.append("self.__post_init__()")
            if init:
                # Este __init__ asigna con object.__setattr__ (_set): necesario si la clase es frozen,
                # porque su propio __setattr__ prohíbe asignar. Para las no frozen se vuelve a generar
                # más abajo con setattr común.
                src = "def __init__(self%s):\n    %s\n" % ("".join(", " + p for p in params),
                                                          "\n    ".join(body) or "pass")
                exec(src, ns)
                cls.__init__ = ns["__init__"]
            names = [f.name for f in fields_.values()]
            # Los métodos que la clase ya define a mano se respetan (como en el original).
            if do_repr and "__repr__" not in cls.__dict__:
                rnames = [f.name for f in fields_.values() if f.repr]
                cls.__repr__ = lambda self: "%s(%s)" % (self.__class__.__name__, ", ".join(
                    "%s=%r" % (n, getattr(self, n)) for n in rnames))
            cnames = [f.name for f in fields_.values() if f.compare]
            if eq and "__eq__" not in cls.__dict__:
                # Dos objetos son iguales si son de la misma clase exacta y coinciden todos los campos
                # que se comparan.
                def __eq__(self, other):
                    if other.__class__ is not self.__class__:
                        return NotImplemented
                    return all(getattr(self, n) == getattr(other, n) for n in cnames)
                cls.__eq__ = __eq__
                # Reglas del original: frozen + eq → se puede usar como clave (hash de los campos);
                # eq sin frozen → no tiene hash (mutable: su hash cambiaría), salvo que la clase defina uno.
                if frozen:
                    cls.__hash__ = lambda self: hash(tuple(getattr(self, n) for n in cnames))
                elif "__hash__" not in cls.__dict__:
                    cls.__hash__ = None
            if frozen:
                def __setattr__(self, name, value):
                    raise FrozenInstanceError("cannot assign to field %r" % name)

                def __delattr__(self, name):
                    raise FrozenInstanceError("cannot delete field %r" % name)
                cls.__setattr__, cls.__delattr__ = __setattr__, __delattr__
            elif init:
                # sin frozen se asigna normal (respeta propiedades con setter)
                src = "def __init__(self%s):\n    %s\n" % ("".join(", " + p for p in params),
                                                          "\n    ".join(b.replace("_set(self, ", "setattr(self, ")
                                                                        for b in body) or "pass")
                exec(src, ns)
                cls.__init__ = ns["__init__"]
            # __match_args__ (3.10) no lo usa nadie en 3.4; se deja por paridad con el original.
            cls.__match_args__ = tuple(names)
            return cls

        # @dataclass y @dataclass(frozen=True, …): sin paréntesis recibe la clase en _cls; con
        # paréntesis devuelve el decorador. order y unsafe_hash se aceptan y se ignoran.
        def dataclass(_cls=None, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False):
            def wrap(cls):
                return _process(cls, frozen, eq, repr, init)
            return wrap if _cls is None else wrap(_cls)

        # Tupla con los Field de la clase o del objeto, en orden.
        def fields(obj):
            return tuple(obj.__dataclass_fields__.values())

        # ¿Es una dataclass (clase u objeto)?
        def is_dataclass(obj):
            return hasattr(obj, "__dataclass_fields__")

        # Convierte un objeto dataclass a diccionario, recursivamente (dataclasses, listas, tuplas y
        # diccionarios anidados). Lo demás se copia con deepcopy, como el original, para que modificar
        # el resultado no toque el objeto.
        def asdict(obj, dict_factory=dict):
            def conv(v):
                if is_dataclass(v) and not isinstance(v, type):
                    return dict_factory((f.name, conv(getattr(v, f.name))) for f in fields(v))
                if isinstance(v, (list, tuple)):
                    return type(v)(conv(x) for x in v)
                if isinstance(v, dict):
                    return type(v)((conv(k), conv(x)) for k, x in v.items())
                import copy
                return copy.deepcopy(v)
            return conv(obj)

        # Copia de «obj» con algunos campos cambiados (llama a __init__, así corre __post_init__). Los
        # campos con init=False no se pasan: los vuelve a calcular __init__.
        def replace(obj, **changes):
            kw = dict((f.name, getattr(obj, f.name)) for f in fields(obj) if f.init)
            kw.update(changes)
            return obj.__class__(**kw)

        # Se arma el módulo «dataclasses» y se registra: desde acá «import dataclasses» y
        # «from dataclasses import dataclass» encuentran esta versión.
        for _n, _v in (("dataclass", dataclass), ("field", field), ("fields", fields), ("asdict", asdict),
                       ("replace", replace), ("is_dataclass", is_dataclass), ("MISSING", MISSING),
                       ("Field", Field), ("FrozenInstanceError", FrozenInstanceError)):
            setattr(dc, _n, _v)
        sys.modules["dataclasses"] = dc
