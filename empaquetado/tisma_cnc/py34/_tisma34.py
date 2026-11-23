# -*- coding: utf-8 -*-
# ruff: noqa  (escrito para Python 3.4 a propósito: %-format, object, sin f-strings)
"""Capa de compatibilidad para correr TISMA CNC en Python 3.4 (el último oficial para Windows XP).

La importa cada módulo traducido por empaquetado/tisma_cnc/a_py34.py. En Python 3.7 o más nuevo no
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

import sys

PY = sys.version_info[:2]
_VIEJOS = []


def _tc_merge(*dicts):
    out = {}
    for d in dicts:
        out.update(d)
    return out


def _tc_list(*parts):
    out = []
    for p in parts:
        out.extend(p)
    return out


if PY < (3, 7):
    import collections
    import math
    import os
    import pathlib
    import types

    # --- math ----------------------------------------------------------------------
    if not hasattr(math, "inf"):
        math.inf = float("inf")
        math.nan = float("nan")
    if not hasattr(math, "isclose"):
        def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
            if a == b:
                return True
            if math.isinf(a) or math.isinf(b):
                return False
            diff = abs(b - a)
            return diff <= abs(rel_tol * b) or diff <= abs(rel_tol * a) or diff <= abs_tol
        math.isclose = isclose

    # --- consola: en XP es cp850/cp437 y 3.4 se cae con «…» o «▶» (3.6+ escribe en Unicode) ------
    if PY < (3, 6):
        import io
        for _n in ("stdout", "stderr"):
            _s = getattr(sys, _n, None)
            if (_s is not None and _s is getattr(sys, "__%s__" % _n, None) and hasattr(_s, "buffer")
                    and getattr(_s, "errors", "") == "strict"):           # solo la consola original
                _VIEJOS.append(_s)              # si el original se libera, cierra la consola
                _s.flush()
                setattr(sys, _n, io.TextIOWrapper(_s.buffer, encoding=_s.encoding or "ascii", errors="replace",
                                                  line_buffering=True))

    # --- json.loads(bytes) (3.6) -------------------------------------------------------
    import json
    if PY < (3, 6):
        _loads = json.loads

        def loads(s, **kw):
            if isinstance(s, (bytes, bytearray)):
                s = bytes(s).decode("utf-8-sig")
            return _loads(s, **kw)
        json.loads = loads

    # --- pathlib ---------------------------------------------------------------------
    P = pathlib.Path
    if not hasattr(P, "read_text"):
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
            with self.open(mode="wb") as f:
                return f.write(bytes(data))

        P.read_text, P.write_text, P.read_bytes, P.write_bytes = read_text, write_text, read_bytes, write_bytes
        _mkdir = P.mkdir

        def mkdir(self, mode=0o777, parents=False, exist_ok=False):
            try:
                _mkdir(self, mode, parents)
            except FileExistsError:
                if not exist_ok or not self.is_dir():
                    raise
        P.mkdir = mkdir
        P.home = classmethod(lambda cls: cls(os.path.expanduser("~")))
    if PY < (3, 6):
        _resolve = P.resolve

        def resolve(self, strict=False):
            try:
                return _resolve(self)
            except (FileNotFoundError, OSError):
                if strict:
                    raise
                return self.__class__(os.path.realpath(str(self)))
        P.resolve = resolve

    # --- http.server ---------------------------------------------------------------
    import http.server
    import socketserver
    if not hasattr(http.server, "ThreadingHTTPServer"):
        class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
        http.server.ThreadingHTTPServer = ThreadingHTTPServer

    # --- subprocess.run (solo lo que usan TISMA CNC y sus pruebas) -------------------------
    import subprocess
    if not hasattr(subprocess, "run"):
        class CompletedProcess(object):
            def __init__(self, args, returncode, stdout=None, stderr=None):
                self.args, self.returncode, self.stdout, self.stderr = args, returncode, stdout, stderr

            def check_returncode(self):
                if self.returncode:
                    raise subprocess.CalledProcessError(self.returncode, self.args, self.stdout)

        def run(args, input=None, timeout=None, check=False, capture_output=False, text=None,
                universal_newlines=False, **kw):
            if capture_output:
                kw["stdout"] = subprocess.PIPE
                kw["stderr"] = subprocess.PIPE
            if input is not None:
                kw["stdin"] = subprocess.PIPE
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

    # --- secrets (3.6) --------------------------------------------------------------
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
            secrets.compare_digest = hmac.compare_digest
            sys.modules["secrets"] = secrets

    # --- dataclasses (3.7) ------------------------------------------------------------
    try:
        import dataclasses  # noqa: F401
    except ImportError:
        dc = types.ModuleType("dataclasses")
        MISSING = object()
        _FACTORY = object()

        class FrozenInstanceError(AttributeError):
            pass

        class Field(object):
            __slots__ = ("name", "default", "default_factory", "init", "repr", "compare")

            def __init__(self, default=MISSING, default_factory=MISSING, init=True, repr=True, compare=True):
                self.name = None
                self.default, self.default_factory = default, default_factory
                self.init, self.repr, self.compare = init, repr, compare

        def field(default=MISSING, default_factory=MISSING, init=True, repr=True, compare=True, hash=None,
                  metadata=None):
            return Field(default, default_factory, init, repr, compare)

        def _process(cls, frozen, eq, do_repr, init):
            fields_ = collections.OrderedDict()
            for base in reversed(cls.__mro__[1:]):
                for f in getattr(base, "__dataclass_fields__", {}).values():
                    fields_[f.name] = f
            for name in cls.__dict__.get("__dc_fields__", ()):
                v = cls.__dict__.get(name, MISSING)
                f = v if isinstance(v, Field) else Field(default=v)
                f.name = name
                if isinstance(v, Field):
                    if f.default is not MISSING:
                        setattr(cls, name, f.default)
                    else:
                        delattr(cls, name)
                fields_[name] = f
            cls.__dataclass_fields__ = fields_
            ns = {"MISSING": MISSING, "_FACTORY": _FACTORY, "_set": object.__setattr__}
            params, body = [], []
            for f in fields_.values():
                if not f.init:
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
                src = "def __init__(self%s):\n    %s\n" % ("".join(", " + p for p in params),
                                                          "\n    ".join(body) or "pass")
                exec(src, ns)
                cls.__init__ = ns["__init__"]
            names = [f.name for f in fields_.values()]
            if do_repr and "__repr__" not in cls.__dict__:
                rnames = [f.name for f in fields_.values() if f.repr]
                cls.__repr__ = lambda self: "%s(%s)" % (self.__class__.__name__, ", ".join(
                    "%s=%r" % (n, getattr(self, n)) for n in rnames))
            cnames = [f.name for f in fields_.values() if f.compare]
            if eq and "__eq__" not in cls.__dict__:
                def __eq__(self, other):
                    if other.__class__ is not self.__class__:
                        return NotImplemented
                    return all(getattr(self, n) == getattr(other, n) for n in cnames)
                cls.__eq__ = __eq__
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
            cls.__match_args__ = tuple(names)
            return cls

        def dataclass(_cls=None, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False):
            def wrap(cls):
                return _process(cls, frozen, eq, repr, init)
            return wrap if _cls is None else wrap(_cls)

        def fields(obj):
            return tuple(obj.__dataclass_fields__.values())

        def is_dataclass(obj):
            return hasattr(obj, "__dataclass_fields__")

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

        def replace(obj, **changes):
            kw = dict((f.name, getattr(obj, f.name)) for f in fields(obj) if f.init)
            kw.update(changes)
            return obj.__class__(**kw)

        for _n, _v in (("dataclass", dataclass), ("field", field), ("fields", fields), ("asdict", asdict),
                       ("replace", replace), ("is_dataclass", is_dataclass), ("MISSING", MISSING),
                       ("Field", Field), ("FrozenInstanceError", FrozenInstanceError)):
            setattr(dc, _n, _v)
        sys.modules["dataclasses"] = dc
