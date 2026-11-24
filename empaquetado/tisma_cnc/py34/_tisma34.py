# -*- coding: utf-8 -*-
# ruff: noqa  (deliberately written for Python 3.4: %-format, object, no f-strings)
"""Compatibility layer to run TISMA CNC on Python 3.4 (the last official one for Windows XP).

Imported by every module translated by empaquetado/tisma_cnc/py34/a_py34.py. On Python 3.7 or newer it
does nothing. On 3.4–3.6 it adds what is missing from the standard library:

* ``dataclasses`` (3.7): dataclass, field, fields, asdict, replace, is_dataclass. The translator leaves
  the list of fields of each class in ``__dc_fields__`` (3.4 does not keep variable annotations).
* ``secrets`` (3.6), ``math.inf``/``math.nan``/``math.isclose`` (3.5),
  ``http.server.ThreadingHTTPServer`` (3.7), ``subprocess.run`` (3.5).
* ``pathlib.Path``: read_text/write_text/read_bytes/write_bytes, mkdir(exist_ok=…), home() (3.5) and
  resolve() without error if the path does not exist (3.6).
* ``_tc_merge`` and ``_tc_list``: what in the original code is ``{**a, **b}`` and ``[*a, *b]`` (3.5).

Deliberately written in Python 3.4: no f-strings or annotations.
"""
# ==================================================================================================
# ========================================= HOW IT FITS IN =========================================
# ==================================================================================================
#
# What it is: a module that, when imported, "patches" (adds attributes to) Python 3.4 standard library
# modules so they resemble those of 3.7, the version TISMA CNC is written in.
#
# Who uses it: every translated module starts with "import _tisma34" (added by a_py34.traducir right
# after the docstring), so the patches are in place before the module uses pathlib, json, etc.
# It is imported only once (Python keeps the module in sys.modules) and the patches apply to everyone.
# construir.armar_xp copies it into the app/ folder of the XP package; preparar_pruebas.py, into the
# root of the translated test folder.
#
# What it uses: only the 3.4 standard library. It is NOT translated: it must be written directly in
# Python 3.4 (no f-strings, annotations, "{**a}", dataclasses…). That is why it uses %-format and
# "class X(object)". To check it: python3.4 -m py_compile _tisma34.py.
#
# Conventions:
#   * Only what is missing is added (hasattr / try-import): on a newer Python nothing is touched and
#     the original version is used, which is more complete and faster.
#   * Only what TISMA CNC and its tests use is implemented, not the full API: if the code starts using
#     something else (another subprocess.run parameter, dataclasses.astuple…), it must be added here,
#     and the translated suite (preparar_pruebas.py + a real Python 3.4) is what will flag it.
#   * Global variable names start with "_" (_n, _s, _loads…): they are internal to the patch.
#
# Example (on Python 3.4):
#     >>> import _tisma34, dataclasses, pathlib
#     >>> pathlib.Path("x.txt").write_text("hola", encoding="utf-8")   # did not exist in 3.4
#     4

import sys

# Running Python version, as a comparable tuple: (3, 4), (3, 7)…
PY = sys.version_info[:2]
# References to the original console objects that get replaced (see "console" below). They are kept
# here so the garbage collector never frees them.
_VIEJOS = []


# ==================================================================================================
# ==================================== UNPACKING (PEP 448, 3.5) ====================================
# ==================================================================================================
# These two functions always exist (also on 3.7+): the translated code calls them even if the Python
# is new, because the translation is the same for any version.

# {**a, **b, "k": v} → _tc_merge(a, b, {"k": v}). New dictionary with all the pairs; if a key repeats
# the last one wins, just like in modern Python. Does not modify the originals.
def _tc_merge(*dicts):
    out = {}
    for d in dicts:
        out.update(d)
    return out


# [*a, *b, x] → _tc_list(a, b, [x]). New list with the elements of all the parts, in order.
# Each part may be any iterable (list, tuple, generator, set…).
def _tc_list(*parts):
    out = []
    for p in parts:
        out.extend(p)
    return out


# ==================================================================================================
# ================================= PATCHES FOR PYTHON 3.4 TO 3.6 ==================================
# ==================================================================================================
# Everything below only runs on Python < 3.7. On 3.7+ the whole block is skipped.

if PY < (3, 7):
    import collections
    import math
    import os
    import pathlib
    import types

    # --- math -------------------------------------------------------------------------------------
    # math.inf/math.nan and math.isclose appeared in 3.5.
    if not hasattr(math, "inf"):
        math.inf = float("inf")
        math.nan = float("nan")
    if not hasattr(math, "isclose"):
        # Copy of the PEP 485 definition: "a and b are close" if the difference is smaller than the
        # relative tolerance (with respect to the larger of the two) or the absolute one. Infinities are
        # only "close" to themselves (handled by the a == b at the start).
        def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
            if a == b:
                return True
            if math.isinf(a) or math.isinf(b):
                return False
            diff = abs(b - a)
            return diff <= abs(rel_tol * b) or diff <= abs(rel_tol * a) or diff <= abs_tol
        math.isclose = isclose

    # --- console: on XP it is cp850/cp437 and 3.4 crashes on "…" or "▶" (3.6+ writes Unicode) -----
    # On 3.4/3.5, print() of a character missing from the console code page raises UnicodeEncodeError
    # and stops the program. stdout/stderr are replaced by new wrappers over the same buffer with
    # errors="replace" (the character comes out as "?") and line_buffering=True (each line shows up
    # immediately, as in the original console).
    if PY < (3, 6):
        import io
        for _n in ("stdout", "stderr"):
            _s = getattr(sys, _n, None)
            # Only if it is the real console (sys.__stdout__) and still in "strict" mode: if pytest or
            # someone else already redirected or changed it, it is left alone.
            if (_s is not None and _s is getattr(sys, "__%s__" % _n, None) and hasattr(_s, "buffer")
                    and getattr(_s, "errors", "") == "strict"):           # original console only
                _VIEJOS.append(_s)              # if the original is freed, it closes the console
                _s.flush()
                setattr(sys, _n, io.TextIOWrapper(_s.buffer, encoding=_s.encoding or "ascii", errors="replace",
                                                  line_buffering=True))

    # --- json.loads(bytes) (3.6) ------------------------------------------------------------------
    # Up to 3.5 json.loads only accepts str. Bytes are decoded as UTF-8; "utf-8-sig" also drops the BOM
    # some Windows editors (Notepad) leave at the start of the file.
    import json
    if PY < (3, 6):
        _loads = json.loads

        def loads(s, **kw):
            if isinstance(s, (bytes, bytearray)):
                s = bytes(s).decode("utf-8-sig")
            return _loads(s, **kw)
        json.loads = loads

    # --- pathlib ----------------------------------------------------------------------------------
    # Added as methods of pathlib.Path (and, by inheritance, of PosixPath/WindowsPath). The presence of
    # read_text tells whether the whole group is needed: the five appeared together in 3.5.
    P = pathlib.Path
    if not hasattr(P, "read_text"):
        # Path.read_text/write_text/read_bytes/write_bytes (3.5): open, read or write everything, close.
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
            # bytes(data): also accepts bytearray/memoryview, like the original.
            with self.open(mode="wb") as f:
                return f.write(bytes(data))

        P.read_text, P.write_text, P.read_bytes, P.write_bytes = read_text, write_text, read_bytes, write_bytes
        _mkdir = P.mkdir

        # Path.mkdir with exist_ok (3.5): if it already exists as a folder and exist_ok=True it is not an
        # error; if it exists but is a file, the error stays (just like in modern Python).
        def mkdir(self, mode=0o777, parents=False, exist_ok=False):
            try:
                _mkdir(self, mode, parents)
            except FileExistsError:
                if not exist_ok or not self.is_dir():
                    raise
        P.mkdir = mkdir
        # Path.home() (3.5): the user's folder, like os.path.expanduser("~").
        P.home = classmethod(lambda cls: cls(os.path.expanduser("~")))
    if PY < (3, 6):
        _resolve = P.resolve

        # Path.resolve(strict=False) (3.6): up to 3.5 resolve() raises FileNotFoundError if the path does
        # not exist, and TISMA CNC resolves data folders before creating them. With strict=False (the
        # norm since 3.6) os.path.realpath is used, which does not require it to exist.
        def resolve(self, strict=False):
            try:
                return _resolve(self)
            except (FileNotFoundError, OSError):
                if strict:
                    raise
                return self.__class__(os.path.realpath(str(self)))
        P.resolve = resolve

    # --- http.server ------------------------------------------------------------------------------
    # http.server.ThreadingHTTPServer (3.7): an HTTPServer that serves each request in its own thread
    # (the browser opens several connections at once). daemon_threads: the threads do not prevent the
    # program from exiting. It is the same definition that 3.7 ships.
    import http.server
    import socketserver
    if not hasattr(http.server, "ThreadingHTTPServer"):
        class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
        http.server.ThreadingHTTPServer = ThreadingHTTPServer

    # --- subprocess.run (only what TISMA CNC and its tests use) -----------------------------------
    # subprocess.run and CompletedProcess appeared in 3.5. This version supports input, timeout, check,
    # capture_output (actually from 3.7), text/universal_newlines and the rest of the Popen parameters
    # (through **kw).
    import subprocess
    if not hasattr(subprocess, "run"):
        # Result of run(): the same attributes as the original (args, returncode, stdout, stderr).
        class CompletedProcess(object):
            def __init__(self, args, returncode, stdout=None, stderr=None):
                self.args, self.returncode, self.stdout, self.stderr = args, returncode, stdout, stderr

            # Raises CalledProcessError if the program exited with a non-zero code. (In 3.4
            # CalledProcessError takes no stderr: only code, command and output.)
            def check_returncode(self):
                if self.returncode:
                    raise subprocess.CalledProcessError(self.returncode, self.args, self.stdout)

        # Runs "args", waits for it to finish and returns a CompletedProcess. If the timeout expires, it
        # kills the process, collects what is left (so no zombie process remains) and re-raises
        # TimeoutExpired, like the original.
        def run(args, input=None, timeout=None, check=False, capture_output=False, text=None,
                universal_newlines=False, **kw):
            if capture_output:
                kw["stdout"] = subprocess.PIPE
                kw["stderr"] = subprocess.PIPE
            if input is not None:
                kw["stdin"] = subprocess.PIPE
            # text=True (3.7) is the same as universal_newlines=True: output as str instead of bytes.
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
    # secrets module (3.6): random numbers suitable for security (the app uses them for the session
    # identifier recorded in the guardian's audit log). A new module is built with the minimum, on top of
    # os.urandom (the operating system's generator, the same one the real secrets uses), and registered
    # in sys.modules so "import secrets" finds it. token_urlsafe: URL-safe base64 without "=" padding.
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
            # Constant-time comparison (does not reveal how many characters match): exists since 3.3.
            secrets.compare_digest = hmac.compare_digest
            sys.modules["secrets"] = secrets

    # --- dataclasses (3.7) ------------------------------------------------------------------------
    # A reduced version of the 3.7 dataclasses module, with what TISMA CNC uses: dataclass (with
    # init, repr, eq and frozen), field (default, default_factory, init, repr, compare), fields, asdict,
    # replace, is_dataclass, MISSING, Field and FrozenInstanceError. It does not implement order,
    # unsafe_hash, metadata or astuple (some parameters are accepted and ignored, so nothing fails).
    #
    # Where the fields come from: "x: int = 0" inside a class does not exist in 3.4, so the translator
    # (a_py34.A34.visit_ClassDef) leaves the tuple __dc_fields__ with the names in order and changes each
    # field to "x = 0" (or removes it if it has no value).
    #
    # Like the original, __init__ is generated as Python source text and compiled with exec: that way it
    # is an ordinary function, with named parameters, and as fast as a hand-written one.
    try:
        import dataclasses  # noqa: F401
    except ImportError:
        dc = types.ModuleType("dataclasses")
        # MISSING: "no value" marker (None may be a valid default value).
        MISSING = object()
        # _FACTORY: default value of the parameters with default_factory; if this one arrives, the factory
        # is called (so each object gets its own list/dictionary rather than a shared one).
        _FACTORY = object()

        # Error when assigning or deleting an attribute of a frozen dataclass (frozen=True).
        class FrozenInstanceError(AttributeError):
            pass

        # Description of a field. __slots__: lightweight, and prevents misspelled attributes.
        class Field(object):
            __slots__ = ("name", "default", "default_factory", "init", "repr", "compare")

            def __init__(self, default=MISSING, default_factory=MISSING, init=True, repr=True, compare=True):
                self.name = None
                self.default, self.default_factory = default, default_factory
                self.init, self.repr, self.compare = init, repr, compare

        # field(...): like the original; hash and metadata are accepted but ignored.
        def field(default=MISSING, default_factory=MISSING, init=True, repr=True, compare=True, hash=None,
                  metadata=None):
            return Field(default, default_factory, init, repr, compare)

        # Turns the class "cls" into a dataclass (adds __dataclass_fields__, __init__, __repr__, __eq__,
        # __hash__ and, if frozen, __setattr__/__delattr__ that forbid changes). Returns the same class.
        def _process(cls, frozen, eq, do_repr, init):
            # Inherited fields first (from the most distant base to the nearest) and then its own:
            # the same order as the original. OrderedDict because in 3.4 dict does not keep order and the
            # field order is the order of the __init__ parameters.
            fields_ = collections.OrderedDict()
            for base in reversed(cls.__mro__[1:]):
                for f in getattr(base, "__dataclass_fields__", {}).values():
                    fields_[f.name] = f
            for name in cls.__dict__.get("__dc_fields__", ()):
                v = cls.__dict__.get(name, MISSING)
                f = v if isinstance(v, Field) else Field(default=v)
                f.name = name
                if isinstance(v, Field):
                    # "x = field(default=3)": the class attribute becomes the default value (3);
                    # without a default (e.g. only default_factory) it is deleted, as in the original.
                    if f.default is not MISSING:
                        setattr(cls, name, f.default)
                    else:
                        delattr(cls, name)
                fields_[name] = f
            cls.__dataclass_fields__ = fields_
            # Namespace for exec: default values and factories are passed as variables (_def_x, _fac_x)
            # and not as text, so any object works.
            ns = {"MISSING": MISSING, "_FACTORY": _FACTORY, "_set": object.__setattr__}
            params, body = [], []
            for f in fields_.values():
                if not f.init:
                    # Field outside __init__: only initialized if it has a factory.
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
                # This __init__ assigns with object.__setattr__ (_set): needed if the class is frozen,
                # because its own __setattr__ forbids assignment. For non-frozen classes it is generated
                # again further below with a plain setattr.
                src = "def __init__(self%s):\n    %s\n" % ("".join(", " + p for p in params),
                                                          "\n    ".join(body) or "pass")
                exec(src, ns)
                cls.__init__ = ns["__init__"]
            names = [f.name for f in fields_.values()]
            # Methods the class already defines by hand are respected (as in the original).
            if do_repr and "__repr__" not in cls.__dict__:
                rnames = [f.name for f in fields_.values() if f.repr]
                cls.__repr__ = lambda self: "%s(%s)" % (self.__class__.__name__, ", ".join(
                    "%s=%r" % (n, getattr(self, n)) for n in rnames))
            cnames = [f.name for f in fields_.values() if f.compare]
            if eq and "__eq__" not in cls.__dict__:
                # Two objects are equal if they are of the exact same class and all compared fields
                # match.
                def __eq__(self, other):
                    if other.__class__ is not self.__class__:
                        return NotImplemented
                    return all(getattr(self, n) == getattr(other, n) for n in cnames)
                cls.__eq__ = __eq__
                # Rules of the original: frozen + eq → usable as a key (hash of the fields);
                # eq without frozen → no hash (mutable: its hash would change), unless the class defines one.
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
                # not frozen: plain assignment (respects properties with a setter)
                src = "def __init__(self%s):\n    %s\n" % ("".join(", " + p for p in params),
                                                          "\n    ".join(b.replace("_set(self, ", "setattr(self, ")
                                                                        for b in body) or "pass")
                exec(src, ns)
                cls.__init__ = ns["__init__"]
            # __match_args__ (3.10) is not used by anything on 3.4; kept for parity with the original.
            cls.__match_args__ = tuple(names)
            return cls

        # @dataclass and @dataclass(frozen=True, …): without parentheses it receives the class in _cls;
        # with parentheses it returns the decorator. order and unsafe_hash are accepted and ignored.
        def dataclass(_cls=None, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False):
            def wrap(cls):
                return _process(cls, frozen, eq, repr, init)
            return wrap if _cls is None else wrap(_cls)

        # Tuple with the Field objects of the class or instance, in order.
        def fields(obj):
            return tuple(obj.__dataclass_fields__.values())

        # Is it a dataclass (class or instance)?
        def is_dataclass(obj):
            return hasattr(obj, "__dataclass_fields__")

        # Converts a dataclass object to a dictionary, recursively (nested dataclasses, lists, tuples and
        # dictionaries). Everything else is copied with deepcopy, like the original, so that modifying
        # the result does not touch the object.
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

        # Copy of "obj" with some fields changed (calls __init__, so __post_init__ runs). Fields with
        # init=False are not passed: __init__ recomputes them.
        def replace(obj, **changes):
            kw = dict((f.name, getattr(obj, f.name)) for f in fields(obj) if f.init)
            kw.update(changes)
            return obj.__class__(**kw)

        # The "dataclasses" module is assembled and registered: from here on "import dataclasses" and
        # "from dataclasses import dataclass" find this version.
        for _n, _v in (("dataclass", dataclass), ("field", field), ("fields", fields), ("asdict", asdict),
                       ("replace", replace), ("is_dataclass", is_dataclass), ("MISSING", MISSING),
                       ("Field", Field), ("FrozenInstanceError", FrozenInstanceError)):
            setattr(dc, _n, _v)
        sys.modules["dataclasses"] = dc
