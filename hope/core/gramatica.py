"""Decodificación con gramática: generar Python o JSON sintácticamente válidos byte a byte.

Hope genera bytes. Un modelo chico se equivoca seguido en la sintaxis (un paréntesis de más,
una coma que falta, un bloque sin cuerpo), y un solo byte malo invalida todo el programa. Acá la
generación se **restringe**: antes de aceptar un byte se comprueba que el texto siga siendo el
comienzo de algo válido. Si no lo es, se prueba el siguiente byte más probable (o se vuelve a
sortear sin ese byte, con temperatura). Así cada byte elegido deja un prefijo viable.

* **JSON** (``JSONConstraint``): autómata de pila incremental exacto (RFC 8259): objetos,
  listas, cadenas con escapes y UTF-8, números, ``true``/``false``/``null``. Cada byte cuesta
  O(1) y el autómata sabe con certeza si el prefijo se puede completar.
* **Python** (``PythonConstraint``): validación incremental con ``tokenize`` (paréntesis,
  caracteres inválidos, indentación, cadenas) y ``ast`` sobre las líneas lógicas ya cerradas
  (un error en una sentencia terminada es definitivo; «falta el bloque» al final no lo es).
  No es una gramática LL completa: es una **aproximación segura del lado de los prefijos**
  (puede dejar pasar un prefijo que después no tenga arreglo), y por eso ``generate_code`` tiene
  **retroceso**: si al final el programa no compila, se corta a la mayor cantidad de líneas que
  compilan y se continúa desde ahí (con otra temperatura) hasta ``retries`` veces.

Uso con el modelo (el modo voraz por defecto no cambia: sin ``constraint`` todo es igual)::

    from hope.core.gramatica import JSONConstraint, generate_code
    out = model.generate(prompt, 200, constraint=JSONConstraint())
    code = generate_code(model, "Escribí la función:\\n", lang="python", stop_seq=b"\\n```")

Las ``terminators`` (p. ej. el cierre de un bloque de código «```») se aceptan solo cuando lo
anterior ya es un programa completo: así el modelo puede cerrar el bloque pero no a la mitad.
"""

from __future__ import annotations

import ast
import codecs
import io
import tokenize
import warnings
from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Interfaz común
# ---------------------------------------------------------------------------


@dataclass
class Constraint:
    """Restricción por prefijos. Las subclases implementan ``viable`` y ``complete``."""

    terminators: tuple[bytes, ...] = ()
    prefix: bytes = b""                        # código ya escrito antes de lo generado (p. ej. la firma)
    max_tries: int = 256                       # bytes que se prueban antes de rendirse
    stats: dict = field(default_factory=lambda: {"bytes": 0, "rechazos": 0, "sin_salida": 0})

    # --- a implementar ---
    def viable(self, text: bytes) -> bool:
        """¿``text`` es el comienzo de algo válido?"""
        raise NotImplementedError

    def complete(self, text: bytes) -> bool:
        """¿``text`` ya es algo válido completo?"""
        raise NotImplementedError

    # --- uso desde la generación ---
    def allowed(self, out: bytes, b: int, stop: bytes = b"\x00") -> bool:
        pre = self.prefix
        if bytes([b]) == stop:
            return self.complete(pre + out)
        cand = out + bytes([b])
        for t in self.terminators:
            for p in range(max(0, len(cand) - len(t)), len(cand)):
                if t.startswith(cand[p:]) and self.complete(pre + cand[:p]):
                    return True
        return self.viable(pre + cand)

    def pick(self, logits: torch.Tensor, temperature: float, out: bytes, stop: bytes = b"\x00") -> int | None:
        """El byte más probable (o sorteado) que deja un prefijo viable; ``None`` si no hay ninguno."""
        self.stats["bytes"] += 1
        logits = logits.float()
        if temperature > 0:
            probs = torch.softmax(logits / temperature, -1).clone()
            for _ in range(self.max_tries):
                if float(probs.sum()) <= 0:
                    break
                b = int(torch.multinomial(probs, 1))
                if self.allowed(out, b, stop):
                    return b
                self.stats["rechazos"] += 1
                probs[b] = 0.0
        for b in torch.argsort(logits, descending=True).tolist()[:self.max_tries]:
            if self.allowed(out, b, stop):
                return b
            self.stats["rechazos"] += 1
        self.stats["sin_salida"] += 1
        return None


def strip_terminator(text: bytes, terminators: tuple[bytes, ...]) -> bytes:
    for t in terminators:
        if t and text.endswith(t):
            return text[:-len(t)]
    return text


# ---------------------------------------------------------------------------
# JSON: autómata de pila exacto
# ---------------------------------------------------------------------------
_WS = frozenset(b" \t\r\n")
_DIG = frozenset(b"0123456789")
_HEX = frozenset(b"0123456789abcdefABCDEF")
_ESC = frozenset(b'"\\/bfnrtu')
_LITS = {ord("t"): b"rue", ord("f"): b"alse", ord("n"): b"ull"}
_NUM_END = frozenset({"cero", "ent", "frac", "expdig"})

# Estado: (pila, modo, sub). pila: tupla de "{" / "["; modo: VAL, VAL_O_CIERRE, CLAVE, CLAVE_O_CIERRE,
# DOSPUNTOS, DESPUES, CAD, NUM, LIT, FIN. sub depende del modo.
JState = tuple


def _value_done(stack: tuple) -> JState:
    return (stack, "FIN", None) if not stack else (stack, "DESPUES", None)


def _start_value(stack: tuple, b: int) -> JState | None:
    if b == ord("{"):
        return (stack + ("{",), "CLAVE_O_CIERRE", None)
    if b == ord("["):
        return (stack + ("[",), "VAL_O_CIERRE", None)
    if b == ord('"'):
        return (stack, "CAD", (False, 0, 0))            # (es_clave, escape: 0 no / 1 sí / 2..5 hex, utf8 pendiente)
    if b == ord("-"):
        return (stack, "NUM", "menos")
    if b == ord("0"):
        return (stack, "NUM", "cero")
    if b in _DIG:
        return (stack, "NUM", "ent")
    if b in _LITS:
        return (stack, "LIT", _LITS[b])
    return None


def json_step(state: JState, b: int) -> JState | None:
    stack, mode, sub = state
    if mode == "CAD":
        key, esc, utf = sub
        if utf:
            return (stack, mode, (key, esc, utf - 1)) if 0x80 <= b <= 0xBF else None
        if esc == 1:
            if b not in _ESC:
                return None
            return (stack, mode, (key, 2 if b == ord("u") else 0, 0))
        if esc >= 2:
            if b not in _HEX:
                return None
            return (stack, mode, (key, 0 if esc == 5 else esc + 1, 0))
        if b == ord('"'):
            return (stack, "DOSPUNTOS", None) if key else _value_done(stack)
        if b == ord("\\"):
            return (stack, mode, (key, 1, 0))
        if b < 0x20:
            return None
        if b < 0x80:
            return state
        if 0xC2 <= b <= 0xDF:
            return (stack, mode, (key, 0, 1))
        if 0xE0 <= b <= 0xEF:
            return (stack, mode, (key, 0, 2))
        if 0xF0 <= b <= 0xF4:
            return (stack, mode, (key, 0, 3))
        return None
    if mode == "NUM":
        nxt = {
            "menos": {"0": "cero", **{d: "ent" for d in "123456789"}},
            "cero": {".": "punto", "e": "exp", "E": "exp"},
            "ent": {**{d: "ent" for d in "0123456789"}, ".": "punto", "e": "exp", "E": "exp"},
            "punto": {d: "frac" for d in "0123456789"},
            "frac": {**{d: "frac" for d in "0123456789"}, "e": "exp", "E": "exp"},
            "exp": {"+": "expsig", "-": "expsig", **{d: "expdig" for d in "0123456789"}},
            "expsig": {d: "expdig" for d in "0123456789"},
            "expdig": {d: "expdig" for d in "0123456789"},
        }[sub]
        ch = chr(b) if b < 128 else ""
        if ch in nxt:
            return (stack, mode, nxt[ch])
        if sub in _NUM_END:                       # el número terminó: el byte se procesa después
            return json_step(_value_done(stack), b)
        return None
    if mode == "LIT":
        if b != sub[0]:
            return None
        return (stack, mode, sub[1:]) if len(sub) > 1 else _value_done(stack)
    if b in _WS:
        return state
    if mode == "VAL":
        return _start_value(stack, b)
    if mode == "VAL_O_CIERRE":
        if b == ord("]"):
            return _value_done(stack[:-1])
        return _start_value(stack, b)
    if mode in ("CLAVE", "CLAVE_O_CIERRE"):
        if b == ord('"'):
            return (stack, "CAD", (True, 0, 0))
        if mode == "CLAVE_O_CIERRE" and b == ord("}"):
            return _value_done(stack[:-1])
        return None
    if mode == "DOSPUNTOS":
        return (stack, "VAL", None) if b == ord(":") else None
    if mode == "DESPUES":
        top = stack[-1]
        if b == ord(","):
            return (stack, "VAL" if top == "[" else "CLAVE", None)
        if (b == ord("]") and top == "[") or (b == ord("}") and top == "{"):
            return _value_done(stack[:-1])
        return None
    return None                                          # FIN: solo espacios


def json_complete(state: JState | None) -> bool:
    if state is None:
        return False
    stack, mode, sub = state
    return mode == "FIN" or (not stack and mode == "NUM" and sub in _NUM_END)


@dataclass
class JSONConstraint(Constraint):
    """JSON válido. ``root="objeto"`` exige que el valor de arriba sea un objeto ``{…}``."""

    root: str = "cualquiera"                    # "cualquiera" | "objeto" | "lista"
    max_ws: int = 8                             # espacios seguidos fuera de cadenas (evita bucles de relleno)
    _cache: dict = field(default_factory=dict, repr=False)

    def _initial(self) -> JState:
        return ((), "VAL", None)

    def state(self, text: bytes) -> JState | None:
        """Estado después de ``text`` (incremental: reusa el del prefijo más largo ya visto)."""
        if text in self._cache:
            return self._cache[text]
        prev = self.state(text[:-1]) if text else None
        if not text:
            st = self._initial()
        elif prev is None:
            st = None
        else:
            st = json_step(prev, text[-1])
            if st is not None and len(text) == len(text.lstrip()) and self.root != "cualquiera":
                first = text.lstrip()[:1]
                if first and first != (b"{" if self.root == "objeto" else b"["):
                    st = None
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[text] = st
        return st

    def viable(self, text: bytes) -> bool:
        if len(text) > self.max_ws and text[-1] in _WS:
            run = len(text) - len(text.rstrip(b" \t\r\n"))
            st = self.state(text[:-run])
            if run > self.max_ws and (st is None or st[1] != "CAD"):
                return False
        if self.root != "cualquiera":
            first = text.lstrip()[:1]
            if first and first != (b"{" if self.root == "objeto" else b"["):
                return False
        return self.state(text) is not None

    def complete(self, text: bytes) -> bool:
        return self.viable(text) and json_complete(self.state(text))


# ---------------------------------------------------------------------------
# Python: tokenize + ast sobre líneas cerradas
# ---------------------------------------------------------------------------
_BAD_OPS = frozenset({"$", "?", "`"})
_PAIRS = {")": "(", "]": "[", "}": "{"}


def _utf8_prefix(text: bytes) -> str | None:
    """Decodifica admitiendo una secuencia UTF-8 incompleta al final; ``None`` si es inválido."""
    dec = codecs.getincrementaldecoder("utf-8")()
    try:
        return dec.decode(text, final=False)
    except UnicodeDecodeError:
        return None


def python_prefix_ok(src: str) -> bool:
    """¿``src`` puede ser el comienzo de un programa Python válido? (aproximación segura)."""
    lines = src.split("\n")
    last_partial = not src.endswith("\n")
    last_no = len(lines)
    toks: list[tokenize.TokenInfo] = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            toks.append(t)
    except IndentationError:
        return False
    except tokenize.TokenError as e:
        msg = str(e.args[0])
        line = e.args[1][0] if len(e.args) > 1 and e.args[1] else last_no
        if "EOF in multi-line" not in msg and not (last_partial and line == last_no):
            return False
    except SyntaxError as e:
        if not (last_partial and (e.lineno or last_no) == last_no):
            return False
    stack: list[str] = []
    last_newline_row = 0
    for t in toks:
        if t.type == tokenize.OP:
            s = t.string
            if s in _BAD_OPS:
                return False
            if s == "!" and not (last_partial and t.end == (last_no, len(lines[-1]))):
                return False
            if s in "([{":
                stack.append(s)
            elif s in _PAIRS:
                if not stack or stack.pop() != _PAIRS[s]:
                    return False
        elif t.type == getattr(tokenize, "ERRORTOKEN", -1) and t.string.strip():
            if not (last_partial and t.start[0] == last_no):
                return False
        elif t.type == tokenize.NEWLINE and t.string == "\n":
            last_newline_row = t.end[0]
    if last_newline_row:
        return _closed_lines_ok("\n".join(lines[:last_newline_row]) + "\n")
    return True


_AST_CACHE: dict[str, bool] = {}


def _closed_lines_ok(code: str) -> bool:
    """Las sentencias terminadas compilan, salvo lo que solo puede estar incompleto (falta el bloque)."""
    if code in _AST_CACHE:
        return _AST_CACHE[code]
    ok = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(code)
    except SyntaxError as e:
        n = code.count("\n")
        code_lines = [ln for ln in code.split("\n") if ln.strip() and not ln.strip().startswith("#")]
        last = code_lines[-1] if code_lines else ""
        msg = e.msg or ""
        if (e.lineno or n) > n:
            ok = True                                            # el error está «después del final»
        elif "expected an indented block" in msg:
            ok = last.split("#")[0].rstrip().endswith(":")       # el encabezado es lo último: falta el cuerpo
        elif "expected 'except' or 'finally'" in msg:
            ok = last[:1] in (" ", "\t")                         # seguimos adentro del try
        else:
            ok = last.lstrip().startswith("@")                   # decorador esperando su def
    if len(_AST_CACHE) > 8192:
        _AST_CACHE.clear()
    _AST_CACHE[code] = ok
    return ok


def python_complete(src: str) -> bool:
    """Programa completo: compila (``compile`` atrapa también lo que ``ast.parse`` deja pasar, como
    ``return`` fuera de una función o una expresión con asterisco suelta)."""
    if not src.strip():
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(src, "<hope>", "exec", dont_inherit=True)
        return True
    except (SyntaxError, ValueError):
        return False


@dataclass
class PythonConstraint(Constraint):
    max_blank_lines: int = 2                    # líneas vacías seguidas (evita bucles de relleno)
    max_indent: int = 32                        # espacios al comienzo de una línea
    _cache: dict = field(default_factory=dict, repr=False)

    def viable(self, text: bytes) -> bool:
        if text in self._cache:
            return self._cache[text]
        if text.endswith(b"\n" * (self.max_blank_lines + 2)):
            return False
        last = text.rsplit(b"\n", 1)[-1]
        if len(last) > self.max_indent and not last.strip():
            return False
        src = _utf8_prefix(text)
        ok = src is not None and "\x00" not in src and python_prefix_ok(src)
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[text] = ok
        return ok

    def complete(self, text: bytes) -> bool:
        try:
            return python_complete(text.decode("utf-8"))
        except UnicodeDecodeError:
            return False


def constraint_for(lang: str, terminators: tuple[bytes, ...] = (), **kw) -> Constraint:
    if lang == "json":
        return JSONConstraint(terminators=terminators, **kw)
    if lang == "python":
        return PythonConstraint(terminators=terminators, **kw)
    raise ValueError("lenguaje: python | json")


# ---------------------------------------------------------------------------
# Generación con retroceso
# ---------------------------------------------------------------------------
def longest_valid_prefix(text: str, lang: str = "python") -> str:
    """La mayor cantidad de líneas iniciales que forman un programa (o JSON) completo."""
    lines = text.split("\n")
    check = python_complete if lang == "python" else (lambda s: JSONConstraint().complete(s.encode()))
    for k in range(len(lines), 0, -1):
        cand = "\n".join(lines[:k]).rstrip()
        if cand and check(cand):
            return cand + ("\n" if lang == "python" else "")
    return ""


@dataclass
class CodeGen:
    text: str
    valid: bool
    backtracks: int
    rejections: int
    raw: str


def generate_code(model, prompt: str | bytes, *, lang: str = "python", max_new: int = 400,
                  temperature: float = 0.0, stop_seq: bytes | None = None, retries: int = 2,
                  retry_temperature: float = 0.7, code_prefix: str = "", generate=None, **gen_kwargs) -> CodeGen:
    """Genera código restringido por la gramática, con retroceso por líneas si al final no compila.

    ``code_prefix``: la parte final del prompt que ya es código (p. ej. la firma y el docstring de la
    función); la gramática se aplica a ``code_prefix`` + lo generado y el resultado lo incluye.
    ``generate``: función de generación (por defecto ``model.generate``) que acepte ``constraint``."""
    gen = generate or model.generate
    pb = prompt.encode("utf-8") if isinstance(prompt, str) else prompt
    terms = (stop_seq,) if stop_seq else ()
    c = constraint_for(lang, terms)
    head = code_prefix.encode("utf-8")
    if head and not pb.endswith(head):
        pb += head
    body, raw, backtracks = head, "", 0
    pb = pb[:len(pb) - len(head)] if head else pb
    budget = max_new
    temp = temperature
    for _ in range(retries + 1):
        c.prefix = body
        out = gen(pb + body, budget, temperature=temp, stop_seq=stop_seq, constraint=c, **gen_kwargs)
        raw += out.decode("utf-8", "replace")
        budget -= len(out)
        body = body + strip_terminator(out, terms)
        c.prefix = b""
        if c.complete(body):
            break
        # Retroceso: volver a la última línea desde la que el programa compila y seguir desde ahí.
        good = longest_valid_prefix(body.decode("utf-8", "replace"), lang).encode("utf-8")
        backtracks += 1
        body = good if len(good) >= len(head) else head
        if budget <= 8:
            break
        temp = retry_temperature
    c.prefix = b""
    text = body.decode("utf-8", "replace")
    valid = c.complete(body)
    if not valid:
        text = longest_valid_prefix(text, lang)
        valid = bool(text) and c.complete(text.encode("utf-8"))
    return CodeGen(text, valid, backtracks, c.stats["rechazos"], raw)
