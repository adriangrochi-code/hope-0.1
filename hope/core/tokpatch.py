"""Parches alineados a los tokens de Qwen, de forma **causal** (``patching="tokens"``).

Idea (pedido del usuario): las fronteras de parche son los finales de token del tokenizador de
Qwen2.5 (BPE a nivel de bytes, Apache 2.0). Así cada parche de Hope coincide con un token del
maestro y la salida global de un parche puede aprender la distribución del **próximo token** del
maestro (destilación de probabilidades, ``hope/train/distill.py``).

**El problema.** La tokenización BPE no es causal: si el byte *t* termina un token depende de los
bytes que vienen después (``" la"`` es un token en ``" la casa"`` pero no en ``" lado"``). Usar la
tokenización del texto completo filtraría el futuro: el modelo sabría que la palabra terminó.

**La regla causal (cierre por pre-token estable).** Qwen corta primero el texto en *pre-tokens*
con una expresión regular (palabras con su espacio inicial, dígitos sueltos, puntuación, espacios)
y aplica BPE **dentro** de cada pre-token; un token nunca cruza un pre-token. Entonces:

1. Se lleva una región «abierta» desde el último byte confirmado hasta el byte actual *c*.
2. En cada byte nuevo se aplica la expresión regular de Qwen a la región abierta (solo bytes ya
   vistos; los bytes finales de un carácter UTF-8 incompleto se ignoran hasta completarlo).
3. Si la región tiene **dos o más** pre-tokens, todos menos el último están completos: se aplica
   BPE a cada uno y sus finales de token se **confirman en el paso c**. La región abierta pasa a
   empezar en el último pre-token.
4. Una frontera confirmada no se revoca nunca (monotonía). Si un pre-token supera
   ``max_open`` bytes (URL, base64, filas de espacios), se confirman todos sus tokens menos el
   último, para acotar el trabajo y el largo de los parches.

Cada frontera ``e`` tiene su paso de confirmación ``c ≥ e + 1``, y la decisión en *c* usa solo los
bytes ≤ *c*: **es causal por construcción**, sea cual sea el texto que venga después. El costo es un
retraso: la salida global de un parche se puede usar recién desde el byte *c* (en el parcheo por
espacios es siempre ``e + 1``). En el caso típico (``" casa"`` seguida de ``" de"``) el parche se
confirma al llegar el espacio siguiente, o sea en ``e + 1``, igual que antes; los tokens interiores
de una palabra larga (``"des" "igual" "dades"``) se confirman todos al terminar la palabra.

En el modelo esto se expresa con ``ctx_id[t]`` = parches confirmados en un paso ≤ *t*: el byte *t*
recibe la salida global del parche ``ctx_id[t] − 1``. Para los otros parcheos ``ctx_id = patch_id``.

**Cuánto se parece a Qwen.** La regla reproduce la tokenización de Qwen salvo en los pre-tokens
largos (> ``max_open``) y en los raros casos en que la expresión regular de un prefijo corta
distinto que la del texto completo; ``agreement`` lo mide (en el corpus de prueba: ver
``docs/resultados/tokens_vs_bytes.md``). La causalidad no depende de ese parecido.

**El tokenizador es elegible** (``HN1Config.tokenizer``): ``"qwen"`` (Qwen2.5, por defecto) o
``"nemotron"`` (NVIDIA Nemotron 3 Nano 4B), un repositorio de Hugging Face cualquiera o la ruta a un
``tokenizer.json``. Los dos usan la misma estructura (expresión regular + BPE a nivel de bytes); la
de Nemotron además corta en los cambios de mayúscula a minúscula (``"DesigualdadesSociales"`` →
``"Des" "igual" "d" "ades" "Social" "es"``). En el corpus de prueba la regla causal coincide con
la tokenización completa en 100 % (Nemotron) y 99,99 % (Qwen) de las fronteras.

**Archivo.** ``tokenizer.json`` de Hugging Face, descargado una vez y guardado en
``~/.cache/hope/tokenizers`` (o ``$HOPE_CACHE``); también acepta la ruta a un archivo propio
(las pruebas entrenan uno chico sin red).
"""

from __future__ import annotations

import json
import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

DEFAULT_TOKENIZER = "Qwen/Qwen2.5-0.5B"
# Tokenizadores conocidos (el usuario corre los dos maestros en casa). Cualquier BPE a nivel de bytes
# con pre-tokenización por expresión regular sirve: la expresión se lee del propio tokenizer.json.
TOKENIZERS = {
    "qwen": "Qwen/Qwen2.5-0.5B",                          # 151.643 tokens, Apache 2.0
    "nemotron": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",  # 131.072 tokens, NVIDIA Open Model License
}
# Expresión regular de pre-tokenización de Qwen2/2.5 (la misma que la de GPT-4, cl100k).
QWEN_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")
HF_FILE = "https://huggingface.co/{repo}/resolve/main/tokenizer.json"


def cache_dir() -> Path:
    return Path(os.environ.get("HOPE_CACHE", Path.home() / ".cache" / "hope")) / "tokenizers"


def resolve_tokenizer(spec: str) -> Path:
    """Ruta a un ``tokenizer.json``: la del argumento si existe; si no, un repositorio de Hugging
    Face (``"Qwen/Qwen2.5-0.5B"``) que se descarga una vez y queda en la caché."""
    spec = TOKENIZERS.get(spec, spec)
    p = Path(spec)
    if p.is_file():
        return p
    if p.is_dir() and (p / "tokenizer.json").is_file():
        return p / "tokenizer.json"
    dst = cache_dir() / spec.replace("/", "--") / "tokenizer.json"
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        with urllib.request.urlopen(HF_FILE.format(repo=spec), timeout=120) as r:
            tmp.write_bytes(r.read())
        tmp.replace(dst)
    return dst


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Correspondencia byte → carácter del BPE a nivel de bytes de GPT-2 (la que usa Qwen)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _complete_utf8(buf: bytes | bytearray) -> int:
    """Largo del prefijo de ``buf`` sin un carácter UTF-8 incompleto al final."""
    n = len(buf)
    for back in range(1, min(4, n) + 1):
        b = buf[n - back]
        if b & 0xC0 == 0x80:                 # byte de continuación: seguir buscando el inicio
            continue
        need = 2 if b & 0xE0 == 0xC0 else 3 if b & 0xF0 == 0xE0 else 4 if b & 0xF8 == 0xF0 else 1
        return n - back if need > back else n
    return n


class TokenPatcher:
    """Fronteras de parche causales alineadas a un tokenizador BPE a nivel de bytes."""

    def __init__(self, spec: str = DEFAULT_TOKENIZER, max_open: int = 48):
        self.spec, self.max_open = spec, max_open
        self._load()

    def _load(self) -> None:
        import regex
        from tokenizers import Tokenizer
        path = resolve_tokenizer(self.spec)
        self.tok = Tokenizer.from_file(str(path))
        pattern = QWEN_PATTERN
        try:
            pre = json.loads(path.read_text(encoding="utf-8")).get("pre_tokenizer") or {}
            for p in pre.get("pretokenizers", [pre]):
                if p.get("type") == "Split" and "Regex" in p.get("pattern", {}):
                    pattern = p["pattern"]["Regex"]
                    break
        except (OSError, ValueError):
            pass
        self.pattern = regex.compile(pattern)
        self.b2u = bytes_to_unicode()
        self._piece_cache: dict[bytes, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    # El tokenizador de Rust no se puede copiar con pickle: se guarda la especificación y se recarga.
    def __getstate__(self):
        return {"spec": self.spec, "max_open": self.max_open}

    def __setstate__(self, state):
        self.spec, self.max_open = state["spec"], state["max_open"]
        self._load()

    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def piece_tokens(self, piece: bytes) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """BPE de un pre-token: (largos en bytes, ids) de sus tokens."""
        got = self._piece_cache.get(piece)
        if got is None:
            mapped = "".join(self.b2u[b] for b in piece)
            toks = self.tok.model.tokenize(mapped)
            got = (tuple(len(t.value) for t in toks), tuple(t.id for t in toks))
            if len(self._piece_cache) > 200_000:
                self._piece_cache.clear()
            self._piece_cache[piece] = got
        return got

    def pieces(self, text: str) -> list[str]:
        """Pre-tokens de ``text`` (incluye los huecos que la expresión no cubre, como «Isolated»)."""
        out, pos = [], 0
        for m in self.pattern.finditer(text):
            if m.start() > pos:
                out.append(text[pos:m.start()])
            out.append(m.group())
            pos = m.end()
        if pos < len(text):
            out.append(text[pos:])
        return out

    def stream(self) -> "TokenStream":
        return TokenStream(self)

    def boundaries(self, seq: bytes | list[int]) -> tuple[list[int], list[int]]:
        """(finales de parche confirmados, paso de confirmación de cada uno) para una secuencia."""
        s = TokenStream(self)
        for b in seq:
            s.push(int(b))
        return s.ends, s.commits

    def tensors(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Para un lote [B, T] de bytes: ``end`` (el byte cierra un parche; el último siempre),
        ``patch_id`` y ``ctx_id`` (parches confirmados en un paso ≤ t)."""
        b, t = x.shape
        rows = x.detach().to("cpu", torch.long).numpy()
        end = np.zeros((b, t), dtype=bool)
        ctx = np.zeros((b, t), dtype=np.int64)
        steps = np.arange(t)
        for i in range(b):
            ends, commits = self.boundaries(rows[i].tolist())
            end[i, ends] = True
            ctx[i] = np.searchsorted(np.asarray(commits, dtype=np.int64), steps, side="right")
        end[:, -1] = True
        end_t = torch.from_numpy(end).to(x.device)
        patch_id = end_t.long().cumsum(dim=1) - end_t.long()
        return end_t, patch_id, torch.from_numpy(ctx).to(x.device)

    def reference_ends(self, text: bytes) -> list[int]:
        """Finales de token de la tokenización **completa** (no causal) de Qwen: para medir el parecido."""
        s = text.decode("utf-8", "surrogateescape")
        ends, pos = [], 0
        for piece in self.pieces(s):
            raw = piece.encode("utf-8", "surrogateescape")
            for n in self.piece_tokens(raw)[0]:
                pos += n
                ends.append(pos - 1)
        return ends

    def agreement(self, text: bytes) -> dict:
        """Coincidencia entre las fronteras causales y las de la tokenización completa."""
        causal, _ = self.boundaries(text)
        ref = self.reference_ends(text)
        a, r = set(causal), set(ref)
        r.discard(len(text) - 1)                       # el último token del texto no se confirma
        inter = len(a & r)
        return {"bytes": len(text), "tokens_ref": len(ref), "parches": len(causal),
                "precision": inter / max(len(a), 1), "recall": inter / max(len(r), 1),
                "bytes_por_parche": len(text) / max(len(causal), 1)}


class TokenStream:
    """Máquina de fronteras incremental (un byte por vez). ``push`` devuelve los finales de parche
    que se confirman en ese paso; ``ends`` y ``commits`` acumulan todo lo confirmado."""

    def __init__(self, patcher: TokenPatcher):
        self.p = patcher
        self.pos = 0                     # bytes consumidos
        self.p0 = 0                      # inicio (absoluto) de la región abierta
        self.buf = bytearray()
        self.ends: list[int] = []
        self.commits: list[int] = []

    def clone(self) -> "TokenStream":
        s = TokenStream(self.p)
        s.pos, s.p0, s.buf = self.pos, self.p0, bytearray(self.buf)
        s.ends, s.commits = list(self.ends), list(self.commits)
        return s

    def _commit(self, lengths, c: int) -> list[int]:
        new, at = [], self.p0
        for n in lengths:
            at += n
            new.append(at - 1)
        self.ends.extend(new)
        self.commits.extend([c] * len(new))
        done = at - self.p0
        self.p0 = at
        del self.buf[:done]
        return new

    def push(self, byte: int) -> list[int]:
        c = self.pos
        self.pos += 1
        self.buf.append(byte)
        k = _complete_utf8(self.buf)
        if k == 0:
            return []
        text = bytes(self.buf[:k]).decode("utf-8", "surrogateescape")
        pieces = self.p.pieces(text)
        if len(pieces) >= 2:
            lengths: list[int] = []
            for piece in pieces[:-1]:
                lengths.extend(self.p.piece_tokens(piece.encode("utf-8", "surrogateescape"))[0])
            return self._commit(lengths, c)
        if len(self.buf) > self.p.max_open:          # pre-token muy largo: confirmar todo menos el último token
            lengths = list(self.p.piece_tokens(bytes(self.buf[:k]))[0])[:-1]
            if lengths:
                return self._commit(lengths, c)
        return []


_PATCHERS: dict[str, TokenPatcher] = {}


def get_patcher(spec: str) -> TokenPatcher:
    """Un ``TokenPatcher`` por tokenizador, compartido (la caché de pre-tokens se reutiliza)."""
    if spec not in _PATCHERS:
        _PATCHERS[spec] = TokenPatcher(spec)
    return _PATCHERS[spec]
