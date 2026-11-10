"""Decodificación contra las repeticiones (pedido del usuario; sin tocar el modo voraz por defecto).

Los modelos de bytes chicos caen en bucles («de la ciudad de la ciudad de la ciudad…»): en los
checkpoints de CPU el voraz repite el 72–82 % de sus 4-gramas de palabras
(``docs/resultados/lenguaje.md``). Este módulo reúne procesadores de logits con la interfaz
``ProcesadorLogits`` de ``hope/core/paso.py`` (``iniciar``, ``__call__(paso, logp)``, ``avanzar``)
para usarlos con ``hope.core.paso.generar`` y el bucle de escritura:

* ``PenalizacionRepeticion``: a nivel de **n-gramas de bytes**, no de bytes sueltos (penalizar la
  «e» porque ya salió no tiene sentido). Por cada byte candidato se cuenta cuántas veces la
  continuación «últimos n−1 bytes + candidato» ya apareció en lo generado (y, opcional, en el
  prompt) y se resta ``alpha · cuenta`` (estilo *frequency penalty*) o ``alpha`` si apareció
  (*presence penalty*).
* ``BloqueoNgramas``: prohíbe el byte que completaría un n-grama de **palabras** ya generado
  (*no_repeat_ngram_size* de Paulus et al. 2017, adaptado a bytes): solo actúa al cerrar una
  palabra, así no rompe palabras a la mitad. Además, con ``max_bytes`` prohíbe completar un
  n-grama de bytes largo ya visto (bucle de caracteres).
* ``MuestreoTipico`` (Meister et al. 2023) y ``MinP`` (Nguyen et al. 2024): truncan la
  distribución antes de muestrear (con temperatura 0 no cambian el voraz, salvo que el máximo
  quede afuera, lo que no pasa).
* ``Contrastiva``: la decodificación contrastiva de ``hope/core/contrastiva.py`` (se re-exporta).

``procesadores_por_nombre("rep:1.5,bloqueo:3,minp:0.05")`` arma una lista desde un texto corto
(para la app y la medición). Con la lista vacía, ``generar`` da exactamente ``model.generate``.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

import torch

from hope.core.contrastiva import ProcesadorContrastivo as Contrastiva  # noqa: F401  (re-export)

_LETRA = re.compile(rb"[A-Za-z\xc0-\xff\x80-\xbf]")


def _es_letra(b: int) -> bool:
    return (65 <= b <= 90) or (97 <= b <= 122) or b >= 128


class PenalizacionRepeticion:
    """``logp[b] -= alpha · f(cuenta de (contexto[-(n−1):] + b))``; ``modo``: "frecuencia" | "presencia".
    Se cuentan los n-gramas de bytes de lo generado y, con ``incluir_prompt``, del prompt."""

    def __init__(self, alpha: float = 1.0, n: int = 6, *, modo: str = "frecuencia", incluir_prompt: bool = False,
                 ventana: int = 2048):
        self.alpha, self.n, self.modo, self.incluir_prompt, self.ventana = alpha, n, modo, incluir_prompt, ventana
        self.seq: list[int] = []
        self.cuentas: dict[bytes, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.desde = 0

    def iniciar(self, seq: list[int]) -> None:
        self.seq = list(seq)
        self.cuentas = defaultdict(lambda: defaultdict(int))
        self.desde = 0 if self.incluir_prompt else len(self.seq)
        if self.incluir_prompt:
            for i in range(self.n - 1, len(self.seq)):
                self._contar(i)

    def _contar(self, i: int) -> None:
        if i - (self.n - 1) < self.desde and not self.incluir_prompt:
            return
        ctx = bytes(self.seq[i - self.n + 1:i])
        self.cuentas[ctx][self.seq[i]] += 1

    def avanzar(self, byte: int) -> None:
        self.seq.append(byte)
        i = len(self.seq) - 1
        if i >= self.n - 1:
            self._contar(i)

    def __call__(self, paso, logp: torch.Tensor) -> torch.Tensor:
        if len(self.seq) < self.n - 1:
            return logp
        ctx = bytes(self.seq[len(self.seq) - self.n + 1:])
        vistos = self.cuentas.get(ctx)
        if not vistos:
            return logp
        out = logp.clone()
        for b, c in vistos.items():
            out[b] -= self.alpha * (c if self.modo == "frecuencia" else 1.0)
        return torch.log_softmax(out, -1)


class BloqueoNgramas:
    """Prohíbe repetir n-gramas de **palabras** (``n``) y, si ``max_bytes`` > 0, n-gramas de bytes de
    ese largo. Nunca deja la distribución vacía.

    A nivel de bytes, la palabra que completaría el n-grama se escribe letra por letra. Si solo se
    prohibiera **cerrarla**, el modelo la alargaría («ciudadanasasis»). Por eso, cuando las n−1
    palabras anteriores coinciden con el comienzo de un n-grama ya generado, se prohíbe la letra que
    seguiría copiando su última palabra a partir de la letra ``min_prefijo + 1`` (las primeras letras
    quedan libres, así «de la c…» puede seguir con «casa») y, si la palabra ya está completa, el cierre."""

    def __init__(self, n: int = 3, *, max_bytes: int = 0, incluir_prompt: bool = False, min_prefijo: int = 2):
        self.n, self.max_bytes, self.incluir_prompt, self.min_prefijo = n, max_bytes, incluir_prompt, min_prefijo
        self.seq: list[int] = []
        self.inicio = 0

    def iniciar(self, seq: list[int]) -> None:
        self.seq = list(seq)
        self.inicio = 0 if self.incluir_prompt else len(self.seq)
        self._reindex()

    @staticmethod
    def _palabras(b: bytes) -> list[bytes]:
        return [w.lower() for w in re.findall(rb"(?:[A-Za-z]|[\x80-\xff])+", b)]

    def _reindex(self) -> None:
        txt = bytes(self.seq[self.inicio:])
        ws = self._palabras(txt)
        self._sig: dict[tuple, set[bytes]] = defaultdict(set)       # n−1 palabras → últimas palabras vistas
        for i in range(len(ws) - self.n + 1):
            self._sig[tuple(ws[i:i + self.n - 1])].add(ws[i + self.n - 1])
        self._bytes: set[bytes] = set()
        if self.max_bytes:
            self._bytes = {txt[i:i + self.max_bytes] for i in range(len(txt) - self.max_bytes + 1)}

    def avanzar(self, byte: int) -> None:
        self.seq.append(byte)
        if not _es_letra(byte):                      # se cerró (o no) una palabra: al día
            self._reindex()
        elif self.max_bytes:
            txt = bytes(self.seq[max(self.inicio, len(self.seq) - self.max_bytes):])
            if len(txt) == self.max_bytes:
                self._bytes.add(txt)

    def __call__(self, paso, logp: torch.Tensor) -> torch.Tensor:
        prohibidos: set[int] = set()
        gen = bytes(self.seq[self.inicio:])
        m = re.search(rb"(?:[A-Za-z]|[\x80-\xff])*$", gen)
        parcial = gen[m.start():].lower()
        antes = self._palabras(gen[:m.start()])
        if len(antes) >= self.n - 1 and self._sig:
            vistas = self._sig.get(tuple(antes[len(antes) - self.n + 1:]) if self.n > 1 else (), ())
            for w in vistas:
                if w == parcial and parcial:
                    prohibidos |= {b for b in range(128) if not _es_letra(b)}          # no cerrarla
                elif w.startswith(parcial) and len(parcial) >= self.min_prefijo:
                    c = w[len(parcial)]
                    prohibidos |= {c, c - 32} if 97 <= c <= 122 else {c}                # no seguir copiándola
        if self.max_bytes and len(gen) >= self.max_bytes - 1:
            pre = gen[-(self.max_bytes - 1):]
            for b in range(256):
                if pre + bytes([b]) in self._bytes:
                    prohibidos.add(b)
        if not prohibidos:
            return logp
        out = logp.clone()
        out[torch.tensor(sorted(prohibidos))] = -math.inf
        if not torch.isfinite(out).any():
            return logp
        return torch.log_softmax(out, -1)


class MuestreoTipico:
    """Muestreo típico: se queda con los bytes cuya sorpresa −log p está más cerca de la entropía,
    hasta juntar masa ``tau``."""

    def __init__(self, tau: float = 0.9):
        self.tau = tau

    def iniciar(self, seq):
        pass

    def avanzar(self, byte):
        pass

    def __call__(self, paso, logp: torch.Tensor) -> torch.Tensor:
        p = logp.exp()
        ent = -(p * torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))).sum()
        dist = (-logp - ent).abs()
        orden = dist.argsort()
        cum = p[orden].cumsum(0)
        k = int((cum < self.tau).sum()) + 1
        keep = orden[:k]
        out = torch.full_like(logp, -math.inf)
        out[keep] = logp[keep]
        return torch.log_softmax(out, -1)


class MinP:
    """min-p: se descartan los bytes con p < ``p_base`` · p_max."""

    def __init__(self, p_base: float = 0.05):
        self.p_base = p_base

    def iniciar(self, seq):
        pass

    def avanzar(self, byte):
        pass

    def __call__(self, paso, logp: torch.Tensor) -> torch.Tensor:
        out = torch.where(logp >= logp.max() + math.log(self.p_base), logp, torch.full_like(logp, -math.inf))
        return torch.log_softmax(out, -1)


def procesadores_por_nombre(spec: str, *, modelo=None) -> list:
    """``"rep:1.5,bloqueo:3,bytes:16,tipico:0.9,minp:0.05,contrastiva"`` → lista de procesadores.
    ``rep:α[/n]`` penalización; ``presencia:α[/n]``; ``bloqueo:n`` n-gramas de palabras; ``bytes:n``
    n-gramas de bytes; ``tipico:τ``; ``minp:p``; ``contrastiva`` (aprendiz: el mismo modelo con 1 iteración)."""
    out: list = []
    bloqueo = None
    for parte in (p.strip() for p in spec.split(",") if p.strip()):
        k, _, v = parte.partition(":")
        k = k.lower()
        if k in ("rep", "presencia"):
            a, _, n = v.partition("/")
            out.append(PenalizacionRepeticion(float(a or 1.0), int(n or 6),
                                              modo="frecuencia" if k == "rep" else "presencia"))
        elif k == "bloqueo":
            bloqueo = bloqueo or BloqueoNgramas(int(v or 3))
            bloqueo.n = int(v or 3)
        elif k == "bytes":
            bloqueo = bloqueo or BloqueoNgramas(10 ** 6)
            bloqueo.max_bytes = int(v or 16)
        elif k == "tipico":
            out.append(MuestreoTipico(float(v or 0.9)))
        elif k == "minp":
            out.append(MinP(float(v or 0.05)))
        elif k == "contrastiva":
            if modelo is None:
                raise ValueError("contrastiva necesita el modelo")
            out.append(Contrastiva(modelo, n_iters_aprendiz=1))
        else:
            raise ValueError(f"procesador desconocido: {k}")
    if bloqueo is not None:
        out.insert(0, bloqueo)
    return out


# ---------------------------------------------------------------------------
# Pérdida «unlikelihood» contra la copia de n-gramas (Welleck et al. 2019, adaptada a bytes)
# ---------------------------------------------------------------------------
def candidatos_copia(x: torch.Tensor, n: int = 8) -> torch.Tensor:
    """Para cada posición t de cada secuencia [B, T], el byte que **copiaría** un bucle: si los
    últimos n−1 bytes hasta t ya aparecieron antes en la secuencia, el byte que siguió esa vez.
    −1 donde no hay candidato o el candidato coincide con el byte correcto (no se penaliza lo bien
    copiado: las repeticiones legítimas, como nombres propios, no se castigan)."""
    b, t = x.shape
    out = torch.full((b, t - 1), -1, dtype=torch.long)
    xs = x.tolist()
    for i, row in enumerate(xs):
        visto: dict[tuple, int] = {}
        for j in range(n - 2, t - 1):                        # predice row[j + 1] desde row[..j]
            ctx = tuple(row[j - n + 2:j + 1])
            if ctx in visto:
                c = visto[ctx]
                if c != row[j + 1]:
                    out[i, j] = c
            if j + 1 < t:
                visto[ctx] = row[j + 1]
    return out


def unlikelihood(logits: torch.Tensor, x: torch.Tensor, n: int = 8) -> torch.Tensor:
    """−mean log(1 − p(candidato de copia)) sobre las posiciones con candidato. ``logits`` [B, T−1, 256]
    predicen ``x[:, 1:]``."""
    cand = candidatos_copia(x.cpu(), n).to(logits.device)
    mask = cand >= 0
    if not bool(mask.any()):
        return logits.new_zeros(())
    p = torch.softmax(logits.float(), -1).gather(-1, cand.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return -(torch.log1p(-p.clamp(max=1 - 1e-6)) * mask).sum() / mask.sum()
