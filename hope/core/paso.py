"""Generación paso a paso con procesadores de logits y cómputo adaptativo por incertidumbre.

Pedido del usuario: que Hope escriba mejor **sin agregar parámetros**. Este módulo es la base
común de las técnicas de decodificación de ``hope/runtime/escritura.py``:

* ``Paso``: una secuencia con cachés (la misma ``_Stream`` de ``hope.core.batched``) que, además de
  los logits del byte siguiente, guarda el **estado del decodificador** (lo usa el kNN-LM), la
  **incertidumbre** que predice la cabeza ``uncertainty`` y la **profundidad** usada por parche.
* ``ComputoAdaptativo`` (técnica 2): el núcleo recurrente itera ``iters_base`` veces por defecto y,
  cuando la incertidumbre de un parche supera ``umbral``, el contexto del parche siguiente se calcula
  con ``iters_profundo`` iteraciones. Nunca pasa de la profundidad entrenada
  (``model.cfg.max_iters``; con ``core_step="embed"`` la tabla tiene exactamente esas filas).
* ``generar``: bucle de muestreo con una lista de ``ProcesadorLogits`` (contrastiva, kNN-LM, …).
  Sin procesadores, sin cómputo adaptativo y en modo voraz da **lo mismo** que ``model.generate``
  (prueba en ``tests/test_escritura.py``): el modo por defecto de Hope no cambia.

Cómo se calcula el contexto profundo sin romper las cachés
-----------------------------------------------------------
La caché del núcleo guarda claves/valores **por iteración**: un parche procesado con 2 iteraciones
no tiene estados para la 3.ª y la 4.ª, así que un parche posterior no puede «iterar más» sobre esa
caché. Por eso hay **dos** ``GlobalCache``: la de base (siempre al día) y la profunda, que se pone
al día **solo cuando hace falta** y, en ese momento, procesa de una vez todos los parches que le
faltan (una pasada con muchos parches cuesta mucho menos que muchas pasadas de uno). Con
``ignorar_parada=False`` el contexto profundo de un parche es exactamente el que daría
``model.generate(n_iters=iters_profundo)``; con la parada ACT ignorada, el de ``forward(full_depth=True)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import torch

from hope.core.batched import _BOUNDARY, _Stream, _local_ok, _to_bytes
from hope.core.hn1 import GlobalCache, HN1Model, natural_boundaries


@dataclass
class ComputoAdaptativo:
    """Más iteraciones del núcleo solo en los parches inciertos.

    ``senal``: ``"cabeza"`` (pérdida predicha por la cabeza ``uncertainty``, en nats por byte) o
    ``"entropia"`` (entropía de la predicción, en nats). Se promedia sobre los bytes del parche
    que se acaba de cerrar; si supera ``umbral``, el parche siguiente usa ``iters_profundo``."""
    umbral: float = 1.5
    iters_base: int = 2
    iters_profundo: int | None = None        # None = profundidad entrenada (cfg.max_iters)
    senal: str = "cabeza"
    # Núcleo con parada ACT: si la parada colapsó a una iteración (pasa en los checkpoints de CPU),
    # pedir más iteraciones no cambia nada. Con ``ignorar_parada`` el modo profundo usa el estado de
    # la última iteración (``full_depth``), que es lo que el entrenamiento usa en la fracción
    # ``full_depth_frac`` de los pasos: sigue dentro de lo entrenado. None = sí, si el núcleo es ACT.
    ignorar_parada: bool | None = None

    def completo(self, model: HN1Model) -> bool:
        if self.ignorar_parada is None:
            return model.cfg.halting == "act"
        return bool(self.ignorar_parada)

    def profundo(self, model: HN1Model) -> int:
        tope = model.cfg.max_iters
        return max(1, min(self.iters_profundo or tope, tope))

    def base(self, model: HN1Model) -> int:
        return max(1, min(self.iters_base, self.profundo(model)))


@dataclass
class EstadisticasPaso:
    bytes: int = 0
    parches: int = 0
    parches_profundos: int = 0
    pasadas_profundas: int = 0

    @property
    def frac_profunda(self) -> float:
        return self.parches_profundos / max(self.parches, 1)


class Paso(_Stream):
    """Secuencia incremental con estado del decodificador e incertidumbre (ver docstring del módulo)."""

    def __init__(self, model: HN1Model, memory=None, n_iters=None, vis=None,
                 adaptativo: ComputoAdaptativo | None = None):
        self.adaptativo = adaptativo
        if adaptativo is not None:
            n_iters = adaptativo.base(model)
        super().__init__(model, memory, n_iters, vis)
        self.deep = (GlobalCache(model, memory, adaptativo.profundo(model), vis, full_depth=adaptativo.completo(model))
                     if adaptativo else None)
        self.hidden: torch.Tensor | None = None       # estado del decodificador del último byte [dl]
        self.unc: float = 0.0                         # pérdida predicha para el byte siguiente (nats)
        self._acum: list[float] = []                  # señal de los bytes del parche actual
        self._prof: dict[int, bool] = {}              # parche → su contexto se calculó en profundo
        self.stats = EstadisticasPaso()

    def clone(self) -> "Paso":
        s = super().clone()
        s._acum, s._prof = list(self._acum), dict(self._prof)
        if self.deep is not None:
            s.deep = GlobalCache(self.m, self.deep.memory, self.deep.n_iters, self.deep.vis, self.deep.full_depth)
            s.deep.restore(self.deep.snapshot())
        s.stats = EstadisticasPaso(**vars(self.stats))
        return s

    def _senal(self, logits: torch.Tensor, unc: torch.Tensor) -> float:
        if self.adaptativo is not None and self.adaptativo.senal == "entropia":
            p = torch.log_softmax(logits.float(), -1)
            return float(-(p.exp() * p).sum())
        return float(unc)

    def _ctx(self, pid: int, h: torch.Tensor) -> torch.Tensor:
        """Contexto global para un byte del parche ``pid`` (= salida del parche ``pid − 1``)."""
        if pid <= 0:
            return h.new_zeros(1, self.m.cfg.d)
        if self._prof.get(pid - 1) and self.deep is not None:
            have = 0 if self.deep.g is None else self.deep.g.shape[1]
            if have < pid:
                self.stats.pasadas_profundas += 1
            return self.deep.upto(self.patch_states, pid)[:, pid - 1]
        return self.inc.upto(self.patch_states, pid)[:, pid - 1]

    @torch.no_grad()
    def feed(self, extra: list[int] = ()) -> int:
        m = self.m
        self.seq.extend(extra)
        new = self.seq[self.n_cached:]
        if not new:
            return 0
        if len(new) > 1 or self.adaptativo is None:
            return self._feed_lote(new)
        # --- un byte (generación): decide la profundidad del parche que se cierra ---
        dev = m.head.weight.device
        ent = m._entropy()
        if ent is None:
            nat = new[0] in _BOUNDARY
        else:
            tail = self.seq[-max(ent[2], 1):]
            nat = bool(natural_boundaries(torch.tensor([tail], device=dev), ent)[0, -1])
        closing = self.last_end
        self.advance_patch(nat)
        if closing:                                       # se cerró el parche cur_pid − 1
            done = self.cur_pid - 1
            sig = sum(self._acum) / max(len(self._acum), 1)
            self._prof[done] = sig > self.adaptativo.umbral
            self.stats.parches += 1
            self.stats.parches_profundos += int(self._prof[done])
            self._acum = []
        h = m.byte_embed(torch.tensor([new], device=dev))
        for blk, c in zip(m.encoder, self.enc):
            h = blk(h, c)
        self.last_h = h[:, -1]
        ctx = self._ctx(self.cur_pid, h)
        y = h + m.to_local(ctx.to(h.dtype)).unsqueeze(1)
        for blk, c in zip(m.decoder, self.dec):
            y = blk(y, c)
        self._salida(m.out_norm(y)[0, -1])
        self.n_cached = len(self.seq)
        self.stats.bytes += 1
        return 1

    def _feed_lote(self, new: list[int]) -> int:
        """Varios bytes de una vez (prompt): como ``_Stream.feed``. Con cómputo adaptativo, los parches
        del prompt deciden su profundidad con la señal de una primera pasada de base (sin tocar las
        cachés) y los bytes que siguen a un parche profundo se decodifican con su contexto profundo."""
        m = self.m
        dev = m.head.weight.device
        h = m.byte_embed(torch.tensor([new], device=dev))
        for blk, c in zip(m.encoder, self.enc):
            h = blk(h, c)
        nat = natural_boundaries(torch.tensor([self.seq], device=dev), m._entropy())[0, self.n_cached:].tolist()
        pids = []
        first_pid = self.cur_pid
        for j, nb in enumerate(nat):
            self.advance_patch(bool(nb))
            pids.append(self.cur_pid)
            self.last_h = h[:, j]
        g = self.inc.upto(self.patch_states, self.cur_pid)
        g_ext = torch.cat([h.new_zeros(1, 1, m.cfg.d), g.to(h.dtype)], dim=1)
        ctx = g_ext[:, pids]
        if self.adaptativo is not None:
            # Primera pasada (con copias de las cachés) con el contexto de base: da la señal de cada
            # byte y con ella se decide la profundidad de los parches del prompt que se cerraron.
            y0 = h + m.to_local(ctx)
            for blk, c in zip(m.decoder, self.dec):
                y0 = blk(y0, dict(c))
            y0 = m.out_norm(y0)[0]
            lg, un = m.head(y0).float(), m.uncertainty(y0).squeeze(-1)
            sig = (un.tolist() if self.adaptativo.senal != "entropia" else
                   (-(torch.log_softmax(lg, -1).exp() * torch.log_softmax(lg, -1)).sum(-1)).tolist())
            por: dict[int, list[float]] = {}
            for p, s in zip(pids, sig):
                por.setdefault(p, []).append(s)
            acum = self._acum
            for p in range(first_pid, self.cur_pid + 1):
                vals = (acum if p == first_pid else []) + por.get(p, [])
                if p < self.cur_pid:                      # parche cerrado: se decide
                    self._prof[p] = sum(vals) / max(len(vals), 1) > self.adaptativo.umbral
                    self.stats.parches += 1
                    self.stats.parches_profundos += int(self._prof[p])
                else:
                    self._acum = vals[:-1]                 # el último lo agrega ``_salida``
            if any(self._prof.get(p - 1) for p in set(pids)):
                ctx = torch.stack([self._ctx(p, h)[0] for p in pids], dim=0).unsqueeze(0).to(h.dtype)
        y = h + m.to_local(ctx)
        for blk, c in zip(m.decoder, self.dec):
            y = blk(y, c)
        self._salida(m.out_norm(y)[0, -1])
        self.n_cached = len(self.seq)
        self.stats.bytes += len(new)
        return len(new)

    def _salida(self, y: torch.Tensor) -> None:
        m = self.m
        self.hidden = y
        self.logits = m.head(y).float()
        unc = m.uncertainty(y).squeeze(-1)
        self.unc = float(unc)
        if self.adaptativo is not None:
            self._acum.append(self._senal(self.logits, unc))


# ---------------------------------------------------------------------------
# Procesadores de logits
# ---------------------------------------------------------------------------
class ProcesadorLogits(Protocol):
    """Modifica la distribución del byte siguiente. Recibe y devuelve **log-probabilidades** [256].

    ``iniciar(seq)`` se llama una vez con el prompt; ``avanzar(byte)`` después de elegir cada byte.
    Pensado para que ``hope/core/decoding.py`` (anti-repetición, otra sesión) se enchufe igual."""

    def iniciar(self, seq: list[int]) -> None: ...

    def __call__(self, paso: Paso, logp: torch.Tensor) -> torch.Tensor: ...

    def avanzar(self, byte: int) -> None: ...


@dataclass
class InformeGeneracion:
    texto: bytes
    segundos: float
    stats: EstadisticasPaso
    extra: dict = field(default_factory=dict)


def _elegir(logp: torch.Tensor, temperature: float, top_p: float | None, gen: torch.Generator | None) -> int:
    if temperature <= 0:
        return int(logp.argmax())
    z = logp / temperature
    if top_p is not None and top_p < 1.0:
        p = torch.softmax(z, -1)
        srt, idx = p.sort(descending=True)
        keep = srt.cumsum(0) - srt < top_p
        z = torch.full_like(z, -math.inf).scatter(0, idx[keep], z[idx[keep]])
    z = torch.where(torch.isfinite(z), z, torch.full_like(z, -1e30))
    return int(torch.multinomial(torch.softmax(z, -1), 1, generator=gen))


@torch.no_grad()
def generar(model: HN1Model, prompt: bytes | str, max_new: int = 256, *, temperature: float = 0.0,
            top_p: float | None = None, procesadores: list | tuple = (), adaptativo: ComputoAdaptativo | None = None,
            stop: bytes = b"\x00", stop_after_line: bytes | None = None, stop_seq: bytes | None = None,
            memory: torch.Tensor | None = None, n_iters: int | None = None, max_context: int = 2048,
            vis: torch.Tensor | None = None, seed: int | None = None, informe: bool = False):
    """Genera con procesadores de logits y/o cómputo adaptativo. Con ``procesadores=()``,
    ``adaptativo=None`` y ``temperature=0`` da lo mismo que ``model.generate``."""
    import time
    t0 = time.perf_counter()
    seq = _to_bytes(prompt)
    if not seq or not _local_ok(model) or len(seq) > max_context:
        # Casos raros (prompt vacío, ventana deslizante): se recorta el prompt a la ventana.
        seq = seq[-max_context // 2:] or [10]
    was = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    paso = Paso(model, memory, n_iters, vis, adaptativo)
    paso.feed(seq)
    for p in procesadores:
        p.iniciar(list(seq))
    out: list[int] = []
    for _ in range(max_new):
        logp = torch.log_softmax(paso.logits, -1)
        for p in procesadores:
            logp = p(paso, logp)
        b = _elegir(logp, temperature, top_p, gen)
        out.append(b)
        if model._should_stop(out, b, stop, stop_after_line, stop_seq):
            break
        for p in procesadores:
            p.avanzar(b)
        if len(paso.seq) + 1 > max_context:              # ventana llena: se vuelve a leer la mitad final
            tail = paso.seq[-max_context // 2:] + [b]
            paso = Paso(model, memory, n_iters, vis, adaptativo)
            paso.feed(tail)
        else:
            paso.feed([b])
    model.train(was)
    if informe:
        return InformeGeneracion(bytes(out), time.perf_counter() - t0, paso.stats)
    return bytes(out)


@torch.no_grad()
def logprobs_paso(model: HN1Model, contexto: bytes | str, texto: bytes | str, *, procesadores: list | tuple = (),
                  adaptativo: ComputoAdaptativo | None = None, n_iters: int | None = None,
                  memory: torch.Tensor | None = None) -> tuple[list[float], EstadisticasPaso]:
    """Log-probabilidad (nats) de cada byte de ``texto`` dado ``contexto``, byte a byte, con los
    mismos procesadores y profundidad adaptativa que la generación (BPB condicional)."""
    ctx, tgt = _to_bytes(contexto), _to_bytes(texto)
    was = model.training
    model.eval()
    paso = Paso(model, memory, n_iters, None, adaptativo)
    paso.feed(ctx or [10])
    for p in procesadores:
        p.iniciar(list(ctx))
    out = []
    for b in tgt:
        logp = torch.log_softmax(paso.logits, -1)
        for p in procesadores:
            logp = p(paso, logp)
        out.append(float(logp[b]))
        for p in procesadores:
            p.avanzar(b)
        paso.feed([b])
    model.train(was)
    return out, paso.stats
