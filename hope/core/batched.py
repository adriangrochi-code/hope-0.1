"""Generación en lote y caché de conversación (ARQUITECTURA_HN1 §3.9.1; pedido del usuario).

Objetivo: aprovechar el ancho de banda de memoria de la GPU **sin agregar parámetros**.
Al generar byte a byte, el costo lo domina leer los pesos de memoria, no las cuentas: una
pasada con 8 secuencias lee los mismos pesos que una con 1. Por eso:

1. ``generate_batch``: genera B continuaciones a la vez (consejo, STaR, GRPO, mejor-de-N).
2. ``Conversation``: guarda las cachés entre turnos para no releer el historial.

**Exactitud.** En modo voraz cada salida de ``generate_batch`` es **igual byte a byte** a
``model.generate`` del mismo prompt, y ``conv.say(t1); conv.say(t2)`` da lo mismo que
``model.generate(t1 + salida1 + t2)`` (pruebas en ``tests/test_batched.py``).

Diseño
------
* **Estado de parches por secuencia** (``_Stream``): cada secuencia lleva su propia máquina
  de fronteras (misma regla que ``patch_boundaries``), sus parches completos y su
  ``GlobalCache``. Las secuencias cierran parches en pasos distintos.
* **Parte local en lote**: en cada paso, el byte nuevo de las B secuencias pasa junto por el
  codificador y el decodificador. Las claves/valores se guardan en búferes con relleno
  ``[B, cabezas, capacidad, dh]``; cada fila tiene **su propia posición** (RoPE por fila) y una
  máscara tapa el relleno (camino por filas de ``CausalSelfAttention``, que solo se activa si
  la caché trae ``"rows"``). Así secuencias de largos distintos comparten cada lectura de los
  pesos locales.
* **Parte global en lote**: en cada paso, todas las secuencias que cerraron un parche se
  procesan en **una sola pasada**, aunque tengan cantidades de parches distintas. Se juntan sus
  estados (``_stack``): las claves/valores de la atención (backbone y núcleo) y del especialista
  relacional se rellenan y usan el camino por filas de esos módulos; el resto (SSM,
  convoluciones, especialistas recurrentes, genomas) tiene forma fija y se concatena. Después
  se separan (``_split``) en el formato normal de ``GlobalCache``. Si las estructuras no
  coinciden, cada secuencia usa su ``GlobalCache``. Se agrupa por paso porque cada secuencia
  necesita la salida global de su parche recién cerrado para el byte siguiente: la pasada
  global no se puede postergar, así que el máximo es una por paso.
* **Prefill por prompt** (B = 1): procesar el prompt ya está limitado por cómputo (muchos
  bytes por lectura de pesos), así que no gana con el lote. Los prompts repetidos (mejor-de-N)
  se procesan **una sola vez** y se copia el estado.
* **Casos que usan ``model.generate`` tal cual** (igual que hace ``generate``): prompt vacío,
  texto más largo que ``max_context`` (ventana deslizante) o codificador/decodificador con
  ventana (``local_kind="swa"``).

La imagen (``image``/``vis``), la memoria y ``n_iters`` son comunes a todo el lote.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from hope.core.hn1 import BOUNDARY_BYTES, GlobalCache, HN1Model, natural_boundaries

_BOUNDARY = frozenset(BOUNDARY_BYTES)


@dataclass
class BatchStats:
    """Contadores para ver cuánto se compartió cada lectura de pesos."""
    prompts: int = 0
    prefills: int = 0                 # prompts procesados (los repetidos se procesan una vez)
    prefill_bytes: int = 0
    steps: int = 0                    # pasadas de la parte local (una por byte, para todo el lote)
    rows: int = 0                     # secuencias·pasos (bytes generados en lote)
    global_calls: int = 0             # pasadas de la parte global
    global_patches: int = 0           # parches procesados por esas pasadas
    fallbacks: int = 0                # prompts generados con ``model.generate``

    @property
    def rows_per_step(self) -> float:
        return self.rows / max(self.steps, 1)

    @property
    def patches_per_global_call(self) -> float:
        return self.global_patches / max(self.global_calls, 1)


def _local_ok(m: HN1Model) -> bool:
    """Lote solo con atención local completa y sin caché comprimida (``hope/core/kvcache.py``):
    en los demás casos cada secuencia usa ``generate`` (mismo resultado, sin lote)."""
    from hope.core.kvcache import kv_active
    return all(b.kind == "attn" for b in [*m.encoder, *m.decoder]) and not kv_active(m)


def _to_bytes(p: bytes | str) -> list[int]:
    return list(p.encode("utf-8") if isinstance(p, str) else p)


# ---------------------------------------------------------------------------
# Estado de una secuencia
# ---------------------------------------------------------------------------
class _Stream:
    """Una secuencia: bytes, cachés locales (diccionarios de ``Block``), máquina de parches y
    ``GlobalCache``. ``seq[:n_cached]`` ya está en las cachés; ``logits`` predice el byte que
    sigue a ``seq[n_cached - 1]``."""

    def __init__(self, model: HN1Model, memory=None, n_iters=None, vis=None):
        self.m = model
        self.seq: list[int] = []
        self.n_cached = 0
        self.enc = [{} for _ in model.encoder]
        self.dec = [{} for _ in model.decoder]
        # Máquina de parches: estado del último byte procesado. Con last_nat = True el primer
        # byte empieza un parche nuevo (in_run = 0), como en ``patch_boundaries``.
        self.last_end, self.last_nat, self.in_run, self.cur_pid = False, True, 0, 0
        self.last_h: torch.Tensor | None = None
        self.patch_states: list[torch.Tensor] = []
        self.inc = GlobalCache(model, memory, n_iters, vis)
        self.logits: torch.Tensor | None = None
        self.out: list[int] = []

    def clone(self) -> "_Stream":
        s = _Stream.__new__(_Stream)
        s.__dict__.update(self.__dict__)
        s.seq, s.out, s.patch_states = list(self.seq), list(self.out), list(self.patch_states)
        s.enc = [dict(c) for c in self.enc]               # los tensores se reemplazan, no se modifican
        s.dec = [dict(c) for c in self.dec]
        s.inc = GlobalCache(self.m, self.inc.memory, self.inc.n_iters, self.inc.vis)
        s.inc.restore(self.inc.snapshot())
        return s

    def advance_patch(self, nat: bool) -> None:
        """Avanza la máquina de parches con un byte nuevo (misma regla que ``patch_boundaries``)."""
        if self.last_end:                                # el byte anterior cerró un parche
            self.patch_states.append(self.last_h)
            self.cur_pid += 1
        self.in_run = 0 if self.last_nat else self.in_run + 1
        self.last_nat = nat
        self.last_end = nat or (self.in_run + 1) % self.m.cfg.max_patch == 0

    @torch.no_grad()
    def feed(self, extra: list[int] = ()) -> int:
        """Agrega ``extra`` y procesa de una vez todo lo que falta en las cachés (prefill
        incremental). Devuelve la cantidad de bytes procesados."""
        m = self.m
        self.seq.extend(extra)
        new = self.seq[self.n_cached:]
        if not new:
            return 0
        dev = m.head.weight.device
        h = m.byte_embed(torch.tensor([new], device=dev))
        for blk, c in zip(m.encoder, self.enc):
            h = blk(h, c)
        nat = natural_boundaries(torch.tensor([self.seq], device=dev), m._entropy())[0, self.n_cached:].tolist()
        pids = []
        for j, nb in enumerate(nat):
            self.advance_patch(bool(nb))
            pids.append(self.cur_pid)
            self.last_h = h[:, j]
        g = self.inc.upto(self.patch_states, self.cur_pid)
        g_ext = torch.cat([h.new_zeros(1, 1, m.cfg.d), g.to(h.dtype)], dim=1)   # contexto = parche anterior
        y = h + m.to_local(g_ext[:, pids])
        for blk, c in zip(m.decoder, self.dec):
            y = blk(y, c)
        self.logits = m.head(m.out_norm(y)[0, -1]).float()
        self.n_cached = len(self.seq)
        return len(new)


# ---------------------------------------------------------------------------
# Cachés en lote: juntar las de varias secuencias (con relleno) y volver a separarlas
# ---------------------------------------------------------------------------
class _Mismatch(Exception):
    pass


def _is_attn(c) -> bool:
    """Caché de ``CausalSelfAttention``: claves/valores [1, cabezas, T, dh] que crecen con el texto."""
    return isinstance(c, dict) and c.keys() == {"k", "v"} and isinstance(c["k"], torch.Tensor) and c["k"].dim() == 4


def _is_rel(c) -> bool:
    """Caché del especialista relacional: ventana de claves/valores y posición entera."""
    return isinstance(c, dict) and c.keys() == {"k", "v", "pos"} and isinstance(c["pos"], int)


def _stack(xs: list, extra: int = 1):
    """Junta los estados de varias secuencias en uno de lote. Lo que depende del largo (claves y
    valores de la atención y del especialista relacional) se rellena y pasa al camino por filas
    de esos módulos; el resto (SSM, convoluciones, núcleo) debe tener la misma forma.
    ``extra``: lugar libre en los búferes de atención (pasos que se van a agregar)."""
    x0 = xs[0]
    if isinstance(x0, dict):
        if any(not isinstance(x, dict) or x.keys() != x0.keys() for x in xs):
            raise _Mismatch
        if all(_is_attn(x) for x in xs):
            ks = [x["k"] for x in xs]
            if any(k.shape[:2] != ks[0].shape[:2] or k.shape[3] != ks[0].shape[3] for k in ks):
                raise _Mismatch
            lens = [k.shape[2] for k in ks]
            _, h, _, dh = ks[0].shape
            kb = ks[0].new_zeros(len(xs), h, max(lens) + extra, dh)
            vb = kb.clone()
            for i, (x, n) in enumerate(zip(xs, lens)):
                kb[i, :, :n], vb[i, :, :n] = x["k"][0], x["v"][0]
            return {"k": kb, "v": vb, "rows": torch.tensor(lens, device=kb.device)}
        if all(_is_rel(x) for x in xs):
            ks = [x["k"] for x in xs]
            n = max(k.shape[2] for k in ks)
            kb = ks[0].new_zeros(len(xs), ks[0].shape[1], n, ks[0].shape[3])
            vb = kb.clone()
            for i, x in enumerate(xs):                       # alineadas a la derecha
                m = x["k"].shape[2]
                kb[i, :, n - m:], vb[i, :, n - m:] = x["k"][0], x["v"][0]
            return {"k": kb, "v": vb, "pos": torch.tensor([x["pos"] for x in xs], device=kb.device)}
        return {k: _stack([x[k] for x in xs], extra) for k in x0}
    if isinstance(x0, list):
        if any(not isinstance(x, list) or len(x) != len(x0) for x in xs):
            raise _Mismatch
        return [_stack([x[i] for x in xs], extra) for i in range(len(x0))]
    if isinstance(x0, torch.Tensor):
        if any(not isinstance(x, torch.Tensor) or x.shape != x0.shape for x in xs) or x0.dim() == 0 \
                or x0.shape[0] != 1:
            raise _Mismatch
        return torch.cat(xs, dim=0)
    if any(x != x0 for x in xs):
        raise _Mismatch
    return x0


def _split(x, i: int):
    """Estado de la secuencia ``i`` de un estado de lote, en el formato normal (sin relleno)."""
    if isinstance(x, dict):
        if "rows" in x:                                     # atención por filas
            n = int(x["rows"][i])
            return {"k": x["k"][i:i + 1, :, :n], "v": x["v"][i:i + 1, :, :n]}
        if isinstance(x.get("pos"), torch.Tensor):           # relacional por filas
            pos, stored = int(x["pos"][i]), x["k"].shape[2]
            n = min(pos, stored)
            return {"k": x["k"][i:i + 1, :, stored - n:], "v": x["v"][i:i + 1, :, stored - n:], "pos": pos}
        return {k: _split(v, i) for k, v in x.items()}
    if isinstance(x, list):
        return [_split(v, i) for v in x]
    if isinstance(x, torch.Tensor):
        return x[i:i + 1]
    return x


def _take(x, idx: torch.Tensor):
    """Se queda con las filas ``idx`` de un estado de lote."""
    if isinstance(x, dict):
        return {k: _take(v, idx) for k, v in x.items()}
    if isinstance(x, list):
        return [_take(v, idx) for v in x]
    if isinstance(x, torch.Tensor):
        return x.index_select(0, idx)
    return x


def _advance_globals(m: HN1Model, streams: list[_Stream], memory, n_iters, vis, st: BatchStats) -> None:
    """Deja la parte global de cada secuencia al día con sus parches completos. Las que cerraron
    un parche en este paso se procesan **juntas**, aunque tengan cantidades de parches distintas
    (atención con relleno y posición por fila)."""
    groups: dict[bool, list[_Stream]] = defaultdict(list)
    for s in streams:
        have = 0 if s.inc.g is None else s.inc.g.shape[1]
        if s.cur_pid == have + 1:
            groups[have > 0].append(s)                     # los estados vacíos van aparte
        elif s.cur_pid > have:
            s.inc.upto(s.patch_states, s.cur_pid)
            st.global_calls += 1
            st.global_patches += s.cur_pid - have
    for grp in groups.values():
        merged = None
        if len(grp) > 1:
            try:
                merged = _stack([s.inc.state for s in grp])
            except _Mismatch:
                merged = None
        if merged is None:
            for s in grp:
                s.inc.upto(s.patch_states, s.cur_pid)
                st.global_calls += 1
                st.global_patches += 1
            continue
        new = torch.cat([s.patch_states[s.cur_pid - 1] for s in grp], dim=0).unsqueeze(1)    # [G, 1, dl]
        mem = memory
        if mem is not None and mem.dim() == 3 and mem.shape[0] == 1:
            mem = mem.expand(len(grp), -1, -1)
        out = m._global(new, mem, n_iters, vis=vis, cache=merged)[0]                         # [G, 1, d]
        st.global_calls += 1
        st.global_patches += len(grp)
        for i, s in enumerate(grp):
            s.inc.state = _split(merged, i)
            gi = out[i:i + 1]
            s.inc.g = gi if s.inc.g is None else torch.cat([s.inc.g, gi], dim=1)


# ---------------------------------------------------------------------------
# Bucle de generación en lote
# ---------------------------------------------------------------------------
def _decode(m: HN1Model, streams: list[_Stream], max_new: int, *, stop: bytes, stop_after_line, stop_seq,
            temperature: float, memory, n_iters, vis, st: BatchStats, keep_cache: bool = False) -> None:
    """Genera hasta ``max_new`` bytes para cada secuencia (deja la salida en ``s.out``). Sigue paso
    a paso el bucle de ``HN1Model.generate``. Con ``keep_cache`` las cachés locales vuelven a
    los diccionarios de cada secuencia (para seguir la conversación)."""
    if max_new <= 0 or not streams:
        return
    dev = m.head.weight.device
    ent = m._entropy()
    tail_n = max(ent[2] if ent else 1, 1)
    active = list(streams)
    enc = [_stack([s.enc[i] for s in active], extra=max_new) for i in range(len(m.encoder))]
    dec = [_stack([s.dec[i] for s in active], extra=max_new) for i in range(len(m.decoder))]
    logits = torch.stack([s.logits for s in active])
    for step in range(max_new):
        if temperature > 0:
            nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1)[:, 0].tolist()
        else:
            nxt = logits.argmax(-1).tolist()
        alive = []
        for i, (s, b) in enumerate(zip(active, nxt)):
            s.seq.append(b)
            s.out.append(b)
            stopped = m._should_stop(s.out, b, stop, stop_after_line, stop_seq)
            if stopped and bytes([b]) == stop:
                s.seq.pop()                               # el byte de parada no forma parte del texto
            if not stopped and step < max_new - 1:
                alive.append(i)
        if len(alive) < len(active):
            if keep_cache:
                for i, s in enumerate(active):
                    if i not in alive:
                        s.enc = [_split(c, i) for c in enc]
                        s.dec = [_split(c, i) for c in dec]
            if not alive:
                break
            rows = torch.tensor(alive, device=dev)
            enc, dec = [_take(c, rows) for c in enc], [_take(c, rows) for c in dec]
            active = [active[i] for i in alive]
            nxt = [nxt[i] for i in alive]
        # --- máquina de parches (depende solo de los bytes) ---
        for s, b in zip(active, nxt):
            if ent is None:
                nat = b in _BOUNDARY
            else:
                nat = bool(natural_boundaries(torch.tensor([s.seq[-tail_n:]], device=dev), ent)[0, -1])
            s.advance_patch(nat)
        # --- codificador local en lote (posición propia por fila) ---
        h = m.byte_embed(torch.tensor(nxt, device=dev)[:, None])
        for blk, c in zip(m.encoder, enc):
            h = blk(h, c)
        for i, s in enumerate(active):
            s.last_h = h[i:i + 1, -1]
        # --- parte global: solo las secuencias que cerraron un parche, todas juntas ---
        _advance_globals(m, active, memory, n_iters, vis, st)
        ctx = torch.cat([s.inc.g[:, s.cur_pid - 1] if s.cur_pid > 0 else h.new_zeros(1, m.cfg.d)
                         for s in active], dim=0)
        # --- decodificador local en lote ---
        y = h + m.to_local(ctx.to(h.dtype)).unsqueeze(1)
        for blk, c in zip(m.decoder, dec):
            y = blk(y, c)
        logits = m.head(m.out_norm(y)[:, -1]).float()
        st.steps += 1
        st.rows += len(active)
        for i, s in enumerate(active):
            s.n_cached += 1
            s.logits = logits[i]


@torch.no_grad()
def generate_batch(model: HN1Model, prompts: list[bytes | str], max_new: int = 256, *, stop: bytes = b"\x00",
                   stop_after_line: bytes | None = None, temperature: float = 0.0,
                   memory: torch.Tensor | None = None, max_context: int = 2048, stop_seq: bytes | None = None,
                   n_iters: int | None = None, image: torch.Tensor | None = None, vis: torch.Tensor | None = None,
                   stats: BatchStats | None = None) -> list[bytes]:
    """Genera una continuación por prompt, todas a la vez. Mismos argumentos que
    ``HN1Model.generate``; en modo voraz cada salida es igual a ``model.generate(prompt)``.
    Para mejor-de-N con muestreo, repetir el prompt N veces: se procesa una sola vez."""
    m = model
    st = stats if stats is not None else BatchStats()
    was_training = m.training
    m.eval()
    if vis is None and image is not None:
        vis = m.encode_image(image)
    results: list[bytes | None] = [None] * len(prompts)
    streams: list[_Stream] = []
    where: list[int] = []
    templates: dict[bytes, _Stream] = {}
    ok = _local_ok(m)
    st.prompts += len(prompts)
    for i, p in enumerate(prompts):
        seq = _to_bytes(p)
        if not seq or len(seq) + max_new > max_context or not ok:
            results[i] = m.generate(bytes(seq), max_new, stop=stop, stop_after_line=stop_after_line,
                                    temperature=temperature, memory=memory, max_context=max_context,
                                    stop_seq=stop_seq, n_iters=n_iters, vis=vis)
            st.fallbacks += 1
            continue
        key = bytes(seq)
        if key in templates:
            s = templates[key].clone()
        else:
            s = _Stream(m, memory, n_iters, vis)
            st.prefill_bytes += s.feed(seq)
            st.prefills += 1
            templates[key] = s.clone()
        streams.append(s)
        where.append(i)
    _decode(m, streams, max_new, stop=stop, stop_after_line=stop_after_line, stop_seq=stop_seq,
            temperature=temperature, memory=memory, n_iters=n_iters, vis=vis, st=st)
    for i, s in zip(where, streams):
        results[i] = bytes(s.out)
    m.train(was_training)
    return results


# ---------------------------------------------------------------------------
# Caché de conversación entre turnos
# ---------------------------------------------------------------------------
@dataclass
class ConversationStats:
    turns: int = 0
    fed_bytes: int = 0                # bytes que se procesaron en los prefill (con caché)
    history_bytes: int = 0            # bytes que habría que releer sin caché (suma por turno)
    fallbacks: int = 0                # turnos generados con ``model.generate`` (texto demasiado largo)

    @property
    def reread_saved(self) -> float:
        """Fracción del historial que no hubo que volver a leer."""
        return 1.0 - self.fed_bytes / max(self.history_bytes, 1)


class Conversation:
    """Conversación con caché entre turnos: claves/valores del codificador y del decodificador
    locales, ``GlobalCache`` y estado de parches. Cada turno procesa solo lo nuevo (el texto del
    usuario y el último byte generado, que todavía no había pasado por el modelo).

    ``conv.say(t1)`` y luego ``conv.say(t2)`` da lo mismo que ``model.generate(t1 + salida1 + t2)``.
    Si el historial más lo pedido supera ``max_context`` (o los bloques locales tienen ventana),
    ese turno usa ``model.generate`` con todo el historial y la caché se pone al día después,
    cuando vuelva a entrar.
    """

    def __init__(self, model: HN1Model, *, memory: torch.Tensor | None = None, n_iters: int | None = None,
                 image: torch.Tensor | None = None, vis: torch.Tensor | None = None, max_context: int = 2048):
        self.model, self.memory, self.n_iters, self.max_context = model, memory, n_iters, max_context
        if vis is None and image is not None:
            with torch.no_grad():
                vis = model.encode_image(image)
        self.vis = vis
        self.history: list[int] = []
        self.stats = ConversationStats()
        self._stream = _Stream(model, memory, n_iters, vis)
        self.batch_stats = BatchStats()

    @property
    def cached_bytes(self) -> int:
        return self._stream.n_cached

    def reset(self) -> None:
        self.history = []
        self._stream = _Stream(self.model, self.memory, self.n_iters, self.vis)

    @torch.no_grad()
    def say(self, text: bytes | str, max_new: int = 256, *, stop: bytes = b"\x00",
            stop_after_line: bytes | None = None, temperature: float = 0.0, stop_seq: bytes | None = None) -> bytes:
        m = self.model
        self.history.extend(_to_bytes(text))
        self.stats.turns += 1
        self.stats.history_bytes += len(self.history)
        was_training = m.training
        m.eval()
        try:
            if not self.history or len(self.history) + max_new > self.max_context or not _local_ok(m):
                out = m.generate(bytes(self.history), max_new, stop=stop, stop_after_line=stop_after_line,
                                 temperature=temperature, memory=self.memory, max_context=self.max_context,
                                 stop_seq=stop_seq, n_iters=self.n_iters, vis=self.vis)
                self.stats.fallbacks += 1
                self.stats.fed_bytes += len(self.history)
                self.history.extend(out)
                return out
            s = self._stream
            self.stats.fed_bytes += s.feed(self.history[len(s.seq):])
            s.out = []
            _decode(m, [s], max_new, stop=stop, stop_after_line=stop_after_line, stop_seq=stop_seq,
                    temperature=temperature, memory=self.memory, n_iters=self.n_iters, vis=self.vis,
                    st=self.batch_stats, keep_cache=True)
            out = bytes(s.out)
            s.out = []
            self.history.extend(out)
            return out
        finally:
            m.train(was_training)
