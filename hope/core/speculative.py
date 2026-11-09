"""Generación especulativa con verificación (mejora de velocidad sin parámetros n.º 1).

Idea (Leviathan et al. 2023; *prompt lookup decoding*, Saxena 2023; Medusa, Cai et al. 2024):
se **adivinan** varios bytes siguientes con algo barato y el modelo los **verifica
todos en una sola pasada**. Se aceptan los que coinciden con lo que el modelo
habría elegido; la salida es **idéntica** a ``HN1Model.generate`` en modo voraz
(prueba en ``tests/test_speed.py``), solo que en menos pasadas.

Borradores, en orden:

1. **Búsqueda de n-gramas** en el propio texto (prompt + lo generado) y en un texto
   de referencia opcional (p. ej. lo recuperado de la memoria): si los últimos
   bytes ya aparecieron antes, se propone lo que siguió entonces. Muy efectivo en
   código, citas, tablas y razonamientos que repiten nombres y números.
2. **Cabezas multi-byte** (si el modelo las tiene): predicen los bytes t+2 … t+1+k
   desde el estado del último byte.

El primer byte de cada borrador es siempre el que el modelo eligió con los logits
ya calculados, así que cada pasada produce **al menos** un byte (nunca es más
lenta en bytes por pasada que la generación normal).

Funciona solo en modo voraz (``temperature = 0``); con muestreo se usa ``generate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hope.core.hn1 import GlobalCache, HN1Model, natural_boundaries
from hope.core.layers import truncate_cache


@dataclass
class SpecStats:
    passes: int = 0
    drafted: int = 0
    accepted: int = 0
    generated: int = 0
    by_source: dict = field(default_factory=lambda: {"ngram": 0, "multibyte": 0})

    @property
    def bytes_per_pass(self) -> float:
        return self.generated / max(self.passes, 1)


def ngram_draft(seq: list[int], k: int, reference: bytes = b"", max_n: int = 8, min_n: int = 2) -> list[int]:
    """Busca el sufijo más largo de ``seq`` que ya apareció y devuelve los ``k`` bytes que lo siguieron."""
    hay = bytes(reference) + bytes(seq)
    for n in range(min(max_n, len(seq)), min_n - 1, -1):
        suffix = bytes(seq[-n:])
        pos = hay.rfind(suffix, 0, len(hay) - 1)
        while pos != -1 and pos + n >= len(hay):
            pos = hay.rfind(suffix, 0, pos)
        if pos != -1:
            cont = hay[pos + n:pos + n + k]
            if cont:
                return list(cont)
    return []


@torch.no_grad()
def generate_speculative(model: HN1Model, prompt: bytes | str, max_new: int = 256, *, stop: bytes = b"\x00",
                         stop_after_line: bytes | None = None, memory: torch.Tensor | None = None,
                         max_context: int = 2048, stop_seq: bytes | None = None, n_iters: int | None = None,
                         image: torch.Tensor | None = None, vis: torch.Tensor | None = None,
                         draft_len: int = 8, reference: bytes | str = b"",
                         stats: SpecStats | None = None) -> bytes:
    m = model
    seq = list(prompt.encode("utf-8") if isinstance(prompt, str) else prompt)
    reference = reference.encode("utf-8") if isinstance(reference, str) else reference
    from hope.core.kvcache import kv_active
    if not seq or len(seq) + max_new + draft_len > max_context or kv_active(m) or any(
            b.kind != "attn" for b in [*m.encoder, *m.decoder]):
        return m.generate(prompt, max_new, stop=stop, stop_after_line=stop_after_line, memory=memory,
                          max_context=max_context, stop_seq=stop_seq, n_iters=n_iters, image=image, vis=vis)
    st = stats if stats is not None else SpecStats()
    was_training = m.training
    m.eval()
    dev = m.head.weight.device
    if vis is None and image is not None:
        vis = m.encode_image(image)
    ent = m._entropy()
    tail_n = max(ent[2] if ent else 1, 1)
    enc_cache = [{} for _ in m.encoder]
    dec_cache = [{} for _ in m.decoder]
    dl, d = m.cfg.d_local, m.cfg.d

    def encode(xb):
        h = m.byte_embed(xb)
        for blk, c in zip(m.encoder, enc_cache):
            h = blk(h, c)
        return h

    def decode(y):
        for blk, c in zip(m.decoder, dec_cache):
            y = blk(y, c)
        return m.out_norm(y)

    # --- prefill (igual que HN1Model.generate) ---
    x = torch.tensor([seq], device=dev)
    h = encode(x)
    nat = natural_boundaries(x, ent)[0].tolist()
    ends, in_run, run_start = [], 0, 0
    for t, nb in enumerate(nat):
        if t > 0 and nat[t - 1]:
            run_start = t
        in_run = t - run_start
        ends.append(bool(nb) or (in_run + 1) % m.cfg.max_patch == 0)
    pid = [0]
    for e in ends[:-1]:
        pid.append(pid[-1] + int(e))
    patch_states = [h[:, t] for t, e in enumerate(ends[:-1]) if e]
    inc = GlobalCache(m, memory, n_iters, vis)

    def global_upto(states: list, n: int) -> torch.Tensor:
        return inc.upto(states, n)

    g = global_upto(patch_states, pid[-1])
    g_ext = torch.cat([h.new_zeros(1, 1, d), g], dim=1)
    ctx = m.to_local(g_ext).gather(1, torch.tensor([pid], device=dev).unsqueeze(-1).expand(-1, -1, dl))
    y = decode(h + ctx)
    hidden = y[0, -1]
    logits = m.head(hidden).float()
    state = {"last_end": ends[-1], "last_nat": nat[-1], "in_run": in_run, "cur_pid": pid[-1],
             "last_h": h[:, -1]}
    out: list[int] = []
    done = False
    while len(out) < max_new and not done:
        first = int(logits.argmax())
        k = min(draft_len, max_new - len(out))
        rest = ngram_draft(seq + [first], k - 1, reference)
        src = "ngram" if rest else None
        if not rest and len(m.mb_heads) and k > 1:
            rest = [int(m.head(hidden + hd(hidden)).argmax()) for hd in m.mb_heads][:k - 1]
            src = "multibyte"
        draft = [first] + rest[:k - 1]
        n_before = len(seq)
        # --- máquina de parches para los bytes del borrador (depende solo de los bytes) ---
        H = encode(torch.tensor([draft], device=dev))                     # [1, k, dl]
        s = dict(state)
        provisional = list(patch_states)
        pids, snapshots = [], []
        tmp_seq = list(seq)
        for j, b in enumerate(draft):
            if s["last_end"]:
                provisional.append(s["last_h"])
                s["cur_pid"] += 1
            s["in_run"] = 0 if s["last_nat"] else s["in_run"] + 1
            tmp_seq.append(b)
            tail = torch.tensor([tmp_seq[-tail_n:]], device=dev)
            s["last_nat"] = bool(natural_boundaries(tail, ent)[0, -1])
            s["last_end"] = s["last_nat"] or (s["in_run"] + 1) % m.cfg.max_patch == 0
            s["last_h"] = H[:, j]
            pids.append(s["cur_pid"])
            snapshots.append((dict(s), len(provisional)))
        need = max(pids)
        snap = inc.snapshot()
        gc = global_upto(provisional, need) if need > 0 else None
        ctxs = []
        for p in pids:
            c = gc[:, p - 1] if p > 0 else H.new_zeros(1, d)
            ctxs.append(m.to_local(c))
        Y = decode(H + torch.stack(ctxs, dim=1))                          # [1, k, dl]
        L = m.head(Y[0]).float()                                          # logits después de cada byte
        # --- aceptación: el byte j+1 del borrador debe ser el que el modelo elige tras el byte j ---
        acc = 1
        while acc < len(draft) and int(L[acc - 1].argmax()) == draft[acc]:
            acc += 1
        st.passes += 1
        st.drafted += len(draft) - 1
        st.accepted += acc - 1
        if src:
            st.by_source[src] += acc - 1
        # --- confirmar los aceptados y descartar el resto ---
        for b in draft[:acc]:
            seq.append(b)
            out.append(b)
            n0 = len(out)
            stopped = m._should_stop(out, b, stop, stop_after_line, stop_seq)
            st.generated += 1 - (n0 - len(out))           # el byte de parada no cuenta como generado
            if stopped:
                done = True
                break
        kept = len(seq) - n_before
        truncate_cache(enc_cache + dec_cache, len(seq))
        s_kept, n_states = snapshots[kept - 1]
        state = s_kept
        patch_states = provisional[:n_states]
        if inc.g is not None and inc.g.shape[1] > len(patch_states):
            # Se procesaron parches que el modelo rechazó: volver al estado anterior. Los parches
            # aceptados se recalculan cuando hagan falta.
            inc.restore(snap)
        hidden = Y[0, kept - 1]
        logits = L[kept - 1]
    m.train(was_training)
    return bytes(out)
