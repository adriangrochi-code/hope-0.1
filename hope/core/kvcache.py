"""Caché de claves/valores para contexto largo (ARQUITECTURA_HN1 §3.9.2; pedido del usuario).

Tres maneras de gastar menos memoria por posición de contexto, sin parámetros nuevos. Se eligen
por configuración (``HN1Config.kv_bits``, ``kv_group``, ``kv_evict``, ``quest_pages``,
``quest_page``) o con ``configure_kv`` sobre un modelo ya armado:

* **Cuantización** (``bits`` = 8 o 4): cada posición guarda sus claves y valores en int8 (o
  int4 empaquetado, dos por byte) con una escala fp16 por grupo de ``group`` canales (int4 lleva
  además el mínimo del grupo: cuantización asimétrica). Se decuantiza al atender. La posición
  nueva siempre se atiende con sus valores exactos; solo lo guardado pierde precisión.
* **Descarte por ventana** (``evict``): en las capas con ventana deslizante (``swa``) lo que ya
  no se puede ver nunca más se borra; quedan los ``sinks`` y la ventana. Memoria constante.
* **Selección de páginas** (``quest_pages`` = k; Quest, Tang et al. 2024): la caché se divide en
  páginas de ``page`` posiciones; para cada página se guarda el mínimo y el máximo de cada canal
  de las claves. Para la consulta q, ``Σ_d max(q_d·mín_d, q_d·máx_d)`` es una cota superior del
  producto q·k de cualquier clave de la página; se atiende solo a las k páginas con cota más alta
  (siempre la primera, por los *sinks*, y la última, la actual). Si k cubre todas las páginas,
  el resultado es **exacto** (se usa el camino completo). Solo en capas sin ventana y en pasos
  de una posición (generación).

Formato de la caché (un diccionario por capa, como el de siempre; los tensores se reemplazan,
nunca se modifican en el lugar, así ``GlobalCache.snapshot`` sigue funcionando):
``n`` posiciones vistas, ``jpos`` posiciones guardadas, ``kq``/``vq`` datos, ``ks``/``vs`` escalas,
``kz``/``vz`` mínimos (int4), ``pmin``/``pmax`` cotas de página (Quest).
Sin opciones activas la atención usa su caché normal: nada cambia.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class KVOptions:
    bits: int = 16          # 16 = sin cuantizar; 8; 4
    group: int = 32         # canales por grupo de cuantización
    evict: bool = False     # capas con ventana: descartar lo que quedó fuera
    quest_pages: int = 0    # páginas que se atienden (0 = todas)
    page: int = 16          # posiciones por página

    def __post_init__(self):
        if self.bits not in (16, 8, 4):
            raise ValueError("kv_bits debe ser 16, 8 o 4")

    @property
    def active(self) -> bool:
        return self.bits != 16 or self.evict or self.quest_pages > 0


def configure_kv(model, bits: int = 16, group: int = 32, evict: bool = False, quest_pages: int = 0,
                 page: int = 16) -> int:
    """Aplica las opciones a todas las capas de atención causal del modelo (y las anota en la
    configuración). Devuelve cuántas capas quedaron con opciones activas."""
    from hope.core.layers import CausalSelfAttention
    opts = KVOptions(bits, group, evict, quest_pages, page)
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        cfg.kv_bits, cfg.kv_group, cfg.kv_evict, cfg.quest_pages, cfg.quest_page = bits, group, evict, quest_pages, page
    n = 0
    for mod in model.modules():
        if isinstance(mod, CausalSelfAttention):
            mod.kv_opts = opts if opts.active else None
            n += int(opts.active)
    return n


def kv_active(model) -> bool:
    from hope.core.layers import CausalSelfAttention
    return any(isinstance(m, CausalSelfAttention) and getattr(m, "kv_opts", None) is not None
               for m in model.modules())


# ---------------------------------------------------------------------------
# Cuantización por grupos (por posición, a lo largo de los canales)
# ---------------------------------------------------------------------------
def _group(dh: int, group: int) -> int:
    return group if group and dh % group == 0 else dh


def quantize(x: torch.Tensor, bits: int, group: int) -> dict:
    """``x`` [..., dh] → partes cuantizadas. Con 16 bits se guarda tal cual."""
    if bits == 16:
        return {"q": x}
    *lead, dh = x.shape
    g = _group(dh, group)
    xg = x.float().reshape(*lead, dh // g, g)
    if bits == 8:
        scale = (xg.abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-8).half().float()
        q = torch.round(xg / scale).clamp(-127, 127).to(torch.int8).reshape(*lead, dh)
        return {"q": q, "s": scale.squeeze(-1).half()}
    if dh % 2:
        raise ValueError("int4 necesita una dimensión de cabeza par")
    lo = xg.amin(-1, keepdim=True).half().float()
    scale = ((xg.amax(-1, keepdim=True) - lo) / 15.0).clamp_min(1e-8).half().float()
    q = torch.round((xg - lo) / scale).clamp(0, 15).to(torch.uint8).reshape(*lead, dh)
    packed = q[..., 0::2] | (q[..., 1::2] << 4)
    return {"q": packed, "s": scale.squeeze(-1).half(), "z": lo.squeeze(-1).half()}


def dequantize(parts: dict, bits: int, group: int, dh: int, dtype) -> torch.Tensor:
    q = parts["q"]
    if bits == 16:
        return q.to(dtype)
    lead = q.shape[:-1]
    g = _group(dh, group)
    if bits == 8:
        x = q.float().reshape(*lead, dh // g, g) * parts["s"].float().unsqueeze(-1)
    else:
        un = torch.stack([q & 15, q >> 4], dim=-1).reshape(*lead, dh).float()
        x = un.reshape(*lead, dh // g, g) * parts["s"].float().unsqueeze(-1) + parts["z"].float().unsqueeze(-1)
    return x.reshape(*lead, dh).to(dtype)


_PARTS = {"q", "s", "z"}


def _get(cache: dict, kind: str) -> dict:
    return {p: cache[kind + p] for p in _PARTS if kind + p in cache}


def _set(cache: dict, kind: str, parts: dict) -> None:
    for p, v in parts.items():
        cache[kind + p] = v


def _cat(old: dict, new: dict) -> dict:
    return {p: torch.cat([old[p], new[p]], dim=2) for p in new}


def _index(parts: dict, idx: torch.Tensor) -> dict:
    """Se queda con las posiciones ``idx`` (dimensión 2). ``idx`` [S] o [B, H, S]."""
    if idx.dim() == 1:
        return {p: v.index_select(2, idx) for p, v in parts.items()}
    return {p: v.gather(2, idx.unsqueeze(-1).expand(*idx.shape, v.shape[-1])) for p, v in parts.items()}


def cache_nbytes(obj) -> int:
    """Bytes de todos los tensores de una caché (diccionarios y listas anidados)."""
    if isinstance(obj, dict):
        return sum(cache_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(cache_nbytes(v) for v in obj)
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    return 0


def attention_caches(obj) -> list[dict]:
    """Las cachés de atención (``k``/``v`` o comprimidas) dentro de un estado anidado."""
    if isinstance(obj, dict):
        if ("k" in obj and isinstance(obj["k"], torch.Tensor) and obj["k"].dim() == 4 and "pos" not in obj) \
                or "kq" in obj:
            return [obj]
        return [c for v in obj.values() for c in attention_caches(v)]
    if isinstance(obj, (list, tuple)):
        return [c for v in obj for c in attention_caches(v)]
    return []


def attention_nbytes(obj) -> int:
    """Bytes de claves/valores (y escalas y cotas de página) de las cachés de atención."""
    return sum(cache_nbytes({k: v for k, v in c.items() if k not in ("jpos", "n")}) for c in attention_caches(obj))


# ---------------------------------------------------------------------------
# Selección de páginas (Quest)
# ---------------------------------------------------------------------------
def page_bounds_update(cache: dict, k_new: torch.Tensor, n0: int, page: int) -> None:
    """Actualiza mínimo y máximo por canal de cada página con las claves nuevas [B, H, t, dh]."""
    b, h, t, dh = k_new.shape
    n = n0 + t
    p_old = cache["pmin"].shape[2] if "pmin" in cache else 0
    p_new = -(-n // page)
    kf = k_new.detach().float()
    pmin = kf.new_full((b, h, p_new, dh), math.inf)
    pmax = kf.new_full((b, h, p_new, dh), -math.inf)
    if p_old:
        pmin[:, :, :p_old] = cache["pmin"].float()
        pmax[:, :, :p_old] = cache["pmax"].float()
    idx = (torch.arange(n0, n, device=k_new.device) // page).view(1, 1, t, 1).expand(b, h, t, dh)
    pmin.scatter_reduce_(2, idx, kf, "amin", include_self=True)
    pmax.scatter_reduce_(2, idx, kf, "amax", include_self=True)
    cache["pmin"], cache["pmax"] = pmin.to(k_new.dtype), pmax.to(k_new.dtype)


def quest_select(q: torch.Tensor, pmin: torch.Tensor, pmax: torch.Tensor, n: int, page: int,
                 k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Posiciones de las ``k`` páginas con cota más alta para la consulta ``q`` [B, H, 1, dh].

    Devuelve (índices [B, Hkv, k·page], válidos [B, Hkv, k·page]). Con GQA, la cota de una
    página para un grupo es la mayor entre sus cabezas de consulta."""
    b, h, _, dh = q.shape
    hkv, p = pmin.shape[1], pmin.shape[2]
    rep = h // hkv
    qg = q[:, :, 0].float().view(b, hkv, rep, 1, dh)
    bound = torch.maximum(qg * pmin.float().unsqueeze(2), qg * pmax.float().unsqueeze(2)).sum(-1)   # [B, Hkv, rep, P]
    score = bound.amax(2)
    score[..., 0] = math.inf                          # primera página (sinks) y página actual: siempre
    score[..., p - 1] = math.inf
    pages = score.topk(min(k, p), dim=-1).indices.sort(-1).values                                 # [B, Hkv, k]
    tok = (pages.unsqueeze(-1) * page + torch.arange(page, device=q.device)).flatten(-2)
    valid = tok < n
    return tok.clamp_max(n - 1), valid


# ---------------------------------------------------------------------------
# Atención con caché comprimida
# ---------------------------------------------------------------------------
def attend(mod, x: torch.Tensor, cache: dict) -> torch.Tensor:
    """Paso de ``CausalSelfAttention`` con la caché comprimida (``mod.kv_opts``).

    Lo guardado se atiende decuantizado y sin gradiente; las posiciones nuevas, con sus valores
    exactos (y con gradiente: sirve también para entrenar por segmentos con memoria)."""
    opts: KVOptions = mod.kv_opts
    b, t, d = x.shape
    H, Hkv, dh = mod.heads, mod.kv_heads, mod.dh
    n0 = int(cache.get("n", 0))
    qkv = mod.qkv(x).view(b, t, H + 2 * Hkv, dh).transpose(1, 2)
    q, k, v = qkv.split([H, Hkv, Hkv], dim=1)
    q, k = mod._rope(q, n0), mod._rope(k, n0)
    q = mod._qscale(q)
    dev = x.device
    newpos = torch.arange(n0, n0 + t, device=dev)
    stored = int(cache["kq"].shape[2]) if "kq" in cache else 0
    evicting = opts.evict and mod.window is not None
    per_head = None                                    # posiciones por cabeza (Quest)
    if stored:
        kp, vp = _get(cache, "k"), _get(cache, "v")
        # Sin descarte, lo guardado son las posiciones 0..n0−1 (no hace falta anotarlas).
        jpos = cache["jpos"].long() if "jpos" in cache else torch.arange(stored, device=dev)
        n_pages = -(-n0 // opts.page)
        if (opts.quest_pages and mod.window is None and t == 1 and "pmin" in cache
                and n_pages > opts.quest_pages):
            tok, valid = quest_select(q, cache["pmin"], cache["pmax"], n0, opts.page, opts.quest_pages)
            kp, vp = _index(kp, tok), _index(vp, tok)
            per_head = (tok, valid)
        kd = dequantize(kp, opts.bits, opts.group, dh, q.dtype)
        vd = dequantize(vp, opts.bits, opts.group, dh, q.dtype)
        kk, vv = torch.cat([kd, k], dim=2), torch.cat([vd, v], dim=2)
    else:
        kk, vv, jpos = k, v, newpos[:0]
    # --- guardar lo nuevo ---
    kq, vq = quantize(k.detach(), opts.bits, opts.group), quantize(v.detach(), opts.bits, opts.group)
    if stored:
        kq, vq = _cat(_get(cache, "k"), kq), _cat(_get(cache, "v"), vq)
    if opts.quest_pages and mod.window is None:
        page_bounds_update(cache, k, n0, opts.page)
    n = n0 + t
    if evicting:
        jall = torch.cat([jpos, newpos]) if stored else newpos
        keep = (jall < mod.sinks) | (jall >= n - mod.window + 1)
        if not bool(keep.all()):
            idx = keep.nonzero(as_tuple=True)[0]
            kq, vq, jall = _index(kq, idx), _index(vq, idx), jall[idx]
        cache["jpos"] = jall.to(torch.int32)
    _set(cache, "k", kq)
    _set(cache, "v", vq)
    cache["n"] = n
    # --- atención ---
    rep = H // Hkv
    if rep > 1:
        kk, vv = kk.repeat_interleave(rep, dim=1), vv.repeat_interleave(rep, dim=1)
    i = newpos[:, None]
    if per_head is None:
        j = torch.cat([jpos, newpos])[None, :]
        allowed = j <= i
        if mod.window is not None:
            allowed &= (i - j < mod.window) | (j < mod.sinks)
        mask = allowed                                                    # [t, S]
    else:
        tok, valid = per_head                                            # t == 1, sin ventana
        cur = torch.ones(b, Hkv, 1, dtype=torch.bool, device=dev)
        mask = torch.cat([valid, cur], dim=-1).repeat_interleave(rep, dim=1).unsqueeze(2)   # [B, H, 1, S]
    y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
    return mod.out(y.transpose(1, 2).reshape(b, t, d))


# ---------------------------------------------------------------------------
# Memoria por posición de contexto (estimación analítica)
# ---------------------------------------------------------------------------
def kv_bytes_per_position(kv_heads: int, dh: int, bits: int = 16, group: int = 32, dtype_bytes: int = 2,
                          quest_page: int = 0) -> float:
    """Bytes de claves + valores que guarda una capa por posición (con escalas y cotas de página)."""
    g = _group(dh, group)
    if bits == 16:
        per = dh * dtype_bytes
    elif bits == 8:
        per = dh + 2 * (dh // g)
    else:
        per = dh // 2 + 4 * (dh // g)
    total = 2 * kv_heads * per
    if quest_page:
        total += 2 * kv_heads * dh * dtype_bytes / quest_page
    return total
