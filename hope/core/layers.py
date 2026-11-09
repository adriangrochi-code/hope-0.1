"""Capas del núcleo HN-1 (ARQUITECTURA_HN1 §3).

Todo está escrito en PyTorch puro, sin kernels CUDA, para que funcione en la
Intel Arc A770 (backend ``xpu``), en GPUs NVIDIA y en CPU.

* ``RMSNorm``, ``rope`` — piezas estándar.
* ``CausalSelfAttention`` — atención causal con ``scaled_dot_product_attention``.
* ``SelectiveSSM`` — modelo de espacio de estados diagonal y selectivo, al estilo
  Mamba pero simplificado. Usa un *scan* por bloques en lugar del kernel CUDA de
  ``mamba-ssm``.
* ``SwiGLU`` y ``HNHSMLP`` — MLP estándar y MLP con poblaciones excitadoras e
  inhibidoras de signo fijo (HNHS §4.1–4.2, ley de Dale). La segunda es una
  **ablación**: se queda solo si mejora (ARQUITECTURA_HN1 §3.6).
* ``MemoryCrossAttention`` — interfaz de memoria (§3.7) con compuerta en cero.
* ``Block`` — mezclador (atención o SSM) más MLP, con residuales y pre-normalización.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_freqs(dh: int, base: float = 10000.0, yarn: tuple[float, float] | None = None,
               device=None) -> torch.Tensor:
    """Frecuencias de RoPE (fp32). Con ``yarn = (escala, largo_original)`` aplica la interpolación
    por bandas de YaRN (Peng et al. 2023): las frecuencias cuya longitud de onda entra muchas veces
    en el largo original (r > β = 32) no se tocan, las que no completan una vuelta (r < α = 1) se
    dividen por la escala (interpolación de posiciones) y las del medio se mezclan en rampa lineal."""
    half = dh // 2
    freqs = base ** (-torch.arange(half, device=device, dtype=torch.float32) / half)
    if yarn is None:
        return freqs
    scale, orig = yarn
    r = orig * freqs / (2 * math.pi)                  # vueltas completas en el largo original
    gamma = ((r - 1.0) / (32.0 - 1.0)).clamp(0.0, 1.0)
    return freqs / scale * (1 - gamma) + freqs * gamma


def yarn_mscale(scale: float) -> float:
    """Ajuste de temperatura de YaRN: los logits de atención se multiplican por (0,1·ln s + 1)²."""
    return 1.0 if scale <= 1.0 else (0.1 * math.log(scale) + 1.0) ** 2


def rope(x: torch.Tensor, base: float = 10000.0, offset: int = 0, freqs: torch.Tensor | None = None) -> torch.Tensor:
    """Rotary position embedding sobre [B, H, T, Dh]; ``offset`` = posición del primer elemento."""
    t, dh = x.shape[-2], x.shape[-1]
    half = dh // 2
    if freqs is None:
        freqs = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(offset, offset + t, device=x.device, dtype=torch.float32)[:, None] * freqs[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:2 * half]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return torch.cat([out, x[..., 2 * half:]], dim=-1) if dh % 2 else out


def rope_rows(x: torch.Tensor, base: float, pos: torch.Tensor, freqs: torch.Tensor | None = None) -> torch.Tensor:
    """``rope`` para un paso [B, H, 1, Dh] con una posición distinta por fila (``pos``: [B]).
    Mismas cuentas que ``rope``: da exactamente los mismos valores para la misma posición."""
    dh = x.shape[-1]
    half = dh // 2
    if freqs is None:
        freqs = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = pos.to(torch.float32)[:, None] * freqs[None]
    cos, sin = ang.cos().to(x.dtype)[:, None, None], ang.sin().to(x.dtype)[:, None, None]
    x1, x2 = x[..., :half], x[..., half:2 * half]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return torch.cat([out, x[..., 2 * half:]], dim=-1) if dh % 2 else out


class CausalSelfAttention(nn.Module):
    """Atención causal con opciones de la investigación (informe «Técnicas de generación rápida»):

    * ``kv_heads`` < ``heads``: GQA (Ainslie et al. 2023), menos claves/valores que guardar;
    * ``window``: atención con ventana deslizante (solo las últimas ``window`` posiciones);
    * ``sinks``: las primeras posiciones siempre visibles (StreamingLLM, Xiao et al. 2023),
      para que la ventana no desestabilice la atención con textos largos.
    Con caché admite avanzar uno o varios pasos (verificación de bytes adivinados).

    Contexto largo (§3.9.2): ``rope_base`` (NTK) o ``rope_yarn = (escala, largo original)`` (YaRN,
    con ajuste de temperatura ``q_scale``) los pone ``set_context_scale``; ``kv_opts``
    (``hope/core/kvcache.py``) activa la caché cuantizada, el descarte por ventana y la selección
    de páginas. Sin esas opciones el cálculo es el de siempre.
    """

    def __init__(self, d: int, heads: int, kv_heads: int | None = None, window: int | None = None,
                 sinks: int = 0):
        super().__init__()
        assert d % heads == 0
        self.heads, self.kv_heads = heads, kv_heads or heads
        assert heads % self.kv_heads == 0
        self.dh = d // heads
        self.window, self.sinks = window, sinks
        self.qkv = nn.Linear(d, (heads + 2 * self.kv_heads) * self.dh, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def _mask(self, t_new: int, t_total: int, device) -> torch.Tensor | None:
        offset = t_total - t_new
        if self.window is None and offset == 0:
            return None                                     # se usa is_causal
        i = torch.arange(offset, t_total, device=device)[:, None]
        j = torch.arange(t_total, device=device)[None, :]
        allowed = j <= i
        if self.window is not None:
            allowed &= (i - j < self.window) | (j < self.sinks)
        return allowed

    def _freqs(self, device) -> torch.Tensor | None:
        yarn = getattr(self, "rope_yarn", None)
        return None if yarn is None else rope_freqs(self.dh, 10000.0, yarn, device)

    def _rope(self, x, offset: int):
        base = getattr(self, "rope_base", 10000.0)        # se escala para extender el contexto (NTK)
        return rope(x, base, offset, self._freqs(x.device))

    def _qscale(self, q):
        s = getattr(self, "q_scale", 1.0)                  # temperatura de YaRN
        return q if s == 1.0 else q * s

    def forward(self, x, cache: dict | None = None):
        if cache is not None and "rows" in cache:
            return self._forward_rows(x, cache)
        if cache is not None and getattr(self, "kv_opts", None) is not None:
            from hope.core.kvcache import attend
            return attend(self, x, cache)
        b, t, d = x.shape
        offset = cache["k"].shape[2] if cache is not None and "k" in cache else 0
        qkv = self.qkv(x).view(b, t, self.heads + 2 * self.kv_heads, self.dh).transpose(1, 2)
        q, k, v = qkv.split([self.heads, self.kv_heads, self.kv_heads], dim=1)
        q, k = self._rope(q, offset), self._rope(k, offset)
        q = self._qscale(q)
        if offset:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        if cache is not None:
            cache["k"], cache["v"] = k, v
        if self.kv_heads != self.heads:
            rep = self.heads // self.kv_heads
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        mask = self._mask(t, k.shape[2], x.device)
        if mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=t > 1)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(b, t, d))

    def _forward_rows(self, x, cache: dict):
        """Un paso para B secuencias de largos distintos (generación en lote, ``hope/core/batched.py``).

        ``cache = {"k", "v": [B, kv_heads, capacidad, dh], "rows": [B]}``: la fila i tiene
        ``rows[i]`` posiciones guardadas al principio del búfer (el resto es relleno). La clave
        nueva se escribe en su posición (el búfer es del lote y se modifica en el lugar) y la
        máscara tapa el relleno. Solo se usa si la caché trae ``"rows"``."""
        b, t, d = x.shape
        assert t == 1, "el camino por filas avanza un paso por vez"
        pos = cache["rows"]
        qkv = self.qkv(x).view(b, 1, self.heads + 2 * self.kv_heads, self.dh).transpose(1, 2)
        q, k, v = qkv.split([self.heads, self.kv_heads, self.kv_heads], dim=1)
        base = getattr(self, "rope_base", 10000.0)
        fr = self._freqs(x.device)
        q, k = rope_rows(q, base, pos, fr), rope_rows(k, base, pos, fr)
        q = self._qscale(q)
        need = int(pos.max()) + 1
        if cache["k"].shape[2] < need:                      # crece de a bloques
            grow = max(need - cache["k"].shape[2], 64)
            pad = cache["k"].new_zeros(b, self.kv_heads, grow, self.dh)
            cache["k"], cache["v"] = torch.cat([cache["k"], pad], dim=2), torch.cat([cache["v"], pad], dim=2)
        rows = torch.arange(b, device=x.device)
        cache["k"][rows, :, pos] = k[:, :, 0]
        cache["v"][rows, :, pos] = v[:, :, 0]
        cache["rows"] = pos + 1
        kk, vv = cache["k"][:, :, :need], cache["v"][:, :, :need]
        if self.kv_heads != self.heads:
            rep = self.heads // self.kv_heads
            kk, vv = kk.repeat_interleave(rep, dim=1), vv.repeat_interleave(rep, dim=1)
        j = torch.arange(need, device=x.device)[None]
        allowed = j <= pos[:, None]
        if self.window is not None:
            allowed &= (pos[:, None] - j < self.window) | (j < self.sinks)
        y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=allowed[:, None, None])
        return self.out(y.transpose(1, 2).reshape(b, 1, d))


def truncate_cache(caches: list[dict], length: int) -> None:
    """Descarta del caché las posiciones a partir de ``length`` (bytes adivinados rechazados)."""
    for c in caches:
        if "jpos" in c:
            raise ValueError("truncate_cache no admite la caché comprimida (hope/core/kvcache.py)")
        if "k" in c:
            c["k"], c["v"] = c["k"][:, :, :length], c["v"][:, :, :length]


def causal_conv(conv: nn.Conv1d, x: torch.Tensor, cache: dict | None = None, key: str = "conv") -> torch.Tensor:
    """Convolución causal por canal sobre [B, T, C]. Con ``cache`` sigue desde los últimos
    k−1 valores vistos (generación incremental); sin caché equivale a rellenar con ceros."""
    t = x.shape[1]
    if cache is None:
        return conv(x.transpose(1, 2))[..., :t].transpose(1, 2)
    k = conv.kernel_size[0]
    hist = cache.get(key)
    if hist is None:
        hist = x.new_zeros(x.shape[0], k - 1, x.shape[2])
    full = torch.cat([hist, x], dim=1)
    y = F.conv1d(full.transpose(1, 2), conv.weight, conv.bias, groups=conv.groups).transpose(1, 2)
    cache[key] = full[:, full.shape[1] - (k - 1):]
    return y


def scan(log_a: torch.Tensor, u: torch.Tensor, cache: dict | None = None, key: str = "h",
         chunk: int = 16) -> torch.Tensor:
    """``linear_scan`` que continúa desde el estado guardado en ``cache`` (si lo hay)."""
    h = linear_scan(log_a, u, chunk, h0=cache.get(key) if cache is not None else None)
    if cache is not None:
        cache[key] = h[:, -1]
    return h


MIN_LOG_DECAY = -5.0   # a_t ≥ e^-5 ≈ 0.0067: más fuerte equivale en la práctica a reiniciar


def linear_scan(log_a: torch.Tensor, u: torch.Tensor, chunk: int = 16,
                h0: torch.Tensor | None = None) -> torch.Tensor:
    """Resuelve ``h_t = a_t * h_{t-1} + u_t`` a lo largo de la dimensión 1.

    Recibe ``log_a`` = log(a_t) ≤ 0, formas [B, T, ...]. Se pasa el logaritmo y
    no ``a``: calcular ``log(exp(-x))`` produce gradientes infinitos cuando
    ``exp(-x)`` se redondea a 0 (eso causaba NaN al entrenar). Dentro de cada bloque usa la forma
    cerrada con productos acumulados en espacio logarítmico; entre bloques
    propaga el estado de forma secuencial (T / chunk iteraciones).

    Estabilidad: con ``log a ≥ MIN_LOG_DECAY`` y bloques de 16 pasos, el rango
    de exponentes dentro de un bloque es ≤ 80, que entra en float32.
    """
    if chunk * -MIN_LOG_DECAY > 80:
        raise ValueError("chunk demasiado grande para el decaimiento mínimo (desborde en float32)")
    b, t = log_a.shape[:2]
    rest = log_a.shape[2:]
    h = torch.zeros((b, *rest), device=log_a.device, dtype=u.dtype) if h0 is None else h0
    outs = []
    log_a = log_a.float().clamp(MIN_LOG_DECAY, 0.0)
    for s in range(0, t, chunk):
        la = log_a[:, s:s + chunk]                       # [B, L, ...]
        uu = u[:, s:s + chunk].float()
        cum = la.cumsum(dim=1)                           # log Π_{r≤t} a_r dentro del bloque
        # h_t = e^{cum_t} · (h0 + Σ_{s≤t} e^{-cum_s} u_s)
        # Para estabilidad, restamos el mínimo del bloque antes de exponenciar.
        shift = cum.amin(dim=1, keepdim=True)
        inner = torch.cumsum(torch.exp(shift - cum) * uu, dim=1)
        hs = torch.exp(cum - shift) * inner + torch.exp(cum) * h.float().unsqueeze(1)
        outs.append(hs.to(u.dtype))
        h = hs[:, -1].to(u.dtype)
    return torch.cat(outs, dim=1)


class SelectiveSSM(nn.Module):
    """SSM diagonal selectivo (simplificación de Mamba; Gu & Dao 2023).

    x → proyección a (u, compuerta, Δ) → convolución causal corta sobre u →
    h_t = exp(-Δ_t·A) ⊙ h_{t-1} + Δ_t·u_t → salida h ⊙ silu(compuerta) → proyección.
    El decaimiento depende de la entrada (selectivo) y es distinto por canal.
    """

    def __init__(self, d: int, expand: int = 2, conv: int = 4, chunk: int = 16):
        super().__init__()
        inner = expand * d
        self.inner, self.chunk = inner, chunk
        self.in_proj = nn.Linear(d, 3 * inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, conv, groups=inner, padding=conv - 1)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, inner)))
        self.dt_bias = nn.Parameter(torch.full((inner,), math.log(math.expm1(0.01))))
        self.out_proj = nn.Linear(inner, d, bias=False)

    def forward(self, x, cache: dict | None = None):
        u, gate, dt = self.in_proj(x).chunk(3, dim=-1)
        u = F.silu(causal_conv(self.conv, u, cache))
        delta = F.softplus(dt + self.dt_bias)                     # [B, T, inner] > 0
        log_a = -delta * torch.exp(self.A_log)                    # log del decaimiento, ≤ 0
        h = scan(log_a, delta * u, cache, chunk=self.chunk)
        return self.out_proj(h * F.silu(gate))


# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w_in = nn.Linear(d, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        g, u = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(g) * u)


class HNHSMLP(nn.Module):
    """MLP con poblaciones excitadoras e inhibidoras (HNHS §4.1–4.2; ley de Dale).

    Las unidades ocultas tienen actividad no negativa. Las excitadoras solo
    pueden sumar a la salida y las inhibidoras solo restar: el signo de cada
    peso de salida está fijado por la población (``softplus(W)·signo``; DANN,
    Cornford et al. 2021). Una normalización divisiva acota la actividad total.
    """

    def __init__(self, d: int, hidden: int, inhibitory_frac: float = 0.25):
        super().__init__()
        self.n_inh = max(1, int(hidden * inhibitory_frac))
        self.w_in = nn.Linear(d, hidden)
        self.w_out_raw = nn.Parameter(torch.randn(d, hidden) * (1.0 / hidden) ** 0.5 - 1.0)
        sign = torch.ones(hidden)
        sign[-self.n_inh:] = -1.0
        self.register_buffer("sign", sign)

    def output_weight(self) -> torch.Tensor:
        return F.softplus(self.w_out_raw) * self.sign

    def forward(self, x):
        r = F.relu(self.w_in(x))                                    # tasas ≥ 0
        r = r / (1.0 + r.mean(dim=-1, keepdim=True))                # normalización divisiva
        return F.linear(r, self.output_weight())


class MemoryCrossAttention(nn.Module):
    """Atención cruzada hacia fragmentos recuperados de AHMC/MDD (§3.7).

    La compuerta empieza en 0: agregar memoria a un modelo ya entrenado no
    cambia su salida hasta que el entrenamiento la habilita.
    """

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x, memory):
        if memory is None:
            return x
        b, t, d = x.shape
        m = memory.shape[1]
        q = self.q(self.norm(x)).view(b, t, self.heads, -1).transpose(1, 2)
        k, v = self.kv(memory).view(b, m, 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return x + torch.tanh(self.gate) * self.out(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    """Mezclador (``"attn"`` o ``"ssm"``) + MLP, ambos con residual."""

    def __init__(self, d: int, kind: str, heads: int, mlp_hidden: int, hnhs: bool = False,
                 kv_heads: int | None = None, window: int = 256, sinks: int = 4):
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(d)
        if kind == "attn":
            self.mixer = CausalSelfAttention(d, heads, kv_heads)
        elif kind == "swa":                                  # ventana deslizante + sinks
            self.mixer = CausalSelfAttention(d, heads, kv_heads, window=window, sinks=sinks)
        elif kind == "ssm":
            self.mixer = SelectiveSSM(d)
        else:
            raise ValueError(f"tipo de bloque desconocido: {kind}")
        self.norm2 = RMSNorm(d)
        self.mlp = HNHSMLP(d, mlp_hidden) if hnhs else SwiGLU(d, mlp_hidden)

    def forward(self, x, cache: dict | None = None):
        x = x + (self.mixer(self.norm1(x), cache) if cache is not None else self.mixer(self.norm1(x)))
        return x + self.mlp(self.norm2(x))
