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


def rope(x: torch.Tensor, base: float = 10000.0, offset: int = 0) -> torch.Tensor:
    """Rotary position embedding sobre [B, H, T, Dh]; ``offset`` = posición del primer elemento."""
    t, dh = x.shape[-2], x.shape[-1]
    half = dh // 2
    freqs = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(offset, offset + t, device=x.device, dtype=torch.float32)[:, None] * freqs[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:2 * half]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return torch.cat([out, x[..., 2 * half:]], dim=-1) if dh % 2 else out


class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, cache: dict | None = None):
        """``cache``: dict con las claves y valores anteriores (generación incremental).

        Sin caché: atención causal normal. Con caché vacía: igual, y guarda K y V.
        Con caché llena: solo admite un paso nuevo (t = 1), que atiende a todo lo anterior.
        """
        b, t, d = x.shape
        offset = cache["k"].shape[2] if cache is not None and "k" in cache else 0
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rope(q, offset=offset), rope(k, offset=offset)
        if offset:
            if t != 1:
                raise ValueError("con caché, la atención incremental avanza de a un paso")
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if cache is not None:
            cache["k"], cache["v"] = k, v
        return self.out(y.transpose(1, 2).reshape(b, t, d))


# ---------------------------------------------------------------------------
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

    def forward(self, x):
        t = x.shape[1]
        u, gate, dt = self.in_proj(x).chunk(3, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :t].transpose(1, 2))
        delta = F.softplus(dt + self.dt_bias)                     # [B, T, inner] > 0
        log_a = -delta * torch.exp(self.A_log)                    # log del decaimiento, ≤ 0
        h = linear_scan(log_a, delta * u, self.chunk)
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

    def __init__(self, d: int, kind: str, heads: int, mlp_hidden: int, hnhs: bool = False):
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(d)
        self.mixer = CausalSelfAttention(d, heads) if kind == "attn" else SelectiveSSM(d)
        self.norm2 = RMSNorm(d)
        self.mlp = HNHSMLP(d, mlp_hidden) if hnhs else SwiGLU(d, mlp_hidden)

    def forward(self, x, cache: dict | None = None):
        if cache is not None and self.kind != "attn":
            raise ValueError("la caché incremental solo está implementada para bloques de atención")
        x = x + (self.mixer(self.norm1(x), cache) if cache is not None else self.mixer(self.norm1(x)))
        return x + self.mlp(self.norm2(x))
