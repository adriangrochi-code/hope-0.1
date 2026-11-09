"""Zonas de especialistas heterogéneos (HNHS §4, §6, §13; AANS §3.3; decisión del usuario).

Hope no es un único tipo de red: cada zona reúne **especialistas de tipos
distintos** y un router aprendido decide, posición por posición, cuáles
trabajan. Sus salidas se combinan; la competencia del router es una forma de
inhibición entre especialistas (HNHS §7.2).

Tipos iniciales (la lista es extensible; AANS puede agregar tipos nuevos):

=============  =================================================  ======================================
tipo           red                                                familia HNHS / uso
=============  =================================================  ======================================
``mlp``        perceptrón con GELU                                transformación general
``glu``        compuerta multiplicativa: σ(Wg x) ⊙ (Wv x)         operadores GATE / MUL (HNHS §6)
``ei``         poblaciones excitadoras/inhibidoras con signo      EN / IN, normalización divisiva
               fijo (ley de Dale) y normalización divisiva
``recurrente`` constante de tiempo dependiente de la entrada      RN; integración temporal
               (inspirada en Liquid/CfC): h ← a·h + (1−a)·u,
               a = exp(−Δt/τ(x))
``conv``       convolución causal en el tiempo + mezcla           DN: detectores de patrones locales
``asociativa`` memoria de Hopfield moderna: ranuras clave→valor   AN / MEM: recuperación por contenido
``relacional`` atención causal local sobre las últimas posiciones relaciones entre elementos cercanos
=============  =================================================  ======================================

Interfaz común de un especialista:
  * ``forward(x)`` con ``x`` de forma [B, T, d] → [B, T, d] (causal);
  * ``sequence_aware``: si necesita la secuencia (recurrente, conv, relacional)
    o trabaja posición por posición;
  * ``widen(extra, seed)``: agrega unidades **sin cambiar la función** (ASG);
  * ``fc2``: capa de salida; su entrada son las «neuronas» que mide el
    diagnóstico LTH (``hope.growth.lth``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hope.core.layers import RMSNorm, causal_conv, rope, rope_rows, scan

KINDS = ("mlp", "glu", "ei", "recurrente", "conv", "asociativa", "relacional")


def _rand_rows(n: int, fan_in: int, g: torch.Generator, like: torch.Tensor) -> torch.Tensor:
    bound = 1.0 / math.sqrt(fan_in)
    return ((torch.rand(n, fan_in, generator=g) * 2 - 1) * bound).to(like)


def _grow_linear_out(lin: nn.Linear, extra: int, g: torch.Generator) -> nn.Linear:
    """Agrega ``extra`` salidas (filas) con pesos aleatorios."""
    new = nn.Linear(lin.in_features, lin.out_features + extra, bias=lin.bias is not None).to(lin.weight)
    with torch.no_grad():
        new.weight.copy_(torch.cat([lin.weight, _rand_rows(extra, lin.in_features, g, lin.weight)]))
        if lin.bias is not None:
            new.bias.copy_(torch.cat([lin.bias, torch.zeros(extra, dtype=lin.bias.dtype, device=lin.bias.device)]))
    return new


def _grow_linear_in(lin: nn.Linear, extra: int) -> nn.Linear:
    """Agrega ``extra`` entradas (columnas) con peso cero: la salida no cambia."""
    new = nn.Linear(lin.in_features + extra, lin.out_features, bias=lin.bias is not None).to(lin.weight)
    with torch.no_grad():
        new.weight.copy_(torch.cat([lin.weight, torch.zeros(lin.out_features, extra, dtype=lin.weight.dtype,
                                                            device=lin.weight.device)], dim=1))
        if lin.bias is not None:
            new.bias.copy_(lin.bias)
    return new


class Specialist(nn.Module):
    kind = "base"
    sequence_aware = False

    @property
    def hidden(self) -> int:
        return self.fc2.in_features

    def zero_output_(self) -> None:
        with torch.no_grad():
            self.fc2.weight.zero_()
            if self.fc2.bias is not None:
                self.fc2.bias.zero_()

    def widen(self, extra: int, seed: int = 0) -> None:
        raise NotImplementedError(f"{self.kind} no implementa widen")


class MLPSpecialist(Specialist):
    kind = "mlp"

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, hidden), nn.Linear(hidden, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

    def widen(self, extra, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.fc1, self.fc2 = _grow_linear_out(self.fc1, extra, g), _grow_linear_in(self.fc2, extra)


class GLUSpecialist(Specialist):
    kind = "glu"

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.fc1, self.gate, self.fc2 = nn.Linear(d, hidden), nn.Linear(d, hidden), nn.Linear(hidden, d)

    def forward(self, x):
        return self.fc2(torch.sigmoid(self.gate(x)) * self.fc1(x))

    def widen(self, extra, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.fc1, self.gate = _grow_linear_out(self.fc1, extra, g), _grow_linear_out(self.gate, extra, g)
        self.fc2 = _grow_linear_in(self.fc2, extra)


class EISpecialist(Specialist):
    """Poblaciones excitadoras e inhibidoras con signo fijo (ley de Dale)."""

    kind = "ei"

    def __init__(self, d: int, hidden: int, inhibitory_frac: float = 0.25):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)                 # magnitudes; el signo lo pone la población
        nn.init.uniform_(self.fc2.weight, 0.0, 1.0 / math.sqrt(hidden))
        sign = torch.ones(hidden)
        sign[hidden - max(1, int(hidden * inhibitory_frac)):] = -1.0
        self.register_buffer("sign", sign)
        # Tamaño de referencia de la normalización: fijo, para que ensanchar no la altere.
        self.register_buffer("norm_ref", torch.tensor(float(hidden)))

    def forward(self, x):
        r = F.relu(self.fc1(x))
        r = r / (1.0 + r.sum(dim=-1, keepdim=True) / self.norm_ref)    # normalización divisiva
        return F.linear(r, self.fc2.weight.abs() * self.sign, self.fc2.bias)

    def widen(self, extra, seed=0):
        """Agrega unidades nuevas (mitad excitadoras, mitad inhibidoras) con salida en cero.

        Las nuevas arrancan inactivas (sesgo muy negativo → ReLU = 0), así que no
        suman a la normalización divisiva; como su tamaño de referencia es fijo,
        la función se preserva exactamente.
        """
        g = torch.Generator().manual_seed(seed)
        self.fc1 = _grow_linear_out(self.fc1, extra, g)
        with torch.no_grad():
            self.fc1.bias[-extra:] = -1e4                  # inactivas al inicio
        self.fc2 = _grow_linear_in(self.fc2, extra)
        new_sign = torch.ones(extra, device=self.sign.device)
        new_sign[extra // 2:] = -1.0
        self.sign = torch.cat([self.sign, new_sign])


class RecurrentSpecialist(Specialist):
    """Constante de tiempo dependiente de la entrada (inspirada en Liquid Time-Constant / CfC)."""

    kind = "recurrente"
    sequence_aware = True

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)         # objetivo u(x)
        self.tau = nn.Linear(d, hidden)         # log τ(x)
        self.fc2 = nn.Linear(hidden, d)

    def forward(self, x, cache: dict | None = None):
        u = torch.tanh(self.fc1(x))
        log_a = -torch.exp(-self.tau(x).float().clamp(-6, 6))     # log a = −Δt/τ con Δt = 1
        a = torch.exp(log_a).to(u.dtype)
        h = scan(log_a, (1 - a) * u, cache)
        return self.fc2(h)

    def widen(self, extra, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.fc1, self.tau = _grow_linear_out(self.fc1, extra, g), _grow_linear_out(self.tau, extra, g)
        self.fc2 = _grow_linear_in(self.fc2, extra)


class ConvSpecialist(Specialist):
    """Detectores de patrones locales: convolución causal por canal + mezcla."""

    kind = "conv"
    sequence_aware = True

    def __init__(self, d: int, hidden: int, kernel: int = 5):
        super().__init__()
        self.kernel = kernel
        self.fc1 = nn.Linear(d, hidden)
        self.conv = nn.Conv1d(hidden, hidden, kernel, groups=hidden, padding=kernel - 1)
        self.fc2 = nn.Linear(hidden, d)

    def forward(self, x, cache: dict | None = None):
        h = causal_conv(self.conv, self.fc1(x), cache)
        return self.fc2(F.gelu(h))

    def widen(self, extra, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.fc1 = _grow_linear_out(self.fc1, extra, g)
        old = self.conv
        new = nn.Conv1d(old.in_channels + extra, old.out_channels + extra, self.kernel,
                        groups=old.groups + extra, padding=self.kernel - 1).to(old.weight)
        with torch.no_grad():
            new.weight[:old.out_channels] = old.weight
            new.bias[:old.out_channels] = old.bias
        self.conv = new
        self.fc2 = _grow_linear_in(self.fc2, extra)


class AssociativeSpecialist(Specialist):
    """Memoria asociativa de Hopfield moderna (Ramsauer et al. 2020): ranuras clave→valor."""

    kind = "asociativa"

    def __init__(self, d: int, slots: int):
        super().__init__()
        self.query = nn.Linear(d, d, bias=False)
        self.keys = nn.Parameter(torch.randn(slots, d) / math.sqrt(d))
        self.slot_bias = nn.Parameter(torch.zeros(slots))
        self.fc2 = nn.Linear(slots, d)                 # valores: una columna por ranura
        self.beta = math.sqrt(d)

    def forward(self, x):
        att = torch.softmax(self.beta * self.query(x) @ self.keys.t() / x.shape[-1] + self.slot_bias, dim=-1)
        return self.fc2(att)

    def widen(self, extra, seed=0):
        """Ranuras nuevas con sesgo muy negativo: no reciben atención hasta que entrenan."""
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.keys = nn.Parameter(torch.cat([self.keys, _rand_rows(extra, self.keys.shape[1], g, self.keys)]))
            self.slot_bias = nn.Parameter(torch.cat([self.slot_bias, torch.full((extra,), -30.0).to(self.slot_bias)]))
        self.fc2 = _grow_linear_in(self.fc2, extra)


class RelationalSpecialist(Specialist):
    """Atención causal local: relaciona cada posición con las ``window`` anteriores."""

    kind = "relacional"
    sequence_aware = True

    def __init__(self, d: int, hidden: int, heads: int = 2, window: int = 32):
        super().__init__()
        hidden = max(heads, hidden - hidden % heads)
        self.heads, self.window = heads, window
        self.fc1 = nn.Linear(d, 3 * hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d)

    def forward(self, x, cache: dict | None = None):
        if cache is not None and isinstance(cache.get("pos"), torch.Tensor):
            return self._forward_rows(x, cache)
        b, t, _ = x.shape
        h = self.fc2.in_features
        q, k, v = self.fc1(x).view(b, t, 3, self.heads, h // self.heads).permute(2, 0, 3, 1, 4)
        off = cache.get("pos", 0) if cache is not None else 0     # posición absoluta del primer elemento
        q, k = rope(q, offset=off), rope(k, offset=off)
        if cache is not None and "k" in cache:
            k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
        if cache is not None:                                    # solo hace falta la ventana
            keep = max(self.window - 1, 0)
            cache["k"], cache["v"] = k[:, :, max(k.shape[2] - keep, 0):], v[:, :, max(v.shape[2] - keep, 0):]
            cache["pos"] = off + t
        i = torch.arange(off, off + t, device=x.device)
        j = torch.arange(off + t - k.shape[2], off + t, device=x.device)
        allowed = (j[None, :] <= i[:, None]) & (i[:, None] - j[None, :] < self.window)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.fc2(y.transpose(1, 2).reshape(b, t, h))

    def _forward_rows(self, x, cache: dict):
        """Un paso para B secuencias en posiciones distintas (``hope/core/batched.py``):
        ``cache["pos"]`` es un tensor [B] y las claves guardadas están alineadas a la derecha
        (la fila i usa las últimas ``min(pos[i], window − 1)``; lo de antes es relleno)."""
        b, t, _ = x.shape
        assert t == 1, "el camino por filas avanza un paso por vez"
        h = self.fc2.in_features
        q, k, v = self.fc1(x).view(b, t, 3, self.heads, h // self.heads).permute(2, 0, 3, 1, 4)
        pos = cache["pos"]
        q, k = rope_rows(q, 10000.0, pos), rope_rows(k, 10000.0, pos)
        if "k" in cache:
            k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
        keep = max(self.window - 1, 0)
        n = k.shape[2]
        valid = pos.clamp(max=keep) + 1                          # guardadas válidas + la nueva
        allowed = torch.arange(n, device=x.device)[None] >= (n - valid)[:, None]
        cache["k"], cache["v"] = k[:, :, max(n - keep, 0):], v[:, :, max(n - keep, 0):]
        cache["pos"] = pos + t
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None, None])
        return self.fc2(y.transpose(1, 2).reshape(b, t, h))

    def widen(self, extra, seed=0):
        raise NotImplementedError("relacional: ensanchar cambia la partición en cabezas; agregar otro especialista")


FACTORY = {"mlp": MLPSpecialist, "glu": GLUSpecialist, "ei": EISpecialist, "recurrente": RecurrentSpecialist,
           "conv": ConvSpecialist, "asociativa": AssociativeSpecialist, "relacional": RelationalSpecialist}


def make_specialist(kind: str, d: int, hidden: int) -> Specialist:
    if kind not in FACTORY:
        raise ValueError(f"tipo de especialista desconocido: {kind} (disponibles: {KINDS})")
    return FACTORY[kind](d, hidden)


class Zone(nn.Module):
    """Zona de especialistas heterogéneos con router top-k (misma interfaz que ``SparseRegion``)."""

    def __init__(self, d: int, kinds: tuple[str, ...], hidden: int, top_k: int, balance: str = "aux",
                 shared_expert: bool = False, bias_rate: float = 1e-3):
        super().__init__()
        self.d, self.top_k, self.balance, self.bias_rate = d, top_k, balance, bias_rate
        self.norm = RMSNorm(d)
        self.router = nn.Linear(d, len(kinds))
        self.experts = nn.ModuleList(make_specialist(k, d, hidden) for k in kinds)
        self.register_buffer("expert_mask", torch.zeros(len(kinds)))
        # Balanceo sin pérdida auxiliar (DeepSeek-V3): un sesgo que solo afecta la elección.
        self.register_buffer("route_bias", torch.zeros(len(kinds)))
        self.shared = make_specialist("mlp", d, hidden) if shared_expert else None

    @property
    def n_experts(self) -> int:
        return len(self.experts)

    @property
    def kinds(self) -> list[str]:
        return [e.kind for e in self.experts]

    def forward(self, x, cache: dict | None = None):
        b, t, d = x.shape
        h = self.norm(x)
        flat = h.reshape(-1, d)
        logits = self.router(flat) + self.expert_mask
        probs = logits.softmax(-1)
        k = min(self.top_k, self.n_experts)
        if self.balance == "sesgo":
            if self.route_bias.numel() != self.n_experts:           # creció la zona
                self.route_bias = torch.cat([self.route_bias, self.route_bias.new_zeros(
                    self.n_experts - self.route_bias.numel())])
            _, top_idx = (logits + self.route_bias).topk(k, dim=-1)
            top_val = logits.gather(-1, top_idx)                    # los pesos usan el puntaje sin sesgo
        else:
            top_val, top_idx = logits.topk(k, dim=-1)
        weights = top_val.softmax(-1)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            hit = top_idx == e
            rows = hit.any(-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                if cache is not None and expert.sequence_aware:
                    expert(h, cache.setdefault(e, {}))  # el estado avanza aunque no lo elijan
                continue
            w = (weights * hit).sum(-1)[rows].unsqueeze(-1)
            if expert.sequence_aware:                   # necesita la secuencia completa (causal)
                y = (expert(h) if cache is None else expert(h, cache.setdefault(e, {}))).reshape(-1, d)[rows]
            else:
                y = expert(flat[rows])
            out.index_add_(0, rows, (w * y).to(out.dtype))
        if self.shared is not None:
            out = out + self.shared(flat).to(out.dtype)
        frac = F.one_hot(top_idx[:, 0], self.n_experts).float().mean(0)
        if self.balance == "sesgo":
            if self.training:
                with torch.no_grad():
                    load = F.one_hot(top_idx, self.n_experts).float().sum((0, 1))
                    self.route_bias += self.bias_rate * torch.sign(load.mean() - load)
            balance = probs.new_zeros(())
        else:
            balance = self.n_experts * (frac * probs.float().mean(0)).sum()
        return out.reshape(b, t, d), balance, top_idx.reshape(b, t, k)
