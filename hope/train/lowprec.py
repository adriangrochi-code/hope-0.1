"""Entrenamiento en baja precisión (decisión del usuario: «de base, baja precisión,
para que al ampliar el tamaño no pese tanto»).

Bytes por parámetro durante el entrenamiento:

======================  ==================================================  ==========
modo                    qué guarda                                          bytes/param
======================  ==================================================  ==========
``fp32``                pesos fp32 + gradientes fp32 + Adam fp32 (m, v)     16
``bf16``                pesos bf16 + gradientes bf16 + Adam fp32            12
``bf16-8bit`` (base)    pesos bf16 + gradientes bf16 + Adam en 8 bits       ~6,1
======================  ==================================================  ==========

Dos técnicas lo hacen posible sin perder el aprendizaje:

* **Redondeo estocástico** al escribir pesos bf16: bf16 tiene solo 8 bits de
  mantisa, así que una actualización pequeña redondeada «al más cercano» se
  pierde siempre. Redondeando al azar con probabilidad proporcional a la
  distancia, el valor esperado es exacto y las actualizaciones chicas se acumulan.
* **Momentos de Adam en 8 bits por bloques** (idea de Dettmers et al. 2022, 8-bit
  optimizers; implementación propia en PyTorch, sin kernels CUDA, para la A770):
  bloques de 256 valores con una escala fp32 por bloque. El segundo momento se
  guarda como su raíz cuadrada, que tiene un rango mucho menor.

La calidad se debe **medir** contra ``fp32`` en cada escala (ablación): la baja
precisión es una hipótesis de ingeniería, no una garantía.
"""

from __future__ import annotations

import math

import torch

MODES = ("fp32", "bf16", "bf16-8bit", "fp32-8bit")
# fp32-8bit: pesos y gradientes fp32 + Adam en 8 bits. Para GPUs sin bf16 (Pascal: GTX 10xx), donde
# bf16 no existe en hardware y fp16 corre a 1/64 de velocidad.
BYTES_PER_PARAM = {"fp32": 16.0, "bf16": 12.0, "bf16-8bit": 6.0 + 8.0 / 256, "fp32-8bit": 10.0 + 8.0 / 256}
BLOCK = 256


def fetch_state(st: dict, device) -> None:
    """Descarga del optimizador (offload): trae el estado de un parámetro a la GPU para actualizarlo."""
    for k, v in list(st.items()):
        if torch.is_tensor(v) and v.device != device:
            st[k] = v.to(device, non_blocking=True)


def park_state(st: dict) -> None:
    """Devuelve el estado a la RAM de la CPU (con 16 GB de RAM entra el de un modelo de 500M)."""
    for k, v in list(st.items()):
        if torch.is_tensor(v) and v.device.type != "cpu":
            st[k] = v.to("cpu")


def stochastic_round_bf16(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """fp32 → bf16 con redondeo estocástico (sin sesgo)."""
    x = x.float().contiguous()
    bits = x.view(torch.int32)
    noise = torch.randint(0, 1 << 16, x.shape, device=x.device, dtype=torch.int32, generator=generator)
    rounded = (bits + noise) & -65536               # descarta los 16 bits bajos
    out = rounded.view(torch.float32)
    out = torch.where(torch.isfinite(x), out, x)    # no tocar inf/nan
    return out.to(torch.bfloat16)


# --- cuantización por bloques -------------------------------------------------
def quantize_blockwise(x: torch.Tensor, signed: bool = True) -> tuple[torch.Tensor, torch.Tensor, int]:
    flat = x.float().reshape(-1)
    n = flat.numel()
    pad = (-n) % BLOCK
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, BLOCK)
    scale = blocks.abs().amax(dim=1).clamp_min(1e-12)
    if signed:
        q = torch.round(blocks / scale[:, None] * 127).clamp(-127, 127).to(torch.int8)
    else:
        q = torch.round(blocks / scale[:, None] * 255).clamp(0, 255).to(torch.uint8)
    return q, scale, n


def dequantize_blockwise(q: torch.Tensor, scale: torch.Tensor, n: int, shape, signed: bool = True) -> torch.Tensor:
    vals = q.float() * (scale[:, None] / (127 if signed else 255))
    return vals.reshape(-1)[:n].view(shape)


class LowPrecisionAdamW(torch.optim.Optimizer):
    """AdamW para pesos bf16 con momentos en 8 bits (o fp32) y redondeo estocástico.

    También acepta parámetros fp32 (los actualiza normalmente), por si alguna
    parte del modelo se deja en alta precisión.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.1, state_bits: int = 8, seed: int = 0, offload: bool = False):
        if state_bits not in (8, 32):
            raise ValueError("state_bits debe ser 8 o 32")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.state_bits = state_bits
        self.offload = offload
        self._gen: dict[str, torch.Generator] = {}
        self._seed = seed

    # Optimizer.__getstate__ solo guarda defaults/state/param_groups: se agregan los atributos propios
    # para que copiar o serializar el optimizador no los pierda.
    def __getstate__(self):
        d = super().__getstate__()
        d["state_bits"], d["_seed"], d["offload"] = self.state_bits, self._seed, self.offload
        return d

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("state_bits", 8)
        self.__dict__.setdefault("offload", False)
        self.__dict__.setdefault("_seed", 0)
        self._gen = {}

    def _generator(self, device: torch.device) -> torch.Generator | None:
        if device.type not in ("cpu", "cuda"):
            return None                      # XPU: usa el generador global del dispositivo
        key = str(device)
        if key not in self._gen:
            self._gen[key] = torch.Generator(device=device).manual_seed(self._seed)
        return self._gen[key]

    def _load_moments(self, st, p):
        # Acepta el estado en cualquiera de los dos formatos (p. ej. al continuar en otra máquina con
        # otra precisión); se guarda de nuevo en el formato de este optimizador.
        if "m" in st:
            return st.pop("m").float(), st.pop("v").float()
        m = dequantize_blockwise(st["m_q"], st["m_s"], p.numel(), p.shape, signed=True)
        v = dequantize_blockwise(st["v_q"], st["v_s"], p.numel(), p.shape, signed=False).square()
        return m, v

    def _store_moments(self, st, m, v):
        if self.state_bits == 32:
            st["m"], st["v"] = m, v
            for k in ("m_q", "m_s", "v_q", "v_s"):
                st.pop(k, None)
        else:
            st["m_q"], st["m_s"], _ = quantize_blockwise(m, signed=True)
            st["v_q"], st["v_s"], _ = quantize_blockwise(v.sqrt(), signed=False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if self.offload:
                    fetch_state(st, p.device)
                if not st:
                    st["step"] = 0
                    zeros = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                    self._store_moments(st, zeros, zeros.clone())
                st["step"] += 1
                t = st["step"]
                g = p.grad.float()
                m, v = self._load_moments(st, p)
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                self._store_moments(st, m, v)
                denom = (v / (1 - b2 ** t)).sqrt_().add_(group["eps"])
                upd = (m / (1 - b1 ** t)) / denom
                w = p.float()
                w.mul_(1 - group["lr"] * group["weight_decay"]).add_(upd, alpha=-group["lr"])
                if p.dtype == torch.bfloat16:
                    p.copy_(stochastic_round_bf16(w, self._generator(p.device)))
                else:
                    p.copy_(w.to(p.dtype))
                if self.offload:
                    park_state(st)
        return loss

    def load_state_dict(self, state_dict):
        """PyTorch convierte los estados flotantes al tipo del parámetro (bf16) al cargar;
        acá se restauran tal cual se guardaron (escalas fp32, momentos int8)."""
        super().load_state_dict(state_dict)
        saved = state_dict["state"]
        for g_saved, g in zip(state_dict["param_groups"], self.param_groups):
            for pid, p in zip(g_saved["params"], g["params"]):
                for k, v in saved.get(pid, {}).items():
                    if torch.is_tensor(v) and k != "step":
                        self.state[p][k] = v.to(p.device).clone()

    def state_bytes(self) -> int:
        total = 0
        for st in self.state.values():
            for v in st.values():
                if torch.is_tensor(v):
                    total += v.numel() * v.element_size()
        return total


def prepare(model: torch.nn.Module, mode: str, *, lr: float, weight_decay: float = 0.1,
            betas=(0.9, 0.95), seed: int = 0, param_groups: list[dict] | None = None,
            optimizer: str = "adamw", offload: bool = False) -> torch.optim.Optimizer:
    """``offload``: el estado del optimizador vive en la RAM de la CPU y se trae por partes en cada
    paso (para entrenar un modelo de 500M en una GPU de 6 GB)."""
    """Pasa el modelo a la precisión del modo y devuelve su optimizador.

    ``param_groups`` (opcional): grupos con ``lr_scale`` (plasticidad por región, ``hope.plasticity.rates``).
    """
    if mode not in MODES:
        raise ValueError(f"modo desconocido: {mode} (opciones: {MODES})")
    model.float() if mode in ("fp32", "fp32-8bit") else model.to(torch.bfloat16)
    params = param_groups if param_groups is not None else [{"params": list(model.parameters()), "lr_scale": 1.0}]
    if optimizer == "muon":                        # hope/train/muon.py (ablación contra AdamW)
        from hope.train.muon import make_muon
        return make_muon(model, lr=lr, weight_decay=weight_decay, seed=seed, param_groups=param_groups,
                         offload=offload)
    if optimizer != "adamw":
        raise ValueError(f"optimizador desconocido: {optimizer}")
    if mode == "fp32":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    return LowPrecisionAdamW(params, lr=lr, betas=betas, weight_decay=weight_decay,
                             state_bits=8 if mode.endswith("8bit") else 32, seed=seed, offload=offload)


def training_memory_gb(params: int, mode: str = "bf16-8bit") -> float:
    return params * BYTES_PER_PARAM[mode] / 1e9


def max_params_for(vram_gb: float, mode: str = "bf16-8bit", activation_reserve: float = 0.2) -> int:
    """Parámetros entrenables que entran dejando una reserva para activaciones."""
    return int(vram_gb * (1 - activation_reserve) * 1e9 / BYTES_PER_PARAM[mode])


def human_bytes(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


__all__ = ["MODES", "BYTES_PER_PARAM", "stochastic_round_bf16", "quantize_blockwise", "dequantize_blockwise",
           "LowPrecisionAdamW", "prepare", "training_memory_gb", "max_params_for", "human_bytes", "math"]
