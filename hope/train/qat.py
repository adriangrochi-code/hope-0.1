"""Entrenamiento consciente de la cuantización (QAT; pedido del usuario: «que Hope se entrene para
funcionar bien cuantizado, así después es eficiente»).

Durante el entrenamiento, cada capa lineal usa sus pesos **redondeados a pocos bits** (cuantización
falsa: se cuantiza y se descuantiza) y el gradiente pasa como si no hubiera redondeo (estimador
«straight-through»). Así el modelo aprende pesos a los que el redondeo no afecta, y al exportarlo
de verdad a int4/int8 (``hope.core.fast.quantize``) pierde muy poca calidad. Es lo que se usó para las
versiones QAT de Gemma 3 y Llama 3.2.

* **Por grupos**: una escala cada ``group`` pesos de una fila (como los formatos Q4 de llama.cpp):
  mucho más preciso que una escala por fila.
* **Matryoshka** (MatQuant, Nair et al. 2025): con ``bits=(4, 8)`` cada paso elige al azar una
  precisión; el mismo modelo queda bien en 4 y en 8 bits (y en lo que haya entre medio).
* **Cuándo**: en la parte final del entrenamiento (el enfriamiento de WSD), como recomiendan los
  trabajos de QAT: antes solo frenaría el aprendizaje.
* No agrega parámetros y no cambia los nombres de los pesos: los checkpoints siguen siendo compatibles.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PREFIXES = ("encoder", "decoder", "backbone", "core", "regions", "memory", "vision", "connections")


def fake_quant(w: torch.Tensor, bits: int, group: int = 64) -> torch.Tensor:
    """Cuantización simétrica por grupos (redondeo + descuantización) con gradiente straight-through.

    Las cuentas se hacen en fp32 aunque los pesos sean bf16, igual que ``hope.core.fast.QuantLinear``:
    en bf16 la división y el redondeo dan otra grilla (hasta un 12 % de diferencia en 4 bits) y el
    modelo aprendería a funcionar con una cuantización distinta de la que se exporta."""
    out_f, in_f = w.shape
    g = group if in_f % group == 0 else in_f
    qmax = 2 ** (bits - 1) - 1
    wf = w.float()
    wg = wf.reshape(out_f, in_f // g, g)
    scale = wg.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    dq = (torch.round(wg.detach() / scale).clamp(-qmax, qmax) * scale).reshape(out_f, in_f)
    return wf + (dq - wf).detach()


class QATLinear(nn.Linear):
    """Mismo módulo (mismos parámetros y nombres) que ``nn.Linear``, con cuantización falsa opcional."""

    qat_bits: int | None = None
    qat_group: int = 64

    def forward(self, x):
        if self.qat_bits is None:
            return super().forward(x)
        w = fake_quant(self.weight, self.qat_bits, self.qat_group)
        return F.linear(x, w.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


def _linears(model: nn.Module, prefixes=DEFAULT_PREFIXES):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.startswith(prefixes) and min(mod.weight.shape) > 1:
            yield name, mod


def enable_qat(model: nn.Module, bits: int = 4, group: int = 64, prefixes=DEFAULT_PREFIXES) -> int:
    """Activa la cuantización falsa en las capas lineales internas (no en embeddings ni cabezas)."""
    n = 0
    for _, mod in _linears(model, prefixes):
        if type(mod) is nn.Linear:
            mod.__class__ = QATLinear
        if isinstance(mod, QATLinear):
            mod.qat_bits, mod.qat_group = bits, group
            n += 1
    return n


def set_qat_bits(model: nn.Module, bits: int | None) -> None:
    for mod in model.modules():
        if isinstance(mod, QATLinear):
            mod.qat_bits = bits


def disable_qat(model: nn.Module) -> None:
    for mod in model.modules():
        if isinstance(mod, QATLinear):
            mod.qat_bits = None
            mod.__class__ = nn.Linear
