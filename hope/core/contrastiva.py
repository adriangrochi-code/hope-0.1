"""Decodificación contrastiva (Li et al. 2022, «Contrastive Decoding: Open-ended Text Generation as
Optimization»; técnica 4 del pedido de lenguaje).

Idea: los errores típicos de un modelo chico (repetir, irse por lo genérico) los comete **todavía
más** un modelo peor. Restar la opinión del «aprendiz» (amateur) a la del «experto» castiga
justamente eso, sin parámetros nuevos::

    puntaje(x) = log p_exp(x) − β · log p_apr(x)    si  p_exp(x) ≥ α · max_w p_exp(w)
               = −∞                                  si no (restricción de plausibilidad)

La restricción de plausibilidad (α = 0,1 en el artículo) es esencial: sin ella ganan bytes que el
experto considera absurdos pero el aprendiz todavía más. ``tau`` es la temperatura del aprendiz
(el artículo usa 0,5–1,0; más baja = aprendiz más «seguro» de sus errores).

Aprendices posibles (todos sin parámetros nuevos para el experto):

* otro checkpoint más viejo o más chico (``hn1-cpu-mini`` contra ``hn1-25m``);
* **el mismo modelo con menos iteraciones del núcleo** (``n_iters=1``): la versión «que piensa
  menos» de sí mismo. No necesita memoria extra (comparte los pesos).

El puntaje no es una distribución: ``ProcesadorContrastivo`` devuelve log-softmax del puntaje
(fuera de la cabeza plausible queda −∞), así que con temperatura 0 elige el máximo del artículo.
"""

from __future__ import annotations

import math

import torch

from hope.core.hn1 import HN1Model
from hope.core.paso import Paso


class ProcesadorContrastivo:
    def __init__(self, aprendiz: HN1Model, *, alpha: float = 0.1, beta: float = 1.0, tau: float = 1.0,
                 n_iters_aprendiz: int | None = None, memory=None):
        self.aprendiz, self.alpha, self.beta, self.tau = aprendiz, alpha, beta, tau
        self.n_iters, self.memory = n_iters_aprendiz, memory
        self.paso: Paso | None = None
        self._pend: list[int] = []

    def iniciar(self, seq: list[int]) -> None:
        self.aprendiz.eval()
        self.paso = Paso(self.aprendiz, self.memory, self.n_iters)
        self.paso.feed(list(seq) or [10])
        self._pend = []

    def avanzar(self, byte: int) -> None:
        self._pend.append(byte)

    @torch.no_grad()
    def __call__(self, paso_exp: Paso, logp: torch.Tensor) -> torch.Tensor:
        if self._pend:                                     # el aprendiz se pone al día perezosamente
            self.paso.feed(self._pend)
            self._pend = []
        lp_apr = torch.log_softmax(self.paso.logits / self.tau, -1)
        cabeza = logp >= logp.max() + math.log(self.alpha)
        score = logp - self.beta * lp_apr
        score = torch.where(cabeza, score, torch.full_like(score, -math.inf))
        return torch.log_softmax(score, -1)


def aprendiz_mismo_modelo(model: HN1Model, n_iters: int = 1, **kw) -> ProcesadorContrastivo:
    """Contrastiva contra el propio modelo con menos iteraciones del núcleo (cero memoria extra)."""
    return ProcesadorContrastivo(model, n_iters_aprendiz=n_iters, **kw)
