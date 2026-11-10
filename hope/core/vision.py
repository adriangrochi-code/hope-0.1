"""Zona visual de Hope (ARQUITECTURA_HN1 §3.9; decisión del usuario).

Hope sigue siendo un modelo de bytes. La visión se agrega como **una zona más**
de la red heterogénea:

    imagen ─► parches de píxeles (conv) ─► codificador bidireccional (ViT pequeño)
          ─► compresión 2×2 (pixel shuffle) ─► fichas visuales [V, d]
          ─► atención cruzada con compuerta en algunos bloques del backbone global

Decisiones y por qué:

* **Atención cruzada con salida en cero** (como Flamingo, Alayrac et al. 2022, pero con
  la proyección de salida en cero en lugar de una compuerta tanh en cero), no fichas
  visuales mezcladas en la secuencia: agregar la zona a un Hope ya entrenado no
  cambia su salida de texto (crecimiento que preserva la función), no toca el
  parcheo ni la causalidad y no alarga la secuencia global.
  Por qué no la compuerta en cero: con la compuerta en 0 el codificador visual no
  recibe gradiente y la compuerta solo ve ruido; en una prueba de 600 pasos desde
  cero quedó en 0,002 y el modelo ignoró la imagen. Con la salida en cero (idea de
  ControlNet) la proyección recibe gradiente desde el primer paso y el resto después.
* **Compresión 2×2** (pixel shuffle, como SmolVLM/Idefics3): 4× menos fichas que
  atender; con imágenes de 224 px y parches de 16 quedan 49 fichas por imagen.
* **Entrenada desde cero con subtítulos**: CapPa (Tschannen et al. 2023) muestra
  que entrenar un codificador visual solo con la tarea de describir la imagen da
  representaciones comparables a las contrastivas. El maestro (Qwen multimodal)
  produce esas descripciones (``hope/teacher/vision.py``).
* **Posiciones factorizadas fila/columna** interpolables: la misma red acepta
  otras resoluciones (documentos, mosaicos) sin reentrenar desde cero.
* **Mosaicos** (``tiles=True``, ``HN1Config.vis_tiling``): un documento de 2S×2S entra como
  miniatura + 4 mosaicos a resolución completa, sin parámetros nuevos (ver ``forward``).

Las imágenes entran como ``uint8`` [B, 3, H, W] (o float en [0, 1]).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hope.core.layers import RMSNorm, SwiGLU

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


class BiAttention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class VisionBlock(nn.Module):
    def __init__(self, d: int, heads: int, hidden: int):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = BiAttention(d, heads)
        self.mlp = SwiGLU(d, hidden)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class VisionEncoder(nn.Module):
    def __init__(self, d_out: int, d: int = 384, layers: int = 6, heads: int = 6, hidden: int | None = None,
                 patch: int = 16, image: int = 224, pool: int = 2):
        super().__init__()
        assert (image // patch) % pool == 0, "la grilla de parches debe ser divisible por pool"
        self.patch, self.image, self.pool = patch, image, pool
        self.stem = nn.Conv2d(3, d, kernel_size=patch, stride=patch)
        g = image // patch
        self.row = nn.Parameter(torch.randn(g, d) * 0.02)
        self.col = nn.Parameter(torch.randn(g, d) * 0.02)
        self.blocks = nn.ModuleList(VisionBlock(d, heads, hidden or 4 * d) for _ in range(layers))
        self.norm = RMSNorm(d * pool * pool)
        self.proj = nn.Linear(d * pool * pool, d_out)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def tokens_per_image(self, h: int | None = None, w: int | None = None) -> int:
        h, w = h or self.image, w or self.image
        return (h // self.patch // self.pool) * (w // self.patch // self.pool)

    def _pos(self, gh: int, gw: int) -> torch.Tensor:
        row, col = self.row, self.col
        if row.shape[0] != gh:
            row = F.interpolate(row.T[None], size=gh, mode="linear", align_corners=False)[0].T
        if col.shape[0] != gw:
            col = F.interpolate(col.T[None], size=gw, mode="linear", align_corners=False)[0].T
        return (row[:, None, :] + col[None, :, :]).reshape(gh * gw, -1)

    def _prep(self, images: torch.Tensor) -> torch.Tensor:
        x = images.to(self.stem.weight.dtype)
        if not images.is_floating_point():
            x = x / 255.0
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

    def _trunk(self, x: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        """Fichas de parche [N, gh·gw, d] (ya con posición) → fichas visuales [N, V, d_out]."""
        for blk in self.blocks:
            x = blk(x)
        b, _, d = x.shape
        p = self.pool                                          # pixel shuffle: 2×2 vecinos → 1 ficha
        x = x.view(b, gh // p, p, gw // p, p, d).permute(0, 1, 3, 2, 4, 5).reshape(b, (gh // p) * (gw // p), p * p * d)
        return self.proj(self.norm(x))

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)                                       # [B, d, gh, gw]
        b, d, gh, gw = x.shape
        return self._trunk(x.flatten(2).transpose(1, 2) + self._pos(gh, gw).to(x.dtype), gh, gw)

    def can_tile(self, h: int, w: int) -> bool:
        s = self.image
        return (h > s or w > s) and h % s == 0 and w % s == 0

    def forward(self, images: torch.Tensor, tiles: bool = False) -> torch.Tensor:
        """[B, 3, H, W] → fichas visuales [B, V, d_out].

        ``tiles=True`` y H, W múltiplos de la resolución de entrenamiento S (mosaicos para
        documentos): una miniatura de toda la imagen a S×S **más** cada mosaico S×S a resolución
        completa, sin parámetros nuevos. Cada mosaico pasa solo por el codificador (costo lineal
        en la cantidad de mosaicos, no cuadrático en la imagen), pero sus parches llevan la
        posición fila/columna **de la imagen entera** (interpolada), así la red sabe dónde está
        cada mosaico. Fichas: (1 + filas·columnas) · V."""
        x = self._prep(images)
        h, w = x.shape[-2:]
        if not (tiles and self.can_tile(h, w)):
            return self._encode(x)
        s = self.image
        r, c = h // s, w // s
        thumb = self._encode(F.interpolate(x, size=(s, s), mode="area"))
        t = self.stem(x)                                       # [B, d, r·g, c·g]
        b, d, gh, gw = t.shape
        g = gh // r
        t = t.permute(0, 2, 3, 1) + self._pos(gh, gw).to(t.dtype).view(gh, gw, d)
        t = t.reshape(b, r, g, c, g, d).permute(0, 1, 3, 2, 4, 5).reshape(b * r * c, g * g, d)
        t = self._trunk(t, g, g)                               # [B·r·c, V, d_out]
        return torch.cat([thumb, t.reshape(b, r * c * t.shape[1], -1)], dim=1)


class VisualCrossAttention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x, vis):
        b, t, d = x.shape
        m = vis.shape[1]
        q = self.q(self.norm(x)).view(b, t, self.heads, -1).transpose(1, 2)
        k, v = self.kv(vis).view(b, m, 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return x + self.out(y.transpose(1, 2).reshape(b, t, d))


class VisualZone(nn.Module):
    """Codificador + atención cruzada (una por bloque elegido del backbone)."""

    def __init__(self, d_model: int, model_heads: int, at_blocks: tuple[int, ...], **enc):
        super().__init__()
        self.encoder = VisionEncoder(d_model, **enc)
        self.at_blocks = tuple(at_blocks)
        self.xattn = nn.ModuleDict({str(i): VisualCrossAttention(d_model, model_heads) for i in self.at_blocks})

    def encode(self, images: torch.Tensor | None, tiles: bool = False) -> torch.Tensor | None:
        if images is None:
            return None
        if images.dim() == 3:
            images = images.unsqueeze(0)
        return self.encoder(images, tiles=tiles)

    def inject(self, j: int, g: torch.Tensor, vis: torch.Tensor | None) -> torch.Tensor:
        key = str(j)
        if vis is None or key not in self.xattn:
            return g
        if vis.shape[0] != g.shape[0]:                         # una imagen para todo el lote
            vis = vis.expand(g.shape[0], -1, -1)
        return self.xattn[key](g, vis)
