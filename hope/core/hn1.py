"""Núcleo neuronal de Hope HN-1 (ARQUITECTURA_HN1 §3).

Flujo::

    bytes ─► codificador local (causal) ─► parches ─► backbone global híbrido (SSM + atención)
          ─► [memoria] ─► núcleo recurrente con parada adaptativa ─► regiones especialistas
          ─► decodificador local de bytes ─► predicción del byte siguiente

**Causalidad.** El byte *t* predice el byte *t + 1* usando:
  * el estado del codificador local hasta *t* (atención causal sobre bytes);
  * la salida global del **parche anterior completo** al parche de *t*.
La frontera de un parche depende solo del byte que la cierra y de la longitud
del parche, así que tampoco filtra información del futuro. Hay una prueba que
lo verifica (``tests/test_hn1.py``).

El mismo código sirve para la configuración de 25M (F1) y la de ~175M (F1.b);
solo cambia ``HN1Config``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hope.core.layers import Block, MemoryCrossAttention, RMSNorm
from hope.core.vision import VisualZone
from hope.core.zones import Zone
from hope.core.toy import Expert
from hope.registry.topology import NodeKind, TopologyRegistry

BYTE_VOCAB = 256
# Bytes que cierran un parche: espacio, salto de línea, tabulación y puntuación ASCII común.
BOUNDARY_BYTES = sorted(set(b" \n\t.,;:!?()[]{}\"'-/\\=+*<>|"))


@dataclass
class HN1Config:
    # Parte local (bytes)
    d_local: int = 256
    local_heads: int = 4
    local_mlp: int = 512
    enc_layers: int = 1
    dec_layers: int = 2
    max_patch: int = 8
    # Backbone global (parches)
    d: int = 384
    heads: int = 6
    mlp_hidden: int = 1408
    pattern: tuple[str, ...] = ("ssm", "ssm", "attn", "ssm", "ssm", "attn")
    # Núcleo recurrente compartido (§3.4)
    core_kind: str = "attn"
    max_iters: int = 4
    ponder_weight: float = 0.01
    halt_eps: float = 0.01
    full_depth_frac: float = 0.5     # fracción de pasos de entrenamiento con todas las iteraciones
    ponder_warmup: int = 2000        # pasos en los que el costo de iterar sube de 0 a su valor
    # Bucle estable (informe «Razonamiento por parámetro» §1.1.1; ``stable_loop``). Los valores por
    # defecto reproducen el núcleo anterior, así los checkpoints existentes cargan igual.
    core_inject: str = "ninguna"     # "ninguna" | "convexa": h ← block(Ā·h + (1−Ā)·x) en cada iteración
    core_decay: float = 0.5          # Ā fijo (no se aprende: cero parámetros), 0 < Ā < 1 ⇒ ρ(Ā) < 1
    core_step: str = "embed"         # "embed" (tabla de max_iters filas) | "senoidal" | "ninguna"
    core_depth: str = "uniforme"     # profundidad al entrenar: "uniforme" 1..max_iters | "lognormal"
    core_mean_iters: float = 4.0     # media de la profundidad log-normal–Poisson (Geiping et al. 2025)
    core_depth_sigma: float = 0.5
    core_train_max: int | None = None    # tope de la profundidad muestreada (None = max_iters)
    core_bptt: int | None = None     # backprop solo por las últimas k iteraciones (None = completo)
    # Zonas de especialistas heterogéneos (§3.5; hope/core/zones.py)
    n_regions: int = 1
    zone_kinds: tuple[str, ...] = ("mlp", "glu", "ei", "recurrente", "conv", "asociativa", "relacional", "mlp")
    n_experts: int = 8               # se ajusta a len(zone_kinds)
    expert_hidden: int = 512
    top_k: int = 2
    balance_weight: float = 0.01
    # Otros
    # Opciones experimentales (informe «Técnicas de generación rápida»; todas apagadas por defecto)
    kv_heads: int | None = None      # GQA en las capas de atención
    window: int = 256                # ventana de los bloques "swa"
    sinks: int = 4                   # posiciones siempre visibles en "swa" (StreamingLLM)
    local_kind: str = "attn"         # "attn" | "swa" en el codificador/decodificador de bytes
    halting: str = "act"             # "act" | "convergencia" (profundidad muestreada; Geiping et al. 2025)
    convergence_tol: float = 1e-3
    balance: str = "aux"             # "aux" (pérdida auxiliar) | "sesgo" (sin pérdida; DeepSeek-V3)
    shared_expert: bool = False      # especialista siempre activo en cada zona (DeepSeekMoE)
    patching: str = "espacios"       # "espacios" | "entropia" (hope/core/patching.py)
    entropy_order: int = 3
    entropy_buckets: int = 65536
    memory: bool = True              # capa de atención cruzada a memoria (§3.7)
    hnhs: bool = False               # poblaciones E/I en los MLP del backbone (ablación §3.6)
    uncertainty_weight: float = 0.01
    # Cabezas multi-byte (Gloeckle et al. 2024; Medusa, Cai et al. 2024): predicen los bytes
    # t+2 … t+1+multibyte como tarea auxiliar y sirven de borrador para la generación especulativa.
    multibyte: int = 0
    # Contexto largo sin parámetros nuevos (§3.9.2): escala de RoPE (NTK) para posiciones más
    # allá de lo entrenado y memoria de segmentos anteriores comprimidos (``long_loss``).
    rope_scale: float = 1.0
    segment_chunk: int = 8                       # parches por resumen de un segmento anterior
    max_segment_memory: int = 256                # resúmenes que se guardan como máximo
    multibyte_weight: float = 0.1
    # Zona visual (§3.9; hope/core/vision.py). Apagada por defecto: se agrega con AddVisualZone.
    vision: bool = False
    vis_d: int = 384
    vis_layers: int = 6
    vis_heads: int = 6
    vis_hidden: int | None = None                 # oculta del MLP visual (None = 8/3·vis_d, como SwiGLU estándar)
    vis_patch: int = 16
    vis_image: int = 224
    vis_pool: int = 2
    vis_blocks: tuple[int, ...] | None = None   # bloques del backbone con atención cruzada (None = los "attn")
    vis_tiling: bool = False                    # mosaicos para imágenes de alta resolución (múltiplos de vis_image)
    # Bloques del backbone con los que «nació» el modelo; los que siguen se agregaron creciendo
    # (AppendBackboneBlock) y son los únicos que el presupuesto puede retirar (§6.5). None = todos.
    base_blocks: int | None = None

    def __post_init__(self):
        self.pattern = tuple(self.pattern)
        self.zone_kinds = tuple(self.zone_kinds)
        if self.vis_blocks is not None:
            self.vis_blocks = tuple(self.vis_blocks)
        self.n_experts = len(self.zone_kinds)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pattern"] = list(self.pattern)
        d["zone_kinds"] = list(self.zone_kinds)
        if self.vis_blocks is not None:
            d["vis_blocks"] = list(self.vis_blocks)
        return d

    def vision_blocks(self) -> tuple[int, ...]:
        if self.vis_blocks is not None:
            return self.vis_blocks
        return tuple(i for i, k in enumerate(self.pattern) if k in ("attn", "swa")) or (len(self.pattern) - 1,)

    def vision_kwargs(self) -> dict:
        return dict(d=self.vis_d, layers=self.vis_layers, heads=self.vis_heads, patch=self.vis_patch,
                    hidden=self.vis_hidden or (8 * self.vis_d // 3 // 64 * 64),
                    image=self.vis_image, pool=self.vis_pool)

    @classmethod
    def from_dict(cls, d: dict) -> "HN1Config":
        d = dict(d)
        if "zone_kinds" not in d:                      # checkpoints anteriores: zona homogénea
            d["zone_kinds"] = ["mlp"] * d.get("n_experts", 8)
        return cls(**d)


def preset(name: str) -> HN1Config:
    """Configuraciones de referencia. Los conteos reales se verifican en las pruebas."""
    if name == "hn1-25m":
        return HN1Config()
    if name == "hn1-200m":
        return HN1Config(d_local=512, local_heads=8, local_mlp=1024, enc_layers=2, dec_layers=3,
                         d=1024, heads=16, mlp_hidden=2048,
                         pattern=("ssm", "ssm", "ssm", "attn", "ssm", "ssm", "ssm", "attn"),
                         expert_hidden=2048, top_k=2)
    if name == "hn1-500m":
        # Destino del crecimiento desde hn1-200m (mismo d y misma parte local, así se llega con
        # operaciones que preservan la función): +4 bloques al final del backbone, 16 especialistas
        # por zona (top-2: el cómputo por parche crece mucho menos que los parámetros) y zona visual.
        base = preset("hn1-200m")
        return HN1Config(**{**base.to_dict(), "base_blocks": len(base.pattern),
                            "pattern": list(base.pattern) + ["ssm", "ssm", "attn", "ssm"],
                            "zone_kinds": list(base.zone_kinds) * 2, "expert_hidden": 4096,
                            "vision": True, "vis_d": 768, "vis_layers": 12, "vis_heads": 12,
                            "vis_hidden": 2048})
    if name in ("hn1-700m", "hn1-1b"):
        # Intermedios alcanzables creciendo desde hn1-200m (pasando por hn1-500m): mismo d=1024,
        # misma parte local y la misma zona visual (atención cruzada en los mismos bloques, 3, 7 y 10).
        # La política de §6.5: primero especialistas (dispersos: casi no suben el cómputo por byte),
        # después bloques al final del backbone (sí lo suben). hn1-700m: 32 especialistas por zona y
        # 16 bloques; hn1-1b: 56 especialistas y 20 bloques.
        base = preset("hn1-500m")
        more, reps = (1, 4) if name == "hn1-700m" else (2, 7)
        return HN1Config(**{**base.to_dict(),
                            "pattern": list(base.pattern) + ["ssm", "ssm", "attn", "ssm"] * more,
                            "zone_kinds": list(preset("hn1-200m").zone_kinds) * reps,
                            "vis_blocks": list(base.vision_blocks())})
    if name == "hn1-cpu":        # ~7,7M: experimentos rápidos en CPU (ablaciones antes de la A770)
        return HN1Config(d_local=192, local_heads=4, local_mlp=384, d=256, heads=4, mlp_hidden=768,
                         pattern=("ssm", "attn", "ssm", "attn"), expert_hidden=256)
    if name == "hn1-tiny":       # para pruebas
        return HN1Config(d_local=32, local_heads=2, local_mlp=64, d=48, heads=4, mlp_hidden=96,
                         pattern=("ssm", "attn"), max_iters=3, expert_hidden=32,
                         zone_kinds=("mlp", "glu", "ei", "recurrente", "conv", "asociativa", "relacional"))
    raise KeyError(name)


def stable_loop(cfg: HN1Config, *, mean_iters: float = 4.0, train_max: int = 12, infer_max: int = 16,
                bptt: int | None = None, decay: float = 0.5, tol: float = 1e-2) -> HN1Config:
    """Variante «bucle estable» del núcleo recurrente (informe «Razonamiento por parámetro» §1.1.1):

    * reinyección de la entrada en cada iteración con una mezcla convexa (Parcae con Ā fijo);
    * sin ``step_embed`` (se puede iterar más en inferencia que al entrenar);
    * profundidad log-normal–Poisson al entrenar (media ``mean_iters``, tope ``train_max``) y,
      opcionalmente, backprop truncado a las últimas ``bptt`` iteraciones;
    * parada por convergencia al inferir, con ``infer_max`` iteraciones como techo.
    No agrega parámetros (quita los ``max_iters·d`` de ``step_embed``)."""
    from dataclasses import replace
    return replace(cfg, halting="convergencia", core_inject="convexa", core_decay=decay, core_step="ninguna",
                   core_depth="lognormal", core_mean_iters=mean_iters, core_train_max=train_max,
                   core_bptt=bptt, max_iters=infer_max, convergence_tol=tol)


# ---------------------------------------------------------------------------
# Parcheo
# ---------------------------------------------------------------------------
def patch_boundaries(x: torch.Tensor, max_patch: int,
                     entropy: tuple[torch.Tensor, float, int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Devuelve (``end``: el byte cierra un parche, ``patch_id`` de cada byte).

    Un parche termina en un byte de frontera o al llegar a ``max_patch`` bytes.
    La frontera es un espacio o signo de puntuación, o, con ``entropy =
    (tabla, umbral, orden)``, un punto donde el próximo byte es difícil de
    predecir (``hope.core.patching``). Vectorizado, sin bucles sobre la secuencia.
    """
    b, t = x.shape
    natural = natural_boundaries(x, entropy)
    pos = torch.arange(t, device=x.device).expand(b, t)
    starts = torch.zeros_like(natural)
    starts[:, 0] = True
    starts[:, 1:] = natural[:, :-1]
    run_start = torch.where(starts, pos, torch.zeros_like(pos)).cummax(dim=1).values
    in_run = pos - run_start
    end = natural | ((in_run + 1) % max_patch == 0)
    end[:, -1] = True
    patch_id = end.long().cumsum(dim=1) - end.long()
    return end, patch_id


def natural_boundaries(x: torch.Tensor, entropy=None) -> torch.Tensor:
    """Bytes que cierran un parche por su contenido (sin la regla de largo máximo)."""
    if entropy is not None:
        from hope.core.patching import entropy_boundaries
        return entropy_boundaries(x, entropy[0], float(entropy[1]), int(entropy[2]))
    table = torch.zeros(BYTE_VOCAB, dtype=torch.bool, device=x.device)
    table[torch.tensor(BOUNDARY_BYTES, device=x.device)] = True
    return table[x.long()]


def gather_patches(states: torch.Tensor, end: torch.Tensor, patch_id: torch.Tensor):
    """Toma el estado del último byte de cada parche. Devuelve [B, P, d] y la máscara de parches válidos."""
    b, t, d = states.shape
    n = end.sum(dim=1)
    p = int(n.max())
    target = torch.where(end, patch_id, torch.full_like(patch_id, p))
    idx = torch.zeros(b, p + 1, dtype=torch.long, device=states.device)
    idx.scatter_(1, target, torch.arange(t, device=states.device).expand(b, t))
    idx = idx[:, :p]
    valid = torch.arange(p, device=states.device)[None] < n[:, None]
    return states.gather(1, idx.unsqueeze(-1).expand(b, p, d)), valid


# ---------------------------------------------------------------------------
# Núcleo recurrente con parada adaptativa (ACT; Graves 2016)
# ---------------------------------------------------------------------------
def sinusoidal_step(i: int, d: int, device=None, dtype=None) -> torch.Tensor:
    """Codificación de la iteración ``i`` sin parámetros (senos y cosenos, como las posiciones de
    Vaswani et al. 2017): sirve para cualquier ``i``, también más allá de lo entrenado."""
    half = d // 2
    freq = torch.exp(-torch.arange(half, device=device, dtype=torch.float32) * (9.210340371976184 / max(half, 1)))
    ang = (i + 1) * freq
    enc = torch.zeros(d, device=device, dtype=torch.float32)
    enc[0:2 * half:2], enc[1:2 * half:2] = ang.sin(), ang.cos()
    return (0.1 * enc).to(dtype or torch.float32)


class _LoopInput(nn.Module):
    """Lo común a los dos núcleos: entrada de cada iteración y profundidad muestreada.

    * ``core_inject="convexa"``: la entrada ``x`` vuelve a entrar en cada iteración,
      ``z_i = Ā·h_i + (1−Ā)·x`` (Parcae, Prairie et al. 2026, con Ā escalar fijo y sin aprender).
      Con 0 < Ā < 1 el radio espectral de la parte lineal es < 1, así el estado residual no
      explota aunque se itere más que al entrenar. La primera iteración ve ``x`` tal cual.
    * ``core_step``: ``"embed"`` (tabla de ``max_iters`` filas, la de siempre), ``"senoidal"``
      (función continua de i) o ``"ninguna"`` (Huginn). Sin la tabla se puede iterar más en
      inferencia que en el entrenamiento.
    """

    def _init_loop(self, cfg: HN1Config):
        self.d = cfg.d
        self.inject, self.decay = cfg.core_inject, float(cfg.core_decay)
        self.step_kind = cfg.core_step
        self.depth_kind, self.mean_iters, self.sigma = cfg.core_depth, cfg.core_mean_iters, cfg.core_depth_sigma
        self.train_max = cfg.core_train_max or cfg.max_iters
        self.bptt = cfg.core_bptt
        if self.inject not in ("ninguna", "convexa"):
            raise ValueError(f"core_inject desconocido: {self.inject}")
        if self.step_kind not in ("embed", "senoidal", "ninguna"):
            raise ValueError(f"core_step desconocido: {self.step_kind}")
        if self.step_kind == "embed":
            self.step_embed = nn.Embedding(cfg.max_iters, cfg.d)

    def _iter_input(self, h: torch.Tensor, x: torch.Tensor, i: int) -> torch.Tensor:
        z = self.decay * h + (1.0 - self.decay) * x if self.inject == "convexa" and i > 0 else h
        if self.step_kind == "embed":
            if i >= self.step_embed.num_embeddings:
                raise ValueError(f"core_step='embed' admite como máximo {self.step_embed.num_embeddings} "
                                 "iteraciones; usar core_step='senoidal' o 'ninguna' para iterar más")
            z = z + self.step_embed.weight[i]
        elif self.step_kind == "senoidal":
            z = z + sinusoidal_step(i, self.d, z.device, z.dtype)
        return z

    def sample_depth(self) -> int:
        """Profundidad de entrenamiento: uniforme en 1..max_iters, o log-normal–Poisson de cola
        pesada (Geiping et al. 2025): τ ~ N(log(r̄) − σ²/2, σ), n = 1 + Poisson(e^τ), E[n] ≈ r̄ + 1.
        Con ``r̄ = core_mean_iters − 1`` la media queda en ``core_mean_iters``."""
        if self.depth_kind == "uniforme":
            return int(torch.randint(1, self.max_iters + 1, (1,)))
        if self.depth_kind != "lognormal":
            raise ValueError(f"core_depth desconocido: {self.depth_kind}")
        rate = max(self.mean_iters - 1.0, 1e-3)
        tau = torch.normal(torch.tensor(math.log(rate) - self.sigma ** 2 / 2), torch.tensor(self.sigma))
        n = 1 + int(torch.poisson(tau.exp()))
        return max(1, min(n, self.train_max))


class ConvergentCore(_LoopInput):
    """Núcleo recurrente con profundidad muestreada al entrenar y parada por convergencia al inferir.

    La investigación (Huginn, TRM, análisis de ARC Prize) muestra que la parada ACT
    ahorra cómputo pero no mejora la precisión, y que su costo empuja a usar una sola
    iteración. Acá, al entrenar se usa una cantidad de iteraciones al azar (sin costo
    de parada) y al inferir cada posición se congela cuando su estado deja de cambiar
    (cambio relativo < ``tol``). Causal: cada posición decide sola.

    Con ``core_bptt = k`` las primeras ``n − k`` iteraciones corren sin gradiente (backprop
    truncado de Huginn): ahorra memoria de activaciones, a costa de entrenar peor el bucle.
    """

    def __init__(self, cfg: HN1Config):
        super().__init__()
        self.max_iters, self.tol = cfg.max_iters, cfg.convergence_tol
        self.block = Block(cfg.d, cfg.core_kind, cfg.heads, cfg.mlp_hidden, kv_heads=cfg.kv_heads)
        self._init_loop(cfg)
        self.halt = nn.Linear(cfg.d, 1)                 # sin uso; mantiene la misma interfaz y registro

    def forward(self, x, n_iters: int | None = None, full_depth: bool = False, cache: list | None = None):
        if self.training and n_iters is None:
            n = self.sample_depth()
        else:
            n = n_iters or self.max_iters
        h = x
        frozen = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        out = x
        iters = torch.zeros(x.shape[:2], device=x.device)
        used = 0
        no_grad_until = n - self.bptt if (self.training and self.bptt and torch.is_grad_enabled()) else 0
        for i in range(n):
            if cache is not None and len(cache) <= i:
                cache.append({})
            if i < no_grad_until:
                with torch.no_grad():
                    new = self.block(self._iter_input(h, x, i))
            else:
                if i == no_grad_until and i > 0:
                    h = h.detach()
                new = self.block(self._iter_input(h, x, i), cache[i] if cache is not None else None)
            used = i + 1
            if self.training:
                h = new
                continue
            change = (new - h).float().norm(dim=-1) / h.float().norm(dim=-1).clamp_min(1e-6)
            out = torch.where(frozen.unsqueeze(-1), out, new)
            iters = torch.where(frozen, iters, torch.full_like(iters, i + 1))
            frozen |= change < self.tol
            h = new
            # Con caché (generación incremental) se calculan todas las iteraciones: las posiciones
            # siguientes atienden a los estados de esta en cada iteración.
            if cache is None and bool(frozen.all()):
                break
        if self.training:
            out, iters = h, torch.full(x.shape[:2], float(n), device=x.device)
        size = max(self.max_iters, n)
        dist = torch.zeros(size, device=x.device)
        for i in range(1, size + 1):
            dist[i - 1] = (iters == i).float().mean()
        return out, {"expected_iters": iters, "iters_used": used, "halt_dist": dist[:max(used, 1)]}


class RecurrentCore(_LoopInput):
    def __init__(self, cfg: HN1Config):
        super().__init__()
        self.max_iters, self.eps = cfg.max_iters, cfg.halt_eps
        self.block = Block(cfg.d, cfg.core_kind, cfg.heads, cfg.mlp_hidden, kv_heads=cfg.kv_heads)
        self._init_loop(cfg)
        self.halt = nn.Linear(cfg.d, 1)
        nn.init.zeros_(self.halt.weight)
        nn.init.constant_(self.halt.bias, -1.0)     # al inicio, ~27% de probabilidad de parar por paso

    def forward(self, x, n_iters: int | None = None, full_depth: bool = False, cache: list | None = None):
        """``full_depth``: ignora la parada y usa todas las iteraciones (entrenamiento),
        para que las iteraciones tardías reciban gradiente y no colapsen.

        Cada posición acumula ``Σ p_n h_n`` y se congela cuando su probabilidad
        pendiente baja de ``eps``. El bucle termina antes solo si todas las
        posiciones se congelaron, lo que no altera ninguna salida: el resultado de
        cada posición depende únicamente de posiciones anteriores (causalidad)."""
        n_max = n_iters or self.max_iters
        h = x
        remain = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
        out = torch.zeros_like(x)
        expected = torch.zeros_like(remain)
        halt_dist: list[torch.Tensor] = []     # peso medio de cada iteración (diagnóstico LTH)
        used = 0
        for n in range(n_max):
            if cache is not None and len(cache) <= n:
                cache.append({})
            h = self.block(self._iter_input(h, x, n), cache[n] if cache is not None else None)
            used = n + 1
            lam = torch.sigmoid(self.halt(h)).squeeze(-1)
            if full_depth:
                lam = torch.zeros_like(lam)
            if n == n_max - 1:
                lam = torch.ones_like(lam)
            p = remain * lam
            out = out + p.unsqueeze(-1) * h
            expected = expected + (n + 1) * p
            remain = remain * (1 - lam)
            halt_dist.append(p.detach().mean())
            # Congelar: el resto pendiente va a este estado y la posición deja de acumular.
            frozen = (remain < self.eps) & (remain > 0)
            if bool(frozen.any()):
                leftover = remain * frozen
                out = out + leftover.unsqueeze(-1) * h
                expected = expected + (n + 1) * leftover
                halt_dist[-1] = halt_dist[-1] + leftover.detach().mean()
                remain = torch.where(frozen, torch.zeros_like(remain), remain)
            if cache is None and not self.training and bool((remain == 0).all()):
                break
        return out, {"expected_iters": expected, "iters_used": used,
                     "halt_dist": torch.stack(halt_dist)}


# ---------------------------------------------------------------------------
# Región especialista con router top-k disperso (§3.5)
# ---------------------------------------------------------------------------
class SparseRegion(nn.Module):
    """Misma interfaz que ``toy.Region`` (router, experts, expert_mask) para que las
    operaciones de crecimiento de ``hope.growth.ops`` funcionen sin cambios, pero
    solo computa los especialistas elegidos."""

    def __init__(self, d: int, n_experts: int, hidden: int, top_k: int):
        super().__init__()
        self.d, self.top_k = d, top_k
        self.norm = RMSNorm(d)
        self.router = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList(Expert(d, hidden) for _ in range(n_experts))
        self.register_buffer("expert_mask", torch.zeros(n_experts))

    @property
    def n_experts(self) -> int:
        return len(self.experts)

    def forward(self, x, cache=None):          # especialistas por posición: no necesitan caché
        shape = x.shape
        h = self.norm(x).reshape(-1, self.d)
        logits = self.router(h) + self.expert_mask
        probs = logits.softmax(-1)
        k = min(self.top_k, self.n_experts)
        top_val, top_idx = logits.topk(k, dim=-1)
        weights = top_val.softmax(-1)
        out = torch.zeros_like(h)
        for e, expert in enumerate(self.experts):
            hit = top_idx == e
            rows = hit.any(-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            w = (weights * hit).sum(-1)[rows].unsqueeze(-1)
            out.index_add_(0, rows, (w * expert(h[rows])).to(out.dtype))
        # Pérdida de balanceo (Switch Transformer): evita que un especialista acapare todo.
        frac = F.one_hot(top_idx[:, 0], self.n_experts).float().mean(0)
        balance = self.n_experts * (frac * probs.float().mean(0)).sum()
        return out.reshape(shape), balance, top_idx.reshape(*shape[:-1], k)


# ---------------------------------------------------------------------------
class HN1Model(nn.Module):
    def __init__(self, cfg: HN1Config):
        super().__init__()
        self.cfg = cfg
        dl, d = cfg.d_local, cfg.d
        self.byte_embed = nn.Embedding(BYTE_VOCAB, dl)
        local = dict(window=cfg.window, sinks=cfg.sinks)
        self.encoder = nn.ModuleList(Block(dl, cfg.local_kind, cfg.local_heads, cfg.local_mlp, **local)
                                     for _ in range(cfg.enc_layers))
        self.to_global = nn.Linear(dl, d)
        self.backbone = nn.ModuleList(Block(d, k, cfg.heads, cfg.mlp_hidden, hnhs=cfg.hnhs, kv_heads=cfg.kv_heads,
                                            window=cfg.window, sinks=cfg.sinks) for k in cfg.pattern)
        # Conectoma dinámico: conexiones dirigidas entre bloques del backbone que se crean y
        # eliminan durante la vida del modelo (hope/plasticity/dynamic.py). Vacío al inicio.
        self.connections = nn.ModuleDict()
        self.memory = MemoryCrossAttention(d, cfg.heads) if cfg.memory else None
        self.vision = (VisualZone(d, cfg.heads, cfg.vision_blocks(), **cfg.vision_kwargs())
                       if cfg.vision else None)
        self.core = ConvergentCore(cfg) if cfg.halting == "convergencia" else RecurrentCore(cfg)
        self.regions = nn.ModuleList(Zone(d, cfg.zone_kinds, cfg.expert_hidden, cfg.top_k,
                                          balance=cfg.balance, shared_expert=cfg.shared_expert)
                                     for _ in range(cfg.n_regions))
        self.global_norm = RMSNorm(d)
        self.to_local = nn.Linear(d, dl)
        self.decoder = nn.ModuleList(Block(dl, cfg.local_kind, cfg.local_heads, cfg.local_mlp, **local)
                                     for _ in range(cfg.dec_layers))
        self.out_norm = RMSNorm(dl)
        self.head = nn.Linear(dl, BYTE_VOCAB, bias=False)
        self.uncertainty = nn.Linear(dl, 1)          # predice la pérdida propia por byte (señal para UAC)
        # Cada cabeza multi-byte: un bloque residual chico (dl × dl) + la cabeza de salida compartida.
        self.mb_heads = nn.ModuleList(nn.Sequential(RMSNorm(dl), nn.Linear(dl, dl), nn.SiLU())
                                      for _ in range(cfg.multibyte))
        set_context_scale(self, cfg.rope_scale)
        if cfg.patching == "entropia":
            self.register_buffer("entropy_table", torch.full((cfg.entropy_buckets,), 8.0, dtype=torch.float16))
            self.register_buffer("entropy_threshold", torch.tensor(8.0))

    def _entropy(self):
        if self.cfg.patching != "entropia":
            return None
        return self.entropy_table, float(self.entropy_threshold), self.cfg.entropy_order

    def fit_patcher(self, data, target_bytes_per_patch: float = 4.5) -> dict:
        """Ajusta el parcheo por entropía con un corpus de bytes. Hacerlo **antes** de entrenar:
        cambiarlo después cambia la estructura de la entrada del modelo global."""
        import numpy as np
        from hope.core.patching import calibrate_threshold, fit_entropy_table
        if self.cfg.patching != "entropia":
            raise ValueError("el modelo no está configurado con patching='entropia'")
        arr = np.asarray(data, dtype=np.uint8)
        table = fit_entropy_table(arr, self.cfg.entropy_order, self.cfg.entropy_buckets)
        thr = calibrate_threshold(arr, table, self.cfg.entropy_order, self.cfg.max_patch, target_bytes_per_patch)
        self.entropy_table.copy_(torch.from_numpy(table))
        self.entropy_threshold.fill_(thr)
        return {"threshold_bits": thr, "target_bytes_per_patch": target_bytes_per_patch}

    @property
    def d(self) -> int:
        return self.cfg.d

    # --- partes del cómputo -------------------------------------------------------
    def _local_encode(self, x: torch.Tensor):
        end, patch_id = patch_boundaries(x, self.cfg.max_patch, self._entropy())
        h = self.byte_embed(x.long())
        for blk in self.encoder:
            h = self._run(blk, h)
        return h, end, patch_id

    def _run(self, blk, x):
        """Aplica un bloque; al entrenar con ``grad_checkpointing`` recalcula sus activaciones
        en el paso hacia atrás en lugar de guardarlas (menos memoria → lotes más grandes)."""
        if self.training and getattr(self, "grad_checkpointing", False) and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(blk, x, use_reentrant=False)
        return blk(x)

    def encode_image(self, image: torch.Tensor | None) -> torch.Tensor | None:
        """Imagen(es) [B, 3, H, W] o [3, H, W] → fichas visuales [B, V, d]. Se calcula una vez
        por imagen y se reutiliza en cada parche (y en toda la generación)."""
        if image is None:
            return None
        if self.vision is None:
            raise ValueError("el modelo no tiene zona visual (agregarla con AddVisualZone)")
        return self.vision.encode(image.to(self.head.weight.device), tiles=self.cfg.vis_tiling)

    def _global(self, patches: torch.Tensor, memory=None, n_iters=None, full_depth=False, vis=None,
                cache: dict | None = None):
        """Parte global sobre parches. Con ``cache`` (``{}`` al empezar) procesa solo los parches
        nuevos y guarda el estado de cada bloque: atención (claves/valores), SSM y especialistas
        secuenciales (estado), núcleo (una caché por iteración). Da lo mismo que recalcular todo."""
        g = self.to_global(patches)
        outs = []
        if cache is not None:
            cache.setdefault("bb", [{} for _ in self.backbone])
            cache.setdefault("core", [])
            cache.setdefault("zones", [{} for _ in self.regions])
            while len(cache["bb"]) < len(self.backbone):
                cache["bb"].append({})
        for j, blk in enumerate(self.backbone):
            for key, conn in self.connections.items():
                src, dst = (int(v) for v in key.split("_"))
                if dst == j:
                    g = g + conn(outs[src])
            g = blk(g, cache["bb"][j]) if cache is not None else self._run(blk, g)
            if self.vision is not None and vis is not None:
                g = self.vision.inject(j, g, vis)
            outs.append(g)
        if self.memory is not None and memory is not None:
            if memory.dim() == 2:                       # [M, d] compartida por todo el lote
                memory = memory.unsqueeze(0).expand(g.shape[0], -1, -1)
            g = self.memory(g, memory)
        g, core_info = self.core(g, n_iters, full_depth, cache["core"] if cache is not None else None)
        balance = g.new_zeros(())
        routes = []
        for ri, region in enumerate(self.regions):
            r, bal, idx = region(g, cache["zones"][ri]) if cache is not None else region(g)
            g = g + r
            balance = balance + bal
            routes.append(idx)
        return self.global_norm(g), core_info, balance, routes

    def _decode(self, h: torch.Tensor, g: torch.Tensor, patch_id: torch.Tensor) -> torch.Tensor:
        # El byte del parche p recibe el contexto global del parche p - 1 (ya completo).
        ctx = torch.cat([torch.zeros_like(g[:, :1]), g[:, :-1]], dim=1)
        ctx = self.to_local(ctx).gather(1, patch_id.unsqueeze(-1).expand(-1, -1, self.cfg.d_local))
        y = h + ctx
        for blk in self.decoder:
            y = self._run(blk, y)
        return self.out_norm(y)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None, n_iters: int | None = None,
                full_depth: bool = False, image: torch.Tensor | None = None, vis: torch.Tensor | None = None):
        """``x``: bytes [B, T]. Devuelve logits del byte siguiente [B, T, 256] y señales auxiliares.
        ``image``: imágenes [B, 3, H, W] (o ``vis`` ya codificadas) que condicionan el texto."""
        if vis is None and image is not None:
            vis = self.encode_image(image)
        h, end, patch_id = self._local_encode(x)
        patches, valid = gather_patches(h, end, patch_id)
        g, core_info, balance, routes = self._global(patches, memory, n_iters, full_depth, vis)
        y = self._decode(h, g, patch_id)
        return {
            "hidden": y,
            "global": g,
            "logits": self.head(y),
            "uncertainty": self.uncertainty(y).squeeze(-1),
            "balance": balance,
            "expected_iters": (core_info["expected_iters"] * valid).sum() / valid.sum(),
            "iters_used": core_info["iters_used"],
            "halt_dist": core_info["halt_dist"],
            "patches_per_byte": valid.sum().item() / x.numel(),
            "routes": routes,
            "patch_id": patch_id,
            "valid": valid,
        }

    # --- generación -----------------------------------------------------------------
    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float) -> int:
        if temperature > 0:
            return int(torch.multinomial(torch.softmax(logits / temperature, -1), 1))
        return int(logits.argmax())

    @staticmethod
    def _should_stop(out: list[int], nxt: int, stop: bytes, stop_after_line: bytes | None,
                     stop_seq: bytes | None = None) -> bool:
        if bytes([nxt]) == stop:
            out.pop()
            return True
        if stop_seq and len(out) >= len(stop_seq) and bytes(out[-len(stop_seq):]) == stop_seq:
            return True
        if stop_after_line and nxt == 10:
            lines = bytes(out).split(b"\n")
            return len(lines) >= 2 and lines[-2].lstrip().startswith(stop_after_line)
        return False

    @torch.no_grad()
    def generate_patch_cache(self, prompt: bytes | str, max_new: int = 256, *, stop: bytes = b"\x00",
                             stop_after_line: bytes | None = None, temperature: float = 0.0,
                             memory: torch.Tensor | None = None, max_context: int = 2048,
                 stop_seq: bytes | None = None, n_iters: int | None = None,
                 image: torch.Tensor | None = None, vis: torch.Tensor | None = None) -> bytes:
        """Genera bytes a continuación de ``prompt`` (propuesta 3: caché por parche).

        La parte global (backbone, núcleo recurrente, zonas) es la más costosa y
        solo cambia cuando **se completa un parche**: los bytes dentro de un
        parche usan el contexto del parche anterior, que no cambia. Por eso se
        guarda y se recalcula solo al cerrar un parche (≈ una vez cada 4–5 bytes).
        La parte local se recalcula en cada byte. Da exactamente lo mismo que el
        cálculo completo (prueba en ``tests/test_generation.py``).
        """
        was_training = self.training
        self.eval()
        dev = self.head.weight.device
        seq = list(prompt.encode("utf-8") if isinstance(prompt, str) else prompt)
        if vis is None and image is not None:
            vis = self.encode_image(image)
        out: list[int] = []
        cache_g, cache_start = None, -1
        for _ in range(max_new):
            start = max(0, len(seq) - max_context)
            if start != cache_start:                     # cambió la ventana: el parcheo se desplaza
                cache_g, cache_start = None, start
            x = torch.tensor([seq[start:]], device=dev)
            h, end, patch_id = self._local_encode(x)
            needed = int(patch_id[0, -1])                # parches completos antes del parche actual
            if needed > 0 and (cache_g is None or cache_g.shape[1] < needed):
                patches, _ = gather_patches(h, end, patch_id)
                cache_g = self._global(patches[:, :needed], memory, n_iters, vis=vis)[0]
            g = cache_g[:, :needed] if needed > 0 else h.new_zeros(1, 0, self.cfg.d)
            g = torch.cat([g, g.new_zeros(1, 1, self.cfg.d)], dim=1)   # el parche actual no aporta contexto
            y = self._decode(h, g, patch_id)
            nxt = self._sample(self.head(y[0, -1]).float(), temperature)
            seq.append(nxt)
            out.append(nxt)
            if self._should_stop(out, nxt, stop, stop_after_line, stop_seq):
                break
        self.train(was_training)
        return bytes(out)

    @torch.no_grad()
    def generate(self, prompt: bytes | str, max_new: int = 256, *, stop: bytes = b"\x00",
                 stop_after_line: bytes | None = None, temperature: float = 0.0,
                 memory: torch.Tensor | None = None, max_context: int = 2048,
                 stop_seq: bytes | None = None, n_iters: int | None = None,
                 image: torch.Tensor | None = None, vis: torch.Tensor | None = None) -> bytes:
        """Generación incremental: caché de claves/valores en la parte local + caché por parche.

        * Codificador y decodificador locales: cada byte nuevo atiende a las claves y
          valores guardados → costo por byte constante, no proporcional al texto.
        * Parte global: se recalcula solo al completarse un parche.
        * Las fronteras de parche se calculan de forma incremental con el mismo
          criterio que ``patch_boundaries``.
        Da lo mismo que ``generate_reference`` (prueba en ``tests/test_generation.py``).
        Si el texto supera ``max_context`` se usa ``generate_patch_cache`` (ventana deslizante).
        """
        seq = list(prompt.encode("utf-8") if isinstance(prompt, str) else prompt)
        if not seq or len(seq) + max_new > max_context or any(b.kind != "attn" for b in [*self.encoder, *self.decoder]):
            return self.generate_patch_cache(prompt, max_new, stop=stop, stop_after_line=stop_after_line,
                                             temperature=temperature, memory=memory, max_context=max_context,
                                             stop_seq=stop_seq, n_iters=n_iters, image=image, vis=vis)
        was_training = self.training
        self.eval()
        dev = self.head.weight.device
        if vis is None and image is not None:
            vis = self.encode_image(image)
        ent = self._entropy()
        enc_cache = [{} for _ in self.encoder]
        dec_cache = [{} for _ in self.decoder]

        def encode(xb):
            h = self.byte_embed(xb)
            for blk, c in zip(self.encoder, enc_cache):
                h = blk(h, c)
            return h

        def decode(y):
            for blk, c in zip(self.decoder, dec_cache):
                y = blk(y, c)
            return self.out_norm(y)

        # --- prefill ---
        x = torch.tensor([seq], device=dev)
        h = encode(x)
        nat = natural_boundaries(x, ent)[0].tolist()
        ends, in_run, run_start = [], 0, 0
        for t, n in enumerate(nat):                      # misma regla que patch_boundaries, sin forzar el último
            if t > 0 and nat[t - 1]:
                run_start = t
            in_run = t - run_start
            ends.append(bool(n) or (in_run + 1) % self.cfg.max_patch == 0)
        pid = [0]
        for e in ends[:-1]:
            pid.append(pid[-1] + int(e))
        patch_states = [h[:, t] for t, e in enumerate(ends[:-1]) if e]       # parches completos
        inc = GlobalCache(self, memory, n_iters, vis)

        def global_ctx(n_complete: int) -> torch.Tensor:
            return inc.upto(patch_states, n_complete)

        g = global_ctx(pid[-1])
        g_ext = torch.cat([h.new_zeros(1, 1, self.cfg.d), g], dim=1)                 # contexto = parche anterior
        ctx = self.to_local(g_ext).gather(1, torch.tensor([pid], device=dev).unsqueeze(-1)
                                          .expand(-1, -1, self.cfg.d_local))
        y = decode(h + ctx)
        logits = self.head(y[0, -1]).float()
        last_end, last_nat = ends[-1], nat[-1]
        cur_pid, in_run = pid[-1], in_run
        last_h = h[:, -1]
        out: list[int] = []
        # --- generación ---
        for _ in range(max_new):
            nxt = self._sample(logits, temperature)
            seq.append(nxt)
            out.append(nxt)
            if self._should_stop(out, nxt, stop, stop_after_line, stop_seq):
                break
            if last_end:                                  # el byte anterior cerró un parche
                patch_states.append(last_h)
                cur_pid += 1
            in_run = 0 if last_nat else in_run + 1
            xb = torch.tensor([[nxt]], device=dev)
            hn = encode(xb)
            tail = torch.tensor([seq[-max(ent[2] if ent else 1, 1):]], device=dev)
            last_nat = bool(natural_boundaries(tail, ent)[0, -1])
            last_end = last_nat or (in_run + 1) % self.cfg.max_patch == 0
            gctx = global_ctx(cur_pid)
            c = gctx[:, cur_pid - 1] if cur_pid > 0 else hn.new_zeros(1, self.cfg.d)
            ctxn = self.to_local(c).unsqueeze(1) if cur_pid > 0 else self.to_local(hn.new_zeros(1, 1, self.cfg.d))
            yn = decode(hn + ctxn)
            logits = self.head(yn[0, -1]).float()
            last_h = hn[:, -1]
        self.train(was_training)
        return bytes(out)

    @torch.no_grad()
    def generate_long(self, prompt: bytes | str, max_new: int = 256, *, segment: int = 1024, **kw) -> bytes:
        """Generación con un contexto más largo que la ventana: lo anterior al último segmento se
        lee por partes y entra comprimido por la capa de memoria; el último segmento, completo."""
        seq = prompt.encode("utf-8") if isinstance(prompt, str) else bytes(prompt)
        if len(seq) <= segment or self.memory is None:
            return self.generate(seq, max_new, **kw)
        cut = len(seq) - segment
        was = self.training
        self.eval()
        x = torch.tensor([list(seq[:cut])], device=self.head.weight.device)
        mem = self.read_segments(x, segment, kw.pop("memory", None))
        self.train(was)
        return self.generate(seq[cut:], max_new, memory=mem, **kw)

    @torch.no_grad()
    def generate_reference(self, prompt: bytes | str, max_new: int = 256, *, stop: bytes = b"\x00",
                           stop_after_line: bytes | None = None, temperature: float = 0.0,
                           memory: torch.Tensor | None = None, max_context: int = 2048,
                 stop_seq: bytes | None = None, n_iters: int | None = None,
                 image: torch.Tensor | None = None, vis: torch.Tensor | None = None) -> bytes:
        """Versión simple (recalcula todo en cada byte); se usa para comprobar ``generate``."""
        was_training = self.training
        self.eval()
        dev = self.head.weight.device
        seq = list(prompt.encode("utf-8") if isinstance(prompt, str) else prompt)
        if vis is None and image is not None:
            vis = self.encode_image(image)
        out: list[int] = []
        for _ in range(max_new):
            x = torch.tensor([seq[-max_context:]], device=dev)
            nxt = self._sample(self.forward(x, memory=memory, n_iters=n_iters, vis=vis)["logits"][0, -1].float(),
                               temperature)
            seq.append(nxt)
            out.append(nxt)
            if self._should_stop(out, nxt, stop, stop_after_line, stop_seq):
                break
        self.train(was_training)
        return bytes(out)

    def encode_memory(self, chunks: list[bytes | str], max_len: int = 256) -> torch.Tensor:
        """Codifica fragmentos recuperados de la memoria (AHMC/MDD) como vectores [M, d]
        para la atención cruzada de memoria (§3.7). Usa el mismo codificador de bytes
        del modelo y promedia los estados de cada fragmento."""
        if not chunks:
            return torch.zeros(0, self.cfg.d, device=self.head.weight.device)
        raw = [c.encode("utf-8") if isinstance(c, str) else c for c in chunks]
        raw = [r[:max_len] or b" " for r in raw]
        width = max(len(r) for r in raw)
        dev = self.head.weight.device
        x = torch.zeros(len(raw), width, dtype=torch.long, device=dev)
        mask = torch.zeros(len(raw), width, device=dev)
        for i, r in enumerate(raw):
            x[i, :len(r)] = torch.tensor(list(r), device=dev)
            mask[i, :len(r)] = 1
        h = self.byte_embed(x)
        for blk in self.encoder:
            h = blk(h)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.to_global(pooled)

    # --- contexto largo con memoria de segmentos (sin parámetros nuevos) ---------------------
    def summarize(self, g: torch.Tensor) -> torch.Tensor:
        """Comprime la salida global de un segmento [B, P, d] en resúmenes [B, P/chunk, d]
        (promedio cada ``segment_chunk`` parches). Van a la capa de memoria que ya existe."""
        c = self.cfg.segment_chunk
        b, p, d = g.shape
        n = max(p // c, 1)
        return g[:, :n * c].reshape(b, n, -1, d).mean(2) if p >= c else g.mean(1, keepdim=True)

    def read_segments(self, x: torch.Tensor, segment: int, memory: torch.Tensor | None = None) -> torch.Tensor | None:
        """Lee un texto largo [B, T] por segmentos y devuelve la memoria comprimida de todo lo leído."""
        mem = memory
        for s in range(0, x.shape[1], segment):
            out = self.forward(x[:, s:s + segment], memory=mem)
            summ = self.summarize(out["global"])
            mem = summ if mem is None else torch.cat([mem if mem.dim() == 3 else mem.unsqueeze(0).expand(
                summ.shape[0], -1, -1), summ], dim=1)[:, -self.cfg.max_segment_memory:]
        return mem

    def long_loss(self, x: torch.Tensor, segment: int, **kw) -> tuple[torch.Tensor, dict]:
        """Pérdida sobre un texto más largo que la ventana: cada segmento ve los anteriores solo a
        través de sus resúmenes (recurrencia por segmentos; Transformer-XL / RMT / Block-Recurrent).
        Los resúmenes se desconectan del gradiente (costo de memoria constante)."""
        total, n, stats, mem = 0.0, 0, {}, None
        nbytes, ce = 0, 0.0
        weights = kw.pop("weights", None)
        for s in range(0, x.shape[1] - 1, segment):
            seg = x[:, s:s + segment + 1]
            if seg.shape[1] < 2:
                break
            w = weights[:, s:s + segment + 1] if weights is not None else None
            loss, stats = self.loss(seg, mem, weights=w, **kw)
            # Cada segmento pesa según los bytes que predice: un último segmento corto no vale
            # lo mismo que uno completo (así la pérdida es el promedio por byte de todo el texto).
            t = seg.shape[1] - 1
            total = total + loss * t
            ce += stats["ce"] * t
            nbytes += t
            n += 1
            summ = self.summarize(self._last_global)
            mem = summ if mem is None else torch.cat([mem, summ], dim=1)[:, -self.cfg.max_segment_memory:]
        if nbytes:
            stats = {**stats, "ce": ce / nbytes, "bpb": ce / nbytes / 0.6931471805599453}
        stats = {**stats, "segmentos": n}
        return total / max(nbytes, 1), stats

    def loss(self, x: torch.Tensor, memory: torch.Tensor | None = None, *, full_depth: bool = False,
             ponder_scale: float = 1.0, image: torch.Tensor | None = None,
             weights: torch.Tensor | None = None, contrastive: float = 0.0) -> tuple[torch.Tensor, dict]:
        """Pérdida de entrenamiento sobre una secuencia de bytes (predice x[1:] desde x[:-1]).

        ``ponder_scale`` permite subir gradualmente el costo de iterar; con
        ``full_depth`` el costo de iterar no se aplica (todas las iteraciones son obligatorias).
        ``contrastive``: peso de la pérdida auxiliar imagen–texto (``contrastive_loss``).
        """
        vis = self.encode_image(image) if image is not None else None
        out = self.forward(x[:, :-1], memory, full_depth=full_depth, vis=vis)
        self._last_global = out["global"].detach()
        target = x[:, 1:].long()
        ce_each = F.cross_entropy(out["logits"].float().reshape(-1, BYTE_VOCAB), target.reshape(-1),
                                  reduction="none").view_as(target)
        if weights is not None:            # p. ej. 0 en la pregunta y 1 en la respuesta del maestro
            w = weights[:, 1:].to(ce_each.dtype)
            ce = (ce_each * w).sum() / w.sum().clamp_min(1.0)
        else:
            ce = ce_each.mean()
        mb = ce.new_zeros(())
        for k, head in enumerate(self.mb_heads, start=1):           # byte t+1+k desde el estado de t
            y = out["hidden"][:, :-k]
            if y.shape[1] == 0:
                continue
            mb = mb + F.cross_entropy(self.head(y + head(y)).float().reshape(-1, BYTE_VOCAB),
                                      x[:, 1 + k:].long().reshape(-1))
        if len(self.mb_heads):
            mb = mb / len(self.mb_heads)
        unc = F.mse_loss(out["uncertainty"].float(), ce_each.detach())
        con = ce.new_zeros(())
        if contrastive and vis is not None and x.shape[0] > 1:
            con = self.contrastive_loss(x, vis, weights)
        total = (ce + contrastive * con + self.cfg.balance_weight * out["balance"]
                 + (0.0 if full_depth or self.cfg.halting != "act" else self.cfg.ponder_weight * ponder_scale)
                 * out["expected_iters"]
                 + self.cfg.uncertainty_weight * unc
                 + self.cfg.multibyte_weight * mb)
        stats = {"ce": ce.item(), "bpb": ce.item() / 0.6931471805599453,
                 "balance": out["balance"].item(), "expected_iters": out["expected_iters"].item(),
                 "uncertainty_mse": unc.item(), "patches_per_byte": out["patches_per_byte"],
                 "multibyte_ce": mb.item(), "contrastiva": con.item()}
        return total, stats

    def contrastive_loss(self, x: torch.Tensor, vis: torch.Tensor, weights: torch.Tensor | None = None,
                         temperature: float = 10.0, bias: float = -10.0) -> torch.Tensor:
        """Pérdida sigmoide imagen–texto estilo SigLIP (Zhai et al. 2023) **sin parámetros nuevos**.

        * Imagen: promedio de las fichas visuales (las mismas que usa la atención cruzada).
        * Texto: promedio de los estados globales de Hope sobre los parches de la respuesta,
          calculados **sin la imagen** y sin gradiente (texto fijo, como LiT pero al revés): el
          codificador visual aprende a llevar cada imagen al espacio donde Hope ya representa
          su descripción.
        * Temperatura y sesgo fijos (los valores iniciales de SigLIP: t = 10, b = −10).
        * Dos filas con la misma respuesta cuentan como par positivo (en las procedurales se
          repiten respuestas cortas como «3»; tratarlas como negativas sería ruido)."""
        with torch.no_grad():
            txt = self.forward(x[:, :-1])
        g, pid = txt["global"].float(), txt["patch_id"]
        m = (weights[:, 1:] > 0) if weights is not None else torch.ones_like(pid, dtype=torch.bool)
        sel = torch.zeros(g.shape[:2], device=g.device).scatter_reduce(1, pid, m.float(), "amax")
        zt = (g * sel.unsqueeze(-1)).sum(1) / sel.sum(1, keepdim=True).clamp_min(1.0)
        zi = vis.float().mean(1)
        logits = temperature * F.normalize(zi, dim=-1) @ F.normalize(zt, dim=-1).T + bias
        ans = x[:, 1:] * m
        labels = 2.0 * (ans[:, None] == ans[None]).all(-1).float() - 1.0
        return -F.logsigmoid(labels * logits).sum() / x.shape[0]


# ---------------------------------------------------------------------------
def set_context_scale(model: "HN1Model", scale: float) -> None:
    """Extiende el alcance de las posiciones de la atención sin reentrenar (RoPE «NTK-aware»,
    bloc97 2023; base de YaRN, Peng et al. 2023): base' = base · s^(dh/(dh−2)). Con s = 4, un modelo
    entrenado con 2 K parches puede atender a ~8 K con poca pérdida; un enfriamiento corto con
    textos largos la recupera. Los bloques SSM no tienen posiciones: su alcance ya es ilimitado."""
    from hope.core.layers import CausalSelfAttention
    model.cfg.rope_scale = scale
    for mod in model.modules():
        if isinstance(mod, CausalSelfAttention):
            dh = mod.dh
            mod.rope_base = 10000.0 * (scale ** (dh / max(dh - 2, 1)) if scale != 1.0 else 1.0)


class GlobalCache:
    """Salida global de los parches ya procesados + estado incremental de cada bloque.

    Al cerrarse un parche se procesa **solo ese parche** (costo constante por parche) en
    lugar de recalcular la parte global entera (costo proporcional al texto).
    ``snapshot``/``restore`` permiten descartar parches provisionales (generación especulativa):
    los estados se reemplazan, nunca se modifican en el lugar, así que copiar los
    diccionarios alcanza."""

    def __init__(self, model: HN1Model, memory=None, n_iters=None, vis=None):
        self.model, self.memory, self.n_iters, self.vis = model, memory, n_iters, vis
        self.state: dict = {}
        self.g: torch.Tensor | None = None

    def upto(self, patch_states: list, n: int) -> torch.Tensor:
        have = 0 if self.g is None else self.g.shape[1]
        if n > have:
            new = torch.stack(patch_states[have:n], dim=1)
            out = self.model._global(new, self.memory, self.n_iters, vis=self.vis, cache=self.state)[0]
            self.g = out if self.g is None else torch.cat([self.g, out], dim=1)
        if self.g is None:
            d = self.model.cfg.d
            return torch.zeros(1, 0, d, device=self.model.head.weight.device, dtype=self.model.head.weight.dtype)
        return self.g[:, :n]

    @staticmethod
    def _copy(x):
        if isinstance(x, dict):
            return {k: GlobalCache._copy(v) for k, v in x.items()}
        if isinstance(x, list):
            return [GlobalCache._copy(v) for v in x]
        return x

    def snapshot(self):
        return self._copy(self.state), self.g

    def restore(self, snap) -> None:
        self.state, self.g = self._copy(snap[0]), snap[1]


def build_registry(model: HN1Model) -> TopologyRegistry:
    """Registro de topología del núcleo con IDs estables (§6.1)."""
    reg = TopologyRegistry()
    root = reg.add(NodeKind.MODEL, "", node_id="mod_hn1_0", meta={"config": model.cfg.to_dict()})
    for path, kind in (("byte_embed", NodeKind.OPERATOR), ("encoder", NodeKind.REGION),
                       ("to_global", NodeKind.OPERATOR), ("to_local", NodeKind.OPERATOR),
                       ("decoder", NodeKind.REGION), ("out_norm", NodeKind.OPERATOR),
                       ("global_norm", NodeKind.OPERATOR), ("head", NodeKind.HEAD),
                       ("uncertainty", NodeKind.HEAD)):
        reg.add(kind, path, parent=root.id, hint=path)
    bb = reg.add(NodeKind.REGION, "backbone", parent=root.id, hint="backbone")
    for i, blk in enumerate(model.backbone):
        reg.add(NodeKind.POPULATION, f"backbone.{i}", parent=bb.id, hint=f"backbone-{blk.kind}-{i}",
                meta={"kind": blk.kind})
    if model.memory is not None:
        reg.add(NodeKind.OPERATOR, "memory", parent=root.id, hint="memory")
    if model.vision is not None:
        register_vision(reg, model, root.id)
    for i in range(len(model.mb_heads)):
        reg.add(NodeKind.HEAD, f"mb_heads.{i}", parent=root.id, hint=f"multibyte-{i + 2}")
    core = reg.add(NodeKind.REGION, "core", parent=root.id, hint="recurrent-core",
                   meta={"shared_weights": True, "max_iters": model.cfg.max_iters})
    reg.add(NodeKind.HEAD, "core.halt", parent=core.id, hint="halt")
    for i, region in enumerate(model.regions):
        r = reg.add(NodeKind.REGION, f"regions.{i}", parent=root.id, hint=f"zone-{i}",
                    meta={"kinds": list(getattr(region, "kinds", []))})
        reg.add(NodeKind.HEAD, f"regions.{i}.router", parent=r.id, hint=f"router-{i}")
        reg.add(NodeKind.OPERATOR, f"regions.{i}.norm", parent=r.id, hint=f"norm-{i}")
        frac = min(region.top_k, region.n_experts) / region.n_experts
        for e, expert in enumerate(region.experts):
            kind = getattr(expert, "kind", "mlp")
            reg.add(NodeKind.EXPERT, f"regions.{i}.experts.{e}", parent=r.id,
                    hint=f"{kind}-{i}", activation=frac, meta={"kind": kind})
    return reg


def register_vision(reg: TopologyRegistry, model: HN1Model, root_id: str) -> None:
    vz = reg.add(NodeKind.REGION, "vision", parent=root_id, hint="vision-zone",
                 meta={"image": model.cfg.vis_image, "patch": model.cfg.vis_patch,
                       "tokens_per_image": model.vision.encoder.tokens_per_image()})
    reg.add(NodeKind.OPERATOR, "vision.encoder", parent=vz.id, hint="vision-encoder")
    for k in model.vision.xattn:
        reg.add(NodeKind.OPERATOR, f"vision.xattn.{k}", parent=vz.id, hint=f"vision-xattn-{k}")


def build_hn1(cfg: HN1Config | str, device: str | torch.device = "cpu") -> tuple[HN1Model, TopologyRegistry]:
    if isinstance(cfg, str):
        cfg = preset(cfg)
    with torch.device(device):
        model = HN1Model(cfg)
    return model, build_registry(model)
