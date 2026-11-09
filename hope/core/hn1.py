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

from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from hope.core.layers import Block, MemoryCrossAttention, RMSNorm
from hope.core.zones import Zone
from hope.core.toy import NEG_INF, Expert
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

    def __post_init__(self):
        self.pattern = tuple(self.pattern)
        self.zone_kinds = tuple(self.zone_kinds)
        self.n_experts = len(self.zone_kinds)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pattern"] = list(self.pattern)
        d["zone_kinds"] = list(self.zone_kinds)
        return d

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
    if name == "hn1-tiny":       # para pruebas
        return HN1Config(d_local=32, local_heads=2, local_mlp=64, d=48, heads=4, mlp_hidden=96,
                         pattern=("ssm", "attn"), max_iters=3, expert_hidden=32,
                         zone_kinds=("mlp", "glu", "ei", "recurrente", "conv", "asociativa", "relacional"))
    raise KeyError(name)


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
class ConvergentCore(nn.Module):
    """Núcleo recurrente con profundidad muestreada al entrenar y parada por convergencia al inferir.

    La investigación (Huginn, TRM, análisis de ARC Prize) muestra que la parada ACT
    ahorra cómputo pero no mejora la precisión, y que su costo empuja a usar una sola
    iteración. Acá, al entrenar se usa una cantidad de iteraciones al azar (sin costo
    de parada) y al inferir cada posición se congela cuando su estado deja de cambiar
    (cambio relativo < ``tol``). Causal: cada posición decide sola.
    """

    def __init__(self, cfg: HN1Config):
        super().__init__()
        self.max_iters, self.tol = cfg.max_iters, cfg.convergence_tol
        self.block = Block(cfg.d, cfg.core_kind, cfg.heads, cfg.mlp_hidden, kv_heads=cfg.kv_heads)
        self.step_embed = nn.Embedding(cfg.max_iters, cfg.d)
        self.halt = nn.Linear(cfg.d, 1)                 # sin uso; mantiene la misma interfaz y registro

    def forward(self, x, n_iters: int | None = None, full_depth: bool = False):
        if self.training and n_iters is None:
            n = int(torch.randint(1, self.max_iters + 1, (1,)))
        else:
            n = n_iters or self.max_iters
        h = x
        frozen = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        out = x
        iters = torch.zeros(x.shape[:2], device=x.device)
        used = 0
        for i in range(n):
            new = self.block(h + self.step_embed.weight[i])
            used = i + 1
            if self.training:
                h = new
                continue
            change = (new - h).norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-6)
            out = torch.where(frozen.unsqueeze(-1), out, new)
            iters = torch.where(frozen, iters, torch.full_like(iters, i + 1))
            frozen |= change < self.tol
            h = new
            if bool(frozen.all()):
                break
        if self.training:
            out, iters = h, torch.full(x.shape[:2], float(n), device=x.device)
        dist = torch.zeros(self.max_iters, device=x.device)
        for i in range(1, self.max_iters + 1):
            dist[i - 1] = (iters == i).float().mean()
        return out, {"expected_iters": iters, "iters_used": used, "halt_dist": dist[:max(used, 1)]}


class RecurrentCore(nn.Module):
    def __init__(self, cfg: HN1Config):
        super().__init__()
        self.max_iters, self.eps = cfg.max_iters, cfg.halt_eps
        self.block = Block(cfg.d, cfg.core_kind, cfg.heads, cfg.mlp_hidden, kv_heads=cfg.kv_heads)
        self.step_embed = nn.Embedding(cfg.max_iters, cfg.d)
        self.halt = nn.Linear(cfg.d, 1)
        nn.init.zeros_(self.halt.weight)
        nn.init.constant_(self.halt.bias, -1.0)     # al inicio, ~27% de probabilidad de parar por paso

    def forward(self, x, n_iters: int | None = None, full_depth: bool = False):
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
            h = self.block(h + self.step_embed.weight[n])
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
            if not self.training and bool((remain == 0).all()):
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

    def forward(self, x):
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
            h = blk(h)
        return h, end, patch_id

    def _global(self, patches: torch.Tensor, memory=None, n_iters=None, full_depth=False):
        g = self.to_global(patches)
        outs = []
        for j, blk in enumerate(self.backbone):
            for key, conn in self.connections.items():
                src, dst = (int(v) for v in key.split("_"))
                if dst == j:
                    g = g + conn(outs[src])
            g = blk(g)
            outs.append(g)
        if self.memory is not None and memory is not None:
            if memory.dim() == 2:                       # [M, d] compartida por todo el lote
                memory = memory.unsqueeze(0).expand(g.shape[0], -1, -1)
            g = self.memory(g, memory)
        g, core_info = self.core(g, n_iters, full_depth)
        balance = g.new_zeros(())
        routes = []
        for region in self.regions:
            r, bal, idx = region(g)
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
            y = blk(y)
        return self.out_norm(y)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None, n_iters: int | None = None,
                full_depth: bool = False):
        """``x``: bytes [B, T]. Devuelve logits del byte siguiente [B, T, 256] y señales auxiliares."""
        h, end, patch_id = self._local_encode(x)
        patches, valid = gather_patches(h, end, patch_id)
        g, core_info, balance, routes = self._global(patches, memory, n_iters, full_depth)
        y = self._decode(h, g, patch_id)
        return {
            "logits": self.head(y),
            "uncertainty": self.uncertainty(y).squeeze(-1),
            "balance": balance,
            "expected_iters": (core_info["expected_iters"] * valid).sum() / valid.sum(),
            "iters_used": core_info["iters_used"],
            "halt_dist": core_info["halt_dist"],
            "patches_per_byte": valid.sum().item() / x.numel(),
            "routes": routes,
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
                 stop_seq: bytes | None = None) -> bytes:
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
                cache_g = self._global(patches[:, :needed], memory)[0]
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
                 stop_seq: bytes | None = None) -> bytes:
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
                                             stop_seq=stop_seq)
        was_training = self.training
        self.eval()
        dev = self.head.weight.device
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
        g_cache = None

        def global_ctx(n_complete: int) -> torch.Tensor:
            nonlocal g_cache
            if n_complete == 0:
                return h.new_zeros(1, 0, self.cfg.d)
            if g_cache is None or g_cache.shape[1] < n_complete:
                g_cache = self._global(torch.stack(patch_states[:n_complete], dim=1), memory)[0]
            return g_cache[:, :n_complete]

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
    def generate_reference(self, prompt: bytes | str, max_new: int = 256, *, stop: bytes = b"\x00",
                           stop_after_line: bytes | None = None, temperature: float = 0.0,
                           memory: torch.Tensor | None = None, max_context: int = 2048,
                 stop_seq: bytes | None = None) -> bytes:
        """Versión simple (recalcula todo en cada byte); se usa para comprobar ``generate``."""
        was_training = self.training
        self.eval()
        dev = self.head.weight.device
        seq = list(prompt.encode("utf-8") if isinstance(prompt, str) else prompt)
        out: list[int] = []
        for _ in range(max_new):
            x = torch.tensor([seq[-max_context:]], device=dev)
            nxt = self._sample(self.forward(x, memory=memory)["logits"][0, -1].float(), temperature)
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

    def loss(self, x: torch.Tensor, memory: torch.Tensor | None = None, *, full_depth: bool = False,
             ponder_scale: float = 1.0) -> tuple[torch.Tensor, dict]:
        """Pérdida de entrenamiento sobre una secuencia de bytes (predice x[1:] desde x[:-1]).

        ``ponder_scale`` permite subir gradualmente el costo de iterar; con
        ``full_depth`` el costo de iterar no se aplica (todas las iteraciones son obligatorias).
        """
        out = self.forward(x[:, :-1], memory, full_depth=full_depth)
        target = x[:, 1:].long()
        ce_each = F.cross_entropy(out["logits"].float().reshape(-1, BYTE_VOCAB), target.reshape(-1),
                                  reduction="none").view_as(target)
        ce = ce_each.mean()
        unc = F.mse_loss(out["uncertainty"].float(), ce_each.detach())
        total = (ce + self.cfg.balance_weight * out["balance"]
                 + (0.0 if full_depth or self.cfg.halting != "act" else self.cfg.ponder_weight * ponder_scale)
                 * out["expected_iters"]
                 + self.cfg.uncertainty_weight * unc)
        stats = {"ce": ce.item(), "bpb": ce.item() / 0.6931471805599453,
                 "balance": out["balance"].item(), "expected_iters": out["expected_iters"].item(),
                 "uncertainty_mse": unc.item(), "patches_per_byte": out["patches_per_byte"]}
        return total, stats


# ---------------------------------------------------------------------------
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


def build_hn1(cfg: HN1Config | str, device: str | torch.device = "cpu") -> tuple[HN1Model, TopologyRegistry]:
    if isinstance(cfg, str):
        cfg = preset(cfg)
    with torch.device(device):
        model = HN1Model(cfg)
    return model, build_registry(model)
