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
def patch_boundaries(x: torch.Tensor, max_patch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Devuelve (``end``: el byte cierra un parche, ``patch_id`` de cada byte).

    Un parche termina en un byte de frontera (espacio, puntuación) o al llegar
    a ``max_patch`` bytes. Vectorizado, sin bucles sobre la secuencia.
    """
    b, t = x.shape
    table = torch.zeros(BYTE_VOCAB, dtype=torch.bool, device=x.device)
    table[torch.tensor(BOUNDARY_BYTES, device=x.device)] = True
    natural = table[x.long()]
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
class RecurrentCore(nn.Module):
    def __init__(self, cfg: HN1Config):
        super().__init__()
        self.max_iters, self.eps = cfg.max_iters, cfg.halt_eps
        self.block = Block(cfg.d, cfg.core_kind, cfg.heads, cfg.mlp_hidden)
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
        self.encoder = nn.ModuleList(Block(dl, "attn", cfg.local_heads, cfg.local_mlp) for _ in range(cfg.enc_layers))
        self.to_global = nn.Linear(dl, d)
        self.backbone = nn.ModuleList(Block(d, k, cfg.heads, cfg.mlp_hidden, hnhs=cfg.hnhs) for k in cfg.pattern)
        self.memory = MemoryCrossAttention(d, cfg.heads) if cfg.memory else None
        self.core = RecurrentCore(cfg)
        self.regions = nn.ModuleList(Zone(d, cfg.zone_kinds, cfg.expert_hidden, cfg.top_k)
                                     for _ in range(cfg.n_regions))
        self.global_norm = RMSNorm(d)
        self.to_local = nn.Linear(d, dl)
        self.decoder = nn.ModuleList(Block(dl, "attn", cfg.local_heads, cfg.local_mlp) for _ in range(cfg.dec_layers))
        self.out_norm = RMSNorm(dl)
        self.head = nn.Linear(dl, BYTE_VOCAB, bias=False)
        self.uncertainty = nn.Linear(dl, 1)          # predice la pérdida propia por byte (señal para UAC)

    @property
    def d(self) -> int:
        return self.cfg.d

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None, n_iters: int | None = None,
                full_depth: bool = False):
        """``x``: bytes [B, T]. Devuelve logits del byte siguiente [B, T, 256] y señales auxiliares."""
        end, patch_id = patch_boundaries(x, self.cfg.max_patch)
        h = self.byte_embed(x.long())
        for blk in self.encoder:
            h = blk(h)
        patches, valid = gather_patches(h, end, patch_id)

        g = self.to_global(patches)
        for blk in self.backbone:
            g = blk(g)
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
        g = self.global_norm(g)

        # El byte del parche p recibe el contexto global del parche p - 1 (ya completo).
        ctx = torch.cat([torch.zeros_like(g[:, :1]), g[:, :-1]], dim=1)
        ctx = self.to_local(ctx).gather(1, patch_id.unsqueeze(-1).expand(-1, -1, self.cfg.d_local))
        y = h + ctx
        for blk in self.decoder:
            y = blk(y)
        y = self.out_norm(y)
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

    @torch.no_grad()
    def generate(self, prompt: bytes | str, max_new: int = 256, *, stop: bytes = b"\x00",
                 stop_after_line: bytes | None = None, temperature: float = 0.0,
                 memory: torch.Tensor | None = None, max_context: int = 2048) -> bytes:
        """Genera bytes a continuación de ``prompt``.

        Para cuando aparece ``stop`` o cuando termina una línea que empieza con
        ``stop_after_line`` (p. ej. b"Resultado:"). v0 recalcula todo en cada paso
        (sin caché): es lento pero simple; la caché llega con el harness (F9).
        """
        was_training = self.training
        self.eval()
        dev = self.head.weight.device
        seq = list(prompt.encode("utf-8") if isinstance(prompt, str) else prompt)
        out: list[int] = []
        for _ in range(max_new):
            x = torch.tensor([seq[-max_context:]], device=dev)
            logits = self.forward(x, memory=memory)["logits"][0, -1].float()
            if temperature > 0:
                nxt = int(torch.multinomial(torch.softmax(logits / temperature, -1), 1))
            else:
                nxt = int(logits.argmax())
            seq.append(nxt)
            out.append(nxt)
            if bytes([nxt]) == stop:
                out.pop()
                break
            if stop_after_line and nxt == 10:
                lines = bytes(out).split(b"\n")
                if len(lines) >= 2 and lines[-2].lstrip().startswith(stop_after_line):
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
                 + (0.0 if full_depth else self.cfg.ponder_weight * ponder_scale) * out["expected_iters"]
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
