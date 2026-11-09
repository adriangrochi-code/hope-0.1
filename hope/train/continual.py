"""Entrenamiento continuo de HN-1 (HLA §10; AMC §8; ASG §2.3).

A diferencia de ``hope.train.lm`` (preentrenamiento desde cero), acá Hope:

* **retoma** desde un checkpoint, con el estado del optimizador y el paso;
* entrena con una **mezcla de fuentes con pesos**: corpus, episodios del maestro,
  repaso de datos anteriores (replay selectivo contra el olvido);
* mide **BPB por fuente** antes y después de cada etapa, para detectar olvido;
* registra cada etapa como una **versión** en el linaje (rama experimental),
  con sus métricas, para poder volver atrás.

Uso típico (consola):

    python -m hope.train.continual --resume runs/hn1_25m --out runs/hn1_25m_etapa2 \\
        --source nuevo=data/libros:0.6 --source repaso=data/f1:0.4 --steps 5000

Los episodios del maestro llegan por el mismo camino: ``hope.teacher.cycle``
los escribe como fuente de bytes y llama a ``Trainer.train``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from hope.core.baseline import LN2
from hope.core.hn1 import HN1Config, HN1Model, build_registry, preset
from hope.data.corpus import ByteData
from hope.device import select_device
from hope.train.lowprec import prepare


@dataclass
class Source:
    """Una fuente de bytes con su peso en la mezcla."""

    name: str
    data: ByteData | np.ndarray
    weight: float

    def sample(self, batch: int, length: int, rng: np.random.Generator) -> torch.Tensor:
        arr = self.data.data if isinstance(self.data, ByteData) else self.data
        length = min(length, len(arr) - 1)
        starts = rng.integers(0, max(len(arr) - length, 1), size=batch)
        return torch.from_numpy(np.stack([arr[s:s + length] for s in starts]).astype(np.int64))

    @staticmethod
    def from_texts(name: str, texts: list[str], weight: float) -> "Source":
        """Fuente en memoria a partir de textos (p. ej. episodios del maestro)."""
        raw = b"\x00".join(t.encode("utf-8") for t in texts) + b"\x00"
        return Source(name, np.frombuffer(raw, dtype=np.uint8).copy(), weight)


class Mixture:
    def __init__(self, sources: list[Source]):
        if not sources:
            raise ValueError("la mezcla necesita al menos una fuente")
        total = sum(s.weight for s in sources)
        self.sources = sources
        self.probs = np.array([s.weight / total for s in sources])

    def batch(self, batch: int, length: int, rng: np.random.Generator) -> tuple[torch.Tensor, list[str]]:
        picks = rng.choice(len(self.sources), size=batch, p=self.probs)
        rows, names = [], []
        for i in picks:
            rows.append(self.sources[i].sample(1, length, rng))
            names.append(self.sources[i].name)
        width = min(r.shape[1] for r in rows)
        return torch.cat([r[:, :width] for r in rows]), names


@dataclass
class TrainConfig:
    lr: float = 3e-4
    warmup: int = 100
    batch: int = 16
    seq_bytes: int = 1024
    clip: float = 1.0
    weight_decay: float = 0.1
    full_depth_frac: float | None = None     # None = el de la config del modelo
    precision: str = "bf16-8bit"             # base en baja precisión (ver hope/train/lowprec.py)
    # Programa de aprendizaje. "wsd" (warmup–stable–decay; Hägele et al. 2024): tasa constante
    # mientras dure el entrenamiento y enfriamiento corto solo al final (``Trainer.anneal``),
    # con los mejores datos. Sirve para entrenar meses sin fijar la duración de antemano:
    # se puede enfriar una copia en cualquier momento y seguir entrenando la rama estable.
    schedule: str = "cosine"                 # "cosine" | "wsd"
    optimizer: str = "adamw"                 # "adamw" | "muon" (hope/train/muon.py; ablación)
    grad_accum: int = 1                      # lotes por paso del optimizador (lote efectivo = batch × grad_accum)
    checkpointing: bool = False              # recalcular activaciones (hope/core/fast.py)
    segment_bytes: int | None = None         # contexto largo: secuencias de seq_bytes leídas por segmentos
    offload_optimizer: bool = False          # estado del optimizador en la RAM de la CPU (GPU de 6 GB)
    qat_bits: tuple[int, ...] | None = None  # QAT (hope/train/qat.py): p. ej. (4, 8) = Matryoshka 4/8 bits
    qat_group: int = 64
    seed: int = 0


@dataclass
class StageReport:
    steps: int
    elapsed_s: float
    bpb_before: dict[str, float]
    bpb_after: dict[str, float]
    train_bpb_last: float
    forgetting: dict[str, float] = field(default_factory=dict)   # sube el BPB = olvido


class Trainer:
    def __init__(self, model: HN1Model, cfg: TrainConfig = TrainConfig(), device: str | None = None):
        self.device = select_device(device)
        self.model = model.to(self.device)
        self.cfg = cfg
        self.step = 0
        self.model.grad_checkpointing = cfg.checkpointing
        self.opt = prepare(self.model, cfg.precision, lr=cfg.lr, weight_decay=cfg.weight_decay, seed=cfg.seed,
                           optimizer=cfg.optimizer, offload=cfg.offload_optimizer)
        self.penalties: list = []           # p. ej. ConsolidationPenalty (protección de lo consolidado)
        self.hooks: list = []               # se llaman después de cada paso (p. ej. hiperplasticidad)
        self.rng = np.random.default_rng(cfg.seed)
        # autocast solo hace falta con pesos fp32 en GPU; en modos bf16 los pesos ya son bf16.
        from hope.device import supports_bf16
        self.use_bf16 = cfg.precision in ("fp32", "fp32-8bit") and supports_bf16(self.device)

    # --- checkpoints --------------------------------------------------------
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # La configuración no alcanza para reconstruir un modelo que creció (especialistas de
        # distinto ancho, zona visual, bloques agregados): se guarda también la estructura.
        with torch.device("meta"):
            fresh = HN1Model(HN1Config.from_dict(self.model.cfg.to_dict()))
        grown = sum(p.numel() for p in fresh.parameters()) != sum(p.numel() for p in self.model.parameters())
        from hope.core.fast import eager
        with eager(self.model):
            self._save(path, grown, extra)

    def _save(self, path: Path, grown: bool, extra: dict | None) -> None:
        torch.save({"state_dict": self.model.state_dict(), "optimizer": self.opt.state_dict(),
                    "structure": self.model if grown else None,
                    "step": self.step, "rng": self.rng.bit_generator.state,
                    "summary": {"config": self.model.cfg.to_dict(), "arch": "hn1",
                                "precision": self.cfg.precision, **(extra or {})}},
                   path / "model.pt")

    @classmethod
    def resume(cls, path: str | Path, cfg: TrainConfig = TrainConfig(), device: str | None = None) -> "Trainer":
        """Retoma desde un checkpoint de ``hope.train.lm`` o de una etapa anterior."""
        ckpt = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=False)
        model = ckpt.get("structure") or HN1Model(HN1Config.from_dict(ckpt["summary"]["config"]))
        model.load_state_dict(ckpt["state_dict"])
        tr = cls(model, cfg, device)
        if "optimizer" in ckpt:
            try:
                tr.opt.load_state_dict(ckpt["optimizer"])
                for g in tr.opt.param_groups:
                    g["lr"] = cfg.lr
            except ValueError:
                pass            # la estructura cambió (crecimiento): el optimizador arranca de nuevo
        tr.step = ckpt.get("step", 0)
        if "rng" in ckpt:
            tr.rng.bit_generator.state = ckpt["rng"]
        return tr

    # --- evaluación ------------------------------------------------------------
    def _amp(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16)

    @torch.no_grad()
    def bpb(self, data: ByteData | np.ndarray | Source, windows: int = 16, seq: int | None = None) -> float:
        """BPB promedio sobre ventanas consecutivas fijas (siempre las mismas)."""
        arr = data.data if isinstance(data, (ByteData, Source)) else data
        if isinstance(arr, ByteData):
            arr = arr.data
        seq = (seq or self.cfg.seq_bytes) + 1
        self.model.eval()
        n = max(1, min(windows, (len(arr) - 1) // seq))
        total, count = 0.0, 0
        for i in range(n):
            w = torch.from_numpy(np.asarray(arr[i * seq:(i + 1) * seq], dtype=np.int64)).unsqueeze(0)
            if w.shape[1] < 2:
                continue
            with self._amp():
                _, st = self.model.loss(w.to(self.device))
            total += st["ce"] * (w.shape[1] - 1)
            count += w.shape[1] - 1
        return total / max(count, 1) / LN2

    @torch.no_grad()
    def text_loss(self, texts: list[str]) -> float:
        """BPB del modelo sobre textos concretos (p. ej. una explicación del maestro)."""
        return self.bpb(Source.from_texts("t", texts, 1.0), windows=64)

    # --- entrenamiento -----------------------------------------------------------
    def set_plasticity(self, registry, policy) -> list[float]:
        """Reconstruye el optimizador con una velocidad de aprendizaje por región."""
        from hope.plasticity.rates import param_groups
        groups = param_groups(self.model, registry, policy)
        self.opt = prepare(self.model, self.cfg.precision, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                           seed=self.cfg.seed, param_groups=groups, optimizer=self.cfg.optimizer, offload=self.cfg.offload_optimizer)
        return [g["lr_scale"] for g in groups]

    def refresh_optimizer(self) -> None:
        """Después de enganchar o quitar adaptadores (o de un cambio estructural)."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = prepare(self.model, self.cfg.precision, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                           seed=self.cfg.seed, param_groups=[{"params": params, "lr_scale": 1.0}],
                           optimizer=self.cfg.optimizer, offload=self.cfg.offload_optimizer)

    def _lr(self, i: int, total: int, decay: bool = False,
            decay_span: tuple[float, float] = (0.0, 1.0)) -> float:
        if self.cfg.schedule == "wsd":
            if self.step < self.cfg.warmup:              # el calentamiento cuenta pasos globales
                return self.cfg.lr * (self.step + 1) / self.cfg.warmup
            if not decay:
                return self.cfg.lr
            # ``decay_span``: qué parte del enfriamiento cubre esta llamada (p. ej. un tramo de
            # 30 min de una etapa de varias horas); por defecto, todo el enfriamiento.
            a, b = decay_span
            p = a + (b - a) * (i + 1) / max(total, 1)
            return self.cfg.lr * max(0.0, 1 - math.sqrt(min(max(p, 0.0), 1.0)))   # enfriamiento 1 − √p
        if i < self.cfg.warmup:
            return self.cfg.lr * (i + 1) / self.cfg.warmup
        p = (i - self.cfg.warmup) / max(total - self.cfg.warmup, 1)
        return self.cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    def train(self, mixture: Mixture, steps: int, *, eval_sources: list[Source] | None = None,
              max_minutes: float | None = None, log=None, vision=None, vision_frac: float = 0.0,
              vision_batch: int | None = None, vision_seq: int = 512, decay: bool = False,
              decay_span: tuple[float, float] = (0.0, 1.0), vision_contrastive: float = 0.0) -> StageReport:
        """``vision``: fuente imagen–texto (``hope.data.vision.ImageTextSource``). En una fracción
        ``vision_frac`` de los pasos se entrena con imágenes; el resto, con texto (así el texto
        hace de repaso y no se olvida mientras aprende a ver). ``vision_contrastive``: peso de la
        pérdida auxiliar imagen–texto estilo SigLIP en los pasos con imagen (0 = apagada)."""
        mcfg = self.model.cfg
        full_frac = mcfg.full_depth_frac if self.cfg.full_depth_frac is None else self.cfg.full_depth_frac
        evals = eval_sources if eval_sources is not None else mixture.sources
        before = {s.name: self.bpb(s) for s in evals}
        t0 = time.time()
        last = float("nan")
        done = 0
        for i in range(steps):
            if max_minutes and time.time() - t0 > max_minutes * 60:
                break
            for g in self.opt.param_groups:
                g["lr"] = self._lr(i, steps, decay, decay_span) * g.get("lr_scale", 1.0)
            self.model.train()
            if self.cfg.qat_bits:
                from hope.train.qat import enable_qat, set_qat_bits
                if not getattr(self, "_qat_on", False):
                    enable_qat(self.model, self.cfg.qat_bits[0], self.cfg.qat_group)
                    self._qat_on = True
                set_qat_bits(self.model, int(self.rng.choice(self.cfg.qat_bits)))   # Matryoshka: una por paso
            elif getattr(self, "_qat_on", False):
                # Una etapa sin QAT después de una con QAT (p. ej. con ``run.py`` el Trainer sigue entre
                # etapas): se apaga la cuantización falsa en lugar de seguir con la última precisión.
                from hope.train.qat import set_qat_bits
                set_qat_bits(self.model, None)
                self._qat_on = False
            self.opt.zero_grad(set_to_none=True)
            accum = max(1, self.cfg.grad_accum)
            for _micro in range(accum):
                img = w = None
                if vision is not None and self.rng.random() < vision_frac:
                    img, x, w = vision.batch(vision_batch or self.cfg.batch, vision_seq + 1, self.rng)
                    img, w = img.to(self.device), w.to(self.device)
                else:
                    x, _ = mixture.batch(self.cfg.batch, self.cfg.seq_bytes + 1, self.rng)
                full = self.rng.random() < full_frac
                with self._amp():
                    ps = min(1.0, self.step / max(mcfg.ponder_warmup, 1))
                    seg = self.cfg.segment_bytes
                    if seg and img is None and x.shape[1] > seg + 1:
                        loss, st = self.model.long_loss(x.to(self.device), seg, full_depth=full, ponder_scale=ps)
                    else:
                        loss, st = self.model.loss(x.to(self.device), full_depth=full, ponder_scale=ps,
                                                   image=img, weights=w,
                                                   contrastive=vision_contrastive if img is not None else 0.0)
                    for pen in self.penalties:
                        loss = loss + pen(self.model).to(loss.device)
                (loss / accum).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip)
            self.opt.step()
            for hook in self.hooks:
                hook(self)
            self.step += 1
            done += 1
            last = st["bpb"]
            if log and i % 100 == 0:
                log(f"paso {self.step}  bpb {last:.3f}")
        after = {s.name: self.bpb(s) for s in evals}
        return StageReport(done, time.time() - t0, before, after, last,
                           {k: after[k] - before[k] for k in before})


def anneal(trainer: "Trainer", best: Mixture, steps: int, **kw) -> StageReport:
    """Enfriamiento final de WSD con los mejores datos (los de nota más alta del filtro de
    calidad, explicaciones de Qwen verificadas y un poco de repaso). Conviene guardar antes
    el checkpoint de la rama estable para poder seguir entrenándola después."""
    return trainer.train(best, steps, decay=True, **kw)


Trainer.anneal = anneal


def _parse_source(spec: str) -> Source:
    """``nombre=carpeta:peso`` → fuente con ``carpeta/train.bin``."""
    name, rest = spec.split("=", 1)
    folder, weight = rest.rsplit(":", 1)
    return Source(name, ByteData(Path(folder) / "train.bin"), float(weight))


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", default=None, help="carpeta con model.pt; sin esto, modelo nuevo")
    ap.add_argument("--preset", default="hn1-25m")
    ap.add_argument("--source", action="append", required=True, help="nombre=carpeta:peso (repetible)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-bytes", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", default="bf16-8bit", choices=["fp32", "bf16", "bf16-8bit", "fp32-8bit"])
    ap.add_argument("--lineage", default=None, help="almacén de linaje donde registrar la etapa")
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"])
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--decay", action="store_true", help="con wsd: esta etapa es el enfriamiento final")
    ap.add_argument("--compile", action="store_true", help="torch.compile de los bloques (hope/core/fast.py)")
    args = ap.parse_args(argv)

    tcfg = TrainConfig(lr=args.lr, batch=args.batch, seq_bytes=args.seq_bytes, precision=args.precision,
                       schedule=args.schedule, optimizer=args.optimizer)
    trainer = (Trainer.resume(args.resume, tcfg, args.device) if args.resume
               else Trainer(HN1Model(preset(args.preset)), tcfg, args.device))
    if args.compile:
        from hope.core.fast import compile_model
        compile_model(trainer.model)
    sources = [_parse_source(s) for s in args.source]
    val_sources = [Source(s.name, ByteData(Path(spec.split("=", 1)[1].rsplit(":", 1)[0]) / "val.bin"), 1.0)
                   for s, spec in zip(sources, args.source)]
    report = trainer.train(Mixture(sources), args.steps, eval_sources=val_sources,
                           max_minutes=args.max_minutes, log=print, decay=args.decay)
    result = {"steps_total": trainer.step, "stage": report.__dict__,
              "sources": {s.name: s.weight for s in sources}}
    trainer.save(args.out, extra={"continual": result})
    if args.lineage:
        from hope.lineage.store import STABLE, LineageStore
        store = LineageStore(args.lineage)
        branch = "experimental/continual"
        if branch not in store.branches():
            if not store.branches():
                store.commit(trainer.model, build_registry(trainer.model), branch=STABLE, message="base")
            store.create_branch(branch, store.head(STABLE))
        result["version"] = store.commit(trainer.model, build_registry(trainer.model), branch=branch,
                                         message=f"etapa continua ({report.steps} pasos)", meta=result)
    (Path(args.out) / "stage.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(report.__dict__, indent=2))
    return result


if __name__ == "__main__":
    main()
