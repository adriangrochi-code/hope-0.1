"""Banco visual sellado de Hope (ARQUITECTURA_HN1 §3.10, §12.1; pedido del usuario: «evaluación primero»).

Imágenes procedurales con **respuesta exacta**, generadas con semillas reservadas
(``hope.data.vision.RESERVED_SEEDS``: ``procedural`` se niega a usarlas para entrenar).
El archivo del banco se puede sellar con ``hope.eval.sealed.SealedRegistry``: si alguien
lo cambia, abrirlo falla, y cada lectura queda registrada.

Tareas (métrica):

* ``contar`` — figuras, resistencias, renglones (exacta).
* ``color_forma`` — color de una forma, color de una porción (exacta).
* ``posicion`` — dónde está una figura, cuál está más arriba (exacta).
* ``texto`` — transcribir texto renderizado y OCR con ruido (**exactitud por carácter**,
  1 − distancia de edición / largo).
* ``valores`` — valores de barras, líneas, tortas, celdas de tablas y la hora de un reloj
  (numérica con tolerancia, o exacta).
* ``razonamiento`` — resistencia equivalente, corriente, sumas de columnas, áreas,
  ángulos, relaciones espaciales, sumar minutos (numérica con tolerancia, o exacta).

Dos medidas por tarea:

1. **exactitud**: Hope genera la respuesta (opcionalmente llamando herramientas: ⟦calc⟧,
   ⟦circuito⟧) y se compara la línea «Respuesta: …» con la verdad.
2. **bpb** de la respuesta de referencia con la imagen correcta y con una imagen cambiada
   (de otra pregunta del mismo tipo). ``ventaja_imagen`` = bpb(cambiada) − bpb(correcta):
   cuánto usa la imagen. Es mucho más sensible que la exactitud en modelos chicos.

Corre en CPU con ``hn1-tiny``/``hn1-cpu`` + visión y en GPU con ``hn1-500m``::

    python -m hope.eval.vision_bench construir --out data/vision_bench/v1.pt --size 224 --n 20 \\
        --sellar data/sealed
    python -m hope.eval.vision_bench evaluar --checkpoint runs/x --bench data/vision_bench/v1.pt \\
        --sellado data/sealed --herramientas --device xpu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hope.data.vision import (DOC_KINDS, GENERATORS, IMG_TAG, RESERVED_SEEDS, augment, encode_pair,
                              resize_square, strip_results)

TASKS = ("contar", "color_forma", "posicion", "texto", "valores", "razonamiento")
BENCH_SEED = RESERVED_SEEDS.start + 17
BENCH_VERSION = 1


@dataclass
class BenchItem:
    image: torch.Tensor            # uint8 [3, H, W]
    prompt: str
    answer: str                    # respuesta final exacta
    reference: str                 # razonamiento + «Respuesta: …» (con resultados de herramientas marcados)
    tarea: str
    subtarea: str
    tol_rel: float = 0.0
    tol_abs: float = 0.0
    scale: int = 1                 # resolución relativa (documentos: doc_scale)


def build(n_per_task: int = 30, size: int = 224, seed: int = BENCH_SEED, *, doc_scale: int = 2,
          augment_p: float = 0.3, per_image: int = 2, kinds: tuple[str, ...] = tuple(GENERATORS),
          max_images: int = 20000) -> list[BenchItem]:
    """Arma el banco con ``n_per_task`` preguntas por tarea. Recorre los generadores por turno
    (así cada tarea mezcla tipos de imagen) y toma hasta ``per_image`` preguntas por imagen,
    priorizando las tareas con menos preguntas. Los documentos van a ``size · doc_scale``."""
    if seed not in RESERVED_SEEDS:
        raise ValueError("el banco debe usar una semilla reservada (no compartida con el entrenamiento)")
    rng = random.Random(seed)
    items: list[BenchItem] = []
    count = dict.fromkeys(TASKS, 0)
    for k in range(max_images):
        if min(count.values()) >= n_per_task:
            break
        kind = kinds[k % len(kinds)]
        arr, ann = GENERATORS[kind](rng, size * doc_scale if kind in DOC_KINDS else size)
        cands = [q for q in ann.get("preguntas", []) if q.get("tarea") in TASKS]
        if ann.get("texto_visible") and kind in ("texto", "ocr"):
            cands.append({"pregunta": "Transcribí el texto de la imagen.", "respuesta": ann["texto_visible"],
                          "tarea": "texto", "subtarea": f"{kind}_transcribir", "_transcribir": True})
        cands = [q for q in cands if count[q["tarea"]] < n_per_task]
        if not cands:
            continue
        rng.shuffle(cands)
        cands.sort(key=lambda q: count[q["tarea"]])
        picked, seen = [], set()
        for q in cands:
            if q["tarea"] not in seen and len(picked) < per_image:
                picked.append(q)
                seen.add(q["tarea"])
        if augment_p and rng.random() < augment_p:
            arr = augment(arr, rng)
        img = torch.from_numpy(np.array(arr, dtype=np.uint8)).permute(2, 0, 1).contiguous()
        for q in picked:
            if q.get("_transcribir"):
                ref = q["respuesta"]
            else:
                ref = (f"{q['razonamiento']}\nRespuesta: {q['respuesta']}" if q.get("razonamiento")
                       else q["respuesta"])
            items.append(BenchItem(img, f"{IMG_TAG}\n{q['pregunta']}\n", str(q["respuesta"]), ref,
                                   q["tarea"], q.get("subtarea", kind), float(q.get("tolerancia", 0.0)),
                                   float(q.get("tolerancia_abs", 0.0)), doc_scale if kind in DOC_KINDS else 1))
            count[q["tarea"]] += 1
    return items


def save(items: list[BenchItem], path: str | Path, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": BENCH_VERSION, "meta": meta or {},
                "items": [{**{k: v for k, v in asdict(it).items() if k != "image"}, "image": it.image}
                          for it in items]}, path)
    return path


def load(path: str | Path, *, registry=None, name: str | None = None, requester: str = "vision_bench",
         decision: bool = False) -> list[BenchItem]:
    """Carga el banco. Con ``registry`` (``SealedRegistry``) y ``name`` verifica el sello y
    registra la lectura (falla si el archivo cambió)."""
    if registry is not None:
        path = registry.open(name or Path(path).stem, purpose="evaluation", requester=requester, decision=decision)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return [BenchItem(**it) for it in blob["items"]]


def image_digest(img: torch.Tensor) -> str:
    return hashlib.sha256(img.contiguous().numpy().tobytes() + str(tuple(img.shape)).encode()).hexdigest()


def contamination(items: list[BenchItem], train_images) -> int:
    """Cuántas imágenes del banco aparecen tal cual entre las de entrenamiento."""
    bench = {image_digest(it.image) for it in items}
    return sum(1 for img in train_images if image_digest(img) in bench)


# ---------------------------------------------------------------------------
# Puntaje
# ---------------------------------------------------------------------------
_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def _norm(s: str) -> str:
    s = s.strip().lower().replace("°", "").replace("%", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .;:«»\"'")


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(pred: str, ref: str) -> float:
    pred, ref = pred.strip(), ref.strip()
    if not ref:
        return float(not pred)
    return max(0.0, 1.0 - edit_distance(pred, ref) / len(ref))


def extract_answer(text: str, transcription: bool = False) -> str:
    text = strip_results(text).split("\x00")[0]
    if transcription:
        return text.strip()
    if "Respuesta:" in text:
        tail = text.rsplit("Respuesta:", 1)[1].strip()
        return tail.splitlines()[0] if tail else ""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""


def score(item: BenchItem, pred: str) -> float:
    if item.tarea == "texto":
        return char_accuracy(pred, item.answer)
    ref = item.answer.strip()
    if _NUM.fullmatch(ref):
        m = _NUM.search(pred)
        if not m:
            return 0.0
        p, r = float(m.group(0).replace(",", ".")), float(ref.replace(",", "."))
        tol = max(item.tol_abs, item.tol_rel * abs(r), 1e-9)
        return float(abs(p - r) <= tol)
    return float(_norm(pred) == _norm(ref))


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def _fit(model, item: BenchItem) -> torch.Tensor:
    """Adapta la imagen del banco al modelo: con mosaicos, los documentos quedan en alta
    resolución (``scale · vis_image``); sin mosaicos, todo se reduce a ``vis_image``."""
    s = model.cfg.vis_image
    side = s * item.scale if model.cfg.vis_tiling and item.scale > 1 else s
    return resize_square(item.image, side)


@torch.no_grad()
def answer_bpb(model, items: list[BenchItem], *, seq: int = 512, batch: int = 16, shuffle_images: bool = False,
               device: str | None = None) -> list[float]:
    """Bits por byte de la respuesta de referencia de cada pregunta (forzado por el maestro).
    ``shuffle_images``: cada pregunta ve la imagen de otra pregunta del mismo tamaño."""
    device = device or next(model.parameters()).device
    out = [float("nan")] * len(items)
    imgs = [_fit(model, it) for it in items]
    groups: dict[tuple, list[int]] = {}
    for i, im in enumerate(imgs):
        groups.setdefault(tuple(im.shape), []).append(i)
    for idx in groups.values():
        src = idx[1:] + idx[:1] if shuffle_images and len(idx) > 1 else idx
        for k in range(0, len(idx), batch):
            ii, ss = idx[k:k + batch], src[k:k + batch]
            xs, ws = zip(*(encode_pair(items[i].prompt, items[i].reference, seq + 1) for i in ii))
            x = torch.from_numpy(np.stack(xs)).to(device)
            w = torch.from_numpy(np.stack(ws)).to(device)
            im = torch.stack([imgs[j] for j in ss]).to(device)
            logits = model(x[:, :-1], image=im)["logits"].float()
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1),
                                 reduction="none").view(x.shape[0], -1)
            wt = w[:, 1:]
            per = (ce * wt).sum(1) / wt.sum(1).clamp_min(1.0) / 0.6931471805599453
            for j, i in enumerate(ii):
                out[i] = per[j].item()
    return out


@torch.no_grad()
def evaluate(model, items: list[BenchItem], *, tools: bool = False, max_new: int = 220, generate: bool = True,
             teacher_forced: bool = True, limit: int | None = None, log=None) -> dict:
    """Evalúa ``model`` (HN1 con zona visual) en el banco. Devuelve métricas por tarea y subtarea."""
    model.eval()
    items = items[:limit] if limit else items
    t0 = time.time()
    runner = None
    if tools:
        from hope.runtime.toolcalls import ToolRunner, generate_with_tools
        from hope.tools.builtin import default_catalog
        runner = ToolRunner(default_catalog())
    scores: list[float] = [float("nan")] * len(items)
    preds: list[str] = [""] * len(items)
    if generate:
        for i, it in enumerate(items):
            vis = model.encode_image(_fit(model, it)[None])
            if runner is not None:
                text = generate_with_tools(model, it.prompt, runner, max_new=max_new, stop_after_line=None, vis=vis)
            else:
                text = model.generate(it.prompt, max_new=max_new, vis=vis).decode("utf-8", errors="replace")
            preds[i] = text
            scores[i] = score(it, extract_answer(text, transcription=it.subtarea.endswith("_transcribir")))
            if log and i % 50 == 0:
                log(f"{i}/{len(items)}")
    right = wrong = [float("nan")] * len(items)
    if teacher_forced:
        right = answer_bpb(model, items)
        wrong = answer_bpb(model, items, shuffle_images=True)

    def agg(keys):
        res = {}
        for key in sorted(set(keys)):
            sel = [i for i, k in enumerate(keys) if k == key]
            r = float(np.nanmean([right[i] for i in sel])) if teacher_forced else None
            wv = float(np.nanmean([wrong[i] for i in sel])) if teacher_forced else None
            res[key] = {"n": len(sel),
                        "exactitud": float(np.mean([scores[i] for i in sel])) if generate else None,
                        "bpb": r, "bpb_imagen_cambiada": wv,
                        "ventaja_imagen": (wv - r) if teacher_forced else None}
        return res

    tareas = agg([it.tarea for it in items])
    report = {"tareas": tareas, "subtareas": agg([it.subtarea for it in items]),
              "promedio_exactitud": float(np.mean([v["exactitud"] for v in tareas.values()])) if generate else None,
              "promedio_ventaja_imagen": float(np.mean([v["ventaja_imagen"] for v in tareas.values()]))
              if teacher_forced else None,
              "preguntas": len(items), "herramientas": tools, "segundos": time.time() - t0,
              "ejemplos": [{"pregunta": it.prompt.split("\n")[1], "verdad": it.answer,
                            "hope": strip_results(preds[i])[:200]} for i, it in enumerate(items[:12])]}
    return report


def table(report: dict) -> str:
    lines = ["| tarea | n | exactitud | bpb respuesta | bpb con imagen cambiada | ventaja de la imagen |",
             "|---|---|---|---|---|---|"]
    f = (lambda v: "—" if v is None else f"{v:.3f}")
    for k, v in report["tareas"].items():
        lines.append(f"| {k} | {v['n']} | {f(v['exactitud'])} | {f(v['bpb'])} | {f(v['bpb_imagen_cambiada'])} | "
                     f"{f(v['ventaja_imagen'])} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict | None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("construir", help="genera (y opcionalmente sella) el banco")
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=40, help="preguntas por tarea")
    b.add_argument("--size", type=int, default=224)
    b.add_argument("--doc-scale", type=int, default=2)
    b.add_argument("--sellar", default=None, help="carpeta del registro de conjuntos sellados")
    b.add_argument("--nombre", default=None)
    e = sub.add_parser("evaluar", help="evalúa un checkpoint")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--bench", default=None, help="archivo del banco; sin esto se arma uno en memoria")
    e.add_argument("--sellado", default=None, help="registro sellado (verifica el hash y registra la lectura)")
    e.add_argument("--nombre", default=None)
    e.add_argument("--n", type=int, default=30, help="preguntas por tarea (si no se da --bench)")
    e.add_argument("--herramientas", action="store_true", help="ejecuta ⟦calc⟧, ⟦circuito⟧… durante la generación")
    e.add_argument("--max-new", type=int, default=220)
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--device", default="cpu")
    e.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "construir":
        items = build(args.n, args.size, doc_scale=args.doc_scale)
        path = save(items, args.out, {"n_por_generador": args.n, "size": args.size, "doc_scale": args.doc_scale})
        print(f"{len(items)} preguntas → {path}")
        if args.sellar:
            from hope.eval.sealed import SealedRegistry
            SealedRegistry(args.sellar).seal(args.nombre or path.stem, path,
                                             description="banco visual procedural (hope.eval.vision_bench)")
            print(f"sellado como «{args.nombre or path.stem}»")
        return None
    from hope.train.continual import TrainConfig, Trainer
    model = Trainer.resume(args.checkpoint, TrainConfig(precision="fp32"), device=args.device).model
    if args.bench:
        reg = None
        if args.sellado:
            from hope.eval.sealed import SealedRegistry
            reg = SealedRegistry(args.sellado)
        items = load(args.bench, registry=reg, name=args.nombre)
    else:
        items = build(args.n, model.cfg.vis_image)
    rep = evaluate(model, items, tools=args.herramientas, max_new=args.max_new, limit=args.limit, log=print)
    print(table(rep))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep


if __name__ == "__main__":
    main()
