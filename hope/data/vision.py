"""Datos imagen–texto para la zona visual (ARQUITECTURA_HN1 §3.9, §10.1.5).

Tres fuentes, con el mismo formato de anotación (el mismo que pide el maestro):

1. **Imágenes procedurales verificables** (``procedural``): formas para contar y
   ubicar, texto renderizado (lectura/OCR) y gráficos de barras con sus valores.
   Infinitas, gratis en CPU y con respuesta exacta. Sirven para arrancar el
   codificador desde cero y para **medir los errores del maestro**: si Qwen
   describe mal una imagen cuya verdad conocemos, se registra.
2. **Imágenes del usuario anotadas por Qwen** (``hope/teacher/vision.py``).
3. Conjuntos públicos imagen–texto que el usuario agregue (mismo índice JSONL).

Almacenamiento eficiente: ``ImageStore`` guarda cada imagen **una sola vez**,
redimensionada al tamaño de entrenamiento y en JPEG (≈ 15–25 KB a 224 px, unas
10 veces menos que el original y que píxeles crudos), con nombre = hash del
contenido (deduplicación exacta). El índice JSONL apunta a esos archivos.

Formato de anotación (dict)::

    {"tipo": "foto|documento|grafico|diagrama|codigo|formas|texto|otro",
     "descripcion_es": "...", "description_en": "...",
     "texto_visible": "...", "objetos": [{"nombre", "cantidad", "posicion"}],
     "datos": {...},                       # valores extraídos (gráficos, tablas)
     "preguntas": [{"pregunta", "respuesta", "razonamiento"}]}
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

IMG_TAG = "⟦imagen⟧"

COLORS = {"rojo": (220, 40, 40), "verde": (40, 170, 60), "azul": (40, 80, 220),
          "amarillo": (235, 200, 30), "negro": (20, 20, 20), "violeta": (140, 60, 180)}
COLORS_EN = {"rojo": "red", "verde": "green", "azul": "blue", "amarillo": "yellow", "negro": "black",
             "violeta": "purple"}
SHAPES_EN = {"círculo": "circle", "cuadrado": "square", "triángulo": "triangle"}
WORDS = ["hope", "red", "zona", "byte", "parche", "memoria", "maestro", "circuito", "energía", "datos",
         "visión", "señal", "modelo", "tensión", "corriente", "física", "código", "prueba", "lógica", "ruido"]


# ---------------------------------------------------------------------------
# Carga y almacenamiento
# ---------------------------------------------------------------------------
def to_tensor(img, size: int) -> torch.Tensor:
    """PIL / ndarray / bytes / ruta → uint8 [3, size, size] (recorte central tras escalar)."""
    from PIL import Image
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    elif isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img))
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).contiguous()


def dhash(img, size: int = 8) -> int:
    """Hash perceptual por diferencias: imágenes casi iguales dan hashes cercanos."""
    from PIL import Image
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class ImageStore:
    """Carpeta de imágenes listas para entrenar + índice JSONL de muestras."""

    def __init__(self, root: str | Path, size: int = 224, quality: int = 90):
        self.root = Path(root)
        (self.root / "img").mkdir(parents=True, exist_ok=True)
        self.size, self.quality = size, quality
        self.index = self.root / "index.jsonl"

    def put(self, img) -> str:
        """Guarda la imagen (redimensionada) y devuelve su id (sha256 del contenido original)."""
        from PIL import Image
        if isinstance(img, (str, Path)):
            raw = Path(img).read_bytes()
        elif isinstance(img, np.ndarray):
            raw = img.tobytes() + str(img.shape).encode()
        else:
            raw = bytes(img)
        sid = hashlib.sha256(raw).hexdigest()[:24]
        path = self.root / "img" / f"{sid}.jpg"
        if not path.exists():
            t = to_tensor(img if not isinstance(img, (bytes, bytearray)) else raw, self.size)
            Image.fromarray(t.permute(1, 2, 0).numpy()).save(path, quality=self.quality)
        return sid

    def load(self, sid: str) -> torch.Tensor:
        return to_tensor(self.root / "img" / f"{sid}.jpg", self.size)

    def add(self, sid: str, annotation: dict, source: str) -> None:
        with self.index.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": sid, "fuente": source, "anotacion": annotation}, ensure_ascii=False) + "\n")

    def entries(self) -> list[dict]:
        if not self.index.exists():
            return []
        return [json.loads(line) for line in self.index.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Anotación → textos de entrenamiento
# ---------------------------------------------------------------------------
def samples_from_annotation(ann: dict) -> list[tuple[str, str]]:
    """Pares (consigna, respuesta). La pérdida se aplica sobre la respuesta.
    De una sola anotación salen varias tareas: más datos por imagen sin otra consulta."""
    out = []
    if ann.get("descripcion_es"):
        out.append((f"{IMG_TAG}\nDescribí la imagen.\n", ann["descripcion_es"]))
    if ann.get("description_en"):
        out.append((f"{IMG_TAG}\nDescribe the image.\n", ann["description_en"]))
    if ann.get("texto_visible"):
        out.append((f"{IMG_TAG}\nTranscribí el texto de la imagen.\n", ann["texto_visible"]))
    if ann.get("objetos"):
        objs = "; ".join(f"{o.get('cantidad', 1)} {o.get('nombre', '?')}"
                         + (f" ({o['posicion']})" if o.get("posicion") else "") for o in ann["objetos"])
        out.append((f"{IMG_TAG}\n¿Qué objetos hay?\n", objs))
    if ann.get("datos"):
        out.append((f"{IMG_TAG}\nExtraé los datos de la imagen en JSON.\n",
                    json.dumps(ann["datos"], ensure_ascii=False, sort_keys=True)))
    for q in ann.get("preguntas", []):
        ans = q.get("respuesta", "")
        if q.get("razonamiento"):
            ans = f"{q['razonamiento']}\nRespuesta: {ans}"
        out.append((f"{IMG_TAG}\n{q.get('pregunta', '')}\n", ans))
    return out


# ---------------------------------------------------------------------------
# Imágenes procedurales verificables
# ---------------------------------------------------------------------------
def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                     # Pillow antiguo
        return ImageFont.load_default()


def _where(x: float, y: float) -> str:
    v = "arriba" if y < 1 / 3 else ("abajo" if y > 2 / 3 else "centro")
    h = "izquierda" if x < 1 / 3 else ("derecha" if x > 2 / 3 else "centro")
    return "centro" if v == h == "centro" else (v if h == "centro" else (h if v == "centro" else f"{v} a la {h}"))


def shapes(rng: random.Random, size: int = 224) -> tuple[np.ndarray, dict]:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), (250, 250, 245))
    d = ImageDraw.Draw(img)
    kinds = rng.sample(list(SHAPES_EN), rng.randint(1, 3))
    objs, placed = [], []
    for kind in kinds:
        color = rng.choice(list(COLORS))
        n = rng.randint(1, 4)
        pos = []
        for _ in range(n):
            for _try in range(50):
                r = rng.randint(size // 16, size // 9)
                x, y = rng.randint(r, size - r), rng.randint(r, size - r)
                if all((x - a) ** 2 + (y - b) ** 2 > (r + c + 4) ** 2 for a, b, c in placed):
                    break
            placed.append((x, y, r))
            pos.append(_where(x / size, y / size))
            box = (x - r, y - r, x + r, y + r)
            if kind == "círculo":
                d.ellipse(box, fill=COLORS[color])
            elif kind == "cuadrado":
                d.rectangle(box, fill=COLORS[color])
            else:
                d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=COLORS[color])
        objs.append({"forma": kind, "color": color, "cantidad": n, "posicion": ", ".join(sorted(set(pos)))})
    total = sum(o["cantidad"] for o in objs)
    es = "; ".join(f"{o['cantidad']} {o['forma']}{'s' if o['cantidad'] > 1 else ''} {o['color']}"
                   f"{'s' if o['cantidad'] > 1 else ''}" for o in objs)
    en = "; ".join(f"{o['cantidad']} {COLORS_EN[o['color']]} {SHAPES_EN[o['forma']]}"
                   f"{'s' if o['cantidad'] > 1 else ''}" for o in objs)
    target = rng.choice(objs)
    ann = {"tipo": "formas", "descripcion_es": f"Fondo claro con {total} figuras: {es}.",
           "description_en": f"Light background with {total} shapes: {en}.",
           "objetos": [{"nombre": f"{o['forma']} {o['color']}", "cantidad": o["cantidad"],
                        "posicion": o["posicion"]} for o in objs],
           "datos": {"total": total},
           "preguntas": [{"pregunta": f"¿Cuántos {target['forma']}s {target['color']}s hay?",
                          "respuesta": str(target["cantidad"]),
                          "razonamiento": f"Cuento las figuras con forma de {target['forma']} y color "
                                          f"{target['color']}: hay {target['cantidad']}."},
                         {"pregunta": "¿Cuántas figuras hay en total?", "respuesta": str(total),
                          "razonamiento": " + ".join(str(o["cantidad"]) for o in objs) + f" = {total}."}]}
    return np.asarray(img), ann


def text_image(rng: random.Random, size: int = 224) -> tuple[np.ndarray, dict]:
    from PIL import Image, ImageDraw
    bg = rng.choice([(255, 255, 255), (245, 240, 225), (230, 238, 250)])
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    lines = [" ".join(rng.sample(WORDS, rng.randint(1, 3))) for _ in range(rng.randint(1, 3))]
    if rng.random() < 0.4:
        lines.append(f"{rng.randint(1, 999)} + {rng.randint(1, 999)}")
    fs = rng.choice([16, 20, 24])
    y = rng.randint(8, size // 4)
    for line in lines:
        d.text((rng.randint(6, 20), y), line, fill=(20, 20, 20), font=_font(fs))
        y += int(fs * 1.5)
    text = "\n".join(lines)
    qs = []
    if "+" in lines[-1]:
        a, b = (int(v) for v in lines[-1].split("+"))
        qs.append({"pregunta": "¿Cuánto da la cuenta escrita?", "respuesta": str(a + b),
                   "razonamiento": f"Leo «{lines[-1]}»: {a} + {b} = {a + b}."})
    ann = {"tipo": "texto", "texto_visible": text,
           "descripcion_es": f"Texto oscuro sobre fondo claro, {len(lines)} línea(s).",
           "description_en": f"Dark text on a light background, {len(lines)} line(s).",
           "preguntas": qs}
    return np.asarray(img), ann


def bar_chart(rng: random.Random, size: int = 224) -> tuple[np.ndarray, dict]:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    n = rng.randint(2, 5)
    labels = rng.sample(["A", "B", "C", "D", "E", "F"], n)
    vals = [rng.randint(1, 10) for _ in range(n)]
    base, top, left = size - 30, 20, 20
    unit = (base - top) / 10
    bw = (size - 2 * left) // n
    for i, (lab, v) in enumerate(zip(labels, vals)):
        x0 = left + i * bw + bw // 6
        d.rectangle((x0, base - v * unit, x0 + bw * 2 // 3, base), fill=COLORS[rng.choice(list(COLORS))])
        d.text((x0 + bw // 4, base + 6), lab, fill=(0, 0, 0), font=_font(14))
    d.line((left, base, size - left, base), fill=(0, 0, 0), width=2)
    mx = labels[int(np.argmax(vals))]
    data = dict(zip(labels, vals))
    ann = {"tipo": "grafico", "descripcion_es": f"Gráfico de barras con {n} categorías.",
           "description_en": f"Bar chart with {n} categories.", "datos": data,
           "preguntas": [{"pregunta": "¿Qué barra es la más alta?", "respuesta": mx,
                          "razonamiento": "Comparo las alturas: " + ", ".join(f"{k}={v}" for k, v in data.items())
                                          + f"; la mayor es {mx}."},
                         {"pregunta": "¿Cuánto suman todas las barras?", "respuesta": str(sum(vals)),
                          "razonamiento": " + ".join(map(str, vals)) + f" = {sum(vals)}."}]}
    return np.asarray(img), ann


GENERATORS = {"formas": shapes, "texto": text_image, "barras": bar_chart}


def procedural(n: int, seed: int = 0, size: int = 224, kinds: tuple[str, ...] = tuple(GENERATORS)):
    rng = random.Random(seed)
    for _ in range(n):
        yield GENERATORS[rng.choice(kinds)](rng, size)


# ---------------------------------------------------------------------------
# Lotes imagen–texto para el entrenamiento
# ---------------------------------------------------------------------------
@dataclass
class ImageTextItem:
    image: torch.Tensor            # uint8 [3, S, S]
    prompt: str
    answer: str


def encode_pair(prompt: str, answer: str, length: int) -> tuple[np.ndarray, np.ndarray]:
    p, a = prompt.encode("utf-8"), answer.encode("utf-8") + b"\x00"
    raw = (p + a)[:length]
    x = np.zeros(length, dtype=np.int64)
    w = np.zeros(length, dtype=np.float32)
    x[:len(raw)] = list(raw)
    w[len(p):len(raw)] = 1.0                      # la pérdida va sobre la respuesta
    return x, w


class ImageTextSource:
    """Fuente de pares imagen–texto para ``Trainer.train(vision=...)``."""

    def __init__(self, items: list[ImageTextItem], name: str = "vision"):
        if not items:
            raise ValueError("fuente visual vacía")
        self.items, self.name = items, name

    @classmethod
    def from_store(cls, store: ImageStore, name: str = "vision") -> "ImageTextSource":
        items = []
        cache: dict[str, torch.Tensor] = {}
        for e in store.entries():
            img = cache.setdefault(e["id"], store.load(e["id"]))
            items += [ImageTextItem(img, p, a) for p, a in samples_from_annotation(e["anotacion"])]
        return cls(items, name)

    @classmethod
    def from_procedural(cls, n: int, seed: int = 0, size: int = 224, name: str = "procedural") -> "ImageTextSource":
        items = []
        for arr, ann in procedural(n, seed, size):
            img = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()
            items += [ImageTextItem(img, p, a) for p, a in samples_from_annotation(ann)]
        return cls(items, name)

    def batch(self, batch: int, length: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self.items), size=batch)
        imgs, xs, ws = [], [], []
        for i in idx:
            it = self.items[i]
            x, w = encode_pair(it.prompt, it.answer, length)
            imgs.append(it.image)
            xs.append(x)
            ws.append(w)
        return torch.stack(imgs), torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ws))
