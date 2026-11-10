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
import re
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
# Fuentes TrueType comunes en Linux y Windows. Si no hay ninguna, se usa la de Pillow:
# los generadores funcionan igual, con menos variedad tipográfica.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf", "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/verdana.ttf", "C:/Windows/Fonts/georgia.ttf",
)
_FONT_FILES: list[str] | None = None


def font_files() -> list[str]:
    global _FONT_FILES
    if _FONT_FILES is None:
        _FONT_FILES = [f for f in FONT_CANDIDATES if Path(f).exists()]
    return _FONT_FILES


def _font(size: int, rng: random.Random | None = None):
    """Fuente de Pillow por defecto; con ``rng``, una TrueType al azar entre las instaladas."""
    from PIL import ImageFont
    files = font_files() if rng is not None else []
    if files:
        return ImageFont.truetype(rng.choice(files), size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                     # Pillow antiguo
        return ImageFont.load_default()


def _pl(word: str) -> str:
    """Plural español simple: «azul» → «azules», «rojo» → «rojos»."""
    return word + ("es" if word[-1] in "lnrdzj" else "s")


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
    es = "; ".join(f"{o['cantidad']} {_pl(o['forma']) if o['cantidad'] > 1 else o['forma']} "
                   f"{_pl(o['color']) if o['cantidad'] > 1 else o['color']}" for o in objs)
    en = "; ".join(f"{o['cantidad']} {COLORS_EN[o['color']]} {SHAPES_EN[o['forma']]}"
                   f"{'s' if o['cantidad'] > 1 else ''}" for o in objs)
    target = rng.choice(objs)
    ann = {"tipo": "formas", "descripcion_es": f"Fondo claro con {total} figuras: {es}.",
           "description_en": f"Light background with {total} shapes: {en}.",
           "objetos": [{"nombre": f"{o['forma']} {o['color']}", "cantidad": o["cantidad"],
                        "posicion": o["posicion"]} for o in objs],
           "datos": {"total": total},
           "preguntas": [{"pregunta": f"¿Cuántos {_pl(target['forma'])} {_pl(target['color'])} hay?",
                          "respuesta": str(target["cantidad"]),
                          "razonamiento": f"Cuento las figuras con forma de {target['forma']} y color "
                                          f"{target['color']}: hay {target['cantidad']}."},
                         {"pregunta": "¿Cuántas figuras hay en total?", "respuesta": str(total),
                          "razonamiento": " + ".join(str(o["cantidad"]) for o in objs) + f" = {total}."}]}
    for q in ann["preguntas"]:
        q["tarea"] = "contar"
    o = rng.choice(objs)
    ann["preguntas"].append({"pregunta": f"¿De qué color son los {_pl(o['forma'])}?", "respuesta": o["color"],
                             "razonamiento": f"Busco los {_pl(o['forma'])}: son de color {o['color']}.",
                             "tarea": "color_forma"})
    single = [o for o in objs if o["cantidad"] == 1]
    if single:
        o = rng.choice(single)
        ann["preguntas"].append({"pregunta": f"¿Dónde está el {o['forma']} {o['color']}?",
                                 "respuesta": o["posicion"],
                                 "razonamiento": f"Ubico el {o['forma']} {o['color']} en la imagen dividida en "
                                                 f"tercios: está {o['posicion']}.", "tarea": "posicion"})
    ann["nivel"] = 1
    return np.asarray(img), ann


def text_image(rng: random.Random, size: int = 224) -> tuple[np.ndarray, dict]:
    from PIL import Image, ImageDraw
    bg = rng.choice([(255, 255, 255), (245, 240, 225), (230, 238, 250)])
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    lines = [" ".join(rng.sample(WORDS, rng.randint(1, 3))) for _ in range(rng.randint(1, 3))]
    if rng.random() < 0.4:
        lines.append(f"{rng.randint(1, 999)} + {rng.randint(1, 999)}")
    fs = max(8, round(rng.choice([16, 20, 24]) * size / 224))
    y = rng.randint(size // 28, size // 4)
    for line in lines:
        d.text((rng.randint(size // 37, size // 11), y), line, fill=(20, 20, 20), font=_font(fs))
        y += int(fs * 1.5)
    text = "\n".join(lines)
    qs = []
    if "+" in lines[-1]:
        a, b = (int(v) for v in lines[-1].split("+"))
        qs.append({"pregunta": "¿Cuánto da la cuenta escrita?", "respuesta": str(a + b),
                   "razonamiento": f"Leo «{lines[-1]}»: {tool_call('calc', f'{a} + {b}')}.", "tarea": "razonamiento"})
    ann = {"tipo": "texto", "texto_visible": text, "nivel": 2,
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
    while vals.count(max(vals)) > 1:                         # «la más alta» sin empates
        vals[vals.index(max(vals))] -= 1
    base, top, left = size - max(12, size * 30 // 224), max(6, size * 20 // 224), max(10, size * 20 // 224)
    unit = (base - top) / 10
    bw = (size - 2 * left) // n
    for i, (lab, v) in enumerate(zip(labels, vals)):
        x0 = left + i * bw + bw // 6
        d.rectangle((x0, base - v * unit, x0 + bw * 2 // 3, base), fill=COLORS[rng.choice(list(COLORS))])
        d.text((x0 + bw // 4, base + 2), lab, fill=(0, 0, 0), font=_font(max(8, size * 14 // 224)))
    d.line((left, base, size - left, base), fill=(0, 0, 0), width=2)
    for v in range(0, 11, 2):                                 # marcas de la escala vertical
        yv = base - v * unit
        d.line((left - 3, yv, left, yv), fill=(0, 0, 0))
        d.line((left, yv, size - left, yv), fill=(225, 225, 225))
        d.text((1, yv - 4), str(v), fill=(0, 0, 0), font=_font(max(7, size * 10 // 224)))
    mx = labels[int(np.argmax(vals))]
    data = dict(zip(labels, vals))
    ask = rng.choice(labels)
    ann = {"tipo": "grafico", "nivel": 1, "descripcion_es": f"Gráfico de barras con {n} categorías.",
           "description_en": f"Bar chart with {n} categories.", "datos": data,
           "preguntas": [{"pregunta": "¿Qué barra es la más alta?", "respuesta": mx,
                          "razonamiento": "Comparo las alturas: " + ", ".join(f"{k}={v}" for k, v in data.items())
                                          + f"; la mayor es {mx}."},
                         {"pregunta": "¿Cuánto suman todas las barras?", "respuesta": str(sum(vals)),
                          "razonamiento": "Leo los valores: " + ", ".join(f"{k}={v}" for k, v in data.items())
                                          + f". {tool_call('calc', ' + '.join(map(str, vals)))}.",
                          "tarea": "razonamiento"},
                         {"pregunta": f"¿Cuánto vale la barra {ask}?", "respuesta": str(data[ask]),
                          "razonamiento": f"La barra {ask} llega a la marca {data[ask]} de la escala.",
                          "tarea": "valores"}]}
    ann["preguntas"][0]["tarea"] = "valores"
    return np.asarray(img), ann


# ---------------------------------------------------------------------------
# Generadores verificables v2 (pedido del usuario: más calidad en imágenes)
# ---------------------------------------------------------------------------
# Cada generador devuelve (imagen, anotación). Cada pregunta trae su respuesta exacta,
# su razonamiento paso a paso y una «tarea» del banco (hope/eval/vision_bench.py):
# contar | color_forma | posicion | texto | valores | razonamiento.
# Las cuentas se escriben como llamadas a herramientas (⟦calc: …⟧ → resultado,
# ⟦circuito: …⟧ → resultado): Hope aprende a leer la imagen y a delegar el cálculo.
# El « → resultado» lo calcula la herramienta real (``tool_call``, el mismo ToolRunner del
# runtime) y queda entre marcas; ``encode_pair`` le pone peso 0: lo escribe el runtime, no Hope.

RES_OPEN, RES_CLOSE = "\ue000", "\ue001"        # marcas (uso privado de Unicode) del resultado
_RUNNER = None


def tool_call(name: str, arg: str) -> str:
    """``⟦nombre: arg⟧ → resultado`` con el resultado real de la herramienta, marcado."""
    global _RUNNER
    if _RUNNER is None:
        from hope.runtime.toolcalls import ToolRunner
        from hope.tools.builtin import default_catalog
        _RUNNER = ToolRunner(default_catalog())
    out = _RUNNER.run(name, arg)
    _RUNNER.calls.clear()
    return f"⟦{name}: {arg}⟧{RES_OPEN} → {out}{RES_CLOSE}"


def strip_results(text: str) -> str:
    """Saca las marcas (deja el texto tal como lo vería el runtime)."""
    return text.replace(RES_OPEN, "").replace(RES_CLOSE, "")

def _num(x: float, nd: int = 2) -> str:
    """Número con hasta ``nd`` decimales, sin ceros de más (punto decimal, como ⟦calc⟧)."""
    r = round(float(x), nd)
    return str(int(r)) if r == int(r) else f"{r:.{nd}f}".rstrip("0").rstrip(".")


def _ohm(r: float) -> str:
    return f"{_num(r / 1000)}k" if r >= 1000 else _num(r)


# Grupos en paralelo con resultado entero (valores ×1, ×10 o ×100): la respuesta es exacta.
_PARALLEL = [(2,), (3,), (4,), (5,), (6,), (8,), (10,), (12,), (15,), (20,), (22,), (33,), (47,),
             (2, 2), (4, 4), (6, 3), (3, 6), (12, 4), (6, 12), (10, 10), (20, 30), (30, 60), (10, 40), (12, 24),
             (6, 6, 6), (2, 3, 6), (4, 4, 2), (20, 30, 60), (12, 12, 6)]


def circuit_image(rng: random.Random, size: int = 224) -> tuple[np.ndarray, dict]:
    """Resistencias en serie y en paralelo entre A y B. Pregunta la resistencia equivalente
    (verificada con el simulador ⟦circuito⟧) y la corriente con una fuente aplicada."""
    from PIL import Image, ImageDraw

    from hope.tools.eng_tools import circuit
    scale = rng.choice([1, 10, 100])
    n_stages = rng.randint(1, 3)
    stages = [[v * scale for v in rng.choice(_PARALLEL)] for _ in range(n_stages)]
    if all(len(st) == 1 for st in stages) and n_stages == 1:
        stages.append([rng.choice([1, 2, 3, 5]) * 10 * scale])
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    fs = max(8, size // 20)
    font = _font(fs, rng)
    ink = (20, 20, 20)
    yc, x0, x1 = size // 2, size // 12, size - size // 12
    d.text((x0 - fs // 2, yc - fs * 2), "A", fill=ink, font=_font(fs + 2, rng))
    d.text((x1 - fs // 2, yc - fs * 2), "B", fill=ink, font=_font(fs + 2, rng))
    d.ellipse((x0 - 3, yc - 3, x0 + 3, yc + 3), fill=ink)
    d.ellipse((x1 - 3, yc - 3, x1 + 3, yc + 3), fill=ink)
    seg = (x1 - x0) / len(stages)
    names, k, netlist = [], 1, []
    nodes = ["a"] + [f"n{i}" for i in range(1, len(stages))] + ["0"]
    for si, st in enumerate(stages):
        sx0, sx1 = x0 + si * seg, x0 + (si + 1) * seg
        mx = (sx0 + sx1) / 2
        rw, rh = seg * 0.42, max(6, size // 28)
        m = len(st)
        gap = min(size * 0.7 / max(m, 1), size // 4)
        ys = [yc + (j - (m - 1) / 2) * gap for j in range(m)]
        if m > 1:                                          # barras de los nodos del paralelo
            bx0, bx1 = mx - rw / 2 - seg * 0.12, mx + rw / 2 + seg * 0.12
            d.line((sx0, yc, bx0, yc), fill=ink, width=2)
            d.line((bx1, yc, sx1, yc), fill=ink, width=2)
            d.line((bx0, ys[0], bx0, ys[-1]), fill=ink, width=2)
            d.line((bx1, ys[0], bx1, ys[-1]), fill=ink, width=2)
        else:
            bx0, bx1 = sx0, sx1
        for j, (r, y) in enumerate(zip(st, ys)):
            d.line((bx0, y, mx - rw / 2, y), fill=ink, width=2)
            d.line((mx + rw / 2, y, bx1, y), fill=ink, width=2)
            d.rectangle((mx - rw / 2, y - rh / 2, mx + rw / 2, y + rh / 2), outline=ink, width=2)
            label = f"R{k}={_ohm(r)}"
            d.text((mx - rw / 2, y - rh / 2 - fs - 2), label, fill=ink, font=font)
            names.append((f"R{k}", r))
            netlist.append(f"R{k} {nodes[si]} {nodes[si + 1]} {r:g}")
            k += 1
    # resistencia equivalente: suma de las etapas; cada etapa, inversa de la suma de inversas
    parts, exprs = [], []
    for st in stages:
        if len(st) == 1:
            parts.append(st[0])
            exprs.append(f"{st[0]:g}")
        else:
            parts.append(1 / sum(1 / r for r in st))
            exprs.append("1/(" + " + ".join(f"1/{r:g}" for r in st) + ")")
    req = sum(parts)
    sim = circuit("V1 a 0 1000; " + "; ".join(netlist))  # verificación independiente (el simulador redondea
    assert abs(1000 / sim["corrientes_fuentes"]["V1"] - req) < 1e-4 * req   # a 9 decimales: 1 kV de prueba)
    desc = []
    k = 1
    for st in stages:
        nm = [f"R{k + j}" for j in range(len(st))]
        desc.append(nm[0] if len(nm) == 1 else "(" + " ∥ ".join(nm) + ")")
        k += len(st)
    topo = " + ".join(desc)
    v = rng.choice([1, 2, 5, 9, 10, 12])
    i_ma = v / req * 1000
    net_v = f"V1 a 0 {v}; " + "; ".join(netlist)
    ann = {"tipo": "diagrama", "nivel": 4,
           "descripcion_es": f"Circuito con {len(names)} resistencias entre A y B: {topo}.",
           "description_en": f"Circuit with {len(names)} resistors between A and B: {topo}.",
           "texto_visible": "  ".join(f"{n}={_ohm(r)}" for n, r in names),
           "datos": {"resistencias": {n: r for n, r in names}, "topologia": topo, "netlist": "; ".join(netlist)},
           "preguntas": [
               {"pregunta": "¿Cuál es la resistencia equivalente entre A y B (en ohm)?", "respuesta": _num(req),
                "razonamiento": f"Leo el circuito: {topo}, con " + ", ".join(f"{n}={r:g}" for n, r in names)
                                + f". En serie se suman; en paralelo, inversa de la suma de inversas. "
                                  f"{tool_call('calc', ' + '.join(exprs))}.",
                "tarea": "razonamiento", "subtarea": "circuito_equivalente", "tolerancia": 0.01},
               {"pregunta": f"Si se aplican {v} V entre A y B, ¿qué corriente circula (en mA)?",
                "respuesta": _num(i_ma), "tarea": "razonamiento", "subtarea": "circuito_corriente",
                "tolerancia": 0.01,
                "razonamiento": f"Armo la netlist y la simulo: {tool_call('circuito', net_v)}. "
                                f"La fuente entrega {_num(v / req, 6)} A = {_num(i_ma)} mA."},
               {"pregunta": "¿Cuántas resistencias hay?", "respuesta": str(len(names)),
                "razonamiento": "Cuento los rectángulos: " + ", ".join(n for n, _ in names) + f": {len(names)}.",
                "tarea": "contar"}]}
    return np.asarray(img), ann


_ROWS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Ana", "Beto", "Caro", "Dani", "Eva",
         "Norte", "Sur", "Este", "Oeste", "Hierro", "Cobre", "Plata"]
_COLS = ["Ventas", "Horas", "Puntos", "Kg", "Litros", "Costo", "Votos", "Goles", "Notas", "Km"]


def table_image(rng: random.Random, size: int = 224) -> tuple[np.ndarray, dict]:
    """Tabla con encabezados: leer una celda y sumar una columna (con ⟦calc⟧)."""
    from PIL import Image, ImageDraw
    nr, nc = rng.randint(2, 4), rng.randint(2, 3)
    rows, cols = rng.sample(_ROWS, nr), rng.sample(_COLS, nc)
    vals = [[rng.randint(1, 99) for _ in range(nc)] for _ in range(nr)]
    img = Image.new("RGB", (size, size), rng.choice([(255, 255, 255), (250, 248, 240)]))
    d = ImageDraw.Draw(img)
    fs = rng.choice([size // 20, size // 18, size // 16])
    font = _font(fs, rng)
    left, top = size // 16, size // 10
    cw = (size - 2 * left) // (nc + 1)
    rh = int(fs * 2.0)
    ink = (20, 20, 20)
    for i in range(nr + 2):
        d.line((left, top + i * rh, left + cw * (nc + 1), top + i * rh), fill=ink, width=1)
    for j in range(nc + 2):
        d.line((left + j * cw, top, left + j * cw, top + rh * (nr + 1)), fill=ink, width=1)
    for j, c in enumerate(cols):
        d.text((left + (j + 1) * cw + 3, top + 3), c, fill=ink, font=font)
    for i, r in enumerate(rows):
        d.text((left + 3, top + (i + 1) * rh + 3), r, fill=ink, font=font)
        for j in range(nc):
            d.text((left + (j + 1) * cw + 3, top + (i + 1) * rh + 3), str(vals[i][j]), fill=ink, font=font)
    ri, ci = rng.randrange(nr), rng.randrange(nc)
    cj = rng.randrange(nc)
    col = [vals[i][cj] for i in range(nr)]
    csv = ",".join(["", *cols]) + "\n" + "\n".join(",".join([r, *map(str, v)]) for r, v in zip(rows, vals))
    ann = {"tipo": "documento", "nivel": 3,
           "descripcion_es": f"Tabla de {nr} filas y {nc} columnas ({', '.join(cols)}).",
           "description_en": f"Table with {nr} rows and {nc} columns.", "texto_visible": csv,
           "datos": {r: dict(zip(cols, v)) for r, v in zip(rows, vals)},
           "preguntas": [
               {"pregunta": f"¿Qué valor tiene {rows[ri]} en la columna {cols[ci]}?", "respuesta": str(vals[ri][ci]),
                "razonamiento": f"Busco la fila {rows[ri]} y la columna {cols[ci]}: la celda dice {vals[ri][ci]}.",
                "tarea": "valores", "subtarea": "tabla_celda"},
               {"pregunta": f"¿Cuánto suma la columna {cols[cj]}?", "respuesta": str(sum(col)),
                "razonamiento": f"Leo la columna {cols[cj]}: " + ", ".join(map(str, col))
                                + f". {tool_call('calc', ' + '.join(map(str, col)))}.",
                "tarea": "razonamiento", "subtarea": "tabla_suma"}]}
    return np.asarray(img), ann


def line_chart(rng: random.Random, size: int = 224) -> tuple[np.ndarray, dict]:
    """Gráfico de líneas con grilla rotulada: leer un punto, el máximo y una diferencia."""
    from PIL import Image, ImageDraw
    n = rng.randint(4, 6)
    xs = list(range(1, n + 1))
    ys = [rng.randint(0, 10) for _ in xs]
    while ys.count(max(ys)) > 1:                             # máximo único
        ys[ys.index(max(ys))] -= 1
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    left, right, top, base = size // 7, size // 14, size // 12, size - size // 7
    unit = (base - top) / 10
    step = (size - left - right) / (n - 1)
    fs = max(8, size // 22)
    for v in range(0, 11, 2):
        yv = base - v * unit
        d.line((left, yv, size - right, yv), fill=(220, 220, 220))
        d.text((3, yv - fs // 2), str(v), fill=(0, 0, 0), font=_font(fs))
    pts = [(left + i * step, base - y * unit) for i, y in enumerate(ys)]
    for i, (px, _) in enumerate(pts):
        d.line((px, top, px, base), fill=(235, 235, 235))
        d.text((px - fs // 3, base + 4), str(xs[i]), fill=(0, 0, 0), font=_font(fs))
    color = COLORS[rng.choice(["rojo", "azul", "verde", "violeta"])]
    d.line(pts, fill=color, width=2)
    for px, py in pts:
        d.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
    d.line((left, base, size - right, base), fill=(0, 0, 0), width=2)
    d.line((left, top, left, base), fill=(0, 0, 0), width=2)
    i = rng.randrange(n)
    a, b = sorted(rng.sample(range(n), 2))
    mx = xs[int(np.argmax(ys))]
    diff = ys[b] - ys[a]
    ann = {"tipo": "grafico", "nivel": 3, "descripcion_es": f"Gráfico de líneas con {n} puntos.",
           "description_en": f"Line chart with {n} points.", "datos": {str(x): y for x, y in zip(xs, ys)},
           "preguntas": [
               {"pregunta": f"¿Cuánto vale la línea en x = {xs[i]}?", "respuesta": str(ys[i]),
                "razonamiento": f"Subo desde x = {xs[i]} hasta la línea y leo la grilla: {ys[i]}.",
                "tarea": "valores", "subtarea": "lineas_punto"},
               {"pregunta": "¿En qué x la línea alcanza su máximo?", "respuesta": str(mx),
                "razonamiento": "Valores: " + ", ".join(f"x={x}: {y}" for x, y in zip(xs, ys))
                                + f"; el mayor es {max(ys)}, en x = {mx}.",
                "tarea": "valores", "subtarea": "lineas_maximo"},
               {"pregunta": f"¿Cuánto cambia la línea de x = {xs[a]} a x = {xs[b]}?", "respuesta": str(diff),
                "razonamiento": f"En x = {xs[a]} vale {ys[a]} y en x = {xs[b]} vale {ys[b]}: "
                                f"{tool_call('calc', f'{ys[b]} - {ys[a]}')}.",
                "tarea": "razonamiento", "subtarea": "lineas_cambio"}]}
    return np.asarray(img), ann


def pie_chart(rng: random.Random, size: int = 224) -> tuple[np.ndarray, dict]:
    """Gráfico de torta con leyenda: porcentajes múltiplos de 5 que suman 100."""
    from PIL import Image, ImageDraw
    n = rng.randint(2, 4)
    cuts = sorted(rng.sample(range(1, 20), n - 1))
    pcts = [(b - a) * 5 for a, b in zip([0] + cuts, cuts + [20])]
    if pcts.count(max(pcts)) > 1:                            # porción mayor única (sigue sumando 100)
        i, j = [k for k, v in enumerate(pcts) if v == max(pcts)][:2]
        pcts[i] += 5
        pcts[j] -= 5
    labels = rng.sample(["A", "B", "C", "D", "E"], n)
    names = rng.sample(["rojo", "verde", "azul", "amarillo", "violeta"], n)
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    r = size * 0.32
    cx, cy = size * 0.38, size * 0.5
    ang = -90.0
    for p, c in zip(pcts, names):
        d.pieslice((cx - r, cy - r, cx + r, cy + r), ang, ang + p * 3.6, fill=COLORS[c], outline=(255, 255, 255))
        ang += p * 3.6
    fs = max(8, size // 18)
    for j, (lab, c) in enumerate(zip(labels, names)):
        y = size * 0.25 + j * fs * 1.8
        d.rectangle((size * 0.76, y, size * 0.76 + fs, y + fs), fill=COLORS[c])
        d.text((size * 0.76 + fs * 1.4, y - 1), lab, fill=(0, 0, 0), font=_font(fs, rng))
    i = rng.randrange(n)
    big = labels[int(np.argmax(pcts))]
    ann = {"tipo": "grafico", "nivel": 3, "descripcion_es": f"Gráfico de torta con {n} porciones.",
           "description_en": f"Pie chart with {n} slices.", "datos": dict(zip(labels, pcts)),
           "preguntas": [
               {"pregunta": f"¿Qué porcentaje ocupa {labels[i]}?", "respuesta": str(pcts[i]),
                "razonamiento": f"{labels[i]} es la porción {names[i]}: ocupa {pcts[i] * 3.6:g}° de 360°, "
                                f"es decir {pcts[i]} %.",
                "tarea": "valores", "subtarea": "torta_porcentaje", "tolerancia_abs": 5},
               {"pregunta": "¿Qué porción es la más grande?", "respuesta": big,
                "razonamiento": "Comparo: " + ", ".join(f"{a}={p} %" for a, p in zip(labels, pcts))
                                + f"; la mayor es {big}.", "tarea": "valores", "subtarea": "torta_mayor"},
               {"pregunta": f"¿De qué color es la porción {labels[i]}?", "respuesta": names[i],
                "razonamiento": f"En la leyenda, {labels[i]} tiene el cuadradito {names[i]}.",
                "tarea": "color_forma"}]}
    return np.asarray(img), ann


_OCR_WORDS = WORDS + ["tierra", "agua", "fuego", "aire", "rápido", "lento", "norte", "sur", "año", "niño",
                      "kilo", "metro", "volt", "ohm", "plano", "nodo", "suma", "resta", "valor", "total"]


def ocr_image(rng: random.Random, size: int = 224) -> tuple[np.ndarray, dict]:
    """Texto en varias líneas con fuentes, tamaños, colores y ruido distintos (OCR)."""
    from PIL import Image, ImageDraw
    bg = tuple(rng.randint(225, 255) for _ in range(3))
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    n = rng.randint(2, 5)
    fs = rng.choice([size // 16, size // 13, size // 11])
    lines = []
    for _ in range(n):
        toks = [rng.choice(_OCR_WORDS) if rng.random() < 0.8 else str(rng.randint(0, 999))
                for _ in range(rng.randint(1, 3))]
        lines.append(" ".join(toks))
    y = rng.randint(4, size // 8)
    drawn = []
    for line in lines:
        if y > size - fs * 1.2:
            break
        font = _font(fs, rng)
        while d.textlength(line, font=font) > size - 16 and " " in line:     # que entre en el ancho
            line = line.rsplit(" ", 1)[0]
        d.text((rng.randint(4, 12), y), line, fill=tuple(rng.randint(0, 80) for _ in range(3)), font=font)
        drawn.append(line)
        y += int(fs * rng.uniform(1.3, 1.7))
    lines = drawn
    arr = np.asarray(img).astype(np.int16)
    arr = np.clip(arr + np.random.default_rng(rng.randrange(2**31)).normal(0, rng.uniform(0, 12), arr.shape), 0, 255)
    text = "\n".join(lines)
    ann = {"tipo": "texto", "nivel": 4, "texto_visible": text,
           "descripcion_es": f"Texto en {len(lines)} líneas, con ruido.",
           "description_en": f"Text in {len(lines)} lines, with noise.",
           "preguntas": [{"pregunta": "¿Cuántas líneas de texto hay?", "respuesta": str(len(lines)),
                          "razonamiento": f"Cuento los renglones: {len(lines)}.", "tarea": "contar"},
                         {"pregunta": "¿Qué dice la primera línea?", "respuesta": lines[0],
                          "razonamiento": f"Leo el primer renglón: «{lines[0]}».", "tarea": "texto"}]}
    return arr.astype(np.uint8), ann


def clock_image(rng: random.Random, size: int = 224) -> tuple[np.ndarray, dict]:
    """Reloj analógico (minutos múltiplos de 5): leer la hora y sumar minutos."""
    import math as _m

    from PIL import Image, ImageDraw
    h, mnt = rng.randint(1, 12), rng.randrange(0, 60, 5)
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    c, r = size / 2, size * 0.42
    d.ellipse((c - r, c - r, c + r, c + r), outline=(20, 20, 20), width=max(2, size // 70), fill=(250, 250, 240))
    fs = max(8, size // 16)
    numbers = rng.random() < 0.7
    for k in range(1, 13):
        a = _m.radians(k * 30 - 90)
        d.line((c + r * 0.88 * _m.cos(a), c + r * 0.88 * _m.sin(a), c + r * _m.cos(a), c + r * _m.sin(a)),
               fill=(20, 20, 20), width=2)
        if numbers:
            d.text((c + r * 0.72 * _m.cos(a) - fs / 3, c + r * 0.72 * _m.sin(a) - fs / 2), str(k),
                   fill=(20, 20, 20), font=_font(fs, rng))
    am = _m.radians(mnt * 6 - 90)
    ah = _m.radians((h % 12) * 30 + mnt * 0.5 - 90)
    d.line((c, c, c + r * 0.85 * _m.cos(am), c + r * 0.85 * _m.sin(am)), fill=(20, 20, 20), width=max(2, size // 90))
    d.line((c, c, c + r * 0.5 * _m.cos(ah), c + r * 0.5 * _m.sin(ah)), fill=(20, 20, 20), width=max(3, size // 45))
    add = rng.choice([10, 15, 20, 25, 30, 40, 45, 50, 70, 90])
    tot = h * 60 + mnt + add
    h2, m2 = ((tot // 60) - 1) % 12 + 1, tot % 60
    hora = f"{h}:{mnt:02d}"
    ann = {"tipo": "diagrama", "nivel": 2, "descripcion_es": f"Reloj analógico que marca las {hora}.",
           "description_en": f"Analog clock showing {hora}.", "datos": {"hora": hora},
           "preguntas": [
               {"pregunta": "¿Qué hora marca el reloj?", "respuesta": hora, "tarea": "valores", "subtarea": "reloj",
                "razonamiento": f"La aguja corta está entre el {h} y el {h % 12 + 1}: son las {h}. La larga apunta "
                                f"al {mnt // 5 if mnt else 12}: {mnt // 5} × 5 = {mnt} minutos."},
               {"pregunta": f"¿Qué hora será dentro de {add} minutos?", "respuesta": f"{h2}:{m2:02d}",
                "tarea": "razonamiento", "subtarea": "reloj_suma",
                "razonamiento": f"Marca las {hora}. {mnt} + {add} = {mnt + add} minutos"
                                + (f" = {(mnt + add) // 60} h y {(mnt + add) % 60} min" if mnt + add >= 60 else "")
                                + f": serán las {h2}:{m2:02d}."}]}
    return np.asarray(img), ann


def geometry_image(rng: random.Random, size: int = 224) -> tuple[np.ndarray, dict]:
    """Figuras con medidas rotuladas: áreas (rectángulo, triángulo, círculo) y ángulos."""
    import math as _m

    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    fs = max(8, size // 16)
    ink = (20, 20, 20)
    fill = COLORS[rng.choice(["azul", "verde", "amarillo", "violeta"])]
    kind = rng.choice(["rectangulo", "triangulo", "circulo", "angulo"])
    if kind == "rectangulo":
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        w, h = size * 0.2 + a * size * 0.045, size * 0.15 + b * size * 0.04
        x0, y0 = (size - w) / 2, (size - h) / 2
        d.rectangle((x0, y0, x0 + w, y0 + h), outline=ink, fill=fill, width=2)
        d.text((x0 + w / 2 - fs, y0 + h + 3), f"{a} cm", fill=ink, font=_font(fs, rng))
        d.text((x0 + w + 4, y0 + h / 2 - fs / 2), f"{b} cm", fill=ink, font=_font(fs, rng))
        q = {"pregunta": "¿Cuál es el área del rectángulo (en cm²)?", "respuesta": str(a * b),
             "razonamiento": f"Base {a} cm y altura {b} cm. Área = base × altura: {tool_call('calc', f'{a} * {b}')}."}
        q2 = {"pregunta": "¿Cuál es el perímetro del rectángulo (en cm)?", "respuesta": str(2 * (a + b)),
              "razonamiento": f"Perímetro = 2 × (base + altura): {tool_call('calc', f'2 * ({a} + {b})')}."}
        desc = f"Rectángulo de {a} cm por {b} cm."
    elif kind == "triangulo":
        b, h = rng.randint(2, 12), rng.randint(2, 12)
        w, hh = size * 0.25 + b * size * 0.04, size * 0.2 + h * size * 0.035
        x0, yb = (size - w) / 2, (size + hh) / 2
        apex = (x0 + w * rng.uniform(0.25, 0.75), yb - hh)
        d.polygon([(x0, yb), (x0 + w, yb), apex], outline=ink, fill=fill)
        d.line((apex[0], apex[1], apex[0], yb), fill=ink, width=1)
        d.text((x0 + w / 2 - fs, yb + 3), f"b = {b}", fill=ink, font=_font(fs, rng))
        d.text((apex[0] + 3, yb - hh / 2), f"h = {h}", fill=ink, font=_font(fs, rng))
        q = {"pregunta": "¿Cuál es el área del triángulo?", "respuesta": _num(b * h / 2),
             "razonamiento": f"Base {b} y altura {h}. Área = base × altura / 2: {tool_call('calc', f'{b} * {h} / 2')}."}
        q2 = None
        desc = f"Triángulo de base {b} y altura {h}."
    elif kind == "circulo":
        r = rng.randint(1, 9)
        rr = size * 0.12 + r * size * 0.025
        c = size / 2
        d.ellipse((c - rr, c - rr, c + rr, c + rr), outline=ink, fill=fill, width=2)
        d.line((c, c, c + rr, c), fill=ink, width=2)
        d.text((c + rr / 4, c - fs - 2), f"r = {r}", fill=ink, font=_font(fs, rng))
        area = _m.pi * r * r
        q = {"pregunta": "¿Cuál es el área del círculo?", "respuesta": _num(area),
             "razonamiento": f"Radio {r}. Área = π × r²: {tool_call('calc', f'3.14159265 * {r} * {r}')}; "
                             f"redondeo: {_num(area)}.", "tolerancia": 0.01}
        q2 = None
        desc = f"Círculo de radio {r}."
    else:
        a1, a2 = rng.randrange(20, 100, 5), rng.randrange(20, 80, 5)
        a3 = 180 - a1 - a2
        if a3 < 10:
            a1, a3 = a1 - 30, a3 + 30
        # triángulo con esos ángulos (ley de los senos), escalado al lienzo
        A, B = _m.radians(a1), _m.radians(a2)
        L = size * 0.6
        px = L * _m.tan(B) / (_m.tan(A) + _m.tan(B))
        py = px * _m.tan(A)
        sc = min(1.0, size * 0.6 / max(py, 1))
        x0, yb = size * 0.2, size * 0.75
        P = [(x0, yb), (x0 + L, yb), (x0 + px, yb - py * sc)]
        d.polygon(P, outline=ink, fill=fill)
        d.text((P[0][0] + 6, P[0][1] - fs - 2), f"{a1}°", fill=ink, font=_font(fs, rng))
        d.text((P[1][0] - fs * 2.2, P[1][1] - fs - 2), f"{a2}°", fill=ink, font=_font(fs, rng))
        d.text((P[2][0] - fs / 2, P[2][1] - fs - 4), "?", fill=ink, font=_font(fs, rng))
        q = {"pregunta": "¿Cuánto mide el ángulo marcado con «?» (en grados)?", "respuesta": str(a3),
             "razonamiento": f"Los ángulos de un triángulo suman 180°. Veo {a1}° y {a2}°: "
                             f"{tool_call('calc', f'180 - {a1} - {a2}')}."}
        q2 = None
        desc = f"Triángulo con ángulos de {a1}° y {a2}° y uno desconocido."
    qs = [q] + ([q2] if q2 else [])
    for x in qs:
        x["tarea"], x["subtarea"] = "razonamiento", f"geometria_{kind}"
    ann = {"tipo": "diagrama", "nivel": 3, "descripcion_es": desc, "description_en": desc, "preguntas": qs}
    return np.asarray(img), ann


def scene_image(rng: random.Random, size: int = 224) -> tuple[np.ndarray, dict]:
    """Escena con relaciones espaciales: «¿qué está a la izquierda del círculo rojo?»."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), (250, 250, 245))
    d = ImageDraw.Draw(img)
    n = rng.randint(3, 5)
    combos = rng.sample([(f, c) for f in SHAPES_EN for c in COLORS if c != "negro"], n)
    r = size // 12
    cols = sorted(rng.sample(range(n * 2), n))          # columnas distintas: «izquierda» es inequívoco
    xs = [int(r + (size - 2 * r) * (c + 0.5) / (n * 2)) for c in cols]
    order = list(range(n))
    rng.shuffle(order)
    objs = []
    for k, (x, (f, c)) in enumerate(zip(xs, [combos[i] for i in order])):
        y = rng.randint(r, size - r)
        box = (x - r, y - r, x + r, y + r)
        if f == "círculo":
            d.ellipse(box, fill=COLORS[c])
        elif f == "cuadrado":
            d.rectangle(box, fill=COLORS[c])
        else:
            d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=COLORS[c])
        objs.append({"nombre": f"{f} {c}", "x": x, "y": y, "forma": f, "color": c})
    i = rng.randrange(1, n)
    ref, left = objs[i], objs[i - 1]
    top = min(objs, key=lambda o: o["y"])
    j = rng.randrange(n)
    right_of = sum(1 for o in objs if o["x"] > objs[j]["x"])
    shapes_count = {}
    for o in objs:
        shapes_count[o["forma"]] = shapes_count.get(o["forma"], 0) + 1
    uniq = [o for o in objs if shapes_count[o["forma"]] == 1]
    orden = ", ".join(o["nombre"] for o in objs)
    qs = [{"pregunta": f"¿Qué figura está inmediatamente a la izquierda del {ref['nombre']}?",
           "respuesta": left["nombre"], "tarea": "razonamiento", "subtarea": "espacial_izquierda",
           "razonamiento": f"De izquierda a derecha: {orden}. Antes del {ref['nombre']} está el {left['nombre']}."},
          {"pregunta": f"¿Cuántas figuras están a la derecha del {objs[j]['nombre']}?", "respuesta": str(right_of),
           "tarea": "razonamiento", "subtarea": "espacial_contar",
           "razonamiento": f"De izquierda a derecha: {orden}. Después del {objs[j]['nombre']} hay {right_of}."},
          {"pregunta": "¿Qué figura está más arriba?", "respuesta": top["nombre"], "tarea": "posicion",
           "razonamiento": f"Comparo la altura de cada figura: la más alta es el {top['nombre']}."},
          {"pregunta": "¿Cuántas figuras hay?", "respuesta": str(n), "tarea": "contar",
           "razonamiento": f"Cuento: {orden}: {n}."}]
    if uniq:
        o = rng.choice(uniq)
        qs.append({"pregunta": f"¿De qué color es el {o['forma']}?", "respuesta": o["color"], "tarea": "color_forma",
                   "razonamiento": f"Hay un solo {o['forma']} y es {o['color']}."})
        o = rng.choice(objs)
        qs.append({"pregunta": f"¿Dónde está el {o['nombre']}?", "respuesta": _where(o["x"] / size, o["y"] / size),
                   "tarea": "posicion", "razonamiento": f"Divido la imagen en tercios: el {o['nombre']} está "
                                                       f"{_where(o['x'] / size, o['y'] / size)}."})
    ann = {"tipo": "formas", "nivel": 2, "descripcion_es": f"Escena con {n} figuras; de izquierda a derecha: {orden}.",
           "description_en": f"Scene with {n} shapes.",
           "objetos": [{"nombre": o["nombre"], "cantidad": 1, "posicion": _where(o["x"] / size, o["y"] / size)}
                       for o in objs], "preguntas": qs}
    return np.asarray(img), ann


def augment(arr: np.ndarray, rng: random.Random, strength: float = 1.0) -> np.ndarray:
    """Aumentos que no cambian la respuesta: rotación leve (±3°), ruido gaussiano y JPEG."""
    from PIL import Image
    img = Image.fromarray(arr)
    if rng.random() < 0.5:
        img = img.rotate(rng.uniform(-3, 3) * strength, resample=Image.BILINEAR, fillcolor=img.getpixel((0, 0)))
    out = np.asarray(img).astype(np.float32)
    if rng.random() < 0.5:
        out += np.random.default_rng(rng.randrange(2**31)).normal(0, rng.uniform(2, 10) * strength, out.shape)
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(40, 90))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return np.asarray(img).copy()


GENERATORS = {"formas": shapes, "texto": text_image, "barras": bar_chart, "circuito": circuit_image,
              "tabla": table_image, "lineas": line_chart, "torta": pie_chart, "ocr": ocr_image,
              "reloj": clock_image, "geometria": geometry_image, "escena": scene_image}
BASIC_KINDS = ("formas", "texto", "barras")                 # los generadores originales (base de la ablación)
DOC_KINDS = ("tabla", "ocr", "circuito")                    # documentos: se generan a mayor resolución
LEVELS = {"formas": 1, "barras": 1, "texto": 2, "escena": 2, "reloj": 2, "lineas": 3, "torta": 3,
          "tabla": 3, "geometria": 3, "ocr": 4, "circuito": 4}
# Semillas reservadas para el banco sellado (hope/eval/vision_bench.py): el entrenamiento no las usa.
RESERVED_SEEDS = range(7_000_000_000, 8_000_000_000)


def procedural(n: int, seed: int = 0, size: int = 224, kinds: tuple[str, ...] = BASIC_KINDS, *,
               doc_scale: int = 1, augment_p: float = 0.0, _allow_reserved: bool = False):
    """``n`` imágenes con su anotación. ``doc_scale``: los documentos (tablas, OCR, circuitos)
    se dibujan a ``size · doc_scale`` (alta resolución, para mosaicos). ``augment_p``:
    fracción con aumentos (rotación leve, ruido, JPEG)."""
    if seed in RESERVED_SEEDS and not _allow_reserved:
        raise ValueError("semilla reservada para el banco sellado de visión")
    rng = random.Random(seed)
    for _ in range(n):
        kind = rng.choice(kinds)
        arr, ann = GENERATORS[kind](rng, size * doc_scale if kind in DOC_KINDS else size)
        ann.setdefault("nivel", LEVELS[kind])
        ann["generador"] = kind
        if augment_p and rng.random() < augment_p:
            arr = augment(arr, rng)
        yield arr, ann


# ---------------------------------------------------------------------------
# Lotes imagen–texto para el entrenamiento
# ---------------------------------------------------------------------------
@dataclass
class ImageTextItem:
    image: torch.Tensor            # uint8 [3, H, W] (H = W = S, o múltiplos de S en documentos)
    prompt: str
    answer: str
    level: int = 1                 # dificultad para el currículo (1 simple … 4 compleja)
    kind: str = ""


def encode_pair(prompt: str, answer: str, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Bytes de consigna + respuesta y pesos de la pérdida: 1 en la respuesta, 0 en la consigna
    y 0 en los resultados de herramientas (entre ``RES_OPEN`` y ``RES_CLOSE``, ver ``tool_call``):
    esos los escribe el runtime al ejecutar la llamada; Hope aprende a hacerla y a usar el resultado."""
    p = prompt.encode("utf-8")
    chunks, weights = [p], [0.0]
    for k, part in enumerate(re.split(f"[{RES_OPEN}{RES_CLOSE}]", answer)):
        chunks.append(part.encode("utf-8"))
        weights.append(0.0 if k % 2 else 1.0)
    chunks.append(b"\x00")
    weights.append(1.0)
    x = np.zeros(length, dtype=np.int64)
    w = np.zeros(length, dtype=np.float32)
    pos = 0
    for c, wt in zip(chunks, weights):
        c = c[:max(0, length - pos)]
        x[pos:pos + len(c)] = list(c)
        w[pos:pos + len(c)] = wt
        pos += len(c)
    return x, w


def resize_square(img: torch.Tensor, size: int) -> torch.Tensor:
    """uint8 [3, H, W] → uint8 [3, size, size] (reducción por área: así ve un documento
    de alta resolución un modelo sin mosaicos)."""
    if img.shape[-2:] == (size, size):
        return img
    import torch.nn.functional as F
    out = F.interpolate(img[None].float(), size=(size, size), mode="area")[0]
    return out.round().clamp(0, 255).to(torch.uint8)


class ImageTextSource:
    """Fuente de pares imagen–texto para ``Trainer.train(vision=...)``.

    * Los lotes se arman con imágenes del mismo tamaño (los documentos en alta resolución
      van juntos).
    * ``image_size`` + ``tiles=False``: las imágenes más grandes se reducen a ese tamaño
      (modelo sin mosaicos); con ``tiles=True`` se dejan enteras (``vis_tiling``).
    * ``curriculum(total)``: de imágenes simples a complejas. El nivel máximo permitido sube
      de 1 a 4 durante la primera parte (``warm``) de las ``total`` llamadas a ``batch``.
    """

    def __init__(self, items: list[ImageTextItem], name: str = "vision", *, image_size: int | None = None,
                 tiles: bool = True):
        if not items:
            raise ValueError("fuente visual vacía")
        if image_size is not None and not tiles:
            cache: dict[int, torch.Tensor] = {}
            for it in items:
                if it.image.shape[-1] != image_size or it.image.shape[-2] != image_size:
                    it.image = cache.setdefault(id(it.image), resize_square(it.image, image_size))
        self.items, self.name = items, name
        keys = {}
        self._shape = np.array([keys.setdefault(tuple(it.image.shape), len(keys)) for it in items])
        self._level = np.array([it.level for it in items])
        self._calls, self._total, self._warm = 0, None, 0.6

    def curriculum(self, total_batches: int, warm: float = 0.6) -> "ImageTextSource":
        self._total, self._warm, self._calls = max(1, total_batches), warm, 0
        return self

    def max_level(self) -> int:
        if self._total is None:
            return 99
        p = self._calls / (self._warm * self._total)
        return 1 + min(3, int(p * 4))

    @classmethod
    def from_store(cls, store: ImageStore, name: str = "vision", **kw) -> "ImageTextSource":
        items = []
        cache: dict[str, torch.Tensor] = {}
        for e in store.entries():
            img = cache.setdefault(e["id"], store.load(e["id"]))
            items += [ImageTextItem(img, p, a) for p, a in samples_from_annotation(e["anotacion"])]
        return cls(items, name, **kw)

    @classmethod
    def from_procedural(cls, n: int, seed: int = 0, size: int = 224, name: str = "procedural", *,
                        kinds: tuple[str, ...] = BASIC_KINDS, doc_scale: int = 1, augment_p: float = 0.0,
                        tiles: bool = True) -> "ImageTextSource":
        items = []
        for arr, ann in procedural(n, seed, size, kinds, doc_scale=doc_scale, augment_p=augment_p):
            img = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()
            items += [ImageTextItem(img, p, a, ann.get("nivel", 1), ann.get("generador", ""))
                      for p, a in samples_from_annotation(ann)]
        return cls(items, name, image_size=size, tiles=tiles)

    def batch(self, batch: int, length: int, rng: np.random.Generator):
        allowed = self._level <= self.max_level()
        if not allowed.any():
            allowed = self._level <= self._level.min()
        self._calls += 1
        pool = np.flatnonzero(allowed)
        first = int(pool[rng.integers(len(pool))])
        same = np.flatnonzero(allowed & (self._shape == self._shape[first]))
        idx = np.concatenate([[first], same[rng.integers(0, len(same), size=batch - 1)]])
        imgs, xs, ws = [], [], []
        for i in idx:
            it = self.items[i]
            x, w = encode_pair(it.prompt, it.answer, length)
            imgs.append(it.image)
            xs.append(x)
            ws.append(w)
        return torch.stack(imgs), torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ws))
