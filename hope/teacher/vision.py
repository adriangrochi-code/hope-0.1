"""Qwen describe imágenes para Hope (ARQUITECTURA_HN1 §10.1.5; decisión del usuario).

El maestro multimodal mira cada imagen y devuelve **una anotación densa** en JSON:
descripción en español e inglés, texto visible, objetos con cantidad y posición,
datos extraídos (gráficos, tablas, circuitos) y preguntas con su razonamiento.
De una sola consulta salen varias muestras de entrenamiento
(``hope.data.vision.samples_from_annotation``).

Para analizar **la mayor cantidad de datos con el menor costo**:

1. **Deduplicación** exacta (hash del archivo) y casi exacta (hash perceptual
   ``dhash``, distancia de Hamming ≤ ``near_dup``): no se paga dos veces la misma foto.
2. **Caché por contenido**: lo ya anotado nunca se vuelve a pedir (se puede cortar
   y retomar).
3. **Selección por diversidad** (``select_diverse``, k-centro voraz sobre
   miniaturas o sobre las fichas visuales de Hope): con un presupuesto de N
   consultas se anotan las N imágenes que más cubren el conjunto, no las primeras N.
4. **Una consulta densa por imagen**: el costo de leer la imagen (la parte cara para
   el maestro) se paga una vez para todas las tareas.
5. **Consultas en paralelo** (``workers``) hacia el servidor local (llama-server,
   vLLM u OpenVINO con varias ranuras).
6. **Verificación del maestro**: en imágenes procedurales se conoce la verdad; se
   compara lo que dice Qwen y se lleva la tasa de error (``audit_teacher``).

Toda consulta pasa por el guardián (``Teacher.ask``): maestro local = nivel 1.
Formato de mensaje: API de chat de OpenAI con ``image_url`` en base64, que
aceptan llama-server, vLLM y Ollama para modelos con visión.
"""

from __future__ import annotations

import base64
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hope.data.vision import ImageStore, dhash, hamming
from hope.teacher.client import Teacher, TeacherConfig

PROMPT = """Sos el maestro de Hope, un modelo que está aprendiendo a ver. Analizá la imagen con cuidado y
respondé SOLO con un objeto JSON con estas claves:
- "tipo": foto | documento | grafico | diagrama | codigo | captura | otro
- "descripcion_es": descripción detallada en español (qué hay, dónde, colores, relaciones, contexto)
- "description_en": la misma descripción en inglés
- "texto_visible": todo el texto legible, transcripto tal cual ("" si no hay)
- "objetos": lista de {"nombre", "cantidad", "posicion"}
- "datos": valores que se puedan extraer (gráficos, tablas, circuitos, fórmulas); {} si no hay
- "preguntas": 3 a 5 preguntas útiles sobre la imagen, cada una con "pregunta", "respuesta" y
  "razonamiento" (cómo se llega a la respuesta mirando la imagen, paso a paso)
No inventes lo que no se ve; si algo es dudoso, decilo en la descripción."""


def image_message(png: bytes, prompt: str = PROMPT) -> list[dict]:
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    return [{"role": "user", "content": [{"type": "text", "text": prompt},
                                         {"type": "image_url", "image_url": {"url": url}}]}]


def _png(img, max_side: int) -> bytes:
    from PIL import Image
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    s = max_side / max(img.size)
    if s < 1:
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def parse_annotation(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("el maestro no devolvió JSON")
    ann = json.loads(m.group(0))
    if not isinstance(ann, dict) or not (ann.get("descripcion_es") or ann.get("texto_visible")):
        raise ValueError("anotación incompleta")
    ann.setdefault("preguntas", [])
    return ann


def _thumb(img, side: int = 16) -> np.ndarray:
    from PIL import Image
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return np.asarray(img.convert("RGB").resize((side, side), Image.BILINEAR), dtype=np.float32).ravel() / 255.0


def select_diverse(features: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """k-centro voraz: elige ``k`` índices que maximizan la cobertura del conjunto."""
    n = len(features)
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(n))]
    dist = np.linalg.norm(features - features[chosen[0]], axis=1)
    for _ in range(k - 1):
        i = int(dist.argmax())
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(features - features[i], axis=1))
    return chosen


@dataclass
class AnnotationReport:
    requested: int = 0
    annotated: int = 0
    cached: int = 0
    duplicates: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


class ImageAnnotator:
    def __init__(self, teacher: Teacher, store: ImageStore, *, near_dup: int = 4, max_side: int = 768,
                 workers: int = 2, source: str = "qwen"):
        self.teacher, self.store = teacher, store
        self.near_dup, self.max_side, self.workers, self.source = near_dup, max_side, workers, source
        self.done = {e["id"] for e in store.entries() if e.get("fuente") == source}
        self.hashes: list[int] = []

    def _is_near_dup(self, h: int) -> bool:
        return any(hamming(h, o) <= self.near_dup for o in self.hashes)

    def _one(self, sid: str, img) -> tuple[str, dict | None, str | None]:
        try:
            reply = self.teacher.ask(image_message(_png(img, self.max_side)), purpose="vision.annotate")
            return sid, parse_annotation(reply.text), None
        except Exception as e:                       # noqa: BLE001 — se registra y se sigue
            return sid, None, f"{sid}: {e}"

    def annotate(self, images: list, budget: int | None = None) -> AnnotationReport:
        """``images``: rutas o arreglos. ``budget``: máximo de consultas al maestro."""
        rep = AnnotationReport(requested=len(images))
        todo = []
        for img in images:
            sid = self.store.put(img)
            if sid in self.done:
                rep.cached += 1
                continue
            h = dhash(img)
            if self._is_near_dup(h):
                rep.duplicates += 1
                continue
            self.hashes.append(h)
            todo.append((sid, img))
        if budget is not None and len(todo) > budget:
            feats = np.stack([_thumb(img) for _, img in todo])
            todo = [todo[i] for i in select_diverse(feats, budget)]
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as ex:
            for sid, ann, err in ex.map(lambda t: self._one(*t), todo):
                if ann is None:
                    rep.failed += 1
                    rep.errors.append(err)
                    continue
                self.store.add(sid, ann, self.source)
                self.done.add(sid)
                rep.annotated += 1
        return rep


# ---------------------------------------------------------------------------
def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def audit_teacher(teacher: Teacher, n: int = 20, seed: int = 0, size: int = 224) -> dict:
    """Mide al maestro con imágenes procedurales cuya respuesta se conoce."""
    from hope.data.vision import procedural
    ok = total = 0
    fails = []
    for arr, truth in procedural(n, seed, size):
        try:
            ann = parse_annotation(teacher.ask(image_message(_png(arr, size)), purpose="vision.audit").text)
        except Exception as e:                       # noqa: BLE001
            fails.append(str(e))
            total += 1
            continue
        checks = []
        if truth.get("texto_visible"):
            checks.append(_norm(ann.get("texto_visible", "")) == _norm(truth["texto_visible"]))
        if truth.get("datos"):
            checks.append(all(_norm(ann.get("datos", {}).get(k)) == _norm(v) for k, v in truth["datos"].items()))
        for q in truth.get("preguntas", [])[:1]:
            got = next((a for a in ann.get("preguntas", []) if _norm(a.get("pregunta")) == _norm(q["pregunta"])), None)
            if got is not None:
                checks.append(_norm(got.get("respuesta")) == _norm(q["respuesta"]))
        total += 1
        if checks and all(checks):
            ok += 1
        else:
            fails.append(truth.get("tipo", "?"))
    return {"imagenes": total, "correctas": ok, "precision": ok / max(total, 1), "fallos": fails[:20]}


class MockVisionTeacher(Teacher):
    """Maestro multimodal simulado para pruebas: devuelve una anotación fija (o la que
    devuelva ``oracle(png)``) y puede equivocarse a propósito."""

    def __init__(self, oracle=None, error_rate: float = 0.0, seed: int = 0, guardian=None):
        super().__init__(TeacherConfig(endpoint="http://127.0.0.1/mock", model="mock-vl"), guardian)
        import random
        self.oracle, self.error_rate, self.rnd = oracle, error_rate, random.Random(seed)

    def _complete(self, messages):
        parts = messages[-1]["content"]
        url = next(p["image_url"]["url"] for p in parts if p.get("type") == "image_url")
        png = base64.b64decode(url.split(",", 1)[1])
        ann = self.oracle(png) if self.oracle else {"tipo": "otro", "descripcion_es": "una imagen",
                                                    "description_en": "an image", "preguntas": []}
        ann = json.loads(json.dumps(ann))
        if self.rnd.random() < self.error_rate:
            ann["texto_visible"] = (ann.get("texto_visible") or "") + " x"
            ann["datos"] = {k: "?" for k in ann.get("datos", {})}
            for q in ann.get("preguntas", []):
                q["respuesta"] = "?"
        return "```json\n" + json.dumps(ann, ensure_ascii=False) + "\n```", {}


# ---------------------------------------------------------------------------
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def main(argv: list[str] | None = None) -> None:
    """``python -m hope.teacher.vision --images carpeta --store data/vision --budget 5000``

    ``--audit 50`` mide primero al maestro con imágenes procedurales."""
    import argparse

    from guardian.cli import load as load_guardian
    from hope.teacher.client import load_teacher
    ap = argparse.ArgumentParser(description="Qwen anota imágenes para la zona visual de Hope")
    ap.add_argument("--images", required=True)
    ap.add_argument("--store", default="data/vision")
    ap.add_argument("--teacher", default="configs/teacher.yaml")
    ap.add_argument("--budget", type=int, default=None, help="máximo de consultas (elige las más diversas)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--audit", type=int, default=0)
    ap.add_argument("--guardian", default="configs/guardian.yaml")
    args = ap.parse_args(argv)
    teacher = load_teacher(args.teacher, load_guardian(args.guardian))
    if args.audit:
        print(json.dumps(audit_teacher(teacher, args.audit), ensure_ascii=False, indent=2))
    paths = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMAGE_EXT)
    rep = ImageAnnotator(teacher, ImageStore(args.store, args.size), workers=args.workers).annotate(paths, args.budget)
    print(json.dumps(rep.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
