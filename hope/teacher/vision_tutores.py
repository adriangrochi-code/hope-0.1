"""Tutores de imágenes locales con verificación cruzada (ARQUITECTURA_HN1 §10.1.4; pedido del usuario).

El usuario preguntó por Google Lens: no tiene API oficial y automatizarlo viola sus términos, así que
Hope aprende a ver con **modelos abiertos que corren en casa** y que se controlan entre sí:

=============  ================================  ============  ===========================================
rol            modelo (por defecto)              licencia      para qué
=============  ================================  ============  ===========================================
principal      Qwen2.5-VL-7B-Instruct            Apache 2.0    describe, transcribe, cuenta y responde
segundo        SmolVLM2 / Moondream2             Apache 2.0    responde las mismas preguntas por su cuenta
florence       Florence-2 (large)                MIT           OCR, detección con cajas, descripción
ocr            Tesseract / PaddleOCR             Apache 2.0    confirma el texto
verificador    SigLIP (o CLIP, MIT)              Apache 2.0    puntúa descripción ↔ imagen
=============  ================================  ============  ===========================================

Cada tutor declara ``licencia`` y ``entrenar_con_salidas`` (``configs/tutores_vision.yaml``). Un tutor con
``entrenar_con_salidas`` distinto de ``true`` (p. ej. Qwen2.5-VL-3B, con licencia propia; PaliGemma, con los
términos de Gemma; YOLO, AGPL) **solo puede verificar**: sus salidas no entran al entrenamiento y la
anotación queda marcada.

Flujo (``TutorPanel``):

1. El tutor principal anota la imagen (descripción es/en, texto visible, objetos con nombre en inglés y
   cantidad, datos, preguntas con respuesta y razonamiento).
2. Verificación cruzada, **campo por campo**:
   * texto: el OCR (Tesseract/PaddleOCR) y Florence-2 lo leen por su cuenta (exactitud por carácter ≥ umbral);
   * objetos y cantidades: las cajas de Florence-2;
   * preguntas: el segundo VLM contesta sin ver la respuesta del principal; tienen que coincidir;
   * descripción: SigLIP la puntúa contra la imagen y contra las descripciones de las otras imágenes
     del lote (tiene que ser la que mejor corresponde y superar un mínimo);
   * datos (valores de tablas/gráficos): cada número tiene que aparecer en el texto que leyó el OCR.
3. Lo que coincide entra al ``ImageStore`` como **verificado** (fuente ``tutores``); lo dudoso queda en
   ``dudosas.jsonl`` con el motivo, para revisarlo o volver a preguntar, y **no se entrena**.

**Turnos de la app** (un modelo de visión por turno en la GTX 1060): el panel trabaja por **etapas**
(``ETAPAS``), cada una con un solo modelo en la VRAM y su resultado guardado en disco. ``siguiente_etapa``
dice qué modelo cargar en el próximo turno; ``cerrar`` combina lo que haya cuando terminan todas.

**Modo tutor** (``VisionTutorSession``, como ``hope/teacher/tutor.py``): Hope intenta describir o contestar;
si se equivoca, el tutor mira la imagen y el intento y da una **pista sin la respuesta** (``safe_hint``);
Hope reintenta. La respuesta verificada (procedural o confirmada por dos tutores) y el reintento corregido
quedan como datos de entrenamiento.

Uso::

    python -m hope.teacher.vision_tutores anotar --images fotos --etapa siguiente      # una etapa por turno
    python -m hope.teacher.vision_tutores anotar --images fotos --etapa todas         # si todo entra junto
    python -m hope.teacher.vision_tutores tutor --resume runs/hope --n 50              # modo tutor
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from hope.data.vision import IMG_TAG, ImageStore, ImageTextItem
from hope.teacher.client import OpenAICompatibleTeacher, Teacher, TeacherConfig
from hope.teacher.vision import _png, image_message, parse_annotation

ETAPAS = ("principal", "segundo", "florence", "ocr", "verificador")

PROMPT_ANOTAR = """Sos tutor de Hope, un modelo que está aprendiendo a ver. Analizá la imagen y respondé SOLO con
un objeto JSON con estas claves:
- "tipo": foto | documento | grafico | diagrama | codigo | captura | otro
- "descripcion_es": descripción detallada en español; "description_en": la misma en inglés
- "texto_visible": todo el texto legible, tal cual ("" si no hay)
- "objetos": lista de {"nombre", "nombre_en", "cantidad", "posicion"} (nombre_en: sustantivo simple en inglés)
- "datos": valores que se puedan extraer (tablas, gráficos, circuitos); {} si no hay
- "preguntas": 3 a 5 preguntas con respuesta CORTA y verificable mirando la imagen, cada una con
  "pregunta", "respuesta" y "razonamiento" (paso a paso)
No inventes lo que no se ve."""

PROMPT_RESPONDER = """Mirá la imagen y respondé la pregunta con una respuesta corta (un número, una palabra o una
frase breve). Respondé SOLO la respuesta.
Pregunta: {pregunta}"""

PROMPT_PISTA = """Sos tutor de Hope, un modelo que está aprendiendo a ver. Mirá la imagen.
Pregunta: {pregunta}
Respuesta correcta (NO la digas): {respuesta}
Intento de Hope: {intento}
Explicá en una o dos frases qué miró mal Hope y dale una PISTA para que lo vuelva a intentar,
sin decir la respuesta. Respondé solo la pista."""


# ---------------------------------------------------------------------------
# Tutores (backends). Todos se cargan perezosamente: importar este módulo no pide transformers.
# ---------------------------------------------------------------------------
@dataclass
class TutorInfo:
    nombre: str
    modelo: str = ""
    licencia: str = ""
    entrenar_con_salidas: bool | str = True      # True | False | "revisar"
    vram_gb: float = 0.0

    @property
    def train_ok(self) -> bool:
        return self.entrenar_con_salidas is True


class VLMTutor:
    """Modelo de visión y lenguaje con API de chat compatible con OpenAI (llama-server con
    ``--mmproj``, LM Studio, vLLM, Ollama). Las consultas pasan por el guardián (``Teacher.ask``)."""

    def __init__(self, teacher: Teacher, info: TutorInfo, max_side: int = 768):
        self.teacher, self.info, self.max_side = teacher, info, max_side

    def _ask(self, img, prompt: str, purpose: str) -> str:
        return self.teacher.ask(image_message(_png(img, self.max_side), prompt), purpose=purpose).text

    def annotate(self, img) -> dict:
        return parse_annotation(self._ask(img, PROMPT_ANOTAR, "vision.tutores.anotar"))

    def answer(self, img, pregunta: str) -> str:
        return self._ask(img, PROMPT_RESPONDER.format(pregunta=pregunta), "vision.tutores.responder").strip()

    def hint(self, img, pregunta: str, intento: str, respuesta: str) -> str:
        return self._ask(img, PROMPT_PISTA.format(pregunta=pregunta, intento=intento[:400], respuesta=respuesta),
                         "vision.tutores.pista").strip()


class TesseractOCR:
    """OCR con el programa ``tesseract`` (Apache 2.0). Sin dependencias de Python."""

    def __init__(self, info: TutorInfo, langs: str = "spa+eng", binary: str = "tesseract"):
        self.info, self.langs, self.binary = info, langs, binary

    def read(self, img) -> str:
        out = subprocess.run([self.binary, "stdin", "stdout", "-l", self.langs], input=_png(img, 2000),
                             capture_output=True, timeout=120, check=True)
        return out.stdout.decode("utf-8", errors="replace").strip()


class PaddleOCRTutor:
    """OCR con PaddleOCR (Apache 2.0); se importa al primer uso."""

    def __init__(self, info: TutorInfo, lang: str = "es"):
        self.info, self.lang, self._ocr = info, lang, None

    def read(self, img) -> str:
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(lang=self.lang)
        res = self._ocr.ocr(np.asarray(_pil(img)))
        return "\n".join(line[1][0] for page in (res or []) for line in (page or []))


class Florence2Tutor:
    """Florence-2 (MIT) con transformers: OCR, detección con cajas y descripción."""

    def __init__(self, info: TutorInfo, device: str = "cpu"):
        self.info, self.device, self._m = info, device, None

    def _run(self, img, task: str):
        if self._m is None:
            from transformers import AutoModelForCausalLM, AutoProcessor
            self._p = AutoProcessor.from_pretrained(self.info.modelo, trust_remote_code=True)
            self._m = AutoModelForCausalLM.from_pretrained(self.info.modelo, trust_remote_code=True).to(self.device)
        im = _pil(img)
        inp = self._p(text=task, images=im, return_tensors="pt").to(self.device)
        ids = self._m.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"], max_new_tokens=512,
                               num_beams=3)
        text = self._p.batch_decode(ids, skip_special_tokens=False)[0]
        return self._p.post_process_generation(text, task=task, image_size=im.size)[task]

    def read(self, img) -> str:
        return str(self._run(img, "<OCR>")).strip()

    def detect(self, img) -> list[dict]:
        r = self._run(img, "<OD>")
        return [{"label": lab, "box": [round(float(v), 1) for v in box]} for lab, box in zip(r["labels"], r["bboxes"])]

    def caption(self, img) -> str:
        return str(self._run(img, "<MORE_DETAILED_CAPTION>")).strip()

    def unload(self) -> None:
        self._m = None


class SigLIPScorer:
    """Coincidencia imagen ↔ texto con SigLIP (Apache 2.0; sigmoide) o CLIP (MIT; softmax)."""

    def __init__(self, info: TutorInfo, device: str = "cpu"):
        self.info, self.device, self._m = info, device, None

    def score(self, img, texts: list[str]) -> list[float]:
        import torch
        if self._m is None:
            from transformers import AutoModel, AutoProcessor
            self._p = AutoProcessor.from_pretrained(self.info.modelo)
            self._m = AutoModel.from_pretrained(self.info.modelo).to(self.device).eval()
        inp = self._p(text=texts, images=_pil(img), padding="max_length", truncation=True,
                      return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self._m(**inp).logits_per_image[0]
        probs = logits.sigmoid() if "siglip" in self.info.modelo.lower() else logits.softmax(-1)
        return [float(p) for p in probs]


def _pil(img):
    from PIL import Image
    if isinstance(img, (str, Path)):
        return Image.open(img).convert("RGB")
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img)).convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    return img.convert("RGB")


def load_panel(path: str | Path = "configs/tutores_vision.yaml", guardian=None, *,
               almacen: str | Path | None = None) -> "TutorPanel":
    """Arma el panel desde la configuración. Los tutores sin servidor o sin paquetes no se cargan hasta usarse."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tutors = {}
    for rol, t in (cfg.get("tutores") or {}).items():
        if not t or t.get("activo", True) is False:
            continue
        info = TutorInfo(rol, t.get("modelo", ""), t.get("licencia", ""), t.get("entrenar_con_salidas", "revisar"),
                         float(t.get("vram_gb", 0)))
        kind = t["tipo"]
        if kind == "vlm_openai":
            tc = TeacherConfig(endpoint=t["endpoint"], model=t.get("alias", t["modelo"]),
                               temperature=float(t.get("temperatura", 0.1)), max_tokens=int(t.get("max_tokens", 1536)),
                               timeout_s=float(t.get("timeout_s", 600)))
            tutors[rol] = VLMTutor(OpenAICompatibleTeacher(tc, guardian), info)
        elif kind == "florence2":
            tutors[rol] = Florence2Tutor(info, t.get("device", "cpu"))
        elif kind == "tesseract":
            tutors[rol] = TesseractOCR(info, t.get("idiomas", "spa+eng"))
        elif kind == "paddleocr":
            tutors[rol] = PaddleOCRTutor(info, t.get("idioma", "es"))
        elif kind in ("siglip", "clip"):
            tutors[rol] = SigLIPScorer(info, t.get("device", "cpu"))
        else:
            raise ValueError(f"tipo de tutor desconocido: {kind}")
    store = ImageStore(almacen or cfg.get("almacen", "data/vision"), int(cfg.get("tamano", 224)))
    return TutorPanel(tutors, store, umbrales=cfg.get("umbrales") or {})


# ---------------------------------------------------------------------------
# Verificación cruzada
# ---------------------------------------------------------------------------
def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower().strip(" .;:«»\"'")


def _char_acc(a: str, b: str) -> float:
    from hope.eval.vision_bench import char_accuracy
    return char_accuracy(_norm(a), _norm(b))


_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def answers_agree(a: str, b: str, rel: float = 0.02) -> bool:
    """¿Dos respuestas cortas dicen lo mismo? Números con tolerancia relativa; texto normalizado."""
    na, nb = _NUM.findall(str(a)), _NUM.findall(str(b))
    if na and nb and (_NUM.fullmatch(_norm(a)) or _NUM.fullmatch(_norm(b))):
        x, y = float(na[0].replace(",", ".")), float(nb[0].replace(",", "."))
        return abs(x - y) <= rel * max(abs(x), abs(y), 1e-9)
    def words(s):                                   # sin artículos ni signos: «El círculo rojo.» = «círculo rojo»
        return [w for w in re.findall(r"\w+", _norm(s)) if w not in _ARTICLES]
    wa, wb = words(a), words(b)
    return bool(wa) and wa == wb


_ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas", "the", "an"}   # «a» no: puede ser la respuesta (barra A)


def _singular(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


@dataclass
class Verdict:
    id: str
    estado: str                          # verificado | parcial | dudoso | solo_verificacion
    anotacion: dict                      # solo los campos verificados
    dudas: list[str] = field(default_factory=list)
    chequeos: dict = field(default_factory=dict)


class TutorPanel:
    """Varios tutores locales que se controlan entre sí. ``tutors``: rol → tutor (ver ``ETAPAS``)."""

    DEFAULT_UMBRALES = {"ocr_min": 0.85, "siglip_min": 0.05, "conteo_tolerancia": 1}

    def __init__(self, tutors: dict, store: ImageStore, *, umbrales: dict | None = None, source: str = "tutores"):
        if "principal" not in tutors:
            raise ValueError("falta el tutor principal")
        self.tutors, self.store, self.source = tutors, store, source
        self.u = {**self.DEFAULT_UMBRALES, **(umbrales or {})}
        self.dir = store.root / "tutores"
        self.dir.mkdir(parents=True, exist_ok=True)

    # --- etapas (un modelo por turno) -------------------------------------------------
    def _cache(self, etapa: str) -> dict[str, object]:
        f = self.dir / f"{etapa}.jsonl"
        if not f.exists():
            return {}
        return {e["id"]: e["salida"] for e in map(json.loads, f.read_text(encoding="utf-8").splitlines()) if e}

    def _save(self, etapa: str, sid: str, out) -> None:
        with (self.dir / f"{etapa}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": sid, "salida": out}, ensure_ascii=False) + "\n")

    def etapas(self) -> list[str]:
        return [e for e in ETAPAS if e in self.tutors]

    def siguiente_etapa(self, images: list) -> str | None:
        """La primera etapa con imágenes pendientes (qué modelo cargar en el próximo turno)."""
        sids = [self.store.put(i) for i in images]
        for e in self.etapas():
            done = self._cache(e)
            if any(s not in done for s in sids):
                return e
        return None

    def run_stage(self, etapa: str, images: list) -> dict:
        """Corre una etapa (un solo modelo) sobre las imágenes que todavía no la tienen."""
        tutor = self.tutors[etapa]
        done = self._cache(etapa)
        prin = self._cache("principal")
        rep = {"etapa": etapa, "hechas": 0, "saltadas": 0, "errores": []}
        batch = [(self.store.put(img), img) for img in images]
        descs = [prin[s].get("description_en", "") for s, _ in batch if isinstance(prin.get(s), dict)]
        for sid, img in batch:
            if sid in done:
                rep["saltadas"] += 1
                continue
            ann = prin.get(sid)
            try:
                if etapa == "principal":
                    out = tutor.annotate(img)
                elif not isinstance(ann, dict):
                    rep["saltadas"] += 1                    # falta la anotación principal
                    continue
                elif etapa == "segundo":
                    out = [tutor.answer(img, q.get("pregunta", "")) for q in ann.get("preguntas", [])]
                elif etapa == "florence":
                    out = {"ocr": tutor.read(img), "objetos": tutor.detect(img)}
                elif etapa == "ocr":
                    out = tutor.read(img)
                elif etapa == "verificador":
                    own = ann.get("description_en", "")
                    others = [d for d in descs if d and d != own][:15]
                    out = {"propia": own, "puntajes": tutor.score(img, [own] + others)} if own else None
                else:
                    raise ValueError(etapa)
            except Exception as e:                       # noqa: BLE001 — se registra y se sigue
                rep["errores"].append(f"{sid}: {str(e)[:200]}")
                out = {"error": str(e)[:200]}
            self._save(etapa, sid, out)
            rep["hechas"] += 1
        if hasattr(tutor, "unload"):
            tutor.unload()                               # libera la VRAM para el próximo turno
        return rep

    # --- combinación -------------------------------------------------------------------
    def verify(self, sid: str) -> Verdict | None:
        ann = self._cache("principal").get(sid)
        if not isinstance(ann, dict) or "error" in ann:
            return None
        seg, flo = self._cache("segundo").get(sid), self._cache("florence").get(sid)
        ocr, ver = self._cache("ocr").get(sid), self._cache("verificador").get(sid)
        flo = flo if isinstance(flo, dict) and "error" not in flo else None
        out, dudas, ch = {"tipo": ann.get("tipo", "otro")}, [], {}
        # texto visible
        reads = [r for r in (ocr if isinstance(ocr, str) else None, flo and flo.get("ocr")) if r is not None]
        txt = ann.get("texto_visible", "") or ""
        if reads:
            best = max(_char_acc(txt, r) for r in reads) if txt else 0.0
            ch["texto"] = round(best, 3)
            if txt and best >= self.u["ocr_min"]:
                out["texto_visible"] = txt
            elif txt:
                dudas.append(f"texto: el OCR no lo confirma (coincidencia {best:.2f})")
            elif any(len(re.findall(r"\w", r)) >= 5 for r in reads):
                dudas.append("texto: el OCR ve texto que el tutor no transcribió")
        elif txt:
            dudas.append("texto: sin OCR para verificarlo")
        # objetos y cantidades
        if flo is not None:
            labels = [_singular(_norm(d["label"])) for d in flo.get("objetos", [])]
            # Florence nombra sin color («square»): se compara el total por sustantivo
            # (un cuadrado rojo + dos verdes = 3 «square»).
            groups: dict[str, list[dict]] = {}
            for o in ann.get("objetos", []) or []:
                groups.setdefault(" ".join(sorted(_singular(w) for w in _norm(o.get("nombre_en", "")).split())),
                                  []).append(o)
            ok_objs = []
            for key, objs in groups.items():
                name = set(key.split())
                n = sum(1 for lab in labels if name and name & set(lab.split()))
                want = sum(int(o.get("cantidad", 1)) if str(o.get("cantidad", 1)).isdigit() else 1 for o in objs)
                if n and abs(n - want) <= (0 if want <= 3 else self.u["conteo_tolerancia"]):
                    ok_objs += objs
                else:
                    dudas.append(f"objetos «{key or '?'}»: Florence ve {n}, el tutor dice {want}")
            if ok_objs:
                out["objetos"] = ok_objs
        elif ann.get("objetos"):
            dudas.append("objetos: sin detector para verificarlos")
        # preguntas: el segundo tutor contesta por su cuenta
        qs = ann.get("preguntas", []) or []
        if isinstance(seg, list) and len(seg) == len(qs):
            good = [q for q, a in zip(qs, seg) if answers_agree(q.get("respuesta", ""), a)]
            for q, a in zip(qs, seg):
                if q not in good:
                    dudas.append(f"pregunta «{q.get('pregunta')}»: principal «{q.get('respuesta')}», segundo «{a}»")
            ch["preguntas"] = f"{len(good)}/{len(qs)}"
            if good:
                out["preguntas"] = good
        elif qs:
            dudas.append("preguntas: sin segundo tutor para verificarlas")
        # descripción: SigLIP/CLIP contra las de las otras imágenes del lote
        if isinstance(ver, dict) and ver.get("puntajes"):
            sc = ver["puntajes"]
            ch["descripcion"] = round(sc[0], 4)
            if sc[0] >= self.u["siglip_min"] and sc[0] >= max(sc):
                out["descripcion_es"], out["description_en"] = ann.get("descripcion_es", ""), ann.get("description_en", "")
            else:
                dudas.append(f"descripción: el verificador no la confirma ({sc[0]:.3f})")
        elif ann.get("descripcion_es"):
            dudas.append("descripción: sin verificador")
        # datos: cada número tiene que aparecer en lo que leyó el OCR
        if ann.get("datos"):
            nums = _NUM.findall(json.dumps(ann["datos"], ensure_ascii=False))
            seen = " ".join(reads)
            if reads and nums and all(n in seen for n in nums):
                out["datos"] = ann["datos"]
            else:
                dudas.append("datos: no todos los valores aparecen en el OCR")
        verified = [k for k in out if k != "tipo"]
        principal = self.tutors["principal"]
        if not getattr(principal, "info", TutorInfo("principal")).train_ok:
            estado = "solo_verificacion"
            dudas.append(f"licencia: «{principal.info.modelo}» ({principal.info.licencia}) no permite entrenar "
                         "con sus salidas sin revisión")
        else:
            estado = "verificado" if verified and not dudas else ("parcial" if verified else "dudoso")
        return Verdict(sid, estado, out, dudas, ch)

    def cerrar(self, images: list) -> dict:
        """Combina las etapas hechas: lo verificado entra al almacén; lo dudoso, a ``dudosas.jsonl``."""
        rep = {"verificado": 0, "parcial": 0, "dudoso": 0, "solo_verificacion": 0, "sin_anotacion": 0}
        known = {e["id"] for e in self.store.entries() if e.get("fuente") == self.source}
        for img in images:
            sid = self.store.put(img)
            v = self.verify(sid)
            if v is None:
                rep["sin_anotacion"] += 1
                continue
            rep[v.estado] += 1
            if v.estado in ("verificado", "parcial") and sid not in known:
                self.store.add(sid, {**v.anotacion, "verificacion": v.chequeos}, self.source)
                known.add(sid)
            if v.dudas:
                with (self.store.root / "dudosas.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(v), ensure_ascii=False) + "\n")
        return rep

    def annotate(self, images: list) -> dict:
        """Todas las etapas seguidas (cuando todos los modelos entran juntos, p. ej. en la A770)."""
        etapas = [self.run_stage(e, images) for e in self.etapas()]
        return {"etapas": etapas, **self.cerrar(images)}


# ---------------------------------------------------------------------------
# Modo tutor: Hope intenta, el tutor da una pista, Hope reintenta
# ---------------------------------------------------------------------------
@dataclass
class Lesson:
    pregunta: str
    respuesta: str
    intento: str
    correcto_al_primero: bool
    pista: str = ""
    reintento: str = ""
    correcto_tras_pista: bool = False
    pista_filtrada: bool = False


class VisionTutorSession:
    """``hope_answer(img, prompt) → texto`` (ver ``hope_answer_fn``). ``tutor``: un ``VLMTutor``.
    La verdad viene de imágenes procedurales o de preguntas confirmadas por dos tutores."""

    def __init__(self, hope_answer: Callable, tutor, *, extract: Callable[[str], str] | None = None):
        from hope.eval.vision_bench import extract_answer
        self.hope_answer, self.tutor = hope_answer, tutor
        self.extract = extract or extract_answer
        self.lessons: list[Lesson] = []
        self.items: list[ImageTextItem] = []

    def lesson(self, img, pregunta: str, respuesta: str, razonamiento: str = "") -> Lesson:
        from hope.teacher.tutor import leaks_answer, safe_hint
        prompt = f"{IMG_TAG}\n{pregunta}\n"
        a1 = self.hope_answer(img, prompt)
        les = Lesson(pregunta, respuesta, a1, answers_agree(self.extract(a1), respuesta))
        gold = f"{razonamiento}\nRespuesta: {respuesta}" if razonamiento else respuesta
        tensor = _tensor(img)
        if not les.correcto_al_primero:
            raw = self.tutor.hint(img, pregunta, a1, respuesta)
            les.pista_filtrada = leaks_answer(raw, respuesta)
            les.pista = safe_hint(raw, respuesta)
            p2 = f"{prompt}Intento anterior: {a1.strip()[:300]}\nPista del tutor: {les.pista}\n"
            les.reintento = self.hope_answer(img, p2)
            les.correcto_tras_pista = answers_agree(self.extract(les.reintento), respuesta)
            # el reintento corregido: con el intento y la pista como contexto, la respuesta verificada
            self.items.append(ImageTextItem(tensor, p2, les.reintento if les.correcto_tras_pista else gold))
        self.items.append(ImageTextItem(tensor, prompt, gold))
        self.lessons.append(les)
        return les

    def run(self, problems: list[tuple]) -> dict:
        """``problems``: (imagen, pregunta, respuesta, razonamiento)."""
        for p in problems:
            self.lesson(*p)
        n = max(len(self.lessons), 1)
        first = sum(le.correcto_al_primero for le in self.lessons)
        fixed = sum(le.correcto_tras_pista for le in self.lessons)
        return {"lecciones": len(self.lessons), "correctas_al_primero": first / n,
                "correctas_tras_pista": (first + fixed) / n, "pistas_filtradas": sum(le.pista_filtrada for le in self.lessons),
                "muestras": len(self.items)}

    @staticmethod
    def hope_answer_fn(model, *, tools: bool = True, max_new: int = 200) -> Callable:
        from hope.data.vision import resize_square
        runner = None
        if tools:
            from hope.runtime.toolcalls import ToolRunner
            from hope.tools.builtin import default_catalog
            runner = ToolRunner(default_catalog())

        def answer(img, prompt: str) -> str:
            from hope.runtime.toolcalls import generate_with_tools
            vis = model.encode_image(resize_square(_tensor(img), model.cfg.vis_image)[None])
            if runner is not None:
                return generate_with_tools(model, prompt, runner, max_new=max_new, stop_after_line=None, vis=vis)
            return model.generate(prompt, max_new=max_new, vis=vis).decode("utf-8", errors="replace")
        return answer


def _tensor(img):
    import torch
    if isinstance(img, torch.Tensor):
        return img
    return torch.from_numpy(np.array(_pil(img), dtype=np.uint8)).permute(2, 0, 1).contiguous()


def problems_from_store(store: ImageStore, source: str = "tutores") -> list[tuple]:
    """Preguntas confirmadas por dos tutores (para el modo tutor con imágenes del usuario)."""
    out = []
    for e in store.entries():
        if e.get("fuente") != source:
            continue
        img = store.load(e["id"])
        for q in e["anotacion"].get("preguntas", []):
            out.append((img, q["pregunta"], str(q["respuesta"]), q.get("razonamiento", "")))
    return out


def problems_procedural(n: int, seed: int = 0, size: int = 224) -> list[tuple]:
    from hope.data.vision import GENERATORS, procedural
    out = []
    for arr, ann in procedural(n, seed, size, tuple(GENERATORS)):
        for q in ann.get("preguntas", [])[:1]:
            out.append((arr, q["pregunta"], str(q["respuesta"]), q.get("razonamiento", "")))
    return out


# ---------------------------------------------------------------------------
# Tutores simulados (pruebas)
# ---------------------------------------------------------------------------
class MockVLM:
    """VLM simulado: ``oracle(png) → anotación verdadera``; ``error_rate`` cambia respuestas a propósito."""

    def __init__(self, oracle, info: TutorInfo | None = None, error_rate: float = 0.0, seed: int = 0,
                 leaky: bool = False):
        import random
        self.oracle, self.error_rate, self.rnd, self.leaky = oracle, error_rate, random.Random(seed), leaky
        self.info = info or TutorInfo("mock", "mock-vl", "Apache-2.0", True)
        self.calls = 0

    def _truth(self, img) -> dict:
        self.calls += 1
        return json.loads(json.dumps(self.oracle(img)))

    def annotate(self, img) -> dict:
        ann = self._truth(img)
        if self.rnd.random() < self.error_rate:
            for q in ann.get("preguntas", []):
                q["respuesta"] = str(q["respuesta"]) + "9"
            ann["texto_visible"] = (ann.get("texto_visible") or "")[::-1] + " zzz"
        return ann

    def answer(self, img, pregunta: str) -> str:
        q = next((q for q in self._truth(img).get("preguntas", []) if q["pregunta"] == pregunta), None)
        return "no sé" if q is None or self.rnd.random() < self.error_rate else str(q["respuesta"])

    def hint(self, img, pregunta, intento, respuesta) -> str:
        self.calls += 1
        return f"Mirá de nuevo con cuidado; la respuesta es {respuesta}." if self.leaky else \
            "Mirá de nuevo la imagen con cuidado y contá otra vez."


class MockOCR:
    def __init__(self, oracle, noise: str = ""):
        self.oracle, self.noise = oracle, noise
        self.info = TutorInfo("ocr", "mock-ocr", "Apache-2.0", True)

    def read(self, img) -> str:
        return (self.oracle(img).get("texto_visible") or "") + self.noise


class MockDetector:
    def __init__(self, oracle):
        self.oracle = oracle
        self.info = TutorInfo("florence", "mock-florence", "MIT", True)

    def read(self, img) -> str:
        return self.oracle(img).get("texto_visible") or ""

    def detect(self, img) -> list[dict]:
        return [{"label": o.get("nombre_en", ""), "box": [0, 0, 1, 1]}
                for o in self.oracle(img).get("objetos", []) for _ in range(int(o.get("cantidad", 1)))]


class MockScorer:
    def __init__(self, oracle):
        self.oracle = oracle
        self.info = TutorInfo("verificador", "mock-siglip", "Apache-2.0", True)

    def score(self, img, texts: list[str]) -> list[float]:
        truth = self.oracle(img).get("description_en", "")
        return [0.9 if t == truth else 0.01 for t in texts]


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> dict:
    from guardian.cli import load as load_guardian
    from hope.teacher.vision import IMAGE_EXT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("anotar")
    a.add_argument("--images", required=True)
    a.add_argument("--etapa", default="siguiente", help=f"siguiente | todas | cerrar | {' | '.join(ETAPAS)}")
    t = sub.add_parser("tutor")
    t.add_argument("--resume", required=True)
    t.add_argument("--n", type=int, default=50)
    t.add_argument("--fuente", default="procedural", help="procedural | almacen")
    t.add_argument("--out", default=None, help="carpeta donde guardar las muestras corregidas (ImageStore)")
    for p in (a, t):
        p.add_argument("--config", default="configs/tutores_vision.yaml")
        p.add_argument("--guardian", default="configs/guardian.yaml")
    args = ap.parse_args(argv)
    panel = load_panel(args.config, load_guardian(args.guardian))
    if args.cmd == "anotar":
        images = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMAGE_EXT)
        etapa = panel.siguiente_etapa(images) if args.etapa == "siguiente" else args.etapa
        if etapa is None or etapa == "cerrar":
            rep = panel.cerrar(images)
        elif etapa == "todas":
            rep = panel.annotate(images)
        else:
            rep = panel.run_stage(etapa, images)
            rep["siguiente"] = panel.siguiente_etapa(images) or "cerrar"
    else:
        from hope.train.continual import TrainConfig, Trainer
        model = Trainer.resume(args.resume, TrainConfig(precision="fp32"), device="cpu").model
        probs = (problems_procedural(args.n, size=model.cfg.vis_image) if args.fuente == "procedural"
                 else problems_from_store(panel.store)[:args.n])
        ses = VisionTutorSession(VisionTutorSession.hope_answer_fn(model), panel.tutors["principal"])
        rep = ses.run(probs)
        if args.out:
            st = ImageStore(args.out, model.cfg.vis_image)
            with (Path(args.out) / "lecciones.jsonl").open("a", encoding="utf-8") as f:
                for it in ses.items:
                    sid = st.put(it.image.permute(1, 2, 0).numpy())
                    f.write(json.dumps({"id": sid, "consigna": it.prompt, "respuesta": it.answer},
                                       ensure_ascii=False) + "\n")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return rep


if __name__ == "__main__":
    main()
