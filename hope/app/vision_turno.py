"""Una etapa de los tutores de imágenes en el turno del maestro (lo lanza la aplicación).

Corre en un proceso aparte para que la VRAM quede libre al terminar (en la GTX 1060 entra un solo modelo de
visión por turno). Usa ``hope.teacher.vision_tutores``: ``panel.siguiente_etapa()`` dice qué modelo toca,
``panel.run_stage()`` lo corre sobre las imágenes que todavía no lo tienen y, cuando ya no falta ninguna etapa,
``panel.cerrar()`` combina todo (lo verificado al almacén; lo dudoso a ``dudosas.jsonl``).

    python -m hope.app.vision_turno --root . --config runs/tutores_vision_turno.yaml --images data/imagenes --etapa siguiente

Con ``HOPE_VISION_SIMULADO=1`` (solo pruebas) los tutores se reemplazan por simulados que leen la anotación
verdadera de ``<imagen>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def imagenes(carpeta: Path) -> list[Path]:
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        return []
    return sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in IMAGE_EXT)


def _oraculo(img) -> dict:
    p = Path(img).with_suffix(".json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def armar_panel(config: Path, guardian=None):
    from hope.teacher import vision_tutores as vt
    if os.environ.get("HOPE_VISION_SIMULADO") == "1":
        from hope.data.vision import ImageStore
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
        roles = [r for r, t in (cfg.get("tutores") or {}).items() if t and t.get("activo", True) is not False]
        hechos = {"principal": lambda: vt.MockVLM(_oraculo), "segundo": lambda: vt.MockVLM(_oraculo, seed=1),
                  "florence": lambda: vt.MockDetector(_oraculo), "ocr": lambda: vt.MockOCR(_oraculo),
                  "verificador": lambda: vt.MockScorer(_oraculo)}
        tutors = {r: hechos[r]() for r in roles if r in hechos}
        return vt.TutorPanel(tutors, ImageStore(cfg.get("almacen", "data/vision"), int(cfg.get("tamano", 224))),
                             umbrales=cfg.get("umbrales") or {})
    return vt.load_panel(config, guardian)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--config", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--etapa", default="siguiente")
    a = ap.parse_args(argv)
    root = Path(a.root)
    guardian = None
    if (root / "configs" / "guardian.yaml").exists():
        from hope.app.guardian_app import cargar
        guardian = cargar(root)
    panel = armar_panel(Path(a.config), guardian)
    imgs = imagenes(Path(a.images))
    etapa = panel.siguiente_etapa(imgs) if a.etapa == "siguiente" else a.etapa
    rep: dict = {"etapa": etapa, "imagenes": len(imgs)}
    if etapa and etapa != "cerrar":
        print(f"PROGRESO 0.05 {etapa}: {len(imgs)} imágenes", flush=True)
        rep.update(panel.run_stage(etapa, imgs))
    sig = panel.siguiente_etapa(imgs)
    rep["siguiente"] = sig
    if sig is None and imgs:
        print("PROGRESO 0.95 combinando las etapas (verificación cruzada)", flush=True)
        rep["cierre"] = panel.cerrar(imgs)
    print("RESULTADO " + json.dumps(rep, ensure_ascii=False), flush=True)
    return rep


if __name__ == "__main__":
    main()
