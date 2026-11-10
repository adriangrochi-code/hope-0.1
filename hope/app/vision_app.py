"""Tutores de imágenes (``hope/teacher/vision_tutores.py``) en los turnos del maestro.

* **Catálogo:** ``configs/tutores_vision.yaml``. La pantalla de maestros muestra cada tutor con su licencia y
  ``entrenar_con_salidas`` (los que no lo permiten solo verifican), y la lista de modelos que no se usan.
* **Imágenes:** las que el usuario deja en ``<datos>/imagenes`` (o ``vision.imagenes`` en ``runs/app.json``).
  El almacén verificado queda en ``<datos>/vision``.
* **Un modelo de visión por turno** (GTX 1060): principal → segundo → florence → ocr → verificador. Cada turno de
  imágenes corre la etapa que falta en un proceso aparte (``hope.app.vision_turno``) y al terminar la última
  combina todo.
* **Modelos con servidor** (principal y segundo, chat compatible con OpenAI): si ya hay uno respondiendo en el
  puerto del yaml (LM Studio, llama-server propio) se usa ese; si no, la app arranca ``llama-server`` con el
  ``.gguf`` y su ``mmproj`` que encuentre en ``<modelos>/vision/<rol>/``. Si no hay ninguno, ese turno lo usa
  un maestro de texto.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
from pathlib import Path

import yaml

from hope.app import descargas, maestros
from hope.app.tareas import Tarea, correr_modulo
from hope.app.vision_turno import armar_panel, imagenes

CON_SERVIDOR = ("vlm_openai",)


def config_path(root: Path) -> Path:
    p = Path(root) / "configs" / "tutores_vision.yaml"
    return p if p.exists() else Path(__file__).resolve().parents[2] / "configs" / "tutores_vision.yaml"


def _cfg(app) -> dict:
    try:
        return yaml.safe_load(config_path(app.cfg.root).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def carpeta_imagenes(app) -> Path:
    v = (app.cfg.data.get("vision") or {}).get("imagenes")
    return app.cfg.ruta(v) if v else app.cfg.carpeta_datos / "imagenes"


def almacen(app) -> Path:
    return app.cfg.carpeta_datos / "vision"


def modelo_local(app, rol: str) -> tuple[Path, Path | None] | None:
    """El .gguf del tutor (y su mmproj) en <modelos>/vision/<rol>/."""
    d = maestros.carpeta_modelos(app) / "vision" / rol
    if not d.is_dir():
        return None
    ggufs = sorted(d.glob("*.gguf"))
    proj = [g for g in ggufs if "mmproj" in g.name.lower()]
    mods = [g for g in ggufs if g not in proj]
    return (mods[0], proj[0] if proj else None) if mods else None


def catalogo(app) -> dict:
    c = _cfg(app)
    tutores = []
    for rol, t in (c.get("tutores") or {}).items():
        t = t or {}
        ent = t.get("entrenar_con_salidas", "revisar")
        tutores.append({"rol": rol, "tipo": t.get("tipo", ""), "modelo": t.get("modelo") or t.get("tipo", ""),
                        "licencia": t.get("licencia", ""), "entrenar_con_salidas": ent, "entrena": ent is True,
                        "vram_gb": t.get("vram_gb"), "activo": t.get("activo", True) is not False,
                        "endpoint": t.get("endpoint"),
                        "local": bool(modelo_local(app, rol)) if t.get("tipo") in CON_SERVIDOR else None})
    return {"tutores": tutores, "no_usar": c.get("no_usar_para_entrenar") or [], "config": str(config_path(app.cfg.root))}


def yaml_turno(app) -> Path:
    """Copia del yaml de tutores con el almacén en la carpeta de datos de la app."""
    c = _cfg(app)
    c["almacen"] = str(almacen(app))
    p = app.cfg.root / "runs" / "tutores_vision_turno.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# Tutores de imágenes de este turno (lo escribe la aplicación)\n"
                 + yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def siguiente(app) -> str | None:
    """La etapa que falta (qué modelo cargar en el próximo turno de imágenes), o None."""
    imgs = imagenes(carpeta_imagenes(app))
    if not imgs or not _cfg(app).get("tutores"):
        return None
    return armar_panel(yaml_turno(app)).siguiente_etapa(imgs)


def tutor(app, rol: str) -> dict:
    return (_cfg(app).get("tutores") or {}).get(rol) or {}


def _puerto(endpoint: str) -> int:
    return urllib.parse.urlparse(endpoint).port or 80


def listo(app, rol: str) -> tuple[bool, str]:
    """¿Se puede correr esta etapa ahora? Los tutores con servidor necesitan uno respondiendo o el modelo local."""
    t = tutor(app, rol)
    if t.get("tipo") not in CON_SERVIDOR or os.environ.get("HOPE_VISION_SIMULADO") == "1":
        return True, ""
    if app.servidor_maestro.responde(_puerto(t.get("endpoint", ""))):
        return True, ""
    if modelo_local(app, rol) and _llama(app):
        return True, ""
    return False, (f"falta {t.get('modelo', rol)}: abrilo en LM Studio en {t.get('endpoint')} o dejá el .gguf y su "
                   f"mmproj en {maestros.carpeta_modelos(app) / 'vision' / rol}")


def _llama(app) -> Path | None:
    ml = app.cfg.data.get("maestro_local") or {}
    if ml.get("exe") and Path(ml["exe"]).exists():
        return Path(ml["exe"])
    return descargas.buscar_servidor(maestros.carpeta_modelos(app) / "llama.cpp")


class TareaVision(Tarea):
    """Una etapa de los tutores de imágenes (un solo modelo en la VRAM)."""

    nombre = "vision"

    def ejecutar(self, etapa: str | None = None) -> None:
        app = self.app
        imgs_dir = carpeta_imagenes(app)
        etapa = etapa or siguiente(app)
        if etapa is None:
            self.log(f"no hay imágenes pendientes en {imgs_dir}")
            return
        t = tutor(app, etapa)
        propio = False
        if t.get("tipo") in CON_SERVIDOR and os.environ.get("HOPE_VISION_SIMULADO") != "1":
            port = _puerto(t.get("endpoint", ""))
            if not app.servidor_maestro.responde(port):
                ml, exe = modelo_local(app, etapa), _llama(app)
                if not ml or not exe:
                    raise RuntimeError(listo(app, etapa)[1])
                gpu = descargas.tipo_gpu((app.cfg.data.get("calibracion") or {}).get("dispositivo"))
                r = app.servidor_maestro.iniciar(exe, ml[0], mmproj=ml[1], capas_gpu=99 if gpu != "cpu" else 0,
                                                 port=port, contexto=4096)
                if not r["ok"]:
                    raise RuntimeError(r["mensaje"])
                propio = True
        self.log(f"tutor de imágenes «{etapa}» ({t.get('modelo') or t.get('tipo')}) — {imgs_dir}")
        res: dict = {}

        def leer(line: str) -> None:
            if line.startswith("RESULTADO "):
                try:
                    res.update(json.loads(line[len("RESULTADO "):]))
                except json.JSONDecodeError:
                    pass
        try:
            code = correr_modulo(self, "hope.app.vision_turno",
                                 ["--root", str(app.cfg.root), "--config", str(yaml_turno(app)),
                                  "--images", str(imgs_dir), "--etapa", etapa], app.cfg.root, al_leer=leer)
        finally:
            if propio:
                app.servidor_maestro.detener()
        self.datos = {**res, "fecha": time.strftime("%Y-%m-%d %H:%M")}
        if code != 0 and not self.detener_pedido():
            raise RuntimeError(f"tutores de imágenes: la etapa {etapa} terminó con código {code}")


_CUENTA: dict = {}


def estado(app) -> dict:
    """Para la pantalla de maestros (se consulta cada 2 s: las imágenes se cuentan cada 30 s)."""
    tv = app.tareas.get("vision")
    d = carpeta_imagenes(app)
    t, n = _CUENTA.get(str(d), (0.0, 0))
    if time.time() - t > 30 or (tv and tv.ocupada()):
        n = len(imagenes(d))
        _CUENTA[str(d)] = (time.time(), n)
    return {**catalogo(app), "imagenes": n, "carpeta": str(d), "tarea": tv.to_dict(10) if tv else None}
