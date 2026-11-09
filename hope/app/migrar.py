"""Paso 5: pasar a la A770. Empaqueta ``runs/hope`` en un .zip y, en la otra PC, lo importa.

El .zip lleva el checkpoint, ``estado.json``, ``registro.jsonl``, el modelo ligero, los episodios
del maestro (``data/maestro``, son chicos), ``plan_a770_continuacion.yaml`` y ``LEEME_A770.txt``.
El corpus no va (pesa decenas de GB): en la A770 se reconstruye con el paso 2 o se copia aparte.
Al importar solo se aceptan rutas dentro de ``runs/hope/``, ``data/maestro/`` y ``LEEME_A770.txt``.
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path, PurePosixPath

from hope.app.tareas import Tarea
from hope.train.run import STOP_FILE

MARCA = "hope_continuacion.json"
LEEME = """Hope HN-1 — continuar en la Intel Arc A770
==========================================

Este archivo trae el entrenamiento hecho en la PC de la GTX 1060 ({origen}).
Último checkpoint: {ultimo}
Horas entrenadas: {horas}

En la PC de la A770:
1. Instalá Hope con instalar_windows.bat a770 (o ./instalar_linux.sh a770): PyTorch con XPU.
2. Abrí la aplicación: iniciar_hope.bat (o python -m hope.app).
3. Paso 5 «Pasar a la A770» → «Importar» y elegí este .zip.
   La aplicación detecta que es una continuación y elige configs/plan_a770_continuacion.yaml
   («desde:» apunta solo al último checkpoint de arriba).
4. Paso 1: calibrar (debe decir dispositivo xpu).
5. Paso 2: reconstruí el corpus en esta PC (o copiá la carpeta data/pre de la otra).
6. Paso 3: Iniciar. Los pesos fp32 de la 1060 pasan solos a bf16.
   Si el modelo vino con 200M, la etapa 3 pide crecer a 500M: aprobalo en el paso 6 (Guardián).

A mano, sin la aplicación:
   python -m hope.train.run --plan configs/plan_a770_continuacion.yaml
"""


class TareaExportar(Tarea):
    nombre = "exportar"

    def ejecutar(self, destino: str = "") -> None:
        app = self.app
        if app.entrenamiento.corriendo():
            raise RuntimeError("pausá el entrenamiento antes de empaquetar (el checkpoint podría estar escribiéndose)")
        cfg = app.cfg
        src = cfg.ruta("runs/hope")
        est_p = src / "estado.json"
        if not est_p.exists():
            raise FileNotFoundError(f"no hay entrenamiento para pasar: falta {est_p}")
        est = json.loads(est_p.read_text(encoding="utf-8"))
        if not est.get("ultimo_checkpoint"):
            raise FileNotFoundError("todavía no hay ningún checkpoint guardado")
        stamp = time.strftime("%Y%m%d-%H%M")
        dest = Path(destino).expanduser() if destino else cfg.root / "runs" / "exportar" / f"hope_para_a770_{stamp}.zip"
        if dest.is_dir():
            dest = dest / f"hope_para_a770_{stamp}.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        files: list[tuple[Path, str]] = []
        for f in sorted(src.rglob("*")):
            if f.is_file() and f.name != STOP_FILE:
                files.append((f, "runs/hope/" + f.relative_to(src).as_posix()))
        maestro = cfg.carpeta_datos / "maestro"
        if maestro.is_dir():
            files += [(f, "data/maestro/" + f.relative_to(maestro).as_posix())
                      for f in sorted(maestro.rglob("*")) if f.is_file() and not f.name.endswith(".tmp")]
        plan = cfg.plan_plantilla("plan_a770_continuacion.yaml")
        total = sum(f.stat().st_size for f, _ in files) or 1
        horas = round(sum(float(e.get("horas", 0)) for e in est.get("etapas", {}).values()), 1)
        marca = {"origen": cfg.data["plan"], "ultimo_checkpoint": est["ultimo_checkpoint"].replace("\\", "/"),
                 "horas": horas, "fecha": time.strftime("%Y-%m-%d %H:%M"), "etapas": est.get("etapas", {})}
        tmp = dest.with_suffix(".zip.part")
        done = 0
        self.log(f"empaquetando {len(files)} archivos ({total / 1e9:.2f} GB) en {dest}")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            z.writestr("LEEME_A770.txt", LEEME.format(origen=cfg.data["plan"], ultimo=marca["ultimo_checkpoint"],
                                                      horas=horas))
            z.writestr("runs/hope/" + MARCA, json.dumps(marca, indent=2, ensure_ascii=False))
            if plan.exists():
                z.write(plan, "configs/plan_a770_continuacion.yaml")
            for f, arc in files:
                if self.detener_pedido():
                    break
                # Los pesos casi no se comprimen: se guardan sin comprimir (rápido); el texto sí.
                comp = zipfile.ZIP_DEFLATED if f.suffix in (".jsonl", ".json", ".log", ".txt") else zipfile.ZIP_STORED
                z.write(f, arc, compress_type=comp)
                done += f.stat().st_size
                self.progreso = done / total
                self.detalle = f"{done / 1e9:.2f} / {total / 1e9:.2f} GB"
        if self.detener_pedido():
            tmp.unlink(missing_ok=True)
            self.log("empaquetado cancelado")
            return
        tmp.replace(dest)
        self.datos = {"zip": str(dest), "gb": round(dest.stat().st_size / 1e9, 2), "marca": marca}
        self.log(f"listo: {dest} ({dest.stat().st_size / 1e9:.2f} GB). Copialo a la PC de la A770.")


PERMITIDOS = ("runs/hope/", "data/maestro/")


class TareaImportar(Tarea):
    nombre = "importar"

    def ejecutar(self, zip_path: str = "") -> None:
        app = self.app
        p = Path(zip_path).expanduser()
        if not p.is_file() or p.suffix.lower() != ".zip":
            raise FileNotFoundError(f"no encuentro el .zip: {p}")
        if app.entrenamiento.corriendo():
            raise RuntimeError("hay un entrenamiento en curso en esta PC: pausalo antes de importar")
        cfg = app.cfg
        dest_runs = cfg.ruta("runs/hope")
        if (dest_runs / "estado.json").exists() and not (dest_runs / MARCA).exists():
            raise FileExistsError(f"{dest_runs} ya tiene un entrenamiento propio de esta PC; movelo antes de importar")
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            if "runs/hope/" + MARCA not in names:
                raise ValueError("el .zip no fue hecho por la aplicación de Hope (falta la marca de continuación)")
            members = []
            for n in names:
                pp = PurePosixPath(n)
                if n.endswith("/") or pp.is_absolute() or ".." in pp.parts or "\\" in n:
                    continue
                if n.startswith(PERMITIDOS) or n == "LEEME_A770.txt":
                    members.append(z.getinfo(n))
            total = sum(i.file_size for i in members) or 1
            done = 0
            for info in members:
                if self.detener_pedido():
                    return
                if info.filename.startswith("data/maestro/"):
                    target = cfg.carpeta_datos / "maestro" / info.filename[len("data/maestro/"):]
                elif info.filename == "LEEME_A770.txt":
                    target = cfg.root / "runs" / "LEEME_A770.txt"
                else:
                    target = cfg.root / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as out:
                    while chunk := src.read(1 << 22):
                        out.write(chunk)
                        done += len(chunk)
                        self.progreso = done / total
                        self.detalle = f"{done / 1e9:.2f} / {total / 1e9:.2f} GB"
        marca = json.loads((dest_runs / MARCA).read_text(encoding="utf-8"))
        cfg.set("plan", "plan_a770_continuacion.yaml", save=False)
        cfg.set("continuacion", True)
        self.datos = {"marca": marca}
        self.log(f"importado: último checkpoint {marca['ultimo_checkpoint']} ({marca['horas']} h). "
                 "Plan elegido: plan_a770_continuacion.yaml")


def continuacion(app) -> dict:
    """¿Esta PC continúa un entrenamiento de otra máquina?"""
    cfg = app.cfg
    marca = cfg.ruta("runs/hope") / MARCA
    info = {"es_continuacion": False}
    if marca.exists():
        try:
            info.update(json.loads(marca.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
        info["es_continuacion"] = True
    elif cfg.data.get("plan") == "plan_a770_continuacion.yaml" and (cfg.ruta("runs/hope") / "estado.json").exists():
        info["es_continuacion"] = True
    cal = cfg.data.get("calibracion") or {}
    info["xpu"] = cal.get("dispositivo") == "xpu"
    return info
