"""Configuración de la aplicación (``runs/app.json``) y plan efectivo que se lanza.

El plan elegido se copia a ``runs/planes/<plan>`` con las rutas de datos apuntando a la carpeta
elegida en la interfaz; en la PC de la A770, ``desde:`` apunta al último checkpoint real.
"""

from __future__ import annotations

import copy
import json
import re
import threading
from pathlib import Path

import yaml

PLANES = {
    "plan_gtx1060.yaml": "GTX 1060 · base simple hn1-200m de pocos días (principal)",
    "plan_gtx1060_500m_vision.yaml": "GTX 1060 · hn1-500m con visión (optimizador en la RAM, mucho más lento)",
    "plan_a770_continuacion.yaml": "Intel Arc A770 · continuar el modelo de la 1060",
    "plan_6m.yaml": "Intel Arc A770 · plan de 6 meses desde cero",
    "plan_cpu.yaml": "Solo CPU (pruebas)",
}

DEFAULT = {
    "version": 1,
    "usuario": "",
    "plan": "plan_gtx1060.yaml",
    "continuacion": False,
    "datos": {
        "carpeta": "data",
        "presupuesto_gb": 40.0,
        # Reparto del presupuesto (fracciones); la interfaz muestra los GB de cada fuente.
        "reparto": {"wiki_es": 0.40, "wiki_en": 0.28, "codigo": 0.14, "repos": 0.10, "propias": 0.08},
        "carpetas_codigo": [],          # vacía = la biblioteca estándar de Python de esta instalación
        "incluir_stdlib": True,
        "repos": [],
        "autor": [],
        "incluir_coautoria_ia": False,
        "propias": [],                  # libros, manuales, apuntes (txt, md, pdf con pdftotext…)
        "modelo_calidad": "",           # .npz de hope.data.quality (opcional)
        "umbral_calidad": 2.5,
        "hechas": {},
    },
    "maestro": {
        "endpoint": "http://127.0.0.1:8080/v1",
        "model": "qwen",
        "temperature": 0.3,
        "max_tokens": 2048,
        "timeout_s": 300,
        "episodios_por_tema": 2,
        "generados_por_tema": 20,
        "max_consultas": 5000,
        "incluir_inferidos": True,
        "mock_error": 0.0,
    },
    "entrenamiento": {"deseado": "detenido", "retomar_al_iniciar": True, "muestras_auto": True},
    "calibracion": None,
}

# Claves que la interfaz puede cambiar con «guardar_config» (lista fija).
EDITABLES = {
    "usuario": str, "plan": str,
    "datos.carpeta": str, "datos.presupuesto_gb": float, "datos.reparto": dict, "datos.carpetas_codigo": list,
    "datos.incluir_stdlib": bool, "datos.repos": list, "datos.autor": list, "datos.incluir_coautoria_ia": bool,
    "datos.propias": list, "datos.modelo_calidad": str, "datos.umbral_calidad": float,
    "maestro.endpoint": str, "maestro.model": str, "maestro.temperature": float, "maestro.max_tokens": int,
    "maestro.timeout_s": float, "maestro.episodios_por_tema": int, "maestro.generados_por_tema": int,
    "maestro.max_consultas": int, "maestro.incluir_inferidos": bool, "maestro.mock_error": float,
    "entrenamiento.retomar_al_iniciar": bool, "entrenamiento.muestras_auto": bool,
}


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "reparto" else v
    return out


class AppConfig:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "runs" / "app.json"
        self._lock = threading.RLock()
        data = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
        self.data = _merge(DEFAULT, data)

    def get(self, key: str, default=None):
        cur = self.data
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, key: str, value, save: bool = True) -> None:
        with self._lock:
            parts = key.split(".")
            cur = self.data
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = value
            if save:
                self.save()

    def update_from_ui(self, cambios: dict) -> list[str]:
        """Aplica solo claves de la lista fija, con su tipo. Devuelve las claves cambiadas."""
        hechos = []
        for key, value in cambios.items():
            if key not in EDITABLES:
                raise KeyError(f"clave no editable: {key}")
            kind = EDITABLES[key]
            if kind is float:
                value = float(value)
            elif kind is int:
                value = int(value)
            elif kind is bool:
                value = bool(value)
            elif kind is list:
                if isinstance(value, str):
                    value = [x.strip() for x in value.splitlines() if x.strip()]
                value = [str(x) for x in value]
            elif kind is dict:
                value = {str(k): float(v) for k, v in dict(value).items()}
            else:
                value = str(value).strip()
            if key == "plan" and not (re.fullmatch(r"plan_[\w.-]+\.yaml", value) and ".." not in value
                                      and (value in PLANES or (self.root / "configs" / value).exists())):
                raise ValueError(f"plan desconocido: {value}")
            self.set(key, value, save=False)
            hechos.append(key)
        self.save()
        return hechos

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)

    # --- rutas -----------------------------------------------------------------
    def ruta(self, p: str | Path) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() else self.root / p

    @property
    def carpeta_datos(self) -> Path:
        return self.ruta(self.data["datos"]["carpeta"] or "data")

    def plan_plantilla(self, nombre: str | None = None) -> Path:
        nombre = nombre or self.data["plan"]
        p = self.root / "configs" / nombre
        if not p.exists():
            p = Path(__file__).resolve().parents[2] / "configs" / nombre
        return p

    def leer_plan(self, nombre: str | None = None) -> dict:
        return yaml.safe_load(self.plan_plantilla(nombre).read_text(encoding="utf-8"))

    def salida(self, nombre: str | None = None) -> Path:
        return self.ruta(self.leer_plan(nombre).get("salida", "runs/hope"))


def _rebase(value: str, carpeta: str) -> str:
    """'data/pre/wiki_es:0.35' → '<carpeta>/pre/wiki_es:0.35' (solo si empieza con data/)."""
    if carpeta in ("", "data") or not value.startswith("data/"):
        return value
    return carpeta.rstrip("/\\").replace("\\", "/") + "/" + value[len("data/"):]


def plan_efectivo(cfg: AppConfig, nombre: str | None = None, escribir: bool = True) -> tuple[Path, dict]:
    """Escribe ``runs/planes/<plan>`` adaptado a esta PC y devuelve (ruta relativa a la raíz, plan)."""
    nombre = nombre or cfg.data["plan"]
    plan = cfg.leer_plan(nombre)
    carpeta = str(cfg.data["datos"]["carpeta"] or "data")
    propias = bool(cfg.data["datos"].get("propias"))
    for etapa in plan.get("etapas", []):
        if "fuentes" in etapa:
            fu = {k: _rebase(str(v), carpeta) for k, v in etapa["fuentes"].items()}
            if propias and "propias" not in fu and not etapa.get("enfriar"):
                fu["propias"] = _rebase("data/pre/propias:0.08", carpeta)
            etapa["fuentes"] = fu
        if isinstance(etapa.get("vision"), str):
            etapa["vision"] = _rebase(etapa["vision"], carpeta)
    if plan.get("desde"):
        # Continuación: el último checkpoint real de la máquina anterior (no siempre el de la etapa 2).
        origen = cfg.ruta("runs/hope") / "estado.json"
        if origen.exists():
            try:
                ult = json.loads(origen.read_text(encoding="utf-8")).get("ultimo_checkpoint")
                if ult:
                    plan["desde"] = str(ult).replace("\\", "/")
            except (OSError, json.JSONDecodeError):
                pass
    dest = cfg.root / "runs" / "planes" / nombre
    if not escribir:
        return dest.relative_to(cfg.root), plan
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("# Generado por la aplicación (python -m hope.app) a partir de configs/" + nombre + "\n"
                    + yaml.safe_dump(plan, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return dest.relative_to(cfg.root), plan
