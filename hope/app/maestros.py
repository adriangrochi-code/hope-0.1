"""Varios maestros: catálogo descargable, servidores locales (LM Studio, Ollama, llama-server), VRAM y rotación.

* ``catalogo(root)`` — ``configs/catalogo_maestros.yaml``: modelos GGUF con su licencia. Los del grupo
  «condiciones» (Llama, Gemma) y «confirmar» (NVIDIA) quedan desactivados hasta que el usuario acepte.
* ``detectar_servidores()`` — prueba LM Studio (127.0.0.1:1234), Ollama (11434) y llama-server (8080) con
  ``GET /v1/models`` y lista sus modelos.
* ``lista(app)`` — los maestros de este equipo: los del catálogo (descargados o por descargar), los de
  servidores agregados a mano y, si existe, ``configs/maestros.yaml``. Se guardan en ``runs/app.json``.
* ``entra_en_vram(...)`` — presupuesto de memoria: compartida (maestro + Hope a la vez) o por turnos.
* ``elegir(maestros, stats, turnos)`` — round-robin ponderado por la tasa de verificación medida: los que
  más aciertan explican más, pero todos aportan variedad.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

import yaml

PUERTOS = {"LM Studio": 1234, "Ollama": 11434, "llama-server": 8080}
HF = "https://huggingface.co/{repo}/resolve/main/{archivo}"
CONTEXTO_GB = 0.6                     # caché de contexto y búferes de llama.cpp con lotes de 4
HOPE_200M_GB = 3.5                    # entrenamiento de hn1-200m en fp32-8bit con checkpointing (medido ≈ 3–4 GB)
HOPE_500M_GB = 5.4


def catalogo(root: Path) -> list[dict]:
    p = Path(root) / "configs" / "catalogo_maestros.yaml"
    if not p.exists():
        p = Path(__file__).resolve().parents[2] / "configs" / "catalogo_maestros.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out = []
    for m in data.get("maestros", []):
        m = dict(m)
        m.setdefault("grupo", "permitido")
        m["url"] = HF.format(repo=m["repo"], archivo=m["archivo"])
        out.append(m)
    return out


def carpeta_modelos(app) -> Path:
    return Path(os.environ.get("HOPE_MODELOS") or app.cfg.root / "runs" / "modelos")


# ---------------------------------------------------------------------------
# Servidores locales compatibles con OpenAI
# ---------------------------------------------------------------------------
def modelos_de(endpoint: str, timeout: float = 1.5) -> list[str] | None:
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=timeout) as r:
            data = json.loads(r.read().decode())
        return [str(m.get("id")) for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
    except (OSError, ValueError):
        return None


def detectar_servidores(puertos: dict | None = None) -> list[dict]:
    out = []
    for nombre, port in (puertos or PUERTOS).items():
        ep = f"http://127.0.0.1:{port}/v1"
        mods = modelos_de(ep)
        if mods is not None:
            out.append({"servidor": nombre, "endpoint": ep, "modelos": mods,
                        "lms": nombre == "LM Studio" and shutil.which("lms") is not None})
    return out


# ---------------------------------------------------------------------------
# Lista de maestros de este equipo
# ---------------------------------------------------------------------------
def lista(app) -> list[dict]:
    cfg = app.cfg
    estado = cfg.data.get("maestros") or {}
    modelos = carpeta_modelos(app)
    out = []
    for m in catalogo(cfg.root):
        st = estado.get(m["id"], {})
        activo = st.get("activo", m.get("activo", False) and m["grupo"] == "permitido")
        if m["grupo"] != "permitido" and not st.get("licencia_aceptada"):
            activo = False
        archivo = modelos / m["archivo"]
        out.append({**m, "tipo": "catalogo", "activo": bool(activo), "descargado": archivo.exists(),
                    "ruta": str(archivo), "licencia_aceptada": bool(st.get("licencia_aceptada"))})
    for s in (cfg.data.get("maestros_servidor") or []):
        out.append({"id": s["id"], "nombre": s["nombre"], "tipo": "servidor", "endpoint": s["endpoint"],
                    "model": s["model"], "servidor": s.get("servidor", ""), "activo": bool(s.get("activo", True)),
                    "licencia": s.get("licencia", "la del modelo (revisala)"), "grupo": "servidor", "gb": s.get("gb")})
    externo = cfg.root / "configs" / "maestros.yaml"         # el de la rama maestro-local, si está
    if externo.exists():
        try:
            data = yaml.safe_load(externo.read_text(encoding="utf-8")) or {}
            for i, m in enumerate(data.get("maestros", []) if isinstance(data, dict) else []):
                if isinstance(m, dict) and m.get("endpoint") and m.get("model"):
                    mid = "yaml-" + str(m.get("id") or m.get("nombre") or i)
                    out.append({"id": mid, "nombre": m.get("nombre", m["model"]), "tipo": "servidor",
                                "endpoint": m["endpoint"], "model": m["model"], "activo": m.get("activo", True)
                                and estado.get(mid, {}).get("activo", True),
                                "licencia": m.get("licencia", "?"), "grupo": "servidor", "origen": "configs/maestros.yaml"})
        except (yaml.YAMLError, OSError):
            pass
    return out


def agregar_servidor(app, servidor: str, endpoint: str, model: str) -> dict:
    from hope.app.maestro import es_local
    if not es_local(endpoint):
        return {"ok": False, "mensaje": "solo servidores locales (127.0.0.1)"}
    lst = list(app.cfg.data.get("maestros_servidor") or [])
    mid = f"srv-{servidor.lower().replace(' ', '')}-{model}"[:80]
    if any(x["id"] == mid for x in lst):
        return {"ok": True, "mensaje": f"{model} ya estaba"}
    lst.append({"id": mid, "nombre": f"{model} ({servidor})", "servidor": servidor, "endpoint": endpoint,
                "model": model, "activo": True})
    app.cfg.set("maestros_servidor", lst)
    return {"ok": True, "mensaje": f"{model} vía {servidor} agregado como maestro"}


def activar(app, mid: str, activo: bool, acepto_licencia: bool = False) -> dict:
    cat = {m["id"]: m for m in catalogo(app.cfg.root)}
    est = dict(app.cfg.data.get("maestros") or {})
    if mid in cat:
        m = cat[mid]
        st = dict(est.get(mid, {}))
        if activo and m["grupo"] != "permitido" and not (st.get("licencia_aceptada") or acepto_licencia):
            return {"ok": False, "licencia": True, "mensaje": f"{m['nombre']}: {m.get('nota', m['licencia'])}. "
                                                              "Para activarlo, confirmá que aceptás esa licencia."}
        if acepto_licencia:
            st["licencia_aceptada"] = True
        st["activo"] = bool(activo)
        est[mid] = st
        app.cfg.set("maestros", est)
        return {"ok": True}
    lst = list(app.cfg.data.get("maestros_servidor") or [])
    for s in lst:
        if s["id"] == mid:
            s["activo"] = bool(activo)
            app.cfg.set("maestros_servidor", lst)
            return {"ok": True}
    if mid.startswith("yaml-"):
        est[mid] = {"activo": bool(activo)}
        app.cfg.set("maestros", est)
        return {"ok": True}
    return {"ok": False, "mensaje": "maestro desconocido"}


def borrar_descarga(app, mid: str) -> dict:
    """Borra el GGUF de un maestro del catálogo desactivado (libera disco)."""
    m = next((x for x in lista(app) if x["id"] == mid and x["tipo"] == "catalogo"), None)
    if m is None:
        return {"ok": False, "mensaje": "maestro desconocido"}
    if m["activo"]:
        return {"ok": False, "mensaje": "desactivalo primero"}
    p = carpeta_modelos(app) / m["archivo"]                 # nombre del catálogo, dentro de la carpeta de modelos
    gb = p.stat().st_size / 1e9 if p.exists() else 0
    p.unlink(missing_ok=True)
    p.with_name(p.name + ".part").unlink(missing_ok=True)
    return {"ok": True, "mensaje": f"se liberaron {gb:.1f} GB"}


# ---------------------------------------------------------------------------
# VRAM
# ---------------------------------------------------------------------------
def necesita_gb(m: dict) -> float:
    return float(m.get("gb") or 2.5) + CONTEXTO_GB


def entra_en_vram(m: dict, vram_gb: float, *, compartida: bool, plan: str = "") -> dict:
    """¿Entra el maestro? ``compartida``: Hope entrena a la vez en la misma GPU (si no, turnos: GPU entera)."""
    hope = (HOPE_500M_GB if "500m" in plan else HOPE_200M_GB) if compartida else 0.0
    need = necesita_gb(m) + hope
    ok = vram_gb > 0 and need <= 0.92 * vram_gb
    out = {"entra": ok, "necesita_gb": round(need, 1), "vram_gb": vram_gb}
    if not ok and compartida:
        out["consejo"] = ("No entran los dos en la GPU a la vez: usá el modo turnos (recomendado), o en LM Studio bajá "
                          "el «GPU offload» del modelo para que parte quede en la CPU.")
    elif not ok:
        out["consejo"] = "Ni en su turno entra en esta GPU: elegí un maestro más chico o usalo en la CPU (lento)."
    return out


# ---------------------------------------------------------------------------
# Rotación
# ---------------------------------------------------------------------------
def tasa(stats: dict, mid: str) -> float | None:
    s = stats.get(mid) or {}
    n = s.get("aceptados", 0) + s.get("rechazados", 0)
    return s.get("aceptados", 0) / n if n else None


def elegir(maestros: list[dict], stats: dict, turnos: dict) -> dict | None:
    """Round-robin ponderado: peso = 0,3 + tasa de verificación (sin medir: 0,8). Se elige el que menos
    turnos tiene en proporción a su peso; así todos explican, y los que más aciertan, más seguido."""
    cands = [m for m in maestros if m.get("activo") and (m["tipo"] != "catalogo" or m.get("descargado"))]
    if not cands:
        return None

    def peso(m):
        t = tasa(stats, m["id"])
        return 0.3 + (0.8 if t is None else t)
    # En empate, el que tuvo menos turnos (más variedad).
    return min(cands, key=lambda m: (round((turnos.get(m["id"], 0) + 1) / peso(m), 9), turnos.get(m["id"], 0), m["id"]))


# ---------------------------------------------------------------------------
# LM Studio: cargar y descargar el modelo de la GPU (CLI «lms»)
# ---------------------------------------------------------------------------
def lms(*args: str) -> dict:
    exe = shutil.which("lms")
    if not exe:
        return {"ok": False, "mensaje": "no está el comando «lms» de LM Studio: cargá/descargá el modelo a mano "
                                        "(botón Eject en LM Studio) o usá un maestro del catálogo, que la app controla"}
    try:
        r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=600)
        return {"ok": r.returncode == 0, "mensaje": (r.stdout + r.stderr).strip()[-500:]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "mensaje": str(e)}
