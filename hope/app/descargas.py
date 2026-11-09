"""Descargas reanudables y el maestro local listo para usar (sin intervención).

* ``bajar(url, destino, ...)`` — descarga con reanudación (encabezado ``Range`` sobre ``<destino>.part``),
  verificación de SHA-256 si se conoce y de espacio libre antes de empezar.
* ``MODELO_MAESTRO`` — Qwen2.5-1.5B-Instruct en GGUF Q4_K_M (licencia Apache 2.0, ~1,1 GB).
  El SHA-256 se toma del encabezado ``X-Linked-Etag`` de Hugging Face.
* ``elegir_llama(release, sistema, gpu)`` — el paquete de llama.cpp de la última versión publicada
  (GitHub releases de ggml-org/llama.cpp) para esta PC: CUDA en una NVIDIA, Vulkan en una Intel Arc,
  CPU si no hay GPU.
* ``ServidorMaestro`` — arranca y vigila ``llama-server`` (un ejecutable descargado a una carpeta fija;
  nunca un comando que llegue desde la web).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

MODELO_MAESTRO = {
    "nombre": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "licencia": "Apache 2.0",
    "gb": 1.2,
}
LLAMA_RELEASE = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
UA = {"User-Agent": "hope-app/1"}


class SinEspacio(OSError):
    pass


class _SinRedireccion(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def info_remota(url: str, timeout: float = 30) -> dict:
    """Tamaño y SHA-256 (si el servidor lo dice: Hugging Face manda X-Linked-Etag/X-Linked-Size)."""
    op = urllib.request.build_opener(_SinRedireccion)
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    try:
        r = op.open(req, timeout=timeout)
        h = r.headers
    except urllib.error.HTTPError as e:              # 302 con los encabezados de LFS
        h = e.headers
    etag = (h.get("X-Linked-Etag") or h.get("ETag") or "").strip('"').removeprefix("W/").strip('"')
    size = h.get("X-Linked-Size") or h.get("Content-Length")
    return {"sha256": etag if re.fullmatch(r"[0-9a-f]{64}", etag or "") else None,
            "bytes": int(size) if size and str(size).isdigit() else None}


def libre_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def bajar(url: str, destino: Path, *, sha: str | None = None, total: int | None = None, progreso=None,
          detener=None, margen_gb: float = 1.0, timeout: float = 60) -> Path:
    """Descarga con reanudación. ``progreso(bytes, total)``; ``detener()`` → True corta (queda el .part)."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and (sha is None or sha256(destino) == sha):
        return destino
    part = destino.with_name(destino.name + ".part")
    hecho = part.stat().st_size if part.exists() else 0
    if total and libre_gb(destino.parent) < (total - hecho) / 1e9 + margen_gb:
        raise SinEspacio(f"faltan {(total - hecho) / 1e9 + margen_gb:.1f} GB libres en {destino.parent}")
    headers = dict(UA)
    if hecho:
        headers["Range"] = f"bytes={hecho}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416 and hecho:                   # ya estaba completo
            part.replace(destino)
            return _verificar(destino, sha)
        raise
    with r:
        if hecho and r.status != 206:                 # el servidor no reanuda: desde cero
            hecho = 0
        modo = "ab" if hecho else "wb"
        largo = r.headers.get("Content-Length")
        total = total or (hecho + int(largo) if largo else None)
        with open(part, modo) as f:
            while True:
                if detener is not None and detener():
                    return part
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                hecho += len(chunk)
                if progreso:
                    progreso(hecho, total)
                if hecho % (256 << 20) < (1 << 20) and libre_gb(destino.parent) < margen_gb:
                    raise SinEspacio(f"el disco se llenó descargando {destino.name}")
    if total and part.stat().st_size < total:
        raise OSError(f"descarga incompleta de {destino.name} ({part.stat().st_size} de {total} bytes): se reanuda")
    part.replace(destino)
    return _verificar(destino, sha)


def _verificar(path: Path, sha: str | None) -> Path:
    if sha and sha256(path) != sha:
        bad = path.with_name(path.name + ".corrupto")
        path.replace(bad)
        raise OSError(f"{path.name}: el SHA-256 no coincide (se apartó como {bad.name}); se vuelve a bajar")
    return path


# ---------------------------------------------------------------------------
# llama.cpp
# ---------------------------------------------------------------------------
PATRONES = {
    ("windows", "cuda"): [r"^llama-.*-bin-win-cuda-12\.\d+-x64\.zip$", r"^cudart-llama-bin-win-cuda-12\.\d+-x64\.zip$"],
    ("windows", "arc"): [r"^llama-.*-bin-win-vulkan-x64\.zip$"],
    ("windows", "cpu"): [r"^llama-.*-bin-win-cpu-x64\.zip$|^llama-.*-bin-win-avx2-x64\.zip$"],
    ("linux", "cuda"): [r"^llama-.*-bin-ubuntu-vulkan-x64\.zip$|^llama-.*-bin-ubuntu-x64\.zip$"],
    ("linux", "arc"): [r"^llama-.*-bin-ubuntu-vulkan-x64\.zip$|^llama-.*-bin-ubuntu-x64\.zip$"],
    ("linux", "cpu"): [r"^llama-.*-bin-ubuntu-x64\.zip$"],
}


def tipo_gpu(dispositivo: str | None) -> str:
    return {"cuda": "cuda", "xpu": "arc"}.get(dispositivo or "", "cpu")


def elegir_llama(release: dict, sistema: str, gpu: str) -> list[dict]:
    """Assets del release que hacen falta (p. ej. binarios CUDA + su runtime). Vacío si no hay."""
    assets = release.get("assets", [])
    out = []
    for patron in PATRONES[(sistema, gpu)]:
        cands = [a for a in assets if re.match(patron, a.get("name", ""))]
        if not cands:
            return [] if not out else out
        # Con varias versiones de CUDA, la más baja (más compatible con drivers viejos de la 1060).
        cands.sort(key=lambda a: a["name"])
        out.append({"name": cands[0]["name"], "url": cands[0]["browser_download_url"], "bytes": cands[0].get("size")})
    return out


def ultimo_release(timeout: float = 30) -> dict:
    req = urllib.request.Request(LLAMA_RELEASE, headers={**UA, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def extraer(zip_path: Path, carpeta: Path) -> None:
    """Extrae sin salir de la carpeta (ignora rutas absolutas o con ..)."""
    carpeta.mkdir(parents=True, exist_ok=True)
    base = carpeta.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            dest = (carpeta / info.filename).resolve()
            if not str(dest).startswith(str(base)) or info.is_dir():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            if os.name != "nt":
                dest.chmod(0o755)


def buscar_servidor(carpeta: Path) -> Path | None:
    nombre = "llama-server.exe" if os.name == "nt" else "llama-server"
    cands = sorted(carpeta.rglob(nombre)) if carpeta.exists() else []
    return cands[0] if cands else None


class ServidorMaestro:
    """``llama-server`` con el modelo del maestro, en 127.0.0.1. Lo maneja la aplicación."""

    def __init__(self, carpeta_runs: Path):
        self.proc: subprocess.Popen | None = None
        self.log = carpeta_runs / "llama_server.log"
        self.port = 8080

    def responde(self, port: int | None = None) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port or self.port}/v1/models", timeout=3) as r:
                return r.status == 200
        except OSError:
            return False

    def corriendo(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def iniciar(self, exe: Path, modelo: Path, *, capas_gpu: int, contexto: int = 4096, hilos: int = 4,
                port: int = 8080, paralelo: int = 1) -> dict:
        self.port = port
        if self.responde():
            return {"ok": True, "mensaje": f"ya hay un servidor del maestro en el puerto {port}"}
        if not Path(exe).is_file() or not Path(modelo).is_file():
            return {"ok": False, "mensaje": "falta llama-server o el modelo del maestro"}
        self.log.parent.mkdir(parents=True, exist_ok=True)
        # Con varios pedidos a la vez (turno del maestro), cada uno tiene su parte del contexto.
        args = [str(exe), "-m", str(modelo), "--host", "127.0.0.1", "--port", str(port),
                "-c", str(contexto * max(1, paralelo)), "-np", str(max(1, paralelo)),
                "-ngl", str(capas_gpu), "-t", str(hilos)]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with open(self.log, "a", encoding="utf-8") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args)}\n")
            log.flush()
            self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=Path(exe).parent,
                                         creationflags=flags)
        for _ in range(120):
            if self.responde():
                return {"ok": True, "mensaje": f"maestro listo en http://127.0.0.1:{port}/v1"}
            if self.proc.poll() is not None:
                return {"ok": False, "mensaje": f"llama-server terminó (código {self.proc.returncode}); ver {self.log}"}
            time.sleep(1)
        return {"ok": False, "mensaje": "llama-server no respondió en 2 minutos"}

    def detener(self) -> None:
        if self.corriendo():
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
