"""Tareas de fondo de la aplicación: estado, progreso, registro y pausa cooperativa."""

from __future__ import annotations

import collections
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ESTADOS = ("inactiva", "en_curso", "pausando", "pausada", "hecha", "error")


class Tarea:
    """Una tarea que corre en un hilo. ``ejecutar`` revisa ``self.detener_pedido()`` a menudo:
    pausar es pedirle que termine en un punto seguro; reanudar es volver a iniciarla (cada tarea
    sabe saltear lo que ya hizo)."""

    nombre = "tarea"

    def __init__(self, app):
        self.app = app
        self.estado = "inactiva"
        self.progreso = 0.0
        self.detalle = ""
        self.error = ""
        self.registro: collections.deque[str] = collections.deque(maxlen=500)
        self.datos: dict = {}                       # resultados para la interfaz
        self._stop = threading.Event()
        self._hilo: threading.Thread | None = None
        self._lock = threading.Lock()

    # --- control -------------------------------------------------------------
    def ocupada(self) -> bool:
        return self._hilo is not None and self._hilo.is_alive()

    def iniciar(self, **params) -> dict:
        with self._lock:
            if self.ocupada():
                return {"ok": False, "mensaje": f"{self.nombre}: ya está en curso"}
            self._stop.clear()
            self.estado, self.error = "en_curso", ""
            self._hilo = threading.Thread(target=self._correr, args=(params,), daemon=True,
                                          name=f"hope-{self.nombre}")
            self._hilo.start()
        return {"ok": True}

    def pausar(self) -> dict:
        if not self.ocupada():
            return {"ok": False, "mensaje": f"{self.nombre}: no está en curso"}
        self._stop.set()
        self.estado = "pausando"
        self.log("pausa pedida: termina en el próximo punto seguro")
        return {"ok": True}

    def detener_pedido(self) -> bool:
        return self._stop.is_set()

    def esperar(self, timeout: float | None = None) -> None:
        if self._hilo is not None:
            self._hilo.join(timeout)

    def _correr(self, params: dict) -> None:
        try:
            self.ejecutar(**params)
            if self._stop.is_set():
                self.estado = "pausada"
                self.log("en pausa")
            elif self.estado == "en_curso":
                self.estado = "hecha"
                self.progreso = 1.0
        except Exception as e:                      # noqa: BLE001 — se muestra en la interfaz
            self.estado, self.error = "error", f"{type(e).__name__}: {e}"
            self.log("ERROR: " + self.error)
            for line in traceback.format_exc().splitlines()[-8:]:
                self.log("  " + line)

    def ejecutar(self, **params) -> None:          # pragma: no cover - lo implementan las subclases
        raise NotImplementedError

    # --- salida --------------------------------------------------------------
    def log(self, texto: str) -> None:
        for line in str(texto).splitlines() or [""]:
            self.registro.append(time.strftime("%H:%M:%S ") + line)

    def to_dict(self, lineas: int = 80) -> dict:
        return {"estado": self.estado, "progreso": round(float(self.progreso), 4), "detalle": self.detalle,
                "error": self.error, "registro": list(self.registro)[-lineas:], "datos": self.datos}


def entorno_hijo(root: Path, **extra: str) -> dict:
    """Variables para los subprocesos: el paquete hope importable aunque la raíz sea otra carpeta."""
    env = dict(os.environ)
    pkg = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra)
    return env


def correr_modulo(tarea: Tarea, modulo: str, args: list[str], root: Path, *, env_extra: dict | None = None,
                  al_leer=None) -> int:
    """Corre ``python -m <modulo>`` (siempre un módulo fijo de Hope), pasa su salida al registro y lo
    termina si se pide pausa. Las líneas ``PROGRESO <0-1> <texto>`` actualizan la barra."""
    if not modulo.startswith("hope."):
        raise ValueError("solo se ejecutan módulos de Hope")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    p = subprocess.Popen([sys.executable, "-u", "-m", modulo, *args], cwd=root, env=entorno_hijo(root, **(env_extra or {})),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", creationflags=flags)

    def vigilar():
        while p.poll() is None:
            if tarea.detener_pedido():
                p.terminate()
                return
            time.sleep(0.3)
    threading.Thread(target=vigilar, daemon=True).start()
    assert p.stdout is not None
    for line in p.stdout:
        line = line.rstrip("\n")
        if line.startswith("PROGRESO "):
            parts = line.split(" ", 2)
            try:
                tarea.progreso = float(parts[1])
            except ValueError:
                pass
            tarea.detalle = parts[2] if len(parts) > 2 else tarea.detalle
            continue
        if al_leer is not None:
            al_leer(line)
        tarea.log(line)
    return p.wait()
