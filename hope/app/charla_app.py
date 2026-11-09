"""Pestaña «Chatear»: maneja el proceso ``hope.app.charla`` y lo mantiene al día con el entrenamiento.

* Antes de cargar, copia ``ligero.pt`` a ``<salida>/charla/paso_<N>.pt`` (cuando el archivo dejó de
  cambiar): el chat nunca lee el archivo que el entrenamiento está escribiendo. Quedan las últimas 5 copias
  para comparar respuestas de dos checkpoints lado a lado.
* Se recarga sola cuando aparece un checkpoint nuevo y avisa «Hope se actualizó: paso N».
* Por defecto en la CPU, cuantizada a 8 bits y con 2 hilos; la GPU solo si sobra memoria (``nvidia-smi``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from hope.app import sistema
from hope.app.tareas import entorno_hijo

MAX_COPIAS = 5
ESTABLE_S = 1.0          # ligero.pt se escribe de forma atómica; esto solo evita copiar en ráfaga


class GestorCharla:
    def __init__(self, app):
        self.app = app
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._rid = 0
        self.cargado: dict | None = None
        self.avisos: list[dict] = []
        self.auto = True
        self.bits = 8
        self.dispositivo = "cpu"
        self.ultimo_error = ""
        self._visto = 0.0

    # --- proceso ---------------------------------------------------------------------
    def _salida(self) -> Path:
        return self.app.entrenamiento._salida()

    def carpeta(self) -> Path:
        return self._salida() / "charla"

    def vivo(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _arrancar(self) -> None:
        if self.vivo():
            return
        env = entorno_hijo(self.app.cfg.root, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        if self.dispositivo == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen([sys.executable, "-u", "-m", "hope.app.charla", "--salida", str(self._salida()),
                                      "--datos", str(self.app.cfg.carpeta_datos), "--raiz", str(self.app.cfg.root)],
                                     cwd=self.app.cfg.root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8", creationflags=flags)
        listo = self.proc.stdout.readline()
        if '"listo"' not in listo:
            raise RuntimeError("el proceso del chat no arrancó")

    def pedir(self, op: str, timeout: float = 600, **kw) -> dict:
        with self._lock:
            try:
                self._arrancar()
            except (OSError, RuntimeError) as e:
                return {"ok": False, "mensaje": str(e)}
            self._rid += 1
            rid = self._rid
            self.proc.stdin.write(json.dumps({"op": op, "rid": rid, **kw}, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            fin = time.time() + timeout
            while time.time() < fin:
                line = self.proc.stdout.readline()
                if not line:
                    self.proc = None
                    self.cargado = None
                    return {"ok": False, "mensaje": "el proceso del chat se cerró; se vuelve a abrir en el próximo mensaje"}
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("rid") == rid:
                    r.pop("rid", None)
                    return r
            return {"ok": False, "mensaje": "el chat tardó demasiado"}

    def cerrar(self) -> None:
        if self.vivo():
            try:
                self.proc.stdin.write('{"op": "salir"}\n')
                self.proc.stdin.flush()
                self.proc.wait(5)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()

    # --- modelos -----------------------------------------------------------------------------
    def paso_actual(self) -> int | None:
        from hope.app.entrenamiento import leer_jsonl
        reg = leer_jsonl(self._salida() / "registro.jsonl", 50)
        pasos = [r["paso"] for r in reg if isinstance(r.get("paso"), int)]
        return pasos[-1] if pasos else None

    def copias(self) -> list[dict]:
        d = self.carpeta()
        if not d.exists():
            return []
        out = []
        for f in sorted(d.glob("paso_*.pt"), key=lambda f: f.stat().st_mtime, reverse=True):
            out.append({"archivo": str(f), "etiqueta": f.stem.replace("_", " "), "fecha":
                        time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))})
        return out

    def _copiar(self) -> Path | None:
        """Copia ligero.pt cuando dejó de cambiar. Devuelve la copia (o None si no hay / está cambiando)."""
        src = self._salida() / "ligero.pt"
        if not src.exists() or time.time() - src.stat().st_mtime < ESTABLE_S:
            return None
        paso = self.paso_actual()
        d = self.carpeta()
        d.mkdir(parents=True, exist_ok=True)
        dest = d / f"paso_{paso if paso is not None else int(src.stat().st_mtime)}.pt"
        if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
            tmp = dest.with_suffix(".tmp")
            shutil.copy2(src, tmp)
            tmp.replace(dest)
        for viejo in sorted(d.glob("paso_*.pt"), key=lambda f: f.stat().st_mtime)[:-MAX_COPIAS]:
            viejo.unlink(missing_ok=True)
        return dest

    def gpu_libre_gb(self) -> float | None:
        nv = sistema.nvidia(cache_s=2)
        if not nv:
            return None
        g = nv[0]
        return (g["memoria_total_mb"] - g["memoria_usada_mb"]) / 1024 if g.get("memoria_total_mb") else None

    def cargar(self, bits: int | None = None, dispositivo: str | None = None, archivo: str | None = None) -> dict:
        if bits is not None:
            if int(bits) not in (0, 4, 8):
                return {"ok": False, "mensaje": "bits: 4, 8 o 0 (sin cuantizar)"}
            self.bits = int(bits)
        if dispositivo is not None:
            if dispositivo not in ("cpu", "cuda", "xpu"):
                return {"ok": False, "mensaje": "dispositivo: cpu, cuda o xpu"}
            if dispositivo == "cuda":
                libre = self.gpu_libre_gb()
                if libre is None or libre < 1.5:
                    return {"ok": False, "mensaje": f"no sobra memoria en la GPU ({libre or 0:.1f} GB libres): "
                                                    "el chat queda en la CPU para no frenar el entrenamiento"}
            if dispositivo != self.dispositivo:
                self.cerrar()                          # el proceso arranca con otra visibilidad de GPU
                self.proc = None
            self.dispositivo = dispositivo
        if archivo:
            p = Path(archivo)
            if p.parent.resolve() != self.carpeta().resolve() or not p.exists():
                return {"ok": False, "mensaje": "solo se pueden cargar las copias de la carpeta del chat"}
        else:
            p = self._copiar()
            if p is None and self.copias():
                p = Path(self.copias()[0]["archivo"])       # se está escribiendo uno nuevo: la última copia estable
            if p is None:
                return {"ok": False, "mensaje": "todavía no hay checkpoint (se guarda con el primero del entrenamiento)"}
        etiqueta = p.stem.replace("_", " ")
        r = self.pedir("cargar", modelo=str(p), bits=self.bits, dispositivo=self.dispositivo, etiqueta=etiqueta)
        if r.get("ok"):
            self.cargado = r
            self._visto = (self._salida() / "ligero.pt").stat().st_mtime if (self._salida() / "ligero.pt").exists() else 0
        else:
            self.ultimo_error = r.get("mensaje", "")
        return r

    def revisar(self) -> None:
        """Lo llama la vigilancia: si hay un checkpoint nuevo y el chat está abierto, recarga y avisa."""
        if not self.auto or not self.vivo() or self.cargado is None:
            return
        src = self._salida() / "ligero.pt"
        if not src.exists() or src.stat().st_mtime <= self._visto or time.time() - src.stat().st_mtime < ESTABLE_S:
            return
        r = self.cargar()
        if r.get("ok"):
            self.avisos.append({"t": time.time(), "texto": f"Hope se actualizó: {r.get('etiqueta')}"})
            self.avisos = self.avisos[-20:]

    def estado(self) -> dict:
        return {"vivo": self.vivo(), "cargado": self.cargado, "bits": self.bits, "dispositivo": self.dispositivo,
                "auto": self.auto, "copias": self.copias(), "avisos": self.avisos[-5:], "gpu_libre_gb": self.gpu_libre_gb(),
                "error": self.ultimo_error}
