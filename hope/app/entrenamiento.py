"""Pasos 3 y 4 (entrenamiento): lanza ``python -m hope.train.run --plan …`` como subproceso.

* **Pausa segura:** crea ``<salida>/DETENER``; el programa termina después del próximo checkpoint.
* **Retoma automática:** si el proceso se cae (código ≠ 0) se relanza solo (hasta 5 veces seguidas);
  si la aplicación se cerró o la PC se reinició mientras entrenaba, al abrirla sigue.
* La salida del proceso va a ``<salida>/salida_app.log`` y el pid a ``runs/app_entrenamiento.json``:
  si se cierra la aplicación, el entrenamiento sigue y al volver a abrirla se lo «adopta».
* Métricas: ``registro.jsonl`` y ``estado.json`` (los escribe ``hope.train.run``), ``nvidia-smi``
  y muestras de texto periódicas con ``load_light`` (subproceso en la CPU: no usa la GPU).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from hope.app import sistema
from hope.app.config import plan_efectivo
from hope.app.tareas import Tarea, correr_modulo, entorno_hijo
from hope.train.run import STOP_FILE

MAX_REINTENTOS = 5


def leer_jsonl(path: Path, ultimas: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for line in lines[-ultimas:] if ultimas else lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def cola(path: Path, n: int = 80) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 64_000))
        data = f.read().decode("utf-8", errors="replace")
    return data.splitlines()[-n:]


class Entrenamiento:
    """Estados: detenido · en_curso · pausando · pausado · esperando_aprobacion · reintentando ·
    terminado · error."""

    def __init__(self, app):
        self.app = app
        self.estado = "detenido"
        self.mensaje = ""
        self.proc: subprocess.Popen | None = None
        self.pid: int | None = None
        self.salida: Path | None = None
        self.plan_path: Path | None = None
        self.reintentos = 0
        self.inicio = 0.0
        self._reintentar_en = 0.0
        self._lock = threading.RLock()
        self._fin = threading.Event()
        self.muestras = TareaMuestras(app)
        self._ligero_visto = 0.0
        self._pidfile = app.cfg.root / "runs" / "app_entrenamiento.json"
        self._adoptar()
        self._hilo = threading.Thread(target=self._vigilar, daemon=True, name="hope-entrenamiento")
        self._hilo.start()

    # --- control ---------------------------------------------------------------
    def corriendo(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        return sistema.proceso_vivo(self.pid)

    def iniciar(self, automatico: bool = False) -> dict:
        with self._lock:
            if self.corriendo():
                return {"ok": False, "mensaje": "el entrenamiento ya está en curso"}
            cfg = self.app.cfg
            rel, plan = plan_efectivo(cfg)
            self.plan_path = rel
            self.salida = cfg.ruta(plan.get("salida", "runs/hope"))
            self.salida.mkdir(parents=True, exist_ok=True)
            (self.salida / STOP_FILE).unlink(missing_ok=True)
            log = open(self.salida / "salida_app.log", "a", encoding="utf-8")
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} inicio ({'automático' if automatico else 'manual'})"
                      f" plan {rel} ===\n")
            log.flush()
            flags = 0
            if os.name == "nt":
                flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            args = [sys.executable, "-u", "-m", "hope.train.run", "--plan", str(rel)]
            if self.app.dispositivo:
                args += ["--device", self.app.dispositivo]
            self.proc = subprocess.Popen(args, cwd=cfg.root, env=entorno_hijo(cfg.root), stdout=log,
                                         stderr=subprocess.STDOUT, creationflags=flags)
            log.close()
            self.pid = self.proc.pid
            self.inicio = time.time()
            self.estado, self.mensaje = "en_curso", ""
            self._pidfile.parent.mkdir(parents=True, exist_ok=True)
            self._pidfile.write_text(json.dumps({"pid": self.pid, "plan": str(rel), "salida": str(self.salida),
                                                 "inicio": self.inicio}), encoding="utf-8")
            cfg.set("entrenamiento.deseado", "en_curso")
            if not automatico:
                self.reintentos = 0
            return {"ok": True, "pid": self.pid}

    def pausar(self) -> dict:
        with self._lock:
            if not self.corriendo():
                return {"ok": False, "mensaje": "el entrenamiento no está en curso"}
            (self.salida / STOP_FILE).write_text("pausa pedida desde la aplicación\n", encoding="utf-8")
            self.estado = "pausando"
            self.mensaje = "se detiene después del próximo checkpoint (como mucho ~30 minutos)"
            self.app.cfg.set("entrenamiento.deseado", "detenido")
            return {"ok": True}

    def detener_ya(self) -> dict:
        with self._lock:
            if not self.corriendo():
                return {"ok": False, "mensaje": "el entrenamiento no está en curso"}
            self.app.cfg.set("entrenamiento.deseado", "detenido")
            if self.salida:
                (self.salida / STOP_FILE).write_text("detenido desde la aplicación\n", encoding="utf-8")
            if self.proc is not None:
                self.proc.terminate()
                try:
                    self.proc.wait(30)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            elif self.pid:
                sistema.terminar_proceso(self.pid)
            self.proc = None
            self._pidfile.unlink(missing_ok=True)
            self.estado = "pausado"
            self.mensaje = "detenido sin esperar al checkpoint: se pierde lo entrenado desde el último"
            return {"ok": True}

    def cerrar(self) -> None:
        self._fin.set()

    # --- vigilancia ---------------------------------------------------------------
    def _adoptar(self) -> None:
        """Si quedó un entrenamiento corriendo de una sesión anterior de la aplicación, lo sigue."""
        if not self._pidfile.exists():
            return
        try:
            info = json.loads(self._pidfile.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if sistema.proceso_vivo(info.get("pid")):
            self.pid, self.salida = int(info["pid"]), Path(info["salida"])
            self.plan_path, self.inicio = Path(info["plan"]), float(info.get("inicio", time.time()))
            self.estado = "pausando" if (self.salida / STOP_FILE).exists() else "en_curso"
            self.mensaje = "entrenamiento que ya estaba corriendo (iniciado en una sesión anterior)"

    def retomar_si_corresponde(self) -> None:
        cfg = self.app.cfg
        if (not self.corriendo() and cfg.get("entrenamiento.deseado") == "en_curso"
                and cfg.get("entrenamiento.retomar_al_iniciar", True)):
            self.mensaje = "retomado al abrir la aplicación"
            self.iniciar(automatico=True)

    def _vigilar(self) -> None:
        despierta = False
        while not self._fin.wait(1.0):
            try:
                corriendo = self.corriendo()
                if corriendo != despierta:
                    sistema.mantener_despierta(corriendo)
                    despierta = corriendo
                if corriendo:
                    self._muestra_auto()
                    continue
                with self._lock:
                    if self.estado in ("en_curso", "pausando"):
                        self._termino()
                    elif self.estado == "reintentando" and time.time() >= self._reintentar_en:
                        self.iniciar(automatico=True)
            except Exception as e:                      # noqa: BLE001 — que la vigilancia no muera
                self.mensaje = f"error en la vigilancia: {e}"

    def _termino(self) -> None:
        code = self.proc.poll() if self.proc is not None else None
        self.proc = None
        self._pidfile.unlink(missing_ok=True)
        est = self.estado_run()
        etapas = est.get("etapas", {})
        plan = self._plan()
        nombres = [e["nombre"] for e in plan.get("etapas", [])]
        pendiente = next((n for n in nombres if etapas.get(n, {}).get("solicitud")
                          and not etapas[n].get("aprobacion") and not etapas[n].get("hecha")), None)
        if nombres and all(etapas.get(n, {}).get("hecha") for n in nombres):
            self.estado, self.mensaje = "terminado", "todas las etapas del plan están hechas"
            self.app.cfg.set("entrenamiento.deseado", "detenido")
        elif self.salida and (self.salida / STOP_FILE).exists():
            self.estado = "pausado"
            self.mensaje = "en pausa: el último checkpoint quedó guardado"
        elif pendiente:
            self.estado = "esperando_aprobacion"
            self.mensaje = (f"la etapa {pendiente} necesita tu aprobación en el guardián "
                            f"(solicitud {etapas[pendiente]['solicitud']}); después, Reanudar")
            self.app.cfg.set("entrenamiento.deseado", "detenido")
        elif code not in (0, None) or (code is None and self.app.cfg.get("entrenamiento.deseado") == "en_curso"):
            if time.time() - self.inicio > 1800:
                self.reintentos = 0                     # corrió un buen rato: el contador vuelve a cero
            self.reintentos += 1
            if self.reintentos <= MAX_REINTENTOS:
                espera = 30 * self.reintentos
                self.estado = "reintentando"
                self._reintentar_en = time.time() + espera
                self.mensaje = (f"el proceso terminó con error (código {code}); se relanza en {espera} s "
                                f"(intento {self.reintentos} de {MAX_REINTENTOS}). Mirá el registro.")
            else:
                self.estado = "error"
                self.mensaje = f"el proceso falló {MAX_REINTENTOS} veces seguidas; revisá el registro"
                self.app.cfg.set("entrenamiento.deseado", "detenido")
        else:
            self.estado, self.mensaje = "detenido", "el proceso terminó"

    def _muestra_auto(self) -> None:
        if not self.app.cfg.get("entrenamiento.muestras_auto", True) or self.salida is None:
            return
        p = self.salida / "ligero.pt"
        if p.exists() and p.stat().st_mtime > self._ligero_visto and not self.muestras.ocupada():
            if self._ligero_visto:                      # no al arrancar: la primera la pide el usuario
                self.muestras.iniciar(modelo=str(p))
            self._ligero_visto = p.stat().st_mtime

    # --- lectura -------------------------------------------------------------------
    def _plan(self) -> dict:
        import yaml
        p = self.app.cfg.root / self.plan_path if self.plan_path else None
        if p and p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8"))
        return self.app.cfg.leer_plan()

    def _salida(self) -> Path:
        return self.salida or self.app.cfg.salida()

    def estado_run(self) -> dict:
        p = self._salida() / "estado.json"
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"etapas": {}}
        except (OSError, json.JSONDecodeError):
            return {"etapas": {}}

    def metricas(self) -> dict:
        salida = self._salida()
        plan = self._plan()
        est = self.estado_run()
        reg = leer_jsonl(salida / "registro.jsonl")
        etapas, offset, offsets = [], 0.0, {}
        actual = None
        for e in plan.get("etapas", []):
            es = est.get("etapas", {}).get(e["nombre"], {})
            horas = float(es.get("horas", 0.0))
            offsets[e["nombre"]] = offset
            item = {"nombre": e["nombre"], "horas_plan": e.get("horas"), "horas": round(horas, 4),
                    "hecha": bool(es.get("hecha")), "preset": e.get("preset"), "crecer_a": e.get("crecer_a"),
                    "solicitud": es.get("solicitud"), "aprobacion": es.get("aprobacion"),
                    "fuentes": list((e.get("fuentes") or {}).keys())}
            if not item["hecha"] and actual is None:
                actual = item
            etapas.append(item)
            offset += horas if not e.get("crecer_a") else 0.0
        serie, ultimo, lote = [], None, None
        for r in reg:
            if "lote_auto" in r:
                lote = r["lote_auto"]
            if "bpb_val" in r and r.get("etapa") in offsets:
                h = offsets[r["etapa"]] + float(r.get("horas", 0))
                serie.append({"h": round(h, 4), "etapa": r["etapa"], "bpb": r.get("bpb_val") or {},
                              "bpb_ent": r.get("bpb_entrenamiento"), "bytes_s": r.get("bytes_por_s"),
                              "paso": r.get("paso")})
                ultimo = r
        eta = None
        if actual and actual.get("horas_plan"):
            eta = max(float(actual["horas_plan"]) - actual["horas"], 0.0)
        gpu = sistema.nvidia()
        return {"estado": self.estado, "mensaje": self.mensaje, "pid": self.pid if self.corriendo() else None,
                "plan": str(self.plan_path or ""), "salida": str(salida), "etapas": etapas,
                "etapa_actual": actual["nombre"] if actual else None, "eta_horas": eta,
                "horas_totales": round(sum(e["horas"] for e in etapas), 2), "lote": lote,
                "paso": ultimo.get("paso") if ultimo else None,
                "bytes_por_s": ultimo.get("bytes_por_s") if ultimo else None,
                "serie": serie[-2000:], "gpu": gpu[0] if gpu else None, "reintentos": self.reintentos,
                "registro": cola(salida / "salida_app.log", 60), "muestras": self.ultimas_muestras(),
                "tarea_muestras": self.muestras.to_dict(10)}

    def ultimas_muestras(self) -> dict | None:
        p = self._salida() / "muestras.json"
        try:
            hist = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        except (OSError, json.JSONDecodeError):
            return None
        return hist[-1] if hist else None

    def saltar_a_etapa(self, nombre: str) -> dict:
        """Marca como hechas las etapas anteriores a ``nombre`` (p. ej. pasar a la del maestro)."""
        with self._lock:
            if self.corriendo():
                return {"ok": False, "mensaje": "pausá el entrenamiento primero"}
            plan = self._plan()
            nombres = [e["nombre"] for e in plan.get("etapas", [])]
            if nombre not in nombres:
                return {"ok": False, "mensaje": f"el plan no tiene la etapa {nombre}"}
            salida = self._salida()
            salida.mkdir(parents=True, exist_ok=True)
            est = self.estado_run()
            for n in nombres[:nombres.index(nombre)]:
                es = est.setdefault("etapas", {}).setdefault(n, {"horas": 0.0, "hecha": False})
                es["hecha"] = True
                es.setdefault("nota", f"terminada a mano desde la aplicación el {time.strftime('%Y-%m-%d %H:%M')}")
            (salida / "estado.json").write_text(json.dumps(est, indent=2, ensure_ascii=False), encoding="utf-8")
            return {"ok": True}


class TareaMuestras(Tarea):
    """Muestras de generación con el modelo ligero (subproceso en CPU, 2 hilos)."""

    nombre = "muestras"

    def ejecutar(self, modelo: str) -> None:
        sal = Path(modelo).parent / "muestras.json"
        code = correr_modulo(self, "hope.app.muestra", ["--modelo", modelo, "--salida", str(sal)],
                             self.app.cfg.root, env_extra={"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                                                           "MKL_NUM_THREADS": "2"})
        if code != 0 and not self.detener_pedido():
            raise RuntimeError(f"la generación de muestras terminó con código {code}")
