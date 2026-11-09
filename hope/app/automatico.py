"""«Empezar (automático)»: de cero a entrenando, sin intervención.

Pasos (cada uno con su progreso; lo hecho queda en ``runs/app.json`` → ``automatico`` y no se repite):

1. **dependencias** — comprueba que estén torch, numpy, pyyaml y pyarrow; instala las que falten (lista fija).
2. **calibración** — mide la PC y elige el plan recomendado.
3. **datos** — Wikipedia en español e inglés y código, con un presupuesto según el espacio libre del disco.
4. **calidad** — las reglas de calidad ya se aplican al construir; si hay un modelo de calidad, también.
5. **maestro** — baja Qwen2.5-1.5B-Instruct Q4_K_M (Apache 2.0) y llama.cpp, arranca ``llama-server``
   (en la GPU si entra junto al entrenamiento, si no en la CPU) y empieza a generar episodios verificados.
   Si algo de esto falla, Hope entrena igual sin maestro y lo avisa.
6. **entrenar** — lanza el plan.

Antes de cada paso se comprueba el espacio libre; si no alcanza, se pausa con un aviso. Las descargas
cortadas se reanudan.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from hope.app import descargas
from hope.app.tareas import Tarea

PASOS = ("dependencias", "calibracion", "datos", "calidad", "maestro", "entrenar")
NOMBRES = {"dependencias": "Instalar lo que falta", "calibracion": "Medir la PC", "datos": "Bajar los textos",
           "calidad": "Filtrar por calidad", "maestro": "Preparar el maestro", "entrenar": "Empezar a entrenar"}
NECESITA_GB = {"dependencias": 3.0, "calibracion": 0.5, "datos": None, "calidad": 0.1, "maestro": 2.5, "entrenar": 5.0}
PAQUETES = {"torch": "torch", "numpy": "numpy", "yaml": "pyyaml", "pyarrow": "pyarrow", "sympy": "sympy", "pint": "pint"}


class TareaAutomatica(Tarea):
    nombre = "automatico"

    def __init__(self, app):
        super().__init__(app)
        self.paso = None

    # --- estado persistente ------------------------------------------------------
    def hechos(self) -> dict:
        return dict(self.app.cfg.data.get("automatico") or {})

    def marcar(self, paso: str, info: str = "") -> None:
        h = self.hechos()
        h[paso] = {"fecha": time.strftime("%Y-%m-%d %H:%M"), "info": info}
        self.app.cfg.set("automatico", h)

    def resumen(self) -> list[dict]:
        h = self.hechos()
        return [{"paso": p, "nombre": NOMBRES[p], "hecho": p in h, "info": (h.get(p) or {}).get("info", ""),
                 "actual": p == self.paso and self.ocupada()} for p in PASOS]

    # --- ejecución ---------------------------------------------------------------------
    def ejecutar(self, rehacer: bool = False) -> None:
        if rehacer:
            self.app.cfg.set("automatico", {})
        for i, paso in enumerate(PASOS):
            if self.detener_pedido():
                return
            if paso in self.hechos():
                continue
            self.paso = paso
            self.progreso = i / len(PASOS)
            self.detalle = NOMBRES[paso]
            self.log(f"— {NOMBRES[paso]}")
            if not self._espacio(paso):
                self.estado = "pausada"
                self._stop.set()
                return
            info = getattr(self, "_" + paso)()
            if self.detener_pedido():
                return
            self.marcar(paso, info or "")
        self.paso = None
        self.detalle = "Hope está entrenando"

    def _espacio(self, paso: str) -> bool:
        cfg = self.app.cfg
        need = NECESITA_GB[paso]
        if paso == "datos":
            need = float(cfg.data["datos"]["presupuesto_gb"]) * 1.6
        libre = descargas.libre_gb(cfg.carpeta_datos if paso == "datos" else cfg.root / "runs")
        if libre < need:
            self.error = (f"Falta espacio para «{NOMBRES[paso]}»: hacen falta ~{need:.0f} GB y hay {libre:.0f} GB libres. "
                          "Liberá espacio o elegí otro disco y tocá Continuar.")
            self.log(self.error)
            return False
        return True

    def _esperar(self, tarea: Tarea, **params) -> None:
        r = tarea.iniciar(**params)
        if not r.get("ok") and not tarea.ocupada():
            raise RuntimeError(r.get("mensaje", f"no se pudo iniciar {tarea.nombre}"))
        while tarea.ocupada():
            if self.detener_pedido():
                tarea.pausar()
                tarea.esperar(60)
                return
            self.detalle = f"{NOMBRES[self.paso]}: {tarea.detalle}"
            time.sleep(0.5)
        if tarea.estado == "error":
            raise RuntimeError(f"{tarea.nombre}: {tarea.error}")

    # --- pasos ---------------------------------------------------------------------------
    def _dependencias(self) -> str:
        faltan = []
        for mod, pkg in PAQUETES.items():
            r = subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True)
            if r.returncode != 0:
                faltan.append(pkg)
        if not faltan:
            return "todo instalado"
        if "torch" in faltan:
            raise RuntimeError("falta PyTorch: corré instalar_windows.bat (o Hope.bat) para instalarlo con la "
                               "variante de tu placa de video")
        self.log("instalando " + " ".join(faltan))
        r = subprocess.run([sys.executable, "-m", "pip", "install", *faltan], capture_output=True, text=True)
        self.log(r.stdout[-2000:] + r.stderr[-2000:])
        if r.returncode != 0:
            raise RuntimeError("no se pudieron instalar: " + " ".join(faltan))
        return "instalado: " + ", ".join(faltan)

    def _calibracion(self) -> str:
        cal = self.app.cfg.data.get("calibracion") or {}
        if not cal.get("plan_recomendado"):
            self._esperar(self.app.tareas["calibracion"])
            cal = self.app.cfg.data.get("calibracion") or {}
        plan = cal.get("plan_recomendado")
        if plan and (self.app.cfg.root / "configs" / plan).exists():
            self.app.cfg.set("plan", plan)
        return f"{cal.get('dispositivo', 'cpu')} · plan {self.app.cfg.data['plan']}"

    def _datos(self) -> str:
        cfg = self.app.cfg
        d = cfg.data["datos"]
        libre = descargas.libre_gb(cfg.carpeta_datos)
        if not cfg.data.get("automatico_presupuesto_fijo"):
            # Unos 40 GB de texto alcanzan para la base; nunca más de un tercio de lo libre.
            gb = max(1.0, min(40.0, libre / 3))
            cfg.set("datos.presupuesto_gb", round(gb, 1))
        self.log(f"presupuesto de texto: {d['presupuesto_gb']} GB ({libre:.0f} GB libres)")
        self._esperar(self.app.tareas["datos"])
        return f"{cfg.data['datos']['presupuesto_gb']} GB de texto"

    def _calidad(self) -> str:
        m = self.app.cfg.data["datos"].get("modelo_calidad")
        return "reglas de calidad + modelo " + m if m else "reglas de calidad (Gopher/C4) aplicadas al construir"

    def _maestro(self) -> str:
        cfg = self.app.cfg
        try:
            return self._preparar_maestro()
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            self.log(f"el maestro no quedó listo ({e}); Hope entrena igual y el maestro se puede preparar después")
            return f"sin maestro por ahora: {e}"
        finally:
            cfg.save()

    def _preparar_maestro(self) -> str:
        cfg = self.app.cfg
        modelos = Path(os.environ.get("HOPE_MODELOS") or cfg.root / "runs" / "modelos")
        m = descargas.MODELO_MAESTRO
        gguf = modelos / m["nombre"]
        if not gguf.exists():
            info = descargas.info_remota(m["url"])
            self.log(f"bajando {m['nombre']} ({(info['bytes'] or 0) / 1e9:.1f} GB, {m['licencia']})")
            descargas.bajar(m["url"], gguf, sha=info["sha256"], total=info["bytes"], detener=self.detener_pedido,
                            progreso=lambda h, t: setattr(self, "detalle", f"modelo del maestro: {h / 1e9:.2f}"
                                                                            + (f" / {t / 1e9:.2f} GB" if t else " GB")))
            if self.detener_pedido():
                return ""
        dispositivo = (cfg.data.get("calibracion") or {}).get("dispositivo")
        gpu = descargas.tipo_gpu(dispositivo)
        sistema = "windows" if os.name == "nt" else "linux"
        carpeta_llama = modelos / "llama.cpp"
        exe = descargas.buscar_servidor(carpeta_llama)
        if exe is None:
            rel = descargas.ultimo_release()
            assets = descargas.elegir_llama(rel, sistema, gpu) or descargas.elegir_llama(rel, sistema, "cpu")
            if not assets:
                raise RuntimeError(f"no encontré llama.cpp para {sistema}/{gpu} en el release {rel.get('tag_name')}")
            for a in assets:
                z = modelos / "descargas" / a["name"]
                self.log(f"bajando {a['name']}")
                descargas.bajar(a["url"], z, total=a.get("bytes"), detener=self.detener_pedido,
                                progreso=lambda h, t, n=a["name"]: setattr(self, "detalle", f"{n}: {h / 1e6:.0f} MB"))
                if self.detener_pedido():
                    return ""
                descargas.extraer(z, carpeta_llama / rel.get("tag_name", "ultimo"))
            exe = descargas.buscar_servidor(carpeta_llama)
            if exe is None:
                raise RuntimeError("no encontré llama-server dentro del paquete descargado")
        # En la 1060 con el plan de 200M entra en la GPU; con 500M o sin GPU, en la CPU.
        plan = cfg.data["plan"]
        capas = 99 if gpu != "cpu" and "500m" not in plan else 0
        hilos = max(2, (os.cpu_count() or 4) - 2)
        r = self.app.servidor_maestro.iniciar(exe, gguf, capas_gpu=capas, hilos=hilos)
        self.log(r["mensaje"])
        if not r["ok"]:
            raise RuntimeError(r["mensaje"])
        cfg.set("maestro.endpoint", f"http://127.0.0.1:{self.app.servidor_maestro.port}/v1", save=False)
        cfg.set("maestro.model", m["nombre"].removesuffix(".gguf"), save=False)
        cfg.set("maestro_local", {"exe": str(exe), "modelo": str(gguf), "capas_gpu": capas, "hilos": hilos})
        if not self.app.tareas["maestro"].ocupada():
            self.app.tareas["maestro"].iniciar()          # episodios verificados mientras Hope entrena
        return f"{m['nombre']} en {'GPU' if capas else 'CPU'} con llama.cpp"

    def _entrenar(self) -> str:
        from hope.app import ajustes
        r = ajustes.empezar(self.app)
        if not r.get("ok") and "ya está en curso" not in r.get("mensaje", ""):
            raise RuntimeError(r.get("mensaje", "no se pudo empezar a entrenar"))
        return f"plan {self.app.cfg.data['plan']} · {platform.node()}"
