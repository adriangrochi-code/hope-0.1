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

# Qwen desde el principio (decisión del usuario): el maestro arranca apenas se baja el modelo, antes de los
# textos y del entrenamiento, y queda explicando en paralelo todo el tiempo.
PASOS = ("dependencias", "calibracion", "maestro", "datos", "calidad", "entrenar")
NOMBRES = {"dependencias": "Instalar lo que falta", "calibracion": "Medir la PC", "datos": "Bajar los textos",
           "calidad": "Filtrar por calidad", "maestro": "Preparar el maestro", "entrenar": "Empezar a entrenar"}
NECESITA_GB = {"dependencias": 3.0, "calibracion": 0.5, "datos": None, "calidad": 0.1, "maestro": 2.5, "entrenar": 5.0}
PAQUETES = {"torch": "torch", "numpy": "numpy", "yaml": "pyyaml", "pyarrow": "pyarrow", "sympy": "sympy", "pint": "pint"}


def maestro_en_gpu(cfg) -> bool:
    """¿Entra Qwen 1,5B (~1,1 GB + contexto) en la GPU junto al entrenamiento? Con el plan de 200M en 6 GB,
    sí (el entrenamiento usa ~3–4 GB); con el de 500M con visión, no: va a la CPU."""
    plan = str(cfg.data.get("plan", ""))
    vram = ((cfg.data.get("calibracion") or {}).get("perfil") or {}).get("vram_gb") or 0
    if "500m" in plan:
        return vram >= 12
    return vram >= 5.5 or vram == 0          # sin dato de VRAM (calibración vieja): se intenta en la GPU


def arrancar_explicaciones(app) -> dict:
    """Pone a Qwen a generar explicaciones verificadas en modo continuo (si no estaba ya)."""
    t = app.tareas["maestro"]
    if t.ocupada():
        return {"ok": True, "mensaje": "Qwen ya está explicando"}
    return t.iniciar(continuo=True)


def asegurar_llama(app, tarea: Tarea) -> Path | None:
    """Baja llama.cpp (la última versión publicada, para esta GPU) si todavía no está. None si se pausó."""
    from hope.app import maestros
    cfg = app.cfg
    modelos = maestros.carpeta_modelos(app)
    carpeta = modelos / "llama.cpp"
    exe = descargas.buscar_servidor(carpeta)
    if exe is not None:
        return exe
    gpu = descargas.tipo_gpu((cfg.data.get("calibracion") or {}).get("dispositivo"))
    sistema = "windows" if os.name == "nt" else "linux"
    rel = descargas.ultimo_release()
    assets = descargas.elegir_llama(rel, sistema, gpu) or descargas.elegir_llama(rel, sistema, "cpu")
    if not assets:
        raise RuntimeError(f"no encontré llama.cpp para {sistema}/{gpu} en el release {rel.get('tag_name')}")
    for a in assets:
        z = modelos / "descargas" / a["name"]
        tarea.log(f"bajando {a['name']}")
        descargas.bajar(a["url"], z, total=a.get("bytes"), detener=tarea.detener_pedido,
                        progreso=lambda h, t, n=a["name"]: setattr(tarea, "detalle", f"{n}: {h / 1e6:.0f} MB"))
        if tarea.detener_pedido():
            return None
        descargas.extraer(z, carpeta / rel.get("tag_name", "ultimo"))
    exe = descargas.buscar_servidor(carpeta)
    if exe is None:
        raise RuntimeError("no encontré llama-server dentro del paquete descargado")
    return exe


class TareaDescargaMaestros(Tarea):
    """Baja (con reanudación y SHA-256) los maestros activos del catálogo que entran en la GPU por turnos."""

    nombre = "descarga_maestros"

    def ejecutar(self) -> None:
        from hope.app import maestros
        cfg = self.app.cfg
        vram = ((cfg.data.get("calibracion") or {}).get("perfil") or {}).get("vram_gb") or 0
        pend = [m for m in maestros.lista(self.app) if m["tipo"] == "catalogo" and m["activo"] and not m["descargado"]]
        for i, m in enumerate(pend):
            if self.detener_pedido():
                return
            if vram and not maestros.entra_en_vram(m, vram, compartida=False)["entra"]:
                self.log(f"{m['nombre']}: no entra en {vram:g} GB de VRAM ni en su turno; se saltea")
                continue
            info = descargas.info_remota(m["url"])
            if not info["bytes"]:
                self.log(f"{m['nombre']}: no se encontró {m['archivo']} en Hugging Face ({m['repo']}); se saltea")
                continue
            self.log(f"bajando {m['nombre']} ({info['bytes'] / 1e9:.1f} GB, {m['licencia']})")
            descargas.bajar(m["url"], Path(m["ruta"]), sha=info["sha256"], total=info["bytes"],
                            detener=self.detener_pedido,
                            progreso=lambda h, t, n=m["nombre"]: (setattr(self, "detalle", f"{n}: {h / 1e9:.2f}"
                                                                          + (f" / {t / 1e9:.2f} GB" if t else " GB")),
                                                                  setattr(self, "progreso", (i + (h / t if t else 0)) / len(pend))))
        self.detalle = "maestros listos"


class TareaAutomatica(Tarea):
    nombre = "automatico"

    def __init__(self, app):
        super().__init__(app)
        self.paso = None
        self.pregunta: dict | None = None          # p. ej. «¿uso LM Studio como maestro?»

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
        """Maestros: si hay LM Studio (u otro servidor local), pregunta si usarlo sin bajar nada. Si no (o si el
        usuario prefiere el catálogo), baja llama.cpp y los maestros activos del catálogo que entran en la GPU."""
        from hope.app import maestros
        cfg = self.app.cfg
        servidores = maestros.detectar_servidores()
        lm = next((x for x in servidores if x["servidor"] == "LM Studio" and x["modelos"]), None)
        if lm and not cfg.data.get("lmstudio_decision"):
            self.pregunta = {"tipo": "lmstudio", "texto": f"Encontré LM Studio con {', '.join(lm['modelos'][:3])}. "
                             "¿Lo uso como maestro sin bajar otro modelo?", "modelos": lm["modelos"],
                             "endpoint": lm["endpoint"]}
            self.detalle = "esperando tu respuesta sobre LM Studio"
            while not self.detener_pedido() and not cfg.data.get("lmstudio_decision"):
                self._stop.wait(1.0)
            self.pregunta = None
            if self.detener_pedido():
                return ""
        usados = []
        if lm and cfg.data.get("lmstudio_decision") == "usar":
            for mod in lm["modelos"][:1]:
                maestros.agregar_servidor(self.app, "LM Studio", lm["endpoint"], mod)
                usados.append(f"{mod} (LM Studio)")
            if not lm["lms"]:
                self.log("sin el comando «lms» la app no puede sacar el modelo de la GPU en el turno de Hope: en LM "
                         "Studio dejá el modelo en la CPU o bajá su «GPU offload» (o instalá lms)")
            return "maestro: " + ", ".join(usados)
        exe = asegurar_llama(self.app, self)
        if exe is None:
            return ""
        t = self.app.tareas["descarga_maestros"]
        self._esperar(t)
        bajados = [m["nombre"] for m in maestros.lista(self.app) if m["tipo"] == "catalogo" and m["activo"]
                   and m["descargado"]]
        if not bajados:
            raise RuntimeError("no se pudo bajar ningún maestro del catálogo")
        cfg.set("maestro_local", {"exe": str(exe)})
        modo = self.app.turnos.cfg()["modo"]
        if modo == "compartida":
            return self._compartida(exe, bajados)
        return "maestros (por turnos): " + ", ".join(bajados)

    def _compartida(self, exe, bajados) -> str:
        """Modo compartido (opcional): el primer maestro queda cargado todo el tiempo junto al entrenamiento."""
        from hope.app import maestros
        cfg = self.app.cfg
        m = next(x for x in maestros.lista(self.app) if x["tipo"] == "catalogo" and x["activo"] and x["descargado"])
        gpu = descargas.tipo_gpu((cfg.data.get("calibracion") or {}).get("dispositivo"))
        vram = ((cfg.data.get("calibracion") or {}).get("perfil") or {}).get("vram_gb") or 0
        cabe = maestros.entra_en_vram(m, vram, compartida=True, plan=cfg.data["plan"])
        if not cabe["entra"]:
            self.log(cabe.get("consejo", ""))
        capas = 99 if gpu != "cpu" and cabe["entra"] else 0
        hilos = max(2, (os.cpu_count() or 4) - 2)
        r = self.app.servidor_maestro.iniciar(exe, Path(m["ruta"]), capas_gpu=capas, hilos=hilos)
        self.log(r["mensaje"])
        if not r["ok"]:
            raise RuntimeError(r["mensaje"])
        cfg.set("maestro.endpoint", f"http://127.0.0.1:{self.app.servidor_maestro.port}/v1", save=False)
        cfg.set("maestro.model", Path(m["archivo"]).stem, save=False)
        cfg.set("maestro_local", {"exe": str(exe), "modelo": m["ruta"], "capas_gpu": capas, "hilos": hilos})
        arrancar_explicaciones(self.app)
        return f"{m['nombre']} en {'GPU' if capas else 'CPU'}, a la vez que Hope"

    def _entrenar(self) -> str:
        from hope.app import ajustes
        if self.app.turnos.cfg()["modo"] == "turnos":
            req = ajustes.requisitos(self.app)
            malos = [x for x in req if not x["ok"]]
            if malos:
                raise RuntimeError("falta: " + "; ".join(f"{x['nombre']} ({x['detalle']})" for x in malos))
            self.app.turnos.iniciar()
            return f"plan {self.app.cfg.data['plan']} por turnos con los maestros · {platform.node()}"
        r = ajustes.empezar(self.app)
        if not r.get("ok") and "ya está en curso" not in r.get("mensaje", ""):
            raise RuntimeError(r.get("mensaje", "no se pudo empezar a entrenar"))
        return f"plan {self.app.cfg.data['plan']} · {platform.node()}"
