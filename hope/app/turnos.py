"""Modo «turnos» (predeterminado; decisión del usuario): Hope y el maestro se ALTERNAN la GPU, no la comparten.

Ciclo::

    Hope entrena ``min_hope`` minutos → pausa segura (después del checkpoint) y libera la VRAM
    → el maestro elegido usa toda la GPU ``min_maestro`` minutos, con pedidos en paralelo
    → se descarga el maestro de la VRAM (se detiene llama-server, o ``lms unload`` en LM Studio)
    → vuelve a entrenar Hope (que en el próximo tramo ya lee las explicaciones nuevas).

* **Maestro de cada turno:** round-robin ponderado por la tasa de verificación de cada maestro
  (``hope.app.maestros.elegir``): todos aportan variedad, y los que más aciertan explican más seguido.
* **Proporción automática:** después de cada turno del maestro se estima cuántas veces Hope repasaría
  las explicaciones con la mezcla actual (``objetivo`` ≈ 20 % de maestro). Si las repetiría más de
  ``max_repeticiones`` veces por turno, el turno del maestro se alarga; si sobran explicaciones, se acorta.
* ``paso(ahora)`` es determinista (las pruebas lo llaman con un reloj simulado); un hilo lo llama cada 2 s.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from hope.app import descargas, maestros

DEFAULT = {"activo": False, "min_hope": 45.0, "min_maestro": 15.0, "objetivo": 0.20, "max_repeticiones": 3.0,
           "auto_ajustar": True, "paralelo": 4, "modo": "turnos"}
LIMITES = {"min_hope": (5, 24 * 60), "min_maestro": (2, 12 * 60), "objetivo": (0.0, 0.8),
           "max_repeticiones": (1, 50), "paralelo": (1, 16)}
MIN_AJUSTE, FACTOR_SUBE, FACTOR_BAJA = 5.0, 1.25, 0.8
PUERTO_TURNOS = 8091            # llama-server de los turnos: no choca con uno propio del usuario en 8080


class Turnos:
    def __init__(self, app):
        self.app = app
        self.fase = "apagado"           # apagado · hope · pausando_hope · maestro · cerrando_maestro
        self.t0 = 0.0
        self.actual: dict | None = None
        self.dados: dict[str, int] = {}
        self.registro: list[str] = []
        self.ultima_estimacion: dict = {}
        self._lock = threading.RLock()
        self._fin = threading.Event()

    # --- configuración -----------------------------------------------------------------
    def cfg(self) -> dict:
        return {**DEFAULT, **(self.app.cfg.data.get("turnos") or {})}

    def configurar(self, cambios: dict) -> dict:
        c = self.cfg()
        for k, v in cambios.items():
            if k == "modo":
                if v not in ("turnos", "compartida"):
                    return {"ok": False, "mensaje": "modo: turnos o compartida"}
                c[k] = v
            elif k == "auto_ajustar":
                c[k] = bool(v)
            elif k in LIMITES:
                lo, hi = LIMITES[k]
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    return {"ok": False, "mensaje": f"{k}: tiene que ser un número"}
                if not lo <= x <= hi:
                    return {"ok": False, "mensaje": f"{k}: entre {lo:g} y {hi:g}"}
                c[k] = int(x) if k == "paralelo" else x
            else:
                return {"ok": False, "mensaje": f"clave desconocida: {k}"}
        self.app.cfg.set("turnos", {k: c[k] for k in DEFAULT})
        return {"ok": True}

    def log(self, texto: str) -> None:
        self.registro = (self.registro + [time.strftime("%H:%M ") + texto])[-50:]

    # --- control -----------------------------------------------------------------------
    def iniciar(self, ahora: float | None = None) -> dict:
        with self._lock:
            ahora = ahora or time.time()
            c = self.cfg()
            c["activo"] = True
            self.app.cfg.set("turnos", {k: c[k] for k in DEFAULT})
            self.fase, self.t0 = "hope", ahora
            self.log("empieza el turno de Hope")
            return self.app.entrenamiento.iniciar()

    def detener(self) -> dict:
        with self._lock:
            c = self.cfg()
            c["activo"] = False
            self.app.cfg.set("turnos", {k: c[k] for k in DEFAULT})
            fase, self.fase = self.fase, "apagado"
            if fase in ("hope", "pausando_hope") and self.app.entrenamiento.corriendo():
                self.app.entrenamiento.pausar()
            if fase in ("maestro", "cerrando_maestro"):
                self.app.tareas["maestro"].pausar()
                self._liberar()
            self.log("turnos detenidos")
            return {"ok": True, "mensaje": "turnos detenidos"}

    def retomar_si_corresponde(self) -> None:
        """Al abrir la aplicación: si estaba en modo turnos, sigue (empezando por Hope)."""
        if self.cfg()["activo"] and self.fase == "apagado":
            self.fase, self.t0 = "hope", time.time()
            if not self.app.entrenamiento.corriendo():
                self.app.entrenamiento.iniciar(automatico=True)

    # --- máquina de estados ----------------------------------------------------------------
    def paso(self, ahora: float | None = None) -> None:
        ahora = ahora or time.time()
        with self._lock:
            c = self.cfg()
            if not c["activo"] or self.fase == "apagado":
                return
            ent = self.app.entrenamiento
            mae = self.app.tareas["maestro"]
            if self.fase == "hope":
                if ent.estado == "terminado":
                    self.log("el plan terminó: fin de los turnos")
                    self.detener()
                    return
                if ent.estado == "esperando_aprobacion":
                    return                                    # espera la decisión del guardián
                if ahora - self.t0 >= c["min_hope"] * 60:
                    if ent.corriendo():
                        ent.pausar()
                        ent.mensaje = "turno del maestro: pausa después del próximo checkpoint"
                    self.fase, self.t0 = "pausando_hope", ahora
                    self.log("pausa de Hope para el turno del maestro")
            elif self.fase == "pausando_hope":
                if not ent.corriendo():
                    self._empezar_maestro(ahora, c)
            elif self.fase == "maestro":
                if ahora - self.t0 >= c["min_maestro"] * 60 or not mae.ocupada():
                    if mae.ocupada():
                        mae.pausar()
                    self.fase, self.t0 = "cerrando_maestro", ahora
            elif self.fase == "cerrando_maestro":
                if not mae.ocupada():
                    self._liberar()
                    self._ajustar(c)
                    self.fase, self.t0 = "hope", ahora
                    self.log("vuelve el turno de Hope")
                    ent.iniciar(automatico=True)

    def _empezar_maestro(self, ahora: float, c: dict) -> None:
        stats = (self.app.tareas["maestro"].conteos().get("por_maestro") or {})
        elegido = maestros.elegir(maestros.lista(self.app), stats, self.dados)
        if elegido is None:
            self.log("no hay maestros activos y descargados: sigue Hope")
            self.fase, self.t0 = "hope", ahora
            self.app.entrenamiento.iniciar(automatico=True)
            return
        r = self._cargar(elegido, c)
        if not r.get("ok"):
            self.log(f"{elegido['nombre']}: no arrancó ({r.get('mensaje')}); sigue Hope")
            self.dados[elegido["id"]] = self.dados.get(elegido["id"], 0) + 1    # que el próximo turno pruebe otro
            self.fase, self.t0 = "hope", ahora
            self.app.entrenamiento.iniciar(automatico=True)
            return
        self.actual = {**elegido, "endpoint": r["endpoint"], "model": r["model"]}
        self.dados[elegido["id"]] = self.dados.get(elegido["id"], 0) + 1
        self.app.tareas["maestro"].iniciar(continuo=True, maestro={"id": elegido["id"], "endpoint": r["endpoint"],
                                                                   "model": r["model"]},
                                           minutos=c["min_maestro"], paralelo=int(c["paralelo"]))
        self.fase, self.t0 = "maestro", ahora
        self.log(f"turno de {elegido['nombre']}")

    def _cargar(self, m: dict, c: dict) -> dict:
        """Pone el modelo del maestro en la GPU. Catálogo: llama-server que maneja la app. Servidor externo:
        LM Studio con «lms load» si está; Ollama u otros cargan solos al recibir el primer pedido."""
        if m["tipo"] == "catalogo":
            exe = self._llama_server()
            if exe is None:
                return {"ok": False, "mensaje": "falta llama.cpp (lo baja el modo automático)"}
            gpu = descargas.tipo_gpu((self.app.cfg.data.get("calibracion") or {}).get("dispositivo"))
            r = self.app.servidor_maestro.iniciar(exe, Path(m["ruta"]), capas_gpu=99 if gpu != "cpu" else 0,
                                                  paralelo=int(c["paralelo"]), port=PUERTO_TURNOS)
            if not r["ok"]:
                return r
            return {"ok": True, "endpoint": f"http://127.0.0.1:{self.app.servidor_maestro.port}/v1",
                    "model": Path(m["archivo"]).stem}
        if m.get("servidor") == "LM Studio":
            r = maestros.lms("load", m["model"], "--gpu", "max", "-y")
            if not r["ok"]:
                self.log("LM Studio: " + r["mensaje"])
        return {"ok": True, "endpoint": m["endpoint"], "model": m["model"]}

    def _liberar(self) -> None:
        a = self.actual or {}
        if a.get("tipo") == "catalogo":
            self.app.servidor_maestro.detener()
        elif a.get("servidor") == "LM Studio":
            r = maestros.lms("unload", "--all")
            if not r["ok"]:
                self.log("LM Studio: " + r["mensaje"])
        self.actual = None

    def _llama_server(self) -> Path | None:
        ml = self.app.cfg.data.get("maestro_local") or {}
        if ml.get("exe") and Path(ml["exe"]).exists():
            return Path(ml["exe"])
        return descargas.buscar_servidor(maestros.carpeta_modelos(self.app) / "llama.cpp")

    def _ajustar(self, c: dict) -> None:
        """Proporción automática: que Hope no repase las explicaciones más de ``max_repeticiones`` veces
        por turno con la mezcla actual (~``objetivo`` de maestro)."""
        tb = self.app.cfg.carpeta_datos / "maestro" / "train.bin"
        maestro_bytes = tb.stat().st_size if tb.exists() else 0
        bps = self.app.entrenamiento.metricas().get("bytes_por_s") or 0
        consumo = bps * c["min_hope"] * 60 * c["objetivo"]
        rep = consumo / maestro_bytes if maestro_bytes else float("inf") if consumo else 0.0
        self.ultima_estimacion = {"repeticiones_por_turno": None if rep == float("inf") else round(rep, 2),
                                  "bytes_maestro": maestro_bytes, "bytes_por_s": bps}
        if not c["auto_ajustar"] or not consumo:
            return
        nuevo = c["min_maestro"]
        if rep > c["max_repeticiones"]:
            nuevo = min(c["min_maestro"] * FACTOR_SUBE, c["min_hope"])
        elif rep < c["max_repeticiones"] / 3:
            nuevo = max(c["min_maestro"] * FACTOR_BAJA, MIN_AJUSTE)
        if abs(nuevo - c["min_maestro"]) > 1e-6:
            self.log(f"turno del maestro: {c['min_maestro']:.0f} → {nuevo:.0f} min (Hope repasaría las explicaciones "
                     f"{'∞' if rep == float('inf') else f'{rep:.1f}'} veces por turno)")
            self.configurar({"min_maestro": round(nuevo, 1)})

    # --- hilo y estado ----------------------------------------------------------------------------
    def arrancar_hilo(self) -> None:
        def bucle():
            while not self._fin.wait(2.0):
                try:
                    self.paso()
                except Exception as e:                # noqa: BLE001 — que el ciclo no muera
                    self.log(f"error en los turnos: {e}")
        threading.Thread(target=bucle, daemon=True, name="hope-turnos").start()

    def cerrar(self) -> None:
        self._fin.set()

    def estado(self) -> dict:
        c = self.cfg()
        mins = (time.time() - self.t0) / 60 if self.fase != "apagado" else 0
        nombre = (self.actual or {}).get("nombre")
        frase = {"hope": f"Turno de Hope: aprendiendo — {mins:.0f} de {c['min_hope']:.0f} min",
                 "pausando_hope": "Hope termina su tramo (checkpoint) para dejarle la GPU al maestro",
                 "maestro": f"Turno de {nombre or 'el maestro'}: explicando — {mins:.0f} de {c['min_maestro']:.0f} min",
                 "cerrando_maestro": f"{nombre or 'El maestro'} libera la GPU", "apagado": ""}[self.fase]
        return {**c, "fase": self.fase, "minutos": round(mins, 1), "maestro": nombre, "frase": frase,
                "turnos_por_maestro": self.dados, "estimacion": self.ultima_estimacion, "registro": self.registro[-10:]}
