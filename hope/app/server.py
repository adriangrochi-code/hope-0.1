"""Servidor de la aplicación: ``python -m hope.app [--puerto 8765] [--raiz .] [--sin-navegador]``.

Seguridad:
* escucha solo en 127.0.0.1;
* rechaza pedidos con un ``Host`` que no sea 127.0.0.1/localhost (contra «DNS rebinding»);
* las acciones (POST ``/api/accion``) exigen el encabezado ``X-Hope: 1`` y JSON, que una página de
  otro origen no puede mandar sin permiso CORS (que este servidor nunca da);
* solo existen las acciones de ``ACCIONES``; ninguna ejecuta comandos arbitrarios.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser

import yaml
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hope.app import ajustes, comandos, discos, guardian_app, red, sistema
from hope.app import maestros as maestros_mod
from hope.app.automatico import TareaAutomatica, TareaDescargaMaestros
from hope.app.turnos import Turnos
from hope.app.charla_app import GestorCharla
from hope.app.descargas import ServidorMaestro
from hope.app.config import PLANES, AppConfig, plan_efectivo
from hope.app.datos import TareaDatos, presupuestos
from hope.app.datos_diarios import TareaDatosDiarios
from hope.app.entrenamiento import Entrenamiento
from hope.app.maestro import TareaMaestro, guardar_yaml, probar
from hope.app.migrar import TareaExportar, TareaImportar, continuacion
from hope.app.tareas import Tarea, correr_modulo

STATIC = Path(__file__).parent / "static"
TIPOS = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}


class TareaCalibracion(Tarea):
    nombre = "calibracion"

    def ejecutar(self, rapido: bool = False) -> None:
        cfg = self.app.cfg
        out = cfg.root / "runs" / "calibracion.json"
        args = ["--salida", str(out)]
        try:
            rel, _ = plan_efectivo(cfg)
            args += ["--plan", str(rel)]
        except (OSError, ValueError) as e:
            self.log(f"aviso: sin plan para buscar el lote ({e})")
        if rapido:
            args.append("--rapido")
        if continuacion(self.app)["es_continuacion"]:
            args.append("--continuacion")
        if self.app.entrenamiento.corriendo():
            raise RuntimeError("pausá el entrenamiento para calibrar (los dos usan la GPU)")
        code = correr_modulo(self, "hope.app.calibrar", args, cfg.root)
        if self.detener_pedido():
            return
        if code != 0:
            raise RuntimeError(f"la calibración terminó con código {code}")
        res = json.loads(out.read_text(encoding="utf-8"))
        cfg.set("calibracion", res)
        self.app.dispositivo = None if res.get("dispositivo") in (None, "cpu") else res["dispositivo"]


class TareaCuantizar(Tarea):
    """Paso 7: exportar el modelo ligero a 4 u 8 bits (``hope.app.cuantizar``, subproceso en CPU)."""

    nombre = "cuantizar"

    def modelo(self) -> Path | None:
        cands = [self.app.cfg.ruta(s) / "ligero.pt" for s in ("runs/hope_a770", "runs/hope")]
        cands = [c for c in cands if c.exists()]
        return max(cands, key=lambda c: c.stat().st_mtime) if cands else None

    def exportados(self) -> list[dict]:
        out = []
        for j in sorted((self.app.cfg.root / "runs" / "exportar").glob("*.pt.json"), reverse=True)[:10]:
            try:
                out.append(json.loads(j.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
        return out

    def ejecutar(self, bits: int = 4) -> None:
        if bits not in (4, 8):
            raise ValueError("bits: 4 u 8")
        modelo = self.modelo()
        if modelo is None:
            raise FileNotFoundError("todavía no hay modelo ligero (se guarda con el primer checkpoint)")
        sal = self.app.cfg.root / "runs" / "exportar" / f"hope_q{bits}_{time.strftime('%Y%m%d-%H%M%S')}.pt"
        code = correr_modulo(self, "hope.app.cuantizar", ["--modelo", str(modelo), "--bits", str(bits),
                                                          "--salida", str(sal)], self.app.cfg.root,
                             env_extra={"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        if self.detener_pedido():
            return
        if code != 0:
            raise RuntimeError(f"la exportación terminó con código {code}")
        self.datos = json.loads(Path(str(sal) + ".json").read_text(encoding="utf-8"))


class App:
    def __init__(self, root: str | Path = "."):
        self.root = Path(root).resolve()
        self.cfg = AppConfig(self.root)
        cal = self.cfg.data.get("calibracion") or {}
        self.dispositivo = cal.get("dispositivo") if cal.get("dispositivo") in ("cuda", "xpu") else None
        self.tareas = {t.nombre: t for t in (TareaCalibracion(self), TareaDatos(self), TareaMaestro(self),
                                             TareaExportar(self), TareaImportar(self), TareaCuantizar(self),
                                             TareaAutomatica(self), TareaDatosDiarios(self),
                                             TareaDescargaMaestros(self))}
        self.entrenamiento = Entrenamiento(self)
        self.servidor_maestro = ServidorMaestro(self.root / "runs")
        self.propuestas: dict = {}                  # chat: propuestas esperando confirmación
        self.charla = GestorCharla(self)            # pestaña «Chatear» (copia del checkpoint, aparte)
        self.turnos = Turnos(self)                  # Hope y el maestro se alternan la GPU
        self.turnos.arrancar_hilo()
        self._servidores = (0.0, [])
        self._fin_charla = threading.Event()
        threading.Thread(target=self._vigilar_charla, daemon=True, name="hope-charla").start()
        self.sesiones = red.Sesiones()              # acceso desde la red local (con PIN)
        self.cerrar_servidor = None                 # lo asigna serve()
        self.escucha = "127.0.0.1"
        self._discos = (0.0, None)

    def discos(self) -> dict:
        """Ubicación elegida y discos (se mide cada 60 s: en Windows consulta PowerShell)."""
        t, v = self._discos
        if v is None or time.time() - t > 60:
            try:
                v = discos.resumen(self.root)
            except OSError as e:
                v = {"error": str(e), "elegida": discos.leer(self.root), "discos": [], "propuesta": None,
                     "en_sistema": []}
            self._discos = (time.time(), v)
        return v

    def guardian(self):
        return guardian_app.cargar(self.root)

    def _vigilar_charla(self) -> None:
        while not self._fin_charla.wait(5.0):
            try:
                self.charla.revisar()
            except Exception as e:                  # noqa: BLE001
                self.charla.ultimo_error = str(e)

    def servidores(self, refrescar: bool = False) -> list:
        t, v = self._servidores
        if refrescar or time.time() - t > 30:
            v = maestros_mod.detectar_servidores()
            self._servidores = (time.time(), v)
        return v

    def maestros_estado(self, cont: dict) -> dict:
        stats = cont.get("por_maestro") or {}
        cal = self.cfg.data.get("calibracion") or {}
        vram = (cal.get("perfil") or {}).get("vram_gb") or 0
        modo = self.turnos.cfg()["modo"]
        lst = []
        for m in maestros_mod.lista(self):
            tasa = maestros_mod.tasa(stats, m["id"])
            s_ = stats.get(m["id"]) or {}
            vr = maestros_mod.entra_en_vram(m, vram, compartida=modo == "compartida", plan=self.cfg.data["plan"]) \
                if m.get("gb") and vram else None
            lst.append({**{k: m.get(k) for k in ("id", "nombre", "tipo", "licencia", "grupo", "nota", "gb", "activo",
                                                  "descargado", "servidor", "model", "endpoint", "licencia_aceptada")},
                        "verificado_pct": None if tasa is None else round(100 * tasa),
                        "episodios": s_.get("aceptados", 0) + s_.get("rechazados", 0), "vram": vr})
        return {"lista": lst, "servidores": self.servidores(), "vram_gb": vram, "modo": modo,
                "descarga": self.tareas["descarga_maestros"].to_dict(10)}

    def cerrar(self) -> None:
        self.turnos.cerrar()
        self._fin_charla.set()
        self.charla.cerrar()
        self.entrenamiento.cerrar()
        self.servidor_maestro.detener()
        for t in self.tareas.values():
            if t.ocupada():
                t.pausar()

    # --- estado completo para la interfaz -----------------------------------------
    def estado(self) -> dict:
        cfg = self.cfg
        t = self.tareas
        datos = t["datos"]
        fuentes = datos.estado_fuentes()
        ent = self.entrenamiento.metricas()
        maestro = t["maestro"]
        cont = maestro.conteos()
        try:
            g = guardian_app.resumen(self.guardian())
        except (OSError, ValueError, KeyError) as e:
            g = {"error": str(e), "pendientes": {}}
        try:
            plan = cfg.leer_plan()
        except OSError as e:
            plan = {"error": str(e)}
        cal = cfg.data.get("calibracion")
        etapas = ent["etapas"]
        idx_maestro = next((i for i, e in enumerate(etapas) if "maestro" in e["nombre"]), None)
        pasos = {
            "calibracion": _paso(t["calibracion"], hecho=bool(cal and cal.get("plan_recomendado"))),
            "datos": _paso(datos, hecho=bool(fuentes) and all(f["hecha"] for f in fuentes.values())),
            "lenguaje": {"estado": ent["estado"], "hecho": bool(etapas) and etapas[0]["hecha"]},
            "maestro": _paso(maestro, hecho=idx_maestro is not None and etapas[idx_maestro]["hecha"]),
            "migrar": _paso(t["exportar"] if t["exportar"].estado != "inactiva" else t["importar"],
                            hecho=bool(t["exportar"].datos.get("zip") or t["importar"].datos.get("marca"))),
            "guardian": {"estado": "pendientes" if g.get("pendientes") else "al_dia", "hecho": not g.get("pendientes")},
            "cuantizar": _paso(t["cuantizar"], hecho=bool(t["cuantizar"].exportados())),
            "automatico": _paso(t["automatico"], hecho=len(t["automatico"].hechos()) >= 6),
            "diarios": _paso(t["datos_diarios"], hecho=t["datos_diarios"].estado == "en_curso"),
        }
        mod = t["cuantizar"].modelo()
        return {
            "version": 1, "hora": time.strftime("%H:%M:%S"),
            "config": cfg.data, "planes": self.planes(), "plan": plan, "pasos": pasos,
            "sistema": sistema.resumen(cfg.carpeta_datos),
            "calibracion": {**t["calibracion"].to_dict(), "resultado": cal},
            "datos": {**datos.to_dict(), "fuentes": fuentes,
                      "presupuestos_gb": {k: round(v / 1e9, 2) for k, v in presupuestos(cfg).items()}},
            "entrenamiento": ent,
            "maestro": {**maestro.to_dict(), "conteos": cont, "etapa": etapas[idx_maestro] if idx_maestro is not None
                        else None},
            "migrar": {"exportar": t["exportar"].to_dict(), "importar": t["importar"].to_dict(),
                       "continuacion": continuacion(self)},
            "guardian": g,
            "diarios": {**t["datos_diarios"].to_dict(), **t["datos_diarios"].resumen()},
            "ajustes": _seguro(lambda: ajustes.estado(self)),
            "inicio": {"frase": _seguro(lambda: comandos.frase_estado(self)),
                       "automatico": {**t["automatico"].to_dict(20), "pasos": t["automatico"].resumen()},
                       "horario": cfg.data.get("horario") or {"activo": False},
                       "maestro_local": self.servidor_maestro.corriendo() or None,
                       "turno": self.turnos.estado(),
                       "pregunta": t["automatico"].pregunta,
                       "qwen": {"explicando": t["maestro"].ocupada(), "hoy": cont.get("hoy_maestro", 0),
                                "hoy_aceptadas": cont.get("hoy_aceptados", 0), "detalle": t["maestro"].detalle,
                                "lee_junto": lee_junto(self)}},
            "discos": self.discos(),
            "charla": self.charla.estado(),
            "maestros": _seguro(lambda: self.maestros_estado(cont)),
            "red": red.estado(self),
            "presupuesto": {"disponible": comandos.presupuesto_disponible(),
                            "techo": (g.get("techo") if isinstance(g, dict) else None),
                            "techo_humano": (g.get("techo_humano") if isinstance(g, dict) else None)},
            "cuantizar": {**t["cuantizar"].to_dict(), "exportados": t["cuantizar"].exportados(),
                          "modelo": str(mod) if mod else None,
                          "mb_modelo": round(mod.stat().st_size / 1e6, 1) if mod else None},
        }

    def planes(self) -> dict:
        out = dict(PLANES)
        for f in sorted((self.root / "configs").glob("plan_*.yaml")):
            out.setdefault(f.name, f.stem.replace("_", " "))
        return out

    # --- acciones (lista fija) ----------------------------------------------------
    def accion(self, nombre: str, p: dict) -> dict:
        if nombre not in ACCIONES:
            raise KeyError(f"acción desconocida: {nombre}")
        return ACCIONES[nombre](self, p or {})


def lee_junto(app) -> dict:
    """«Qwen lee junto a Hope» (logits de Qwen precalculados, destilación): llega con la rama de tokens."""
    import importlib.util
    disponible = any(importlib.util.find_spec(m) is not None for m in
                     ("hope.train.distill", "hope.train.destilacion", "hope.data.logits_qwen"))
    return {"disponible": disponible, "activo": bool(app.cfg.data.get("qwen_lee_junto")) and disponible}


def _seguro(fn):
    try:
        return fn()
    except (OSError, ValueError, KeyError, yaml.YAMLError) as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _cerrar(app: App, p: dict) -> dict:
    if not _confirmado(p):
        return {"ok": False, "mensaje": "falta confirmar"}
    if app.cerrar_servidor is None:
        return {"ok": False, "mensaje": "no se puede cerrar desde acá"}
    threading.Timer(0.5, app.cerrar_servidor).start()
    return {"ok": True, "mensaje": "la aplicación se cierra; el entrenamiento en curso sigue"}


def _charla_decir(app: App, p: dict) -> dict:
    texto = str(p.get("texto", "")).strip()[:4000]
    if not texto:
        return {"ok": False, "mensaje": "escribí algo"}
    if app.charla.cargado is None:
        r = app.charla.cargar()
        if not r.get("ok"):
            return r
    obj = p.get("objetivo_tok_s")
    return app.charla.pedir("decir", conv=str(p.get("conv", "principal"))[:40], texto=texto,
                            max_bytes=max(16, min(int(p.get("max_bytes", 300)), 4000)),
                            herramientas=bool(p.get("herramientas", True)), contexto=bool(p.get("contexto", True)),
                            temperatura=max(0.0, min(float(p.get("temperatura", 0.0)), 2.0)),
                            objetivo_tok_s=float(obj) if obj else None)


def _charla_comparar(app: App, p: dict) -> dict:
    texto = str(p.get("texto", "")).strip()[:4000]
    archivo = Path(str(p.get("archivo", "")))
    if not texto or not archivo.exists() or archivo.parent.resolve() != app.charla.carpeta().resolve():
        return {"ok": False, "mensaje": "elegí una pregunta y una copia de checkpoint del chat"}
    if app.charla.cargado is None:
        r = app.charla.cargar()
        if not r.get("ok"):
            return r
    return app.charla.pedir("comparar", texto=texto, modelo_b=str(archivo), bits=app.charla.bits)


def _servidor_agregar(app: App, p: dict) -> dict:
    if not _confirmado(p):
        return {"ok": False, "mensaje": "falta confirmar"}
    det = {x["endpoint"]: x for x in app.servidores()}
    ep, model = str(p.get("endpoint", "")), str(p.get("model", ""))
    if ep not in det or model not in det[ep]["modelos"]:
        return {"ok": False, "mensaje": "ese modelo no está en los servidores detectados (tocá «Buscar» de nuevo)"}
    return maestros_mod.agregar_servidor(app, det[ep]["servidor"], ep, model)


def _auto_responder(app: App, p: dict) -> dict:
    r = str(p.get("respuesta", ""))
    if r not in ("usar", "descargar"):
        return {"ok": False, "mensaje": "respuesta: usar o descargar"}
    app.cfg.set("lmstudio_decision", r)
    return {"ok": True, "mensaje": "LM Studio como maestro" if r == "usar" else "se bajan los maestros del catálogo"}


def _lee_junto(app: App, p: dict) -> dict:
    info = lee_junto(app)
    if not info["disponible"]:
        return {"ok": False, "mensaje": "«Qwen lee junto a Hope» llega con la rama de tokens (destilación de "
                                        "probabilidades); todavía no está en esta versión"}
    app.cfg.set("qwen_lee_junto", bool(p.get("activo")))
    return {"ok": True, "mensaje": "Qwen lee junto a Hope: " + ("activado" if p.get("activo") else "desactivado")}


def _pausar_todo(app: App) -> dict:
    r = {"ok": False, "mensaje": "no hay nada en curso"}
    if app.tareas["automatico"].ocupada():
        r = app.tareas["automatico"].pausar()
    if app.entrenamiento.corriendo():
        r = app.entrenamiento.pausar()
    return r


def _disco_usar(app: App, p: dict) -> dict:
    if not _confirmado(p):
        return {"ok": False, "mensaje": "falta confirmar"}
    if app.entrenamiento.corriendo() or any(t.ocupada() for t in app.tareas.values()):
        return {"ok": False, "mensaje": "pausá el entrenamiento y las tareas en curso antes de cambiar de disco"}
    info = app.discos()
    carpeta = str(p.get("carpeta") or ((info.get("propuesta") or {}).get("carpeta")) or "").strip()
    if not carpeta:
        return {"ok": False, "mensaje": "falta la carpeta"}
    try:
        r = discos.preparar(carpeta, app.root, discos.disco_de(carpeta, info.get("discos") or []))
    except (OSError, ValueError) as e:
        return {"ok": False, "mensaje": f"no pude usar {carpeta}: {e}"}
    discos.aplicar_entorno(app.root)
    app._discos = (0.0, None)
    notas = [f"{k}: {v}" for k, v in r["enlaces"].items() if v not in ("enlazado", "ya estaba")]
    return {"ok": True, "mensaje": f"Hope usa {r['carpeta']}" + ("; " + "; ".join(notas) if notas else "")}


def _presupuesto_plan(app: App, p: dict) -> dict:
    """Plan para llevar el modelo entrenado a otro tamaño (módulo de la rama de presupuesto)."""
    from guardian.cli import human, parse_count
    try:
        n = parse_count(str(p.get("valor", "")))
    except ValueError as e:
        return {"ok": False, "mensaje": str(e)}
    out = {"ok": True, "valor": n, "humano": human(n), "frase": f"AUTORIZO {human(n)}",
           "disponible": comandos.presupuesto_disponible()}
    if not out["disponible"]:
        out["mensaje"] = ("el cambio de tamaño del modelo ya entrenado todavía no está en esta versión "
                          "(hope.growth.budget_target); por ahora solo se puede ajustar el techo del guardián")
        return out
    est = app.entrenamiento.estado_run()
    ck = est.get("ultimo_checkpoint")
    if not ck:
        out["mensaje"] = "todavía no hay un modelo entrenado: el tamaño sale del plan"
        return out
    tarea = Tarea(app)
    tarea.nombre = "presupuesto"
    code = correr_modulo(tarea, "hope.growth.budget_target", ["--checkpoint", str(ck), "--objetivo", str(p["valor"])],
                         app.root, env_extra={"CUDA_VISIBLE_DEVICES": ""})
    out["plan"] = "\n".join(x.split(" ", 1)[1] if x[:2].isdigit() else x for x in tarea.registro)
    out["ok"] = code == 0
    return out


def _presupuesto_aplicar(app: App, p: dict) -> dict:
    if not _confirmado(p):
        return {"ok": False, "mensaje": "falta confirmar"}
    plan = _presupuesto_plan(app, p)
    if not plan.get("ok"):
        return plan
    cambios = {"techo": {"valor": str(plan["valor"]), "confirmacion": str(p.get("frase", ""))}}
    r = ajustes.aplicar(app, cambios, confirmar=True)
    if not r.get("ok") or not plan["disponible"]:
        return r
    r2 = ajustes.agregar_etapa_presupuesto(app, plan["humano"])
    return {**r, "mensaje": r["mensaje"] + "; " + r2["mensaje"], "ok": r2["ok"]}


def _paso(t: Tarea, hecho: bool) -> dict:
    return {"estado": t.estado, "hecho": hecho, "progreso": t.progreso}


def _confirmado(p: dict) -> bool:
    return p.get("confirmar") is True


def _guardar(app: App, p: dict) -> dict:
    cambios = app.cfg.update_from_ui(p.get("cambios", {}))
    if any(k.startswith("maestro.") for k in cambios):
        guardar_yaml(app.cfg)
    return {"ok": True, "cambios": cambios}


def _pasar_etapa_maestro(app: App, p: dict) -> dict:
    if not _confirmado(p):
        return {"ok": False, "mensaje": "falta confirmar"}
    plan = app.cfg.leer_plan()
    nombre = next((e["nombre"] for e in plan.get("etapas", []) if "maestro" in e["nombre"]), None)
    if nombre is None:
        return {"ok": False, "mensaje": "el plan elegido no tiene etapa del maestro"}
    r = app.entrenamiento.saltar_a_etapa(nombre)
    if r["ok"] and p.get("lanzar", True):
        return app.entrenamiento.iniciar()
    return r


def _detener_ya(app: App, p: dict) -> dict:
    if not _confirmado(p):
        return {"ok": False, "mensaje": "falta confirmar"}
    return app.entrenamiento.detener_ya()


def _muestra(app: App, p: dict) -> dict:
    sal = app.entrenamiento._salida() / "ligero.pt"
    if not sal.exists():
        return {"ok": False, "mensaje": "todavía no hay modelo ligero (se guarda con el primer checkpoint)"}
    return app.entrenamiento.muestras.iniciar(modelo=str(sal))


ACCIONES = {
    "guardar_config": _guardar,
    "calibrar_iniciar": lambda a, p: a.tareas["calibracion"].iniciar(rapido=bool(p.get("rapido"))),
    "calibrar_detener": lambda a, p: a.tareas["calibracion"].pausar(),
    "datos_iniciar": lambda a, p: a.tareas["datos"].iniciar(rehacer=_confirmado(p) and bool(p.get("rehacer"))),
    "datos_pausar": lambda a, p: a.tareas["datos"].pausar(),
    "entrenar_iniciar": lambda a, p: a.entrenamiento.iniciar(),
    "entrenar_pausar": lambda a, p: a.entrenamiento.pausar(),
    "entrenar_detener_ya": _detener_ya,
    "muestra_generar": _muestra,
    "maestro_probar": lambda a, p: probar(a.cfg, p.get("endpoint"), p.get("model")),
    "maestro_iniciar": lambda a, p: a.tareas["maestro"].iniciar(rondas=int(p["rondas"]) if p.get("rondas") else None),
    "maestro_pausar": lambda a, p: a.tareas["maestro"].pausar(),
    "maestro_pasar_etapa": _pasar_etapa_maestro,
    "exportar_iniciar": lambda a, p: a.tareas["exportar"].iniciar(destino=str(p.get("destino", ""))),
    "exportar_pausar": lambda a, p: a.tareas["exportar"].pausar(),
    "importar_iniciar": lambda a, p: a.tareas["importar"].iniciar(zip_path=str(p.get("zip", ""))),
    "cuantizar_iniciar": lambda a, p: a.tareas["cuantizar"].iniciar(bits=int(p.get("bits", 4))),
    "cuantizar_pausar": lambda a, p: a.tareas["cuantizar"].pausar(),
    "diarios_iniciar": lambda a, p: a.tareas["datos_diarios"].iniciar(repetir=not p.get("una_vez")),
    "diarios_pausar": lambda a, p: a.tareas["datos_diarios"].pausar(),
    "config_previsualizar": lambda a, p: ajustes.previsualizar(a, p.get("cambios") or {}),
    "config_aplicar": lambda a, p: ajustes.aplicar(a, p.get("cambios") or {}, _confirmado(p)),
    "empezar_entrenar": lambda a, p: ajustes.empezar(a),
    "cerrar_app": _cerrar,
    "maestro_activar": lambda a, p: maestros_mod.activar(a, str(p.get("id", "")), bool(p.get("activo")),
                                                         bool(p.get("acepto_licencia"))),
    "maestro_borrar": lambda a, p: maestros_mod.borrar_descarga(a, str(p.get("id", ""))) if _confirmado(p)
                                   else {"ok": False, "mensaje": "falta confirmar"},
    "maestros_descargar": lambda a, p: a.tareas["descarga_maestros"].iniciar(),
    "maestros_descargar_pausar": lambda a, p: a.tareas["descarga_maestros"].pausar(),
    "servidores_detectar": lambda a, p: {"ok": True, "servidores": a.servidores(refrescar=True)},
    "servidor_agregar": lambda a, p: _servidor_agregar(a, p),
    "turnos_iniciar": lambda a, p: a.turnos.iniciar(),
    "turnos_detener": lambda a, p: a.turnos.detener(),
    "turnos_config": lambda a, p: a.turnos.configurar(dict(p.get("cambios") or {})),
    "auto_responder": lambda a, p: _auto_responder(a, p),
    "qwen_explicar": lambda a, p: (__import__("hope.app.automatico", fromlist=["x"]).arrancar_explicaciones(a)
                                   if p.get("activo", True) else a.tareas["maestro"].pausar()),
    "qwen_lee_junto": lambda a, p: _lee_junto(a, p),
    "auto_iniciar": lambda a, p: a.tareas["automatico"].iniciar(rehacer=_confirmado(p) and bool(p.get("rehacer"))),
    "auto_pausar": lambda a, p: _pausar_todo(a),
    "disco_usar": lambda a, p: _disco_usar(a, p),
    "chat": lambda a, p: comandos.interpretar(a, str(p.get("texto", ""))[:500]),
    "chat_confirmar": lambda a, p: comandos.confirmar(a, str(p.get("id", "")), str(p.get("frase", ""))),
    "presupuesto_plan": lambda a, p: _presupuesto_plan(a, p),
    "presupuesto_aplicar": lambda a, p: _presupuesto_aplicar(a, p),
    "red_configurar": lambda a, p: red.configurar(a, p, desde_local=False),   # el servidor lo atiende aparte
    "red_hosts": lambda a, p: red.configurar_hosts(a, p, desde_local=False),
    "charla_cargar": lambda a, p: a.charla.cargar(p.get("bits"), p.get("dispositivo"), p.get("archivo")),
    "charla_auto": lambda a, p: setattr(a.charla, "auto", bool(p.get("auto"))) or {"ok": True},
    "charla_decir": lambda a, p: _charla_decir(a, p),
    "charla_comparar": lambda a, p: _charla_comparar(a, p),
    "charla_feedback": lambda a, p: a.charla.pedir("feedback", pregunta=str(p.get("pregunta", ""))[:4000],
                                                   respuesta=str(p.get("respuesta", ""))[:8000],
                                                   voto=str(p.get("voto", "")), correccion=str(p.get("correccion", ""))[:8000],
                                                   usados=[str(x) for x in (p.get("usados") or [])][:50]),
    "charla_reiniciar": lambda a, p: a.charla.pedir("reiniciar", conv=str(p.get("conv", "principal"))),
    "guardian_aprobar": lambda a, p: guardian_app.resolver(a, str(p.get("id", "")), True, str(p.get("nota", ""))),
    "guardian_denegar": lambda a, p: guardian_app.resolver(a, str(p.get("id", "")), False, str(p.get("nota", ""))),
    "guardian_techo": lambda a, p: guardian_app.subir_techo(a, str(p.get("valor", "")), str(p.get("confirmacion", "")),
                                                            p.get("modo")),
}


# ---------------------------------------------------------------------------
def make_handler(app: App, port_ref: dict):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HopeApp/1"

        def log_message(self, fmt, *args):         # silencioso: la interfaz consulta cada 2 s
            pass

        def _host_ok(self) -> bool:
            return red.host_ok(self.headers.get("Host") or "", port_ref["port"], app.escucha != "127.0.0.1",
                               (app.cfg.data.get("red") or {}).get("hosts") or ())

        def _local(self) -> bool:
            """La propia PC: conexión desde 127.0.0.1 y con Host local. Un túnel (Tailscale serve, cloudflared)
            también se conecta desde 127.0.0.1, pero con otro Host: ese pide PIN."""
            return red.es_local(self.client_address[0]) and red.host_local(self.headers.get("Host") or "",
                                                                             port_ref["port"])

        def _autorizado(self) -> bool:
            """Desde la propia PC, siempre; desde la red local, con la cookie de sesión del PIN."""
            if self._local():
                return True
            for part in (self.headers.get("Cookie") or "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == "hope_sesion" and app.sesiones.valida(v):
                    return True
            return False

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"ok": False, "mensaje": "host no permitido"})
            path = self.path.split("?", 1)[0]
            if path == "/api/estado":
                if not self._autorizado():
                    return self._json(401, {"ok": False, "login": True, "mensaje": "escribí el PIN"})
                return self._json(200, app.estado())
            if path in ("/", "/index.html"):
                path = "/index.html"
            name = path.lstrip("/")
            f = STATIC / name
            if "/" in name or "\\" in name or not f.is_file():
                return self._json(404, {"ok": False, "mensaje": "no existe"})
            self._send(200, f.read_bytes(), TIPOS.get(f.suffix, "application/octet-stream"))

        def do_POST(self):
            if not self._host_ok():
                return self._json(403, {"ok": False, "mensaje": "host no permitido"})
            if self.path not in ("/api/accion", "/api/login") or self.headers.get("X-Hope") != "1" or \
                    not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._json(403, {"ok": False, "mensaje": "pedido no permitido"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > 1_000_000:
                    return self._json(413, {"ok": False, "mensaje": "pedido demasiado grande"})
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/login":
                    ok, token = app.sesiones.login(app, self.client_address[0], str(body.get("pin", "")))
                    if not ok:
                        return self._json(401, {"ok": False, "mensaje": token})
                    self.send_response(200)
                    self.send_header("Set-Cookie", f"hope_sesion={token}; HttpOnly; SameSite=Strict; Path=/; "
                                                   f"Max-Age={red.DURACION_SESION}")
                    data = b'{"ok": true}'
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return None
                if not self._autorizado():
                    return self._json(401, {"ok": False, "login": True, "mensaje": "escribí el PIN"})
                accion = str(body.get("accion", ""))
                params = body.get("params") or {}
                if accion == "red_configurar":
                    return self._json(200, red.configurar(app, params, desde_local=self._local()))
                if accion == "red_hosts":
                    return self._json(200, red.configurar_hosts(app, params, desde_local=self._local()))
                res = app.accion(accion, params)
                return self._json(200, res)
            except KeyError as e:
                return self._json(400, {"ok": False, "mensaje": str(e).strip("'\"")})
            except (ValueError, TypeError) as e:
                return self._json(400, {"ok": False, "mensaje": str(e)})
            except Exception as e:                  # noqa: BLE001
                return self._json(500, {"ok": False, "mensaje": f"{type(e).__name__}: {e}"})

    return Handler


def serve(root: str | Path = ".", port: int = 8765, retomar: bool = True, host: str | None = None):
    """Crea la app y el servidor (sin bloquear). Devuelve (app, servidor, hilo).

    Escucha en 127.0.0.1 salvo que en la configuración esté activado el acceso desde la red local con PIN."""
    discos.aplicar_entorno(Path(root).resolve())
    app = App(root)
    rcfg = app.cfg.data.get("red") or {}
    if host is None:
        host = "0.0.0.0" if rcfg.get("lan") and rcfg.get("pin_hash") else "127.0.0.1"
    app.escucha = host
    ref = {"port": port}
    httpd = ThreadingHTTPServer((host, port), make_handler(app, ref))
    httpd.daemon_threads = True
    ref["port"] = httpd.server_address[1]
    app.puerto = ref["port"]
    th = threading.Thread(target=httpd.serve_forever, daemon=True, name="hope-http")
    th.start()
    app.cerrar_servidor = httpd.shutdown
    if retomar:
        app.entrenamiento.retomar_si_corresponde()
        ml = app.cfg.data.get("maestro_local") or {}
        app.turnos.retomar_si_corresponde()
        if ml.get("exe") and ml.get("modelo") and app.turnos.cfg()["modo"] == "compartida":
            def _maestro():
                r = app.servidor_maestro.iniciar(Path(ml["exe"]), Path(ml["modelo"]),
                                                 capas_gpu=int(ml.get("capas_gpu", 0)), hilos=int(ml.get("hilos", 4)))
                if r.get("ok") and app.cfg.data.get("maestro_continuo", True):
                    from hope.app.automatico import arrancar_explicaciones
                    arrancar_explicaciones(app)
            threading.Thread(target=_maestro, daemon=True).start()
    return app, httpd, th


def ya_corriendo(port: int) -> bool:
    """¿Hay otra instancia de la aplicación en ese puerto? (doble clic dos veces = abrir la misma)."""
    import http.client
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        c.request("GET", "/api/estado", headers={"Host": f"127.0.0.1:{port}"})
        r = c.getresponse()
        body = r.read(200)
        c.close()
        return r.status == 200 and body.startswith(b"{")
    except OSError:
        return False


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Aplicación de Hope: entrenar en casa con pasos guiados")
    ap.add_argument("--puerto", type=int, default=8765)
    ap.add_argument("--raiz", default=".", help="carpeta del proyecto (donde están configs/ y runs/)")
    ap.add_argument("--sin-navegador", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.raiz).resolve()
    if not (root / "configs").is_dir():
        raise SystemExit(f"no encuentro {root / 'configs'}: corré la aplicación desde la carpeta de Hope "
                         "o pasá --raiz <carpeta>")
    url = f"http://127.0.0.1:{args.puerto}"
    if ya_corriendo(args.puerto):
        print(f"Hope ya está abierta en {url}: abro el navegador.", flush=True)
        if not args.sin_navegador:
            webbrowser.open(url)
        return
    app, httpd, th = serve(root, args.puerto)
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    print(f"Hope: aplicación en {url}  (Ctrl+C para cerrar la interfaz; el entrenamiento sigue)", flush=True)
    if not args.sin_navegador:
        webbrowser.open(url)
    try:
        while th.is_alive():
            th.join(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        app.cerrar()
        httpd.shutdown()
