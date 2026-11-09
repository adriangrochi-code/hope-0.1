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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hope.app import guardian_app, sistema
from hope.app.config import PLANES, AppConfig, plan_efectivo
from hope.app.datos import TareaDatos, presupuestos
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


class App:
    def __init__(self, root: str | Path = "."):
        self.root = Path(root).resolve()
        self.cfg = AppConfig(self.root)
        cal = self.cfg.data.get("calibracion") or {}
        self.dispositivo = cal.get("dispositivo") if cal.get("dispositivo") in ("cuda", "xpu") else None
        self.tareas = {t.nombre: t for t in (TareaCalibracion(self), TareaDatos(self), TareaMaestro(self),
                                             TareaExportar(self), TareaImportar(self))}
        self.entrenamiento = Entrenamiento(self)

    def guardian(self):
        return guardian_app.cargar(self.root)

    def cerrar(self) -> None:
        self.entrenamiento.cerrar()
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
        }
        return {
            "version": 1, "hora": time.strftime("%H:%M:%S"),
            "config": cfg.data, "planes": PLANES, "plan": plan, "pasos": pasos,
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
        }

    # --- acciones (lista fija) ----------------------------------------------------
    def accion(self, nombre: str, p: dict) -> dict:
        if nombre not in ACCIONES:
            raise KeyError(f"acción desconocida: {nombre}")
        return ACCIONES[nombre](self, p or {})


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
    "maestro_probar": lambda a, p: probar(a.cfg),
    "maestro_iniciar": lambda a, p: a.tareas["maestro"].iniciar(rondas=int(p["rondas"]) if p.get("rondas") else None),
    "maestro_pausar": lambda a, p: a.tareas["maestro"].pausar(),
    "maestro_pasar_etapa": _pasar_etapa_maestro,
    "exportar_iniciar": lambda a, p: a.tareas["exportar"].iniciar(destino=str(p.get("destino", ""))),
    "exportar_pausar": lambda a, p: a.tareas["exportar"].pausar(),
    "importar_iniciar": lambda a, p: a.tareas["importar"].iniciar(zip_path=str(p.get("zip", ""))),
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
            host = (self.headers.get("Host") or "").strip().lower()
            return host in {f"127.0.0.1:{port_ref['port']}", f"localhost:{port_ref['port']}"}

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
            if self.path != "/api/accion" or self.headers.get("X-Hope") != "1" or \
                    not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._json(403, {"ok": False, "mensaje": "pedido no permitido"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > 1_000_000:
                    return self._json(413, {"ok": False, "mensaje": "pedido demasiado grande"})
                body = json.loads(self.rfile.read(n) or b"{}")
                res = app.accion(str(body.get("accion", "")), body.get("params") or {})
                return self._json(200, res)
            except KeyError as e:
                return self._json(400, {"ok": False, "mensaje": str(e).strip("'\"")})
            except (ValueError, TypeError) as e:
                return self._json(400, {"ok": False, "mensaje": str(e)})
            except Exception as e:                  # noqa: BLE001
                return self._json(500, {"ok": False, "mensaje": f"{type(e).__name__}: {e}"})

    return Handler


def serve(root: str | Path = ".", port: int = 8765, retomar: bool = True):
    """Crea la app y el servidor (sin bloquear). Devuelve (app, servidor, hilo)."""
    app = App(root)
    ref = {"port": port}
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, ref))
    httpd.daemon_threads = True
    ref["port"] = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True, name="hope-http")
    th.start()
    if retomar:
        app.entrenamiento.retomar_si_corresponde()
    return app, httpd, th


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
