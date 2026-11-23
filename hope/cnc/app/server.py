"""Servidor local de la app Hope CNC.

Seguridad (igual que ``hope/app``):

* escucha solo en 127.0.0.1 y rechaza un ``Host`` que no sea local (contra «DNS rebinding»);
* las acciones son POST ``/api/accion`` con ``X-Hope: 1`` y JSON, que una página de otro origen no
  puede mandar sin permiso CORS (que este servidor nunca da);
* solo existen las acciones de ``ACTIONS``; ninguna ejecuta comandos del sistema;
* mover una máquina real pasa por el guardián (ver :mod:`hope.cnc.session`).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hope.cnc.backends.base import BackendError
from hope.cnc.session import CncSession, SessionError

STATIC = Path(__file__).parent / "static"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
MAX_BODY = 32 << 20               # un programa de G-code grande


def _b(v) -> bool:
    return v in (True, 1, "1", "true", "si", "sí")


ACTIONS = {
    "reset": lambda s, p: s.reset(),
    "aprobar": lambda s, p: s.approve(str(p["id"])),
    "cancelar": lambda s, p: s.cancel(),
    "cargar": lambda s, p: s.load(str(p.get("nombre") or "programa.nc")[:200], str(p["texto"])),
    "cerrar_programa": lambda s, p: s.unload(),
    "ciclo": lambda s, p: s.cycle_start(),
    "pausa": lambda s, p: s.feed_hold(),
    "parar": lambda s, p: s.stop(),
    "rebobinar": lambda s, p: s.rewind(),
    "jog": lambda s, p: s.jog(str(p["eje"]), int(p.get("dir", 1)), float(p["paso"]) if p.get("paso") else None,
                              _b(p.get("rapido")), float(p.get("porcentaje", 30))),
    "jog_parar": lambda s, p: s.jog_stop(),
    "referenciar": lambda s, p: s.home(),
    "desbloquear": lambda s, p: s.unlock(),
    "mdi": lambda s, p: s.mdi(str(p.get("linea", ""))),
    "cero": lambda s, p: s.zero(str(p["eje"])),
    "ir_a_cero": lambda s, p: s.goto_zero(),
    "override": lambda s, p: s.override(str(p["tipo"]), str(p["accion"])),
    "husillo": lambda s, p: s.spindle(_b(p.get("on")), float(p["rpm"]) if p.get("rpm") else None),
    "refrigerante": lambda s, p: s.coolant(),
    "sensar_altura": lambda s, p: s.plasma_probe(),
    "opcion": lambda s, p: s.toggle(str(p["nombre"])),
    "linea_siguiente": lambda s, p: s.set_next_line(int(p["linea"]), _b(p.get("desde_aqui"))),
    "abrir_reciente": lambda s, p: s.load_recent(str(p["sha"])),
    "perfil_elegir": lambda s, p: s.choose_profile(p.get("tipo"), p.get("nombre"),
                                                   float(p["espesor"]) if p.get("espesor") is not None else None),
    "perfil_guardar": lambda s, p: s.profile_save(str(p["tipo"]), str(p["nombre"]), dict(p.get("datos") or {}),
                                                  p.get("hereda") or None),
    "perfil_borrar": lambda s, p: s.profile_delete(str(p["tipo"]), str(p["nombre"])),
    "perfil_importar": lambda s, p: s.profile_import(str(p["tipo"]), str(p["texto"])),
    "disparo_prueba": lambda s, p: s.test_fire(),
}


def _host_ok(host: str, port: int) -> bool:
    return host in (f"127.0.0.1:{port}", f"localhost:{port}", "127.0.0.1", "localhost")


def make_handler(session: CncSession, port_ref: dict):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HopeCNC/1"

        def log_message(self, fmt, *args):
            pass

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
            if not _host_ok(self.headers.get("Host") or "", port_ref["port"]):
                return self._json(403, {"ok": False, "mensaje": "host no permitido"})
            path, _, query = self.path.partition("?")
            if path == "/api/perfil":
                from urllib.parse import parse_qs
                q = parse_qs(query)
                try:
                    return self._json(200, session.profile_detail(q.get("tipo", [""])[0], q.get("nombre", [""])[0]))
                except SessionError as e:
                    return self._json(404, {"ok": False, "mensaje": str(e)})
            api = {"/api/perfiles": session.profile_list,"/api/estado": session.state, "/api/programa": session.program_lines,
                   "/api/trayectoria": session.toolpath, "/api/config": session.config,
                   "/api/origenes": session.offsets_table, "/api/recientes": session.recent_files}
            if path in api:
                return self._json(200, api[path]())
            name = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
            f = STATIC / name
            if "/" in name or "\\" in name or not f.is_file():
                return self._json(404, {"ok": False, "mensaje": "no existe"})
            self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))

        def do_POST(self):
            if not _host_ok(self.headers.get("Host") or "", port_ref["port"]):
                return self._json(403, {"ok": False, "mensaje": "host no permitido"})
            if self.path != "/api/accion" or self.headers.get("X-Hope") != "1" or \
                    not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._json(403, {"ok": False, "mensaje": "pedido no permitido"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    return self._json(413, {"ok": False, "mensaje": "programa demasiado grande (máx. 32 MB)"})
                body = json.loads(self.rfile.read(n) or b"{}")
                name = str(body.get("accion", ""))
                if name not in ACTIONS:
                    return self._json(400, {"ok": False, "mensaje": f"acción desconocida: {name}"})
                return self._json(200, ACTIONS[name](session, body.get("params") or {}))
            except (SessionError, BackendError, KeyError, ValueError, TypeError) as e:
                return self._json(400, {"ok": False, "mensaje": str(e).strip("'\"")})
            except Exception as e:                       # noqa: BLE001
                return self._json(500, {"ok": False, "mensaje": f"{type(e).__name__}: {e}"})

    return Handler


def serve(session: CncSession, port: int = 8766):
    """Arranca el servidor en un hilo; devuelve (servidor, hilo, url)."""
    port_ref = {"port": port}
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(session, port_ref))
    port_ref["port"] = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True, name="hope-cnc-http")
    th.start()
    return srv, th, f"http://127.0.0.1:{port_ref['port']}/"


BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "microsoft-edge", "brave-browser",
]


def open_window(url: str, profile_dir: Path) -> str:
    """Abre la pantalla en una ventana propia (modo app); devuelve cómo se abrió."""
    for b in BROWSERS:
        exe = b if os.path.isabs(b) and os.path.exists(b) else shutil.which(b)
        if exe:
            profile_dir.mkdir(parents=True, exist_ok=True)
            subprocess.Popen([exe, f"--app={url}", "--window-size=1440,900", f"--user-data-dir={profile_dir}",
                              "--no-first-run", "--no-default-browser-check"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"ventana de {Path(exe).stem}"
    webbrowser.open(url)
    return "navegador por defecto"


def run_app(machine_path: str | None = None, port: int = 8766, window: bool = True,
            guardian_path: str = "configs/guardian.yaml") -> int:
    import json

    from hope.cnc.links import make_link
    from hope.cnc.profiles import ProfileStore, machine_profile_from_config
    user = Path("runs") / "cnc"
    store = ProfileStore(user_dir=user / "perfiles")
    selection: dict = {}
    try:
        selection = json.loads((user / "perfiles" / "seleccion.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if machine_path:                       # configuración YAML de antes: se importa como perfil de máquina
        import yaml
        cfg = yaml.safe_load(Path(machine_path).read_text(encoding="utf-8")) or {}
        prof = machine_profile_from_config(cfg)
        name = str(cfg.get("nombre") or Path(machine_path).stem)
        if name in store.names("maquina") and store.raw("maquina", name)[1] == "sistema":
            name += " (YAML)"
        store.save("maquina", name, prof)
        selection = {"maquina": name}
    from hope.cnc.controller import load_guardian
    guardian = load_guardian(guardian_path)  # mover una máquina real pide aprobación del guardián

    from hope.cnc.profiles import effective, fix_selection
    sel = fix_selection(store, selection)
    m, _ = effective(store, sel)
    session = CncSession(m, make_link(m), guardian, "perfiles", recent_dir=str(user / "recientes"),
                         profiles=store, selection=sel, link_factory=make_link)
    srv, th, url = serve(session, port)
    how = open_window(url, Path("runs") / "cnc_ventana") if window else "sin ventana"
    from hope.cnc.session import BRAND
    print(f"{BRAND} en {url} ({how}); máquina «{session.m.name}», conexión {session.link.name}. Ctrl+C para salir.")
    try:
        th.join()
    except KeyboardInterrupt:
        pass
    finally:
        if session.enabled:
            session.link.stop()
        session.link.close()
        srv.shutdown()
    return 0
