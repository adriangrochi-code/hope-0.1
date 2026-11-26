"""Local server of the Hope CNC app (TISMA CNC): the modern «Mach3» that runs on the PC itself.

What it is
----------
A minimal HTTP server (standard library only: ``http.server``) that bridges the screen
(``static/index.html`` + ``static/cnc.js``, which mimic the Mach3Mill layout) and the CNC session
(:class:`hope.cnc.session.CncSession`), which is what talks to the real controller (grbl,
simulator, etc.) through a «link» (:mod:`hope.cnc.links`).

How it fits in
--------------
* Started by ``python -m hope.cnc app`` (see :mod:`hope.cnc.__main__`), which calls :func:`run_app`.
* :func:`run_app` sets up the data folder, the slicer-style profiles (:mod:`hope.cnc.profiles`), the
  guardian (:func:`hope.cnc.controller.load_guardian`) and the session; then :func:`serve` starts the
  server in a thread and :func:`open_window` opens the screen in its own window.
* The page polls the state with GET ``/api/estado`` several times per second and sends each button
  as POST ``/api/accion`` with ``{"accion": name, "params": {...}}`` (see ``ACTIONS``).

HTTP routes
-----------
* GET ``/`` and ``/index.html`` and the files in ``static/`` (no subfolders).
* GET ``/api/estado``, ``/api/config``, ``/api/programa``, ``/api/trayectoria``, ``/api/origenes``,
  ``/api/recientes``, ``/api/perfiles`` and ``/api/perfil?tipo=…&nombre=…``: read-only JSON.
* POST ``/api/accion``: the only way to change anything.

Security (same as ``hope/app``)
-------------------------------
* listens only on 127.0.0.1 and rejects a non-local ``Host`` (against «DNS rebinding»);
* actions are POST ``/api/accion`` with ``X-Hope: 1`` and JSON, which a page from another origin
  cannot send without CORS permission (which this server never grants);
* only the actions in ``ACTIONS`` exist; none of them runs system commands;
* moving a real machine goes through the guardian (see :mod:`hope.cnc.session`);
* every response carries a strict content policy (CSP): only our own scripts, no ``eval`` and no
  inline scripts, and the page cannot be embedded in a frame from another site.

Conventions
-----------
All API responses are JSON with ``"ok": true/false`` and, on failure, a ``"mensaje"`` in Spanish
to show the operator as-is. Units: mm, mm/min and seconds, as in the session.

Example::

    python -m hope.cnc app --puerto 8766            # opens the window; Ctrl+C to quit
    curl -H "X-Hope: 1" -H "Content-Type: application/json" \\
         -d '{"accion": "ciclo"}' http://127.0.0.1:8766/api/accion     # = Cycle Start button
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

# ==================================================================================================
# ======================================== CONFIGURATION ===========================================
# ==================================================================================================

# Folder with the page (index.html, cnc.js, cnc.css, compat.*, logo.svg). Only files sitting directly
# in it are served: no subfolders and no paths with «..» (see do_GET).
STATIC = Path(__file__).parent / "static"
# MIME type by extension. With «X-Content-Type-Options: nosniff» the browser trusts only this, so a
# .js must go out as text/javascript or it will not run. Anything not listed here is sent as
# application/octet-stream (downloaded, never interpreted).
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
# Maximum POST body size (bytes): 32 MiB. Enough for a large G-code program (hundreds of thousands
# of lines from a plasma CAM). Anything larger is rejected with 413 before reading it, so the memory
# of an old PC does not fill up. cnc.js checks the same limit before sending the file.
MAX_BODY = 32 << 20               # a large G-code program


def _b(v) -> bool:
    """Interprets a boolean parameter that may arrive as ``true``, ``1``, ``"1"``, ``"true"`` or
    ``"si"``/``"sí"``. Anything else (including ``None``) is ``False``: when in doubt, do not enable."""
    return v in (True, 1, "1", "true", "si", "sí")


# ==================================================================================================
# ============================================ ACTIONS =============================================
# ==================================================================================================

# Whitelist of actions the screen can request with POST /api/accion. Each one receives the session
# ``s`` and the JSON parameter dictionary ``p``, and returns the dict sent back as the response.
# Parameters are converted here (str/int/float/_b) so the session never gets odd types; if a required
# one is missing (KeyError) or cannot be converted (ValueError/TypeError), do_POST answers 400.
# Mach3 equivalents in comments.
ACTIONS = {
    "reset": lambda s, p: s.reset(),                                    # RESET button (enable/disable)
    "aprobar": lambda s, p: s.approve(str(p["id"])),                    # operator approves the guardian request
    "cancelar": lambda s, p: s.cancel(),
    # Load program (Load G-Code): the name is cut to 200 characters; it is only used for display.
    "asistente": lambda s, p: s.run_wizard(str(p["tipo"]), dict(p.get("datos") or {})),
    "palpado": lambda s, p: s.run_probe(str(p["tipo"]), dict(p.get("datos") or {})),   # probing routine
    "latencia": lambda s, p: s.latency_test(float(p.get("segundos") or 5.0)),   # parallel port DriverTest
    "aplicar_mapa": lambda s, p: s.apply_height_map(float(p.get("tramo") or 5.0)),   # height map → program
    "cargar": lambda s, p: s.load(str(p.get("nombre") or "programa.nc")[:200], str(p["texto"])),
    "cerrar_programa": lambda s, p: s.unload(),                         # Close G-Code
    "ciclo": lambda s, p: s.cycle_start(),                              # Cycle Start (Alt-R)
    "pausa": lambda s, p: s.feed_hold(),                                # Feed Hold (space bar)
    "parar": lambda s, p: s.stop(),                                     # Stop (Alt-S / Esc)
    "rebobinar": lambda s, p: s.rewind(),                               # Rewind (Ctrl-W)
    # Jog: axis, direction (+1/-1), step in mm (None = continuous while held down), rapid
    # (Shift in Mach3) and percentage of the axis maximum speed (Mach3 Slow Jog Rate).
    "jog": lambda s, p: s.jog(str(p["eje"]), int(p.get("dir", 1)), float(p["paso"]) if p.get("paso") else None,
                              _b(p.get("rapido")), float(p.get("porcentaje", 30))),
    "jog_parar": lambda s, p: s.jog_stop(),                             # the continuous-jog key/button was released
    "referenciar": lambda s, p: s.home(),                               # Ref All Home
    "desbloquear": lambda s, p: s.unlock(),
    "mdi": lambda s, p: s.mdi(str(p.get("linea", ""))),                 # one line typed on the MDI screen
    "cero": lambda s, p: s.zero(str(p["eje"])),                         # Zero X/Y/Z… ("todo" = all axes)
    "ir_a_cero": lambda s, p: s.goto_zero(),                            # Goto Zero
    "override": lambda s, p: s.override(str(p["tipo"]), str(p["accion"])),   # FRO/SRO/rapids: +, -, reset
    "husillo": lambda s, p: s.spindle(_b(p.get("on")), float(p["rpm"]) if p.get("rpm") else None),  # F5
    "refrigerante": lambda s, p: s.coolant(),                           # Flood (Ctrl-F)
    "sensar_altura": lambda s, p: s.plasma_probe(),                     # plasma: initial height sensing (IHS)
    "opcion": lambda s, p: s.toggle(str(p["nombre"])),                  # Single BLK, Block Delete, Opt Stop
    # Set Next Line / Run From Here: the next Cycle Start begins at that line.
    "linea_siguiente": lambda s, p: s.set_next_line(int(p["linea"]), _b(p.get("desde_aqui"))),
    "abrir_reciente": lambda s, p: s.load_recent(str(p["sha"])),        # Recent File (by the file's sha256)
    # Slicer-style profiles (machine + material + consumable + process): choose, save, delete, import.
    "perfil_elegir": lambda s, p: s.choose_profile(p.get("tipo"), p.get("nombre"),
                                                   float(p["espesor"]) if p.get("espesor") is not None else None),
    "perfil_guardar": lambda s, p: s.profile_save(str(p["tipo"]), str(p["nombre"]), dict(p.get("datos") or {}),
                                                  p.get("hereda") or None),
    "perfil_borrar": lambda s, p: s.profile_delete(str(p["tipo"]), str(p["nombre"])),
    "perfil_importar": lambda s, p: s.profile_import(str(p["tipo"]), str(p["texto"])),
    "disparo_prueba": lambda s, p: s.test_fire(),                       # plasma: fire the torch for 0.3 s
}


# ==================================================================================================
# ========================================== HTTP SERVER ===========================================
# ==================================================================================================

def _host_ok(host: str, port: int) -> bool:
    """Is the request's ``Host`` header this very local server?

    Defense against «DNS rebinding»: a malicious page can make its domain resolve to 127.0.0.1 and
    so talk to this server «from the same origin». But the browser sends the attacker's domain name
    in ``Host``, not ``127.0.0.1``/``localhost``, and it is rejected here.

    host: header value (may be empty). port: actual port the server listens on.
    Returns True only for ``127.0.0.1`` or ``localhost``, with or without that port.
    """
    return host in (f"127.0.0.1:{port}", f"localhost:{port}", "127.0.0.1", "localhost")


def make_handler(session: CncSession, port_ref: dict):
    """Creates the HTTP request handler class bound to a session.

    ``http.server`` instantiates one class per request and does not allow passing arguments to it,
    so a new class is built here that «sees» ``session`` and ``port_ref`` through a closure.

    session: the CNC session asked for state and actions.
    port_ref: ``{"port": n}``; it is a dict (not an int) because with port 0 the system picks a free
              one and :func:`serve` fills it in after creating the server, when the class already exists.
    Returns the class (a subclass of ``BaseHTTPRequestHandler``).
    """
    class Handler(BaseHTTPRequestHandler):
        """Handles one request: GET for the page and queries, POST only for ``/api/accion``."""

        server_version = "HopeCNC/1"

        def log_message(self, fmt, *args):
            # Silences console logging: the screen polls the state several times per second and would
            # flood the terminal (and on Windows XP writing to the console is slow).
            pass

        # --- responses ----------------------------------------------------------------------------

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            """Sends a complete response with the usual security headers.

            code: HTTP status code. body: content already in bytes. ctype: MIME type (Content-Type).
            """
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # Nothing cached: the state changes all the time and, when the app is updated, the new page
            # must load without the operator having to clear the browser cache.
            self.send_header("Cache-Control", "no-store")
            # The browser does not guess types: it only runs as script what goes out as text/javascript.
            self.send_header("X-Content-Type-Options", "nosniff")
            # No page can put the screen inside an <iframe> (prevents «clickjacking» on Cycle Start).
            self.send_header("X-Frame-Options", "DENY")
            # Strict CSP: everything comes from this same server; no inline scripts and no eval (that is why
            # viejo.js detects old browsers by probing functions, without eval). Inline styles are allowed:
            # the HTML built by cnc.js and viejo.js has style="..." attributes. img data: allows embedded images.
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            """Sends ``obj`` as UTF-8 JSON (accents unescaped; anything not serializable, such as
            dates or paths, goes as text via ``str``)."""
            self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"),
                       "application/json; charset=utf-8")

        # --- GET: page and queries -----------------------------------------------------------------

        def do_GET(self):
            """Serves the page files and the read-only queries (``/api/...``).

            A GET never changes anything (actions are POST only), so a link or an image on another
            page pointing here cannot move the machine.
            """
            if not _host_ok(self.headers.get("Host") or "", port_ref["port"]):
                return self._json(403, {"ok": False, "mensaje": "host no permitido"})
            path, _, query = self.path.partition("?")
            if path == "/api/perfil":                    # profile detail: takes parameters in the URL
                from urllib.parse import parse_qs
                q = parse_qs(query)
                try:
                    return self._json(200, session.profile_detail(q.get("tipo", [""])[0], q.get("nombre", [""])[0]))
                except SessionError as e:
                    return self._json(404, {"ok": False, "mensaje": str(e)})
            # Parameterless queries: route → session method that returns the dict to send.
            api = {"/api/perfiles": session.profile_list,"/api/estado": session.state, "/api/programa": session.program_lines,
                   "/api/trayectoria": session.toolpath, "/api/config": session.config,
                   "/api/origenes": session.offsets_table, "/api/recientes": session.recent_files,
                   "/api/asistentes": session.wizards, "/api/palpado": session.probe_forms}
            if path in api:
                return self._json(200, api[path]())
            name = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
            f = STATIC / name
            # Only plain files in static/: with every slash rejected there is no way to escape the
            # folder («../../secreto») or reach subfolders.
            if "/" in name or "\\" in name or not f.is_file():
                return self._json(404, {"ok": False, "mensaje": "no existe"})
            self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))

        # --- POST: actions -------------------------------------------------------------------------

        def do_POST(self):
            """Runs an action from ``ACTIONS`` (what each screen button does).

            Requires, besides the local Host: path ``/api/accion``, header ``X-Hope: 1`` and a JSON body
            (``Content-Type: application/json``). A page from another site can send a POST form to
            127.0.0.1 without permission (CSRF), but not with a custom header or with JSON: for that the
            browser first asks via CORS and this server never says yes.

            Responses: 200 with what the action returns; 400 if the action does not exist, the parameters
            are invalid or the session rejects it; 403 if the request does not meet the above; 413 if the
            body exceeds ``MAX_BODY``; 500 on an unexpected error (with its type, for diagnosis).
            """
            if not _host_ok(self.headers.get("Host") or "", port_ref["port"]):
                return self._json(403, {"ok": False, "mensaje": "host no permitido"})
            if self.path != "/api/accion" or self.headers.get("X-Hope") != "1" or \
                    not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._json(403, {"ok": False, "mensaje": "pedido no permitido"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:                         # rejected without reading it: no memory spent
                    return self._json(413, {"ok": False, "mensaje": "programa demasiado grande (máx. 32 MB)"})
                body = json.loads((self.rfile.read(n) or b"{}").decode("utf-8"))
                name = str(body.get("accion", ""))
                if name not in ACTIONS:
                    return self._json(400, {"ok": False, "mensaje": f"acción desconocida: {name}"})
                return self._json(200, ACTIONS[name](session, body.get("params") or {}))
            except (SessionError, BackendError, KeyError, ValueError, TypeError) as e:
                # «Expected» errors (machine disabled, missing parameter…): the message is shown to the
                # operator. KeyError wraps the name in quotes ('eje'): they are stripped.
                return self._json(400, {"ok": False, "mensaje": str(e).strip("'\"")})
            except Exception as e:                       # noqa: BLE001
                # Anything else is a program bug: it is reported instead of dropping the connection,
                # so the screen shows it and the server keeps running.
                return self._json(500, {"ok": False, "mensaje": f"{type(e).__name__}: {e}"})

    return Handler


def serve(session: CncSession, port: int = 8766):
    """Starts the server in a thread; returns (server, thread, url).

    session: CNC session to publish. port: local port (8766 by default, the same one
    ``python -m hope.cnc app`` uses; 0 = let the system pick a free one, useful in tests).
    Listens only on 127.0.0.1: it cannot be reached from another PC on the network. The thread is a
    «daemon» so Ctrl+C closes the program even while the server is still serving. ``ThreadingHTTPServer``
    serves each request in its own thread: a slow request (loading a large program) does not freeze the state.
    """
    port_ref = {"port": port}
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(session, port_ref))
    port_ref["port"] = srv.server_address[1]     # the actual one (if 0 was asked, the one the system picked)
    th = threading.Thread(target=srv.serve_forever, daemon=True, name="hope-cnc-http")
    th.start()
    return srv, th, f"http://127.0.0.1:{port_ref['port']}/"


# ==================================================================================================
# ============================================= WINDOW =============================================
# ==================================================================================================

# Browsers with «app» mode (--app=URL: a window without address bar or tabs, like any other program).
# They are tried in this order: fixed Windows and macOS paths first, then names looked up in the PATH
# (Linux). Supermium is a current Chromium that still runs on Windows XP/Vista.
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Supermium\chrome.exe", r"C:\Program Files (x86)\Supermium\chrome.exe",   # Vista/XP
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "microsoft-edge", "brave-browser",
]
# without Chrome/Edge (e.g. Windows 7 or an old Linux with Firefox ESR): a new Firefox window
FIREFOX = [r"C:\Program Files\Mozilla Firefox\firefox.exe", r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
           "firefox", "firefox-esr"]


def open_window(url: str, profile_dir: Path) -> str:
    """Opens the screen in its own window (app mode); returns how it was opened.

    url: local address of the screen. profile_dir: the browser's own profile folder (created if
    missing). With a separate profile, the window does not mix with the operator's tabs, remembers its
    size and does not inherit extensions that might get in the way.
    Order: Chrome/Edge/Chromium in app mode (1440×900), otherwise Firefox in a new window, otherwise the
    default browser. Returns a short text for the startup message («ventana de msedge», etc.).
    """
    for b in BROWSERS:
        exe = b if os.path.isabs(b) and os.path.exists(b) else shutil.which(b)
        if exe:
            profile_dir.mkdir(parents=True, exist_ok=True)
            # Popen without waiting: the browser stays open even if this process ends, and its output
            # does not clutter the console.
            subprocess.Popen([exe, f"--app={url}", "--window-size=1440,900", f"--user-data-dir={profile_dir}",
                              "--no-first-run", "--no-default-browser-check"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"ventana de {Path(exe).stem}"
    for b in FIREFOX:
        exe = b if os.path.isabs(b) and os.path.exists(b) else shutil.which(b)
        if exe:
            subprocess.Popen([exe, "-new-window", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "ventana de Firefox"
    webbrowser.open(url)
    return "navegador por defecto"


# ==================================================================================================
# ============================================ STARTUP =============================================
# ==================================================================================================

def data_dir(arg: str | None = None) -> Path:
    """Data folder (own profiles, recent files, guardian): --datos, TISMA_CNC_DATOS or runs/cnc.
    When installed, it points to %APPDATA%\\TISMA CNC (Program Files is not writable).

    arg: value of ``--datos`` (None if not given). Returns the path (does not create it).
    """
    return Path(arg or os.environ.get("TISMA_CNC_DATOS") or Path("runs") / "cnc")


def guardian_config(datos: Path, explicit: str | None) -> str | None:
    """Guardian configuration: the one given, the repository's, or our own in the data folder
    (JSON, so pyyaml is not needed), so that approvals are recorded in an installed copy too.

    datos: data folder (see :func:`data_dir`). explicit: ``--guardian`` if given.
    Returns the path of the file to use. ``configs/guardian.yaml`` is only used when running from the
    repository with the default data folder (installations are not mixed with the repo). If there is
    none, creates ``<datos>/guardian/guardian.json`` with the workdir at ``datos`` and its own files
    (budget, audit, approvals) protected inside ``<datos>/guardian``.
    """
    if explicit and Path(explicit).exists():
        return explicit
    if explicit is None and Path("configs/guardian.yaml").exists() and datos == Path("runs") / "cnc":
        return "configs/guardian.yaml"
    cfg = datos / "guardian" / "guardian.json"
    if not cfg.exists():
        g = (datos / "guardian").resolve()
        g.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({"workdir": str(datos.resolve()), "protected_paths": [str(g)],
                                   "budget_path": str(g / "presupuesto.yaml"), "audit_path": str(g / "auditoria.jsonl"),
                                   "approvals_path": str(g / "aprobaciones.json")}, indent=2), encoding="utf-8")
    return str(cfg)


def already_running(port: int) -> bool:
    """Is a TISMA CNC already running on this port? (second double click: only the window opens).

    Asks for ``/api/config`` (with a valid Host, otherwise the other server would say 403) and checks
    that the response contains ``marca``: so it is not mistaken for another program on the same port.
    Waits 2 s at most. Returns False if nobody answers or something else answers.
    """
    import http.client
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        c.request("GET", "/api/config", headers={"Host": f"127.0.0.1:{port}"})
        r = c.getresponse()
        return r.status == 200 and b"marca" in r.read()
    except OSError:
        return False


def run_app(machine_path: str | None = None, port: int = 8766, window: bool = True,
            guardian_path: str | None = None, datos: str | None = None) -> int:
    """Entry point of ``python -m hope.cnc app``: sets everything up, opens the window and waits for Ctrl+C.

    machine_path: machine YAML configuration from earlier versions (``--maquina``); it is imported as
                  an own machine profile and becomes the selected one. None = the saved selection.
    port: local port (``--puerto``). window: open the window (False with ``--sin-ventana``).
    guardian_path: guardian configuration (``--guardian``). datos: data folder (``--datos``).
    Returns the process exit code (0).

    Steps: if one is already open on the port, just opens another window and returns; otherwise loads
    profiles and selection, the guardian, creates the session with the effective machine and serves the
    screen. On exit, stops the machine if it was enabled, and closes the connection to the controller.
    """
    from hope.cnc.links import make_link
    from hope.cnc.profiles import ProfileStore, machine_profile_from_config
    user = data_dir(datos)
    if already_running(port):
        url = f"http://127.0.0.1:{port}/"
        how = open_window(url, user / "ventana") if window else "sin ventana"
        print(f"TISMA CNC ya estaba abierto en {url} ({how}).")
        return 0
    user.mkdir(parents=True, exist_ok=True)
    store = ProfileStore(user_dir=user / "perfiles")
    selection: dict = {}
    try:                                   # the profile selection from last time (if any)
        selection = json.loads((user / "perfiles" / "seleccion.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if machine_path:                       # old YAML configuration: imported as a machine profile
        import yaml
        cfg = yaml.safe_load(Path(machine_path).read_text(encoding="utf-8")) or {}
        prof = machine_profile_from_config(cfg)
        name = str(cfg.get("nombre") or Path(machine_path).stem)
        # Never overwrite a system profile with the same name: it is saved separately.
        if name in store.names("maquina") and store.raw("maquina", name)[1] == "sistema":
            name += " (YAML)"
        store.save("maquina", name, prof)
        selection = {"maquina": name}
    from hope.cnc.controller import load_guardian
    guardian = load_guardian(guardian_config(user, guardian_path))   # moving a real machine requires approval

    from hope.cnc.profiles import effective, fix_selection
    sel = fix_selection(store, selection)      # drops profiles that no longer exist
    m, _ = effective(store, sel)               # machine + material + consumable + process → effective machine
    session = CncSession(m, make_link(m), guardian, "perfiles", recent_dir=str(user / "recientes"),
                         profiles=store, selection=sel, link_factory=make_link)
    srv, th, url = serve(session, port)
    how = open_window(url, user / "ventana") if window else "sin ventana"
    from hope.cnc.session import BRAND
    print(f"{BRAND} en {url} ({how}); máquina «{session.m.name}», conexión {session.link.name}. Ctrl+C para salir.")
    try:
        th.join()
    except KeyboardInterrupt:
        pass
    finally:
        if session.enabled:                # on close, the machine is not left moving
            session.link.stop()
        session.link.close()
        srv.shutdown()
    return 0
