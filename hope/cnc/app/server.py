"""Servidor local de la app Hope CNC (TISMA CNC): el «Mach3» moderno que corre en la propia PC.

Qué es
------
Un servidor HTTP mínimo (solo biblioteca estándar: ``http.server``) que hace de puente entre la
pantalla (``static/index.html`` + ``static/cnc.js``, que imitan la distribución de Mach3Mill) y la
sesión CNC (:class:`hope.cnc.session.CncSession`), que es la que habla con el controlador real
(grbl, simulador, etc.) a través de un «link» (:mod:`hope.cnc.links`).

Cómo encaja
-----------
* Lo arranca ``python -m hope.cnc app`` (ver :mod:`hope.cnc.__main__`), que llama a :func:`run_app`.
* :func:`run_app` arma la carpeta de datos, los perfiles tipo slicer (:mod:`hope.cnc.profiles`), el
  guardián (:func:`hope.cnc.controller.load_guardian`) y la sesión; después :func:`serve` levanta el
  servidor en un hilo y :func:`open_window` abre la pantalla en una ventana propia.
* La página consulta el estado con GET ``/api/estado`` varias veces por segundo y manda cada botón
  como POST ``/api/accion`` con ``{"accion": nombre, "params": {...}}`` (ver ``ACTIONS``).

Rutas HTTP
----------
* GET ``/`` e ``/index.html`` y los archivos de ``static/`` (sin subcarpetas).
* GET ``/api/estado``, ``/api/config``, ``/api/programa``, ``/api/trayectoria``, ``/api/origenes``,
  ``/api/recientes``, ``/api/perfiles`` y ``/api/perfil?tipo=…&nombre=…``: JSON de solo lectura.
* POST ``/api/accion``: la única forma de cambiar algo.

Seguridad (igual que ``hope/app``)
----------------------------------
* escucha solo en 127.0.0.1 y rechaza un ``Host`` que no sea local (contra «DNS rebinding»);
* las acciones son POST ``/api/accion`` con ``X-Hope: 1`` y JSON, que una página de otro origen no
  puede mandar sin permiso CORS (que este servidor nunca da);
* solo existen las acciones de ``ACTIONS``; ninguna ejecuta comandos del sistema;
* mover una máquina real pasa por el guardián (ver :mod:`hope.cnc.session`);
* cada respuesta lleva una política de contenido (CSP) estricta: solo scripts propios, sin ``eval``
  ni scripts en línea, y la página no se puede meter en un marco de otro sitio.

Convenciones
------------
Todas las respuestas de la API son JSON con ``"ok": true/false`` y, si falla, ``"mensaje"`` en
español para mostrarle al operador tal cual. Unidades: mm, mm/min y segundos, como en la sesión.

Ejemplo::

    python -m hope.cnc app --puerto 8766            # abre la ventana; Ctrl+C para salir
    curl -H "X-Hope: 1" -H "Content-Type: application/json" \\
         -d '{"accion": "ciclo"}' http://127.0.0.1:8766/api/accion     # = botón Cycle Start
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
# ========================================= CONFIGURACIÓN ==========================================
# ==================================================================================================

# Carpeta con la página (index.html, cnc.js, cnc.css, compat.*, logo.svg). Se sirven solo los
# archivos que están directamente acá: nada de subcarpetas ni rutas con «..» (ver do_GET).
STATIC = Path(__file__).parent / "static"
# Tipo MIME por extensión. Con «X-Content-Type-Options: nosniff» el navegador confía solo en esto, así
# que un .js tiene que salir como text/javascript o no se ejecuta. Lo que no esté acá se manda como
# application/octet-stream (se descarga, nunca se interpreta).
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
# Tamaño máximo del cuerpo de un POST (bytes): 32 MiB. Alcanza para un programa de G-code grande
# (cientos de miles de líneas de un CAM de plasma). Más grande se rechaza con 413 antes de leerlo, para
# no llenar la memoria de una PC vieja. cnc.js revisa el mismo límite antes de mandar el archivo.
MAX_BODY = 32 << 20               # un programa de G-code grande


def _b(v) -> bool:
    """Interpreta un parámetro booleano que puede llegar como ``true``, ``1``, ``"1"``, ``"true"`` o
    ``"si"``/``"sí"``. Cualquier otra cosa (incluido ``None``) es ``False``: ante la duda, no activar."""
    return v in (True, 1, "1", "true", "si", "sí")


# ==================================================================================================
# ============================================ ACCIONES ============================================
# ==================================================================================================

# Lista blanca de acciones que la pantalla puede pedir con POST /api/accion. Cada una recibe la sesión
# ``s`` y el diccionario de parámetros ``p`` del JSON, y devuelve el dict que se manda como respuesta.
# Los parámetros se convierten acá (str/int/float/_b) para que la sesión nunca reciba tipos raros; si falta
# uno obligatorio (KeyError) o no se puede convertir (ValueError/TypeError), do_POST responde 400.
# Equivalencias con Mach3 entre paréntesis.
ACTIONS = {
    "reset": lambda s, p: s.reset(),                                    # botón RESET (habilitar/deshabilitar)
    "aprobar": lambda s, p: s.approve(str(p["id"])),                    # el operador aprueba lo que pidió el guardián
    "cancelar": lambda s, p: s.cancel(),
    # Cargar programa (Load G-Code): el nombre se recorta a 200 caracteres, solo se usa para mostrarlo.
    "cargar": lambda s, p: s.load(str(p.get("nombre") or "programa.nc")[:200], str(p["texto"])),
    "cerrar_programa": lambda s, p: s.unload(),                         # Close G-Code
    "ciclo": lambda s, p: s.cycle_start(),                              # Cycle Start (Alt-R)
    "pausa": lambda s, p: s.feed_hold(),                                # Feed Hold (barra espaciadora)
    "parar": lambda s, p: s.stop(),                                     # Stop (Alt-S / Esc)
    "rebobinar": lambda s, p: s.rewind(),                               # Rewind (Ctrl-W)
    # Jog: eje, sentido (+1/-1), paso en mm (None = continuo mientras se mantiene apretado), rápido
    # (Shift en Mach3) y porcentaje de la velocidad máxima del eje (Slow Jog Rate de Mach3).
    "jog": lambda s, p: s.jog(str(p["eje"]), int(p.get("dir", 1)), float(p["paso"]) if p.get("paso") else None,
                              _b(p.get("rapido")), float(p.get("porcentaje", 30))),
    "jog_parar": lambda s, p: s.jog_stop(),                             # se soltó la tecla/botón de jog continuo
    "referenciar": lambda s, p: s.home(),                               # Ref All Home
    "desbloquear": lambda s, p: s.unlock(),
    "mdi": lambda s, p: s.mdi(str(p.get("linea", ""))),                 # una línea escrita en la pantalla MDI
    "cero": lambda s, p: s.zero(str(p["eje"])),                         # Zero X/Y/Z… ("todo" = todos los ejes)
    "ir_a_cero": lambda s, p: s.goto_zero(),                            # Goto Zero
    "override": lambda s, p: s.override(str(p["tipo"]), str(p["accion"])),   # FRO/SRO/rápidos: +, -, reset
    "husillo": lambda s, p: s.spindle(_b(p.get("on")), float(p["rpm"]) if p.get("rpm") else None),  # F5
    "refrigerante": lambda s, p: s.coolant(),                           # Flood (Ctrl-F)
    "sensar_altura": lambda s, p: s.plasma_probe(),                     # plasma: sensado de altura inicial (IHS)
    "opcion": lambda s, p: s.toggle(str(p["nombre"])),                  # Single BLK, Block Delete, Opt Stop
    # Set Next Line / Run From Here: el próximo Cycle Start arranca en esa línea.
    "linea_siguiente": lambda s, p: s.set_next_line(int(p["linea"]), _b(p.get("desde_aqui"))),
    "abrir_reciente": lambda s, p: s.load_recent(str(p["sha"])),        # Recent File (por sha256 del archivo)
    # Perfiles tipo slicer (máquina + material + consumible + proceso): elegir, guardar, borrar, importar.
    "perfil_elegir": lambda s, p: s.choose_profile(p.get("tipo"), p.get("nombre"),
                                                   float(p["espesor"]) if p.get("espesor") is not None else None),
    "perfil_guardar": lambda s, p: s.profile_save(str(p["tipo"]), str(p["nombre"]), dict(p.get("datos") or {}),
                                                  p.get("hereda") or None),
    "perfil_borrar": lambda s, p: s.profile_delete(str(p["tipo"]), str(p["nombre"])),
    "perfil_importar": lambda s, p: s.profile_import(str(p["tipo"]), str(p["texto"])),
    "disparo_prueba": lambda s, p: s.test_fire(),                       # plasma: encender la antorcha 0,3 s
}


# ==================================================================================================
# ========================================= SERVIDOR HTTP ==========================================
# ==================================================================================================

def _host_ok(host: str, port: int) -> bool:
    """¿El encabezado ``Host`` del pedido es este mismo servidor local?

    Defensa contra «DNS rebinding»: una página maliciosa puede hacer que su dominio resuelva a
    127.0.0.1 y así hablarle a este servidor «desde el mismo origen». Pero el navegador manda en
    ``Host`` el nombre del dominio atacante, no ``127.0.0.1``/``localhost``, y acá se rechaza.

    host: valor del encabezado (puede venir vacío). port: puerto real en el que escucha el servidor.
    Devuelve True solo para ``127.0.0.1`` o ``localhost``, con o sin ese puerto.
    """
    return host in (f"127.0.0.1:{port}", f"localhost:{port}", "127.0.0.1", "localhost")


def make_handler(session: CncSession, port_ref: dict):
    """Crea la clase manejadora de pedidos HTTP atada a una sesión.

    ``http.server`` instancia una clase por pedido y no deja pasarle argumentos, por eso se arma acá
    una clase nueva que «ve» ``session`` y ``port_ref`` por cierre (closure).

    session: la sesión CNC a la que se le piden estado y acciones.
    port_ref: ``{"port": n}``; es un dict (y no un int) porque con puerto 0 el sistema elige uno libre
              y :func:`serve` lo completa después de crear el servidor, cuando la clase ya existe.
    Devuelve la clase (subclase de ``BaseHTTPRequestHandler``).
    """
    class Handler(BaseHTTPRequestHandler):
        """Atiende un pedido: GET para la página y las consultas, POST solo para ``/api/accion``."""

        server_version = "HopeCNC/1"

        def log_message(self, fmt, *args):
            # Silencia el registro por consola: la pantalla consulta el estado varias veces por segundo y
            # llenaría la terminal (y en Windows XP escribir en consola es lento).
            pass

        # --- respuestas ---------------------------------------------------------------------------

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            """Manda una respuesta completa con los encabezados de seguridad de siempre.

            code: código HTTP. body: contenido ya en bytes. ctype: tipo MIME (Content-Type).
            """
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # Nada en caché: el estado cambia todo el tiempo y, al actualizar la app, la página nueva
            # tiene que cargarse sin que el operador tenga que borrar la caché del navegador.
            self.send_header("Cache-Control", "no-store")
            # El navegador no adivina tipos: solo ejecuta como script lo que sale como text/javascript.
            self.send_header("X-Content-Type-Options", "nosniff")
            # Ninguna página puede meter la pantalla en un <iframe> (evita «clickjacking» sobre Cycle Start).
            self.send_header("X-Frame-Options", "DENY")
            # CSP estricta: todo sale de este mismo servidor; sin scripts en línea ni eval (por eso viejo.js
            # detecta navegadores viejos probando funciones, sin eval). Estilos en línea sí: el HTML que arman
            # cnc.js y viejo.js trae atributos style="...". img data: deja usar imágenes embebidas.
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            """Manda ``obj`` como JSON UTF-8 (acentos sin escapar; lo que no sea serializable, como
            fechas o rutas, va como texto con ``str``)."""
            self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"),
                       "application/json; charset=utf-8")

        # --- GET: página y consultas ---------------------------------------------------------------

        def do_GET(self):
            """Sirve los archivos de la página y las consultas de solo lectura (``/api/...``).

            Un GET nunca cambia nada (las acciones son solo POST), así un enlace o una imagen de otra
            página que apunte acá no puede mover la máquina.
            """
            if not _host_ok(self.headers.get("Host") or "", port_ref["port"]):
                return self._json(403, {"ok": False, "mensaje": "host no permitido"})
            path, _, query = self.path.partition("?")
            if path == "/api/perfil":                    # detalle de un perfil: lleva parámetros en la URL
                from urllib.parse import parse_qs
                q = parse_qs(query)
                try:
                    return self._json(200, session.profile_detail(q.get("tipo", [""])[0], q.get("nombre", [""])[0]))
                except SessionError as e:
                    return self._json(404, {"ok": False, "mensaje": str(e)})
            # Consultas sin parámetros: ruta → método de la sesión que devuelve el dict a mandar.
            api = {"/api/perfiles": session.profile_list,"/api/estado": session.state, "/api/programa": session.program_lines,
                   "/api/trayectoria": session.toolpath, "/api/config": session.config,
                   "/api/origenes": session.offsets_table, "/api/recientes": session.recent_files}
            if path in api:
                return self._json(200, api[path]())
            name = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
            f = STATIC / name
            # Solo archivos sueltos de static/: con cualquier barra afuera no hay forma de escaparse de la
            # carpeta («../../secreto») ni de llegar a subcarpetas.
            if "/" in name or "\\" in name or not f.is_file():
                return self._json(404, {"ok": False, "mensaje": "no existe"})
            self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))

        # --- POST: acciones ------------------------------------------------------------------------

        def do_POST(self):
            """Ejecuta una acción de ``ACTIONS`` (lo que hace cada botón de la pantalla).

            Pide, además del Host local: ruta ``/api/accion``, encabezado ``X-Hope: 1`` y cuerpo JSON
            (``Content-Type: application/json``). Una página de otro sitio puede mandar un formulario
            POST a 127.0.0.1 sin permiso (CSRF), pero no con un encabezado propio ni con JSON: para eso el
            navegador primero pregunta por CORS y este servidor nunca responde que sí.

            Respuestas: 200 con lo que devuelve la acción; 400 si la acción no existe o los parámetros
            son inválidos o la sesión la rechaza; 403 si el pedido no cumple lo anterior; 413 si el
            cuerpo pasa ``MAX_BODY``; 500 ante un error inesperado (con su tipo, para diagnosticarlo).
            """
            if not _host_ok(self.headers.get("Host") or "", port_ref["port"]):
                return self._json(403, {"ok": False, "mensaje": "host no permitido"})
            if self.path != "/api/accion" or self.headers.get("X-Hope") != "1" or \
                    not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._json(403, {"ok": False, "mensaje": "pedido no permitido"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:                         # se rechaza sin leerlo: no se gasta memoria
                    return self._json(413, {"ok": False, "mensaje": "programa demasiado grande (máx. 32 MB)"})
                body = json.loads((self.rfile.read(n) or b"{}").decode("utf-8"))
                name = str(body.get("accion", ""))
                if name not in ACTIONS:
                    return self._json(400, {"ok": False, "mensaje": f"acción desconocida: {name}"})
                return self._json(200, ACTIONS[name](session, body.get("params") or {}))
            except (SessionError, BackendError, KeyError, ValueError, TypeError) as e:
                # Errores «esperables» (máquina deshabilitada, parámetro faltante…): el mensaje se muestra
                # al operador. KeyError trae el nombre entre comillas ('eje'): se le sacan.
                return self._json(400, {"ok": False, "mensaje": str(e).strip("'\"")})
            except Exception as e:                       # noqa: BLE001
                # Cualquier otra cosa es un error del programa: se informa en vez de cortar la conexión,
                # así la pantalla lo muestra y el servidor sigue andando.
                return self._json(500, {"ok": False, "mensaje": f"{type(e).__name__}: {e}"})

    return Handler


def serve(session: CncSession, port: int = 8766):
    """Arranca el servidor en un hilo; devuelve (servidor, hilo, url).

    session: sesión CNC a publicar. port: puerto local (8766 por defecto, el mismo que usa
    ``python -m hope.cnc app``; 0 = que el sistema elija uno libre, útil en pruebas).
    Escucha solo en 127.0.0.1: desde otra PC de la red no se puede entrar. El hilo es «daemon» para que
    Ctrl+C cierre el programa aunque el servidor siga atendiendo. ``ThreadingHTTPServer`` atiende cada
    pedido en su propio hilo: una consulta lenta (cargar un programa grande) no congela el estado.
    """
    port_ref = {"port": port}
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(session, port_ref))
    port_ref["port"] = srv.server_address[1]     # el real (si se pidió 0, el que eligió el sistema)
    th = threading.Thread(target=srv.serve_forever, daemon=True, name="hope-cnc-http")
    th.start()
    return srv, th, f"http://127.0.0.1:{port_ref['port']}/"


# ==================================================================================================
# ============================================ VENTANA =============================================
# ==================================================================================================

# Navegadores con modo «app» (--app=URL: ventana sin barra de direcciones ni pestañas, como un programa
# más). Se prueban en este orden: rutas fijas de Windows y macOS primero, después nombres que se buscan
# en el PATH (Linux). Supermium es un Chromium actual que todavía anda en Windows XP/Vista.
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
# sin Chrome/Edge (p. ej. Windows 7 o Linux viejo con Firefox ESR): ventana nueva de Firefox
FIREFOX = [r"C:\Program Files\Mozilla Firefox\firefox.exe", r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
           "firefox", "firefox-esr"]


def open_window(url: str, profile_dir: Path) -> str:
    """Abre la pantalla en una ventana propia (modo app); devuelve cómo se abrió.

    url: dirección local de la pantalla. profile_dir: carpeta de perfil propia del navegador (se crea si
    falta). Con un perfil aparte, la ventana no se mezcla con las pestañas del operador, recuerda su
    tamaño y no hereda extensiones que puedan molestar.
    Orden: Chrome/Edge/Chromium en modo app (1440×900), si no Firefox en ventana nueva, si no el navegador
    por defecto. Devuelve un texto corto para el mensaje de arranque («ventana de msedge», etc.).
    """
    for b in BROWSERS:
        exe = b if os.path.isabs(b) and os.path.exists(b) else shutil.which(b)
        if exe:
            profile_dir.mkdir(parents=True, exist_ok=True)
            # Popen sin esperar: el navegador sigue abierto aunque este proceso termine, y su salida no
            # ensucia la consola.
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
# ============================================ ARRANQUE ============================================
# ==================================================================================================

def data_dir(arg: str | None = None) -> Path:
    """Carpeta de datos (perfiles propios, recientes, guardián): --datos, TISMA_CNC_DATOS o runs/cnc.
    Instalada, apunta a %APPDATA%\\TISMA CNC (Program Files no se puede escribir).

    arg: valor de ``--datos`` (None si no se pasó). Devuelve la ruta (no la crea).
    """
    return Path(arg or os.environ.get("TISMA_CNC_DATOS") or Path("runs") / "cnc")


def guardian_config(datos: Path, explicit: str | None) -> str | None:
    """Configuración del guardián: la indicada, la del repositorio, o una propia en la carpeta de datos
    (JSON, así no hace falta pyyaml), para que las aprobaciones queden registradas también instalada.

    datos: carpeta de datos (ver :func:`data_dir`). explicit: ``--guardian`` si se pasó.
    Devuelve la ruta del archivo a usar. La de ``configs/guardian.yaml`` solo se toma corriendo desde el
    repositorio con la carpeta de datos por defecto (no se mezclan instalaciones con el repo). Si no hay
    ninguna, crea ``<datos>/guardian/guardian.json`` con el trabajo en ``datos`` y sus propios archivos
    (presupuesto, auditoría, aprobaciones) protegidos dentro de ``<datos>/guardian``.
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
    """¿Ya hay un TISMA CNC en este puerto? (segundo doble clic: solo se abre la ventana).

    Pregunta ``/api/config`` (con un Host válido, si no el otro servidor diría 403) y mira que la
    respuesta tenga ``marca``: así no se confunde con otro programa que use el mismo puerto.
    Espera como mucho 2 s. Devuelve False si no contesta nadie o contesta otra cosa.
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
    """Punto de entrada de ``python -m hope.cnc app``: arma todo, abre la ventana y espera a Ctrl+C.

    machine_path: configuración YAML de máquina de versiones anteriores (``--maquina``); se importa como
                  perfil de máquina propio y queda elegida. None = la selección guardada.
    port: puerto local (``--puerto``). window: abrir la ventana (False con ``--sin-ventana``).
    guardian_path: configuración del guardián (``--guardian``). datos: carpeta de datos (``--datos``).
    Devuelve el código de salida del proceso (0).

    Pasos: si ya hay uno abierto en el puerto, solo abre otra ventana y termina; si no, carga perfiles
    y selección, el guardián, crea la sesión con la máquina efectiva y sirve la pantalla. Al salir, si
    la máquina estaba habilitada la detiene, y cierra la conexión con el controlador.
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
    try:                                   # la selección de perfiles de la última vez (si hay)
        selection = json.loads((user / "perfiles" / "seleccion.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if machine_path:                       # configuración YAML de antes: se importa como perfil de máquina
        import yaml
        cfg = yaml.safe_load(Path(machine_path).read_text(encoding="utf-8")) or {}
        prof = machine_profile_from_config(cfg)
        name = str(cfg.get("nombre") or Path(machine_path).stem)
        # Nunca pisar un perfil del sistema con el mismo nombre: se guarda aparte.
        if name in store.names("maquina") and store.raw("maquina", name)[1] == "sistema":
            name += " (YAML)"
        store.save("maquina", name, prof)
        selection = {"maquina": name}
    from hope.cnc.controller import load_guardian
    guardian = load_guardian(guardian_config(user, guardian_path))   # mover una máquina real pide aprobación

    from hope.cnc.profiles import effective, fix_selection
    sel = fix_selection(store, selection)      # descarta perfiles que ya no existen
    m, _ = effective(store, sel)               # máquina + material + consumible + proceso → máquina efectiva
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
        if session.enabled:                # al cerrar, la máquina no queda moviéndose
            session.link.stop()
        session.link.close()
        srv.shutdown()
    return 0
