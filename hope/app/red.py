"""Acceso a la aplicación desde el celular, en la misma red local (opcional, con PIN).

Por defecto la aplicación escucha **solo en 127.0.0.1**. Con «Usar desde el celular» activado (y un PIN de 6
dígitos o más), al volver a abrirla escucha en la red local:

* desde la propia PC (127.0.0.1) no se pide PIN;
* desde otro equipo, ``POST /api/login`` con el PIN da una cookie de sesión (``HttpOnly``,
  ``SameSite=Strict``, 30 días); sin ella, la API responde 401;
* 5 PIN equivocados seguidos desde una dirección la bloquean 5 minutos;
* el PIN se guarda como hash PBKDF2-SHA256 con sal, nunca en texto;
* activar o cambiar esto solo se puede desde la propia PC, no desde el celular.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import secrets
import socket
import threading
import time

ITERACIONES = 200_000
DURACION_SESION = 30 * 24 * 3600
MAX_FALLOS, BLOQUEO_S = 5, 300


def hash_pin(pin: str, sal: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(sal), ITERACIONES).hex()


def es_local(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip.split("%")[0]).is_loopback
    except ValueError:
        return False


LOCALES = ("127.0.0.1", "localhost", "[::1]")


def host_local(host: str, port: int) -> bool:
    return (host or "").strip().lower() in tuple(f"{h}:{port}" for h in LOCALES)


def host_ok(host: str, port: int, lan: bool, extra: list[str] | tuple = ()) -> bool:
    """Contra «DNS rebinding»: solo localhost/127.0.0.1, (con la red local activada) una IP literal, o un nombre
    agregado a mano para un túnel seguro (Tailscale o Cloudflare Tunnel; ver docs/APP.md)."""
    host = (host or "").strip().lower()
    if host_local(host, port):
        return True
    nombre = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    if nombre and nombre in [h.strip().lower() for h in extra]:
        return True
    if not lan:
        return False
    name, _, p = host.rpartition(":")
    if p != str(port):
        return False
    try:
        ipaddress.ip_address(name.strip("[]"))
        return True
    except ValueError:
        return False


def ips_locales() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))            # no manda nada: solo elige la interfaz
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            ips.add(ip)
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


class Sesiones:
    def __init__(self):
        self.tokens: dict[str, float] = {}
        self.fallos: dict[str, list] = {}
        self._lock = threading.Lock()

    def valida(self, token: str | None) -> bool:
        with self._lock:
            exp = self.tokens.get(token or "")
            if exp and exp > time.time():
                return True
            self.tokens.pop(token or "", None)
            return False

    def login(self, app, ip: str, pin: str) -> tuple[bool, str]:
        cfg = app.cfg.data.get("red") or {}
        with self._lock:
            f = self.fallos.get(ip, [0, 0.0])
            if f[0] >= MAX_FALLOS and time.time() - f[1] < BLOQUEO_S:
                return False, f"demasiados intentos: esperá {int(BLOQUEO_S - (time.time() - f[1])) // 60 + 1} min"
            ok = bool(cfg.get("pin_hash")) and hmac.compare_digest(hash_pin(str(pin), cfg["sal"]), cfg["pin_hash"])
            if not ok:
                f = [f[0] + 1 if time.time() - f[1] < BLOQUEO_S else 1, time.time()]
                self.fallos[ip] = f
                return False, "PIN incorrecto"
            self.fallos.pop(ip, None)
            token = secrets.token_urlsafe(32)
            self.tokens[token] = time.time() + DURACION_SESION
            return True, token


def estado(app) -> dict:
    cfg = app.cfg.data.get("red") or {}
    activa = app.escucha != "127.0.0.1"
    port = getattr(app, "puerto", 8765)
    return {"lan": bool(cfg.get("lan")), "pin_configurado": bool(cfg.get("pin_hash")), "escuchando_en_red": activa,
            "urls": [f"http://{ip}:{port}" for ip in ips_locales()] if activa else [],
            "hosts": cfg.get("hosts") or [], "tailscale": tailscale()}


def configurar(app, p: dict, desde_local: bool = True) -> dict:
    if not desde_local:
        return {"ok": False, "mensaje": "esto solo se cambia desde la propia PC"}
    if p.get("confirmar") is not True:
        return {"ok": False, "mensaje": "falta confirmar"}
    lan = bool(p.get("lan"))
    red = dict(app.cfg.data.get("red") or {})
    pin = str(p.get("pin") or "")
    if pin:
        if not pin.isdigit() or len(pin) < 6:
            return {"ok": False, "mensaje": "el PIN tiene que tener al menos 6 números"}
        red["sal"] = secrets.token_hex(16)
        red["pin_hash"] = hash_pin(pin, red["sal"])
        app.sesiones.tokens.clear()               # PIN nuevo: se cierran las sesiones abiertas
    if lan and not red.get("pin_hash"):
        return {"ok": False, "mensaje": "para usarla desde el celular hace falta un PIN"}
    red["lan"] = lan
    app.cfg.set("red", red)
    try:
        app.guardian().audit.append("app.red_local", by=app.cfg.data.get("usuario") or "?", lan=lan,
                                    pin_cambiado=bool(pin))
    except (OSError, ValueError):
        pass
    cambia = (app.escucha != "127.0.0.1") != lan
    return {"ok": True, "mensaje": ("acceso desde la red local " + ("activado" if lan else "desactivado"))
            + ("; vale al volver a abrir la aplicación" if cambia else "")}


HOST_RE = __import__("re").compile(r"^([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")


def configurar_hosts(app, p: dict, desde_local: bool = True) -> dict:
    """Nombres permitidos para entrar por un túnel (p. ej. mipc.tailnet.ts.net). Exige PIN configurado."""
    if not desde_local:
        return {"ok": False, "mensaje": "esto solo se cambia desde la propia PC"}
    if p.get("confirmar") is not True:
        return {"ok": False, "mensaje": "falta confirmar"}
    red = dict(app.cfg.data.get("red") or {})
    hosts = p.get("hosts") or []
    if isinstance(hosts, str):
        hosts = [h for h in hosts.split() if h]
    hosts = [str(h).strip().lower().rstrip(".") for h in hosts]
    malos = [h for h in hosts if not HOST_RE.match(h)]
    if malos:
        return {"ok": False, "mensaje": "nombres inválidos: " + ", ".join(malos)}
    if hosts and not red.get("pin_hash"):
        return {"ok": False, "mensaje": "para entrar desde afuera hace falta un PIN"}
    red["hosts"] = hosts
    app.cfg.set("red", red)
    try:
        app.guardian().audit.append("app.red_hosts", by=app.cfg.data.get("usuario") or "?", hosts=hosts)
    except (OSError, ValueError):
        pass
    return {"ok": True, "mensaje": ("se puede entrar por: " + ", ".join(hosts)) if hosts else "sin túneles"}


_TS: dict = {"t": 0.0, "v": None}


def tailscale() -> dict | None:
    """Si Tailscale está instalado y conectado: nombre de la máquina y el comando para publicar la app con HTTPS
    solo dentro de la red privada (``tailscale serve``)."""
    import json
    import shutil
    import subprocess
    if time.time() - _TS["t"] < 60:
        return _TS["v"]
    v = None
    exe = shutil.which("tailscale")
    if exe:
        try:
            r = subprocess.run([exe, "status", "--json"], capture_output=True, text=True, timeout=10)
            st = json.loads(r.stdout or "{}")
            me = st.get("Self") or {}
            nombre = str(me.get("DNSName") or "").rstrip(".")
            v = {"instalado": True, "conectado": st.get("BackendState") == "Running", "nombre": nombre,
                 "ips": me.get("TailscaleIPs") or [], "comando": "tailscale serve --bg 8765",
                 "url": f"https://{nombre}" if nombre else None}
        except (OSError, subprocess.TimeoutExpired, ValueError):
            v = {"instalado": True, "conectado": False}
    _TS.update(t=time.time(), v=v)
    return v
