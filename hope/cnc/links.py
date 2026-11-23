"""Conexiones en vivo para la app (hope/cnc/app): simulador en tiempo real y grbl con hilos.

A diferencia de ``backends`` (envío por lotes desde la consola), una conexión en vivo atiende a la
vez el envío del programa, el estado cada 200 ms, el jog, MDI y los ajustes de avance/husillo
(overrides), como Mach3. Interfaz común: :class:`Link`.

* :class:`SimLink` — simulador: ejecuta los bloques del planificador en tiempo real (aceleración,
  esquinas, overrides, pausas M0, retención, jog, referenciado). Sirve para aprender la app y
  probar programas sin máquina.
* :class:`GrblLink` — grbl / grblHAL / FluidNC: un hilo lector despacha ``ok``/``error``/estado y
  otro pide ``?``; el programa se envía con conteo de caracteres.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque

from hope.cnc.backends.base import BackendError
from hope.cnc.backends.grbl import describe, open_port
from hope.cnc.gcode import Interpreter, Op
from hope.cnc.machine import Machine, as_controller
from hope.cnc.planner import distance_at, plan

# Overrides en tiempo real de grbl 1.1
OVERRIDES = {
    ("avance", "100"): 0x90, ("avance", "+10"): 0x91, ("avance", "-10"): 0x92, ("avance", "+1"): 0x93,
    ("avance", "-1"): 0x94, ("rapido", "100"): 0x95, ("rapido", "50"): 0x96, ("rapido", "25"): 0x97,
    ("husillo", "100"): 0x99, ("husillo", "+10"): 0x9A, ("husillo", "-10"): 0x9B, ("husillo", "+1"): 0x9C,
    ("husillo", "-1"): 0x9D,
}
JOG_CANCEL = 0x85
FLOOD_TOGGLE = 0xA0


def _job(total=0):
    return {"activo": False, "enviadas": 0, "confirmadas": 0, "total": total, "error": None, "terminado": False}


class Link:
    physical = True
    name = "base"

    def open(self) -> None: ...

    def close(self) -> None: ...

    def snapshot(self) -> dict:
        raise NotImplementedError

    def offsets(self) -> dict[str, dict[str, float]]:
        return {}

    def start(self, lines: list[str]) -> None:
        raise NotImplementedError

    def hold(self) -> None: ...

    def resume(self) -> None: ...

    def stop(self) -> None: ...

    def jog(self, moves: dict[str, float], feed: float) -> None:
        raise NotImplementedError

    def jog_cancel(self) -> None: ...

    def home(self) -> None:
        raise NotImplementedError

    def unlock(self) -> None: ...

    def mdi(self, line: str) -> None:
        raise NotImplementedError

    def override(self, kind: str, action: str) -> None: ...

    def coolant_toggle(self) -> None: ...

    single = False

    def set_single(self, on: bool) -> None:
        """Bloque a bloque (Single BLK de Mach3): cada Cycle Start ejecuta una línea y queda en pausa."""
        self.single = bool(on)

    def modal(self) -> str:
        return ""

    def tool(self) -> int | None:
        return None

    def clear_job(self) -> None:
        """Rebobinar (Mach3): olvida el avance del último programa si no hay uno en marcha."""
        job = getattr(self, "job", None)
        if job is not None and not job["activo"]:
            self.job = _job()


# ---------------------------------------------------------------------------
class SimLink(Link):
    physical = False
    name = "simulador"

    def __init__(self, machine: Machine, speed: float = 1.0, tick: float = 0.02, autostart: bool = True):
        self.m = as_controller(machine)               # recibe el programa ya traducido
        self.speed = speed
        self.tick = tick
        self.lock = threading.RLock()
        self.interp = Interpreter(self.m)             # estado modal y orígenes, como el del controlador
        self._pos = dict(self.interp.pos)             # posición física simulada (el intérprete va por delante)
        self.state = "Idle"
        self.homed = False
        self.ov = {"avance": 100, "rapido": 100, "husillo": 100}
        self.spindle = ("off", 0.0)
        self.coolant = False
        self.items: deque = deque()
        self.cur = None
        self.t_in = 0.0
        self.v = 0.0
        self.job = _job()
        self.jogging = False
        self.message = ""
        self.alarm = ""
        self._home_left = 0.0
        self._last_line = None
        self._single_go = False
        self._run = True
        self._thread = None
        if autostart:
            self.open()

    @property
    def pos(self) -> dict[str, float]:
        return self._pos

    def open(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="hope-cnc-sim")
            self._thread.start()

    def close(self) -> None:
        self._run = False

    def _loop(self):
        last = time.monotonic()
        while self._run:
            time.sleep(self.tick)
            now = time.monotonic()
            self.step((now - last) * self.speed)
            last = now

    # --- ejecución ----------------------------------------------------------------
    def _queue(self, ops: list[Op], line_base: int = 0):
        events: list = []
        pl = plan(ops, self.m, events=events)
        ev = deque(events)
        for i, b in enumerate(pl.blocks):
            while ev and ev[0][0] == i:
                self.items.append(("op", ev.popleft()[1]))
            self.items.append(("block", b))
        while ev:
            self.items.append(("op", ev.popleft()[1]))

    def step(self, dt: float) -> None:
        with self.lock:
            if self.state == "Home":
                self._home_left -= dt
                if self._home_left <= 0:
                    for a, ax in self.m.axes.items():
                        self.pos[a] = (ax.max if ax.max is not None else 0.0) if a == "Z" else \
                            (ax.min if ax.min is not None else 0.0)
                    self.homed, self.state = True, "Idle"
                return
            while dt > 1e-12 and self.state in ("Run", "Jog"):
                if self.cur is None:
                    if not self.items:
                        self._finish()
                        return
                    nxt = self.items[0][1]
                    if self.single and self.job["activo"] and self._last_line is not None \
                            and nxt.line != self._last_line:
                        if not self._single_go:
                            self.state, self.v = "Hold", 0.0     # bloque a bloque: espera Cycle Start
                            return
                        self._single_go = False
                    self._last_line = nxt.line
                    self.cur, self.t_in = self.items.popleft(), 0.0
                kind, it = self.cur
                if self.job["activo"]:
                    self.job["confirmadas"] = max(self.job["confirmadas"], it.line - 1)
                if kind == "block":
                    scale = (self.ov["rapido"] if it.kind == "rapid" else self.ov["avance"]) / 100.0
                    self.t_in += dt * scale
                    if self.t_in >= it.time:
                        dt = (self.t_in - it.time) / scale
                        for i, a in enumerate(self.m.axis_names):
                            self.pos[a] = it.end[i]
                        self.cur = None
                    else:
                        d = distance_at(it, self.t_in)
                        self.v = (distance_at(it, self.t_in + 1e-3) - d) / 1e-3 * scale
                        for i, a in enumerate(self.m.axis_names):
                            self.pos[a] = it.start[i] + it.unit[i] * d
                        dt = 0.0
                else:
                    dt = self._event(it, dt)

    def _event(self, op: Op, dt: float) -> float:
        if op.kind == "dwell":
            self.t_in += dt
            if self.t_in < float(op.value):
                return 0.0
            dt, self.cur = self.t_in - float(op.value), None
            return dt
        self.cur = None
        if op.kind == "spindle":
            self.spindle = op.value
        elif op.kind == "coolant":
            self.coolant = bool(op.value)
        elif op.kind in ("pause", "tool"):
            self.message = str(op.value) if op.kind == "pause" else f"cambiar a la herramienta T{op.value}"
            self.state, self.v = "Hold", 0.0
            return 0.0
        elif op.kind == "end":
            self.items.clear()
        return dt

    def _finish(self):
        self.cur, self.v = None, 0.0
        if self.job["activo"]:
            self.job.update(activo=False, terminado=True, confirmadas=self.job["total"])
        self.state = "Idle"
        self.jogging = False

    # --- interfaz -----------------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            return {"estado": self.state, "mpos": dict(self.pos), "wco": {a: self.interp._off(a) for a in self.pos},
                    "avance": round(self.v * 60.0, 1), "husillo": self.spindle[1] * self.ov["husillo"] / 100.0
                    if self.spindle[0] != "off" else 0.0, "husillo_dir": self.spindle[0],
                    "refrigerante": self.coolant, "ov": dict(self.ov), "referenciado": self.homed,
                    "trabajo": dict(self.job), "mensaje": self.message, "alarma": self.alarm,
                    "wcs": self.interp.wcs, "conexion": self.name,
                    "pausa_g4": bool(self.cur and self.cur[0] == "op" and self.cur[1].kind == "dwell"),
                    "avance_prog": round(self.cur[1].v_nominal * 60.0, 1)
                    if self.cur and self.cur[0] == "block" and self.cur[1].kind == "feed" else None,
                    "bloque_a_bloque": self.single}

    def set_machine(self, machine: Machine) -> None:
        """Otra máquina (perfil): nuevos límites, velocidades y orígenes; la posición se conserva."""
        with self.lock:
            self.m = as_controller(machine)
            pos = dict(self._pos)
            self.interp = Interpreter(self.m)
            self._pos = {a: pos.get(a, 0.0) for a in self.m.axis_names}
            self.interp.pos = dict(self._pos)

    def modal(self) -> str:
        from hope.cnc.gcode import modal_text
        with self.lock:
            return modal_text(self.interp)

    def tool(self) -> int | None:
        return self.interp.tool

    def offsets(self) -> dict[str, dict[str, float]]:
        with self.lock:
            out = {w: dict(v) for w, v in self.interp.offsets.items()}
            out["G92"] = dict(self.interp.g92)
            return out

    def _idle(self, what: str):
        if self.state != "Idle":
            raise BackendError(f"{what}: la máquina no está en reposo ({self.state})")

    def _run_text(self, text: str) -> list[Op]:
        it = self.interp
        it.ops, it.diags, it.ended = [], [], False
        it.pos = dict(self._pos)
        it.run(text)
        if any(d.level == "error" for d in it.diags):
            raise BackendError("; ".join(str(d) for d in it.diags if d.level == "error"))
        return it.ops

    def start(self, lines: list[str]) -> None:
        with self.lock:
            self._idle("ciclo")
            ops = self._run_text("\n".join(lines))
            self.items.clear()
            self._queue(ops)
            self.job = _job(len(lines))
            self.job.update(activo=True, enviadas=len(lines))
            self.message = ""
            self._last_line, self._single_go = None, False
            self.state = "Run"

    def hold(self) -> None:
        with self.lock:
            if self.state == "Run":
                self.state, self.v = "Hold", 0.0

    def resume(self) -> None:
        with self.lock:
            if self.state == "Hold":
                self.message = ""
                self._single_go = self.single
                self.state = "Run"

    def stop(self) -> None:
        with self.lock:
            self.items.clear()
            self.cur, self.v = None, 0.0
            if self.job["activo"]:
                self.job.update(activo=False, error="detenido por el operador")
            self.spindle = ("off", self.spindle[1])
            self.coolant = False
            self.state = "Idle"

    def jog(self, moves: dict[str, float], feed: float) -> None:
        with self.lock:
            if self.state == "Jog":
                self.jog_cancel()
            self._idle("jog")
            target = dict(self.pos)
            for a, d in moves.items():
                target[a] += d
            ax_bad = [a for a in moves if not self._inside(a, target[a])]
            if ax_bad:
                raise BackendError(f"el jog sale del recorrido en {', '.join(ax_bad)}")
            self.items.clear()
            self._queue([Op("feed", 0, dict(self.pos), target, feed)])
            self.state, self.jogging = "Jog", True

    def _inside(self, a: str, v: float) -> bool:
        ax = self.m.axes[a]
        return (ax.min is None or v >= ax.min - 1e-6) and (ax.max is None or v <= ax.max + 1e-6)

    def jog_cancel(self) -> None:
        with self.lock:
            if self.state == "Jog":
                self.items.clear()
                self.cur, self.v = None, 0.0
                self.state, self.jogging = "Idle", False

    def home(self) -> None:
        with self.lock:
            if self.state not in ("Idle", "Alarm"):
                raise BackendError("referenciar: la máquina no está en reposo")
            self.state, self._home_left, self.alarm = "Home", 1.5, ""

    def unlock(self) -> None:
        with self.lock:
            if self.state == "Alarm":
                self.state, self.alarm = "Idle", ""

    def mdi(self, line: str) -> None:
        with self.lock:
            self._idle("MDI")
            ops = self._run_text(line)
            self.items.clear()
            self._queue(ops)
            if self.items and all(k == "op" and it.kind != "dwell" for k, it in self.items):
                while self.items:                          # sin movimiento: se aplica en el acto
                    self._event(self.items.popleft()[1], 0.0)
            if self.items:
                self.state = "Run"

    def override(self, kind: str, action: str) -> None:
        with self.lock:
            if (kind, action) not in OVERRIDES:
                raise BackendError(f"ajuste desconocido: {kind} {action}")
            lo, hi = (10, 200) if kind != "rapido" else (25, 100)
            if action in ("100", "50", "25"):
                self.ov[kind] = int(action)
            else:
                self.ov[kind] = max(lo, min(hi, self.ov[kind] + int(action)))

    def coolant_toggle(self) -> None:
        with self.lock:
            self.coolant = not self.coolant


# ---------------------------------------------------------------------------
STATUS = re.compile(r"<([^|>]+)\|(.*)>")


class GrblLink(Link):
    name = "grbl"

    def __init__(self, port, machine: Machine, rx_buffer: int = 128, poll: float = 0.2, autostart: bool = True):
        self.port = port
        self.m = machine
        self.axes = machine.axis_names
        self.rx = int(rx_buffer)
        self.poll = poll
        self.wlock = threading.Lock()
        self.lock = threading.RLock()
        self.st = {"estado": "Desconectado", "mpos": {a: 0.0 for a in self.axes}, "avance": 0.0, "husillo": 0.0,
                   "husillo_dir": "off", "refrigerante": False, "ov": {"avance": 100, "rapido": 100, "husillo": 100},
                   "pines": ""}
        self.wco = {a: 0.0 for a in self.axes}
        self.homed = False
        self.banner = ""
        self.message = ""
        self.alarm = ""
        self.lines: list[str] = []
        self.pending: deque[int] = deque()
        self.job = _job()
        self.waiter: dict | None = None
        self._tokens = 0
        self.log: deque[str] = deque(maxlen=200)
        self._run = False
        if autostart:
            self.open()

    @classmethod
    def from_config(cls, cfg: dict, machine: Machine) -> GrblLink:
        if not cfg.get("puerto"):
            raise BackendError("controlador grbl sin «puerto» (p. ej. /dev/ttyUSB0, COM3 o tcp://192.168.1.50:23)")
        return cls(open_port(str(cfg["puerto"]), int(cfg.get("baudios", 115200))), machine,
                   rx_buffer=int(cfg.get("bufer", 128)))

    # --- hilos --------------------------------------------------------------------
    def open(self) -> None:
        if self._run:
            return
        self._run = True
        self._write(b"\r\n\r\n")
        threading.Thread(target=self._reader, daemon=True, name="hope-cnc-grbl-rx").start()
        threading.Thread(target=self._poller, daemon=True, name="hope-cnc-grbl-poll").start()

    def close(self) -> None:
        self._run = False
        try:
            self.port.close()
        except Exception as e:                             # noqa: BLE001
            self.log.append(f"al cerrar: {e}")

    def _write(self, data: bytes) -> None:
        with self.wlock:
            self.port.write(data)

    def _poller(self):
        while self._run:
            try:
                self._write(b"?")
            except Exception as e:                          # noqa: BLE001
                self.alarm = f"conexión perdida: {e}"
                self.st["estado"] = "Desconectado"
                return
            time.sleep(self.poll)

    def _reader(self):
        while self._run:
            try:
                raw = self.port.readline()
            except Exception as e:                          # noqa: BLE001
                self.alarm = f"conexión perdida: {e}"
                self.st["estado"] = "Desconectado"
                return
            if not raw:
                time.sleep(0.002)
                continue
            self.handle(raw.decode("ascii", errors="replace").strip())

    # --- protocolo -----------------------------------------------------------------
    def handle(self, line: str) -> None:
        if not line:
            return
        with self.lock:
            if line.startswith("<"):
                self._status(line)
                return
            self.log.append(line)
            if line == "ok" or line.startswith("error:"):
                if self.job["activo"] and self.pending:
                    self.pending.popleft()
                    self.job["confirmadas"] += 1
                    if line != "ok":
                        n = self.job["confirmadas"]
                        self.job.update(activo=False, error=f"línea {n} «{self.lines[n - 1]}»: {describe(line)}")
                        self._write(b"\x18")
                        return
                    if self.job["confirmadas"] == self.job["total"]:
                        self.job.update(activo=False, terminado=True)
                    self._fill()
                elif self.waiter is not None:
                    self.waiter["error"] = None if line == "ok" else describe(line)
                    self.waiter["event"].set()
            elif line.startswith("ALARM:"):
                self.alarm = describe(line)
                self.st["estado"] = "Alarm"
                if self.job["activo"]:
                    self.job.update(activo=False, error=self.alarm)
                if self.waiter is not None:
                    self.waiter["error"] = self.alarm
                    self.waiter["event"].set()
            elif line.lower().startswith(("grbl", "fluidnc")) or "grbl" in line.lower():
                self.banner = line
                self.pending.clear()
                if self.job["activo"]:
                    self.job.update(activo=False, error="el controlador se reinició")
            elif line.startswith("[MSG:"):
                self.message = line[5:].rstrip("]").strip()
            elif self.waiter is not None:
                self.waiter["lines"].append(line)

    def _status(self, line: str) -> None:
        m = STATUS.match(line)
        if not m:
            return
        self.st["estado"] = m.group(1).split(":")[0]
        fields = dict(f.split(":", 1) for f in m.group(2).split("|") if ":" in f)
        vec = lambda s: {a: float(v) for a, v in zip(self.axes, s.split(","))}
        if "WCO" in fields:
            self.wco = vec(fields["WCO"])
        if "MPos" in fields:
            self.st["mpos"] = vec(fields["MPos"])
        elif "WPos" in fields:
            w = vec(fields["WPos"])
            self.st["mpos"] = {a: w[a] + self.wco.get(a, 0.0) for a in w}
        fs = fields.get("FS") or fields.get("F")
        if fs:
            p = fs.split(",")
            self.st["avance"] = float(p[0])
            self.st["husillo"] = float(p[1]) if len(p) > 1 else 0.0
        if "Ov" in fields:
            f, r, s = (int(x) for x in fields["Ov"].split(","))
            self.st["ov"] = {"avance": f, "rapido": r, "husillo": s}
        if "Ov" in fields or "A" in fields:
            acc = fields.get("A", "")
            self.st["husillo_dir"] = "cw" if "S" in acc else "ccw" if "C" in acc else "off"
            self.st["refrigerante"] = "F" in acc or "M" in acc
        self.st["pines"] = fields.get("Pn", "")

    def _fill(self) -> None:
        while self.job["enviadas"] < self.job["total"]:
            if self.single:                        # bloque a bloque: una línea por Cycle Start
                if self.pending or self._tokens <= 0:
                    break
                self._tokens -= 1
            s = self.lines[self.job["enviadas"]]
            if sum(self.pending) + len(s) + 1 > self.rx:
                break
            self._write((s + "\n").encode("ascii"))
            self.pending.append(len(s) + 1)
            self.job["enviadas"] += 1

    def _command(self, line: str, timeout: float = 5.0) -> list[str]:
        with self.lock:
            if self.job["activo"]:
                raise BackendError("hay un programa en marcha")
            if self.waiter is not None:
                raise BackendError("el controlador está atendiendo otro comando")
            self.waiter = {"event": threading.Event(), "lines": [], "error": None}
            w = self.waiter
            self._write((line + "\n").encode("ascii"))
        try:
            if not w["event"].wait(timeout):
                raise BackendError(f"sin respuesta a «{line}»")
            if w["error"]:
                raise BackendError(w["error"])
            return w["lines"]
        finally:
            with self.lock:
                self.waiter = None

    # --- interfaz -----------------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            return {**{k: (dict(v) if isinstance(v, dict) else v) for k, v in self.st.items()},
                    "wco": dict(self.wco), "referenciado": self.homed, "trabajo": dict(self.job),
                    "mensaje": self.message, "alarma": self.alarm, "wcs": "G54", "conexion": self.banner or self.name,
                    "pausa_g4": False, "bloque_a_bloque": self.single}

    def offsets(self) -> dict[str, dict[str, float]]:
        out = {}
        for line in self._command("$#"):
            m = re.match(r"\[(G5[4-9]|G28|G30|G92):([^\]]+)\]", line)
            if m:
                out[m.group(1)] = {a: float(v) for a, v in zip(self.axes, m.group(2).split(","))}
        return out

    def start(self, lines: list[str]) -> None:
        clean = []
        for ln in lines:
            s = re.sub(r"\(.*?\)|;.*", "", ln).strip().upper().replace(" ", "")
            if s:
                if len(s) + 1 > self.rx:
                    raise BackendError(f"línea más larga que el búfer del controlador: {s[:40]}…")
                clean.append(s)
        with self.lock:
            if self.job["activo"]:
                raise BackendError("ya hay un programa en marcha")
            if self.st["estado"] not in ("Idle",):
                raise BackendError(f"la máquina no está en reposo ({self.st['estado']})")
            self.lines = clean
            self.pending.clear()
            self.job = _job(len(clean))
            self.job["activo"] = bool(clean)
            self.job["terminado"] = not clean
            self.message = ""
            self._tokens = 1
            self._fill()

    def hold(self) -> None:
        self._write(b"!")

    def set_machine(self, machine: Machine) -> None:
        self.m = machine

    def set_single(self, on: bool) -> None:
        with self.lock:
            self.single = bool(on)
            if not self.single and self.job["activo"]:
                self._fill()                       # al apagar bloque a bloque sigue enviando de corrido

    def resume(self) -> None:
        self.message = ""
        self._write(b"~")
        with self.lock:
            if self.single and self.job["activo"]:
                self._tokens += 1
                self._fill()

    def stop(self) -> None:
        self._write(b"!")
        time.sleep(0.25)                                   # frenar antes de reiniciar: no se pierde la posición
        with self.lock:
            if self.job["activo"]:
                self.job.update(activo=False, error="detenido por el operador")
            self.pending.clear()
        self._write(b"\x18")

    def jog(self, moves: dict[str, float], feed: float) -> None:
        axes = " ".join(f"{a}{v:.4f}" for a, v in moves.items() if abs(v) > 1e-9)
        if axes:
            self._command(f"$J=G91 G21 {axes} F{feed:.0f}")

    def jog_cancel(self) -> None:
        self._write(bytes([JOG_CANCEL]))

    def home(self) -> None:
        self._command("$H", timeout=180.0)
        self.homed = True

    def unlock(self) -> None:
        self._command("$X")
        self.alarm = ""

    def mdi(self, line: str) -> None:
        s = re.sub(r"\(.*?\)|;.*", "", line).strip()
        if s:
            self._command(s, timeout=30.0)

    def override(self, kind: str, action: str) -> None:
        b = OVERRIDES.get((kind, action))
        if b is None:
            raise BackendError(f"ajuste desconocido: {kind} {action}")
        self._write(bytes([b]))

    def coolant_toggle(self) -> None:
        self._write(bytes([FLOOD_TOGGLE]))


def make_link(machine: Machine, cfg: dict | None = None) -> Link:
    cfg = cfg if cfg is not None else machine.controller
    kind = str(cfg.get("tipo", "sim")).lower()
    if kind == "sim":
        return SimLink(machine, speed=float(cfg.get("velocidad", 1.0)))
    if kind in ("grbl", "grblhal", "fluidnc"):
        return GrblLink.from_config(cfg, machine)
    raise BackendError(f"la app todavía no maneja el controlador «{kind}» en vivo (sim o grbl); "
                       "para LinuxCNC usar «python -m hope.cnc enviar»")

