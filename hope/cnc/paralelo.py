"""Parallel port (LPT) connection: TISMA drives the breakout board directly, like Mach3 did.

What it is
----------
``LptLink`` is a live connection (``links.Link``) for machines wired to the PC's **parallel
port** through a Mach3-style breakout board, on Windows XP to 11 — 32 **and** 64 bits (Mach3's
parallel driver only works on 32-bit Windows). TISMA interprets and plans the program as always
(acceleration, corners, the same check as every controller); the real-time part — the step and
direction pulses, the switches, the probe — runs in a small separate program, ``tisma_lpt``
(``hope/cnc/lpt/tisma_lpt.c``), so Python's pauses never reach the motors.

    TISMA (Python)                          tisma_lpt.exe (C, TIME_CRITICAL thread)
    interpreter → planner → blocks  ──M──▶  live trapezoid + DDA → step/dir pins
    snapshot()  ◀──S, PRB, HOMED, ALARM──   inputs: limits, home, probe, e-stop

How it fits
-----------
``links.make_link`` creates it for ``controlador.tipo: paralelo``. It inherits from ``SimLink``
everything that happens before the machine moves (the interpreter that keeps the modal state and
the work offsets, the planner that turns the program into blocks, MDI, offsets, modal text,
tool) and replaces the «moving» part: instead of advancing a simulated clock it sends the blocks
to the engine and reads the real position back. The session, the screens, probing and the
guardian approval see one more ``Link`` (``physical = True``).

Configuration (``controlador`` of the machine profile; Spanish keys, like Mach3's «Ports & Pins»)
--------------------------------------------------------------------------------------------------
    tipo: paralelo
    puerto: "0x378"          LPT base address. 0x378 = LPT1 on the motherboard; PCI/PCIe cards
                             use another one (see Device Manager → Resources), e.g. 0xD010.
    frecuencia: 25000        kernel frequency (Hz), like Mach3's «Kernel Speed». Max step rate per
                             axis = frecuencia / 2. 25000 for most PCs; 35000–45000 only if the
                             latency test allows it. Too high = late ticks, rough motion.
    pulso_us: 5              step pulse width (µs). 2–5 for modern drivers, 5–10 for old
                             optocoupled boards. Must be well below 1e6/frecuencia.
    pines: [                 one row per signal: {senal, pin, bajo}; «bajo» = 1 if active LOW
      {senal: "X paso", pin: 2, bajo: 0}, {senal: "X dir", pin: 3, bajo: 0}, ...
      {senal: "Parada de emergencia", pin: 15, bajo: 1}, {senal: "Sonda", pin: 13, bajo: 0}, ...]
                             signals: «<eje> paso», «<eje> dir» (bajo = invert direction),
                             «Habilitar», «Husillo», «Husillo antihorario», «Refrigerante»,
                             «Niebla», «PWM husillo», «Bomba de carga», «Parada de emergencia»,
                             «Sonda», «Límite <eje>», «Referencia <eje>». Output pins: 1–9, 14, 16,
                             17; input pins: 10, 11, 12, 13, 15. Without «pines»: X 2/3, Y 4/5,
                             Z 6/7, A 8/9 (Mach3's default) and no inputs or outputs.
    referencia:              homing per axis (only axes with a «Referencia» input home):
      {X: {direccion: -1, velocidad: 1200, lenta: 120, retiro: 3}}
                             direccion −1 = towards the minimum (X, Y), +1 = towards the maximum
                             (Z by default); velocidad/lenta in mm/min (seek / latch), retiro mm.
    pwm_hz: 100              spindle PWM frequency; bomba_hz: 12500 charge pump frequency.
    permitir_un_nucleo: false   see «Single-core PCs» below.
    motor: path              another tisma_lpt executable (tests: the simulator build).

Units: mm and mm/min outside, steps inside (``steps_per_mm`` of each axis, rotary: steps/°).

Safety
------
* The emergency stop input stops the engine at once and switches spindle and coolant off; the
  machine stays in alarm until the button is released and RESET is pressed. It is a help, never
  a substitute for the physical emergency stop that cuts the drivers' power.
* Limit inputs stop everything at once (alarm); homing ignores only the switch it is looking for.
* If TISMA closes or crashes, the engine sees its input closed and stops the machine.
* Axis speeds are capped so no axis asks for more than frecuencia/2 steps per second.

Single-core PCs
---------------
The engine busy-waits on one CPU core (Windows has no real-time timer finer than 1 ms). With a
single core that would freeze the screen and the rest of Windows during motion, so it is refused
unless ``permitir_un_nucleo: true`` — on such PCs LinuxCNC (real-time) or an external controller
(grblHAL) is the right choice.

Example
-------
::

    link = make_link(machine, {"tipo": "paralelo", "puerto": "0x378"})
    link.mdi("G0 X10")
"""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from hope.cnc.backends.base import BackendError
from hope.cnc.gcode import Op
from hope.cnc.links import SimLink, _job
from hope.cnc.machine import Machine
from hope.cnc.planner import plan

# Engine axis index of each axis name (fixed, whatever the machine's axis order).
ENGINE_AXES = ("X", "Y", "Z", "A", "B", "C")
# Engine state letter → state shown on screen (same names as grbl, see links).
STATES = {"I": "Idle", "R": "Run", "H": "Hold", "A": "Alarm"}
# How many items may be queued in the engine at once (its queue holds 512): enough for ~0.5 s of
# very short CAM segments, and a Stop discards little.
WINDOW = 400
# Output signal (Spanish, as in the pin table) → engine output name.
OUTPUTS = {"habilitar": "ENABLE", "husillo": "SPINDLE_CW", "husillo antihorario": "SPINDLE_CCW",
           "refrigerante": "FLOOD", "niebla": "MIST", "pwm husillo": "PWM", "bomba de carga": "CHARGEPUMP"}
# Mach3's default step/dir pins (data pins 2..9).
DEFAULT_PINS = [{"senal": f"{a} {k}", "pin": p, "bajo": 0}
                for a, base in (("X", 2), ("Y", 4), ("Z", 6), ("A", 8))
                for k, p in (("paso", base), ("dir", base + 1))]


# ==================================================================================================
# ======================================= ENGINE AND PIN MAP =======================================
# ==================================================================================================

def engine_path(cfg: dict) -> Path:
    """Executable of the pulse engine for this PC.

    ``cfg["motor"]`` if given; on Windows ``tisma_lpt64.exe`` on a 64-bit Windows (even with a
    32-bit Python: ``PROCESSOR_ARCHITEW6432`` tells WOW64 apart) or ``tisma_lpt32.exe`` otherwise,
    looked for next to the app (packages) and in ``hope/cnc/lpt`` (built from source).

    Raises:
        BackendError: not Windows (on Linux the parallel port goes through LinuxCNC), or not built.
    """
    if cfg.get("motor"):
        return Path(str(cfg["motor"]))
    if not sys.platform.startswith("win"):
        raise BackendError("el puerto paralelo directo es para Windows; en Linux se usa LinuxCNC (controlador «linuxcnc»)")
    bits = "64" if "64" in (os.environ.get("PROCESSOR_ARCHITEW6432", "") + platform.machine()) else "32"
    name = f"tisma_lpt{bits}.exe"
    here = Path(__file__).resolve().parent
    for folder in (Path(sys.executable).resolve().parent.parent, here / "lpt", here.parent.parent):
        if (folder / name).exists():
            return folder / name
    raise BackendError(f"falta {name} (el motor del puerto paralelo): compilarlo con hope/cnc/lpt/compilar.sh")


def engine_config(cfg: dict, machine: Machine) -> list[str]:
    """The ``CFG`` lines for the engine from the profile's ``controlador`` section.

    Raises:
        BackendError: unknown signal, pin of the wrong kind or used twice (Spanish message).
    """
    try:
        port = int(str(cfg.get("puerto", "0x378")), 0)
    except ValueError:
        raise BackendError(f"puerto paralelo: «{cfg.get('puerto')}» no es una dirección (ej. 0x378)") from None
    freq = float(cfg.get("frecuencia", 25000))
    pulse = float(cfg.get("pulso_us", 5))
    if not 5000 <= freq <= 100000:
        raise BackendError("puerto paralelo: la frecuencia va de 5000 a 100000 Hz (25000 es lo normal)")
    if not 0.5 <= pulse <= 0.5e6 / freq:
        raise BackendError(f"puerto paralelo: el pulso va de 0,5 a {0.5e6 / freq:.0f} µs con {freq:.0f} Hz")
    lines = [f"CFG FREQ {freq:g}", f"CFG PULSE {pulse:g}", f"CFG PORT {port}",
             f"CFG PWMFREQ {float(cfg.get('pwm_hz', 100)):g}", f"CFG CPFREQ {float(cfg.get('bomba_hz', 12500)):g}"]
    axes = {}
    used = {}
    for row in cfg.get("pines") or DEFAULT_PINS:
        signal = " ".join(str(row.get("senal", "")).split())
        pin = int(row.get("pin") or 0)
        low = 1 if row.get("bajo") in (1, "1", True, "si", "sí") else 0
        if not pin:
            continue
        key = signal.lower()
        if pin in used and not ("límite" in key or "referencia" in key or "limite" in key):
            # A limit and a home switch often share one input (Mach3 «home/limit» on one pin).
            raise BackendError(f"puerto paralelo: el pin {pin} está en «{used[pin]}» y en «{signal}»")
        used.setdefault(pin, signal)
        parts = key.split()
        if len(parts) == 2 and parts[0].upper() in ENGINE_AXES and parts[1] in ("paso", "dir"):
            axes.setdefault(parts[0].upper(), {})[parts[1]] = (pin, low)
        elif key in OUTPUTS:
            lines.append(f"CFG OUT {OUTPUTS[key]} {pin} {low}")
        elif key == "parada de emergencia":
            lines.append(f"CFG IN ESTOP {pin} {low}")
        elif key == "sonda":
            lines.append(f"CFG IN PROBE {pin} {low}")
        elif len(parts) == 2 and parts[0] in ("límite", "limite", "referencia") and parts[1].upper() in ENGINE_AXES:
            kind = "HOME" if parts[0] == "referencia" else "LIMIT"
            lines.append(f"CFG IN {kind}{ENGINE_AXES.index(parts[1].upper())} {pin} {low}")
        else:
            raise BackendError(f"puerto paralelo: señal desconocida «{signal}»")
    for a in machine.axis_names:
        if a not in axes or "paso" not in axes[a] or "dir" not in axes[a]:
            raise BackendError(f"puerto paralelo: faltan los pines de paso y dirección del eje {a}")
        (sp, sl), (dp, di) = axes[a]["paso"], axes[a]["dir"]
        lines.append(f"CFG AXIS {ENGINE_AXES.index(a)} STEP {sp} DIR {dp} STEPLOW {sl} DIRINV {di}")
    return lines


def cap_rates(machine: Machine, freq: float) -> tuple:
    """Machine copy whose axis speeds fit in the engine's step rate (frecuencia / 2 per axis).

    Returns ``(machine, notes)``; notes are Spanish lines for the operator («X limitado a …»).
    Why half the frequency: one step per tick is the hard maximum; at half there is always one
    idle tick between pulses, so the pulse shape stays clean even if a tick is late.
    """
    import dataclasses
    axes, notes = {}, []
    for a in machine.axis_names:                          # axis order, not dict order (Python 3.4)
        ax = machine.axes[a]
        top = freq / 2.0 / ax.steps_per_mm * 60.0          # mm/min the engine can step
        if ax.max_rate > top:
            notes.append(f"{a} limitado a {top:.0f} mm/min por la frecuencia del puerto ({freq:.0f} Hz, "
                         f"{ax.steps_per_mm:g} pasos)")
            ax = dataclasses.replace(ax, max_rate=top)
        axes[a] = ax
    return dataclasses.replace(machine, axes=axes), notes


# ==================================================================================================
# =========================================== THE LINK =============================================
# ==================================================================================================

class LptLink(SimLink):
    """Live connection through the parallel port (see the module header).

    Parameters:
        machine: machine profile (axes with ``steps_per_mm``, limits, speeds).
        cfg:     the ``controlador`` section (see the header).
        autostart: start the engine now (the tests may start it later with ``open``).

    Raises:
        BackendError: bad configuration, engine missing, single-core PC, engine not answering.
    """

    physical = True
    name = "puerto paralelo"

    def __init__(self, machine: Machine, cfg: dict, autostart: bool = True):
        self.cfg = dict(cfg)
        self.freq = float(self.cfg.get("frecuencia", 25000))
        machine, self.notes = cap_rates(machine, self.freq)
        super().__init__(machine, autostart=False)      # no simulated clock: the engine moves
        self.cfg_lines = engine_config(self.cfg, self.m)
        if (os.cpu_count() or 1) < 2 and not self.cfg.get("permitir_un_nucleo"):
            raise BackendError("PC de un solo núcleo: el puerto paralelo directo congelaría Windows mientras se mueve; "
                               "usar LinuxCNC o un controlador externo (o «permitir_un_nucleo» en el perfil)")
        self.proc = None
        self.eng = {"state": "I", "done": 0, "steps": [0] * 6, "inputs": 0, "v": 0.0, "overruns": 0}
        self._cmd = [0] * 6                  # commanded absolute steps (where the queued moves end)
        self._out: deque = deque()           # (id, engine line) waiting to be sent
        self._next_id = 0                    # last id handed out
        self._sent_id = 0                    # last id written to the engine
        self._line_of: dict = {}             # id → program line (job progress) while a job runs
        self._pause_msg: dict = {}           # id of a pause item → message for the operator
        self._floor = 0                      # ids up to here were cancelled (Stop): never «done»
        self._reports = 0                    # status reports received (to wait for a fresh one)
        self._homing = False
        self._homed_ids: set = set()
        self._latency = None
        self._ready = threading.Event()
        self._wake = threading.Event()
        self.message = "; ".join(self.notes)
        if autostart:
            self.open()

    # --- engine process -----------------------------------------------------------------------------
    def open(self) -> None:
        """Starts the engine, sends the configuration and waits for READY (5 s).
        Raises BackendError if it does not start or reports an alarm right away."""
        if self.proc is not None:
            return
        cmd = [str(engine_path(self.cfg))]
        extra = list(self.cfg.get("motor_extra") or [])       # tests: «CFG INTERACTIVE 50», SIMIN rules…
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, bufsize=0)
        except OSError as e:
            raise BackendError(f"no se pudo iniciar el motor del puerto paralelo: {e}") from None
        threading.Thread(target=self._reader, daemon=True, name="hope-cnc-lpt-lector").start()
        self._write(self.cfg_lines + extra + ["START"])
        if not self._ready.wait(5.0):
            raise BackendError(f"el motor del puerto paralelo no respondió: {self.alarm or 'sin respuesta'}")
        if self.alarm:
            raise BackendError(self.alarm)
        threading.Thread(target=self._sender, daemon=True, name="hope-cnc-lpt-envio").start()

    def close(self) -> None:
        """Stops the engine (it stops the machine and releases the port)."""
        p, self.proc = self.proc, None
        if p is not None:
            try:
                p.stdin.write(b"QUIT\n")
                p.stdin.close()
                p.wait(2.0)
            except (OSError, subprocess.TimeoutExpired):
                p.kill()
        self._wake.set()

    def _write(self, lines: list) -> None:
        """Writes lines to the engine's input (raises BackendError if it died)."""
        p = self.proc
        if p is None:
            raise BackendError("el motor del puerto paralelo no está en marcha")
        try:
            p.stdin.write(("\n".join(lines) + "\n").encode("ascii"))
            p.stdin.flush()
        except OSError:
            raise BackendError("se cortó la comunicación con el motor del puerto paralelo") from None

    def _reader(self) -> None:
        """Reads the engine's reports (thread): status, probe, homing, alarms, latency."""
        p = self.proc
        for raw in iter(p.stdout.readline, b""):
            parts = raw.decode("ascii", "replace").split()
            if not parts:
                continue
            with self.lock:
                kind = parts[0]
                if kind == "S" and len(parts) >= 12:
                    e = self.eng
                    e["state"], e["done"] = parts[1], int(parts[2])
                    e["steps"] = [int(x) for x in parts[3:9]]
                    e["inputs"], e["v"], e["overruns"] = int(parts[9]), float(parts[10]), int(parts[11])
                    self._reports += 1
                    self._sync_state()
                    self._wake.set()
                elif kind == "READY":
                    self._ready.set()
                elif kind == "PRB" and len(parts) >= 8:
                    n = (self._probe["n"] if self._probe else 0) + 1
                    self._probe = {"n": n, "pos": self._mm([int(x) for x in parts[2:8]]), "ok": parts[1] == "1"}
                elif kind == "HOMED":
                    self._homed_ids.add(int(parts[1]))
                elif kind == "ALARM":
                    self.alarm = " ".join(parts[1:])
                    self.eng["state"] = "A"
                    self._out.clear()
                    if self.job["activo"]:
                        self.job.update(activo=False, error=self.alarm)
                    self._ready.set()                     # an alarm before READY also unblocks open()
                elif kind == "LAT" and len(parts) >= 5:
                    self._latency = {"ticks": int(parts[1]), "max_us": float(parts[2]),
                                     "p99_us": float(parts[3]), "p999_us": float(parts[4])}
                elif kind == "ERR":
                    self.message = " ".join(parts[1:])
        with self.lock:                                    # the engine ended: nothing moves anymore
            if self.proc is not None:
                self.alarm = self.alarm or "el motor del puerto paralelo se cerró"
                self.eng["state"] = "A"
                self.state = "Alarm"
            self._ready.set()

    def _sender(self) -> None:
        """Feeds queued items to the engine, keeping at most WINDOW not yet finished (thread)."""
        while self.proc is not None:
            batch = []
            with self.lock:
                while self._out and self._out[0][0] - self.eng["done"] <= WINDOW:
                    i, line = self._out.popleft()
                    batch.append(line)
                    self._sent_id = i
            if batch:
                try:
                    self._write(batch)
                except BackendError as e:
                    with self.lock:
                        self.alarm = str(e)
                    return
            self._wake.wait(0.05)
            self._wake.clear()

    # --- conversions ---------------------------------------------------------------------------------
    def _spm(self, a: str) -> float:
        return self.m.axes[a].steps_per_mm

    def _mm(self, steps: list) -> dict:
        """Engine step counters → machine position (mm / °) of the machine's axes."""
        return {a: steps[ENGINE_AXES.index(a)] / self._spm(a) for a in self.m.axis_names}

    def _sync_state(self) -> None:
        """Updates ``state``/``_pos``/job progress from the last report (lock held)."""
        e = self.eng
        self._pos = self._mm(e["steps"])
        self.v = e["v"]
        busy = self._next_id > max(e["done"], self._floor)  # items handed out and not finished yet
        st = STATES.get(e["state"], "Idle")
        if st == "Idle" and busy and e["state"] != "A":
            st = "Run"
        if self._homing:
            st = "Home"
        elif self.jogging and st == "Run":
            st = "Jog"
        elif self.jogging and st == "Idle":
            self.jogging = False
        self.state = st
        if st == "Hold":
            nxt = e["done"] + 1
            if nxt in self._pause_msg:
                self.message = self._pause_msg[nxt]
        if self.job["activo"]:
            line = self._line_of.get(e["done"])
            if line:
                self.job["confirmadas"] = max(self.job["confirmadas"], line - 1)
            if e["done"] >= self._next_id and e["state"] == "I" and not self._out:
                self.job.update(activo=False, terminado=True, confirmadas=self.job["total"])

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _queue(self, ops: list, line_base: int = 0):
        """Plans ``ops`` and hands the blocks and events to the engine (lock held).

        Probing moves are planned to their full travel (the interpreter assumed a contact) and
        flagged so the engine stops at the trigger. Steps are computed from absolute targets
        (round(position × steps/mm)), so rounding never accumulates along a long program.
        """
        if not self._out and self._next_id <= max(self.eng["done"], self._floor):
            self._cmd = list(self.eng["steps"])           # at rest: start from where the engine is
        for op in ops:
            if op.kind == "probe" and isinstance(op.value, dict):
                op.end = dict(op.value)
        events: list = []
        pl = plan(ops, self.m, events=events)
        ev = deque(events)
        for i, b in enumerate(pl.blocks):
            while ev and ev[0][0] == i:
                self._event_out(ev.popleft()[1])
            self._block_out(b, ops)
        while ev:
            self._event_out(ev.popleft()[1])
        self._wake.set()

    def _block_out(self, b, ops: list, flags: str = "") -> None:
        """One planned block → «M id L vc ve a rapid s0..s5 [flags]»."""
        steps = [0] * 6
        for k, a in enumerate(self.m.axis_names):
            j = ENGINE_AXES.index(a)
            target = round(b.end[k] * self._spm(a))
            steps[j] = target - self._cmd[j]
            self._cmd[j] = target
        op = ops[b.op] if 0 <= b.op < len(ops) else None
        if op is not None and op.kind == "probe":
            flags += " P3" if op.source == "G38.3" else " P2"
        ve = 0.0 if b.stop_after or (op is not None and op.kind == "probe") else b.v_exit
        i = self._new_id()
        self._line_of[i] = b.line
        self._out.append((i, f"M {i} {b.length:.6f} {b.v_nominal:.6f} {ve:.6f} {b.accel:.6f} "
                             f"{1 if b.kind == 'rapid' else 0} " + " ".join(str(s) for s in steps) + flags))

    def _event_out(self, op: Op) -> None:
        """A non-motion operation → engine items (outputs, dwell, pause) in program order."""
        out = []
        if op.kind == "spindle":
            direction, rpm = op.value
            duty = 0 if direction == "off" else round(1000 * min(1.0, rpm / (self.m.spindle_max or rpm or 1.0)))
            self.spindle = op.value
            out = [("SPINDLE_CW", 1 if direction == "cw" else 0), ("SPINDLE_CCW", 1 if direction == "ccw" else 0),
                   ("PWM", duty * self.ov["husillo"] // 100)]
        elif op.kind == "coolant":
            names = op.value or set()
            self.coolant = bool(names)
            out = [("FLOOD", 1 if "inundación" in names else 0), ("MIST", 1 if "niebla" in names else 0)]
        elif op.kind == "dwell":
            i = self._new_id()
            self._line_of[i] = op.line
            self._out.append((i, f"W {i} {float(op.value):.4f}"))
        elif op.kind in ("pause", "tool"):
            i = self._new_id()
            self._line_of[i] = op.line
            self._pause_msg[i] = str(op.value) if op.kind == "pause" else f"cambiar a la herramienta T{op.value}"
            self._out.append((i, f"P {i}"))
        for name, value in out:
            i = self._new_id()
            self._line_of[i] = op.line
            self._out.append((i, f"O {i} {name} {value}"))

    # --- interface (see links.Link) ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        """Nothing: the engine moves in real time (SimLink's clock is not used)."""

    def snapshot(self) -> dict:
        """See :meth:`Link.snapshot`, from the engine's last report. Adds ``tiempo_real``: late
        ticks (a growing number means this PC cannot keep up with the configured frequency)."""
        with self.lock:
            snap = super().snapshot()
            snap["conexion"] = f"{self.name} {self.cfg.get('puerto', '0x378')}"
            snap["referenciado"] = self.homed
            snap["avance"] = round(self.v * 60.0, 1)
            snap["avance_prog"] = None
            snap["pausa_g4"] = False
            snap["tiempo_real"] = {"retrasos": self.eng["overruns"], "frecuencia": self.freq}
            return snap

    def start(self, lines: list[str]) -> None:
        """See :meth:`Link.start`: interprets, plans and streams the program to the engine."""
        with self.lock:
            self._idle("ciclo")
            ops = self._run_text("\n".join(lines))
            self._line_of, self._pause_msg = {}, {}
            self.message = ""
            self._queue(ops)
            self.job = _job(len(lines))
            self.job.update(activo=True, enviadas=len(lines))
            self.state = "Run"

    def mdi(self, line: str) -> None:
        """See :meth:`Link.mdi`: one line, planned and sent like a short program."""
        with self.lock:
            self._idle("MDI")
            ops = self._run_text(line)
            self._queue(ops)
            if self._next_id > self.eng["done"]:
                self.state = "Run"

    def hold(self) -> None:
        """Feed hold: the engine brakes inside the current block (position kept)."""
        self._write(["HOLD"])

    def resume(self) -> None:
        """Cycle Start after a hold or an M0/M1/M6 pause."""
        with self.lock:
            self.message = ""
        self._write(["RESUME"])

    def stop(self) -> None:
        """Stop: brake (no lost position), discard everything queued, spindle and coolant off."""
        with self.lock:
            self._out.clear()
            if self.job["activo"]:
                self.job.update(activo=False, error="detenido por el operador")
            self.message = ""
        self._write(["CANCEL", "SET SPINDLE_CW 0", "SET SPINDLE_CCW 0", "SET PWM 0", "SET FLOOD 0", "SET MIST 0"])
        self.spindle, self.coolant = ("off", 0.0), False
        self._wait_stopped(30.0)

    def _wait_stopped(self, timeout: float) -> None:
        """After a CANCEL: waits for two fresh status reports showing the engine stopped, then
        marks every item handed out as finished (``_floor``): the cancelled ones never will be."""
        t0 = time.monotonic()
        with self.lock:
            seen = self._reports
        while time.monotonic() - t0 < timeout:
            with self.lock:
                e = self.eng
                if self._reports >= seen + 2 and e["state"] != "R" and e["v"] == 0:
                    self._floor = self._next_id
                    self._sync_state()
                    return
            time.sleep(0.01)

    def jog(self, moves: dict, feed: float) -> None:
        """See :meth:`Link.jog`: one move, checked against the travel (like grbl's $J=)."""
        with self.lock:
            if self.state == "Jog":
                self.jog_cancel()
            self._idle("jog")
            target = dict(self._pos)
            for a, d in moves.items():
                target[a] += d
            bad = [a for a in moves if not self._inside(a, target[a])]
            if bad:
                raise BackendError(f"el jog sale del recorrido en {', '.join(bad)}")
            self._queue([Op("feed", 0, dict(self._pos), target, feed)])
            self.state, self.jogging = "Jog", True

    def jog_cancel(self) -> None:
        """Releasing the jog key: brake and drop the rest of the jog."""
        with self.lock:
            if not self.jogging:
                return
            self._out.clear()
        self._write(["CANCEL"])
        self._wait_stopped(5.0)
        with self.lock:
            self.jogging = False
            self._sync_state()

    def unlock(self) -> None:
        """Clears an alarm (the engine refuses while the emergency stop is pressed)."""
        self._write(["RESET"])
        with self.lock:
            self.alarm = ""
            self._out.clear()
            self._floor = self._next_id         # an alarm discarded the engine's queue

    def override(self, kind: str, action: str) -> None:
        """See :meth:`Link.override`: feed / rapid to the engine; spindle rescales the PWM."""
        super().override(kind, action)
        cmd = {"avance": f"OVF {self.ov['avance']}", "rapido": f"OVR {self.ov['rapido']}"}.get(kind)
        if cmd:
            self._write([cmd])
        elif kind == "husillo" and self.spindle[0] != "off":
            duty = round(1000 * min(1.0, self.spindle[1] / (self.m.spindle_max or 1.0)))
            self._write([f"SET PWM {duty * self.ov['husillo'] // 100}"])

    def coolant_toggle(self) -> None:
        """Flood on/off at once, even with a program running."""
        with self.lock:
            self.coolant = not self.coolant
            on = 1 if self.coolant else 0
        self._write([f"SET FLOOD {on}"])

    # --- homing --------------------------------------------------------------------------------------
    def home(self) -> None:
        """Ref All Home: Z first (out of the way), then X, Y and the rotary axes.

        Per axis with a «Referencia» input: fast seek towards the switch, back off, slow seek
        (precise), back off, and the counters take the travel end on that side (minus the back
        off). Axes without a switch keep their position as the reference (like Mach3 without
        home switches). Blocks until done (the session runs it in a thread).

        Raises:
            BackendError: not at rest, a switch not found, or an alarm while homing.
        """
        with self.lock:
            if self.state not in ("Idle", "Alarm"):
                raise BackendError(f"referenciar: la máquina no está en reposo ({self.state})")
            self._homing = True
            self.alarm = ""
        try:
            self._write(["RESET"])
            order = [a for a in ("Z", "X", "Y", "A", "B", "C") if a in self.m.axes]
            for a in order:
                if f"IN HOME{ENGINE_AXES.index(a)} " in " ".join(self.cfg_lines) + " ":
                    self._home_axis(a)
            with self.lock:
                self.homed = True
        finally:
            with self.lock:
                self._homing = False
                self._sync_state()

    def _home_axis(self, a: str) -> None:
        """Homing sequence of one axis (see :meth:`home`)."""
        ax = self.m.axes[a]
        ref = dict((self.cfg.get("referencia") or {}).get(a) or {})
        direction = int(ref.get("direccion", 1 if a == "Z" else -1))
        fast = float(ref.get("velocidad", min(ax.max_rate / 3.0, 1500.0))) / 60.0
        slow = float(ref.get("lenta", fast * 60.0 / 10.0)) / 60.0
        back = float(ref.get("retiro", 3.0))
        end = (ax.max if direction > 0 else ax.min)
        span = (ax.max - ax.min) if ax.max is not None and ax.min is not None else 1000.0
        j = ENGINE_AXES.index(a)

        def move(dist, speed, flags):
            """One homing move of ``dist`` mm along the axis; waits for it; returns True if the
            home switch stopped it."""
            with self.lock:
                i = self._new_id()
                steps = [0] * 6
                steps[j] = round(dist * self._spm(a))
                self._out.append((i, f"M {i} {abs(dist):.6f} {speed:.6f} 0 {ax.accel:.6f} 0 "
                                     + " ".join(str(s) for s in steps) + flags))
                self._wake.set()
            t0 = time.monotonic()
            while time.monotonic() - t0 < abs(dist) / speed + 30:
                with self.lock:
                    if self.eng["state"] == "A":
                        raise BackendError(f"referencia del eje {a}: {self.alarm}")
                    if self.eng["done"] >= i and self.eng["state"] == "I":
                        return i in self._homed_ids
                time.sleep(0.01)
            raise BackendError(f"referencia del eje {a}: no terminó a tiempo")

        if not move(direction * span * 1.5, fast, f" H{j}"):
            raise BackendError(f"referencia del eje {a}: no encontró el fin de carrera")
        move(-direction * back, fast, f" I{1 << j}")
        if not move(direction * back * 2, slow, f" H{j}"):
            raise BackendError(f"referencia del eje {a}: no volvió a encontrar el fin de carrera")
        move(-direction * back, slow, f" I{1 << j}")
        with self.lock:
            steps = list(self.eng["steps"])
            steps[j] = round(((end if end is not None else 0.0) - direction * back) * self._spm(a))
            self._write(["SETPOS " + " ".join(str(s) for s in steps)])
            seen = self._reports
        # Wait for reports sent after SETPOS: an older one arriving late would show the old count.
        _until_fresh = time.monotonic() + 5.0
        while time.monotonic() < _until_fresh:
            with self.lock:
                if self._reports >= seen + 2 and self.eng["steps"][j] == steps[j]:
                    self._cmd = list(self.eng["steps"])
                    return
            time.sleep(0.01)
        raise BackendError(f"referencia del eje {a}: el motor no confirmó la posición")

    # --- latency test --------------------------------------------------------------------------------
    def latency_test(self, seconds: float = 5.0) -> dict:
        """Like Mach3's DriverTest: measures how late the engine's ticks wake up on this PC.

        Returns ``{"ticks", "max_us", "p99_us", "p999_us", "frecuencia_sugerida", "texto"}``. The
        suggested frequency keeps one tick longer than 99.9 % of the delays (and ≤ the configured
        one). Only at rest. Raises BackendError if busy or no answer.
        """
        with self.lock:
            self._idle("prueba de latencia")
            self._latency = None
        self._write([f"TEST {seconds:g}"])
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds + 10:
            with self.lock:
                lat = self._latency
            if lat is not None:
                worst = max(lat["p999_us"], 1.0)
                hz = min(self.freq, max(5000.0, math.floor(1e6 / (2 * worst) / 5000) * 5000))
                ok = lat["max_us"] < 0.5e6 / self.freq
                lat.update(frecuencia_sugerida=hz, texto=(
                    f"Latencia: máxima {lat['max_us']:.0f} µs, 99,9 % por debajo de {lat['p999_us']:.0f} µs. "
                    + ("Esta PC sirve para la frecuencia configurada." if ok else
                       f"Hubo retrasos largos: conviene {hz:.0f} Hz o menos, cerrar programas y desactivar "
                       "el ahorro de energía de la CPU.")))
                return lat
            time.sleep(0.05)
        raise BackendError("la prueba de latencia no respondió")
