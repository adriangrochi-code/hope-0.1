"""grbl / grblHAL / FluidNC over a serial port or TCP.

**Character-counting** protocol (the one grbl recommends): lines are sent as long as they fit in the
controller's receive buffer (128 bytes in grbl 1.1; 1024 or more in grblHAL) and each ``ok`` or
``error:N`` frees the oldest line. That way the controller's planner never runs out of blocks and
the machine does not stutter, not even with an old PC.

Real-time commands: ``?`` status, ``!`` hold, ``~`` resume, ``Ctrl-X`` reset.

The port is any object with ``write(bytes)``, ``readline() -> bytes`` (with a timeout: ``b""`` if
nothing arrived) and ``close()``; ``open_port`` creates a serial one (pyserial) or a TCP one.

WHY CHARACTER COUNTING (and not «send and wait for ok»)
    With «send and wait» the PC sends one line, waits for the ``ok`` and only then sends the next:
    between lines grbl's buffer drains and, with short segments (curves, 3D), the machine slows down
    constantly. With character counting there are always several lines waiting in the controller, so
    its planner can look ahead and keep the speed up. The only thing to guarantee is never exceeding
    the buffer size: if it overflows, grbl loses characters and executes garbage.

HOW IT FITS
    ``make_backend`` creates a ``GrblBackend`` with ``from_config``; the ``Controller`` hands it the
    program translated to the grbl dialect and an ``on_ack`` that handles M0 pauses. ``hope.cnc.links``
    (the app's live connection) reuses ``open_port`` and ``describe``.

EXAMPLE
    >>> b = GrblBackend.from_config({"puerto": "/dev/ttyUSB0", "baudios": 115200, "bufer": 128})
    >>> b.connect(); b.banner                  # e.g. «Grbl 1.1h ['$' for help]»
    >>> b.stream(["G21 G90", "G0 X10", "G1 X20 F500"]).ok
    True
"""

from __future__ import annotations

import re
import socket
import time
from collections import deque

from hope.cnc.backends.base import Backend, BackendError, Status, StreamResult

# ==================================================================================================
# ===================================== ERROR AND ALARM CODES ======================================
# ==================================================================================================

# Translation of the most common grbl 1.1 ``error:N`` codes (the full list is in the grbl wiki,
# «Interface → Error codes»). The ones not listed are shown as «ver la documentación de grbl».
# grblHAL and FluidNC use the same numbers for these codes.
ERRORS = {
    1: "falta la letra del comando", 2: "número mal formado", 3: "comando $ no reconocido",
    9: "bloqueado por alarma (desbloquear con $X o referenciar con $H)", 15: "el jog sale del recorrido",
    20: "comando G no soportado", 22: "avance no definido", 24: "dos comandos con los mismos ejes",
    33: "objetivo del arco inválido", 36: "palabras sin usar en el bloque",
}
# Translation of grbl 1.1 ``ALARM:N`` codes. An alarm locks everything until $X or $H; alarms 1, 2
# and 3 mean the position is no longer reliable and the machine has to be homed again.
ALARMS = {
    1: "final de carrera activado (hard limit): revisar posición y referenciar",
    2: "el movimiento sale del recorrido (soft limit)", 3: "reinicio durante el movimiento: posición perdida",
    4: "falló la sonda (estado inicial)", 5: "la sonda no tocó", 6: "referenciado abortado",
    7: "puerta abierta durante el referenciado", 8: "no se pudo liberar el final de carrera",
    9: "no se encontró el final de carrera",
}
# grbl 1.1 status report: ``<State|Field:value|Field:value…>``, e.g.
# ``<Idle|MPos:0.000,0.000,0.000|FS:0,0|WCO:0.000,0.000,0.000>``. Group 1 = state (may carry a
# substate: «Hold:0»), group 2 = the rest of the fields separated by «|».
STATUS = re.compile(r"<([^|>]+)\|(.*)>")


# ==================================================================================================
# ============================================= PORTS ==============================================
# ==================================================================================================

class TcpPort:
    """TCP port with the same interface as ``serial.Serial`` (what ``GrblBackend`` uses).

    For grblHAL over Ethernet and FluidNC over Wi-Fi («telnet» service, port 23 by default).
    TCP delivers a stream of bytes with no line boundaries, so data is accumulated in ``_buf`` until
    there is a complete line.
    """

    def __init__(self, host: str, port: int, timeout: float = 0.2):
        """Connect to ``host:port``. ``timeout`` (s) is the ``readline`` wait (not the connection
        timeout, which is fixed at 5 s). Raises ``OSError`` if it cannot connect."""
        self.sock = socket.create_connection((host, port), timeout=5)
        self.sock.settimeout(timeout)
        self._buf = b""

    def write(self, data: bytes):
        """Send all the bytes (``sendall`` retries until everything is sent)."""
        self.sock.sendall(data)

    def readline(self) -> bytes:
        """Return a complete line with its ``\\n``, or ``b""`` if nothing arrived in time (like pyserial).

        Raises ``BackendError`` if the controller closed the connection.
        """
        while b"\n" not in self._buf:
            try:
                chunk = self.sock.recv(4096)
            except TimeoutError:                       # no data within ``timeout``: same as pyserial
                return b""
            if not chunk:                              # empty recv = the other end closed
                raise BackendError("el controlador cerró la conexión")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line + b"\n"

    def close(self):
        """Close the connection."""
        self.sock.close()


def open_port(port: str, baud: int = 115200, timeout: float = 0.2):
    """Open the controller's port.

    Parameters:
        port:    ``tcp://host[:port]`` (port 23 if not given) or the serial port name
                 (``COM3`` on Windows, ``/dev/ttyUSB0`` or ``/dev/ttyACM0`` on Linux).
        baud:    serial port baud rate (115200 is the grbl 1.1 standard; ignored for TCP).
        timeout: maximum wait for each read (s). 0.2 s lets the interface respond quickly without
                 burning CPU; much higher makes detecting «no response» slow.

    Returns an object with ``write``/``readline``/``close``. Raises: ``BackendError`` if pyserial is
    missing; ``OSError``/``serial.SerialException`` if the port does not exist or is busy.
    """
    if port.startswith("tcp://"):
        host, _, p = port[6:].partition(":")
        return TcpPort(host, int(p or 23), timeout)
    try:
        # pyserial is imported only when needed (TCP and the simulator do not need it)
        import serial
    except ImportError:
        raise BackendError("falta pyserial para el puerto serie: pip install pyserial") from None
    return serial.Serial(port, baud, timeout=timeout)


# ==================================================================================================
# ============================================ PARSING =============================================
# ==================================================================================================

def parse_status(line: str, axes: tuple[str, ...] = ("X", "Y", "Z")) -> Status:
    """Turn a grbl status report (the answer to ``?``) into a ``Status``.

    Parameters:
        line: e.g. ``<Run|MPos:10.000,5.000,-1.000|FS:500,12000|WCO:0,0,-20>``.
        axes: axis names in the order grbl sends the numbers.

    grbl sends **either** ``MPos`` (machine) **or** ``WPos`` (work), depending on ``$10``, and the
    ``WCO`` (active work offset) only every so many reports. With ``WCO`` the missing one is computed:
    WPos = MPos − WCO. If this report did not include ``WCO``, the other one stays empty.

    ``FS:feed,rpm`` (or ``F:feed`` on grbl without variable spindle) gives the feed (mm/min) and rpm.

    Raises ``BackendError`` if the line does not have the form ``<…|…>``.
    """
    m = STATUS.match(line.strip())
    if not m:
        raise BackendError(f"estado no reconocido: {line!r}")
    st = Status(state=m.group(1).split(":")[0], raw=line.strip())          # «Hold:0» → «Hold»
    fields = dict(f.split(":", 1) for f in m.group(2).split("|") if ":" in f)
    vec = lambda s: {a: float(v) for a, v in zip(axes, s.split(","))}
    wco = vec(fields["WCO"]) if "WCO" in fields else None
    if "MPos" in fields:
        st.mpos = vec(fields["MPos"])
        if wco:
            st.wpos = {a: st.mpos[a] - wco.get(a, 0.0) for a in st.mpos}
    if "WPos" in fields:
        st.wpos = vec(fields["WPos"])
        if wco:
            st.mpos = {a: st.wpos[a] + wco.get(a, 0.0) for a in st.wpos}
    fs = fields.get("FS") or fields.get("F")
    if fs:
        parts = fs.split(",")
        st.feed = float(parts[0])
        st.spindle = float(parts[1]) if len(parts) > 1 else 0.0
    return st


def describe(resp: str) -> str:
    """Append the Spanish explanation to ``error:N`` / ``ALARM:N``; any other response comes back unchanged.

    Example: ``describe("error:22")`` → ``"error:22 (avance no definido)"``.
    """
    m = re.match(r"(error|ALARM):(\d+)", resp)
    if not m:
        return resp
    table = ERRORS if m.group(1) == "error" else ALARMS
    return f"{resp} ({table.get(int(m.group(2)), 'ver la documentación de grbl')})"


# ==================================================================================================
# =========================================== CONTROLLER ===========================================
# ==================================================================================================

class GrblBackend(Backend):
    """grbl 1.1 / grblHAL / FluidNC controller (they all speak the same text protocol).

    Attributes:
        port:        the open port (serial or ``TcpPort``).
        rx_buffer:   size of the controller's receive buffer (bytes), see ``__init__``.
        axes:        axis names in grbl's order.
        ack_timeout: maximum wait without responses during a send (s), see ``__init__``.
        banner:      the controller's greeting read in ``connect`` (empty if it did not greet).
        messages:    lines that arrived and were not ``ok``/``error``/status (``[MSG:…]``, ``[GC:…]``…),
                     kept to show or log them.
    """

    name = "grbl"
    physical = True

    def __init__(self, port, rx_buffer: int = 128, axes: tuple[str, ...] = ("X", "Y", "Z"),
                 ack_timeout: float = 300.0):
        """Set up the controller on an already open port (see ``open_port``).

        Parameters:
            port:        object with ``write``/``readline``/``close`` (a fake one in tests).
            rx_buffer:   controller receive buffer (bytes). 128 in grbl 1.1 (Arduino UNO);
                         1024 or more in grblHAL/FluidNC. Smaller than the real one = works but sends
                         less ahead; **larger than the real one = overflows the buffer and characters
                         are lost** (wrong moves). When in doubt, 128.
            axes:        axis names, in the order grbl sends the positions.
            ack_timeout: seconds without any response before giving the send up as lost. Long (300 s)
                         because a G4 (dwell) or a slow move can keep the buffer full for a long time
                         without an ``ok`` arriving. Too short aborts healthy jobs.
        """
        self.port = port
        self.rx_buffer = int(rx_buffer)
        self.axes = axes
        self.ack_timeout = ack_timeout
        self.banner = ""
        self.messages: list[str] = []

    @classmethod
    def from_config(cls, cfg: dict) -> GrblBackend:
        """Create the controller from the profile's ``controlador`` section and open the port.

        Keys: ``puerto`` (required), ``baudios`` (115200), ``bufer`` (128), ``ejes`` (X, Y, Z).
        Raises ``BackendError`` if the port or pyserial is missing.
        """
        if not cfg.get("puerto"):
            raise BackendError("controlador grbl sin «puerto» (p. ej. /dev/ttyUSB0, COM3 o tcp://192.168.1.50:23)")
        port = open_port(str(cfg["puerto"]), int(cfg.get("baudios", 115200)))
        return cls(port, rx_buffer=int(cfg.get("bufer", 128)), axes=tuple(cfg.get("ejes", ("X", "Y", "Z"))))

    # --- low level ---------------------------------------------------------------
    def _readline(self) -> str:
        """One line from the controller as text without surrounding spaces or line ending; ``""`` if
        nothing arrived.

        ``errors="replace"``: electrical noise on the USB cable can bring non-ASCII bytes; a «?» is
        better than an exception in the middle of a cut.
        """
        return self.port.readline().decode("ascii", errors="replace").strip()

    def _command(self, cmd: str, timeout: float = 5.0) -> list[str]:
        """Send a line command and collect the responses up to ``ok``/``error``.

        For query commands outside a program send (``$#``, ``$$``, ``$G``). ``timeout`` in seconds.
        Returns the lines received before the ``ok``. Raises ``BackendError`` on ``error:``,
        ``ALARM:`` or if the ``ok`` does not arrive in time.
        """
        self.port.write((cmd + "\n").encode("ascii"))
        out, deadline = [], time.monotonic() + timeout     # monotonic clock: unaffected by clock changes
        while time.monotonic() < deadline:
            line = self._readline()
            if not line:
                continue
            if line == "ok":
                return out
            if line.startswith(("error:", "ALARM:")):
                raise BackendError(describe(line))
            out.append(line)
        raise BackendError(f"sin respuesta a «{cmd}»")

    # --- interface ---------------------------------------------------------------
    def connect(self, wait: float = 3.0) -> None:
        """Wake the controller and read its greeting («Grbl 1.1h», «GrblHAL 1.1f», «FluidNC…»).

        Sends two line breaks (the classic way to «wake up» grbl) and waits for the greeting for up to
        ``wait`` seconds: opening the serial port resets the Arduino, which takes ~2 s to boot.
        No greeting is not an error (it may already have been running); ``banner`` stays empty.
        """
        self.port.write(b"\r\n\r\n")
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and not self.banner:
            line = self._readline()
            if line.lower().startswith(("grbl", "fluidnc")) or "grbl" in line.lower():
                self.banner = line

    def close(self) -> None:
        """Close the port."""
        self.port.close()

    def status(self) -> Status:
        """Ask for the status with ``?`` (a real-time command: it takes no buffer space and does not
        wait its turn).

        Lines arriving meanwhile are stored in ``messages``. Raises ``BackendError`` if no status
        report arrives within 2 s.
        """
        self.port.write(b"?")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            line = self._readline()
            if line.startswith("<"):
                return parse_status(line, self.axes)
            if line:
                self.messages.append(line)
        raise BackendError("el controlador no respondió al pedido de estado")

    def work_offsets(self) -> dict[str, dict[str, float]]:
        """The controller's actual offsets via ``$#`` (G54–G59 plus G28, G30, G92).

        grbl answers lines like ``[G54:0.000,0.000,-20.000]``. Returns ``{"G54": {"X": mm, …}, …}``.
        """
        out = {}
        for line in self._command("$#"):
            m = re.match(r"\[(G5[4-9]|G28|G30|G92):([^\]]+)\]", line)
            if m:
                out[m.group(1)] = {a: float(v) for a, v in zip(self.axes, m.group(2).split(","))}
        return out

    def hold(self) -> None:
        """Feed hold (``!``): decelerates with the configured acceleration without losing position."""
        self.port.write(b"!")

    def resume(self) -> None:
        """Resume after a hold or an M0 (``~``, «cycle start»)."""
        self.port.write(b"~")

    def reset(self) -> None:
        """Software reset (Ctrl-X, 0x18): empties the buffer and stops everything.

        If the machine was moving, grbl enters ALARM:3 (position lost) and it has to be homed.
        It does not replace the physical emergency stop.
        """
        self.port.write(b"\x18")

    def stream(self, lines, progress=None, stop_on_error=True, on_ack=None) -> StreamResult:
        """Send the program with the character-counting protocol (see the module header).

        Parameters:
            lines:         G-code already translated to the grbl dialect.
            progress:      ``progress(sent, acknowledged, total)`` on every acknowledgement.
            stop_on_error: on an ``error:N``, reset (Ctrl-X) and abort. The safe choice: grbl
                           discards the faulty line but carries on with the following ones, which may
                           depend on it (e.g. a G90 that was not applied).
            on_ack:        ``on_ack(index, line) -> bool`` after each acknowledgement; if it returns
                           False the controller is reset and sending aborts. The ``Controller`` uses
                           it to handle M0s (wait for the operator and resume with ``~``).

        Lines are cleaned before sending: no ``( )`` or ``;`` comments, upper case and no spaces
        (they take less buffer; grbl ignores them anyway). Empty ones are skipped, so the
        ``sent``/``acknowledged`` numbers count clean lines, not the original ones.

        Returns ``StreamResult``. An ``ALARM:`` always aborts (the machine has already stopped by
        itself). Raises ``BackendError`` if a line does not fit in the buffer (it could never be sent).
        """
        clean = []
        for ln in lines:
            s = re.sub(r"\(.*?\)|;.*", "", ln).strip().upper().replace(" ", "")
            if s:
                if len(s) + 1 > self.rx_buffer:                # +1 for the «\n» added when sending
                    raise BackendError(f"línea más larga que el búfer del controlador: {s[:40]}…")
                clean.append(s)
        pending: deque[int] = deque()            # sizes of the unacknowledged lines
        sent = acked = 0
        errors: list[str] = []
        last = time.monotonic()                  # last time something arrived (for ``ack_timeout``)
        while acked < len(clean):
            # Fill the buffer: keep sending while the unacknowledged bytes + the new line still fit.
            while sent < len(clean) and sum(pending) + len(clean[sent]) + 1 <= self.rx_buffer:
                self.port.write((clean[sent] + "\n").encode("ascii"))
                pending.append(len(clean[sent]) + 1)
                sent += 1
            line = self._readline()
            if not line:
                if time.monotonic() - last > self.ack_timeout:
                    return StreamResult(sent, acked, errors + ["sin respuesta del controlador"], aborted=True)
                continue
            last = time.monotonic()
            if line == "ok" or line.startswith("error:"):
                # grbl answers in order: each ok/error belongs to the oldest unacknowledged line
                # and frees exactly its bytes from the buffer.
                pending.popleft()
                acked += 1
                if line != "ok":
                    errors.append(f"línea enviada {acked} «{clean[acked - 1]}»: {describe(line)}")
                    if stop_on_error:
                        self.reset()
                        return StreamResult(sent, acked, errors, aborted=True)
                if progress:
                    progress(sent, acked, len(clean))
                if on_ack and not on_ack(acked - 1, clean[acked - 1]):
                    self.reset()
                    return StreamResult(sent, acked, errors, aborted=True)
            elif line.startswith("ALARM:"):
                return StreamResult(sent, acked, errors, aborted=True, alarm=describe(line))
            else:
                self.messages.append(line)       # [MSG:…], stray status reports, etc.
        return StreamResult(sent, acked, errors)
