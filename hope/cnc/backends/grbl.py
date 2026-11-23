"""grbl / grblHAL / FluidNC por puerto serie o TCP.

Protocolo de **conteo de caracteres** (el recomendado por grbl): se envían líneas mientras quepan en
el búfer de recepción del controlador (128 bytes en grbl 1.1; 1024 o más en grblHAL) y cada ``ok`` o
``error:N`` libera la línea más vieja. Así el planificador del controlador nunca se queda sin bloques
y la máquina no tartamudea, ni siquiera con una PC vieja.

Comandos en tiempo real: ``?`` estado, ``!`` retención, ``~`` reanudar, ``Ctrl-X`` reinicio.

El puerto es cualquier objeto con ``write(bytes)``, ``readline() -> bytes`` (con tiempo de espera:
``b""`` si no llegó nada) y ``close()``; ``open_port`` crea uno serie (pyserial) o TCP.
"""

from __future__ import annotations

import re
import socket
import time
from collections import deque

from hope.cnc.backends.base import Backend, BackendError, Status, StreamResult

ERRORS = {
    1: "falta la letra del comando", 2: "número mal formado", 3: "comando $ no reconocido",
    9: "bloqueado por alarma (desbloquear con $X o referenciar con $H)", 15: "el jog sale del recorrido",
    20: "comando G no soportado", 22: "avance no definido", 24: "dos comandos con los mismos ejes",
    33: "objetivo del arco inválido", 36: "palabras sin usar en el bloque",
}
ALARMS = {
    1: "final de carrera activado (hard limit): revisar posición y referenciar",
    2: "el movimiento sale del recorrido (soft limit)", 3: "reinicio durante el movimiento: posición perdida",
    4: "falló la sonda (estado inicial)", 5: "la sonda no tocó", 6: "referenciado abortado",
    7: "puerta abierta durante el referenciado", 8: "no se pudo liberar el final de carrera",
    9: "no se encontró el final de carrera",
}
STATUS = re.compile(r"<([^|>]+)\|(.*)>")


class TcpPort:
    def __init__(self, host: str, port: int, timeout: float = 0.2):
        self.sock = socket.create_connection((host, port), timeout=5)
        self.sock.settimeout(timeout)
        self._buf = b""

    def write(self, data: bytes):
        self.sock.sendall(data)

    def readline(self) -> bytes:
        while b"\n" not in self._buf:
            try:
                chunk = self.sock.recv(4096)
            except TimeoutError:
                return b""
            if not chunk:
                raise BackendError("el controlador cerró la conexión")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line + b"\n"

    def close(self):
        self.sock.close()


def open_port(port: str, baud: int = 115200, timeout: float = 0.2):
    if port.startswith("tcp://"):
        host, _, p = port[6:].partition(":")
        return TcpPort(host, int(p or 23), timeout)
    try:
        import serial
    except ImportError:
        raise BackendError("falta pyserial para el puerto serie: pip install pyserial") from None
    return serial.Serial(port, baud, timeout=timeout)


def parse_status(line: str, axes: tuple[str, ...] = ("X", "Y", "Z")) -> Status:
    m = STATUS.match(line.strip())
    if not m:
        raise BackendError(f"estado no reconocido: {line!r}")
    st = Status(state=m.group(1).split(":")[0], raw=line.strip())
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
    m = re.match(r"(error|ALARM):(\d+)", resp)
    if not m:
        return resp
    table = ERRORS if m.group(1) == "error" else ALARMS
    return f"{resp} ({table.get(int(m.group(2)), 'ver la documentación de grbl')})"


class GrblBackend(Backend):
    name = "grbl"
    physical = True

    def __init__(self, port, rx_buffer: int = 128, axes: tuple[str, ...] = ("X", "Y", "Z"),
                 ack_timeout: float = 300.0):
        self.port = port
        self.rx_buffer = int(rx_buffer)
        self.axes = axes
        self.ack_timeout = ack_timeout
        self.banner = ""
        self.messages: list[str] = []

    @classmethod
    def from_config(cls, cfg: dict) -> GrblBackend:
        if not cfg.get("puerto"):
            raise BackendError("controlador grbl sin «puerto» (p. ej. /dev/ttyUSB0, COM3 o tcp://192.168.1.50:23)")
        port = open_port(str(cfg["puerto"]), int(cfg.get("baudios", 115200)))
        return cls(port, rx_buffer=int(cfg.get("bufer", 128)), axes=tuple(cfg.get("ejes", ("X", "Y", "Z"))))

    # --- bajo nivel --------------------------------------------------------------
    def _readline(self) -> str:
        return self.port.readline().decode("ascii", errors="replace").strip()

    def _command(self, cmd: str, timeout: float = 5.0) -> list[str]:
        """Envía un comando de línea y junta las respuestas hasta ``ok``/``error``."""
        self.port.write((cmd + "\n").encode("ascii"))
        out, deadline = [], time.monotonic() + timeout
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

    # --- interfaz ---------------------------------------------------------------
    def connect(self, wait: float = 3.0) -> None:
        """Despierta el controlador y lee el saludo («Grbl 1.1h», «GrblHAL 1.1f», «FluidNC…»)."""
        self.port.write(b"\r\n\r\n")
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and not self.banner:
            line = self._readline()
            if line.lower().startswith(("grbl", "fluidnc")) or "grbl" in line.lower():
                self.banner = line

    def close(self) -> None:
        self.port.close()

    def status(self) -> Status:
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
        out = {}
        for line in self._command("$#"):
            m = re.match(r"\[(G5[4-9]|G28|G30|G92):([^\]]+)\]", line)
            if m:
                out[m.group(1)] = {a: float(v) for a, v in zip(self.axes, m.group(2).split(","))}
        return out

    def hold(self) -> None:
        self.port.write(b"!")

    def resume(self) -> None:
        self.port.write(b"~")

    def reset(self) -> None:
        self.port.write(b"\x18")

    def stream(self, lines, progress=None, stop_on_error=True, on_ack=None) -> StreamResult:
        clean = []
        for ln in lines:
            s = re.sub(r"\(.*?\)|;.*", "", ln).strip().upper().replace(" ", "")
            if s:
                if len(s) + 1 > self.rx_buffer:
                    raise BackendError(f"línea más larga que el búfer del controlador: {s[:40]}…")
                clean.append(s)
        pending: deque[int] = deque()            # tamaños de las líneas sin confirmar
        sent = acked = 0
        errors: list[str] = []
        last = time.monotonic()
        while acked < len(clean):
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
                self.messages.append(line)
        return StreamResult(sent, acked, errors)
