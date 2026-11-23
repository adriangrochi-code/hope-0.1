"""grbl / grblHAL / FluidNC por puerto serie o TCP.

Protocolo de **conteo de caracteres** (el recomendado por grbl): se envían líneas mientras quepan en
el búfer de recepción del controlador (128 bytes en grbl 1.1; 1024 o más en grblHAL) y cada ``ok`` o
``error:N`` libera la línea más vieja. Así el planificador del controlador nunca se queda sin bloques
y la máquina no tartamudea, ni siquiera con una PC vieja.

Comandos en tiempo real: ``?`` estado, ``!`` retención, ``~`` reanudar, ``Ctrl-X`` reinicio.

El puerto es cualquier objeto con ``write(bytes)``, ``readline() -> bytes`` (con tiempo de espera:
``b""`` si no llegó nada) y ``close()``; ``open_port`` crea uno serie (pyserial) o TCP.

POR QUÉ CONTEO DE CARACTERES (y no «enviar y esperar ok»)
    Con «enviar y esperar» la PC manda una línea, espera el ``ok`` y recién ahí manda la siguiente:
    entre línea y línea el búfer de grbl se vacía y, con tramos cortos (curvas, 3D), la máquina frena
    a cada rato. Contando caracteres siempre hay varias líneas esperando en el controlador, así que su
    planificador ve hacia adelante y mantiene la velocidad. Lo único que hay que garantizar es no
    pasarse del tamaño del búfer: si se desborda, grbl pierde caracteres y ejecuta basura.

CÓMO ENCAJA
    ``make_backend`` crea un ``GrblBackend`` con ``from_config``; el ``Controller`` le pasa el programa
    traducido al dialecto grbl y un ``on_ack`` que atiende las pausas M0. ``hope.cnc.links`` (la
    conexión en vivo de la app) reutiliza ``open_port`` y ``describe``.

EJEMPLO
    >>> b = GrblBackend.from_config({"puerto": "/dev/ttyUSB0", "baudios": 115200, "bufer": 128})
    >>> b.connect(); b.banner                  # p. ej. «Grbl 1.1h ['$' for help]»
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
# ================================== CÓDIGOS DE ERROR Y ALARMA =====================================
# ==================================================================================================

# Traducción de los ``error:N`` más comunes de grbl 1.1 (la lista completa está en la wiki de grbl,
# «Interface → Error codes»). Los que no están se muestran como «ver la documentación de grbl».
# grblHAL y FluidNC usan los mismos números para estos códigos.
ERRORS = {
    1: "falta la letra del comando", 2: "número mal formado", 3: "comando $ no reconocido",
    9: "bloqueado por alarma (desbloquear con $X o referenciar con $H)", 15: "el jog sale del recorrido",
    20: "comando G no soportado", 22: "avance no definido", 24: "dos comandos con los mismos ejes",
    33: "objetivo del arco inválido", 36: "palabras sin usar en el bloque",
}
# Traducción de los ``ALARM:N`` de grbl 1.1. Una alarma bloquea todo hasta $X o $H; las 1, 2 y 3
# implican que la posición ya no es confiable y hay que volver a referenciar.
ALARMS = {
    1: "final de carrera activado (hard limit): revisar posición y referenciar",
    2: "el movimiento sale del recorrido (soft limit)", 3: "reinicio durante el movimiento: posición perdida",
    4: "falló la sonda (estado inicial)", 5: "la sonda no tocó", 6: "referenciado abortado",
    7: "puerta abierta durante el referenciado", 8: "no se pudo liberar el final de carrera",
    9: "no se encontró el final de carrera",
}
# Informe de estado de grbl 1.1: ``<Estado|Campo:valor|Campo:valor…>``, p. ej.
# ``<Idle|MPos:0.000,0.000,0.000|FS:0,0|WCO:0.000,0.000,0.000>``. Grupo 1 = estado (puede traer
# subestado: «Hold:0»), grupo 2 = el resto de los campos separados por «|».
STATUS = re.compile(r"<([^|>]+)\|(.*)>")


# ==================================================================================================
# ============================================= PUERTOS ============================================
# ==================================================================================================

class TcpPort:
    """Puerto TCP con la misma interfaz que ``serial.Serial`` (lo que usa ``GrblBackend``).

    Para grblHAL por Ethernet y FluidNC por Wi-Fi (servicio «telnet», puerto 23 por defecto).
    TCP entrega un chorro de bytes sin cortes de línea, así que se acumula en ``_buf`` hasta tener
    una línea completa.
    """

    def __init__(self, host: str, port: int, timeout: float = 0.2):
        """Conecta a ``host:port``. ``timeout`` (s) es la espera de ``readline`` (no la de conexión,
        que es fija en 5 s). Lanza ``OSError`` si no se puede conectar."""
        self.sock = socket.create_connection((host, port), timeout=5)
        self.sock.settimeout(timeout)
        self._buf = b""

    def write(self, data: bytes):
        """Envía todos los bytes (``sendall`` reintenta hasta mandar todo)."""
        self.sock.sendall(data)

    def readline(self) -> bytes:
        """Devuelve una línea completa con su ``\\n``, o ``b""`` si no llegó nada a tiempo (como pyserial).

        Lanza ``BackendError`` si el controlador cerró la conexión.
        """
        while b"\n" not in self._buf:
            try:
                chunk = self.sock.recv(4096)
            except TimeoutError:                       # sin datos dentro de ``timeout``: igual que pyserial
                return b""
            if not chunk:                              # recv vacío = el otro extremo cerró
                raise BackendError("el controlador cerró la conexión")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line + b"\n"

    def close(self):
        """Cierra la conexión."""
        self.sock.close()


def open_port(port: str, baud: int = 115200, timeout: float = 0.2):
    """Abre el puerto del controlador.

    Parámetros:
        port:    ``tcp://host[:puerto]`` (puerto 23 si no se indica) o el nombre del puerto serie
                 (``COM3`` en Windows, ``/dev/ttyUSB0`` o ``/dev/ttyACM0`` en Linux).
        baud:    baudios del puerto serie (115200 es lo estándar en grbl 1.1; ignorado en TCP).
        timeout: espera máxima de cada lectura (s). 0.2 s deja responder rápido a la interfaz sin
                 gastar CPU; mucho más alto hace lenta la detección de «sin respuesta».

    Devuelve un objeto con ``write``/``readline``/``close``. Errores: ``BackendError`` si falta
    pyserial; ``OSError``/``serial.SerialException`` si el puerto no existe o está ocupado.
    """
    if port.startswith("tcp://"):
        host, _, p = port[6:].partition(":")
        return TcpPort(host, int(p or 23), timeout)
    try:
        # pyserial se importa solo si hace falta (TCP y el simulador no lo necesitan)
        import serial
    except ImportError:
        raise BackendError("falta pyserial para el puerto serie: pip install pyserial") from None
    return serial.Serial(port, baud, timeout=timeout)


# ==================================================================================================
# ======================================== INTERPRETACIÓN ==========================================
# ==================================================================================================

def parse_status(line: str, axes: tuple[str, ...] = ("X", "Y", "Z")) -> Status:
    """Convierte un informe de estado de grbl (respuesta a ``?``) en un ``Status``.

    Parámetros:
        line: p. ej. ``<Run|MPos:10.000,5.000,-1.000|FS:500,12000|WCO:0,0,-20>``.
        axes: nombres de los ejes en el orden en que grbl manda los números.

    grbl manda **o** ``MPos`` (máquina) **o** ``WPos`` (trabajo), según ``$10``, y el ``WCO`` (origen
    de trabajo activo) solo cada tantos informes. Con ``WCO`` se calcula el que falta:
    WPos = MPos − WCO. Si no vino ``WCO`` en este informe, el otro queda vacío.

    ``FS:avance,rpm`` (o ``F:avance`` en grbl sin husillo variable) da el avance (mm/min) y las rpm.

    Lanza ``BackendError`` si la línea no tiene la forma ``<…|…>``.
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
    """Agrega la explicación en español a ``error:N`` / ``ALARM:N``; cualquier otra respuesta vuelve igual.

    Ejemplo: ``describe("error:22")`` → ``"error:22 (avance no definido)"``.
    """
    m = re.match(r"(error|ALARM):(\d+)", resp)
    if not m:
        return resp
    table = ERRORS if m.group(1) == "error" else ALARMS
    return f"{resp} ({table.get(int(m.group(2)), 'ver la documentación de grbl')})"


# ==================================================================================================
# ========================================== CONTROLADOR ===========================================
# ==================================================================================================

class GrblBackend(Backend):
    """Controlador grbl 1.1 / grblHAL / FluidNC (todos hablan el mismo protocolo de texto).

    Atributos:
        port:        el puerto abierto (serie o ``TcpPort``).
        rx_buffer:   tamaño del búfer de recepción del controlador (bytes), ver ``__init__``.
        axes:        nombres de los ejes en el orden de grbl.
        ack_timeout: espera máxima sin respuestas durante un envío (s), ver ``__init__``.
        banner:      el saludo del controlador leído en ``connect`` (vacío si no saludó).
        messages:    líneas que llegaron y no eran ``ok``/``error``/estado (``[MSG:…]``, ``[GC:…]``…),
                     guardadas para mostrarlas o registrarlas.
    """

    name = "grbl"
    physical = True

    def __init__(self, port, rx_buffer: int = 128, axes: tuple[str, ...] = ("X", "Y", "Z"),
                 ack_timeout: float = 300.0):
        """Prepara el controlador sobre un puerto ya abierto (ver ``open_port``).

        Parámetros:
            port:        objeto con ``write``/``readline``/``close`` (en pruebas, uno falso).
            rx_buffer:   búfer de recepción del controlador (bytes). 128 en grbl 1.1 (Arduino UNO);
                         1024 o más en grblHAL/FluidNC. Más chico de lo real = funciona pero envía
                         menos adelantado; **más grande de lo real = desborda el búfer y se pierden
                         caracteres** (movimientos equivocados). Ante la duda, 128.
            axes:        nombres de los ejes, en el orden en que grbl manda las posiciones.
            ack_timeout: segundos sin ninguna respuesta antes de dar el envío por perdido. Largo (300 s)
                         porque un G4 (pausa) o un movimiento lento pueden tener el búfer lleno mucho
                         rato sin que llegue un ``ok``. Muy corto aborta trabajos sanos.
        """
        self.port = port
        self.rx_buffer = int(rx_buffer)
        self.axes = axes
        self.ack_timeout = ack_timeout
        self.banner = ""
        self.messages: list[str] = []

    @classmethod
    def from_config(cls, cfg: dict) -> GrblBackend:
        """Crea el controlador desde la sección ``controlador`` del perfil y abre el puerto.

        Claves: ``puerto`` (obligatoria), ``baudios`` (115200), ``bufer`` (128), ``ejes`` (X, Y, Z).
        Lanza ``BackendError`` si falta el puerto o pyserial.
        """
        if not cfg.get("puerto"):
            raise BackendError("controlador grbl sin «puerto» (p. ej. /dev/ttyUSB0, COM3 o tcp://192.168.1.50:23)")
        port = open_port(str(cfg["puerto"]), int(cfg.get("baudios", 115200)))
        return cls(port, rx_buffer=int(cfg.get("bufer", 128)), axes=tuple(cfg.get("ejes", ("X", "Y", "Z"))))

    # --- bajo nivel --------------------------------------------------------------
    def _readline(self) -> str:
        """Una línea del controlador como texto sin espacios ni fin de línea; ``""`` si no llegó nada.

        ``errors="replace"``: ruido eléctrico en el cable USB puede traer bytes no ASCII; mejor un «?»
        que una excepción en medio de un corte.
        """
        return self.port.readline().decode("ascii", errors="replace").strip()

    def _command(self, cmd: str, timeout: float = 5.0) -> list[str]:
        """Envía un comando de línea y junta las respuestas hasta ``ok``/``error``.

        Para comandos de consulta fuera de un envío (``$#``, ``$$``, ``$G``). ``timeout`` en segundos.
        Devuelve las líneas recibidas antes del ``ok``. Lanza ``BackendError`` ante ``error:``,
        ``ALARM:`` o si no llega el ``ok`` a tiempo.
        """
        self.port.write((cmd + "\n").encode("ascii"))
        out, deadline = [], time.monotonic() + timeout     # reloj monótono: no le afectan cambios de hora
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
        """Despierta el controlador y lee el saludo («Grbl 1.1h», «GrblHAL 1.1f», «FluidNC…»).

        Manda dos saltos de línea (la forma clásica de «despertar» grbl) y espera el saludo hasta
        ``wait`` segundos: al abrir el puerto serie, el Arduino se reinicia y tarda ~2 s en arrancar.
        Si no saluda no es error (puede que ya estuviera andando); ``banner`` queda vacío.
        """
        self.port.write(b"\r\n\r\n")
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and not self.banner:
            line = self._readline()
            if line.lower().startswith(("grbl", "fluidnc")) or "grbl" in line.lower():
                self.banner = line

    def close(self) -> None:
        """Cierra el puerto."""
        self.port.close()

    def status(self) -> Status:
        """Pide el estado con ``?`` (comando en tiempo real: no ocupa el búfer ni espera turno).

        Las líneas que lleguen mientras tanto se guardan en ``messages``. Lanza ``BackendError`` si
        no llega un informe de estado en 2 s.
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
        """Orígenes reales del controlador con ``$#`` (G54–G59 y además G28, G30, G92).

        grbl responde líneas como ``[G54:0.000,0.000,-20.000]``. Devuelve ``{"G54": {"X": mm, …}, …}``.
        """
        out = {}
        for line in self._command("$#"):
            m = re.match(r"\[(G5[4-9]|G28|G30|G92):([^\]]+)\]", line)
            if m:
                out[m.group(1)] = {a: float(v) for a, v in zip(self.axes, m.group(2).split(","))}
        return out

    def hold(self) -> None:
        """Retención de avance (``!``): frena con la aceleración configurada sin perder posición."""
        self.port.write(b"!")

    def resume(self) -> None:
        """Reanuda después de una retención o de un M0 (``~``, «cycle start»)."""
        self.port.write(b"~")

    def reset(self) -> None:
        """Reinicio por software (Ctrl-X, 0x18): vacía el búfer y detiene todo.

        Si la máquina se estaba moviendo, grbl entra en ALARM:3 (posición perdida) y hay que
        referenciar. No reemplaza la parada de emergencia física.
        """
        self.port.write(b"\x18")

    def stream(self, lines, progress=None, stop_on_error=True, on_ack=None) -> StreamResult:
        """Envía el programa con el protocolo de conteo de caracteres (ver el encabezado).

        Parámetros:
            lines:         G-code ya traducido al dialecto grbl.
            progress:      ``progress(enviadas, confirmadas, total)`` en cada confirmación.
            stop_on_error: ante un ``error:N``, reiniciar (Ctrl-X) y abortar. Lo seguro: grbl
                           descarta la línea errónea pero sigue con las siguientes, que pueden
                           depender de ella (p. ej. un G90 que no se aplicó).
            on_ack:        ``on_ack(índice, línea) -> bool`` tras cada confirmación; si devuelve
                           False se reinicia y se aborta. El ``Controller`` lo usa para atender los M0
                           (esperar al operador y reanudar con ``~``).

        Antes de enviar se limpian las líneas: sin comentarios ``( )`` ni ``;``, en mayúsculas y sin
        espacios (ocupan menos búfer; grbl los ignora igual). Las vacías se saltean, así que los
        números de ``sent``/``acknowledged`` cuentan líneas limpias, no las originales.

        Devuelve ``StreamResult``. Una ``ALARM:`` aborta siempre (la máquina ya se detuvo sola).
        Lanza ``BackendError`` si alguna línea no entra en el búfer (no se podría enviar nunca).
        """
        clean = []
        for ln in lines:
            s = re.sub(r"\(.*?\)|;.*", "", ln).strip().upper().replace(" ", "")
            if s:
                if len(s) + 1 > self.rx_buffer:                # +1 por el «\n» que se agrega al enviar
                    raise BackendError(f"línea más larga que el búfer del controlador: {s[:40]}…")
                clean.append(s)
        pending: deque[int] = deque()            # tamaños de las líneas sin confirmar
        sent = acked = 0
        errors: list[str] = []
        last = time.monotonic()                  # última vez que llegó algo (para ``ack_timeout``)
        while acked < len(clean):
            # Llenar el búfer: mandar mientras lo ya enviado sin confirmar + la línea nueva quepan.
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
                # grbl responde en orden: cada ok/error corresponde a la línea más vieja sin confirmar
                # y libera exactamente sus bytes del búfer.
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
                self.messages.append(line)       # [MSG:…], informes de estado sueltos, etc.
        return StreamResult(sent, acked, errors)
