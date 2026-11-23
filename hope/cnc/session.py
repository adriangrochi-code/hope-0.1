"""Sesión de la app Hope CNC: el «tablero» de Mach3 por detrás de la pantalla.

Une la máquina, la conexión en vivo (:mod:`hope.cnc.links`), el programa cargado (revisado y
traducido) y el guardián. Flujo de uso, como en Mach3:

1. Al abrir, la máquina está en **RESET** (deshabilitada): el botón parpadea. Pulsarlo habilita la
   máquina para el operador (jog, MDI, referenciar, ceros). En una máquina real eso es
   ``hardware.control`` con modo manual: el guardián pide aprobación y el operador la da en la
   misma pantalla (queda en la auditoría).
2. **Cargar G-code** → se revisa (errores, límites, tiempo) y se dibuja la trayectoria.
3. **Cycle Start** → se vuelve a revisar con los orígenes reales del controlador y, en una máquina
   real, el guardián pide aprobación de ESE programa (hash + envolvente). Durante la marcha:
   Feed Hold, Stop y los ajustes de avance/husillo.

Volver a pulsar RESET (o Esc) detiene todo y deshabilita la máquina.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import secrets
import threading
import time
from collections import deque
from pathlib import Path

from hope.cnc.backends.base import BackendError
from hope.cnc.check import check_program, report_text
from hope.cnc.dialect import HEADER, emit
from hope.cnc.gcode import Interpreter, Program, interpret, linearize, modal_text
from hope.cnc.links import Link
from hope.cnc.machine import WCS, Machine, as_controller
from hope.cnc.planner import plan

MAX_PATH_POINTS = 250_000
BRAND = "TISMA CNC"
JOG_STEPS = (0.001, 0.01, 0.1, 1.0, 10.0)


class SessionError(ValueError):
    pass


class CncSession:
    def __init__(self, machine: Machine, link: Link, guardian=None, machine_path: str = "",
                 recent_dir: str | None = None, profiles=None, selection: dict | None = None, link_factory=None):
        """``profiles`` (ProfileStore) y ``selection`` activan los perfiles tipo slicer: la máquina efectiva
        sale de máquina + material + consumible + proceso. ``link_factory(machine)`` crea la conexión
        cuando se elige otra máquina con otro controlador."""
        self.m = machine
        self.link = link
        self.profiles = profiles
        self.link_factory = link_factory
        self.selection: dict | None = None
        self.profile_summary: dict | None = None
        if profiles is not None:
            from hope.cnc.profiles import effective, fix_selection
            self.selection = fix_selection(profiles, selection or {})
            self.m, self.profile_summary = effective(profiles, self.selection)
        self.guardian = guardian
        self.machine_path = machine_path
        self.id = secrets.token_hex(6)
        self.enabled = False
        self.manual_grant = None          # aprobación del modo manual (máquina real)
        self.program = None               # dict con el programa cargado
        self.pending = None               # aprobación que espera al operador {"id", "tipo", "resumen"}
        self.started_at = None
        self.paused_time = 0.0
        self.log: deque[tuple[float, str]] = deque(maxlen=300)
        self.mdi_history: deque[str] = deque(maxlen=30)
        self.lock = threading.RLock()
        self.mdi_interp = Interpreter(as_controller(self.m))    # estado modal de MDI (G90/G91, unidades…)
        # botones de Mach3 con LED: Single BLK, Block Delete, M1 Optional Stop
        self.opts = {"bloque_a_bloque": False, "borrado_bloque": False, "parada_opcional": False}
        self.next_line = 0                # «Set Next Line» / «Run From Here» (0 = desde el principio)
        self.from_here = False
        self.run_map: list[int] | None = None   # línea enviada → índice en las emitidas del programa
        self.recent_dir = Path(recent_dir) if recent_dir else None
        self.recent: list[dict] = self._read_recent()
        self._recent_mem: dict[str, str] = {}
        self.say(f"{BRAND} listo. Pulsá RESET para habilitar la máquina.")

    # --- utilidades -------------------------------------------------------------
    def say(self, msg: str) -> None:
        self.log.append((time.time(), msg))

    def _require_enabled(self, what: str):
        if not self.enabled:
            raise SessionError(f"{what}: la máquina está en RESET (pulsá RESET para habilitarla)")

    def _machine_real(self) -> Machine:
        try:
            real = self.link.offsets()
        except BackendError:
            real = {}
        if not real:
            return self.m
        return dataclasses.replace(self.m, offsets={**self.m.offsets, **{k: dict(v) for k, v in real.items()}})

    def _guard(self, action):
        """None si está permitido; si no, deja la aprobación pendiente para la pantalla."""
        if not self.link.physical:
            return None
        if self.guardian is None:
            raise SessionError("sin guardián no se mueve una máquina real (configs/guardian.yaml)")
        d = self.guardian.decide(action)
        if d.allowed:
            return None
        if not d.pending_request:
            raise SessionError("; ".join(d.reasons))
        return d.pending_request

    # --- RESET (habilitar / deshabilitar) -------------------------------------------
    def reset(self) -> dict:
        with self.lock:
            if self.enabled:
                self.link.stop()
                self.enabled = False
                self.manual_grant = None
                self.say("RESET: máquina deshabilitada.")
                return {"ok": True, "habilitada": False}
            if self.link.physical:
                from guardian.core import Action
                act = Action("hardware.control", {"dispositivo": "cnc", "modo": "manual", "maquina": self.m.name,
                                                  "controlador": self.link.name, "sesion": self.id},
                             requested_by="hope.cnc.app", justification="habilitar la máquina para el operador")
                req = self._guard(act)
                if req:
                    self.pending = {"id": req, "tipo": "manual", "accion": act,
                                    "resumen": f"Habilitar «{self.m.name}» para jog, MDI y referenciado"}
                    return {"ok": True, "aprobar": self._pending_public()}
            self.enabled = True
            self.say("Máquina habilitada.")
            return {"ok": True, "habilitada": True}

    def _pending_public(self) -> dict | None:
        p = self.pending
        return None if p is None else {k: v for k, v in p.items() if k in ("id", "tipo", "resumen", "informe")}

    def approve(self, request_id: str) -> dict:
        """El operador aprueba en la pantalla la solicitud que creó el guardián."""
        with self.lock:
            p = self.pending
            if p is None or p["id"] != request_id:
                raise SessionError("no hay ninguna aprobación pendiente con ese número")
            self.guardian.approve(request_id, "operador (app Hope CNC)")
            d = self.guardian.decide(p["accion"], approval_id=request_id)
            self.pending = None
            if not d.allowed:
                raise SessionError("; ".join(d.reasons))
            if p["tipo"] == "manual":
                self.enabled = True
                self.say("Máquina habilitada (aprobado por el operador).")
                return {"ok": True, "habilitada": True}
            return self._launch(p["lineas"], p["hash"], p.get("mapa"))

    def cancel(self) -> dict:
        with self.lock:
            if self.pending and self.guardian is not None:
                self.guardian.deny(self.pending["id"], "operador (app Hope CNC)")
            self.pending = None
            return {"ok": True}

    # --- programa ---------------------------------------------------------------
    def load(self, name: str, text: str) -> dict:
        with self.lock:
            if self.link.snapshot()["trabajo"]["activo"]:
                raise SessionError("no se puede cargar otro programa con uno en marcha")
            m = self._machine_real()
            prog = interpret(text, m, block_delete=self.opts["borrado_bloque"],
                             optional_stop=self.opts["parada_opcional"])
            report = check_program(prog, m)
            report["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if self.profile_summary:
                report["perfil"] = self._profile_line()
            emitted = emit(prog, m, "grbl") if prog.ok else []
            # tiempo acumulado por línea enviada → tiempo restante en vivo
            cum = [0.0]
            if emitted:
                mc = as_controller(m)
                pl = plan(interpret("\n".join(t for t, _ in emitted), mc), mc)
                for i in range(1, len(emitted) + 1):
                    cum.append(cum[-1] + pl.time_by_line.get(i, 0.0))
                dwell_lines = {i + 1: float(t.split("P", 1)[1]) for i, (t, _) in enumerate(emitted)
                               if t.startswith("G4 P")}
                acc = 0.0
                for i in range(1, len(cum)):
                    acc += dwell_lines.get(i, 0.0)
                    cum[i] += acc
            pierce_idx = [i for i, (t, _) in enumerate(emitted) if t.startswith(("M3", "M4"))]
            self.program = {"nombre": name, "texto": text, "lineas": text.splitlines(), "informe": report,
                            "programa": prog, "emitidas": emitted, "acumulado": cum, "maquina": m,
                            "perforaciones": pierce_idx}
            self.started_at = None
            if self.next_line > len(self.program["lineas"]):
                self.next_line = 0
            self._remember(name, text, report["sha256"])
            self.say(f"Cargado «{name}»: " + report_text(report).splitlines()[1])
            return {"ok": True, "informe": report, "texto_informe": report_text(report)}

    def unload(self) -> dict:
        with self.lock:
            if self.link.snapshot()["trabajo"]["activo"]:
                raise SessionError("hay un programa en marcha")
            self.program = None
            return {"ok": True}

    def toolpath(self) -> dict:
        """Trayectoria en coordenadas G54 para la vista: [[x, y, z, tipo, línea], …] (tipo 0 rápido, 1 corte)."""
        p = self.program
        if p is None:
            return {"puntos": [], "limites": self._limits_work()}
        m = p["maquina"]
        g54 = m.offset("G54")
        pts, pierces = [], []
        on = False
        op_pos: dict[str, float] = {}
        for op in p["programa"].ops:
            if op.kind == "spindle":
                if op.value[0] != "off" and not on and m.is_plasma:
                    pierces.append([round(op_pos.get(a, 0.0) - g54.get(a, 0.0), 4) for a in "XYZ"])
                on = op.value[0] != "off"
                continue
            if op.kind not in ("rapid", "feed", "arc", "probe"):
                continue
            # 0 rápido, 1 corte (husillo/antorcha encendidos), 3 avance sin cortar (sensado, bajadas en vacío)
            kind = 0 if op.kind == "rapid" else 1 if on and op.kind != "probe" else 3
            if not pts or any(abs(pts[-1][i] - (op.start.get(a, 0.0) - g54.get(a, 0.0))) > 1e-9
                              for i, a in enumerate("XYZ")):
                pts.append([*(round(op.start.get(a, 0.0) - g54.get(a, 0.0), 4) for a in "XYZ"), -1, op.line])
            seq = linearize(op, max(m.arc_tolerance, 0.01)) if op.kind == "arc" else [op.end]
            for q in seq:
                pts.append([*(round(q.get(a, 0.0) - g54.get(a, 0.0), 4) for a in "XYZ"), kind, op.line])
            op_pos = op.end
            if len(pts) > MAX_PATH_POINTS:
                break
        return {"puntos": pts, "perforaciones": pierces, "recortado": len(pts) > MAX_PATH_POINTS,
                "limites": self._limits_work()}

    def _limits_work(self) -> dict:
        g54 = self.m.offset("G54")
        return {a: [None if ax.min is None else ax.min - g54[a], None if ax.max is None else ax.max - g54[a]]
                for a, ax in self.m.axes.items()}

    # --- ciclo ---------------------------------------------------------------------
    def cycle_start(self) -> dict:
        with self.lock:
            self._require_enabled("Cycle Start")
            st = self.link.snapshot()
            if st["estado"] == "Hold" or (self.link.single and st["trabajo"]["activo"]):
                self.link.resume()
                self.say("Reanudado." if not self.link.single else "Bloque siguiente.")
                return {"ok": True}
            if st["trabajo"]["activo"]:
                return {"ok": True}
            if self.program is None:
                raise SessionError("no hay programa cargado (Cargar G-code)")
            # revisar otra vez con los orígenes reales: el G54 pudo cambiar desde que se cargó
            r = self.load(self.program["nombre"], self.program["texto"])["informe"]
            if not r["runnable"]:
                raise SessionError("el programa tiene errores: " + "; ".join(r["errores"][:3]))
            lines, run_map = self._lines_to_run()
            partial = self.next_line > 0
            if self.link.physical:
                from guardian.core import Action
                params = {"dispositivo": "cnc", "maquina": self.m.name, "controlador": self.link.name,
                          "programa_sha256": r["sha256"], "envolvente_maquina": r["envolvente_maquina"]}
                if partial:
                    params.update(desde_linea=self.next_line, preparacion=self.from_here,
                                  lineas_sha256=hashlib.sha256("\n".join(lines).encode()).hexdigest())
                act = Action("hardware.control", params, requested_by="hope.cnc.app",
                             justification=f"ejecutar «{self.program['nombre']}» ({r['lineas']} líneas)"
                                           + (f" desde la línea {self.next_line}" if partial else ""))
                req = self._guard(act)
                if req:
                    self.pending = {"id": req, "tipo": "programa", "accion": act, "lineas": lines, "mapa": run_map,
                                    "hash": r["sha256"], "informe": r,
                                    "resumen": f"Ejecutar «{self.program['nombre']}»" +
                                               (f" desde la línea {self.next_line}" +
                                                (" (Run From Here: sube Z, va al punto y baja)" if self.from_here
                                                 else "") if partial else "")}
                    return {"ok": True, "aprobar": self._pending_public()}
            return self._launch(lines, r["sha256"], run_map)

    def _lines_to_run(self) -> tuple[list[str], list[int]]:
        """Programa a enviar según «Set Next Line» / «Run From Here» y el mapa a las líneas emitidas."""
        p = self.program
        emitted = p["emitidas"]
        texts = [t for t, _ in emitted]
        n = self.next_line
        if n <= 0:
            return texts, list(range(len(texts)))
        i = next((k for k in range(1, len(emitted)) if emitted[k][1] >= n), None)
        if i is None:
            raise SessionError(f"desde la línea {n} no queda nada que ejecutar")
        mc = as_controller(p["maquina"])

        def replay(upto: int) -> Interpreter:
            it = Interpreter(mc)
            it.run("\n".join(texts[:upto]))
            return it

        it = replay(i)
        if self.m.is_plasma and it.spindle[0] != "off":
            # a mitad de un corte: volver a la perforación de ese corte (no se puede seguir con el arco apagado)
            k = max(j for j in range(1, i) if texts[j].startswith(("M3", "M4")))
            while k - 1 >= 1 and emitted[k - 1][1] == emitted[k][1]:
                k -= 1
            i = k
            it = replay(i)
        if not self.from_here:
            return [HEADER, *texts[i:]], [i] + list(range(i, len(texts)))
        g54 = it.offsets["G54"]
        work = {a: it.pos[a] - g54[a] for a in it.axes}
        prep = [HEADER]
        zax = self.m.axes.get("Z")
        if zax is not None and zax.max is not None:
            prep.append(f"G53 G0 Z{zax.max:g}")
        prep.append("G0 " + " ".join(f"{a}{work[a]:.4f}" for a in it.axes if a != "Z"))
        if it.spindle[0] != "off" and not self.m.is_plasma:
            prep.append(("M3" if it.spindle[0] == "cw" else "M4") + (f" S{it.rpm:g}" if it.rpm else ""))
        for c in sorted(it.coolant):
            prep.append("M7" if c == "niebla" else "M8")
        if "Z" in work:
            plunge = min(it.feed or 300.0, zax.max_rate if zax else 300.0)
            prep.append(f"G1 Z{work['Z']:.4f} F{plunge:g}")
        if it.feed:
            prep.append(f"F{it.feed:g}")
        return prep + texts[i:], [i] * len(prep) + list(range(i, len(texts)))

    def _launch(self, lines: list[str], sha: str, run_map: list[int] | None = None) -> dict:
        if self.program is None or self.program["informe"]["sha256"] != sha:
            raise SessionError("el programa cambió: volvé a pulsar Cycle Start")
        self.link.start(lines)
        self.run_map = run_map
        self.started_at = time.monotonic()
        self.say(f"Cycle Start: «{self.program['nombre']}»" +
                 (f" desde la línea {self.next_line}." if self.next_line else "."))
        self.next_line, self.from_here = 0, False
        return {"ok": True, "en_marcha": True}

    # --- botones de Mach3: Single BLK, Block Delete, M1, Set Next Line, Run From Here -----
    def toggle(self, name: str) -> dict:
        if name not in self.opts:
            raise SessionError(f"opción desconocida: {name}")
        with self.lock:
            self.opts[name] = not self.opts[name]
            if name == "bloque_a_bloque":
                self.link.set_single(self.opts[name])
            elif self.program is not None and not self.link.snapshot()["trabajo"]["activo"]:
                self.load(self.program["nombre"], self.program["texto"])     # cambia lo que se ejecuta
            return {"ok": True, "opciones": dict(self.opts)}

    def set_next_line(self, line: int, from_here: bool = False) -> dict:
        with self.lock:
            if self.program is None:
                raise SessionError("no hay programa cargado")
            if self.link.snapshot()["trabajo"]["activo"]:
                raise SessionError("hay un programa en marcha")
            n = int(line)
            if not 0 <= n <= len(self.program["lineas"]):
                raise SessionError(f"línea fuera del programa (1 a {len(self.program['lineas'])})")
            self.next_line, self.from_here = n, bool(from_here) and n > 0
            self.link.clear_job()
            if n:
                self.say(f"{'Run From Here' if self.from_here else 'Set Next Line'}: el próximo Cycle Start "
                         f"empieza en la línea {n}.")
            return {"ok": True, "linea": n, "desde_aqui": self.from_here}

    # --- archivos recientes ---------------------------------------------------------------
    def _read_recent(self) -> list[dict]:
        if self.recent_dir is None:
            return []
        try:
            return json.loads((self.recent_dir / "recientes.json").read_text(encoding="utf-8"))[:10]
        except (OSError, ValueError):
            return []

    def _remember(self, name: str, text: str, sha: str) -> None:
        self.recent = [r for r in self.recent if r["sha"] != sha]
        self.recent.insert(0, {"nombre": name, "sha": sha, "fecha": time.time(), "lineas": text.count("\n") + 1})
        self.recent = self.recent[:10]
        keep = {r["sha"] for r in self.recent}
        if self.recent_dir is None:
            self._recent_mem[sha] = text
            self._recent_mem = {k: v for k, v in self._recent_mem.items() if k in keep}
            return
        try:
            self.recent_dir.mkdir(parents=True, exist_ok=True)
            (self.recent_dir / f"{sha[:16]}.nc").write_text(text, encoding="utf-8")
            (self.recent_dir / "recientes.json").write_text(json.dumps(self.recent, ensure_ascii=False), encoding="utf-8")
            for f in self.recent_dir.glob("*.nc"):
                if not any(f.stem == k[:16] for k in keep):
                    f.unlink()
        except OSError as e:
            self.say(f"no se pudo guardar en recientes: {e}")

    def recent_files(self) -> dict:
        return {"recientes": self.recent}

    def load_recent(self, sha: str) -> dict:
        r = next((x for x in self.recent if x["sha"] == sha), None)
        if r is None:
            raise SessionError("ese archivo ya no está en recientes")
        if self.recent_dir is None:
            text = self._recent_mem.get(sha)
        else:
            f = self.recent_dir / f"{sha[:16]}.nc"
            text = f.read_text(encoding="utf-8") if f.is_file() else None
        if text is None:
            raise SessionError("no se encontró el texto del archivo reciente")
        return self.load(r["nombre"], text)

    def rewind(self) -> dict:
        self.link.clear_job()
        self.started_at = None
        return {"ok": True}

    def feed_hold(self) -> dict:
        self.link.hold()
        self.say("Feed Hold.")
        return {"ok": True}

    def stop(self) -> dict:
        self.link.stop()
        self.say("Stop.")
        return {"ok": True}

    # --- manual ------------------------------------------------------------------
    def jog(self, axis: str, direction: int, step: float | None, fast: bool, percent: float = 30.0) -> dict:
        self._require_enabled("jog")
        axis = axis.upper()
        if axis not in self.m.axes:
            raise SessionError(f"eje {axis} no configurado")
        ax = self.m.axes[axis]
        feed = ax.max_rate * (1.0 if fast else max(1.0, min(100.0, percent)) / 100.0)
        pos = self.link.snapshot()["mpos"][axis]
        sign = 1 if direction > 0 else -1
        if step:
            d = sign * float(step)
        else:                                               # continuo: hasta el límite (se corta al soltar)
            lim = ax.max if sign > 0 else ax.min
            d = (lim - pos) if lim is not None else sign * 1000.0
            if sign * d <= 1e-6:
                raise SessionError(f"{axis} ya está en el límite")
        self.link.jog({axis: d}, feed)
        return {"ok": True}

    def jog_stop(self) -> dict:
        self.link.jog_cancel()
        return {"ok": True}

    def home(self) -> dict:
        self._require_enabled("referenciar")
        self.say("Referenciando todos los ejes…")
        threading.Thread(target=self._safe, args=(self.link.home, "referenciado"), daemon=True).start()
        return {"ok": True}

    def _safe(self, fn, what):
        try:
            fn()
            self.say(f"{what.capitalize()} terminado.")
        except BackendError as e:
            self.say(f"Error en {what}: {e}")

    def unlock(self) -> dict:
        self.link.unlock()
        return {"ok": True}

    def mdi(self, line: str) -> dict:
        self._require_enabled("MDI")
        line = line.strip()
        if not line:
            return {"ok": True}
        if len(line) > 200:
            raise SessionError("línea MDI demasiado larga")
        r = self._check_mdi(line)
        if r["errores"]:
            raise SessionError(r["errores"][0].replace("línea 1: ", ""))
        self.link.mdi(line)
        if not self.mdi_history or self.mdi_history[-1] != line:
            self.mdi_history.append(line)
        self.say(f"MDI: {line}")
        return {"ok": True}

    def _check_mdi(self, line: str) -> dict:
        """Revisa la línea con el estado modal de MDI (G91 sigue valiendo en la línea siguiente, como en
        el controlador) y con los orígenes reales; si hay error, el estado modal no cambia."""
        import copy
        try:
            real = self.link.offsets()
        except BackendError:
            real = {}
        it = copy.deepcopy(self.mdi_interp)
        for w, v in real.items():
            if w in it.offsets:
                it.offsets[w] = {a: float(v.get(a, 0.0)) for a in it.axes}
            elif w == "G92":
                it.g92 = {a: float(v.get(a, 0.0)) for a in it.axes}
        it.pos = dict(self.link.snapshot()["mpos"])
        it.ops, it.diags, it.ended, it.line = [], [], False, 0
        it.run(line)
        m = dataclasses.replace(it.m, offsets={w: dict(o) for w, o in it.offsets.items()})
        r = check_program(Program(it.ops, it.diags, 1, [], True), m)
        if not r["errores"]:
            self.mdi_interp = it
        return r

    def zero(self, axis: str) -> dict:
        axes = list(self.m.axis_names) if axis.lower() in ("todo", "all") else [axis.upper()]
        if any(a not in self.m.axes for a in axes):
            raise SessionError(f"eje {axis} no configurado")
        return self.mdi("G10 L20 P1 " + " ".join(f"{a}0" for a in axes))

    def goto_zero(self) -> dict:
        """Como «Goto Z» de Mach3 pero prudente: sube Z al máximo y va a X0 Y0 (no baja Z)."""
        self._require_enabled("ir a cero")
        zmax = self.m.axes["Z"].max if "Z" in self.m.axes else None
        if zmax is not None:
            self.mdi(f"G53 G0 Z{zmax:g}")
            self._wait_idle()
        return self.mdi("G90 G0 X0 Y0")

    def _wait_idle(self, timeout: float = 120.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.link.snapshot()["estado"] == "Idle":
                return
            time.sleep(0.05)

    def override(self, kind: str, action: str) -> dict:
        self.link.override(kind, action)
        return {"ok": True}

    def spindle(self, on: bool, rpm: float | None = None) -> dict:
        if on:
            rpm = rpm or self.m.spindle_max / 2
            return self.mdi(f"M3 S{rpm:g}")
        return self.mdi("M5")

    # --- plasma -------------------------------------------------------------------
    def _plasma(self):
        if not self.m.is_plasma or self.m.plasma is None:
            raise SessionError("la máquina no es de plasma (tipo: plasma en la configuración)")
        return self.m.plasma

    def _sequence(self, lines: list[str], what: str):
        def run():
            for ln in lines:
                self._wait_idle()
                self.mdi(ln)
            self._wait_idle()
        threading.Thread(target=self._safe, args=(run, what), daemon=True).start()

    def plasma_probe(self) -> dict:
        """Sensado de altura (IHS) en el lugar: baja hasta tocar la chapa y pone Z0 en la superficie."""
        pl = self._plasma()
        self._require_enabled("sensar altura")
        self.say("Sensando la altura de la chapa…")
        self._sequence(["G91", f"G38.2 Z-{pl.probe_depth:g} F{pl.probe_feed:g}", "G90",
                        f"G10 L20 P1 Z-{pl.float_offset:g}", f"G0 Z{pl.travel_height:g}"], "sensado")
        return {"ok": True}

    def test_fire(self) -> dict:
        """Disparo de prueba: antorcha 0,3 s en el lugar (para ver el arco y la presión de aire)."""
        self._plasma()
        self._require_enabled("disparo de prueba")
        if self.link.snapshot()["trabajo"]["activo"]:
            raise SessionError("hay un programa en marcha")
        self.say("Disparo de prueba.")
        self._sequence(["M3", "G4 P0.3", "M5"], "disparo de prueba")
        return {"ok": True}

    def coolant(self) -> dict:
        self._require_enabled("refrigerante")
        st = self.link.snapshot()
        if st["trabajo"]["activo"]:
            self.link.coolant_toggle()
            return {"ok": True}
        return self.mdi("M9" if st["refrigerante"] else "M8")

    # --- estado para la pantalla ----------------------------------------------------
    def state(self) -> dict:
        st = self.link.snapshot()
        wpos = {a: st["mpos"][a] - st["wco"].get(a, 0.0) for a in st["mpos"]}
        job = st["trabajo"]
        prog = self.program
        info = None
        if prog is not None:
            emitted, cum = prog["emitidas"], prog["acumulado"]
            ran = job["activo"] or job["terminado"] or job["error"]
            done = min(job["confirmadas"], len(emitted)) if ran else 0
            rm = self.run_map
            if ran and rm:
                done = len(emitted) if job["terminado"] else rm[min(job["confirmadas"], len(rm) - 1)]
            src = 0
            if ran and emitted:
                src = emitted[min(done, len(emitted) - 1)][1]
            total = cum[-1] if cum else 0.0
            elapsed = (time.monotonic() - self.started_at) if self.started_at and job["activo"] else None
            info = {"nombre": prog["nombre"], "lineas_total": len(prog["lineas"]), "linea_actual": src,
                    "tiempo_total": round(total, 1),
                    "tiempo_restante": round(max(0.0, total - cum[min(done, len(cum) - 1)]), 1) if cum else 0.0,
                    "transcurrido": None if elapsed is None else round(elapsed, 1),
                    "progreso": (cum[min(done, len(cum) - 1)] / total) if total else 0.0,
                    "runnable": prog["informe"]["runnable"], "errores": prog["informe"]["errores"][:50],
                    "avisos": prog["informe"]["avisos"][:50], "informe": report_text(prog["informe"]),
                    "sha256": prog["informe"]["sha256"]}
        plasma = None
        if self.m.is_plasma:
            idx = prog["perforaciones"] if prog else []
            ran = prog is not None and (job["activo"] or job["terminado"] or job["error"])
            k = job["confirmadas"]
            if ran and self.run_map:
                k = len(prog["emitidas"]) if job["terminado"] else self.run_map[min(k, len(self.run_map) - 1)]
            done_n = sum(1 for i in idx if i < k) if ran else 0
            plasma = {"antorcha": st.get("husillo_dir", "off") != "off", "perforaciones_total": len(idx),
                      "perforaciones_hechas": min(done_n, len(idx))}
        return {"maquina": self.m.name, "marca": BRAND, "tipo": self.m.kind, "plasma": plasma,
                "ejes": list(self.m.axis_names), "habilitada": self.enabled,
                "fisica": self.link.physical, "estado": st, "trabajo_pos": {a: round(v, 4) for a, v in wpos.items()},
                "programa": info, "pendiente": self._pending_public(),
                "mensajes": [{"t": t, "m": m} for t, m in list(self.log)[-40:]],
                "mdi_historial": list(self.mdi_history), "opciones": dict(self.opts),
                "linea_siguiente": self.next_line, "desde_aqui": self.from_here,
                "modal": self.link.modal() or modal_text(self.mdi_interp), "herramienta": self._tool_info(st),
                "perfiles": None if self.selection is None else {"seleccion": self.selection,
                                                                 "resumen": self.profile_summary}}

    def _tool_info(self, st: dict) -> dict:
        n = self.link.tool()
        if n is None:
            n = self.mdi_interp.tool
        t = self.m.tool(n) if n is not None else None
        return {"n": n or 0, "diametro": t.diameter if t else 0.0, "largo": t.length if t else 0.0,
                "nombre": t.name if t else "", "cambio": str(st.get("mensaje", "")).startswith("cambiar a la herramienta")}

    def program_lines(self) -> dict:
        p = self.program
        return {"nombre": p["nombre"], "lineas": p["lineas"]} if p else {"nombre": "", "lineas": []}

    # --- perfiles tipo slicer ------------------------------------------------------------
    def _profile_line(self) -> str:
        sel, s = self.selection, self.profile_summary
        names = " + ".join(x for x in (sel.get("material"), sel.get("consumible"), sel.get("proceso")) if x)
        return f"{names}: {s['texto']}"

    def _selection_file(self):
        d = getattr(self.profiles, "user_dir", None)
        return None if d is None else Path(d) / "seleccion.json"

    def choose_profile(self, kind: str | None = None, name: str | None = None, thickness: float | None = None) -> dict:
        """Elegir un perfil (como los desplegables de un slicer) o el espesor; recalcula todo."""
        from hope.cnc.profiles import KINDS, ProfileError, effective, fix_selection
        if self.profiles is None:
            raise SessionError("los perfiles no están activos")
        with self.lock:
            if self.link.snapshot()["trabajo"]["activo"]:
                raise SessionError("no se cambian perfiles con un programa en marcha")
            sel = dict(self.selection)
            if kind is not None:
                if kind not in KINDS:
                    raise SessionError(f"tipo de perfil desconocido: {kind}")
                if name not in self.profiles.names(kind):
                    raise SessionError(f"no existe el perfil «{name}»")
                sel[kind] = name
                if kind == "maquina":
                    sel = {"maquina": name, "espesor": sel.get("espesor")}
            if thickness is not None:
                sel["espesor"] = float(thickness)
            try:
                sel = fix_selection(self.profiles, sel)
                m, summary = effective(self.profiles, sel)
            except (ProfileError, ValueError) as e:
                raise SessionError(str(e)) from None
            self._apply_machine(m)
            self.selection, self.profile_summary = sel, summary
            f = self._selection_file()
            if f is not None:
                try:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps(sel, ensure_ascii=False), encoding="utf-8")
                except OSError as e:
                    self.say(f"no se pudo guardar la selección: {e}")
            if self.program is not None:
                self.load(self.program["nombre"], self.program["texto"])
            self.say(f"Perfiles: {sel['maquina']} · {self._profile_line()}")
            return {"ok": True, "seleccion": sel, "resumen": summary}

    def _apply_machine(self, m: Machine) -> None:
        changed_machine = m.name != self.m.name
        if m.controller != self.m.controller and self.link_factory is not None:
            if self.enabled:
                self.link.stop()
            self.link.close()
            self.link = self.link_factory(m)
            self.enabled = False
            self.say(f"Conexión nueva: {self.link.name}. Pulsá RESET para habilitar.")
        elif changed_machine and self.enabled:
            self.enabled = False                  # otra máquina: volver a habilitar a conciencia
            self.say("Máquina cambiada: pulsá RESET para habilitarla.")
        self.m = m
        if hasattr(self.link, "set_machine"):
            self.link.set_machine(m)
        self.mdi_interp = Interpreter(as_controller(m))

    def profile_list(self) -> dict:
        from hope.cnc.profiles import KIND_NAMES, KINDS, thickness_options
        if self.profiles is None:
            return {"activos": False}
        kind = self.m.kind
        out = {k: self.profiles.list(k, kind) for k in KINDS}
        out["maquina"] = self.profiles.list("maquina")
        esp = []
        if self.m.is_plasma and self.selection.get("material"):
            esp = thickness_options(self.profiles.resolve("material", self.selection["material"]),
                                    self.profiles.resolve("consumible", self.selection["consumible"])
                                    if self.selection.get("consumible") else None)
        return {"activos": True, "tipos": out, "nombres": KIND_NAMES, "seleccion": self.selection,
                "resumen": self.profile_summary, "espesores": esp}

    def profile_detail(self, kind: str, name: str) -> dict:
        from hope.cnc.profiles import SCHEMA, ProfileError
        if self.profiles is None:
            raise SessionError("los perfiles no están activos")
        try:
            raw, origin = self.profiles.raw(kind, name)
            data = self.profiles.resolve(kind, name)
            parent = self.profiles.resolve(kind, raw["hereda"]) if raw.get("hereda") else None
            own = self.profiles.own_keys(kind, name)
        except ProfileError as e:
            raise SessionError(str(e)) from None
        return {"tipo": kind, "nombre": name, "origen": origin, "hereda": raw.get("hereda"), "datos": data,
                "padre": parent, "propios": own, "esquema": SCHEMA[kind], "crudo": raw}

    def profile_save(self, kind: str, name: str, data: dict, inherits: str | None) -> dict:
        from hope.cnc.profiles import ProfileError
        if self.profiles is None:
            raise SessionError("los perfiles no están activos")
        try:
            out = self.profiles.save(kind, name, data, inherits)
        except (ProfileError, ValueError) as e:
            raise SessionError(str(e)) from None
        self.say(f"Perfil guardado: {name}")
        if self.selection and self.selection.get(kind) == name or kind == "maquina" and self.selection["maquina"] == name:
            self.choose_profile()                 # el perfil en uso cambió: recalcular
        return {"ok": True, "perfil": out}

    def profile_delete(self, kind: str, name: str) -> dict:
        from hope.cnc.profiles import ProfileError
        try:
            self.profiles.delete(kind, name)
        except (ProfileError, AttributeError) as e:
            raise SessionError(str(e)) from None
        if self.selection.get(kind) == name:
            self.choose_profile()
        return {"ok": True}

    def profile_import(self, kind: str, text: str) -> dict:
        from hope.cnc.profiles import ProfileError
        try:
            out = self.profiles.import_text(kind, text)
        except (ProfileError, ValueError, AttributeError) as e:
            raise SessionError(str(e)) from None
        return {"ok": True, "perfil": out}

    def config(self) -> dict:
        m = self.m
        return {"nombre": m.name, "tipo": m.kind, "archivo": self.machine_path,
                "ejes": {a: {"pasos_mm": ax.steps_per_mm, "vel_max": ax.max_rate, "acel": ax.accel, "min": ax.min,
                             "max": ax.max} for a, ax in m.axes.items()},
                "desviacion_union": m.junction_deviation, "tolerancia_arco": m.arc_tolerance,
                "husillo": {"rpm_min": m.spindle_min, "rpm_max": m.spindle_max},
                "herramientas": {n: {"largo": t.length, "diametro": t.diameter, "nombre": t.name}
                                 for n, t in sorted(m.tools.items())},
                "controlador": m.controller, "pasos_jog": JOG_STEPS, "marca": BRAND,
                "plasma": None if m.plasma is None or not m.is_plasma else {
                    "altura_perforacion": m.plasma.pierce_height, "altura_corte": m.plasma.cut_height,
                    "retardo_perforacion": m.plasma.pierce_delay, "altura_traslado": m.plasma.travel_height,
                    "avance_bajada": m.plasma.plunge_feed, "perforacion_auto": m.plasma.auto_pierce,
                    "sensado": m.plasma.probe, "holgura_flotante": m.plasma.float_offset}}

    def offsets_table(self) -> dict:
        try:
            real = self.link.offsets()
        except BackendError as e:
            return {"ok": False, "mensaje": str(e)}
        rows = {w: real.get(w, self.m.offset(w)) for w in WCS}
        if "G92" in real:
            rows["G92"] = real["G92"]
        return {"ok": True, "origenes": rows}

