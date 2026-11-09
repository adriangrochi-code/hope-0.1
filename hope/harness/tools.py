"""Herramientas de carpeta del harness F9: leer, listar, buscar, editar y ejecutar.

Hope trabaja sobre **una carpeta local fija** (no un repositorio remoto). Toda acción pasa
por el guardián (``guardian.core.Guardian.decide``), cuyo directorio de trabajo es esa carpeta:

* ``file.read`` (leer, listar, buscar) — dentro de la carpeta: permitido; fuera: aprobación humana.
* ``file.write`` (aplicar una edición) — dentro: permitido; fuera: aprobación humana;
  rutas protegidas (configuración, guardián, ``.git`` de la carpeta): **denegado** siempre.
* ``tool.execute`` (comandos y pruebas) — solo en el sandbox (``hope/tools/sandbox.py``) y
  sobre una **copia** de la carpeta: un comando nunca modifica los archivos originales.

Las ediciones no se escriben de inmediato: ``propose_edit`` devuelve un diff revisable y
``apply_edit`` lo escribe después de que la persona lo acepta (y si el archivo no cambió
mientras tanto). Si el guardián deja una acción pendiente de aprobación, se lanza
``NeedsApproval`` con el id de la solicitud; tras aprobarla, la misma llamada con
``approval_id`` se ejecuta una sola vez.

Cada acción (permitida, pendiente o denegada) queda en PESAM además de la auditoría del guardián.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from guardian.core import Action, Guardian
from hope.tools.code_tools import diff as unified_diff, search_code
from hope.tools.sandbox import IGNORE_IN_COPY, RunResult, run_command

MAX_READ = 200_000          # bytes por lectura
MAX_LIST = 300              # entradas por listado


class HarnessDenied(PermissionError):
    """El guardián rechazó la acción (rutas protegidas, acciones prohibidas)."""


class NeedsApproval(PermissionError):
    """La acción queda pendiente de aprobación humana; ``request_id`` la identifica."""

    def __init__(self, request_id: str, reasons, action: str, path: str = ""):
        self.request_id, self.reasons, self.action, self.path = request_id, tuple(reasons), action, path
        super().__init__(f"{action} {path}: {'; '.join(self.reasons)} (solicitud {request_id})")


@dataclass
class Edit:
    path: str                 # ruta absoluta resuelta
    rel: str                  # ruta como se pidió (para mostrar)
    old: str | None           # contenido actual (None = archivo nuevo)
    new: str
    diff: str
    source: str = "hope"
    old_hash: str = ""


@dataclass
class FolderTools:
    root: Path
    guardian: Guardian
    pesam: object | None = None                  # hope.memory.pesam.ExperienceLog
    requested_by: str = "hope"
    timeout_s: float = 60.0
    actions: list[dict] = field(default_factory=list)
    # Solicitudes pendientes (id → huella de la acción) y aprobaciones otorgadas sin usar (huella → id).
    # Una aprobación vale para **exactamente** esa acción y una sola vez (lo comprueba el guardián).
    requests: dict[str, str] = field(default_factory=dict)
    grants: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.root = Path(self.root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"la carpeta de trabajo no existe: {self.root}")
        if Path(self.guardian.config.workdir).resolve() != self.root:
            raise ValueError("el directorio de trabajo del guardián debe ser la carpeta del harness")

    # --- permisos -------------------------------------------------------------------
    def resolve(self, path: str | os.PathLike) -> Path:
        p = Path(os.path.expanduser(str(path)))
        return (p if p.is_absolute() else self.root / p).resolve()

    def inside(self, path: str | os.PathLike) -> bool:
        p = self.resolve(path)
        return p == self.root or self.root in p.parents

    def _record(self, kind: str, params: dict, outcome: str, reasons=()) -> None:
        row = {"accion": kind, **params, "resultado": outcome, "motivos": list(reasons)}
        self.actions.append(row)
        if self.pesam is not None:
            self.pesam.record("harness.accion", **row)

    def grant(self, request_id: str) -> bool:
        """Lo llama la sesión **después** de que una persona aprobó la solicitud en el guardián:
        la próxima vez que se pida exactamente esa acción, se usa la aprobación."""
        fp = self.requests.pop(request_id, None)
        if fp is None:
            return False
        self.grants[fp] = request_id
        return True

    def _check(self, kind: str, params: dict, approval_id: str | None = None) -> None:
        action = Action(kind, params, requested_by=self.requested_by)
        fp = action.fingerprint()
        if approval_id is None and fp in self.grants:
            approval_id = self.grants.pop(fp)
        d = self.guardian.decide(action, approval_id=approval_id)
        if d.allowed:
            self._record(kind, params, "permitido", d.reasons)
            return
        if d.pending_request:
            self.requests[d.pending_request] = fp
            self._record(kind, params, "pendiente", d.reasons)
            raise NeedsApproval(d.pending_request, d.reasons, kind, str(params.get("path", "")))
        self._record(kind, params, "denegado", d.reasons)
        raise HarnessDenied("; ".join(d.reasons) or f"{kind} denegado")

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root)) if (p == self.root or self.root in p.parents) else str(p)

    # --- leer -----------------------------------------------------------------------
    def read(self, path: str, *, approval_id: str | None = None, max_bytes: int = MAX_READ) -> str:
        p = self.resolve(path)
        self._check("file.read", {"path": str(p)}, approval_id)
        data = p.read_bytes()
        text = data[:max_bytes].decode("utf-8", errors="replace")
        return text + (f"\n… (truncado: {len(data)} bytes)" if len(data) > max_bytes else "")

    def list(self, path: str = ".", *, approval_id: str | None = None, recursive: bool = False) -> list[str]:
        p = self.resolve(path)
        self._check("file.read", {"path": str(p), "op": "listar"}, approval_id)
        out: list[str] = []
        if recursive:
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_IN_COPY)
                for f in sorted(filenames):
                    out.append(str(Path(dirpath, f).relative_to(p)))
                    if len(out) >= MAX_LIST:
                        return out
            return out
        for e in sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name)):
            out.append(e.name + ("/" if e.is_dir() else ""))
            if len(out) >= MAX_LIST:
                break
        return out

    def search(self, pattern: str, path: str = ".", *, approval_id: str | None = None,
               max_results: int = 30) -> list[dict]:
        """Busca con ripgrep si está instalado; si no, con ``re`` en Python (mismo formato de salida)."""
        p = self.resolve(path)
        self._check("file.read", {"path": str(p), "op": "buscar", "patron": pattern}, approval_id)
        if shutil.which("rg"):
            hits = search_code(pattern, str(p), max_results=max_results)
            for h in hits:
                h["path"] = self.rel(Path(h["path"]).resolve())
            return hits
        rx, hits = re.compile(pattern), []
        files = [p] if p.is_file() else [Path(d, f) for d, ds, fs in os.walk(p)
                                         for f in fs if not set(Path(d).parts) & set(IGNORE_IN_COPY)]
        for f in sorted(files):
            try:
                lines = f.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(lines, 1):
                if rx.search(line):
                    hits.append({"path": self.rel(f.resolve()), "line": i, "text": line.rstrip()[:200]})
                    if len(hits) >= max_results:
                        return hits
        return hits

    # --- editar (diff revisable) ---------------------------------------------------------
    def propose_edit(self, path: str, new_text: str, *, source: str = "hope",
                     approval_id: str | None = None) -> Edit:
        """Prepara la edición y su diff. No escribe nada (leer el original sí pasa por el guardián)."""
        p = self.resolve(path)
        old = None
        if p.exists():
            old = self.read(str(p), approval_id=approval_id, max_bytes=10 * MAX_READ)
        d = unified_diff(old or "", new_text, name=self.rel(p))
        h = hashlib.sha256((old or "").encode()).hexdigest()
        return Edit(str(p), self.rel(p), old, new_text, d, source, h)

    def apply_edit(self, edit: Edit, *, approval_id: str | None = None) -> None:
        p = Path(edit.path)
        current = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
        if hashlib.sha256((current or "").encode()).hexdigest() != edit.old_hash:
            raise RuntimeError(f"{edit.rel} cambió desde que se propuso la edición; volvé a pedirla")
        self._check("file.write", {"path": str(p), "sha256_nuevo": hashlib.sha256(edit.new.encode()).hexdigest()},
                    approval_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(edit.new, encoding="utf-8")

    def propose_search_replace(self, text: str, *, source: str = "hope",
                               approval_id: str | None = None) -> list[Edit]:
        """Ediciones a partir de bloques buscar/reemplazar (``hope/tools/diff_edits.py``). Cada texto
        buscado tiene que aparecer una sola vez; si no, ``EditError`` con el motivo. No escribe nada."""
        from hope.tools.diff_edits import apply_blocks, parse_blocks

        def read(rel: str):
            p = self.resolve(rel)
            return self.read(str(p), approval_id=approval_id, max_bytes=10 * MAX_READ) if p.exists() else None

        new = apply_blocks(parse_blocks(text), read)
        return [self.propose_edit(rel, content, source=source, approval_id=approval_id) for rel, content in new.items()]

    def apply_edits_checked(self, edits: list[Edit], test_command: str | list[str] | None, *,
                            approval_id: str | None = None, timeout_s: float | None = None) -> dict:
        """Escribe las ediciones, corre las pruebas (sandbox, sobre una copia de la carpeta ya editada) y
        **vuelve atrás** si fallan: restaura el contenido anterior (o borra los archivos nuevos). Las
        escrituras y la vuelta atrás pasan por el guardián."""
        done: list[Edit] = []
        try:
            for e in edits:
                self.apply_edit(e, approval_id=approval_id)
                done.append(e)
        except Exception:
            self._rollback(done)
            raise
        if not test_command:
            return {"aplicado": True, "revertido": False, "pruebas": None, "archivos": [e.rel for e in edits]}
        r = self.run(test_command, timeout_s=timeout_s)
        tests = {"ok": r.ok, "codigo": r.returncode, "salida": (r.stdout + r.stderr)[-2000:], "tiempo_agotado": r.timed_out}
        if r.ok:
            return {"aplicado": True, "revertido": False, "pruebas": tests, "archivos": [e.rel for e in edits]}
        self._rollback(done)
        return {"aplicado": False, "revertido": True, "pruebas": tests, "archivos": [e.rel for e in edits]}

    def _rollback(self, done: list[Edit]) -> None:
        for e in reversed(done):
            p = Path(e.path)
            if e.old is None:
                self._check("file.write", {"path": str(p), "op": "deshacer (borrar archivo nuevo)"})
                p.unlink(missing_ok=True)
            else:
                self._check("file.write", {"path": str(p), "op": "deshacer",
                                           "sha256_nuevo": hashlib.sha256(e.old.encode()).hexdigest()})
                p.write_text(e.old, encoding="utf-8")
            self._record("file.write", {"path": str(p)}, "revertido")

    # --- ejecutar (sandbox, sobre una copia) ------------------------------------------------
    def run(self, command: str | list[str], *, overlay: dict[str, str] | None = None,
            approval_id: str | None = None, timeout_s: float | None = None) -> RunResult:
        """Ejecuta en el sandbox sobre una copia de la carpeta (con ``overlay`` escrito encima).

        ``overlay`` son ediciones candidatas (ruta relativa → contenido) para probarlas sin tocar
        los archivos reales. Las rutas que salen de la carpeta se ignoran."""
        argv = shlex.split(command) if isinstance(command, str) else list(command)
        if not argv:
            raise ValueError("comando vacío")
        files = {}
        for rel, content in (overlay or {}).items():
            if self.inside(rel):
                files[self.rel(self.resolve(rel))] = content
        self._check("tool.execute", {"tool": "harness.ejecutar", "argv": argv, "sandboxed": True,
                                     "copia_de": str(self.root), "overlay": sorted(files)}, approval_id)
        return run_command(argv, files=files, copy_tree=str(self.root), timeout_s=timeout_s or self.timeout_s)
