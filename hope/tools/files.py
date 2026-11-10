"""Acceso a archivos de las herramientas, siempre a través del guardián (docs/HERRAMIENTAS_HN1.md §8).

* Leer (``file.read``): dentro de la carpeta de trabajo del guardián, permitido; fuera, queda
  pendiente de aprobación humana. Tras ``python -m guardian.cli approve <id>``, repetir la misma
  llamada usa esa aprobación una sola vez.
* Escribir (``file.write``): solo los resultados de las herramientas (gráficos, recortes), en
  ``<carpeta>/runs/herramientas/``; las rutas protegidas se deniegan siempre.
* Límites de tamaño por lectura (``MAX_FILE``): un archivo enorme no se carga entero.

Sin guardián (pruebas unitarias, generadores sin permisos) la carpeta de trabajo es el
directorio actual y no se pide permiso.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from guardian.core import Action
from hope.tools.catalog import ToolDenied

MAX_FILE = 64 << 20          # 64 MB por archivo leído por una herramienta


@dataclass
class ToolFiles:
    guardian: object | None = None
    requested_by: str = "hte"

    @property
    def root(self) -> Path:
        return Path(self.guardian.config.workdir if self.guardian is not None else os.getcwd()).resolve()

    @property
    def out_dir(self) -> Path:
        return self.root / "runs" / "herramientas"

    def resolve(self, path: str | os.PathLike) -> Path:
        p = Path(os.path.expanduser(str(path).strip().strip('"').strip("'")))
        return (p if p.is_absolute() else self.root / p).resolve()

    def _check(self, kind: str, path: Path) -> None:
        if self.guardian is None:
            return
        action = Action(kind, {"path": str(path)}, requested_by=self.requested_by)
        approved = self.guardian.approved_request(action)
        if approved is None:                       # no repetir la misma pregunta a la persona
            fp = action.fingerprint()
            same = next((rid for rid, req in self.guardian.pending().items() if req["fingerprint"] == fp), None)
            if same is not None:
                raise ToolDenied(f"{kind} {path}: pendiente de aprobación humana (solicitud {same}; "
                                 "aprobar y repetir la llamada)")
        d = self.guardian.decide(action, approval_id=approved)
        if d.allowed:
            return
        if d.pending_request:
            raise ToolDenied(f"{'; '.join(d.reasons)}: pendiente de aprobación humana "
                             f"(solicitud {d.pending_request}; aprobar y repetir la llamada)")
        raise ToolDenied("; ".join(d.reasons))

    def check_read(self, path: str | os.PathLike) -> Path:
        """Ruta absoluta ya autorizada para leer (para pasarla a un proceso del sandbox)."""
        p = self.resolve(path)
        self._check("file.read", p)
        if not p.exists():
            raise FileNotFoundError(f"no existe: {path}")
        if p.is_file() and p.stat().st_size > MAX_FILE:
            raise ValueError(f"archivo demasiado grande ({p.stat().st_size >> 20} MB > {MAX_FILE >> 20} MB)")
        return p

    def read_bytes(self, path: str | os.PathLike, offset: int = 0, length: int | None = None) -> bytes:
        p = self.check_read(path)
        if not p.is_file():
            raise IsADirectoryError(f"no es un archivo: {path}")
        with open(p, "rb") as f:
            f.seek(max(0, offset))
            return f.read(MAX_FILE if length is None else min(length, MAX_FILE))

    def read_text(self, path: str | os.PathLike, max_bytes: int = 4 << 20) -> str:
        return self.read_bytes(path, 0, max_bytes).decode("utf-8", errors="replace")

    def write_output(self, kind: str, data: bytes, suffix: str = ".png") -> Path:
        """Guarda un resultado (gráfico, recorte) con nombre por contenido; devuelve la ruta."""
        p = self.out_dir / kind / (hashlib.sha256(data).hexdigest()[:16] + suffix)
        self._check("file.write", p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)


def looks_like_path(arg: str, exts: tuple[str, ...] = ()) -> bool:
    """¿El argumento es una ruta (y no contenido pegado en la llamada)?"""
    a = arg.strip()
    if not a or "\n" in a or len(a) > 400:
        return False
    if exts and a.lower().endswith(exts):
        return True
    return "/" in a or "\\" in a or os.path.splitext(a)[1] != "" and " " not in a
