"""Versiones, linaje y vuelta atrás (ASG §2.3 y §7; ARQUITECTURA_HN1 §6.1).

Cada versión guarda el modelo completo, su registro de topología y los
metadatos del cambio (diagnóstico, operación, métricas, decisión). Las ramas
apuntan a una versión:

* ``stable`` — la arquitectura consolidada; solo avanza con ``promote``.
* ``experimental/<nombre>`` — copias donde ASG/AANS prueban cambios.

Nada se borra: volver atrás mueve el puntero de la rama y queda registrado.

Formato en disco::

    <root>/branches.json
    <root>/events.jsonl                  # registro de solo agregado
    <root>/versions/<vid>/model.pt       # módulo completo (la estructura cambia entre versiones)
    <root>/versions/<vid>/registry.json
    <root>/versions/<vid>/meta.json

Nota: ``model.pt`` guarda el módulo serializado con pickle para poder
restaurar estructuras distintas. Solo deben cargarse versiones creadas por
este mismo sistema.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from hope.registry.topology import TopologyRegistry

STABLE = "stable"
EXPERIMENTAL_PREFIX = "experimental/"


class LineageError(RuntimeError):
    pass


@dataclass
class Version:
    id: str
    parent: str | None
    branch: str
    message: str
    created: float
    meta: dict


class LineageStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "versions").mkdir(parents=True, exist_ok=True)
        self._branches_path = self.root / "branches.json"
        self._events_path = self.root / "events.jsonl"
        if not self._branches_path.exists():
            self._write_branches({})

    # --- ramas -------------------------------------------------------------
    def _read_branches(self) -> dict[str, str | None]:
        return json.loads(self._branches_path.read_text(encoding="utf-8"))

    def _write_branches(self, data: dict) -> None:
        tmp = self._branches_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._branches_path)

    def branches(self) -> dict[str, str | None]:
        return self._read_branches()

    def head(self, branch: str) -> str | None:
        b = self._read_branches()
        if branch not in b:
            raise LineageError(f"rama inexistente: {branch}")
        return b[branch]

    def create_branch(self, name: str, from_version: str | None) -> None:
        b = self._read_branches()
        if name in b:
            raise LineageError(f"la rama {name} ya existe")
        if from_version is not None:
            self._version_dir(from_version)
        b[name] = from_version
        self._write_branches(b)
        self._event("create_branch", branch=name, version=from_version)

    def experiment(self, name: str, from_branch: str = STABLE) -> str:
        """Crea una rama experimental a partir de la cabeza de otra rama."""
        branch = EXPERIMENTAL_PREFIX + name
        self.create_branch(branch, self.head(from_branch))
        return branch

    # --- versiones -----------------------------------------------------------
    def commit(self, model: nn.Module, registry: TopologyRegistry, *, branch: str,
               message: str, meta: dict | None = None) -> str:
        b = self._read_branches()
        if branch not in b:
            if branch == STABLE and not b:
                b[branch] = None            # primera versión de todo el almacén
            else:
                raise LineageError(f"rama inexistente: {branch}")
        if branch == STABLE and b[branch] is not None:
            raise LineageError("no se commitea directo en 'stable': usar una rama experimental y promote()")
        parent = b[branch]

        buf = io.BytesIO()
        torch.save(model, buf)
        blob = buf.getvalue()
        reg_json = registry.to_json()
        meta = dict(meta or {})
        h = hashlib.sha256()
        for part in (parent or "", branch, message, reg_json, json.dumps(meta, sort_keys=True)):
            h.update(part.encode())
        h.update(hashlib.sha256(blob).digest())
        h.update(str(time.time_ns()).encode())
        vid = h.hexdigest()[:16]

        vdir = self.root / "versions" / vid
        vdir.mkdir(parents=True)
        (vdir / "model.pt").write_bytes(blob)
        (vdir / "registry.json").write_text(reg_json, encoding="utf-8")
        info = {"id": vid, "parent": parent, "branch": branch, "message": message,
                "created": time.time(), "meta": meta,
                "model_sha256": hashlib.sha256(blob).hexdigest()}
        (vdir / "meta.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

        b[branch] = vid
        self._write_branches(b)
        self._event("commit", branch=branch, version=vid, parent=parent, message=message)
        return vid

    def _version_dir(self, vid: str) -> Path:
        d = self.root / "versions" / vid
        if not d.is_dir():
            raise LineageError(f"versión inexistente: {vid}")
        return d

    def info(self, vid: str) -> Version:
        raw = json.loads((self._version_dir(vid) / "meta.json").read_text(encoding="utf-8"))
        return Version(raw["id"], raw["parent"], raw["branch"], raw["message"], raw["created"], raw["meta"])

    def load(self, vid: str, map_location: str | torch.device = "cpu") -> tuple[nn.Module, TopologyRegistry]:
        d = self._version_dir(vid)
        raw = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        blob = (d / "model.pt").read_bytes()
        if hashlib.sha256(blob).hexdigest() != raw["model_sha256"]:
            raise LineageError(f"la versión {vid} está corrupta (hash distinto)")
        model = torch.load(io.BytesIO(blob), map_location=map_location, weights_only=False)
        registry = TopologyRegistry.from_json((d / "registry.json").read_text(encoding="utf-8"))
        return model, registry

    def history(self, branch: str) -> list[Version]:
        out, vid = [], self.head(branch)
        while vid is not None:
            v = self.info(vid)
            out.append(v)
            vid = v.parent
        return out

    def _ancestors(self, vid: str | None) -> set[str]:
        seen = set()
        while vid is not None:
            seen.add(vid)
            vid = self.info(vid).parent
        return seen

    # --- vuelta atrás y consolidación ---------------------------------------
    def rollback(self, branch: str, to_version: str, reason: str) -> None:
        """Mueve la cabeza de una rama a una versión anterior de su propio linaje."""
        if to_version not in self._ancestors(self.head(branch)):
            raise LineageError(f"{to_version} no pertenece al linaje de {branch}")
        b = self._read_branches()
        old = b[branch]
        b[branch] = to_version
        self._write_branches(b)
        self._event("rollback", branch=branch, version=to_version, previous=old, reason=reason)

    def promote(self, src_branch: str, *, approval_ref: str, decision: dict | None = None,
                dst_branch: str = STABLE) -> str:
        """Consolida la cabeza de una rama experimental en ``stable``.

        ``approval_ref`` identifica la decisión que lo autorizó (aprobación humana
        o evaluación automática permitida por el guardián). Es obligatorio.
        La nueva cabeza de ``stable`` debe descender de la actual: no se pisa
        trabajo consolidado.
        """
        if not approval_ref:
            raise LineageError("promote requiere approval_ref")
        vid = self.head(src_branch)
        if vid is None:
            raise LineageError(f"la rama {src_branch} no tiene versiones")
        b = self._read_branches()
        current = b.get(dst_branch)
        if current is not None and current not in self._ancestors(vid):
            raise LineageError(f"{src_branch} no desciende de la cabeza actual de {dst_branch}")
        b[dst_branch] = vid
        self._write_branches(b)
        self._event("promote", src=src_branch, dst=dst_branch, version=vid, previous=current,
                    approval_ref=approval_ref, decision=decision or {})
        return vid

    # --- eventos -----------------------------------------------------------
    def _event(self, kind: str, **data) -> None:
        rec = {"ts": time.time(), "event": kind, **data}
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def events(self) -> list[dict]:
        if not self._events_path.exists():
            return []
        return [json.loads(line) for line in self._events_path.read_text(encoding="utf-8").splitlines() if line]
