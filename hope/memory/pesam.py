"""PESAM v0 — registro persistente de experiencias (glosario §1).

Registro de solo agregado en JSONL. Cada experiencia lleva su tipo, su estado
epistémico y los datos que la describen. Es la fuente común para el
aprendizaje del propio crecimiento (ASG §13), el análisis retrospectivo (RRA)
y el metaaprendizaje (MLE).

v0 es deliberadamente simple. La versión completa se integra con MDD y AHMC.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from hope.epistemic import EpistemicStatus


class ExperienceLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def record(self, kind: str, status: EpistemicStatus = EpistemicStatus.OBSERVADO, **data) -> str:
        eid = f"exp_{uuid.uuid4().hex[:12]}"
        row = {"id": eid, "ts": time.time(), "kind": kind, "status": status.value, "data": data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return eid

    def query(self, kind: str | None = None) -> list[dict]:
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]
        return [r for r in rows if kind is None or r["kind"] == kind]
