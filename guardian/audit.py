"""Registro de auditoría encadenado por hash (HSA; Carta art. 6.5 y art. 12).

Cada entrada incluye el hash de la anterior. Modificar, borrar o reordenar
entradas rompe la cadena y ``verify()`` lo detecta. No impide que alguien con
acceso al archivo lo reescriba entero; para eso, en producción el archivo
pertenece al usuario del guardián y el proceso del modelo no tiene permiso
de escritura. Opcionalmente se puede anclar el último hash fuera de la máquina.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

GENESIS = "0" * 64


def _digest(prev: str, body: dict) -> str:
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256((prev + payload).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _last_hash(self) -> str:
        last = GENESIS
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)["hash"]
        return last

    def append(self, event: str, **data) -> str:
        prev = self._last_hash()
        body = {"ts": time.time(), "event": event, "data": data, "prev": prev}
        h = _digest(prev, body)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**body, "hash": h}, ensure_ascii=False) + "\n")
        return h

    def entries(self) -> list[dict]:
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def verify(self) -> tuple[bool, int | None]:
        """Devuelve (íntegro, índice de la primera entrada inválida)."""
        prev = GENESIS
        for i, e in enumerate(self.entries()):
            body = {k: v for k, v in e.items() if k != "hash"}
            if e.get("prev") != prev or _digest(prev, body) != e.get("hash"):
                return False, i
            prev = e["hash"]
        return True, None
