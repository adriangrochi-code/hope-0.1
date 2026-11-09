"""Conjuntos de evaluación sellados (ASG §8; AANS §10; ARQUITECTURA_HN1 §12.1).

* Cada conjunto se registra con el hash SHA-256 de su archivo. Si el archivo
  cambia, abrirlo falla: no se puede «retocar» la evaluación.
* Cada lectura queda registrada con su propósito. El propósito ``training`` se
  rechaza siempre.
* Si un conjunto se consulta demasiadas veces para decidir cambios, se marca
  como desgastado y conviene renovarlo (sobreoptimización del evaluador).
* ``contamination_report`` detecta solapamiento por n-gramas entre datos de
  entrenamiento y un conjunto sellado (HLA §12: la evaluación no debe repetir
  ejemplos de enseñanza).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ALLOWED_PURPOSES = {"evaluation", "contamination_check", "audit"}


class SealError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class SealedSet:
    name: str
    path: str
    sha256: str
    max_decision_reads: int


class SealedRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest = self.root / "manifest.json"
        self._access = self.root / "access.jsonl"
        if not self._manifest.exists():
            self._manifest.write_text("{}", encoding="utf-8")

    def _read(self) -> dict:
        return json.loads(self._manifest.read_text(encoding="utf-8"))

    def seal(self, name: str, path: str | Path, *, max_decision_reads: int = 20,
             description: str = "") -> SealedSet:
        data = self._read()
        if name in data:
            raise SealError(f"el conjunto '{name}' ya está sellado; para renovarlo usar otro nombre")
        digest = sha256_file(path)
        data[name] = {"path": str(Path(path).resolve()), "sha256": digest, "created": time.time(),
                      "max_decision_reads": max_decision_reads, "description": description}
        self._manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return SealedSet(name, data[name]["path"], digest, max_decision_reads)

    def open(self, name: str, *, purpose: str, requester: str, decision: bool = False) -> Path:
        """Devuelve la ruta del conjunto tras verificar integridad y registrar el acceso.

        ``decision=True`` indica que el resultado se usará para decidir un cambio
        (consolidar, revertir, elegir arquitectura). Esas lecturas desgastan el conjunto.
        """
        if purpose not in ALLOWED_PURPOSES:
            raise SealError(f"propósito no permitido para un conjunto sellado: '{purpose}'")
        data = self._read()
        if name not in data:
            raise SealError(f"conjunto inexistente: {name}")
        entry = data[name]
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise SealError(f"el conjunto '{name}' fue modificado después de sellarlo")
        with self._access.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "name": name, "purpose": purpose,
                                "requester": requester, "decision": decision}) + "\n")
        return Path(entry["path"])

    def accesses(self, name: str | None = None) -> list[dict]:
        if not self._access.exists():
            return []
        rows = [json.loads(line) for line in self._access.read_text(encoding="utf-8").splitlines() if line]
        return [r for r in rows if name is None or r["name"] == name]

    def is_worn(self, name: str) -> bool:
        """¿Se usó demasiadas veces para tomar decisiones? Entonces hay que renovarlo."""
        limit = self._read()[name]["max_decision_reads"]
        return sum(1 for r in self.accesses(name) if r["decision"]) >= limit


# --- contaminación -------------------------------------------------------------
def _ngrams(text: str, n: int) -> set[int]:
    words = text.lower().split()
    return {hash(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)}


def contamination_report(train_texts: Iterable[str], eval_texts: Iterable[str], n: int = 13) -> dict:
    """Fracción de ejemplos de evaluación que comparten algún n-grama con el entrenamiento.

    13 palabras es el umbral usual para detectar copias literales. Para textos
    cortos (problemas sintéticos) conviene bajar ``n``.
    """
    train: set[int] = set()
    for t in train_texts:
        train |= _ngrams(t, n)
    eval_list = list(eval_texts)
    flagged = [i for i, t in enumerate(eval_list) if _ngrams(t, n) & train]
    return {"n": n, "eval_items": len(eval_list), "contaminated": len(flagged),
            "fraction": len(flagged) / max(len(eval_list), 1), "indices": flagged}
