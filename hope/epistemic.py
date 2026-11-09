"""Estados epistémicos y de ciclo de vida compartidos (ARQUITECTURA_HN1 §9).

Todas las piezas de Hope (memoria, ideas del sueño, restricciones, modelos del
mundo, herramientas, arquitecturas) usan estas dos enumeraciones en lugar de
listas propias.

Regla central: un estado solo «sube» con evidencia registrada. Imaginar o
repetir algo no aumenta su confianza (SCA §2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EpistemicStatus(str, Enum):
    OBSERVADO = "OBSERVADO"                    # dato de una fuente, con procedencia
    INFERIDO = "INFERIDO"                      # derivado por razonamiento, sin verificar
    GENERADO_EN_SUENO = "GENERADO_EN_SUENO"    # producido por SCA
    SIMULADO = "SIMULADO"                      # resultado de un modelo interno
    HIPOTESIS = "HIPOTESIS"                    # propuesta pendiente de prueba
    PROBADO = "PROBADO"                        # sometido a pruebas, sin veredicto suficiente
    RESPALDADO = "RESPALDADO"                  # evidencia favorable, falta validación independiente
    VALIDADO_CONDICIONAL = "VALIDADO_CONDICIONAL"  # válido bajo condiciones registradas
    VALIDADO = "VALIDADO"                      # superó pruebas independientes en un dominio
    REFUTADO = "REFUTADO"                      # contradicho por evidencia
    INCONCLUSO = "INCONCLUSO"                  # información insuficiente


class LifecycleStatus(str, Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    EN_VALIDACION = "EN_VALIDACION"
    APROBADO_CON_LIMITACIONES = "APROBADO_CON_LIMITACIONES"
    ESTABLE = "ESTABLE"
    DEPRECADO = "DEPRECADO"
    BLOQUEADO = "BLOQUEADO"


# Nivel de respaldo: subir de nivel exige evidencia. Los estados sin respaldo
# propio (imaginado, simulado, inferido) comparten el nivel más bajo.
_SUPPORT_LEVEL = {
    EpistemicStatus.GENERADO_EN_SUENO: 0,
    EpistemicStatus.SIMULADO: 0,
    EpistemicStatus.INFERIDO: 0,
    EpistemicStatus.HIPOTESIS: 0,
    EpistemicStatus.INCONCLUSO: 0,
    EpistemicStatus.REFUTADO: 0,
    EpistemicStatus.PROBADO: 1,
    EpistemicStatus.OBSERVADO: 2,
    EpistemicStatus.RESPALDADO: 2,
    EpistemicStatus.VALIDADO_CONDICIONAL: 3,
    EpistemicStatus.VALIDADO: 4,
}

# Estados a los que solo se llega con verificación independiente.
_REQUIRES_INDEPENDENT = {EpistemicStatus.VALIDADO, EpistemicStatus.VALIDADO_CONDICIONAL}


class EvidenceRequired(ValueError):
    """Se intentó subir el estado epistémico sin la evidencia necesaria."""


@dataclass(frozen=True)
class Evidence:
    """Referencia a la evidencia que justifica un cambio de estado."""

    ref: str                      # id de experimento, prueba, fuente o registro PESAM
    independent: bool = False     # verificador independiente del que generó la idea
    description: str = ""


@dataclass
class EpistemicRecord:
    """Estado epistémico con historial de transiciones."""

    status: EpistemicStatus
    history: list[tuple[str, str, str]] = field(default_factory=list)  # (de, a, ref)

    def transition(self, new: EpistemicStatus, evidence: Evidence | None = None) -> None:
        old = self.status
        rising = _SUPPORT_LEVEL[new] > _SUPPORT_LEVEL[old]
        if rising and evidence is None:
            raise EvidenceRequired(f"{old.value} → {new.value} requiere evidencia")
        if new in _REQUIRES_INDEPENDENT and not (evidence and evidence.independent):
            raise EvidenceRequired(f"{new.value} requiere verificación independiente")
        self.history.append((old.value, new.value, evidence.ref if evidence else ""))
        self.status = new
