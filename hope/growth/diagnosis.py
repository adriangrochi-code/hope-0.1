"""Diagnóstico unificado de carencias (ARQUITECTURA_HN1 §7; ASG §4–5; Herramientas §12).

Antes de crecer hay que descartar causas más baratas. El orden de las reglas
sigue el principio de la §2.2: datos → herramienta → memoria → optimización /
capacidad. Optimización y capacidad se distinguen con una sonda: una copia con
más capacidad y un control, ambos con el mismo entrenamiento adicional. La expansión es una de varias soluciones, no la respuesta automática.

Esta versión (v0) es **un conjunto de reglas diseñadas a mano** sobre señales
medidas. El documento de ASG pide que Hope aprenda a diagnosticar con la
experiencia (§13); eso requiere muchos registros en PESAM y llega después.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from hope.epistemic import EpistemicStatus


class Cause(str, Enum):
    SIN_PROBLEMA = "SIN_PROBLEMA"
    DATOS_INSUFICIENTES = "DATOS_INSUFICIENTES"
    ETIQUETAS_ERRONEAS = "ETIQUETAS_ERRONEAS"
    OPTIMIZACION = "OPTIMIZACION"
    REPRESENTACION = "REPRESENTACION"
    CONOCIMIENTO_FALTANTE = "CONOCIMIENTO_FALTANTE"
    HERRAMIENTA_FALTANTE = "HERRAMIENTA_FALTANTE"
    ESTRATEGIA_RAZONAMIENTO = "ESTRATEGIA_RAZONAMIENTO"
    MEMORIA_RECUPERACION = "MEMORIA_RECUPERACION"
    CONEXIONES = "CONEXIONES"
    CAPACIDAD_INSUFICIENTE = "CAPACIDAD_INSUFICIENTE"
    INCONCLUSO = "INCONCLUSO"


# A qué subsistema se envía cada causa (ARQUITECTURA_HN1 §7.2).
ROUTE = {
    Cause.SIN_PROBLEMA: "ninguna acción",
    Cause.DATOS_INSUFICIENTES: "pipeline de datos",
    Cause.ETIQUETAS_ERRONEAS: "pipeline de datos (limpieza)",
    Cause.OPTIMIZACION: "seguir entrenando / ajustar optimizador",
    Cause.REPRESENTACION: "AANS (reorganizar)",
    Cause.CONOCIMIENTO_FALTANTE: "HLA + ATI",
    Cause.HERRAMIENTA_FALTANTE: "HTE (usar herramienta)",
    Cause.ESTRATEGIA_RAZONAMIENTO: "ARS / MRE",
    Cause.MEMORIA_RECUPERACION: "MDD / AHMC (recuperación)",
    Cause.CONEXIONES: "AANS (reconectar)",
    Cause.CAPACIDAD_INSUFICIENTE: "ASG (crecer dentro del techo)",
    Cause.INCONCLUSO: "recolectar más evidencia",
}


@dataclass
class Signals:
    """Señales medidas para una familia de tareas. ``None`` = no medida."""

    task: str
    region: str | None = None             # ID estable de la región que atiende la tarea
    train_acc: float | None = None
    val_acc: float | None = None
    loss_trend: float | None = None       # mejora relativa de la pérdida en la ventana final
    label_conflict_rate: float | None = None
    best_tool: str | None = None
    tool_fit: float | None = None         # exactitud de la mejor herramienta sobre los datos de entrenamiento
    store_fit: float | None = None        # exactitud de la búsqueda en memoria externa
    capacity_probe_gain: float | None = None  # mejora de una copia ensanchada vs. control
    optimization_gain: float | None = None    # mejora del control (mismo entrenamiento, sin cambios)
    evidence: list[str] = field(default_factory=list)


@dataclass
class Thresholds:
    target_val_acc: float = 0.90
    max_label_conflict: float = 0.05
    tool_fit: float = 0.99
    store_fit: float = 0.99
    min_probe_gain: float = 0.05


@dataclass
class Hypothesis:
    cause: Cause
    confidence: float
    test: str


@dataclass
class Diagnosis:
    """Registro de diagnóstico (esquema de ARQUITECTURA_HN1 §7.1)."""

    task: str
    region: str | None
    cause: Cause
    route: str
    hypotheses: list[Hypothesis]
    observed: dict
    alternatives_to_growth: list[str]
    needs_capacity_probe: bool = False
    status: EpistemicStatus = EpistemicStatus.INFERIDO

    def as_dict(self) -> dict:
        d = asdict(self)
        d["cause"] = self.cause.value
        d["status"] = self.status.value
        for h in d["hypotheses"]:
            h["cause"] = h["cause"].value
        return d


def diagnose(s: Signals, th: Thresholds = Thresholds()) -> Diagnosis:
    """Aplica las reglas en orden. Si llega a la capacidad sin sonda, la pide."""
    observed = {k: v for k, v in asdict(s).items() if v is not None and k not in ("task", "region")}
    alts: list[str] = []

    def result(cause: Cause, conf: float, test: str, probe: bool = False,
               status: EpistemicStatus = EpistemicStatus.RESPALDADO) -> Diagnosis:
        return Diagnosis(s.task, s.region, cause, ROUTE[cause], [Hypothesis(cause, conf, test)],
                         observed, alts, needs_capacity_probe=probe, status=status)

    if s.val_acc is not None and s.val_acc >= th.target_val_acc:
        return result(Cause.SIN_PROBLEMA, 0.9, "val_acc ≥ objetivo")

    if s.label_conflict_rate is not None and s.label_conflict_rate > th.max_label_conflict:
        alts.append("limpiar o reetiquetar datos")
        return result(Cause.ETIQUETAS_ERRONEAS, min(1.0, s.label_conflict_rate * 4),
                      "entradas idénticas con etiquetas distintas")

    if s.tool_fit is not None and s.tool_fit >= th.tool_fit:
        alts.append(f"usar la herramienta '{s.best_tool}'")
        return result(Cause.HERRAMIENTA_FALTANTE, s.tool_fit,
                      f"la herramienta '{s.best_tool}' reproduce las etiquetas")

    if s.store_fit is not None and s.store_fit >= th.store_fit:
        alts.append("recuperar de la memoria externa")
        return result(Cause.MEMORIA_RECUPERACION, s.store_fit,
                      "la memoria externa contiene la respuesta")

    # Optimización y capacidad se separan con la misma sonda: dos copias entrenadas
    # igual, una con más capacidad. La pendiente de la pérdida es demasiado ruidosa.
    if s.capacity_probe_gain is None:
        return result(Cause.INCONCLUSO, 0.0, "falta la sonda de capacidad", probe=True,
                      status=EpistemicStatus.HIPOTESIS)

    opt = s.optimization_gain or 0.0
    if opt >= th.min_probe_gain:
        alts.append("entrenar más antes de modificar la estructura")
    if s.capacity_probe_gain >= th.min_probe_gain:
        d = result(Cause.CAPACIDAD_INSUFICIENTE, min(1.0, 0.5 + s.capacity_probe_gain),
                   "una copia con más capacidad aprende más que el control con el mismo entrenamiento")
        if opt >= th.min_probe_gain:
            d.hypotheses.append(Hypothesis(Cause.OPTIMIZACION, min(1.0, 0.5 + opt),
                                           "el control también mejora entrenando más"))
        return d
    if opt >= th.min_probe_gain:
        return result(Cause.OPTIMIZACION, min(1.0, 0.5 + opt),
                      "el control mejora entrenando más; la capacidad extra no agrega")

    return result(Cause.INCONCLUSO, 0.3, "ninguna regla explica el fallo",
                  status=EpistemicStatus.INCONCLUSO)
