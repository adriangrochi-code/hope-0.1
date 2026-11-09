"""Guardián de seguridad H-SVF (Carta Fundamental; ARQUITECTURA_HN1 §8).

Decide si una acción propuesta por Hope puede ejecutarse. Reglas:

* **Lo que no está autorizado se rechaza.** Una acción de tipo desconocido se
  deniega (Carta art. 5: «la ausencia de autorización no es permiso»).
* **Tres niveles** (Carta art. 5, HED §13):
  1. autonomía interna — se permite y se registra;
  2. autonomía supervisada — se permite dentro de permisos y límites;
  3. autorización humana obligatoria — queda pendiente hasta que una persona la apruebe.
* **Invariantes (HSI)** que ninguna política puede desactivar: techo de
  parámetros, modos de crecimiento, rutas protegidas, prohibición de que Hope
  modifique sus propios límites o la configuración de seguridad.
* **Todo** queda en el registro de auditoría encadenado.

Despliegue: en F0 el guardián es una biblioteca dentro del mismo proceso. El
diseño final lo ejecuta como **proceso separado**, dueño de su configuración y
de su registro, sin permisos de escritura para el proceso del modelo. La
interfaz (``decide`` / ``approve``) no cambia.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path

from guardian.audit import AuditLog
from hope.growth.budget import (GrowthBudget, ProposedChange, check_change,
                                needs_human_approval)


class Level(IntEnum):
    INTERNAL = 1
    SUPERVISED = 2
    HUMAN = 3


@dataclass(frozen=True)
class Action:
    kind: str
    params: dict = field(default_factory=dict)
    requested_by: str = "hope"
    justification: str = ""

    def fingerprint(self) -> str:
        payload = json.dumps({"kind": self.kind, "params": self.params}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class Decision:
    allowed: bool
    level: Level | None
    reasons: tuple[str, ...]
    pending_request: str | None = None     # id de solicitud si espera aprobación humana

    @property
    def needs_human(self) -> bool:
        return self.pending_request is not None


# Tipos de acción conocidos y su nivel base.
BASE_LEVELS: dict[str, Level] = {
    "teacher.query": Level.INTERNAL,              # ATI → Qwen
    "memory.write": Level.INTERNAL,
    "experiment.train": Level.SUPERVISED,         # entrenar una copia experimental
    "structural.experiment": Level.SUPERVISED,    # cambio estructural en rama experimental
    "structural.consolidate": Level.SUPERVISED,   # sube a HUMAN si el cambio es importante
    "tool.execute": Level.SUPERVISED,             # solo en sandbox; fuera → HUMAN
    "tool.consolidate": Level.HUMAN,              # herramienta creada (TCG) → catálogo estable
    "file.write": Level.SUPERVISED,               # fuera del directorio de trabajo → HUMAN
    "file.read": Level.INTERNAL,                  # fuera del directorio de trabajo → HUMAN
    "network.access": Level.HUMAN,
    "hardware.control": Level.HUMAN,
    "external.publish": Level.HUMAN,
}

# Acciones que Hope nunca puede ejecutar por iniciativa propia (Carta art. 6).
FORBIDDEN = {
    "budget.change": "Hope no puede modificar sus límites de crecimiento (ASG §14)",
    "safety.modify": "Hope no puede modificar la configuración de seguridad (Carta art. 6)",
    "audit.modify": "el registro de auditoría es de solo agregado",
    "network.write": "Hope solo puede leer en la red: nada de enviar, publicar, iniciar sesión ni subir (configs/web.yaml)",
}


@dataclass
class GuardianConfig:
    workdir: str
    protected_paths: list[str]
    budget_path: str
    audit_path: str
    approvals_path: str
    web_path: str | None = None          # permisos de red del usuario (configs/web.yaml)


class Guardian:
    def __init__(self, config: GuardianConfig):
        self.config = config
        self.audit = AuditLog(config.audit_path)
        self._approvals_path = Path(config.approvals_path)
        if not self._approvals_path.exists():
            self._approvals_path.parent.mkdir(parents=True, exist_ok=True)
            self._approvals_path.write_text("{}", encoding="utf-8")

    # --- configuración -----------------------------------------------------
    def budget(self) -> GrowthBudget:
        """Se relee en cada decisión: el presupuesto vive en un archivo del usuario."""
        return GrowthBudget.from_yaml(Path(self.config.budget_path).read_text(encoding="utf-8"))

    # --- red (configs/web.yaml) ------------------------------------------------
    @property
    def _web_usage_path(self) -> Path:
        return self._approvals_path.parent / "web_usage.json"

    def web_config(self):
        """Se relee en cada decisión: los permisos de red son del usuario, no de Hope."""
        from guardian.web import WebPermissions
        if not self.config.web_path or not Path(self.config.web_path).exists():
            return WebPermissions()                       # sin archivo: modo «preguntar», sin lista
        return WebPermissions.from_yaml(Path(self.config.web_path).read_text(encoding="utf-8"))

    def _web_usage(self) -> dict:
        p = self._web_usage_path
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        now = time.time()
        data["accesos"] = [t for t in data.get("accesos", []) if now - t < 3600]
        data["bytes"] = [b for b in data.get("bytes", []) if now - b[0] < 86400]
        return data

    def _save_web_usage(self, data: dict) -> None:
        self._web_usage_path.parent.mkdir(parents=True, exist_ok=True)
        self._web_usage_path.write_text(json.dumps(data), encoding="utf-8")

    def record_network_bytes(self, host: str, nbytes: int) -> None:
        """La herramienta informa lo descargado; cuenta para el límite diario de bytes."""
        data = self._web_usage()
        data["bytes"].append([time.time(), int(nbytes)])
        self._save_web_usage(data)
        self.audit.append("network.bytes", host=host, bytes=int(nbytes))

    def network_usage(self) -> dict:
        data, cfg = self._web_usage(), self.web_config()
        used = sum(b for _, b in data["bytes"])
        return {"consultas_ultima_hora": len(data["accesos"]), "max_consultas_por_hora": cfg.max_queries_per_hour,
                "bytes_ultimo_dia": used, "max_bytes_por_dia": cfg.max_bytes_per_day,
                "bytes_restantes": max(0, cfg.max_bytes_per_day - used)}

    def find_approval(self, action: Action) -> str | None:
        """Una aprobación humana sin usar para exactamente esta acción (para reintentar después)."""
        fp = action.fingerprint()
        for rid, req in self._read_approvals().items():
            if req["status"] == "APPROVED" and req["fingerprint"] == fp:
                return rid
        return None

    def _evaluate_network(self, action: Action) -> Decision:
        from guardian.web import READ_METHODS
        cfg = self.web_config()
        host = str(action.params.get("host", "")).lower()
        method = str(action.params.get("metodo", "GET")).upper()
        if method not in READ_METHODS:          # invariante: solo lectura, en cualquier modo
            return Decision(False, None, (f"método {method} prohibido: el acceso a la red es solo de lectura "
                                          f"(GET/HEAD)",))
        if action.params.get("cuerpo") or action.params.get("credenciales") or action.params.get("cookies"):
            return Decision(False, None, ("enviar datos, credenciales o cookies está prohibido (solo lectura)",))
        if cfg.is_forbidden_host(host):
            return Decision(False, None, (f"{host} está en la lista «prohibido» de configs/web.yaml",))
        if cfg.mode == "apagado":
            return Decision(False, None, ("acceso a la red apagado por el usuario (configs/web.yaml)",))
        if not host:
            return Decision(False, None, ("acceso a la red sin dominio declarado",))
        if cfg.is_blocked(host):
            return Decision(False, None, (f"dominio bloqueado por el usuario: {host}",))
        usage = self.network_usage()
        if usage["consultas_ultima_hora"] >= cfg.max_queries_per_hour:
            return Decision(False, None, (f"límite de {cfg.max_queries_per_hour} consultas por hora alcanzado",))
        if usage["bytes_restantes"] <= 0:
            return Decision(False, None, (f"límite de {cfg.max_bytes_per_day} bytes por día alcanzado",))
        if cfg.mode == "lista_permitida" and cfg.is_allowed(host):
            return Decision(True, Level.HUMAN, (f"aprobación humana delegada por configs/web.yaml "
                                                f"(modo lista_permitida, dominio {host})",))
        why = ("dominio fuera de la lista permitida" if cfg.mode == "lista_permitida"
               else "modo «preguntar»: cada acceso a la red requiere aprobación humana")
        return Decision(False, Level.HUMAN, (why,))

    def _is_protected(self, path: str) -> bool:
        p = Path(path).resolve()
        guarded = [Path(x).resolve() for x in self.config.protected_paths]
        guarded += [Path(self.config.budget_path).resolve(), Path(self.config.audit_path).resolve(),
                    Path(self.config.approvals_path).resolve(), self._web_usage_path.resolve()]
        if self.config.web_path:
            guarded.append(Path(self.config.web_path).resolve())
        return any(p == g or g in p.parents for g in guarded)

    def _inside_workdir(self, path: str) -> bool:
        p, w = Path(path).resolve(), Path(self.config.workdir).resolve()
        return p == w or w in p.parents

    # --- decisión -----------------------------------------------------------
    def decide(self, action: Action, *, approval_id: str | None = None) -> Decision:
        decision = self._evaluate(action)
        if decision.allowed or decision.level != Level.HUMAN:
            if decision.allowed and action.kind == "network.access":
                self._count_network_access()
                self._log(action, decision, delegated_by=self.config.web_path)
                return decision
            self._log(action, decision)
            return decision

        # Nivel 3: ¿hay una aprobación humana válida para exactamente esta acción?
        if approval_id and self._consume_approval(approval_id, action):
            decision = Decision(True, Level.HUMAN, decision.reasons + ("aprobado por humano",))
            if action.kind == "network.access":
                self._count_network_access()
            self._log(action, decision, approval_id=approval_id)
            return decision

        req = self._create_request(action, decision.reasons)
        decision = Decision(False, Level.HUMAN, decision.reasons, pending_request=req)
        self._log(action, decision)
        return decision

    def _evaluate(self, action: Action) -> Decision:
        if action.kind in FORBIDDEN:
            return Decision(False, None, (FORBIDDEN[action.kind],))
        if action.kind not in BASE_LEVELS:
            return Decision(False, None, (f"acción desconocida '{action.kind}': sin autorización explícita",))

        if action.kind == "network.access":
            return self._evaluate_network(action)
        if action.kind == "external.publish" and str(action.requested_by).startswith(("web", "hope.web")):
            return Decision(False, None, ("la herramienta web es de solo lectura: no publica nada",))

        level = BASE_LEVELS[action.kind]
        p = action.params
        reasons: list[str] = []

        if action.kind in ("structural.experiment", "structural.consolidate"):
            budget = self.budget()
            change = ProposedChange(**p["change"])
            check = check_change(
                budget, current_total=p["current_total"], current_active=p.get("current_active", 0.0),
                current_regions=p.get("current_regions", 0),
                current_connections=p.get("current_connections", 0),
                change=change, experimental=action.kind == "structural.experiment")
            if not check.allowed:
                return Decision(False, None, check.reasons)    # invariante: nunca se aprueba
            if action.kind == "structural.consolidate" and needs_human_approval(budget, check):
                level = Level.HUMAN
                reasons.append("consolidación que requiere aprobación humana "
                               f"(modo {budget.mode.value}, importante={check.important})")

        elif action.kind == "file.write":
            path = p.get("path", "")
            if self._is_protected(path):
                return Decision(False, None, (f"ruta protegida: {path}",))
            if not self._inside_workdir(path):
                level = Level.HUMAN
                reasons.append("escritura fuera del directorio de trabajo")
            else:
                level = Level.INTERNAL

        elif action.kind == "file.read":
            if not self._inside_workdir(p.get("path", "")):
                level = Level.HUMAN
                reasons.append("lectura fuera del directorio de trabajo")

        elif action.kind == "teacher.query":
            if not p.get("local", True):
                level = Level.HUMAN
                reasons.append("el maestro es remoto: la consulta sale a la red")

        elif action.kind == "tool.execute":
            if not p.get("sandboxed", False):
                level = Level.HUMAN
                reasons.append("ejecución fuera del sandbox")

        if level == Level.HUMAN:
            return Decision(False, level, tuple(reasons) or ("requiere autorización humana",))
        return Decision(True, level, tuple(reasons))

    # --- aprobaciones humanas ------------------------------------------------
    def _read_approvals(self) -> dict:
        return json.loads(self._approvals_path.read_text(encoding="utf-8"))

    def _write_approvals(self, data: dict) -> None:
        self._approvals_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _count_network_access(self) -> None:
        data = self._web_usage()
        data["accesos"].append(time.time())
        self._save_web_usage(data)

    def _create_request(self, action: Action, reasons) -> str:
        data = self._read_approvals()
        if action.kind == "network.access":             # no repetir la misma pregunta al usuario
            fp = action.fingerprint()
            for rid, req in data.items():
                if req["status"] == "PENDING" and req["fingerprint"] == fp:
                    return rid
        rid = uuid.uuid4().hex[:12]
        data[rid] = {"action": asdict(action), "fingerprint": action.fingerprint(),
                     "reasons": list(reasons), "status": "PENDING", "created": time.time()}
        self._write_approvals(data)
        self.audit.append("approval.requested", request=rid, kind=action.kind)
        return rid

    def pending(self) -> dict:
        return {k: v for k, v in self._read_approvals().items() if v["status"] == "PENDING"}

    def approve(self, request_id: str, approver: str, note: str = "") -> None:
        """Llamado por una persona (CLI), nunca por el modelo."""
        self._resolve(request_id, "APPROVED", approver, note)

    def deny(self, request_id: str, approver: str, note: str = "") -> None:
        self._resolve(request_id, "DENIED", approver, note)

    def _resolve(self, request_id: str, status: str, approver: str, note: str) -> None:
        data = self._read_approvals()
        if request_id not in data or data[request_id]["status"] != "PENDING":
            raise KeyError(f"solicitud inexistente o ya resuelta: {request_id}")
        data[request_id].update(status=status, approver=approver, note=note, resolved=time.time())
        self._write_approvals(data)
        self.audit.append(f"approval.{status.lower()}", request=request_id, approver=approver, note=note)

    def _consume_approval(self, request_id: str, action: Action) -> bool:
        data = self._read_approvals()
        req = data.get(request_id)
        if not req or req["status"] != "APPROVED" or req["fingerprint"] != action.fingerprint():
            return False
        req["status"] = "USED"
        self._write_approvals(data)
        return True

    # --- auditoría ----------------------------------------------------------
    def _log(self, action: Action, decision: Decision, **extra) -> None:
        self.audit.append("decision", action=asdict(action), allowed=decision.allowed,
                          level=int(decision.level) if decision.level else None,
                          reasons=list(decision.reasons), pending=decision.pending_request, **extra)
