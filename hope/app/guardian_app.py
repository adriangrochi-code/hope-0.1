"""Paso 6: el guardián desde la interfaz (mismas funciones que ``python -m guardian.cli``)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from guardian.cli import human, parse_count, set_limit, status
from guardian.core import Guardian, GuardianConfig


def cargar(root: Path) -> Guardian:
    p = root / "configs" / "guardian.yaml"
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))

    def abs_(x: str) -> str:
        return str(x) if Path(x).is_absolute() else str(root / x)
    raw["workdir"] = abs_(raw["workdir"])
    raw["protected_paths"] = [abs_(x) for x in raw.get("protected_paths", [])]
    for k in ("budget_path", "audit_path", "approvals_path"):
        raw[k] = abs_(raw[k])
    return Guardian(GuardianConfig(**raw))


def resumen(g: Guardian) -> dict:
    ok, idx = g.audit.verify()
    return {"estado": status(g), "pendientes": g.pending(), "auditoria_integra": ok, "auditoria_alterada_en": idx,
            "techo": g.budget().max_total_params, "techo_humano": human(g.budget().max_total_params)}


def resolver(app, request_id: str, aprobar: bool, nota: str = "") -> dict:
    by = (app.cfg.data.get("usuario") or "").strip()
    if not by:
        return {"ok": False, "mensaje": "escribí tu nombre (arriba a la derecha): queda en la auditoría"}
    g = app.guardian()
    try:
        (g.approve if aprobar else g.deny)(request_id, by, nota or "desde la aplicación")
    except KeyError as e:
        return {"ok": False, "mensaje": str(e)}
    out = {"ok": True, "mensaje": f"{'aprobada' if aprobar else 'denegada'} {request_id}"}
    if aprobar:
        # Si la pidió el programa de entrenamiento (crecer), se anota en estado.json para que siga.
        for salida in ("runs/hope", "runs/hope_a770"):
            p = app.cfg.ruta(salida) / "estado.json"
            if not p.exists():
                continue
            st = json.loads(p.read_text(encoding="utf-8"))
            for nombre, es in st.get("etapas", {}).items():
                if es.get("solicitud") == request_id:
                    es["aprobacion"] = request_id
                    p.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
                    out["mensaje"] += f"; la etapa {nombre} sigue al reanudar el entrenamiento"
    return out


def subir_techo(app, valor: str, confirmacion: str, modo: str | None = None) -> dict:
    by = (app.cfg.data.get("usuario") or "").strip()
    if not by:
        return {"ok": False, "mensaje": "escribí tu nombre (arriba a la derecha): queda en la auditoría"}
    try:
        n = parse_count(valor)
    except ValueError as e:
        return {"ok": False, "mensaje": str(e)}
    esperado = f"AUTORIZO {human(n)}"
    if (confirmacion or "").strip().upper() != esperado.upper():
        return {"ok": False, "mensaje": f"para confirmar escribí exactamente: {esperado}", "confirmacion": esperado}
    g = app.guardian()
    avisos = set_limit(g, n, by=by, mode=modo or None, note="desde la aplicación")
    return {"ok": True, "mensaje": f"techo: {human(n)}", "avisos": avisos, "estado": status(g)}
