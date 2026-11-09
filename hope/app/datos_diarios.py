"""Paso 8: datos diarios desde internet, por vías legales (``hope.data.web_sources``).

Corre ``python -m hope.data.web_sources --diario --repetir`` en segundo plano: cada día trae la
cuota configurada en ``configs/fuentes_web.yaml`` (dumps oficiales primero), la filtra, deduplica
y la suma al corpus con procedencia y licencia. Todo acceso pasa por el guardián
(``configs/web.yaml``): si el modo es «preguntar», las solicitudes aparecen en el paso 6.
"""

from __future__ import annotations

import json
from pathlib import Path

from hope.app.tareas import Tarea, correr_modulo


class TareaDatosDiarios(Tarea):
    nombre = "datos_diarios"

    def _registro(self):
        from hope.data.web_sources import Registry, table
        p = self.app.root / "configs" / "fuentes_web.yaml"
        if not p.exists():
            return None, []
        reg = Registry.load(p)
        return reg, table(reg)

    def resumen(self) -> dict:
        reg, rows = self._registro()
        out = {"fuentes": rows, "ultimo_informe": None}
        if reg is not None:
            root = self.app.root / reg.daily.get("carpeta", reg.daily.get("salida", "data/web"))
            informes = sorted(Path(root).glob("*/informe.json")) if root.exists() else []
            if informes:
                try:
                    out["ultimo_informe"] = json.loads(informes[-1].read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
            from hope.data.web_sources import daily_bytes
            out["cuota"] = {"gb_dia": round(daily_bytes(reg.daily) / 1e9, 2),
                            "docs_dia": reg.daily.get("paginas_por_dia")}
        return out

    def ejecutar(self, repetir: bool = True) -> None:
        self.detalle = "trayendo la cuota de hoy"
        args = ["--diario"] + (["--repetir"] if repetir else [])
        code = correr_modulo(self, "hope.data.web_sources", args, self.app.root)
        if code not in (0, None) and not self.detener_pedido():
            raise RuntimeError(f"web_sources terminó con código {code}")
        self.detalle = "pausado" if self.detener_pedido() else "cuota del día lista"
