"""Permisos de acceso a la red que fija el usuario (``configs/web.yaml``; ARQUITECTURA_HN1 §8).

La Carta exige aprobación humana para la red (acción ``network.access``, nivel 3). El usuario
decide **de antemano** cómo se da esa aprobación:

* ``preguntar`` (por defecto): cada acceso queda como solicitud pendiente hasta que una persona
  la apruebe con ``python -m guardian.cli approve <id> --by <nombre>``;
* ``lista_permitida``: para los dominios de la lista, la aprobación queda **delegada** en este
  archivo del usuario (cada acceso se registra igual en la auditoría); fuera de la lista, pendiente;
* ``apagado``: ningún acceso; ni siquiera se crean solicitudes.

Además: máximo de consultas por hora y de bytes por día (invariantes: al llegar al límite se
deniega, no se pregunta) y dominios bloqueados. Este archivo es **propiedad del usuario**: está
dentro de una ruta protegida y Hope no puede escribirlo (``file.write`` → denegado).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

MODES = ("preguntar", "lista_permitida", "apagado")


def _match(host: str, domain: str) -> bool:
    host, domain = host.lower().rstrip("."), domain.lower().strip().lstrip(".")
    return host == domain or host.endswith("." + domain)


@dataclass
class WebPermissions:
    mode: str = "preguntar"
    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=list)
    max_queries_per_hour: int = 30
    max_bytes_per_day: int = 20_000_000
    raw: dict = field(default_factory=dict)          # el resto (backends, caché, tiempos) lo usa la herramienta

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"modo de red desconocido «{self.mode}»; opciones: {', '.join(MODES)}")

    @classmethod
    def from_yaml(cls, text: str) -> "WebPermissions":
        d = yaml.safe_load(text) or {}
        return cls(mode=str(d.get("modo", "preguntar")),
                   allowed_domains=[str(x) for x in d.get("dominios_permitidos") or []],
                   blocked_domains=[str(x) for x in d.get("dominios_bloqueados") or []],
                   max_queries_per_hour=int(d.get("max_consultas_por_hora", 30)),
                   max_bytes_per_day=int(d.get("max_bytes_por_dia", 20_000_000)), raw=d)

    def is_allowed(self, host: str) -> bool:
        return not self.is_blocked(host) and any(_match(host, d) for d in self.allowed_domains)

    def is_blocked(self, host: str) -> bool:
        return any(_match(host, d) for d in self.blocked_domains)
