"""Registro de topología y conteo de parámetros (ARQUITECTURA_HN1 §6.1, ASG §3).

El registro es la fuente de verdad sobre la estructura de Hope:

* Cada región, población, especialista, conexión u operador tiene un
  **identificador estable** que sobrevive a cambios de estructura y migraciones.
* Cada nodo se asocia a un submódulo de PyTorch por su ruta (``"regions.0.experts.2"``),
  de modo que el registro es serializable sin depender de objetos en memoria.
* El conteo de parámetros distingue **totales**, **activos esperados** por paso
  y **entrenables** (ASG §3.A–C). Estados, cachés y memoria externa no se cuentan.

El total global se calcula siempre desde el modelo real, no sumando nodos,
para detectar parámetros que nadie registró.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable

import torch.nn as nn

from hope.epistemic import LifecycleStatus


class NodeKind(str, Enum):
    MODEL = "MODEL"
    REGION = "REGION"
    POPULATION = "POPULATION"
    EXPERT = "EXPERT"
    CONNECTION = "CONNECTION"
    OPERATOR = "OPERATOR"
    HEAD = "HEAD"


@dataclass
class Node:
    id: str
    kind: NodeKind
    module_path: str | None          # ruta en el modelo; None si es puramente lógico
    parent: str | None = None
    # Fracción esperada de pasos en que el nodo está activo (1.0 = siempre).
    # Para un especialista con router top-k entre n: k / n.
    activation: float = 1.0
    status: LifecycleStatus = LifecycleStatus.ESTABLE
    protected: bool = False          # circuitos protegidos (tareas raras, ASG §5)
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ParamReport:
    total: int
    active_expected: float
    trainable: int
    unregistered: int                # parámetros que no pertenecen a ningún nodo hoja

    def as_dict(self) -> dict:
        return asdict(self)


class UnknownNode(KeyError):
    pass


class TopologyRegistry:
    """Registro de nodos estructurales con identificadores estables."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._counters: dict[str, int] = {}

    # --- identificadores -------------------------------------------------
    def new_id(self, kind: NodeKind, hint: str = "") -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", hint.lower()).strip("-") or kind.value.lower()
        prefix = f"{kind.value[:3].lower()}_{slug}"
        n = self._counters.get(prefix, 0)
        self._counters[prefix] = n + 1
        return f"{prefix}_{n}"

    # --- alta / baja ------------------------------------------------------
    def add(self, kind: NodeKind, module_path: str | None, *, parent: str | None = None,
            hint: str = "", node_id: str | None = None, **kwargs) -> Node:
        if parent is not None and parent not in self._nodes:
            raise UnknownNode(parent)
        nid = node_id or self.new_id(kind, hint or (module_path or ""))
        if nid in self._nodes:
            raise ValueError(f"id duplicado: {nid}")
        node = Node(id=nid, kind=kind, module_path=module_path, parent=parent, **kwargs)
        self._nodes[nid] = node
        return node

    def remove(self, node_id: str) -> Node:
        if any(n.parent == node_id for n in self._nodes.values()):
            raise ValueError(f"{node_id} tiene hijos; retirarlos primero")
        return self._nodes.pop(node_id)

    def get(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError as e:
            raise UnknownNode(node_id) from e

    def nodes(self, kind: NodeKind | None = None) -> list[Node]:
        return [n for n in self._nodes.values() if kind is None or n.kind == kind]

    def children(self, node_id: str) -> list[Node]:
        return [n for n in self._nodes.values() if n.parent == node_id]

    def find_by_path(self, module_path: str) -> Node | None:
        return next((n for n in self._nodes.values() if n.module_path == module_path), None)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    # --- conteo de parámetros ----------------------------------------------
    def _effective_activation(self, node: Node) -> float:
        a = node.activation
        p = node.parent
        while p is not None:
            parent = self._nodes[p]
            a *= parent.activation
            p = parent.parent
        return a

    def _leaf_owner(self, model: nn.Module) -> dict[int, Node]:
        """Asigna cada parámetro al nodo registrado con la ruta más específica."""
        bound = sorted((n for n in self._nodes.values() if n.module_path is not None),
                       key=lambda n: len(n.module_path), reverse=True)
        owner: dict[int, Node] = {}
        for name, p in model.named_parameters():
            for node in bound:
                path = node.module_path
                if path == "" or name == path or name.startswith(path + "."):
                    owner.setdefault(id(p), node)
                    break
        return owner

    def node_params(self, model: nn.Module, node_id: str) -> int:
        node = self.get(node_id)
        if node.module_path is None:
            return 0
        mod = model.get_submodule(node.module_path) if node.module_path else model
        return sum(p.numel() for p in mod.parameters())

    def report(self, model: nn.Module) -> ParamReport:
        owner = self._leaf_owner(model)
        seen: set[int] = set()
        total = trainable = unregistered = 0
        active = 0.0
        for _, p in model.named_parameters():
            if id(p) in seen:          # parámetros compartidos se cuentan una vez
                continue
            seen.add(id(p))
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
            node = owner.get(id(p))
            if node is None or node.kind == NodeKind.MODEL:
                unregistered += n
                active += n
            else:
                active += n * self._effective_activation(node)
        return ParamReport(total=total, active_expected=active, trainable=trainable,
                           unregistered=unregistered)

    # --- serialización -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "nodes": [
                {**asdict(n), "kind": n.kind.value, "status": n.status.value}
                for n in self._nodes.values()
            ],
            "counters": dict(self._counters),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TopologyRegistry":
        reg = cls()
        for raw in data["nodes"]:
            raw = dict(raw)
            raw["kind"] = NodeKind(raw["kind"])
            raw["status"] = LifecycleStatus(raw["status"])
            reg._nodes[raw["id"]] = Node(**raw)
        reg._counters = dict(data.get("counters", {}))
        return reg

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TopologyRegistry":
        return cls.from_dict(json.loads(text))

    def validate(self, model: nn.Module) -> list[str]:
        """Problemas de consistencia entre el registro y el modelo real."""
        problems = []
        names = {name for name, _ in model.named_modules()}
        for n in self._nodes.values():
            if n.module_path is not None and n.module_path not in names:
                problems.append(f"{n.id}: ruta inexistente '{n.module_path}'")
            if n.parent is not None and n.parent not in self._nodes:
                problems.append(f"{n.id}: padre inexistente '{n.parent}'")
        return problems


def ids(nodes: Iterable[Node]) -> list[str]:
    return [n.id for n in nodes]
