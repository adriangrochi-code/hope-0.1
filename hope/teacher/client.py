"""ATI — interacción con el maestro (glosario §5; Herramientas §5; HLA §11–14).

El maestro es intercambiable. La interfaz es una sola: ``ask(messages) -> str``.

* ``OpenAICompatibleTeacher`` — cualquier servidor con la API de chat de OpenAI:
  llama.cpp (``llama-server``), Ollama, OpenVINO Model Server, vLLM o una API
  remota. Cambiar a Qwen 4 = cambiar ``model`` (y el endpoint) en
  ``configs/teacher.yaml``.
* ``MockTeacher`` — maestro simulado y determinista para pruebas: enseña
  aritmética y unidades con episodios bien formados y puede inyectar errores
  a propósito para comprobar que la verificación los atrapa.

Toda consulta pasa por el guardián: un maestro local es acción de nivel 1; uno
remoto implica salir a la red y requiere aprobación humana (Carta art. 5).

Sobre el razonamiento (decisión del usuario): Qwen explica **todo** el proceso,
pero como explicación didáctica estructurada (episodio). Los bloques de
pensamiento crudo (``<think>…</think>``) se separan y no se usan para entrenar
salvo que ``keep_raw_thinking`` esté activado: no están verificados paso a paso.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import yaml

from guardian.core import Action, Guardian

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class TeacherDenied(PermissionError):
    pass


@dataclass
class TeacherReply:
    text: str                       # respuesta sin el pensamiento crudo
    raw_thinking: str | None = None
    model: str = ""
    usage: dict = field(default_factory=dict)


@dataclass
class TeacherConfig:
    endpoint: str = "http://127.0.0.1:8080/v1"
    model: str = "qwen"
    api_key: str | None = None
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout_s: float = 300.0
    keep_raw_thinking: bool = False
    no_think: bool = False          # Qwen3: agrega «/no_think» al último mensaje (respuesta directa, más rápida en CPU)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TeacherConfig":
        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @property
    def is_local(self) -> bool:
        return (urlparse(self.endpoint).hostname or "") in LOCAL_HOSTS


def split_thinking(text: str) -> tuple[str, str | None]:
    thoughts = THINK_RE.findall(text)
    return THINK_RE.sub("", text).strip(), ("\n".join(t.strip() for t in thoughts) or None)


class Teacher:
    """Interfaz común. Las subclases implementan ``_complete``."""

    def __init__(self, config: TeacherConfig, guardian: Guardian | None = None):
        self.config = config
        self.guardian = guardian
        self.calls = 0

    def ask(self, messages: list[dict], *, purpose: str = "") -> TeacherReply:
        if self.guardian is not None:
            d = self.guardian.decide(Action("teacher.query", {
                "endpoint": self.config.endpoint, "model": self.config.model,
                "local": self.config.is_local, "purpose": purpose}, requested_by="ati"))
            if not d.allowed:
                msg = "; ".join(d.reasons) or "consulta al maestro denegada"
                if d.pending_request:
                    msg += f" (solicitud pendiente de aprobación: {d.pending_request})"
                raise TeacherDenied(msg)
        self.calls += 1
        text, usage = self._complete(messages)
        clean, thinking = split_thinking(text)
        return TeacherReply(clean, thinking if self.config.keep_raw_thinking else None,
                            self.config.model, usage)

    def _complete(self, messages: list[dict]) -> tuple[str, dict]:
        raise NotImplementedError


class OpenAICompatibleTeacher(Teacher):
    def _complete(self, messages: list[dict]) -> tuple[str, dict]:
        if self.config.no_think and messages and messages[-1].get("role") == "user":
            messages = messages[:-1] + [{**messages[-1], "content": messages[-1]["content"] + "\n/no_think"}]
        body = json.dumps({"model": self.config.model, "messages": messages,
                           "temperature": self.config.temperature,
                           "max_tokens": self.config.max_tokens}).encode()
        req = urllib.request.Request(self.config.endpoint.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        if self.config.api_key:
            req.add_header("Authorization", f"Bearer {self.config.api_key}")
        with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"], data.get("usage", {})


def load_teacher(path: str | Path = "configs/teacher.yaml", guardian: Guardian | None = None) -> Teacher:
    cfg = TeacherConfig.from_yaml(path)
    if cfg.model == "mock":
        return MockTeacher(cfg, guardian)
    return OpenAICompatibleTeacher(cfg, guardian)


# ---------------------------------------------------------------------------
class MockTeacher(Teacher):
    """Maestro simulado para pruebas. Entiende pedidos de episodios sobre temas
    simples y responde con JSON válido. ``error_rate`` > 0 introduce un resultado
    intermedio equivocado en una fracción de los episodios."""

    def __init__(self, config: TeacherConfig | None = None, guardian: Guardian | None = None,
                 error_rate: float = 0.0, seed: int = 0):
        super().__init__(config or TeacherConfig(endpoint="http://127.0.0.1/mock", model="mock"), guardian)
        import random
        self.error_rate = error_rate
        self.rnd = random.Random(seed)

    def _complete(self, messages: list[dict]) -> tuple[str, dict]:
        prompt = messages[-1]["content"]
        m = re.search(r"TEMA:\s*(.+)", prompt)
        topic = (m.group(1).strip() if m else "suma").lower()
        a, b = self.rnd.randint(10, 99), self.rnd.randint(10, 99)
        wrong = self.rnd.random() < self.error_rate
        if "potencia" in topic or "ohm" in topic:
            r, i = self.rnd.choice([100, 220, 330, 470]), self.rnd.choice([0.01, 0.02, 0.05])
            v = round(r * i, 6)
            shown = v + 1 if wrong else v
            ep = {
                "tema": topic, "idioma": "es",
                "problema": f"¿Qué tensión cae en un resistor de {r} ohm con {i} A?",
                "datos": [{"valor": r, "unidad": "ohm"}, {"valor": i, "unidad": "A"}],
                "supuestos": ["resistor óhmico", "corriente continua"],
                "plan": "Aplicar la ley de Ohm V = I·R.",
                "pasos": [{"n": 1, "operacion": "V = I · R", "resultado_intermedio": f"{shown} V",
                           "justificacion": "ley de Ohm",
                           "verificacion": {"tipo": "unidades", "expr": f"{r} ohm * {i} A",
                                            "esperado": f"{shown} V"}}],
                "resultado": f"{shown} V",
                "verificacion_inversa": f"{shown} V / {r} ohm debe dar {i} A.",
                "errores_frecuentes": ["confundir miliamperios con amperios"],
                "variaciones": [{"problema": f"¿Y con {r * 2} ohm?", "respuesta": f"{round(r * 2 * i, 6)} V"}],
            }
        else:
            total = a + b + (1 if wrong else 0)
            ep = {
                "tema": topic, "idioma": "es", "problema": f"¿Cuánto es {a} + {b}?",
                "datos": [{"valor": a}, {"valor": b}], "supuestos": [],
                "plan": "Sumar unidades y decenas por separado.",
                "pasos": [
                    {"n": 1, "operacion": f"{a % 10} + {b % 10}", "resultado_intermedio": str(a % 10 + b % 10),
                     "justificacion": "suma de unidades",
                     "verificacion": {"tipo": "aritmetica", "expr": f"{a % 10} + {b % 10}",
                                      "esperado": str(a % 10 + b % 10)}},
                    {"n": 2, "operacion": f"{a} + {b}", "resultado_intermedio": str(total),
                     "justificacion": "se suman las decenas y el acarreo",
                     "verificacion": {"tipo": "aritmetica", "expr": f"{a} + {b}", "esperado": str(total)}}],
                "resultado": str(total),
                "verificacion_inversa": f"{total} - {b} debe dar {a}.",
                "errores_frecuentes": ["olvidar el acarreo"],
                "variaciones": [{"problema": f"¿Cuánto es {b} + {a}?", "respuesta": str(a + b)},
                                {"problema": f"¿Cuánto es {a + 1} + {b}?", "respuesta": str(a + 1 + b)}],
            }
        thinking = "<think>borrador interno que no se usa para entrenar</think>\n"
        return thinking + "```json\n" + json.dumps(ep, ensure_ascii=False) + "\n```", {"mock": True}
