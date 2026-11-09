"""Autoentrenamiento con soluciones propias verificadas (propuesta 1; STaR, Zelikman et al. 2022).

Para cada problema de práctica (con respuesta conocida):

1. Hope genera hasta ``samples`` soluciones con temperatura (exploración).
2. Se queda con la primera que llega a la respuesta correcta **y** pasa la crítica
   (CRE + HCRI): acertar por casualidad con un razonamiento roto no cuenta
   (Herramientas §10: «resultado correcto por casualidad»).
3. Si ninguna sirve, **racionalización**: se le da la respuesta como pista y se le
   pide que reconstruya el razonamiento. Si la explicación es correcta y pasa la
   crítica, se guarda **sin la pista**, como si lo hubiera resuelto.

Los textos aceptados se entrenan en la siguiente consolidación. Así cada consulta
a Qwen rinde más: Hope practica sola con problemas generados y aprende de sus
propios aciertos verificados, y depende cada vez menos del maestro.

Los problemas de práctica no pueden ser los de evaluación (lo controla el ciclo).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from hope.constraints.critic import critique


@dataclass
class StarConfig:
    samples: int = 4
    temperature: float = 0.8
    rationalize: bool = True
    max_new: int = 400


@dataclass
class StarReport:
    problems: int = 0
    solved_directly: int = 0
    rationalized: int = 0
    rejected_lucky: int = 0          # respuesta correcta con razonamiento que no pasa la crítica
    texts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "texts"} | {"accepted": len(self.texts)}


def self_improve(generate: Callable[[str, float], str], problems: list[dict], *,
                 extract: Callable[[str], str | None], match: Callable[[str | None, str], bool],
                 cfg: StarConfig = StarConfig()) -> StarReport:
    """``generate(prompt, temperature) -> continuación``; ``problems``: {tema, problema, respuesta}."""
    rep = StarReport()
    for p in problems:
        rep.problems += 1
        prompt = f"Tema: {p['tema']}\nProblema: {p['problema']}\n"
        accepted = None
        for _ in range(cfg.samples):
            text = generate(prompt, cfg.temperature)
            ans = extract(text)
            if not match(ans, p["respuesta"]):
                continue
            if critique(p["problema"], text, ans).verdict in ("ok", "incierto"):
                accepted = text
                rep.solved_directly += 1
                break
            rep.rejected_lucky += 1
        if accepted is None and cfg.rationalize:
            hinted = prompt + f"(Pista: la respuesta correcta es {p['respuesta']}. Explicá cómo se llega.)\n"
            text = generate(hinted, 0.0)
            ans = extract(text)
            if match(ans, p["respuesta"]) and critique(p["problema"], text, ans).verdict in ("ok", "incierto"):
                accepted = text
                rep.rationalized += 1
        if accepted is not None:
            rep.texts.append(prompt + accepted.strip())        # sin la pista
    return rep
