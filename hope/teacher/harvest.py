"""Cosecha de episodios del maestro, tema por tema (ARQUITECTURA_HN1 §10.1.2–§10.1.3.1).

Pide al maestro (Qwen local) episodios de todo el currículo, los verifica con
las herramientas (``hope.tools.verify``) y los guarda con procedencia, sin
entrenar a Hope: sirve para adelantar datos cuando el maestro es lento (CPU) y
el entrenamiento corre en otra máquina o en otro momento.

Salida en ``--out``:

* ``episodios_verificados.jsonl`` — RESPALDADO (peso 1,0): se entrenan.
* ``episodios_inferidos.jsonl``   — INFERIDO (peso 0,3): nada verificable; se guardan aparte.
* ``episodios_refutados.jsonl``   — REFUTADO: no se entrenan; quedan como ejemplos de error (RRA).
* ``invalidos.jsonl``             — respuestas que no se pudieron leer como episodio.
* ``textos_entrenamiento.txt``    — ``Episode.training_text`` de los verificados (separados por línea en blanco doble).
* ``textos_inferidos.txt``        — ídem para los inferidos.
* ``estadisticas.json``           — conteos y tasas por tema.

Cada registro JSONL lleva: modelo, tema, nivel, idioma, estado, peso, detalle de
la verificación, el episodio completo, el texto de entrenamiento, segundos y tokens.
Es reanudable: al volver a correr, lee lo ya cosechado y evita repetir problemas.
La consulta pasa por el guardián (maestro local = nivel 1; remoto = requiere aprobación).

Uso::

    python -m hope.teacher.harvest --teacher configs/teacher_local.yaml --out datos_maestro \\
        --rounds 10 --max-hours 3
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from hope.teacher.client import Teacher, TeacherDenied
from hope.teacher.cycle import Topic, load_curriculum
from hope.teacher.episode import Episode, EpisodeParseError, curriculum_request, parse_episode
from hope.tools.verify import verify_episode

FILES = {"RESPALDADO": "episodios_verificados.jsonl", "INFERIDO": "episodios_inferidos.jsonl",
         "REFUTADO": "episodios_refutados.jsonl"}
TEXTS = {"RESPALDADO": "textos_entrenamiento.txt", "INFERIDO": "textos_inferidos.txt"}

# Contextos para variar los problemas de un mismo tema (el maestro tiende a repetir el ejemplo típico).
FOCUS = [
    "un problema de la vida cotidiana", "un contexto de electrónica o robótica", "un contexto de cocina o recetas",
    "un contexto de viajes y distancias", "un contexto de finanzas personales", "un contexto de deporte",
    "un contexto de construcción o carpintería", "un contexto de ciencia de laboratorio",
    "un contexto de agricultura", "un contexto de programación o computadoras", "un caso con un error típico",
    "un caso con números decimales", "un caso un poco más difícil que el básico", "un caso con varios pasos",
]


@dataclass
class HarvestStats:
    per_topic: dict = field(default_factory=lambda: defaultdict(Counter))
    seconds: float = 0.0
    completion_tokens: int = 0

    def as_dict(self) -> dict:
        topics = {}
        for t, c in sorted(self.per_topic.items()):
            asked = sum(c.values())
            topics[t] = {**dict(c), "pedidos": asked,
                         "tasa_verificado": c["RESPALDADO"] / asked if asked else None,
                         "tasa_refutado": c["REFUTADO"] / asked if asked else None,
                         "tasa_invalido": c["INVALIDO"] / asked if asked else None}
        total = Counter()
        for c in self.per_topic.values():
            total.update(c)
        n = sum(total.values())
        return {"total": {**dict(total), "pedidos": n,
                          "tasa_verificado": total["RESPALDADO"] / n if n else None},
                "segundos_maestro": round(self.seconds, 1), "tokens_generados": self.completion_tokens,
                "tokens_por_segundo": round(self.completion_tokens / self.seconds, 2) if self.seconds else None,
                "por_tema": topics}


class Harvester:
    def __init__(self, teacher: Teacher, catalog, out_dir: str | Path, *, log=print, seed: int = 0):
        self.teacher, self.catalog, self.log = teacher, catalog, log
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.stats = HarvestStats()
        self.seen: dict[str, list[str]] = defaultdict(list)       # tema → problemas ya enseñados
        self._load_previous()

    def _load_previous(self) -> None:
        for status, name in list(FILES.items()) + [("INVALIDO", "invalidos.jsonl")]:
            p = self.out / name
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                self.stats.per_topic[r["tema"]][status] += 1
                self.stats.seconds += r.get("segundos", 0.0)
                self.stats.completion_tokens += r.get("tokens", {}).get("completion_tokens", 0)
                if status != "INVALIDO":
                    self.seen[r["tema"]].append(r["episodio"]["problema"])

    def _append(self, name: str, record: dict) -> None:
        with open(self.out / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def ask(self, topic: Topic) -> dict:
        """Un pedido al maestro → registro guardado. Devuelve el registro (con 'estado')."""
        focus = self.rng.choice(FOCUS) if self.seen[topic.name] else ""
        messages = curriculum_request(topic.name, topic.level, topic.language,
                                      avoid=self.seen[topic.name], focus=focus)
        t0 = time.time()
        reply = self.teacher.ask(messages, purpose=f"cosecha:{topic.name}")
        dt = time.time() - t0
        self.stats.seconds += dt
        self.stats.completion_tokens += int(reply.usage.get("completion_tokens", 0) or 0)
        base = {"modelo": reply.model, "tema": topic.name, "nivel": topic.level, "idioma": topic.language,
                "enfoque": focus, "segundos": round(dt, 1), "tokens": reply.usage,
                "fecha": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            ep = Episode(parse_episode(reply.text), reply.model, reply.raw_thinking)
        except EpisodeParseError as e:
            rec = {**base, "estado": "INVALIDO", "error": str(e), "respuesta": reply.text[:4000]}
            self._append("invalidos.jsonl", rec)
            self.stats.per_topic[topic.name]["INVALIDO"] += 1
            return rec
        verify_episode(self.catalog, ep)
        status = ep.verification["status"]
        rec = {**base, "estado": status, "peso": ep.verification["weight"], "verificacion": ep.verification,
               "episodio": ep.data, "texto_entrenamiento": ep.training_text(tool_calls=True)}
        self._append(FILES[status], rec)
        if status in TEXTS:
            with open(self.out / TEXTS[status], "a", encoding="utf-8") as f:
                f.write(rec["texto_entrenamiento"] + "\n\n\n")
        self.stats.per_topic[topic.name][status] += 1
        self.seen[topic.name].append(ep.data["problema"])
        return rec

    def run(self, topics: list[Topic], *, rounds: int = 1, per_topic: int = 1, max_hours: float | None = None,
            max_calls: int | None = None) -> dict:
        deadline = time.time() + max_hours * 3600 if max_hours else None
        calls = 0
        for r in range(rounds):
            for t in topics:
                for _ in range(per_topic):
                    if (deadline and time.time() > deadline) or (max_calls is not None and calls >= max_calls):
                        return self.save_stats()
                    try:
                        rec = self.ask(t)
                    except TeacherDenied as e:
                        self.log(f"maestro denegado: {e}")
                        return self.save_stats()
                    except Exception as e:                  # servidor caído, timeout: se sigue con el próximo
                        self.log(f"  error con «{t.name}»: {type(e).__name__}: {e}")
                        time.sleep(2)
                        continue
                    calls += 1
                    self.log(f"ronda {r} · {t.name}: {rec['estado']} ({rec['segundos']} s)")
            self.save_stats()
        return self.save_stats()

    def save_stats(self) -> dict:
        d = self.stats.as_dict()
        (self.out / "estadisticas.json").write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        return d


def reverify(out_dir: str | Path, catalog) -> dict:
    """Vuelve a verificar todo lo cosechado (p. ej. tras mejorar el verificador) y reescribe los archivos."""
    out = Path(out_dir)
    records = []
    for name in FILES.values():
        p = out / name
        if p.exists():
            records += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    moved = Counter()
    still_invalid = []
    inv = out / "invalidos.jsonl"
    for r in ([json.loads(x) for x in inv.read_text(encoding="utf-8").splitlines() if x.strip()]
              if inv.exists() else []):
        try:                                   # el parser pudo haber mejorado: se reintenta leer la respuesta
            data = parse_episode(r["respuesta"])
        except EpisodeParseError:
            still_invalid.append(r)
            continue
        r = {k: v for k, v in r.items() if k not in ("error", "respuesta")}
        records.append({**r, "estado": "INVALIDO", "episodio": data})
    for name in list(FILES.values()) + list(TEXTS.values()) + ["invalidos.jsonl"]:
        (out / name).unlink(missing_ok=True)
    for r in still_invalid:
        with open(inv, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in records:
        ep = verify_episode(catalog, Episode(r["episodio"], r.get("modelo", "")))
        new = ep.verification["status"]
        if new != r["estado"]:
            moved[f"{r['estado']}→{new}"] += 1
        r.update(estado=new, peso=ep.verification["weight"], verificacion=ep.verification,
                 texto_entrenamiento=ep.training_text(tool_calls=True))
        with open(out / FILES[new], "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if new in TEXTS:
            with open(out / TEXTS[new], "a", encoding="utf-8") as f:
                f.write(r["texto_entrenamiento"] + "\n\n\n")
    return dict(moved)


def main(argv: list[str] | None = None) -> dict:
    from guardian.cli import load as load_guardian
    from hope.teacher.client import load_teacher
    from hope.tools.builtin import default_catalog

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="configs/teacher_local.yaml")
    ap.add_argument("--curriculum", default="configs/curriculum.yaml")
    ap.add_argument("--guardian", default="configs/guardian.yaml")
    ap.add_argument("--out", default="datos_maestro")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--per-topic", type=int, default=1)
    ap.add_argument("--topics", default=None, help="subcadenas separadas por coma para elegir temas")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reverificar", action="store_true", help="solo re-verificar lo ya cosechado (sin maestro)")
    args = ap.parse_args(argv)
    guardian = load_guardian(args.guardian)
    if args.reverificar:
        moved = reverify(args.out, default_catalog(guardian))
        print(json.dumps(moved, ensure_ascii=False))
        h = Harvester.__new__(Harvester)
        h.out, h.stats, h.seen = Path(args.out), HarvestStats(), defaultdict(list)
        h._load_previous()
        return h.save_stats()
    topics = load_curriculum(args.curriculum)
    if args.topics:
        keys = [k.strip().lower() for k in args.topics.split(",")]
        topics = [t for t in topics if any(k in t.name.lower() for k in keys)]
    h = Harvester(load_teacher(args.teacher, guardian), default_catalog(guardian), args.out, seed=args.seed)
    stats = h.run(topics, rounds=args.rounds, per_topic=args.per_topic, max_hours=args.max_hours,
                  max_calls=args.max_calls)
    print(json.dumps(stats["total"], ensure_ascii=False))
    return stats


if __name__ == "__main__":
    main()
