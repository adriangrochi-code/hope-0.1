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

**Modo continuo** (``--continuo``; lo usa la aplicación, pedido del usuario: «Qwen desde el principio»):
explica sin parar, hasta que lo detengan o se cumpla ``--max-hours``:

* alterna temas del currículo y **pasajes del corpus** (``--corpus data/pre/wiki_es ...``): el maestro
  explica el pasaje paso a paso (conceptos, y cálculos o código con su ``verificacion``);
* **varios maestros**: ``--teacher`` acepta un YAML con ``maestros: [{id, endpoint, model, ...}, ...]``
  (``configs/maestros.yaml``); cada pedido va al próximo maestro en un round-robin ponderado por su tasa
  de verificación, y cada episodio guarda el ``maestro`` que lo hizo;
* ``--paralelo N`` pedidos a la vez (el servidor tiene que aceptarlos: ``llama-server -np N``);
* ``--max-calls-dia``: al llegar, espera al día siguiente;
* escribe ``<out>/train.bin`` y ``val.bin`` (formato del corpus: textos separados por el byte 0;
  partición determinista por hash) de forma **atómica y por lotes** (``--lote``): ``hope.train.run``
  los vuelve a leer en cada tramo. Es reanudable: al empezar relee lo cosechado y reconstruye los .bin.

    python -m hope.teacher.harvest --continuo --teacher configs/maestros.yaml --out data/maestro \\
        --corpus data/pre/wiki_es data/pre/codigo --paralelo 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from hope.teacher.client import Teacher, TeacherConfig, TeacherDenied
from hope.teacher.cycle import Topic, load_curriculum
from hope.teacher.episode import (FORMAT_HINT, SYSTEM_PROMPT, Episode, EpisodeParseError, curriculum_request,
                                  parse_episode)
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
        self.pasajes_vistos: set[str] = set()
        self.por_maestro: dict[str, Counter] = defaultdict(Counter)
        self._lock = threading.Lock()
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
                self.por_maestro[r.get("maestro") or r.get("modelo", "?")][status] += 1
                if r.get("pasaje_sha"):
                    self.pasajes_vistos.add(r["pasaje_sha"])
                self.stats.seconds += r.get("segundos", 0.0)
                self.stats.completion_tokens += r.get("tokens", {}).get("completion_tokens", 0)
                if status != "INVALIDO":
                    self.seen[r["tema"]].append(r["episodio"]["problema"])

    def _append(self, name: str, record: dict) -> None:
        with self._lock, open(self.out / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def ask(self, topic: Topic, *, teacher: Teacher | None = None, maestro: str | None = None,
            messages: list[dict] | None = None, extra: dict | None = None) -> dict:
        """Un pedido al maestro → registro guardado. Devuelve el registro (con 'estado').

        ``teacher``/``maestro``: el maestro de este pedido (modo continuo con varios maestros).
        ``messages``/``extra``: otro tipo de pedido (p. ej. explicar un pasaje del corpus)."""
        teacher = teacher or self.teacher
        focus = self.rng.choice(FOCUS) if self.seen[topic.name] and messages is None else ""
        if messages is None:
            messages = curriculum_request(topic.name, topic.level, topic.language,
                                          avoid=self.seen[topic.name], focus=focus)
        t0 = time.time()
        reply = teacher.ask(messages, purpose=f"cosecha:{topic.name}")
        dt = time.time() - t0
        self.stats.seconds += dt
        self.stats.completion_tokens += int(reply.usage.get("completion_tokens", 0) or 0)
        base = {"modelo": reply.model, "tema": topic.name, "nivel": topic.level, "idioma": topic.language,
                "enfoque": focus, "segundos": round(dt, 1), "tokens": reply.usage,
                "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"), "maestro": maestro or reply.model, **(extra or {})}
        try:
            ep = Episode(parse_episode(reply.text), reply.model, reply.raw_thinking)
        except EpisodeParseError as e:
            rec = {**base, "estado": "INVALIDO", "error": str(e), "respuesta": reply.text[:4000]}
            self._append("invalidos.jsonl", rec)
            with self._lock:
                self.stats.per_topic[topic.name]["INVALIDO"] += 1
                self.por_maestro[rec["maestro"]]["INVALIDO"] += 1
            return rec
        verify_episode(self.catalog, ep)
        status = ep.verification["status"]
        rec = {**base, "estado": status, "peso": ep.verification["weight"], "verificacion": ep.verification,
               "episodio": ep.data, "texto_entrenamiento": ep.training_text(tool_calls=True)}
        self._append(FILES[status], rec)
        with self._lock:
            if status in TEXTS:
                with open(self.out / TEXTS[status], "a", encoding="utf-8") as f:
                    f.write(rec["texto_entrenamiento"] + "\n\n\n")
            self.stats.per_topic[topic.name][status] += 1
            self.por_maestro[rec["maestro"]][status] += 1
            self.seen[topic.name].append(ep.data["problema"])
            if rec.get("pasaje_sha"):
                self.pasajes_vistos.add(rec["pasaje_sha"])
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


# ---------------------------------------------------------------------------
# Modo continuo
# ---------------------------------------------------------------------------
def load_teachers(path: str | Path, guardian=None) -> list[tuple[str, Teacher]]:
    """Un YAML de maestro (formato de ``configs/teacher.yaml``) o uno con ``maestros: [...]`` (varios)."""
    from hope.teacher.client import load_teacher
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "maestros" not in raw:
        t = load_teacher(path, guardian)
        return [(str(raw.get("id") or t.config.model), t)]
    import tempfile
    campos = set(TeacherConfig.__dataclass_fields__)
    out = []
    for i, m in enumerate(raw["maestros"]):
        if not isinstance(m, dict) or m.get("activo") is False:
            continue
        cfg = {k: v for k, v in {**{k: v for k, v in raw.items() if k in campos}, **m}.items() if k in campos}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        try:
            out.append((str(m.get("id") or m.get("nombre") or cfg.get("model") or i), load_teacher(f.name, guardian)))
        finally:
            Path(f.name).unlink(missing_ok=True)
    if not out:
        raise ValueError(f"{path}: no hay maestros activos")
    return out


PASAJE_PEDIDO = (
    "PASAJE (de {fuente}):\n<<<\n{texto}\n>>>\n\n"
    "TEMA: explicar un pasaje ({fuente})\nIDIOMA: {idioma}\n"
    "Explicá este pasaje paso a paso a un estudiante: qué dice, los conceptos y las palabras difíciles, y por qué. "
    "Si tiene cifras, cuentas, unidades o código, resolvelos y agregá en cada paso su 'verificacion' (aritmética, "
    "unidades o código). En 'problema' poné la pregunta que responde el pasaje; en 'resultado', la idea principal. "
    "Agregá variaciones con respuesta. Formato:\n{formato}")


def pasajes(corpus: list[str | Path], rng: random.Random, max_chars: int = 1200):
    """Pasajes al azar de los train.bin del corpus (cortados en límites de documento y de oración)."""
    files = []
    for c in corpus:
        c = Path(c)
        f = c / "train.bin" if c.is_dir() else c
        if f.exists() and f.stat().st_size > 2048:
            files.append(f)
    if not files:
        return
    while True:
        f = rng.choice(files)
        size = f.stat().st_size
        with open(f, "rb") as fh:
            fh.seek(rng.randrange(0, max(1, size - 8000)))
            raw = fh.read(8000)
        docs = [d for d in raw.split(bytes([0]))[1:-1] if len(d) > 300] or [d for d in raw.split(bytes([0])) if len(d) > 300]
        if not docs:
            continue
        texto = rng.choice(docs).decode("utf-8", errors="ignore").strip()
        if len(texto) > max_chars:
            ini = rng.randrange(0, len(texto) - max_chars + 1)
            texto = texto[ini:ini + max_chars]
            corte = texto.rfind(". ")
            texto = texto[:corte + 1] if corte > max_chars // 2 else texto
        yield f.parent.name, texto


def _idioma(texto: str) -> str:
    t = " " + texto.lower() + " "
    es = sum(t.count(w) for w in (" el ", " la ", " de ", " que ", " los ", " es "))
    en = sum(t.count(w) for w in (" the ", " of ", " and ", " is ", " to "))
    return "en" if en > es else "es"


def escribir_bins(out: Path, *, incluir_inferidos: bool = True, val_percent: float = 3.0) -> dict:
    """Reconstruye ``train.bin``/``val.bin`` con lo aceptado (escritura atómica: temporal + reemplazo)."""
    import hashlib
    from hope.data.corpus import EOD, split_of
    out = Path(out)
    bufs = {"train": bytearray(), "val": bytearray()}
    estados = ["RESPALDADO"] + (["INFERIDO"] if incluir_inferidos else [])
    for est in estados:
        p = out / FILES[est]
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = r.get("texto_entrenamiento")
            if t:
                h = hashlib.sha256(t.encode()).hexdigest()
                bufs[split_of(h, val_percent)] += t.encode("utf-8") + bytes([EOD])
    app = out / "episodios.jsonl"                 # lo que generó la aplicación (generadores, maestro de la app)
    if app.exists():
        for line in app.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("aceptado") and r.get("texto") and (incluir_inferidos or r.get("estado") != "INFERIDO"):
                bufs[split_of(r.get("sha") or hashlib.sha256(r["texto"].encode()).hexdigest(), val_percent)] += \
                    r["texto"].encode("utf-8") + bytes([EOD])
    tam = {}
    for split, buf in bufs.items():
        tmp = out / f"{split}.bin.tmp"
        tmp.write_bytes(bytes(buf))
        try:
            os.replace(tmp, out / f"{split}.bin")
        except PermissionError:          # Windows: el entrenamiento lo tiene abierto; se actualiza en el próximo lote
            tmp.unlink(missing_ok=True)
        tam[split] = len(buf)
    return tam


def elegir_maestro(maestros: list[tuple[str, Teacher]], por_maestro: dict, usados: Counter) -> tuple[str, Teacher]:
    """Round-robin ponderado por la tasa de verificación (sin medir: 0,8): todos aportan variedad."""
    def peso(mid):
        c = por_maestro.get(mid) or Counter()
        n = sum(c.values())
        return 0.3 + (c["RESPALDADO"] / n if n else 0.8)
    return min(maestros, key=lambda m: (round((usados[m[0]] + 1) / peso(m[0]), 9), usados[m[0]], m[0]))


def continuo(h: Harvester, maestros: list[tuple[str, Teacher]], topics: list[Topic], *, corpus=(),
             fraccion_pasajes: float = 0.5, paralelo: int = 1, lote: int = 8, max_hours: float | None = None,
             max_calls_dia: int | None = None, incluir_inferidos: bool = True, progreso: bool = False,
             detener=None) -> dict:
    """Explica sin parar (hasta ``detener()``, ``max_hours`` o una señal). Ver el docstring del módulo."""
    from concurrent.futures import ThreadPoolExecutor
    deadline = time.time() + max_hours * 3600 if max_hours else None
    fuente = pasajes(list(corpus), h.rng) if corpus else None
    usados: Counter = Counter()
    dia, hoy = time.strftime("%Y-%m-%d"), 0
    nuevos = 0
    escribir_bins(h.out, incluir_inferidos=incluir_inferidos)        # reanudar: lo cosechado ya está disponible
    i = 0

    def fin() -> bool:
        return (detener is not None and detener()) or (deadline is not None and time.time() >= deadline)

    def tarea(spec):
        topic, mid, teacher, messages, extra = spec
        try:
            return h.ask(topic, teacher=teacher, maestro=mid, messages=messages, extra=extra)
        except TeacherDenied as e:
            return {"estado": "DENEGADO", "error": str(e)}
        except Exception as e:                       # servidor caído, timeout: se sigue
            return {"estado": "ERROR", "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=max(1, paralelo)) as pool:
        while not fin():
            if time.strftime("%Y-%m-%d") != dia:
                dia, hoy = time.strftime("%Y-%m-%d"), 0
            if max_calls_dia is not None and hoy >= max_calls_dia:
                msg = "límite de consultas de hoy alcanzado: sigue mañana"
                h.log(msg)
                if progreso:
                    print(f"PROGRESO 1.0 {msg}", flush=True)
                while not fin() and time.strftime("%Y-%m-%d") == dia:
                    time.sleep(min(60.0, max(0.1, (deadline - time.time()) if deadline else 60.0)))
                continue
            specs = []
            n = max(1, paralelo) if max_calls_dia is None else min(max(1, paralelo), max_calls_dia - hoy)
            for _ in range(n):
                mid, teacher = elegir_maestro(maestros, h.por_maestro, usados)
                usados[mid] += 1
                if fuente is not None and h.rng.random() < fraccion_pasajes:
                    for _intento in range(20):
                        nombre, texto = next(fuente)
                        import hashlib
                        sha = hashlib.sha256(texto.encode()).hexdigest()
                        if sha not in h.pasajes_vistos:
                            break
                    idioma = _idioma(texto)
                    tema = Topic(f"pasaje: {nombre}", 1, idioma)
                    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": PASAJE_PEDIDO.format(fuente=nombre, texto=texto, idioma=idioma,
                                                                             formato=FORMAT_HINT)}]
                    specs.append((tema, mid, teacher, msgs, {"pasaje_sha": sha, "pasaje_fuente": nombre}))
                else:
                    specs.append((topics[i % len(topics)], mid, teacher, None, None))
                    i += 1
            for rec in pool.map(tarea, specs):
                hoy += 1
                est = rec.get("estado")
                if est == "DENEGADO":
                    h.log(f"maestro denegado: {rec['error']}")
                    return h.save_stats()
                if est == "ERROR":
                    h.log(f"  error: {rec['error']}")
                    time.sleep(1)
                    continue
                if est == "RESPALDADO" or (est == "INFERIDO" and incluir_inferidos):
                    nuevos += 1
                h.log(f"{rec.get('maestro')} · {rec.get('tema')}: {est} ({rec.get('segundos')} s)")
            if nuevos >= lote:
                tam = escribir_bins(h.out, incluir_inferidos=incluir_inferidos)
                nuevos = 0
                h.save_stats()
                if progreso:
                    print(f"PROGRESO 0.5 explicaciones aceptadas: {tam['train'] + tam['val']} bytes; hoy {hoy} consultas",
                          flush=True)
    escribir_bins(h.out, incluir_inferidos=incluir_inferidos)
    return h.save_stats()


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
    ap.add_argument("--continuo", action="store_true", help="explicar sin parar y escribir train.bin/val.bin")
    ap.add_argument("--corpus", nargs="*", default=[], help="carpetas con train.bin para explicar pasajes")
    ap.add_argument("--fraccion-pasajes", type=float, default=0.5)
    ap.add_argument("--paralelo", type=int, default=1)
    ap.add_argument("--lote", type=int, default=8, help="reescribir los .bin cada N explicaciones aceptadas")
    ap.add_argument("--max-calls-dia", type=int, default=None)
    ap.add_argument("--sin-inferidos", action="store_true", help="solo lo verificado entra a train.bin")
    ap.add_argument("--progreso", action="store_true", help="líneas PROGRESO para la aplicación")
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
    if args.continuo:
        maestros = load_teachers(args.teacher, guardian)
        h = Harvester(maestros[0][1], default_catalog(guardian), args.out, seed=args.seed)
        stats = continuo(h, maestros, topics, corpus=args.corpus, fraccion_pasajes=args.fraccion_pasajes,
                         paralelo=args.paralelo, lote=args.lote, max_hours=args.max_hours,
                         max_calls_dia=args.max_calls_dia, incluir_inferidos=not args.sin_inferidos,
                         progreso=args.progreso)
        print(json.dumps(stats["total"], ensure_ascii=False))
        return stats
    h = Harvester(load_teacher(args.teacher, guardian), default_catalog(guardian), args.out, seed=args.seed)
    stats = h.run(topics, rounds=args.rounds, per_topic=args.per_topic, max_hours=args.max_hours,
                  max_calls=args.max_calls)
    print(json.dumps(stats["total"], ensure_ascii=False))
    return stats


if __name__ == "__main__":
    main()
