"""Banco de contexto largo (ARQUITECTURA_HN1 §3.9.2; pedido del usuario).

Tareas sintéticas en español, con relleno sin números y nombres inventados (no se pueden
responder con lo que el modelo ya sabe):

* ``aguja`` — «aguja en el pajar» (passkey): una clave de 5 cifras escondida a cierta profundidad;
* ``varias`` — varias claves de personas distintas (NIAH con varias claves); se pregunta una: las
  otras claves del mismo texto son los distractores, así que hay que recuperar **la** correcta;
* ``combinar`` — dos datos lejanos que hay que unir: «P vive en C» al principio y «C queda en X»
  al final, con parejas distractoras; se pregunta en qué país vive P.

Medidas por ejemplo:

* **ranking**: log-probabilidad de la respuesta correcta frente a 3 distractores (azar = 25 %); da
  señal aunque el modelo sea chico;
* **generación** exacta (voraz) de la respuesta.

Modos de lectura del contexto: ``completo`` (todo de una vez), ``cache`` (por tramos que siguen
desde las cachés: Transformer-XL, mismo resultado que ``completo``), ``cache-int8``/``cache-int4``
(lo guardado cuantizado), ``resumen`` (tramos anteriores comprimidos por la capa de memoria),
``ventana`` (solo el último tramo: la referencia de «sin memoria») y, para generar, ``gen``,
``gen-int8``, ``gen-quest``.

Además, ``estimate`` y ``max_context`` calculan la memoria de inferencia por byte de contexto y el
contexto máximo que entra en la A770 (16 GB) y en la GTX 1060 (6 GB) para cada configuración.

Uso::

    python -m hope.eval.contexto_bench --memoria                     # tabla de memoria
    python -m hope.eval.contexto_bench --entrenar-min 15 --largos 1K,2K,4K,8K --salida r.json
    python -m hope.eval.contexto_bench --datos data/largos/sintetico --bytes 50M   # datos de entrenamiento
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Generación de ejemplos
# ---------------------------------------------------------------------------
SUJETOS = ["el viejo pescador", "la maestra del pueblo", "un perro callejero", "la abuela", "el panadero",
           "una niña curiosa", "el cartero", "la vecina del tercer piso", "un gato gris", "el guardabosques",
           "la médica", "el músico ciego", "una tortuga", "el herrero", "la jardinera", "un viajero cansado"]
VERBOS = ["camina hacia", "mira con calma", "recuerda", "pinta", "limpia", "busca", "cuida", "dibuja",
          "visita", "describe", "ordena", "escucha el ruido de", "cruza", "observa", "arregla", "rodea"]
OBJETOS = ["el río", "la plaza", "un árbol enorme", "la vieja estación", "el mercado", "una casa blanca",
           "el puente de piedra", "la biblioteca", "un campo de trigo", "la orilla del lago", "el faro",
           "una colina verde", "el patio", "la escuela", "un camino de tierra", "el muelle"]
CIERRES = ["mientras cae la tarde", "sin apuro", "antes de la lluvia", "como todos los días", "en silencio",
           "con una sonrisa", "al amanecer", "después del almuerzo", "bajo el sol", "cuando sopla el viento"]
NOMBRES = ["Ardelia", "Bruno Vexo", "Calista", "Damián Orl", "Eulalia", "Fermín Taro", "Gisela", "Hilario",
           "Ileana", "Joaquín Pell", "Kalina", "Lisandro", "Marcela Fu", "Nicanor", "Odalis", "Pacífico"]
CIUDADES = ["Tremonte", "Valdoria", "Quintasol", "Brumal", "Osterra", "Lindavía", "Carrusco", "Pelagra",
            "Morvento", "Silvara", "Durnay", "Feldaro"]
PAISES = ["Varnia", "Toleda", "Mirsania", "Quendor", "Ostralia", "Belmira", "Zandria", "Corvelia",
          "Puntaria", "Eldoria", "Nortesia", "Galvara"]


@dataclass
class Example:
    tarea: str
    largo: int
    profundidad: float
    contexto: str
    respuesta: str
    distractores: list[str]
    meta: dict = field(default_factory=dict)


def filler(rng: np.random.Generator, n_bytes: int) -> str:
    out, size = [], 0
    while size < n_bytes:
        s = (f"{SUJETOS[rng.integers(len(SUJETOS))].capitalize()} {VERBOS[rng.integers(len(VERBOS))]} "
             f"{OBJETOS[rng.integers(len(OBJETOS))]} {CIERRES[rng.integers(len(CIERRES))]}. ")
        out.append(s)
        size += len(s.encode())
    return "".join(out)


def _place(rng, length: int, facts: list[tuple[float, str]], question: str) -> str:
    """Relleno de ~``length`` bytes con los hechos insertados a sus profundidades (0–1)."""
    budget = max(length - len(question.encode()) - sum(len(f.encode()) for _, f in facts), 0)
    text = filler(rng, budget)
    raw = text.encode()[:budget].decode("utf-8", "ignore")
    pieces, last = [], 0
    for depth, fact in sorted(facts):
        target = max(int(len(raw) * depth), 0)
        before, after = raw.rfind(". ", 0, target), raw.find(". ", target)
        options = [c + 2 for c in (before, after) if c >= 0] or [0]
        cut = max(min(options, key=lambda c: abs(c - target)), last)    # fin de oración más cercano
        pieces.append(raw[last:cut])
        pieces.append(fact + " ")
        last = cut
    pieces.append(raw[last:])
    return "".join(pieces) + question


def _num(rng) -> str:
    return str(int(rng.integers(10000, 100000)))


def task_aguja(rng, length: int, depth: float) -> Example:
    key = _num(rng)
    q = "\nPregunta: ¿cuál es la clave secreta?\nRespuesta: la clave secreta es "
    ctx = _place(rng, length, [(depth, f"La clave secreta es {key}.")], q)
    others = set()
    while len(others) < 3:
        n = _num(rng)
        if n != key:
            others.add(n)
    return Example("aguja", length, depth, ctx, key, sorted(others))


def task_varias(rng, length: int, depth: float, n_keys: int = 4) -> Example:
    names = list(rng.choice(NOMBRES, size=n_keys, replace=False))
    keys = []
    while len(keys) < n_keys:
        k = _num(rng)
        if k not in keys:
            keys.append(k)
    target = int(rng.integers(n_keys))
    depths = [float(d) for d in np.linspace(0.05, 0.95, n_keys)]
    depths[target] = depth                              # la que se pregunta va a la profundidad pedida
    facts = [(d + 1e-6 * i, f"La clave de {nm} es {k}.") for i, (d, nm, k) in enumerate(zip(depths, names, keys))]
    q = f"\nPregunta: ¿cuál es la clave de {names[target]}?\nRespuesta: la clave de {names[target]} es "
    return Example("varias", length, depth, _place(rng, length, facts, q), keys[target],
                   [k for i, k in enumerate(keys) if i != target][:3])


def task_combinar(rng, length: int, depth: float, n_pairs: int = 4) -> Example:
    people = list(rng.choice(NOMBRES, size=n_pairs, replace=False))
    cities = list(rng.choice(CIUDADES, size=n_pairs, replace=False))
    countries = list(rng.choice(PAISES, size=n_pairs, replace=False))
    target = int(rng.integers(n_pairs))
    facts = []
    for i in range(n_pairs):
        d1 = depth * 0.3 if i == target else float(rng.uniform(0.02, 0.45))    # dónde vive: al principio
        d2 = 0.6 + 0.35 * float(rng.random())                                  # dónde queda: al final
        facts.append((d1 + 1e-6 * i, f"{people[i]} vive en {cities[i]}."))
        facts.append((d2 + 1e-6 * i, f"{cities[i]} queda en {countries[i]}."))
    q = f"\nPregunta: ¿en qué país vive {people[target]}?\nRespuesta: {people[target]} vive en "
    return Example("combinar", length, depth, _place(rng, length, facts, q), countries[target],
                   [c for i, c in enumerate(countries) if i != target][:3])


TASKS = {"aguja": task_aguja, "varias": task_varias, "combinar": task_combinar}


def make_examples(tasks=("aguja", "varias", "combinar"), lengths=(1024, 2048), depths=(0.1, 0.5, 0.9),
                  n: int = 2, seed: int = 0) -> list[Example]:
    rng = np.random.default_rng(seed)
    return [TASKS[t](rng, L, d) for t in tasks for L in lengths for d in depths for _ in range(n)]


def write_training_data(out: str | Path, n_bytes: int, lengths=(2048, 8192, 32768, 131072), seed: int = 10**6,
                        val_frac: float = 0.01) -> dict:
    """Datos de entrenamiento con las mismas tareas (semillas distintas de las de evaluación), en el
    formato de ``hope.data.largos`` (documentos enteros con índice): cada documento es un ejemplo
    con su respuesta. Así la etapa larga incluye tareas que solo se resuelven con la memoria."""
    from hope.data.largos import _Writer
    out = Path(out)
    rng = np.random.default_rng(seed)
    w = {"train": _Writer(out, "train"), "val": _Writer(out, "val")}
    while w["train"].pos < n_bytes:
        t = list(TASKS)[int(rng.integers(len(TASKS)))]
        ex = TASKS[t](rng, int(rng.choice(lengths)), float(rng.uniform(0.02, 0.98)))
        split = "val" if rng.random() < val_frac else "train"
        w[split].add((ex.contexto + ex.respuesta + ".\n").encode())
    summary = {k: v.close() for k, v in w.items()}
    (out / "manifest.json").write_text(json.dumps({"fuente": "sintetico_contexto", **summary}, indent=2),
                                       encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Puntaje
# ---------------------------------------------------------------------------
def _bytes(s: str) -> list[int]:
    return list(s.encode("utf-8"))


@torch.no_grad()
def _prefill_cache(model, ctx: list[int], segment: int):
    """Lee el contexto por tramos (``segment_forward``). Devuelve (estado, última salida oculta)."""
    from hope.core.hn1 import patch_boundaries
    dev = model.head.weight.device
    x = torch.tensor([ctx], device=dev)
    # Fronteras sin forzar la última (se agrega un byte que no es frontera y se descarta).
    end = patch_boundaries(torch.tensor([ctx + [97]], device=dev), model.cfg.max_patch, model._entropy())[0][:, :-1]
    if not bool(end[0, -1]):
        raise ValueError("el contexto debe terminar en una frontera de parche (p. ej. un espacio)")
    state = model.new_segment_state()
    a, t = 0, len(ctx)
    out = None
    while a < t:
        b = min(a + segment, t)
        if b < t:
            e = end[0, a:b].nonzero()
            if len(e):
                b = a + int(e[-1]) + 1
        out = model.segment_forward(x[:, a:b], end[:, a:b], state)
        a = b
    return state, out["hidden"][:, -1:]


@torch.no_grad()
def _cand_logp_cache(model, state, last_hidden, cand: list[int]) -> float:
    from hope.core.hn1 import GlobalCache, patch_boundaries
    dev = model.head.weight.device
    st = GlobalCache._copy(state)
    x = torch.tensor([cand], device=dev)
    end = patch_boundaries(x, model.cfg.max_patch, model._entropy())[0]
    out = model.segment_forward(x[:, :-1], end[:, :-1], st) if len(cand) > 1 else None
    hid = last_hidden if out is None else torch.cat([last_hidden, out["hidden"]], dim=1)
    logp = F.log_softmax(model.head(hid).float(), -1)[0]
    return float(logp.gather(1, x[0, :, None]).sum())


@torch.no_grad()
def _cand_logp_forward(model, ctx: list[int], cand: list[int], memory=None) -> float:
    dev = model.head.weight.device
    x = torch.tensor([ctx + cand], device=dev)
    logits = model(x[:, :-1], memory=memory)["logits"][0, len(ctx) - 1:]
    logp = F.log_softmax(logits.float(), -1)
    return float(logp.gather(1, x[0, len(ctx):, None]).sum())


@torch.no_grad()
def rank(model, ex: Example, mode: str = "cache", segment: int = 1024) -> dict:
    """¿La respuesta correcta tiene más log-probabilidad que los distractores?"""
    ctx = _bytes(ex.contexto)
    cands = [ex.respuesta] + list(ex.distractores)
    if mode == "completo":
        lps = [_cand_logp_forward(model, ctx, _bytes(c)) for c in cands]
    elif mode == "ventana":
        lps = [_cand_logp_forward(model, ctx[-segment:], _bytes(c)) for c in cands]
    elif mode == "resumen":
        cut = max(len(ctx) - segment, 0)
        mem = model.read_segments(torch.tensor([ctx[:cut]], device=model.head.weight.device), segment) if cut else None
        lps = [_cand_logp_forward(model, ctx[cut:], _bytes(c), memory=mem) for c in cands]
    elif mode.startswith("cache"):
        state, last = _prefill_cache(model, ctx, segment)
        lps = [_cand_logp_cache(model, state, last, _bytes(c)) for c in cands]
    else:
        raise ValueError(mode)
    return {"ok": int(np.argmax(lps) == 0), "logp": lps[0], "margen": lps[0] - max(lps[1:])}


@torch.no_grad()
def generate_answer(model, ex: Example, chunk: int = 1024) -> dict:
    out = model.generate(ex.contexto, len(ex.respuesta.encode()) + 1, max_context=len(ex.contexto.encode()) + 64,
                         prefill_chunk=chunk)
    text = out.decode("utf-8", "ignore")
    return {"ok": int(text.startswith(ex.respuesta)), "salida": text}


# ---------------------------------------------------------------------------
# Memoria de inferencia
# ---------------------------------------------------------------------------
def estimate(cfg, n_bytes: int, *, kv_bits: int = 16, dtype_bytes: int = 2, weight_bytes: float = 2.0,
             evict: bool | None = None, quest_page: int = 0, n_params: int | None = None,
             prefill_chunk: int = 4096) -> dict:
    """Memoria de inferencia (bytes) para generar con ``n_bytes`` de contexto. Analítica; coincide
    con lo que miden ``generate(kv_stats=...)`` y ``measure`` (prueba en ``tests/test_contexto.py``)."""
    from hope.core.kvcache import kv_bytes_per_position
    evict = cfg.kv_evict if evict is None else evict
    bpp = cfg.rope_bytes_per_patch
    patches = n_bytes / bpp
    out = {"pesos": (n_params or 0) * weight_bytes}

    def layer(positions, kind, window, heads, kvh, dh):
        stored = min(positions, window + cfg.sinks) if (kind == "swa" and evict) else positions
        qp = quest_page if kind == "attn" else 0
        return stored * kv_bytes_per_position(kvh, dh, kv_bits, cfg.kv_group, dtype_bytes, qp)

    lw = cfg.local_window or cfg.window
    dl_h = cfg.d_local // cfg.local_heads
    local = (cfg.enc_layers + cfg.dec_layers) * layer(n_bytes, cfg.local_kind, lw, cfg.local_heads,
                                                       cfg.local_heads, dl_h)
    dh = cfg.d // cfg.heads
    kvh = cfg.kv_heads or cfg.heads
    glob = sum(layer(patches, k, cfg.window, cfg.heads, kvh, dh) for k in cfg.pattern if k in ("attn", "swa"))
    core = cfg.max_iters * layer(patches, cfg.core_kind, cfg.window, cfg.heads, kvh, dh) \
        if cfg.core_kind in ("attn", "swa") else 0
    out["kv_local"], out["kv_global"], out["kv_nucleo"] = local, glob, core
    # Salidas globales guardadas por GlobalCache: con prefill por tramos solo las últimas 128.
    long_path = n_bytes > 8192
    out["salidas_globales"] = (min(patches, 128) if long_path else patches) * cfg.d * dtype_bytes
    # Listas de Python de generate (bytes del texto y fronteras): ~80 bytes por byte de contexto.
    out["listas_python"] = 80 * n_bytes
    # Activaciones del prefill por tramos (pico de una atención local: cabezas × tramo × (tramo + ventana)).
    t = min(prefill_chunk, n_bytes)
    keys = t + (min(n_bytes, lw + cfg.sinks) if cfg.local_kind == "swa" else n_bytes)
    out["activaciones_prefill"] = cfg.local_heads * t * keys * 4 + 20 * t * cfg.d_local * 4
    out["total"] = sum(v for k, v in out.items())
    out["kv_por_byte"] = (local + glob + core) / max(n_bytes, 1)
    return out


def max_context(cfg, budget: float, *, n_params: int, weight_bytes: float = 2.0, kv_bits: int = 16,
                dtype_bytes: int = 2, quest_page: int = 0, cap: int = 1 << 30) -> int:
    """El contexto más largo (bytes) que entra en ``budget`` bytes (búsqueda binaria)."""
    lo, hi = 0, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        e = estimate(cfg, mid, kv_bits=kv_bits, dtype_bytes=dtype_bytes, weight_bytes=weight_bytes,
                     n_params=n_params, quest_page=quest_page)
        if e["total"] <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


GPUS = {"A770 (16 GB)": 16e9 * 0.9, "GTX 1060 (6 GB)": 6e9 * 0.85}


def memory_table(presets=("hn1-200m", "hn1-200m-largo", "hn1-500m", "hn1-500m-largo"),
                 kv=(16, 8, 4), contexts=(32_768, 131_072)) -> list[dict]:
    from hope.core.hn1 import HN1Model, preset
    rows = []
    for name in presets:
        cfg = preset(name)
        with torch.device("meta"):
            n = sum(p.numel() for p in HN1Model(cfg).parameters())
        for bits in kv:
            row = {"preset": name, "parametros": n, "kv_bits": bits}
            for c in contexts:
                e = estimate(cfg, c, kv_bits=bits, n_params=n)
                row[f"kv_MB@{c // 1024}K"] = round((e["kv_local"] + e["kv_global"] + e["kv_nucleo"]) / 1e6, 1)
                row[f"kv_bytes_por_byte@{c // 1024}K"] = round(e["kv_por_byte"], 1)
            for gpu, budget in GPUS.items():
                for wb, wname in ((2.0, "bf16"), (0.56, "int4")):
                    m = max_context(cfg, budget, n_params=n, weight_bytes=wb, kv_bits=bits)
                    row[f"max_{gpu.split()[0]}_{wname}"] = m
            rows.append(row)
    return rows


@torch.no_grad()
def measure(model, n_bytes: int, seed: int = 0) -> dict:
    """Mide los bytes de las cachés después de leer ``n_bytes`` de contexto con ``generate``."""
    rng = np.random.default_rng(seed)
    text = filler(rng, n_bytes).encode()[:n_bytes]
    st: dict = {}
    model.generate(text, 1, max_context=n_bytes + 64, kv_stats=st, prefill_chunk=min(4096, n_bytes))
    return st


def _train_step_peak(preset_name: str, length: int, mode: str, segment: int) -> dict:
    """Un paso de entrenamiento (adelante + atrás) con un texto de ``length`` bytes; se corre en un
    subproceso nuevo y devuelve el pico de memoria residente (CPU)."""
    import resource
    from hope.core.hn1 import HN1Model, preset
    torch.manual_seed(0)
    m = HN1Model(preset(preset_name)).train()
    x = torch.randint(97, 123, (1, length + 1))
    x[0, ::6] = 32
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.time()
    if mode == "completo":
        loss, _ = m.loss(x)
        loss.backward()
    elif mode == "completo+trozos+ckpt":
        m.loss_chunk, m.grad_checkpointing = 1024, True
        loss, _ = m.loss(x)
        loss.backward()
    elif mode in ("segmentos-cache", "segmentos-resumen"):
        m.loss_chunk, m.grad_checkpointing = 1024, True
        m.long_loss(x, segment, memoria=mode.split("-")[1], backward_scale=1.0)
    else:
        raise ValueError(mode)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"preset": preset_name, "largo": length, "modo": mode, "segmento": segment,
            "pico_MB": round(peak / 1024, 1), "pico_sobre_modelo_MB": round((peak - base) / 1024, 1),
            "segundos": round(time.time() - t0, 1)}


def train_memory(preset_name: str = "hn1-cpu-largo", lengths=(2048, 8192, 32768),
                 modes=("completo", "completo+trozos+ckpt", "segmentos-cache"), segment: int = 2048,
                 timeout: float = 1800) -> list[dict]:
    import subprocess
    import sys
    rows = []
    for L in lengths:
        for mode in modes:
            cmd = [sys.executable, "-m", "hope.eval.contexto_bench", "--_paso", f"{preset_name},{L},{mode},{segment}"]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                row = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else \
                    {"preset": preset_name, "largo": L, "modo": mode, "error": r.stderr.strip().splitlines()[-1][:200]}
            except subprocess.TimeoutExpired:
                row = {"preset": preset_name, "largo": L, "modo": mode, "error": "tiempo agotado"}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


# ---------------------------------------------------------------------------
# Entrenamiento corto en CPU (para tener resultados con un modelo chico)
# ---------------------------------------------------------------------------
def _answer_weights(ex: Example, full: list[int], answer_w: float = 1.0, other_w: float = 0.1) -> list[float]:
    w = [other_w] * len(full)
    a = len(_bytes(ex.contexto))
    for i in range(a, len(full)):
        w[i] = answer_w
    return w


def train_cpu(model, minutes: float, lengths=(256, 512, 1024), batch: int = 8, lr: float = 2e-3, seed: int = 123,
              log=print, segment: int | None = None) -> dict:
    """Entrena el modelo unos minutos con las tareas (semillas distintas de las de evaluación).
    La pérdida pesa 1 en la respuesta y 0,1 en el resto. Con ``segment``, los textos más largos se
    leen por segmentos con caché (como en la etapa larga del plan)."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    t0, steps, hist = time.time(), 0, []
    total_steps = None
    while time.time() - t0 < minutes * 60:
        L = int(rng.choice(lengths))
        rows, ws = [], []
        for _ in range(batch):
            t = list(TASKS)[int(rng.integers(len(TASKS)))]
            ex = TASKS[t](rng, L, float(rng.uniform(0.02, 0.98)))
            full = _bytes(ex.contexto + ex.respuesta + ".")
            rows.append(full)
            ws.append(_answer_weights(ex, full))
        width = min(len(r) for r in rows)
        # Recortar por la izquierda para que todas tengan el mismo largo (la respuesta queda al final).
        x = torch.tensor([r[-width:] for r in rows])
        w = torch.tensor([v[-width:] for v in ws])
        frac = (time.time() - t0) / (minutes * 60)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (steps + 1) / 50) * max(0.1, 1 - frac)
        opt.zero_grad(set_to_none=True)
        if segment and x.shape[1] > segment + 1:
            loss, st = model.long_loss(x, segment, memoria="cache", weights=w, backward_scale=1.0)
        else:
            loss, st = model.loss(x, weights=w)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps += 1
        hist.append(st["ce"])
        if log and steps % 50 == 0:
            log(f"paso {steps}  largo {L}  ce respuesta {np.mean(hist[-50:]):.3f}  {time.time() - t0:.0f} s")
    model.eval()
    return {"pasos": steps, "segundos": round(time.time() - t0, 1), "ce_final": float(np.mean(hist[-50:]))}


# ---------------------------------------------------------------------------
def run(model, lengths, depths=(0.1, 0.5, 0.9), n: int = 4, modes=("completo", "cache", "cache-int8",
        "cache-int4", "resumen", "ventana"), gen_modes=("gen", "gen-int8", "gen-quest"), segment: int = 512,
        full_max: int = 8192, seed: int = 0, tasks=("aguja", "varias", "combinar"), quest_pages: int = 8,
        train_bytes: int = 1024, log=print) -> list[dict]:
    """Modos de ranking: ``completo``, ``cache``, ``cache-int8``, ``cache-int4``, ``cache-yarn``,
    ``cache-ntk`` (RoPE escalado por largo / ``train_bytes``), ``resumen``, ``ventana``.
    Modos de generación: ``gen``, ``gen-int8``, ``gen-quest``."""
    from hope.core.hn1 import set_context_scale
    from hope.core.kvcache import configure_kv
    cfg = model.cfg
    rope0 = (cfg.rope_scale, cfg.rope_kind, cfg.rope_train_bytes)
    base_kv = dict(bits=cfg.kv_bits, group=cfg.kv_group, evict=cfg.kv_evict, quest_pages=cfg.quest_pages,
                   page=cfg.quest_page)
    exs = make_examples(tasks, lengths, depths, n, seed)
    rows = []
    for mode in [*modes, *gen_modes]:
        bits = {"cache-int8": 8, "cache-int4": 4, "gen-int8": 8}.get(mode, 16)
        quest = quest_pages if mode == "gen-quest" else 0
        configure_kv(model, **{**base_kv, "bits": bits, "quest_pages": quest, "page": 16})
        for L in lengths:
            if mode == "completo" and L > full_max:
                continue
            if mode in ("cache-yarn", "cache-ntk"):
                set_context_scale(model, max(1.0, L / train_bytes), mode.split("-")[1], train_bytes)
            for task in tasks:
                sub = [e for e in exs if e.largo == L and e.tarea == task]
                t0 = time.time()
                rmode = "cache" if mode in ("cache-yarn", "cache-ntk") else mode
                res = [generate_answer(model, e, chunk=segment) if mode.startswith("gen") else rank(model, e, rmode, segment)
                       for e in sub]
                row = {"modo": mode, "largo": L, "tarea": task, "n": len(res),
                       "acierto": round(float(np.mean([r["ok"] for r in res])), 3),
                       "segundos": round(time.time() - t0, 1)}
                if not mode.startswith("gen"):
                    row["margen_medio"] = round(float(np.mean([r["margen"] for r in res])), 3)
                for d in depths:
                    row[f"prof_{d}"] = round(float(np.mean([r["ok"] for r, e in zip(res, sub) if e.profundidad == d])), 3)
                rows.append(row)
                if log:
                    log(json.dumps(row, ensure_ascii=False))
            set_context_scale(model, *rope0)
    configure_kv(model, **base_kv)
    return rows


def _size(s: str) -> int:
    from hope.data.sources import parse_size
    return parse_size(s.replace("KB", "K"))


def main(argv: list[str] | None = None) -> dict:
    from hope.core.hn1 import HN1Model, preset
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--memoria", action="store_true", help="tabla de memoria y contexto máximo por GPU")
    ap.add_argument("--datos", default=None, help="escribir datos de entrenamiento sintéticos en esta carpeta")
    ap.add_argument("--bytes", default="50M")
    ap.add_argument("--preset", default="hn1-cpu-largo")
    ap.add_argument("--modelo", default=None, help="ligero.pt (export_light) para evaluar")
    ap.add_argument("--entrenar-min", type=float, default=0.0)
    ap.add_argument("--largos-entrenamiento", default="256,512,1K")
    ap.add_argument("--largos", default="1K,2K,4K")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--segmento", type=int, default=512)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--modos", default="completo,cache,cache-int8,cache-int4,cache-yarn,resumen,ventana")
    ap.add_argument("--modos-gen", default="gen,gen-int8,gen-quest")
    ap.add_argument("--guardar", default=None, help="guardar el modelo entrenado (export_light)")
    ap.add_argument("--hilos", type=int, default=None)
    ap.add_argument("--memoria-entrenamiento", default=None, help="preset: pico de memoria al entrenar por largo")
    ap.add_argument("--_paso", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args._paso:
        name, L, mode, seg = args._paso.split(",")
        print(json.dumps(_train_step_peak(name, int(L), mode, int(seg))))
        return {}
    if args.hilos:
        torch.set_num_threads(args.hilos)
    result: dict = {}
    if args.memoria_entrenamiento:
        result["memoria_entrenamiento"] = train_memory(args.memoria_entrenamiento,
                                                       [_size(s) for s in args.largos.split(",")])
    if args.datos:
        result["datos"] = write_training_data(args.datos, _size(args.bytes))
    if args.memoria:
        result["memoria"] = memory_table()
        for r in result["memoria"]:
            print(json.dumps(r, ensure_ascii=False))
    if args.modelo or args.entrenar_min:
        torch.manual_seed(0)
        if args.modelo:
            from hope.train.run import load_light
            model = load_light(args.modelo)
        else:
            model = HN1Model(preset(args.preset))
        if args.entrenar_min:
            result["entrenamiento"] = train_cpu(model, args.entrenar_min,
                                                [_size(s) for s in args.largos_entrenamiento.split(",")])
        result["config"] = {k: v for k, v in model.cfg.to_dict().items()
                            if k in ("kv_heads", "local_kind", "local_window", "window", "pattern", "core_kind",
                                     "kv_evict", "rope_kind", "rope_scale")}
        result["parametros"] = sum(p.numel() for p in model.parameters())
        if args.guardar:
            from hope.train.run import export_light
            export_light(model, Path(args.guardar))
        tb = max(_size(s) for s in args.largos_entrenamiento.split(",")) if args.entrenar_min else model.cfg.rope_train_bytes
        result["banco"] = run(model, [_size(s) for s in args.largos.split(",")], n=args.n, segment=args.segmento,
                              modes=[m for m in args.modos.split(",") if m],
                              gen_modes=[m for m in args.modos_gen.split(",") if m], train_bytes=tb)
        result["medida_cache"] = {L: measure(model, L) for L in (4096, 16384, 65536)}
    if args.salida:
        Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
        Path(args.salida).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
