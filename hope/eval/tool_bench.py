"""Banco sellado de uso de herramientas (docs/HERRAMIENTAS_HN1.md §7.1).

Mide si Hope usa **la herramienta correcta, en el momento justo, rápido y verificado**.
Corre con cualquier ``generate(texto, max_new=..., stop_after_line=..., temperature=...,
stop_seq=...) -> bytes`` (``HN1Model.generate``, un modelo cuantizado, un servidor, o la
política de referencia del currículo), a través de ``generate_with_tools``: cada llamada
``⟦nombre: entrada⟧`` se ejecuta de verdad con el catálogo y el guardián.

Ítems (los arma ``hope/data/tool_curriculum.py`` con una semilla propia y conceptos del
glosario reservados al banco; el currículo excluye estos problemas):

========== ===================================================================
categoría  qué mide
========== ===================================================================
calculo, fraccion, estadistica, unidades, ecuacion, circuito
           elegir la herramienta correcta para el tipo de problema
cadena     encadenar circuito → calc → unidades
trivial    NO llamar herramientas cuando responder directo es mejor
error      recuperarse: la primera llamada falla a propósito y hay que reintentar
definicion consultar ⟦nota⟧/⟦memoria⟧ en vez de inventar la definición
codigo     escribir código y usar ⟦pruebas⟧; se juzga con pruebas ocultas
========== ===================================================================

Métricas (``summarize``):

* ``herramienta_correcta``: la primera llamada (en ítems que necesitan herramienta) es una
  de las correctas.
* ``argumentos_validos``: llamadas que la herramienta aceptó (sin contar las fallas inyectadas).
* ``argumentos_del_enunciado``: la entrada de la primera llamada sale del enunciado (sus números
  están en el problema, o el concepto de ``⟦nota⟧`` aparece en él). Separa «eligió bien la
  herramienta pero copió mal los datos» de «eligió mal».
* ``exactitud``: respuesta final correcta (número con unidad, texto, o pruebas ocultas en código).
* ``llamadas_innecesarias``: en ítems triviales, fracción con alguna llamada.
* ``llamadas_faltantes``: en ítems que necesitan herramienta, fracción sin ninguna llamada;
  ``cadena_completa``: en ``cadena``, se usaron las tres herramientas.
* ``segundos`` y ``bytes_generados`` (solo los que escribe Hope; los resultados no cuentan).
* ``recuperacion``: en ``error``, volvió a llamar después de la falla **y** acertó.

Sellado: el archivo del banco tiene su SHA-256 al lado (``.sha256``); si cambia, ``load_bench``
falla. Con ``--registro`` también se usa ``SealedRegistry`` (propósito «evaluation», con registro
de accesos). Uso::

    python -m hope.eval.tool_bench crear                    # escribe y sella evals/banco_herramientas_v1.jsonl
    python -m hope.eval.tool_bench correr --modelo runs/x/ligero.pt
    python -m hope.eval.tool_bench correr --referencia      # cota superior: la solución de referencia
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from hope.data.tool_curriculum import (BUILDERS, FaultInjectingRunner, ToolExample, Writer, answer_ok,
                                       default_notebook, extract_answer)
from hope.runtime.toolcalls import CLOSE, generate_with_tools

DEFAULT_BENCH = "evals/banco_herramientas_v1.jsonl"
BENCH_SEED = "banco-herramientas-v1"
CATEGORIES = ("calculo", "fraccion", "estadistica", "unidades", "ecuacion", "circuito", "cadena", "trivial",
              "error", "definicion", "codigo")


class BenchTampered(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Construcción, guardado y sellado
# ---------------------------------------------------------------------------
def build_bench(per_category: int = 10, seed: str = BENCH_SEED, runner=None) -> list[ToolExample]:
    from hope.data.tool_curriculum import make_runner
    runner = runner or make_runner()
    rng = random.Random(f"tool_bench:{seed}")
    items, seen = [], set()
    for cat in CATEGORIES:
        made, tries = 0, 0
        while made < per_category and tries < per_category * 20:
            tries += 1
            w = Writer(runner)
            if cat == "definicion":
                ex = BUILDERS[cat](rng, w, split="banco")
            elif cat == "error":                       # en el banco la falla siempre es inyectada
                base = rng.choice([BUILDERS[c] for c in ("calculo", "unidades", "fraccion", "ecuacion")])
                from hope.data.tool_curriculum import _RetryWriter
                w = Writer(FaultInjectingRunner(runner.catalog, memory=runner.memory, fail_next=1))
                ex = base(rng, _RetryWriter(w))
                if ex is not None:
                    ex.categoria, ex.falla_inyectada, ex.segmentos = "error", 1, w.segs
            else:
                ex = BUILDERS[cat](rng, w)
            if ex is None or ex.problema in seen:
                continue
            seen.add(ex.problema)
            ex.origen = "banco"
            items.append(ex)
            made += 1
    return items


def subset_per_category(items: list[ToolExample], n: int) -> list[ToolExample]:
    """Los primeros ``n`` ítems de cada categoría (evaluaciones rápidas)."""
    count: dict[str, int] = {}
    out = []
    for it in items:
        if count.get(it.categoria, 0) < n:
            count[it.categoria] = count.get(it.categoria, 0) + 1
            out.append(it)
    return out


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_bench(items: list[ToolExample], path: str | Path = DEFAULT_BENCH) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(it.to_json(), ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    digest = sha256_of(p)
    p.with_suffix(p.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def load_bench(path: str | Path = DEFAULT_BENCH, *, check: bool = True, registry=None, name: str | None = None,
               requester: str = "tool_bench") -> list[ToolExample]:
    """Lee el banco verificando el sello. Con ``registry`` (``SealedRegistry``) registra el acceso."""
    if registry is not None:
        path = registry.open(name or "banco_herramientas_v1", purpose="evaluation", requester=requester)
    p = Path(path)
    if check:
        side = p.with_suffix(p.suffix + ".sha256")
        if not side.exists() or side.read_text().strip() != sha256_of(p):
            raise BenchTampered(f"el banco {p} no coincide con su sello")
    return [ToolExample.from_json(json.loads(x)) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
@dataclass
class ItemResult:
    categoria: str
    problema: str
    esperado: str
    respuesta: str | None
    correcta: bool
    llamadas: list[dict]
    herramienta_correcta: bool | None
    necesita: bool
    cadena_completa: bool | None
    recupero: bool | None
    argumentos_del_enunciado: bool | None
    segundos: float
    bytes_generados: int
    texto: str = field(repr=False, default="")


def reference_generate(items: list[ToolExample]) -> Callable:
    """Política de referencia: reproduce la solución del currículo (cota superior del banco)."""
    by_prompt = {it.prompt: it.model_segments for it in items}

    def gen(text, max_new=400, **kw) -> bytes:
        prompt = next(p for p in sorted(by_prompt, key=len, reverse=True) if text.startswith(p))
        produced = text[len(prompt):]
        k = produced.count(CLOSE)
        segs = by_prompt[prompt]
        return segs[k].encode("utf-8")[:max_new] if k < len(segs) else b""
    return gen


def model_generate(model) -> Callable:
    return model.generate


_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def grounded_args(problem: str, call_input: str) -> bool:
    """¿La entrada de la primera llamada sale del enunciado? Números: todos los de la llamada están
    en el problema. Sin números (⟦nota⟧): el concepto aparece en el problema."""
    from hope.data.tool_curriculum import _norm_text
    nums = [n.replace(",", ".") for n in _NUM.findall(call_input)]
    if nums:
        have = {n.replace(",", ".") for n in _NUM.findall(problem)}
        return all(n in have for n in nums)
    return bool(_norm_text(call_input)) and _norm_text(call_input) in _norm_text(problem)


def _check_code(item: ToolExample, calls) -> bool:
    from hope.tools.code_tools import run_tests
    code_calls = [c for c in calls if c.name == "pruebas"]
    if not code_calls:
        return False
    code = code_calls[-1].input.split("|||")[0].strip()
    try:
        return bool(run_tests(code, item.pruebas_ocultas, timeout_s=30)["ok"])
    except Exception:
        return False


def run_item(generate: Callable, item: ToolExample, *, catalog, notebook, max_new: int = 700, max_calls: int = 6,
             tlm=None, temperature: float = 0.0) -> ItemResult:
    runner = FaultInjectingRunner(catalog, memory=notebook, fail_next=item.falla_inyectada)
    prompt = item.prompt
    if tlm is not None and not item.pista:
        hint = tlm.context_hint(item.tema, item.problema)
        prompt += f"{hint}\n" if hint else ""
    t0 = time.perf_counter()
    text = generate_with_tools(None, prompt, runner, max_new=max_new, max_calls=max_calls, temperature=temperature,
                               generate=generate)
    secs = time.perf_counter() - t0
    tool_bytes = sum(len((" → " + c.output).encode()) for c in runner.calls)
    calls = runner.calls
    real = [c for i, c in enumerate(calls) if i not in runner.injected]
    ans = extract_answer(text)
    if item.comparar == "codigo":
        correct = _check_code(item, real)
    else:
        correct = answer_ok(ans, item.respuesta, item.comparar)
    first = calls[0].name if calls else None
    tool_ok = (first in item.herramientas_ok) if item.necesita_herramienta and calls else (
        False if item.necesita_herramienta else None)
    chain = (set(item.cadena) <= {c.name for c in real}) if item.categoria == "cadena" else None
    grounded = (grounded_args(item.problema, real[0].input)
                if real and item.necesita_herramienta and item.categoria != "codigo" else None)
    recovered = None
    if item.falla_inyectada:
        recovered = len(calls) > len(runner.injected) and correct
    if tlm is not None:
        if real:
            tlm.record(real, item.tema, correct)
        elif not calls:
            tlm.record_direct(item.tema, correct)
    return ItemResult(item.categoria, item.problema, item.respuesta, ans, correct,
                      [{"herramienta": c.name, "entrada": c.input[:200], "salida": c.output[:200], "ok": c.ok,
                        "inyectada": i in runner.injected, "segundos": round(c.seconds, 4)}
                       for i, c in enumerate(calls)],
                      tool_ok, item.necesita_herramienta, chain, recovered, grounded, secs,
                      max(len(text.encode()) - tool_bytes, 0), text)


def run_bench(generate: Callable, items: list[ToolExample], *, catalog=None, max_new: int = 700, max_calls: int = 6,
              tlm=None, log=None) -> dict:
    from hope.tools.builtin import default_catalog
    catalog = catalog or default_catalog()
    notebook = default_notebook()
    results = []
    for i, it in enumerate(items):
        results.append(run_item(generate, it, catalog=catalog, notebook=notebook, max_new=max_new,
                                max_calls=max_calls, tlm=tlm))
        if log and (i + 1) % 20 == 0:
            log(f"  banco: {i + 1}/{len(items)}")
    return {"resumen": summarize(results), "items": [r.__dict__ for r in results]}


def _rate(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(map(bool, xs)) / len(xs), 4) if xs else None


def summarize(results: list[ItemResult]) -> dict:
    need = [r for r in results if r.necesita]
    trivial = [r for r in results if not r.necesita]
    calls = [c for r in results for c in r.llamadas if not c["inyectada"]]
    n = max(len(results), 1)
    out = {
        "items": len(results),
        "exactitud": _rate(r.correcta for r in results),
        "herramienta_correcta": _rate(r.herramienta_correcta for r in need),
        "argumentos_validos": _rate(c["ok"] for c in calls),
        "argumentos_del_enunciado": _rate(r.argumentos_del_enunciado for r in results),
        "llamadas_por_item": round(len(calls) / n, 3),
        "llamadas_innecesarias": _rate(len(r.llamadas) > 0 for r in trivial),
        "llamadas_faltantes": _rate(len(r.llamadas) == 0 for r in need),
        "cadena_completa": _rate(r.cadena_completa for r in results),
        "recuperacion": _rate(r.recupero for r in results),
        "segundos_por_item": round(sum(r.segundos for r in results) / n, 4),
        "bytes_generados_por_item": round(sum(r.bytes_generados for r in results) / n, 1),
        "por_categoria": {},
    }
    for cat in sorted({r.categoria for r in results}):
        rs = [r for r in results if r.categoria == cat]
        out["por_categoria"][cat] = {"items": len(rs), "exactitud": _rate(r.correcta for r in rs),
                                     "herramienta_correcta": _rate(r.herramienta_correcta for r in rs),
                                     "llamadas_por_item": round(sum(len(r.llamadas) for r in rs) / len(rs), 2),
                                     "bytes": round(sum(r.bytes_generados for r in rs) / len(rs), 1)}
    return out


def main(argv: list[str] | None = None) -> dict:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crear")
    c.add_argument("--out", default=DEFAULT_BENCH)
    c.add_argument("--por-categoria", type=int, default=10)
    c.add_argument("--registro", default=None, help="carpeta de SealedRegistry donde sellarlo")
    r = sub.add_parser("correr")
    r.add_argument("--banco", default=DEFAULT_BENCH)
    r.add_argument("--modelo", default=None, help="pesos livianos (hope.train.run.export_light)")
    r.add_argument("--referencia", action="store_true", help="usar la solución de referencia (cota superior)")
    r.add_argument("--tlm", default=None, help="memoria de uso (JSON) para sugerir herramientas en el contexto")
    r.add_argument("--max-new", type=int, default=700)
    r.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "crear":
        if Path(args.out).exists():                # un banco sellado no se rehace: se crea uno nuevo (v2, …)
            raise SystemExit(f"{args.out} ya existe; para renovarlo usar otro nombre (--out …_v2.jsonl)")
        items = build_bench(args.por_categoria)
        digest = save_bench(items, args.out)
        if args.registro:
            from hope.eval.sealed import SealedRegistry
            SealedRegistry(args.registro).seal("banco_herramientas_v1", args.out,
                                               description="banco de uso de herramientas")
        res = {"items": len(items), "sha256": digest, "archivo": args.out}
        print(json.dumps(res, ensure_ascii=False))
        return res
    items = load_bench(args.banco)
    if args.referencia:
        gen = reference_generate(items)
    else:
        from hope.train.run import load_light
        gen = model_generate(load_light(args.modelo))
    tlm = None
    if args.tlm:
        from hope.tools.tlm import ToolUsageMemory
        tlm = ToolUsageMemory(args.tlm)
    res = run_bench(gen, items, max_new=args.max_new, tlm=tlm, log=print)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res["resumen"], indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
