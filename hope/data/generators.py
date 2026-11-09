"""Generadores procedurales de problemas verificables (propuesta 2, aprobada por el usuario).

Producen episodios **con el mismo formato que los de Qwen** (``hope.teacher.episode``):
problema, datos, plan, pasos con su verificación, resultado, comprobación
inversa, errores frecuentes y variaciones. La respuesta es exacta y cada paso se
puede comprobar con herramientas, así que salen ``RESPALDADO`` sin depender del
maestro. Son gratis e ilimitados; Qwen queda para explicar lo que no se puede
generar así (HLA §16: priorizar ejemplos que enseñan principios).

Familias (español e inglés): aritmética, porcentajes, ecuaciones lineales,
ley de Ohm y potencia, resistencias en serie y paralelo, conversión de
unidades, lógica proposicional, predicción de salida de programas Python y
problemas con datos faltantes o contradictorios (enseñan a declarar la limitación).
"""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Callable

from hope.teacher.episode import Episode


def _fmt(x: float | Fraction) -> str:
    if isinstance(x, Fraction):
        x = float(x)
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def _ep(topic: str, lang: str, problem: str, steps: list[dict], result: str, *, data=None, assumptions=None,
        plan: str = "", back: str = "", errors=None, variations=None) -> dict:
    return {"tema": topic, "idioma": lang, "problema": problem, "datos": data or [], "supuestos": assumptions or [],
            "plan": plan, "pasos": steps, "resultado": result, "verificacion_inversa": back,
            "errores_frecuentes": errors or [], "variaciones": variations or [], "origen": "generador"}


def _step(n: int, op: str, res: str, why: str, check: dict | None = None) -> dict:
    s = {"n": n, "operacion": op, "resultado_intermedio": res, "justificacion": why}
    if check:
        s["verificacion"] = check
    return s


# --- familias ---------------------------------------------------------------------
def arithmetic(rng: random.Random, level: int, lang: str) -> dict:
    op = rng.choice(["+", "-", "*", "/"] if level > 1 else ["+", "-", "*"])
    a = rng.randint(2, 10 ** (level + 1))
    b = rng.randint(2, 10 ** level)
    if op == "/":
        a = a * b                                     # división exacta
    val = {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op]
    word = {"+": ("suma", "addition"), "-": ("resta", "subtraction"), "*": ("multiplicación", "multiplication"),
            "/": ("división", "division")}[op][lang == "en"]
    q = f"¿Cuánto es {a} {op} {b}?" if lang == "es" else f"What is {a} {op} {b}?"
    var = [{"problema": (f"¿Cuánto es {a + 1} {op} {b}?" if lang == "es" else f"What is {a + 1} {op} {b}?"),
            "respuesta": _fmt(eval(f"{a + 1}{op}{b}"))}] if op != "/" else []    # noqa: S307 (números propios)
    return _ep(f"aritmética: {word}" if lang == "es" else f"arithmetic: {word}", lang, q,
               [_step(1, f"{a} {op} {b}", str(val), word, {"tipo": "aritmetica", "expr": f"{a} {op} {b}",
                                                             "esperado": str(val)})], str(val),
               data=[{"valor": a}, {"valor": b}], plan=f"Aplicar la {word}." if lang == "es" else f"Apply {word}.",
               back=(f"{val} {'-+'[op == '-'] if op in '+-' else ('/' if op == '*' else '*')} {b} = {a}"),
               variations=var)


def percentage(rng: random.Random, level: int, lang: str) -> dict:
    p = rng.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 75])
    base = rng.randint(2, 50) * 20
    val = Fraction(p * base, 100)
    q = (f"¿Cuánto es el {p}% de {base}?" if lang == "es" else f"What is {p}% of {base}?")
    return _ep("porcentajes" if lang == "es" else "percentages", lang, q,
               [_step(1, f"{p} / 100", _fmt(Fraction(p, 100)), "pasar el porcentaje a fracción",
                      {"tipo": "aritmetica", "expr": f"{p} / 100", "esperado": _fmt(Fraction(p, 100))}),
                _step(2, f"{_fmt(Fraction(p, 100))} * {base}", _fmt(val), "multiplicar por la cantidad",
                      {"tipo": "aritmetica", "expr": f"{_fmt(Fraction(p, 100))} * {base}", "esperado": _fmt(val)})],
               _fmt(val), plan="p% de N = p/100 · N", back=f"{_fmt(val)} / {base} · 100 = {p}",
               errors=["dividir por 10 en lugar de 100"],
               variations=[{"problema": (f"¿Cuánto es el {p}% de {base * 2}?" if lang == "es"
                                         else f"What is {p}% of {base * 2}?"), "respuesta": _fmt(val * 2)}])


def linear_equation(rng: random.Random, level: int, lang: str) -> dict:
    a, x = rng.choice([2, 3, 4, 5, 6, 7, 8, 9]), rng.randint(-10, 20)
    b = rng.randint(-20, 20)
    c = a * x + b
    q = (f"Resolver {a}x + {b} = {c}" if lang == "es" else f"Solve {a}x + {b} = {c}").replace("+ -", "- ")
    return _ep("ecuaciones lineales" if lang == "es" else "linear equations", lang, q,
               [_step(1, f"{c} - {b}", str(c - b), "restar el término independiente a ambos lados",
                      {"tipo": "aritmetica", "expr": f"{c} - ({b})", "esperado": str(c - b)}),
                _step(2, f"{c - b} / {a}", str(x), "dividir por el coeficiente",
                      {"tipo": "aritmetica", "expr": f"({c - b}) / {a}", "esperado": str(x)}),
                _step(3, f"{a}*x + {b} - {c}", "0 con x = " + str(x), "comprobar la solución",
                      {"tipo": "simbolico", "lhs": f"{a}*({x}) + ({b})", "rhs": str(c)})],
               f"x = {x}", plan="Despejar x.", back=f"{a}·{x} + {b} = {c}",
               errors=["cambiar mal el signo al pasar de lado"],
               variations=[{"problema": (f"Resolver {a}x + {b} = {c + a}" if lang == "es"
                                         else f"Solve {a}x + {b} = {c + a}").replace("+ -", "- "),
                            "respuesta": f"x = {x + 1}"}])


def ohm(rng: random.Random, level: int, lang: str) -> dict:
    r = rng.choice([10, 47, 100, 150, 220, 330, 470, 1000])
    i = rng.choice([0.01, 0.02, 0.05, 0.1, 0.2])
    v = r * i
    kind = rng.choice(["tension", "corriente", "potencia"] if level > 1 else ["tension", "corriente"])
    if kind == "tension":
        q = f"¿Qué tensión cae en un resistor de {r} ohm con {_fmt(i)} A?"
        steps = [_step(1, f"{r} * {_fmt(i)}", _fmt(v), "ley de Ohm V = I · R",
                       {"tipo": "unidades", "expr": f"{r} ohm * {_fmt(i)} A", "esperado": f"{_fmt(v)} V"})]
        res, back = f"{_fmt(v)} V", f"{_fmt(v)} V / {r} ohm = {_fmt(i)} A"
        var = f"¿Qué tensión cae en un resistor de {r * 2} ohm con {_fmt(i)} A?", f"{_fmt(2 * v)} V"
    elif kind == "corriente":
        q = f"¿Qué corriente circula por un resistor de {r} ohm con {_fmt(v)} V?"
        steps = [_step(1, f"{_fmt(v)} / {r}", _fmt(i), "ley de Ohm I = V / R",
                       {"tipo": "unidades", "expr": f"{_fmt(v)} V / ({r} ohm)", "esperado": f"{_fmt(i)} A"})]
        res, back = f"{_fmt(i)} A", f"{_fmt(i)} A · {r} ohm = {_fmt(v)} V"
        var = f"¿Qué corriente circula por un resistor de {r} ohm con {_fmt(2 * v)} V?", f"{_fmt(2 * i)} A"
    else:
        p = i * i * r
        q = f"¿Qué potencia disipa un resistor de {r} ohm si circulan {_fmt(i)} A?"
        steps = [_step(1, f"{_fmt(i)} * {_fmt(i)}", _fmt(i * i), "elevar la corriente al cuadrado",
                       {"tipo": "aritmetica", "expr": f"{_fmt(i)} * {_fmt(i)}", "esperado": _fmt(i * i)}),
                 _step(2, f"{_fmt(i * i)} * {r}", _fmt(p), "P = I² · R",
                       {"tipo": "unidades", "expr": f"({_fmt(i)} A)**2 * {r} ohm", "esperado": f"{_fmt(p)} W"})]
        res, back = f"{_fmt(p)} W", f"{_fmt(p)} W = {_fmt(v)} V · {_fmt(i)} A"
        var = f"¿Qué potencia disipa un resistor de {r} ohm si circulan {_fmt(2 * i)} A?", f"{_fmt(4 * p)} W"
    return _ep("ley de Ohm y potencia eléctrica", "es", q, steps, res,
               data=[{"valor": r, "unidad": "ohm"}], assumptions=["resistor óhmico", "corriente continua"],
               plan="Identificar la incógnita y aplicar la ley correspondiente.", back=back,
               errors=["confundir miliamperios con amperios", "dar el resultado sin unidades"],
               variations=[{"problema": var[0], "respuesta": var[1]}])


def resistor_network(rng: random.Random, level: int, lang: str) -> dict:
    r1, r2 = rng.choice([100, 220, 330, 470, 1000]), rng.choice([100, 220, 330, 470, 1000])
    if rng.random() < 0.5:
        req = r1 + r2
        return _ep("resistencias en serie y en paralelo", "es",
                   f"¿Cuál es la resistencia equivalente de {r1} ohm y {r2} ohm en serie?",
                   [_step(1, f"{r1} + {r2}", str(req), "en serie se suman",
                          {"tipo": "aritmetica", "expr": f"{r1} + {r2}", "esperado": str(req)})],
                   f"{req} ohm", plan="Serie: R = R1 + R2.", back=f"{req} - {r2} = {r1}",
                   errors=["usar la fórmula de paralelo"],
                   variations=[{"problema": f"¿Cuál es la resistencia equivalente de {r1} ohm y {r1} ohm en serie?",
                                "respuesta": f"{2 * r1} ohm"}])
    req = Fraction(r1 * r2, r1 + r2)
    return _ep("resistencias en serie y en paralelo", "es",
               f"¿Cuál es la resistencia equivalente de {r1} ohm y {r2} ohm en paralelo?",
               [_step(1, f"{r1} * {r2}", str(r1 * r2), "producto",
                      {"tipo": "aritmetica", "expr": f"{r1} * {r2}", "esperado": str(r1 * r2)}),
                _step(2, f"{r1} + {r2}", str(r1 + r2), "suma",
                      {"tipo": "aritmetica", "expr": f"{r1} + {r2}", "esperado": str(r1 + r2)}),
                _step(3, f"{r1 * r2} / {r1 + r2}", _fmt(req), "producto sobre suma",
                      {"tipo": "aritmetica", "expr": f"{r1 * r2} / {r1 + r2}", "esperado": _fmt(req),
                       "tolerancia": 1e-6})],
               f"{_fmt(req)} ohm", plan="Paralelo de dos: R = R1·R2 / (R1 + R2).",
               back="la equivalente en paralelo es menor que la menor de las dos",
               errors=["sumar las resistencias en paralelo"],
               variations=[{"problema": f"¿Cuál es la resistencia equivalente de {r1} ohm y {r1} ohm en paralelo?",
                            "respuesta": f"{_fmt(Fraction(r1, 2))} ohm"}])


def voltage_divider(rng: random.Random, level: int, lang: str) -> dict:
    vin = rng.choice([5, 9, 12, 24])
    r1, r2 = rng.choice([1000, 2200, 4700, 10000]), rng.choice([1000, 2200, 4700, 10000])
    vout = Fraction(vin * r2, r1 + r2)
    net = f"V1 in 0 {vin}; R1 in out {r1}; R2 out 0 {r2}"
    return _ep("divisores de tensión y corriente", "es",
               f"Un divisor con R1 = {r1} ohm (arriba) y R2 = {r2} ohm (abajo) se alimenta con {vin} V. "
               f"¿Qué tensión hay en la salida?",
               [_step(1, f"{r1} + {r2}", str(r1 + r2), "resistencia total en serie",
                      {"tipo": "aritmetica", "expr": f"{r1} + {r2}", "esperado": str(r1 + r2)}),
                _step(2, f"{vin} * {r2} / {r1 + r2}", _fmt(vout), "Vout = Vin · R2 / (R1 + R2)",
                      {"tipo": "aritmetica", "expr": f"{vin} * {r2} / {r1 + r2}", "esperado": _fmt(vout),
                       "tolerancia": 1e-6}),
                _step(3, "simular el circuito", f"{_fmt(vout)} V", "verificación independiente por simulación",
                      {"tipo": "circuito", "netlist": net, "nodo": "out", "esperado": _fmt(vout)})],
               f"{_fmt(vout)} V", data=[{"valor": vin, "unidad": "V"}], assumptions=["salida sin carga"],
               plan="Divisor de tensión sin carga.", back=f"{_fmt(vout)} V + caída en R1 = {vin} V",
               errors=["usar R1 en el numerador", "olvidar que una carga cambia la salida"],
               variations=[{"problema": f"Un divisor con R1 = {r2} ohm (arriba) y R2 = {r1} ohm (abajo) se alimenta "
                                        f"con {vin} V. ¿Qué tensión hay en la salida?",
                            "respuesta": f"{_fmt(Fraction(vin * r1, r1 + r2))} V"}])


def unit_conversion(rng: random.Random, level: int, lang: str) -> dict:
    cases = [("mA", "A", 0.001), ("kohm", "ohm", 1000), ("mV", "V", 0.001), ("km", "m", 1000), ("ms", "s", 0.001)]
    a, b, f = rng.choice(cases)
    x = rng.randint(1, 999)
    y = x * f
    return _ep("unidades del SI y conversiones", "es", f"¿Cuántos {b} son {x} {a}?",
               [_step(1, f"{x} * {_fmt(f)}", _fmt(y), f"1 {a} = {_fmt(f)} {b}",
                      {"tipo": "unidades", "expr": f"{x} {a}", "esperado": f"{_fmt(y)} {b}"})],
               f"{_fmt(y)} {b}", plan="Multiplicar por el factor de conversión.",
               back=f"{_fmt(y)} {b} / {_fmt(f)} = {x} {a}", errors=["multiplicar en lugar de dividir"],
               variations=[{"problema": f"¿Cuántos {b} son {x * 2} {a}?", "respuesta": f"{_fmt(2 * y)} {b}"}])


def propositional_logic(rng: random.Random, level: int, lang: str) -> dict:
    p, q = rng.choice([True, False]), rng.choice([True, False])
    form = rng.choice(["p and q", "p or q", "not p or q", "(p or q) and not (p and q)"])
    val = eval(form, {}, {"p": p, "q": q})        # noqa: S307 (fórmulas fijas propias)
    tf = {True: "verdadero", False: "falso"}
    pretty = form.replace("and", "y").replace("or", "o").replace("not ", "no ")
    return _ep("lógica proposicional: implicación y contraejemplos", "es",
               f"Si p es {tf[p]} y q es {tf[q]}, ¿cuál es el valor de «{pretty}»?",
               [_step(1, f"{pretty} con p={tf[p]}, q={tf[q]}", tf[val], "evaluar con la tabla de verdad",
                      {"tipo": "codigo", "codigo": f"p, q = {p}, {q}\nprint({form})", "salida_esperada": str(val)})],
               tf[val], plan="Reemplazar y evaluar de adentro hacia afuera.",
               errors=["confundir «o» inclusivo con exclusivo"],
               variations=[{"problema": f"Si p es {tf[not p]} y q es {tf[q]}, ¿cuál es el valor de «{pretty}»?",
                            "respuesta": tf[eval(form, {}, {'p': not p, 'q': q})]}])     # noqa: S307


def python_output(rng: random.Random, level: int, lang: str) -> dict:
    n, k = rng.randint(3, 8), rng.randint(1, 4)
    kind = rng.choice(["suma", "lista", "condicional"])
    if kind == "suma":
        code = f"t = 0\nfor i in range({n}):\n    t += i * {k}\nprint(t)"
        out = sum(i * k for i in range(n))
    elif kind == "lista":
        code = f"xs = [i * {k} for i in range({n}) if i % 2 == 0]\nprint(len(xs))"
        out = len([i for i in range(n) if i % 2 == 0])
    else:
        code = f"x = {n * k}\nif x > 10:\n    print(x - 10)\nelse:\n    print(x + 10)"
        out = n * k - 10 if n * k > 10 else n * k + 10
    return _ep("Python: variables, condiciones y bucles", "es", f"¿Qué imprime este programa?\n{code}",
               [_step(1, "ejecutar mentalmente el programa línea por línea", str(out), "seguir el valor de cada variable",
                      {"tipo": "codigo", "codigo": code, "salida_esperada": str(out)})],
               str(out), plan="Rastrear las variables en cada iteración.",
               errors=["olvidar que range(n) termina en n − 1"],
               variations=[{"problema": f"¿Qué imprime este programa?\n{code.replace(str(k), str(k + 1), 1)}",
                            "respuesta": "verificar ejecutando"}] if False else [])


def missing_or_contradictory(rng: random.Random, level: int, lang: str) -> dict:
    r = rng.choice([100, 220, 470])
    if rng.random() < 0.5:
        q = f"¿Qué corriente circula por un resistor de {r} ohm?"
        return _ep("detectar datos faltantes o contradictorios en un problema", "es", q,
                   [_step(1, "identificar datos", f"solo R = {r} ohm", "la ley de Ohm necesita dos de V, I, R")],
                   "faltan datos: la tensión aplicada", plan="Comprobar si los datos alcanzan antes de calcular.",
                   errors=["inventar un valor de tensión"],
                   variations=[{"problema": f"¿Qué tensión cae en un resistor de {r * 2} ohm?",
                                "respuesta": "faltan datos: la corriente"}])
    v, i = rng.choice([5, 9, 12]), rng.choice([1, 2, 3])
    return _ep("detectar datos faltantes o contradictorios en un problema", "es",
               f"Un resistor de {r} ohm tiene {v} V y circulan {i} A. ¿Qué potencia disipa?",
               [_step(1, f"{r} * {i}", str(r * i), "con R e I la tensión debería ser R·I",
                      {"tipo": "aritmetica", "expr": f"{r} * {i}", "esperado": str(r * i)}),
                _step(2, f"comparar {r * i} V con {v} V", "no coinciden", "los datos violan la ley de Ohm")],
               "datos contradictorios: no cumplen la ley de Ohm", plan="Verificar la consistencia de los datos.",
               errors=["calcular igual con dos de los tres datos"],
               variations=[{"problema": f"Un resistor de {r * 2} ohm tiene {v} V y circulan {i} A. ¿Qué potencia disipa?",
                            "respuesta": "datos contradictorios"}])


FAMILIES: dict[str, Callable[[random.Random, int, str], dict]] = {
    "aritmetica": arithmetic, "porcentajes": percentage, "ecuaciones": linear_equation, "ohm": ohm,
    "resistencias": resistor_network, "divisores": voltage_divider, "unidades": unit_conversion, "logica": propositional_logic,
    "python": python_output, "datos_faltantes": missing_or_contradictory,
}

# Palabras del nombre de un tema del currículo → familia
TOPIC_KEYWORDS = {
    "aritmetica": ["suma", "resta", "multiplica", "divisi", "arithmetic"], "porcentajes": ["porcentaje", "percent"],
    "ecuaciones": ["ecuacion", "ecuación", "equation"], "ohm": ["ohm", "potencia eléctrica"],
    "resistencias": ["serie", "paralelo"], "divisores": ["divisor"], "unidades": ["unidades", "conversion", "conversión"],
    "logica": ["lógica", "logica", "logic"], "python": ["python"], "datos_faltantes": ["faltantes", "contradictorios"],
}


def family_for(topic: str) -> str | None:
    t = topic.lower()
    for fam, words in TOPIC_KEYWORDS.items():
        if any(w in t for w in words):
            return fam
    return None


def generate(family: str, n: int, *, seed: int = 0, level: int = 1, lang: str = "es") -> list[Episode]:
    rng = random.Random(f"{family}:{seed}:{level}:{lang}")
    return [Episode(FAMILIES[family](rng, level, lang), source_model="generador") for _ in range(n)]


def main(argv: list[str] | None = None) -> dict:
    """Escribe un corpus generado: python -m hope.data.generators --n 20000 --out data/generado"""
    import argparse
    import hashlib
    import json
    from pathlib import Path

    from hope.tools.builtin import default_catalog
    from hope.tools.verify import verify_episode

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--n", type=int, default=1000, help="episodios por familia, nivel e idioma")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--langs", default="es,en")
    ap.add_argument("--val-percent", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sin-herramientas", action="store_true",
                    help="escribir los pasos sin llamadas a herramientas")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cat = default_catalog()
    files = {"train": open(out / "train.bin", "wb"), "val": open(out / "val.bin", "wb")}
    stats = {"train": 0, "val": 0, "refutados": 0}
    with open(out / "episodes.jsonl", "w", encoding="utf-8") as js:
        for fam in FAMILIES:
            for level in map(int, args.levels.split(",")):
                for lang in args.langs.split(","):
                    for ep in generate(fam, args.n, seed=args.seed, level=level, lang=lang):
                        verify_episode(cat, ep)
                        if ep.verification["status"] == "REFUTADO":
                            stats["refutados"] += 1
                            continue
                        text = ep.training_text(tool_calls=not args.sin_herramientas)
                        h = hashlib.sha256(text.encode()).hexdigest()
                        split = "val" if int(h[:8], 16) % 10_000 < args.val_percent * 100 else "train"
                        files[split].write(text.encode("utf-8") + b"\x00")
                        stats[split] += 1
                        js.write(json.dumps({"split": split, **ep.data, "verificacion": ep.verification},
                                            ensure_ascii=False) + "\n")
    for f in files.values():
        f.close()
    print(json.dumps(stats))
    return stats


if __name__ == "__main__":
    main()
