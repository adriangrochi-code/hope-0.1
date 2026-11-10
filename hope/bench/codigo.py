"""Currículo y banco de código: problemas con pruebas ocultas, pass@k, tiempo y tokens; refuerzo.

**Problemas** (estilo HumanEval/MBPP, escritos para este repositorio, licencia del repo):

* ``bank()`` — ejercicios propios en español: firma, enunciado, solución de referencia y
  entradas. Las salidas esperadas **no se escriben a mano**: se calculan ejecutando la solución
  de referencia en el sandbox (``validate``), así una prueba oculta nunca está mal.
* ``generated_problems(n)`` — familias paramétricas (múltiplos, filtros, sucesiones, cadenas…)
  con parámetros al azar: currículo ilimitado para el refuerzo, separado de la evaluación.
* ``load_mbpp(ruta)`` — si se descarga MBPP (CC BY 4.0) o un JSONL con el mismo formato
  (``text``, ``code``, ``test_list``), se convierte al mismo esquema con su licencia.

Cada problema tiene **ejemplos públicos** (los primeros 2: van en el enunciado y son las anclas
de la autodepuración) y **pruebas ocultas** (el resto: solo para puntuar).

**Métricas** (``evaluate``): pass@1 y pass@k con el estimador insesgado de Chen et al. (2021),
% de muestras sintácticamente válidas, segundos y tokens (bytes generados / 4) por problema.
Modos: ``plano`` (generación directa), ``gramatica`` (``hope/core/gramatica.py``) y
``autodepuracion`` (``hope/runtime/autodepurar.py``), combinables.

**Refuerzo** (``rl_step``): recompensa de ejecución = fracción de pruebas **públicas** que pasan
(las ocultas no se usan para entrenar), con ``grpo_step_tools`` de ``hope/train/rlvr.py``
(misma pérdida Dr. GRPO, recompensa con costo opcional). ``--rl --prueba-cpu`` corre un paso
con un modelo diminuto para comprobar que todo funciona en CPU.

Uso::

    python -m hope.bench.codigo --checkpoint checkpoints/hn1-25m-cpu.pt --n 5 --k 1 5 \\
        --modos plano gramatica autodepuracion --out docs/resultados/codigo.json
    python -m hope.bench.codigo --rl --prueba-cpu
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from hope.runtime.autodepurar import CodeTask, run_tests_detailed

# ---------------------------------------------------------------------------
# Banco propio: (id, firma, enunciado, solución, entradas)
# ---------------------------------------------------------------------------
_RAW = [
    ("suma_lista", "def suma_lista(xs):", "Devuelve la suma de los números de la lista xs.",
     "def suma_lista(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t\n",
     ["[1, 2, 3]", "[]", "[-5, 5, 10]", "[7]", "[100, 200, 300, 400]", "[-1, -2, -3]"]),
    ("maximo", "def maximo(xs):", "Devuelve el mayor elemento de la lista no vacía xs, sin usar max.",
     "def maximo(xs):\n    m = xs[0]\n    for x in xs:\n        if x > m:\n            m = x\n    return m\n",
     ["[3, 9, 2]", "[-4, -1, -7]", "[5]", "[1, 1, 1]", "[0, 10, 10, 3]", "[-100, 50, 49]"]),
    ("contar_vocales", "def contar_vocales(s):", "Cuenta las vocales (a, e, i, o, u, sin tilde, mayúsculas o minúsculas) de s.",
     "def contar_vocales(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
     ["'hola'", "'xyz'", "'AEIOU'", "''", "'murcielago'", "'Programar En Python'"]),
    ("es_palindromo", "def es_palindromo(s):", "Devuelve True si s se lee igual de izquierda a derecha que al revés.",
     "def es_palindromo(s):\n    return s == s[::-1]\n",
     ["'neuquen'", "'hola'", "''", "'a'", "'abba'", "'abca'"]),
    ("invertir_palabras", "def invertir_palabras(s):", "Devuelve las palabras de s (separadas por espacios) en orden inverso, unidas por un espacio.",
     "def invertir_palabras(s):\n    return ' '.join(reversed(s.split()))\n",
     ["'hola mundo'", "'uno'", "'a b c d'", "''", "'el perro come'", "'  dos   espacios '"]),
    ("factorial", "def factorial(n):", "Devuelve n! para n >= 0.",
     "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n",
     ["0", "1", "5", "3", "10", "7"]),
    ("fibonacci", "def fibonacci(n):", "Devuelve el n-ésimo número de Fibonacci (fibonacci(0) = 0, fibonacci(1) = 1).",
     "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
     ["0", "1", "2", "10", "20", "7"]),
    ("es_primo", "def es_primo(n):", "Devuelve True si n es un número primo.",
     "def es_primo(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True\n",
     ["7", "1", "2", "9", "97", "100", "0", "13"]),
    ("pares", "def pares(xs):", "Devuelve una lista con los números pares de xs, en el mismo orden.",
     "def pares(xs):\n    return [x for x in xs if x % 2 == 0]\n",
     ["[1, 2, 3, 4]", "[1, 3]", "[]", "[0, -2, 5]", "[8, 8, 7]"]),
    ("cuadrados", "def cuadrados(n):", "Devuelve la lista de los cuadrados de 1 a n inclusive.",
     "def cuadrados(n):\n    return [i * i for i in range(1, n + 1)]\n",
     ["3", "0", "1", "5", "7"]),
    ("mcd", "def mcd(a, b):", "Devuelve el máximo común divisor de los enteros positivos a y b.",
     "def mcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
     ["12, 18", "7, 3", "10, 5", "100, 75", "17, 17", "1, 9"]),
    ("contar_palabras", "def contar_palabras(s):", "Devuelve un diccionario palabra → cantidad de veces que aparece en s (separar por espacios).",
     "def contar_palabras(s):\n    d = {}\n    for p in s.split():\n        d[p] = d.get(p, 0) + 1\n    return d\n",
     ["'a b a'", "''", "'sol sol sol'", "'uno dos tres'", "'x y x y z'"]),
    ("aplanar", "def aplanar(xss):", "Recibe una lista de listas y devuelve una sola lista con todos los elementos en orden.",
     "def aplanar(xss):\n    return [x for xs in xss for x in xs]\n",
     ["[[1, 2], [3]]", "[]", "[[], [1], []]", "[[1], [2], [3, 4, 5]]", "[['a'], ['b', 'c']]"]),
    ("sin_repetidos", "def sin_repetidos(xs):", "Devuelve los elementos de xs sin repetir, en el orden de su primera aparición.",
     "def sin_repetidos(xs):\n    vistos = []\n    for x in xs:\n        if x not in vistos:\n            vistos.append(x)\n    return vistos\n",
     ["[1, 2, 1, 3]", "[]", "[5, 5, 5]", "['a', 'b', 'a']", "[3, 2, 1, 2, 3]"]),
    ("segundo_mayor", "def segundo_mayor(xs):", "Devuelve el segundo valor distinto más grande de xs, o None si no existe.",
     "def segundo_mayor(xs):\n    vals = sorted(set(xs), reverse=True)\n    return vals[1] if len(vals) > 1 else None\n",
     ["[1, 3, 2]", "[5, 5]", "[]", "[10, 9, 10, 8]", "[-1, -2]", "[4]"]),
    ("binario", "def binario(n):", "Devuelve la representación en binario de n >= 0 como cadena, sin prefijo.",
     "def binario(n):\n    if n == 0:\n        return '0'\n    s = ''\n    while n:\n        s = str(n % 2) + s\n        n //= 2\n    return s\n",
     ["5", "0", "1", "8", "255", "10"]),
    ("digitos", "def suma_digitos(n):", "Devuelve la suma de los dígitos del entero n (si es negativo, ignorar el signo).",
     "def suma_digitos(n):\n    return sum(int(c) for c in str(abs(n)))\n",
     ["123", "0", "-45", "9999", "1000", "7"]),
    ("capitalizar", "def capitalizar(s):", "Devuelve s con la primera letra de cada palabra en mayúscula y el resto en minúscula.",
     "def capitalizar(s):\n    return ' '.join(p[:1].upper() + p[1:].lower() for p in s.split(' '))\n",
     ["'hola mundo'", "'PYTHON'", "''", "'a b'", "'buenos AIRES'"]),
    ("rotar", "def rotar(xs, k):", "Rota la lista xs k posiciones hacia la derecha y devuelve la lista nueva.",
     "def rotar(xs, k):\n    if not xs:\n        return []\n    k %= len(xs)\n    return xs[-k:] + xs[:-k] if k else list(xs)\n",
     ["[1, 2, 3, 4], 1", "[1, 2, 3], 0", "[], 3", "[1, 2, 3], 4", "['a', 'b'], 1"]),
    ("fizzbuzz", "def fizzbuzz(n):", "Devuelve 'Fizz' si n es múltiplo de 3, 'Buzz' si de 5, 'FizzBuzz' si de ambos y si no str(n).",
     "def fizzbuzz(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)\n",
     ["3", "5", "15", "7", "30", "9", "10"]),
    ("promedio", "def promedio(xs):", "Devuelve el promedio de xs, o 0 si la lista está vacía.",
     "def promedio(xs):\n    return sum(xs) / len(xs) if xs else 0\n",
     ["[1, 2, 3]", "[]", "[10]", "[1, 2]", "[-3, 3]"]),
    ("anagramas", "def son_anagramas(a, b):", "Devuelve True si a y b tienen exactamente las mismas letras con la misma cantidad.",
     "def son_anagramas(a, b):\n    return sorted(a) == sorted(b)\n",
     ["'roma', 'amor'", "'ab', 'abb'", "'', ''", "'listen', 'silent'", "'aab', 'abb'"]),
    ("comprimir", "def comprimir(s):", "Compresión por corridas: 'aaabcc' → 'a3b1c2'.",
     "def comprimir(s):\n    out = ''\n    i = 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n        out += s[i] + str(j - i)\n        i = j\n    return out\n",
     ["'aaabcc'", "''", "'a'", "'abc'", "'zzzz'", "'aabbaa'"]),
    ("matriz_transpuesta", "def transpuesta(m):", "Devuelve la transpuesta de la matriz m (lista de filas de igual largo).",
     "def transpuesta(m):\n    return [list(f) for f in zip(*m)]\n",
     ["[[1, 2], [3, 4]]", "[[1, 2, 3]]", "[]", "[[1], [2]]", "[[1, 2], [3, 4], [5, 6]]"]),
    ("contar_mayores", "def contar_mayores(xs, k):", "Cuenta cuántos elementos de xs son estrictamente mayores que k.",
     "def contar_mayores(xs, k):\n    return sum(1 for x in xs if x > k)\n",
     ["[1, 5, 7], 4", "[], 0", "[3, 3], 3", "[-1, 0, 1], -1", "[10, 20, 30], 100"]),
    ("potencia", "def potencia(b, e):", "Devuelve b elevado a e (e entero >= 0) sin usar ** ni pow.",
     "def potencia(b, e):\n    r = 1\n    for _ in range(e):\n        r *= b\n    return r\n",
     ["2, 10", "5, 0", "3, 3", "-2, 3", "1, 100", "10, 2"]),
    ("indice_de", "def indice_de(xs, x):", "Devuelve el índice de la primera aparición de x en xs, o -1 si no está.",
     "def indice_de(xs, x):\n    for i, y in enumerate(xs):\n        if y == x:\n            return i\n    return -1\n",
     ["[4, 5, 6], 5", "[1, 2], 3", "[], 1", "[7, 7], 7", "['a', 'b'], 'b'"]),
    ("mezclar_ordenadas", "def mezclar(a, b):", "Recibe dos listas ordenadas y devuelve una lista ordenada con todos sus elementos.",
     "def mezclar(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i])\n            i += 1\n        else:\n            out.append(b[j])\n            j += 1\n    return out + a[i:] + b[j:]\n",
     ["[1, 3], [2, 4]", "[], [1]", "[1, 2], []", "[1, 1], [1]", "[0, 5, 9], [3, 4, 10]"]),
    ("busqueda_binaria", "def buscar(xs, x):", "xs está ordenada; devuelve True si x está en xs (búsqueda binaria).",
     "def buscar(xs, x):\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n        m = (lo + hi) // 2\n        if xs[m] == x:\n            return True\n        if xs[m] < x:\n            lo = m + 1\n        else:\n            hi = m - 1\n    return False\n",
     ["[1, 3, 5, 7], 5", "[1, 3, 5, 7], 4", "[], 1", "[2], 2", "[1, 2, 3, 4, 5, 6], 6", "[1, 2, 3], 0"]),
    ("celsius", "def a_fahrenheit(c):", "Convierte grados Celsius a Fahrenheit (F = C · 9/5 + 32).",
     "def a_fahrenheit(c):\n    return c * 9 / 5 + 32\n",
     ["0", "100", "-40", "37", "20"]),
    ("acumulada", "def suma_acumulada(xs):", "Devuelve la lista de sumas acumuladas de xs.",
     "def suma_acumulada(xs):\n    out, t = [], 0\n    for x in xs:\n        t += x\n        out.append(t)\n    return out\n",
     ["[1, 2, 3]", "[]", "[5]", "[1, -1, 1]", "[10, 0, 10]"]),
    ("parentesis", "def balanceado(s):", "Devuelve True si los paréntesis de s están balanceados (solo '(' y ')').",
     "def balanceado(s):\n    n = 0\n    for c in s:\n        if c == '(':\n            n += 1\n        elif c == ')':\n            n -= 1\n            if n < 0:\n                return False\n    return n == 0\n",
     ["'(())'", "')('", "''", "'(()'", "'()()'", "'a(b)c'"]),
    ("mayusculas", "def contar_mayusculas(s):", "Cuenta las letras mayúsculas de s.",
     "def contar_mayusculas(s):\n    return sum(1 for c in s if c.isupper())\n",
     ["'Hola Mundo'", "'abc'", "'ABC'", "''", "'aBcD'"]),
    ("multiplos", "def suma_multiplos(n):", "Suma los números menores que n que son múltiplos de 3 o de 5.",
     "def suma_multiplos(n):\n    return sum(i for i in range(n) if i % 3 == 0 or i % 5 == 0)\n",
     ["10", "0", "1", "16", "100"]),
    ("dividir_en_partes", "def partes(xs, n):", "Divide xs en trozos consecutivos de largo n (el último puede ser más corto).",
     "def partes(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs), n)]\n",
     ["[1, 2, 3, 4, 5], 2", "[], 3", "[1, 2], 5", "[1, 2, 3], 1", "['a', 'b', 'c', 'd'], 2"]),
    ("collatz", "def pasos_collatz(n):", "Cuenta los pasos hasta llegar a 1 aplicando: si n es par n//2, si no 3n+1.",
     "def pasos_collatz(n):\n    p = 0\n    while n != 1:\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        p += 1\n    return p\n",
     ["1", "2", "3", "6", "7", "27"]),
    ("producto_digitos", "def producto_lista(xs):", "Devuelve el producto de los números de xs (1 si está vacía).",
     "def producto_lista(xs):\n    r = 1\n    for x in xs:\n        r *= x\n    return r\n",
     ["[1, 2, 3]", "[]", "[5, 0]", "[-1, 4]", "[2, 2, 2, 2]"]),
    ("ultima_palabra", "def largo_ultima(s):", "Devuelve el largo de la última palabra de s (0 si no hay palabras).",
     "def largo_ultima(s):\n    ps = s.split()\n    return len(ps[-1]) if ps else 0\n",
     ["'hola mundo'", "''", "'uno  '", "'a'", "'  vamos ya  '"]),
    ("intercalar", "def intercalar(a, b):", "Intercala los elementos de a y b (a[0], b[0], a[1], …); lo que sobre va al final.",
     "def intercalar(a, b):\n    out = []\n    for i in range(max(len(a), len(b))):\n        if i < len(a):\n            out.append(a[i])\n        if i < len(b):\n            out.append(b[i])\n    return out\n",
     ["[1, 3], [2, 4]", "[], [1]", "[1, 2, 3], ['a']", "[], []", "['x'], ['y', 'z']"]),
    ("cuenta_regresiva", "def cuenta_regresiva(n):", "Devuelve la lista n, n-1, …, 1, 0.",
     "def cuenta_regresiva(n):\n    return list(range(n, -1, -1))\n",
     ["3", "0", "1", "5"]),
]


def _call(name: str, args: str) -> str:
    return f"{name}({args})"


def validate(raw: list[tuple], *, n_public: int = 2) -> list[dict]:
    """Calcula las salidas esperadas ejecutando cada solución de referencia en el sandbox."""
    from hope.tools.sandbox import run_python
    progs = []
    for _pid, sig, _statement, sol, inputs in raw:
        name = re.match(r"def\s+(\w+)", sig).group(1)
        lines = [f"print(repr({_call(name, a)}))" for a in inputs]
        progs.append(sol + "\n" + "\n".join(lines))
    script = "import json, sys, io, contextlib\nouts = []\nfor p in json.loads(sys.stdin.read()):\n" \
             "    buf = io.StringIO()\n    with contextlib.redirect_stdout(buf):\n        exec(p, {})\n" \
             "    outs.append(buf.getvalue().splitlines())\nprint(json.dumps(outs))"
    r = run_python(script, stdin=json.dumps(progs), timeout_s=60)
    if not r.ok:
        raise RuntimeError(f"las soluciones de referencia fallaron: {r.stderr[-500:]}")
    outs = json.loads(r.stdout.strip().splitlines()[-1])
    problems = []
    for (pid, sig, statement, sol, inputs), out in zip(raw, outs):
        name = re.match(r"def\s+(\w+)", sig).group(1)
        asserts = [f"assert {_call(name, a)} == {o}" for a, o in zip(inputs, out)]
        problems.append({"id": pid, "nombre": name, "firma": sig, "enunciado": statement, "solucion": sol,
                         "ejemplos": asserts[:n_public], "ocultas": asserts[n_public:], "pruebas": asserts,
                         "licencia": "propia (licencia del repositorio)", "fuente": "banco"})
    return problems


_BANK: list[dict] | None = None


def bank() -> list[dict]:
    global _BANK
    if _BANK is None:
        _BANK = validate(_RAW)
    return _BANK


# ---------------------------------------------------------------------------
# Problemas generados (currículo ilimitado, para practicar)
# ---------------------------------------------------------------------------
def _gen_multiplos(r: random.Random):
    a, b = r.sample([2, 3, 4, 5, 7], 2)
    return (f"suma_mult_{a}_{b}", "def suma_mult(n):", f"Suma los enteros de 0 a n-1 que son múltiplos de {a} o de {b}.",
            f"def suma_mult(n):\n    return sum(i for i in range(n) if i % {a} == 0 or i % {b} == 0)\n",
            [str(r.randint(0, 60)) for _ in range(6)])


def _gen_lineal(r: random.Random):
    k, c = r.randint(2, 9), r.randint(-5, 9)
    return (f"transformar_{k}_{c}", "def transformar(xs):", f"Devuelve una lista con cada elemento x de xs cambiado por {k}·x + ({c}).",
            f"def transformar(xs):\n    return [{k} * x + ({c}) for x in xs]\n",
            [str([r.randint(-9, 9) for _ in range(r.randint(0, 5))]) for _ in range(6)])


def _gen_filtro_largo(r: random.Random):
    k = r.randint(2, 5)
    words = ["sol", "mar", "casa", "luz", "perro", "gato", "río", "nube", "pan", "camino", "a", "yo"]
    return (f"largas_{k}", "def largas(palabras):", f"Devuelve las palabras con al menos {k} letras, en el mismo orden.",
            f"def largas(palabras):\n    return [p for p in palabras if len(p) >= {k}]\n",
            [str(r.sample(words, r.randint(0, 5))) for _ in range(6)])


def _gen_sucesion(r: random.Random):
    a0, m, c = r.randint(0, 5), r.randint(1, 3), r.randint(-2, 4)
    return (f"sucesion_{a0}_{m}_{c}", "def termino(n):", f"a(0) = {a0} y a(n) = {m}·a(n-1) + ({c}). Devuelve a(n).",
            f"def termino(n):\n    a = {a0}\n    for _ in range(n):\n        a = {m} * a + ({c})\n    return a\n",
            [str(r.randint(0, 8)) for _ in range(6)])


def _gen_reemplazo(r: random.Random):
    x, y = r.sample("aeiou", 2)
    words = ["casa", "perro", "murcielago", "", "ola", "auto", "pera"]
    return (f"reemplazar_{x}_{y}", "def cambiar(s):", f"Devuelve s con cada letra '{x}' reemplazada por '{y}'.",
            f"def cambiar(s):\n    return s.replace('{x}', '{y}')\n", [repr(r.choice(words)) for _ in range(6)])


def _gen_contar(r: random.Random):
    k = r.randint(0, 9)
    return (f"contar_{k}", "def contar(xs):", f"Cuenta cuántas veces aparece el número {k} en xs.",
            f"def contar(xs):\n    return sum(1 for x in xs if x == {k})\n",
            [str([r.randint(0, 9) for _ in range(r.randint(0, 8))]) for _ in range(6)])


FAMILIES = [_gen_multiplos, _gen_lineal, _gen_filtro_largo, _gen_sucesion, _gen_reemplazo, _gen_contar]


def generated_problems(n: int, *, seed: int = 0) -> list[dict]:
    rng = random.Random(f"codigo:{seed}")
    raw, seen = [], set()
    while len(raw) < n:
        item = rng.choice(FAMILIES)(rng)
        ins = list(dict.fromkeys(item[4]))
        if item[0] in seen or len(ins) < 3:
            continue
        seen.add(item[0])
        raw.append((item[0], item[1], item[2], item[3], ins))
    out = validate(raw)
    for p in out:
        p["fuente"] = "generador"
    return out


def load_mbpp(path: str | Path, *, n_public: int = 1) -> list[dict]:
    """JSONL con el formato de MBPP (``text``, ``code``, ``test_list``). Licencia: CC BY 4.0."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        m = re.search(r"def\s+(\w+)\s*\((.*?)\)\s*:", d["code"])
        if not m:
            continue
        tests = [t.strip() for t in d["test_list"]]
        out.append({"id": f"mbpp_{d.get('task_id', len(out))}", "nombre": m.group(1),
                    "firma": f"def {m.group(1)}({m.group(2)}):", "enunciado": d["text"], "solucion": d["code"],
                    "ejemplos": tests[:n_public], "ocultas": tests[n_public:], "pruebas": tests,
                    "licencia": "CC BY 4.0 (MBPP, Austin et al. 2021)", "fuente": "mbpp"})
    return out


def task_of(p: dict) -> CodeTask:
    return CodeTask(p["enunciado"], p["nombre"], p["firma"], list(p["ejemplos"]))


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimador insesgado de pass@k (Chen et al. 2021): 1 − C(n−c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def empty_body(code: str) -> bool:
    """¿Las funciones del código no tienen más que su docstring (o ``pass``)?"""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return bool(funcs) and all(all(isinstance(b, ast.Pass) or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))
                                   for b in f.body) for f in funcs)


def is_valid_python(code: str) -> bool:
    """Sintácticamente válido: compila (sin ejecutarlo)."""
    from hope.core.gramatica import python_complete
    return python_complete(code)


@dataclass
class Sample:
    code: str
    passed: bool
    hidden_ok: int
    hidden: int
    valid: bool
    seconds: float
    tokens: float
    empty: bool = False              # la función quedó sin cuerpo (solo firma y docstring)


@dataclass
class ProblemResult:
    id: str
    samples: list[Sample] = field(default_factory=list)

    def pass_k(self, k: int) -> float:
        n = len(self.samples)
        return pass_at_k(n, sum(s.passed for s in self.samples), k) if n >= k else float("nan")


def score_codes(problem: dict, codes: list[str]) -> list[tuple[bool, int]]:
    """Pruebas ocultas (y las públicas) contra cada código: (pasa todas, cuántas ocultas pasan)."""
    tests = problem["ejemplos"] + problem["ocultas"]
    rows = run_tests_detailed(codes, tests, timeout_s=10 + 2 * len(codes))
    out = []
    for row in rows:
        ok = [p["ok"] for p in row["pruebas"]] if row["carga"] is None else [False] * len(tests)
        out.append((all(ok) and len(ok) == len(tests), sum(ok[len(problem["ejemplos"]):])))
    return out


def solve_fn(model, mode: str, *, temperature: float, max_new: int = 300, rounds: int = 3):
    """Devuelve ``f(problema, i) -> (código, bytes generados)`` para un modo de evaluación.

    Modos: ``plano``, ``gramatica``, ``autodepuracion`` y ``autodepuracion+gramatica``."""
    from hope.runtime.autodepurar import ModelCoder, SelfDebugger
    grammar = "gramatica" in mode
    coder = ModelCoder(model, grammar=grammar)

    def f(p: dict, i: int) -> tuple[str, int]:
        b0 = coder.stats["bytes"]
        task = task_of(p)
        temp = temperature if i > 0 or temperature > 0 else 0.0
        if "autodepuracion" in mode:
            dbg = SelfDebugger(coder, rounds=rounds, n_tests=3, reflect=False, max_new=max_new, timeout_s=8)
            # La muestra i usa su temperatura en el primer intento (diversidad entre muestras); las
            # correcciones, al menos 0,6.
            orig, calls = coder.code, [0]

            def first_code(prompt, *, code_prefix="", temperature=0.0, max_new=max_new):
                t = temp if calls[0] == 0 else max(temperature, 0.6)
                calls[0] += 1
                return orig(prompt, code_prefix=code_prefix, temperature=t, max_new=max_new)
            coder.code = first_code
            try:
                rep = dbg.solve(task)
            finally:
                coder.code = orig
            return rep.code, coder.stats["bytes"] - b0
        code = coder.code("", code_prefix=task.head(), temperature=temp, max_new=max_new)
        return code, coder.stats["bytes"] - b0
    return f


def evaluate(problems: list[dict], solve, *, n: int = 5, ks: tuple[int, ...] = (1, 5), log=None) -> dict:
    results = []
    for j, p in enumerate(problems):
        pr = ProblemResult(p["id"])
        codes, times, toks = [], [], []
        for i in range(n):
            t0 = time.perf_counter()
            code, nbytes = solve(p, i)
            times.append(time.perf_counter() - t0)
            toks.append(nbytes / 4)
            codes.append(code)
        for code, (ok, hid), s, t in zip(codes, score_codes(p, codes), times, toks):
            pr.samples.append(Sample(code, ok, hid, len(p["ocultas"]), is_valid_python(code), s, t,
                                     empty_body(code)))
        results.append(pr)
        if log:
            log(f"{j + 1}/{len(problems)} {p['id']}: {sum(s.passed for s in pr.samples)}/{n}")
    allS = [s for r in results for s in r.samples]
    summary = {f"pass@{k}": round(sum(r.pass_k(k) for r in results) / len(results), 4) for k in ks if k <= n}
    summary.update({
        "problemas": len(results), "muestras": len(allS),
        "sintaxis_valida": round(sum(s.valid for s in allS) / max(len(allS), 1), 4),
        # Válida y con algo más que la firma y el docstring (una función vacía compila pero no hace nada).
        "valida_con_cuerpo": round(sum(s.valid and not s.empty for s in allS) / max(len(allS), 1), 4),
        "ocultas_pasadas": round(sum(s.hidden_ok for s in allS) / max(sum(s.hidden for s in allS), 1), 4),
        "segundos_por_problema": round(sum(s.seconds for s in allS) / len(results), 3),
        "tokens_por_problema": round(sum(s.tokens for s in allS) / len(results), 1),
    })
    return {"resumen": summary, "por_problema": [
        {"id": r.id, "aciertos": sum(s.passed for s in r.samples), "validas": sum(s.valid for s in r.samples),
         "ejemplo": r.samples[0].code[:400]} for r in results]}


# ---------------------------------------------------------------------------
# Refuerzo con recompensa de ejecución
# ---------------------------------------------------------------------------
def code_reward(problem: dict, completion: str) -> float:
    """Fracción de pruebas **públicas** que pasa el código (cabecera + lo generado). Las ocultas no se usan."""
    from hope.runtime.autodepurar import cut_function
    code = cut_function(task_of(problem).head() + completion)
    row = run_tests_detailed([code], problem["ejemplos"], timeout_s=8)[0]
    if row["carga"] is not None or not row["pruebas"]:
        return 0.0
    return row["pasan"] / len(row["pruebas"])


def rl_step(rl, problems: list[dict], *, cost=None) -> dict:
    """Un paso GRPO sobre problemas de código con ``grpo_step_tools`` (Hope puede usar ⟦python: …⟧)."""
    from hope.runtime.toolcalls import ToolRunner
    from hope.tools.builtin import default_catalog
    from hope.train.rlvr import grpo_step_tools
    cat = default_catalog()
    prompts = [task_of(p).head() for p in problems]
    return grpo_step_tools(rl, prompts, lambda i, text: code_reward(problems[i], text),
                           lambda: ToolRunner(cat), w=cost)


def load_model(path: str):
    import torch

    from hope.core.hn1 import HN1Config, HN1Model
    p = Path(path)
    ckpt = torch.load(p / "model.pt" if p.is_dir() else p, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or ckpt["summary"]["config"]
    m = HN1Model(HN1Config.from_dict(cfg))
    m.load_state_dict(ckpt["state_dict"])
    return m.eval()


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Banco de código: pass@k con y sin gramática y autodepuración")
    ap.add_argument("--checkpoint", default="checkpoints/hn1-25m-cpu.pt")
    ap.add_argument("--modos", nargs="+", default=["plano", "gramatica", "autodepuracion"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--temperatura", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--problemas", type=int, default=0, help="limitar a los primeros N del banco (0 = todos)")
    ap.add_argument("--generados", type=int, default=0, help="agregar N problemas generados")
    ap.add_argument("--mbpp", default=None, help="JSONL con formato MBPP (opcional)")
    ap.add_argument("--hilos", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rl", action="store_true", help="refuerzo con recompensa de ejecución")
    ap.add_argument("--prueba-cpu", action="store_true", help="con --rl: modelo diminuto y un paso (humo)")
    ap.add_argument("--pasos", type=int, default=1)
    a = ap.parse_args(argv)
    import torch
    if a.hilos:
        torch.set_num_threads(a.hilos)
    probs = bank()
    if a.problemas:
        probs = probs[:a.problemas]
    if a.generados:
        probs = probs + generated_problems(a.generados)
    if a.mbpp:
        probs = probs + load_mbpp(a.mbpp)
    if a.rl:
        from hope.core.hn1 import HN1Model, preset
        from hope.train.rlvr import GRPO, RLConfig
        torch.manual_seed(0)
        if a.prueba_cpu:
            model = HN1Model(preset("hn1-tiny"))
            cfg = RLConfig(group=2, temperature=1.0, max_new=24, curriculum=(), lr=1e-3)
            train = generated_problems(2)
        else:
            model = load_model(a.checkpoint)
            cfg = RLConfig(group=4, temperature=0.8, max_new=a.max_new, curriculum=(), lr=1e-5)
            train = generated_problems(max(8, a.generados or 8))
        rl = GRPO(model, cfg)
        hist = []
        for s in range(a.pasos):
            st = rl_step(rl, train[(s * 2) % len(train):(s * 2) % len(train) + 2])
            hist.append(st)
            print(json.dumps({"paso": s, **st}, ensure_ascii=False))
        return {"rl": hist}
    model = load_model(a.checkpoint)
    report = {"checkpoint": a.checkpoint, "n": a.n, "temperatura": a.temperatura, "modos": {}}
    for mode in a.modos:
        t0 = time.time()
        r = evaluate(probs, solve_fn(model, mode, temperature=a.temperatura, max_new=a.max_new), n=a.n,
                     ks=tuple(a.k), log=lambda m, mode=mode: print(f"[{mode}] {m}", flush=True))
        r["resumen"]["minutos"] = round((time.time() - t0) / 60, 1)
        report["modos"][mode] = r
        print(json.dumps({mode: r["resumen"]}, ensure_ascii=False), flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
