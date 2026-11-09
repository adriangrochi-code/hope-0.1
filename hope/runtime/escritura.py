"""Escribir mejor sin más parámetros: planificar, criticar, reescribir, elegir y recordar.

Pedido del usuario: que Hope genere lenguaje con la calidad de un modelo mucho más grande **sin
agregar parámetros**, con bucles de razonamiento, selección y memoria. Todo está detrás de opciones:
``model.generate`` (voraz) no cambia.

Piezas (las de decodificación están en ``hope/core``):

1. **Bucle de escritura** (``Escritor``): esquema → borrador → crítica con verificadores de lengua y
   de contenido → reescritura **solo de los fragmentos marcados** → otra ronda mientras haya
   presupuesto. El presupuesto sale de la velocidad objetivo (``SpeedTarget`` de
   ``hope/runtime/budget.py``): como en ``BudgetedResponder``, cada fase se estima con lo que tardó
   la anterior y **no se empieza si no entra** (la respuesta nunca llega tarde). ``agente_escritor``
   lo envuelve como ``Agent`` para usarlo dentro de ``BudgetedResponder``.
2. **Cómputo adaptativo** (``hope.core.paso.ComputoAdaptativo``).
3. **Mejor de N con MBR** (``mejor_de_n``/``elegir_mbr``): N candidatos con ``generate_batch`` y se
   elige el de **riesgo de Bayes mínimo** (el que más se parece, en chrF por bytes, a los demás),
   combinado con las marcas de los verificadores.
4. **Contrastiva** (``hope.core.contrastiva``) y 5. **kNN-LM** (``hope.core.knn``).
6. **Ejemplos de estilo** (``EjemplosEstilo``): pasajes bien escritos (nota ≥ 4 del maestro) sobre el
   tema, recuperados por similitud y antepuestos como guía.
7. **Resumen continuo** (``ResumenContinuo`` + ``ConversacionConResumen``): un resumen extractivo de
   la conversación en el cuaderno (``Notebook``); cuando el historial ya no entra en la ventana,
   la conversación sigue desde «resumen + último turno» en lugar de olvidar lo anterior.

Verificadores
-------------
Protocolo ``Verificador``: ``nombre`` y ``revisar(texto, ctx) -> list[Marca]``. Una ``Marca`` señala
un tramo ``[inicio, fin)`` del texto (en caracteres) con un tipo, un detalle y una severidad; el
bucle reescribe los fragmentos (oraciones) que tocan. Los de este módulo son básicos y sin
parámetros (repeticiones, fluidez por n-gramas de bytes del corpus, léxico, contenido). Las
herramientas de lengua de la otra sesión (ortografía, gramática, coherencia; rama ``…-catalogo``)
se enchufan con ``adaptar`` sin tocar el bucle.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

# ---------------------------------------------------------------------------
# Texto: oraciones, palabras, métricas
# ---------------------------------------------------------------------------
_PALABRA = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
_FIN_ORACION = re.compile(r"(?<=[.!?…])\s+|\n+")
VACIAS = frozenset("""a al algo ante antes como con contra cual cuando de del desde donde durante e el ella ellas
ellos en entre era eran es esa ese eso esta este esto fue fueron ha han hasta hay la las le les lo los mas más me
mi muy no nos o otra otro para pero por que qué se sea ser si sí sin sobre su sus también tiene tienen todo un una
uno unos unas y ya the of and to in is was for on that with as by it at from are an be this which""".split())


def oraciones(texto: str) -> list[tuple[int, int]]:
    """Tramos ``[inicio, fin)`` de cada oración (los separadores quedan fuera)."""
    out, start = [], 0
    for m in _FIN_ORACION.finditer(texto):
        if m.start() > start:
            out.append((start, m.start()))
        start = m.end()
    if start < len(texto):
        out.append((start, len(texto)))
    return out


def palabras(texto: str) -> list[str]:
    return [w.lower() for w in _PALABRA.findall(texto)]


def tasa_repeticion(texto: str, n: int = 4) -> float:
    """Fracción de n-gramas de palabras que ya habían aparecido antes en el mismo texto (rep-n)."""
    w = palabras(texto)
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def distinct_n(textos: list[str], n: int = 2) -> float:
    """Diversidad: n-gramas de palabras distintos / totales, sobre todas las salidas juntas."""
    grams = []
    for t in textos:
        w = palabras(t)
        grams += [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def _ngramas_bytes(b: bytes, n: int) -> Counter:
    return Counter(b[i:i + n] for i in range(len(b) - n + 1))


def chrf(hip: str | bytes, ref: str | bytes, n_max: int = 6, beta: float = 2.0) -> float:
    """chrF (Popović 2015) sobre **bytes**: F-β promedio de precisión/recall de n-gramas 1..n_max."""
    h = hip.encode("utf-8") if isinstance(hip, str) else hip
    r = ref.encode("utf-8") if isinstance(ref, str) else ref
    ps, rs = [], []
    for n in range(1, n_max + 1):
        ch, cr = _ngramas_bytes(h, n), _ngramas_bytes(r, n)
        if not ch or not cr:
            continue
        inter = sum((ch & cr).values())
        ps.append(inter / sum(ch.values()))
        rs.append(inter / sum(cr.values()))
    if not ps:
        return 0.0
    p, r_ = sum(ps) / len(ps), sum(rs) / len(rs)
    if p + r_ == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r_ / (b2 * p + r_)


class ModeloNgramas:
    """Modelo de lenguaje de n-gramas de **bytes** con interpolación Witten-Bell. Sin parámetros
    entrenables: cuenta el corpus local. Sirve para medir fluidez (bits por byte del texto generado
    según el corpus) y como verificador de fluidez por oración."""

    def __init__(self, orden: int = 5):
        self.orden = orden
        self.c: list[dict] = [defaultdict(Counter) for _ in range(orden)]   # c[k][contexto de k bytes][byte]

    def entrenar(self, textos: list[str | bytes]) -> "ModeloNgramas":
        for t in textos:
            b = t.encode("utf-8") if isinstance(t, str) else t
            for i in range(len(b)):
                for k in range(self.orden):
                    if i - k < 0:
                        break
                    self.c[k][b[i - k:i]][b[i]] += 1
        self._tot = [{ctx: (sum(cnt.values()), len(cnt)) for ctx, cnt in ck.items()} for ck in self.c]
        return self

    def prob(self, ctx: bytes, x: int) -> float:
        p = 1.0 / 256
        for k in range(self.orden):
            if k > len(ctx):
                break
            h = ctx[len(ctx) - k:] if k else b""
            tot = self._tot[k].get(h)
            if not tot:
                break
            n, t = tot
            lam = n / (n + t)
            p = lam * self.c[k][h][x] / n + (1 - lam) * p
        return p

    def bits_por_byte(self, texto: str | bytes, contexto: str | bytes = b"") -> float:
        b = texto.encode("utf-8") if isinstance(texto, str) else texto
        c = contexto.encode("utf-8") if isinstance(contexto, str) else contexto
        full = c + b
        if not b:
            return 0.0
        tot = 0.0
        for i in range(len(c), len(full)):
            tot -= math.log2(self.prob(full[max(0, i - self.orden + 1):i], full[i]))
        return tot / len(b)


# ---------------------------------------------------------------------------
# Verificadores
# ---------------------------------------------------------------------------
@dataclass
class Marca:
    inicio: int
    fin: int
    tipo: str                 # repeticion | fluidez | lexico | contenido | ortografia | gramatica | coherencia …
    detalle: str
    severidad: float = 1.0
    verificador: str = ""


@dataclass
class ContextoRevision:
    """Lo que un verificador puede necesitar además del texto."""
    tema: str = ""
    pedido: str = ""
    esquema: "Esquema | None" = None
    prefijo: str = ""         # texto anterior (prompt), por si el verificador mira continuidad


class Verificador(Protocol):
    nombre: str

    def revisar(self, texto: str, ctx: ContextoRevision) -> list[Marca]: ...


class Repeticiones:
    """Marca las repeticiones: n-gramas de palabras repetidos y tramos de bytes repetidos (bucles)."""
    nombre = "repeticiones"

    def __init__(self, n: int = 3, min_chars: int = 16):
        self.n, self.min_chars = n, min_chars

    def revisar(self, texto: str, ctx: ContextoRevision) -> list[Marca]:
        marcas = []
        toks = [(m.group().lower(), m.start(), m.end()) for m in _PALABRA.finditer(texto)]
        visto: dict[tuple, int] = {}
        for i in range(len(toks) - self.n + 1):
            g = tuple(t[0] for t in toks[i:i + self.n])
            if all(w in VACIAS for w in g):
                continue
            if g in visto and visto[g] + self.n <= i:          # sin contar solapes consigo mismo
                marcas.append(Marca(toks[i][1], toks[i + self.n - 1][2], "repeticion",
                                    f"repite «{' '.join(g)}»", 1.0, self.nombre))
            visto.setdefault(g, i)
        # bucles de caracteres (p. ej. «de la ciudad de la ciudad de la ciudad»)
        for m in re.finditer(r"(.{%d,80}?)\1+" % max(4, self.min_chars // 2), texto, flags=re.S):
            if m.end() - m.start() >= self.min_chars and m.group(1).strip():
                marcas.append(Marca(m.start() + len(m.group(1)), m.end(), "repeticion",
                                    f"bucle «{m.group(1)[:30]}…»", 2.0, self.nombre))
        return marcas


class Fluidez:
    """Oraciones mucho menos probables que el corpus según un modelo de n-gramas de bytes, y bytes
    inválidos (UTF-8 roto)."""
    nombre = "fluidez"

    def __init__(self, ngramas: ModeloNgramas, umbral_bits: float = 3.2, min_chars: int = 12):
        self.ng, self.umbral, self.min_chars = ngramas, umbral_bits, min_chars

    def revisar(self, texto: str, ctx: ContextoRevision) -> list[Marca]:
        marcas = []
        for a, b in oraciones(texto):
            frag = texto[a:b]
            if "�" in frag:
                marcas.append(Marca(a, b, "fluidez", "caracteres inválidos", 2.0, self.nombre))
                continue
            if b - a < self.min_chars:
                continue
            bits = self.ng.bits_por_byte(frag, texto[max(0, a - 8):a])
            if bits > self.umbral:
                marcas.append(Marca(a, b, "fluidez", f"poco natural ({bits:.2f} bits/byte)",
                                    min(2.0, bits - self.umbral + 0.5), self.nombre))
        return marcas


class Lexico:
    """Palabras que no aparecen en el corpus (aproximación de ortografía sin diccionario externo)."""
    nombre = "lexico"

    def __init__(self, vocabulario: set[str], max_frac: float = 0.2, min_len: int = 4):
        self.voc, self.max_frac, self.min_len = vocabulario, max_frac, min_len

    @classmethod
    def desde_corpus(cls, textos: list[str], min_cuenta: int = 2, **kw) -> "Lexico":
        c = Counter(w for t in textos for w in palabras(t))
        return cls({w for w, n in c.items() if n >= min_cuenta}, **kw)

    def revisar(self, texto: str, ctx: ContextoRevision) -> list[Marca]:
        marcas = []
        for a, b in oraciones(texto):
            ws = [w for w in palabras(texto[a:b]) if len(w) >= self.min_len]
            raras = [w for w in ws if w not in self.voc]
            if ws and len(raras) / len(ws) > self.max_frac:
                marcas.append(Marca(a, b, "lexico", "palabras desconocidas: " + ", ".join(raras[:4]),
                                    0.5 + len(raras) / len(ws), self.nombre))
        return marcas


class Contenido:
    """Contenido: oraciones fuera de tema (sin ninguna palabra clave del esquema ni del pedido) y
    palabras clave del esquema que el texto no cubre (marca sin tramo: ``inicio = fin = -1``)."""
    nombre = "contenido"

    def __init__(self, min_palabras: int = 6, marcar_faltantes: bool = True):
        self.min_palabras, self.marcar_faltantes = min_palabras, marcar_faltantes

    def revisar(self, texto: str, ctx: ContextoRevision) -> list[Marca]:
        claves = set(ctx.esquema.palabras_clave if ctx.esquema else []) | {
            w for w in palabras(ctx.pedido + " " + ctx.tema) if w not in VACIAS and len(w) > 3}
        if not claves:
            return []
        raices = {w[:5] for w in claves}
        marcas = []
        for a, b in oraciones(texto):
            ws = [w for w in palabras(texto[a:b]) if w not in VACIAS]
            if len(ws) >= self.min_palabras and not any(w[:5] in raices for w in ws):
                marcas.append(Marca(a, b, "contenido", "se va del tema", 0.7, self.nombre))
        if self.marcar_faltantes and ctx.esquema:
            hay = {w[:5] for w in palabras(texto)}
            falta = [w for w in ctx.esquema.palabras_clave if w[:5] not in hay]
            if falta:
                marcas.append(Marca(-1, -1, "contenido", "no menciona: " + ", ".join(falta[:5]),
                                    0.3 * len(falta), self.nombre))
        return marcas


class _Adaptado:
    def __init__(self, fn: Callable, nombre: str):
        self.fn, self.nombre = fn, nombre

    def revisar(self, texto: str, ctx: ContextoRevision) -> list[Marca]:
        out = []
        for r in self.fn(texto) or []:
            if isinstance(r, Marca):
                out.append(r)
            elif isinstance(r, dict):
                out.append(Marca(int(r.get("inicio", -1)), int(r.get("fin", -1)), str(r.get("tipo", self.nombre)),
                                 str(r.get("detalle", "")), float(r.get("severidad", 1.0)), self.nombre))
            else:                                       # (inicio, fin, detalle[, severidad])
                out.append(Marca(int(r[0]), int(r[1]), self.nombre, str(r[2]),
                                 float(r[3]) if len(r) > 3 else 1.0, self.nombre))
        return out


def adaptar(v, nombre: str | None = None) -> Verificador:
    """Convierte un verificador externo (p. ej. las herramientas de lengua de la otra sesión) al
    protocolo: un objeto con ``revisar(texto, ctx)`` se usa tal cual; una función ``f(texto)`` que
    devuelve marcas, diccionarios ``{inicio, fin, tipo, detalle, severidad}`` o tuplas
    ``(inicio, fin, detalle[, severidad])`` se envuelve."""
    if hasattr(v, "revisar"):
        if not hasattr(v, "nombre"):
            v.nombre = nombre or type(v).__name__.lower()
        return v
    return _Adaptado(v, nombre or getattr(v, "__name__", "externo"))


def revisar(texto: str, verificadores: list, ctx: ContextoRevision | None = None) -> list[Marca]:
    ctx = ctx or ContextoRevision()
    out: list[Marca] = []
    for v in verificadores:
        try:
            out += v.revisar(texto, ctx)
        except Exception as e:                             # noqa: BLE001 — un verificador roto no frena la escritura
            out.append(Marca(-1, -1, "error", f"{getattr(v, 'nombre', v)} falló: {e}", 0.0, "escritura"))
    return out


def penalidad(marcas: list[Marca]) -> float:
    return sum(m.severidad for m in marcas)


def verificadores_basicos(corpus: list[str] | None = None, ngramas: ModeloNgramas | None = None) -> list:
    vs: list = [Repeticiones(), Contenido()]
    if corpus:
        ngramas = ngramas or ModeloNgramas(5).entrenar(corpus)
        vs += [Lexico.desde_corpus(corpus)]
    if ngramas is not None:
        vs.insert(1, Fluidez(ngramas))
    return vs


# ---------------------------------------------------------------------------
# Corpus local (pasajes bien escritos) y ejemplos de estilo
# ---------------------------------------------------------------------------
def _es_espanol(t: str) -> bool:
    w = palabras(t[:2000])
    es = sum(x in ("de", "la", "el", "que", "los", "las", "del", "en", "y") for x in w)
    en = sum(x in ("the", "of", "and", "is", "to", "was", "which") for x in w)
    return es > 2 * en and es >= 5


def corpus_local(raiz: str | Path = ".", *, min_nota: float = 0.0, espanol: bool = True,
                 particion: str | None = None) -> list[dict]:
    """Pasajes de ``datos_maestro/notas_calidad.jsonl`` (Wikipedia con nota de calidad del maestro),
    sin duplicados. ``particion``: ``"eval"`` (1 de cada 10, por hash del texto), ``"almacen"`` (el
    resto) o ``None`` (todo). La partición por hash hace que nada de evaluación entre al almacén."""
    p = Path(raiz) / "datos_maestro" / "notas_calidad.jsonl"
    if not p.exists():
        return []
    vistos, out = set(), []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = r.get("texto", "")
        h = hashlib.sha1(t.encode()).hexdigest()
        if not t or h in vistos or (r.get("nota") or 0) < min_nota or (espanol and not _es_espanol(t)):
            continue
        vistos.add(h)
        es_eval = int(h[:8], 16) % 10 == 0
        if particion == "eval" and not es_eval or particion == "almacen" and es_eval:
            continue
        out.append({"texto": t, "nota": r.get("nota"), "titulo": t.split("\n", 1)[0][:80]})
    return out


def pasajes(texto: str, max_chars: int = 500, min_chars: int = 120) -> list[str]:
    """Párrafos (o trozos de párrafo cortados en una oración) de largo razonable."""
    out = []
    for par in re.split(r"\n\s*\n", texto):
        par = " ".join(par.split())
        if len(par) < min_chars:
            continue
        while len(par) > max_chars:
            cut = par.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > min_chars else max_chars
            out.append(par[:cut].strip())
            par = par[cut:].strip()
        if len(par) >= min_chars:
            out.append(par)
    return out


class EjemplosEstilo:
    """Recupera pasajes bien escritos sobre el tema y los antepone como guía (técnica 6).

    Usa el ``HashingEmbedder`` de la memoria (n-gramas de caracteres; sin entrenamiento) sobre los
    pasajes de nota ≥ ``min_nota``. ``store``: un ``MemoryStore`` opcional; si se da, los pasajes se
    guardan ahí como engramas ``estilo`` (y se pueden recuperar con ``retrieve``)."""

    def __init__(self, textos: list[str], *, max_chars: int = 500, store=None):
        import numpy as np
        from hope.memory.embed import HashingEmbedder
        self.emb = HashingEmbedder()
        self.pas = [p for t in textos for p in pasajes(t, max_chars)]
        self.mat = self.emb.embed(self.pas) if self.pas else np.zeros((0, self.emb.dim), dtype=np.float32)
        if store is not None:
            from hope.epistemic import EpistemicStatus
            from hope.memory.store import Level
            for p in self.pas:
                store.add_engram(kind="estilo", level=Level.SEMANTICA, concept=p[:60], content=p,
                                 description="pasaje bien escrito (nota del maestro)", importance=0.5,
                                 confidence=0.8, status=EpistemicStatus.OBSERVADO)
            store.commit()

    @classmethod
    def desde_corpus(cls, raiz: str | Path = ".", *, min_nota: float = 4.0, particion: str | None = "almacen",
                     **kw) -> "EjemplosEstilo":
        return cls([r["texto"] for r in corpus_local(raiz, min_nota=min_nota, particion=particion)], **kw)

    def buscar(self, tema: str, k: int = 1, excluir: str = "") -> list[str]:
        if not self.pas:
            return []
        q = self.emb.embed([tema])[0]
        sims = self.mat @ q
        out = []
        for i in sims.argsort()[::-1]:
            p = self.pas[int(i)]
            if excluir and (p[:80] in excluir or excluir[:80] in p):
                continue                                   # no «copiar» el propio texto de referencia
            out.append(p)
            if len(out) >= k:
                break
        return out

    def anteponer(self, prompt: str, tema: str, k: int = 1, max_bytes: int = 700, excluir: str = "") -> str:
        ej = self.buscar(tema or prompt[-300:], k, excluir=excluir)
        if not ej:
            return prompt
        guia = "\n\n".join(ej)
        guia = guia.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
        return f"{guia}\n\n{prompt}"


# ---------------------------------------------------------------------------
# Selección: mejor de N con riesgo de Bayes mínimo
# ---------------------------------------------------------------------------
@dataclass
class Eleccion:
    indice: int
    texto: str
    utilidad: list[float]          # utilidad esperada (chrF medio contra los demás) de cada candidato
    penalidades: list[float]
    puntajes: list[float]


def elegir_mbr(candidatos: list[str], verificadores: list | None = None, *, peso: float = 0.15,
               ctx: ContextoRevision | None = None) -> Eleccion:
    """MBR: ``argmax_i Σ_j chrF(c_i, c_j) / (N − 1) − peso · penalidad(c_i)``. Los candidatos vacíos
    pierden siempre."""
    n = len(candidatos)
    util = [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            s = 0.5 * (chrf(candidatos[i], candidatos[j]) + chrf(candidatos[j], candidatos[i]))
            util[i] += s
            util[j] += s
    util = [u / max(n - 1, 1) for u in util]
    pens = [penalidad(revisar(c, verificadores, ctx)) / max(len(c) / 100, 1.0) if verificadores else 0.0
            for c in candidatos]
    score = [u - peso * p if c.strip() else -1e9 for u, p, c in zip(util, pens, candidatos)]
    best = max(range(n), key=lambda i: score[i])
    return Eleccion(best, candidatos[best], util, pens, score)


def mejor_de_n(model, prompt: str, n: int = 4, max_new: int = 200, *, temperature: float = 0.8,
               verificadores: list | None = None, peso: float = 0.15, ctx: ContextoRevision | None = None,
               incluir_voraz: bool = True, n_iters: int | None = None) -> Eleccion:
    """N candidatos en lote (``generate_batch``: el prompt se procesa una vez) y elección MBR.
    Con ``incluir_voraz`` el primer candidato es el voraz (así MBR nunca es peor que no elegir… salvo
    que los verificadores digan lo contrario)."""
    from hope.core.batched import generate_batch
    outs: list[bytes] = []
    if incluir_voraz:
        outs += generate_batch(model, [prompt], max_new, n_iters=n_iters)
    k = n - len(outs)
    if k > 0:
        outs += generate_batch(model, [prompt] * k, max_new, temperature=temperature, n_iters=n_iters)
    cands = [o.decode("utf-8", errors="ignore") for o in outs]
    return elegir_mbr(cands, verificadores, peso=peso, ctx=ctx)


# ---------------------------------------------------------------------------
# Bucle de escritura
# ---------------------------------------------------------------------------
@dataclass
class Esquema:
    palabras_clave: list[str] = field(default_factory=list)
    guia: list[str] = field(default_factory=list)          # pasajes de estilo recuperados


def planificar(tema: str, pedido: str, estilo: EjemplosEstilo | None = None, k_estilo: int = 1,
               max_claves: int = 6) -> Esquema:
    """Esquema barato y sin modelo: palabras clave del pedido y de los pasajes sobre el tema
    (las más frecuentes que no son vacías). Con un modelo de bytes de pocos millones de parámetros,
    pedirle que escriba un esquema da texto sin estructura; el esquema sirve para **verificar**
    cobertura y orientar las reescrituras."""
    guia = estilo.buscar(tema or pedido, k_estilo, excluir=pedido) if estilo else []
    base = [w for w in palabras(f"{tema} {pedido}") if w not in VACIAS and len(w) > 3]
    c = Counter(w for g in guia for w in palabras(g) if w not in VACIAS and len(w) > 4)
    claves = list(dict.fromkeys(base))[:max_claves]
    for w, _n in c.most_common():
        if len(claves) >= max_claves:
            break
        if w[:5] not in {x[:5] for x in claves}:
            claves.append(w)
    return Esquema(claves, guia)


@dataclass
class Ronda:
    fase: str
    segundos: float
    penalidad: float
    cambios: int = 0
    detalle: str = ""


@dataclass
class InformeEscritura:
    texto: str
    borrador: str
    rondas: list[Ronda]
    marcas_finales: list[Marca]
    presupuesto_s: float | None
    segundos: float
    esquema: Esquema
    motivo: str = ""

    @property
    def mejoro(self) -> bool:
        return self.texto != self.borrador


Generador = Callable[[list[str], int, float], list[bytes]]


def generador_lote(model, *, n_iters: int | None = None) -> Generador:
    from hope.core.batched import generate_batch
    return lambda prompts, max_new, temp: generate_batch(model, prompts, max_new, temperature=temp, n_iters=n_iters)


def generador_paso(model, *, procesadores_factory: Callable[[], list] | None = None, adaptativo=None,
                   n_iters: int | None = None, top_p: float | None = None) -> Generador:
    """Generador con ``hope.core.paso.generar`` (contrastiva, kNN-LM, cómputo adaptativo). En serie."""
    from hope.core.paso import generar

    def gen(prompts, max_new, temp):
        return [generar(model, p, max_new, temperature=temp, top_p=top_p, adaptativo=adaptativo, n_iters=n_iters,
                        procesadores=procesadores_factory() if procesadores_factory else ()) for p in prompts]
    return gen


def _primera_oracion(t: str, minimo: int = 8) -> str:
    """Corta en el primer fin de oración después de ``minimo`` caracteres (sin cortar UTF-8)."""
    m = re.search(r"[.!?…](?=\s|$)|\n", t[minimo:])
    return t[:minimo + m.end()] if m else t


class Escritor:
    """Bucle de escritura (técnica 1). ``modelo`` se usa para puntuar fluidez (log-prob) y, si no se
    da ``generador``, para generar en lote.

    Fases y costo estimado (cada fase se estima con lo que tardó la última vez; si no entra en el
    presupuesto, no se empieza):

    1. ``planificar`` (sin modelo).
    2. ``borrador``: voraz (o con los procesadores del generador).
    3. ``mbr`` (opcional, ``n_candidatos > 1``): más candidatos del texto entero y elección MBR.
    4. rondas de ``reescritura``: crítica → por cada fragmento marcado (los peores primero, hasta
       ``max_fragmentos``) ``n_alternativas`` reescrituras de esa oración con el texto anterior como
       contexto; gana la de menos marcas y, a igualdad, la de mayor log-probabilidad de la oración
       **y de lo que sigue** (así la reescritura no rompe la continuidad).
    """

    def __init__(self, modelo, *, verificadores: list | None = None, generador: Generador | None = None,
                 estilo: EjemplosEstilo | None = None, objetivo=None, n_candidatos: int = 1,
                 n_alternativas: int = 4, temperatura: float = 0.8, max_rondas: int = 3, max_fragmentos: int = 3,
                 peso_mbr: float = 0.15, k_estilo: int = 1, clock=time.perf_counter):
        self.model = modelo
        self.verificadores = verificadores if verificadores is not None else verificadores_basicos()
        self.gen = generador or generador_lote(modelo)
        self.estilo, self.objetivo = estilo, objetivo
        self.n_candidatos, self.n_alt, self.temp = n_candidatos, n_alternativas, temperatura
        self.max_rondas, self.max_frag, self.peso_mbr, self.k_estilo = max_rondas, max_fragmentos, peso_mbr, k_estilo
        self.clock = clock
        self.costos: dict[str, float] = {}

    # --- puntaje del modelo ---------------------------------------------------------------
    def _logprob(self, prefijo: str, textos: list[str]) -> list[float]:
        """Log-probabilidad media (nats/byte) de cada ``texto`` después de ``prefijo`` (una pasada en lote)."""
        import torch
        pre = list(prefijo.encode("utf-8"))[-1536:] or [10]
        tb = [list(t.encode("utf-8"))[:400] for t in textos]
        width = len(pre) + max((len(t) for t in tb), default=0)
        if not any(tb):
            return [0.0] * len(textos)
        dev = self.model.head.weight.device
        x = torch.zeros(len(tb), width, dtype=torch.long, device=dev)
        for i, t in enumerate(tb):
            row = pre + t
            x[i, :len(row)] = torch.tensor(row)
        with torch.no_grad():
            self.model.eval()
            lp = torch.log_softmax(self.model(x[:, :-1])["logits"].float(), -1)
        out = []
        for i, t in enumerate(tb):
            if not t:
                out.append(-10.0)
                continue
            idx = torch.arange(len(pre) - 1, len(pre) - 1 + len(t), device=dev)
            out.append(float(lp[i, idx, torch.tensor(t, device=dev)].mean()))
        return out

    # --- fases ----------------------------------------------------------------------------
    def _presupuesto(self, largo: int) -> float | None:
        return self.objetivo.seconds_for(largo) * self.objetivo.safety if self.objetivo is not None else None

    def _entra(self, fase: str, t0: float, presupuesto: float | None, factor: float = 1.0) -> bool:
        if presupuesto is None:
            return True
        est = self.costos.get(fase, self.costos.get("borrador", 0.0) * 2.0) * factor
        return self.clock() - t0 + est <= presupuesto

    def escribir(self, inicio: str, *, tema: str = "", largo: int = 300) -> InformeEscritura:
        t0 = self.clock()
        presupuesto = self._presupuesto(largo)
        esquema = planificar(tema, inicio, self.estilo, self.k_estilo)
        guia = "\n\n".join(esquema.guia)
        guia = guia.encode("utf-8")[:700].decode("utf-8", errors="ignore")
        contexto = f"{guia}\n\n{inicio}" if guia else inicio
        ctx = ContextoRevision(tema=tema, pedido=inicio, esquema=esquema, prefijo=inicio)
        rondas: list[Ronda] = []
        # borrador
        t = self.clock()
        borrador = self.gen([contexto], largo, 0.0)[0].decode("utf-8", errors="ignore")
        self.costos["borrador"] = self.clock() - t
        texto = borrador
        marcas = revisar(texto, self.verificadores, ctx)
        rondas.append(Ronda("borrador", self.costos["borrador"], penalidad(marcas)))
        motivo = "sin marcas"
        # mejor de N
        if self.n_candidatos > 1 and marcas and self._entra("mbr", t0, presupuesto,
                                                             1.0 if "mbr" in self.costos else 0.6 * self.n_candidatos):
            t = self.clock()
            otros = self.gen([contexto] * (self.n_candidatos - 1), largo, self.temp)
            el = elegir_mbr([texto] + [o.decode("utf-8", errors="ignore") for o in otros], self.verificadores,
                            peso=self.peso_mbr, ctx=ctx)
            self.costos["mbr"] = self.clock() - t
            texto = el.texto
            marcas = revisar(texto, self.verificadores, ctx)
            rondas.append(Ronda("mbr", self.costos["mbr"], penalidad(marcas), int(el.indice != 0),
                                f"eligió el candidato {el.indice}"))
        # reescritura de fragmentos marcados
        for r in range(self.max_rondas):
            frags = self._fragmentos(texto, marcas)
            if not frags:
                motivo = "sin marcas" if not any(m.inicio >= 0 for m in marcas) else "nada para reescribir"
                break
            if not self._entra("reescritura", t0, presupuesto):
                motivo = "presupuesto de tiempo cumplido"
                break
            t = self.clock()
            texto, cambios = self._reescribir(contexto, texto, frags, ctx)
            self.costos["reescritura"] = self.clock() - t
            marcas = revisar(texto, self.verificadores, ctx)
            rondas.append(Ronda(f"reescritura {r + 1}", self.costos["reescritura"], penalidad(marcas), cambios))
            if cambios == 0:
                motivo = "ninguna alternativa mejoró"
                break
        else:
            motivo = "máximo de rondas"
        return InformeEscritura(texto, borrador, rondas, marcas, presupuesto, self.clock() - t0, esquema, motivo)

    def _fragmentos(self, texto: str, marcas: list[Marca]) -> list[tuple[int, int, float]]:
        """Oraciones que tocan alguna marca, de mayor a menor severidad."""
        sev: dict[tuple[int, int], float] = defaultdict(float)
        spans = oraciones(texto)
        for m in marcas:
            if m.inicio < 0:
                continue
            for a, b in spans:
                if a < max(m.fin, m.inicio + 1) and m.inicio < b:
                    sev[(a, b)] += m.severidad
        orden = sorted(sev.items(), key=lambda kv: -kv[1])[:self.max_frag]
        return sorted(((a, b, s) for (a, b), s in orden), key=lambda x: -x[0])   # del final al principio

    def _reescribir(self, contexto: str, texto: str, frags, ctx: ContextoRevision) -> tuple[str, int]:
        cambios = 0
        for a, b, _sev in frags:                           # del final al principio: los tramos no se corren
            prev, frag, sig = texto[:a], texto[a:b], texto[b:]
            falta = [m for m in revisar(texto, self.verificadores, ctx) if m.inicio < 0 and m.tipo == "contenido"]
            pista = ""
            if falta and ctx.esquema:                     # orienta una alternativa hacia lo que falta
                hay = {w[:5] for w in palabras(texto)}
                faltan = [w for w in ctx.esquema.palabras_clave if w[:5] not in hay]
                pista = faltan[0].capitalize() + " " if faltan and not prev.rstrip().endswith(",") else ""
            prompts = [contexto + prev] * max(self.n_alt - (1 if pista else 0), 1)
            if pista:
                prompts.append(contexto + prev + pista)
            outs = self.gen(prompts, min(max(2 * len(frag.encode()), 48), 400), self.temp)
            alts = []
            for i, o in enumerate(outs):
                s = o.decode("utf-8", errors="ignore")
                if pista and i == len(outs) - 1:
                    s = pista + s
                s = _primera_oracion(s).strip()
                if s and s != frag:
                    alts.append(s)
            if not alts:
                continue
            def total(f: str) -> tuple[float, str]:
                nuevo = prev + f + sig
                return penalidad(revisar(nuevo, self.verificadores, ctx)), nuevo
            base_pen, _ = total(frag)
            cand = [(total(s), s) for s in alts]
            lps = self._logprob(contexto + prev, [s + sig[:64] for s in [frag] + [c[1] for c in cand]])
            base_lp, lps = lps[0], lps[1:]
            best = min(range(len(cand)), key=lambda i: (cand[i][0][0], -lps[i]))
            (pen, nuevo), s = cand[best]
            if pen < base_pen or (pen == base_pen and lps[best] > base_lp + 0.05):
                texto = nuevo
                cambios += 1
        return texto, cambios


def agente_escritor(escritor: Escritor, nombre: str = "escritor", largo: int = 300):
    """``Agent`` para ``BudgetedResponder``: su «solución» es el texto del bucle de escritura (sin
    respuesta numérica). Así la escalada de calidad del presupuesto puede incluir la escritura."""
    from hope.council.agents import Agent, AgentProfile

    def solver(topic, problem, profile):
        inf = escritor.escribir(problem, tema=topic, largo=profile.max_bytes)
        return inf.texto, None
    return Agent(AgentProfile(nombre, "escribir bien", "", max_bytes=largo, use_tools=False), solver)


# ---------------------------------------------------------------------------
# Resumen continuo de la conversación (técnica 7)
# ---------------------------------------------------------------------------
def resumir_extractivo(textos: list[str], max_chars: int = 600, excluir: set[str] | None = None) -> str:
    """Resumen extractivo sin modelo: las oraciones con más palabras frecuentes de la conversación
    (frecuencia normalizada, sin vacías), en el orden original, hasta ``max_chars``."""
    sents = []
    for ti, t in enumerate(textos):
        for a, b in oraciones(t):
            s = t[a:b].strip()
            if len(s) > 15 and (not excluir or s not in excluir):
                sents.append((ti, a, s))
    if not sents:
        return ""
    freq = Counter(w for _, _, s in sents for w in palabras(s) if w not in VACIAS and len(w) > 2)
    top = max(freq.values(), default=1)

    def score(s: str) -> float:
        ws = [w for w in palabras(s) if w not in VACIAS and len(w) > 2]
        return sum(freq[w] / top for w in set(ws)) / (1 + 0.1 * len(ws)) if ws else 0.0
    ranked = sorted(range(len(sents)), key=lambda i: -score(sents[i][2]))
    keep, used = [], 0
    for i in ranked:
        s = sents[i][2]
        if used + len(s) + 1 > max_chars:
            continue
        keep.append(i)
        used += len(s) + 1
    return " ".join(sents[i][2] for i in sorted(keep))


class ResumenContinuo:
    """Resumen de una conversación guardado en el cuaderno (``Notebook``) y actualizado en cada turno.

    La nota se llama «resumen de la conversación <id>»; cada versión nueva deja la anterior en el
    historial de la nota. Guarda además los últimos turnos completos (``recientes``)."""

    def __init__(self, notebook=None, conv_id: str = "principal", *, max_chars: int = 600, recientes: int = 2):
        self.nb, self.conv_id = notebook, conv_id
        self.max_chars, self.recientes = max_chars, recientes
        self.turnos: list[tuple[str, str]] = []
        self.resumen = ""
        if notebook is not None:
            n = notebook._find(self.concepto)
            if n is not None:
                self.resumen = notebook.store.get(n).content

    @property
    def concepto(self) -> str:
        return f"resumen de la conversación {self.conv_id}"

    def actualizar(self, usuario: str, respuesta: str) -> str:
        self.turnos.append((usuario.strip(), respuesta.strip()))
        viejos = self.turnos[:-self.recientes] if len(self.turnos) > self.recientes else []
        textos = ([self.resumen] if self.resumen else []) + [f"{u} {r}" for u, r in viejos[-4:]]
        if viejos:
            self.resumen = resumir_extractivo(textos, self.max_chars)
            if self.nb is not None and self.resumen:
                self.nb.write(self.concepto, self.resumen, source="resumen de conversación")
        return self.resumen

    def texto(self) -> str:
        return self.resumen


class ConversacionConResumen:
    """``Conversation`` que no olvida: cuando el historial más lo pedido ya no entra en
    ``max_context``, la conversación se reinicia desde «resumen + últimos turnos» (y la caché se
    reconstruye una sola vez) en lugar de caer en la ventana deslizante, que pierde el principio.

    Expone ``history``, ``reset`` y ``say`` como ``Conversation``, así sirve en ``charla.py`` tal cual."""

    def __init__(self, model, resumen: ResumenContinuo, *, max_context: int = 2048, margen: float = 0.9,
                 prefijo_resumen: str = "Resumen de la conversación hasta ahora: ", **kw):
        from hope.core.batched import Conversation
        self.model, self.resumen, self.max_context, self.margen = model, resumen, max_context, margen
        self.prefijo = prefijo_resumen
        self.conv = Conversation(model, max_context=max_context, **kw)
        self.compactaciones = 0
        self._turnos_txt: list[bytes] = []

    @property
    def history(self) -> list[int]:
        return self.conv.history

    @property
    def stats(self):
        return self.conv.stats

    def reset(self) -> None:
        self.conv.reset()
        self._turnos_txt = []

    def _compactar(self, nuevo: int, max_new: int) -> None:
        if len(self.conv.history) + nuevo + max_new <= self.margen * self.max_context:
            return
        base = (self.prefijo + self.resumen.texto() + "\n\n").encode("utf-8") if self.resumen.texto() else b""
        ult = b"".join(self._turnos_txt[-self.resumen.recientes:])
        libre = max(int(self.margen * self.max_context) - nuevo - max_new, 0)
        base = base[:libre // 2].decode("utf-8", errors="ignore").encode("utf-8") if len(base) > libre // 2 else base
        espacio = libre - len(base)
        ult = ult[-espacio:].decode("utf-8", errors="ignore").encode("utf-8") if espacio > 0 else b""
        self.conv.reset()
        self.conv.history.extend(base + ult)
        self.compactaciones += 1

    def say(self, text: bytes | str, max_new: int = 256, **kw) -> bytes:
        tb = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        self._compactar(len(tb), max_new)
        out = self.conv.say(tb, max_new, **kw)
        self._turnos_txt.append(tb + out)
        self.resumen.actualizar(tb.decode("utf-8", errors="ignore"), out.decode("utf-8", errors="ignore"))
        return out
