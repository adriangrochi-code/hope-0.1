"""Filtro de calidad del corpus (ARQUITECTURA_HN1 §10.1.1; mejora sin parámetros n.º 1).

Con cómputo limitado no se usa todo el TB: se elige **lo mejor**. Dos etapas:

1. **Reglas** (estilo Gopher/C4, Rae et al. 2021; Raffel et al. 2020): descartan basura
   evidente sin costo — texto muy corto, palabras de largo absurdo, exceso de símbolos,
   líneas repetidas, pocas letras. El código tiene reglas propias (no se le exige
   puntuación de prosa).
2. **Clasificador** (estilo FineWeb-Edu, Penedo et al. 2024): Qwen puntúa una muestra
   de 0 a 5 según su valor educativo; con esas notas se entrena un regresor lineal
   sobre n-gramas de bytes con hashing (rápido en CPU, sin GPU, al estilo fastText)
   y se aplica al resto del corpus. FineWeb-Edu mostró que quedarse con lo mejor
   puntuado rinde como varias veces más datos sin filtrar.

Uso::

    python -m hope.data.quality puntuar --input muestra.txt --out notas.jsonl --n 20000
    python -m hope.data.quality entrenar --notas notas.jsonl --out data/calidad.npz
    python -m hope.data.quality filtrar --modelo data/calidad.npz --input corpus/ --out filtrado/ --umbral 2.5

Los documentos se separan por línea en blanco doble o por archivo.
"""

from __future__ import annotations

import json
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCORE_PROMPT = """Puntuá de 0 a 5 el valor de este texto para enseñarle a un modelo que aprende a razonar
(ciencia, ingeniería, matemática, programación, idioma, conocimiento general):
0 = basura, spam o sin sentido; 1 = publicidad o texto vacío; 2 = algo de información, mal escrito;
3 = informativo y correcto; 4 = muy claro, explica conceptos; 5 = excelente material didáctico.
Respondé SOLO con JSON: {"nota": <0-5>, "motivo": "<breve>"}

TEXTO:
"""

CODE_HINT = re.compile(r"(def |class |#include|function |return |import |=>|\{\s*$|;\s*$)", re.M)


# ---------------------------------------------------------------------------
@dataclass
class RuleReport:
    keep: bool
    reasons: list[str]
    is_code: bool


def rules(text: str, *, min_chars: int = 200, lengua: float | None = None) -> RuleReport:
    """``lengua``: máximo de errores de lengua cada 100 palabras (``hope.tools.lenguaje.evaluar_lengua``);
    None = no se revisa (lo de siempre). Solo se aplica a prosa, no a código."""
    reasons = []
    lines = [line for line in text.splitlines() if line.strip()]
    is_code = len(CODE_HINT.findall(text)) >= max(3, len(lines) // 10)
    if len(text) < min_chars:
        reasons.append("corto")
    words = re.findall(r"\w+", text)
    if words:
        mean = sum(len(w) for w in words) / len(words)
        if not is_code and not 2.5 <= mean <= 12:
            reasons.append("largo de palabra anómalo")
    else:
        reasons.append("sin palabras")
    letters = sum(ch.isalpha() for ch in text)
    if letters / max(len(text), 1) < (0.35 if is_code else 0.6):
        reasons.append("pocas letras")
    if lines:
        dup = 1 - len(set(lines)) / len(lines)
        if dup > 0.3:
            reasons.append("líneas repetidas")
        if not is_code and len(lines) >= 5:
            ends = sum(line.rstrip()[-1:] in ".!?:;»\")" for line in lines) / len(lines)
            short = sum(len(line) < 30 for line in lines) / len(lines)
            if ends < 0.1 and short > 0.7:
                reasons.append("lista o menú sin prosa")
    # Repetición: texto que se comprime demasiado es casi todo repetido.
    raw = text.encode("utf-8", "ignore")
    if len(raw) > 500 and len(zlib.compress(raw)) / len(raw) < 0.12:
        reasons.append("muy repetitivo")
    if re.search(r"(lorem ipsum|click here|haz clic aquí|suscríbete|cookies)", text, re.I):
        reasons.append("frases de plantilla")
    if lengua is not None and not is_code and not reasons:
        from hope.tools.lenguaje import evaluar_lengua
        ev = evaluar_lengua(text[:4000], gramatica_completa=False)
        if ev["errores_cada_100"] > lengua:
            reasons.append(f"muchos errores de lengua ({ev['errores_cada_100']:.1f} cada 100 palabras)")
    return RuleReport(not reasons, reasons, is_code)


# ---------------------------------------------------------------------------
def _features(text: str, dim: int, ngrams: tuple[int, ...]) -> np.ndarray:
    """Conteo de n-gramas de bytes con hashing, vectorizado (≈ 1 ms por documento de 20 KB)."""
    raw = np.frombuffer(text.encode("utf-8", "ignore")[:20000].lower(), dtype=np.uint8).astype(np.uint64)
    v = np.zeros(dim, dtype=np.float32)
    with np.errstate(over="ignore"):
        for n in ngrams:
            if len(raw) < n:
                continue
            h = np.full(len(raw) - n + 1, np.uint64(n), dtype=np.uint64)
            for i in range(n):
                h = h * np.uint64(1099511628211) + raw[i:len(raw) - n + 1 + i]
            h ^= h >> np.uint64(29)
            v += np.bincount((h % np.uint64(dim)).astype(np.int64), minlength=dim).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm else v


class QualityClassifier:
    """Regresor lineal (ridge) sobre n-gramas de bytes con hashing. Predice la nota 0–5."""

    def __init__(self, dim: int = 1 << 16, ngrams: tuple[int, ...] = (3, 4, 5), l2: float = 1.0):
        self.dim, self.ngrams, self.l2 = dim, tuple(ngrams), l2
        self.w = np.zeros(dim, dtype=np.float32)
        self.b = 0.0

    def featurize(self, texts: list[str]) -> np.ndarray:
        return np.stack([_features(t, self.dim, self.ngrams) for t in texts])

    def fit(self, texts: list[str], scores: list[float], epochs: int = 30, lr: float = 0.5) -> dict:
        x = self.featurize(texts)
        y = np.asarray(scores, dtype=np.float32)
        self.b = float(y.mean())
        n = len(y)
        for _ in range(epochs):                      # descenso por gradiente completo (pocas muestras)
            err = x @ self.w + self.b - y
            self.w -= lr * (x.T @ err / n + self.l2 * self.w / n)
            self.b -= lr * float(err.mean())
        pred = x @ self.w + self.b
        return {"mae": float(np.abs(pred - y).mean()), "n": n}

    def score(self, texts: list[str]) -> np.ndarray:
        return self.featurize(texts) @ self.w + self.b

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, w=self.w, b=self.b, dim=self.dim, ngrams=np.array(self.ngrams), l2=self.l2)

    @classmethod
    def load(cls, path: str | Path) -> "QualityClassifier":
        d = np.load(path)
        c = cls(int(d["dim"]), tuple(int(v) for v in d["ngrams"]), float(d["l2"]))
        c.w, c.b = d["w"].astype(np.float32), float(d["b"])
        return c


NOTA_RE = re.compile(r'"nota"\s*:\s*"?([0-5](?:\.\d+)?)')


def teacher_score(teacher, text: str, max_chars: int = 3000) -> float | None:
    """Nota 0–5 del maestro. Lee la nota aunque el JSON quede cortado (``max_tokens`` chico
    para no pagar el «motivo» en CPU)."""
    reply = teacher.ask([{"role": "user", "content": SCORE_PROMPT + text[:max_chars]}], purpose="data.quality")
    m = re.search(r"\{.*\}", reply.text, re.DOTALL)
    try:
        if m:
            return float(json.loads(m.group(0))["nota"])
    except (ValueError, KeyError, TypeError):
        pass
    m = NOTA_RE.search(reply.text)
    return float(m.group(1)) if m else None


@dataclass
class FilterReport:
    seen: int = 0
    kept: int = 0
    rule_drops: Counter = None
    low_score: int = 0

    def __post_init__(self):
        self.rule_drops = self.rule_drops or Counter()


def filter_docs(docs, classifier: QualityClassifier | None = None, threshold: float = 2.5,
                batch: int = 256):
    """Generador: devuelve (documento, nota) de los que pasan reglas y umbral. Lleva un informe."""
    report = FilterReport()
    pending: list[str] = []

    def flush():
        if not pending:
            return []
        scores = classifier.score(pending) if classifier is not None else np.full(len(pending), np.inf)
        out = [(d, float(s)) for d, s in zip(pending, scores) if s >= threshold]
        report.low_score += len(pending) - len(out)
        pending.clear()
        return out

    for doc in docs:
        report.seen += 1
        r = rules(doc)
        if not r.keep:
            report.rule_drops.update(r.reasons)
            continue
        pending.append(doc)
        if len(pending) >= batch:
            for item in flush():
                report.kept += 1
                yield item
    for item in flush():
        report.kept += 1
        yield item
    filter_docs.last_report = report


def iter_docs(path: str | Path):
    p = Path(path)
    files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        for doc in re.split(r"\n\s*\n\s*\n", text):
            if doc.strip():
                yield doc.strip()


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Filtro de calidad del corpus")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("puntuar")
    a.add_argument("--input", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=20000)
    a.add_argument("--teacher", default="configs/teacher.yaml")
    a.add_argument("--guardian", default="configs/guardian.yaml")
    a.add_argument("--max-chars", type=int, default=3000, help="caracteres del documento que ve el maestro")
    b = sub.add_parser("entrenar")
    b.add_argument("--notas", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--epocas", type=int, default=30)
    b.add_argument("--lr", type=float, default=0.5)
    b.add_argument("--validacion", type=float, default=0.0,
                   help="fracción apartada para medir el error en notas no vistas (luego se reentrena con todo)")
    c = sub.add_parser("filtrar")
    c.add_argument("--modelo", default=None)
    c.add_argument("--input", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--umbral", type=float, default=2.5)
    args = ap.parse_args(argv)
    if args.cmd == "puntuar":
        from guardian.cli import load as load_guardian
        from hope.teacher.client import load_teacher
        t = load_teacher(args.teacher, load_guardian(args.guardian))
        out = Path(args.out)
        # Reanudable: los documentos ya vistos (puntuados o no) se saltean por su índice.
        done = {json.loads(line).get("i") for line in out.read_text(encoding="utf-8").splitlines()
                if line.strip()} if out.exists() else set()
        with open(out, "a", encoding="utf-8") as f:
            for i, doc in enumerate(d for d in iter_docs(args.input) if rules(d).keep):
                if i >= args.n:
                    break
                if i in done:
                    continue
                s = teacher_score(t, doc, max_chars=args.max_chars)
                f.write(json.dumps({"i": i, "texto": doc[:args.max_chars], "nota": s,
                                    "modelo": t.config.model}, ensure_ascii=False) + "\n")
                f.flush()
    elif args.cmd == "entrenar":
        rows = [json.loads(line) for line in Path(args.notas).read_text(encoding="utf-8").splitlines() if line]
        rows = [r for r in rows if r.get("nota") is not None]
        report = {}
        if args.validacion > 0:
            import random
            shuffled = rows[:]
            random.Random(0).shuffle(shuffled)
            k = max(1, int(len(shuffled) * args.validacion))
            val, tr = shuffled[:k], shuffled[k:]
            clf = QualityClassifier()
            clf.fit([r["texto"] for r in tr], [r["nota"] for r in tr], epochs=args.epocas, lr=args.lr)
            y = np.asarray([r["nota"] for r in val], dtype=np.float32)
            mean = float(np.mean([r["nota"] for r in tr]))
            report["validacion"] = {"n": k, "mae": float(np.abs(clf.score([r["texto"] for r in val]) - y).mean()),
                                    "mae_prediciendo_la_media": float(np.abs(y - mean).mean())}
        clf = QualityClassifier()
        report["entrenamiento"] = clf.fit([r["texto"] for r in rows], [r["nota"] for r in rows],
                                          epochs=args.epocas, lr=args.lr)
        print(json.dumps(report, ensure_ascii=False))
        clf.save(args.out)
    else:
        clf = QualityClassifier.load(args.modelo) if args.modelo else None
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "filtrado.txt", "w", encoding="utf-8") as f:
            for doc, _ in filter_docs(iter_docs(args.input), clf, args.umbral):
                f.write(doc + "\n\n\n")
        r = filter_docs.last_report
        print(json.dumps({"vistos": r.seen, "conservados": r.kept, "nota_baja": r.low_score,
                          "reglas": dict(r.rule_drops)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
