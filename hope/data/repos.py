"""Datos de entrenamiento a partir de los repositorios del usuario (pedido del usuario).

De cada repositorio (clon local) salen dos fuentes:

* **código** — los archivos de código y documentación en la versión actual;
* **historial** — cada commit como episodio ``mensaje → archivos → diff``. Los commits que
  arreglan algo (palabras como *fix, arreglo, corrige, error, bug, falla*) se marcan como
  ``arreglo``: muestran el código con el error, la descripción y la corrección. Es
  exactamente lo que necesita Hope Code para aprender a depurar.

**Autoría.** Los commits con coautoría de una IA (``Co-Authored-By: Claude``,
``noreply@anthropic.com``, ``Claude-Session``, «Generated with Claude Code») se
**excluyen por defecto**, y también los archivos cuyo historial completo es de commits así.
Motivo: los términos de uso de varios proveedores de IA, incluido Anthropic, restringen
usar sus salidas para entrenar otros modelos. Revisá los términos que te aplican; si
corresponde, ``--incluir-coautoria-ia`` los incorpora. En repos que son forks (Marlin,
PrusaSlicer…) ``--autor`` limita el historial a tus commits; el código queda con la
licencia del proyecto original, que se registra en la procedencia.

Uso::

    python -m hope.data.repos --repos ~/git/* --out data/repos --autor adriangrochi
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from hope.data.sources import CODE_EXT, build, parse_size

DOC_EXT = {".md", ".txt", ".rst", ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini"}
AI_MARKS = re.compile(r"(co-authored-by:\s*claude|noreply@anthropic\.com|claude-session:|generated with \[?claude code)",
                      re.I)
FIX_WORDS = re.compile(r"\b(fix(e[sd])?|bug|arregl\w*|corrig\w*|correc\w*|error(es)?|falla\w*|crash|solucion\w*|"
                       r"hotfix|parche|repar\w*)\b", re.I)


@dataclass
class Commit:
    sha: str
    author: str
    email: str
    subject: str
    body: str
    files: list[str]

    @property
    def ai(self) -> bool:
        return bool(AI_MARKS.search(self.body) or AI_MARKS.search(self.subject))

    @property
    def is_fix(self) -> bool:
        return bool(FIX_WORDS.search(self.subject) or FIX_WORDS.search(self.body[:500]))


def _git(repo: Path, *args: str, timeout: float = 120) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout,
                       errors="ignore")
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])
    return r.stdout


def commits(repo: Path, max_commits: int = 5000) -> list[Commit]:
    raw = _git(repo, "log", "--no-merges", f"-n{max_commits}", "--name-only",
               "--format=%x1e%H%x00%an%x00%ae%x00%s%x00%b%x00", timeout=600)
    out = []
    for rec in raw.split("\x1e")[1:]:
        parts = rec.split("\x00")
        if len(parts) < 6:
            continue
        sha, an, ae, subj, body, files = parts[:6]
        out.append(Commit(sha, an, ae, subj.strip(), body.strip(), [f for f in files.split("\n") if f.strip()]))
    return out


def _wanted(c: Commit, include_ai: bool, authors: list[str] | None) -> bool:
    if c.ai and not include_ai:
        return False
    if authors and not any(a.lower() in (c.author + " " + c.email).lower() for a in authors):
        return False
    return True


def history_docs(repo: Path, *, include_ai: bool = False, authors: list[str] | None = None,
                 max_diff: int = 12_000, max_commits: int = 5000):
    name = repo.name
    for c in commits(repo, max_commits):
        if not _wanted(c, include_ai, authors):
            continue
        try:
            diff = _git(repo, "show", "--format=", "--unified=3", "--no-color", "--no-ext-diff", c.sha)
        except (RuntimeError, subprocess.TimeoutExpired):
            continue
        if not diff.strip() or "\x00" in diff:
            continue
        if len(diff) > max_diff:
            diff = diff[:max_diff] + "\n[… diff recortado …]\n"
        kind = "arreglo" if c.is_fix else "cambio"
        msg = c.subject + (f"\n{c.body}" if c.body else "")
        msg = AI_MARKS.sub("", msg).strip()
        text = (f"⟦repo: {name}⟧ ⟦{kind}⟧\n"
                f"{'Problema' if kind == 'arreglo' else 'Tarea'}: {msg}\n"
                f"Archivos: {', '.join(c.files[:20])}\n"
                f"{'Corrección' if kind == 'arreglo' else 'Cambio'}:\n{diff}")
        yield text, {"fuente": "repo-historial", "repo": name, "commit": c.sha, "tipo": kind, "ia": c.ai}


def code_docs(repo: Path, *, include_ai: bool = False, authors: list[str] | None = None,
              max_file: int = 200_000):
    name = repo.name
    touched: dict[str, list[bool]] = {}
    try:
        for c in commits(repo):
            for f in c.files:
                touched.setdefault(f, []).append(c.ai)
    except RuntimeError:
        pass
    files = _git(repo, "ls-files").splitlines()
    for rel in files:
        p = repo / rel
        if p.suffix.lower() not in CODE_EXT | DOC_EXT or not p.is_file() or p.stat().st_size > max_file:
            continue
        flags = touched.get(rel, [])
        if not include_ai and flags and all(flags):
            continue                                   # todo su historial es de commits con IA
        text = p.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            yield f"⟦repo: {name}⟧ ⟦archivo: {rel}⟧\n{text}", {"fuente": "repo-codigo", "repo": name, "ruta": rel}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Corpus de entrenamiento a partir de repositorios git propios")
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--autor", action="append", default=None, help="solo commits de este autor (nombre o email)")
    ap.add_argument("--incluir-coautoria-ia", action="store_true")
    ap.add_argument("--max-codigo", default="500M")
    ap.add_argument("--max-historial", default="500M")
    args = ap.parse_args(argv)
    repos = [Path(r).expanduser() for r in args.repos if (Path(r).expanduser() / ".git").exists()]

    def all_code():
        for r in repos:
            yield from code_docs(r, include_ai=args.incluir_coautoria_ia)

    def all_history():
        for r in repos:
            yield from history_docs(r, include_ai=args.incluir_coautoria_ia, authors=args.autor)

    # El historial no pasa las reglas de prosa: un diff no «parece» texto, pero es valioso.
    s1 = build(args.out, [("repos_codigo", all_code(), parse_size(args.max_codigo))])
    s2 = build(args.out, [("repos_historial", all_history(), parse_size(args.max_historial))], apply_rules=False)
    print(json.dumps({**s1, **s2}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
