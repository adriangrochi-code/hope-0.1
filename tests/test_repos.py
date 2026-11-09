"""Corpus a partir de repositorios git propios (hope/data/repos.py)."""

import subprocess

from hope.data.repos import code_docs, commits, history_docs, main


def _repo(tmp_path):
    r = tmp_path / "mirepo"
    r.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(r), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "yo@example.com")
    run("config", "user.name", "Yo")
    (r / "calc.py").write_text("def div(a, b):\n    return a / b\n")
    run("add", ".")
    run("commit", "-qm", "calculadora inicial")
    (r / "calc.py").write_text("def div(a, b):\n    if b == 0:\n        raise ValueError('b = 0')\n    return a / b\n")
    run("commit", "-qam", "fix: división por cero")
    (r / "ia.py").write_text("def hecho_con_ia():\n    return 42\n")
    run("add", ".")
    run("commit", "-qm", "agregar helper\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
    return r


def test_history_marks_fixes_and_excludes_ai_by_default(tmp_path):
    r = _repo(tmp_path)
    cs = commits(r)
    assert [c.is_fix for c in cs] == [False, True, False] and cs[0].ai
    docs = list(history_docs(r))
    assert len(docs) == 2
    fix = next(t for t, m in docs if m["tipo"] == "arreglo")
    assert "Problema: fix: división por cero" in fix and "+    if b == 0:" in fix
    assert len(list(history_docs(r, include_ai=True))) == 3
    assert len(list(history_docs(r, authors=["otro"]))) == 0
    paths = [m["ruta"] for _, m in code_docs(r)]
    assert paths == ["calc.py"]                       # ia.py solo tiene historial con IA
    assert sorted(m["ruta"] for _, m in code_docs(r, include_ai=True)) == ["calc.py", "ia.py"]


def test_cli_builds_bins(tmp_path):
    r = _repo(tmp_path)
    main(["--repos", str(r), "--out", str(tmp_path / "out")])
    hist = (tmp_path / "out" / "repos_historial")
    data = (hist / "train.bin").read_bytes() + (hist / "val.bin").read_bytes()
    assert b"Correcci" in data and b"Co-Authored-By" not in data
