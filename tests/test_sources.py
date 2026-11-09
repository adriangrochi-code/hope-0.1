"""Corpus de bytes y fuentes locales (hope/data/corpus.py, hope/data/sources.py)."""

import numpy as np

from hope.data.corpus import ByteData, BinWriter
from hope.data.sources import code_docs, main, parse_size, split_spec
from hope.train.run import _sources


def test_parse_size_and_spec():
    assert parse_size("120M") == 120_000_000 and parse_size("2K") == 2000 and parse_size("1.5G") == 1_500_000_000
    assert split_spec("/usr/lib/python3.13:60M") == ("/usr/lib/python3.13", 60_000_000)


def test_writer_budget_and_split(tmp_path):
    w = BinWriter(tmp_path / "f", budget=1000, val_every=4)
    n = 0
    while w.add(f"documento {n} " * 5):
        n += 1
    s = w.close()
    assert s["bytes"] >= 1000 and s["train_bytes"] + s["val_bytes"] == s["bytes"]
    val = (tmp_path / "f" / "val.bin").read_bytes()
    assert val.count(b"\x00") == s["docs"] // 4
    d = ByteData(tmp_path / "f" / "train.bin")
    assert d.data.dtype == np.uint8 and len(d) == s["train_bytes"]


def test_code_source_feeds_the_run_plan(tmp_path):
    src = tmp_path / "src"
    (src / "pkg" / "__pycache__").mkdir(parents=True)
    (src / "pkg" / "a.py").write_text("def f():\n    return 1\n" * 300)
    (src / "pkg" / "__pycache__" / "x.py").write_text("basura")
    (src / "notas.txt").write_text("no es código")
    assert [m["ruta"] for _, m in code_docs(src)] == ["pkg/a.py"]
    for i in range(200):
        (src / f"m{i}.py").write_text(f"x = {i}\n" * 50)
    main(["--out", str(tmp_path / "pre"), "--code", f"{src}:50K", "--val-cada", "10"])
    folder = tmp_path / "pre" / "codigo_src"
    train = (folder / "train.bin").read_bytes()
    assert train.startswith(b"# archivo: m0.py\n") and 40_000 <= len(train) < 55_000   # respeta el presupuesto
    fuentes = _sources({"codigo": f"{folder}:1.0"})
    assert fuentes[0].name == "codigo" and len(fuentes[0].data) > 1024
