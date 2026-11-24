"""TISMA CNC on old computers: Python 3.7 (Windows Vista), browsers from Firefox 52 ESR / Chrome 57 on, and
only the standard library for the core. These tests keep a change from breaking that promise."""

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

STATIC = Path("hope/cnc/app/static")


def test_compat_css_is_up_to_date():
    from hope.cnc.app.compat_css import generate
    assert (STATIC / "compat.css").read_text(encoding="utf-8") == generate(), \
        "regenerar con: python -m hope.cnc.app.compat_css"


@pytest.mark.parametrize("name", ["cnc.js", "pathview.js"])
def test_js_avoids_features_missing_in_old_browsers(name):
    js = (STATIC / name).read_text(encoding="utf-8")
    code = re.sub(r"^\s*/\*.*?\*/", "", js, flags=re.DOTALL | re.MULTILINE)
    code = re.sub(r"(^|\s)//.*", "", code)
    for bad in ("?.", "??", ".replaceAll(", ".at(", "Object.fromEntries", ".flatMap(", "structuredClone",
                ".text()", "new ResizeObserver"):
        assert bad not in code, f"{name} usa {bad}: no existe en Firefox 52 / Chrome 57 (usar compat.js)"
    if name == "cnc.js":
        assert "pointerdown" not in code, "usar onPress() de compat.js (ratón/táctil sin pointer events)"
    else:
        assert "window.PointerEvent" in code              # pointer events only with their fallback


def test_css_has_fallbacks():
    css = (STATIC / "cnc.css").read_text(encoding="utf-8")
    assert "inset:" not in css                           # Chrome < 87, Firefox < 66
    assert not re.search(r"(?<![\w-])(min|max)\(", css)    # grid minmax() does exist since 2017
    for m in re.finditer(r"[^;{]*clamp\(", css):         # clamp() always after a fixed value
        before = css[:m.start()].rsplit("{", 1)[1]
        assert "font" in before
    for m in re.finditer(r"([\w-]+):\s*color-mix", css):
        prop = m.group(1)
        rule = css[:m.start()].rsplit("{", 1)[1]
        assert prop == "text-shadow" or f"{prop}:" in rule, f"color-mix sin alternativa en {prop}"
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert html.index("compat.js") < html.index("pathview.js") < html.index("cnc.js")
    assert "compat.css" in html


def test_core_needs_only_the_standard_library():
    code = ("import sys, hope.cnc.app.server, hope.cnc.links, hope.cnc.session, hope.cnc.profiles, hope.cnc.__main__\n"
            "bad = sorted({m.split('.')[0] for m in sys.modules} & {'yaml', 'numpy', 'torch', 'serial', 'scipy'})\n"
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


@pytest.mark.skipif(shutil.which("vermin") is None and importlib.util.find_spec("vermin") is None,
                    reason="vermin no instalado (pip install vermin)")
def test_python_37_compatible():
    exe = shutil.which("vermin") or None
    cmd = [exe] if exe else [sys.executable, "-m", "vermin"]
    out = subprocess.run(cmd + ["--no-tips", "-q", "-t=3.7-", "--violations", "hope/cnc"],
                         capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stdout + out.stderr
