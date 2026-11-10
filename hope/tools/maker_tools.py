"""Herramientas de electrónica e impresión 3D (docs/HERRAMIENTAS_HN1.md §3, catálogo ampliado).

* ``resistor`` — código de colores (3–6 bandas) en ambos sentidos, series E12/E24/E96 y
  resistencia para un LED.
* ``gcode_analyze`` — análisis de G-code (Marlin/Klipper): tiempo estimado sin aceleración,
  capas, longitud y peso de filamento, temperaturas, movimientos fuera de la cama, comandos
  desconocidos y macros de Klipper.
* ``stl_analyze`` — STL ASCII o binario con NumPy puro: volumen, área, dimensiones, malla cerrada,
  agujeros, aristas no variedad, normales invertidas o inconsistentes, cuerpos.
* ``marlin_config`` — lee Configuration.h / Configuration_adv.h y reporta valores clave e
  incoherencias comunes.
* ``material`` — tabla de materiales de impresión con temperaturas y velocidades típicas y su
  fuente; puede leer perfiles del usuario (PrusaSlicer .ini u OrcaSlicer/Bambu .json).

Los analizadores de archivos (G-code, STL, Marlin, perfiles) se ejecutan en el sandbox desde el
catálogo, porque los archivos no son confiables; acá están como funciones puras.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Resistencias

_DIGIT = {"negro": 0, "marron": 1, "cafe": 1, "rojo": 2, "naranja": 3, "amarillo": 4, "verde": 5, "azul": 6,
          "violeta": 7, "morado": 7, "gris": 8, "blanco": 9,
          "black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4, "green": 5, "blue": 6, "violet": 7,
          "grey": 8, "gray": 8, "white": 9}
_MULT_EXTRA = {"dorado": -1, "oro": -1, "gold": -1, "plateado": -2, "plata": -2, "silver": -2}
_TOL = {1: 1.0, 2: 2.0, 5: 0.5, 6: 0.25, 7: 0.1, 8: 0.05, -1: 5.0, -2: 10.0}       # IEC 60062
_TCR = {1: 100, 2: 50, 3: 15, 4: 25}                                               # ppm/K
_NAMES = ["negro", "marrón", "rojo", "naranja", "amarillo", "verde", "azul", "violeta", "gris", "blanco"]

E12 = [10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82]
E24 = [10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91]
E96 = [100, 102, 105, 107, 110, 113, 115, 118, 121, 124, 127, 130, 133, 137, 140, 143, 147, 150, 154, 158,
       162, 165, 169, 174, 178, 182, 187, 191, 196, 200, 205, 210, 215, 221, 226, 232, 237, 243, 249, 255,
       261, 267, 274, 280, 287, 294, 301, 309, 316, 324, 332, 340, 348, 357, 365, 374, 383, 392, 402, 412,
       422, 432, 442, 453, 464, 475, 487, 499, 511, 523, 536, 549, 562, 576, 590, 604, 619, 634, 649, 665,
       681, 698, 715, 732, 750, 768, 787, 806, 825, 845, 866, 887, 909, 931, 953, 976]
SERIES = {"E12": E12, "E24": E24, "E96": E96}


def _fold(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def parse_ohms(tok: str) -> float:
    """«4k7», «4.7k», «470R», «2R2», «1M», «0,22», «10 kΩ»."""
    t = tok.strip().replace("Ω", "").replace("ohm", "").replace(",", ".").replace(" ", "")
    m = re.fullmatch(r"(\d+)([rRkKmMgG])(\d+)", t)
    if m:
        mult = {"r": 1, "k": 1e3, "m": 1e6, "g": 1e9}[m.group(2).lower()]
        return float(f"{m.group(1)}.{m.group(3)}") * mult
    m = re.fullmatch(r"(\d*\.?\d+(?:e[-+]?\d+)?)([rRkKmMgG]?)", t)
    if not m:
        raise ValueError(f"valor de resistencia inválido: {tok}")
    mult = {"": 1, "r": 1, "k": 1e3, "m": 1e6, "g": 1e9}[m.group(2).lower()]
    return float(m.group(1)) * mult


def fmt_ohms(v: float) -> str:
    for div, suf in ((1e9, " GΩ"), (1e6, " MΩ"), (1e3, " kΩ"), (1, " Ω")):
        if abs(v) >= div or div == 1:
            return f"{v / div:.6g}{suf}"
    return f"{v:g} Ω"


def colors_to_value(bands: list[str]) -> dict:
    b = [_fold(x) for x in bands]
    if len(b) not in (3, 4, 5, 6):
        raise ValueError("entre 3 y 6 bandas")
    nd = 2 if len(b) <= 4 else 3
    try:
        digits = [_DIGIT[x] for x in b[:nd]]
    except KeyError as e:
        raise ValueError(f"color de cifra inválido: {e.args[0]}") from None
    mb = b[nd]
    exp = _DIGIT.get(mb, _MULT_EXTRA.get(mb))
    if exp is None:
        raise ValueError(f"color de multiplicador inválido: {mb}")
    value = int("".join(map(str, digits))) * 10.0 ** exp
    value = round(value, 6)
    tol = 20.0
    if len(b) > nd + 1:
        tb = b[nd + 1]
        code = _DIGIT.get(tb, _MULT_EXTRA.get(tb))
        if code not in _TOL:
            raise ValueError(f"color de tolerancia inválido: {tb}")
        tol = _TOL[code]
    out = {"ohm": value, "texto": fmt_ohms(value), "tolerancia_pct": tol,
           "min": value * (1 - tol / 100), "max": value * (1 + tol / 100)}
    if len(b) == 6:
        out["tcr_ppm_K"] = _TCR.get(_DIGIT.get(b[5], -9))
    return out


def value_to_colors(ohms: float, tol_pct: float = 5.0) -> dict:
    if ohms < 0.1 or ohms >= 1e11:
        raise ValueError("fuera del rango del código de colores (0,1 Ω a 99 GΩ)")
    tolname = next((("dorado" if c == -1 else "plateado" if c == -2 else _NAMES[c])
                    for c, t in _TOL.items() if t == tol_pct), "dorado")
    out = {}
    for nd in (2, 3):
        e = math.floor(math.log10(ohms)) - (nd - 1)
        sig = round(ohms / 10 ** e)
        if sig >= 10 ** nd:
            sig //= 10
            e += 1
        exact = abs(sig * 10.0 ** e - ohms) <= 1e-9 * ohms
        if e < -2 or e > 9:
            continue
        mname = {-1: "dorado", -2: "plateado"}.get(e) or _NAMES[e]
        digits = [_NAMES[int(c)] for c in str(sig).zfill(nd)]
        out[f"{nd + 2}_bandas"] = {"bandas": digits + [mname, tolname], "exacto": exact}
    return out


def nearest_series(ohms: float, series: str = "E24") -> dict:
    vals = SERIES[series.upper()]
    dec = 10 ** math.floor(math.log10(ohms))
    base = 100 if series.upper() == "E96" else 10
    cands = sorted({round(v * dec * f / base, 9) for v in vals for f in (0.1, 1, 10)})
    lower = max((c for c in cands if c <= ohms * (1 + 1e-12)), default=None)
    upper = min((c for c in cands if c >= ohms * (1 - 1e-12)), default=None)
    near = min((c for c in (lower, upper) if c), key=lambda c: abs(math.log(c / ohms)))
    return {"serie": series.upper(), "cercano": near, "inferior": lower, "superior": upper,
            "error_pct": (near - ohms) / ohms * 100}


_WATTS = [0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]


def led_resistor(v_supply: float, v_led: float, current_a: float, series: str = "E24", n_leds: int = 1) -> dict:
    drop = v_supply - v_led * n_leds
    if drop <= 0:
        raise ValueError("la fuente no alcanza para la tensión directa del LED")
    if current_a <= 0:
        raise ValueError("la corriente debe ser positiva")
    r = drop / current_a
    near = nearest_series(r, series)
    chosen = near["superior"] or near["cercano"]                # el de arriba: nunca más corriente
    i_real = drop / chosen
    p = i_real ** 2 * chosen
    return {"r_calculada": r, "r_comercial": chosen, "serie": series.upper(), "corriente_real_a": i_real,
            "potencia_w": p, "potencia_nominal_w": next((w for w in _WATTS if w >= 2 * p), _WATTS[-1])}


def _num(tok: str) -> float:
    from hope.tools.eng_tools import parse_value
    return parse_value(re.sub(r"(?i)(v|a)$", "", tok.strip()) or tok)


def resistor_text(arg: str) -> str:
    a = arg.strip()
    toks = a.replace(",", " ").split() if not re.search(r"\d,\d", a) else a.split()
    head = _fold(toks[0]) if toks else ""
    if head == "led":
        nums = [x for x in toks[1:] if not re.fullmatch(r"(?i)e\d+", x)]
        ser = next((x.upper() for x in toks if re.fullmatch(r"(?i)e(12|24|96)", x)), "E24")
        if len(nums) < 3:
            raise ValueError("formato: ⟦resistencias: led <V fuente> <V led> <corriente>⟧, p. ej. led 5 2 20mA")
        r = led_resistor(_num(nums[0]), _num(nums[1]), _num(nums[2]), ser, int(nums[3]) if len(nums) > 3 else 1)
        return (f"R = {fmt_ohms(r['r_calculada'])} → comercial {fmt_ohms(r['r_comercial'])} ({r['serie']}), "
                f"I = {r['corriente_real_a'] * 1000:.4g} mA, P = {r['potencia_w'] * 1000:.4g} mW "
                f"(usar ≥ {r['potencia_nominal_w']:g} W)")
    if re.fullmatch(r"e(12|24|96)", head):
        n = nearest_series(parse_ohms(toks[1]), head.upper())
        return (f"{n['serie']}: más cercano {fmt_ohms(n['cercano'])} ({n['error_pct']:+.3g} %); "
                f"inferior {fmt_ohms(n['inferior'])}, superior {fmt_ohms(n['superior'])}")
    if head in ("colores", "color", "bandas"):
        tol = float(toks[2].rstrip("%")) if len(toks) > 2 else 5.0
        c = value_to_colors(parse_ohms(toks[1]), tol)
        return "; ".join(f"{k.replace('_', ' ')}: {' '.join(v['bandas'])}" + ("" if v["exacto"] else " (redondeado)")
                         for k, v in c.items())
    if all(_fold(t) in _DIGIT or _fold(t) in _MULT_EXTRA for t in toks):
        r = colors_to_value(toks)
        s = f"{r['texto']} ±{r['tolerancia_pct']:g} % ({fmt_ohms(r['min'])} a {fmt_ohms(r['max'])})"
        return s + (f", {r['tcr_ppm_K']} ppm/K" if r.get("tcr_ppm_K") else "")
    raise ValueError("formato: colores («marrón negro rojo dorado»), «colores 4k7», «E24 4800» o «led 5 2 20mA»")


# ---------------------------------------------------------------------------
# G-code

KNOWN_G = {0, 1, 2, 3, 4, 10, 11, 17, 18, 19, 20, 21, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 53, 54, 60, 61,
           76, 80, 90, 91, 92}
KNOWN_M = {0, 1, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 42, 73, 75, 76, 77, 80, 81, 82, 83,
           84, 85, 92, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 121, 140,
           141, 145, 149, 150, 155, 190, 191, 200, 201, 203, 204, 205, 206, 207, 208, 209, 211, 217, 218, 220, 221,
           226, 240, 250, 251, 280, 290, 300, 301, 302, 303, 304, 400, 401, 402, 404, 405, 406, 407, 410, 412, 413,
           420, 421, 428, 500, 501, 502, 503, 504, 510, 511, 512, 524, 540, 569, 575, 593, 600, 603, 851, 862, 876,
           900, 906, 907, 911, 912, 913, 914, 915, 928, 993, 994, 995, 997, 999}
WORD = re.compile(r"([A-Za-z])\s*([-+]?(?:\d+\.?\d*|\.\d+))")


def _parse_bed(s: str | None) -> tuple[float, float, float]:
    if not s:
        return (220.0, 220.0, 250.0)
    v = [float(x) for x in re.split(r"[x×,\s]+", s.strip()) if x]
    return tuple((v + [250.0])[:3])  # type: ignore[return-value]


def _slicer_time(line: str) -> float | None:
    m = re.match(r";\s*TIME\s*:\s*(\d+)", line)                          # Cura
    if m:
        return float(m.group(1))
    m = re.match(r";\s*estimated printing time[^=]*=\s*(.+)", line)       # PrusaSlicer / Orca
    if m:
        t = 0.0
        for n, u in re.findall(r"(\d+)\s*([dhms])", m.group(1)):
            t += int(n) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[u]
        return t
    return None


def gcode_analyze(text: str, bed: str | None = None, diameter: float = 1.75, density: float = 1.24,
                  max_hotend: float = 300.0, max_bed: float = 120.0, max_lines: int = 5_000_000) -> dict:
    bx, by, bz = _parse_bed(bed)
    pos = {"X": 0.0, "Y": 0.0, "Z": 0.0, "E": 0.0}
    abs_xyz, abs_e, scale = True, True, 1.0
    feed, feed_seen = 1500.0, False
    t_move = t_dwell = 0.0
    extruded = retracted = 0.0
    n_retract = 0
    dist_print = dist_travel = 0.0
    hotend, bedt, warnings = [], [], []
    out_of_range, unknown, macros = [], {}, {}
    layer_z: set[float] = set()
    layer_comments = 0
    heated, cold_extrusion_line = False, None
    slicer_time = slicer_fil = None
    mins = {"X": math.inf, "Y": math.inf, "Z": math.inf}
    maxs = {"X": -math.inf, "Y": -math.inf, "Z": -math.inf}
    lines = text.splitlines()
    if len(lines) > max_lines:
        raise ValueError(f"demasiadas líneas ({len(lines)} > {max_lines})")
    for ln, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if stripped.startswith(";"):
            low = stripped.lower()
            if low.startswith((";layer:", ";layer_change", "; layer_change", ";layer change")):
                layer_comments += 1
            st = _slicer_time(stripped)
            if st is not None:
                slicer_time = st
            m = re.match(r";\s*filament used\s*(?:\[mm\])?\s*[:=]\s*([\d.]+)\s*(m|mm)?", stripped, re.I)
            if m:
                slicer_fil = float(m.group(1)) * (1000 if (m.group(2) or "mm") == "m" else 1)
        code = raw.split(";", 1)[0].strip()
        if not code:
            continue
        code = re.sub(r"^N\d+\s+", "", code)
        code = re.sub(r"\*\d+$", "", code).strip()
        m = re.match(r"([GMT])(\d+)(?:\.(\d+))?", code, re.I)
        if not m:
            name = code.split()[0].upper()
            if re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
                macros[name] = macros.get(name, 0) + 1
                if any(k in name for k in ("START", "PRINT_START", "HEAT")):
                    heated = True                                    # la macro probablemente calienta
            else:
                unknown[name] = unknown.get(name, 0) + 1
            continue
        letter, num = m.group(1).upper(), int(m.group(2))
        words = {k.upper(): float(v) for k, v in WORD.findall(code[m.end():])}
        cmd = f"{letter}{num}"
        if letter == "G" and num not in KNOWN_G or letter == "M" and num not in KNOWN_M:
            unknown[cmd] = unknown.get(cmd, 0) + 1
            continue
        if cmd in ("G0", "G1", "G2", "G3"):
            if "F" in words:
                feed, feed_seen = words["F"] * scale, True
            new = dict(pos)
            for ax in "XYZ":
                if ax in words:
                    new[ax] = words[ax] * scale + (pos[ax] if not abs_xyz else 0.0)
            if "E" in words:
                new["E"] = words["E"] * scale + (pos["E"] if not abs_e else 0.0)
            dx, dy, dz = new["X"] - pos["X"], new["Y"] - pos["Y"], new["Z"] - pos["Z"]
            de = new["E"] - pos["E"]
            if cmd in ("G2", "G3") and ("I" in words or "J" in words or "R" in words):
                if "R" in words:
                    r = abs(words["R"] * scale)
                    chord = math.hypot(dx, dy)
                    ang = 2 * math.asin(min(1.0, chord / (2 * r))) if r else 0.0
                else:
                    cx, cy = pos["X"] + words.get("I", 0) * scale, pos["Y"] + words.get("J", 0) * scale
                    r = math.hypot(pos["X"] - cx, pos["Y"] - cy)
                    a0 = math.atan2(pos["Y"] - cy, pos["X"] - cx)
                    a1 = math.atan2(new["Y"] - cy, new["X"] - cx)
                    ang = (a0 - a1) if cmd == "G2" else (a1 - a0)
                    ang %= 2 * math.pi
                    if ang == 0:
                        ang = 2 * math.pi                            # círculo completo
                dist = math.hypot(r * ang, dz)
            else:
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            move = dist if dist > 0 else abs(de)
            if move > 0 and feed > 0:
                t_move += move / (feed / 60.0)
            if de > 0:
                extruded += de
                if not heated and cold_extrusion_line is None:
                    cold_extrusion_line = ln
                if dist > 0:
                    dist_print += dist
                    if dx or dy:
                        layer_z.add(round(new["Z"], 4))
            elif de < 0:
                retracted += -de
                n_retract += 1
            if de <= 0 and dist > 0:
                dist_travel += dist
            for ax in "XYZ":
                if ax in words or dist > 0:
                    mins[ax] = min(mins[ax], new[ax])
                    maxs[ax] = max(maxs[ax], new[ax])
            lim = {"X": bx, "Y": by, "Z": bz}
            bad = [f"{ax}={new[ax]:g}" for ax in "XYZ" if ax in words and not (-0.5 <= new[ax] <= lim[ax] + 0.5)]
            if bad and len(out_of_range) < 50:
                out_of_range.append({"linea": ln, "fuera": bad})
            pos = new
        elif cmd == "G4":
            t_dwell += words.get("P", 0) / 1000.0 + words.get("S", 0)
        elif cmd == "G20":
            scale = 25.4
        elif cmd == "G21":
            scale = 1.0
        elif cmd == "G28":
            axes = [a for a in "XYZ" if a in code[m.end():].upper()] or list("XYZ")
            for a in axes:
                pos[a] = 0.0
        elif cmd == "G90":
            abs_xyz, abs_e = True, True
        elif cmd == "G91":
            abs_xyz, abs_e = False, False
        elif cmd == "M82":
            abs_e = True
        elif cmd == "M83":
            abs_e = False
        elif cmd == "G92":
            for ax in "XYZE":
                if ax in words:
                    pos[ax] = words[ax] * scale
        elif cmd in ("M104", "M109"):
            t = words.get("S", words.get("R"))
            if t is not None:
                hotend.append(t)
                heated = heated or t >= 150
                if t > max_hotend:
                    warnings.append(f"línea {ln}: hotend a {t:g} °C (> {max_hotend:g} °C)")
        elif cmd in ("M140", "M190"):
            t = words.get("S", words.get("R"))
            if t is not None:
                bedt.append(t)
                if t > max_bed:
                    warnings.append(f"línea {ln}: cama a {t:g} °C (> {max_bed:g} °C)")
    net = extruded - retracted
    area = math.pi * (diameter / 2) ** 2
    if not feed_seen and t_move:
        warnings.append("no hay F en los movimientos: se supuso 1500 mm/min")
    if cold_extrusion_line is not None:
        warnings.append(f"línea {cold_extrusion_line}: se extruye antes de calentar el hotend (≥150 °C)")
    if out_of_range:
        warnings.append(f"{len(out_of_range)} movimientos fuera de la cama {bx:g}×{by:g}×{bz:g} mm")
    if unknown:
        warnings.append("comandos desconocidos: " + ", ".join(f"{k}×{v}" for k, v in sorted(unknown.items())))
    return {
        "lineas": len(lines), "tiempo_s": round(t_move + t_dwell, 3),
        "tiempo_nota": "suma distancia/velocidad sin aceleración: cota inferior del tiempo real",
        "tiempo_slicer_s": slicer_time,
        "capas": len(layer_z), "capas_por_comentario": layer_comments,
        "altura_max_mm": max(layer_z) if layer_z else 0.0,
        "filamento_mm": round(net, 4), "extruido_mm": round(extruded, 4), "retraido_mm": round(retracted, 4),
        "retracciones": n_retract, "filamento_slicer_mm": slicer_fil,
        "peso_g": round(net * area / 1000.0 * density, 4), "diametro_mm": diameter, "densidad_g_cm3": density,
        "distancia_impresion_mm": round(dist_print, 3), "distancia_viaje_mm": round(dist_travel, 3),
        "temp_hotend": sorted(set(hotend)), "temp_cama": sorted(set(bedt)),
        "extremos": {ax: [mins[ax], maxs[ax]] for ax in "XYZ" if mins[ax] != math.inf},
        "cama_mm": [bx, by, bz], "fuera_de_rango": out_of_range, "desconocidos": unknown,
        "macros_klipper": macros, "avisos": warnings,
    }


def _hms(s: float) -> str:
    s = round(s)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return (f"{h} h " if h else "") + (f"{m} min " if h or m else "") + f"{sec} s"


def gcode_text(r: dict) -> str:
    s = (f"tiempo ≈ {_hms(r['tiempo_s'])} (sin aceleración)"
         + (f", slicer {_hms(r['tiempo_slicer_s'])}" if r["tiempo_slicer_s"] else "")
         + f"; {r['capas']} capas; filamento {r['filamento_mm'] / 1000:.4g} m ≈ {r['peso_g']:.4g} g"
         + f" (Ø{r['diametro_mm']:g} mm, {r['densidad_g_cm3']:g} g/cm³); {r['retracciones']} retracciones")
    if r["temp_hotend"] or r["temp_cama"]:
        s += f"; hotend {'/'.join(f'{t:g}' for t in r['temp_hotend']) or '—'} °C, cama " \
             f"{'/'.join(f'{t:g}' for t in r['temp_cama']) or '—'} °C"
    if r["macros_klipper"]:
        s += "; macros: " + ", ".join(r["macros_klipper"])
    if r["avisos"]:
        s += " · AVISOS: " + " | ".join(r["avisos"][:5])
    return s


# ---------------------------------------------------------------------------
# STL

def stl_read(data: bytes) -> tuple[np.ndarray, np.ndarray, str]:
    """(triángulos (n,3,3), normales guardadas (n,3), formato)."""
    if len(data) >= 84:
        n = int.from_bytes(data[80:84], "little")
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                                count=n, offset=84)
            return rec["v"].astype(np.float64), rec["n"].astype(np.float64), "binario"
    text = data.decode("ascii", errors="replace")
    if not text.lstrip().lower().startswith("solid"):
        raise ValueError("no es un STL válido (ni binario con tamaño coherente ni ASCII «solid …»)")
    num = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    verts = np.array(re.findall(rf"vertex\s+{num}\s+{num}\s+{num}", text), dtype=np.float64)
    norms = np.array(re.findall(rf"facet\s+normal\s+{num}\s+{num}\s+{num}", text), dtype=np.float64)
    if len(verts) == 0 or len(verts) % 3:
        raise ValueError(f"STL ASCII con {len(verts)} vértices (debe ser múltiplo de 3)")
    tris = verts.reshape(-1, 3, 3)
    if len(norms) != len(tris):
        norms = np.zeros((len(tris), 3))
    return tris, norms, "ascii"


def _components(n: int, pairs: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def stl_analyze(data: bytes, density: float | None = None, max_tris: int = 5_000_000) -> dict:
    tris, stored, fmt = stl_read(data)
    n = len(tris)
    if n == 0:
        raise ValueError("el STL no tiene triángulos")
    if n > max_tris:
        raise ValueError(f"demasiados triángulos ({n} > {max_tris})")
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    cross = np.cross(v1 - v0, v2 - v0)
    dbl = np.linalg.norm(cross, axis=1)
    area = float(dbl.sum() / 2)
    signed_vol = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6)
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    degenerate = int((dbl <= 1e-12 * diag * diag).sum())
    # Soldar vértices iguales (tolerancia relativa al tamaño de la pieza).
    q = np.round(tris.reshape(-1, 3) / (1e-7 * diag)).astype(np.int64)
    _, vid = np.unique(q, axis=0, return_inverse=True)
    vid = vid.reshape(-1, 3)
    directed = np.concatenate([vid[:, [0, 1]], vid[:, [1, 2]], vid[:, [2, 0]]])
    und = np.sort(directed, axis=1)
    _, inv, counts = np.unique(und, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    boundary = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    # Orientación: en una malla bien orientada cada arista dirigida aparece una sola vez.
    _, dcounts = np.unique(directed, axis=0, return_counts=True)
    inconsistent = int((dcounts > 1).sum())
    # Agujeros: componentes conexas de las aristas de borde.
    holes = 0
    if boundary:
        bedges = np.unique(und[np.isin(inv, np.nonzero(counts == 1)[0])], axis=0)
        verts_b, local = np.unique(bedges, return_inverse=True)
        comp = _components(len(verts_b), local.reshape(-1, 2))
        holes = len(set(comp.tolist()))
    # Cuerpos: triángulos conectados por aristas compartidas.
    tri_of_edge = np.tile(np.arange(n), 3)
    order = np.argsort(inv, kind="stable")
    se, st = inv[order], tri_of_edge[order]
    same = se[1:] == se[:-1]
    shells = len(set(_components(n, np.stack([st[:-1][same], st[1:][same]], 1)).tolist()))
    # Normales guardadas contra la orientación de los vértices.
    has_n = np.linalg.norm(stored, axis=1) > 0
    flipped = int(((np.einsum("ij,ij->i", stored, cross) < 0) & has_n).sum())
    closed = boundary == 0 and nonmanifold == 0
    out = {"formato": fmt, "triangulos": n, "dimensiones_mm": [float(x) for x in hi - lo],
           "min": [float(x) for x in lo], "max": [float(x) for x in hi],
           "area_mm2": area, "volumen_mm3": abs(signed_vol) if closed and not inconsistent else None,
           "volumen_con_signo_mm3": signed_vol, "cerrada": closed, "agujeros": holes,
           "aristas_borde": boundary, "aristas_no_variedad": nonmanifold,
           "aristas_orientacion_inconsistente": inconsistent, "normales_invertidas": flipped,
           "triangulos_degenerados": degenerate, "cuerpos": shells,
           "orientacion_global_invertida": bool(closed and signed_vol < 0)}
    if density and out["volumen_mm3"] is not None:
        out["peso_g"] = abs(signed_vol) / 1000.0 * density
    return out


def stl_text(r: dict) -> str:
    d = r["dimensiones_mm"]
    s = f"{r['triangulos']} triángulos ({r['formato']}); {d[0]:.4g} × {d[1]:.4g} × {d[2]:.4g} mm; " \
        f"área {r['area_mm2']:.6g} mm²; "
    s += (f"volumen {r['volumen_mm3']:.6g} mm³ ({r['volumen_mm3'] / 1000:.4g} cm³)" if r["volumen_mm3"] is not None
          else "malla ABIERTA (volumen no fiable)" if not r["cerrada"] else "volumen no fiable (orientación inconsistente)")
    if r.get("peso_g") is not None:
        s += f" ≈ {r['peso_g']:.4g} g"
    probs = []
    if r["agujeros"]:
        probs.append(f"{r['agujeros']} agujeros ({r['aristas_borde']} aristas de borde)")
    if r["aristas_no_variedad"]:
        probs.append(f"{r['aristas_no_variedad']} aristas no variedad")
    if r["aristas_orientacion_inconsistente"]:
        probs.append(f"{r['aristas_orientacion_inconsistente']} aristas con orientación inconsistente")
    if r["normales_invertidas"]:
        probs.append(f"{r['normales_invertidas']} normales guardadas invertidas")
    if r["orientacion_global_invertida"]:
        probs.append("toda la malla está al revés (volumen negativo)")
    if r["triangulos_degenerados"]:
        probs.append(f"{r['triangulos_degenerados']} triángulos degenerados")
    if r["cuerpos"] > 1:
        probs.append(f"{r['cuerpos']} cuerpos separados")
    return s + ("; cerrada y bien orientada" if not probs and r["cerrada"] else "; problemas: " + ", ".join(probs))


def stl_analyze_file(path: str, density: float | None = None) -> dict:
    return stl_analyze(Path(path).read_bytes(), density)


# ---------------------------------------------------------------------------
# Marlin

THERMISTORS = {0: "sin sensor", 1: "100k EPCOS (el más común)", 5: "100k ATC Semitec 104GT-2 / 104NT-4",
               11: "100k beta 3950 1 %", 13: "100k Hisens 3950", 147: "PT100 con resistencia de 4k7",
               1047: "PT1000 con resistencia de 4k7", 998: "simulado (pruebas)", 999: "simulado (pruebas)",
               -5: "PT100/PT1000 con MAX31865", -2: "termocupla con MAX6675", -3: "termocupla con MAX31855"}
_DEFINE = re.compile(r"^\s*#define\s+([A-Za-z_]\w*)(?:\s+(.*?))?\s*(?://.*)?$")


def marlin_parse(text: str) -> dict:
    """Valores de los ``#define`` activos (los comentados con // no cuentan)."""
    out = {}
    for line in text.splitlines():
        m = _DEFINE.match(line)
        if m:
            v = (m.group(2) or "").strip()
            out[m.group(1)] = v if v else True
    return out


def _arr(v) -> list[float]:
    if not isinstance(v, str):
        return []
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v)]


def _num1(v) -> float | None:
    a = _arr(v)
    return a[0] if len(a) == 1 else None


def marlin_check(text: str, adv_text: str = "") -> dict:
    d = marlin_parse(text)
    d_adv = marlin_parse(adv_text)
    allv = {**d_adv, **d}
    key = {}
    for k in ("MOTHERBOARD", "SERIAL_PORT", "BAUDRATE", "EXTRUDERS", "TEMP_SENSOR_0", "TEMP_SENSOR_BED",
              "HEATER_0_MAXTEMP", "BED_MAXTEMP", "EXTRUDE_MINTEMP", "X_BED_SIZE", "Y_BED_SIZE", "Z_MAX_POS",
              "X_MIN_POS", "Y_MIN_POS", "X_MAX_POS", "Y_MAX_POS", "DEFAULT_AXIS_STEPS_PER_UNIT",
              "DEFAULT_MAX_FEEDRATE", "DEFAULT_MAX_ACCELERATION", "DEFAULT_ACCELERATION", "PREHEAT_1_TEMP_HOTEND",
              "PREHEAT_1_TEMP_BED", "PREHEAT_2_TEMP_HOTEND", "X_DRIVER_TYPE", "E0_DRIVER_TYPE",
              "NOZZLE_TO_PROBE_OFFSET", "BLTOUCH", "FIX_MOUNTED_PROBE", "AUTO_BED_LEVELING_BILINEAR",
              "AUTO_BED_LEVELING_UBL", "MESH_BED_LEVELING", "LIN_ADVANCE", "THERMAL_PROTECTION_HOTENDS",
              "THERMAL_PROTECTION_BED", "PREVENT_COLD_EXTRUSION"):
        if k in allv:
            key[k] = allv[k]
    issues = []
    ext = _num1(allv.get("EXTRUDERS", "1"))
    ext = 1 if ext is None else ext
    t0 = _num1(allv.get("TEMP_SENSOR_0"))
    if ext >= 1 and t0 == 0:
        issues.append("TEMP_SENSOR_0 = 0 con extrusor: el hotend no tiene termistor configurado")
    for name, val in (("TEMP_SENSOR_0", t0), ("TEMP_SENSOR_BED", _num1(allv.get("TEMP_SENSOR_BED")))):
        if val is not None:
            key[name + "_descripcion"] = THERMISTORS.get(int(val), "tipo no reconocido por esta tabla (ver thermistors.h)")
            if int(val) in (998, 999):
                issues.append(f"{name} = {int(val)}: sensor SIMULADO, no lee temperatura real")
    if "THERMAL_PROTECTION_HOTENDS" not in allv and "THERMAL_PROTECTION_HOTEND" not in allv:
        issues.append("THERMAL_PROTECTION_HOTENDS desactivado: PELIGRO de incendio ante un termistor suelto")
    bed_sensor = _num1(allv.get("TEMP_SENSOR_BED"))
    if bed_sensor and "THERMAL_PROTECTION_BED" not in allv:
        issues.append("cama caliente sin THERMAL_PROTECTION_BED: PELIGRO de incendio")
    if "PREVENT_COLD_EXTRUSION" not in allv:
        issues.append("PREVENT_COLD_EXTRUSION desactivado: se puede extruir en frío y dañar el extrusor")
    maxt, overshoot = _num1(allv.get("HEATER_0_MAXTEMP")), _num1(allv.get("HOTEND_OVERSHOOT")) or 15
    for k in ("PREHEAT_1_TEMP_HOTEND", "PREHEAT_2_TEMP_HOTEND"):
        p = _num1(allv.get(k))
        if p is not None and maxt is not None and p > maxt - overshoot:
            issues.append(f"{k} = {p:g} supera HEATER_0_MAXTEMP − HOTEND_OVERSHOOT = {maxt - overshoot:g}")
    bmax = _num1(allv.get("BED_MAXTEMP"))
    pb = _num1(allv.get("PREHEAT_1_TEMP_BED"))
    if pb is not None and bmax is not None and pb > bmax - 10:
        issues.append(f"PREHEAT_1_TEMP_BED = {pb:g} demasiado cerca de BED_MAXTEMP = {bmax:g}")
    mint = _num1(allv.get("EXTRUDE_MINTEMP"))
    if mint is not None and mint < 150:
        issues.append(f"EXTRUDE_MINTEMP = {mint:g} °C es muy bajo (lo usual es 170)")
    steps = _arr(allv.get("DEFAULT_AXIS_STEPS_PER_UNIT"))
    if steps:
        if len(steps) < 3 + ext:
            issues.append(f"DEFAULT_AXIS_STEPS_PER_UNIT tiene {len(steps)} valores; se esperan X, Y, Z y {int(ext)} E")
        for ax, s, lo, hi in zip("XYZ", steps[:3], (20, 20, 50), (1000, 1000, 6400)):
            if not lo <= s <= hi:
                issues.append(f"pasos/mm de {ax} = {s:g} fuera de lo habitual ({lo}–{hi})")
        if len(steps) > 3 and not 50 <= steps[3] <= 2000:
            issues.append(f"pasos/mm del extrusor = {steps[3]:g} fuera de lo habitual (50–2000)")
    feeds = _arr(allv.get("DEFAULT_MAX_FEEDRATE"))
    if len(feeds) >= 3 and feeds[2] > 60:
        issues.append(f"velocidad máxima de Z = {feeds[2]:g} mm/s es alta para un husillo (lo usual ≤ 20)")
    if len(feeds) >= 4 and feeds[3] > 150:
        issues.append(f"velocidad máxima del extrusor = {feeds[3]:g} mm/s es inusualmente alta")
    acc = _arr(allv.get("DEFAULT_MAX_ACCELERATION"))
    if len(acc) >= 2 and max(acc[:2]) > 20000:
        issues.append(f"aceleración máxima XY = {max(acc[:2]):g} mm/s² inusualmente alta")
    for ax, size_k, max_k, min_k in (("X", "X_BED_SIZE", "X_MAX_POS", "X_MIN_POS"),
                                     ("Y", "Y_BED_SIZE", "Y_MAX_POS", "Y_MIN_POS")):
        size, mx = _num1(allv.get(size_k)), _num1(allv.get(max_k))
        mn = _num1(allv.get(min_k)) or 0
        if size is not None and mx is not None and mx - mn < size:
            issues.append(f"{max_k} − {min_k} = {mx - mn:g} es menor que {size_k} = {size:g}: no alcanza toda la cama")
    probe = any(k in allv for k in ("BLTOUCH", "FIX_MOUNTED_PROBE", "Z_PROBE_SERVO_NR", "PROBE_MANUALLY",
                                     "NOZZLE_AS_PROBE", "TOUCH_MI_PROBE", "BIQU_MICROPROBE_V1", "Z_PROBE_SLED"))
    abl = any(k in allv for k in ("AUTO_BED_LEVELING_BILINEAR", "AUTO_BED_LEVELING_UBL",
                                  "AUTO_BED_LEVELING_LINEAR", "AUTO_BED_LEVELING_3POINT"))
    if abl and not probe:
        issues.append("nivelación automática activada sin sonda (BLTOUCH, FIX_MOUNTED_PROBE, …)")
    if probe and not abl and "MESH_BED_LEVELING" not in allv:
        issues.append("hay sonda pero ninguna nivelación de cama activada")
    if "BLTOUCH" in allv and "NOZZLE_TO_PROBE_OFFSET" not in allv:
        issues.append("BLTOUCH sin NOZZLE_TO_PROBE_OFFSET")
    if "Z_MIN_PROBE_USES_Z_MIN_ENDSTOP_PIN" in allv and "USE_PROBE_FOR_Z_HOMING" in allv and not probe:
        issues.append("se usa la sonda para Z sin sonda definida")
    return {"valores": key, "incoherencias": issues, "defines": len(d) + len(d_adv)}


def marlin_check_files(path: str) -> dict:
    p = Path(path)
    if p.is_dir():
        main = p / "Configuration.h"
        adv = p / "Configuration_adv.h"
        if not main.exists():
            found = sorted(p.rglob("Configuration.h"))
            if not found:
                raise FileNotFoundError("no se encontró Configuration.h en la carpeta")
            main, adv = found[0], found[0].with_name("Configuration_adv.h")
    else:
        main, adv = p, p.with_name("Configuration_adv.h")
    for f in (main, adv):
        if f.exists() and f.stat().st_size > (8 << 20):
            raise ValueError(f"{f.name} es demasiado grande")
    r = marlin_check(main.read_text(encoding="utf-8", errors="replace"),
                     adv.read_text(encoding="utf-8", errors="replace") if adv.exists() else "")
    r["archivos"] = [str(main)] + ([str(adv)] if adv.exists() else [])
    return r


def marlin_text(r: dict) -> str:
    v = r["valores"]
    parts = []
    for k, lab in (("MOTHERBOARD", "placa"), ("DEFAULT_AXIS_STEPS_PER_UNIT", "pasos/mm"),
                   ("DEFAULT_MAX_FEEDRATE", "vel. máx"), ("DEFAULT_MAX_ACCELERATION", "acel. máx"),
                   ("TEMP_SENSOR_0", "termistor hotend"), ("TEMP_SENSOR_BED", "termistor cama"),
                   ("HEATER_0_MAXTEMP", "máx hotend"), ("BED_MAXTEMP", "máx cama"),
                   ("X_BED_SIZE", "cama X"), ("Y_BED_SIZE", "cama Y"), ("Z_MAX_POS", "Z máx")):
        if k in v:
            extra = f" ({v[k + '_descripcion']})" if k + "_descripcion" in v else ""
            parts.append(f"{lab} {v[k]}{extra}")
    s = "; ".join(parts) or "sin valores clave"
    return s + (" · INCOHERENCIAS: " + " | ".join(r["incoherencias"]) if r["incoherencias"]
                else " · sin incoherencias detectadas")


# ---------------------------------------------------------------------------
# Materiales

MATERIAL_SOURCE = ("rangos típicos de las fichas técnicas de fabricantes (Prusament, Polymaker, eSun) y de los "
                   "perfiles genéricos de PrusaSlicer/Cura; la ficha del carrete manda")
MATERIALS = {
    # nombre: boquilla °C, cama °C, velocidad mm/s, ventilador %, densidad g/cm³, cerramiento, notas
    "PLA": ((190, 220), (50, 65), (40, 100), (100, 100), 1.24, False, "fácil; se deforma por encima de ~55 °C"),
    "PLA+": ((200, 230), (50, 65), (40, 100), (100, 100), 1.24, False, "PLA modificado, algo más tenaz"),
    "PETG": ((220, 250), (70, 90), (30, 80), (30, 60), 1.27, False, "se pega mucho al PEI liso: usar separador"),
    "ABS": ((230, 260), (90, 110), (30, 80), (0, 30), 1.04, True, "emite estireno: ventilar; se contrae y despega"),
    "ASA": ((235, 260), (90, 110), (30, 80), (0, 30), 1.07, True, "como ABS pero resistente a UV"),
    "TPU": ((210, 240), (30, 60), (15, 35), (50, 100), 1.21, False, "flexible (95A); extrusor directo y lento"),
    "PA": ((240, 280), (70, 100), (30, 60), (0, 30), 1.14, True, "nailon: higroscópico, secar antes (70 °C, 8 h)"),
    "PC": ((260, 300), (100, 120), (30, 60), (0, 30), 1.20, True, "muy tenaz; boquilla de alta temperatura"),
    "HIPS": ((220, 250), (90, 110), (30, 60), (0, 30), 1.04, True, "soporte soluble en limoneno para ABS"),
    "PVA": ((185, 210), (45, 60), (20, 40), (100, 100), 1.23, False, "soporte soluble en agua; muy higroscópico"),
    "PLA-CF": ((200, 230), (50, 65), (30, 80), (100, 100), 1.29, False, "abrasivo: boquilla endurecida"),
    "PETG-CF": ((230, 260), (70, 90), (30, 70), (30, 60), 1.30, False, "abrasivo: boquilla endurecida"),
}
_ALIASES = {"NYLON": "PA", "NAILON": "PA", "POLICARBONATO": "PC", "FLEX": "TPU", "FLEXIBLE": "TPU",
            "PLA PLUS": "PLA+", "PLAPLUS": "PLA+", "PET-G": "PETG", "PA6": "PA", "PA12": "PA", "CF-PLA": "PLA-CF"}


def _material_row(name, t, src) -> dict:
    noz, bed, spd, fan, dens, encl, notes = t
    return {"material": name, "boquilla_c": list(noz), "cama_c": list(bed), "velocidad_mm_s": list(spd),
            "ventilador_pct": list(fan), "densidad_g_cm3": dens, "cerramiento": encl, "notas": notes, "fuente": src}


def _profile_rows(path: str) -> list[dict]:
    """Perfiles de filamento del usuario: PrusaSlicer/SuperSlicer (.ini) u OrcaSlicer/Bambu (.json)."""
    p = Path(path)
    files = sorted([*p.rglob("*.ini"), *p.rglob("*.json")])[:500] if p.is_dir() else [p]
    rows = []
    for f in files:
        if f.stat().st_size > (2 << 20):
            continue
        txt = f.read_text(encoding="utf-8", errors="replace")
        kv: dict = {}
        if f.suffix.lower() == ".json":
            try:
                kv = json.loads(txt)
            except json.JSONDecodeError:
                continue
            if not isinstance(kv, dict):
                continue
        else:
            for line in txt.splitlines():
                if "=" in line and not line.lstrip().startswith(("#", ";", "[")):
                    k, _, v = line.partition("=")
                    kv[k.strip()] = v.strip()

        def first(*keys):
            for k in keys:
                v = kv.get(k)
                if isinstance(v, list):
                    v = v[0] if v else None
                if v not in (None, "", "nil"):
                    try:
                        return float(str(v).split(",")[0].split(";")[0])
                    except ValueError:
                        return str(v)
            return None

        ftype = first("filament_type")
        noz = first("temperature", "nozzle_temperature")
        if ftype is None and noz is None:
            continue
        rows.append({"material": str(ftype or "?").upper(), "perfil": kv.get("name") or f.stem,
                     "boquilla_c": noz, "primera_capa_c": first("first_layer_temperature",
                                                                 "nozzle_temperature_initial_layer"),
                     "cama_c": first("bed_temperature", "hot_plate_temp", "textured_plate_temp", "eng_plate_temp"),
                     "densidad_g_cm3": first("filament_density"),
                     "velocidad_max_volumetrica_mm3_s": first("filament_max_volumetric_speed"),
                     "ventilador_pct": first("max_fan_speed", "fan_max_speed"),
                     "fuente": f"perfil del usuario: {f}"})
    return rows


def material(query: str, profiles: str | None = None) -> list[dict]:
    q = query.strip().upper()
    if q in ("", "LISTA", "TODOS", "TODAS"):
        names = list(MATERIALS)
    else:
        q = q.replace("COMPARAR", "").strip()
        names = []
        for tok in (t for t in re.split(r"[\s,;]+", q) if t not in ("", "VS", "VS.", "Y", "CON", "O")):
            tok = _ALIASES.get(tok, tok)
            if tok in MATERIALS and tok not in names:
                names.append(tok)
        if not names and q in _ALIASES:
            names = [_ALIASES[q]]
    rows = [_material_row(n, MATERIALS[n], MATERIAL_SOURCE) for n in names]
    if profiles:
        user = _profile_rows(profiles)
        if names:
            user = [u for u in user if any(u["material"].startswith(n) for n in names)]
        rows = user + rows                          # lo del usuario va primero: es su máquina
    return rows


def material_text(rows: list[dict]) -> str:
    if not rows:
        return "material no encontrado; disponibles: " + ", ".join(MATERIALS)
    out = []
    for r in rows[:6]:
        if "perfil" in r:
            out.append(f"{r['material']} [{r['perfil']}]: boquilla {r['boquilla_c']} °C, cama {r['cama_c']} °C"
                       + (f", densidad {r['densidad_g_cm3']}" if r["densidad_g_cm3"] else "") + f" ({r['fuente']})")
            continue
        out.append(f"{r['material']}: boquilla {r['boquilla_c'][0]}–{r['boquilla_c'][1]} °C, cama "
                   f"{r['cama_c'][0]}–{r['cama_c'][1]} °C, {r['velocidad_mm_s'][0]}–{r['velocidad_mm_s'][1]} mm/s, "
                   f"ventilador {r['ventilador_pct'][0]}–{r['ventilador_pct'][1]} %, {r['densidad_g_cm3']} g/cm³"
                   + (", con cerramiento" if r["cerramiento"] else "") + f"; {r['notas']}")
    return " | ".join(out) + (f" · fuente: {MATERIAL_SOURCE}" if any("perfil" not in r for r in rows[:6]) else "")
