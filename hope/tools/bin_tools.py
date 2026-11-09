"""Herramientas para binarios y análisis de BIOS (docs/HERRAMIENTAS_HN1.md §3, catálogo ampliado).

* ``hexdump`` — volcado hexadecimal + ASCII de un rango.
* ``find_pattern`` — buscar secuencias de bytes (con comodines ``??``) o cadenas ASCII/UTF-16LE.
* ``entropy`` — entropía de Shannon por bloques; agrupa zonas vacías, de código/datos y
  comprimidas/cifradas (útil para ubicar volúmenes y módulos dentro de una imagen de BIOS).
* ``hashes`` — MD5, SHA-1, SHA-256 y CRC32.
* ``disassemble`` — Capstone (x86 16/32/64, ARM, Thumb, ARM64) si está instalado; corre en el sandbox.

Todo opera sobre bytes ya leídos a través del guardián (``hope/tools/files.py``); acá no se
abren archivos salvo en ``disassemble_file``, que ejecuta el sandbox con una ruta ya autorizada.
"""

from __future__ import annotations

import hashlib
import math
import re
import zlib

import numpy as np

MAX_DUMP = 4096
MAX_HITS = 200


def parse_int(s: str | int | None, default: int = 0) -> int:
    if s is None or (isinstance(s, str) and not s.strip()):
        return default
    if isinstance(s, int):
        return s
    t = s.strip().replace("_", "")
    if re.fullmatch(r"[0-9A-Fa-f]+[hH]", t):
        return int(t[:-1], 16)
    return int(t, 0)


def hexdump(data: bytes, base: int = 0, width: int = 16) -> str:
    lines = []
    for i in range(0, len(data), width):
        chunk = data[i:i + width]
        hx = " ".join(f"{b:02X}" for b in chunk)
        asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{base + i:08X}  {hx:<{width * 3 - 1}}  |{asc}|")
    return "\n".join(lines)


def pattern_regex(pattern: str) -> tuple[re.Pattern, str]:
    """``55 AA``, ``E9 ?? ?? 00``, ``"$IBIOSI$"`` (ASCII), ``u"BIOS"`` (UTF-16LE) o ``re:…`` (regex de bytes)."""
    p = pattern.strip()
    if p.startswith(("u\"", "u'")) and len(p) >= 3 and p[-1] == p[1]:
        lit = p[2:-1].encode("utf-16-le")
        return re.compile(re.escape(lit), re.DOTALL), f"UTF-16LE «{p[2:-1]}»"
    if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'":
        return re.compile(re.escape(p[1:-1].encode("latin-1")), re.DOTALL), f"ASCII «{p[1:-1]}»"
    toks = p.replace(",", " ").split()
    if toks and all(re.fullmatch(r"[0-9A-Fa-f]{2}|\?\?|\?", t) for t in toks) or re.fullmatch(r"(?:[0-9A-Fa-f]{2})+", p):
        if len(toks) == 1 and len(p) > 2:
            toks = [p[i:i + 2] for i in range(0, len(p), 2)]
        rx = b"".join(b"." if t.startswith("?") else re.escape(bytes([int(t, 16)])) for t in toks)
        if not rx.strip(b"."):
            raise ValueError("el patrón no puede ser solo comodines")
        return re.compile(rx, re.DOTALL), "bytes " + " ".join(t.upper() for t in toks)
    return re.compile(re.escape(p.encode("latin-1")), re.DOTALL), f"ASCII «{p}»"


def find_pattern(data: bytes, pattern: str, base: int = 0, max_hits: int = MAX_HITS) -> dict:
    rx, desc = pattern_regex(pattern)
    hits, total, pos = [], 0, 0
    while True:                                # también coincidencias superpuestas
        m = rx.search(data, pos)
        if not m:
            break
        total += 1
        if len(hits) < max_hits:
            hits.append(base + m.start())
        pos = m.start() + 1
    return {"patron": desc, "coincidencias": total, "offsets": hits}


def entropy(data: bytes, block: int = 4096, high: float = 7.5, low: float = 1.0) -> dict:
    if block < 16:
        raise ValueError("bloque mínimo de 16 bytes")
    arr = np.frombuffer(data, dtype=np.uint8)
    n = math.ceil(len(arr) / block)
    values = []
    for i in range(n):
        b = arr[i * block:(i + 1) * block]
        c = np.bincount(b, minlength=256).astype(np.float64)
        p = c[c > 0] / len(b)
        values.append(float(-(p * np.log2(p)).sum()) if len(b) else 0.0)

    def kind(i: int, e: float) -> str:
        b = arr[i * block:(i + 1) * block]
        if e < low:
            fill = int(np.bincount(b, minlength=256).argmax())
            return f"vacío/relleno (0x{fill:02X})"
        return "comprimido/cifrado" if e >= high else "código/datos"

    regions = []
    for i, e in enumerate(values):
        k = kind(i, e)
        if regions and regions[-1]["tipo"] == k:
            r = regions[-1]
            r["fin"] = min(len(arr), (i + 1) * block)
            r["_e"].append(e)
        else:
            regions.append({"inicio": i * block, "fin": min(len(arr), (i + 1) * block), "tipo": k, "_e": [e]})
    for r in regions:
        es = r.pop("_e")
        r["entropia_media"] = round(sum(es) / len(es), 4)
    total = np.bincount(arr, minlength=256).astype(np.float64)
    p = total[total > 0] / max(1, len(arr))
    return {"tamano": len(arr), "bloque": block, "entropia_global": float(-(p * np.log2(p)).sum()) if len(arr) else 0.0,
            "por_bloque": [round(v, 4) for v in values], "zonas": regions}


def entropy_text(r: dict, max_regions: int = 12) -> str:
    z = "; ".join(f"0x{x['inicio']:X}–0x{x['fin']:X} {x['tipo']} ({x['entropia_media']:.2f})"
                  for x in r["zonas"][:max_regions])
    more = f"; … {len(r['zonas']) - max_regions} zonas más" if len(r["zonas"]) > max_regions else ""
    return f"{r['tamano']} bytes, entropía global {r['entropia_global']:.3f} bits/byte; bloques de {r['bloque']}: {z}{more}"


def hashes(data: bytes, algos: str = "") -> dict:
    want = [a.strip().lower().replace("-", "") for a in algos.replace(",", " ").split()] or [
        "md5", "sha1", "sha256", "crc32"]
    out = {}
    for a in want:
        if a == "crc32":
            out["crc32"] = f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"
        elif a in ("md5", "sha1", "sha256", "sha512", "sha224", "sha384"):
            out[a] = hashlib.new(a, data).hexdigest()
        else:
            raise ValueError(f"algoritmo desconocido: {a} (md5, sha1, sha256, sha512, crc32)")
    return out


ARCHS = {"x86-16": ("CS_ARCH_X86", "CS_MODE_16"), "x86": ("CS_ARCH_X86", "CS_MODE_32"),
         "x86-32": ("CS_ARCH_X86", "CS_MODE_32"), "x86-64": ("CS_ARCH_X86", "CS_MODE_64"),
         "x64": ("CS_ARCH_X86", "CS_MODE_64"), "arm": ("CS_ARCH_ARM", "CS_MODE_ARM"),
         "thumb": ("CS_ARCH_ARM", "CS_MODE_THUMB"), "arm64": ("CS_ARCH_ARM64", "CS_MODE_ARM"),
         "aarch64": ("CS_ARCH_ARM64", "CS_MODE_ARM")}


def capstone_available() -> bool:
    try:
        import capstone  # noqa: F401
        return True
    except ImportError:
        return False


def disassemble(code_hex: str, arch: str = "x86-64", address: int = 0, max_insns: int = 64) -> dict:
    """Se ejecuta en el sandbox. ``code_hex``: bytes en hexadecimal."""
    a = arch.strip().lower()
    if a not in ARCHS:
        raise ValueError(f"arquitectura: {', '.join(ARCHS)}")
    try:
        import capstone
    except ImportError:
        return {"error": "Capstone no está instalado: pip install capstone (extra «herramientas» de pyproject)"}
    arch_c, mode_c = ARCHS[a]
    arch_v = getattr(capstone, arch_c, None) if arch_c != "CS_ARCH_ARM64" else getattr(
        capstone, "CS_ARCH_ARM64", getattr(capstone, "CS_ARCH_AARCH64", None))
    md = capstone.Cs(arch_v, getattr(capstone, mode_c))
    data = bytes.fromhex(code_hex)
    out, used = [], 0
    for ins in md.disasm(data, address):
        out.append({"dir": ins.address, "bytes": ins.bytes.hex(), "ins": f"{ins.mnemonic} {ins.op_str}".strip()})
        used = ins.address - address + ins.size
        if len(out) >= max_insns:
            break
    return {"arquitectura": a, "instrucciones": out, "bytes_usados": used, "bytes_totales": len(data)}


def disasm_text(r: dict) -> str:
    if "error" in r:
        return r["error"]
    s = "\n".join(f"{i['dir']:08X}: {i['ins']}" for i in r["instrucciones"]) or "ninguna instrucción válida"
    if r["bytes_usados"] < r["bytes_totales"] and len(r["instrucciones"]) < 64:
        s += f"\n(se detuvo en el byte {r['bytes_usados']}: instrucción inválida para {r['arquitectura']})"
    return s
