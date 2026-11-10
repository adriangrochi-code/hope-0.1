"""Herramientas de imagen (docs/HERRAMIENTAS_HN1.md §3, catálogo ampliado).

* ``image_info`` — tamaño, modo, formato, EXIF principal e histograma por canal.
* ``crop`` — recorte y zoom de una región, para leer detalles con la zona visual
  (``hope/data/vision.py``): el resultado es un PNG nuevo que Hope puede mirar.

Se ejecutan en el sandbox desde el catálogo (una imagen puede ser una «bomba de
descompresión»): Pillow con límite de píxeles. Sin Pillow, ``image_info`` lee al menos el
tamaño de PNG, GIF, BMP y JPEG desde la cabecera.
"""

from __future__ import annotations

import base64
import io
import struct
from pathlib import Path

MAX_PIXELS = 80_000_000
MAX_OUT_SIDE = 4096


def _header_size(data: bytes) -> tuple[str, int, int] | None:
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return "PNG", w, h
    if data[:6] in (b"GIF87a", b"GIF89a"):
        w, h = struct.unpack("<HH", data[6:10])
        return "GIF", w, h
    if data[:2] == b"BM" and len(data) >= 26:
        w, h = struct.unpack("<ii", data[18:26])
        return "BMP", w, abs(h)
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            seglen = struct.unpack(">H", data[i + 2:i + 4])[0]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return "JPEG", w, h
            i += 2 + seglen
    return None


def _open(path: str):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    img = Image.open(path)
    if img.width * img.height > MAX_PIXELS:
        raise ValueError(f"imagen demasiado grande ({img.width}×{img.height})")
    return img


def image_info(path: str, bins: int = 8) -> dict:
    data = Path(path).read_bytes()
    try:
        from PIL import ExifTags
    except ImportError:
        hs = _header_size(data)
        if hs is None:
            raise ValueError("formato no reconocido sin Pillow (pip install Pillow)") from None
        return {"formato": hs[0], "ancho": hs[1], "alto": hs[2], "bytes": len(data),
                "nota": "sin Pillow: solo tamaño (pip install Pillow para EXIF e histograma)"}
    img = _open(path)
    out = {"formato": img.format, "ancho": img.width, "alto": img.height, "modo": img.mode, "bytes": len(data),
           "cuadros": getattr(img, "n_frames", 1)}
    exif = {}
    try:
        raw = img.getexif()
        for k, v in raw.items():
            name = ExifTags.TAGS.get(k, str(k))
            if name in ("Make", "Model", "DateTime", "Orientation", "Software", "ExifOffset", "XResolution",
                        "YResolution", "Artist", "Copyright", "GPSInfo", "ImageDescription"):
                exif[name] = v if isinstance(v, (int, float, str)) else str(v)[:80]
        try:                                                   # sub-IFD de EXIF: cámara y toma
            sub = raw.get_ifd(0x8769)
            for k, v in sub.items():
                name = ExifTags.TAGS.get(k, str(k))
                if name in ("DateTimeOriginal", "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength",
                            "LensModel", "FocalLengthIn35mmFilm", "ExifImageWidth", "ExifImageHeight"):
                    exif[name] = float(v) if hasattr(v, "numerator") else (v if isinstance(v, (int, str)) else str(v)[:80])
        except Exception:                                      # noqa: BLE001 — EXIF incompleto
            pass
    except Exception:                                          # noqa: BLE001
        pass
    exif.pop("ExifOffset", None)
    out["exif"] = exif
    rgb = img.convert("RGB") if img.mode not in ("L", "RGB") else img
    hist = rgb.histogram()
    chans = "L" if rgb.mode == "L" else "RGB"
    stats = {}
    step = 256 // bins
    for i, ch in enumerate(chans):
        h = hist[256 * i:256 * (i + 1)]
        n = sum(h) or 1
        mean = sum(k * c for k, c in enumerate(h)) / n
        stats[ch] = {"media": round(mean, 3), "min": next(k for k, c in enumerate(h) if c) if any(h) else 0,
                     "max": max(k for k, c in enumerate(h) if c) if any(h) else 0,
                     "bins": [sum(h[j:j + step]) for j in range(0, 256, step)]}
    out["histograma"] = stats
    return out


def image_info_text(r: dict) -> str:
    s = f"{r['formato']} {r['ancho']}×{r['alto']}" + (f" {r['modo']}" if "modo" in r else "") + f", {r['bytes']} bytes"
    if r.get("cuadros", 1) > 1:
        s += f", {r['cuadros']} cuadros"
    if r.get("exif"):
        s += "; EXIF: " + ", ".join(f"{k}={v}" for k, v in list(r["exif"].items())[:8])
    if r.get("histograma"):
        s += "; medias " + ", ".join(f"{k} {v['media']:.1f}" for k, v in r["histograma"].items())
        s += "; rango " + ", ".join(f"{k} {v['min']}–{v['max']}" for k, v in r["histograma"].items())
    return s + (f" ({r['nota']})" if "nota" in r else "")


def crop(path: str, box: list[float], zoom: float = 1.0) -> dict:
    """``box`` = [x, y, ancho, alto] en píxeles, o en fracciones (0–1) si todos son ≤ 1."""
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("recortar necesita Pillow (pip install Pillow)") from None
    img = _open(path)
    x, y, w, h = box
    if all(0 <= v <= 1 for v in box) and any(0 < v < 1 for v in box):
        x, y, w, h = x * img.width, y * img.height, w * img.width, h * img.height
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(img.width, int(round(x + w))), min(img.height, int(round(y + h)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"región vacía o fuera de la imagen ({img.width}×{img.height})")
    if not 0.1 <= zoom <= 16:
        raise ValueError("zoom entre 0,1 y 16")
    out = img.crop((x0, y0, x1, y1))
    nw, nh = max(1, round((x1 - x0) * zoom)), max(1, round((y1 - y0) * zoom))
    if max(nw, nh) > MAX_OUT_SIDE:
        raise ValueError(f"el resultado superaría {MAX_OUT_SIDE} px de lado")
    if zoom != 1:
        out = out.resize((nw, nh), Image.LANCZOS if zoom < 1 else Image.NEAREST if zoom >= 4 else Image.BICUBIC)
    if out.mode not in ("RGB", "RGBA", "L"):
        out = out.convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return {"png_b64": base64.b64encode(buf.getvalue()).decode(), "region": [x0, y0, x1 - x0, y1 - y0],
            "tamano": [out.width, out.height], "original": [img.width, img.height]}
