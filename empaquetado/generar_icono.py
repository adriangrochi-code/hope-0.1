"""Genera ``hope.ico`` (y ``hope.png``) sin dependencias: cuadrado azul redondeado con una «H» blanca.

    python empaquetado/generar_icono.py
"""

import struct
import zlib
from pathlib import Path


def png(size: int) -> bytes:
    azul, blanco = (47, 111, 221, 255), (255, 255, 255, 255)
    r = size * 0.2                                   # radio de las esquinas
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            cx = min(max(x + 0.5, r), size - r)
            cy = min(max(y + 0.5, r), size - r)
            dentro = (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= r * r
            px = (0, 0, 0, 0)
            if dentro:
                px = azul
                u, v = x / size, y / size
                if 0.22 <= v <= 0.78 and (0.26 <= u <= 0.38 or 0.62 <= u <= 0.74 or (0.38 < u < 0.62 and 0.45 <= v <= 0.56)):
                    px = blanco
            row += bytes(px)
        rows.append(bytes(row))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


def ico(sizes=(256, 64, 48, 32, 16)) -> bytes:
    imgs = [png(s) for s in sizes]
    out = struct.pack("<HHH", 0, 1, len(imgs))
    off = 6 + 16 * len(imgs)
    for s, data in zip(sizes, imgs):
        out += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), off)
        off += len(data)
    return out + b"".join(imgs)


if __name__ == "__main__":
    d = Path(__file__).parent
    (d / "hope.ico").write_bytes(ico())
    (d / "hope.png").write_bytes(png(256))
    print("hope.ico y hope.png listos")
