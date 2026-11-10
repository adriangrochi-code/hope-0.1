"""Dónde vive Hope: elegir un disco que NO sea el del sistema operativo y poner todo ahí.

Pedido del usuario: «lo más automático posible» y «que no toque el SSD del sistema operativo».
Solo biblioteca estándar: se usa antes de instalar nada (``Hope.bat`` / ``hope.sh`` / ``Hope.exe``).

* ``detectar()`` — discos con su espacio libre y tipo (HDD, SSD, externo). En Windows, letras de unidad
  (``GetDriveTypeW`` + ``Get-PhysicalDisk`` si está); en Linux, puntos de montaje de ``/proc/mounts`` con
  ``/sys/block/*/queue/rotational``. Marca el disco del sistema (``%SystemDrive%`` o el de ``/``).
* ``proponer(discos)`` — el disco con más espacio libre fuera del sistema, prefiriendo HDD o externo.
  Si solo está el del sistema, lo propone con un aviso claro.
* ``preparar(carpeta, repo)`` — crea ``<carpeta>/{venv,runs,data,cache,tmp,modelos}`` y enlaza
  ``.venv``, ``runs`` y ``data`` del proyecto a esas carpetas (unión de directorios en Windows, enlace
  simbólico en Linux): todo lo que el código escribe con rutas relativas termina en el disco elegido.
  Si ya había datos en el proyecto, se mueven.
* ``entorno(carpeta)`` — variables para que las cachés y temporales también vayan ahí: ``PIP_CACHE_DIR``,
  ``HF_HOME``, ``TORCH_HOME``, ``XDG_CACHE_HOME``, ``TMP``/``TEMP``/``TMPDIR`` y ``HOPE_OFFLOAD_DIR``
  (estado del optimizador en disco si no alcanza la RAM).
* La elección queda en ``hope_ubicacion.json`` en la carpeta del proyecto (un archivo chico).

Uso desde la consola (lo llaman los lanzadores)::

    python -m hope.app.discos --elegir          # propone, pide confirmar (Enter) o escribir otra carpeta
    python -m hope.app.discos --elegir --si     # acepta la propuesta sin preguntar
    python -m hope.app.discos --entorno         # imprime CLAVE=VALOR para los lanzadores
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WINDOWS = os.name == "nt"
ARCHIVO = "hope_ubicacion.json"
SUBCARPETAS = ("venv", "runs", "data", "cache", "tmp", "modelos")
ENLACES = {".venv": "venv", "runs": "runs", "data": "data"}
FS_REALES = {"ext4", "ext3", "ext2", "xfs", "btrfs", "ntfs", "ntfs3", "exfat", "vfat", "fuseblk", "f2fs", "zfs",
             "jfs", "reiserfs", "hfsplus", "apfs"}
GB = 1e9


def repo_por_defecto() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Detección
# ---------------------------------------------------------------------------
def _uso(ruta: str) -> tuple[float, float]:
    u = shutil.disk_usage(ruta)
    return round(u.free / GB, 1), round(u.total / GB, 1)


def _base_dispositivo(dev: str) -> str:
    """/dev/sda2 → sda · /dev/nvme0n1p3 → nvme0n1 · /dev/mmcblk0p1 → mmcblk0."""
    name = Path(dev).name
    if name.startswith(("nvme", "mmcblk")) and "p" in name:
        return name.rsplit("p", 1)[0]
    return name.rstrip("0123456789") or name


def _linux() -> list[dict]:
    try:
        mounts = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return []
    raiz_dev = None
    filas = []
    for line in mounts:
        parts = line.split()
        if len(parts) < 3:
            continue
        dev, mnt, fs = parts[0], parts[1].replace("\\040", " "), parts[2]
        if mnt == "/":
            raiz_dev = _base_dispositivo(dev) if dev.startswith("/dev/") else dev
        if fs not in FS_REALES or not dev.startswith("/dev/"):
            continue
        if mnt.startswith(("/boot", "/snap", "/var/snap", "/efi")):
            continue
        filas.append((dev, mnt))
    vistos, out = set(), []
    for dev, mnt in filas:
        base = _base_dispositivo(dev)
        if (dev, mnt) in vistos:
            continue
        vistos.add((dev, mnt))
        try:
            libre, total = _uso(mnt)
        except OSError:
            continue
        sysb = Path("/sys/block") / base
        rot = (sysb / "queue" / "rotational").read_text().strip() if (sysb / "queue" / "rotational").exists() else ""
        rem = (sysb / "removable").read_text().strip() if (sysb / "removable").exists() else ""
        externo = rem == "1" or mnt.startswith(("/media/", "/run/media/"))
        tipo = "externo" if externo else ("HDD" if rot == "1" else "SSD" if rot == "0" else "?")
        out.append({"ruta": mnt, "dispositivo": base, "libre_gb": libre, "total_gb": total, "tipo": tipo,
                    "sistema": base == raiz_dev or mnt == "/"})
    # Un mismo disco montado en varios lugares: queda el punto de montaje más corto.
    por_dev: dict[str, dict] = {}
    for d in out:
        k = d["dispositivo"] + str(d["total_gb"])
        if k not in por_dev or len(d["ruta"]) < len(por_dev[k]["ruta"]):
            por_dev[k] = d
    return list(por_dev.values())


def _windows() -> list[dict]:
    import ctypes
    import string
    k32 = ctypes.windll.kernel32
    sistema = (os.environ.get("SystemDrive") or "C:").upper().rstrip("\\")
    tipos_ps = _tipos_windows()
    out = []
    mask = k32.GetLogicalDrives()
    for i, letra in enumerate(string.ascii_uppercase):
        if not mask >> i & 1:
            continue
        raiz = f"{letra}:\\"
        t = k32.GetDriveTypeW(ctypes.c_wchar_p(raiz))
        if t not in (2, 3):                       # 2 extraíble, 3 fijo (no red, CD ni RAM)
            continue
        try:
            libre, total = _uso(raiz)
        except OSError:
            continue
        tipo = "externo" if t == 2 else tipos_ps.get(letra, "?")
        out.append({"ruta": raiz, "dispositivo": f"{letra}:", "libre_gb": libre, "total_gb": total, "tipo": tipo,
                    "sistema": f"{letra}:" == sistema})
    return out


def _tipos_windows() -> dict[str, str]:
    """Letra → HDD / SSD / externo según Get-PhysicalDisk (si PowerShell responde)."""
    ps = ("Get-Partition | Where-Object DriveLetter | ForEach-Object { $d = Get-Disk -Number $_.DiskNumber; "
          "$p = Get-PhysicalDisk | Where-Object DeviceId -eq $_.DiskNumber; "
          "[pscustomobject]@{L=[string]$_.DriveLetter; M=[string]$p.MediaType; B=[string]$d.BusType} } | ConvertTo-Json")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=30,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        data = json.loads(r.stdout or "[]")
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return {}
    if isinstance(data, dict):
        data = [data]
    out = {}
    for d in data:
        bus, media = str(d.get("B", "")).upper(), str(d.get("M", "")).upper()
        out[str(d.get("L", "")).upper()] = ("externo" if bus in ("USB", "1394", "SD") else
                                           "HDD" if media == "HDD" else "SSD" if media == "SSD" else "?")
    return out


def detectar() -> list[dict]:
    return _windows() if WINDOWS else _linux()


def proponer(discos: list[dict], minimo_gb: float = 30.0) -> dict:
    """Elige el disco. Devuelve {"carpeta", "disco", "aviso", "alternativas"}."""
    otros = [d for d in discos if not d["sistema"] and d["libre_gb"] >= minimo_gb]
    lentos = [d for d in otros if d["tipo"] in ("HDD", "externo") and d["libre_gb"] >= 100]
    aviso = ""
    if lentos:
        elegido = max(lentos, key=lambda d: d["libre_gb"])
    elif otros:
        elegido = max(otros, key=lambda d: d["libre_gb"])
    else:
        sistema = [d for d in discos if d["sistema"]]
        elegido = max(sistema or discos, key=lambda d: d["libre_gb"]) if (sistema or discos) else None
        aviso = ("Solo encontré el disco del sistema operativo (o los otros tienen menos de "
                 f"{minimo_gb:g} GB libres). Hope lo va a usar si confirmás; los datos y checkpoints pesan decenas "
                 "de GB y escriben mucho.")
    if elegido is None:
        return {"carpeta": None, "disco": None, "aviso": "no encontré ningún disco", "alternativas": []}
    base = Path(elegido["ruta"])
    if elegido["sistema"] and not WINDOWS:
        base = Path.home()                       # en Linux, la raíz no es escribible: la carpeta personal
    return {"carpeta": str(base / "Hope"), "disco": elegido, "aviso": aviso,
            "alternativas": sorted(discos, key=lambda d: (d["sistema"], -d["libre_gb"]))}


def disco_de(ruta: str | Path, discos: list[dict]) -> dict | None:
    """El disco (de la lista) que contiene esa ruta: el punto de montaje más largo que la prefija."""
    p = str(Path(ruta).resolve() if Path(ruta).exists() else Path(ruta).absolute())
    norm = (lambda s: s.lower().replace("/", "\\")) if WINDOWS else (lambda s: s)
    mejores = [d for d in discos if norm(p).startswith(norm(d["ruta"]).rstrip("\\/") + ("\\" if WINDOWS else "/"))
               or norm(p) == norm(d["ruta"]).rstrip("\\/") or d["ruta"] == "/"]
    return max(mejores, key=lambda d: len(d["ruta"])) if mejores else None


# ---------------------------------------------------------------------------
# Preparar la carpeta
# ---------------------------------------------------------------------------
def leer(repo: Path | None = None) -> dict | None:
    p = (repo or repo_por_defecto()) / ARCHIVO
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None


def _es_enlace(p: Path) -> bool:
    if p.is_symlink():
        return True
    if WINDOWS and p.exists():
        try:
            return bool(os.lstat(p).st_file_attributes & 0x400)     # FILE_ATTRIBUTE_REPARSE_POINT (unión)
        except (OSError, AttributeError):
            return False
    return False


def _destino_enlace(p: Path) -> Path | None:
    try:
        return Path(os.path.realpath(p))
    except OSError:
        return None


def _enlazar(link: Path, target: Path) -> str:
    """Hace que ``link`` apunte a ``target``. Si ``link`` era una carpeta con contenido, lo mueve."""
    target.mkdir(parents=True, exist_ok=True)
    if _es_enlace(link):
        if _destino_enlace(link) == target.resolve():
            return "ya estaba"
        if WINDOWS:
            os.rmdir(link)                                # quita la unión, no el contenido
        else:
            link.unlink()
    elif link.exists():
        if link.resolve() == target.resolve():
            return "es la misma carpeta"
        for item in link.iterdir():                       # mover lo que ya había
            dest = target / item.name
            if dest.exists():
                dest = target / f"{item.name}.anterior_{time.strftime('%Y%m%d-%H%M%S')}"
            shutil.move(str(item), str(dest))
        link.rmdir()
    if WINDOWS:
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
        if r.returncode != 0:
            raise OSError(f"no pude crear la unión {link} → {target}: {r.stdout} {r.stderr}")
    else:
        os.symlink(target, link, target_is_directory=True)
    return "enlazado"


def preparar(carpeta: str | Path, repo: Path | None = None, disco: dict | None = None) -> dict:
    repo = (repo or repo_por_defecto()).resolve()
    carpeta = Path(carpeta).expanduser()
    if not carpeta.is_absolute():
        raise ValueError("la carpeta tiene que ser una ruta completa (p. ej. D:\\Hope)")
    carpeta.mkdir(parents=True, exist_ok=True)
    prueba = carpeta / ".hope_prueba_escritura"
    prueba.write_text("ok")
    prueba.unlink()
    for sub in SUBCARPETAS:
        (carpeta / sub).mkdir(exist_ok=True)
    for sub in ("pip", "huggingface", "torch", "xdg"):
        (carpeta / "cache" / sub).mkdir(parents=True, exist_ok=True)
    acciones = {}
    if carpeta.resolve() != repo:
        for link, sub in ENLACES.items():
            real = repo / link
            if link == ".venv" and real.exists() and not _es_enlace(real) and any(real.iterdir()):
                # Un entorno virtual no se puede mover (tiene rutas absolutas adentro): queda donde está.
                acciones[link] = ("queda en la carpeta del proyecto: para pasarlo al disco elegido, borrá .venv "
                                  "y volvé a abrir Hope (se reinstala ahí)")
                continue
            acciones[link] = _enlazar(real, carpeta / sub)
    info = {"carpeta": str(carpeta), "fecha": time.strftime("%Y-%m-%d %H:%M"), "disco": disco, "enlaces": acciones}
    (repo / ARCHIVO).write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def entorno(carpeta: str | Path) -> dict[str, str]:
    c = Path(carpeta)
    return {"HOPE_DISCO": str(c), "PIP_CACHE_DIR": str(c / "cache" / "pip"), "HF_HOME": str(c / "cache" / "huggingface"),
            "TORCH_HOME": str(c / "cache" / "torch"), "XDG_CACHE_HOME": str(c / "cache" / "xdg"),
            "TMP": str(c / "tmp"), "TEMP": str(c / "tmp"), "TMPDIR": str(c / "tmp"),
            "HOPE_OFFLOAD_DIR": str(c / "runs" / "optimizador_en_disco"), "HOPE_MODELOS": str(c / "modelos")}


def aplicar_entorno(repo: Path | None = None) -> dict | None:
    """Si hay ubicación elegida, pone las variables en este proceso (y sus hijos)."""
    info = leer(repo)
    if not info or not info.get("carpeta"):
        return None
    env = entorno(info["carpeta"])
    for k, v in env.items():
        if k in ("TMP", "PIP_CACHE_DIR", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "HOPE_MODELOS"):
            Path(v).mkdir(parents=True, exist_ok=True)
        os.environ[k] = v
    import tempfile
    tempfile.tempdir = env["TMP"]
    return info


def resumen(repo: Path | None = None) -> dict:
    """Para la interfaz: ubicación elegida, propuesta y si algo pesado quedó en el disco del sistema."""
    repo = (repo or repo_por_defecto()).resolve()
    discos = detectar()
    info = leer(repo)
    out = {"elegida": info, "discos": discos, "propuesta": None, "en_sistema": []}
    if not info:
        out["propuesta"] = proponer(discos)
        return out
    for link in ENLACES:
        p = repo / link
        if p.exists():
            d = disco_de(os.path.realpath(p), discos)
            if d and d["sistema"] and not (disco_de(info["carpeta"], discos) or {}).get("sistema"):
                out["en_sistema"].append(link)
    return out


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Elegir el disco de Hope (fuera del disco del sistema)")
    ap.add_argument("--elegir", action="store_true")
    ap.add_argument("--si", action="store_true", help="aceptar la propuesta sin preguntar")
    ap.add_argument("--carpeta", default=None, help="usar esta carpeta (p. ej. D:\\Hope)")
    ap.add_argument("--entorno", action="store_true", help="imprimir CLAVE=VALOR de las cachés")
    ap.add_argument("--repo", default=None)
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve() if args.repo else repo_por_defecto()
    if args.entorno:
        info = leer(repo)
        if info:
            for k, v in entorno(info["carpeta"]).items():
                print(f"{k}={v}")
        return 0
    if args.elegir:
        info = leer(repo)
        if info and not args.carpeta:
            print(f"Hope usa {info['carpeta']}")
            preparar(info["carpeta"], repo, info.get("disco"))
            return 0
        discos = detectar()
        if args.carpeta:
            carpeta, disco = args.carpeta, disco_de(args.carpeta, discos)
        else:
            p = proponer(discos)
            if not p["carpeta"]:
                print("No encontré discos. Escribí la carpeta a mano con --carpeta.")
                return 1
            d = p["disco"]
            print(f"\nVoy a usar {p['carpeta']} ({d['libre_gb']:.0f} GB libres, {d['tipo']}).")
            if p["aviso"]:
                print("AVISO: " + p["aviso"])
            print("Ahí van los datos, los checkpoints, el entorno de Python y las cachés (nada pesado en el disco del "
                  "sistema).")
            carpeta, disco = p["carpeta"], d
            if not args.si:
                for o in p["alternativas"]:
                    print(f"   {o['ruta']:<14} {o['libre_gb']:>8.0f} GB libres  {o['tipo']:<8}"
                          f"{'  (sistema)' if o['sistema'] else ''}")
                r = input("Enter para usarla, o escribí otra carpeta (p. ej. E:\\Hope): ").strip().strip('"')
                if r:
                    carpeta, disco = r, disco_de(r, discos)
        try:
            info = preparar(carpeta, repo, disco)
        except (OSError, ValueError) as e:
            print(f"No pude preparar {carpeta}: {e}")
            return 1
        print(f"Listo: Hope usa {info['carpeta']}")
        return 0
    print(json.dumps(resumen(repo), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
