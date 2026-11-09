"""Datos del sistema sin depender de torch: CPU, RAM, disco, GPU (``nvidia-smi``), procesos."""

from __future__ import annotations

import ctypes
import functools
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

WINDOWS = os.name == "nt"


def ram_gb() -> float:
    if WINDOWS:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        try:
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.ullTotalPhys / 1e9, 1)
        except Exception:
            return 0.0
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        return 0.0


def cpu_nombre() -> str:
    try:
        if WINDOWS:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def disco(path: str | Path) -> dict:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        u = shutil.disk_usage(p)
        return {"ruta": str(p), "libre_gb": round(u.free / 1e9, 1), "total_gb": round(u.total / 1e9, 1)}
    except OSError:
        return {"ruta": str(p), "libre_gb": None, "total_gb": None}


_GPU_CACHE: dict = {"t": 0.0, "v": None}


def nvidia(cache_s: float = 5.0) -> list[dict] | None:
    """GPUs NVIDIA según ``nvidia-smi`` (None si no está). Memoria en MB, temperatura en °C."""
    if time.time() - _GPU_CACHE["t"] < cache_s:
        return _GPU_CACHE["v"]
    exe = shutil.which("nvidia-smi")
    out = None
    if exe:
        try:
            r = subprocess.run([exe, "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu,"
                                     "driver_version", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                out = []
                for line in r.stdout.strip().splitlines():
                    c = [x.strip() for x in line.split(",")]
                    if len(c) >= 6:
                        def num(x):
                            try:
                                return float(x)
                            except ValueError:
                                return None
                        out.append({"nombre": c[0], "memoria_usada_mb": num(c[1]), "memoria_total_mb": num(c[2]),
                                    "temperatura_c": num(c[3]), "uso_pct": num(c[4]), "driver": c[5]})
        except (OSError, subprocess.TimeoutExpired):
            out = None
    _GPU_CACHE.update(t=time.time(), v=out)
    return out


@functools.lru_cache(maxsize=1)
def gpus_por_nombre() -> tuple[str, ...]:
    """Nombres de las placas de video que ve el sistema operativo (para detectar una Arc sin torch)."""
    names: list[str] = []
    try:
        if WINDOWS:
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                               capture_output=True, text=True, timeout=20)
            names = [x.strip() for x in r.stdout.splitlines() if x.strip()]
        elif shutil.which("lspci"):
            r = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10)
            names = [x.split(":", 2)[-1].strip() for x in r.stdout.splitlines()
                     if "VGA" in x or "3D controller" in x or "Display" in x]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return tuple(names)


def proceso_vivo(pid: int | None) -> bool:
    if not pid:
        return False
    if WINDOWS:
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259               # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    try:                                                     # un zombi (hijo terminado sin recoger) no cuenta
        st = Path(f"/proc/{int(pid)}/stat").read_text()
        return st.rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def terminar_proceso(pid: int) -> None:
    try:
        if WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30)
        else:
            import signal
            os.kill(int(pid), signal.SIGTERM)
    except (OSError, subprocess.TimeoutExpired):
        pass


def mantener_despierta(activo: bool) -> None:
    """En Windows, evita que la PC se suspenda mientras entrena (no apaga la pantalla)."""
    if not WINDOWS:
        return
    try:
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if activo else 0))
    except Exception:
        pass


def resumen(carpeta_datos: str | Path) -> dict:
    return {"sistema": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "cpu": cpu_nombre(), "nucleos": os.cpu_count(), "ram_gb": ram_gb(), "disco": disco(carpeta_datos),
            "nvidia": nvidia(), "placas": list(gpus_por_nombre())}
