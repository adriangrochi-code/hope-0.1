"""Ejecución aislada de código (Herramientas §9; Carta art. 5).

El código que llega del maestro o que genera Hope es **no confiable**. Se
ejecuta en un proceso separado con:

* intérprete en modo aislado (``python -I``: sin variables de entorno ni sitio del usuario);
* directorio temporal propio como carpeta de trabajo, que se borra al terminar;
* entorno vacío salvo ``PATH``;
* límite de tiempo; en Linux/macOS también de memoria, CPU, tamaño de archivos y procesos.

Límite honesto: esto **no bloquea la red** ni es un contenedor. Para eso hace
falta aislamiento del sistema operativo (contenedor, ``unshare``, usuario sin
privilegios), que se agrega en el despliegue del harness (F9).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class RunResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool = False


def _limits(mem_mb: int, cpu_s: int):
    def apply():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            resource.setrlimit(resource.RLIMIT_FSIZE, (10 << 20, 10 << 20))
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except Exception:
            pass
    return apply if os.name == "posix" else None


def run_python(code: str, *, timeout_s: float = 10.0, mem_mb: int = 1024, stdin: str = "") -> RunResult:
    with tempfile.TemporaryDirectory(prefix="hope_sandbox_") as tmp:
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp, input=stdin,
                                  capture_output=True, text=True, timeout=timeout_s,
                                  env={"PATH": os.environ.get("PATH", "")},
                                  preexec_fn=_limits(mem_mb, int(timeout_s) + 1))
        except subprocess.TimeoutExpired as e:
            return RunResult(False, e.stdout or "", e.stderr or "", None, timed_out=True)
    return RunResult(proc.returncode == 0, proc.stdout, proc.stderr, proc.returncode)


IGNORE_IN_COPY = (".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache")


def run_command(argv: list[str], *, files: dict[str, str] | None = None, timeout_s: float = 20.0,
                mem_mb: int = 1024, stdin: str = "", copy_tree: str | None = None,
                max_copy_mb: float = 200.0) -> RunResult:
    """Ejecuta un programa en una carpeta temporal con ``files`` escritos adentro, con los mismos límites.

    ``argv`` puede referirse a los archivos por su nombre relativo. Con ``copy_tree`` primero se
    copia esa carpeta (sin ``.git``, entornos virtuales ni cachés) y después se escriben ``files``
    encima: el programa trabaja sobre una **copia** y la carpeta original no se modifica (harness F9).
    """
    with tempfile.TemporaryDirectory(prefix="hope_sandbox_") as tmp:
        if copy_tree is not None:
            import shutil
            total = 0
            for dirpath, dirnames, filenames in os.walk(copy_tree):
                dirnames[:] = [d for d in dirnames if d not in IGNORE_IN_COPY]
                total += sum(os.path.getsize(os.path.join(dirpath, f)) for f in filenames
                             if os.path.isfile(os.path.join(dirpath, f)))
            if total > max_copy_mb * (1 << 20):
                return RunResult(False, "", f"la carpeta ocupa {total / (1 << 20):.0f} MB (> {max_copy_mb:.0f} MB)", None)
            shutil.copytree(copy_tree, tmp, dirs_exist_ok=True, symlinks=True,
                            ignore=shutil.ignore_patterns(*IGNORE_IN_COPY))
        for name, content in (files or {}).items():
            path = os.path.join(tmp, name)
            if os.path.dirname(name):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        try:
            proc = subprocess.run(argv, cwd=tmp, input=stdin, capture_output=True, text=True, timeout=timeout_s,
                                  env={"PATH": os.environ.get("PATH", ""), "HOME": tmp},
                                  preexec_fn=_limits(mem_mb, int(timeout_s) + 1))
        except subprocess.TimeoutExpired as e:
            return RunResult(False, e.stdout or "", e.stderr or "", None, timed_out=True)
        except FileNotFoundError:
            return RunResult(False, "", f"programa no disponible: {argv[0]}", None)
    return RunResult(proc.returncode == 0, proc.stdout, proc.stderr, proc.returncode)
