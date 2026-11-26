#!/bin/sh
# ==================================================================================================
# compilar.sh — builds the parallel port pulse engine (tisma_lpt.c).
# ==================================================================================================
# Outputs (in the folder given as $1, default: this folder):
#   tisma_lpt32.exe   Windows XP .. 11, 32-bit (PE subsystem 5.01 so XP accepts it).
#   tisma_lpt64.exe   Windows XP x64 .. 11, 64-bit (subsystem 5.02, XP x64 / Server 2003).
#   tisma_lpt_sim     the simulator build for this OS (virtual time, simulated port): the tests.
# Needs gcc, and mingw-w64 for the Windows builds (Debian/Ubuntu: apt install gcc-mingw-w64);
# a missing cross compiler skips that build with a notice.
# Static runtime (-static): nothing to install next to the exe except InpOut's DLL.
# --no-insert-timestamp: the same source gives the same bytes (reproducible builds).
# Usage: sh hope/cnc/lpt/compilar.sh [output folder]
set -e
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-$here}
mkdir -p "$out"
gcc -std=c99 -O2 -Wall -Wextra -DTISMA_SIM -o "$out/tisma_lpt_sim" "$here/tisma_lpt.c" -lm
for spec in "i686 32 5.01 1" "x86_64 64 5.02 2"; do
    set -- $spec
    cc=$1-w64-mingw32-gcc
    if command -v "$cc" >/dev/null 2>&1; then
        "$cc" -O2 -Wall -Wextra -o "$out/tisma_lpt$2.exe" "$here/tisma_lpt.c" -lwinmm -static \
            -Wl,--subsystem,console:$3 -Wl,--major-os-version,5 -Wl,--minor-os-version,$4 -Wl,--no-insert-timestamp
    else
        echo "aviso: falta $cc (apt install gcc-mingw-w64): no se compila tisma_lpt$2.exe"
    fi
done
