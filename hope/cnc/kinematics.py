"""Rotary-axis kinematics: where a point of the workpiece ends up when the table axes turn.

What it is
----------
The math shared by the toolpath view and the tilted work plane (G68.2):

* ``table_rotations``: which rotary axes turn the WORKPIECE (mounted on the table) and around which
  centre, in work (G54) coordinates.
* ``to_part`` (machine → workpiece) and ``to_machine`` (workpiece → machine): rotate a point by the
  table angles around those centres.
* ``align_angles``: the table angles that make a direction of the workpiece (the normal of a tilted
  plane) point straight up, at the tool — what G53.1 does.
* ``euler_zxz``: the rotation of a G68.2 plane from its I J K angles.

How it fits
-----------
Used by ``gcode.py`` (G68.2 / G53.1: programmed points on a tilted plane → machine coordinates) and
by ``session.py`` (toolpath view: machine positions → workpiece, so 4th-axis wrapping and 3+2 work
are drawn on the part). It uses ``machine.py`` (axes, mounting, centre of rotation).

Conventions
-----------
* Points are ``(x, y, z)`` tuples in work coordinates (machine minus the G54 origin), mm.
* Angles are WORK angles in degrees (machine angle minus the G54 value of that axis).
* RS274/NGC right-hand rule: A turns about X (Y towards Z), B about Y (Z towards X), C about Z
  (X towards Y): ``ROTARY_PLANE``.
* Machine order of the axes is A, B, C, and a table axis listed later rides on the earlier ones
  (on a trunnion A+C table the C platter sits on the A cradle). The chain is therefore
  machine = A(B(C(part))) and part = C⁻¹(B⁻¹(A⁻¹(machine))).
* Head-mounted axes (``montaje: cabezal``) are not part of the chain: they tilt the tool.

Example
-------
::

    rots = table_rotations(machine, machine.offset("G54"))
    p_machine = to_machine((0.0, -20.0, -20.0), {"A": -90.0, "C": 0.0}, rots)
    p_part = to_part(p_machine, {"A": -90.0, "C": 0.0}, rots)       # back to (0, -20, -20)
"""

from __future__ import annotations

import math

from hope.cnc.machine import ROTARY, ROTARY_PLANE, Machine

AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


# ==================================================================================================
# ===================================== TABLE ROTARY AXES ==========================================
# ==================================================================================================

def table_rotations(m: Machine, g54: dict) -> list:
    """Rotary axes that turn the WORKPIECE (mounted on the table), as ``[[axis, cu, cv], …]``: the
    centre of rotation in work coordinates (G54, mm) on the two linear axes of the axis's plane
    (``ROTARY_PLANE``). Head-mounted axes are left out: they tilt the tool, the part does not move.
    Without a configured centre the axis passes through the G54 zero (``[axis, 0, 0]``)."""
    out = []
    for a in m.axis_names:
        ax = m.axes[a]
        if a not in ROTARY or ax.mount == "cabezal":
            continue
        u, v = ROTARY_PLANE[a]
        cu, cv = (ax.pivot[0] - g54.get(u, 0.0), ax.pivot[1] - g54.get(v, 0.0)) if ax.pivot else (0.0, 0.0)
        out.append([a, round(cu, 4), round(cv, 4)])
    return out


def _turn(p: list, axis: str, cu: float, cv: float, degrees: float) -> None:
    """Rotates point ``p`` (list [x, y, z], changed in place) by ``degrees`` about ``axis`` through
    the centre (cu, cv) of its plane. u' = cu + (u−cu)·cos θ − (v−cv)·sin θ, v' = cv + (u−cu)·sin θ +
    (v−cv)·cos θ, which is the right-hand rule for A, B and C with the ``ROTARY_PLANE`` order."""
    u, v = ROTARY_PLANE[axis]
    iu, iv = AXIS_INDEX[u], AXIS_INDEX[v]
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    du, dv = p[iu] - cu, p[iv] - cv
    p[iu], p[iv] = cu + du * c - dv * s, cv + du * s + dv * c


def to_part(point, angles: dict, rots: list) -> tuple:
    """Machine work point → workpiece point: undoes the table rotations, A first (part = C⁻¹B⁻¹A⁻¹·m).
    ``angles``: work angles in degrees by axis (missing = 0)."""
    p = list(point)
    for a, cu, cv in rots:
        _turn(p, a, cu, cv, -angles.get(a, 0.0))
    return (p[0], p[1], p[2])


def to_machine(point, angles: dict, rots: list) -> tuple:
    """Workpiece point → machine work point at the given table angles: m = A(B(C(part))), so the
    LAST axis in machine order is applied first."""
    p = list(point)
    for a, cu, cv in reversed(rots):
        _turn(p, a, cu, cv, angles.get(a, 0.0))
    return (p[0], p[1], p[2])


def _direction(vec, angles: dict, rots: list) -> tuple:
    """A direction (not a point) carried to the machine: rotations without the centres."""
    return to_machine(vec, angles, [[a, 0.0, 0.0] for a, _, _ in rots])


# ==================================================================================================
# ===================================== TILTED WORK PLANE ==========================================
# ==================================================================================================

def euler_zxz(i_deg: float, j_deg: float, k_deg: float) -> tuple:
    """Rotation matrix (3 rows) of a G68.2 plane from its Euler angles I, J, K (degrees), Fanuc's
    «rotate I about Z, then J about the new X, then K about the new Z» (R = Rz(I)·Rx(J)·Rz(K)).
    Column 0 is the plane's X direction, column 1 its Y and column 2 its normal (the tool side),
    all expressed in work coordinates."""
    a, b, c = (math.radians(x) for x in (i_deg, j_deg, k_deg))

    def rz(t):
        return ((math.cos(t), -math.sin(t), 0.0), (math.sin(t), math.cos(t), 0.0), (0.0, 0.0, 1.0))

    def rx(t):
        return ((1.0, 0.0, 0.0), (0.0, math.cos(t), -math.sin(t)), (0.0, math.sin(t), math.cos(t)))

    def mul(p, q):
        return tuple(tuple(sum(p[r][k] * q[k][col] for k in range(3)) for col in range(3)) for r in range(3))
    return mul(mul(rz(a), rx(b)), rz(c))


def plane_to_work(rot: tuple, origin: tuple, p) -> tuple:
    """Point of the tilted plane (its own X Y Z) → work coordinates: origin + R·p."""
    return tuple(origin[r] + sum(rot[r][k] * p[k] for k in range(3)) for r in range(3))


def work_to_plane(rot: tuple, origin: tuple, w) -> tuple:
    """Work coordinates → point of the tilted plane: Rᵀ·(w − origin) (R is a rotation: Rᵀ = R⁻¹)."""
    d = [w[r] - origin[r] for r in range(3)]
    return tuple(sum(rot[k][col] * d[k] for k in range(3)) for col in range(3))


def align_angles(normal, m: Machine, rots: list, g54: dict) -> dict:
    """Table angles (work degrees) that make ``normal`` (a workpiece direction, the tilted plane's
    normal) point at the tool, +Z of the machine — what G53.1 does.

    Supported table chains (closed-form, two solutions each; the one inside the axis limits and with
    the smallest total turn is chosen):
        A + C:  C first aligns the normal with the Y-Z plane, then A tilts it up.
        B + C:  C first aligns it with the Z-X plane, then B tilts it up.
        A only: the normal must have no X component.   B only: no Y component.
        C only: only the plain «normal = +Z» case.

    Raises:
        ValueError: (Spanish message for the operator) unsupported axis combination, a normal the
        axes cannot reach, or every solution outside the axis limits.
    """
    nx, ny, nz = normal
    names = [a for a, _, _ in rots]
    cands = []

    def deg(t):
        return math.degrees(t)
    if set(names) == {"A", "C"}:
        c = math.atan2(nx, ny)                     # turn the normal into the Y-Z plane, on +Y
        r = math.hypot(nx, ny)
        cands = [{"C": deg(c), "A": deg(math.atan2(r, nz))},
                 {"C": deg(c + math.pi), "A": deg(math.atan2(-r, nz))}]
    elif set(names) == {"B", "C"}:
        c = math.atan2(-ny, nx)                    # turn the normal into the Z-X plane, on +X
        r = math.hypot(nx, ny)
        cands = [{"C": deg(c), "B": deg(math.atan2(-r, nz))},
                 {"C": deg(c + math.pi), "B": deg(math.atan2(r, nz))}]
    elif names == ["A"]:
        if abs(nx) > 1e-9:
            raise ValueError("este plano necesita un eje C: con solo A no se puede inclinar hacia X")
        cands = [{"A": deg(math.atan2(ny, nz))}]
    elif names == ["B"]:
        if abs(ny) > 1e-9:
            raise ValueError("este plano necesita un eje C: con solo B no se puede inclinar hacia Y")
        cands = [{"B": deg(math.atan2(-nx, nz))}]
    elif names == ["C"] or not names:
        if abs(nx) > 1e-9 or abs(ny) > 1e-9:
            raise ValueError("la máquina no tiene un eje de mesa que incline la pieza")
        cands = [{n: 0.0 for n in names}]
    else:
        raise ValueError("G53.1 todavía no sabe orientar esta combinación de ejes de mesa: "
                         + "+".join(names))
    ok = []
    for cand in cands:
        cand = {a: (v + 180.0) % 360.0 - 180.0 for a, v in cand.items()}     # −180…180
        up = _direction(normal, cand, rots)
        if abs(up[0]) > 1e-6 or abs(up[1]) > 1e-6 or up[2] < 0:
            continue                               # safety net: must really point up
        inside = True
        for a, v in cand.items():
            ax = m.axes[a]
            machine_angle = v + g54.get(a, 0.0)
            if (ax.min is not None and machine_angle < ax.min - 1e-6) or \
                    (ax.max is not None and machine_angle > ax.max + 1e-6):
                inside = False
        if inside:
            ok.append(cand)
    if not ok:
        raise ValueError("los ejes rotativos no llegan a orientar este plano dentro de sus límites")
    return min(ok, key=lambda c: sum(abs(v) for v in c.values()))
