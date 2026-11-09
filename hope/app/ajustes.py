"""Pantalla «Configuración»: formularios validados en lugar de editar YAML a mano.

Flujo: la interfaz manda los cambios a ``previsualizar`` (valida y devuelve un diff por archivo, sin
escribir nada) y después a ``aplicar`` (valida de nuevo, guarda una **copia de seguridad** de cada
archivo en ``runs/respaldos/`` y escribe). Las ediciones de los YAML son **por línea**: se cambia solo
el valor y se conservan los comentarios.

Archivos que toca:

* ``configs/<plan>.yaml`` — secuencia, optimizador, precisión, horas y lote máximo por etapa, pesos
  de las fuentes (deben sumar 1). La precisión bf16 se rechaza si la calibración dice que la GPU no la
  soporta (Pascal).
* ``runs/app.json`` — carpeta de datos, maestro (endpoint local y modelo).
* ``configs/web.yaml`` — reglas de la web para Hope. **Siempre de solo lectura**: ``acceso:
  solo_lectura`` y métodos ``GET``/``HEAD``; cualquier otro valor se rechaza.
* ``configs/fuentes_web.yaml`` — cuota diaria de datos de la web.
* ``configs/growth_budget.yaml`` — techo de parámetros, con la frase de confirmación y auditoría del
  guardián (``guardian.cli.set_limit``).
"""

from __future__ import annotations

import copy
import difflib
import json
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

import yaml

from hope.app.config import EDITABLES
from hope.train.lowprec import BYTES_PER_PARAM

OPTIMIZADORES = ("adamw", "muon")
PRECISIONES = tuple(BYTES_PER_PARAM)
MODOS_WEB = ("desactivado", "solo_lista", "busqueda")
METODOS_LECTURA = ("GET", "HEAD")
HOST_RE = re.compile(r"^(\*\.)?([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")

WEB_DEFAULT = {
    "modo": "desactivado",
    "acceso": "solo_lectura",
    "metodos": ["GET", "HEAD"],
    "sitios_permitidos": ["es.wikipedia.org", "en.wikipedia.org", "docs.python.org"],
    "sitios_bloqueados": [],
    "limites": {"consultas_por_hora": 60, "paginas_por_dia": 500, "mb_por_pagina": 2, "segundos_por_pedido": 20},
    "respetar_robots": True,
}
WEB_LIMITES = {"consultas_por_hora": (1, 3600), "paginas_por_dia": (1, 100_000), "mb_por_pagina": (0.01, 100),
               "segundos_por_pedido": (1, 300)}
FUENTES_WEB_DEFAULT = {"cuota_diaria_mb": 200, "paginas_por_dia": 2000, "carpeta": "data/web",
                       "horario": {"desde": 0, "hasta": 24}}

WEB_ENCABEZADO = ("# Reglas de la web para Hope (editadas desde la aplicación: python -m hope.app → Configuración).\n"
                  "# SOLO LECTURA: Hope puede buscar y leer páginas (GET/HEAD), nunca enviar formularios, publicar\n"
                  "# ni escribir en la red. La aplicación rechaza cualquier otro valor de «acceso» o «metodos».\n"
                  "# modo: desactivado | solo_lista (solo sitios_permitidos) | busqueda (buscar y leer, sin los bloqueados)\n")
FUENTES_WEB_ENCABEZADO = ("# Cuota diaria de datos de la web para el corpus de Hope (editada desde la aplicación).\n")


class ErrorValidacion(ValueError):
    def __init__(self, errores: list[str]):
        super().__init__("; ".join(errores))
        self.errores = errores


# ---------------------------------------------------------------------------
# Edición de YAML por línea (conserva comentarios)
# ---------------------------------------------------------------------------
def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() and abs(v) >= 1 else repr(round(v, 6))
    return str(v)


def _set_line(line: str, key: str, value: str) -> str:
    """``  key: viejo   # comentario`` → ``  key: nuevo   # comentario``."""
    m = re.match(r"^(\s*-?\s*" + re.escape(key) + r":\s*)([^#\n]*?)(\s*)(#.*)?$", line)
    if not m:
        return line
    if not m.group(4):
        return m.group(1) + value
    # El comentario queda en la misma columna (o a un espacio si el valor nuevo es más largo).
    col = len(m.group(1)) + len(m.group(2)) + len(m.group(3))
    return m.group(1) + value + " " * max(col - len(m.group(1)) - len(value), 1) + m.group(4)


def editar_plan(texto: str, cambios: dict) -> str:
    """Aplica cambios a un plan conservando comentarios. ``cambios`` = {secuencia, optimizador, precision,
    etapas: {nombre: {horas, lote_max, fuentes: {fuente: peso}}}}."""
    lines = texto.splitlines(keepends=False)
    top = {k: cambios[k] for k in ("secuencia", "optimizador", "precision") if k in cambios}
    etapas = cambios.get("etapas", {})
    out: list[str] = []
    actual = None                 # etapa en la que estamos
    en_fuentes = False
    vistos: dict[str, set] = {}
    i_nombre: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if indent == 0 and stripped and not stripped.startswith("#"):
            actual, en_fuentes = None, False
            key = stripped.split(":", 1)[0]
            if key in top:
                line = _set_line(line, key, _fmt(top.pop(key)))
        m = re.match(r"^\s*-\s*nombre:\s*([^#\s]+)", line)
        if m:
            actual, en_fuentes = m.group(1), False
            i_nombre[actual] = len(out)
            vistos[actual] = set()
        elif actual in etapas and stripped and not stripped.startswith("#"):
            ec = etapas[actual]
            key = stripped.split(":", 1)[0]
            if key == "fuentes":
                en_fuentes = True
            elif en_fuentes and indent >= 6 and key in ec.get("fuentes", {}):
                m2 = re.match(r"^(\s*" + re.escape(key) + r":\s*)([^#\s]+)(.*)$", line)
                if m2:
                    path = m2.group(2).rsplit(":", 1)[0]
                    line = m2.group(1) + f"{path}:{_fmt(float(ec['fuentes'][key]))}" + m2.group(3)
                    vistos[actual].add("fuente:" + key)
            elif indent <= 4:
                en_fuentes = False
                if key in ("horas", "lote_max") and key in ec:
                    line = _set_line(line, key, _fmt(ec[key]))
                    vistos[actual].add(key)
        out.append(line)
    # Claves que no estaban en el archivo: se agregan debajo de «- nombre:» de su etapa.
    for nombre in sorted(etapas, key=lambda n: -i_nombre.get(n, -1)):
        if nombre not in i_nombre:
            continue
        faltan = [k for k in ("horas", "lote_max") if k in etapas[nombre] and k not in vistos[nombre]]
        for k in reversed(faltan):
            out.insert(i_nombre[nombre] + 1, f"    {k}: {_fmt(etapas[nombre][k])}")
    for key, v in top.items():
        out.append(f"{key}: {_fmt(v)}")
    return "\n".join(out) + ("\n" if texto.endswith("\n") else "")


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------
def _num(v, kind, lo, hi, nombre, errores):
    try:
        x = kind(v)
        if isinstance(v, bool) or (kind is int and float(v) != int(float(v))):
            raise ValueError
    except (TypeError, ValueError):
        errores.append(f"{nombre}: tiene que ser un número{' entero' if kind is int else ''}")
        return None
    if not lo <= x <= hi:
        errores.append(f"{nombre}: tiene que estar entre {lo:g} y {hi:g}")
        return None
    return x


def validar_plan(plan: dict, c: dict, bf16: bool | None) -> tuple[dict, list[str], list[str]]:
    """Devuelve (cambios normalizados, errores, avisos)."""
    errores, avisos, norm = [], [], {}
    if "secuencia" in c:
        s = _num(c["secuencia"], int, 64, 65536, "secuencia", errores)
        if s is not None and s % 64:
            errores.append("secuencia: tiene que ser múltiplo de 64")
        norm["secuencia"] = s
    if "optimizador" in c:
        if c["optimizador"] not in OPTIMIZADORES:
            errores.append(f"optimizador: {', '.join(OPTIMIZADORES)}")
        norm["optimizador"] = c["optimizador"]
    if "precision" in c:
        p = c["precision"]
        if p not in PRECISIONES:
            errores.append(f"precisión: {', '.join(PRECISIONES)}")
        elif p.startswith("bf16") and bf16 is False:
            errores.append("precisión: esta GPU no soporta bf16 (Pascal/GTX 10xx); usá fp32-8bit")
        elif p.startswith("bf16") and bf16 is None:
            avisos.append("precisión bf16 sin calibrar: no se sabe si la GPU la soporta (calibrá en el paso 1)")
        norm["precision"] = p
    etapas_plan = {e["nombre"]: e for e in plan.get("etapas", [])}
    norm["etapas"] = {}
    for nombre, ec in (c.get("etapas") or {}).items():
        if nombre not in etapas_plan:
            errores.append(f"el plan no tiene la etapa {nombre}")
            continue
        e = etapas_plan[nombre]
        ne = {}
        if "horas" in ec:
            if "crecer_a" in e:
                errores.append(f"{nombre}: es una etapa de crecimiento, no tiene horas")
            else:
                ne["horas"] = _num(ec["horas"], float, 0.01, 20_000, f"{nombre}: horas", errores)
        if "lote_max" in ec:
            ne["lote_max"] = _num(ec["lote_max"], int, 1, 1024, f"{nombre}: lote máximo", errores)
        if "fuentes" in ec:
            fu = {}
            for k, v in ec["fuentes"].items():
                if k not in (e.get("fuentes") or {}):
                    errores.append(f"{nombre}: la fuente {k} no está en el plan")
                    continue
                fu[k] = _num(v, float, 0, 1, f"{nombre}: peso de {k}", errores)
            if fu and None not in fu.values():
                total = sum(fu.values()) + sum(float(str(v).rsplit(":", 1)[1]) for k, v in e["fuentes"].items()
                                               if k not in fu)
                actuales = {k: float(str(v).rsplit(":", 1)[1]) for k, v in e["fuentes"].items()}
                sin_tocar = all(abs(actuales[k] - v) < 1e-9 for k, v in fu.items())
                if abs(total - 1.0) > 1e-3:
                    if sin_tocar:      # ya estaban así en el archivo: no bloquea otros cambios
                        avisos.append(f"{nombre}: los pesos suman {total:.3f} en el plan (Hope los normaliza "
                                      "al entrenar); conviene corregirlos para que sumen 1")
                    else:
                        errores.append(f"{nombre}: los pesos de las fuentes suman {total:.3f}; tienen que sumar 1")
            ne["fuentes"] = fu
        norm["etapas"][nombre] = ne
    return norm, errores, avisos


def validar_app(c: dict) -> tuple[dict, list[str]]:
    from hope.app.maestro import es_local
    errores, norm = [], {}
    for k, v in c.items():
        if k not in EDITABLES:
            errores.append(f"clave no editable: {k}")
            continue
        norm[k] = v
    if "datos.carpeta" in norm and not str(norm["datos.carpeta"]).strip():
        errores.append("carpeta de datos: no puede quedar vacía")
    if "maestro.endpoint" in norm:
        ep = str(norm["maestro.endpoint"]).strip()
        if urlparse(ep).scheme not in ("http", "https") or not urlparse(ep).hostname:
            errores.append("maestro: el endpoint tiene que ser una URL http://…/v1")
        elif not es_local(ep):
            errores.append("maestro: solo servidores locales (127.0.0.1 o localhost); uno remoto necesita "
                           "aprobación del guardián desde la línea de comandos")
    if "maestro.model" in norm and not str(norm["maestro.model"]).strip():
        errores.append("maestro: falta el nombre del modelo")
    return norm, errores


def validar_web(c: dict, base: dict | None = None) -> tuple[dict, list[str]]:
    errores = []
    w = copy.deepcopy(base or WEB_DEFAULT)
    w = {k: w.get(k, copy.deepcopy(v)) for k, v in WEB_DEFAULT.items()}
    w.update({k: v for k, v in c.items() if k in WEB_DEFAULT})
    for k in c:
        if k not in WEB_DEFAULT:
            errores.append(f"web: clave desconocida {k}")
    if w["acceso"] != "solo_lectura":
        errores.append("web: el acceso es siempre de solo lectura (Hope no escribe en la red)")
    metodos = [str(m).upper() for m in (w["metodos"] or [])]
    malos = [m for m in metodos if m not in METODOS_LECTURA]
    if malos:
        errores.append(f"web: métodos no permitidos {', '.join(malos)}: solo GET y HEAD (lectura)")
    w["metodos"] = [m for m in metodos if m in METODOS_LECTURA] or ["GET"]
    if w["modo"] not in MODOS_WEB:
        errores.append(f"web: modo {', '.join(MODOS_WEB)}")
    for lista in ("sitios_permitidos", "sitios_bloqueados"):
        sitios = w[lista]
        if isinstance(sitios, str):
            sitios = [x.strip() for x in sitios.splitlines() if x.strip()]
        limpio = []
        for s in sitios:
            s = str(s).strip().lower()
            if not HOST_RE.match(s):
                errores.append(f"web: «{s}» no es un sitio válido (solo el dominio, p. ej. es.wikipedia.org)")
            else:
                limpio.append(s)
        w[lista] = limpio
    if w["modo"] == "solo_lista" and not w["sitios_permitidos"]:
        errores.append("web: en modo solo_lista hace falta al menos un sitio permitido")
    lim = {**WEB_DEFAULT["limites"], **(w.get("limites") or {})}
    for k, v in list(lim.items()):
        if k not in WEB_LIMITES:
            errores.append(f"web: límite desconocido {k}")
            continue
        lo, hi = WEB_LIMITES[k]
        lim[k] = _num(v, float if k == "mb_por_pagina" else int, lo, hi, f"web: {k}", errores)
    w["limites"] = lim
    w["respetar_robots"] = bool(w["respetar_robots"])
    return w, errores


def validar_fuentes_web(c: dict, base: dict | None = None) -> tuple[dict, list[str]]:
    errores = []
    f = copy.deepcopy(base or FUENTES_WEB_DEFAULT)
    f = {k: f.get(k, copy.deepcopy(v)) for k, v in FUENTES_WEB_DEFAULT.items()}
    for k, v in c.items():
        if k not in FUENTES_WEB_DEFAULT:
            errores.append(f"cuota web: clave desconocida {k}")
        else:
            f[k] = v
    f["cuota_diaria_mb"] = _num(f["cuota_diaria_mb"], float, 0, 1_000_000, "cuota diaria (MB)", errores)
    f["paginas_por_dia"] = _num(f["paginas_por_dia"], int, 0, 10_000_000, "páginas por día", errores)
    if not str(f["carpeta"]).strip():
        errores.append("cuota web: falta la carpeta")
    h = f.get("horario") or {}
    d = _num(h.get("desde", 0), int, 0, 24, "horario: desde", errores)
    a = _num(h.get("hasta", 24), int, 0, 24, "horario: hasta", errores)
    if d is not None and a is not None and d >= a:
        errores.append("horario: «desde» tiene que ser menor que «hasta»")
    f["horario"] = {"desde": d, "hasta": a}
    return f, errores


# ---------------------------------------------------------------------------
# Lectura para el formulario
# ---------------------------------------------------------------------------
def leer_yaml(path: Path, default: dict) -> tuple[dict, bool]:
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return {**copy.deepcopy(default), **data}, True
        except yaml.YAMLError:
            pass
    return copy.deepcopy(default), False


def estado(app) -> dict:
    cfg = app.cfg
    plan = cfg.leer_plan()
    web, web_existe = leer_yaml(cfg.root / "configs" / "web.yaml", WEB_DEFAULT)
    fw, fw_existe = leer_yaml(cfg.root / "configs" / "fuentes_web.yaml", FUENTES_WEB_DEFAULT)
    cal = cfg.data.get("calibracion") or {}
    bf16 = (cal.get("perfil") or {}).get("bf16")
    etapas = []
    for e in plan.get("etapas", []):
        etapas.append({"nombre": e["nombre"], "horas": e.get("horas"), "lote_max": e.get("lote_max"),
                       "crecer": "crecer_a" in e, "preset": e.get("preset"),
                       "fuentes": {k: float(str(v).rsplit(":", 1)[1]) for k, v in (e.get("fuentes") or {}).items()},
                       "rutas": {k: str(v).rsplit(":", 1)[0] for k, v in (e.get("fuentes") or {}).items()}})
    respaldos = sorted((cfg.root / "runs" / "respaldos").glob("*"), reverse=True)[:12] \
        if (cfg.root / "runs" / "respaldos").exists() else []
    return {"plan": {"nombre": cfg.data["plan"], "secuencia": plan.get("secuencia"),
                     "optimizador": plan.get("optimizador", "adamw"), "precision": plan.get("precision", "bf16-8bit"),
                     "etapas": etapas},
            "bf16": bf16, "precisiones": PRECISIONES, "optimizadores": OPTIMIZADORES,
            "web": web, "web_existe": web_existe, "modos_web": MODOS_WEB,
            "fuentes_web": fw, "fuentes_web_existe": fw_existe,
            "respaldos": [p.name for p in respaldos], "listo": requisitos(app)}


# ---------------------------------------------------------------------------
# Previsualizar / aplicar
# ---------------------------------------------------------------------------
def _diff(nombre: str, antes: str, despues: str) -> str:
    return "".join(difflib.unified_diff(antes.splitlines(keepends=True), despues.splitlines(keepends=True),
                                        fromfile=f"{nombre} (actual)", tofile=f"{nombre} (nuevo)"))


def _web_texto(w: dict) -> str:
    return WEB_ENCABEZADO + yaml.safe_dump(w, allow_unicode=True, sort_keys=False)


def _techo_texto(texto: str, valor: int, modo: str | None) -> str:
    texto, n = re.subn(r"(?m)^(max_total_params:\s*)[^#\n]*", rf"\g<1>{valor} ", texto)
    if n == 0:
        texto = f"max_total_params: {valor}\n" + texto
    if modo:
        texto = re.sub(r"(?m)^(mode:\s*)\S+", rf"\g<1>{modo}", texto)
    return texto


def planificar(app, cambios: dict) -> dict:
    """Valida y calcula los archivos nuevos. No escribe nada."""
    cfg = app.cfg
    errores, avisos, archivos = [], [], []     # archivos: (ruta, texto_actual, texto_nuevo)
    cal = cfg.data.get("calibracion") or {}
    bf16 = (cal.get("perfil") or {}).get("bf16")
    app_cambios = dict(cambios.get("app") or {})
    if "plan" in cambios:
        pc = dict(cambios["plan"])
        nombre = pc.pop("nombre", None) or cfg.data["plan"]
        if nombre != cfg.data["plan"]:
            app_cambios["plan"] = nombre
        try:
            path = cfg.plan_plantilla(nombre)
            texto = path.read_text(encoding="utf-8")
            plan = yaml.safe_load(texto)
        except OSError:
            errores.append(f"no existe el plan {nombre}")
            plan, texto, path = None, "", None
        if plan is not None:
            norm, e, a = validar_plan(plan, pc, bf16)
            errores += e
            avisos += a
            if not e:
                nuevo = editar_plan(texto, norm)
                if _verificar_plan(nuevo, norm):
                    # Plan con otra forma (p. ej. escrito por yaml.safe_dump): se reescribe entero,
                    # conservando los comentarios del encabezado.
                    nuevo = _reescribir_plan(texto, plan, norm)
                errores += [f"{m} en {path.name}" for m in _verificar_plan(nuevo, norm)]
                if nuevo != texto:
                    archivos.append((cfg.root / "configs" / nombre, texto, nuevo))
    if app_cambios:
        norm, e = validar_app(app_cambios)
        errores += e
        if "plan" in norm and not re.fullmatch(r"plan_[\w.-]+\.yaml", str(norm["plan"])):
            errores.append("plan inválido")
        if not e:
            antes = json.dumps(cfg.data, indent=2, ensure_ascii=False)
            data = copy.deepcopy(cfg.data)
            for k, v in norm.items():
                cur = data
                parts = k.split(".")
                for p in parts[:-1]:
                    cur = cur.setdefault(p, {})
                cur[parts[-1]] = v
            despues = json.dumps(data, indent=2, ensure_ascii=False)
            if despues != antes:
                archivos.append((cfg.path, antes, despues))
    if "web" in cambios:
        w, e = validar_web(cambios["web"], leer_yaml(cfg.root / "configs" / "web.yaml", WEB_DEFAULT)[0])
        errores += e
        if not e:
            p = cfg.root / "configs" / "web.yaml"
            antes = p.read_text(encoding="utf-8") if p.exists() else ""
            nuevo = _web_texto(w)
            if nuevo != antes:
                archivos.append((p, antes, nuevo))
    if "fuentes_web" in cambios:
        f, e = validar_fuentes_web(cambios["fuentes_web"],
                                   leer_yaml(cfg.root / "configs" / "fuentes_web.yaml", FUENTES_WEB_DEFAULT)[0])
        errores += e
        if not e:
            p = cfg.root / "configs" / "fuentes_web.yaml"
            antes = p.read_text(encoding="utf-8") if p.exists() else ""
            nuevo = FUENTES_WEB_ENCABEZADO + yaml.safe_dump(f, allow_unicode=True, sort_keys=False)
            if nuevo != antes:
                archivos.append((p, antes, nuevo))
    techo = None
    if cambios.get("techo") and str(cambios["techo"].get("valor", "")).strip():
        from guardian.cli import human, parse_count
        from hope.growth.budget import GrowthMode
        t = cambios["techo"]
        try:
            n = parse_count(str(t["valor"]))
            modo = t.get("modo") or None
            if modo:
                GrowthMode(modo)
            frase = f"AUTORIZO {human(n)}"
            techo = {"valor": n, "modo": modo, "frase": frase, "confirmacion": str(t.get("confirmacion", ""))}
            g = app.guardian()
            p = Path(g.config.budget_path)
            antes = p.read_text(encoding="utf-8")
            archivos.append((p, antes, _techo_texto(antes, n, modo)))
            if not (cfg.data.get("usuario") or "").strip():
                errores.append("techo: escribí tu nombre arriba a la derecha (queda en la auditoría)")
        except ValueError as e:
            errores.append(f"techo: {e}")
    return {"errores": errores, "avisos": avisos, "archivos": archivos, "techo": techo}


def _verificar_plan(nuevo: str, norm: dict) -> list[str]:
    """Red de seguridad: el YAML nuevo tiene que leerse y tener exactamente esos valores."""
    try:
        check = yaml.safe_load(nuevo)
    except yaml.YAMLError as e:
        return [f"el plan editado no se puede leer ({e})"]
    errores = []
    for k in ("secuencia", "optimizador", "precision"):
        if k in norm and check.get(k) != norm[k]:
            errores.append(f"no se pudo editar {k}")
    for en, ec in norm.get("etapas", {}).items():
        et = next((x for x in check.get("etapas", []) if x.get("nombre") == en), None)
        if et is None:
            errores.append(f"no se encontró la etapa {en}")
            continue
        for k in ("horas", "lote_max"):
            if k in ec and (et.get(k) is None or float(et.get(k)) != float(ec[k])):
                errores.append(f"no se pudo editar {en}.{k}")
        for f, w in ec.get("fuentes", {}).items():
            if abs(float(str(et["fuentes"][f]).rsplit(":", 1)[1]) - w) > 1e-9:
                errores.append(f"no se pudo editar el peso de {f}")
    return errores


def _reescribir_plan(texto: str, plan: dict, norm: dict) -> str:
    plan = copy.deepcopy(plan)
    for k in ("secuencia", "optimizador", "precision"):
        if k in norm:
            plan[k] = norm[k]
    for e in plan.get("etapas", []):
        ec = norm.get("etapas", {}).get(e.get("nombre"))
        if not ec:
            continue
        for k in ("horas", "lote_max"):
            if k in ec:
                e[k] = ec[k]
        for f, w in ec.get("fuentes", {}).items():
            e["fuentes"][f] = f"{str(e['fuentes'][f]).rsplit(':', 1)[0]}:{_fmt(float(w))}"
    cabecera = []
    for line in texto.splitlines():
        if line.startswith("#") or not line.strip():
            cabecera.append(line)
        else:
            break
    cabecera = "\n".join(cabecera).strip("\n")
    cabecera = cabecera + "\n" if cabecera else ""
    return cabecera + yaml.safe_dump(plan, allow_unicode=True, sort_keys=False)


def previsualizar(app, cambios: dict) -> dict:
    plan = planificar(app, cambios)
    return {"ok": not plan["errores"], "errores": plan["errores"], "avisos": plan["avisos"],
            "mensaje": "hay valores para corregir" if plan["errores"] else "",
            "frase_techo": plan["techo"]["frase"] if plan["techo"] else None,
            "diffs": [{"archivo": str(p.relative_to(app.cfg.root)) if p.is_relative_to(app.cfg.root) else str(p),
                       "diff": _diff(p.name, a, b), "nuevo": not a} for p, a, b in plan["archivos"]]}


def respaldar(app, path: Path) -> Path | None:
    if not path.exists():
        return None
    d = app.cfg.root / "runs" / "respaldos"
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"{time.strftime('%Y%m%d-%H%M%S')}_{path.name}"
    k = 1
    while dest.exists():
        dest = d / f"{time.strftime('%Y%m%d-%H%M%S')}-{k}_{path.name}"
        k += 1
    shutil.copy2(path, dest)
    return dest


def aplicar(app, cambios: dict, confirmar: bool = False) -> dict:
    if not confirmar:
        return {"ok": False, "mensaje": "falta confirmar (revisá las diferencias primero)"}
    plan = planificar(app, cambios)
    techo = plan["techo"]
    if techo and techo["confirmacion"].strip().upper() != techo["frase"].upper():
        plan["errores"].append(f"techo: para confirmar escribí exactamente «{techo['frase']}»")
    if plan["errores"]:
        return {"ok": False, "mensaje": "; ".join(plan["errores"]), "errores": plan["errores"]}
    if app.entrenamiento.corriendo() and any(p.name.startswith("plan_") for p, _, _ in plan["archivos"]):
        plan["avisos"].append("el entrenamiento en curso sigue con el plan anterior: los cambios valen al reanudar")
    respaldos = []
    for path, _antes, nuevo in plan["archivos"]:
        b = respaldar(app, path)
        if b:
            respaldos.append(b.name)
        if techo and path.name == Path(app.guardian().config.budget_path).name:
            # El techo pasa por el guardián: misma función que la CLI, queda auditado.
            from guardian.cli import set_limit
            plan["avisos"] += set_limit(app.guardian(), techo["valor"], by=app.cfg.data["usuario"].strip(),
                                        mode=techo["modo"], note="desde la aplicación (Configuración)")
            continue
        if path == app.cfg.path:
            app.cfg.data = json.loads(nuevo)
            app.cfg.save()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(nuevo, encoding="utf-8")
        tmp.replace(path)
    return {"ok": True, "mensaje": f"guardado: {len(plan['archivos'])} archivo(s); copias en runs/respaldos",
            "respaldos": respaldos, "avisos": plan["avisos"]}


# ---------------------------------------------------------------------------
# «Empezar a entrenar»
# ---------------------------------------------------------------------------
def requisitos(app) -> list[dict]:
    from hope.app.config import plan_efectivo
    cfg = app.cfg
    out = []
    cal = cfg.data.get("calibracion") or {}
    out.append({"nombre": "calibración", "ok": bool(cal.get("plan_recomendado")),
                "detalle": f"dispositivo {cal.get('dispositivo', '?')} ({cal.get('fecha', '')})" if cal
                else "falta calibrar (paso 1)"})
    problemas = (cal.get("torch") or {}).get("problemas") or []
    if problemas:
        out.append({"nombre": "PyTorch y la GPU", "ok": False, "detalle": "; ".join(problemas)})
    try:
        plan = cfg.leer_plan()
        _, rel_plan = plan_efectivo(cfg, escribir=False)
        est = app.entrenamiento.estado_run()
        etapa = next((e for e in rel_plan.get("etapas", []) if not est.get("etapas", {}).get(e["nombre"], {}).get("hecha")
                      and "fuentes" in e), None)
        if etapa is None:
            out.append({"nombre": "datos", "ok": True, "detalle": "no quedan etapas con datos por entrenar"})
        else:
            hay, faltan = [], []
            for k, v in etapa["fuentes"].items():
                d = cfg.ruta(str(v).rsplit(":", 1)[0])
                ok = (d / "train.bin").exists() and (d / "train.bin").stat().st_size > 1024
                (hay if ok else faltan).append(k)
            out.append({"nombre": f"datos de la etapa {etapa['nombre']}", "ok": bool(hay),
                        "detalle": (f"listas: {', '.join(hay)}" if hay else "ninguna fuente construida (paso 2)")
                        + (f" · faltan (se ignoran): {', '.join(faltan)}" if faltan and hay else "")})
        if plan.get("desde"):
            d = cfg.ruta(str(rel_plan["desde"]))
            out.append({"nombre": "checkpoint de origen", "ok": (d / "model.pt").exists()
                        or (cfg.ruta(rel_plan.get("salida", "runs/hope")) / "estado.json").exists(),
                        "detalle": str(rel_plan["desde"])})
    except (OSError, ValueError, KeyError) as e:
        out.append({"nombre": "plan", "ok": False, "detalle": str(e)})
    return out


def empezar(app) -> dict:
    req = requisitos(app)
    malos = [r for r in req if not r["ok"]]
    if malos:
        return {"ok": False, "mensaje": "falta: " + "; ".join(f"{r['nombre']} ({r['detalle']})" for r in malos),
                "requisitos": req}
    r = app.entrenamiento.iniciar()
    return {**r, "requisitos": req}


def agregar_etapa_presupuesto(app, valor: str) -> dict:
    """Agrega ``- nombre: presupuesto_<X>`` con ``presupuesto: X`` después de la etapa en curso (o al final),
    con copia de seguridad del plan. La interpreta ``hope.train.run`` (rama de presupuesto)."""
    cfg = app.cfg
    path = cfg.root / "configs" / cfg.data["plan"]
    texto = path.read_text(encoding="utf-8")
    plan = yaml.safe_load(texto)
    if any(e.get("presupuesto") == valor for e in plan.get("etapas", [])):
        return {"ok": True, "mensaje": f"el plan ya tiene una etapa para llegar a {valor}"}
    est = app.entrenamiento.estado_run().get("etapas", {})
    nombres = [e["nombre"] for e in plan.get("etapas", [])]
    actual = next((n for n in nombres if not est.get(n, {}).get("hecha")), None)
    lines = texto.splitlines()
    idx = [i for i, line in enumerate(lines) if re.match(r"^\s*-\s*nombre:", line)]
    pos = len(lines)
    if actual is not None:
        k = nombres.index(actual)
        if k + 1 < len(idx):
            pos = idx[k + 1]
    nombre = f"presupuesto_{valor.lower()}"
    bloque = [f"  - nombre: {nombre}       # agregada desde la aplicación ({time.strftime('%Y-%m-%d')})",
              f"    presupuesto: {valor}"]
    nuevo = "\n".join(lines[:pos] + bloque + lines[pos:]) + "\n"
    check = yaml.safe_load(nuevo)
    if not any(e.get("nombre") == nombre for e in check.get("etapas", [])):
        return {"ok": False, "mensaje": "no se pudo agregar la etapa al plan"}
    respaldar(app, path)
    path.write_text(nuevo, encoding="utf-8")
    return {"ok": True, "mensaje": f"etapa {nombre} agregada al plan {path.name}"}
