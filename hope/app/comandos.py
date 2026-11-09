"""Chat de control: frases en español → comandos de una LISTA FIJA, siempre con confirmación.

El usuario escribe «entrená 2 semanas», «pausá a la noche», «agregá Wikipedia en inglés», «¿cómo va?»…
``interpretar`` lo traduce por reglas y palabras clave (sin modelo, sin ejecutar nada) a:

* ``{"tipo": "respuesta", "texto": …}`` — preguntas de estado o ayuda: no cambian nada;
* ``{"tipo": "pregunta", "texto": …}`` — la frase es ambigua o le falta un dato: se pide aclarar;
* ``{"tipo": "propuesta", "propuesta": {id, comando, args, resumen, sensible, frase}}`` — qué va a cambiar.
  Nada se aplica hasta ``confirmar(id)``. Los cambios sensibles (techo de parámetros, reglas de la web)
  exigen además escribir la frase de confirmación y quedan en la auditoría del guardián.

Solo existen los comandos de ``COMANDOS``; lo que no encaja en ninguno se contesta con la ayuda.
"""

from __future__ import annotations

import re
import time
import unicodedata
import uuid

COMANDOS = {
    "estado": "contar cómo va el entrenamiento",
    "ayuda": "mostrar qué se puede pedir",
    "empezar": "empezar en modo automático (o continuar)",
    "pausar": "pausar el entrenamiento (después del próximo checkpoint)",
    "continuar": "seguir entrenando",
    "duracion": "cambiar cuánto dura la etapa actual",
    "horario": "pausar todos los días a una hora y seguir a otra",
    "quitar_horario": "sacar el horario de pausa",
    "fuente_agregar": "agregar una fuente de textos",
    "fuente_quitar": "quitar una fuente de textos",
    "texto_gb": "cambiar cuántos GB de texto se bajan",
    "parametros": "cambiar los parámetros disponibles (techo del guardián)",
    "web": "activar o desactivar la lectura de la web (siempre solo lectura)",
    "cuantizar": "exportar a Hope en 4 u 8 bits",
    "calibrar": "volver a medir la PC",
}
FUENTES = {"wiki_es": "Wikipedia en español", "wiki_en": "Wikipedia en inglés", "codigo": "código",
           "repos": "tus repositorios", "propias": "tus carpetas (libros, manuales)"}
NUMEROS = {"un": 1, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
           "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "quince": 15, "veinte": 20, "treinta": 30,
           "media": 0.5, "medio": 0.5}
UNIDAD_H = {"hora": 1, "horas": 1, "h": 1, "dia": 24, "dias": 24, "semana": 168, "semanas": 168, "mes": 720, "meses": 720}
EXPIRA_S = 600
AYUDA = ("Puedo: «¿cómo va?», «empezá», «pausá», «seguí», «entrená 2 semanas», «pausá a la noche» o "
         "«pausá de 23 a 7», «sacá el horario», «agregá Wikipedia en inglés», «quitá el código», "
         "«usá 60 GB de texto», «usá 1B de parámetros», «activá la web» / «desactivá la web», "
         "«exportá en 4 bits», «calibrá de nuevo».")


def normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[¿?¡!.,;:]+(?!\d)", " ", t)).strip()


def _num(tok: str) -> float | None:
    tok = tok.replace(",", ".")
    if re.fullmatch(r"\d+(\.\d+)?", tok):
        return float(tok)
    return NUMEROS.get(tok)


def duracion_horas(t: str) -> float | None:
    m = re.search(r"(\d+(?:[.,]\d+)?|" + "|".join(NUMEROS) + r")\s*(horas?|h|dias?|semanas?|mes(?:es)?)\b", t)
    if not m:
        return None
    n = _num(m.group(1))
    return None if n is None else n * UNIDAD_H[m.group(2)]


def parametros(t: str) -> int | None:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(mil millones|millones|[kmbg])\b", t)
    if not m:
        return None
    mult = {"k": 1e3, "m": 1e6, "millones": 1e6, "b": 1e9, "g": 1e9, "mil millones": 1e9}[m.group(2)]
    return int(float(m.group(1).replace(",", ".")) * mult)


def _horas_del_dia(t: str) -> tuple[int | None, int | None, bool]:
    """(pausa, seguir, eran_por_defecto)."""
    m = re.search(r"de (?:las )?(\d{1,2})(?:[:h]\d{2})? (?:a|hasta) (?:las )?(\d{1,2})", t)
    if m:
        return int(m.group(1)) % 24, int(m.group(2)) % 24, False
    m = re.search(r"a las (\d{1,2})", t)
    if m:
        return int(m.group(1)) % 24, None, False
    if "noche" in t:
        return 23, 7, True
    return None, None, False


# ---------------------------------------------------------------------------
def interpretar(app, texto: str) -> dict:
    t = normalizar(texto)
    if not t:
        return {"tipo": "pregunta", "texto": "¿Qué querés hacer? " + AYUDA}
    cands = _candidatos(t)
    if not cands:
        return {"tipo": "respuesta", "texto": "No entendí. " + AYUDA}
    if len(cands) > 1:
        return {"tipo": "pregunta", "texto": "¿Qué querés hacer: " + " o ".join(COMANDOS[c] for c in cands) + "?"}
    return _armar(app, cands[0], t)


def _candidatos(t: str) -> list[str]:
    c = []
    if re.search(r"\b(como va|como vas|estado|progreso|cuanto falta|que tal|como anda)\b", t):
        return ["estado"]
    if re.search(r"\b(ayuda|que (puedo|podes) (hacer|pedir)|comandos)\b", t):
        return ["ayuda"]
    if re.search(r"\b(saca|quita|borra|elimina|desactiva)\w* (el |los )?horario", t):
        return ["quitar_horario"]
    pausa = re.search(r"\b(pausa|pausar|pausalo|pausala|para|parar|paralo|detene|detener|detenelo|frena|frenar|"
                      r"apaga|apagar)\b", t)
    if pausa and (re.search(r"\b(noche|a las \d|de \d{1,2} a \d)", t) or "todos los dias" in t):
        c.append("horario")
    elif pausa and not re.search(r"\bpara que\b", t):
        c.append("pausar")
    if re.search(r"\b(segui|continua|reanuda|retoma|volve a entrenar)\w*\b", t):
        c.append("continuar")
    if re.search(r"\b(empeza|empezar|arranca|inicia|comenza)\w*\b", t) and not c:
        c.append("empezar")
    if re.search(r"\bparametros?\b", t) or re.search(r"\b(tamano|mas grande|mas chico|achica|agranda)\b", t):
        c.append("parametros")
    if re.search(r"\b(web|internet)\b", t):
        c.append("web")
    if re.search(r"\b(cuantiza|4 bits|8 bits|exporta)\w*", t):
        c.append("cuantizar")
    if re.search(r"\b(calibra|medi la pc|medir la pc)\w*", t):
        c.append("calibrar")
    if re.search(r"\b(entrena|que dure|durante|por)\b", t) and duracion_horas(t) and "parametros" not in c:
        c.append("duracion")
    elif re.search(r"\bentrena\w*\b", t) and not c:
        c.append("duracion")                    # «entrená más» sin cuánto: se pregunta
    if re.search(r"\b\d+(?:[.,]\d+)?\s*gb\b", t) and re.search(r"\b(texto|datos|corpus|baja)\w*", t):
        c.append("texto_gb")
    if re.search(r"\b(agrega|suma|anade|incorpora|inclui|usa tambien)\w*\b", t) and "parametros" not in c and "web" not in c:
        c.append("fuente_agregar")
    if re.search(r"\b(quita|saca|elimina|sin)\b", t) and _fuente(t) and "quitar_horario" not in c:
        c.append("fuente_quitar")
    # Sin duplicados, respetando el orden.
    return list(dict.fromkeys(c))


def _fuente(t: str) -> str | None:
    if "wikipedia" in t or "wiki" in t:
        if re.search(r"\b(ingles|english|en ingles)\b", t):
            return "wiki_en"
        if re.search(r"\b(espanol|castellano)\b", t):
            return "wiki_es"
        return "wiki?"
    if re.search(r"\b(repos?|repositorios?|git)\b", t):
        return "repos"
    if re.search(r"\b(codigo|programas?|python)\b", t):
        return "codigo"
    if re.search(r"\b(libros?|manuales?|carpetas?|apuntes|documentos)\b", t):
        return "propias"
    return None


def _propuesta(app, comando: str, args: dict, resumen: str, *, sensible: bool = False, frase: str | None = None,
               diffs: list | None = None) -> dict:
    pid = uuid.uuid4().hex[:10]
    p = {"id": pid, "comando": comando, "args": args, "resumen": resumen, "sensible": sensible, "frase": frase,
         "diffs": diffs or [], "creada": time.time()}
    props = app.propuestas
    for k in [k for k, v in props.items() if time.time() - v["creada"] > EXPIRA_S]:
        props.pop(k, None)
    props[pid] = p
    return {"tipo": "propuesta", "propuesta": {k: v for k, v in p.items() if k != "creada"}}


def _etapa_actual(app) -> dict | None:
    m = app.entrenamiento.metricas()
    return next((e for e in m["etapas"] if not e["hecha"] and e.get("horas_plan")), None)


def _armar(app, cmd: str, t: str) -> dict:
    from hope.app import ajustes
    cfg = app.cfg
    if cmd == "ayuda":
        return {"tipo": "respuesta", "texto": AYUDA}
    if cmd == "estado":
        return {"tipo": "respuesta", "texto": frase_estado(app)}
    if cmd == "empezar":
        return _propuesta(app, cmd, {}, "Empezar en modo automático: medir la PC, bajar los textos, preparar el maestro "
                                        "y entrenar (lo que ya esté hecho no se repite).")
    if cmd == "pausar":
        return _propuesta(app, cmd, {}, "Pausar el entrenamiento después del próximo checkpoint (no se pierde nada).")
    if cmd == "continuar":
        return _propuesta(app, cmd, {}, "Seguir entrenando desde el último checkpoint.")
    if cmd == "calibrar":
        return _propuesta(app, cmd, {}, "Volver a medir la PC (hay que pausar el entrenamiento: los dos usan la GPU).")
    if cmd == "quitar_horario":
        return _propuesta(app, cmd, {}, "Sacar el horario de pausa: entrena todo el día.")
    if cmd == "horario":
        pausa, seguir, defecto = _horas_del_dia(t)
        if pausa is None:
            return {"tipo": "pregunta", "texto": "¿A qué hora pauso? Por ejemplo: «pausá de 23 a 7»."}
        if seguir is None:
            return {"tipo": "pregunta", "texto": f"Pauso a las {pausa:02d}:00. ¿A qué hora sigo? Por ejemplo: "
                                                 f"«pausá de {pausa} a 7»."}
        if pausa == seguir:
            return {"tipo": "pregunta", "texto": "La hora de pausa y la de seguir son iguales. ¿Cuáles querés?"}
        extra = " (entendí «a la noche» como de 23 a 7; si querés otras horas, decí «pausá de 22 a 6»)" if defecto else ""
        return _propuesta(app, cmd, {"pausa": pausa, "seguir": seguir},
                          f"Pausar todos los días a las {pausa:02d}:00 y seguir a las {seguir:02d}:00{extra}.")
    if cmd == "duracion":
        h = duracion_horas(t)
        if h is None:
            return {"tipo": "pregunta", "texto": "¿Cuánto tiempo? Por ejemplo: «entrená 2 semanas» o «entrená 3 días»."}
        e = _etapa_actual(app)
        if e is None:
            return {"tipo": "respuesta", "texto": "El plan elegido no tiene una etapa con horas pendiente."}
        if h <= e["horas"]:
            return {"tipo": "pregunta", "texto": f"La etapa {e['nombre']} ya lleva {e['horas']:.0f} h. ¿Querés que dure "
                                                 f"más que eso en total?"}
        cambios = {"plan": {"etapas": {e["nombre"]: {"horas": round(h, 2)}}}}
        prev = ajustes.previsualizar(app, cambios)
        if not prev["ok"]:
            return {"tipo": "respuesta", "texto": "No se puede: " + "; ".join(prev["errores"])}
        return _propuesta(app, cmd, cambios, f"La etapa {e['nombre']} va a durar {_humano_h(h)} en total "
                          f"(ya lleva {_humano_h(e['horas'])}; antes eran {_humano_h(e['horas_plan'])}).",
                          diffs=prev["diffs"])
    if cmd in ("fuente_agregar", "fuente_quitar"):
        f = _fuente(t)
        if f is None:
            return {"tipo": "pregunta", "texto": "¿Qué fuente? " + ", ".join(FUENTES.values()) + "."}
        if f == "wiki?":
            return {"tipo": "pregunta", "texto": "¿Wikipedia en español o en inglés?"}
        rep = dict(cfg.data["datos"]["reparto"])
        if cmd == "fuente_agregar":
            if f in ("repos", "propias") and not cfg.data["datos"].get(f):
                return {"tipo": "respuesta", "texto": f"Para agregar {FUENTES[f]} escribí las carpetas en «Más opciones → "
                                                      "Datos» (por seguridad no se toman rutas desde el chat)."}
            if rep.get(f, 0) > 0:
                return {"tipo": "respuesta", "texto": f"{FUENTES[f]} ya está incluida ({rep[f]:.0%} del texto)."}
            rep[f] = 0.15
            texto = f"Agregar {FUENTES[f]} (15 % del texto) y bajarla ahora."
        else:
            if rep.get(f, 0) <= 0:
                return {"tipo": "respuesta", "texto": f"{FUENTES[f]} no está incluida."}
            rep[f] = 0.0
            texto = f"Quitar {FUENTES[f]} de los textos que se bajan (lo ya bajado queda en el disco)."
        return _propuesta(app, cmd, {"reparto": rep, "fuente": f}, texto)
    if cmd == "texto_gb":
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*gb", t)
        gb = float(m.group(1).replace(",", "."))
        if not 0.1 <= gb <= 5000:
            return {"tipo": "respuesta", "texto": "Tiene que ser entre 0,1 y 5000 GB."}
        return _propuesta(app, cmd, {"gb": gb}, f"Bajar hasta {gb:g} GB de texto (antes: "
                                                f"{cfg.data['datos']['presupuesto_gb']:g} GB).")
    if cmd == "parametros":
        n = parametros(t)
        if n is None:
            return {"tipo": "pregunta", "texto": "¿Cuántos parámetros? Por ejemplo: «usá 500M de parámetros» o «usá 1B»."}
        from guardian.cli import human
        try:
            actual = app.guardian().budget().max_total_params
        except (OSError, ValueError):
            actual = None
        disp = presupuesto_disponible()
        resumen = (f"Parámetros disponibles: {human(n)} (techo actual del guardián: {human(actual) if actual else '?'}). "
                   "El techo es un máximo autorizado, no una meta.")
        resumen += (" Además se agrega al plan una etapa para llevar el modelo a ese tamaño." if disp else
                    " El cambio de tamaño del modelo ya entrenado todavía no está disponible en esta versión: por "
                    "ahora solo se ajusta el techo.")
        return _propuesta(app, cmd, {"valor": n}, resumen, sensible=True, frase=f"AUTORIZO {human(n)}")
    if cmd == "web":
        activar = not re.search(r"\b(desactiva|apaga|saca|quita|sin|no uses|corta)\w*", t)
        modo = "lista_permitida" if activar else "apagado"
        prev = ajustes.previsualizar(app, {"web": {"modo": modo}})
        if not prev["ok"]:
            return {"tipo": "respuesta", "texto": "No se puede: " + "; ".join(prev["errores"])}
        texto = ("Activar la lectura de la web solo en los sitios permitidos (siempre solo lectura: Hope lee, nunca "
                 "escribe ni envía nada)." if activar else "Desactivar la web: Hope no lee nada de internet.")
        return _propuesta(app, cmd, {"modo": modo}, texto, sensible=True,
                          frase="ACTIVAR WEB" if activar else "DESACTIVAR WEB", diffs=prev["diffs"])
    if cmd == "cuantizar":
        bits = 8 if re.search(r"\b8\b", t) else 4 if re.search(r"\b4\b", t) else None
        if bits is None:
            return {"tipo": "pregunta", "texto": "¿En 4 bits (más chico) o en 8 bits (más fiel)?"}
        return _propuesta(app, cmd, {"bits": bits}, f"Exportar el último modelo en {bits} bits a runs/exportar.")
    return {"tipo": "respuesta", "texto": AYUDA}


def confirmar(app, pid: str, frase: str = "") -> dict:
    p = app.propuestas.get(pid)
    if p is None or time.time() - p["creada"] > EXPIRA_S:
        app.propuestas.pop(pid, None)
        return {"ok": False, "mensaje": "esa propuesta ya no vale; pedilo de nuevo"}
    if p["sensible"] and (frase or "").strip().upper() != (p["frase"] or "").upper():
        return {"ok": False, "mensaje": f"para confirmar escribí exactamente: {p['frase']}"}
    app.propuestas.pop(pid, None)
    r = _ejecutar(app, p)
    if r.get("ok") and p["sensible"]:
        try:
            app.guardian().audit.append("app.chat_confirmado", by=app.cfg.data.get("usuario") or "?",
                                        comando=p["comando"], args=p["args"], resumen=p["resumen"])
        except (OSError, ValueError):
            pass
    return r


def _ejecutar(app, p: dict) -> dict:
    from hope.app import ajustes
    cmd, a = p["comando"], p["args"]
    cfg = app.cfg
    if cmd == "empezar":
        return app.tareas["automatico"].iniciar()
    if cmd == "pausar":
        r = app.entrenamiento.pausar()
        if app.tareas["automatico"].ocupada():
            app.tareas["automatico"].pausar()
            return {"ok": True, "mensaje": "pausado"}
        return r
    if cmd == "continuar":
        if not app.tareas["automatico"].ocupada() and set(app.tareas["automatico"].hechos()) != set(
                __import__("hope.app.automatico", fromlist=["PASOS"]).PASOS):
            return app.tareas["automatico"].iniciar()
        return app.entrenamiento.iniciar()
    if cmd == "calibrar":
        return app.tareas["calibracion"].iniciar()
    if cmd == "horario":
        cfg.set("horario", {"activo": True, "pausa": a["pausa"], "seguir": a["seguir"]})
        return {"ok": True, "mensaje": f"horario: pausa {a['pausa']:02d}:00, sigue {a['seguir']:02d}:00"}
    if cmd == "quitar_horario":
        cfg.set("horario", {"activo": False})
        return {"ok": True, "mensaje": "sin horario de pausa"}
    if cmd == "duracion":
        return ajustes.aplicar(app, a, confirmar=True)
    if cmd in ("fuente_agregar", "fuente_quitar"):
        cfg.set("datos.reparto", a["reparto"])
        hechas = dict(cfg.data["datos"].get("hechas") or {})
        if cmd == "fuente_agregar":
            hechas.pop(a["fuente"], None)
            cfg.set("datos.hechas", hechas)
            if not app.tareas["datos"].ocupada():
                app.tareas["datos"].iniciar()
        return {"ok": True, "mensaje": p["resumen"]}
    if cmd == "texto_gb":
        cfg.set("datos.presupuesto_gb", float(a["gb"]))
        cfg.set("automatico_presupuesto_fijo", True)
        return {"ok": True, "mensaje": f"hasta {a['gb']:g} GB de texto"}
    if cmd == "parametros":
        from guardian.cli import human, set_limit
        by = (cfg.data.get("usuario") or "").strip() or "usuario (chat)"
        avisos = set_limit(app.guardian(), int(a["valor"]), by=by, note="desde el chat de la aplicación")
        msg = f"parámetros disponibles: {human(a['valor'])}"
        if presupuesto_disponible():
            msg += "; la etapa de tamaño se agrega desde «Más opciones → Configuración → Parámetros disponibles»"
        return {"ok": True, "mensaje": msg, "avisos": avisos}
    if cmd == "web":
        return ajustes.aplicar(app, {"web": {"modo": a["modo"]}}, confirmar=True)
    if cmd == "cuantizar":
        return app.tareas["cuantizar"].iniciar(bits=int(a["bits"]))
    return {"ok": False, "mensaje": "comando desconocido"}


# ---------------------------------------------------------------------------
def _humano_h(h: float | None) -> str:
    if h is None:
        return "—"
    if h < 1:
        return f"{h * 60:.0f} min"
    if h < 48:
        return f"{h:.0f} h"
    if h >= 336 and abs(h / 168 - round(h / 168)) < 0.05:
        return f"{round(h / 168):.0f} semanas"
    return f"{h / 24:.1f} días".replace(".0 ", " ")


ETAPAS_LEGIBLES = (("lenguaje", "aprendiendo idiomas y código"), ("maestro", "aprendiendo del maestro"),
                   ("crecer", "creciendo"), ("vision", "aprendiendo a ver"), ("multimodal", "aprendiendo a ver y leer"),
                   ("enfriamiento", "afinando lo aprendido"))


def frase_estado(app) -> str:
    m = app.entrenamiento.metricas()
    auto = app.tareas.get("automatico")
    if auto is not None and auto.ocupada() and auto.paso and auto.paso != "entrenar":
        return f"Hope se está preparando: {auto.detalle or auto.paso}."
    e = next((x for x in m["etapas"] if x["nombre"] == m.get("etapa_actual")), None)
    if e is None:
        return "Hope terminó todas las etapas del plan." if m["etapas"] else "Hope todavía no empezó."
    que = next((v for k, v in ETAPAS_LEGIBLES if k in e["nombre"]), f"en la etapa {e['nombre']}")
    pct = f" — {min(e['horas'] / e['horas_plan'], 1):.0%} de la etapa" if e.get("horas_plan") else ""
    falta = f", faltan {_humano_h(m['eta_horas'])}" if m.get("eta_horas") else ""
    estados = {"en_curso": f"Hope está {que}{pct}{falta}.", "pausando": f"Hope se está por pausar ({que}{pct}).",
               "pausado": f"Hope está en pausa ({que}{pct}{falta}).",
               "esperando_aprobacion": "Hope espera tu aprobación para crecer (guardián).",
               "reintentando": "Hope tuvo un problema y se vuelve a lanzar solo.",
               "terminado": "Hope terminó todas las etapas del plan.", "error": "Hope se detuvo por un error: mirá el registro."}
    return estados.get(m["estado"], f"Hope está detenido ({que}{pct}).")


def presupuesto_disponible() -> bool:
    """¿Está el módulo que lleva un modelo entrenado a otro tamaño (rama de presupuesto)?"""
    try:
        import importlib
        mod = importlib.import_module("hope.growth.budget_target")
        return hasattr(mod, "plan_to_budget")
    except ImportError:
        return False
