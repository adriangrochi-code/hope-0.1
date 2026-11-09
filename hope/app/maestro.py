"""Paso 4: el maestro local (llama.cpp u Ollama con Qwen en la CPU) y sus episodios verificados.

* ``probar(cfg)`` — prueba la conexión: lista de modelos y un pedido mínimo. Solo servidores locales:
  un maestro remoto implica salir a la red y requiere aprobación del guardián (CLI).
* ``TareaMaestro`` — recorre el currículo (``configs/curriculum.yaml``): pide un episodio por tema,
  lo **verifica con las herramientas** (``hope.tools.verify``) y lo acepta o lo rechaza; suma
  episodios de los generadores verificables (gratis, sin maestro). Todo queda en
  ``<datos>/maestro/episodios.jsonl`` y los aceptados forman ``<datos>/maestro/train.bin``/``val.bin``,
  la fuente ``maestro`` de la etapa 2 del plan. Se puede pausar y reanudar: los conteos se releen.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import yaml

from hope.app.tareas import Tarea
from hope.data.corpus import EOD, split_of
from hope.teacher.client import LOCAL_HOSTS, MockTeacher, OpenAICompatibleTeacher, TeacherConfig, TeacherDenied


_HARVEST: dict = {}


def harvest_continuo_disponible() -> bool:
    """¿El cosechador del maestro ya tiene modo continuo? (se pregunta una vez a su --help)."""
    if "v" not in _HARVEST:
        import subprocess
        import sys
        try:
            r = subprocess.run([sys.executable, "-m", "hope.teacher.harvest", "--help"], capture_output=True,
                               text=True, timeout=60)
            _HARVEST["v"] = "--continuo" in r.stdout
        except (OSError, subprocess.TimeoutExpired):
            _HARVEST["v"] = False
    return _HARVEST["v"]


def teacher_config(cfg) -> TeacherConfig:
    m = cfg.data["maestro"]
    return TeacherConfig(endpoint=m["endpoint"], model=m["model"], temperature=float(m["temperature"]),
                         max_tokens=int(m["max_tokens"]), timeout_s=float(m["timeout_s"]))


def guardar_yaml(cfg) -> Path:
    """``runs/maestro.yaml``: la misma configuración en el formato de ``configs/teacher.yaml``."""
    tc = teacher_config(cfg)
    p = cfg.root / "runs" / "maestro.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# Generado por la aplicación (python -m hope.app)\n" +
                 yaml.safe_dump(tc.__dict__, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def es_local(endpoint: str) -> bool:
    return (urlparse(endpoint).hostname or "") in LOCAL_HOSTS


def probar(cfg, endpoint: str | None = None, model: str | None = None) -> dict:
    """Prueba la conexión con lo guardado o con un endpoint/modelo que todavía no se guardó."""
    tc = teacher_config(cfg)
    if endpoint:
        tc.endpoint = str(endpoint).strip()
    if model:
        tc.model = str(model).strip()
    if tc.model == "mock":
        return {"ok": True, "mensaje": "maestro simulado (model: mock), solo para pruebas", "modelos": ["mock"]}
    if urlparse(tc.endpoint).scheme not in ("http", "https"):
        return {"ok": False, "mensaje": "el endpoint tiene que empezar con http:// o https://"}
    if not es_local(tc.endpoint):
        return {"ok": False, "mensaje": "la aplicación solo usa maestros locales (127.0.0.1 / localhost). Un maestro "
                                        "remoto sale a la red y necesita aprobación del guardián desde la línea de "
                                        "comandos."}
    out: dict = {"ok": False}
    base = tc.endpoint.rstrip("/")
    try:
        with urllib.request.urlopen(base + "/models", timeout=5) as r:
            data = json.loads(r.read().decode())
        out["modelos"] = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
    except (urllib.error.URLError, OSError, ValueError) as e:
        out["mensaje"] = (f"no responde {base}: {e}. ¿Está corriendo el servidor? llama.cpp: "
                          "llama-server -m qwen.gguf --port 8080 · Ollama: ollama serve (puerto 11434)")
        return out
    t0 = time.time()
    try:
        teacher = OpenAICompatibleTeacher(TeacherConfig(**{**tc.__dict__, "max_tokens": 16,
                                                           "timeout_s": min(tc.timeout_s, 120)}))
        reply = teacher.ask([{"role": "user", "content": "Respondé solo con la palabra: listo"}], purpose="prueba")
        out.update(ok=True, respuesta=reply.text[:200], segundos=round(time.time() - t0, 1),
                   mensaje=f"el maestro respondió en {time.time() - t0:.1f} s")
        if out["modelos"] and tc.model not in out["modelos"]:
            out["aviso"] = f"el servidor no lista el modelo «{tc.model}»; modelos: {', '.join(map(str, out['modelos']))}"
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        out["mensaje"] = f"el servidor responde pero el pedido de prueba falló: {e}"
    return out


class TareaMaestro(Tarea):
    nombre = "maestro"

    def carpeta(self) -> Path:
        return self.app.cfg.carpeta_datos / "maestro"

    def conteos(self) -> dict:
        c = {"aceptados": 0, "rechazados": 0, "invalidos": 0, "del_maestro": 0, "generados": 0,
             "respaldados": 0, "inferidos": 0, "hoy_maestro": 0, "hoy_aceptados": 0, "por_tema": {},
             "por_maestro": {}}
        p = self.carpeta() / "episodios.jsonl"
        if not p.exists():
            return c
        st = p.stat()
        clave = (str(p), st.st_mtime_ns, st.st_size, time.strftime("%Y-%m-%d"))
        tb = self.carpeta() / "train.bin"
        if getattr(self, "_cache_conteos", (None,))[0] == clave:     # la interfaz consulta cada 2 s
            return {**self._cache_conteos[1], "bytes_train": tb.stat().st_size if tb.exists() else 0}
        hoy = time.strftime("%Y-%m-%d")
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = "aceptados" if r.get("aceptado") else ("invalidos" if r.get("estado") == "INVALIDO" else "rechazados")
                c[key] += 1
                c["del_maestro" if r.get("origen") == "maestro" else "generados"] += 1
                if r.get("estado") == "RESPALDADO":
                    c["respaldados"] += 1
                elif r.get("estado") == "INFERIDO":
                    c["inferidos"] += 1
                if r.get("origen") == "maestro":
                    pm = c["por_maestro"].setdefault(r.get("maestro") or "?", {"aceptados": 0, "rechazados": 0})
                    pm["aceptados" if r.get("aceptado") else "rechazados"] += 1
                if r.get("origen") == "maestro" and str(r.get("fecha", "")).startswith(hoy):
                    c["hoy_maestro"] += 1
                    c["hoy_aceptados"] += bool(r.get("aceptado"))
                t = c["por_tema"].setdefault(r.get("tema", "?"), {"aceptados": 0, "rechazados": 0})
                t["aceptados" if r.get("aceptado") else "rechazados"] += 1
        c["bytes_train"] = (self.carpeta() / "train.bin").stat().st_size if (self.carpeta() / "train.bin").exists() else 0
        self._cache_conteos = (clave, c)
        return c

    def ejecutar(self, rondas: int | None = None, continuo: bool = False, maestro: dict | None = None,
                 minutos: float | None = None, paralelo: int = 1) -> None:
        """``continuo``: no termina nunca (Qwen explica mientras Hope entrena). El máximo de consultas pasa a
        ser por día; sin episodios nuevos, espera un minuto y sigue.

        ``maestro``: {id, endpoint, model} de este turno (modo turnos con varios maestros); cada episodio
        queda con el id de su maestro. ``minutos``: termina al cumplirse (fin del turno). ``paralelo``:
        pedidos simultáneos al servidor (en el turno del maestro la GPU es toda suya: lotes grandes)."""
        if continuo and maestro is None and harvest_continuo_disponible():
            return self._harvest_continuo()
        fin = time.time() + minutos * 60 if minutos else None
        self.maestro_actual = maestro
        from hope.data.generators import family_for, generate
        from hope.teacher.cycle import load_curriculum
        from hope.teacher.episode import Episode, EpisodeParseError, curriculum_request, parse_episode
        from hope.tools.builtin import default_catalog
        from hope.tools.verify import verify_episode
        cfg = self.app.cfg
        m = cfg.data["maestro"]
        tc = teacher_config(cfg)
        if maestro:
            tc.endpoint, tc.model = maestro.get("endpoint", tc.endpoint), maestro.get("model", tc.model)
        mid = (maestro or {}).get("id") or tc.model
        guardar_yaml(cfg)
        guardian = self.app.guardian()
        if tc.model == "mock":
            teacher = MockTeacher(tc, guardian, error_rate=float(m.get("mock_error", 0.0)), seed=int(time.time()))
        elif not es_local(tc.endpoint):
            raise PermissionError("solo maestros locales desde la aplicación (ver «Probar conexión»)")
        else:
            teacher = OpenAICompatibleTeacher(tc, guardian)
        catalog = default_catalog(guardian)
        cur = cfg.ruta("configs/curriculum.yaml")
        if not cur.exists():
            cur = Path(__file__).resolve().parents[2] / "configs" / "curriculum.yaml"
        topics = load_curriculum(cur)
        folder = self.carpeta()
        folder.mkdir(parents=True, exist_ok=True)
        jsonl = folder / "episodios.jsonl"
        vistos = set()
        if jsonl.exists():
            for line in jsonl.open(encoding="utf-8"):
                try:
                    vistos.add(json.loads(line).get("sha"))
                except json.JSONDecodeError:
                    pass
        max_calls = int(m["max_consultas"])
        ronda = 0
        llamadas_previas = self.datos.get("consultas", 0)
        dia = time.strftime("%Y-%m-%d")
        def vence() -> bool:
            return self.detener_pedido() or (fin is not None and time.time() >= fin)

        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=max(1, int(paralelo)))

        def pedir(topic):
            try:
                reply = teacher.ask(curriculum_request(topic.name, topic.level, topic.language), purpose="app.episodios")
            except TeacherDenied:
                raise
            except (OSError, ValueError, KeyError) as e:
                return ("error", str(e))
            try:
                return ("ok", Episode(parse_episode(reply.text), reply.model))
            except EpisodeParseError as e:
                return ("invalido", str(e))

        while not vence() and (rondas is None or ronda < rondas):
            ronda += 1
            nuevos = 0
            for k, topic in enumerate(topics):
                if vence():
                    break
                self.progreso = (k + 1) / len(topics)
                self.detalle = f"ronda {ronda}: {topic.name}"
                eps: list[tuple[str, Episode | None, str]] = []
                n = min(int(m["episodios_por_tema"]), max(0, max_calls - llamadas_previas - teacher.calls))
                if n and not vence():
                    try:
                        resultados = list(pool.map(pedir, [topic] * n))
                    except TeacherDenied as e:
                        self.log(f"maestro denegado por el guardián: {e}")
                        pool.shutdown(wait=False)
                        return
                    for estado_r, val in resultados:
                        if estado_r == "error":
                            self.log(f"{topic.name}: el maestro no respondió ({val})")
                        elif estado_r == "invalido":
                            eps.append(("maestro", None, val))
                        else:
                            eps.append(("maestro", val, ""))
                fam = family_for(topic.name)
                if fam and int(m["generados_por_tema"]) > 0 and not (continuo and ronda > 1):
                    for ep in generate(fam, int(m["generados_por_tema"]), seed=hash((ronda, k, time.time())) % 10**9,
                                       level=topic.level, lang=topic.language):
                        eps.append(("generador", ep, ""))
                with jsonl.open("a", encoding="utf-8") as f:
                    for origen, ep, err in eps:
                        if ep is None:
                            rec = {"tema": topic.name, "origen": origen, "estado": "INVALIDO", "aceptado": False,
                                   "error": err}
                        else:
                            ep = verify_episode(catalog, ep)
                            st = ep.verification["status"]
                            ok = st == "RESPALDADO" or (st == "INFERIDO" and m.get("incluir_inferidos", True))
                            text = ep.training_text(tool_calls=True)
                            sha = hashlib.sha256(text.encode()).hexdigest()
                            if sha in vistos:
                                continue
                            vistos.add(sha)
                            rec = {"tema": topic.name, "origen": origen, "estado": st, "aceptado": ok,
                                   "peso": ep.verification["weight"], "sha": sha, "texto": text,
                                   "pasos": ep.verification["steps"]}
                        rec["fecha"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        if origen == "maestro":
                            rec["maestro"] = mid
                        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                        nuevos += 1
                self.datos = {"conteos": self.conteos(), "consultas": llamadas_previas + teacher.calls}
            self.reconstruir()
            self.log(f"ronda {ronda}: {nuevos} episodios nuevos; consultas al maestro: "
                     f"{llamadas_previas + teacher.calls}")
            if continuo and time.strftime("%Y-%m-%d") != dia:   # día nuevo: el límite vuelve a empezar
                dia, llamadas_previas = time.strftime("%Y-%m-%d"), -teacher.calls
            if llamadas_previas + teacher.calls >= max_calls:
                if not continuo:
                    self.log("se alcanzó el máximo de consultas al maestro")
                    break
                self.log("se alcanzó el máximo de consultas de hoy: sigue mañana")
                self.detalle = "límite de consultas de hoy alcanzado: sigue mañana"
                while not self.detener_pedido() and time.strftime("%Y-%m-%d") == dia:
                    self._stop.wait(60)
                continue
            if nuevos == 0:
                if not continuo:
                    self.log("ronda sin episodios nuevos: termina")
                    break
                self._stop.wait(60 if fin is None else max(0.0, min(60.0, fin - time.time())))
        pool.shutdown(wait=False)
        self.datos = {"conteos": self.conteos(), "consultas": llamadas_previas + teacher.calls}

    def _harvest_continuo(self) -> None:
        """Si ``hope.teacher.harvest`` tiene ``--continuo`` (rama maestro-local), lo usa directamente."""
        from hope.app.tareas import correr_modulo
        cfg_yaml = guardar_yaml(self.app.cfg)
        code = correr_modulo(self, "hope.teacher.harvest", ["--continuo", "--teacher", str(cfg_yaml),
                                                             "--out", str(self.carpeta())], self.app.cfg.root)
        if code != 0 and not self.detener_pedido():
            raise RuntimeError(f"hope.teacher.harvest --continuo terminó con código {code}")

    def reconstruir(self) -> dict:
        """Reescribe train.bin / val.bin con los episodios aceptados (partición determinista)."""
        folder = self.carpeta()
        p = folder / "episodios.jsonl"
        if not p.exists():
            return {}
        bufs = {"train": bytearray(), "val": bytearray()}
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("aceptado") and r.get("texto"):
                    # Validación chica pero suficiente (>1 KB) para medir BPB del maestro.
                    bufs[split_of(r["sha"], 3.0)] += r["texto"].encode("utf-8") + bytes([EOD])
        out = {}
        for split, buf in bufs.items():
            dest = folder / f"{split}.bin"
            tmp = folder / f"{split}.bin.tmp"
            tmp.write_bytes(bytes(buf))
            try:
                os.replace(tmp, dest)
            except PermissionError:                 # Windows: el entrenamiento lo tiene abierto
                self.log(f"{dest.name} está en uso por el entrenamiento: se actualiza en la próxima pausa")
            out[split] = len(buf)
        return out
