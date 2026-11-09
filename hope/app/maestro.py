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


def probar(cfg) -> dict:
    tc = teacher_config(cfg)
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
             "respaldados": 0, "inferidos": 0, "por_tema": {}}
        p = self.carpeta() / "episodios.jsonl"
        if not p.exists():
            return c
        st = p.stat()
        clave = (str(p), st.st_mtime_ns, st.st_size)
        tb = self.carpeta() / "train.bin"
        if getattr(self, "_cache_conteos", (None,))[0] == clave:     # la interfaz consulta cada 2 s
            return {**self._cache_conteos[1], "bytes_train": tb.stat().st_size if tb.exists() else 0}
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
                t = c["por_tema"].setdefault(r.get("tema", "?"), {"aceptados": 0, "rechazados": 0})
                t["aceptados" if r.get("aceptado") else "rechazados"] += 1
        c["bytes_train"] = (self.carpeta() / "train.bin").stat().st_size if (self.carpeta() / "train.bin").exists() else 0
        self._cache_conteos = (clave, c)
        return c

    def ejecutar(self, rondas: int | None = None) -> None:
        from hope.data.generators import family_for, generate
        from hope.teacher.cycle import load_curriculum
        from hope.teacher.episode import Episode, EpisodeParseError, curriculum_request, parse_episode
        from hope.tools.builtin import default_catalog
        from hope.tools.verify import verify_episode
        cfg = self.app.cfg
        m = cfg.data["maestro"]
        tc = teacher_config(cfg)
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
        while not self.detener_pedido() and (rondas is None or ronda < rondas):
            ronda += 1
            nuevos = 0
            for k, topic in enumerate(topics):
                if self.detener_pedido():
                    break
                self.progreso = (k + 1) / len(topics)
                self.detalle = f"ronda {ronda}: {topic.name}"
                eps: list[tuple[str, Episode | None, str]] = []
                for _ in range(int(m["episodios_por_tema"])):
                    if llamadas_previas + teacher.calls >= max_calls or self.detener_pedido():
                        break
                    try:
                        reply = teacher.ask(curriculum_request(topic.name, topic.level, topic.language),
                                            purpose="app.episodios")
                    except TeacherDenied as e:
                        self.log(f"maestro denegado por el guardián: {e}")
                        return
                    except (OSError, ValueError, KeyError) as e:
                        self.log(f"{topic.name}: el maestro no respondió ({e})")
                        continue
                    try:
                        eps.append(("maestro", Episode(parse_episode(reply.text), reply.model), ""))
                    except EpisodeParseError as e:
                        eps.append(("maestro", None, str(e)))
                fam = family_for(topic.name)
                if fam and int(m["generados_por_tema"]) > 0:
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
                        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                        nuevos += 1
                self.datos = {"conteos": self.conteos(), "consultas": llamadas_previas + teacher.calls}
            self.reconstruir()
            self.log(f"ronda {ronda}: {nuevos} episodios nuevos; consultas al maestro: "
                     f"{llamadas_previas + teacher.calls}")
            if llamadas_previas + teacher.calls >= max_calls:
                self.log("se alcanzó el máximo de consultas al maestro")
                break
            if nuevos == 0:
                self.log("ronda sin episodios nuevos: termina")
                break
        self.datos = {"conteos": self.conteos(), "consultas": llamadas_previas + teacher.calls}

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
