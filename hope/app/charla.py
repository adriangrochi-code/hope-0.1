"""Proceso de «Chatear»: Hope responde con una COPIA del último checkpoint, sin tocar el entrenamiento.

Corre aparte (``python -m hope.app.charla``) y habla con la aplicación por líneas JSON en stdin/stdout:

* ``{"op": "cargar", "modelo": ruta, "bits": 8|4|0, "dispositivo": "cpu"|"cuda"|"xpu", "etiqueta": "paso 1200"}``
  — carga la copia con ``load_light`` y la cuantiza (``hope.core.fast.quantize``) para ocupar poco.
  Por defecto en la CPU con 2 hilos: no le quita memoria ni cómputo a la GPU que entrena.
* ``{"op": "decir", "conv": id, "texto": …, "max_bytes": 300, "herramientas": true, "contexto": true}``
  — un turno de conversación con caché (``hope.core.batched.Conversation``). Con ``contexto`` usa la base
  de conocimiento y el cuaderno (solo lectura); con ``herramientas``, las del catálogo a través del
  guardián (las de la web, solo lectura según ``configs/web.yaml``). Devuelve la respuesta y la traza
  («cómo lo pensó»: herramientas llamadas, contexto usado, verificación, velocidad).
* ``{"op": "comparar", "texto": …, "modelo_b": ruta}`` — la misma pregunta a otro checkpoint.
* ``{"op": "feedback", "pregunta", "respuesta", "voto": "bien"|"mal"|"corregir", "correccion", "usados"}``
  — 👍 suma uso a lo que ayudó y guarda la respuesta como ejemplo; 👎 baja la confianza; «corregir» guarda la
  corrección como ejemplo de entrenamiento y como lección (verificada con herramientas si es una cuenta).
* ``{"op": "reiniciar", "conv": id}`` · ``{"op": "salir"}``.

Todo turno y toda valoración quedan en PESAM (``<salida>/pesam_chat.jsonl``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

PREFIJO_U, PREFIJO_H = "Usuario: ", "Hope: "
CORTE = "\nUsuario:"


class Charla:
    def __init__(self, salida: Path, datos: Path, root: Path):
        self.salida, self.datos, self.root = Path(salida), Path(datos), Path(root)
        self.model = None
        self.modelo_b = (None, None)                 # (ruta, modelo) para comparar
        self.info: dict = {}
        self.convs: dict = {}
        self._kb = self._catalog = self._pesam = self._runner = None

    # --- recursos (perezosos) ----------------------------------------------------
    def kb(self):
        if self._kb is None:
            from hope.memory.knowledge import KnowledgeBase
            from hope.memory.store import MemoryStore
            self.salida.mkdir(parents=True, exist_ok=True)
            self._kb = KnowledgeBase(MemoryStore(self.salida / "conocimiento.sqlite"))
        return self._kb

    def pesam(self):
        if self._pesam is None:
            from hope.memory.pesam import ExperienceLog
            self._pesam = ExperienceLog(self.salida / "pesam_chat.jsonl")
        return self._pesam

    def guardian(self):
        from hope.app.guardian_app import cargar
        return cargar(self.root)

    def catalog(self):
        if self._catalog is None:
            from hope.tools.builtin import default_catalog
            g = self.guardian()
            self._catalog = default_catalog(g)
            try:                                       # web: solo lectura, con los permisos de configs/web.yaml
                from hope.tools import web
                self._web = web.load(g)
                web.register(self._catalog, self._web)
            except Exception:                          # noqa: BLE001 — sin web, el chat sigue
                self._web = None
        return self._catalog

    # --- operaciones ---------------------------------------------------------------------
    def cargar(self, modelo: str, bits: int = 8, dispositivo: str = "cpu", etiqueta: str = "") -> dict:
        t0 = time.time()
        m = self._abrir(modelo, bits, dispositivo)
        self.model = m
        self.convs = {}                                # las cachés eran del modelo anterior
        params = sum(p.numel() for p in m.parameters()) + sum(b.numel() for n, b in m.named_buffers()
                                                                if n.endswith("qweight"))
        self.info = {"modelo": modelo, "bits": bits, "dispositivo": dispositivo, "etiqueta": etiqueta,
                     "parametros": params, "segundos_carga": round(time.time() - t0, 1),
                     "mtime": Path(modelo).stat().st_mtime}
        return {"ok": True, **self.info}

    def _abrir(self, modelo: str, bits: int, dispositivo: str):
        from hope.train.run import load_light
        m = load_light(modelo, "cpu")
        if bits in (4, 8):
            from hope.core.fast import quantize
            quantize(m, bits=bits)
        if dispositivo != "cpu":
            m = m.to(dispositivo)
        return m.eval()

    def _prompt(self, texto: str, contexto: bool) -> tuple[str, list[str]]:
        usados, extra = [], ""
        if contexto:
            try:
                ctx = self.kb().context("chat", texto)
                if ctx.text:
                    extra, usados = ctx.text + "\n\n", ctx.used
            except Exception:                          # noqa: BLE001
                pass
        return f"{extra}{PREFIJO_U}{texto}\n{PREFIJO_H}", usados

    def decir(self, conv: str, texto: str, max_bytes: int = 300, herramientas: bool = True, contexto: bool = True,
              temperatura: float = 0.0, objetivo_tok_s: float | None = None) -> dict:
        if self.model is None:
            return {"ok": False, "mensaje": "todavía no hay modelo cargado"}
        from hope.core.batched import Conversation
        c = self.convs.get(conv)
        if c is None:
            c = self.convs[conv] = Conversation(self.model)
        prompt, usados = self._prompt(texto, contexto)
        t0 = time.time()
        traza: dict = {"contexto_usado": len(usados), "herramientas": [], "verificacion": None,
                       "consejo": "no se usa en el chat (respuesta directa)"}
        if herramientas:
            from hope.runtime.toolcalls import ToolRunner, generate_with_tools
            runner = ToolRunner(self.catalog(), memory=self.kb().store, web=getattr(self, "_web", None))

            def gen(text, max_new, **kw):              # adapta Conversation al generador de herramientas
                tb = text.encode("utf-8")
                hist = bytes(c.history)
                if not tb.startswith(hist):
                    c.reset()
                    hist = b""
                kw.pop("stop_after_line", None)
                return c.say(tb[len(hist):], max_new, **{k: v for k, v in kw.items() if k in ("temperature", "stop_seq")})
            base = bytes(c.history).decode("utf-8", errors="replace")
            out = generate_with_tools(self.model, base + prompt, runner, max_new=max_bytes, stop_after_line=None,
                                      temperature=temperatura, generate=gen)
            traza["herramientas"] = [{"nombre": x.name, "entrada": x.input, "salida": x.output[:300], "ok": x.ok}
                                     for x in runner.calls]
        else:
            out = c.say(prompt, max_bytes, temperature=temperatura, stop_seq=CORTE.encode()).decode(
                "utf-8", errors="replace")
        respuesta = out.split(CORTE)[0].strip()
        seg = max(time.time() - t0, 1e-6)
        nbytes = len(out.encode("utf-8"))
        traza.update(segundos=round(seg, 2), bytes_s=round(nbytes / seg, 1), tokens_s=round(nbytes / 4 / seg, 1))
        if objetivo_tok_s:
            traza["objetivo_tok_s"] = objetivo_tok_s
            traza["cumple_objetivo"] = traza["tokens_s"] >= objetivo_tok_s
        traza["verificacion"] = verificar(texto, respuesta, self.catalog() if herramientas else None)
        self.pesam().record("chat_turno", pregunta=texto, respuesta=respuesta, modelo=self.info.get("etiqueta"),
                            traza=traza)
        return {"ok": True, "respuesta": respuesta, "traza": traza, "usados": usados,
                "modelo": self.info.get("etiqueta")}

    def comparar(self, texto: str, modelo_b: str, max_bytes: int = 300, bits: int = 8) -> dict:
        if self.model is None:
            return {"ok": False, "mensaje": "todavía no hay modelo cargado"}
        ruta, mb = self.modelo_b
        if ruta != modelo_b or mb is None:
            mb = self._abrir(modelo_b, bits, "cpu")
            self.modelo_b = (modelo_b, mb)
        prompt, _ = self._prompt(texto, False)
        out = {}
        for nombre, m in (("actual", self.model), ("otro", mb)):
            t0 = time.time()
            r = m.generate(prompt, max_new=max_bytes, stop_seq=CORTE.encode()).decode("utf-8", errors="replace")
            out[nombre] = {"respuesta": r.split(CORTE)[0].strip(), "segundos": round(time.time() - t0, 2)}
        return {"ok": True, **out}

    def feedback(self, pregunta: str, respuesta: str, voto: str, correccion: str = "", usados=None) -> dict:
        kb = self.kb()
        usados = list(usados or [])
        res = {"ok": True}
        if voto == "bien":
            if usados:
                kb.feedback(usados, True)
            kb.learn_solution("chat", pregunta, respuesta, None, verified=False, source="usuario (👍)")
            ejemplo(self.datos, pregunta, respuesta, "aprobado", verificado=None)
            res["mensaje"] = "guardado como buen ejemplo"
        elif voto == "mal":
            if usados:
                kb.feedback(usados, False)
            res["mensaje"] = "anotado: esa respuesta no sirvió"
        elif voto == "corregir":
            if not correccion.strip():
                return {"ok": False, "mensaje": "falta la corrección"}
            v = verificar(pregunta, correccion, self.catalog())
            if v and v.get("ok") is False:
                return {"ok": False, "mensaje": f"la corrección no pasa la verificación: {v['detalle']}", "verificacion": v}
            kb.learn_lesson("chat", pregunta, respuesta, [f"la respuesta correcta es: {correccion.strip()[:300]}"],
                            source="usuario")
            kb.learn_solution("chat", pregunta, correccion, None, verified=bool(v and v.get("ok")),
                              source="corrección del usuario")
            ejemplo(self.datos, pregunta, correccion, "correccion", verificado=v.get("ok") if v else None)
            res["mensaje"] = "corrección guardada como ejemplo y lección" + (" (verificada)" if v and v.get("ok") else "")
            res["verificacion"] = v
        else:
            return {"ok": False, "mensaje": "voto desconocido"}
        self.pesam().record("chat_feedback", pregunta=pregunta, respuesta=respuesta, voto=voto, correccion=correccion)
        return res


# ---------------------------------------------------------------------------
EXPR = re.compile(r"(?:cu[aá]nto (?:es|da|vale)|calcul[aá]|how much is|what is)\s*([-+*/()\d\s.,^]+)\??", re.I)


def verificar(pregunta: str, respuesta: str, catalog) -> dict | None:
    """Si la pregunta es una cuenta («¿cuánto es 23 * 4?»), se verifica la respuesta con la herramienta."""
    if catalog is None:
        return None
    m = EXPR.search(pregunta)
    if not m:
        return None
    expr = m.group(1).strip().replace("^", "**").replace(",", ".")
    if not re.search(r"\d\s*[-+*/]", expr):
        return None
    try:
        from hope.tools.builtin import numbers_equal
        valor = catalog.call("hte_arith_v1", expr=expr)
        nums = re.findall(r"-?\d+(?:[.,]\d+)?", respuesta)
        ok = any(numbers_equal(valor, n.replace(",", ".")) for n in nums)
        return {"ok": ok, "detalle": f"{expr} = {valor}", "herramienta": "hte_arith_v1"}
    except Exception as e:                             # noqa: BLE001
        return {"ok": None, "detalle": f"no se pudo verificar ({e})"}


def ejemplo(datos: Path, pregunta: str, respuesta: str, tipo: str, verificado) -> None:
    """Ejemplo de entrenamiento: ``<datos>/charla/ejemplos.jsonl`` y ``train.bin``/``val.bin`` (fuente «charla»)."""
    from hope.data.corpus import EOD, split_of
    d = Path(datos) / "charla"
    d.mkdir(parents=True, exist_ok=True)
    texto = f"{PREFIJO_U}{pregunta.strip()}\n{PREFIJO_H}{respuesta.strip()}"
    sha = hashlib.sha256(texto.encode()).hexdigest()
    with (d / "ejemplos.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"tipo": tipo, "verificado": verificado, "texto": texto, "sha": sha,
                            "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False) + "\n")
    bufs = {"train": bytearray(), "val": bytearray()}
    for line in (d / "ejemplos.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("verificado") is not False:
            bufs[split_of(r["sha"], 5.0)] += r["texto"].encode("utf-8") + bytes([EOD])
    for split, buf in bufs.items():
        tmp = d / f"{split}.bin.tmp"
        tmp.write_bytes(bytes(buf))
        try:
            tmp.replace(d / f"{split}.bin")
        except PermissionError:
            pass


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--salida", required=True)
    ap.add_argument("--datos", required=True)
    ap.add_argument("--raiz", default=".")
    ap.add_argument("--hilos", type=int, default=2)
    args = ap.parse_args(argv)
    import torch
    torch.set_num_threads(args.hilos)
    ch = Charla(Path(args.salida), Path(args.datos), Path(args.raiz))
    ops = {"cargar": ch.cargar, "decir": ch.decir, "comparar": ch.comparar, "feedback": ch.feedback}
    print(json.dumps({"listo": True}), flush=True)
    for line in sys.stdin:
        try:
            msg = json.loads(line)
            op = msg.pop("op")
            rid = msg.pop("rid", None)
            if op == "salir":
                break
            if op == "reiniciar":
                ch.convs.pop(msg.get("conv"), None)
                res = {"ok": True}
            elif op in ops:
                res = ops[op](**msg)
            else:
                res = {"ok": False, "mensaje": f"operación desconocida: {op}"}
        except Exception as e:                         # noqa: BLE001 — el proceso sigue vivo
            rid = locals().get("rid")
            res = {"ok": False, "mensaje": f"{type(e).__name__}: {e}"}
        print(json.dumps({"rid": rid, **res}, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
