"""Aplicación con interfaz (hope.app): API JSON, pasos, entrenamiento con pausa y retoma, maestro,
paso a la A770 y guardián. El servidor corre en un puerto libre sobre una carpeta temporal."""

import http.client
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import pytest
import yaml

from guardian.core import Action

REPO = Path(__file__).resolve().parents[1]
TEXTO = ("La lluvia cae sobre el campo verde y los ríos llevan el agua hasta el mar. " * 8 + "\n\n")


def _esperar(cond, timeout=120.0, paso=0.25):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(paso)
    raise AssertionError("tiempo agotado esperando la condición")


@pytest.fixture()
def raiz(tmp_path):
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    # Corpus mínimo para el plan de juguete (dos fuentes con validación).
    for name, text in (("wiki_es", TEXTO), ("codigo", "def suma(a, b):\n    return a + b\n" * 40)):
        d = tmp_path / "data" / "pre" / name
        d.mkdir(parents=True)
        (d / "train.bin").write_bytes((text * 40).encode())
        (d / "val.bin").write_bytes((text * 4).encode())
    plan = {"salida": "runs/hope", "precision": "fp32-8bit", "optimizador": "adamw", "lr": 2e-3, "warmup": 5,
            "secuencia": 64, "checkpointing": False, "compilar": False, "checkpoint_min": 0.02,
            "etapas": [{"nombre": "1_lenguaje", "preset": "hn1-tiny", "horas": 1, "lote": 4,
                        "fuentes": {"wiki_es": "data/pre/wiki_es:0.6", "codigo": "data/pre/codigo:0.4"}},
                       {"nombre": "2_maestro_local", "horas": 1, "lote": 4,
                        "fuentes": {"wiki_es": "data/pre/wiki_es:0.6", "maestro": "data/maestro:0.4"}}]}
    (tmp_path / "configs" / "plan_juguete.yaml").write_text(yaml.safe_dump(plan), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def servidor(raiz):
    from hope.app.server import serve
    app, httpd, th = serve(raiz, port=0, retomar=False)
    yield app, httpd.server_address[1]
    if app.entrenamiento.corriendo():
        app.entrenamiento.detener_ya()
    app.cerrar()
    httpd.shutdown()


def _pedir(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    h = {"Host": f"127.0.0.1:{port}"}
    h.update(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    raw = r.read()
    c.close()
    try:
        return r.status, json.loads(raw)
    except json.JSONDecodeError:
        return r.status, raw.decode()


def accion(port, nombre, **params):
    return _pedir(port, "POST", "/api/accion", {"accion": nombre, "params": params}, {"X-Hope": "1"})[1]


def estado(port):
    return _pedir(port, "GET", "/api/estado")[1]


# ---------------------------------------------------------------------------
def test_api_estado_pagina_y_seguridad(servidor):
    app, port = servidor
    code, s = _pedir(port, "GET", "/api/estado")
    assert code == 200
    assert set(s["pasos"]) == {"calibracion", "datos", "lenguaje", "maestro", "migrar", "guardian"}
    assert s["pasos"]["calibracion"]["estado"] == "inactiva" and not s["pasos"]["datos"]["hecho"]
    assert s["entrenamiento"]["estado"] == "detenido"
    code, html = _pedir(port, "GET", "/")
    assert code == 200 and "Calibración" in html and "/app.js" in html
    assert _pedir(port, "GET", "/app.js")[0] == 200
    assert _pedir(port, "GET", "/../server.py")[0] == 404
    # Otro Host (DNS rebinding) y POST sin el encabezado propio: rechazados.
    assert _pedir(port, "GET", "/api/estado", headers={"Host": "evil.example:80"})[0] == 403
    assert _pedir(port, "POST", "/api/accion", {"accion": "entrenar_iniciar"})[0] == 403
    # Solo acciones de la lista fija; solo claves editables de la configuración.
    code, r = _pedir(port, "POST", "/api/accion", {"accion": "ejecutar", "params": {"cmd": "rm -rf /"}}, {"X-Hope": "1"})
    assert code == 400 and "desconocida" in r["mensaje"]
    code, r = _pedir(port, "POST", "/api/accion", {"accion": "guardar_config", "params": {"cambios": {"calibracion": 1}}},
                     {"X-Hope": "1"})
    assert code == 400
    assert httpd_escucha_solo_local(port)


def httpd_escucha_solo_local(port):
    import socket
    s = socket.socket()
    s.settimeout(1)
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip.startswith("127."):
            return True
        return s.connect_ex((ip, port)) != 0
    finally:
        s.close()


def test_config_se_guarda_en_app_json(servidor, raiz):
    app, port = servidor
    r = accion(port, "guardar_config", cambios={"usuario": "adrian", "plan": "plan_gtx1060_200m.yaml",
                                                "datos.presupuesto_gb": "12.5", "datos.repos": "a\nb\n"})
    assert r["ok"]
    data = json.loads((raiz / "runs" / "app.json").read_text(encoding="utf-8"))
    assert data["usuario"] == "adrian" and data["plan"] == "plan_gtx1060_200m.yaml"
    assert data["datos"]["presupuesto_gb"] == 12.5 and data["datos"]["repos"] == ["a", "b"]
    assert accion(port, "guardar_config", cambios={"plan": "../../etc/passwd"})["ok"] is False
    # Al reabrir la aplicación la configuración sigue ahí.
    from hope.app.config import AppConfig
    assert AppConfig(raiz).get("datos.presupuesto_gb") == 12.5
    # El plan efectivo apunta a la carpeta de datos elegida.
    from hope.app.config import plan_efectivo
    app.cfg.set("datos.carpeta", "D:/hope_datos")
    rel, plan = plan_efectivo(app.cfg)
    assert plan["etapas"][0]["fuentes"]["wiki_es"].startswith("D:/hope_datos/pre/wiki_es:")


def test_calibracion_rapida_en_cpu(servidor):
    app, port = servidor
    assert accion(port, "calibrar_iniciar", rapido=True)["ok"]
    s = _esperar(lambda: (lambda s: s if s["calibracion"]["estado"] in ("hecha", "error") else None)(estado(port)))
    assert s["calibracion"]["estado"] == "hecha", s["calibracion"]
    res = s["calibracion"]["resultado"]
    assert res["plan_recomendado"] and {r["preset"] for r in res["recomendacion"]} >= {"hn1-200m", "hn1-500m"}
    assert s["pasos"]["calibracion"]["hecho"]


def test_datos_con_carpetas_propias(servidor, raiz):
    app, port = servidor
    libros = raiz / "libros"
    libros.mkdir()
    frases = [f"En el capítulo {i} se explica cómo medir la tensión de la fuente número {i * 7} con un "
              f"multímetro, anotando el valor y comparándolo con el esperado de {i * 3} voltios." for i in range(12)]
    (libros / "manual.txt").write_text("\n".join(frases), encoding="utf-8")
    (libros / "basura.txt").write_text("x", encoding="utf-8")
    codigo = raiz / "micodigo"
    codigo.mkdir()
    (codigo / "a.py").write_text("def f(x):\n    return x * 2\n" * 30, encoding="utf-8")
    r = accion(port, "guardar_config", cambios={
        "datos.carpeta": "datos_app", "datos.presupuesto_gb": 0.001, "datos.incluir_stdlib": False,
        "datos.carpetas_codigo": [str(codigo)], "datos.propias": [str(libros)],
        "datos.reparto": {"wiki_es": 0, "wiki_en": 0, "codigo": 0.5, "repos": 0, "propias": 0.5}})
    assert r["ok"]
    assert accion(port, "datos_iniciar")["ok"]
    s = _esperar(lambda: (lambda s: s if s["datos"]["estado"] in ("hecha", "error") else None)(estado(port)))
    assert s["datos"]["estado"] == "hecha", s["datos"]
    f = s["datos"]["fuentes"]
    assert f["codigo"]["hecha"] and f["propias"]["hecha"] and s["pasos"]["datos"]["hecho"]
    assert f["propias"]["resumen"]["docs"] == 1 and f["propias"]["resumen"]["descartados_reglas"] == 1
    assert (raiz / "datos_app" / "pre" / "propias" / "train.bin").exists() or \
        (raiz / "datos_app" / "pre" / "propias" / "val.bin").stat().st_size > 0


def test_datos_repos_excluye_coautoria_ia(servidor, raiz):
    app, port = servidor
    repo = raiz / "mirepo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=yo", "-c", "user.email=yo@x"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "humano.py").write_text("def humano():\n    return 'hecho a mano'\n" * 20, encoding="utf-8")
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-qm", "arreglo del cálculo"], check=True)
    (repo / "ia.py").write_text("def ia():\n    return 'hecho con IA'\n" * 20, encoding="utf-8")
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-qm", "agrega ia\n\nCo-Authored-By: Claude <noreply@anthropic.com>"], check=True)
    accion(port, "guardar_config", cambios={
        "datos.presupuesto_gb": 0.001, "datos.repos": [str(repo)], "datos.incluir_stdlib": False,
        "datos.reparto": {"wiki_es": 0, "wiki_en": 0, "codigo": 0, "repos": 1.0, "propias": 0}})
    accion(port, "datos_iniciar")
    s = _esperar(lambda: (lambda s: s if s["datos"]["estado"] in ("hecha", "error") else None)(estado(port)))
    assert s["datos"]["estado"] == "hecha", s["datos"]
    out = raiz / "data" / "pre" / "repos"
    todo = b"".join((out / n / f).read_bytes() for n in ("repos_codigo", "repos_historial")
                    for f in ("train.bin", "val.bin") if (out / n / f).exists())
    assert b"hecho a mano" in todo and b"hecho con IA" not in todo


def _registro(raiz):
    p = raiz / "runs" / "hope" / "registro.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_entrenar_pausar_y_retomar_plan_de_juguete(servidor, raiz):
    app, port = servidor
    assert accion(port, "guardar_config", cambios={"plan": "plan_juguete.yaml",
                                                   "entrenamiento.muestras_auto": False})["ok"]
    assert accion(port, "entrenar_iniciar")["ok"]
    assert estado(port)["entrenamiento"]["estado"] == "en_curso"
    assert accion(port, "entrenar_iniciar")["ok"] is False           # no se lanza dos veces
    _esperar(lambda: any("paso" in r and "bpb_val" in r for r in _registro(raiz)), timeout=180)
    assert accion(port, "entrenar_pausar")["ok"]
    s = _esperar(lambda: (lambda s: s if s["entrenamiento"]["estado"] == "pausado" else None)(estado(port)), timeout=180)
    assert (raiz / "runs" / "hope" / "DETENER").exists()
    reg = _registro(raiz)
    assert reg[-1].get("evento") == "pausa"
    paso_pausa = max(r["paso"] for r in reg if "paso" in r)
    n_antes = len(reg)
    e = s["entrenamiento"]
    assert e["serie"] and "wiki_es" in e["serie"][-1]["bpb"] and e["etapa_actual"] == "1_lenguaje"
    assert e["etapas"][0]["horas"] > 0 and e["eta_horas"] < 1
    assert json.loads((raiz / "runs" / "hope" / "estado.json").read_text())["ultimo_checkpoint"]
    # Reanudar: sigue desde el checkpoint (el paso no vuelve a cero).
    assert accion(port, "entrenar_iniciar")["ok"]
    assert not (raiz / "runs" / "hope" / "DETENER").exists()
    nuevo = _esperar(lambda: next((r for r in _registro(raiz)[n_antes:] if "bpb_val" in r), None), timeout=180)
    assert nuevo["paso"] > paso_pausa and nuevo["pasos"] < nuevo["paso"]
    # Muestra de generación con el modelo ligero.
    assert accion(port, "muestra_generar")["ok"]
    _esperar(lambda: estado(port)["entrenamiento"]["muestras"], timeout=180)
    m = estado(port)["entrenamiento"]["muestras"]
    assert [x["idioma"] for x in m["muestras"]] == ["es", "en", "código"]
    assert accion(port, "entrenar_detener_ya")["ok"] is False          # sin confirmar no hace nada
    assert accion(port, "entrenar_detener_ya", confirmar=True)["ok"]
    assert not app.entrenamiento.corriendo()


def test_retoma_automatica_al_caerse(servidor, raiz, monkeypatch):
    """Si el proceso termina con error, la aplicación lo relanza sola."""
    import hope.app.entrenamiento as ent
    app, port = servidor
    accion(port, "guardar_config", cambios={"plan": "plan_juguete.yaml", "entrenamiento.muestras_auto": False})
    accion(port, "entrenar_iniciar")
    _esperar(lambda: _registro(raiz), timeout=180)
    pid1 = app.entrenamiento.pid
    with app.entrenamiento._lock:
        app.entrenamiento.proc.kill()                                    # «se corta la luz»
        app.entrenamiento.proc.wait()
        app.entrenamiento._termino()
    assert app.entrenamiento.estado == "reintentando"
    app.entrenamiento._reintentar_en = 0                                 # sin esperar los 30 s
    _esperar(lambda: app.entrenamiento.pid != pid1 and app.entrenamiento.corriendo(), timeout=30)
    assert app.entrenamiento.reintentos == 1


def test_maestro_mock_y_etapa_2(servidor, raiz):
    app, port = servidor
    accion(port, "guardar_config", cambios={"plan": "plan_juguete.yaml", "maestro.model": "mock",
                                            "maestro.episodios_por_tema": 2, "maestro.generados_por_tema": 1,
                                            "maestro.mock_error": 0.5, "maestro.max_consultas": 20})
    assert accion(port, "maestro_probar")["ok"]
    assert accion(port, "maestro_iniciar", rondas=1)["ok"]
    s = _esperar(lambda: (lambda s: s if s["maestro"]["estado"] in ("hecha", "error") else None)(estado(port)),
                 timeout=300)
    assert s["maestro"]["estado"] == "hecha", s["maestro"]
    c = s["maestro"]["conteos"]
    assert c["aceptados"] > 0 and c["rechazados"] > 0 and c["del_maestro"] > 0
    assert (raiz / "data" / "maestro" / "train.bin").stat().st_size > 0
    # Los rechazados no entran al entrenamiento.
    eps = [json.loads(x) for x in (raiz / "data" / "maestro" / "episodios.jsonl").read_text().splitlines()]
    train = (raiz / "data" / "maestro" / "train.bin").read_bytes() + (raiz / "data" / "maestro" / "val.bin").read_bytes()
    for e in eps:
        if e.get("estado") == "REFUTADO":
            assert e["texto"].encode() not in train
    # Un maestro remoto no se usa desde la aplicación.
    accion(port, "guardar_config", cambios={"maestro.model": "qwen", "maestro.endpoint": "http://example.org/v1"})
    assert accion(port, "maestro_probar")["ok"] is False
    # Pasar a la etapa 2 sin entrenar la 1 (con confirmación), sin lanzar.
    assert accion(port, "maestro_pasar_etapa")["ok"] is False
    assert accion(port, "maestro_pasar_etapa", confirmar=True, lanzar=False)["ok"]
    est = json.loads((raiz / "runs" / "hope" / "estado.json").read_text())
    assert est["etapas"]["1_lenguaje"]["hecha"]
    assert estado(port)["entrenamiento"]["etapa_actual"] == "2_maestro_local"


def test_exportar_e_importar_en_la_a770(servidor, raiz, tmp_path_factory):
    app, port = servidor
    hope = raiz / "runs" / "hope"
    (hope / "1_lenguaje").mkdir(parents=True)
    (hope / "1_lenguaje" / "model.pt").write_bytes(b"\x00" * 1000)
    (hope / "estado.json").write_text(json.dumps({"etapas": {"1_lenguaje": {"horas": 12.0, "hecha": False}},
                                                  "ultimo_checkpoint": "runs\\hope\\1_lenguaje"}))
    (hope / "registro.jsonl").write_text('{"etapa": "1_lenguaje", "horas": 12.0}\n')
    (hope / "DETENER").write_text("x")
    (raiz / "data" / "maestro").mkdir(parents=True)
    (raiz / "data" / "maestro" / "train.bin").write_bytes(b"episodio")
    assert accion(port, "exportar_iniciar")["ok"]
    s = _esperar(lambda: (lambda s: s if s["migrar"]["exportar"]["estado"] in ("hecha", "error") else None)(estado(port)))
    assert s["migrar"]["exportar"]["estado"] == "hecha", s["migrar"]["exportar"]
    zpath = Path(s["migrar"]["exportar"]["datos"]["zip"])
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
    assert {"runs/hope/1_lenguaje/model.pt", "runs/hope/estado.json", "runs/hope/hope_continuacion.json",
            "configs/plan_a770_continuacion.yaml", "LEEME_A770.txt", "data/maestro/train.bin"} <= names
    assert "runs/hope/DETENER" not in names
    # Agregamos una entrada maliciosa: no debe salir de la carpeta.
    with zipfile.ZipFile(zpath, "a") as z:
        z.writestr("runs/hope/../../fuera.txt", "x")
        z.writestr("configs/guardian.yaml", "pisado")

    # «Otra PC»: carpeta nueva con los configs del repositorio.
    otra = tmp_path_factory.mktemp("a770")
    shutil.copytree(REPO / "configs", otra / "configs")
    from hope.app.server import serve
    app2, httpd2, _ = serve(otra, port=0, retomar=False)
    try:
        p2 = httpd2.server_address[1]
        assert not estado(p2)["migrar"]["continuacion"]["es_continuacion"]
        assert accion(p2, "importar_iniciar", zip=str(zpath))["ok"]
        s2 = _esperar(lambda: (lambda s: s if s["migrar"]["importar"]["estado"] in ("hecha", "error") else None)(estado(p2)))
        assert s2["migrar"]["importar"]["estado"] == "hecha", s2["migrar"]["importar"]
        assert (otra / "runs" / "hope" / "1_lenguaje" / "model.pt").exists()
        assert (otra / "data" / "maestro" / "train.bin").read_bytes() == b"episodio"
        assert not (otra.parent / "fuera.txt").exists() and not (otra / "fuera.txt").exists()
        assert (otra / "configs" / "guardian.yaml").read_text() != "pisado"
        s2 = estado(p2)
        assert s2["migrar"]["continuacion"]["es_continuacion"] and s2["config"]["plan"] == "plan_a770_continuacion.yaml"
        from hope.app.config import plan_efectivo
        _, plan = plan_efectivo(app2.cfg)
        assert plan["desde"] == "runs/hope/1_lenguaje" and plan["precision"] == "bf16-8bit"
    finally:
        app2.cerrar()
        httpd2.shutdown()


def test_guardian_aprobar_y_techo(servidor, raiz):
    app, port = servidor
    g = app.guardian()
    d = g.decide(Action("network.access", {"host": "example.org"}))
    rid = d.pending_request
    assert rid
    est = raiz / "runs" / "hope_a770"
    est.mkdir(parents=True)
    (est / "estado.json").write_text(json.dumps({"etapas": {"3_crecer_500m": {"horas": 0, "hecha": False,
                                                                               "solicitud": rid}}}))
    s = estado(port)
    assert rid in s["guardian"]["pendientes"] and s["pasos"]["guardian"]["estado"] == "pendientes"
    assert accion(port, "guardian_aprobar", id=rid)["ok"] is False            # falta el nombre
    accion(port, "guardar_config", cambios={"usuario": "adrian"})
    assert accion(port, "guardian_aprobar", id=rid)["ok"]
    assert json.loads((est / "estado.json").read_text())["etapas"]["3_crecer_500m"]["aprobacion"] == rid
    assert not estado(port)["guardian"]["pendientes"]
    # Subir el techo exige la frase exacta.
    r = accion(port, "guardian_techo", valor="1B", confirmacion="si")
    assert not r["ok"] and r["confirmacion"] == "AUTORIZO 1B"
    r = accion(port, "guardian_techo", valor="1B", confirmacion="AUTORIZO 1B")
    assert r["ok"]
    assert app.guardian().budget().max_total_params == 10**9
    ok, _ = app.guardian().audit.verify()
    assert ok
