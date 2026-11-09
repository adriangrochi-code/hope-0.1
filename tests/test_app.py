"""Aplicación con interfaz (hope.app): API JSON, pasos, entrenamiento con pausa y retoma, maestro,
paso a la A770 y guardián. El servidor corre en un puerto libre sobre una carpeta temporal."""

import http.client
import os
import time
import json
import shutil
import subprocess
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
    assert set(s["pasos"]) == {"calibracion", "datos", "lenguaje", "maestro", "migrar", "guardian", "cuantizar",
                                "diarios", "automatico"}
    assert s["diarios"]["estado"] == "inactiva" and any(f["fuente"] == "arxiv" for f in s["diarios"]["fuentes"])
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
    r = accion(port, "guardar_config", cambios={"usuario": "adrian", "plan": "plan_gtx1060_500m_vision.yaml",
                                                "datos.presupuesto_gb": "12.5", "datos.repos": "a\nb\n"})
    assert r["ok"]
    data = json.loads((raiz / "runs" / "app.json").read_text(encoding="utf-8"))
    assert data["usuario"] == "adrian" and data["plan"] == "plan_gtx1060_500m_vision.yaml"
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
    import hope.app.entrenamiento as ent  # noqa: F401  (carga el módulo antes de parchearlo)
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


def test_exportar_cuantizado_4_bits(servidor, raiz):
    from hope.app.cuantizar import cargar, perdida
    from hope.core.hn1 import HN1Model, preset
    from hope.train.run import export_light
    app, port = servidor
    assert accion(port, "cuantizar_iniciar", bits=4)["ok"]            # sin modelo todavía: termina con error
    s = _esperar(lambda: (lambda s: s if s["cuantizar"]["estado"] in ("hecha", "error") else None)(estado(port)))
    assert s["cuantizar"]["estado"] == "error" and "ligero" in s["cuantizar"]["error"]
    export_light(HN1Model(preset("hn1-tiny")), raiz / "runs" / "hope" / "ligero.pt")
    assert estado(port)["cuantizar"]["modelo"].endswith("ligero.pt")
    assert accion(port, "cuantizar_iniciar", bits=4)["ok"]
    s = _esperar(lambda: (lambda s: s if s["cuantizar"]["estado"] in ("hecha", "error") else None)(estado(port)))
    assert s["cuantizar"]["estado"] == "hecha", s["cuantizar"]
    x = s["cuantizar"]["exportados"][0]
    assert x["bits"] == 4 and x["mb_pesos_lineales_despues"] < x["mb_pesos_lineales_antes"] / 3
    # La pérdida informada se mide sobre el modelo tal como queda en el archivo (lo no cuantizado en bf16,
    # las escalas en float32): al reabrir da lo mismo.
    assert abs(x["perdida_al_reabrir"] - x["perdida_cuantizado"]) < 2e-4 and s["pasos"]["cuantizar"]["hecho"]
    m = cargar(x["archivo"])
    assert abs(perdida(m) - x["perdida_al_reabrir"]) < 1e-3


# ---------------------------------------------------------------------------
# Configuración (hope/app/ajustes.py)
# ---------------------------------------------------------------------------
def _plan_base(raiz):
    p = raiz / "configs" / "plan_juguete.yaml"
    p.write_text("# plan de prueba\nsalida: runs/hope\nprecision: fp32-8bit   # Pascal\noptimizador: adamw\n"
                 "secuencia: 64\ncompilar: false\ncheckpointing: false\ncheckpoint_min: 0.02\nlr: 0.002\nwarmup: 5\n"
                 "etapas:\n  - nombre: 1_lenguaje   # la primera\n    preset: hn1-tiny\n    horas: 1   # una hora\n"
                 "    lote: 4\n    fuentes:\n      wiki_es: data/pre/wiki_es:0.6\n      codigo: data/pre/codigo:0.4\n"
                 "  - nombre: 2_maestro_local\n    horas: 1\n    lote: 4\n    fuentes:\n"
                 "      wiki_es: data/pre/wiki_es:0.6\n      maestro: data/maestro:0.4\n", encoding="utf-8")
    return p


def test_config_previsualiza_aplica_con_respaldo_y_conserva_comentarios(servidor, raiz):
    app, port = servidor
    p = _plan_base(raiz)
    accion(port, "guardar_config", cambios={"plan": "plan_juguete.yaml"})
    antes = p.read_text()
    cambios = {"plan": {"secuencia": 128, "optimizador": "muon",
                        "etapas": {"1_lenguaje": {"horas": 2.5, "lote_max": 8,
                                                  "fuentes": {"wiki_es": 0.7, "codigo": 0.3}}}}}
    r = accion(port, "config_previsualizar", cambios=cambios)
    assert r["ok"] and r["diffs"] and "+secuencia: 128" in r["diffs"][0]["diff"]
    assert p.read_text() == antes                                       # previsualizar no escribe
    assert accion(port, "config_aplicar", cambios=cambios)["ok"] is False   # sin confirmar
    r = accion(port, "config_aplicar", cambios=cambios, confirmar=True)
    assert r["ok"], r
    nuevo = p.read_text()
    assert "# una hora" in nuevo and "# la primera" in nuevo and "# Pascal" in nuevo
    plan = yaml.safe_load(nuevo)
    e = plan["etapas"][0]
    assert plan["secuencia"] == 128 and plan["optimizador"] == "muon"
    assert e["horas"] == 2.5 and e["lote_max"] == 8 and e["fuentes"]["wiki_es"] == "data/pre/wiki_es:0.7"
    resp = list((raiz / "runs" / "respaldos").glob("*_plan_juguete.yaml"))
    assert len(resp) == 1 and resp[0].read_text() == antes
    assert estado(port)["ajustes"]["plan"]["secuencia"] == 128


def test_config_rechaza_valores_invalidos(servidor, raiz):
    app, port = servidor
    _plan_base(raiz)
    accion(port, "guardar_config", cambios={"plan": "plan_juguete.yaml"})
    malos = [
        {"plan": {"secuencia": 100}},                                           # no múltiplo de 64
        {"plan": {"secuencia": "mucho"}},
        {"plan": {"optimizador": "sgd"}},
        {"plan": {"precision": "fp8"}},
        {"plan": {"etapas": {"1_lenguaje": {"horas": -3}}}},
        {"plan": {"etapas": {"1_lenguaje": {"lote_max": 2.5}}}},
        {"plan": {"etapas": {"1_lenguaje": {"fuentes": {"wiki_es": 0.9, "codigo": 0.4}}}}},   # suman 1,3
        {"plan": {"etapas": {"1_lenguaje": {"fuentes": {"inventada": 1.0}}}}},
        {"plan": {"etapas": {"9_no_existe": {"horas": 1}}}},
        {"app": {"maestro.endpoint": "http://example.org/v1"}},                 # maestro remoto
        {"app": {"datos.carpeta": "  "}},
        {"app": {"calibracion": {}}},
        {"fuentes_web": {"cuota_diaria_mb": -1}},
        {"fuentes_web": {"horario": {"desde": 20, "hasta": 8}}},
        {"techo": {"valor": "mucho"}},
    ]
    for c in malos:
        r = accion(port, "config_previsualizar", cambios=c)
        assert r["ok"] is False and r["errores"], c
        assert accion(port, "config_aplicar", cambios=c, confirmar=True)["ok"] is False, c
    assert not (raiz / "runs" / "respaldos").exists()
    # Pesos que ya venían mal en el archivo: aviso (no bloquea); si se tocan, tienen que sumar 1.
    pj = raiz / "configs" / "plan_juguete.yaml"
    pj.write_text(pj.read_text().replace("data/maestro:0.4", "data/maestro:0.5"))
    r = accion(port, "config_previsualizar", cambios={"plan": {"etapas": {"2_maestro_local": {
        "horas": 3, "fuentes": {"wiki_es": 0.6, "maestro": 0.5}}}}})
    assert r["ok"] and any("normaliza" in a for a in r["avisos"])
    r = accion(port, "config_previsualizar", cambios={"plan": {"etapas": {"2_maestro_local": {
        "fuentes": {"wiki_es": 0.7, "maestro": 0.5}}}}})
    assert not r["ok"]
    # bf16 bloqueado si la calibración dice que la GPU no lo soporta.
    app.cfg.set("calibracion", {"perfil": {"bf16": False}, "plan_recomendado": "plan_juguete.yaml"})
    r = accion(port, "config_previsualizar", cambios={"plan": {"precision": "bf16-8bit"}})
    assert not r["ok"] and "bf16" in r["errores"][0]
    app.cfg.set("calibracion", {"perfil": {"bf16": True}, "plan_recomendado": "plan_juguete.yaml"})
    assert accion(port, "config_previsualizar", cambios={"plan": {"precision": "bf16-8bit"}})["ok"]


def test_config_web_siempre_solo_lectura(servidor, raiz):
    app, port = servidor
    antes = (raiz / "configs" / "web.yaml").read_text()          # el del repositorio (copiado por la fixture)
    for c in ({"acceso": "lectura_escritura"}, {"metodos": ["GET", "POST"]}, {"metodos": ["PUT"]},
              {"metodos": ["delete"]}, {"escribir": True}, {"modo": "todo"},
              {"sitios_permitidos": ["https://example.org/login"]}, {"limites": {"consultas_por_hora": 0}},
              {"modo": "lista_permitida", "sitios_permitidos": []}, {"modo": "solo_lista", "sitios_permitidos": []}):
        r = accion(port, "config_aplicar", cambios={"web": c}, confirmar=True)
        assert r["ok"] is False, c
    assert (raiz / "configs" / "web.yaml").read_text() == antes and not (raiz / "runs" / "respaldos").exists()
    r = accion(port, "config_aplicar", confirmar=True, cambios={"web": {
        "modo": "solo_lista", "sitios_permitidos": ["es.wikipedia.org", "*.python.org"], "metodos": ["get"]}})
    assert r["ok"], r
    w = yaml.safe_load((raiz / "configs" / "web.yaml").read_text())
    assert w["acceso"] == "solo_lectura" and set(w["metodos"]) <= {"GET", "HEAD"}
    assert w["modo"] == "lista_permitida" and "*.python.org" in w["sitios_permitidos"]   # nombre anterior → actual
    assert "SOLO LECTURA" in (raiz / "configs" / "web.yaml").read_text()
    # Un web.yaml editado a mano con escritura se corrige al guardar (o se rechaza el cambio).
    (raiz / "configs" / "web.yaml").write_text("modo: busqueda\nacceso: lectura_escritura\nmetodos: [GET, POST]\n")
    r = accion(port, "config_previsualizar", cambios={"web": {"modo": "busqueda"}})
    assert r["ok"] is False
    r = accion(port, "config_aplicar", confirmar=True, cambios={"web": {"modo": "busqueda", "acceso": "solo_lectura",
                                                                     "metodos": ["GET"]}})
    assert r["ok"], r
    w = yaml.safe_load((raiz / "configs" / "web.yaml").read_text())
    assert w["acceso"] == "solo_lectura" and w["metodos"] == ["GET"]
    resp = list((raiz / "runs" / "respaldos").glob("*_web.yaml"))      # una copia por cada vez que se aplicó
    assert len(resp) == 2 and any("lectura_escritura" in r.read_text() for r in resp)
    assert any("prohibido" in r.read_text() for r in resp)


def test_config_cuota_web_app_y_techo_auditado(servidor, raiz):
    app, port = servidor
    r = accion(port, "config_aplicar", confirmar=True, cambios={
        "fuentes_web": {"cuota_diaria_mb": 50, "paginas_por_dia": 300},
        "app": {"datos.carpeta": "D:/hope_datos", "maestro.model": "qwen2.5-1.5b"}})
    assert r["ok"], r
    fw = yaml.safe_load((raiz / "configs" / "fuentes_web.yaml").read_text())
    assert fw["diario"]["cuota_diaria_mb"] == 50 and fw["diario"]["paginas_por_dia"] == 300
    data = json.loads((raiz / "runs" / "app.json").read_text())
    assert data["datos"]["carpeta"] == "D:/hope_datos" and data["maestro"]["model"] == "qwen2.5-1.5b"
    # Techo: necesita nombre y la frase exacta; queda en la auditoría.
    c = {"techo": {"valor": "500M", "confirmacion": "AUTORIZO 500M"}}
    assert accion(port, "config_aplicar", cambios=c, confirmar=True)["ok"] is False      # sin nombre
    accion(port, "guardar_config", cambios={"usuario": "adrian"})
    r = accion(port, "config_previsualizar", cambios=c)
    assert r["ok"] and r["frase_techo"] == "AUTORIZO 500M" and "max_total_params: 500000000" in r["diffs"][0]["diff"]
    bad = accion(port, "config_aplicar", cambios={"techo": {"valor": "500M", "confirmacion": "si"}}, confirmar=True)
    assert bad["ok"] is False
    assert app.guardian().budget().max_total_params != 500_000_000
    assert accion(port, "config_aplicar", cambios=c, confirmar=True)["ok"]
    g = app.guardian()
    assert g.budget().max_total_params == 500_000_000
    audit = (raiz / "runs" / "guardian" / "audit.jsonl").read_text()
    assert "budget.changed_by_human" in audit and "adrian" in audit and g.audit.verify()[0]
    assert list((raiz / "runs" / "respaldos").glob("*_growth_budget.yaml"))


def test_empezar_a_entrenar_valida_requisitos(servidor, raiz):
    app, port = servidor
    accion(port, "guardar_config", cambios={"plan": "plan_juguete.yaml", "entrenamiento.muestras_auto": False})
    r = accion(port, "empezar_entrenar")
    assert r["ok"] is False and "calibración" in r["mensaje"]
    assert not app.entrenamiento.corriendo()
    app.cfg.set("calibracion", {"plan_recomendado": "plan_juguete.yaml", "dispositivo": "cpu", "torch": {"problemas": []}})
    shutil.rmtree(raiz / "data" / "pre")
    r = accion(port, "empezar_entrenar")
    assert r["ok"] is False and "datos" in r["mensaje"]
    # Con datos, arranca.
    for name in ("wiki_es", "codigo"):
        d = raiz / "data" / "pre" / name
        d.mkdir(parents=True)
        (d / "train.bin").write_bytes((TEXTO * 40).encode())
        (d / "val.bin").write_bytes((TEXTO * 4).encode())
    r = accion(port, "empezar_entrenar")
    assert r["ok"], r
    assert all(x["ok"] for x in r["requisitos"]) and app.entrenamiento.corriendo()
    assert estado(port)["ajustes"]["listo"]


def test_lanzador_encuentra_la_carpeta(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("lanzador", REPO / "empaquetado" / "lanzador.py")
    lz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lz)
    assert lz.carpeta_hope() == REPO
    assert lz.variante_torch() in ("cu126", "a770", "cpu")
    assert lz.comprobar(REPO) == 0
    (REPO / "hope_lanzador.log").unlink(missing_ok=True)
    assert (REPO / "empaquetado" / "hope.ico").read_bytes()[:4] == b"\x00\x00\x01\x00"


# ---------------------------------------------------------------------------
# Disco: nada pesado en el disco del sistema (dos discos simulados)
# ---------------------------------------------------------------------------
def test_disco_elige_el_que_no_es_del_sistema_y_pone_todo_ahi(tmp_path, monkeypatch):
    from hope.app import discos
    sistema = tmp_path / "C"
    datos = tmp_path / "D"
    for d in (sistema, datos):
        d.mkdir()
    repo = sistema / "hope"                       # el proyecto está en el disco del sistema
    shutil.copytree(REPO / "configs", repo / "configs")
    (repo / "runs").mkdir()
    (repo / "runs" / "viejo.txt").write_text("lo que ya había")
    lista = [{"ruta": str(sistema), "dispositivo": "C:", "libre_gb": 400.0, "total_gb": 500.0, "tipo": "SSD",
              "sistema": True},
             {"ruta": str(datos), "dispositivo": "D:", "libre_gb": 1800.0, "total_gb": 2000.0, "tipo": "HDD",
              "sistema": False},
             {"ruta": str(tmp_path / "E"), "dispositivo": "E:", "libre_gb": 900.0, "total_gb": 1000.0, "tipo": "SSD",
              "sistema": False}]
    p = discos.proponer(lista)
    assert p["disco"]["dispositivo"] == "D:" and p["carpeta"] == str(datos / "Hope") and not p["aviso"]
    # Con un solo disco (el del sistema): se propone igual, pero con aviso claro.
    solo = discos.proponer(lista[:1])
    assert solo["disco"]["sistema"] and "sistema" in solo["aviso"]
    info = discos.preparar(p["carpeta"], repo, p["disco"])
    assert (repo / discos.ARCHIVO).exists() and info["carpeta"] == str(datos / "Hope")
    for link in ("runs", "data", ".venv"):
        real = Path(os.path.realpath(repo / link))
        assert str(real).startswith(str(datos)), link
    assert (datos / "Hope" / "runs" / "viejo.txt").read_text() == "lo que ya había"
    env = discos.entorno(info["carpeta"])
    for k in ("PIP_CACHE_DIR", "HF_HOME", "TORCH_HOME", "TMP", "TEMP", "TMPDIR", "HOPE_OFFLOAD_DIR", "HOPE_MODELOS",
              "XDG_CACHE_HOME"):
        assert env[k].startswith(str(datos)), k
    # Lo que escribe la aplicación (configuración, entrenamiento, datos) termina en D.
    monkeypatch.setattr(discos, "detectar", lambda: lista)
    for k in list(env):                      # setenv + delenv: pytest las restaura (las borra) al terminar
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)
    from hope.app.server import serve
    app, httpd, _ = serve(repo, port=0, retomar=False)
    try:
        assert os.environ["TMPDIR"].startswith(str(datos)) and os.environ["HF_HOME"].startswith(str(datos))
        app.cfg.set("usuario", "adrian")
        assert Path(os.path.realpath(app.cfg.path)).is_relative_to(datos)
        ds = app.discos()
        assert ds["elegida"]["carpeta"] == str(datos / "Hope") and ds["en_sistema"] == []
        # Repetir no rompe nada (los enlaces ya están).
        assert discos.preparar(p["carpeta"], repo, p["disco"])["enlaces"]["runs"] == "ya estaba"
    finally:
        app.cerrar()
        httpd.shutdown()
        import tempfile
        tempfile.tempdir = None
    # Nada pesado quedó en C (solo el código, los configs y el archivo de ubicación).
    pesados = [f for f in sistema.rglob("*") if f.is_file() and not os.path.islink(f)
               and "configs" not in f.parts and f.name != discos.ARCHIVO]
    assert pesados == []


def test_venv_existente_no_se_mueve(tmp_path):
    from hope.app import discos
    repo = tmp_path / "repo"
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin")
    info = discos.preparar(tmp_path / "D" / "Hope", repo)
    assert "queda" in info["enlaces"][".venv"] and (repo / ".venv" / "pyvenv.cfg").exists()
    assert Path(os.path.realpath(repo / "runs")).is_relative_to(tmp_path / "D")


def test_offload_del_optimizador_a_disco(tmp_path):
    import torch
    from hope.train.lowprec import fetch_state, offload_to_disk, park_state
    offload_to_disk(tmp_path / "opt")
    try:
        st = {"m": torch.randn(64, 32), "v": torch.rand(64, 32), "paso": 3}
        ref = {k: v.clone() for k, v in st.items() if torch.is_tensor(v)}
        park_state(st)
        assert len(list((tmp_path / "opt").iterdir())) == 2
        assert all(torch.equal(st[k], ref[k]) for k in ref)
        fetch_state(st, torch.device("cpu"))
        st["m"] += 1
        park_state(st)                                   # reutiliza el mismo archivo
        assert len(list((tmp_path / "opt").iterdir())) == 2 and torch.equal(st["m"], ref["m"] + 1)
    finally:
        offload_to_disk(None)


# ---------------------------------------------------------------------------
# Descargas reanudables (servidor HTTP local, sin internet)
# ---------------------------------------------------------------------------
def test_descarga_reanuda_y_verifica(tmp_path):
    import hashlib
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from hope.app import descargas
    data = bytes(range(256)) * 20000                   # 5 MB: varios pedazos de 1 MB
    sha = hashlib.sha256(data).hexdigest()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            start = int(rng.split("=")[1].split("-")[0]) if rng else 0
            body = data[start:]
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/modelo.gguf"
    dest = tmp_path / "modelo.gguf"
    try:
        corte = {"n": 0}

        def detener():
            corte["n"] += 1
            return corte["n"] > 2                         # se corta a mitad de camino
        parcial = descargas.bajar(url, dest, sha=sha, total=len(data), detener=detener)
        tam = parcial.stat().st_size
        assert parcial.name.endswith(".part") and 0 < tam < len(data)
        vistos = []
        descargas.bajar(url, dest, sha=sha, total=len(data), progreso=lambda h, t: vistos.append(h))
        assert dest.read_bytes() == data and vistos[0] > tam          # siguió desde lo ya bajado
        assert not (tmp_path / "modelo.gguf.part").exists()
        dest.unlink()
        with pytest.raises(OSError):                      # SHA equivocado: se aparta
            descargas.bajar(url, dest, sha="0" * 64, total=len(data))
        assert (tmp_path / "modelo.gguf.corrupto").exists()
        with pytest.raises(descargas.SinEspacio):         # no hay lugar: ni empieza
            descargas.bajar(url, tmp_path / "x.bin", total=10**15)
    finally:
        srv.shutdown()


def test_elige_llama_cpp_para_cada_pc():
    from hope.app import descargas
    rel = {"tag_name": "b5000", "assets": [{"name": n, "browser_download_url": "https://x/" + n, "size": 1} for n in (
        "llama-b5000-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "llama-b5000-bin-win-vulkan-x64.zip", "llama-b5000-bin-win-cpu-x64.zip", "llama-b5000-bin-ubuntu-x64.zip",
        "llama-b5000-bin-macos-arm64.zip")]}
    assert [a["name"] for a in descargas.elegir_llama(rel, "windows", "cuda")] == [
        "llama-b5000-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]
    assert descargas.elegir_llama(rel, "windows", "arc")[0]["name"].endswith("vulkan-x64.zip")
    assert descargas.elegir_llama(rel, "linux", "cpu")[0]["name"] == "llama-b5000-bin-ubuntu-x64.zip"
    assert descargas.tipo_gpu("cuda") == "cuda" and descargas.tipo_gpu(None) == "cpu"


# ---------------------------------------------------------------------------
# Modo automático (con los pasos de red simulados)
# ---------------------------------------------------------------------------
def test_automatico_encadena_y_pausa_si_falta_espacio(servidor, raiz, monkeypatch):
    from hope.app import automatico, descargas
    app, port = servidor
    auto = app.tareas["automatico"]
    app.cfg.set("calibracion", {"plan_recomendado": "plan_juguete.yaml", "dispositivo": "cpu", "torch": {"problemas": []}})
    llamados = []
    monkeypatch.setattr(auto, "_dependencias", lambda: llamados.append("dependencias") or "ok")
    monkeypatch.setattr(auto, "_datos", lambda: llamados.append("datos") or "ok")
    monkeypatch.setattr(auto, "_preparar_maestro", lambda: llamados.append("maestro") or "simulado")
    monkeypatch.setattr(auto, "_entrenar", lambda: llamados.append("entrenar") or "ok")
    # Disco casi lleno: se pausa antes de bajar los textos, con un aviso.
    monkeypatch.setattr(descargas, "libre_gb", lambda p: 5.0)
    assert accion(port, "auto_iniciar")["ok"]
    s = _esperar(lambda: (lambda s: s if s["inicio"]["automatico"]["estado"] in ("pausada", "hecha", "error") else None)(estado(port)))
    a = s["inicio"]["automatico"]
    assert a["estado"] == "pausada" and "espacio" in a["error"], a
    # El maestro va antes que los textos (Qwen desde el principio): ya quedó listo.
    assert llamados == ["dependencias", "maestro"] and app.cfg.data["plan"] == "plan_juguete.yaml"
    assert [p["hecho"] for p in a["pasos"]][:3] == [True, True, True]
    # Con espacio sigue desde donde quedó (no repite lo hecho).
    monkeypatch.setattr(descargas, "libre_gb", lambda p: 5000.0)
    assert accion(port, "auto_iniciar")["ok"]
    s = _esperar(lambda: (lambda s: s if s["inicio"]["automatico"]["estado"] in ("pausada", "hecha", "error") else None)(estado(port)))
    assert s["inicio"]["automatico"]["estado"] == "hecha", s["inicio"]["automatico"]
    assert llamados == ["dependencias", "maestro", "datos", "entrenar"]
    s = estado(port)
    assert all(p["hecho"] for p in s["inicio"]["automatico"]["pasos"]) and s["pasos"]["automatico"]["hecho"]
    assert set(automatico.PASOS) == set(app.cfg.data["automatico"])


def test_vigilancia_de_disco_pausa_el_entrenamiento(servidor, raiz, monkeypatch):
    import shutil as sh
    import hope.app.entrenamiento as ent  # noqa: F401  (carga el módulo antes de parchearlo)
    app, port = servidor
    accion(port, "guardar_config", cambios={"plan": "plan_juguete.yaml", "entrenamiento.muestras_auto": False})
    assert accion(port, "entrenar_iniciar")["ok"]
    real = sh.disk_usage
    monkeypatch.setattr(sh, "disk_usage", lambda p: real(p)._replace(free=int(1e9)))
    app.entrenamiento._disco_visto = 0
    app.entrenamiento._vigilar_disco()
    assert app.entrenamiento.estado == "pausando" and "llenando" in app.entrenamiento.mensaje
    assert (raiz / "runs" / "hope" / "DETENER").exists()


# ---------------------------------------------------------------------------
# Chat de control
# ---------------------------------------------------------------------------
def _chat(port, texto):
    return accion(port, "chat", texto=texto)


def test_chat_frases_a_comandos(servidor, raiz):
    app, port = servidor
    accion(port, "guardar_config", cambios={"plan": "plan_juguete.yaml"})
    casos = {
        "¿cómo va?": ("respuesta", None), "ayuda": ("respuesta", None),
        "entrená 2 semanas": ("propuesta", "duracion"), "pausá": ("propuesta", "pausar"),
        "pará el entrenamiento": ("propuesta", "pausar"), "seguí": ("propuesta", "continuar"),
        "empezá": ("propuesta", "empezar"), "pausá a la noche": ("propuesta", "horario"),
        "pausá de 22 a 6": ("propuesta", "horario"), "sacá el horario": ("propuesta", "quitar_horario"),
        "agregá Wikipedia en inglés": ("respuesta", None),          # ya está incluida por defecto
        "quitá el código": ("propuesta", "fuente_quitar"), "usá 60 GB de texto": ("propuesta", "texto_gb"),
        "usá 1B de parámetros": ("propuesta", "parametros"), "activá la web": ("propuesta", "web"),
        "desactivá la web": ("propuesta", "web"), "exportá en 4 bits": ("propuesta", "cuantizar"),
        "calibrá de nuevo": ("propuesta", "calibrar"),
    }
    for frase, (tipo, cmd) in casos.items():
        r = _chat(port, frase)
        assert r["tipo"] == tipo, (frase, r)
        if cmd:
            assert r["propuesta"]["comando"] == cmd, (frase, r)
    r = _chat(port, "entrená 2 semanas")["propuesta"]
    assert r["args"]["plan"]["etapas"]["1_lenguaje"]["horas"] == 336 and r["diffs"] and "2 semanas" in r["resumen"]
    r = _chat(port, "pausá a la noche")["propuesta"]
    assert r["args"] == {"pausa": 23, "seguir": 7} and "23" in r["resumen"]
    assert _chat(port, "usá 1B de parámetros")["propuesta"]["sensible"]
    assert _chat(port, "activá la web")["propuesta"]["frase"] == "ACTIVAR WEB"


def test_chat_ambiguo_pregunta_y_nada_fuera_de_la_lista(servidor, raiz):
    app, port = servidor
    for frase in ("entrená más", "agregá datos", "agregá Wikipedia", "usá más parámetros", "exportá cuantizado",
                  "pausá y exportá en 4 bits", "pausá a las 23", ""):
        r = _chat(port, frase)
        assert r["tipo"] == "pregunta", (frase, r)
    for frase in ("ejecutá rm -rf /", "borrá todo", "hacé un café", "abrí una terminal", "mandá un mail"):
        r = _chat(port, frase)
        assert r["tipo"] == "respuesta" and "No entendí" in r["texto"], (frase, r)
    from hope.app import comandos
    assert set(app.propuestas.values()) == set() or all(p["comando"] in comandos.COMANDOS for p in app.propuestas.values())


def test_chat_confirmar_aplica_y_lo_sensible_pide_frase(servidor, raiz):
    app, port = servidor
    accion(port, "guardar_config", cambios={"plan": "plan_juguete.yaml", "usuario": "adrian"})
    p = _chat(port, "entrená 3 días")["propuesta"]
    plan = yaml.safe_load((raiz / "configs" / "plan_juguete.yaml").read_text())
    assert plan["etapas"][0]["horas"] == 1                              # no cambia hasta confirmar
    assert accion(port, "chat_confirmar", id=p["id"])["ok"]
    plan = yaml.safe_load((raiz / "configs" / "plan_juguete.yaml").read_text())
    assert plan["etapas"][0]["horas"] == 72
    assert accion(port, "chat_confirmar", id=p["id"])["ok"] is False    # no se confirma dos veces
    p = _chat(port, "pausá de 22 a 6")["propuesta"]
    assert accion(port, "chat_confirmar", id=p["id"])["ok"]
    assert app.cfg.data["horario"] == {"activo": True, "pausa": 22, "seguir": 6}
    # Sensible: sin la frase exacta no hace nada.
    p = _chat(port, "usá 500M de parámetros")["propuesta"]
    assert accion(port, "chat_confirmar", id=p["id"], frase="si")["ok"] is False
    assert app.guardian().budget().max_total_params != 500_000_000
    assert accion(port, "chat_confirmar", id=p["id"], frase="AUTORIZO 500M")["ok"]
    assert app.guardian().budget().max_total_params == 500_000_000
    audit = (raiz / "runs" / "guardian" / "audit.jsonl").read_text()
    assert "app.chat_confirmado" in audit and "budget.changed_by_human" in audit
    p = _chat(port, "activá la web")["propuesta"]
    assert accion(port, "chat_confirmar", id=p["id"], frase="ACTIVAR WEB")["ok"]
    w = yaml.safe_load((raiz / "configs" / "web.yaml").read_text())
    assert w["modo"] == "lista_permitida" and w["acceso"] == "solo_lectura"
    assert accion(port, "chat_confirmar", id="inventado")["ok"] is False
    s = estado(port)
    assert isinstance(s["inicio"]["frase"], str) and s["inicio"]["horario"]["activo"]


def test_horario_pausa_y_sigue(servidor, raiz, monkeypatch):
    app, port = servidor
    ent = app.entrenamiento
    llamados = []
    monkeypatch.setattr(ent, "pausar", lambda: llamados.append("pausar") or {"ok": True})
    monkeypatch.setattr(ent, "iniciar", lambda automatico=False: llamados.append("iniciar") or {"ok": True})
    hora = time.localtime().tm_hour
    app.cfg.set("horario", {"activo": True, "pausa": hora, "seguir": (hora + 8) % 24})
    ent.estado = "en_curso"
    ent._horario(corriendo=True)
    ent._horario(corriendo=True)                       # una sola vez por hora
    assert llamados == ["pausar"]
    app.cfg.set("horario", {"activo": True, "pausa": (hora + 8) % 24, "seguir": hora})
    ent._horario_visto = None
    ent.estado = "pausado"
    ent._horario(corriendo=False)
    assert llamados == ["pausar", "iniciar"]


# ---------------------------------------------------------------------------
# Celular: red local con PIN
# ---------------------------------------------------------------------------
def test_red_local_con_pin(raiz):
    from hope.app import red
    from hope.app.server import serve
    assert red.host_ok("127.0.0.1:8765", 8765, False) and not red.host_ok("192.168.0.5:8765", 8765, False)
    assert red.host_ok("192.168.0.5:8765", 8765, True) and not red.host_ok("evil.example:8765", 8765, True)
    assert not red.host_ok("192.168.0.5:9999", 8765, True)
    app, httpd, _ = serve(raiz, port=0, retomar=False)
    port = httpd.server_address[1]
    try:
        assert accion(port, "red_configurar", lan=True, confirmar=True)["ok"] is False      # sin PIN
        assert accion(port, "red_configurar", lan=True, pin="12", confirmar=True)["ok"] is False
        assert accion(port, "red_configurar", lan=True, pin="482913", confirmar=True)["ok"]
        rc = app.cfg.data["red"]
        assert rc["lan"] and "482913" not in json.dumps(rc)                                # solo el hash
        assert estado(port)["red"]["pin_configurado"]
    finally:
        app.cerrar()
        httpd.shutdown()
    # Al volver a abrir, escucha en la red. Simulamos un celular: dirección no local.
    import hope.app.red as red_mod
    orig = red_mod.es_local
    app, httpd, _ = serve(raiz, port=0, retomar=False)
    port = httpd.server_address[1]
    try:
        assert app.escucha == "0.0.0.0"
        red_mod.es_local = lambda ip: False
        code, r = _pedir(port, "GET", "/api/estado")
        assert code == 401 and r["login"]
        assert _pedir(port, "POST", "/api/accion", {"accion": "entrenar_iniciar"}, {"X-Hope": "1"})[0] == 401
        code, r = _pedir(port, "POST", "/api/login", {"pin": "000000"}, {"X-Hope": "1"})
        assert code == 401
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        c.request("POST", "/api/login", body=json.dumps({"pin": "482913"}),
                  headers={"Host": f"127.0.0.1:{port}", "X-Hope": "1", "Content-Type": "application/json"})
        resp = c.getresponse()
        resp.read()
        cookie = resp.getheader("Set-Cookie")
        assert resp.status == 200 and "HttpOnly" in cookie and "SameSite=Strict" in cookie
        token = cookie.split(";")[0]
        assert _pedir(port, "GET", "/api/estado", headers={"Cookie": token})[0] == 200
        # Desde el celular no se puede cambiar la configuración de red.
        r = _pedir(port, "POST", "/api/accion", {"accion": "red_configurar", "params": {"lan": False, "confirmar": True}},
                   {"X-Hope": "1", "Cookie": token})[1]
        assert r["ok"] is False and "propia PC" in r["mensaje"]
        # 5 PIN equivocados: bloqueo.
        for _ in range(5):
            _pedir(port, "POST", "/api/login", {"pin": "111111"}, {"X-Hope": "1"})
        code, r = _pedir(port, "POST", "/api/login", {"pin": "482913"}, {"X-Hope": "1"})
        assert code == 401 and "intentos" in r["mensaje"]
    finally:
        red_mod.es_local = orig
        app.cerrar()
        httpd.shutdown()


# ---------------------------------------------------------------------------
# Parámetros disponibles
# ---------------------------------------------------------------------------
def test_parametros_disponibles(servidor, raiz):
    from hope.app import ajustes, comandos
    app, port = servidor
    _plan_base(raiz)
    accion(port, "guardar_config", cambios={"plan": "plan_juguete.yaml", "usuario": "adrian"})
    r = accion(port, "presupuesto_plan", valor="1B")
    assert r["ok"] and r["frase"] == "AUTORIZO 1B" and r["disponible"] == comandos.presupuesto_disponible()
    if not r["disponible"]:
        assert "todavía no está" in r["mensaje"]
    assert accion(port, "presupuesto_aplicar", valor="1B", frase="AUTORIZO 1B")["ok"] is False   # sin confirmar
    assert accion(port, "presupuesto_aplicar", valor="1B", frase="no", confirmar=True)["ok"] is False
    assert accion(port, "presupuesto_aplicar", valor="1B", frase="AUTORIZO 1B", confirmar=True)["ok"]
    assert app.guardian().budget().max_total_params == 10**9
    # La etapa de presupuesto se agrega después de la etapa en curso, con copia del plan.
    r = ajustes.agregar_etapa_presupuesto(app, "1B")
    assert r["ok"]
    plan = yaml.safe_load((raiz / "configs" / "plan_juguete.yaml").read_text())
    assert [e["nombre"] for e in plan["etapas"]] == ["1_lenguaje", "presupuesto_1b", "2_maestro_local"]
    assert plan["etapas"][1]["presupuesto"] == "1B"
    assert ajustes.agregar_etapa_presupuesto(app, "1B")["mensaje"].startswith("el plan ya tiene")


def test_config_web_y_fuentes_conservan_lo_que_no_edita_la_app(servidor, raiz):
    """La pantalla edita modo, sitios, límites y la cuota; el guardián y web_sources leen lo mismo, y
    las claves que la pantalla no muestra (prohibido, contacto, lista de fuentes) no se pierden."""
    import shutil
    from guardian.web import WebPermissions
    from hope.data.web_sources import Registry, daily_bytes
    app, port = servidor
    for f in ("web.yaml", "fuentes_web.yaml"):
        shutil.copy(Path("configs") / f, raiz / "configs" / f)
    r = accion(port, "config_aplicar", confirmar=True, cambios={
        "web": {"modo": "lista_permitida", "sitios_permitidos": ["es.wikipedia.org", "arxiv.org"],
                "limites": {"consultas_por_hora": 12, "paginas_por_dia": 99, "mb_por_pagina": 1, "segundos_por_pedido": 9}},
        "fuentes_web": {"cuota_diaria_mb": 750, "paginas_por_dia": 40, "carpeta": "data/web2",
                        "horario": {"desde": 1, "hasta": 7}}})
    assert r["ok"], r
    texto = (raiz / "configs" / "web.yaml").read_text()
    perms = WebPermissions.from_yaml(texto)
    assert perms.mode == "lista_permitida" and perms.is_allowed("export.arxiv.org")
    assert perms.max_queries_per_hour == 12 and perms.max_pages_per_day == 99
    assert perms.max_bytes_per_day == 20_000_000                     # mb_por_dia no estaba en el cambio: se conserva
    w = yaml.safe_load(texto)
    assert w["prohibido"]["hosts_exactos"] == ["google.com", "www.google.com"] and "contacto" in w
    assert w["acceso"] == "solo_lectura" and w["buscador"] == "wikipedia"
    reg = Registry.load(raiz / "configs" / "fuentes_web.yaml")
    assert daily_bytes(reg.daily) == 750_000_000 and reg.daily["carpeta"] == "data/web2"
    assert reg.daily["horario"] == {"desde": 1, "hasta": 7} and len(reg.sources) >= 10   # las fuentes siguen ahí
    assert reg.daily["licencias_entrenamiento"]
    s = estado(port)
    assert s["ajustes"]["web"]["limites"]["consultas_por_hora"] == 12
    assert s["ajustes"]["fuentes_web"]["cuota_diaria_mb"] == 750
    # Un archivo escrito por la primera versión de la app (claves arriba, modo «desactivado») se migra.
    (raiz / "configs" / "fuentes_web.yaml").write_text("cuota_diaria_mb: 5\npaginas_por_dia: 1\ncarpeta: x\n")
    (raiz / "configs" / "web.yaml").write_text("modo: desactivado\nacceso: solo_lectura\nmetodos: [GET]\n"
                                               "dominios_permitidos: [pypi.org]\n")
    s = estado(port)
    assert s["ajustes"]["fuentes_web"]["cuota_diaria_mb"] == 5 and s["ajustes"]["web"]["modo"] == "apagado"
    assert s["ajustes"]["web"]["sitios_permitidos"] == ["pypi.org"]
    assert WebPermissions.from_yaml((raiz / "configs" / "web.yaml").read_text()).mode == "apagado"


# ---------------------------------------------------------------------------
# Chatear con Hope (copia del checkpoint, aparte del entrenamiento)
# ---------------------------------------------------------------------------
def _ligero(raiz, seed=0, hace=10.0):
    import torch
    from hope.core.hn1 import HN1Model, preset
    from hope.train.run import export_light
    torch.manual_seed(seed)
    p = raiz / "runs" / "hope" / "ligero.pt"
    export_light(HN1Model(preset("hn1-tiny")), p)
    t = time.time() - hace
    os.utime(p, (t, t))
    return p


def test_chatear_responde_con_una_copia_y_guarda_feedback(servidor, raiz):
    app, port = servidor
    r = accion(port, "charla_decir", texto="hola")
    assert r["ok"] is False and "checkpoint" in r["mensaje"]          # todavía no hay modelo
    _ligero(raiz)
    r = accion(port, "charla_cargar", bits=8)
    assert r["ok"] and r["bits"] == 8 and r["dispositivo"] == "cpu", r
    copia = Path(r["modelo"])
    assert copia.parent == raiz / "runs" / "hope" / "charla" and copia.exists()     # nunca lee ligero.pt directo
    r = accion(port, "charla_decir", texto="¿cuánto es 23 * 4?", max_bytes=40, objetivo_tok_s=5)
    assert r["ok"] and isinstance(r["respuesta"], str), r
    t = r["traza"]
    assert t["verificacion"]["detalle"] == "23 * 4 = 92" and "tokens_s" in t and "herramientas" in t
    r2 = accion(port, "charla_decir", texto="y ahora?", max_bytes=20, herramientas=False, contexto=False)
    assert r2["ok"]
    # 👍, 👎 y corregir (la corrección se verifica con la herramienta).
    assert accion(port, "charla_feedback", pregunta="hola", respuesta="hola!", voto="bien")["ok"]
    assert accion(port, "charla_feedback", pregunta="hola", respuesta="xx", voto="mal")["ok"]
    malo = accion(port, "charla_feedback", pregunta="¿cuánto es 23 * 4?", respuesta="x", voto="corregir", correccion="93")
    assert malo["ok"] is False and "verificación" in malo["mensaje"]
    ok = accion(port, "charla_feedback", pregunta="¿cuánto es 23 * 4?", respuesta="x", voto="corregir", correccion="Son 92.")
    assert ok["ok"] and ok["verificacion"]["ok"]
    ej = [json.loads(x) for x in (raiz / "data" / "charla" / "ejemplos.jsonl").read_text().splitlines()]
    assert [e["tipo"] for e in ej] == ["aprobado", "correccion"] and ej[1]["verificado"] is True
    assert (raiz / "data" / "charla" / "train.bin").exists() or (raiz / "data" / "charla" / "val.bin").exists()
    pesam = (raiz / "runs" / "hope" / "pesam_chat.jsonl").read_text()
    assert "chat_turno" in pesam and "chat_feedback" in pesam
    assert (raiz / "runs" / "hope" / "conocimiento.sqlite").exists()
    # Solo se cargan copias de la carpeta del chat (no rutas arbitrarias).
    assert accion(port, "charla_cargar", archivo="/etc/passwd")["ok"] is False
    assert accion(port, "charla_comparar", texto="hola", archivo="/etc/passwd")["ok"] is False


def test_chatear_se_recarga_con_un_checkpoint_nuevo_y_compara(servidor, raiz):
    app, port = servidor
    _ligero(raiz, seed=0, hace=20)
    (raiz / "runs" / "hope" / "registro.jsonl").write_text('{"etapa": "1_lenguaje", "paso": 100, "bpb_val": {}}\n')
    assert accion(port, "charla_cargar")["ok"]
    assert app.charla.cargado["etiqueta"] == "paso 100"
    # Aparece un checkpoint nuevo (paso 200): mientras se está escribiendo no se toca; después, se recarga.
    p = _ligero(raiz, seed=1, hace=0)
    with (raiz / "runs" / "hope" / "registro.jsonl").open("a") as f:
        f.write('{"etapa": "1_lenguaje", "paso": 200, "bpb_val": {}}\n')
    app.charla.revisar()
    assert app.charla.cargado["etiqueta"] == "paso 100"                 # recién escrito: espera
    t = time.time() - 10
    os.utime(p, (t, t))
    app.charla.revisar()
    assert app.charla.cargado["etiqueta"] == "paso 200"
    assert any("paso 200" in a["texto"] for a in estado(port)["charla"]["avisos"])
    copias = [c["etiqueta"] for c in estado(port)["charla"]["copias"]]
    assert copias[:2] == ["paso 200", "paso 100"]
    vieja = next(c["archivo"] for c in estado(port)["charla"]["copias"] if c["etiqueta"] == "paso 100")
    r = accion(port, "charla_comparar", texto="hola", archivo=vieja)
    assert r["ok"] and "respuesta" in r["actual"] and "respuesta" in r["otro"], r
    # Sin recarga automática, no cambia.
    accion(port, "charla_auto", auto=False)
    _ligero(raiz, seed=2, hace=10)
    app.charla.revisar()
    assert app.charla.cargado["etiqueta"] == "paso 200"
    # GPU solo si sobra memoria: en esta máquina no hay, así que se rechaza.
    assert accion(port, "charla_cargar", dispositivo="cuda")["ok"] is False


def test_chatear_no_interfiere_con_el_entrenamiento(servidor, raiz):
    app, port = servidor
    accion(port, "guardar_config", cambios={"plan": "plan_juguete.yaml", "entrenamiento.muestras_auto": False})
    assert accion(port, "entrenar_iniciar")["ok"]
    _esperar(lambda: (raiz / "runs" / "hope" / "ligero.pt").exists() and _registro(raiz), timeout=180)
    time.sleep(1.2)
    r = accion(port, "charla_cargar", bits=4)
    assert r["ok"], r
    antes = max(x["paso"] for x in _registro(raiz) if "paso" in x)
    for _ in range(3):
        assert accion(port, "charla_decir", texto="contame algo", max_bytes=30)["ok"]
    assert app.entrenamiento.corriendo()                                # el entrenamiento sigue
    _esperar(lambda: max(x["paso"] for x in _registro(raiz) if "paso" in x) > antes, timeout=120)
    assert app.charla.proc.pid != app.entrenamiento.pid                 # procesos separados


def test_tunel_pide_pin_aunque_venga_de_127(raiz):
    """Tailscale serve / cloudflared se conectan desde 127.0.0.1 con otro Host: eso NO cuenta como la propia PC."""
    from hope.app.server import serve
    app, httpd, _ = serve(raiz, port=0, retomar=False)
    port = httpd.server_address[1]
    try:
        assert accion(port, "red_hosts", hosts=["mipc.tailnet.ts.net"], confirmar=True)["ok"] is False   # sin PIN
        assert accion(port, "red_configurar", lan=False, pin="482913", confirmar=True)["ok"]
        assert accion(port, "red_hosts", hosts=["https://malo/"], confirmar=True)["ok"] is False
        assert accion(port, "red_hosts", hosts=["mipc.tailnet.ts.net"], confirmar=True)["ok"]
        tunel = {"Host": "mipc.tailnet.ts.net"}
        assert _pedir(port, "GET", "/api/estado", headers={"Host": "otro.example"})[0] == 403
        assert _pedir(port, "GET", "/api/estado", headers=tunel)[0] == 401
        code, _ = _pedir(port, "POST", "/api/accion", {"accion": "red_hosts", "params": {"hosts": [], "confirmar": True}},
                         {"X-Hope": "1", **tunel})
        assert code == 401
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        c.request("POST", "/api/login", body=json.dumps({"pin": "482913"}),
                  headers={**tunel, "X-Hope": "1", "Content-Type": "application/json"})
        resp = c.getresponse()
        resp.read()
        token = resp.getheader("Set-Cookie").split(";")[0]
        assert _pedir(port, "GET", "/api/estado", headers={**tunel, "Cookie": token})[0] == 200
        r = _pedir(port, "POST", "/api/accion", {"accion": "red_hosts", "params": {"hosts": [], "confirmar": True}},
                   {"X-Hope": "1", "Cookie": token, **tunel})[1]
        assert r["ok"] is False and "propia PC" in r["mensaje"]          # desde el túnel no se cambia
        assert app.escucha == "127.0.0.1"                                 # sin abrir puertos
    finally:
        app.cerrar()
        httpd.shutdown()


# ---------------------------------------------------------------------------
# Qwen desde el principio
# ---------------------------------------------------------------------------
def test_lo_nuevo_del_maestro_entra_mientras_se_entrena(servidor, raiz):
    """hope.train.run vuelve a leer las fuentes en cada tramo: data/maestro, vacía al empezar, entra después."""
    app, port = servidor
    plan = yaml.safe_load((raiz / "configs" / "plan_juguete.yaml").read_text())
    plan["etapas"][0]["fuentes"]["maestro"] = "data/maestro:0.3"
    (raiz / "configs" / "plan_juguete.yaml").write_text(yaml.safe_dump(plan), encoding="utf-8")
    accion(port, "guardar_config", cambios={"plan": "plan_juguete.yaml", "entrenamiento.muestras_auto": False})
    assert accion(port, "entrenar_iniciar")["ok"]
    _esperar(lambda: any("bpb_val" in r for r in _registro(raiz)), timeout=180)
    assert all("maestro" not in r.get("bpb_val", {}) for r in _registro(raiz))     # todavía no hay
    d = raiz / "data" / "maestro"
    d.mkdir(parents=True, exist_ok=True)
    ep = "Tema: suma\nProblema: ¿Cuánto es 12 + 30?\nPaso 1: 12 + 30 = 42 (suma)\nResultado: 42\n\x00"
    (d / "train.bin").write_bytes((ep * 60).encode())
    (d / "val.bin").write_bytes((ep * 30).encode())                     # más de 1 KB: si no, se ignora
    _esperar(lambda: any("maestro" in r.get("bpb_val", {}) for r in _registro(raiz)), timeout=180)
    assert app.entrenamiento.corriendo()                   # sin reiniciar el entrenamiento
    accion(port, "entrenar_detener_ya", confirmar=True)


def test_plan_1060_tiene_al_maestro_desde_la_etapa_1():
    plan = yaml.safe_load((REPO / "configs" / "plan_gtx1060.yaml").read_text())
    for e in plan["etapas"]:
        total = sum(float(str(v).rsplit(":", 1)[1]) for v in e["fuentes"].values())
        assert abs(total - 1) < 1e-6, e["nombre"]
    assert plan["etapas"][0]["fuentes"]["maestro"] == "data/maestro:0.20"


def test_qwen_explica_en_continuo_y_se_ve_en_inicio(servidor, raiz):
    app, port = servidor
    accion(port, "guardar_config", cambios={"maestro.model": "mock", "maestro.episodios_por_tema": 1,
                                            "maestro.generados_por_tema": 1, "maestro.max_consultas": 30})
    r = accion(port, "qwen_explicar", activo=True)
    assert r["ok"], r
    s = _esperar(lambda: (lambda s: s if s["inicio"]["qwen"]["hoy"] >= 18 else None)(estado(port)), timeout=300)
    q = s["inicio"]["qwen"]
    assert q["explicando"] and q["hoy_aceptadas"] > 0
    # Llega al máximo de consultas del día y NO termina: espera al día siguiente.
    _esperar(lambda: "mañana" in app.tareas["maestro"].detalle, timeout=300)
    assert app.tareas["maestro"].ocupada()
    assert (raiz / "data" / "maestro" / "train.bin").stat().st_size > 0
    assert accion(port, "qwen_explicar", activo=False)["ok"]
    app.tareas["maestro"].esperar(30)
    assert not app.tareas["maestro"].ocupada()
    # «Qwen lee junto a Hope» todavía no está: el lugar en la interfaz existe, pero no se puede activar.
    assert estado(port)["inicio"]["qwen"]["lee_junto"]["disponible"] is False
    assert accion(port, "qwen_lee_junto", activo=True)["ok"] is False


def test_maestro_en_gpu_segun_el_plan_y_la_vram(servidor):
    from hope.app.automatico import PASOS, maestro_en_gpu
    app, _ = servidor
    assert PASOS.index("maestro") < PASOS.index("datos") < PASOS.index("entrenar")
    app.cfg.set("calibracion", {"perfil": {"vram_gb": 6.4}})
    app.cfg.set("plan", "plan_gtx1060.yaml")
    assert maestro_en_gpu(app.cfg)                         # 200M + Qwen 1,5B entran en 6 GB
    app.cfg.set("plan", "plan_gtx1060_500m_vision.yaml")
    assert not maestro_en_gpu(app.cfg)                     # con 500M, Qwen va a la CPU
    app.cfg.set("calibracion", {"perfil": {"vram_gb": 16.0}})
    assert maestro_en_gpu(app.cfg)                         # A770


# ---------------------------------------------------------------------------
# Varios maestros: catálogo, servidores locales, VRAM, rotación y turnos (todo simulado)
# ---------------------------------------------------------------------------
def _servidor_falso(rutas: dict):
    """Servidor HTTP local: GET/HEAD de rutas fijas → (cuerpo, encabezados)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _r(self, cuerpo_tambien):
            if self.path not in rutas:
                self.send_response(404)
                self.end_headers()
                return
            body, headers = rutas[self.path]
            self.send_response(200)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if cuerpo_tambien:
                self.wfile.write(body)

        def do_GET(self):
            self._r(True)

        def do_HEAD(self):
            self._r(False)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_detecta_lm_studio_y_lo_agrega_como_maestro(servidor, raiz, monkeypatch):
    from hope.app import maestros
    app, port = servidor
    srv, p = _servidor_falso({"/v1/models": (json.dumps({"data": [{"id": "nvidia-nemotron-3-nano-4b"}]}).encode(), {})})
    try:
        monkeypatch.setattr(maestros, "PUERTOS", {"LM Studio": p, "Ollama": 1})
        r = accion(port, "servidores_detectar")
        assert r["ok"] and r["servidores"][0]["servidor"] == "LM Studio"
        assert r["servidores"][0]["modelos"] == ["nvidia-nemotron-3-nano-4b"]
        assert accion(port, "servidor_agregar", endpoint=f"http://127.0.0.1:{p}/v1", model="otro", confirmar=True)["ok"] is False
        assert accion(port, "servidor_agregar", endpoint=f"http://127.0.0.1:{p}/v1",
                      model="nvidia-nemotron-3-nano-4b", confirmar=True)["ok"]
        lst = estado(port)["maestros"]["lista"]
        srvs = [m for m in lst if m["tipo"] == "servidor"]
        assert srvs and srvs[0]["servidor"] == "LM Studio" and srvs[0]["activo"]
    finally:
        srv.shutdown()


def test_catalogo_licencias_y_borrado(servidor, raiz):
    from hope.app import maestros
    app, port = servidor
    lst = {m["id"]: m for m in estado(port)["maestros"]["lista"]}
    assert "qwen2.5-3b" not in lst                                      # el 3B de Qwen2.5 no es Apache
    assert lst["qwen2.5-1.5b"]["licencia"] == "Apache 2.0" and lst["qwen2.5-1.5b"]["activo"]
    assert lst["deepseek-r1-distill-qwen-1.5b"]["licencia"] == "MIT"
    for mid in ("llama-3.2-3b", "gemma-2-2b", "nemotron-nano-4b"):
        assert not lst[mid]["activo"] and lst[mid]["grupo"] in ("condiciones", "confirmar")
        r = accion(port, "maestro_activar", id=mid, activo=True)
        assert r["ok"] is False and r["licencia"], r                    # pide aceptar la licencia
    assert accion(port, "maestro_activar", id="gemma-2-2b", activo=True, acepto_licencia=True)["ok"]
    assert {m["id"]: m for m in estado(port)["maestros"]["lista"]}["gemma-2-2b"]["activo"]
    # Borrar el GGUF de un maestro desactivado.
    mod = maestros.carpeta_modelos(app)
    mod.mkdir(parents=True, exist_ok=True)
    (mod / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"x" * 1000)
    assert accion(port, "maestro_borrar", id="qwen3-4b", confirmar=True)["ok"] is False   # está activo
    accion(port, "maestro_activar", id="qwen3-4b", activo=False)
    assert accion(port, "maestro_borrar", id="qwen3-4b")["ok"] is False                  # sin confirmar
    assert accion(port, "maestro_borrar", id="qwen3-4b", confirmar=True)["ok"]
    assert not (mod / "Qwen3-4B-Q4_K_M.gguf").exists()


def test_presupuesto_de_vram():
    from hope.app import maestros
    nemo = {"gb": 2.6}
    r = maestros.entra_en_vram(nemo, 6.0, compartida=True, plan="plan_gtx1060.yaml")
    assert not r["entra"] and "GPU offload" in r["consejo"] and "turnos" in r["consejo"]
    assert maestros.entra_en_vram(nemo, 6.0, compartida=False)["entra"]                 # en su turno, sí
    assert maestros.entra_en_vram({"gb": 4.7}, 6.0, compartida=False)["entra"]         # 7B Q4 en la 1060
    assert not maestros.entra_en_vram({"gb": 5.0}, 6.0, compartida=False)["entra"]
    assert maestros.entra_en_vram({"gb": 1.1}, 16.0, compartida=True, plan="plan_a770_continuacion.yaml")["entra"]


def test_rotacion_ponderada_por_verificacion():
    from hope.app import maestros
    ms = [{"id": "a", "tipo": "servidor", "activo": True}, {"id": "b", "tipo": "servidor", "activo": True},
          {"id": "c", "tipo": "catalogo", "activo": True, "descargado": False},          # sin bajar: no se elige
          {"id": "d", "tipo": "servidor", "activo": False}]
    stats = {"a": {"aceptados": 90, "rechazados": 10}, "b": {"aceptados": 20, "rechazados": 80}}
    dados = {}
    for _ in range(20):
        m = maestros.elegir(ms, stats, dados)
        dados[m["id"]] = dados.get(m["id"], 0) + 1
    assert set(dados) == {"a", "b"} and dados["a"] > dados["b"] > 0


def test_maestro_de_cada_episodio_y_por_maestro(servidor, raiz):
    app, port = servidor
    accion(port, "guardar_config", cambios={"maestro.model": "mock", "maestro.episodios_por_tema": 2,
                                            "maestro.generados_por_tema": 0, "maestro.mock_error": 0.4})
    t = app.tareas["maestro"]
    assert t.iniciar(rondas=1, maestro={"id": "srv-a", "endpoint": "http://127.0.0.1:1/v1", "model": "mock"},
                     paralelo=4)["ok"]
    t.esperar(120)
    assert t.estado == "hecha", t.error
    eps = [json.loads(x) for x in (raiz / "data" / "maestro" / "episodios.jsonl").read_text().splitlines()]
    assert all(e["maestro"] == "srv-a" for e in eps if e["origen"] == "maestro")
    pm = t.conteos()["por_maestro"]["srv-a"]
    assert pm["aceptados"] > 0 and pm["rechazados"] > 0
    lst = {m["id"]: m for m in estado(port)["maestros"]["lista"]}
    assert all(m["verificado_pct"] is None for m in lst.values())       # srv-a no está en la lista: no se mezcla


def test_turnos_alternan_la_gpu(servidor, raiz, monkeypatch):
    """Hope 45 min → pausa tras checkpoint → maestro con la GPU 15 min → se descarga → vuelve Hope."""
    from hope.app import maestros
    app, port = servidor
    ent, mae, tur = app.entrenamiento, app.tareas["maestro"], app.turnos
    tur.cerrar()                                                        # el hilo no interfiere: paso() a mano
    log = []
    est = {"corre": False}
    monkeypatch.setattr(ent, "iniciar", lambda automatico=False: (log.append("hope:inicia"), est.update(corre=True))
                        and {"ok": True} or {"ok": True})
    monkeypatch.setattr(ent, "pausar", lambda: log.append("hope:pausa") or {"ok": True})
    monkeypatch.setattr(ent, "corriendo", lambda: est["corre"])
    monkeypatch.setattr(ent, "metricas", lambda: {"bytes_por_s": 5000.0, "etapas": [], "estado": "en_curso"})
    ocup = {"v": False}
    monkeypatch.setattr(mae, "iniciar", lambda **kw: (log.append(f"maestro:{kw['maestro']['id']}:{kw['paralelo']}"),
                                                      ocup.update(v=True)) and {"ok": True} or {"ok": True})
    monkeypatch.setattr(mae, "ocupada", lambda: ocup["v"])
    monkeypatch.setattr(mae, "pausar", lambda: (log.append("maestro:pausa"), ocup.update(v=False)) and {"ok": True})
    monkeypatch.setattr(mae, "conteos", lambda: {"por_maestro": {"m1": {"aceptados": 9, "rechazados": 1},
                                                                 "m2": {"aceptados": 3, "rechazados": 7}}})
    monkeypatch.setattr(maestros, "lista", lambda a: [
        {"id": "m1", "nombre": "Qwen", "tipo": "servidor", "activo": True, "endpoint": "http://127.0.0.1:9/v1", "model": "q"},
        {"id": "m2", "nombre": "Nemotron", "tipo": "servidor", "activo": True, "endpoint": "http://127.0.0.1:9/v1",
         "model": "n", "servidor": "LM Studio"}])
    lms = []
    monkeypatch.setattr(maestros, "lms", lambda *a: lms.append(a) or {"ok": True, "mensaje": ""})
    tb = raiz / "data" / "maestro" / "train.bin"
    tb.parent.mkdir(parents=True, exist_ok=True)
    tb.write_bytes(b"x" * 1000)                       # pocas explicaciones: Hope las repetiría mucho
    tur.configurar({"min_hope": 45, "min_maestro": 15, "max_repeticiones": 3, "paralelo": 6})
    t = 1_000_000.0
    tur.iniciar(ahora=t)
    assert log == ["hope:inicia"] and tur.fase == "hope"
    tur.paso(t + 44 * 60)
    assert tur.fase == "hope"
    tur.paso(t + 45 * 60)
    assert tur.fase == "pausando_hope" and log[-1] == "hope:pausa"
    tur.paso(t + 46 * 60)
    assert tur.fase == "pausando_hope"                 # todavía guarda el checkpoint
    est["corre"] = False
    tur.paso(t + 47 * 60)
    assert tur.fase == "maestro" and log[-1] == "maestro:m1:6"          # el que más verifica, con 6 pedidos a la vez
    assert "Turno de Qwen: explicando" in estado(port)["inicio"]["turno"]["frase"]
    tur.paso(t + 62 * 60)
    assert tur.fase == "cerrando_maestro" and log[-1] == "maestro:pausa"
    tur.paso(t + 62 * 60 + 5)
    assert tur.fase == "hope" and log[-1] == "hope:inicia"
    # Proporción automática: con 5 kB/s y 1 kB de explicaciones, el turno del maestro se alarga.
    assert tur.cfg()["min_maestro"] > 15
    # Segundo ciclo: rota a Nemotron vía LM Studio (lms load / unload).
    t2 = t + 62 * 60 + 5
    tur.paso(t2 + 45 * 60)
    est["corre"] = False
    tur.paso(t2 + 46 * 60)
    hechos = [x for x in log if x.startswith("maestro:m")]
    assert hechos[-1].startswith("maestro:m2") and lms and lms[0][0] == "load"
    tur.paso(t2 + 46 * 60 + tur.cfg()["min_maestro"] * 60)
    tur.paso(t2 + 47 * 60 + tur.cfg()["min_maestro"] * 60)
    assert lms[-1] == ("unload", "--all") and tur.fase == "hope"
    # Con muchas explicaciones nuevas, el turno del maestro se acorta.
    tb.write_bytes(b"x" * 50_000_000)
    antes = tur.cfg()["min_maestro"]
    tur._ajustar(tur.cfg())
    assert tur.cfg()["min_maestro"] < antes
    # Detener: no deja nada corriendo.
    est["corre"] = True
    assert tur.detener()["ok"] and tur.fase == "apagado" and log[-1] == "hope:pausa"
    assert accion(port, "turnos_config", cambios={"min_hope": 0})["ok"] is False
    assert accion(port, "turnos_config", cambios={"modo": "raro"})["ok"] is False


def test_descarga_de_maestros_del_catalogo(servidor, raiz, monkeypatch):
    import hashlib
    from hope.app import maestros
    app, port = servidor
    datos = b"GGUF" + bytes(range(256)) * 3000
    sha = hashlib.sha256(datos).hexdigest()
    rutas = {"/Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf":
             (datos, {"X-Linked-Etag": sha, "X-Linked-Size": str(len(datos))})}
    srv, p = _servidor_falso(rutas)
    try:
        monkeypatch.setattr(maestros, "HF", f"http://127.0.0.1:{p}/{{repo}}/{{archivo}}")
        for mid in ("qwen3-4b", "phi-4-mini"):                       # dejamos uno activo con archivo y otro que no existe
            accion(port, "maestro_activar", id=mid, activo=False)
        accion(port, "maestro_activar", id="mistral-7b-v0.3", activo=True)   # no está en el servidor falso: 404
        app.cfg.set("calibracion", {"perfil": {"vram_gb": 6.0}})
        assert accion(port, "maestros_descargar")["ok"]
        t = app.tareas["descarga_maestros"]
        t.esperar(120)
        assert t.estado == "hecha", t.error
        f = maestros.carpeta_modelos(app) / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        assert f.read_bytes() == datos
        assert any("no se encontró" in x for x in t.registro)
        lst = {m["id"]: m for m in estado(port)["maestros"]["lista"]}
        assert lst["qwen2.5-1.5b"]["descargado"] and not lst["mistral-7b-v0.3"]["descargado"]
    finally:
        srv.shutdown()


def test_automatico_pregunta_si_usar_lm_studio(servidor, raiz, monkeypatch):
    from hope.app import descargas, maestros
    app, port = servidor
    auto = app.tareas["automatico"]
    app.cfg.set("calibracion", {"plan_recomendado": "plan_juguete.yaml", "dispositivo": "cpu", "torch": {"problemas": []}})
    monkeypatch.setattr(descargas, "libre_gb", lambda p: 5000.0)
    monkeypatch.setattr(auto, "_dependencias", lambda: "ok")
    monkeypatch.setattr(auto, "_datos", lambda: "ok")
    monkeypatch.setattr(auto, "_entrenar", lambda: "ok")
    monkeypatch.setattr(maestros, "detectar_servidores", lambda puertos=None: [
        {"servidor": "LM Studio", "endpoint": "http://127.0.0.1:1234/v1", "modelos": ["nemotron-nano-4b"], "lms": False}])
    assert accion(port, "auto_iniciar")["ok"]
    s = _esperar(lambda: (lambda s: s if s["inicio"]["pregunta"] else None)(estado(port)), timeout=60)
    assert "LM Studio" in s["inicio"]["pregunta"]["texto"]
    assert accion(port, "auto_responder", respuesta="cualquiera")["ok"] is False
    assert accion(port, "auto_responder", respuesta="usar")["ok"]
    _esperar(lambda: not auto.ocupada(), timeout=60)
    assert auto.estado == "hecha", auto.error
    srvs = [m for m in estado(port)["maestros"]["lista"] if m["tipo"] == "servidor"]
    assert srvs and srvs[0]["model"] == "nemotron-nano-4b"
    assert not list(maestros.carpeta_modelos(app).glob("*.gguf"))       # no se bajó otro modelo
