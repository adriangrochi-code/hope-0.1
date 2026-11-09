"""Paso 2: datos. Descarga y construye las fuentes del plan con su presupuesto en GB.

Fuentes (carpetas dentro de ``<carpeta>/pre``, las mismas que leen los planes):

* ``wiki_es`` / ``wiki_en`` — Wikipedia (Parquet de Hugging Face; se guardan en ``<carpeta>/raw/wiki``
  y no se vuelven a bajar);
* ``codigo`` — biblioteca estándar de Python de esta instalación + carpetas de código elegidas;
* ``repos/repos_codigo`` y ``repos/repos_historial`` — repositorios git propios
  (``hope.data.repos``; los commits con coautoría de una IA se excluyen salvo que se pida);
* ``propias`` — libros, manuales y apuntes (txt, md, pdf con ``pdftotext``…).

Todas pasan las reglas de calidad (``hope.data.quality.rules``) salvo el historial de commits; con
un modelo de ``hope.data.quality`` se descarta además la prosa por debajo del umbral.
Pausar deja hechas las fuentes terminadas; al reanudar se rehace solo la que quedó a medias.
"""

from __future__ import annotations

import json
import sysconfig
import time
import urllib.request
from pathlib import Path

from hope.app.tareas import Tarea
from hope.data import repos as repos_mod
from hope.data import sources
from hope.data.corpus import TEXT_EXT, read_document

GB = 1e9
TEXTO_EXT = TEXT_EXT | {".pdf", ".markdown", ".html", ".htm", ".org", ".adoc"}


def presupuestos(cfg) -> dict[str, int]:
    d = cfg.data["datos"]
    total = float(d["presupuesto_gb"]) * GB
    rep = d["reparto"]
    out = {"wiki_es": rep.get("wiki_es", 0), "wiki_en": rep.get("wiki_en", 0), "codigo": rep.get("codigo", 0)}
    if d.get("repos"):
        out["repos_codigo"] = rep.get("repos", 0) * 0.6
        out["repos_historial"] = rep.get("repos", 0) * 0.4
    if d.get("propias"):
        out["propias"] = rep.get("propias", 0)
    return {k: int(v * total) for k, v in out.items() if v > 0}


def carpeta_de(name: str, base: Path) -> Path:
    return base / "pre" / ("repos/" + name if name.startswith("repos_") else name)


class TareaDatos(Tarea):
    nombre = "datos"

    def estado_fuentes(self) -> dict:
        cfg = self.app.cfg
        base = cfg.carpeta_datos
        out = {}
        for name, budget in presupuestos(cfg).items():
            d = carpeta_de(name, base)
            size = sum((d / f).stat().st_size for f in ("train.bin", "val.bin") if (d / f).exists())
            hecha = cfg.data["datos"]["hechas"].get(name)
            out[name] = {"presupuesto_gb": round(budget / GB, 2), "bytes_gb": round(size / GB, 3),
                         "progreso": 1.0 if hecha else min(size / budget, 1.0) if budget else 0.0,
                         "hecha": bool(hecha), "resumen": hecha, "carpeta": str(d)}
        return out

    # ------------------------------------------------------------------------
    def ejecutar(self, rehacer: bool = False) -> None:
        cfg = self.app.cfg
        d = cfg.data["datos"]
        base = cfg.carpeta_datos
        base.mkdir(parents=True, exist_ok=True)
        if rehacer:
            cfg.set("datos.hechas", {})
        budgets = presupuestos(cfg)
        from hope.app.sistema import disco
        libre = disco(base)["libre_gb"]
        need = sum(budgets.values()) / GB * 1.6          # .bin + Parquet de Wikipedia en caché
        if libre is not None and libre < need:
            self.log(f"aviso: hay {libre} GB libres en {base}; conviene tener ~{need:.0f} GB")
        clf = None
        if d.get("modelo_calidad"):
            from hope.data.quality import QualityClassifier
            p = cfg.ruta(d["modelo_calidad"])
            if p.exists():
                clf = QualityClassifier.load(p)
                self.log(f"filtro de calidad: {p} (umbral {d['umbral_calidad']})")
            else:
                self.log(f"aviso: no existe el modelo de calidad {p}; solo se aplican las reglas")
        names = list(budgets)
        for i, name in enumerate(names):
            if self.detener_pedido():
                return
            if cfg.data["datos"]["hechas"].get(name):
                continue
            self._actual = (i, len(names), name, budgets[name])
            self.detalle = f"{name}: empezando"
            self.log(f"— {name}: presupuesto {budgets[name] / GB:.2f} GB")
            docs, rules, prose = self._docs(name, base)
            if docs is None:
                continue
            if clf is not None and prose:
                docs = self._calidad(docs, clf, float(d["umbral_calidad"]))
            out = carpeta_de(name, base).parent
            summary = sources.build(out, [(name, self._vigilado(docs, carpeta_de(name, base)), budgets[name])],
                                    apply_rules=rules)[name]
            if self.detener_pedido():
                self.log(f"{name}: quedó a medias ({summary['bytes_train'] / GB:.2f} GB); al reanudar se rehace")
                return
            summary["fecha"] = time.strftime("%Y-%m-%d %H:%M")
            cfg.data["datos"]["hechas"][name] = summary
            cfg.save()
            self.log(f"{name}: {json.dumps(summary, ensure_ascii=False)}")
        self.detalle = "datos listos"

    # ------------------------------------------------------------------------
    def _docs(self, name: str, base: Path):
        """(documentos, aplicar reglas, es prosa)."""
        d = self.app.cfg.data["datos"]
        if name.startswith("wiki_"):
            return self._wiki(name[5:], base / "raw" / "wiki"), True, True
        if name == "codigo":
            folders = [Path(p).expanduser() for p in d.get("carpetas_codigo", [])]
            if d.get("incluir_stdlib", True):
                folders.insert(0, Path(sysconfig.get_paths()["stdlib"]))
            folders = [f for f in folders if f.is_dir()]
            if not folders:
                self.log("codigo: no hay carpetas de código")
                return None, True, False
            self.log("codigo: " + ", ".join(map(str, folders)))
            return sources.code_files(folders), True, False
        if name.startswith("repos_"):
            repos = [Path(r).expanduser() for r in d.get("repos", [])]
            ok = [r for r in repos if (r / ".git").exists()]
            for r in set(repos) - set(ok):
                self.log(f"aviso: {r} no es un repositorio git (no tiene .git); se saltea")
            if not ok:
                return None, True, False
            ia = bool(d.get("incluir_coautoria_ia"))
            if not ia:
                self.log("se excluyen los commits con coautoría de una IA (y los archivos hechos solo con ellos)")
            authors = d.get("autor") or None
            if name == "repos_codigo":
                return (doc for r in ok for doc in repos_mod.code_docs(r, include_ai=ia)), True, False
            return (doc for r in ok for doc in repos_mod.history_docs(r, include_ai=ia, authors=authors)), False, False
        if name == "propias":
            return self._propias([Path(p).expanduser() for p in d.get("propias", [])]), True, True
        return None, True, False

    def _propias(self, folders: list[Path]):
        for root in folders:
            if not root.exists():
                self.log(f"aviso: no existe {root}")
                continue
            files = [root] if root.is_file() else sorted(f for f in root.rglob("*") if f.is_file())
            for f in files:
                if self.detener_pedido():
                    return
                suf = f.suffix.lower()
                if suf in sources.CODE_EXT:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                elif suf in TEXTO_EXT:
                    text = read_document(f) if suf in TEXT_EXT | {".pdf"} else f.read_text(encoding="utf-8",
                                                                                          errors="ignore")
                    if text is None and suf == ".pdf":
                        self.log(f"aviso: {f.name}: falta pdftotext (poppler) para leer PDF")
                else:
                    continue
                if text and text.strip():
                    yield text, {"fuente": "propias", "ruta": str(f)}

    def _wiki(self, lang: str, cache: Path):
        n = sources.WIKI_SHARDS[lang]
        cache.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            if self.detener_pedido():
                return
            path = cache / f"{lang}-{i:05d}.parquet"
            if not path.exists():
                if not self._bajar(sources.HF_WIKI.format(lang=lang, i=i, n=n), path, f"wiki_{lang} {i + 1}/{n}"):
                    return
            yield from sources.wikipedia(lang, shards=[i], cache=cache)

    def _bajar(self, url: str, path: Path, label: str) -> bool:
        tmp = path.with_suffix(".part")
        self.log(f"descargando {label}: {url}")
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                if self.detener_pedido():
                    f.close()
                    tmp.unlink(missing_ok=True)
                    return False
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                self.detalle = (f"descargando {label}: {got / 1e6:.0f}" + (f" / {total / 1e6:.0f} MB" if total else " MB"))
        tmp.replace(path)
        return True

    def _calidad(self, docs, clf, threshold: float):
        bajos = 0
        for text, meta in docs:
            if float(clf.score([text])[0]) < threshold:
                bajos += 1
                if bajos % 1000 == 0:
                    self.log(f"filtro de calidad: {bajos} documentos por debajo del umbral")
                continue
            yield text, meta

    def _vigilado(self, docs, folder: Path):
        """Corta si se pide pausa y actualiza el progreso mirando cuánto se escribió."""
        i, n, name, budget = self._actual
        t = 0.0
        for k, item in enumerate(docs):
            if self.detener_pedido():
                return
            if k % 200 == 0 and time.time() - t > 1.0:
                t = time.time()
                size = sum((folder / f).stat().st_size for f in ("train.bin", "val.bin") if (folder / f).exists())
                frac = min(size / budget, 1.0) if budget else 1.0
                self.progreso = (i + frac) / n
                self.detalle = f"{name}: {size / GB:.2f} / {budget / GB:.2f} GB"
            yield item
