"""Catálogo ampliado: 22 herramientas más para los proyectos del usuario y el razonamiento general.

Ola 1: bases, ajuste, ode, constantes, resistencias, gcode, stl, java, validar, regex, graficar,
preguntar_maestro. Ola 2: hex, patron_bin, entropia, hash, desensamblar, marlin, material,
git_leer, imagen_info, recortar. Fichas, permisos y límites: docs/HERRAMIENTAS_HN1.md §8.

Reglas comunes:
* **Archivos**: solo a través del guardián (``ToolFiles``): dentro de la carpeta de trabajo,
  libre; fuera, aprobación humana. Los resultados (gráficos, recortes) se escriben en
  ``<carpeta>/runs/herramientas/``.
* **Sandbox**: todo lo que interpreta una entrada no confiable (G-code, STL, Marlin, perfiles,
  imágenes, XML/YAML/TOML, regex, expresiones de ode/graficar, Capstone, Java, git) corre en un
  proceso aparte con límites de tiempo y memoria (``call_function`` / ``run_command``).
* **Límites**: 64 MB por archivo, tiempos por herramienta (``TIMEOUTS``) y salida recortada.
"""

from __future__ import annotations

import base64
import re

from hope.tools import bin_tools as bt
from hope.tools import maker_tools as mk
from hope.tools import sci_tools as sc
from hope.tools import text_tools as tt
from hope.tools.catalog import Catalog, ToolSpec
from hope.tools.files import ToolFiles, looks_like_path
from hope.tools.sandbox import call_function

TIMEOUTS = {"ode": 30.0, "plot": 30.0, "gcode": 60.0, "stl": 60.0, "regex": 5.0, "validate": 10.0,
            "marlin": 20.0, "material": 20.0, "image": 30.0, "disasm": 20.0, "java": 60.0, "git": 20.0}
TESTS = ["tests/test_catalogo_ampliado.py"]


def _sb(module: str, func: str, timeout: float, mem_mb: int = 1024, **kw):
    return call_function(f"hope.tools.{module}", func, kw, timeout_s=timeout, mem_mb=mem_mb)


def _opts(parts: list[str]) -> tuple[list[str], dict]:
    """Separa las partes «clave=valor» (opciones) de las posicionales."""
    pos, opts = [], {}
    for p in parts:
        toks = p.split()
        if toks and all(re.fullmatch(r"[\wáéíóúñ]+=\S+", t) for t in toks):
            opts.update(t.split("=", 1) for t in toks)
        else:
            pos.append(p)
    return pos, opts


class ExtendedTools:
    """Funciones que se registran en el catálogo (cada una recibe argumentos ya separados)."""

    def __init__(self, guardian=None):
        self.files = ToolFiles(guardian)

    # --- ola 1 ----------------------------------------------------------------------
    def bases(self, expr: str, width: int | None = None) -> dict:
        return sc.bases(expr, width)

    def fit(self, model: str, xs: list[float], ys: list[float]) -> dict:
        return sc.fit(model, xs, ys)

    def ode(self, equations: str, initial: str, span: str) -> dict:
        return _sb("sci_tools", "ode", TIMEOUTS["ode"], equations=equations, initial=initial, span=span)

    def constants(self, query: str) -> list[dict]:
        return sc.constant(query)

    def resistor(self, query: str) -> str:
        return mk.resistor_text(query)

    def gcode(self, source: str, bed: str | None = None, diameter: float = 1.75, density: float = 1.24) -> dict:
        text = self.files.read_text(source, max_bytes=64 << 20) if looks_like_path(source, (".gcode", ".gco", ".g", ".nc")) \
            else source
        return _sb("maker_tools", "gcode_analyze", TIMEOUTS["gcode"], mem_mb=2048, text=text, bed=bed,
                   diameter=float(diameter), density=float(density))

    def stl(self, path: str, density: float | None = None) -> dict:
        p = self.files.check_read(path)
        return _sb("maker_tools", "stl_analyze_file", TIMEOUTS["stl"], mem_mb=3072, path=str(p),
                   density=None if density is None else float(density))

    def java(self, code: str, stdin: str = "") -> dict:
        return tt.java_run(code, stdin=stdin, timeout_s=TIMEOUTS["java"])

    def validate(self, fmt: str, text: str) -> dict:
        return _sb("text_tools", "validate", TIMEOUTS["validate"], mem_mb=512, fmt=fmt, text=text)

    def regex(self, pattern: str, cases: list[str], flags: str = "") -> dict:
        return _sb("text_tools", "regex_test", TIMEOUTS["regex"], mem_mb=512, pattern=pattern, cases=cases, flags=flags)

    def plot(self, spec: str) -> dict:
        r = _sb("sci_tools", "plot", TIMEOUTS["plot"], spec=spec)
        p = self.files.write_output("graficos", base64.b64decode(r.pop("png_b64")))
        return {**r, "ruta": self.files.rel(p), "ruta_absoluta": str(p)}

    def ask_teacher(self, question: str, teacher=None, run_tool=None) -> dict:
        from hope.tools.ask_teacher import ask_teacher
        return ask_teacher(teacher, question, run_tool)

    # --- ola 2 ----------------------------------------------------------------------
    def hexdump(self, path: str, offset: int = 0, length: int = 64) -> str:
        length = max(1, min(int(length), bt.MAX_DUMP))
        return bt.hexdump(self.files.read_bytes(path, offset, length), base=offset)

    def find_bytes(self, path: str, pattern: str, start: int = 0, end: int | None = None) -> dict:
        data = self.files.read_bytes(path, start, None if end is None else max(0, end - start))
        return bt.find_pattern(data, pattern, base=start)

    def entropy(self, path: str, block: int = 4096) -> dict:
        return bt.entropy(self.files.read_bytes(path), block=int(block))

    def hashes(self, path: str | None = None, text: str | None = None, algos: str = "") -> dict:
        data = text.encode("utf-8") if text is not None else self.files.read_bytes(path)
        return bt.hashes(data, algos)

    def disasm(self, arch: str, code_hex: str | None = None, path: str | None = None, offset: int = 0,
               length: int = 64) -> dict:
        if code_hex is None:
            code_hex = self.files.read_bytes(path, offset, max(1, min(int(length), 4096))).hex()
        return _sb("bin_tools", "disassemble", TIMEOUTS["disasm"], code_hex=code_hex, arch=arch, address=int(offset))

    def marlin(self, path: str) -> dict:
        p = self.files.check_read(path)
        if p.is_file() and p.with_name("Configuration_adv.h").exists():
            self.files.check_read(p.with_name("Configuration_adv.h"))
        return _sb("maker_tools", "marlin_check_files", TIMEOUTS["marlin"], path=str(p))

    def material(self, query: str, profiles: str | None = None) -> list[dict]:
        if profiles is None:
            return mk.material(query)
        p = self.files.check_read(profiles)
        return _sb("maker_tools", "material", TIMEOUTS["material"], query=query, profiles=str(p))

    def git(self, sub: str, repo: str = ".", args: list[str] | None = None, max_count: int = 20) -> dict:
        rp = self.files.check_read(repo)
        args = list(args or [])
        if sub.strip().lower() == "blame" and args:
            fp = self.files.check_read(rp / args[-1] if not args[-1].startswith("/") else args[-1])
            try:
                args[-1] = str(fp.relative_to(rp))
            except ValueError:
                raise ValueError("el archivo de blame debe estar dentro del repositorio") from None
        return tt.git_read(str(rp), sub, args, max_count=max_count, timeout_s=TIMEOUTS["git"])

    def image_info(self, path: str) -> dict:
        p = self.files.check_read(path)
        return _sb("image_tools", "image_info", TIMEOUTS["image"], mem_mb=2048, path=str(p))

    def crop(self, path: str, box: list[float], zoom: float = 1.0) -> dict:
        p = self.files.check_read(path)
        r = _sb("image_tools", "crop", TIMEOUTS["image"], mem_mb=2048, path=str(p), box=[float(v) for v in box],
                zoom=float(zoom))
        out = self.files.write_output("recortes", base64.b64decode(r.pop("png_b64")))
        return {**r, "ruta": self.files.rel(out), "ruta_absoluta": str(out)}


# ---------------------------------------------------------------------------
# Fichas (formato de Herramientas §16)

def _spec(tid, name, domain, objective, description, inputs, outputs, examples, *, execution="en proceso",
          deps=(), limits=(), risks=(), perms=("tool.execute",), dont=(), precision="", units="", cost="bajo"):
    return ToolSpec(tid, name, "1.0", domain, objective, description, dict(inputs), dict(outputs),
                    units_and_formats=units, dependencies=list(deps), execution=execution, cost=cost,
                    expected_precision=precision, limitations=list(limits), risks=list(risks),
                    permissions=list(perms), tests=list(TESTS), usage_examples=list(examples),
                    do_not_use_when=list(dont), provenance="catálogo ampliado (HERRAMIENTAS_HN1 §8)")


FILE = ("tool.execute", "file.read (fuera de la carpeta de trabajo: aprobación humana)")
WRITE = ("tool.execute", "file.write en runs/herramientas/")


def register_extended(cat: Catalog, guardian=None) -> ExtendedTools:
    x = ExtendedTools(guardian)
    specs = [
        (_spec("hte_bases_v1", "Bases numéricas y bits", "matemática",
               "convertir binario octal decimal hexadecimal bits máscaras operaciones and or xor desplazamiento",
               "Convierte enteros entre bases y evalúa operaciones de bits (& | ^ ~ << >>) con ancho opcional "
               "en complemento a dos. Acepta 0x.., 0b.., 0o.., sufijo h (BIOS/ASM) y «base N: dígitos».",
               {"expr": "str", "width": "int?"}, {"dec": "int", "hex": "str", "oct": "str", "bin": "str"},
               ["⟦bases: 255⟧ → 255 = 0xFF = 0o377 = 0b11111111 (8 unos)",
                "⟦bases: 0xF0 xor 0x3C⟧ → 204 = 0xCC = …", "⟦bases: ~0x0F | 8⟧ → 240 = 0xF0 … con signo -16"],
               precision="exacta (enteros de precisión arbitraria)", limits=["desplazamientos y exponentes ≤ 4096"],
               dont=["números con decimales: usar calc"]), x.bases),
        (_spec("hte_fit_v1", "Ajuste de curvas", "matemática",
               "ajuste regresión lineal polinómica exponencial potencia logarítmica r2 datos tendencia",
               "Mínimos cuadrados (NumPy): lineal, polinómica de grado N, exponencial, potencial, logarítmica o "
               "«auto» (el más simple entre los de mejor R²). R² en el espacio original de y.",
               {"model": "str", "xs": "list[float]", "ys": "list[float]"},
               {"coef": "dict", "ecuacion": "str", "r2": "float"},
               ["⟦ajuste: lineal | 1 2 3 4 | 3 5 7 9⟧ → y = 2·x + 1 (R² = 1, n = 4, …)",
                "⟦ajuste: exp | 0 1 2 | 3 4.946 8.155⟧", "⟦ajuste: auto | 1 2 3 4 5 | 1 4 9 16 25⟧"],
               deps=["numpy"], limits=["exp y potencia ajustan en escala logarítmica (necesitan y > 0)",
                                       "2 a 1 000 000 puntos"], precision="doble precisión"), x.fit),
        (_spec("hte_ode_v1", "Ecuaciones diferenciales ordinarias", "matemática",
               "ecuación diferencial ordinaria edo sistema dinámico transitorio oscilador integrar en el tiempo",
               "Integra y' = f(t, y), sistemas y orden superior (y'' = …). SciPy solve_ivp (DOP853, rtol 1e-11) si "
               "está; si no, RK4 propio. Expresiones con evaluador propio (sin eval), en sandbox.",
               {"equations": "str", "initial": "str", "span": "str"}, {"final": "dict", "puntos": "list"},
               ["⟦ode: y' = -2*y | y(0) = 1 | t = 0..1⟧ → y(1) = 0.1353352832",
                "⟦ode: y'' = -y | y(0)=1, y'(0)=0 | 0..pi⟧ → y(3.14159) = -1",
                "⟦ode: x' = v; v' = -x - 0.1*v | x(0)=1, v(0)=0 | 0..10⟧"],
               execution="sandbox", deps=["numpy", "scipy (opcional)"], cost="medio",
               limits=["orden ≤ 6", "sistemas rígidos: RK4 puede necesitar SciPy", "30 s"],
               precision="SciPy ~1e-10; RK4 ~1e-9 en problemas suaves"), x.ode),
        (_spec("hte_constants_v1", "Constantes físicas (CODATA)", "física",
               "constante física valor planck boltzmann avogadro carga electrón velocidad de la luz",
               "Constantes CODATA 2018 (NIST) con unidad, incertidumbre y si son exactas por definición del SI.",
               {"query": "str"}, {"valor": "float", "unidad": "str", "incertidumbre": "float", "fuente": "str"},
               ["⟦constantes: planck⟧ → h = 6.62607015e-34 J·s (exacta; …)", "⟦constantes: G⟧",
                "⟦constantes: lista⟧"], perms=("ninguno",),
               limits=["28 constantes de uso común; CODATA 2022 cambia algunas medidas en la última cifra"]),
         x.constants),
        (_spec("hte_resistor_v1", "Resistencias", "electrónica",
               "resistencia código de colores bandas serie e12 e24 e96 led valor comercial",
               "Código de colores de 3 a 6 bandas (IEC 60062) en ambos sentidos, valor comercial más cercano en "
               "E12/E24/E96 y resistencia para LED con potencia recomendada.",
               {"query": "str"}, {"resultado": "str"},
               ["⟦resistencias: marrón negro rojo dorado⟧ → 1 kΩ ±5 % (950 Ω a 1.05 kΩ)",
                "⟦resistencias: colores 4k7⟧", "⟦resistencias: E24 4800⟧ → E24: más cercano 4.7 kΩ …",
                "⟦resistencias: led 5 2 20mA⟧ → R = 150 Ω → comercial 150 Ω (E24), …"],
               perms=("ninguno",), precision="exacta"), x.resistor),
        (_spec("hte_gcode_v1", "Análisis de G-code", "impresión 3D",
               "gcode impresión 3d tiempo capas filamento peso temperaturas marlin klipper slicer",
               "Analiza G-code (Marlin/Klipper): tiempo estimado (distancia/velocidad, sin aceleración), capas, "
               "filamento neto y peso, retracciones, temperaturas, movimientos fuera de la cama, comandos "
               "desconocidos, macros de Klipper y extrusión en frío. Compara con el tiempo del slicer si viene.",
               {"source": "str (ruta o contenido)", "bed": "str?", "diameter": "float?", "density": "float?"},
               {"tiempo_s": "float", "capas": "int", "filamento_mm": "float", "peso_g": "float", "avisos": "list"},
               ["⟦gcode: pieza.gcode⟧", "⟦gcode: pieza.gcode | cama=235x235x250 diametro=1.75 densidad=1.27⟧"],
               execution="sandbox", perms=FILE, cost="medio",
               limits=["el tiempo es una cota inferior (no modela aceleración ni jerk)", "5 M líneas, 64 MB"],
               units="mm, mm/min (F), °C", precision="exacta para longitudes y capas"), x.gcode),
        (_spec("hte_stl_v1", "Análisis de STL", "impresión 3D",
               "stl malla volumen área dimensiones cerrada agujeros normales fotogrametría pieza",
               "Lee STL ASCII o binario con NumPy: volumen, área, dimensiones, malla cerrada, agujeros (lazos de "
               "borde), aristas no variedad, orientación inconsistente, normales invertidas, cuerpos y peso.",
               {"path": "str", "density": "float?"},
               {"volumen_mm3": "float?", "area_mm2": "float", "dimensiones_mm": "list", "cerrada": "bool"},
               ["⟦stl: pieza.stl⟧ → 12 triángulos (binario); 10 × 10 × 10 mm; área 600 mm²; volumen 1000 mm³ …",
                "⟦stl: scan.stl | PETG⟧ (peso con la densidad del material)"],
               execution="sandbox", deps=["numpy"], perms=FILE, cost="medio",
               limits=["5 M triángulos", "el volumen solo se informa si la malla es cerrada y orientada"]), x.stl),
        (_spec("hte_java_v1", "Compilar y ejecutar Java", "programación",
               "java compilar ejecutar javac mods minecraft programa",
               "Compila con javac y ejecuta con java en una carpeta temporal del sandbox (memoria y tiempo "
               "limitados). Varios archivos con «// archivo: Nombre.java».",
               {"code": "str", "stdin": "str?"}, {"ok": "bool", "stdout": "str"},
               ["⟦java: public class Hola { public static void main(String[] a){ System.out.println(2+3); } }⟧ → 5"],
               execution="sandbox", deps=["JDK (javac, java) del sistema"], cost="medio",
               limits=["sin Minecraft ni Forge/Fabric: solo lógica pura", "60 s", "sin JDK devuelve cómo instalarlo"],
               risks=["el sandbox no bloquea la red"]), x.java),
        (_spec("hte_validate_v1", "Validar formatos", "datos",
               "validar json yaml xml toml sintaxis error línea columna configuración",
               "Valida JSON, YAML, XML o TOML y da la línea y columna del error.",
               {"fmt": "str", "text": "str"}, {"valido": "bool", "linea": "int?", "columna": "int?"},
               ["⟦validar: json ||| {\"a\": 1,}⟧ → JSON inválido en línea 1, columna 8: …",
                "⟦validar: configs/web.yaml⟧"],
               execution="sandbox", perms=FILE, limits=["4 MB", "XML con <!ENTITY> se rechaza"]), x.validate),
        (_spec("hte_regex_v1", "Probar expresiones regulares", "programación",
               "expresión regular regex patrón probar casos coincidencia grupos",
               "Compila el patrón y lo prueba contra casos: coincidencia completa, búsquedas con posición y grupos. "
               "En sandbox con 5 s: un retroceso catastrófico se corta.",
               {"pattern": "str", "cases": "list[str]"}, {"casos": "list"},
               ["⟦regex: (\\d{4})-(\\d{2}) ||| 2024-05 ||| x 1999-12 y⟧"],
               execution="sandbox", limits=["sintaxis de Python «re»", "50 casos, 20 coincidencias por caso"]),
         x.regex),
        (_spec("hte_plot_v1", "Graficar", "datos",
               "gráfico graficar función datos curva png visualizar",
               "Gráfico simple a PNG (matplotlib si está; si no, rasterizador propio). Devuelve la ruta en "
               "runs/herramientas/graficos/ para mirarlo con la zona visual (⟦imagen_info⟧, ⟦recortar⟧).",
               {"spec": "str"}, {"ruta": "str", "y": "list"},
               ["⟦graficar: sin(x); cos(x) | -3.14..3.14⟧", "⟦graficar: datos | 1 2 3 4 | 1 4 9 16⟧"],
               execution="sandbox", deps=["matplotlib (opcional)"], perms=WRITE,
               limits=["hasta 5 funciones de x", "400 puntos por función"]), x.plot),
        (_spec("hte_ask_teacher_v1", "Preguntar al maestro", "meta",
               "preguntar al maestro consultar qwen duda explicación",
               "Consulta al maestro configurado a través del guardián. Si el maestro propone una comprobación "
               "con una herramienta determinista, se ejecuta y la respuesta queda VERIFICADA o REFUTADA; si no, "
               "NO VERIFICADA. El texto vuelve como dato neutralizado.",
               {"question": "str"}, {"respuesta": "str", "estado": "str"},
               ["⟦preguntar_maestro: ¿qué resistencia necesita un LED rojo a 5 V y 20 mA?⟧ → ⟪maestro · VERIFICADA …⟫"],
               perms=("teacher.query (remoto: aprobación humana)",), cost="alto",
               dont=["lo resuelve una herramienta determinista directamente"],
               risks=["el maestro puede equivocarse: por eso se comprueba"]), x.ask_teacher),
        (_spec("hte_hexdump_v1", "Volcado hexadecimal", "binarios",
               "hex volcado hexadecimal bytes binario bios firmware offset",
               "Volcado hexadecimal + ASCII de un rango de un archivo.",
               {"path": "str", "offset": "int", "length": "int"}, {"volcado": "str"},
               ["⟦hex: bios.rom | 0x0 | 64⟧", "⟦hex: firmware.bin | 0FFF0h | 16⟧ (vector de reset)"],
               perms=FILE, limits=["4096 bytes por volcado; la salida de una llamada se recorta a 600 caracteres"]),
         x.hexdump),
        (_spec("hte_binpattern_v1", "Buscar patrones binarios", "binarios",
               "buscar bytes patrón firma cadena binario bios offset",
               "Busca secuencias de bytes con comodines (?? ), cadenas ASCII o UTF-16LE; devuelve los offsets.",
               {"path": "str", "pattern": "str"}, {"coincidencias": "int", "offsets": "list[int]"},
               ["⟦patron_bin: bios.rom | \"$IBIOSI$\"⟧", "⟦patron_bin: bios.rom | 55 AA⟧",
                "⟦patron_bin: bios.rom | u\"BIOS\"⟧"], perms=FILE, limits=["64 MB", "200 offsets"]), x.find_bytes),
        (_spec("hte_entropy_v1", "Entropía por bloques", "binarios",
               "entropía bloques comprimido cifrado bios zonas firmware",
               "Entropía de Shannon por bloques y zonas: vacío/relleno, código/datos, comprimido/cifrado.",
               {"path": "str", "block": "int"}, {"zonas": "list", "entropia_global": "float"},
               ["⟦entropia: bios.rom⟧", "⟦entropia: bios.rom | 65536⟧"], deps=["numpy"], perms=FILE,
               limits=["comprimido y cifrado no se distinguen solo con entropía (> 7,5 bits/byte)"]), x.entropy),
        (_spec("hte_hash_v1", "Hashes", "binarios",
               "hash md5 sha1 sha256 crc32 suma de verificación integridad",
               "MD5, SHA-1, SHA-256 (y SHA-512) y CRC32 de un archivo o de un texto.",
               {"path": "str?", "text": "str?"}, {"md5": "str", "sha1": "str", "sha256": "str", "crc32": "str"},
               ["⟦hash: texto: abc⟧ → md5 900150983cd24fb0d6963f7d28e17f72, …", "⟦hash: bios.rom | sha256⟧"],
               perms=FILE, precision="exacta"), x.hashes),
        (_spec("hte_disasm_v1", "Desensamblar", "binarios",
               "desensamblar instrucciones x86 arm código máquina bios ensamblador",
               "Capstone (x86 16/32/64, ARM, Thumb, ARM64) sobre bytes en hexadecimal o un rango de archivo, en "
               "sandbox. Sin Capstone devuelve cómo instalarlo.",
               {"arch": "str", "code_hex": "str?", "path": "str?", "offset": "int", "length": "int"},
               {"instrucciones": "list"},
               ["⟦desensamblar: x86-64 | 55 48 89 e5 c3⟧ → push rbp; mov rbp, rsp; ret",
                "⟦desensamblar: x86-16 | bios.rom | 0xFFF0 | 16⟧"],
               execution="sandbox", deps=["capstone (opcional)"], perms=FILE), x.disasm),
        (_spec("hte_marlin_v1", "Revisar configuración de Marlin", "impresión 3D",
               "marlin firmware configuration.h pasos por mm termistor límites velocidades incoherencias",
               "Lee Configuration.h y Configuration_adv.h (#define activos) y reporta valores clave e "
               "incoherencias comunes: protección térmica, termistores, precalentado vs máximos, pasos/mm, "
               "límites de la cama, velocidades y sonda/nivelación.",
               {"path": "str"}, {"valores": "dict", "incoherencias": "list"},
               ["⟦marlin: Marlin/⟧", "⟦marlin: Marlin/Configuration.h⟧"], execution="sandbox", perms=FILE,
               limits=["no evalúa #if/#ifdef: toma todos los #define no comentados",
                       "los rangos «habituales» son orientativos"]), x.marlin),
        (_spec("hte_material_v1", "Materiales de impresión", "impresión 3D",
               "material filamento temperatura boquilla cama velocidad pla petg abs tpu nylon perfil",
               "Tabla de materiales (temperaturas, velocidades, ventilador, densidad, cerramiento) con su fuente; "
               "con «perfiles=ruta» lee los perfiles del usuario (PrusaSlicer .ini u OrcaSlicer/Bambu .json, por "
               "ejemplo un repositorio como Tisma-profiles) y los pone primero.",
               {"query": "str", "profiles": "str?"}, {"materiales": "list"},
               ["⟦material: PETG⟧", "⟦material: comparar PLA PETG ASA⟧", "⟦material: PETG | perfiles=Tisma-profiles⟧"],
               perms=FILE, limits=["rangos típicos: la ficha del carrete manda"]), x.material),
        (_spec("hte_git_v1", "Leer un repositorio git", "programación",
               "git historia log show blame diff autor cambios repositorio",
               "log/show/blame/diff/status de un repositorio local, solo lectura y en sandbox: sin pager, sin diff "
               "externos ni textconv, sin hooks, sin red; argumentos que empiezan con «-» se rechazan.",
               {"sub": "str", "repo": "str", "args": "list[str]"}, {"salida": "str"},
               ["⟦git_leer: log | . | 5⟧", "⟦git_leer: show | . | HEAD~1⟧", "⟦git_leer: blame | . | hope/tools/catalog.py⟧",
                "⟦git_leer: diff | . | HEAD~2 HEAD⟧"],
               execution="sandbox", deps=["git"], perms=FILE), x.git),
        (_spec("hte_image_info_v1", "Información de imagen", "imagen",
               "imagen tamaño exif histograma foto fotogrametría cámara",
               "Tamaño, modo, formato, EXIF principal (cámara, lente, focal, exposición) e histograma por canal.",
               {"path": "str"}, {"ancho": "int", "alto": "int", "exif": "dict", "histograma": "dict"},
               ["⟦imagen_info: fotos/IMG_0001.jpg⟧"], execution="sandbox", deps=["Pillow"], perms=FILE,
               limits=["80 M píxeles", "sin Pillow solo tamaño"]), x.image_info),
        (_spec("hte_crop_v1", "Recortar y ampliar", "imagen",
               "recortar zoom ampliar región detalle imagen leer",
               "Recorta una región (píxeles o fracciones) y la amplía; guarda un PNG en runs/herramientas/recortes/ "
               "para leer el detalle con la zona visual.",
               {"path": "str", "box": "list[float]", "zoom": "float"}, {"ruta": "str", "tamano": "list"},
               ["⟦recortar: placa.jpg | 100 50 200 120 | 3⟧", "⟦recortar: captura.png | 0.5 0 0.5 0.5⟧"],
               execution="sandbox", deps=["Pillow"], perms=(*FILE, "file.write en runs/herramientas/"),
               limits=["resultado ≤ 4096 px de lado", "zoom 0,1–16"]), x.crop),
    ]
    for spec, fn in specs:
        cat.register(spec, fn)
    return x


# ---------------------------------------------------------------------------
# Llamadas dentro del texto: ⟦nombre: argumento⟧

HELP = {
    "bases": "bases y bits: ⟦bases: 255⟧, ⟦bases: 0xF0 xor 0x3C⟧, ⟦bases: ~0x0F | 8⟧",
    "ajuste": "regresión con R²: ⟦ajuste: lineal | 1 2 3 4 | 3 5 7 9⟧ (poli 2, exp, potencia, log, auto)",
    "ode": "ecuaciones diferenciales: ⟦ode: y' = -2*y | y(0) = 1 | t = 0..1⟧",
    "constantes": "constantes CODATA: ⟦constantes: planck⟧",
    "resistencias": "resistencias: ⟦resistencias: marrón negro rojo dorado⟧, ⟦resistencias: led 5 2 20mA⟧",
    "gcode": "analizar G-code: ⟦gcode: pieza.gcode | cama=235x235x250⟧",
    "stl": "analizar una malla STL: ⟦stl: pieza.stl | PETG⟧",
    "java": "Java en sandbox: ⟦java: public class Main { public static void main(String[] a){ … } }⟧",
    "validar": "validar formatos: ⟦validar: json ||| {\"a\": 1}⟧ o ⟦validar: configs/web.yaml⟧",
    "regex": "probar regex: ⟦regex: \\d+ ||| abc 12 ||| 345⟧",
    "graficar": "gráfico PNG: ⟦graficar: sin(x) | -3..3⟧ o ⟦graficar: datos | 1 2 3 | 1 4 9⟧",
    "preguntar_maestro": "consultar al maestro (con verificación): ⟦preguntar_maestro: ¿…?⟧",
    "hex": "volcado hexadecimal: ⟦hex: bios.rom | 0x100 | 64⟧",
    "patron_bin": "buscar bytes: ⟦patron_bin: bios.rom | 55 AA⟧ o ⟦patron_bin: bios.rom | \"$IBIOSI$\"⟧",
    "entropia": "entropía por bloques: ⟦entropia: bios.rom | 4096⟧",
    "hash": "hashes: ⟦hash: archivo.bin⟧ o ⟦hash: texto: abc⟧",
    "desensamblar": "desensamblar: ⟦desensamblar: x86-64 | 55 48 89 e5 c3⟧ o ⟦desensamblar: x86-16 | bios.rom | 0xFFF0 | 16⟧",
    "marlin": "revisar Marlin: ⟦marlin: Marlin/Configuration.h⟧",
    "material": "materiales de impresión: ⟦material: PETG⟧ o ⟦material: PLA | perfiles=Tisma-profiles⟧",
    "git_leer": "git solo lectura: ⟦git_leer: log | . | 5⟧, show, blame, diff",
    "imagen_info": "información de imagen: ⟦imagen_info: foto.jpg⟧",
    "recortar": "recortar y ampliar: ⟦recortar: foto.jpg | x y ancho alto | zoom⟧",
}


def dispatch(runner, name: str, arg: str) -> str:
    c = runner.catalog.call
    parts = [p.strip() for p in arg.split("|")]
    big = [p.strip() for p in arg.split("|||")]
    raw = [p[1:] if p.startswith(" ") else p for p in arg.split("|||")]      # contenido tal cual (validar, regex)
    raw = [p[:-1] if p.endswith(" ") else p for p in raw]
    if name == "bases":
        expr, _, w = arg.partition("|")
        m = re.search(r"\d+", w)
        r = c("hte_bases_v1", expr=expr, width=int(m.group()) if m else None)
        s = f"{r['dec']} = {r['hex']} = {r['oct']} = {r['bin']} ({r['unos']} unos"
        return s + (f"; {r['ancho']} bits, con signo {r['con_signo']})" if "ancho" in r else ")")
    if name == "ajuste":
        if len(parts) == 2:
            pairs = [p for p in re.split(r";|\n", parts[1]) if p.strip()]
            xy = [sc.parse_numbers(p.replace(",", " ") if p.count(",") == 1 else p) for p in pairs]
            parts = [parts[0], " ".join(str(p[0]) for p in xy), " ".join(str(p[1]) for p in xy)]
        if len(parts) != 3:
            raise ValueError("formato: ⟦ajuste: lineal | x1 x2 … | y1 y2 …⟧")
        r = c("hte_fit_v1", model=parts[0], xs=sc.parse_numbers(parts[1]), ys=sc.parse_numbers(parts[2]))
        s = f"{r['ecuacion']} (R² = {r['r2']:.6g}, n = {r['n']}, modelo {r['modelo']})"
        return s + ("; R² de cada modelo: " + ", ".join(f"{k} {v}" for k, v in r["candidatos"].items())
                    if "candidatos" in r else "")
    if name == "ode":
        if len(parts) != 3:
            raise ValueError("formato: ⟦ode: y' = … | y(0) = … | t = 0..1⟧")
        return sc.ode_text(c("hte_ode_v1", equations=parts[0], initial=parts[1], span=parts[2]))
    if name == "constantes":
        c("hte_constants_v1", query=arg)                       # pasa por el catálogo (TLM, guardián)
        return sc.constant_text(arg)
    if name == "resistencias":
        return c("hte_resistor_v1", query=arg)
    if name == "gcode":
        pos, o = _opts(parts)
        return mk.gcode_text(c("hte_gcode_v1", source="|".join(pos), bed=o.get("cama") or o.get("bed"),
                               diameter=float(o.get("diametro", o.get("diámetro", 1.75))),
                               density=float(o.get("densidad", 1.24))))
    if name == "stl":
        pos, o = _opts(parts)
        dens = o.get("densidad")
        if dens is None and len(pos) > 1:
            mat = mk.material(pos[1])
            dens = mat[0]["densidad_g_cm3"] if mat else float(pos[1].replace(",", "."))
        return mk.stl_text(c("hte_stl_v1", path=pos[0], density=None if dens is None else float(dens)))
    if name == "java":
        return tt.java_text(c("hte_java_v1", code=big[0], stdin=big[1] if len(big) > 1 else ""))
    if name == "validar":
        if len(big) > 1:
            fmt, text = big[0], raw[1]
        else:
            fmt, _, path = arg.partition("|") if "|" in arg else ("", "", arg)
            path = path.strip()
            fmt = fmt.strip() or path.rsplit(".", 1)[-1]
            text = runner_files(runner).read_text(path)
        return tt.validate_text(c("hte_validate_v1", fmt=fmt, text=text))
    if name == "regex":
        if len(big) < 2:
            raise ValueError("formato: ⟦regex: patrón ||| caso1 ||| caso2⟧")
        return tt.regex_text(c("hte_regex_v1", pattern=raw[0].strip(), cases=raw[1:]))
    if name == "graficar":
        r = c("hte_plot_v1", spec=arg)
        return (f"gráfico guardado en {r['ruta']} ({r['motor']}; x ∈ [{r['x'][0]:.4g}, {r['x'][1]:.4g}], "
                f"y ∈ [{r['y'][0]:.4g}, {r['y'][1]:.4g}]); miralo con ⟦imagen_info: {r['ruta']}⟧ o ⟦recortar⟧")
    if name == "preguntar_maestro":
        from hope.tools.ask_teacher import VERIFIERS, ask_teacher_text

        def verify(tool: str, a: str) -> str:
            if tool not in VERIFIERS:
                return "error: herramienta no admitida para comprobar"
            return runner.run(tool, a)

        return ask_teacher_text(c("hte_ask_teacher_v1", question=arg, teacher=getattr(runner, "teacher", None),
                                  run_tool=verify))
    if name == "hex":
        return c("hte_hexdump_v1", path=parts[0], offset=bt.parse_int(parts[1] if len(parts) > 1 else None),
                 length=bt.parse_int(parts[2] if len(parts) > 2 else None, 64))
    if name == "patron_bin":
        if len(parts) < 2:
            raise ValueError("formato: ⟦patron_bin: archivo | patrón⟧")
        r = c("hte_binpattern_v1", path=parts[0], pattern=parts[1],
              start=bt.parse_int(parts[2] if len(parts) > 2 else None),
              end=bt.parse_int(parts[3]) if len(parts) > 3 else None)
        offs = ", ".join(f"0x{o:X}" for o in r["offsets"][:20])
        return f"{r['coincidencias']} coincidencias de {r['patron']}" + (f": {offs}" if offs else "") + (
            " …" if r["coincidencias"] > 20 else "")
    if name == "entropia":
        return bt.entropy_text(c("hte_entropy_v1", path=parts[0],
                                 block=bt.parse_int(parts[1] if len(parts) > 1 else None, 4096)))
    if name == "hash":
        if arg.strip().lower().startswith("texto:"):
            r = c("hte_hash_v1", text=arg.strip()[6:].lstrip())
        else:
            r = c("hte_hash_v1", path=parts[0], algos=parts[1] if len(parts) > 1 else "")
        return ", ".join(f"{k} {v}" for k, v in r.items())
    if name == "desensamblar":
        if len(parts) < 2:
            raise ValueError("formato: ⟦desensamblar: x86-64 | 55 48 89 e5 c3⟧")
        hx = re.sub(r"\s|0x|,", "", parts[1])
        if len(parts) == 2 and re.fullmatch(r"(?:[0-9A-Fa-f]{2})+", hx):
            r = c("hte_disasm_v1", arch=parts[0], code_hex=hx)
        else:
            r = c("hte_disasm_v1", arch=parts[0], path=parts[1],
                  offset=bt.parse_int(parts[2] if len(parts) > 2 else None),
                  length=bt.parse_int(parts[3] if len(parts) > 3 else None, 64))
        return bt.disasm_text(r)
    if name == "marlin":
        return mk.marlin_text(c("hte_marlin_v1", path=arg.strip()))
    if name == "material":
        pos, o = _opts(parts)
        return mk.material_text(c("hte_material_v1", query=" ".join(pos),
                                  profiles=o.get("perfiles") or o.get("profiles")))
    if name == "git_leer":
        sub = parts[0]
        repo = parts[1] if len(parts) > 1 and parts[1] else "."
        rest = " ".join(parts[2:]).split()
        n = 20
        if sub.lower() == "log" and rest and rest[0].isdigit():
            n, rest = int(rest[0]), rest[1:]
        r = c("hte_git_v1", sub=sub, repo=repo, args=rest, max_count=n)
        return r["salida"] or "(sin salida)"
    if name == "imagen_info":
        from hope.tools.image_tools import image_info_text
        return image_info_text(c("hte_image_info_v1", path=arg.strip()))
    if name == "recortar":
        if len(parts) < 2:
            raise ValueError("formato: ⟦recortar: imagen | x y ancho alto | zoom⟧")
        box = sc.parse_numbers(parts[1])
        if len(box) != 4:
            raise ValueError("la región son 4 números: x y ancho alto")
        zoom = float(parts[2].replace(",", ".").lstrip("x×")) if len(parts) > 2 else 1.0
        r = c("hte_crop_v1", path=parts[0], box=box, zoom=zoom)
        return (f"recorte {r['region'][2]}×{r['region'][3]} desde ({r['region'][0]}, {r['region'][1]}) ampliado a "
                f"{r['tamano'][0]}×{r['tamano'][1]}, guardado en {r['ruta']}")
    raise KeyError(name)


def runner_files(runner) -> ToolFiles:
    return ToolFiles(runner.catalog.guardian)
