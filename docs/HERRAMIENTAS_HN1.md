# Hope HN-1 — Plan del ecosistema de herramientas

**Objetivo (decisión del usuario):** que Hope dé resultados mucho más fiables y aprenda mucho más de sus propios datos, apoyándose en herramientas que van desde una calculadora hasta simuladores de ingeniería, en lugar de depender solo de lo que sus pesos memorizan.

Base: documento de Herramientas (HTE, TCG, TVE, TLM, TEO, HTO), Carta (permisos) y lo ya implementado en `hope/tools/`.

---

## 1. Por qué las herramientas multiplican lo que Hope puede aprender

1. **Fiabilidad.** Lo que una herramienta determinista calcula no depende de que Hope "recuerde bien". Una red de 200M que delega la aritmética, el álgebra o la simulación puede dar resultados exactos.
2. **Datos propios verificados.** Cada problema que Hope resuelve **y una herramienta confirma** se convierte en un ejemplo de entrenamiento correcto: autoentrenamiento STaR, generadores y verificación de episodios de Qwen. Cuantas más herramientas, más tipos de problemas puede practicar sola, sin depender del maestro.
3. **Aprender a usar herramientas.** Las trazas de uso (qué herramienta, con qué entrada, qué salió, si sirvió) alimentan la TLM. Hope aprende *cuándo* usar cada herramienta, no solo cómo.
4. **Menos parámetros para lo mismo.** Lo procedimental y exacto queda en software, y la red se dedica a entender, planificar e interpretar (Herramientas §8).

## 2. Cómo usa Hope una herramienta mientras razona

**Protocolo de llamada dentro del texto (a implementar):** Hope escribe una llamada, el runtime la ejecuta a través del guardián y escribe el resultado a continuación; Hope sigue razonando con ese resultado.

```
Paso 1: ⟦calc: 220 * 0,05⟧ → 11
Paso 2: ⟦unidades: 11 V / 0,05 A en ohm⟧ → 220 ohm
```

- **Generación:** cuando Hope genera `⟧`, el bucle de generación ejecuta la herramienta y agrega `→ resultado`. Con la caché por parche + KV, insertar el resultado cuesta lo mismo que un prompt corto.
- **Entrenamiento:** los generadores y los episodios de Qwen incluyen llamadas en ese formato, así Hope aprende a pedirlas. STaR acepta una solución solo si las llamadas fueron correctas y el resultado final pasa la crítica.
- **Seguridad:** cada llamada pasa por el guardián con el nivel de permiso de la herramienta.

**Implementado** (`hope/runtime/toolcalls.py`):
- La generación se detiene al cerrar `⟧`, ejecuta la herramienta a través del catálogo y del guardián, inserta ` → resultado` y retoma.
- Cada llamada queda registrada (entrada, salida, éxito, tiempo) para la TLM.
- Los episodios se escriben con llamadas (`Episode.training_text(tool_calls=True)`, por defecto en los generadores y en el ciclo).
- Hope responde y practica (STaR) usando herramientas.

Nombres disponibles: `calc`, `fraccion`, `estad`, `unidades`, `resolver`, `derivar`, `integrar`, `limite`, `matriz`, `logica`, `python`, `memoria`, `crear_herramienta`, `componer` (y las de los niveles 3–5).

### 2.1 Crear y componer herramientas (TCG/TVE y HTO)

```
⟦crear_herramienta: contar vocales de un texto⟧ → creada tcg_contar_vocales_v1 (EXPERIMENTAL; texto: str → cantidad: int)
⟦componer: invertir > contar_vocales > doble | murciélago⟧ → 10
```

**TCG** (`hope/tools/creation.py`, `ToolForge`). Cuando Hope repite una tarea sin herramienta (`observe`, por defecto 3 veces) o la pide, el maestro devuelve:
- una especificación: nombre, entradas y salidas tipadas, dominio y limitaciones;
- el código Python;
- las pruebas pytest.

**TVE** valida la herramienta sin creerle al maestro:
1. **Especificación:** nombre válido, tipos conocidos, la función existe y acepta las entradas declaradas.
2. **Seguridad (AST, lista permitida):** solo módulos de cálculo (`math`, `fractions`, `re`, `numpy`, `sympy`, …). Sin red, archivos, procesos, `eval`/`exec`/`__import__` ni atributos `__internos__`. Se revisan el código y las pruebas. Si falla, no se ejecuta nada.
3. **Análisis estático:** ruff, si está instalado. Los errores graves (sintaxis, nombres indefinidos) la rechazan.
4. **Pruebas propias** del maestro, en el sandbox, con límites de tiempo y memoria.
5. **Pruebas independientes**, generadas a partir de casos conocidos que aporta Hope y no el maestro (Herramientas §18). Sin casos conocidos, la herramienta queda EXPERIMENTAL pero no se puede consolidar.

**Registro:**
- Si pasa, queda en el catálogo como **EXPERIMENTAL**, con su ficha, procedencia (maestro, pedido y sha256 del código) y registro en disco.
- Se ejecuta siempre en un proceso aparte del sandbox, y se comprueban los tipos de entrada y salida.
- **Consolidarla** (ESTABLE) pasa por el guardián con la acción `tool.consolidate`, de nivel 3. La aprobación humana queda atada al hash exacto del código.
- Si falla, el motivo se guarda y se le muestra al maestro en el próximo intento.

**HTO** (`hope/tools/compose.py`, `Composer`):
- **Tipos:** antes de ejecutar se comprueba con las fichas que la salida de cada paso cabe en la entrada del siguiente (`int` cabe en `number`, `int` no cabe en `str`). Con `h.campo` se elige un campo de una salida con varios.
- **Ejecución:** paso a paso a través del catálogo y del guardián, comprobando el tipo real de cada resultado.
- **TLM:** registra el tiempo de cada paso y del total, y si la composición funcionó y fue exacta. Con eso, `best_compositions(tema)` prefiere las composiciones rápidas y exactas.

## 3. Catálogo por niveles

Estado: ✅ implementado · 🔜 propuesto para la próxima etapa · ⏳ más adelante.

Permiso (Carta art. 5):
- **1**: autónomo; cálculo puro, sin efectos.
- **2**: sandbox y límites.
- **3**: requiere aprobación humana (red, hardware, acciones externas).

### Nivel 0 — Cálculo básico

| Herramienta | Para qué | Implementación | Permiso | Estado |
|---|---|---|---|---|
| Calculadora decimal exacta | aritmética sin errores de redondeo | evaluador propio sobre `Decimal` (sin `eval`) | 1 | ✅ |
| Unidades y dimensiones | conversiones y chequeo dimensional | `pint` | 1 | ✅ |
| Fracciones y números exactos | probabilidades, proporciones | `fractions` | 1 | ✅ |
| Estadística descriptiva | media, desvío, mediana, percentiles | `statistics` / `numpy` | 1 | ✅ |
| Fechas y tiempos | duraciones, calendarios | `datetime` | 1 | 🔜 |

### Nivel 1 — Matemática

| Herramienta | Para qué | Implementación | Permiso | Estado |
|---|---|---|---|---|
| Equivalencia simbólica | verificar transformaciones algebraicas | SymPy en sandbox | 2 | ✅ |
| Resolver ecuaciones y sistemas | despejar, sistemas lineales y no lineales | SymPy `solve` / `nsolve` | 2 | ✅ |
| Cálculo | derivadas, integrales, límites, series | SymPy | 2 | ✅ (sin series) |
| Álgebra lineal | matrices, determinantes, autovalores | NumPy | 2 | ✅ |
| Optimización | mínimos, ajustes de curvas | SciPy `optimize` | 2 | ⏳ |
| Ecuaciones diferenciales | sistemas dinámicos, transitorios | SciPy `integrate` / SymPy `dsolve` | 2 | ⏳ |
| Probabilidad y estadística inferencial | distribuciones, pruebas de hipótesis | SciPy `stats` | 2 | ⏳ |

### Nivel 2 — Verificación y lógica

| Herramienta | Para qué | Implementación | Permiso | Estado |
|---|---|---|---|---|
| Restricciones e invariantes | datos contradictorios o faltantes, valor esperado | HCRI (`hope/constraints`) | 1 | ✅ |
| Crítica de razonamientos | revisar paso a paso | CRE (`hope/constraints/critic.py`) | 1 | ✅ |
| SAT/SMT | lógica, restricciones combinatorias, verificación de programas chicos | `z3-solver` | 2 | ✅ |
| Tablas de verdad | lógica proposicional | Python en sandbox | 2 | ✅ (vía Python) |
| Expresiones regulares | extraer y validar formatos | `re` con límite de tiempo | 2 | ⏳ |
| Demostrador formal | pruebas matemáticas verificadas | Lean 4 (externo) | 2 | ⏳ (complejo) |

### Nivel 3 — Programación (Hope Code)

| Herramienta | Para qué | Implementación | Permiso | Estado |
|---|---|---|---|---|
| Ejecutar Python | probar código, calcular | sandbox (`hope/tools/sandbox.py`) | 2 | ✅ |
| Ejecutar pruebas | verificar cambios de código | `pytest` en sandbox | 2 | ✅ |
| Análisis estático | errores sin ejecutar | `ruff` (mypy ⏳) | 2 | ✅ |
| Compilar C/C++ | Hope Code en C/C++ | `gcc`/`g++` en sandbox | 2 | ✅ |
| Buscar en código | ubicar funciones y usos | `ripgrep` + permiso de lectura del guardián | 1 | ✅ |
| Diff y parches | proponer cambios revisables | `difflib` | 2 | ✅ |
| Git local (lectura) | historia, autores, cambios | `git log/show` | 2 | ⏳ |
| Perfilador | dónde se va el tiempo | `cProfile` | 2 | ⏳ |

### Nivel 4 — Datos y documentos

| Herramienta | Para qué | Implementación | Permiso | Estado |
|---|---|---|---|---|
| Memoria propia | recuperar hechos, pasajes y fuentes exactas | `MemoryStore` + recuperación híbrida | 1 | ✅ |
| Leer PDF | papers, manuales | `pdftotext`; Marker/Nougat para fórmulas | 2 | ✅ (texto) / ⏳ (fórmulas) |
| Tablas | extraer y consultar tablas | SQL sobre CSV | 2 | ✅ |
| Consultas SQL | datos estructurados | SQLite en sandbox (solo SELECT) | 2 | ✅ |
| OCR | imágenes y escaneos | Tesseract | 2 | ⏳ |
| Conversión de formatos | CSV/JSON/YAML | biblioteca estándar | 1 | ✅ |

### Nivel 5 — Ciencia e ingeniería

| Herramienta | Para qué | Implementación | Permiso | Estado |
|---|---|---|---|---|
| Simulación de circuitos | verificar análisis eléctricos (DC, AC) | simulador propio MNA en NumPy (ngspice y transitorios ⏳) | 2 | ✅ |
| Filtros y señales | respuesta en frecuencia, corte, polos | NumPy (SciPy `signal` y FFT ⏳) | 2 | ✅ |
| Mecánica | estática, cinemática | SymPy `physics` | 2 | ⏳ |
| Térmica | disipación, temperaturas de juntura | fórmulas + SciPy | 2 | ⏳ |
| Química | moléculas, estequiometría | RDKit (opcional) | 2 | ⏳ |
| G-code y CAD | analizar trayectorias y piezas | parser propio, CadQuery | 2 | ⏳ |
| Control de hardware real | impresoras 3D, equipos | interfaz con límites físicos independientes | **3** | ⏳ (solo con aprobación) |

### Nivel 6 — Investigación externa

| Herramienta | Para qué | Implementación | Permiso | Estado |
|---|---|---|---|---|
| Búsqueda de papers | arXiv, Semantic Scholar | API web | **3** | ⏳ |
| Descarga y lectura de fuentes | citar con procedencia | HTTP + procesamiento de PDF | **3** | ⏳ |

### Nivel 7 — Meta-herramientas (el ecosistema que se mejora a sí mismo)

| Herramienta | Para qué | Implementación | Permiso | Estado |
|---|---|---|---|---|
| Catálogo y búsqueda por objetivo | elegir la herramienta adecuada | HTE (`hope/tools/catalog.py`) | 1 | ✅ |
| Verificación de episodios | aceptar o refutar explicaciones | TVE (`hope/tools/verify.py`) | 1 | ✅ |
| Memoria de uso (TLM) | cuándo conviene cada herramienta, por tema | `hope/tools/tlm.py`, alimentada por el ciclo y el banco; sugiere en el contexto por temas parecidos, con éxito y tiempo (§7.3) | 1 | ✅ |
| Creación de herramientas (TCG) | Hope y Qwen diseñan una herramienta nueva | `hope/tools/creation.py`: especificación + código + pruebas del maestro; queda EXPERIMENTAL con ficha y procedencia; consolidar requiere aprobación humana (guardián, `tool.consolidate`) | 2 (3 para consolidar) | ✅ |
| Validación independiente (TVE) | pruebas que no escribió quien hizo la herramienta | revisión de seguridad por lista permitida, ruff, pruebas propias y casos conocidos de Hope, todo en sandbox con límites | 2 | ✅ |
| Composición (HTO) | encadenar herramientas en flujos | `hope/tools/compose.py`: tuberías lineales con tipos de la ficha y tiempos en la TLM (grafos con ramas ⏳) | 2 | ✅ |
| Optimización (TEO) | detectar herramientas lentas o redundantes | métricas de uso | 2 | ⏳ |

## 4. Cómo cada nivel produce datos de entrenamiento propios

| Herramienta | Datos que genera |
|---|---|
| Calculadora y unidades | millones de problemas aritméticos y físicos con respuesta exacta (generadores) |
| SymPy | ecuaciones, derivadas e integrales con solución verificada |
| z3 | problemas lógicos y de restricciones con respuesta y contraejemplos |
| Python, pytest y compilador | programas cuya salida o cuyas pruebas confirman si el código de Hope es correcto (STaR para código) |
| ngspice y SciPy `signal` | circuitos y filtros cuyo comportamiento simulado valida el análisis de Hope |
| Memoria y documentos | preguntas cuya respuesta exacta está en una fuente con posición |

La regla es siempre la misma: **solo entra al entrenamiento lo que una herramienta independiente confirmó**. Lo no verificable pesa menos, y lo refutado queda como ejemplo de error.

## 5. Orden de implementación propuesto

1. **Protocolo de llamadas dentro del texto** (`⟦herramienta: entrada⟧ → resultado`) en la generación, el guardián y el formato de los episodios. Sin esto, Hope no puede usar herramientas *mientras razona*.
2. **Generadores con llamadas**: que los episodios generados muestren el uso correcto de herramientas, y que STaR acepte soluciones con llamadas verificadas.
3. **Nivel 0–1 completo y z3** (fracciones, estadística, resolver ecuaciones, cálculo, álgebra lineal, SAT/SMT).
4. **Nivel 3** para Hope Code: pytest, ruff, gcc, ripgrep y diff, todo en sandbox.
5. **Nivel 5** para electrónica: ngspice y SciPy `signal`.
6. **TLM**: registro de uso de herramientas y aprendizaje de cuándo usar cada una.
7. ✅ **TCG/TVE y HTO**: creación de herramientas con Qwen, validación independiente, aprobación humana para consolidar y composición de herramientas con tiempos en la TLM.
8. ✅ **Usar bien las herramientas** (§7): banco sellado de uso, currículo con llamadas ejecutadas, sugerencias de la TLM por temas parecidos (éxito y tiempo) y refuerzo con recompensa verificada menos costo.

## 6. Requisitos de seguridad comunes

- Toda herramienta tiene ficha en el catálogo (formato de Herramientas §16), estado de ciclo de vida y nivel de permiso.
- Lo que ejecuta código corre en el sandbox: proceso aparte, sin entorno, con límites de tiempo, memoria y procesos. El bloqueo de red requiere aislamiento del sistema operativo, que se suma con el harness (F9).
- Nada del nivel 3 (red, hardware, acciones externas) se ejecuta sin aprobación humana.
- Una herramienta creada por Hope o por Qwen es código no confiable hasta que pasa la validación independiente.

## 7. Usar bien las herramientas: evaluación, currículo, selección aprendida y refuerzo

**Objetivo (pedido del usuario):** que Hope use **la herramienta correcta, en el momento justo, rápido y verificado**. Tener herramientas no alcanza: hay que medir cómo las usa, enseñarle con ejemplos ejecutados de verdad, sugerirle las que rindieron y reforzar lo que funciona.

### 7.1 Evaluación primero: banco sellado (`hope/eval/tool_bench.py`)

- **110 ítems** (10 por categoría) en `evals/banco_herramientas_v1.jsonl`, con su SHA-256 al lado (`.sha256`). Si el archivo cambia, `load_bench` falla. Con `--registro` también queda en `SealedRegistry` (propósito «evaluation», accesos registrados).
- **Corre con cualquier `generate`**: `HN1Model.generate`, un modelo cuantizado, un servidor o la solución de referencia. Las llamadas se ejecutan de verdad con `generate_with_tools`, el catálogo y el guardián.
- Las categorías y lo que mide cada una:

| Categoría | Qué mide |
|---|---|
| `calculo`, `fraccion`, `estadistica`, `unidades`, `ecuacion`, `circuito` | elegir la herramienta correcta para el tipo de problema |
| `cadena` | encadenar `circuito → calc → unidades` |
| `trivial` | **no** llamar herramientas cuando responder directo es mejor |
| `error` | la primera llamada falla a propósito: hay que reintentar |
| `definicion` | consultar `⟦nota⟧`/`⟦memoria⟧` (conceptos reservados al banco) |
| `codigo` | escribir código y usar `⟦pruebas⟧`; se juzga con **pruebas ocultas** |

- **Métricas:** `herramienta_correcta` (primera llamada), `argumentos_validos` (llamadas que la herramienta aceptó, sin contar fallas inyectadas), `exactitud`, `llamadas_innecesarias` (en lo trivial), `llamadas_faltantes` (cuando hacía falta), `cadena_completa`, `recuperacion` (reintentó tras la falla **y** acertó), `segundos_por_item` y `bytes_generados_por_item` (solo lo que escribe Hope; los resultados de herramientas no cuentan).

```
python -m hope.eval.tool_bench crear --out evals/banco_herramientas_v2.jsonl   # uno nuevo; el v1 ya está en el repo y no se pisa
python -m hope.eval.tool_bench correr --modelo runs/hope/ligero.pt [--tlm herramientas_uso.json]
python -m hope.eval.tool_bench correr --referencia     # cota superior: 100 % en todo
```

### 7.2 Currículo de uso de herramientas (`hope/data/tool_curriculum.py`)

- **Llamadas reales:** cada `⟦nombre: entrada⟧ → resultado` se ejecuta al escribir el ejemplo, y el resultado insertado es el que devolvió la herramienta. Un ejemplo entra solo si la respuesta final coincide con la calculada por separado (`fractions`) o, en código, si pasa las pruebas.
- **Mismas categorías que el banco**, con otra semilla, sin sus problemas y sin sus conceptos del glosario. Además:
  - errores **reales**: unidades dentro de `calc` → cambiar a `unidades`; un paréntesis sin cerrar en `python` → corregir;
  - `⟦anotar⟧` + `⟦nota⟧` para guardar y comprobar un dato;
  - código que **falla primero**, se corrige y se vuelve a probar hasta que pasa;
  - episodios de los **generadores** y de la **cosecha del maestro** (`datos_maestro/episodios_verificados.jsonl`) reescritos con las llamadas ejecutadas; si una herramienta contradice al episodio, el episodio no entra.
- **Mezcla por defecto:** calc 13 %, trivial 14 %, unidades 11 %, error 9 %, cadena 9 %, ecuación 7 %, definición 7 %, código 6 %, fracción 6 %, estadística 6 %, circuito 5 %, generadores 5 %, anotar 2 %. Lo trivial pesa bastante a propósito: **no usar** también se aprende.
- **Exportación** como fuente de entrenamiento (`train.bin`/`val.bin`, documentos separados por el byte 0):

```
python -m hope.data.tool_curriculum --out data/herramientas --n 20000
```

- **En los planes:** `herramientas: data/herramientas:0.10` en la etapa 2 de `configs/plan_gtx1060.yaml`; 0,08, 0,07 y 0,10 en las etapas 4, 5 y 6 de `configs/plan_a770_continuacion.yaml`. Es mucho más que el ~1 % que alcanza para que el refuerzo posterior transfiera (reports/Razonamiento por parámetro para Hope.md §1.4) y no le quita lugar al idioma. Si la carpeta no existe, `hope.train.run` la saltea.
- El formato es el de los episodios: `Tema:` / `Problema:` / (herramientas sugeridas) / pasos con llamadas / `Resultado:`. Así sirve igual en el ciclo con el maestro, en el banco y en el refuerzo.

### 7.3 Selección aprendida: la TLM sugiere en el contexto (`hope/tools/tlm.py`)

- La memoria de uso ya registraba, por tema, usos, éxitos, respuestas correctas y segundos de cada herramienta. Ahora también registra **responder directo** (`record_direct`, que aparece como `ninguna`).
- `rank(tema, problema)` combina el tema y los **temas parecidos** (fracción de las palabras del tema guardado que aparecen en el tema y el enunciado nuevos):
  - puntaje = Σ similitud · (qué tan seguido se usa en ese tema) · utilidad de Laplace / Σ similitud;
  - después se divide por (1 + segundos medios): entre dos igual de exactas, gana la rápida.
- `context_hint(tema, problema)` escribe en el contexto una línea con **formato fijo**, que `parse_hint` lee:

```
Herramientas sugeridas: unidades (éxito 97 %, 0,02 s), calc (éxito 80 %, 0,00 s)
Herramientas sugeridas: ninguna (éxito 100 %, 0,00 s), calc (éxito 100 %, 0,00 s)
```

  Hasta 3 herramientas con puntaje ≥ 0,1, en orden. «éxito» es la fracción de usos con respuesta correcta; los segundos son el tiempo medio real de la herramienta. Si no hay datos parecidos, no se escribe nada.
- **Dónde se usa:**
  - en `TeachingCycle.answer_reviewed`, que antes ponía la lista sin tiempos y solo del tema exacto;
  - en el banco con `--tlm`, que además aprende en línea de cada ítem;
  - en el ~40 % de los ejemplos del currículo, armada con la TLM que se llena con las llamadas reales durante la construcción. Así Hope aprende a leer esa línea.

### 7.4 Refuerzo (`hope/train/tool_rl.py`)

- Usa `grpo_step_tools` (`hope/train/rlvr.py`, sin cambios) sobre problemas del currículo (semilla `rl`, sin los del banco).
- **Recompensa = calidad verificada − costo**:
  - calidad: 1 si acierta con una llamada que funcionó o si el problema era trivial y respondió directo; 0,5 si acertó «a ojo» algo que pedía herramienta; 0 si se equivocó; en código, las pruebas ocultas;
  - costo: bytes propios, con tope 0,5, cobrado **solo si la respuesta es correcta** (`rewarded`). Si se cobra siempre, con la mayoría de las respuestas mal la única señal del grupo es el largo y el modelo aprende a no usar herramientas. Pasó: `docs/resultados/uso_herramientas.md` §4.
- Los problemas con falla inyectada van en lotes aparte, para que el grupo que reintenta gane.
- Mide el banco antes y después.

```
python -m hope.train.tool_rl --prueba                                  # humo en CPU con hn1-tiny (~10 s)
python -m hope.train.tool_rl --modelo runs/hope/ligero.pt --pasos 300 --salida runs/tool_rl
```

- **Cuándo usarlo:** **después** del entrenamiento supervisado con el currículo. Un modelo que nunca vio `⟦…⟧` casi no genera llamadas y los grupos no dan señal. KL por defecto 1·10⁻³ (reports/Razonamiento por parámetro §1.4).

### 7.5 Resultados

Experimento en CPU con y sin el currículo: `docs/resultados/uso_herramientas.md`.

Resumen (Hope-mini 7,9M, 1.500 pasos): con el currículo la exactitud sube de 8 % a 36 %, la herramienta correcta de 37 % a 88 %, las llamadas innecesarias bajan de 100 % a 0 % y la recuperación tras fallas sube de 0 % a 70 %. Queda un cuello de botella: copiar bien los datos del enunciado a la llamada (54 %).

**Otro cambio:** `⟦calc⟧` ahora escribe los números sin notación científica (`30`, no `3E+1`; `1321920`, no `1.32192E+6`). Hope aprende a copiar lo que devuelve la herramienta, así que el formato importa.
