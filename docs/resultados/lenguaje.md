# Escribir mejor sin más parámetros — medición en CPU

Código: `hope/runtime/escritura.py`, `hope/runtime/lengua.py`, `hope/core/paso.py`, `hope/core/contrastiva.py`, `hope/core/knn.py`.
Mediciones: `python -m hope.bench.lenguaje` (técnicas de escritura) y `python -m hope.bench.lengua` (enfoque «como código»).
Datos crudos: `docs/resultados/lenguaje_datos.json` y `docs/resultados/lengua_datos.json`.

## Resumen honesto

- **Los dos checkpoints de CPU escriben mal sin importar la técnica.** El voraz entra en bucles («de la construcción de la construcción…»): rep-4 de 0,82 (mini) y 0,72 (25M). Ninguna técnica de decodificación convierte un modelo de 2,2–2,35 BPB en uno que escriba como uno grande. Lo que sí se consigue es **sacar los bucles**, **bajar mucho el BPB con memoria (kNN-LM)** y **detectar errores conocidos** con verificadores baratos.
- **Lo que funcionó:**
  - **kNN-LM**: el BPB condicional baja un **20 %** (mini: 2,353 → 1,875) y un **17 %** (25M: 2,264 → 1,880), sin parámetros nuevos, con 82 MB de memoria y 0,25 ms por byte con el índice IVF. Con el almacén, **el mini empata al 25M**.
  - **Bucle de escritura y MBR**: rep-4 baja de 0,72–0,82 a **0,00** y la diversidad (distinct-2) sube de ~0,1 a ~0,8. Cuesta 2–3× (MBR) o 10× (bucle) el tiempo del voraz.
  - **Verificadores**: detectan el **90 % (es)** y el **80 % (en)** de los errores introducidos a propósito (detalle abajo).
- **Lo que no funcionó (resultados negativos):**
  - **Cómputo adaptativo**: la parada ACT de los dos checkpoints **colapsó a una iteración**. Pedir más iteraciones no cambia nada, e ignorar la parada cambia el BPB en ±0,003. La señal sí sirve (la cabeza de incertidumbre correlaciona con la pérdida real: Spearman 0,66 en el mini y 0,73 en el 25M), pero no hay profundidad útil en la que gastarla. **Hace falta entrenar con `stable_loop`** (§3.4.1).
  - **Contrastiva**: con el propio modelo de aprendiz (1 iteración) el aprendiz es casi idéntico al experto y la salida degenera («a a a…»). Con el mini de aprendiz del 25M desaparecen los bucles, pero el 89 % de las palabras no existen. Li et al. la evalúan con expertos de más de 1.000 M de parámetros; a esta escala el experto todavía no sabe lo suficiente como para que «restar al aprendiz» ayude.
  - **Ejemplos de estilo**: anteponer un pasaje bien escrito **empeora** el BPB (mini: 2,35 → 2,74; 25M: 2,26 → 2,32). A esta escala no hay aprendizaje en contexto.
  - **Combinar todo** no suma: la técnica combinada tarda 34 s por respuesta y queda peor que el bucle solo, porque arrastra la contrastiva.
- **Presupuesto:** a 25 tokens/s (200 bytes → 2 s × 0,9 de margen), el bucle **nunca tuvo tiempo** de reescribir en esta CPU de 4 núcleos y entregó el borrador. Para que el bucle actúe dentro del presupuesto, hace falta un objetivo de ~5 tokens/s en CPU o la GPU.

## Protocolo

- **Evaluación:** 16 pasajes en español de `datos_maestro/notas_calidad.jsonl` (Wikipedia con nota del maestro), de la partición `eval`: 1 de cada 10 documentos por hash, sin duplicados. Prompt de ~200 bytes cortado en un espacio; la referencia son los 200 bytes siguientes; se generan 200 bytes.
- **Almacén kNN y ejemplos de estilo:** solo la partición `almacen` (281 documentos, 331 kB). Revisé la fuga: de 30 trozos de 40 caracteres por documento de evaluación, casi ninguno aparece en el almacén (0–5, en fichas botánicas de plantilla). La ganancia del kNN es real, aunque en buena parte viene de que almacén y evaluación tienen el **mismo estilo enciclopédico**.
- **Métricas:**
  - **rep-4**: 4-gramas de palabras repetidos dentro de la salida (más bajo es mejor);
  - **n-gramas bits/byte**: fluidez de superficie según un modelo de 5-gramas de bytes del almacén. Más bajo = más parecido al corpus, pero un bucle de palabras frecuentes también da valores bajos, así que se lee junto con rep-4;
  - **palabras desconocidas**: palabras de 4 letras o más que no están en el corpus;
  - **distinct-2**: diversidad de la salida;
  - **BPB condicional** de la referencia, solo cuando la técnica define una distribución;
  - **segundos** por respuesta.
- **Máquina:** CPU de 4 núcleos, 4 hilos, fp32, sin cuantizar.

## Resultados por técnica

### `hn1-cpu-mini` (7,9 M)

| Técnica | rep-4 ↓ | n-gramas bits/byte | desconocidas ↓ | distinct-2 ↑ | BPB cond. ↓ | s/respuesta |
|---|---|---|---|---|---|---|
| voraz | 0,82 | 1,91 | 0,44 | 0,08 | 2,353 | 0,66 |
| voraz 1 iteración | 0,82 | 1,91 | 0,44 | 0,08 | 2,353 | 0,53 |
| voraz 2 iteraciones | 0,82 | 1,91 | 0,44 | 0,08 | 2,353 | 0,67 |
| muestreo T=0,8 top-p 0,95 | 0,00 | 3,11 | 0,63 | 0,90 | — | 1,28 |
| siempre 2 iteraciones completas (sin parada) | 0,76 | 1,60 | 0,43 | 0,09 | 2,354 | 0,74 |
| adaptativo 1→2 completas (67 % de parches profundos) | 0,76 | 1,63 | 0,49 | 0,09 | 2,354 | 0,64 |
| adaptativo 1→4 completas (62 %) | 0,76 | 1,61 | 0,44 | 0,09 | 2,355 | 0,75 |
| MBR N=4 (chrF + verificadores) | 0,00 | 1,88 | 0,60 | 0,85 | — | 1,52 |
| contrastiva (aprendiz: sí mismo, 1 iteración) | 0,61 | 4,21 | 0,45 | 0,08 | — | 1,61 |
| **kNN-LM λ=0,4** | 0,69 | 1,41 | **0,19** | 0,17 | **1,875** | 0,72 |
| ejemplos de estilo | 0,19 | 1,88 | 0,56 | 0,54 | 2,736 | 0,67 |
| bucle de escritura (MBR + 2 rondas) | 0,00 | 2,06 | 0,49 | 0,80 | — | 6,87 |
| bucle de escritura a 25 tok/s | 0,77 | 1,95 | 0,40 | 0,11 | — | 0,67 |
| combinada (estilo + contrastiva + kNN + adaptativo + bucle) | 0,00 | 2,83 | 0,67 | 0,93 | — | 34,54 |
| contrastiva + kNN (voraz) | 0,35 | 1,49 | **0,11** | 0,35 | — | 1,28 |

### `hn1-25m-cpu` (25 M)

| Técnica | rep-4 ↓ | n-gramas bits/byte | desconocidas ↓ | distinct-2 ↑ | BPB cond. ↓ | s/respuesta |
|---|---|---|---|---|---|---|
| voraz | 0,72 | 2,23 | 0,58 | 0,11 | 2,264 | 0,82 |
| voraz 1 iteración | 0,72 | 2,23 | 0,58 | 0,11 | 2,264 | 0,68 |
| voraz 2 iteraciones | 0,72 | 2,23 | 0,58 | 0,11 | 2,264 | 0,75 |
| muestreo T=0,8 top-p 0,95 | 0,00 | 2,93 | 0,60 | 0,92 | — | 0,83 |
| siempre 2 iteraciones completas (sin parada) | 0,77 | 2,24 | 0,58 | 0,09 | 2,262 | 1,02 |
| adaptativo 1→2 completas (70 %) | 0,76 | 2,24 | 0,57 | 0,09 | 2,261 | 0,99 |
| adaptativo 1→4 completas (69 %) | 0,76 | 2,22 | 0,50 | 0,09 | 2,265 | 1,03 |
| MBR N=4 (chrF + verificadores) | 0,00 | 1,89 | 0,55 | 0,79 | — | 2,26 |
| contrastiva (aprendiz: sí mismo, 1 iteración) | 0,40 | 3,38 | 0,62 | 0,04 | — | 2,64 |
| contrastiva (aprendiz: hn1-cpu-mini) | 0,00 | 4,83 | 0,89 | 0,97 | — | 1,39 |
| **kNN-LM λ=0,4** | 0,56 | **1,39** | 0,40 | 0,15 | **1,880** | 1,02 |
| ejemplos de estilo | 0,75 | 2,17 | 0,67 | 0,10 | 2,319 | 0,89 |
| **bucle de escritura (MBR + 2 rondas)** | **0,00** | 2,07 | **0,39** | 0,81 | — | 9,35 |
| bucle de escritura a 25 tok/s | 0,72 | 2,23 | 0,58 | 0,11 | — | 0,73 |
| combinada (estilo + contrastiva + kNN + adaptativo + bucle) | 0,00 | 2,43 | 0,47 | 0,92 | — | 34,44 |
| contrastiva + kNN (voraz) | 0,00 | 2,75 | 0,48 | 0,78 | — | 1,64 |

**Muestras** (25M, primer caso, «Ascomycota»):

- **Voraz:** «la provincia de la construcción de los procesos de la construcción de la construcción de la construcción…»
- **kNN λ=0,4:** «la internación de la construcción de la construcción de la construcción…»
- **MBR:** «la división de los directos tienen consistentes son las corportaciones de la conversión de la Construcción Agr…»
- **Bucle de escritura:** «explicación de las acciones históricas de los tiempos de la situación más tiempo haber de los siglos providera…»

## kNN-LM: costo y efecto de λ

| | mini | 25M |
|---|---|---|
| Claves (estado del decodificador → byte siguiente) | 330.412 | 330.412 |
| Dimensión (PCA) | 64 | 64 |
| Construcción | 19 s | 31 s |
| Memoria, búsqueda exacta | 82 MB | 82 MB |
| Memoria, con índice IVF (2.299 listas) | 168 MB | 168 MB |
| Búsqueda exacta (k=16) | 2,8 ms/byte | 3,0 ms/byte |
| Búsqueda IVF (nprobe=8) | 0,25 ms/byte | 0,27 ms/byte |
| BPB λ = 0 / 0,1 / 0,25 / 0,4 | 2,353 / 2,095 / 1,952 / **1,875** | 2,264 / 2,067 / 1,944 / **1,880** |

- La mejora sigue creciendo hasta λ = 0,4, el máximo que probé: con un modelo tan chico, la memoria sabe más que los pesos sobre este dominio.
- Con IVF, generar cuesta ~10–25 % más que el voraz.
- **Límite:** el kNN **no saca los bucles** del voraz (rep-4 0,56–0,69). Para eso hace falta combinarlo con selección (MBR o el bucle de escritura) o con anti-repetición (`hope/core/decoding.py`, otra sesión, que se enchufa como `ProcesadorLogits`).

## Cómputo adaptativo

Tanto `forward(n_iters=1)` como `n_iters=4` dan `halt_dist = [1.0]`: la parada ACT corta en la primera iteración en el 100 % de los parches. Por eso `generate(n_iters=k)` da lo mismo para cualquier k.

Para que el modo profundo sirva, `ComputoAdaptativo(ignorar_parada=None)` usa por defecto `full_depth` con núcleos ACT: el estado de la última iteración, que es lo que el entrenamiento usa en la mitad de los pasos (`full_depth_frac=0,5`). Así sigue dentro de lo entrenado. Aun así:

| BPB en 8 textos de evaluación (forward completo) | mini | 25M |
|---|---|---|
| ACT (lo normal) | 2,3778 | 2,2041 |
| todas las iteraciones, 2 | 2,3749 | 2,2028 |
| todas las iteraciones, 3 | 2,3797 | 2,2062 |
| todas las iteraciones, 4 | 2,3769 | 2,2070 |

- La diferencia está en la tercera cifra decimal: las iteraciones 2–4 casi no aprendieron nada.
- La mecánica funciona y está probada: los parches inciertos (62–70 % con el umbral en la mediana) van a la caché profunda, que se pone al día en una sola pasada; con umbral −∞ da exactamente `generate(n_iters=max)`. El resultado depende de un núcleo entrenado con `stable_loop`.

## Resumen continuo de la conversación

- `ConversacionConResumen` da lo mismo que `Conversation` mientras el historial entra en la ventana (prueba).
- Cuando no entra, reinicia desde «Resumen de la conversación hasta ahora: … + últimos turnos». El resumen es extractivo (sin modelo: las oraciones con las palabras más frecuentes de la conversación) y se guarda en el cuaderno como la nota «resumen de la conversación <id>», con historial de versiones.
- En la pestaña Chatear: casilla «recordar conversaciones largas (resumen continuo en el cuaderno)». La traza muestra el resumen y cuántas veces se compactó.
- No medí calidad de recuerdo con usuarios: con estos checkpoints, la respuesta no usa el resumen de forma útil. Lo que sí garantiza es que lo dicho al principio **sigue en el contexto**, en lugar de perderse con la ventana deslizante.

<!-- PARTE_LENGUA -->
