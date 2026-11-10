# Escribir mejor sin más parámetros — medición en CPU

Código: `hope/runtime/escritura.py`, `hope/runtime/lengua.py`, `hope/core/paso.py`, `hope/core/contrastiva.py`, `hope/core/knn.py`, `hope/core/decoding.py`, `hope/tools/lenguaje.py`, `hope/data/ejercicios_lengua.py`.

Mediciones (todas con los checkpoints integrados de la rama principal al 10/11: `hn1-cpu-mini` 7,9 M y `hn1-25m-cpu` 25 M):

- `python -m hope.bench.lenguaje` (técnicas de escritura) → `lenguaje_datos.json`
- `python -m hope.bench.lengua` (enfoque «como código») → `lengua_datos.json`
- `python -m hope.bench.repeticion` (antirrepetición) → `repeticion_datos.json`
- tiempos de kNN con la CPU libre → `knn_tiempos.json`

<!-- RESUMEN -->

## Protocolo

- **Evaluación:** 16 pasajes en español de `datos_maestro/notas_calidad.jsonl` (Wikipedia con nota del maestro), de la partición `eval`: 1 de cada 10 documentos por hash, sin duplicados. Prompt de ~200 bytes cortado en un espacio; la referencia son los 200 bytes siguientes; se generan 200 bytes.
- **Almacén kNN y ejemplos de estilo:** solo la partición `almacen`. Revisé la fuga: de 30 trozos de 40 caracteres por documento de evaluación, casi ninguno aparece en el almacén (0–5, en fichas botánicas de plantilla). La ganancia del kNN es real, aunque en buena parte viene de que almacén y evaluación tienen el **mismo estilo enciclopédico**.
- **Métricas:**
  - **rep-4**: 4-gramas de palabras repetidos dentro de la salida (más bajo es mejor);
  - **n-gramas bits/byte**: fluidez de superficie según un modelo de 5-gramas de bytes del almacén. Un bucle de palabras frecuentes también da valores bajos, así que se lee junto con rep-4;
  - **palabras desconocidas**: palabras de 4 letras o más que no están en el corpus;
  - **distinct-2**: diversidad de la salida;
  - **BPB condicional** de la referencia, solo cuando la técnica define una distribución;
  - **segundos** por respuesta.
- **Máquina:** CPU de 4 núcleos, 4 hilos, fp32, sin cuantizar.

## 1. Técnicas de escritura

### `hn1-cpu-mini` (7,9 M)

| Técnica | rep-4 ↓ | n-gramas bits/byte | desconocidas ↓ | distinct-2 ↑ | BPB cond. ↓ | s/respuesta |
|---|---|---|---|---|---|---|
| voraz | 0,80 | 1,74 | 0,24 | 0,08 | 2,222 | 0,61 |
| voraz 1 iteración | 0,80 | 1,74 | 0,24 | 0,08 | 2,222 | 0,50 |
| voraz 2 iteraciones | 0,80 | 1,74 | 0,24 | 0,08 | 2,222 | 0,54 |
| muestreo T=0,8 top-p 0,95 | 0,00 | 2,93 | 0,59 | 0,85 | — | 0,75 |
| siempre 2 iteraciones completas (sin parada) | 0,74 | 1,90 | 0,14 | 0,07 | 2,223 | 0,91 |
| adaptativo 1→2 completas (55 % de parches profundos) | 0,73 | 1,94 | 0,17 | 0,08 | 2,222 | 0,76 |
| adaptativo 1→4 completas (62 %) | 0,79 | 1,68 | 0,26 | 0,08 | 2,222 | 0,74 |
| MBR N=4 (chrF + verificadores) | 0,00 | 2,21 | 0,51 | 0,78 | — | 2,07 |
| contrastiva (aprendiz: sí mismo, 1 iteración) | 0,35 | 4,65 | 0,66 | 0,07 | — | 2,13 |
| **kNN-LM λ=0,4** | 0,82 | **1,26** | **0,10** | 0,12 | **1,842** | 0,78 |
| ejemplos de estilo | 0,25 | 1,98 | 0,57 | 0,41 | 2,575 | 0,64 |
| **bucle de escritura (MBR + 2 rondas)** | **0,07** | 1,96 | 0,36 | 0,68 | — | 6,11 |
| bucle de escritura a 25 tok/s | 0,74 | 1,83 | 0,26 | 0,09 | — | 0,68 |
| combinada (estilo + contrastiva + kNN + adaptativo + bucle) | 0,00 | 2,62 | 0,56 | 0,91 | — | 26,96 |
| contrastiva + kNN (voraz) | 0,43 | 1,31 | **0,05** | 0,36 | — | 1,26 |

### `hn1-25m-cpu` (25 M)

| Técnica | rep-4 ↓ | n-gramas bits/byte | desconocidas ↓ | distinct-2 ↑ | BPB cond. ↓ | s/respuesta |
|---|---|---|---|---|---|---|
| voraz | 0,87 | 2,16 | 0,42 | 0,08 | 1,988 | 0,82 |
| voraz 1 iteración | 0,87 | 2,16 | 0,42 | 0,08 | 1,988 | 0,76 |
| voraz 2 iteraciones | 0,87 | 2,16 | 0,42 | 0,08 | 1,988 | 0,74 |
| muestreo T=0,8 top-p 0,95 | 0,00 | 2,60 | 0,45 | 0,88 | — | 0,84 |
| siempre 2 iteraciones completas (sin parada) | 0,86 | 2,11 | 0,36 | 0,08 | 1,984 | 0,99 |
| adaptativo 1→2 completas (74 %) | 0,86 | 2,11 | 0,36 | 0,08 | 1,984 | 0,98 |
| adaptativo 1→4 completas (74 %) | 0,86 | 2,13 | 0,31 | 0,08 | 1,985 | 1,07 |
| **MBR N=4 (chrF + verificadores)** | **0,00** | **1,85** | **0,29** | 0,80 | — | 2,27 |
| contrastiva (aprendiz: sí mismo, 1 iteración) | 0,63 | 2,35 | 0,47 | 0,04 | — | 2,34 |
| contrastiva (aprendiz: hn1-cpu-mini) | 0,11 | 4,19 | 0,69 | 0,80 | — | 1,78 |
| **kNN-LM λ=0,4** | 0,72 | **1,70** | **0,21** | 0,13 | **1,672** | 0,98 |
| ejemplos de estilo | 0,81 | 2,39 | 0,55 | 0,12 | 2,027 | 0,92 |
| bucle de escritura (MBR + 2 rondas) | 0,05 | 2,00 | 0,30 | 0,77 | — | 7,28 |
| bucle de escritura a 25 tok/s | 0,87 | 2,16 | 0,42 | 0,08 | — | 0,76 |
| combinada (estilo + contrastiva + kNN + adaptativo + bucle) | 0,00 | 2,14 | 0,32 | 0,90 | — | 30,35 |
| contrastiva + kNN (voraz) | 0,13 | 2,23 | 0,30 | 0,65 | — | 1,55 |

**Muestras** (25M, primer caso):

- **Voraz:** «el contenido de la producción de la producción de la producción de la producción…»
- **kNN λ=0,4:** «la construcción de la familia de la región de la región de la región de la construcción…»
- **MBR:** «la división de los de los años con la familia de la base de los remostres pueden hacer el régimen de un composición prop…»
- **Bucle de escritura:** «el contrario del perú en la tabla de las producciones tablas en cada producción de energía en caso, que entre este estad…»
- **Contrastiva + kNN:** «los nuevos microsporancionados de la familia de las plantas son una región de divisas o anuales, tal que se refiere a lo…»

### kNN-LM: costo y efecto de λ

| | mini | 25M |
|---|---|---|
| Claves (estado del decodificador → byte siguiente) | 400.000 | 400.000 |
| Dimensión (PCA) | 64 | 64 |
| Memoria, búsqueda exacta / con índice IVF | 100 MB / 203 MB | 100 MB / 203 MB |
| Búsqueda exacta / IVF (k=16, nprobe=8), CPU libre | <!-- KNN_MINI --> | <!-- KNN_25M --> |
| BPB λ = 0 / 0,1 / 0,25 / 0,4 | 2,222 / 2,022 / 1,903 / **1,842** | 1,988 / 1,814 / 1,718 / **1,672** |

- Baja el BPB un **17 %** (mini) y un **16 %** (25M), sin parámetros nuevos. **Con el almacén, el mini (1,84) queda mejor que el 25M sin almacén (1,99).**
- La mejora sigue creciendo hasta λ = 0,4, el máximo que probé.
- **Límite:** el kNN no saca los bucles del voraz (rep-4 0,72–0,82). Para eso se combina con antirrepetición (§3) o con selección (MBR o el bucle de escritura).

### Cómputo adaptativo

La parada ACT de los dos checkpoints corta en la **primera iteración** en el 100 % de los parches: `halt_dist = [1.0]`, así que `generate(n_iters=k)` da lo mismo para cualquier k (las filas «1 iteración» y «2 iteraciones» son idénticas al voraz).

`ComputoAdaptativo` ignora la parada en el modo profundo (`full_depth`): es lo que el entrenamiento usa en la mitad de los pasos, así que no sale de lo entrenado. Aun así, el BPB cambia en la tercera cifra decimal (25M: 1,988 → 1,984), porque las iteraciones 2–4 casi no aprendieron nada.

La señal sí sirve: la cabeza de incertidumbre correlaciona con la pérdida real (Spearman 0,67 en el mini y 0,75 en el 25M). Falta un núcleo con profundidad útil en la que gastarla (`stable_loop`, §3.4.1 de la arquitectura).

### Resumen continuo de la conversación

- `ConversacionConResumen` da lo mismo que `Conversation` mientras el historial entra en la ventana (prueba).
- Cuando no entra, sigue desde «Resumen de la conversación hasta ahora: … + últimos turnos». El resumen es extractivo y queda en el cuaderno como la nota «resumen de la conversación <id>», con historial de versiones.
- En la pestaña Chatear: casilla «recordar conversaciones largas».
- No medí cuánto recuerda en la práctica: con estos checkpoints, la respuesta no aprovecha el resumen. Lo que sí garantiza es que lo dicho al principio **sigue en el contexto**.

## Estado (pausa del 10/11)

Medición **en pausa** a pedido del usuario. Lo que falta, para retomar:

1. `python -m hope.bench.lengua --pasos 300`: las partes A y B del mini ya están en `lengua_datos.json` (parcial); faltan el bucle con requisitos del 25M, la parte C del 25M y el ajuste fino de trazas y FIM (parte D).
2. Tiempos de kNN con la CPU libre: `knn_tiempos.py`. Los de `lenguaje_datos.json` para el mini están inflados porque corrieron en paralelo con las pruebas.
3. `python -m hope.bench.repeticion`: antirrepetición (decoding.py) en los dos checkpoints y la comparación con y sin unlikelihood.
4. Completar el resumen honesto y las secciones 2 y 3 de este informe con esos datos.

Resultados parciales ya medidos (enfoque «como código», mini):

- **Mutaciones**: todas las comprobaciones juntas detectan el 90 % (es) y el 78 % (en) de los errores introducidos, con 12 % de falsas alarmas. Sin la concordancia: 80 % y 0 %.
- **Requisitos sobre lo generado**: voraz 0,62, MBR 0,61, bucle 0,61. Ningún caso cumple todos los requisitos: el modelo no logra cubrir las palabras pedidas ni el largo.
- **Concordancia** (errores cada 100 palabras): bucle 1,09 → bucle con requisitos y concordancia 0,34.
- **Relleno del medio por puntaje**: chrF 0,139 → 0,152 frente al voraz desde el prefijo.

