# Uso de herramientas: con y sin currículo (CPU, Hope-mini 7,9M)

**Pregunta:** ¿el currículo de herramientas (`hope/data/tool_curriculum.py`) hace que Hope use mejor las herramientas que los datos que ya tenía (episodios de los generadores con llamadas)?

**Respuesta corta:** sí, y bastante, en **cuándo** y **cuál** herramienta usar. Con el currículo:
- deja de llamar herramientas en lo trivial (100 % → 0 %);
- elige la herramienta correcta mucho más seguido (37 % → 88 %);
- reintenta tras una falla (0 % → 70 %);
- la exactitud final sube de 8 % a 36 %.

El cuello de botella que queda es **copiar bien los datos del enunciado a la llamada**: con 7,9M parámetros y 1.500 pasos, la mitad de las primeras llamadas lleva números que no están en el problema.

El refuerzo (`hope/train/tool_rl.py`) en su primera versión **empeoró** al modelo (ver §4); se corrigió y se midió de nuevo.

## 1. Montaje

- **Modelo de partida:** Hope-mini (`checkpoints/hn1-cpu-mini.pt`, preset `hn1-cpu`, 7,9M), preentrenado ~1 h en CPU con Wikipedia y código.
- **Entrenamiento:** el mismo para las dos variantes. Muon, lr 3·10⁻⁴ con coseno, warmup 50, lote 8 × 512 bytes, **1.500 pasos** (~18–19 min por variante en 4 núcleos), fp32.
- **Datos** (`python -m hope.bench.tool_use_experiment --pasos 1500`):

  | Variante | Mezcla | Textos | Bytes |
  |---|---|---|---|
  | **sin** | episodios de los 10 generadores con llamadas (`Episode.training_text(tool_calls=True)`), niveles 1–2 | 5.946 | 2,29 MB |
  | **con** | 50 % lo mismo + 50 % currículo de herramientas | + 5.142 | + 1,29 MB |

  El currículo pedido era de 6.000 ejemplos. Quedaron 5.142 porque `circuito` y `cadena` se quedaron sin problemas distintos (solo había ~100 divisores posibles). Ya se amplió a ~2.000 circuitos.
- **Banco:** `evals/banco_herramientas_v1.jsonl`, sellado (SHA-256), 110 ítems (10 por categoría). Ninguno de sus problemas ni de sus conceptos de glosario está en los datos de entrenamiento. Generación voraz, hasta 700 bytes y 6 llamadas.
- **«con + TLM»:** el mismo modelo «con», con la línea `Herramientas sugeridas: …` de la memoria de uso armada durante la construcción del currículo. La TLM sigue aprendiendo en línea durante el banco.

## 2. Resultados del banco

| Métrica | base | sin | **con** | con + TLM | referencia |
|---|---|---|---|---|---|
| exactitud | 0 % | 8,2 % | **36,4 %** | 33,6 % | 100 % |
| herramienta correcta (1.ª llamada) | 0 % | 37 % | **88 %** | 90 % | 100 % |
| argumentos válidos (la herramienta los aceptó) | — | 96,6 % | **99,3 %** | 95,4 % | 100 % |
| argumentos del enunciado (1.ª llamada) | — | 27,7 % | **53,9 %** | 46,1 % | 100 % |
| llamadas innecesarias (en lo trivial) | 0 % | 100 % | **0 %** | 10 % | 0 % |
| llamadas faltantes (cuando hacían falta) | 100 % | 5 % | **1 %** | 1 % | 0 % |
| cadena completa (circuito → calc → unidades) | 0 % | 0 % | **50 %** | 50 % | 100 % |
| recuperación tras una falla | 0 % | 0 % | **70 %** | 60 % | 100 % |
| llamadas por ítem | 0 | 1,34 | 1,31 | 1,38 | 1,13 |
| bytes generados por ítem | 700 | 197 | 227 | 217 | 114 |
| segundos por ítem | (20,9)\* | 0,58 | 0,74 | 0,74 | 0,06 |

\* El modelo base se midió mientras corrían otras pruebas en la misma CPU, así que su tiempo no es comparable. Además nunca escribe `Resultado:` y agota los 700 bytes.

«Argumentos del enunciado» se calculó después, con los textos guardados, cuando ya se había agregado al banco (`grounded_args`).

**Por categoría** (exactitud / herramienta correcta):

| Categoría | sin | con | con + TLM |
|---|---|---|---|
| calculo | 0 / 1,0 | 0,5 / 1,0 | 0,5 / 1,0 |
| fraccion | 0 / 0 | **0,9 / 1,0** | 0,9 / 1,0 |
| estadistica | 0 / 0 | 0 / 0,9 | 0,1 / 1,0 |
| unidades | 0,1 / 1,0 | 0,3 / 1,0 | 0,2 / **0,2** |
| ecuacion | 0,3 / 0 | 0,8 / 0,5 | **1,0 / 1,0** |
| circuito | 0,1 / 1,0 | 0,1 / 1,0 | 0 / 1,0 |
| cadena | 0 / 0 | 0 / 1,0 | 0 / 1,0 |
| trivial | 0,4 / — | 0,3 / — | 0,3 / — |
| error | 0 / 0,7 | **0,7 / 1,0** | 0,6 / 0,9 |
| definicion | 0 / 0 | 0 / 0,4 | 0 / **0,9** |
| codigo | 0 / 0 | **0,4 / 1,0** | 0,1 / 1,0 |

## 3. Lectura honesta

**Lo que el currículo sí enseñó (decisiones):**
- **No usar herramientas en lo trivial.** «sin» llama a `calc` en el 100 % de los ítems triviales (gasta 220 bytes donde alcanzaban 15). «con» responde directo en todos, con 15 bytes. Que acierte solo el 30 % de esas cuentas de una cifra es un problema de aritmética del modelo, no de decisión.
- **Elegir la herramienta.** «sin» solo conoce `calc`, `unidades` y `python` (lo que traen los episodios) y usa `calc` para todo. «con» usa `fraccion`, `estad`, `resolver`, `pruebas` y `nota` donde corresponde (88 %).
- **Reintentar.** Ningún dato de «sin» mostraba una falla; «con» reintenta y acierta en 7 de 10.
- **Código con pruebas.** 4 de 10 funciones pasan pruebas ocultas que Hope nunca vio.

**Lo que no alcanzó (copiar y encadenar):**
- **Copiar los datos del enunciado.** Ejemplo real de «con»: el problema dice 680 ohm y Hope escribe `⟦unidades: 33 V / (470 ohm) en A⟧`. Elige bien la herramienta, pero con valores de otros ejemplos. Solo el 54 % de las primeras llamadas usa los números del enunciado. Por eso `unidades` y `calculo` tienen la herramienta correcta al 100 % y la exactitud en 30–50 %. Es la limitación esperable de un modelo de bytes de 7,9M con 1.500 pasos. Copiar exacto mejora con escala y más pasos. Es lo próximo a medir en la 1060.
- **Cadenas.** Arranca bien (`circuito`) en el 100 %, pero en el paso 2 copia mal el número que devolvió el simulador o mezcla la fórmula (I = V/R en vez de P = V²/R). Ninguna cadena terminó con el resultado exacto.
- **Definiciones nuevas.** Usa `⟦nota⟧` (40 %, 90 % con la TLM), pero consulta un concepto equivocado: los conceptos del banco nunca se vieron, y el modelo no copia el nombre del concepto del enunciado.

**La sugerencia de la TLM (§8.3), mixta:**
- **Ayudó** donde la decisión era el problema: definiciones (herramienta correcta 40 % → 90 %), ecuaciones (50 % → 100 %, exactitud 80 % → 100 %), estadística (90 % → 100 %).
- **Empeoró** `unidades` (100 % → 20 %). La línea era correcta (`unidades` primero, `circuito` segundo, por el parecido con «divisores de tensión y corriente»), pero un modelo de 7,9M reacciona a que `circuito` *aparezca* en la línea y arranca la plantilla de cadena. También aparecieron llamadas innecesarias en lo trivial (10 %) cuando la línea nombraba `calc` en segundo lugar.
- **Conclusión:** la sugerencia sirve, pero conviene mostrar solo la primera herramienta cuando la segunda tiene mucho menos puntaje. Vale la pena repetir la medición con un modelo más grande antes de activarla por defecto en el banco. En el ciclo con el maestro ya estaba activa (antes, como lista sin tiempos).

## 4. Refuerzo (`hope/train/tool_rl.py`) sobre el modelo «con»

Las dos corridas de esta sección usaron `rlvr.py` **antes** de integrar la rama `bucle-GRPO`, que cambia la normalización a Dr. GRPO (ventaja centrada y suma ÷ `max_new`). `tool_rl` ya usa la versión nueva, con `w=None` y la recompensa de `rewarded`. Conviene repetir la medición con ella.

**Primera versión** (costo de bytes y tiempo siempre, como `grpo_step_tools` por defecto): 40 pasos, lote 4, grupo 4, lr 2·10⁻⁵, KL 10⁻³, 200 bytes.

| | exactitud | herramienta correcta | llamadas faltantes | bytes por ítem |
|---|---|---|---|---|
| antes | 30 % | 82 % | 7 % | 132 |
| después | **3,6 %** | **12 %** | **88 %** | 36 |

**El refuerzo colapsó:** Hope aprendió a no usar herramientas y responder corto.
- **Causa:** con la mayoría de las respuestas mal, todas tenían calidad 0 y la única diferencia dentro de cada grupo era el costo. GRPO premió lo más corto, y lo más corto es no llamar a nada. La recompensa media pasó de 0,37 (paso 11) a −0,02, y el uso de herramientas de 0,63 a 0.
- Es exactamente lo que advierten reports/Razonamiento por parámetro §1.4 y Fatemi et al.: la penalización de costo va en una fase posterior y sobre problemas resolubles.
- La exactitud «antes» es 30 % y no 36 % porque esta corrida limita la respuesta a 200 bytes (el código queda cortado).

**Corrección:** el costo se cobra **solo si la respuesta es correcta** (`rewarded`): recompensa = calidad · (1 − costo) si acierta y 0 si no. Así una respuesta incorrecta corta no le gana a una incorrecta que llamó a la herramienta. Lo que queda de señal sale de los aciertos.

**Segunda corrida, con la corrección** (mismos 40 pasos y parámetros):

| | exactitud | herramienta correcta | argumentos del enunciado | llamadas innecesarias | llamadas faltantes | recuperación | bytes por ítem |
|---|---|---|---|---|---|---|---|
| antes | 30 % | 82 % | 53,4 % | 0 % | 7 % | 70 % | 132 |
| después | 28,2 % | 85 % | 53,6 % | **60 %** | 11 % | **40 %** | 126 |

- **Ya no colapsa:** la recompensa media sube de 0,12 a ~0,5 en los pasos con señal, y el uso de herramientas se mantiene.
- **Tampoco mejora el banco:** 40 pasos × 4 problemas son 160 problemas, y en 110 ítems ±4 puntos es ruido.
- **Las llamadas innecesarias suben de 0 % a 60 %.** El refuerzo aprendió a llamar a `calc` en las cuentas de una cifra. Para *este* modelo es lo racional: su aritmética propia acierta el 30 % de esas cuentas y con `calc` la respuesta sale verificada. El costo de ~30 bytes (−0,03) no alcanza para compensarlo.
  - El banco cuenta como «innecesaria» una llamada que un modelo más capaz no necesitaría.
  - Es la tensión real entre «rápido» y «verificado»: con un modelo chico, gana verificar.
  - Si se quiere que responda directo lo trivial, hay que subir el costo por byte en lo trivial o medir la aritmética propia primero, no forzarlo con el banco.
- **La recuperación baja (70 % → 40 %)** con solo ~10 % de problemas con falla en el lote. Hay pocos ejemplos de reintento para que el refuerzo los sostenga. Conviene subir esa fracción.
- **Conclusión:** a esta escala el refuerzo no agrega nada medible sobre el currículo, que es lo que coincide con la evidencia (CMU 100M y L20 135M en reports §1.4). El script queda listo para la 1060/A770, con el costo condicionado a acertar.

## 5. Qué hacer con esto

1. **Entrenar con el currículo** en los planes reales (ya está como fuente `herramientas` al 10 % en la etapa 2 de la 1060 y al 7–10 % en la A770). Es lo que más movió las decisiones de uso.
2. **Medir con el banco** cada checkpoint: `python -m hope.eval.tool_bench correr --modelo runs/hope/ligero.pt`. La métrica a mirar con modelos más grandes es `argumentos_del_enunciado`, porque hoy es el cuello de botella.
3. **Refuerzo solo después**, con el costo condicionado a acertar, y solo sobre problemas que el modelo resuelve a veces (pass@1 entre ~0,1 y ~0,8).
4. **TLM:** mostrar solo herramientas con puntaje cercano al mejor; volver a medir con más parámetros.

## 6. Cómo reproducir

```
python -m hope.bench.tool_use_experiment --pasos 1500 --salida runs/exp_herramientas   # ~1 h 15 min en 4 núcleos
python -m hope.train.tool_rl --modelo runs/exp_herramientas/con.pt --pasos 40 --problemas 300 --lr 2e-5 --salida runs/tool_rl
```

Resúmenes completos en `docs/resultados/uso_herramientas.json`.
