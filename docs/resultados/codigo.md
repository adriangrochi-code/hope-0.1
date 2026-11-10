# Capacidad de programar — medición en CPU (informe preliminar)

**Estado:** las mediciones del banco están completas para el checkpoint actual. El experimento de ajuste con trazas + FIM se midió solo con el checkpoint anterior: la corrida con el actual quedó **pendiente** porque la tarea se pausó.

Datos completos: `docs/resultados/codigo.json`. Código: `docs/HERRAMIENTAS_HN1.md` §10.

## Configuración

- **Modelo:** `checkpoints/hn1-25m-cpu.pt`, hn1-25m de 25 M parámetros.
  - **Actual:** paso ~7.700, después del enfriamiento WSD.
  - **Anterior:** paso ~3.733. Se midió antes de que la integración trajera el actual.
- **Banco:** `hope/bench/codigo.py`, 40 problemas propios estilo HumanEval/MBPP en español.
  - Cada problema tiene 2 ejemplos públicos y 3 a 6 pruebas ocultas.
  - Las salidas esperadas se calcularon ejecutando la solución de referencia.
- **Prompt:** firma + docstring con los ejemplos (estilo HumanEval).
- **Muestreo:** temperatura 0,8 y hasta 300 bytes nuevos por intento.
- **pass@k:** estimador insesgado de Chen et al. 2021.
- **Máquina:** CPU de 4 núcleos, con dos procesos en paralelo de 2 hilos cada uno. Los tiempos están inflados por esa competencia.
- **Métricas:**
  - «sintaxis válida»: el código compila.
  - «válida con cuerpo»: compila **y** la función tiene algo más que la firma y el docstring. Una función vacía compila pero no hace nada.

## Resultados — checkpoint actual

### 40 problemas × 5 muestras

| Modo | pass@1 | pass@5 | sintaxis válida | válida con cuerpo | pruebas ocultas | s/problema | tokens/problema |
|---|---|---|---|---|---|---|---|
| plano | 0 % | 0 % | 31 % | 26 % | 0,3 % | 10,2 | 318 |
| gramática | 0 % | 0 % | 99,5 % | 62 % | 0,7 % | 11,4 | 335 |

### 20 problemas × 2 muestras (los cuatro modos)

| Modo | pass@1 | pass@2 | sintaxis válida | válida con cuerpo | pruebas ocultas | s/problema | tokens/problema |
|---|---|---|---|---|---|---|---|
| plano | 0 % | 0 % | 25 % | 17,5 % | 0 % | 4,5 | 133 |
| gramática | 0 % | 0 % | 100 % | 75 % | 1,3 % | 4,2 | 134 |
| autodepuración | 0 % | 0 % | 22,5 % | 20 % | 1,3 % | 17,0 | 437 |
| autodepuración + gramática | 0 % | 0 % | 100 % | 70 % | 2,6 % | 17,5 | 437 |

## Resultados — checkpoint anterior (paso ~3.733)

| Modo (40 × 5) | pass@1 | pass@5 | sintaxis válida | pruebas ocultas | s/problema |
|---|---|---|---|---|---|
| plano | 0 % | 0 % | 2 % | 0 % | 22,9 |
| gramática | 0 % | 0 % | 100 % (con cuerpo: ~43 % en una re-medición de 20 × 2) | 1,4 % | 9,3 |

Con 20 × 2, la autodepuración dio 0 % de pass@k, con y sin gramática.

### Ajuste corto con trazas + FIM (checkpoint anterior)

- **Datos:** 300 pasos, lote 8 × 512 bytes, `fim_rate` 0,5, 3.040 programas trazados para entrenar.
- **Evaluación:** sobre 26 programas nuevos.

| | antes | después |
|---|---|---|
| BPB en trazas nuevas | 3,66 | 0,23 |
| predecir la salida (exacta) | 0 % | 0 % |
| completar una línea con FIM (exacta) | 0 % | 0 % |

## Lectura honesta

1. **A 25 M parámetros, Hope todavía no resuelve ningún problema** (pass@k = 0 en todos los modos).
   - El código generado es «código de aspecto Python» sin relación con el enunciado: `self.`, `dtype`, nombres inventados.
   - Las pocas pruebas ocultas que pasan son casuales: una función que devuelve `None` donde se esperaba `None`.
   - Ninguna técnica de inferencia compensa un modelo que no sabe programar todavía. Están para multiplicar lo que el modelo ya sabe, y van a rendir con el 200 M y con más datos de código.
2. **La gramática hace lo que promete.**
   - La sintaxis válida pasa del 31 % al 99,5–100 %, y la válida con cuerpo del 26 % al 62–75 %.
   - El costo en tiempo es igual o apenas mayor.
   - La diferencia entre «válida» y «válida con cuerpo» muestra un límite: cuando el modelo escribe algo irrecuperable, el retroceso corta hasta la firma y deja una función vacía. Por eso se reporta aparte.
3. **El enfriamiento del preentrenamiento mejoró mucho la sintaxis sin ayuda:** del 2 % al 31 % válido en modo plano.
4. **La autodepuración no ayuda todavía y cuesta unas 4 veces más tiempo y tokens.**
   - El bucle funciona: ejecuta, informa «obtuvo X, se esperaba Y» y pide la corrección (está probado en `tests/test_codigo.py` con un programador guionado).
   - Pero un modelo de 25 M no sabe usar ese informe para corregir.
   - Las pruebas que genera Hope casi nunca compilan, así que la validación por mutación casi no actúa.
5. **Trazas + FIM:**
   - El modelo aprende el **formato** muy rápido: el BPB baja de 3,66 a 0,23. Pero esos textos son muy repetitivos, así que el BPB exagera lo aprendido.
   - En 300 pasos no aprendió a **ejecutar mentalmente** ni a completar líneas exactas.
   - No hay corrida de control; falta repetirla con el checkpoint actual y más pasos.

## Pendiente (al retomar)

- Repetir el ajuste trazas + FIM con el checkpoint actual. El experimento ya guarda ejemplos de salidas: `python -m hope.bench.codigo_entreno --pasos 300 --eval 60`.
- Hacer una corrida de control sin FIM y otra sin trazas, para atribuir la mejora.
- Medir con más pasos y con el modelo de 200 M en la GPU.
- Probar el refuerzo con recompensa de ejecución (`python -m hope.bench.codigo --rl`) sobre problemas generados. Solo tiene sentido cuando haya algún acierto, porque sin varianza GRPO no aprende.
