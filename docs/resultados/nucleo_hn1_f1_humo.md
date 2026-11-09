# Núcleo HN-1 — prueba de humo de F1 (2026-11-09)

**No es la comparación de F1.** Es una verificación de que todo funciona de punta a punta, en CPU (4 núcleos), con un corpus de 650 KB (especificaciones de Hope en español + código Python). La comparación real se hace en la A770 con texto de Wikipedia/libros.

## Qué se verificó

- HN-1 de 23,9M parámetros (21,5M activos) entrena de forma estable: BPB de validación 8,1 → 2,28 en 6 minutos.
- La línea base BPE (20,8M, 8 capas de atención, tokenizador propio de 4.096) entrena con el mismo script y se evalúa en la misma métrica (bits por byte).
- Causalidad del núcleo: cambiar bytes futuros no altera ninguna predicción anterior (prueba automática, en entrenamiento y en inferencia).
- Las operaciones de crecimiento (agregar especialista, ensanchar) funcionan sobre las regiones reales de HN-1 y preservan la salida.

## Comparación indicativa (mismo tiempo: 6 min en CPU)

| modelo | parámetros | bytes/s | bytes vistos | val BPB |
|---|---|---|---|---|
| línea base BPE | 20,8M | 10.300 | 3,7M | 1,50 |
| HN-1 (bytes) | 23,9M | 2.750 | 1,0M | 2,28 |

La línea base procesa ~3,8× más bytes por segundo. Es el costo de leer byte a byte, que el diseño ya anticipaba (§3.9). Con un corpus de 650 KB, la línea base además sobreajusta (train 0,6 vs. val 1,5). **No sacar conclusiones de esta tabla.** En la A770 hay que medir: (1) velocidad real de cada uno; (2) BPB con el mismo tiempo de cómputo; (3) BPB con los mismos bytes vistos.

## Problemas encontrados y corregidos

1. **NaN al entrenar.** El SSM calculaba `log(exp(−x))`. Cuando `exp(−x)` se redondeaba a 0, el gradiente era infinito. Ahora el scan recibe el logaritmo del decaimiento directamente. Hay una prueba con decaimientos extremos.
2. **Desborde numérico del scan** con decaimientos fuertes: bloques de 16 pasos y decaimiento mínimo e⁻⁵ por paso.
3. **Causalidad en inferencia.** El bucle recurrente paraba cuando *todas* las posiciones habían terminado, así que una posición dependía (muy poco) de las siguientes. Ahora cada posición se congela por separado.
4. **Colapso del bucle interno**, detectado con el diagnóstico de uso (LTH): la iteración 1 se llevaba el 99,99% del peso y las iteraciones 2–4 aportaban 0 a la pérdida. Corrección: la mitad de los pasos de entrenamiento usa todas las iteraciones y el costo de iterar sube gradualmente. Después de la corrección las iteraciones 2–4 reciben peso (5,6%), pero **todavía no mejoran la pérdida**. Con este corpus y este tiempo de entrenamiento no se puede saber si el bucle sirve; queda como pregunta abierta para la A770.

## Diagnóstico de uso (después de la corrección)

- Especialistas: uso desbalanceado (27% el más usado vs. 5,5% el menos usado). Vigilar en entrenamientos reales; si persiste, subir el peso de la pérdida de balanceo.
- Neuronas: prácticamente ninguna ociosa a esta escala (< 1% por población).
