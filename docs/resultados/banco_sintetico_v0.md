# Banco sintético de ASG v0 — resultados (2026-11-09)

Experimento de validación de la §12.2 de `ARQUITECTURA_HN1.md`. Corrido en CPU.
Reproducir: `python -m hope.bench.synthetic --seeds 3 --first-seed 100`.

## Resumen

- **Diagnóstico:** 15 de 15 familias bien diagnosticadas en las semillas nuevas (100%).
- **Crecimiento innecesario:** ninguno. ASG solo creció en la región de la familia «capacity».
- **Dirigido vs. uniforme**, con los mismos parámetros agregados (+5.200, de 17.020 a 22.220) y el mismo entrenamiento:
  - exactitud en «capacity»: **0,359 dirigido** vs. 0,194 uniforme vs. 0,140 sin crecer;
  - promedio general: 0,291 vs. 0,259 vs. 0,246;
  - ninguna otra familia empeoró más de 0,007.
- **Guardián:** ASG decidió CONSOLIDAR en las tres semillas, pero el cambio agrega 30% de parámetros (> 10%). El guardián lo dejó **pendiente de aprobación humana**, como corresponde.

## Cómo se llegó (para no sobrevalorar el resultado)

Los parámetros del banco se ajustaron mirando las semillas 0–2. Ese proceso encontró dos problemas reales:

1. **Net2WiderNet no agrega capacidad.** Duplicar unidades preserva la función, pero las copias reciben gradientes idénticos y nunca se diferencian. Ensanchar la región con +80 unidades no mejoró nada (0,241 → 0,240). Se agregó el modo `zero_out`: unidades nuevas con entrada aleatoria y salida en cero. También preserva la función exactamente, y con él la misma ampliación llevó la exactitud de entrenamiento de 0,24 a 0,50. Es el modo por defecto desde ahora.
2. **La pendiente de la pérdida es un mal detector de «falta entrenar».** Con ruido, la familia «capacity» se diagnosticaba como OPTIMIZACION o quedaba en el límite. Se reemplazó por la sonda: dos copias entrenadas igual, una con más capacidad. El control mide cuánto aporta entrenar más; la copia ampliada mide cuánto aporta la capacidad *por encima* de eso.
3. La sonda con 32 unidades dio +0,0496, justo debajo del umbral de 0,05. Se pasó a probar el mismo crecimiento que se aplicaría (80 unidades), en vez de bajar el umbral.

Después de esos cambios, la validación se hizo con **semillas nuevas (100–102)** que no se usaron para ajustar nada. Los resultados de arriba son de esas semillas.

## Límites

- Las causas están **construidas**, y por eso son conocidas. El banco valida la *mecánica* del ciclo (señales → diagnóstico → sonda → crecimiento dirigido → decisión → guardián), no la hipótesis central sobre datos reales.
- Las familias «noisy», «tool» y «memory» tienen exactitud baja en los tres brazos, como se espera: su solución no es crecer, sino limpiar datos, usar la herramienta o recuperar de memoria. Esas rutas todavía no están implementadas; el diagnóstico solo las señala.
- Las reglas del diagnóstico son manuales (v0). Aprender a diagnosticar con la experiencia (ASG §13) requiere muchos más registros en PESAM.

## Resultados por semilla (validación, semillas 100–102)

## Diagnóstico por familia (semilla por semilla)

| familia | esperado | semilla 100 | semilla 101 | semilla 102 |
|---|---|---|---|---|
| easy | SIN_PROBLEMA | ✅ SIN_PROBLEMA | ✅ SIN_PROBLEMA | ✅ SIN_PROBLEMA |
| capacity | CAPACIDAD_INSUFICIENTE | ✅ CAPACIDAD_INSUFICIENTE | ✅ CAPACIDAD_INSUFICIENTE | ✅ CAPACIDAD_INSUFICIENTE |
| noisy | ETIQUETAS_ERRONEAS | ✅ ETIQUETAS_ERRONEAS | ✅ ETIQUETAS_ERRONEAS | ✅ ETIQUETAS_ERRONEAS |
| tool | HERRAMIENTA_FALTANTE | ✅ HERRAMIENTA_FALTANTE | ✅ HERRAMIENTA_FALTANTE | ✅ HERRAMIENTA_FALTANTE |
| memory | MEMORIA_RECUPERACION | ✅ MEMORIA_RECUPERACION | ✅ MEMORIA_RECUPERACION | ✅ MEMORIA_RECUPERACION |

## Exactitud de validación (media ± desvío entre semillas)

| brazo | parámetros | promedio | easy | capacity | noisy | tool | memory |
|---|---|---|---|---|---|---|---|
| dirigido | 22220 | 0.291 ± 0.007 | 0.991 ± 0.006 | 0.359 ± 0.029 | 0.050 ± 0.005 | 0.039 ± 0.005 | 0.015 ± 0.001 |
| fijo | 17020 | 0.246 ± 0.004 | 0.990 ± 0.006 | 0.140 ± 0.015 | 0.041 ± 0.005 | 0.042 ± 0.005 | 0.016 ± 0.001 |
| uniforme | 22220 | 0.259 ± 0.003 | 0.991 ± 0.006 | 0.194 ± 0.017 | 0.053 ± 0.003 | 0.039 ± 0.007 | 0.018 ± 0.000 |

## Decisión de ASG y guardián

- semilla 100: CONSOLIDAR (mejora objetivo +0.226; peor cambio en otras tareas -0.005) — guardián: requiere aprobación humana
- semilla 101: CONSOLIDAR (mejora objetivo +0.232; peor cambio en otras tareas -0.002) — guardián: requiere aprobación humana
- semilla 102: CONSOLIDAR (mejora objetivo +0.198; peor cambio en otras tareas -0.002) — guardián: requiere aprobación humana


## Actualización: cada diagnóstico enviado a su remedio (semillas 100–102)

Con el sistema de memoria v0 y el catálogo de herramientas, cada causa se envía a su solución en lugar de solo señalarla:

| familia | ruta | sin remedio | con remedio |
|---|---|---|---|
| capacity | crecimiento dirigido (ASG) | 0,140 | 0,359 |
| tool | herramienta `mul_add_mod` (HTE) | 0,042 | **1,000** |
| memory | consulta al almacén de hechos (`MemoryStore`) | 0,016 | **1,000** |
| noisy | limpieza por voto mayoritario + reentrenamiento | 0,041 | 0,048 |

**Hallazgo en «noisy»: dos causas encadenadas.** La limpieza funciona sobre los datos (etiquetas correctas: 61,1% → 87,4%), pero el modelo igual no aprende la función `(3·x1 + x2) mod 64` con su capacidad actual. Arreglar la primera causa deja al descubierto una segunda. Consecuencia de diseño: **después de aplicar un remedio, el ciclo tiene que volver a diagnosticar**, en lugar de dar el problema por resuelto. Queda pendiente para ASG v1.
