# Bucle estable y GRPO corregido: implementación y ablación en CPU

Recomendaciones 2 y 4 del informe «Razonamiento por parámetro para Hope» (`reports/`), sin agregar parámetros. Diseño en ARQUITECTURA_HN1 §3.4.1 (bucle) y §4.2.4 (GRPO).

## Qué se implementó

- **Bucle estable** (`stable_loop` en `hope/core/hn1.py`):
  - reinyección convexa de la entrada en cada iteración, `h ← block(Ā·h + (1−Ā)·x)`, con Ā = 0,5 fijo;
  - sin `step_embed`, así que se puede iterar más en inferencia que al entrenar;
  - profundidad log-normal–Poisson al entrenar (media 4, tope 12), con backprop truncado opcional;
  - parada por convergencia (techo 16 en inferencia).
  
  Por defecto el núcleo sigue igual que antes y los checkpoints cargan sin cambios. La variante tiene `max_iters·d` parámetros **menos**.
- **EMA de pesos** opcional en el Trainer (`TrainConfig.ema`).
- **GRPO corregido** (`hope/train/rlvr.py`):
  - Dr. GRPO: ventaja centrada y pérdida ÷ `max_new`;
  - KL por byte de 1·10⁻³;
  - relleno de los grupos sin varianza con `ProblemPool` (retira lo fácil, posterga lo imposible, lista la franja borde);
  - currículo de largo 512 → 1.024 → 2.048 bytes;
  - interfaz `sampler`: usa la generación en lote de `hope.core.batched` si está, si no genera en serie;
  - log-probabilidades del grupo en una sola pasada;
  - costo como fase final opcional.
  
  Está probado con pruebas unitarias (`tests/test_rlvr.py`), **no con una corrida real de RL**: hace falta antes el SFT (§1.4 del informe).
- **Cadenas** (`chain_problem`/`chain_set`/`chain_corpus` en `hope/data/generators.py`): operaciones de un dígito módulo 10, con la respuesta escrita **sin pasos intermedios**. El cómputo tiene que ocurrir dentro del modelo. Sirven para medir la generalización a más pasos que los entrenados.

## Ablación 1: corpus real + 20 % de cadenas

**Configuración:**
- **Datos:** `python -m hope.data.sources --out data/pre --wiki es:30M --wiki en:20M --code /usr/lib/python3.13:7M`, concatenado en `data/pre/mix` (56 MB de entrenamiento y 0,87 MB de validación).
- **Cadenas:** 20 % de la mezcla, de 1 a 4 pasos (`python -m hope.data.generators --cadenas --n 300000 --pasos 1,2,3,4 --out data/cadenas`).
- **Entrenamiento:** preset hn1-cpu (7,75M), 1.500 pasos, lote 8 × 512 bytes, `bf16-8bit`, AdamW y coseno, semillas 0 y 1.
- **Máquina:** CPU de 4 núcleos.
- **Comando:**

```
python -m hope.bench.ablation --data data/pre/mix --preset hn1-cpu --steps 1500 --batch 8 --seq-bytes 512 \
  --variants base,bucle_estable,bucle_estable_ema --seeds 0,1 --extra cadenas=data/cadenas:0.2 \
  --val-dirs data/pre/wiki_es,data/pre/wiki_en,data/pre/codigo_python3.13 --depth-eval --out runs/ablacion_bucle
```

### BPB de validación

| Variante | Semilla 0 | Semilla 1 | Media | Δ vs base | wiki en (media) | código (media) | cadenas (media) | Generación |
|---|---|---|---|---|---|---|---|---|
| base (ACT, 1–4 iteraciones) | 2,765 | 2,701 | 2,733 | — | 2,530 | 2,403 | 0,932 | 222 B/s |
| bucle_estable | 2,821 | 2,727 | 2,774 | +0,041 | 2,550 | 2,423 | 0,927 | 149 B/s |
| bucle_estable_ema (0,998) | 2,813 | 2,736 | 2,774 | +0,041 | 2,555 | 2,436 | 0,927 | 154 B/s |

La columna «val BPB» mide las primeras 32 ventanas de `mix/val.bin`, que son de Wikipedia en español. Por eso coincide con el BPB por fuente de wiki_es.

### Generalización de profundidad

Exactitud en 200 cadenas por cantidad de pasos, con la respuesta voraz de un byte (el azar es 10 %). Media de las dos semillas.

| Variante | Iteraciones en inferencia | 1 paso | 2 | 3 | 4 | 5 | 6 | 8 | Iteraciones medias |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 / 2 / 4 | 0,16 | 0,21 | 0,17 | 0,19 | 0,17 | 0,16 | 0,16 | 1,0 |
| base | 8 / 16 / 32 | no se puede (`step_embed` tiene 4 filas) | | | | | | | |
| bucle_estable (convergencia) | 1 | 0,17 | 0,19 | 0,17 | 0,19 | 0,15 | 0,21 | 0,15 | 1,0 |
| bucle_estable (convergencia) | 4 | 0,17 | 0,17 | 0,16 | 0,20 | 0,15 | 0,20 | 0,15 | 4,0 |
| bucle_estable (convergencia) | 32 | 0,16 | 0,17 | 0,17 | 0,21 | 0,16 | 0,21 | 0,16 | 5,3 |
| bucle_estable (fija) | 32 | 0,16 | 0,18 | 0,16 | 0,20 | 0,15 | 0,20 | 0,15 | 32,0 |
| bucle_estable_ema (convergencia) | 32 | 0,16 | 0,18 | 0,17 | 0,18 | 0,15 | 0,20 | 0,16 | 5,2 |

Las tablas completas, por iteración y semilla, están en `runs/ablacion_bucle/report.md` y `results.json`.

### Lectura honesta

- **En BPB, el bucle estable no mejora a esta escala: queda 0,041 peor en promedio** (+0,056 con la semilla 0 y +0,026 con la 1).
  - La diferencia entre semillas de la misma variante (0,064 en la base) es del mismo orden. Con dos semillas no se puede afirmar que empeore, pero sí que no mejora.
  - Coincide con lo que el informe anticipaba para modelos chicos: MoR queda por debajo del transformer común a 135M, y el bucle mejora el razonamiento más que la perplejidad.
  - En código, una semilla mejora y la otra empeora.
- **La EMA es neutra:** −0,008 con la semilla 0, +0,009 con la 1. Era esperable: con un programa coseno que termina en 0,1·lr, los últimos pesos ya están «promediados». La EMA tendría que medirse con WSD o con tasa constante, y en corridas largas.
- **La base ACT colapsó a 1,0 iteración** en las dos semillas: es el colapso documentado en §3.4. Sus iteraciones 2–4 no cambian nada.
  - El bucle estable no colapsa. Con la parada por convergencia (tolerancia 1·10⁻²) usa 5,2–5,3 iteraciones en promedio.
  - Iterar hasta 32, más del doble del tope de entrenamiento (12), no rompe nada. La exactitud y la salida quedan estables, que era el objetivo de la reinyección convexa.
- **La prueba de profundidad no es informativa en esta mezcla.** Ningún modelo aprendió las cadenas en 1.500 pasos: la exactitud es de 15–21 % en todos los casos, y la cadena de 1 paso no anda mejor que la de 8. Ese 15–21 % por encima del 10 % del azar sale de sesgos (por ejemplo, «*0» da siempre 0), no de componer. Con un 20 % de cadenas sobre 6 MB vistos (~1,2 MB, unas 50.000 cadenas) no alcanzó. Por eso se corrió la ablación 2.
- **Costo:** el tiempo de entrenamiento fue parecido (~1.040 s por corrida en las que no compartieron la CPU; las de 3.200 s compartieron la CPU con la batería de pruebas). La generación es ~30 % más lenta, porque con caché el bucle calcula todas las iteraciones hasta el techo (16) en vez de 4. Bajar el techo de inferencia a 6–8 lo recupera casi todo, porque la convergencia llega en ~5,3.

## Ablación 2: solo cadenas

PENDIENTE
