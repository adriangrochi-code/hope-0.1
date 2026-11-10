# Contexto largo con menos memoria (§3.9.2.1) — informe PARCIAL

> **Estado: en pausa (pedido del usuario).** El código, las pruebas, la tabla de memoria y los
> planes están hechos. Falta: el experimento de CPU con un modelo chico entrenado (banco de
> contexto a 1–8 KB) y la medición de memoria de entrenamiento por largo. Las dos corridas se
> cortaron (reinicio de la máquina y la pausa) y no hay números todavía: no se inventan acá.

## Qué se implementó

| pieza | archivo | estado |
|---|---|---|
| Currículo de longitud 8 → 32 → 128 KB, ≥ 50 % de datos cortos (ProLong) | `configs/plan_a770_continuacion.yaml` (etapas 5–5c), etapa opcional en `configs/plan_gtx1060.yaml` | hecho |
| Datos largos de verdad (libros, Wikipedia completa, repos concatenados con la ruta de cada archivo) | `hope/data/largos.py` (`LongDocSource`) | hecho, probado con datos de juguete; no se armaron los datos reales |
| RoPE YaRN (bandas de frecuencia + temperatura), elegible por config | `hope/core/layers.py`, `set_context_scale(..., "yarn")` | hecho |
| Caché KV int8 / int4 por grupos | `hope/core/kvcache.py` | hecho |
| GQA + ventana deslizante con descarte en parte de las capas (presets `*-largo`, `to_long_context`) | `hope/core/hn1.py` | hecho |
| Selección de páginas (Quest), en `generate` y en la parte global incremental | `hope/core/kvcache.py` | hecho |
| Pérdida por trozos, checkpointing del núcleo, segmentos con memoria de Transformer-XL y backward por segmento | `hope/core/hn1.py` (`long_loss(memoria="cache")`), `hope/train/continual.py` | hecho |
| Banco de contexto (aguja, varias claves, combinar datos lejanos) y estimación de memoria | `hope/eval/contexto_bench.py` | código hecho; falta correrlo con un modelo entrenado |

## Exactitud (medido; `tests/test_contexto.py`, `tests/test_speed.py`)

- Con las opciones apagadas, la generación es **idéntica** a la de antes (`tests/test_speed.py` pasa entero).
- Leer por tramos con cachés (`segment_forward`) da lo mismo que el texto entero: error máximo en logits ~1e-6, en 5 configuraciones.
- `long_loss(memoria="cache")` en evaluación: misma entropía cruzada que `loss` sobre el texto entero (±1e-4). El backward por segmento da el mismo gradiente que un solo backward.
- `generate` con prefill por tramos y con ventana local + descarte: igual byte a byte a `generate_reference`.
- Quest con k páginas que cubren todo: exacto. La cota mín/máx es ≥ al producto q·k real (probado).
- Los bytes medidos de cada caché coinciden exactamente con la fórmula `kv_bytes_per_position` (16/8/4 bits).
- Suite completa: 498 pruebas pasan (sin las marcadas como lentas).

## Memoria de KV por contexto (estimación analítica, validada contra lo medido)

KV en bf16 salvo indicación; `docs/resultados/contexto_memoria.json` tiene la tabla completa.

| preset | KV | 32 KB | 128 KB | bytes de KV por byte (128 KB) |
|---|---|---|---|---|
| hn1-200m (actual) | bf16 | 515 MB | 2058 MB | 15 701 |
| hn1-200m-largo | bf16 | 21 MB | 43 MB | 328 |
| hn1-200m-largo | int8 | 11 MB | 23 MB | 174 |
| hn1-200m-largo | int4 | 6 MB | 13 MB | 103 |
| hn1-500m (actual) | bf16 | 544 MB | 2177 MB | 16 612 |
| hn1-500m-largo | int8 | 15 MB | 39 MB | 295 |

- En el modelo actual lo que más pesa es la **atención local completa sobre bytes** (5 capas guardan cada byte). La variante «largo» la limita a una ventana de 1 KB; el contexto lejano lo llevan la parte global (por parches) y los SSM.
- **Contexto máximo que entra** (pesos bf16): hn1-200m ≈ 90 KB en la A770 y ≈ 27 KB en la 1060, limitado por el pico de la atención local completa al leer el prompt. Con hn1-200m-largo la memoria deja de ser el límite (decenas de MB de contexto en teoría). Ese número **no** significa que el modelo use bien ese contexto: la precisión hay que medirla con el banco después del currículo.
- Quest no ahorra memoria (la caché completa sigue guardada). Ahorra lo que se lee en cada paso.

## Pendiente al retomar

1. Experimento de CPU: `python -m hope.eval.contexto_bench --preset hn1-cpu-largo --entrenar-min 25 --largos 1K,2K,4K,8K --n 3 --segmento 512 --salida docs/resultados/contexto_cpu.json` (correrlo **solo**: compartir la CPU lo vuelve ~15 veces más lento).
2. Memoria de entrenamiento: `python -m hope.eval.contexto_bench --memoria-entrenamiento hn1-cpu-largo --largos 2K,8K,32K`.
3. Completar este informe con esos resultados.
