# Bytes vs. parches = tokens de Qwen (y destilación) en CPU — **PARCIAL, en pausa**

> **Estado:** comparación pausada a pedido del usuario (hasta el miércoles). Hay **una sola semilla**
> de 4 de las 6 variantes. Faltan: `bpe` y `tokens_nemotron` (semilla 0) y toda la semilla 1.
> **No sacar conclusiones todavía**: las diferencias de abajo son del orden del ruido entre semillas
> que vimos en ablaciones anteriores (~0,03–0,05 BPB). La recomendación para la 1060 queda para el
> informe final.

## Configuración

- **Datos** (`data/tvb/mix`, `hope.data.sources --wiki-rows`): 12,2 MB de entrenamiento (6,8 MB de
  Wikipedia en español, 3,4 MB en inglés, 1,9 MB de código Python) y 368 KB de validación.
- **Modelo:** preset `hn1-cpu` (7,75M parámetros), lote 8 × 512 bytes, `bf16-8bit`.
- **Mismo tiempo:** 15 min de entrenamiento por corrida en la misma CPU (4 núcleos), una corrida por
  vez. **Muon + WSD** para todas (enfriamiento 1 − √p en el último 20 % del tiempo), lr 1e-3.
- **BPB de validación** sobre las mismas 256 ventanas de 512 bytes (repartidas entre español, inglés
  y código) para todas las variantes, incluida la BPE.
- **Maestro:** Qwen2.5-0.5B en CPU (bf16, `transformers`), top-8 por token, 3,22M tokens del
  corpus de entrenamiento (3,77 bytes/token) en 86 min; 174 MB en disco. Vocabulario reducido
  K = 4096 (+ «otro»): cubre el 87,1 % de la masa del maestro; la cabeza suma 1,05M parámetros que
  solo se usan al entrenar. λ = 0,25 (KL medio por parche alineado).
- Comando: `python -m hope.bench.tokens_vs_bytes --data data/tvb/mix --logits data/tvb/maestro_qwen05 --preset hn1-cpu --minutes 15 --seeds 0,1`

## Resultados parciales (semilla 0)

| variante | val BPB | Δ vs espacios | MB vistos en 15 min | bytes/s entren. | bytes/s gen. | parámetros (inferencia) | bytes/parche |
|---|---|---|---|---|---|---|---|
| espacios | 2,492 | — | 4,17 | 4.629 | 158 | 7,75M | 4,10 |
| entropía | **2,431** | −0,061 | 4,72 | 5.246 | 204 | 7,75M | 4,46 |
| tokens (Qwen) | 2,558 | +0,066 | 3,92 | 4.357 | 145 | 7,75M | 3,95 |
| tokens + KL | 2,706 | +0,214 | 3,52 | 3.905 | 166 | 8,80M (7,75M) | 3,50 |
| tokens (Nemotron) | pendiente | | | | | | |
| BPE | pendiente | | | | | | |

(«bytes/parche» es el del último lote de entrenamiento, no un promedio del corpus.)

**Curva:** val BPB vs. MB vistos (32 ventanas; cada corrida entra en su enfriamiento a ~80 % de
*su* tiempo, así que después de ~3 MB los puntos mezclan fases distintas):

| MB vistos | espacios | entropía | tokens | tokens + KL |
|---|---|---|---|---|
| 0,5 | 3,741 | 3,782 | 3,693 | 3,689 |
| 1 | 3,490 | 3,534 | 3,468 | 3,455 |
| 1,5 | 3,354 | 3,377 | 3,332 | 3,328 |
| 2 | 3,062 | 3,082 | 3,108 | 3,149 |
| 2,5 | 2,827 | 2,852 | 2,867 | 2,924 |
| 3 | 2,691 | 2,731 | 2,715 | 2,744 |
| 3,5 | 2,562 | 2,603 | 2,562 | 2,674 |

## Lectura provisional (una semilla)

- **Por dato**, los parches = tokens y la destilación van apenas adelante al principio (hasta 1,5 MB,
  −0,02 a −0,05 BPB), y después quedan iguales o atrás. La KL bajó a 2,9 nats por parche con 99,3 %
  de parches alineados con el maestro: la cabeza aprende, pero en este tamaño y duración no se
  traduce en mejores bytes.
- **Por tiempo** pierden: parches más cortos (más pasadas globales por byte), el parcheo causal en
  Python por lote y, en `tokens_kl`, la cabeza de 4097 salidas → 6–16 % menos bytes vistos.
- El parcheo **por entropía** es el mejor por ahora, sobre todo porque procesa más bytes por segundo.

## Para retomar

- Estado guardado: `docs/resultados/tokens_vs_bytes_estado/results.json` (copiar a
  `runs/tvb/results.json`) y el registro de la corrida. El banco retoma solo, saltando lo hecho.
- `tokens_nemotron` semilla 0 quedó cortada a 3,7 min: se rehace entera.
- Los logits del maestro están en `data/tvb/maestro_qwen05` (no se versionan, 174 MB). Si el
  contenedor se perdió: `python -m hope.data.sources --out data/tvb --wiki-rows es:7M --wiki-rows en:3.5M --code /usr/lib/python3.13:2M --val-percent 3`,
  concatenar `train.bin`/`val.bin` de las tres fuentes en `data/tvb/mix` y
  `python -m hope.train.distill logits --data data/tvb/mix/train.bin --out data/tvb/maestro_qwen05 --k 8` (~1,5 h).
  La muestra de Wikipedia es determinista (semilla fija), pero depende de que el volcado no cambie.
