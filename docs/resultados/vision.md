# Calidad en imágenes: ablación en CPU (PARCIAL; en pausa hasta el miércoles)

**Estado:** 7 de 12 corridas terminadas. La ablación quedó en pausa por pedido del usuario. Faltan
`contrastiva` semilla 1, `curriculo` (las dos semillas) y `todo` (las dos semillas). Los números de
abajo son preliminares: con 30 preguntas por tarea, una diferencia de 0,033 es **una pregunta**.

Datos crudos: `docs/resultados/vision_ablacion_parcial.json`.

## Montaje

- **Modelo:** `hn1-cpu` (backbone d = 256, 4 bloques) + zona visual chica: ViT de 128 × 2 capas, imágenes de 96 px, parches de 8, 36 fichas por imagen. **8,8 M parámetros en todas las variantes**: ninguna mejora agrega parámetros.
- **Entrenamiento:** 1500 pasos, lote 16, secuencias de 384 bytes, AdamW lr 1·10⁻³ con coseno, fp32, CPU de 4 núcleos. El 90 % de los pasos son con imagen y el 10 % de repaso de texto.
- **Datos:** 3000 imágenes procedurales por corrida (≈ 17 000 pares pregunta–respuesta), semilla de datos 100 + semilla. Las imágenes idénticas píxel a píxel a alguna del banco se sacaron del entrenamiento: 0–8 por corrida, sobre todo relojes y tortas, que tienen pocas combinaciones.
- **Banco** (`hope/eval/vision_bench.py`):
  - sellado con `SealedRegistry`, semilla reservada `RESERVED_SEEDS.start + 17`, 30 preguntas por tarea (180 en total), 96 px y documentos a 192 px;
  - se mide la exactitud de la respuesta generada (con ⟦calc⟧ y ⟦circuito⟧ ejecutándose) y la **ventaja de la imagen** = bpb de la respuesta de referencia con una imagen cambiada − bpb con la correcta.
- **Variantes:** cada mejora se suma a `datos` por separado.
  - `base`: generadores originales (formas, texto, barras);
  - `datos`: + 8 generadores nuevos y aumentos; los documentos se reducen a 96 px;
  - `mosaicos`: `datos` + documentos a 192 px en mosaicos (miniatura + 2×2);
  - `contrastiva`: `datos` + pérdida sigmoide estilo SigLIP con peso 0,1;
  - `curriculo`: `datos` + de simple a complejo durante el 60 % inicial;
  - `todo`: las tres juntas.
- **Formato de la tabla:** media de las semillas ± media amplitud entre semillas.

## Resultados parciales

#### Exactitud (generación con herramientas; en texto, exactitud por carácter)

| variante | semillas | contar | color/forma | posición | texto | valores | razonamiento | promedio |
|---|---|---|---|---|---|---|---|---|
| base | 2 | 0,233 ±0,200 | 0,183 ±0,017 | 0,100 ±0,000 | 0,060 ±0,017 | 0,050 ±0,050 | 0,000 | 0,104 ±0,042 |
| datos | 2 | 0,250 ±0,117 | 0,183 ±0,017 | 0,133 ±0,033 | 0,027 ±0,014 | 0,117 ±0,017 | 0,000 | 0,118 ±0,027 |
| mosaicos | 2 | 0,283 ±0,017 | 0,317 ±0,017 | 0,083 ±0,017 | 0,031 ±0,009 | 0,100 ±0,033 | 0,000 | 0,136 ±0,001 |
| contrastiva | 1 | 0,300 | 0,267 | 0,100 | 0,080 | 0,100 | 0,000 | 0,141 |

#### Ventaja de la imagen (bpb con imagen cambiada − bpb con la correcta; más es mejor)

| variante | contar | color/forma | posición | texto | valores | razonamiento | promedio |
|---|---|---|---|---|---|---|---|
| base | 0,056 ±0,012 | −0,100 ±0,026 | −0,023 ±0,008 | 0,058 ±0,017 | 0,033 ±0,034 | 0,220 ±0,036 | 0,041 ±0,008 |
| datos | 0,007 ±0,003 | 0,024 ±0,014 | 0,002 ±0,001 | 0,145 ±0,052 | 0,017 ±0,009 | 0,018 ±0,014 | 0,036 ±0,015 |
| mosaicos | 0,002 ±0,001 | 0,004 ±0,000 | 0,001 ±0,001 | 0,180 ±0,003 | 0,003 ±0,002 | 0,031 ±0,012 | 0,037 ±0,002 |
| contrastiva | 0,002 | 0,036 | 0,007 | 0,210 | 0,034 | 0,027 | 0,053 |

#### bpb de la respuesta de referencia con la imagen correcta (menos es mejor)

| variante | contar | color/forma | posición | texto | valores | razonamiento |
|---|---|---|---|---|---|---|
| base | 2,993 | 3,484 | 2,833 | 2,929 | 4,945 | 4,542 |
| datos | 0,083 | 0,047 | 0,073 | 1,043 | 0,221 | 0,289 |
| mosaicos | 0,082 | 0,046 | 0,070 | 1,048 | 0,220 | 0,324 |
| contrastiva | 0,090 | 0,046 | 0,069 | 1,001 | 0,211 | 0,266 |

## Lectura honesta (provisoria)

1. **Todo sigue bajo.** Con 8,8 M parámetros y 1500 pasos la exactitud promedio es del 10–14 %. **Razonamiento queda en 0** en todas las variantes: el modelo arma llamadas ⟦calc⟧ con la forma correcta pero con números mal leídos (p. ej. «⟦calc: 5 + 10 + 10⟧» para barras 5, 1 y 10). Esta escala sirve para **comparar** variantes, no para juzgar la calidad final; eso va en la A770 con `hn1-500m`.
2. **`datos` vs `base`:**
   - la exactitud sube poco (0,104 → 0,118; dentro del ruido entre semillas); lo más claro es *valores* (0,050 → 0,117 en las dos semillas);
   - el salto enorme en bpb (≈ 3–5 → 0,05–1) **no** es ver mejor: 102 de los 180 textos de referencia del banco aparecen tal cual en el entrenamiento de `datos`, con *otras* imágenes, porque las plantillas de razonamiento son cortas. `base` nunca vio esas plantillas. Por eso, entre `base` y el resto, el bpb mide sobre todo familiaridad con el formato; las medidas que valen son la exactitud y la ventaja de la imagen (que compara el mismo texto con la imagen correcta y con otra, así las plantillas se cancelan).
3. **`mosaicos`:**
   - promedio 0,136 con las dos semillas casi iguales (±0,001);
   - en *texto* la ventaja de la imagen sube (0,145 → 0,180): con los mosaicos el modelo usa más la imagen de los documentos. Pero la exactitud de transcripción **no** mejora todavía (0,027 → 0,031): ve más, pero no alcanza para leer;
   - la suba en *color/forma* (0,183 → 0,317, consistente en las dos semillas) **no puede venir directamente de los mosaicos**, porque esas preguntas no son documentos y se codifican igual que en `datos`. Es un efecto indirecto del entrenamiento (otros lotes de documentos) o suerte; con 30 preguntas son 4 respuestas. No lo cuento como resultado.
4. **`contrastiva`** (una sola semilla): el mejor promedio (0,141) y la mayor ventaja de la imagen (0,053; en texto 0,210). Es coherente con lo que se esperaba (el codificador visual recibe una señal más directa), pero **sin la segunda semilla no es concluyente**.
5. **`curriculo` y `todo`:** todavía sin datos.

## Para retomar

Las corridas guardan resultados incrementales y saltean lo que ya está hecho. Hay que correrlas con el
**mismo código** (commit `e23287e`, anterior a la integración de `origin/ccr-976b883c-06ef2m`, que cambió el
núcleo; así las 12 corridas son comparables). Como `runs/` no está en git, primero hay que reponer los
resultados de este archivo:

```bash
git worktree add /home/user/hope-abl e23287e
mkdir -p runs/vision_ablation runs/vision_ablation_s1
python - <<'EOF'
import json
r = json.load(open("docs/resultados/vision_ablacion_parcial.json"))
json.dump([x for x in r if x["semilla"] == 0], open("runs/vision_ablation/results.json", "w"))
json.dump([x for x in r if x["semilla"] == 1], open("runs/vision_ablation_s1/results.json", "w"))
EOF
cd /home/user/hope-abl
OMP_NUM_THREADS=2 python -m hope.bench.vision_ablation --out /home/user/hope-0.1/runs/vision_ablation --seeds 0 &
OMP_NUM_THREADS=2 python -m hope.bench.vision_ablation --out /home/user/hope-0.1/runs/vision_ablation_s1 --seeds 1 &
```

El banco es determinista (semilla reservada): si falta el archivo sellado, se reconstruye igual y se
vuelve a sellar. Faltan 5 corridas, unas 2,5 h con las dos semillas en paralelo.
