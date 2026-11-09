# Ablación en CPU con datos reales (hn1-cpu, 7,7M parámetros)

**Configuración:**
- **Datos:** 72 MB de entrenamiento con 40 MB de Wikipedia en español, 25 MB en inglés y 7 MB de código Python (`hope/data/sources.py`), y 1 MB de validación.
- **Entrenamiento:** 1.500 pasos, lote 8 × 512 bytes (6,1 MB vistos), `bf16-8bit`, una semilla.
- **Máquina:** CPU de 4 núcleos (contenedor de la sesión).
- **Comando:** `python -m hope.bench.ablation --data data/pre/mix --preset hn1-cpu --variants base,muon,wsd,multibyte --steps 1500 --batch 8 --seq-bytes 512`

| Variante | Descripción | BPB de validación | Diferencia con la base |
|---|---|---|---|
| base | AdamW, coseno | 2,752 | — |
| muon | Muon en las matrices ocultas | 2,679 | **−0,073** |
| wsd | tasa constante + enfriamiento 1−√p en el último 20 % | 2,659 | **−0,093** |
| multibyte | cabezas que predicen 2–4 bytes adelante | 2,743 | −0,008 |

**Lectura honesta:**
- **Muon y WSD** mejoran de forma clara para este tamaño y esta duración. Los dos se adoptan para el preentrenamiento (`configs/plan_cpu.yaml`) y se recomiendan para la A770.
- Falta confirmarlos con más semillas y a 200M: las ganancias de un optimizador pueden cambiar con la escala.
- **Multi-byte** es neutro en BPB a esta escala. Se esperaba: Gloeckle et al. 2024 reportan beneficios sobre todo en modelos grandes y en código. Se mantiene porque sus predicciones sirven de borrador para la generación especulativa, por un 1,4 % de parámetros.
- **Tiempos no comparables:** la variante base compartió la CPU con otras pruebas (4836 s contra ~1000 s de las demás), así que los tiempos y los bytes/s de generación de esta tabla no sirven para comparar velocidad.
