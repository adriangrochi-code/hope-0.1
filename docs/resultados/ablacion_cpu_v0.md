# Ablación de la red experimental — prueba en CPU (indicativa)

hn1-tiny (0,17M), 300 pasos, corpus de humo de 650 KB, fp32, CPU. Diferencias de BPB de ±0,02 están dentro del ruido a esta escala: **solo sirve para comprobar que cada variante entrena y genera**. La decisión se toma con `python -m hope.bench.ablation --data data/f1 --preset hn1-25m` en la A770.

| variante | descripción | val BPB | Δ BPB vs base | bytes/s generación | parámetros |
|---|---|---|---|---|---|
| base | configuración actual | 3.959 | +0.000 | 290.3 | 0.17M |
| ventana_gqa | atención global solo en la última capa, ventana + sinks en el resto, GQA | 3.974 | +0.015 | 141.6 | 0.17M |
| local_ventana | codificador/decodificador de bytes con ventana deslizante | 3.954 | -0.005 | 243.3 | 0.17M |
| convergencia | profundidad muestreada + parada por convergencia | 3.975 | +0.016 | 143.2 | 0.17M |
| sesgo_compartido | balanceo por sesgo + especialista compartido | 3.942 | -0.017 | 400.8 | 0.17M |
| entropia | parches por entropía (estilo BLT) | 4.029 | +0.070 | 650.9 | 0.17M |
| rigl | conectividad dinámica en las zonas (densidad 50 %) | 3.961 | +0.002 | 293.8 | 0.17M |
