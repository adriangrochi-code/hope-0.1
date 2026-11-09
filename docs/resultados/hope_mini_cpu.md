# Hope-mini (hn1-cpu, 7,9M) — preentrenamiento en CPU (en curso)

**Configuración** (`configs/plan_cpu.yaml`):
- **Datos:** Wikipedia en español e inglés, código Python, y el código e historial de Klipper, COLMAP y la parte humana de los repos del usuario.
- **Entrenamiento:** Muon + WSD, cabezas multi-byte, `bf16-8bit`, en la CPU de 4 núcleos de la sesión, compartida con otras tareas.

| Horas | Paso | BPB wiki_es | BPB wiki_en | BPB código | BPB repos (código) | BPB repos (historial) | bytes/s |
|---|---|---|---|---|---|---|---|
| 0.34 | 618 | 2.99 | 3.21 | 3.21 | 3.94 | 3.41 | 2095 |
| 0.73 | 765 | 2.77 | 2.96 | 2.94 | 3.67 | 3.14 | 425 |
| 1.16 | 1105 | 2.52 | 2.68 | 2.56 | 3.40 | 2.68 | 904 |
| 1.49 | 2441 | 2.12 | 2.39 | 2.16 | 2.78 | 2.06 | 4518 |
| 1.89 | 3493 | 2.02 | 2.27 | 2.01 | 2.52 | 1.86 | 3058 |

**Muestras** (voraz, 120 bytes, último checkpoint):

```
La historia de la ciudad de la ciudad de la ciudad de la ciudad de la ciudad de la ciudad de la ciudad de la ciudad de la ciudad de la 
```
```
The city of the city of the city of the city of the city of the city of the city of the city of the city of the city of the city of 
```
```
def suma(a, b):
        """Return the sumal sumalized by the sumalized by the sumalized by the sumalized by the sumalized by the sumaliz
```

**Lectura honesta:** es una corrida muy corta en CPU (decenas de MB vistos) con un modelo diminuto. Sirve para validar todo el camino con datos reales y como referencia, no como modelo útil.
Los pesos livianos están en `checkpoints/hn1-cpu-mini.pt`; se cargan con `hope.train.run.load_light`.
