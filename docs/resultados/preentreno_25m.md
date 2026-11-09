# Preentrenamiento F1 de hn1-25m en CPU

**Resumen:** corrida corta de prueba en una CPU de 4 núcleos (contenedor de la sesión).

- **Duración:** 4,95 h en total (4,5 h estables y 0,45 h de enfriamiento).
- **Pasos y datos:** 7.699 pasos de 8 × 512 bytes, unos 31,5 MB vistos.
- **Resultado:** el BPB de validación bajó de unos 8 (modelo sin entrenar) a 1,98 en Wikipedia en español, 1,91 en inglés y 1,30 en código de la biblioteca estándar de Python.
- **Calidad:** el modelo aprendió ortografía, formato y estructura superficial (docstrings, indentación, secciones de Wikipedia), pero todavía no produce texto con sentido.

## Configuración

| | |
|---|---|
| Modelo | `hn1-25m` + `multibyte: 3`, 25.049.995 parámetros |
| Plan | `configs/plan_cpu_25m.yaml` (copia de `plan_cpu.yaml`) |
| Optimizador | Muon (ganó la ablación, `docs/resultados/ablacion_cpu.md`), lr 6e-4, warmup 200 |
| Programa | WSD: tasa constante en la etapa 1 y enfriamiento en la etapa 2 (`2_enfriamiento`) |
| Precisión | `bf16-8bit` (la CPU tiene AMX/AVX-512 bf16) |
| Lote | 8 secuencias × 512 bytes = 4.096 bytes por paso, sin acumulación |
| Máquina | 4 núcleos, 15 GB de RAM, sin GPU |
| Comando | `python -m hope.train.run --plan configs/plan_cpu_25m.yaml --device cpu` |

### Corpus (`data/pre/`, no versionado)

| Fuente | Origen | train | val | Peso en la mezcla |
|---|---|---|---|---|
| `wiki_es` | Wikipedia es (`wikimedia/wikipedia` 20231101), `hope.data.sources --wiki es:120M` | 118,8 MB | 1,23 MB | 0,40 |
| `wiki_en` | Wikipedia en, `--wiki en:80M` | 79,4 MB | 0,70 MB | 0,28 |
| `codigo` | Biblioteca estándar de Python 3.13 (`/usr/local/lib/python3.13`) | 9,4 MB | 0,12 MB | 0,12 |
| `codigo_terceros` | `site-packages` instalados (torch, numpy, pyarrow, pip…) | 49,5 MB | 0,49 MB | 0,20 |

**Diferencias con el pedido:**
- **Código:** la imagen solo trae unos 10 MB de `.py` en la biblioteca estándar (no incluye `test/`). Los 50 MB restantes del presupuesto de código salieron de las bibliotecas de terceros instaladas.
- **Repos propios:** no se usaron (el paso de Klipper/colmap era opcional y no se hizo).
- **Wikipedia:** los artículos salen de los primeros shards del volcado y no de una muestra aleatoria. Por eso hay sesgo hacia los artículos más antiguos o de id bajo.

## Curva de BPB de validación por fuente

En cada tramo de unos 20 min, el BPB se mide al final sobre 16 ventanas fijas de 513 bytes al comienzo de cada `val.bin`, es decir unos 8 KB por fuente.

Es una muestra chica. Las oscilaciones de ±0,05 entre tramos vecinos son ruido; por ejemplo, `codigo_terceros` pasa de 1,445 a 1,486 al final de la etapa 1. La tendencia sí es clara.

| Etapa | Horas | Paso | wiki_es | wiki_en | codigo | codigo_terceros | BPB entren. | bytes/s |
|---|---|---|---|---|---|---|---|---|
| 1_texto_codigo | 0,34 | 535 | 3,442 | 3,477 | 3,210 | 3,430 | 3,508 | 1809 |
| 1_texto_codigo | 0,67 | 1070 | 2,816 | 2,775 | 2,243 | 2,568 | 2,284 | 1810 |
| 1_texto_codigo | 1,01 | 1609 | 2,577 | 2,571 | 1,939 | 2,204 | 2,251 | 1824 |
| 1_texto_codigo | 1,34 | 2143 | 2,415 | 2,463 | 1,831 | 2,075 | 2,264 | 1806 |
| 1_texto_codigo | 1,68 | 2669 | 2,369 | 2,378 | 1,739 | 1,956 | 1,924 | 1779 |
| 1_texto_codigo | 2,02 | 3194 | 2,255 | 2,319 | 1,719 | 1,887 | 1,950 | 1777 |
| 1_texto_codigo | 2,35 | 3733 | 2,214 | 2,287 | 1,618 | 1,803 | 2,090 | 1826 |
| 1_texto_codigo | 2,69 | 4259 | 2,222 | 2,257 | 1,592 | 1,674 | 1,880 | 1779 |
| 1_texto_codigo | 3,03 | 4797 | 2,172 | 2,146 | 1,479 | 1,588 | 1,973 | 1822 |
| 1_texto_codigo | 3,36 | 5308 | 2,146 | 2,118 | 1,445 | 1,571 | 1,947 | 1730 |
| 1_texto_codigo | 3,70 | 5810 | 2,102 | 2,059 | 1,396 | 1,516 | 1,949 | 1698 |
| 1_texto_codigo | 4,04 | 6304 | 2,063 | 2,012 | 1,389 | 1,503 | 1,903 | 1670 |
| 1_texto_codigo | 4,37 | 6800 | 2,059 | 2,003 | 1,392 | 1,445 | 1,843 | 1677 |
| 1_texto_codigo | 4,50 | 6998 | 2,070 | 2,008 | 1,382 | 1,486 | 1,908 | 1720 |
| 2_enfriamiento | 0,34 | 7523 | 1,986 | 1,915 | 1,303 | 1,389 | 1,663 | 1775 |
| **2_enfriamiento** | **0,45** | **7699** | **1,975** | **1,906** | **1,296** | **1,381** | 1,853 | 1740 |

El enfriamiento final de 0,45 h bajó el BPB de validación entre 0,08 y 0,10 en las cuatro fuentes, el efecto esperado de WSD. Los datos crudos por tramo están en `preentreno_25m_registro.jsonl`.

## Velocidad y datos vistos

- **Entrenamiento:** unos 1.770 bytes/s en promedio (rango de 1.670 a 1.830), con 2,35–2,38 s por paso. La cifra incluye la evaluación de cada tramo.
- **Datos vistos:** unos 31,5 MB (7.699 × 4.096 bytes). Es alrededor del 12 % del corpus de entrenamiento (257 MB), así que no hubo ni una época completa y no hay riesgo de memorización por repetición.
- **Generación:** con el modelo liviano (`ligero.pt`, pesos bf16 pasados a fp32) y `model.generate`, unos 280–300 bytes/s en texto y 180–270 bytes/s en código, en la misma CPU.

## Muestras de generación

Se generaron con `hope.train.run.load_light("checkpoints/hn1-25m-cpu.pt")` y `model.generate(prompt, max_new=200, temperature=…)`, con semilla 0, y se transcriben sin editar. El texto en negrita es el prompt.

**Español, temperatura 0** (codicioso):
> **La ciudad de Buenos Aires es** el primer medio de la provincia de la provincia de la provincia de la provincia de la provincia …

**Español, temperatura 0,8:**
> **La ciudad de Buenos Aires es** la propiedad de escribir o marcardio en el año 620 metros de la región de la venforma de la braza del AIC por el arte del nuevo se encuentra la teoría de las posibilidades de esta y millones de l

> **La fotosíntesis es el proceso por el cual** girariación nosteriormente dulcana se funda una propiedad de dos dos publicación de contrarios de complión entre procesos de este proceso del país, y fue un oveno de cada de son dos resultados de

**Inglés, temperatura 0,8:**
> **The history of the Roman Empire** is a program of 14th techniques which eventually industrially also early 1202, like the History of Mary Muttich. In 1917, at the Mary Ship of the Balley and his postumenth during his clement and his

> **In mathematics, a prime number is** the last of International of our and primary anarch culture complexity.
>
> See also
>  Other of the tropical of the tropical opera of processors

**Python, temperatura 0 y 0,8:**
```python
def fibonacci(n):
        """
        Return a string file file file file file.

        The file file file file file file file file file file file.
        """
        if n == "file":
            return None
```
```python
import os

def list_files(path):
    """This fold on examples which is batched in the mapping of orders.

    Parameters
    ----------
    path): path
      arc path of program conversion of path path is conversion (default
```

Hay más muestras con los dos tipos de decodificación en `runs/muestras.txt`, que no se sube al repo.

## Conclusiones

1. **El camino completo funciona con datos reales:** construcción del corpus, plan por etapas con retoma, Muon, WSD con enfriamiento, checkpoints cada 20 min, exportación liviana y carga para generar. No hubo cortes ni errores en 4,95 h.
2. **El modelo aprendió lo superficial:** palabras mayormente bien escritas en los dos idiomas, sin mezclarlos, concordancias cortas («la región», «del país») y formato de Wikipedia («See also»). En Python reproduce la estructura de una función, con docstring, sección `Parameters` estilo numpydoc, indentación, `if … return`. Todavía no hay semántica: las frases no tienen sentido y en modo codicioso entra en bucles («de la provincia de la provincia…»), lo normal en un modelo de bytes con esta poca exposición.
3. **El código es más fácil que la prosa:** 1,30–1,38 contra 1,9–2,0 BPB, por ser más repetitivo y predecible. `codigo` (stdlib) queda por debajo de `codigo_terceros`, aunque tiene la mitad de peso en la mezcla. Puede deberse a un estilo más homogéneo, aunque con 8 KB de validación la diferencia es indicativa.
4. **El modelo seguía mejorando:** la curva se aplanaba (≈ −0,05 BPB por hora al final de la etapa estable) pero no se había estancado. Con 31,5 MB vistos, un modelo de 25M está muy lejos del régimen que se considera razonable (del orden de 20 bytes o más por parámetro, unos 500 MB). El cuello de botella es la CPU, con unos 6 MB por hora. Este resultado sirve como referencia de que todo funciona y como punto de partida para retomar en la A770, no como modelo usable.
5. **Limitaciones:**
   - una sola semilla;
   - validación chica, de unos 8 KB por fuente;
   - sin comparación contra una corrida AdamW del mismo largo;
   - Wikipedia tomada de los primeros shards y no muestreada;
   - código mayormente de bibliotecas de terceros y no de repos propios.

## Archivos

- `checkpoints/hn1-25m-cpu.pt`: pesos livianos finales en bf16, unos 50 MB. Se cargan con `hope.train.run.load_light`.
- `docs/resultados/preentreno_25m_registro.jsonl`: registro por tramo (BPB de validación por fuente, BPB de entrenamiento y bytes/s).
- `configs/plan_cpu_25m.yaml`: el plan usado.
