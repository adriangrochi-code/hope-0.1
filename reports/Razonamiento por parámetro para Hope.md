# Hope razona más por parámetro con datos procedurales, un bucle bien entrenado y verificación barata

La literatura de 2024–2026 coincide en que, en modelos de menos de 1B, el razonamiento por parámetro mejora sobre todo con tres palancas que no agregan parámetros densos. La primera son los **datos de razonamiento desde el preentrenamiento**: procedimientos, matemática, código y problemas sintéticos verificables con errores corregidos. La segunda es la **profundidad recurrente entrenada para que se use**. Los modelos en bucle no guardan más conocimiento (siguen en ~2 bits por parámetro), pero lo manipulan mejor: Ouro-1,4B iguala a Qwen3-4B en la mayoría de los benchmarks. La tercera es la **verificación en inferencia con herramientas**, que convierte muestras baratas en aciertos. La destilación del maestro rinde más que el refuerzo en modelos chicos, siempre que se adapte al alumno (cadenas cortas y pérdida solo en la respuesta). El refuerzo con recompensas verificables (RLVR) da ganancias reales solo sobre problemas al borde de lo que el modelo ya sabe y después de una etapa supervisada; a 135M, un GRPO directo sobre GSM8K empeoró el resultado. En memoria, lo que más sirve a un modelo de bytes es barato y específico: **embeddings hash de n-gramas de bytes** (BLT, Engram). Las memorias paramétricas grandes ayudan sobre todo a los hechos, no al razonamiento, y el kNN-LM incluso lo empeora. En contexto largo, Hope ya tiene casi todas las piezas. Le faltan atención con compuerta, una mezcla de datos largos bien hecha y tareas que obliguen a usar la memoria. La advertencia central es de escala: Hope procesará del orden de 0,07–0,15 billones (T) de tokens equivalentes, mientras que MobileLLM-R1-140M, con 4,2T tokens, saca 6,2 en MATH500. La meta realista no es competir en GSM8K o MATH generales. Es lograr **razonamiento composicional en dominios verificables estrechos** (aritmética, circuitos, Python), medido por generalización a problemas más complejos que los entrenados.

## Qué se puede esperar con 200M–500M parámetros y una A770

El presupuesto de datos manda sobre todo lo demás. Según ARQUITECTURA_HN1 §3.9, el modelo de 200M cuesta del orden de 3·10²⁰ FLOPs por TB de bytes. Seis meses son ~1,6·10⁷ segundos. Con un rendimiento efectivo de 10–20 TFLOPS en la A770 (a medir en F0), eso da 0,5–1 TB de bytes a 200M, y menos después de crecer a 500M y de reservar tiempo para la visión y el maestro. La estimación (no medición) queda en **0,3–0,6 TB, unos 70–150 mil millones de tokens BPE equivalentes**. Es de 5 a 10 veces más que los estudios hechos con una sola GPU, pero de 25 a 60 veces menos que MobileLLM-R1 (4,2T) y más de 70 veces menos que LFM2 (11T):

| Modelo | Parámetros | Tokens | Razonamiento reportado |
|---|---:|---:|---|
| MobileLLM-R1-140M ([Meta 2025](https://arxiv.org/html/2509.24945v3)) | 140M | 4,2T | MATH500 6,2; GSM8K 4,1 (instruct) |
| MobileLLM-R1-360M | 360M | 4,2T | MATH500 28,4; GSM8K 24,5 |
| MobileLLM-R1-950M | 950M | 4,2T | MATH500 74,0; GSM8K 67,5; AIME24 15,5 |
| LFM2-350M ([Liquid AI 2025](https://arxiv.org/html/2511.23404v1)) | 350M | 10T + 1T | GSM8K 30,1; MATH500 24,2 |
| L20-Edu-135M ([2026](https://arxiv.org/abs/2606.22189)) | 135M | 13B | GSM8K 1,8 → 1,6 tras GRPO |
| Modelo de 100M sobre datos sintéticos ([Zhang, Neubig y Yue 2025](https://arxiv.org/html/2512.07783)) | 100M | 10B | aprende composición; RL hasta +60% pass@128 fuera de distribución |
| **Hope HN-1 (estimación)** | 200–500M | ~0,07–0,15T | — |

El segundo límite es de capacidad. Allen-Zhu y Li midieron que un modelo guarda como máximo **~2 bits de conocimiento por parámetro**, incluso cuantizado a int8 ([arXiv:2404.05405](https://arxiv.org/abs/2404.05405)). Ouro confirmó que los modelos en bucle quedan en el mismo ~2 bits/parámetro: el bucle no agrega almacenamiento, mejora la composición de hechos y el razonamiento de varios saltos ([Zhu et al. 2025](https://arxiv.org/html/2510.25741v2)). Para 200M, el techo son ~4·10⁸ bits, unos **50 MB de hechos**, frente a decenas de GB de Wikipedia en español e inglés. La conclusión práctica, por inferencia, coincide con la decisión de Hope de dejar el conocimiento fuera de los pesos: los parámetros tienen que dedicarse a procedimientos, y los hechos a la base de conocimiento.

La evaluación debería seguir el diseño del estudio de CMU, que entrenó desde cero un modelo de 100M ([arXiv:2512.07783](https://arxiv.org/html/2512.07783)). Ese estudio separa los problemas por complejidad (cantidad de operaciones) en tres franjas: dentro de la distribución, borde (el modelo falla a pass@1 pero acierta a pass@128) y difícil (cerca de cero). También mide pass@1 y pass@k por separado. Los generadores de Hope ya producen problemas por nivel. Agregarles un parámetro de complejidad compositiva permite ver si el bucle, la destilación o el RL enseñan a pensar o solo a imitar.

## 1. Razonamiento por parámetro en modelos chicos

### 1.1 Profundidad recurrente: el bucle mejora la manipulación, no el almacenamiento

**Evidencia.** Saunshi et al. muestran que un modelo de k capas aplicado L veces casi iguala en tareas de razonamiento a uno de kL capas, con L veces menos parámetros y peor perplejidad. Prueban además que T iteraciones pueden simular T pasos de cadena de pensamiento ([ICLR 2025, arXiv:2502.17416](https://arxiv.org/abs/2502.17416)). Es decir, el bucle tiene un sesgo hacia el razonamiento que la perplejidad no muestra. Ouro lo llevó a escala con 7,7T tokens. Ouro-1,4B con 4 pasos supera a Qwen3-4B-Base en GSM8K (78,9 frente a 72,9) y en MATH500 (82,4 frente a 59,6), y pierde en MMLU (67,4 frente a 73,2), que mide conocimiento ([Zhu et al. 2025](https://arxiv.org/html/2510.25741v2)). Mixture-of-Recursions, a igual cómputo de entrenamiento (16,5·10¹⁸ FLOPs), logra con 167M parámetros únicos una pérdida menor y una exactitud few-shot mayor (43,1% frente a 42,3%) que un transformer de 315M ([Bae et al. 2025](https://arxiv.org/html/2507.10524v3)). El Inner Thinking Transformer llega con 162M al 96,5% del rendimiento de uno de 466M, con 43% menos datos ([Chen et al. 2025, ACL](https://arxiv.org/abs/2502.13842)). MobileLLM, a 125M–350M, gana 2,7 y 4,3 puntos sobre el estado del arte previo combinando profundidad sobre ancho, embeddings compartidos y GQA, y otros 0,7–0,8 repitiendo cada bloque dos veces sin agregar parámetros ([Liu et al. 2024, ICML](https://arxiv.org/abs/2402.14905)).

**Lo que cuesta.** El bucle no es gratis en cómputo. La ley de escala iso-profundidad de 2026 estima un exponente de equivalencia φ = 0,46: con 4 recurrencias, un modelo en bucle de 410M rinde como uno sin bucle de 580M, pero **cuesta entrenarlo como uno de 1B** ([Schwethelm, Rueckert y Kaissis 2026, arXiv:2604.21106](https://arxiv.org/abs/2604.21106)). El mismo trabajo encontró que el backprop truncado en el tiempo baja φ a 0,38 (el bucle queda mal entrenado aunque la pérdida baje), y que las hiperconexiones lo suben a 0,65. Parcae explica la inestabilidad clásica de estos modelos (explosión del estado residual y picos de pérdida): el radio espectral de la inyección llega a 1 o más. Lo corrige con una matriz diagonal negativa discretizada que garantiza ρ < 1. Con eso baja la perplejidad hasta 6,3% frente a modelos recurrentes previos, y su versión de 770M casi alcanza a un transformer de 1,3B (Core 25,07 frente a 25,45) ([Prairie et al. 2026, arXiv:2604.12946](https://arxiv.org/abs/2604.12946); [blog](https://together.ai/blog/parcae)). Bajo un presupuesto fijo, lo óptimo es subir la recurrencia y los datos juntos.

**Lo que no hace.** Ouro rinde mejor exactamente en la profundidad entrenada (T = 4) y empeora más allá. En el modelo de razonamiento de 1,4B, AIME24 cae de 65,0 a 38,7 al pasar de 4 a 8 pasos ([Zhu et al. 2025](https://arxiv.org/html/2510.25741v2)). Con 8 pasos el entrenamiento era inestable y tuvieron que bajar a 4. Huginn sí escala más allá porque se entrenó con profundidad aleatoria de cola pesada (Poisson log-normal) y backprop solo por las últimas 8 iteraciones. Con 3,5B parámetros y 800B tokens, mejora hasta un cómputo equivalente a 50B parámetros ([Geiping et al. 2025, arXiv:2502.05171](https://arxiv.org/abs/2502.05171)). MoR, por su parte, queda por debajo del transformer común a 135M y lo iguala o supera recién desde ~360M ([Bae et al. 2025](https://arxiv.org/html/2507.10524v3)). **Hope a 200M está justo en esa frontera.**

**Parada.** Un estudio de 2026 con un modelo recurrente de clase 135M, entrenado en una sola RTX 4090 en ~100 horas (una escala comparable a Hope), encontró que cada token converge a un punto fijo. El token mediano converge en la iteración 6, los espacios primero y las palabras de contenido al final. Una regla sin entrenamiento, que detiene cada token cuando su salida deja de cambiar (KL entre iteraciones), iguala la calidad de profundidad 8 con 4,94 iteraciones en promedio (−38%). En cambio, un router lineal entrenado para lo mismo no ahorró nada ([Logan 2026, arXiv:2607.14427](https://arxiv.org/abs/2607.14427)). La pérdida bajó de 3,80 con 1 iteración a 3,20 con 8 y se mantuvo estable hasta 32. Ouro usa una alternativa aprendida en dos etapas. Primero entrena el modelo con la pérdida esperada sobre la distribución de salida menos β·entropía (β de 0,1 a 0,05), lo que evita el colapso a la iteración más profunda. Después congela el modelo y entrena solo la compuerta con etiquetas de "mejoró la pérdida" ([Zhu et al. 2025](https://arxiv.org/html/2510.25741v2)).

**TRM, HRM y HRM-Text.** TRM, con 7M parámetros y 2 capas, logra 44,6% en ARC-AGI-1, 7,8% en ARC-AGI-2 y 87,4% en Sudoku-Extreme (HRM: 55,0%), con supervisión profunda y EMA ([Jolicoeur-Martineau 2025, arXiv:2510.04871](https://arxiv.org/abs/2510.04871)). Pero usa ~1000× de aumentación por ejemplo y son rompecabezas, no lenguaje. HRM-Text (mayo de 2026) es el primer resultado de la familia HRM en lenguaje: un modelo de 1B entrenado desde cero con 40B tokens únicos de pares instrucción-respuesta, por ~1.500 USD. Reporta GSM8K 84,5 y MATH 56,2 ([Wang et al. 2026, arXiv:2605.20613](https://arxiv.org/html/2605.20613)). Son cifras autorreportadas, de un solo grupo y con una señal marginal de contaminación en DROP. Lo transferible de ese trabajo son tres técnicas: un horizonte de gradiente que crece de 2 a 5 pasos, atención con compuerta sigmoide y pesos EMA 0,9999. Su ablación sobre la pérdida está en §1.3.

**Madurez:** media–alta para el bucle en sí (varios grupos, 2025–2026, escalas de 100M a 2,6B); baja para la parada por convergencia (una escala, una semilla) y para HRM-Text.

#### 1.1.1 Qué cambiar en el núcleo de Hope

Leyendo `hope/core/hn1.py`, el `ConvergentCore` actual aplica `h = block(h + step_embed[i])`, con profundidad uniforme entre 1 y `max_iters = 4` y backprop completo. La evidencia sugiere seis cambios. Ninguno agrega más que unos miles de parámetros, salvo el adaptador opcional:

1. **Inyección de la entrada en cada iteración, de forma estable.** Hoy la entrada `x` solo entra en la primera iteración. Huginn concatena la entrada con el estado en cada paso. En aritmética, la inyección de la entrada redujo los errores de generalización a la mitad, y junto con el bucle llevó la suma de 100 dígitos al 99% ([McLeish et al. 2024, arXiv:2405.17399](https://arxiv.org/abs/2405.17399)). La versión estable es la de Parcae: `h ← block(Ā·h + B̄·x)` con `Ā = exp(Δ·A)` y A diagonal negativa. Cuesta ~2·d parámetros, o 2·d² ≈ 2M si se usa un adaptador de concatenación como el de Huginn.
2. **Quitar el embedding por iteración de tamaño fijo.** `step_embed` tiene `max_iters` filas, así que impide usar más iteraciones en inferencia que en entrenamiento. Se puede eliminar (Huginn no lo usa) o reemplazar por una función continua de i.
3. **Profundidad muestreada con cola pesada**, con media 4–6 y máximo 12–16, en vez de uniforme 1–4. Si la memoria alcanza, conviene mantener el backprop completo (el truncado baja φ). Si no alcanza, el horizonte puede crecer de 2 a 5 pasos como en HRM-Text.
4. **`halting: "convergencia"` como valor por defecto y retirar ACT.** El colapso a una iteración que documenta §3.4 es el modo de falla que la literatura atribuye al costo de ponder. Si hace falta una compuerta aprendida para planificar el cómputo, se entrena aparte con el modelo congelado (etapa II de Ouro).
5. **Bucle más grande y modelo más profundo.** El núcleo es ~22M de ~174M. Ouro pone en bucle toda la pila y MoR concluye que lo mejor es compartir el medio y dejar únicos los extremos ("Middle-Cycle"). Es un experimento a decidir antes de la corrida de 6 meses: un backbone más angosto y profundo (estilo MobileLLM) con 2–3 bloques dentro del bucle. El preset hn1-200m (d = 1024, 8 bloques) es ancho y poco profundo para su tamaño.
6. **EMA de pesos y reutilización de KV en decodificación.** La EMA (TRM, HRM-Text) cuesta 0,4–0,8 GB a 200M. En decodificación, Ouro reutiliza el KV de la última iteración para todas: 4 veces menos memoria con GSM8K 78,85 frente a 78,92, aunque en el prefill hacen falta todos ([Zhu et al. 2025](https://arxiv.org/html/2510.25741v2)). Hoy Hope calcula todas las iteraciones con caché.

El criterio de éxito tiene que ser **que la pérdida y la exactitud mejoren de verdad entre 1 y N iteraciones** en problemas de la franja "borde". Si no mejoran a 200M, como en MoR a 135M, el bucle debe quedar para el modelo de 500M.

### 1.2 Razonamiento latente continuo y tokens de pausa: todavía no

**Evidencia.** Coconut reemplaza los pasos de la cadena de pensamiento por estados ocultos. Supera a la cadena explícita en planificación lógica (ProsQA), pero queda por debajo en GSM8K ([Hao et al. 2024, arXiv:2412.06769](https://arxiv.org/abs/2412.06769)). CODI, que destila el estado de un token designado desde la cadena explícita, iguala la cadena de pensamiento en GSM8K con GPT-2 (43,7%, ~99%) comprimiendo 3,1 veces, y llega al ~90% con Llama-3.2-1B. Es 2,7–5,9 veces más rápido ([Shen et al. 2025, EMNLP](https://arxiv.org/html/2502.21074v3)). Los tokens de pausa, si se usan desde el preentrenamiento, dan +18% de EM en SQuAD a 1B, pero solo +1 en GSM8K (sobre una base de 7,5%), y aplicados solo en el ajuste fino pueden empeorar ([Goyal et al. 2024, ICLR](https://arxiv.org/abs/2310.02226)). PonderLM, que realimenta una mezcla de embeddings, reporta igualar a Pythia con ~la mitad de parámetros, pero a costa de pasadas extra por token ([arXiv:2505.20674](https://arxiv.org/abs/2505.20674)).

**Para Hope:** no por ahora. El razonamiento latente pierde lo que Hope más usa, que es la **verificación paso a paso** (crítica CRE, recompensa de 0,2 a los pasos rotos, base de lecciones). Además, a la escala de Hope rinde menos que la cadena explícita. El núcleo recurrente ya cubre el "pensar sin emitir bytes". Una posible excepción futura es comprimir subpasos triviales (como la aritmética que ya verificó la calculadora), recién cuando el modelo de 500M razone bien en forma explícita.

### 1.3 Destilación: adaptar el maestro al alumno

**Brecha de aprendibilidad.** Los modelos de 3B o menos no mejoran de forma consistente con cadenas largas ni con maestros mucho más grandes. Mezclar cadenas largas y cortas en proporción 1:4 (Mix-Long), o respuestas de maestros grandes y chicos (Mix-Large), mejoró a Qwen2.5-3B más de 8 puntos en MATH y AMC frente a entrenar solo con cadenas largas ([Li et al. 2025, ICML](https://arxiv.org/abs/2502.12143)). Para Hope, un alumno de 200M con un maestro de 8B, la brecha es todavía mayor. Hay además un beneficio práctico: un Qwen de 1,5–3B genera varias veces más rápido en la A770, y el maestro es el cuello de botella declarado (§10.1.1).

**SFT antes que RL.** MobileLLM-R1 encontró que en modelos chicos el SFT sobre datos destilados supera al RL desde la base: en su ablación, el 950M con SFT superó con claridad al mismo modelo base entrenado solo con RL (74,0 frente a 57,0 según el apéndice; las cifras del trabajo no son del todo consistentes entre secciones). Agregar RL después del SFT no dio una ganancia significativa e incluso puede degradar ([Meta 2025, arXiv:2509.24945](https://arxiv.org/html/2509.24945v3)). También usó destilación de logits de Llama-3.1-8B en el mid-training, que para Hope no es directa por el tokenizador.

**Pérdida solo en la respuesta y PrefixLM.** En la ablación de HRM-Text a 1B y a igual cómputo, pasar de pérdida sobre todo el texto a pérdida solo sobre la respuesta subió MMLU de 40,6 a 47,7 en un transformer común. Agregar atención bidireccional sobre el enunciado (PrefixLM) lo llevó a 53,2. En el modelo recurrente, el salto fue de 43,7 a 60,7 ([Wang et al. 2026](https://arxiv.org/html/2605.20613)). Es un solo trabajo, pero el cambio es barato: los episodios del maestro de Hope ya separan problema y explicación.

**Soluciones como programas, más un verificador.** TinyGSM logró 81,5% en GSM8K con un generador de 1,3B y un verificador de 1,3B, usando 12,3M problemas sintéticos con soluciones en Python, por encima de su maestro GPT-3.5 (77,4%). Agrandar el verificador de 125M a 1,3B rindió +7,2 puntos, más que agrandar el generador ([Liu et al. 2023, arXiv:2312.09241](https://arxiv.org/abs/2312.09241)). Encaja con la recompensa de costo de Hope, que ya premia delegar en `⟦calc⟧`.

**Destilación de logits entre tokenizadores (ALM).** ALM alinea fragmentos de texto que ambos tokenizadores cortan igual y compara sus probabilidades con una f-divergencia binarizada. Necesita solo la probabilidad de los tokens observados que da una pasada del maestro, no su distribución completa ([Minixhofer, Vulić y Ponti 2025, NeurIPS](https://arxiv.org/html/2503.20083v3)). Es el primer método que destiló de subpalabras a bytes. La ganancia sobre el SFT es modesta: en matemática, Gemma2-2B con un maestro OpenMath2-8B llegó a 53,3 de promedio con ALM+SFT, frente a 51,7 con SFT, y en la transferencia a bytes quedó una brecha frente al modelo original. Bolmo convirtió OLMo-2-1B y Olmo-3-7B a bytes con menos del 1% del costo de preentrenamiento, destilando de los logits del modelo fuente ([arXiv:2512.15586](https://arxiv.org/abs/2512.15586)). Para Hope, Bolmo no es el camino (se entrena desde cero), pero un Qwen chico convertido a bytes sería una **línea base fuerte** contra la cual medirse.

**Destilación on-policy.** El alumno genera y el maestro califica cada token de esa muestra con KL inversa: es una señal densa, sobre las propias muestras del alumno y difícil de explotar. Thinking Machines la midió 9–30 veces más barata que el SFT. El informe técnico de Qwen3 reporta 74,4% en AIME24 con 1.800 GPU-h, frente a 67,6% del RL con 17.920 GPU-h ([Lu et al. 2025, Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)). Todo eso es con el mismo tokenizador y con 4 muestras por problema. Para Hope tiene una ventaja estructural, que es una inferencia y está sin medir. Calificar una muestra exige solo un **prefill** del maestro, que procesa todos los tokens en paralelo y es mucho más rápido que generar. Así, el cuello de botella pasa de la generación del maestro a su prefill. Combinada con la alineación por fragmentos de ALM, la KL se calcularía por fragmento en vez de por token. No hay evidencia publicada de destilación on-policy hacia un alumno de bytes, así que es un experimento.

**Para Hope:**
- Pedir a Qwen explicaciones **cortas por defecto**, con ~20% de explicaciones largas, y mezclar respuestas de un Qwen de 1,5–3B.
- Entrenar los episodios con pérdida solo en la respuesta y PrefixLM sobre el enunciado.
- Pedir y verificar soluciones en forma de programa cuando el tema lo permita (aritmética, circuitos, Python).
- Agregar una cabeza verificadora sobre los estados de Hope, entrenada con los pares (solución, correcta) que ya producen STaR y RLVR.
- Como experimento, cachear offline las log-probabilidades del maestro sobre los textos (ALM). Más adelante, puntuar con prefill las muestras de Hope (on-policy).

### 1.4 RL con recompensas verificables: solo al borde de la competencia

**Evidencia en modelos chicos.** El estudio de CMU con un modelo de 100M entrenado desde cero es el más cercano a Hope ([Zhang, Neubig y Yue 2025, arXiv:2512.07783](https://arxiv.org/html/2512.07783)). Encontró que el RL da ganancias reales (pass@128) solo si el preentrenamiento deja margen y los problemas de RL están en el borde de lo que el modelo resuelve. Para que el RL transfiera a un contexto nuevo, alcanza con **~1% de exposición en el preentrenamiento** (con 0,1% no transfiere), y entonces la ganancia llega hasta +60% pass@128. Un mid-training enfocado más RL liviano rinde más que solo RL al mismo cómputo, y en los problemas difíciles el mid-training más RL supera al RL solo por 10,8%. La mejor recompensa combinó 0,2 de resultado y 0,8 de proceso. "Echo Chamber" entrenó desde cero modelos de 150M y 1B y encontró que el RL converge a un único formato de salida ya presente en el preentrenamiento, distinto según la escala ([Zhao et al. 2025, COLM](https://arxiv.org/abs/2504.07912)). A 135M, el GRPO directo sobre GSM8K bajó la exactitud de 1,82% a 1,59% ([L20-Edu-135M, 2026](https://arxiv.org/abs/2606.22189)). Los éxitos famosos son a ≥1,5B y sobre modelos ya destilados. DeepScaleR-1.5B pasó de 28,8% a 43,1% en AIME24 alargando el contexto de 8K a 16K y a 24K tokens ([Agentica 2025](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)). Dang y Ngo llevaron un modelo de 1,5B ya destilado de 63% a 80% en AMC23 con 7.000 problemas, 4 A40 y 24 horas ([arXiv:2503.16219](https://arxiv.org/abs/2503.16219)).

**Advertencias.** Con recompensas al azar, Qwen2.5-Math-7B ganó 21,4 puntos en MATH500, casi lo mismo que con recompensas correctas (29,1). En Llama y OLMo, esas recompensas no rindieron nada ([Shao et al. 2025, arXiv:2506.10947](https://arxiv.org/abs/2506.10947)). Muchas ganancias de RL publicadas sobre Qwen amplifican capacidades que ya estaban en el preentrenamiento. Hope, entrenado desde cero, no las trae, así que las cifras de Qwen no son una expectativa válida.

**Ajustes que funcionaron.** DAPO, sobre Qwen2.5-32B, pasó de 30 a 50 en AIME24 con cuatro ajustes ([Yu et al. 2025, arXiv:2503.14476](https://arxiv.org/abs/2503.14476)): techo de recorte más alto (ε de 0,2 a 0,28) para no colapsar la entropía; muestreo dinámico, que descarta los grupos todos correctos o todos incorrectos y rellena el lote; pérdida por token; y penalización suave del exceso de largo. Dr. GRPO mostró que normalizar por el largo de cada respuesta alarga las respuestas incorrectas, y que normalizar por el desvío del grupo sesga hacia preguntas muy fáciles o muy difíciles ([Liu et al. 2025, arXiv:2503.20783](https://arxiv.org/abs/2503.20783)). Fatemi et al. atribuyen la verbosidad a la pérdida sobre respuestas incorrectas en problemas que el modelo no puede resolver. Proponen una segunda fase corta de RL sobre problemas resolubles, que acorta sin perder exactitud ([arXiv:2504.05185](https://arxiv.org/abs/2504.05185)).

| Trabajo | Tamaño | Grupo | Largo máximo | KL | Otros |
|---|---|---:|---|---|---|
| CMU, 100M desde cero | 100M | 6 | 1.024 tokens | 1·10⁻³ | LR 1·10⁻⁶; lote 1.024; problemas del borde |
| Thinking Machines (on-policy) | 8B | 4 | — | — | 64 problemas por lote |
| DeepScaleR | 1,5B | 8–16 (no verificado) | 8K → 16K → 24K tokens | — | 40K problemas |
| DAPO | 32B | (no verificado) | 16K + 4K de margen | sin KL | ε 0,2/0,28; muestreo dinámico |
| Hope hoy (`hope/train/rlvr.py`) | 200M | 8 | **200 bytes (~50 tokens)** | 0,02 | normaliza por largo y por desvío; genera el grupo en serie |

**Para Hope.** El RLVR va **después** del SFT con episodios del maestro y de los generadores, y solo sobre problemas con pass@1 entre ~0,1 y ~0,8 y pass@8 > 0, recalibrados cada ronda. Cambios concretos sobre `rlvr.py`:
- Subir `max_new` con un currículo de 512 → 1.024 → 2.048 bytes. Con 200 bytes no entra un razonamiento de varios pasos.
- Generar el grupo de 8 **en un solo lote**. El costo es casi el de una muestra, porque la A770 está limitada por el ancho de banda (§3.9.1).
- Sumar log-probabilidades y dividir por una constante (`max_new`) en vez de por el largo de cada muestra, y centrar la ventaja sin dividir por el desvío (Dr. GRPO).
- Rellenar el lote cuando se descartan grupos sin varianza (DAPO).
- Bajar `kl_weight` a ~1·10⁻³ (CMU) y seguir la KL como métrica.
- Mantener el esquema 1 / 0,2 / 0, que equivale a la variante estricta de proceso de CMU.
- Activar la penalización de costo recién en una fase corta posterior, sobre problemas resolubles (Fatemi).

La expectativa honesta es una ganancia chica en pass@1 dentro de la distribución y una mayor en el borde. Si a 200M el RL no mejora pass@k en la franja borde, conviene dejarlo para 500M.

### 1.5 Cómputo en inferencia: cuándo le gana a un modelo más grande

**Evidencia.** A igual cómputo, un modelo chico con búsqueda y verificador óptimos superó a uno **14 veces más grande**, pero solo en problemas donde el chico ya tenía una tasa de acierto no trivial. En los difíciles rindió más el preentrenamiento ([Snell et al. 2024, arXiv:2408.03314](https://arxiv.org/abs/2408.03314)). Un Llama de 1B con un buen modelo de recompensa de procesos superó a uno de 405B en MATH500 con N = 512, y la estrategia óptima depende del verificador y de la dificultad ([Liu et al. 2025, arXiv:2502.06703](https://arxiv.org/abs/2502.06703)). El muestreo repetido llevó a DeepSeek-Coder-V2 de 15,9% a 56% en SWE-bench Lite con 250 muestras, porque había verificación automática. Sin verificador, la votación y los modelos de recompensa se estancan ([Brown et al. 2024, arXiv:2407.21787](https://arxiv.org/abs/2407.21787)). rStar-Math llevó a Qwen2.5-Math-7B de 58,8% a 90,0% en MATH con búsqueda MCTS y un modelo de preferencias de procesos que se mejoraba a sí mismo ([Guan et al. 2025, arXiv:2501.04519](https://arxiv.org/abs/2501.04519)). En contra, Kinetics muestra que las leyes medidas en FLOPs sobreestiman a los modelos chicos. Al contar los accesos a memoria, el costo dominante es la atención sobre el KV, no los parámetros. La atención dispersa resulta esencial: según la versión del trabajo, da 45–60 puntos más en AIME24 en el régimen de bajo costo ([Sadhukhan et al. 2025, NeurIPS](https://arxiv.org/abs/2506.05333)).

**Cuándo conviene en Hope.** Hope está mejor parado que un transformer para el cómputo en inferencia: los parches dan ~4,5 veces menos posiciones globales, los SSM no tienen KV y la generación especulativa ya está hecha. La regla que sale de la evidencia tiene tres casos según el acuerdo entre las primeras 2–4 muestras, generadas en lote:
- Si coinciden y la herramienta verifica, se responde.
- Si hay desacuerdo, se gasta el tiempo sobrante en más muestras y en verificación con herramientas. Esa es la franja donde el cómputo en inferencia le gana a un modelo más grande.
- Si ninguna es plausible, el problema está fuera de su alcance y conviene recuperar de la base de conocimiento o marcar "no sé". Más muestras ahí no ayudan.

La profundidad latente se usa solo dentro del rango entrenado, porque Ouro empeora fuera de él. El verificador más rentable es una herramienta exacta. Donde no la haya, se usa la cabeza verificadora de §1.3: TinyGSM muestra que escalar el verificador rinde más que escalar el generador.

## 2. Memoria externa y recuperación durante el entrenamiento

### 2.1 La memoria ayuda a los hechos, no al razonamiento

**RETRO.** La reproducción controlada de NVIDIA (148M a 9,5B, el mismo decoder que GPT y el mismo corpus, recuperando de 330B tokens) encontró que RETRO **supera claramente a GPT en tareas de conocimiento y queda a la par en las demás**. Genera con menos repetición y algo más de exactitud factual ([Wang et al. 2023, EMNLP](https://arxiv.org/abs/2304.06762)). **kNN-LM.** En 22 conjuntos, mejora las tareas de memoria pero **empeora las de razonamiento**, sobre todo en GSM8K. Ni con recuperación perfecta resuelve los casos de razonamiento ([Geng, Zhao y Rush 2025, NAACL](https://arxiv.org/abs/2408.11815)). **Memorizing Transformers.** La memoria kNN sobre claves y valores propios mejora el modelado de lenguaje hasta 262K tokens en papers de arXiv, código y teoremas, y permite usar funciones y teoremas definidos en el mismo documento ([Wu et al. 2022, ICLR](https://arxiv.org/abs/2203.08913)).

**Memoria paramétrica dispersa.** Las *memory layers* de Meta (product keys con compuerta SiLU y qk-norm, en 3 capas con una memoria compartida) superan a modelos densos con más del doble de cómputo, y a MoE con iguales parámetros y cómputo ([Berges et al. 2024, arXiv:2412.09764](https://arxiv.org/html/2412.09764)). A 134M, TriviaQA sube de 7,7 a 18,8 y NaturalQuestions de 0,9 a 3,2. Las ganancias son sobre todo factuales, y MBPP bajó levemente. Son capas limitadas por el ancho de banda y necesitaron un kernel CUDA propio 6 veces más rápido que el `EmbeddingBag` de PyTorch. Apple separa un "ancla" chica, que guarda el conocimiento común y el razonamiento, de bloques de memoria que se traen según el documento. Un modelo de 160M con 18M de memoria, tomada de un banco de 4,6B, rinde como uno de más del doble de parámetros. En los hechos más raros (números atómicos, con un modelo base de 1,4B), la exactitud pasa de 17% a 83% con 10% más de parámetros de memoria ([Pouransari et al. 2025, arXiv:2510.02375](https://arxiv.org/abs/2510.02375)).

**Engram**, la excepción relevante. DeepSeek indexa una tabla de embeddings con n-gramas del contexto local, mediante hash y búsqueda O(1), con compuerta y una convolución corta ([Cheng et al. 2026, arXiv:2601.07372](https://arxiv.org/html/2601.07372v1)). A igual cantidad de parámetros y FLOPs que un MoE de 27B, mejora **también el razonamiento**: BBH 50,9 → 55,9, ARC-C 70,1 → 73,8, HumanEval +3,0, GSM8K +2,2, MATH +2,4. En contexto largo, Multi-Query NIAH pasa de 84,2 a 97,0. El análisis interno muestra que las primeras capas dejan de reconstruir patrones estáticos: su capa 5 se parece a la capa 12 del MoE, lo que equivale a más profundidad efectiva. La pérdida tiene forma de U según cómo se reparten los parámetros dispersos, y el óptimo deja **~20–25% para la memoria y el resto para los expertos**. Una tabla de 100B en la RAM del host costó 1,9–2,8% de throughput en H800, sin optimizar.

### 2.2 Cómo encaja en un modelo de bytes

Lo más natural para Hope es la versión de bytes de Engram, que BLT ya validó: **embeddings hash de n-gramas de bytes** sumados en el codificador local. En BLT de 1B, la mejor configuración de la ablación (n = 3 a 8, ~2M entradas en total) bajó el bpb 0,061 en Wikipedia, 0,028 en GitHub y 0,021 en Common Crawl. Los autores lo describen como "vital" para igualar a los modelos con tokenizador, con rendimientos decrecientes por encima de ~300K hashes y ganancias complementarias a la atención cruzada ([Pagnoni et al. 2024, arXiv:2412.09871](https://arxiv.org/html/2412.09871v1)). Para Hope tiene cuatro ventajas:
- Los índices dependen solo de los bytes de entrada, así que se calculan antes del forward. Eso permite dejar la tabla en RAM y traer por adelantado solo las filas del lote, como hace Engram.
- El gradiente es disperso.
- Hope ya calcula hashes de n-gramas para el parcheo por entropía (`entropy_buckets = 65536`).
- No suma FLOPs densos.

Una configuración inicial razonable (estimación) son n ∈ {3,…,8}, 32K entradas por n y dimensión 128 proyectada a `d_local`. Son ~25M parámetros, que entran en la reserva de ~26M sin asignar (§3.8), y su estado de optimizador ocupa unos 150 MB.

El presupuesto de parámetros es una decisión del usuario, no técnica. `growth_budget.yaml` cuenta parámetros totales (techo de 200M), y una tabla hash los suma aunque no agregue cómputo. Si la tabla se considera memoria fuera del conteo, como el AHMC (ASG §3.E), puede crecer hasta el punto de rendimientos decrecientes de BLT sin tocar el resto. Lo mismo vale, con más fuerza, para las *memory layers* y para el banco de Apple. Ese banco mide 4,6B parámetros, ~9 GB en bf16, y entraría en los 64 GB de RAM, pero no en el techo actual.

**Para Hope:**
- **Sí:** embeddings hash de n-gramas de bytes en el codificador local, primero en VRAM con la reserva y medidos en bpb y en exactitud en la franja borde.
- **Mantener** la recuperación por bloque con referencia a la fuente (RETRO/AHMC, Self-Route) para hechos, números y fórmulas.
- **No** interpolar la salida con kNN-LM en tareas de razonamiento.
- **Después**, *memory layers* o memoria por documento estilo Apple, solo si el diagnóstico muestra que Hope falla por hechos raros y no por procedimiento. La base de conocimiento ya cubre ese caso con fuentes.

La atención cruzada de memoria de Hope tiene que **abrirse** durante el entrenamiento. Sin tareas que solo se resuelvan con lo recuperado, la compuerta puede quedar cerrada, como en la reproducción fallida de Infini-attention ([HF](https://huggingface.co/blog/infini-attention)).

## 3. Contexto largo sin más parámetros

Hope ya tiene un backbone SSM más atención, escala NTK de RoPE, memoria de segmentos y una parte global incremental. El informe anterior ya cubrió StreamingLLM, YaRN, Quest, BABILong y la advertencia de RULER: la longitud efectiva de muchos modelos es menor a 4K aunque la nominal sea 128K ([Hsieh et al. 2024](https://arxiv.org/html/2404.06654)). Esta sección agrega lo nuevo y lo específico de modelos chicos.

**Híbridos chicos.** Hymba-1.5B pone cabezas de atención y de SSM en paralelo dentro de cada capa, más tokens "meta" aprendidos al inicio. Supera a Llama-3.2-3B (61,1% frente a 59,7% de promedio) y reporta 11,7–14,7 veces menos caché que un transformer comparable ([Dong et al. 2024, arXiv:2411.13676](https://arxiv.org/abs/2411.13676)). Los meta tokens absorben la atención que iría al primer token. Un análisis sistemático de Meta encontró que los híbridos dentro de la capa y entre capas superan a transformers, Mamba y ventana deslizante puros, y que los híbridos dentro de la capa están en la frontera de Pareto entre calidad y eficiencia ([Bae et al. 2025, arXiv:2510.04800](https://arxiv.org/abs/2510.04800)). LFM2, en cambio, buscó la arquitectura midiendo en el dispositivo y eligió 10 convoluciones cortas con compuerta (núcleo 3) más 6 capas GQA, para tamaños de 350M a 1,2B. Agregar SSM o atención lineal **no mejoró la calidad** una vez que había algunas capas de atención ([Liquid AI 2025](https://arxiv.org/html/2511.23404v1)). Para Hope, la implicancia es de velocidad. La capa Mamba-2 corre como un scan en PyTorch puro sobre XPU, mientras que una convolución corta es trivialmente rápida. Merece una ablación (Mamba-2 frente a convolución con compuerta, a iguales parámetros), midiendo bpb, recuerdo a 8K y bytes por segundo en la A770.

**Atención con compuerta.** Una compuerta sigmoide por cabeza, aplicada a la salida de la atención, mejora la perplejidad y la estabilidad, tolera tasas de aprendizaje más altas, elimina el *attention sink* y las activaciones masivas, y mejora la extrapolación a contextos más largos. Se probó en 30 variantes, de 1,7B denso y de 15B MoE, con 3,5T tokens ([Qiu et al. 2025, NeurIPS](https://arxiv.org/abs/2505.06708)). La usan Qwen3-Next y HRM-Text. Su costo es ~d² parámetros por capa en la variante por elemento (~1M a d = 1024) o ~d·cabezas en la variante por cabeza (~16K). Hope debería ponerla en las capas de atención del backbone, en el núcleo y en la atención cruzada de memoria. Con ella, los sinks fijos de la ventana (`sinks: 4`) pasan a ser un respaldo.

**Memoria aprendida en inferencia.** Titans fue reimplementado independientemente con resultados mixtos: no siempre supera a las líneas base, en parte por el chunking, aunque la memoria neural mejora sistemáticamente a los modelos de solo atención ([Di Nepi et al. 2025, arXiv:2510.09551](https://arxiv.org/abs/2510.09551)). Google presentó en NeurIPS 2025 una arquitectura llamada justamente **HOPE**, dentro de *Nested Learning*: una variante de Titans que se automodifica, con memorias que se actualizan a distintas frecuencias ([Behrouz et al. 2025, blog](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)). Sus resultados públicos son de blog y no tienen reproducción. Conviene saberlo para no confundir referencias, pero **no adoptarlo todavía**.

**Bucle y KV.** Si el núcleo crece en iteraciones, su KV crece con él. Ouro reutiliza el KV de la última iteración en decodificación (4 veces menos memoria, casi sin pérdida). MoR, que comparte el KV entre recursiones, lo divide por la cantidad de recursiones con una pérdida chica (41,9% frente a 42,6%).

**Atención dispersa para razonamientos largos.** Quest selecciona las páginas del caché según la consulta, con hasta 7,03 veces menos latencia ([Tang et al. 2024](https://arxiv.org/abs/2406.10774)). Kinetics muestra que es lo que vuelve rentable el cómputo en inferencia. Para Hope importa cuando las trazas superen ~16K parches, y no antes.

**Datos para contexto largo.** ProLong encontró que los repositorios de código y los libros son las mejores fuentes largas, pero **solo mezclados con datos cortos de alta calidad**. También que entrenar con secuencias más largas que las de evaluación ayuda, y que para el ajuste alcanzan instrucciones cortas ([Gao et al. 2024, arXiv:2410.02660](https://arxiv.org/abs/2410.02660)). La etapa 4 del plan (8 KB en segmentos de 2 KB) debería armar las secuencias largas concatenando archivos del mismo repositorio y documentos largos completos, mantener al menos la mitad de datos cortos de calidad e incluir tareas que solo se resuelvan con la memoria de segmentos: passkey entre segmentos, NIAH con varias claves y seguimiento de variables.

## 4. Datos: currículos y mezclas que más mejoraron el razonamiento

**Razonamiento desde el preentrenamiento.** NVIDIA midió **+19% de promedio** al incluir datos de razonamiento en el preentrenamiento (8B híbrido Mamba-2, 1T tokens), una ventaja que el SFT posterior no recupera ni con más datos. La diversidad de patrones importa más en el preentrenamiento, y la calidad más en el ajuste ([Akter et al. 2025, arXiv:2510.03264](https://arxiv.org/abs/2510.03264)). Ruis et al. rastrearon con funciones de influencia qué documentos sostienen el razonamiento. Para preguntas de hechos influyen documentos con la respuesta; para razonamiento influyen documentos con **procedimientos** (fórmulas y código), no con la respuesta ([Ruis et al. 2024, arXiv:2411.12580](https://arxiv.org/abs/2411.12580)). OctoThinker mostró que un mid-training en matemática de calidad (MegaMath-Web-Pro mejor que FineMath-4+) es lo que vuelve a Llama receptivo al RL, aunque el modelo base no muestre la ganancia. Más presupuesto de mid-training dio mejor RL ([Wang et al. 2025, arXiv:2506.20512](https://arxiv.org/abs/2506.20512)).

**Proporciones reportadas.**
- **Código:** 25% fue óptimo para el razonamiento en lenguaje natural (+3,4% relativo frente a solo texto, y hasta +8,2% con la mejor receta), sostenido hasta 75% y en caída por encima. El conocimiento general baja a medida que sube el código: −3,4% con 25%. El código sintético de calidad sumó +9% en razonamiento cuando el preentrenamiento era solo de código, y +2% en preentrenamiento continuo ([Aryabumi et al. 2024, ICLR 2025](https://arxiv.org/html/2408.10914v1)).
- **SmolLM2-1.7B:** etapa 1 con web 60/40 FineWeb-Edu/DCLM, 10% de código y sin matemática; etapa 2 con 75/20/5; enfriamiento con 58% de web, **24% de Stack-Edu, 14% de matemática** (FineMath-4+, InfiWebMath-3+) y 4% de Cosmopedia. Para 135M y 360M usaron **una sola etapa con datos de calidad desde el inicio**, con matemática y código incluidos ([Allal et al. 2025, arXiv:2502.02737](https://arxiv.org/html/2502.02737v1)). FineMath-4+ dio ~2 veces más en GSM8K y ~6 veces más en MATH que InfiMM-WebMath.
- **MobileLLM-R1:** fase 1 con 64% de FineWeb-Edu, 11% de código y ~9% de matemática; fase 2 con **~37% de matemática y razonamiento**; mid-training final con ~39% de código y ~48% de matemática ([Meta 2025](https://arxiv.org/html/2509.24945v3)).

**Datos sintéticos verificables y con errores.** En el estudio sintético iGSM, preentrenar con datos de "reintento" (un paso erróneo seguido de un token `[BACK]` y la corrección) subió la exactitud de 78% a 94–95%. No hizo falta enmascarar los errores, ni siquiera con 50% de pasos con reintento. El modelo resultante casi no reintenta y sus soluciones siguen siendo cortas ([Ye et al. 2024, ICLR 2025](https://arxiv.org/html/2408.16293v1)). Una variante barata, que inserta un paso futuro al azar seguido de `[BACK]`, rinde casi igual. **Agregarlo en el ajuste fino con LoRA no sirve: tiene que estar en el preentrenamiento.** Reescribir un corpus de matemática con un modelo del mismo tamaño (0,8B), en forma de notas de estudio o de proceso de pensamiento, logró el rendimiento final con **3–6 veces menos tokens** en GSM8K ([Saponati 2026, Tufa Labs](https://matteosaponati.github.io/synth-pretrain-very-small-reasoning-models/)).

**Selección y etiquetas.** Rho-1 entrena solo con los tokens donde el modelo falla y un modelo de referencia de calidad acierta. Logró +6,8% de promedio en TinyLlama-1B con 80B tokens y hasta +30% en matemática few-shot ([Lin et al. 2024, arXiv:2404.07965](https://arxiv.org/abs/2404.07965)). Anteponer el nombre del dominio de origen (como `wikipedia.org`) a cada documento **aumenta significativamente la capacidad de conocimiento** ([Allen-Zhu y Li 2024](https://arxiv.org/abs/2404.05405)).

**Números en un modelo de bytes.** La tokenización de los números cambia la aritmética. Agrupar dígitos de derecha a izquierda llevó a GPT-3.5 de 75,6% a 97,8% ([Singh y Strouse 2024, arXiv:2402.14903](https://arxiv.org/abs/2402.14903)). Las Abacus Embeddings, que codifican la posición de cada dígito dentro de su número, combinadas con un bucle e inyección de la entrada, generalizaron de 20 a 100 dígitos con ~99% ([McLeish et al. 2024](https://arxiv.org/abs/2405.17399)). Hope trabaja en bytes, así que cada dígito ya es un símbolo. Pero el parcheo por espacios puede juntar hasta 8 dígitos en un parche, y entonces el backbone ve un solo vector por tramo de número. Es una hipótesis a medir: forzar cortes de parche en los dígitos (o en grupos de 3 desde la derecha) y agregar una posición "dentro del número" en el codificador local cuesta casi nada y apunta directo al currículo de Hope (aritmética, porcentajes, ley de Ohm).

**Mezcla propuesta para `plan_6m.yaml` (estimación a validar con ablaciones cortas):**

| Fuente | Etapa 1 (hoy) | Etapa 1 propuesta | Enfriamiento propuesto | Base de evidencia |
|---|---:|---:|---:|---|
| Wikipedia ES/EN + web educativa filtrada | 60% | 45% | 25% | SmolLM2, 2 bits/parámetro |
| Código filtrado (estilo Stack-Edu) | 25% | 25% | 25% | Aryabumi; SmolLM2 24% |
| Matemática web (FineMath-4+, MegaMath-Web-Pro; ES donde haya) | 0% | 10% | 15–20% | SmolLM2, MobileLLM-R1, OctoThinker |
| Generadores procedurales verificables, con 20–50% de pasos de reintento y soluciones-programa | 0% (solo en el ciclo) | 5% | 10% | CMU (≥1%), iGSM, TinyGSM |
| Episodios del maestro (cortos, con pérdida en la respuesta) | 15% | 15% | 20–25% | Front-loading, Mix-Long |

Todas las fuentes llevan una etiqueta de dominio en bytes al inicio de cada documento. Los generadores cubren a propósito los "primitivos" (operaciones de un paso en cada contexto) con al menos 1% de densidad, para que el RL posterior pueda componerlos.

## 5. Tabla resumen

Las cifras vienen de otros modelos, escalas y hardware. Ninguna fue medida en Hope ni en una Intel Arc.

| # | Técnica | Ganancia reportada | Parámetros | Cómputo / memoria | Madurez | Hope: ¿aplicar y cómo? |
|---|---|---|---|---|---|---|
| 1 | Bucle de profundidad (looped, LoopLM, MoR, Parcae) | Ouro-1,4B ≈ Qwen3-4B en la mayoría; MoR 167M ≥ 315M a igual cómputo; φ = 0,46 | 0 (+2d con inyección estable) | ×r en la parte en bucle; KV ×r salvo reutilización | Media–alta | Sí: reentrenar el núcleo (§1.1.1) |
| 2 | Profundidad antes que ancho + bloques compartidos (MobileLLM) | +2,7/+4,3 pts a 125M/350M; +0,7/+0,8 compartiendo | 0 | más latencia secuencial | Alta | Sí, antes de fijar el preset de la corrida larga |
| 3 | Parada por convergencia, sin cabeza aprendida | −38% de iteraciones a igual calidad (135M) | 0 | ahorra cómputo en inferencia | Baja | Sí: hacerla el valor por defecto |
| 4 | Razonamiento latente (Coconut, CODI) | CODI ≈ CoT en GPT-2 con 3,1× de compresión; 90% de CoT a 1B | 0 | 2,7–5,9× más rápido | Baja–media | No por ahora (pierde verificación de pasos) |
| 5 | Tokens de pausa / PonderLM | +1 pt en GSM8K (1B); PonderLM ≈ la mitad de parámetros con más pasadas | ~0 | pasadas extra por token | Baja–media | Postergar |
| 6 | SFT adaptado al alumno (CoT corta, Mix-Long 1:4, maestros chicos) | +8 pts en MATH/AMC (3B) frente a CoT larga | 0 | menos bytes del maestro | Media | Sí |
| 7 | Pérdida solo en la respuesta + PrefixLM | MMLU 40,6 → 53,2 (1B, igual cómputo) | 0 | máscara de atención | Baja–media | Sí, barato |
| 8 | Soluciones como programas + verificador | 1,3B + verificador 1,3B: 81,5% en GSM8K; verificador más grande: +7,2 | cabeza verificadora | N muestras | Media | Sí |
| 9 | Destilación de logits entre tokenizadores (ALM) | +1–1,6 pts sobre SFT; subpalabras → bytes viable | 0 | prefill del maestro; log-probs en disco | Media | Experimento |
| 10 | Destilación on-policy | 74,4 frente a 67,6 en AIME con 1/10 de GPU-h; 9–30× más barata que SFT | 0 | prefill del maestro sobre muestras del alumno | Media (mismo tokenizador); sin evidencia en bytes | Experimento con alineación ALM |
| 11 | GRPO al borde de la competencia (+DAPO, Dr. GRPO) | 100M sintético: hasta +60% pass@128 fuera de distribución; 1,5B: AIME 28,8 → 43,1; 135M en web: empeoró | 0 (+ copia de referencia) | dominado por la generación | Alta ≥1,5B; baja <500M | Sí, después del SFT (§1.4) |
| 12 | Muestras en lote + verificación con herramientas | 15,9 → 56% con 250 muestras; ≈ modelo 14× mayor en problemas fáciles/medios | 0 | N muestras (casi gratis en lote) | Alta | Sí, con la regla de §1.5 |
| 13 | Embeddings hash de n-gramas de bytes (BLT) / Engram | −0,02 a −0,06 bpb (BLT 1B); Engram: BBH +5,0, GSM8K +2,2 | tabla dispersa (~25M propuesta) | lookup O(1); puede vivir en RAM | Media–alta | Sí |
| 14 | Memory layers (product keys) | > denso con 2× cómputo; 134M: TriviaQA 7,7 → 18,8 | grande (≥67M a d = 1024) | limitado por ancho de banda; sin kernel XPU | Media | Después, solo para hechos |
| 15 | Memoria jerárquica por documento (Apple) | 160M + 18M de 4,6B ≈ modelo 2× mayor | banco grande en RAM | traer bloques por documento | Baja–media | Experimento posterior |
| 16 | RETRO / kNN-LM | RETRO: mejor en conocimiento, igual en el resto; kNN-LM empeora el razonamiento | 0–10M | índice | Alta | Mantener recuperación por bloque; no kNN-LM |
| 17 | Atención con compuerta sigmoide (+ meta tokens) | mejor perplejidad y estabilidad, sin sink, mejor extrapolación | ~1M o ~16K por capa | despreciable | Alta (Qwen3-Next) | Sí |
| 18 | Convolución corta + pocas capas GQA en vez de SSM (LFM2) | igual calidad a 350M–1,2B, más rápido en dispositivo | similar | sin scan | Media | Ablación en la A770 |
| 19 | Atención dispersa por páginas (Quest) | 7,03× menos latencia; clave para el cómputo en inferencia | 0 | gather/top-k | Media | Con trazas de más de ~16K parches |
| 20 | Memoria aprendida en inferencia (Titans, HOPE de Nested Learning) | reproducción con resultados mixtos | módulo extra | actualizaciones en inferencia | Baja | No por ahora |
| 21 | Datos de razonamiento desde el preentrenamiento | +19% de promedio (8B); reintentos 78 → 94% (iGSM); 1% de exposición para que el RL transfiera | 0 | CPU para generar | Media–alta | Sí, prioridad 1 |
| 22 | Código ~25% | +3,4% relativo en razonamiento con 25% óptimo; el conocimiento baja | 0 | — | Media–alta | Ya está en 25%; filtrarlo por calidad |
| 23 | Etiquetas de dominio | "aumenta significativamente" la capacidad de conocimiento | 0 | unos bytes por documento | Media (sintético) | Sí, trivial |
| 24 | Selección de tokens (Rho-1) | +6,8% de promedio; hasta +30% en matemática few-shot | 0 | modelo de referencia | Media | Más adelante |
| 25 | Números: cortes por dígito y posición dentro del número | 75,6 → 97,8% (GPT-3.5, de derecha a izquierda); Abacus + bucle 99% a 100 dígitos | ~0 | — | Media | Sí, cortes de parche en dígitos |

## 6. Top 8 priorizado para Hope HN-1

El orden pondera la ganancia esperada en razonamiento por parámetro, el costo en la A770 (16 GB, una GPU, sin kernels CUDA), la madurez y lo que ya existe en el repositorio.

1. **Datos de razonamiento desde el preentrenamiento.** Matemática web de calidad (10% → 15–20% en el enfriamiento), generadores verificables con pasos de reintento y soluciones-programa (≥1–5%), código filtrado al 25% y etiquetas de dominio. No suma parámetros y es la palanca con más evidencia (+19%; 78 → 94% en iGSM).
2. **Reentrenar el núcleo recurrente como un bucle estable.** Inyección de la entrada con Ā diagonal negativa (Parcae), sin `step_embed` fijo, profundidad muestreada de cola pesada, backprop completo o con horizonte creciente, parada por convergencia y EMA. En paralelo, probar un backbone más profundo y angosto con más bloques dentro del bucle.
3. **SFT con el maestro adaptado a un alumno chico.** Explicaciones cortas por defecto (Mix-Long 1:4), respuestas de un Qwen de 1,5–3B mezcladas, pérdida solo en la respuesta con PrefixLM y soluciones-programa verificadas.
4. **GRPO corregido y apuntado al borde de la competencia, después del SFT.** Grupo de 8 generado en lote, currículo de 512 → 2.048 bytes, sin normalizar por largo ni por desvío, relleno de grupos sin varianza, KL ~1·10⁻³, recompensa de proceso y la penalización de costo en una fase corta al final.
5. **Representación de bytes con memoria barata.** Embeddings hash de n-gramas de bytes en el codificador local (~25M, de la reserva o en RAM), más cortes de parche en los dígitos con posición dentro del número.
6. **Cómputo en inferencia con muestras en lote y verificación por herramientas.** Escalar solo en los problemas de dificultad media (desacuerdo entre muestras), usar la profundidad latente solo dentro del rango entrenado y entrenar una cabeza verificadora con los datos de STaR y RLVR.
7. **Destilación on-policy con prefill del maestro y alineación por fragmentos (ALM), como experimento.** Convierte el cuello de botella de la generación del maestro en un prefill paralelo y da una señal densa sobre las muestras de Hope.
8. **Atención con compuerta sigmoide** en las capas de atención, el núcleo y la memoria. Acompañarla con un contexto largo estilo ProLong (repositorios y documentos largos con al menos 50% de datos cortos de calidad) y con tareas que obliguen a abrir la memoria de segmentos.

## 7. Lo que conviene no hacer, o postergar

- **ACT con costo de ponder como mecanismo de entrenamiento.** Ya provocó el colapso a una iteración. La literatura de 2025–2026 entrena con profundidad muestreada y detiene por convergencia o con una compuerta entrenada aparte.
- **Esperar que el bucle extrapole más allá de lo entrenado.** Ouro empeora fuera de T = 4. Solo Huginn, con profundidad de cola pesada en el entrenamiento, escala.
- **RL directo sobre el modelo base, o sobre problemas que el modelo no resuelve nunca.** A 135M empeoró. Los resultados con Qwen no son una referencia para un modelo entrenado desde cero (recompensas espurias).
- **Explicaciones largas del maestro como formato único.** Los modelos de 3B o menos aprenden peor de ellas, y además cuestan más tiempo de la A770.
- **Razonamiento latente (Coconut/CODI) e interpolación kNN-LM** para razonar. El primero pierde la verificación de pasos y el segundo empeora el razonamiento medido.
- **Titans o el HOPE de Nested Learning** como núcleo de memoria, hasta que haya reproducciones independientes favorables.
- **Memory layers grandes o un banco de memoria de miles de millones de parámetros** antes de que el diagnóstico muestre fallas por hechos raros. Ayudan sobre todo a los hechos y no tienen kernel eficiente en XPU.
- **Comparar a Hope con GSM8K y MATH generales como criterio de éxito.** Con el presupuesto de tokens de una A770, la señal útil está en la generalización composicional dentro de los dominios del currículo, medida por franjas de complejidad.

## Referencias

**Profundidad recurrente y razonamiento latente**
- Saunshi, Dikkala, Li, Kumar y Reddi (2025). *Reasoning with Latent Thoughts: On the Power of Looped Transformers*. ICLR. [arXiv:2502.17416](https://arxiv.org/abs/2502.17416)
- Zhu, Wang, Hua, Zhang et al. (2025). *Scaling Latent Reasoning via Looped Language Models* (Ouro). [arXiv:2510.25741](https://arxiv.org/html/2510.25741v2)
- Bae et al. (2025). *Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation*. NeurIPS. [arXiv:2507.10524](https://arxiv.org/html/2507.10524v3)
- Schwethelm, Rueckert y Kaissis (2026). *How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models*. [arXiv:2604.21106](https://arxiv.org/abs/2604.21106)
- Prairie, Novack, Berg-Kirkpatrick y Fu (2026). *Parcae: Scaling Laws for Stable Looped Language Models*. [arXiv:2604.12946](https://arxiv.org/abs/2604.12946); [blog de Together AI](https://together.ai/blog/parcae)
- Logan (2026). *Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers*. [arXiv:2607.14427](https://arxiv.org/abs/2607.14427)
- Geiping et al. (2025). *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (Huginn). [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)
- McLeish et al. (2025). *Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence*. [arXiv:2511.07384](https://arxiv.org/abs/2511.07384)
- Chen et al. (2025). *Inner Thinking Transformer*. ACL. [arXiv:2502.13842](https://arxiv.org/abs/2502.13842)
- Liu et al. (2024). *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases*. ICML. [arXiv:2402.14905](https://arxiv.org/abs/2402.14905)
- Jolicoeur-Martineau (2025). *Less is More: Recursive Reasoning with Tiny Networks* (TRM). [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)
- Wang et al. (2026). *HRM-Text: Efficient Pretraining Beyond Scaling*. [arXiv:2605.20613](https://arxiv.org/html/2605.20613)
- Hao et al. (2024). *Training Large Language Models to Reason in a Continuous Latent Space* (Coconut). [arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
- Shen et al. (2025). *CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation*. EMNLP. [arXiv:2502.21074](https://arxiv.org/html/2502.21074v3)
- Goyal et al. (2024). *Think before you speak: Training Language Models With Pause Tokens*. ICLR. [arXiv:2310.02226](https://arxiv.org/abs/2310.02226)
- *Pretraining Language Models to Ponder in Continuous Space* (PonderLM, 2025). [arXiv:2505.20674](https://arxiv.org/abs/2505.20674)
- McLeish et al. (2024). *Transformers Can Do Arithmetic with the Right Embeddings* (Abacus). [arXiv:2405.17399](https://arxiv.org/abs/2405.17399)

**Destilación y refuerzo**
- Li, Yue et al. (2025). *Small Models Struggle to Learn from Strong Reasoners*. ICML. [arXiv:2502.12143](https://arxiv.org/abs/2502.12143)
- Meta AI (2025). *MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes*. ICLR 2026. [arXiv:2509.24945](https://arxiv.org/html/2509.24945v3)
- Minixhofer, Vulić y Ponti (2025). *Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching*. NeurIPS. [arXiv:2503.20083](https://arxiv.org/html/2503.20083v3)
- AI2 y colaboradores (2025). *Bolmo: Byteifying the Next Generation of Language Models*. [arXiv:2512.15586](https://arxiv.org/abs/2512.15586)
- Lu et al. (2025). *On-Policy Distillation*. Thinking Machines. [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Liu et al. (2023). *TinyGSM: achieving >80% on GSM8k with small language models*. [arXiv:2312.09241](https://arxiv.org/abs/2312.09241)
- Zhang, Neubig y Yue (2025). *On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models*. ICML 2026. [arXiv:2512.07783](https://arxiv.org/html/2512.07783)
- Zhao, Meterez, Kakade, Pehlevan, Jelassi y Malach (2025). *Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining*. COLM. [arXiv:2504.07912](https://arxiv.org/abs/2504.07912)
- L20-Edu-135M (2026). *An Auditable Single-GPU Study of Data-Efficient Small Language Modeling*. [arXiv:2606.22189](https://arxiv.org/abs/2606.22189)
- Agentica (2025). *DeepScaleR-1.5B-Preview*. [ficha del modelo](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)
- Dang y Ngo (2025). *Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't*. [arXiv:2503.16219](https://arxiv.org/abs/2503.16219)
- Zeng et al. (2025). *SimpleRL-Zoo*. [arXiv:2503.18892](https://arxiv.org/abs/2503.18892)
- Yu et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
- Liu et al. (2025). *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO). [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)
- Fatemi, Rafiee, Tang y Talamadupula (2025). *Concise Reasoning via Reinforcement Learning*. [arXiv:2504.05185](https://arxiv.org/abs/2504.05185)
- Shao et al. (2025). *Spurious Rewards: Rethinking Training Signals in RLVR*. [arXiv:2506.10947](https://arxiv.org/abs/2506.10947)

**Cómputo en inferencia**
- Snell, Lee, Xu y Kumar (2024). *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*. [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
- Liu et al. (2025). *Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling*. [arXiv:2502.06703](https://arxiv.org/abs/2502.06703)
- Brown et al. (2024). *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*. [arXiv:2407.21787](https://arxiv.org/abs/2407.21787)
- Sadhukhan et al. (2025). *Kinetics: Rethinking Test-Time Scaling Laws*. NeurIPS. [arXiv:2506.05333](https://arxiv.org/abs/2506.05333)
- Guan et al. (2025). *rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking*. ICML. [arXiv:2501.04519](https://arxiv.org/abs/2501.04519)

**Memoria y recuperación**
- Wang et al. (2023). *Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study*. EMNLP. [arXiv:2304.06762](https://arxiv.org/abs/2304.06762)
- Geng, Zhao y Rush (2025). *Great Memory, Shallow Reasoning: Limits of kNN-LMs*. NAACL. [arXiv:2408.11815](https://arxiv.org/abs/2408.11815)
- Wu, Rabe, Hutchins y Szegedy (2022). *Memorizing Transformers*. ICLR. [arXiv:2203.08913](https://arxiv.org/abs/2203.08913)
- Berges et al. (2024). *Memory Layers at Scale*. [arXiv:2412.09764](https://arxiv.org/html/2412.09764)
- Pouransari, Grangier et al. (2025). *Pretraining with Hierarchical Memories: Separating Long-Tail and Common Knowledge*. ICLR 2026. [arXiv:2510.02375](https://arxiv.org/abs/2510.02375)
- Cheng et al. (2026). *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models* (Engram). [arXiv:2601.07372](https://arxiv.org/html/2601.07372v1)
- Pagnoni et al. (2024). *Byte Latent Transformer: Patches Scale Better Than Tokens*. [arXiv:2412.09871](https://arxiv.org/html/2412.09871v1)
- Hugging Face (2024). *A failed experiment: Infini-Attention*. [blog](https://huggingface.co/blog/infini-attention)

**Contexto largo**
- Qiu et al. (2025). *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*. NeurIPS. [arXiv:2505.06708](https://arxiv.org/abs/2505.06708)
- Dong et al. (2024). *Hymba: A Hybrid-head Architecture for Small Language Models*. ICLR 2025. [arXiv:2411.13676](https://arxiv.org/abs/2411.13676)
- Bae et al. (2025). *Hybrid Architectures for Language Models: Systematic Analysis and Design Insights*. [arXiv:2510.04800](https://arxiv.org/abs/2510.04800)
- Liquid AI (2025). *LFM2 Technical Report*. [arXiv:2511.23404](https://arxiv.org/html/2511.23404v1)
- Di Nepi, Siciliano y Silvestri (2025). *Titans Revisited: A Lightweight Reimplementation and Critical Analysis of a Test-Time Memory Model*. [arXiv:2510.09551](https://arxiv.org/abs/2510.09551)
- Behrouz, Razaviyayn, Zhong y Mirrokni (2025). *Nested Learning: The Illusion of Deep Learning Architectures*. NeurIPS. [blog de Google Research](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)
- Tang et al. (2024). *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference*. [arXiv:2406.10774](https://arxiv.org/abs/2406.10774)
- Hsieh et al. (2024). *RULER: What's the Real Context Size of Your Long-Context Language Models?* [arXiv:2404.06654](https://arxiv.org/html/2404.06654)
- Gao, Wettig, Yen y Chen (2024). *How to Train Long-Context Language Models (Effectively)*. [arXiv:2410.02660](https://arxiv.org/abs/2410.02660)

**Datos**
- Akter, Prabhumoye et al. (2025). *Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data*. [arXiv:2510.03264](https://arxiv.org/abs/2510.03264)
- Ruis et al. (2024). *Procedural Knowledge in Pretraining Drives Reasoning in Large Language Models*. [arXiv:2411.12580](https://arxiv.org/abs/2411.12580)
- Wang, Zhou, Li y Liu (2025). *OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling*. [arXiv:2506.20512](https://arxiv.org/abs/2506.20512)
- Aryabumi et al. (2024). *To Code, or Not To Code? Exploring Impact of Code in Pre-training*. ICLR 2025. [arXiv:2408.10914](https://arxiv.org/html/2408.10914v1)
- Allal et al. (2025). *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model*. [arXiv:2502.02737](https://arxiv.org/html/2502.02737v1)
- Ye, Xu, Li y Allen-Zhu (2024). *Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems*. ICLR 2025. [arXiv:2408.16293](https://arxiv.org/html/2408.16293v1)
- Allen-Zhu y Li (2024). *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws*. [arXiv:2404.05405](https://arxiv.org/abs/2404.05405)
- Saponati (2026). *Synthetic pretraining for very small reasoning models*. Tufa Labs. [entrada](https://matteosaponati.github.io/synth-pretrain-very-small-reasoning-models/)
- Lin et al. (2024). *Rho-1: Not All Tokens Are What You Need*. [arXiv:2404.07965](https://arxiv.org/abs/2404.07965)
- Singh y Strouse (2024). *Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs*. [arXiv:2402.14903](https://arxiv.org/abs/2402.14903)
