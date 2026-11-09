# Decodificación especulativa y multi-token prediction (MTP) para generación rápida — aplicabilidad a Hope HN-1

Notas de investigación (nov. 2026). Todas las cifras de speedup son las reportadas por los autores salvo indicación; casi todas están medidas en GPUs NVIDIA, batch 1, modelos de 7B–70B. "Lossless" = la distribución de salida es idéntica a la del modelo objetivo (verificación por rejection sampling / greedy match).

## 1. Draft-model speculative decoding (Leviathan et al. 2023; Chen et al. 2023)

### Takeaway
El método clásico (modelo draft pequeño + verificación paralela por el modelo objetivo) da 2–3x en modelos grandes sin cambiar la salida, pero requiere un draft mucho más barato que el objetivo y con alta tasa de aceptación; para un objetivo de 25M–200M no existe un draft separado "mucho más barato" que merezca la pena, así que para Hope solo tienen sentido variantes self-drafting o drafts sin red (n-gram / prompt lookup).

### Cited Findings
- Leviathan, Kalman, Matias (ICML 2023): un modelo de aproximación más pequeño propone tokens que el modelo grande verifica en paralelo; salidas idénticas, sin reentrenar ni cambiar arquitectura; 2X–3X de aceleración sobre la implementación estándar T5X en T5-XXL. — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Chen et al. (DeepMind, 2023), "speculative sampling": rejection sampling modificado que preserva la distribución del objetivo; 2–2.5x de speedup en Chinchilla 70B en un setup distribuido. — [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Spec-Bench (Xia et al., ACL Findings 2024), Vicuna-7B, greedy, batch 1, RTX 3090: autoregresivo 36.74 tok/s (1.00x); SpS (draft model) 1.77x (64.85 tok/s); en A100 SpS 1.59x. — [Spec-Bench, arXiv:2401.07851](https://arxiv.org/html/2401.07851)
- Spec-Bench: los speedups de casi todos los métodos son mayores en A100 que en RTX 3090 (más cómputo ocioso disponible para el drafting); bajan al subir la temperatura de muestreo; en FP32 caen notablemente (EAGLE 2.39x→1.74x, PLD →1.01x); batch>1 queda como problema abierto (el batch avanza al ritmo de la muestra más lenta). — [arXiv:2401.07851](https://arxiv.org/html/2401.07851)
- Spec-Bench: la mayoría de métodos rinden mal en traducción; Prompt Lookup Decoding (PLD, drafts por coincidencia de n-gramas en el prompt, sin red) logra 1.62x en 3090 pero gana poco en traducción y QA. — [arXiv:2401.07851](https://arxiv.org/html/2401.07851)
- MambaByte (Wang et al., COLM 2024): modelo SSM a nivel de bytes; adaptan decodificación especulativa con *drafting tokenizado* (draft subword Mamba) y *verificación a nivel de byte*, obteniendo 2.6x de speedup sobre MambaByte estándar y eficiencia similar al Mamba subword. — [arXiv:2401.13660](https://arxiv.org/abs/2401.13660)

### Inferences
- Fórmula clave (del análisis de Leviathan et al.): con tasa de aceptación por token α y γ tokens draft, tokens esperados por paso de verificación = (1−α^(γ+1))/(1−α); el speedup real se divide por (γ·c + 1) donde c = coste relativo draft/objetivo. Con c no despreciable (p.ej. draft de 5M para objetivo de 25M, c≈0.2+ en práctica por overhead de Python/lanzamiento de kernels, que no escala con tamaño) la ganancia desaparece. Por eso falla para objetivos pequeños: el coste por paso de un modelo de 25M es dominado por overhead fijo (dispatch, memoria), no por FLOPs, y un draft más pequeño apenas es más barato.
- Lo que sí escala bien en modelos pequeños es la mitad "verificación": verificar k posiciones en un forward cuesta casi lo mismo que 1 posición cuando el hardware está infrautilizado (caso del A770 con 25M a batch 1). Esto favorece drafts *casi gratuitos* (PLD/n-gram, cabezas MTP, early-exit).
- MambaByte es el precedente más cercano a Hope (byte-level + SSM): demuestra que la verificación especulativa con estado SSM es viable (requiere guardar/restaurar el estado recurrente en caso de rechazo).

### Gaps
- No encontré medidas publicadas de speculative decoding con objetivos <500M parámetros ni en Intel Arc/XPU.
- Spec-Bench no reporta tasas de aceptación por método en el texto revisado.

## 2. Medusa, EAGLE-1/2/3, Hydra, Lookahead, self-speculative / LayerSkip

### Takeaway
Las cabezas de draft acopladas al propio modelo (Medusa, Hydra, EAGLE) son el estado del arte en producción (vLLM, SGLang, TensorRT-LLM), con 2–6.5x en modelos 7B–70B; EAGLE-3 es el más rápido reportado. Para Hope lo más transferible son: cabezas tipo Medusa/Hydra sobre el decoder local de bytes y el drafting por early-exit (LayerSkip/Draft&Verify), que encaja de forma natural con el núcleo recurrente con halting adaptativo.

### Cited Findings
**Medusa (Cai et al., ICML 2024)**
- Añade cabezas de decodificación extra que predicen varios tokens siguientes en paralelo; tree attention para verificar varios candidatos en un paso; sin modelo draft. Medusa-1 (cabezas sobre backbone congelado, lossless) >2.2x; Medusa-2 (fine-tune conjunto con receta especial) 2.3–3.6x; propone self-distillation cuando no hay datos y "typical acceptance" (no estrictamente lossless) para subir la aceptación. — [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- Spec-Bench: Medusa 1.48x en RTX 3090 y 2.42x en A100 (Vicuna-7B); Medusa y Lookahead son los que más ganan con mejor hardware. — [arXiv:2401.07851](https://arxiv.org/html/2401.07851)

**Hydra (Ankner et al., 2024)**
- Cabezas de draft *secuencialmente dependientes* (cada predicción condiciona en los tokens ya drafteados), a diferencia de las cabezas independientes de Medusa; la receta Hydra++ da hasta 1.31x más throughput que Medusa y hasta 2.70x sobre decodificación autoregresiva. — [arXiv:2402.05109](https://arxiv.org/abs/2402.05109)

**EAGLE-1/2/3 (Li et al., ICML 2024 / EMNLP 2024 / 2025)**
- EAGLE-1: autoregresión a nivel de *features* (penúltima capa) más el token desplazado un paso para reducir incertidumbre; 2.7x–3.5x de latencia en LLaMA2-Chat 70B, throughput duplicado, lossless. — [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
- EAGLE-2: árbol de draft dinámico y dependiente del contexto usando la confianza del draft como proxy de la aceptación; 3.05x–4.26x, 20–40% más rápido que EAGLE-1, lossless. — [arXiv:2406.16858](https://arxiv.org/abs/2406.16858)
- EAGLE-3: abandona la predicción de features, predice tokens directamente, fusiona features de varias capas ("training-time test"); speedup hasta 6.5x (~1.4x sobre EAGLE-2); 1.38x de throughput en SGLang con batch 64. — [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)
- Repo oficial: draft de ~0.24B para Vicuna-7B, 0.25B para LLaMA3-8B, 1.05B para Qwen2-72B (≈3–4% del objetivo, un bloque decoder); entrenable en 1–2 días en 8x RTX 3090; integrado en vLLM, SGLang, TensorRT-LLM, MLC-LLM, AMD ROCm, NeMo; EAGLE-3 ~5.6x en Vicuna-13B (2x RTX 3090, fp16). — [GitHub SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)
- Spec-Bench: EAGLE es el mejor en RTX 3090 (2.08x, 76.30 tok/s, Vicuna-7B); en A100 2.39x. — [arXiv:2401.07851](https://arxiv.org/html/2401.07851)

**Lookahead decoding (Fu et al., ICML 2024)**
- Decodificación paralela exacta (Jacobi + n-gram pool), sin modelo draft ni datastore; hasta 1.8x en MT-bench y hasta 4x en code completion con escalado en varias GPUs. — [arXiv:2402.02057](https://arxiv.org/abs/2402.02057)
- Spec-Bench (medición independiente): Lookahead solo 1.11x en RTX 3090 y 1.77x en A100 — necesita mucho cómputo ocioso. — [arXiv:2401.07851](https://arxiv.org/html/2401.07851)

**Self-speculative / early-exit**
- Draft & Verify (Zhang et al., ACL 2024): el draft se genera saltando selectivamente capas intermedias del mismo LLM; verificación con el modelo completo en un forward; salida idéntica, sin entrenamiento adicional ni memoria extra; hasta 1.99x en LLaMA-2. — [arXiv:2309.08168](https://arxiv.org/abs/2309.08168)
- LayerSkip (Elhoushi et al., ACL 2024): entrenamiento con layer dropout creciente con la profundidad + early-exit loss con cabeza de salida compartida; drafting con capas tempranas y verificación con las restantes reutilizando cómputo y activaciones (KV) → menor memoria que otros métodos; hasta 2.16x (CNN/DM), 1.82x (código), 2.0x (TOPv2). Requiere (pre/continued/fine-)training con la receta. — [arXiv:2404.16710](https://arxiv.org/abs/2404.16710)

### Inferences
- Madurez: EAGLE(-3) y Medusa = producción (integrados en motores de serving); Hydra, Lookahead, LayerSkip, Draft&Verify = investigación con código abierto, uso en producción limitado.
- Coste de entrenamiento: Medusa-1/EAGLE/Hydra entrenan solo cabezas (horas–1–2 días GPU para 7B–13B; para un modelo de 25M–200M sería de minutos–horas). LayerSkip y Medusa-2 tocan el backbone (más caro, riesgo de calidad).
- Complejidad: chain drafting (lineal) es trivial; tree attention requiere máscaras de atención custom y manejo de KV por rama — implementable en PyTorch puro con `scaled_dot_product_attention(attn_mask=...)`, pero con SSM (Mamba) los árboles son mucho más difíciles porque el estado recurrente no se "ramifica" barato. Para Hope: preferir drafts lineales (o pocas ramas con estados SSM duplicados).
- Encaje especial con Hope: el núcleo recurrente de pesos compartidos con halting adaptativo es literalmente un modelo de profundidad variable → draft con 1–2 iteraciones del núcleo y verificación con el número completo de iteraciones (análogo directo a LayerSkip/Draft&Verify). Entrenar con pérdida en salidas tempranas (como LayerSkip) aumentaría la aceptación.

### Gaps
- No hay resultados públicos de Medusa/EAGLE/Hydra en modelos <1B ni en arquitecturas byte-level jerárquicas.
- No verifiqué cifras de Hydra en Spec-Bench (no aparecen en el texto revisado).

## 3. Multi-token prediction: Gloeckle et al. 2024 y DeepSeek-V3

### Takeaway
MTP como objetivo de pre-entrenamiento mejora la calidad a gran escala (≥~1–3B) y permite self-speculation de ~3x con 4 tokens (7B) y 6.4x con 8 bytes en un modelo byte-level de 7B — el resultado más relevante para Hope; pero en modelos pequeños (0.3B–0.6B) MTP *empeora* la calidad, así que para 25M–200M conviene usarlo con peso bajo / profundidad 1 (estilo DeepSeek) o entrenar cabezas sobre backbone congelado.

### Cited Findings
**Gloeckle et al. (Meta, ICML 2024), "Better & Faster LLMs via Multi-token Prediction"**
- Arquitectura: tronco transformer compartido + n cabezas independientes (cada una una capa transformer) + matriz de unembedding compartida; las cabezas predicen los n tokens siguientes en paralelo. — [arXiv:2404.19737](https://arxiv.org/html/2404.19737v1)
- Entrenamiento eficiente en memoria: forward/backward de cada cabeza secuencialmente acumulando gradiente en el tronco, liberando logits → memoria pico de O(nV+d) a O(V+d); tiempo de entrenamiento 1.02–1.22x respecto a next-token para n=2,4 (overhead atribuido a FSDP). — [arXiv:2404.19737](https://arxiv.org/html/2404.19737v1)
- Calidad: modelos de 13B resuelven ~12% más HumanEval y ~17% más MBPP; beneficio crece con el tamaño del modelo. — [arXiv:2404.19737](https://arxiv.org/abs/2404.19737)
- Velocidad: modelo de 7B con 4 tokens, self-speculative greedy: 3.0x en código (2.5 tokens aceptados de 3 sugeridos de media), 2.7x en texto; tabla S2: 2.74x Wikipedia, 2.67x Books, 3.05x Code (3.12, 3.09, 3.50 tokens por forward); speedup constante con batch hasta 42. Cabezas pre-entrenadas con MTP son más precisas que añadirlas por fine-tuning a un modelo next-token. — [arXiv:2404.19737](https://arxiv.org/html/2404.19737v1)
- **Byte-level**: modelo de 7B a nivel de bytes, 314B bytes (~116B tokens), predicción de 8 bytes: +67% MBPP pass@1 y +20% HumanEval pass@1 frente a next-byte (MBPP@1 19.3→32.3; HumanEval@1 18.1→21.8); self-speculative decoding da **6.4x** (6.39 a batch 16); los autores dicen que lo acerca a modelos tokenizados pese a 1.7x menos datos. — [arXiv:2404.19737](https://arxiv.org/html/2404.19737v1)
- **Tamaño pequeño**: MTP es "peor que el baseline en tamaños pequeños, pero mejor a escala". 0.3B: n=2 y n=4 por debajo de n=1 (MBPP@1 1.7 y 1.0 vs 1.8; HumanEval@1 1.5 y 1.2 vs 1.9). 0.6B: n=4 claramente peor (MBPP@1 3.0 vs 4.7), n=2 ~empate. 1.3B: n=2/n=4 ligeramente mejores (MBPP@1 7.3/7.4 vs 6.8). En tareas de inducción la ventaja de 2 tokens desaparece hacia ~100M parámetros no-embedding. — [arXiv:2404.19737](https://arxiv.org/html/2404.19737v1)
- Lenguaje natural (7B, 200B tokens): n=2 empata con el baseline en 6 benchmarks estándar, n=4 regresa ligeramente; en resumen (ROUGE-L) ambos mejoran; n=4 fue el mejor en código. — [arXiv:2404.19737](https://arxiv.org/html/2404.19737v1)

**DeepSeek-V3 (DeepSeek-AI, dic. 2024)**
- Módulos MTP *secuenciales* (mantienen la cadena causal, a diferencia de las cabezas paralelas de Gloeckle): cada módulo = embedding compartido + cabeza de salida compartida + un bloque transformer + matriz de proyección M_k ∈ R^(d×2d) que combina (concatenando, tras RMSNorm) la representación de la profundidad anterior con el embedding del token en offset k. — [DeepSeek-V3 report, arXiv:2412.19437](https://arxiv.org/html/2412.19437v2)
- Profundidad D=1 (predice 1 token extra); peso de pérdida λ=0.3 durante los primeros 10T tokens y 0.1 durante los 4.8T restantes. — [arXiv:2412.19437](https://arxiv.org/html/2412.19437v2)
- Ablación: mejora consistente en la mayoría de benchmarks tanto en MoE de 15.7B (1.33T tokens) como de 228.7B (540B tokens); el módulo MTP puede descartarse en inferencia (coste de inferencia sin cambios). — [arXiv:2412.19437](https://arxiv.org/html/2412.19437v2)
- Usado para speculative decoding: tasa de aceptación del segundo token 85–90% según tema; ~1.8x TPS. — [arXiv:2412.19437](https://arxiv.org/html/2412.19437v2)

### Inferences
- Para Hope (bytes): los bytes son mucho más predecibles que tokens BPE (muchos bytes dentro de una palabra están casi determinados), por lo que la aceptación de cabezas de k bytes debería ser alta → la evidencia de 6.4x con 8 bytes (7B) es la más alentadora, aunque no está demostrada a 25M–200M.
- Riesgo de calidad a pequeña escala: con 25M–200M los datos de Gloeckle sugieren que n≥4 como objetivo principal perjudica. Opciones de menor riesgo: (a) MTP estilo DeepSeek (D=1, λ 0.1–0.3) como pérdida auxiliar; (b) cabezas entrenadas sobre backbone congelado (Medusa-1, lossless, sin tocar calidad); (c) en todo caso, con verificación lossless la *calidad de salida* nunca cambia — solo la aceptación/speedup.
- La estructura secuencial de DeepSeek (cada módulo condiciona en el token anterior) es equivalente en espíritu a Hydra y suele aceptar más que cabezas independientes; para bytes, una cabeza "mini-decoder" recurrente muy pequeña (GRU/1 capa) sobre el estado del decoder local es barata.

### Gaps
- No hay resultados publicados de MTP a nivel de bytes por debajo de 7B.
- DeepSeek no detalla el setup exacto de medición del 1.8x TPS.

## 4. ¿Qué funciona en modelos muy pequeños (25M–2B) y qué solo compensa en grandes? + implementación en PyTorch puro / XPU

### Takeaway
En modelos diminutos el coste por paso está dominado por overhead fijo (dispatch Python/kernels, lectura de memoria), así que lo que compensa son drafts casi gratuitos verificados en un único forward: cabezas MTP/Medusa sobre el decoder de bytes, early-exit del núcleo recurrente, y n-gram/prompt lookup; un modelo draft separado y lookahead (que necesita mucho cómputo ocioso) solo compensan claramente en modelos grandes. Todo es implementable en PyTorch puro sin kernels CUDA si se usan drafts lineales.

### Cited Findings
- La pérdida de calidad de MTP como objetivo principal aparece en 0.3B–0.6B y la ganancia empieza ~1.3B. — [Gloeckle et al. 2024](https://arxiv.org/html/2404.19737v1)
- Los speedups especulativos dependen fuertemente del hardware (más en A100 que 3090), caen en FP32 y con temperatura alta; batch>1 reduce el beneficio. — [Spec-Bench](https://arxiv.org/html/2401.07851)
- Lookahead pasa de 1.8x/4x (autores, multi-GPU) a 1.11x en una RTX 3090 en medición independiente. — [Fu et al. 2024](https://arxiv.org/abs/2402.02057); [Spec-Bench](https://arxiv.org/html/2401.07851)
- PLD (sin red, pura búsqueda de n-gramas) 1.62x en 3090 con Vicuna-7B, sin entrenamiento. — [Spec-Bench](https://arxiv.org/html/2401.07851)
- Draft&Verify no necesita entrenamiento ni memoria extra (solo saltar capas). — [Zhang et al.](https://arxiv.org/abs/2309.08168)
- Verificación byte-level con estado SSM demostrada (MambaByte, 2.6x). — [Wang et al. 2024](https://arxiv.org/abs/2401.13660)
- EAGLE puede combinarse con FlashAttention/cuantización; el repo no declara dependencia de kernels custom para el algoritmo. — [GitHub SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)

### Inferences
**Ranking para Hope HN-1 (25M–200M, bytes, A770/XPU)** — todo esto es inferencia, no demostrado:
1. **Cabezas MTP de bytes en el decoder local (Medusa-1/Hydra-style, k=2–4 bytes)**, verificadas en un forward del decoder local con KV cache. Barato de entrenar (backbone congelado, minutos–horas), lossless, sin cambios de calidad. Variante Hydra/DeepSeek (cabeza secuencial diminuta) probablemente mejor aceptación.
2. **Early-exit del núcleo recurrente (LayerSkip/Draft&Verify nativo)**: draft con 1 iteración del core compartido, verificación con el halting completo; reutiliza activaciones. Añadir pérdida en iteraciones tempranas durante entrenamiento.
3. **Prompt-lookup / n-gram de bytes** (copiar continuaciones del contexto): coste cero, pure PyTorch/Python, muy útil en código/texto repetitivo; combinable con 1–2.
4. **Especulación a nivel de patch**: si el cuello de botella es el paso global por patch, predecir los bytes del patch siguiente con el decoder local (o una cabeza) y verificar varios patches a la vez con el backbone global. Requiere checkpoint/rollback del estado Mamba y de la caché por patch; más complejo (análogo a MambaByte "tokenized drafting").
5. **No recomendado** para este tamaño: draft model separado (c demasiado alto relativo), Lookahead (necesita cómputo sobrante y su ganancia independiente es baja), árboles grandes tipo EAGLE-2/3 (máscaras + estados SSM ramificados).
- Antes de nada: perfilar los ~233 bytes/s en CPU para saber qué fracción es overhead Python por byte vs paso global; si el overhead Python por paso domina, la especulación (k bytes por forward) ataca exactamente eso, y también ayuda `torch.compile`/CUDA-graph-equivalentes en XPU si están disponibles.
- Viabilidad PyTorch puro/XPU: verificación greedy = comparar argmax de k posiciones (trivial); speculative sampling lossless = rejection sampling con `torch.rand` y min(1,p/q), residuo normalizado (pocas líneas); tree attention = `scaled_dot_product_attention` con máscara booleana; rollback de KV = truncar tensores; rollback de SSM = guardar estado por posición draft (memoria k×estado, despreciable a 25M). Ninguna pieza requiere kernels CUDA.
- Calidad: con verificación lossless la salida es idéntica a la decodificación normal (salvo efectos numéricos FP16 señalados por Spec-Bench); "typical acceptance" de Medusa cambia la distribución y debe evitarse si se quiere exactitud.
- Expectativa realista (especulativa): 1.5–3x en bytes/s para texto con cabezas de 3–4 bytes si la aceptación por byte es ≥0.7; la cifra de 6.4x de Gloeckle es a 7B y con MTP pre-entrenado, no extrapolable directamente.

### Gaps
- No encontré benchmarks de ningún método especulativo en Intel Arc / PyTorch XPU, ni en modelos <300M, ni en arquitecturas byte-patch jerárquicas (BLT-like) con especulación a nivel de patch.
- No hay datos públicos sobre tasas de aceptación por byte para cabezas MTP en modelos pequeños; habría que medirlas experimentalmente en Hope.
