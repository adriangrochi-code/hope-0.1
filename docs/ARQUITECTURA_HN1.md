# HOPE HN-1 — Arquitectura técnica consolidada

**Versión:** 0.1 (borrador de diseño)
**Estado:** propuesta. Todavía no hay código ni resultados experimentales.
**Fuentes:** los 14 documentos de `docs/especificaciones/`. Las siglas están en `docs/GLOSARIO.md`.

> Este documento convierte las especificaciones conceptuales en una arquitectura que se pueda implementar.
> Cuando una decisión es una propuesta mía y no algo que diga una especificación, lo marco como **[Propuesta]**.
> Cuando una cifra es una estimación que hay que medir, lo marco como **[Estimación]**.
> Ninguna capacidad descrita aquí está demostrada. Cada una es una hipótesis de ingeniería que debe pasar las pruebas de la §12.

---

## 1. Decisión central: tres cosas distintas que los documentos mezclan

Las especificaciones nombran unos 40 "módulos". Si cada uno fuera una red neuronal, 200M de parámetros no alcanzarían y el sistema sería imposible de entrenar y depurar. Los propios documentos lo advierten (ARS §3, SCA §3, MDD §12, HCRI §7: *"no debe convertirse en un modelo independiente"*).

**[Propuesta]** Separar Hope en **cinco capas**, y asignar cada módulo a la capa donde realmente vive:

| Capa | Qué es | ¿Cuenta como parámetros? | Módulos que viven aquí |
|---|---|---|---|
| **A. Núcleo neuronal** | La red entrenable (~200M) | Sí | Language Core, núcleo recurrente de razonamiento, regiones especialistas, familias HNHS, cabezas de control (router AACA, parada del Loop Controller, incertidumbre UAC, detectores CRE, reconocedor RRE de MRC) |
| **B. Runtime cognitivo** | Código que orquesta el núcleo en cada tarea | No (solo cabezas pequeñas de la capa A) | ARS/ARC, MSRE, Loop Controller, AIS, HTO, CRE (lógica), HCRI (solvers), MRC (transformaciones), ATI |
| **C. Almacenes persistentes** | Bases de datos e índices externos | No | AHMC, MDD (n-gramas, engramas, índice asociativo), PESAM, DLM, TLM, IRM, catálogo HTE, registro de ideas de SCA |
| **D. Evolución y experimentación** | Procesos fuera de línea que modifican la capa A en copias | No | ASG, AANS (NAG/NSG/NSE/NVE/NIM), HED, SCA, RRA, MLE, TEO, TCG, TVE |
| **E. Seguridad (H-SVF)** | Guardián determinista e independiente | No | HPA, HSI, HSA, HRA, HIA, sandbox, límites de presupuesto |

Consecuencias:

- El "módulo" de una especificación es una **función**. Puede implementarse como cabeza neuronal, código, almacén o proceso, según lo que sea más fiable y barato.
- Solo la capa A consume el presupuesto de parámetros. Esto cumple ASG §3.E y Language Core §9: *"no cuentes memoria externa como parámetros"*.
- La capa E **no forma parte del modelo**. Corre como proceso separado, con permisos del sistema operativo que el modelo no puede modificar (Carta art. 6.6 y art. 13).

```mermaid
flowchart TB
  subgraph E[E · Seguridad H-SVF · proceso separado]
    HPA[Permisos HPA] --- HSI[Invariantes HSI] --- HSA[Auditoría HSA]
  end
  subgraph B[B · Runtime cognitivo]
    ARC[ARC / ARS] --> LC[Loop Controller]
    ARC --> HTO
    LC --> CRE
    CRE --> HCRI[HCRI solvers]
    ARC --> MRC
    AIS --> ARC
  end
  subgraph A[A · Núcleo neuronal ~200M]
    ENC[Codificador de bytes] --> BB[Backbone híbrido SSM+atención]
    BB --> RC[Núcleo recurrente compartido]
    RC <--> EXP[Regiones especialistas]
    RC --> DEC[Decodificador de bytes]
    CTRL[Cabezas de control: router, parada, incertidumbre, detectores]
  end
  subgraph C[C · Almacenes]
    AHMC --- MDD --- PESAM
    HTE[Catálogo HTE] --- IRM
  end
  subgraph D[D · Evolución fuera de línea]
    ASG --- AANS --- HED --- SCA --- RRA
  end
  B <--> A
  A <-->|recuperación| C
  B <--> C
  D -->|copias experimentales| A
  B -->|toda acción externa| E
  D -->|toda consolidación| E
  E -->|bloquea / autoriza| B
```

---

## 2. Principios de diseño que restringen todas las decisiones

Son los principios que se repiten en las 14 especificaciones, reducidos a reglas verificables:

1. **Techo, no meta.** El límite de parámetros autorizado por el usuario es un máximo. Crecer requiere evidencia (ASG §1, AANS §1, Evolución §2.1).
2. **Optimizar antes de crecer.** Orden obligatorio de soluciones: datos → conocimiento → herramienta → estrategia → memoria → reorganización → crecimiento (Herramientas §12, AANS §7, Evolución §2.1).
3. **Toda modificación permanente pasa por una copia experimental, una evaluación con conjuntos reservados y un linaje reversible** (ASG §7, AANS §9, HNHS §20, SCA §10).
4. **El estado epistémico es obligatorio y único.** Todo dato, idea, restricción, modelo o herramienta lleva un estado que distingue lo observado, lo inferido, lo imaginado y lo verificado (SCA §2, MDD §4.4, HCRI §9, HED §16). Ver §9.
5. **El sistema nunca es el único juez de sus propias mejoras.** Se usan evaluadores independientes, verificadores deterministas y conjuntos reservados sellados (AANS §10, ASG §8).
6. **La seguridad es independiente del optimizador.** Ninguna métrica de rendimiento puede desactivar un control (Carta art. 6 y art. 14).
7. **Interfaces estables, independientes del tamaño.** Para que HN-1 pueda migrar a HN-2 sin reconstruir las herramientas ni las memorias (Herramientas §15, HNHS §16).

---

## 3. Capa A — Núcleo neuronal

### 3.1 Vista general **[Propuesta]**

```
bytes ──► Codificador local de bytes ──► parches ──► Backbone global híbrido ──► Núcleo recurrente ──► Decodificador local ──► bytes
                                                         ▲          (N iteraciones, parada adaptativa)
                                                         │                 ▲ │
                                                  Recuperación de          │ ▼
                                                  memoria (AHMC/MDD)   Regiones especialistas (router AACA)
```

### 3.2 Representación de entrada: bytes con parches (Language Core §2)

**Decisión:** procesamiento a nivel de **bytes** (UTF-8), con un codificador local que agrupa los bytes en **parches** de longitud variable. Un backbone global trabaja sobre los parches y un decodificador local reconstruye los bytes.

- **Por qué:** maneja de forma nativa palabras desconocidas, errores ortográficos, símbolos matemáticos, código y cualquier idioma, sin vocabulario fijo. La tabla de embeddings es mínima (256 entradas) y no consume ~25M de parámetros como un vocabulario BPE de 32k.
- **Estado de la técnica:** demostrado a escala grande por *Byte Latent Transformer* (BLT, Pagnoni et al., 2024) y *MegaByte* (Yu et al., 2023). **No está demostrado a 200M.** Hay riesgo.
- **Mitigación:**
  - v0 usa parches por fronteras simples (espacios y puntuación, con un máximo de 8 bytes).
  - El parcheo por entropía de BLT queda como experimento F1.b.
  - Se entrena en paralelo una **línea base con BPE** del mismo tamaño para comparar (Language Core §13 exige comparaciones con presupuestos equivalentes).
- **Costo:** con un promedio de ~4,5 bytes por parche **[Estimación]**, 8.192 bytes de contexto equivalen a ~1.800 posiciones globales.
- **Representación morfológica, semántica y contextual** (Language Core §2): emerge en las capas del backbone. No se programan reglas lingüísticas a mano (Language Core §4).

**Generación con caché por parche (propuesta 3, implementada).**
- La parte global (backbone, núcleo recurrente y zonas) solo se recalcula cuando se completa un parche. Los bytes dentro de un parche usan el contexto del parche anterior, que no cambia.
- Da exactamente el mismo resultado que el cálculo completo (prueba automática).
- Medido en CPU con el modelo de 25M: 3,2 veces más rápido.
- **Caché de claves y valores en la parte local (implementada):** cada byte nuevo atiende a lo guardado, y las fronteras de parche se calculan de forma incremental con la misma regla.
- Medido en CPU con 25M y 96 bytes: 53 → 138 → **233 bytes/s** (sin caché → caché por parche → ambas). Mismo resultado exacto que el cálculo completo.
- Si el texto supera la ventana de contexto, se usa la caché por parche con ventana deslizante.

**Parcheo por entropía (propuesta 4, implementado; `hope/core/patching.py`).**
- Un modelo de n-gramas de bytes (una tabla de 128 KB dentro del modelo) estima la dificultad del próximo byte. El parche se cierra donde es alta.
- Es causal, y el umbral se calibra para un largo medio de parche objetivo.
- En el corpus de prueba los cortes caen al inicio de las palabras y los tramos predecibles quedan agrupados (`uaciones`, `Desigual`).
- Se activa con `--patching entropia` (se ajusta antes de entrenar).
- Pendiente: medir en la A770 si mejora el BPB por unidad de cómputo frente a `espacios`.

### 3.3 Backbone global híbrido SSM + atención

**Decisión:** bloques **Mamba-2** (modelo de espacio de estados) intercalados con bloques de **atención** en proporción ~3:1.

- **Por qué:**
  - Los SSM procesan secuencias largas a costo lineal. Cubren el "procesamiento temporal SSM/Liquid" de ASG §16.
  - La atención, en pocas capas, da **recuperación exacta** de datos dentro del contexto. Cubre la "atención precisa para datos que requieren exactitud" de Language Core §5.
  - Así no es un Transformer convencional (Language Core §15), pero usa componentes probados: Jamba (Lieber et al., 2024), Samba (Ren et al., 2024), Zamba.
- **Alternativas descartadas:**
  - Solo Transformer: costo cuadrático y contradice el pedido.
  - Solo SSM: la recuperación exacta es débil (está documentado en la literatura).
  - Liquid/CfC puro: menos maduro para entrenar lenguaje a esta escala. Queda como candidato de AANS.

### 3.4 Núcleo recurrente de razonamiento (TRM)

**Decisión:** un **bloque de pesos compartidos** que se aplica de forma iterativa sobre el estado latente, con **parada adaptativa** decidida por una cabeza de halting (estilo ACT/PonderNet).

- **Por qué:** da profundidad de cómputo sin parámetros nuevos. Es la "capacidad por parámetro" que piden Evolución §5.3 y AANS §5.3, y es el sustrato físico del Loop Controller (ARS §7).
- **Referencias:** Universal Transformers (Dehghani et al., 2018), ACT (Graves, 2016), PonderNet (Banino et al., 2021), HRM (Wang et al., 2025), TRM (*Less is More: Recursive Reasoning with Tiny Networks*, Jolicoeur-Martineau, 2025).
- **Diseño:**
  - Iteraciones `n ∈ [1, n_max]`, con `n_max` fijado por el presupuesto de AIS.
  - La cabeza de parada recibe: cambio del estado entre iteraciones, incertidumbre estimada y señales de los detectores de contradicción.
  - La recompensa de entrenamiento penaliza el cómputo **solo cuando la respuesta es correcta**. Una respuesta rápida pero incorrecta nunca recibe premio (ASG §10).
- **Separación clave** (ASG §11): este bucle es **adaptación temporal de la tarea**. Nunca modifica pesos ni estructura.
- **Riesgo comprobado: colapso a una iteración.** En la prueba de humo la parada aprendió a usar siempre una sola iteración, y las demás nunca aprendieron a aportar. Mitigación implementada: la mitad de los pasos de entrenamiento usa todas las iteraciones, y el costo de iterar sube gradualmente (`full_depth_frac`, `ponder_warmup`). El diagnóstico de uso (§6.3) vigila el peso y la ganancia de cada iteración.

#### 3.4.1 Variante «bucle estable» (informe «Razonamiento por parámetro», recomendación 2; `stable_loop` en `hope/core/hn1.py`)

Opciones nuevas de `HN1Config`. Los valores por defecto reproducen el núcleo anterior, así que los checkpoints existentes cargan igual (hay una prueba). `stable_loop(cfg)` las activa juntas:

| Opción | Por defecto | En `stable_loop` | Qué hace |
|---|---|---|---|
| `core_inject` | `"ninguna"` | `"convexa"` | La entrada `x` del núcleo vuelve a entrar en cada iteración: `h ← block(Ā·h + (1−Ā)·x)` (Parcae con Ā **fijo**, sin aprender). Con 0 < Ā < 1 el radio espectral de la parte lineal es < 1 y el estado no explota al iterar más que lo entrenado |
| `core_decay` | 0,5 | 0,5 | Ā |
| `core_step` | `"embed"` | `"ninguna"` | Sin la tabla `step_embed` de `max_iters` filas; también hay `"senoidal"` (función continua de la iteración, sin parámetros) |
| `core_depth` | `"uniforme"` | `"lognormal"` | Profundidad de entrenamiento de cola pesada: τ ~ N(log r̄ − σ²/2, σ), n = 1 + Poisson(e^τ), con media `core_mean_iters` (4) y tope `core_train_max` (12) (Huginn, Geiping et al. 2025) |
| `core_bptt` | `None` | `None` | Backprop solo por las últimas k iteraciones. Apagado por defecto: el informe señala que el truncado empeora el bucle (φ 0,46 → 0,38); queda para cuando falte memoria |
| `halting` | `"act"` | `"convergencia"` | Al inferir, cada posición se congela cuando su estado cambia menos de `convergence_tol` (1·10⁻² en la variante: con pesos bf16 un umbral de 1·10⁻³ no se alcanza nunca) |
| `max_iters` | 4 | 16 | En la variante es solo el techo de inferencia; se puede pasar `n_iters` mayor |

- **Sin parámetros nuevos:** la variante tiene `max_iters·d` parámetros **menos** (no tiene `step_embed`).
- **EMA de pesos** (`TrainConfig.ema`, `Trainer.ema_weights()`, `copy_ema_to_model()`): copia fp32 de los pesos (+4 bytes por parámetro), con decaída que arranca baja ((1+t)/(10+t)) y sube hasta el valor pedido. Se guarda en el checkpoint.
- **Generación incremental:** sigue siendo exacta (la variante está en `CONFIGS` de `tests/test_speed.py`). Con caché se calculan todas las iteraciones hasta el techo, así que generar con techo 16 cuesta más que con 4 en la parte del núcleo.
- **Resultados en CPU:** `docs/resultados/bucle_grpo.md`.

### 3.5 Regiones especialistas y router (AACA)

**Decisión:** un conjunto de **regiones especialistas** (bloques MLP/SwiGLU con identificador estable) seleccionadas por un **router aprendido** con top-k disperso (estilo mezcla de expertos).

- **Por qué:**
  - Es la forma más práctica de lograr "activación selectiva" (AANS §5.4) y la diferencia entre "parámetros activos y totales" (ASG §3.B).
  - Es la **unidad natural de crecimiento**: ASG puede agregar, dividir, fusionar o retirar especialistas sin tocar el resto (ASG §6.3, AANS §3.3).
- **Referencias:** Shazeer et al. (2017), Switch Transformer (Fedus et al., 2021).
- **Inicio:** 8 regiones genéricas. **No** se etiquetan como "matemáticas", "código", etc. La especialización tiene que emerger y medirse. ASG §2.2 prohíbe fijar porcentajes por dominio.
- **El router es AACA en su forma neuronal.** La parte de política (presupuesto, cuándo usar herramientas) vive en la capa B.

### 3.5.1 Zonas de especialistas heterogéneos (decisión del usuario; `hope/core/zones.py`)

Hope no es un único tipo de red. Cada zona combina **especialistas de tipos distintos**, y un router aprendido decide, posición por posición, cuáles trabajan (top-k). Sus salidas se combinan, y la competencia del router actúa como inhibición entre especialistas.

| tipo | red | familia HNHS |
|---|---|---|
| `mlp` | perceptrón GELU | general |
| `glu` | compuerta multiplicativa | operadores GATE/MUL |
| `ei` | poblaciones excitadoras/inhibidoras con signo fijo y normalización divisiva | EN/IN |
| `recurrente` | constante de tiempo dependiente de la entrada (inspirada en Liquid/CfC) | RN |
| `conv` | convolución causal + mezcla | DN (detectores) |
| `asociativa` | memoria de Hopfield moderna (ranuras clave→valor) | AN/MEM |
| `relacional` | atención causal local | relaciones |

- Todos los tipos se pueden **ensanchar sin cambiar la función**, salvo `relacional`, que crece agregando otro especialista.
- `AddExpert(zona, kind=...)` agrega un especialista del tipo que se elija.
- El registro guarda el tipo de cada uno, y el diagnóstico LTH mide sus neuronas y su uso.
- Así ASG/AANS pueden decidir con datos qué tipo de red le falta a cada zona.
- Una zona homogénea (`zone_kinds = ("mlp",)*8`) queda disponible como línea base para la ablación.

**Composición automática de zonas (propuesta 5, implementada; `hope/growth/zone_composer.py`).**
- Con el uso de cada especialista y su tipo, propone:
  - agregar un especialista del tipo sobrecargado (más de 1,5 veces el reparto parejo);
  - reemplazar uno casi sin uso (menos de 0,25 veces) por uno del tipo más demandado.
- Cada propuesta se evalúa con dos copias entrenadas igual, control y cambio, y se decide con la regla de ASG.
- Nada se aplica sin esa evidencia, y el guardián controla la consolidación como en cualquier cambio estructural.

### 3.5.2 Red experimental e hiperplasticidad (decisión del usuario)

**Opciones experimentales**, todas apagadas por defecto y comparables con `hope/bench/ablation.py`:
- `kv_heads`: GQA;
- bloques `swa`: ventana deslizante con *sinks*;
- `local_kind="swa"`;
- `halting="convergencia"`: profundidad muestreada al entrenar y parada por convergencia al inferir;
- `balance="sesgo"`: balanceo sin pérdida auxiliar;
- `shared_expert`;
- `patching="entropia"`.

Salen del informe `reports/Técnicas de generación rápida para Hope.md`.

**Hiperplasticidad** (`hope/plasticity/dynamic.py`):
1. Conexiones dentro de las capas que se podan y recrecen con densidad constante (RigL).
2. Conectoma entre bloques del backbone: conexiones dirigidas que nacen con compuerta 0 y se eliminan si no aportan.
3. Cambio gradual de tipo de bloque (`MorphBlock`, p. ej. SSM → atención con ventana): arranca sin cambiar la función.
4. Especialistas con arquitectura definida por un **genoma** de operadores HNHS (`hope/core/genome.py`), mutable, para explorar redes chicas y especializadas que no están en el catálogo inicial.

Todo pasa por el ciclo de evaluación con control y el guardián antes de consolidarse.

### 3.6 HNHS: familias neuronales implementadas en GPU

**Tensión que hay que resolver:** HNHS describe neuronas individuales heterogéneas (`HN_i = (T, G, X, P, S, M)`) con operadores distintos cada una. Implementado literalmente, como un grafo de objetos por neurona, eso es **órdenes de magnitud más lento** en una GPU que operaciones vectorizadas. Haría inviable entrenar 200M de parámetros.

**[Propuesta]** Implementar las familias como **poblaciones vectorizadas** (grupos de unidades que comparten una plantilla de operadores) dentro de cada bloque:

| Familia HNHS | Implementación concreta | Referencia |
|---|---|---|
| EN excitadoras / IN inhibidoras | Canales con signo fijo en sus pesos de salida (parametrización `softplus(W)·signo`), al estilo de la ley de Dale | DANN, Cornford et al., 2021 |
| Inhibición competitiva / lateral | k-WTA y normalización divisiva sobre grupos de unidades. Con temperatura, para no suprimir prematuramente alternativas cuando hay incertidumbre (HNHS §7.2) | — |
| MN moduladoras | Red pequeña que produce ganancias FiLM (γ, β) por región y **multiplicadores acotados de tasa de plasticidad**. Con límites duros (`tanh`, clamp) para que una sola señal no modifique toda la arquitectura (HNHS §8) | FiLM, Perez et al., 2018 |
| DN detectoras | Cabezas sigmoides dispersas: contradicción, anomalía, novedad. Alimentan a CRE y UAC | — |
| RN recurrentes | Estado SSM y núcleo recurrente (§3.4) | — |
| INE integradoras | Capas de fusión de las salidas de especialistas y de recuperación | — |
| MEM memoria | Ranuras de estado persistente entre iteraciones. **No** reemplazan a AHMC/PESAM | — |
| CN coordinación | El router (§3.5) | — |
| SN especializadas | Nuevas plantillas construidas con la biblioteca de operadores de HNHS §6 (SUM, MUL, MAX, GATE, …), creadas solo por AANS en experimentos | — |

- **Granularidad del registro:** el registro de topología (§6.1) rastrea **regiones → microcircuitos → poblaciones**. El detalle por neurona se maneja con **máscaras**, que es lo que necesitan LTH y la poda.
- **Conexiones como entidades** (HNHS §5): las conexiones entre regiones son objetos del registro con peso, tipo, dirección, compuerta y plasticidad. Dentro de una población son matrices densas o dispersas.
- **v0 de HN-1** (HNHS §21): solo poblaciones excitadoras, inhibidoras, integradoras, detectoras, recurrentes y moduladoras. Las asociativas, de memoria y de coordinación explícitas llegan en "HN-1 avanzado".
- **Cuestión abierta para medir:** si las restricciones de signo de Dale y la inhibición competitiva **mejoran o empeoran** el rendimiento frente a un MLP estándar del mismo tamaño. Es el primer experimento de ablación de HNHS (§12).

### 3.7 Interfaz de memoria (puente con la capa C)

- **Atención cruzada de recuperación:** el backbone consulta fragmentos que trae AHMC/MDD (estilo RETRO, Borgeaud et al., 2022, y Memorizing Transformers, Wu et al., 2022).
- **Proyecciones de claves para engramas:** codifican estados del modelo como claves del índice asociativo.
- La memoria recuperada entra **con su referencia a la fuente original** para cumplir ASG §12 y Language Core §5: *"para números, fórmulas y unidades, recuperar la fuente"*.

### 3.8 Presupuesto inicial de parámetros **[Estimación — se fija en F0 con el contador del registro]**

Dimensión global d = 1024. Dimensión local d = 512.

| Componente | Totales | Activos por paso | Entrenables en preentrenamiento | Notas |
|---|---:|---:|---:|---|
| Codificador + decodificador de bytes (2+3 capas, d=512) | ~16M | ~16M | sí | |
| Backbone global (6 bloques Mamba-2 + 2 de atención, con MLP) | ~96M | ~96M | sí | |
| Núcleo recurrente compartido (2 bloques) | ~22M | ~22M × iteraciones de **cómputo** | sí | Los parámetros no se multiplican al iterar |
| Regiones especialistas (8 × ~3,1M, top-2) | ~25M | ~6M | sí | Unidad de crecimiento de ASG |
| Interfaz de memoria (2 capas de atención cruzada + proyecciones) | ~10M | ~10M | sí | |
| Cabezas de control (router, parada, incertidumbre, detectores, RRE) | ~5M | ~5M | sí | |
| **Total asignado** | **~174M** | **~155M** | | |
| **Reserva sin asignar** | ~26M | — | — | Techo autorizado: 200M. Solo ASG la usa, con evidencia |

- **Atribución al Language Core:** codificador/decodificador más backbone dan ≈ 112M. Coincide con los ~110M de referencia de Language Core §9. Pero esos parámetros son **compartidos** con razonamiento y memoria. No se duplican redes (Language Core §9).
- **Fuera del conteo** (ASG §3.E): estados recurrentes, caché de atención, AHMC, MDD, PESAM y documentos.
- **Multimodal:** no tiene asignación en v0. Se agrega cuando haya una necesidad medida.

### 3.8.1 Baja precisión de base (decisión del usuario; `hope/train/lowprec.py`)

Para que crecer no pese tanto, Hope entrena de base en **`bf16-8bit`**:
- pesos y gradientes en bf16;
- momentos de AdamW cuantizados a 8 bits por bloques de 256 (el segundo momento se guarda como raíz);
- redondeo estocástico al escribir los pesos, para que las actualizaciones chicas no se pierdan.

| modo | bytes por parámetro | parámetros entrenables en 16 GB (reserva de 20% para activaciones) |
|---|---|---|
| fp32 | 16 | ~0,8B |
| bf16 | 12 | ~1,07B |
| **bf16-8bit (base)** | ~6 | **~2,1B** |

- Se elige con `--precision` en el preentrenamiento y en el entrenamiento continuo, y con `training_precision` en el presupuesto, que usa el guardián para estimar memoria.
- En el modelo diminuto aprende igual que fp32 (prueba automática), con unas 4 veces menos memoria de optimizador.
- **Hay que medirlo a escala real en la A770** contra fp32: es una hipótesis de ingeniería.

### 3.9 Hardware y costo de entrenamiento **[Estimación — medir en F0]**

**Hardware confirmado:** 1 × Intel Arc A770 (16 GB VRAM) y 64 GB de RAM. Corpus disponible: varios TB.

**Consecuencias:**

- **Memoria:** entrenar 200M en bf16 con AdamW ocupa ~3,2 GB, más las activaciones. **Entra en 16 GB** con gradient checkpointing y lotes moderados.
- **Software (riesgo principal):** la A770 no usa CUDA.
  - PyTorch soporta GPUs Intel mediante el backend `xpu`, con `torch.compile` y Triton para XPU.
  - Los kernels CUDA de `mamba-ssm` y de FlashAttention **no funcionan**.
  - **Decisión:** la capa SSM se implementa como una recurrencia lineal por bloques (chunked scan) en PyTorch/Triton. La atención usa `scaled_dot_product_attention` de PyTorch.
  - El rendimiento real hay que **medirlo en la primera semana** con un benchmark de matmul bf16, SDPA y un bloque completo.
- **Cómputo:** ~6 × parámetros activos × posiciones procesadas.
  - Para el modelo de 200M a nivel de bytes, el costo es del orden de **3·10²⁰ FLOPs por TB**.
  - Con el rendimiento efectivo esperable en la A770 (decenas de TFLOPS en el mejor caso; a medir), eso son **muchos meses por TB**.
  - **Usar varios TB enteros no es viable en este hardware.**
- **Estrategia de datos [Propuesta]:** usar los TB como **fuente para seleccionar**. Filtrar por calidad, deduplicar y quedarse con un subconjunto de 20–100 GB de alta calidad por corrida. Lo que no se use queda para escalar más adelante o con cómputo alquilado.
- **Bytes vs. BPE bajo cómputo limitado:**
  - El procesamiento por bytes paga las capas locales en **cada byte**.
  - Con poco cómputo, la línea base BPE será más barata por dato.
  - La comparación de F1 decide cuál se usa para la corrida de 200M.
- **Maestro Qwen:**
  - Un modelo de 7–8B cuantizado a 4 bits corre en la A770 mediante llama.cpp (backends SYCL o Vulkan) u OpenVINO/IPEX-LLM.
  - No puede compartir los 16 GB con el entrenamiento, así que se **alterna**: generar episodios en lotes y después entrenar.
- **Escalado futuro:** la corrida grande con todo el corpus puede hacerse con GPUs alquiladas **una vez validado el diseño**. El código debe ser portable entre `xpu`, `cuda` y `cpu`.
- **Expectativa honesta** (Language Core §13, Herramientas §8):
  - SmolLM2-135M se entrenó con ~2 billones de tokens en clústeres de GPU.
  - En una A770, HN-1 procesará una fracción de eso.
  - Las comparaciones deben hacerse **con cómputo y datos equivalentes**, y por tarea. Ninguna equivalencia se afirma sin medición.

### 3.9.1 Velocidad y memoria sin más parámetros (pedido del usuario; `hope/core/speculative.py`, `hope/core/fast.py`)

| mejora | qué hace | medido (CPU, 25M, sin entrenar) |
|---|---|---|
| **Parte global incremental** (`GlobalCache`) | al cerrarse un parche se procesa solo ese parche: claves/valores en atención, estado en SSM, especialistas secuenciales (recurrente, convolución, relacional, genomas) y una caché por iteración del núcleo | 2× con 300 bytes de contexto, **3,7×** con 1500 |
| **Generación especulativa** | adivina hasta 8 bytes (n-gramas del propio texto o de la memoria, o cabezas multi-byte) y los verifica en una pasada; salida **idéntica** a `generate` | 2,8–3,2 bytes por pasada; **5,1×** en total con 1500 bytes |
| **Cabezas multi-byte** (`multibyte`) | predicen los bytes t+2…t+4 como tarea auxiliar; sirven de borrador | +3·(d_local² + 2·d_local) parámetros (0,8M en 200M) |
| `compile_model` | `torch.compile` por bloque (fusiona operaciones; `reduce-overhead` graba grafos) | medir en la A770 |
| `quantize_int8` | pesos int8 con escala por fila para inferir | memoria de pesos ÷2 frente a bf16 |
| `checkpointing`, `grad_accum`, `find_batch_size` | entrenar con el lote más grande que entra (limitado por cómputo, no por memoria) | — |
| **Generación en lote** (`generate_batch`; `hope/core/batched.py`) | B continuaciones a la vez: cada lectura de los pesos sirve a B bytes. Parte local en lote con una posición por fila; parte global en lote para todas las secuencias que cierran un parche en el mismo paso, aunque tengan historiales de distinto largo; prompts repetidos se procesan una vez; salida **idéntica** a `generate` | CPU, 25M sin entrenar, 64 bytes por secuencia: 325 bytes/s con `generate` → **696** (B = 8, prompts distintos), **850** (B = 8, mejor-de-N con muestreo), **1549** (B = 8, mismo prompt voraz) |
| **Caché de conversación** (`Conversation`) | guarda claves/valores locales, `GlobalCache` y estado de parches entre turnos; cada turno procesa solo el texto nuevo y el último byte generado; da lo mismo que `generate` con todo el historial | primer byte con 2087 bytes de historial: **23 ms** frente a 199 ms (8,7×); con 570 bytes: 19 frente a 66 ms; el turno procesa 33 bytes en lugar de 2087 |

- **Exactitud:** la referencia (recalcular todo), la generación incremental y la especulativa dan **la misma salida byte por byte** en 7 configuraciones (SSM, convergencia, ventana+GQA, sesgo+compartido, visión, multi-byte, conectoma dinámico + genoma). Prueba: `tests/test_speed.py`.
- **Error encontrado en el camino:** un especialista secuencial al que el router no elegía en un parche no avanzaba su estado. Se corrigió: el estado avanza siempre.
- **Ancho de banda:** al generar, cada lectura de los pesos ahora produce varios bytes (especulativa), la parte global no relee todo el contexto y **varias muestras se generan a la vez** (consejo, STaR, GRPO, mejor-de-N): en una GPU limitada por ancho de banda, 8 muestras cuestan casi lo mismo que una.

**Generación en lote y caché de conversación** (`hope/core/batched.py`; medición: `python -m hope.bench.batch_speed`):

- **Estado de parches por secuencia:** cada secuencia lleva su máquina de fronteras (la misma regla que `patch_boundaries`), sus parches completos y su `GlobalCache`; las fronteras caen en pasos distintos en cada una.
- **Parte local en lote:** el byte nuevo de las B secuencias pasa junto por el codificador y el decodificador. Las claves/valores van en búferes con relleno y cada fila tiene su posición (RoPE por fila y máscara que tapa el relleno). Es un camino opcional de `CausalSelfAttention` que solo se activa si la caché trae `"rows"`; el comportamiento anterior no cambia.
- **Parte global en lote:** en cada paso, todas las secuencias que cerraron un parche se procesan **en una sola pasada**, aunque tengan cantidades de parches distintas. Se juntan sus estados: la atención (backbone y núcleo) y el especialista relacional se rellenan y usan el camino por filas; el SSM, las convoluciones, los especialistas recurrentes y los genomas tienen estado de forma fija y se concatenan. Después se separan. Si dos estados no tienen la misma estructura, cada secuencia usa su `GlobalCache` (nunca se pierde exactitud).
  - Elección: agrupar por paso y no rellenar «hacia adelante». La secuencia necesita la salida global de su parche recién cerrado para el byte siguiente, así que la pasada global no se puede postergar; por eso el máximo es una pasada global por paso. Con parches de ~4,5 bytes y B = 8, cierran ~2 secuencias por paso (medido: 2,45 parches por pasada con prompts distintos, 8 con el mismo prompt voraz).
- **Prefill por prompt (B = 1):** procesar un prompt ya usa muchos bytes por lectura de pesos, así que no gana con el lote. Los prompts repetidos (mejor-de-N, GRPO) se procesan **una vez** y se copia el estado.
- **Casos que usan `generate` tal cual:** texto más largo que `max_context` (ventana deslizante), codificador/decodificador con ventana (`local_kind="swa"`) o prompt vacío. La imagen, la memoria y `n_iters` son comunes al lote.
- **Exactitud:** en modo voraz, cada salida de `generate_batch` es igual byte a byte a `generate` del mismo prompt, y `Conversation.say(t1)`, `say(t2)` da lo mismo que `generate(t1 + salida1 + t2)`, incluidas las paradas por byte, por secuencia y por línea. Prueba: `tests/test_batched.py`, en 9 configuraciones (base, SSM, convergencia, ventana+GQA, sesgo+compartido, visión, multi-byte, parches cortos, relacional con ventana corta).
- **Medido en CPU** (4 núcleos, sin entrenar, 64 bytes por secuencia, sin byte de parada):

| modo | B | hn1-cpu (7,7M) bytes/s | hn1-25m bytes/s |
|---|---|---|---|
| `generate`, una por vez | — | 382 | 325 |
| `generate_batch`, prompts distintos | 1 / 4 / 8 | 501 / 753 / 893 | 302 / 570 / 696 |
| `generate_batch`, mismo prompt (voraz) | 1 / 4 / 8 | 390 / 1379 / 2667 | 367 / 825 / 1549 |
| `generate_batch`, mismo prompt, muestreo T = 1 | 1 / 4 / 8 | 436 / 848 / 1038 | 321 / 595 / 850 |

| historial | bytes procesados en el turno | primer byte con caché / sin caché | turno de 32 bytes con caché / sin caché |
|---|---|---|---|
| 570 bytes (25M) | 33 | 19 / 66 ms | 145 / 146 ms |
| 2087 bytes (25M) | 33 | 23 / 199 ms | 127 / 308 ms |

- En CPU el cálculo pesa más que la lectura de los pesos y el costo fijo de Python por paso es alto, así que la ganancia del lote es menor que la esperada en la A770, donde un paso de B = 8 lee los mismos pesos que uno de B = 1. Hay que medirlo allí: `python -m hope.bench.batch_speed --preset hn1-25m --device xpu`.
- La caché de conversación ahorra sobre todo la latencia hasta el primer byte, que deja de crecer con el historial.

### 3.9.2 Más contexto sin crecer la red (pedido del usuario)

**Cuatro piezas, ninguna agrega parámetros:**

1. **Bloques SSM:** no tienen posiciones y su estado resume todo lo anterior, así que su alcance ya es ilimitado. Con la parte global incremental (§3.9.1), cada parche nuevo cuesta lo mismo aunque el texto sea largo.
2. **Escala de RoPE** (`set_context_scale`; NTK-aware, base de YaRN):
   - estira las posiciones de la atención para leer más allá de lo entrenado sin reentrenar;
   - con escala 4, lo entrenado con 2 K se estira a ~8 K;
   - un tramo corto de entrenamiento con textos largos recupera la calidad.
3. **Memoria de segmentos** (`long_loss`, `read_segments`, `generate_long`):
   - un texto largo se lee por segmentos;
   - cada segmento leído se comprime en resúmenes (promedio cada 8 parches) que entran por la **capa de memoria que ya existe**;
   - al entrenar (`TrainConfig.segment_bytes`), Hope aprende a usar esos resúmenes;
   - es la idea de Transformer-XL, RMT y Block-Recurrent, con costo de memoria constante.
4. **Conocimiento fuera de los pesos** (§4.2.4): lo que no entra en el contexto se guarda y se recupera de la base de conocimiento.

**En el plan:** la etapa 4 entrena con secuencias de 8 KB leídas en segmentos de 2 KB, para que Hope aprenda a sostener razonamientos largos.

**Hecho:** la caché de la conversación se guarda entre turnos (`Conversation`, §3.9.1): el turno siguiente no relee el historial.

**Pendientes:**
- selección de páginas de la caché según la consulta (Quest), para contextos muy largos.

### 3.9.3 Entrenar en casa con una GTX 1060 y continuar en la A770 (pedido del usuario)

**Máquina de casa:** GTX 1060 de 6 GB (Pascal), i5-9400F y 16 GB de RAM. Se puede dejar entrenando semanas.

**Diferencias con la A770, y qué se hizo:**
- **Sin bf16 en hardware** (y fp16 a 1/64 de velocidad):
  - modo de precisión nuevo `fp32-8bit`: pesos y gradientes fp32, Adam o Muon con estado en 8 bits (~10 bytes por parámetro);
  - `hope.device.supports_bf16` decide si se usa autocast; en Pascal no se usa.
- **Sin `torch.compile`:** Triton no soporta Pascal, así que los planes lo apagan.
- **6 GB de VRAM:** el modelo de 500M con visión necesita `offload_optimizador`.
  - El estado del optimizador vive en la RAM y se trae a la GPU parámetro por parámetro en cada paso (`fetch_state`/`park_state`). Da el mismo resultado que sin descarga (está probado).
  - En la GPU quedan ~3,7 GB de pesos y gradientes más las activaciones, con checkpointing.
- **PyTorch:** las GTX 10xx necesitan la variante CUDA 12.6 (`pip install torch --index-url https://download.pytorch.org/whl/cu126`). La calibración avisa si la instalación no incluye `sm_61`.

**Calibración:** `python -m hope.bench.hardware --calibrar`.
- Mide la GPU real, la memoria, los núcleos y la RAM.
- Recomienda qué preset entra (con y sin descarga del optimizador), cuánto texto alcanza a ver y una velocidad de generación aproximada.
- Hay perfiles para la GTX 1060 (6 y 3 GB), la A770, la RTX 4090 y la RTX 5090.

**Planes:**

| Plan | Máquina | Modelo | Notas |
|---|---|---|---|
| `configs/plan_gtx1060.yaml` (principal) | GTX 1060 | hn1-200m | **Base simple de pocos días** (5 días de idioma y código, 2 de maestro local), la decisión del usuario. Qwen 1,5B cuantizado entra en la misma GPU como maestro |
| `configs/plan_gtx1060_500m_vision.yaml` | GTX 1060 | hn1-500m con visión | Optimizador en RAM. Más lento por byte. Útil si se quiere la visión desde el inicio en la 1060 |
| `configs/plan_a770_continuacion.yaml` | A770 | el mismo modelo | `desde:` el último checkpoint de la 1060. Pasa solo a bf16. Si vino con 200M, crece a 500M con tu aprobación; si ya es de 500M, sigue directo. Después vienen multimodal con contexto largo y enfriamiento |

**Ejecutar no es entrenar** (pregunta del usuario: «corro modelos de 1,5B sin problemas»):
- **Ejecutar** un modelo cuantizado a 4 bits ocupa ~0,6 bytes por parámetro: 1,5B ≈ 1 GB.
- **Entrenarlo** ocupa ~10 bytes por parámetro (pesos y gradientes fp32 + optimizador) más las activaciones: 1,5B ≈ 15 GB.
- **Cómputo:** en una 1060, un 1,5B vería ~0,2 GB de texto por día. Para una base de pocos días, el tamaño óptimo según Chinchilla ronda los 100–250M.
- **Dónde rinde lo que ya corrés:** como **maestro** en la misma máquina.

**Continuar entre máquinas:**
- El optimizador acepta el estado guardado en 32 u 8 bits, así que la A770 retoma el entrenamiento de la 1060 aunque cambie la precisión.
- Está probado: entrenar en fp32-8bit, guardar y retomar en bf16-8bit.

**Aplicación** (`python -m hope.app`; guía en [`APP.md`](APP.md)):
- Interfaz web local (solo `127.0.0.1`) con pasos guiados: calibración, datos, lenguaje, maestro, pasar a la A770, guardián y exportación cuantizada en 4/8 bits (`hope.core.fast.quantize`).
- Lanza `hope.train.run` como subproceso. «Pausar» crea `salida/DETENER` y el programa termina después del próximo checkpoint. Si el proceso se cae, lo relanza solo.
- Empaqueta `runs/hope` en un .zip para la A770. En la otra PC lo importa y elige `plan_a770_continuacion.yaml` apuntando al último checkpoint.

### 3.9.4 Hope cuantizado: entrenar para funcionar bien en pocos bits (pedido del usuario; `hope/train/qat.py`)

- **QAT:** en la etapa final (enfriamiento WSD), cada capa lineal usa sus pesos redondeados a pocos bits por grupos de 64, y el gradiente pasa como si no hubiera redondeo (estimador straight-through). El modelo aprende pesos a los que el redondeo no afecta. Es la técnica de las versiones QAT de Gemma 3 y Llama 3.2.
- **Matryoshka** (`qat_bits: [4, 8]`): cada paso elige 4 u 8 bits. El mismo modelo queda bien en las dos precisiones (MatQuant, Nair et al. 2025).
- **Exportación** (`hope.core.fast.quantize(model, bits=4|8)`):
  - pesos por grupos, con 4 bits empaquetados (dos por byte, ~0,56 bytes por parámetro con la escala);
  - coincide exactamente con la cuantización usada al entrenar;
  - un modelo de 500M queda en ~0,3 GB de pesos en 4 bits.
- No agrega parámetros ni cambia los nombres de los pesos: los checkpoints siguen siendo compatibles.
- **Prueba de efectividad** (`tests/test_qat.py`): a partir del mismo modelo, seguir entrenando con QAT deja menos pérdida al exportar a 4 bits que seguir entrenando normal.
- **En los planes:** el enfriamiento de `plan_a770_continuacion.yaml` y de `plan_6m.yaml` usa `qat_bits: [4, 8]`.
- **Pendiente:** un kernel que multiplique directamente en 4 bits (IPEX o Triton) para que, además de ocupar menos, sea más rápido. Hoy se descuantiza antes de multiplicar.

### 3.10 Zona visual (decisión del usuario; `hope/core/vision.py`)

Hope sigue siendo un modelo de bytes; la visión es **una zona más** de la red heterogénea:

```
imagen ─► parches de píxeles (conv 16×16) ─► codificador bidireccional (ViT, posiciones fila/columna)
      ─► compresión 2×2 (pixel shuffle) ─► 49 fichas por imagen de 224 px
      ─► atención cruzada en los bloques de atención del backbone global
```

- **Por qué atención cruzada y no fichas en la secuencia:** no toca el parcheo ni la causalidad, no alarga la secuencia global y se puede agregar a un Hope ya entrenado.
- **Preserva la función:** la proyección de salida de cada atención cruzada empieza en **cero**, así que agregar la zona no cambia ni un byte del texto.
  - Primero se probó una compuerta tanh en cero (Flamingo), pero con la compuerta en 0 el codificador visual no recibe gradiente. En 600 pasos desde cero la compuerta quedó en 0,002 y el modelo ignoraba la imagen.
  - Con la salida en cero (idea de ControlNet), la misma prueba da una pérdida de 0,35 con la imagen correcta contra 0,62 con imágenes mezcladas: **el modelo usa la imagen**. Esto quedó como prueba en `tests/test_vision.py`.
- **Desde cero con descripciones:** CapPa (Tschannen et al. 2023) muestra que un codificador entrenado solo para describir imágenes rinde como uno contrastivo. Las descripciones las da Qwen (§10.1.4) y las imágenes procedurales.
- **Costo:** con el preset de 500M, una imagen cuesta ≈ 1·10¹¹ FLOPs de entrenamiento, lo mismo que ≈ 240 bytes de texto. Las fichas se calculan una vez por imagen y se reutilizan en toda la generación.
- **Crecimiento** (`hope/growth/hn1_ops.py`): `AddVisualZone`, `AppendBackboneBlock` (bloque nuevo = identidad) y `plan_growth_to(modelo, "hn1-500m")`, que arma toda la secuencia desde `hn1-200m` con operaciones que preservan la función. Cada paso pasa por el presupuesto y el guardián. Los especialistas nuevos quedan enmascarados hasta habilitarlos (`unmask_expert`).
- **Pendiente:** mosaicos de alta resolución para documentos (las posiciones ya se interpolan); medir en la A770.

#### Preset `hn1-500m` (464M)

| parte | hn1-200m | hn1-500m |
|---|---|---|
| backbone | 8 bloques (109M) | 12 bloques (164M) |
| zonas | 8 especialistas de 2048 (43M) | 16 especialistas de 4096 (128M), top-2 |
| visión | — | ViT 768×12 + atención cruzada (101M) |
| total | 181M | **464M** (deja ~36M de margen bajo un techo de 500M) |
| cómputo de texto por byte | 1× | **1,36×** |

Los parámetros casi se triplican pero el cómputo de texto sube solo un 36%: las zonas son dispersas (cada parche usa 2 de 16 especialistas) y la visión se paga solo con imágenes.

---

## 4. Capa B — Runtime cognitivo

### 4.1 Ciclo de una tarea

```
1. Entrada → RRE (MRC) clasifica representaciones presentes → UAC estima incertidumbre/carencias
2. ARC (ARS) clasifica el problema y elige estrategia + presupuesto con AIS
3. HCRI extrae restricciones del problema ANTES de resolver (HCRI §3)
4. RMI recupera de AHMC/MDD/PESAM: problemas y estrategias similares
5. Loop Controller ejecuta iteraciones del núcleo / herramientas / transformaciones MRC
     en cada iteración: CRE + HCRI.CCE revisan; se miden progreso, novedad, costo
6. Parada: criterio suficiente | verificación OK | sin progreso | presupuesto agotado |
           falta información externa | requiere humano
7. Toda acción externa pasa por el guardián (capa E)
8. Salida con: respuesta, estado epistémico, supuestos, confianza, fuentes
9. PESAM registra la experiencia (estrategia, costo, resultado, errores)
```

### 4.2 Componentes

| Componente | Implementación | Notas |
|---|---|---|
| **ARC / MSRE** (ARS §3) | Política en código. En v0 son reglas y heurísticas; luego es una política aprendida con los registros de PESAM | MRE (meta-razonamiento) actualiza la política fuera de línea |
| **Loop Controller** | Código y la cabeza de parada del núcleo | Detecta repetición (hash de estados y acciones), ausencia de progreso, rendimiento marginal decreciente y presupuesto. Es el mismo controlador para ARS, SCA §9 y la recuperación de MDD |
| **AIS** | Asignador de presupuesto: tiempo, iteraciones, memoria y llamadas a herramientas | Lo consumen ARS, SCA, HED y HCRI |
| **CRE** (razonamiento crítico) | Detectores neuronales + verificaciones simbólicas + el mismo núcleo con un "objetivo adversarial" (ARS §6) | Sin un modelo crítico separado |
| **HCRI** | Solvers deterministas: SymPy, z3 y verificación dimensional con `pint` | La red **no** resuelve lo que un solver verifica mejor (HCRI §2.3) |
| **MRC** | Representaciones tipadas compartidas (§4.3) más traductores (neuronales o herramientas) | RCE verifica la compatibilidad entre representaciones |
| **HTO** | Grafo de subtareas (DAG) con dependencias, estados y método de verificación (ARS §8) | |
| **ATI** | Cliente del maestro Qwen. Arma consultas estructuradas (Herramientas §4, etapa 4) | Solo durante entrenamiento y desarrollo. El uso normal no depende de Qwen |

### 4.2.1 HCRI, CRE y Loop Controller (implementado, v0)

- **HCRI** (`hope/constraints/`): reconoce magnitudes con unidad y la incógnita, y elige los invariantes cuyo dominio aplica (ley de Ohm, potencias, energía, rangos físicos). **Antes de resolver** detecta:
  - datos contradictorios: por ejemplo, 100 Ω con 5 V y 2 A, o dos leyes que dan valores distintos;
  - datos faltantes, diciendo qué dato falta;
  - valores fuera de rango.

  Con eso deriva el valor esperado con SymPy en sandbox, y las restricciones aplicables se le dan a Hope como guía.
- **Restricciones nuevas** (`dynamic.py`): empiezan como HIPÓTESIS. Pasan a RESPALDADO con al menos 5 casos sin contraejemplos y a VALIDADO_CONDICIONAL solo con un verificador independiente. Un contraejemplo las refuta.
- **CRE** (`critic.py`): revisa el razonamiento paso a paso:
  - aritmética exacta y unidades;
  - que el resultado tenga la dimensión de la incógnita;
  - que el resultado cumpla los invariantes;
  - pasos repetidos y conclusiones sin pasos.

  Si el enunciado es contradictorio o incompleto, la respuesta correcta es **decirlo**, no dar un número. Cada observación indica si fue comprobada con herramienta o es una sospecha.
- **Loop Controller** (`hope/runtime/loop.py`): intento → crítica → reintento con la crítica. Se detiene si queda verificado, si Hope repite la misma respuesta (sin progreso) o si se agota el presupuesto. Nunca modifica pesos ni estructura.
- **En el ciclo maestro–alumno:** Hope responde con las restricciones como guía y con revisión. Cuando le pregunta a Qwen, le pasa lo que detectó su propia revisión.

### 4.2.2 Consejo interno: agentes con objetivos distintos y coordinador–verificador (decisión del usuario; `hope/council/`)

Hope resuelve con **cuatro agentes internos** de personalidades distintas. **Todo corre local en una sola GPU** (decisión del usuario): hay un único modelo en VRAM y cada agente es un adaptador LoRA liviano sobre ese modelo compartido.

| agente | objetivo | cómo trabaja |
|---|---|---|
| innovador | caminos nuevos | temperatura alta, sin memoria, busca otra representación |
| conservador | no equivocarse | memoria, restricciones, hasta 3 revisiones de CRE |
| optimizado | menos pasos | herramientas, respuestas cortas |
| veloz | responder ya | una pasada, sin revisión; usa la calculadora y otras herramientas rápidas cuando son más rápidas y exactas que calcular |

El **coordinador–verificador** es la «isla» donde se reúnen las propuestas:
1. Elige a quién consultar: a todos, o a los de mejor historial en el tema, con exploración UCB1.
2. Los agentes resuelven por turnos sobre el mismo modelo (en una sola GPU no hay ganancia en paralelizar; `parallel=False` por defecto).
3. Verifica cada propuesta de forma independiente con CRE/HCRI y, si se conoce, con la respuesta esperada.
4. Puntúa: verificación > consenso > velocidad.
5. Elige la mejor y aprende qué agente rinde en cada tema.

Ningún agente se aprueba a sí mismo.

**Una sola GPU (`build_local_council`):**
- `attach_named_adapters` engancha en cada capa lineal de regiones y backbone un `MultiLoRALinear` con un adaptador por agente (rango 4: unos pocos MB cada uno, frente a cientos de MB del modelo).
- Antes de cada turno, `use_adapter(modelo, agente)` activa el adaptador del agente; al terminar se vuelve al modelo base. Un `Lock` serializa el uso del modelo.
- Los adaptadores empiezan en cero (al inicio los cuatro agentes son el modelo base con distinta configuración) y se especializan entrenando cada uno con sus propios datos.
- `core_iterations` por agente se pasa como `n_iters` al núcleo recurrente: el veloz usa 1 iteración; el conservador, las del preset.

**Servidores (opcional):** `hope/council/server.py` expone un agente por HTTP (127.0.0.1 por defecto) y `RemoteAgent` lo usa igual que a uno local. Solo para pruebas en CPU o si algún día hay más de una máquina.

**Pendiente:**
- integrar el consejo en el ciclo maestro–alumno.

### 4.2.3 Velocidad objetivo: el tiempo sobrante se usa para mejorar la calidad (pedido del usuario; `hope/runtime/budget.py`)

El usuario fija una velocidad equivalente, por ejemplo **200 tokens/s** (1 token ≈ 4 bytes). Eso da un presupuesto de tiempo: largo de la respuesta ÷ velocidad. Por ejemplo, 400 bytes = 100 tokens = 0,5 s.

Si Hope genera más rápido, **no responde antes**: usa lo que sobra para pensar más.

1. Borrador del agente veloz.
2. Mientras haya tiempo, sube la calidad en este orden:
   - agente optimizado (herramientas);
   - agente conservador (memoria, 3 revisiones del crítico);
   - agente innovador (otro enfoque);
   - rondas cada vez más profundas: más iteraciones del núcleo, más revisiones, otra temperatura.
3. Cada candidato se verifica (CRE/HCRI) y se compara con los demás. Gana verificación > consenso.
4. **Nunca llega tarde:** antes de cada ronda se estima su duración y no se empieza si no entra.
5. **Corte anticipado** si la mejor respuesta está verificada y otra independiente coincide. Con `consistent=True` usa todo el presupuesto igual.

El informe dice qué agente ganó, cuántas rondas hubo, la velocidad efectiva percibida y por qué se detuvo. En la prueba, un borrador rápido con un error de cálculo se corrige dentro de los 0,5 s; con un presupuesto mínimo, se entrega el borrador a tiempo.

### 4.2.4 Máxima capacidad por parámetro (decisión del usuario)

El objetivo no es crecer sino **razonar mejor con los mismos parámetros**, para que cuando la red crezca ese «saber pensar» se amplifique. Las palancas, en orden de impacto esperado:

1. **Refuerzo con recompensas verificables** (`hope/train/rlvr.py`, estilo GRPO):
   - Hope genera grupos de soluciones a problemas con respuesta exacta.
   - Recompensa: 1 si acierta y el razonamiento pasa la crítica; 0,2 si acierta por casualidad con pasos rotos; 0 si se equivoca.
   - Se refuerza lo mejor que el promedio del grupo, con una penalización KL que no deja alejarse del modelo de referencia.
   - No necesita un modelo crítico aparte, lo que ahorra memoria.
   - En modelos de 1,5B fue lo que más rindió en razonamiento (DeepSeek-R1-Distill, DeepScaleR). A 135M, un GRPO directo sobre el modelo base empeoró: va **después del SFT** y solo sobre problemas al borde de lo que el modelo resuelve.
   - **GRPO corregido** (informe «Razonamiento por parámetro», recomendación 4):
     - Ventaja centrada `r − media`, **sin dividir por el desvío**, y pérdida `Σ log p / max_new` (constante), **sin normalizar por el largo** de cada muestra (Dr. GRPO). `loss_norm="largo"` y `adv_norm="desvio"` reproducen el GRPO original para comparar.
     - KL k3 **por byte**, con `kl_weight` = 1·10⁻³ (antes 0,02); se reporta como métrica.
     - Grupos sin varianza: se descartan y el lote se **rellena** con otros problemas (DAPO). `ProblemPool` retira los que salen siempre bien, posterga los que salen siempre mal y lista la franja borde (`edge(0,1, 0,8)`).
     - Currículo de largo: `max_new` empieza en 512 bytes (antes 200) y sube a 1.024 y 2.048 cuando ≥ 20 % de las respuestas se cortan, o cada `curriculum_every` pasos.
     - El grupo se genera con un `sampler` (interfaz `Sampler`). `default_sampler` usa `hope.core.batched.generate_batch` (rama `ccr-976b883c-06ef2m-lote-cache`) si está disponible, y si no genera en serie. Las log-probabilidades del grupo se calculan en una sola pasada (relleno a la derecha; el modelo es causal).
     - La recompensa con costo (`grpo_step_tools` con `efficient_reward`) queda como **fase final opcional y corta**, con la misma pérdida; con `w=None` es solo calidad.
2. **Explicaciones de Qwen** (destilación de razonamiento) y **datos de alta calidad** (§10.1.3.1).
3. **Conocimiento fuera de los pesos** (`hope/memory/knowledge.py`): los hechos y las soluciones van a una base consultable, y los parámetros quedan para razonar (§5).
4. **Profundidad sin parámetros:** el núcleo recurrente itera más cuando hace falta, y la velocidad objetivo (§4.2.3) convierte el tiempo sobrante en revisiones.
5. **Rapidez además de calidad** (pedido del usuario; `rollout_with_tools`, `efficient_reward`, `grpo_step_tools`):
   - La recompensa resta el costo: bytes generados por Hope (los resultados de herramientas no cuentan) y tiempo total.
   - El castigo tiene un tope de 0,5, así que una respuesta correcta y lenta siempre le gana a una rápida e incorrecta.
   - Entre dos correctas gana la más barata. Por ejemplo, `⟦calc: 6 * 7⟧ → 42` le gana a multiplicar paso a paso.
   - Así Hope aprende cuándo conviene delegar en una herramienta y cuándo responder directo, según lo que de verdad cuesta en tiempo cada herramienta.
   - Medición y datos para esto: banco sellado de uso de herramientas, currículo con llamadas ejecutadas y `hope/train/tool_rl.py` (docs/HERRAMIENTAS_HN1.md §8).
6. **Entrenar más tiempo un modelo chico:** con 200–500M, 6 meses dan cientos de tokens por parámetro, muy por encima del óptimo de Chinchilla. Los modelos chicos siguen mejorando con más datos buenos.

#### Base de conocimiento de trabajo (`hope/memory/knowledge.py`)

- **Qué guarda:**
  - **soluciones** con su razonamiento: RESPALDADO si pasaron la crítica, INFERIDO si no;
  - **lecciones**: los errores que la crítica encontró en respuestas descartadas.
- **Cómo la usa:**
  - antes de responder, recupera las soluciones y lecciones más parecidas; lo refutado nunca entra;
  - al reanalizar dentro de una misma respuesta, cada ronda nueva recibe los errores de los intentos anteriores.
- **Optimización:**
  - sin duplicados: lo repetido refuerza el recuerdo existente;
  - una solución verificada que contradice a otra la deja REFUTADA, con la relación registrada;
  - la utilidad se mide: lo usado en respuestas verificadas suma usos y lo que no ayudó pierde confianza;
  - `optimize()` consolida (lo usado y respaldado pasa a semántico; lo viejo, poco usado y poco importante se archiva) y poda las estadísticas.
- **Prueba:** con un presupuesto mínimo, la primera vez Hope se equivoca y guarda la lección; la segunda, con el mismo presupuesto, acierta gracias a ella.

#### Cuaderno de notas con repaso espaciado (pedido del usuario; `Notebook` en `hope/memory/knowledge.py`)

Hope estudia como una persona: anota, consulta cuando no se acuerda, repasa y, con la repetición, lo automatiza.

1. **Anotar:**
   - durante el ciclo con Qwen, Hope anota el procedimiento de cada tema y los errores frecuentes que explicó el maestro;
   - también puede anotar por su cuenta con `⟦anotar: concepto | texto⟧`;
   - no duplica, y cada versión nueva deja la anterior en el historial.
2. **Consultar:** `⟦nota: concepto⟧` busca por concepto exacto o parecido. Cada consulta cuenta.
3. **Repasar:**
   - las notas vencidas, primero las más consultadas, se convierten en preguntas de repaso («¿Qué es X?» → definición);
   - esas preguntas entran al entrenamiento, y así el conocimiento pasa de la base a los pesos.
4. **Comprobar sin mirar:**
   - si Hope acierta sin consultar, el intervalo de repaso se duplica (1, 2, 4, 8… días); si falla, vuelve a 1 día;
   - tras varios aciertos seguidos, la nota queda **automatizada** (VALIDADO_CONDICIONAL): deja de repasarse, pero sigue guardada por si hace falta.

El efecto buscado es el de una persona que estudia: lo que usa seguido se vuelve mecánico y no le cuesta tiempo; lo que usa poco lo busca cuando lo necesita.

#### Grafo de conocimiento y correlaciones (pedido del usuario; `hope/memory/graph.py`)

Las relaciones entre datos se calculan **al guardar** y se refuerzan **al usar**. Así, encontrar lo relacionado es recorrer enlaces, no buscar de cero.

- **Enlaces automáticos:**
  - SIMILARIDAD con los recuerdos más parecidos;
  - DEPENDENCIA cuando un texto menciona el concepto de otra nota. Por ejemplo, «potencia» menciona la «ley de Ohm», así que traer la potencia trae también la definición que necesita.
- **Aprendizaje hebbiano:** lo que se usa junto en una respuesta verificada refuerza su enlace (COMPLEMENTARIEDAD). Lo que no se refuerza se debilita (`decay`) y se poda.
- **Propagación de activación** (`expand`):
  - desde lo encontrado se recorren 1–2 saltos multiplicando las confianzas;
  - el contexto de cada respuesta incluye una sección «Relacionado»;
  - lo refutado y las contradicciones no se propagan.
- **Patrones** (`patterns`): conceptos centrales, grupos de conocimiento (componentes por enlaces fuertes) y pares que siempre se usan juntos.
- **Datos numéricos:** `⟦correlacion: <csv>⟧` calcula Pearson (lineal) y Spearman (monótona) entre columnas, ordenadas por intensidad. Siempre aclara que correlación no implica causalidad y cuántas filas hay.

### 4.2.5 Harness de trabajo F9 (pedido del usuario; `hope/harness/`) — en progreso

Interfaz de consola tipo Claude Code. La persona escribe un pedido y Hope trabaja sobre **una carpeta local fija** (no un repositorio remoto):

```bash
python -m hope.harness --carpeta ~/proyecto --modelo runs/hn1_200m \
    [--maestro configs/teacher.yaml] [--tokens-por-s 200] [--pruebas "python -m pytest -q"]
```

**Herramientas de carpeta** (`hope/harness/tools.py`, `FolderTools`). Todas pasan por `Guardian.decide`, y el directorio de trabajo del guardián es la carpeta elegida:

| Herramienta | Acción del guardián | Dentro de la carpeta | Fuera | Rutas protegidas |
|---|---|---|---|---|
| leer, listar, buscar (ripgrep si está; si no, `re`) | `file.read` | permitido | aprobación humana | — |
| editar: `propose_edit` → diff, `apply_edit` → escribe | `file.write` | permitido, después de que la persona acepta el diff | aprobación humana | **denegado** (configuración, guardián, `.git` de la carpeta) |
| ejecutar comandos y pruebas | `tool.execute` con `sandboxed=True` | solo en el sandbox (`hope/tools/sandbox.py`), sobre una **copia** de la carpeta | — | — |

- Las rutas se resuelven con enlaces simbólicos incluidos: un enlace que apunta afuera cuenta como afuera.
- Una edición no se aplica si el archivo cambió desde que se propuso.
- Los comandos nunca modifican los archivos originales. Las ediciones candidatas se prueban como capa encima de la copia (`overlay`).
- Una aprobación humana (`/aprobar <id>` en la consola, o `python -m guardian.cli approve`) vale para **exactamente esa acción y una sola vez**.
- Hope también puede usar estas herramientas mientras razona: `⟦leer: ruta⟧`, `⟦listar: ruta⟧`, `⟦buscar: patrón | ruta⟧`, `⟦ejecutar: comando⟧` (`HarnessToolRunner`, conectado al consejo con `build_local_council(..., runner_factory=...)`).

**Ciclo de una tarea** (`hope/harness/session.py`, `HarnessSession.task`):

1. **Contexto:** listado de la carpeta, los archivos que el pedido menciona (leídos a través del guardián) y la nota más parecida del cuaderno. Lo que está fuera de la carpeta no se lee: queda como solicitud pendiente.
2. **Hope y Qwen en paralelo:**
   - Hope responde con `BudgetedResponder` a la velocidad objetivo (§4.2.3), con la base de conocimiento y el consejo (§4.2.2, §4.2.4).
   - Qwen responde por el cliente OpenAI-compatible (§10.1.2), también con permiso del guardián.
   - El contexto de los archivos va aparte del enunciado: la crítica analiza solo el pedido, así los números de un archivo no se confunden con los datos del problema.
3. **Comparación con la misma vara para los dos:**
   - crítica CRE/HCRI de los pasos;
   - consenso (¿coinciden el resultado y las ediciones?);
   - pruebas en el sandbox con las ediciones de cada uno, si hay comando de pruebas;
   - respuesta esperada, si se conoce;
   - si Qwen respondió con un episodio JSON, su verificación paso a paso con herramientas (`verify_episode`).
   - un episodio con pasos coherentes pero sobre **otro** problema (no contiene los números del pedido) no cuenta como verificado: los pasos coherentes no prueban que responda a este pedido.
4. **Elección:** **gana la respuesta verificada**. Si las dos están verificadas y coinciden, gana Hope. Si ninguna lo está, se muestra la de mayor puntaje marcada «sin verificar».
5. **Registro en PESAM** (`harness.sesion`, `harness.tarea`, `harness.accion`, `harness.edicion`, `harness.aprobacion`, `harness.episodio`): pedido, archivos de contexto, ambas respuestas con veredicto, problemas, pruebas, elección, motivos y cada acción sobre la carpeta (permitida, pendiente o denegada). La auditoría encadenada del guardián registra además cada decisión.
6. **Aprendizaje:**
   - Si las respuestas **difieren** y la elegida **pasó la verificación**, la diferencia se guarda como episodio de entrenamiento en `episodios.jsonl`: respuesta elegida, respuesta descartada con sus errores, verificación y texto de entrenamiento.
   - Sin verificación no se guarda nada para entrenar (el maestro no es la autoridad, §10.1).
   - Si gana Qwen, su solución entra a la base de conocimiento como RESPALDADO y los errores de Hope como lección.
7. **Ediciones:** las de la respuesta elegida quedan **propuestas** con su diff (`/cambios`). Se escriben solo con `/aplicar <n>`, otra vez a través del guardián.

**Comandos de la consola:** `/ayuda`, `/notas [concepto]`, `/aprobaciones`, `/aprobar <id>`, `/denegar <id>`, `/cambios`, `/aplicar <n>`, `/descartar <n>`, `/pruebas [comando]`, `/salir`. El registro (PESAM, base de conocimiento y episodios) va en `--registro` (por defecto `runs/harness/`).

**Límites honestos:**
- El sandbox limita tiempo, memoria y procesos, pero no bloquea la red ni es un contenedor (§8.2). Para usar el harness con código no confiable conviene correrlo dentro de un contenedor o con un usuario sin privilegios.
- En F0 el guardián es una biblioteca en el mismo proceso (§8.1). La consola cambia su directorio de trabajo a la carpeta elegida porque esa elección la hace la persona al lanzar el harness, no el modelo.
- Hope y Qwen corren en paralelo: Qwen en un hilo aparte (es una llamada HTTP) y Hope en el hilo principal, porque la base de conocimiento usa SQLite.
- Mientras no exista el modelo de 200M entrenado, `--preset hn1-tiny` sirve solo para probar el circuito: las respuestas de Hope no tienen sentido todavía.

### 4.3 Representación compartida (MRC §4) **[Propuesta]**

Un **sobre común** para cualquier representación, para que los módulos se comuniquen sin duplicar estructuras:

```yaml
Representacion:
  id: rep_…                     # estable
  tipo: linguistica | matematica | logica | estructural | causal | temporal |
        espacial | procedimental | experimental | neuronal
  contenido: …                  # texto, expresión SymPy, grafo, vector, programa…
  variables: [{nombre, unidad, dominio}]
  supuestos: […]
  condiciones_validez: […]
  procedencia: {fuente, offset, documento_id}
  derivada_de: [rep_…]          # linaje de traducciones
  perdida_informacion: "…"      # obligatorio si la traducción no es reversible (MRC §3.3)
  estado_epistemico: …          # §9
  confianza: 0..1
```

---

## 5. Capa C — Memoria y almacenes

**Principio:** la memoria persistente **no son parámetros** y **no es una ventana de atención** (ASG §12). El modelo recupera fragmentos y nunca carga todo.

| Nivel (AHMC) | Contenido | Implementación v0 |
|---|---|---|
| Inmediata | Datos de la operación actual | Contexto del modelo y caché |
| Trabajo | Objetivos, variables y relaciones de la tarea | Estructura en el runtime y ranuras MEM |
| Episódica | Experiencias, documentos y proyectos | PESAM + engramas episódicos |
| Semántica | Conceptos y relaciones consolidadas | Engramas semánticos + índice asociativo |
| Procedimental | Métodos y estrategias | Engramas procedimentales + TLM (herramientas) |
| Archivo | Documentos y datos originales | Almacén de archivos con hash y offsets |

**Implementación v0 [Propuesta]:**
- SQLite para registros estructurados: engramas, relaciones, PESAM e ideas.
- Índice vectorial (FAISS) para la recuperación por similitud.
- Índice de n-gramas por tipo de unidad: bytes, palabras, operaciones o código (MDD §3.2).
- Todo con `id`, procedencia, estado epistémico y versión.

**Del documento MDD:**
- **N-gramas:** se almacenan de forma **selectiva**, por novedad, utilidad, frecuencia y costo (MDD §3.3), con un límite de tamaño y poda.
- **Engramas:** el esquema completo de MDD §4.2.
- **Índice asociativo compartido:** relaciones tipadas (similaridad, equivalencia, causalidad hipotética o verificada, contradicción, …). La causalidad **siempre** se marca como hipotética o verificada (MDD §5).
- **Continuidad persistente** (MDD §11): un registro de estado del proyecto con objetivos, decisiones, hipótesis pendientes y próximos pasos. Se valida antes de retomar.

**Implementación v0** (`hope/memory/`):
- `store.py`: archivo de originales con recuperación exacta por posición, engramas con el esquema de MDD §4.2, hechos estructurados que **no se sobrescriben** ante contradicciones (se guardan ambos y se relacionan), relaciones tipadas, índice de trigramas, consolidación que archiva sin borrar.
- `retrieval.py`: recuperación híbrida (vectores + n-gramas + asociaciones + importancia + estado epistémico). Marca las contradicciones y declara «no encontrado» cuando nada supera el umbral.
- `working.py`: memoria de trabajo acotada y estado de proyecto con validación de archivos.
- `embed.py`: codificador sin entrenamiento (n-gramas de caracteres hasheados). Se reemplazará por los estados del núcleo una vez entrenado.
- Puente con el núcleo: `HN1Model.encode_memory` codifica los fragmentos recuperados con el propio codificador de bytes y los entrega a la atención de memoria (§3.7).
- Comprobado en el banco sintético: la tarea de «memoria» pasa de 0,016 a 1,000 consultando el almacén.

**PESAM:** un registro de experiencias de solo agregado (append-only). Es la fuente común de datos para MRE, MLE, RRA, ASG §13 y TLM.

---

## 6. Capa D — Evolución estructural (ASG + AANS + HNHS)

### 6.1 Infraestructura obligatoria (ASG §17, fase 0)

1. **Registro de topología:** cada región, población, conexión y operador tiene un **ID estable**, su tipo, sus dimensiones y su conteo de parámetros (totales, activos esperados y entrenables). Se serializa con el checkpoint. Es la fuente de verdad del presupuesto.
2. **Presupuesto de crecimiento:**
   ```yaml
   GrowthBudget:
     max_parametros_totales: 200_000_000
     max_parametros_activos: null        # opcional
     max_regiones: …
     max_conexiones: …
     presupuesto_entrenamiento: {gpu_horas: …, tokens: …}
     presupuesto_memoria: {vram_gb: …, ram_gb: …, disco_gb: …}
     limite_tiempo_experimento: …
     modo: EXPANSION_DISABLED | EXPERIMENT_ONLY | USER_APPROVAL_REQUIRED |
           AUTO_CONSOLIDATION_WITHIN_LIMIT | PAUSED
   ```
   Lo **hace cumplir el guardián (capa E)**, no el modelo.
3. **Linaje de versiones:** un grafo de checkpoints con una rama `estable` y ramas `experimental/*`. Cada nodo guarda el diff estructural, el diagnóstico que lo motivó, los datos y métricas, y la decisión tomada. Rollback en una sola operación.

### 6.2 Operaciones estructurales que preservan la función **[Propuesta]**

Todas las operaciones de crecimiento se diseñan para que **en el instante de aplicarlas el modelo calcule exactamente lo mismo**. Así no se destruye lo aprendido (ASG §2.3):

| Operación | Técnica |
|---|---|
| Agregar especialista | Router con logit inicial muy bajo y proyección de salida inicializada en cero |
| Ensanchar una región | Unidades nuevas con entrada aleatoria y **salida en cero** (`zero_out`). Net2WiderNet (Chen et al., 2015) preserva la función pero **no agrega capacidad efectiva**: las copias reciben gradientes idénticos y nunca se diferencian. Comprobado en el banco sintético (`docs/resultados/banco_sintetico_v0.md`) |
| Agregar conexión entre regiones | Compuerta escalar inicializada en 0 |
| Agregar una iteración o profundidad | Bloque residual con salida en cero (Net2DeeperNet) |
| Dividir un especialista | Duplicar y perturbar levemente; el router aprende a separar |
| Fusionar especialistas | Promedio ponderado o destilación hacia uno, validado con NVE |
| Retirar | **Primero enmascarar** y evaluar; eliminar solo tras consolidar. Protege los circuitos de tareas raras (ASG §5 y §9) |

- **Estados del optimizador:** al agregar parámetros se crean sus estados de Adam en cero. Al eliminar, se descartan. El mapeo usa los IDs del registro (ASG §19).
- **Prueba automática obligatoria:** después de cada operación, la diferencia de salidas sobre un lote fijo tiene que ser menor que ε.

### 6.3 Ciclo ASG (ASG §4 y §7, unificado con AANS §7 y HNHS §12)

```
Diagnóstico (§7.1) → Hipótesis → Rama experimental → Operación estructural (§6.2)
→ Entrenamiento con replay + protección de lo consolidado → Evaluación contra:
     (a) versión estable, (b) control sin modificación entrenado igual tiempo,
     (c) conjuntos reservados, (d) regresión en tareas anteriores
→ Decisión: consolidar | ajustar | revertir | mantener como experimento
→ Registro en PESAM (alimenta el aprendizaje del propio crecimiento, ASG §13)
```

- **El control (b) es obligatorio.** Sin él no se puede distinguir "mejoró por la estructura" de "mejoró por entrenar más".
- **Protección contra el olvido:** replay selectivo, más regularización tipo EWC (Kirkpatrick et al., 2017) sobre los parámetros de alta importancia, más pruebas de retención.
- **Uso de la capacidad (decisión del usuario):** LTH se usa para saber qué partes están menos usadas y mejorar su uso, tanto en neuronas como en los bucles internos. Implementado en `hope/growth/lth.py`:
  - importancia de Taylor por neurona y **por familia de tareas**, para proteger circuitos de tareas raras;
  - uso de cada especialista;
  - peso y ganancia de cada iteración interna;
  - tickets por poda iterativa con rebobinado y su solapamiento entre tareas;
  - reinicialización de unidades ociosas sin cambiar el total de parámetros.
- **Lottery Ticket Theory** (Frankle & Carbin, 2019): **solo como diagnóstico**, con máscaras y rewinding, para encontrar subredes redundantes y capacidad reasignable. Nunca para reducir Hope de forma permanente (ASG §9). La sparsity no se toma como mejora de velocidad salvo que se mida.

### 6.4 Velocidades de plasticidad (HNHS §9, AANS §6, HLA §10)

| Nivel | Qué cambia | Cuándo | Autorización |
|---|---|---|---|
| 1. Actividad | Estados, compuertas, umbrales | En inferencia | Automática, dentro de límites |
| 2. Rápida (FAE) | Adaptadores tipo LoRA por sesión o tarea, memoria episódica | Durante la tarea; se pueden descartar | Automática; no se consolida sin pruebas |
| 3. Pesos | Pesos del núcleo | Sesiones de entrenamiento | Automática con replay + pruebas de regresión |
| 4. Estructural local | Poblaciones, conexiones, especialistas | Solo en ramas experimentales | Según el modo de crecimiento (§8.3) |
| 5. Regional / global | Rutas, regiones fundamentales, arquitectura | Solo en ramas experimentales | **Siempre requiere aprobación humana** (Carta art. 6.7) |

**Implementado** (`hope/plasticity/`):
- **Nivel 2, adaptación rápida (FAE):** adaptadores de bajo rango (LoRA) que se enganchan sin cambiar la función. Se entrenan con la base congelada y después se consolidan en los pesos (`merge_adapters`) o se descartan volviendo exactamente al estado anterior (`discard_adapters`). Las operaciones estructurales se niegan si hay adaptadores enganchados.
- **Nivel 3, protección de lo consolidado:** penalización tipo EWC con la diagonal de Fisher, guardada en bf16 (`ConsolidationPenalty`). Complementa al repaso; en la prueba reduce el olvido.
- **Plasticidad modulada por región:** cada región aprende a su ritmo según su estado en el registro (protegida ×0,05, estable ×1, experimental ×3, con límites duros de 0 a 5) (`PlasticityPolicy`, `Trainer.set_plasticity`).

---

## 7. Diagnóstico unificado de carencias

Las especificaciones definen el mismo procedimiento cinco veces: Herramientas §4 y §12, ASG §4, AANS etapa 1, ARS §11 y HLA §14. **[Propuesta]** Un único esquema y un único enrutador.

### 7.1 Registro de diagnóstico

```yaml
Diagnostico:
  id: dx_…
  evidencia_observada: […]          # ejemplos concretos, ids PESAM
  tareas_afectadas: […]
  frecuencia: …
  reproducibilidad: …               # ¿se repite con otras semillas/datos?
  hipotesis_causa:                  # ordenadas por el orden obligatorio (§2.2)
    - {tipo: DATOS_INSUFICIENTES | ETIQUETAS_ERRONEAS | OPTIMIZACION |
              REPRESENTACION | CONOCIMIENTO_FALTANTE | HERRAMIENTA_FALTANTE |
              ESTRATEGIA_RAZONAMIENTO | MEMORIA_RECUPERACION | CONEXIONES |
              CAPACIDAD_INSUFICIENTE,
       confianza: 0..1, prueba_para_confirmar: "…"}
  alternativas_a_la_expansion: […]
  decision: …
```

### 7.2 Enrutamiento

| Causa | Se envía a |
|---|---|
| Datos / etiquetas | Pipeline de datos (pedir datos o limpiar) |
| Conocimiento | HLA (con consulta dirigida a Qwen por ATI, HLA §11) |
| Herramienta | HTE → TCG → TVE (ciclo de 9 etapas de Herramientas §4) |
| Estrategia | ARS/MRE |
| Memoria / recuperación | MDD/AHMC |
| Conexiones / representación | AANS (reorganizar sin crecer) |
| Capacidad realmente insuficiente | ASG (crecer dentro del techo) |

**Criterio de éxito 1 de ASG §18:** que este enrutador distinga correctamente la causa. Se mide con **fallos inducidos de causa conocida** (§12.2).

---

## 8. Capa E — Seguridad (Carta Fundamental, H-SVF)

La Carta declara jerarquía **superior** a todos los objetivos operativos. **[Propuesta]** Traducirla a mecanismos verificables (art. 13):

### 8.1 Guardián independiente

- Es un **proceso separado** del modelo, con su propia configuración en archivos **de solo lectura** para el proceso del modelo (permisos del sistema operativo).
- Toda acción con efecto fuera del sandbox pasa por él: herramientas, red, archivos fuera del directorio de trabajo, hardware, consolidaciones y cambios de presupuesto.
- Clasifica cada acción en tres niveles (Carta art. 5, HED §13):
  - **Nivel 1:** se registra.
  - **Nivel 2:** permisos predefinidos + sandbox.
  - **Nivel 3:** aprobación humana explícita.
- **Lo que no está autorizado se rechaza.** La ausencia de autorización no es permiso (art. 5).

### 8.2 Componentes

- **HPA:** tabla de permisos y flujo de aprobación humana.
- **HSI (invariantes):** reglas deterministas que ninguna política puede desactivar. Ejemplos:
  - no superar `max_parametros_totales`;
  - no escribir en la configuración de seguridad;
  - no ejecutar código nuevo fuera del sandbox;
  - no controlar hardware físico sin aprobación.
- **HSA (auditoría):** registro de solo agregado **encadenado por hash**, para detectar manipulaciones (art. 12). Registra también los intentos de modificar las garantías (art. 6.5).
- **Sandbox:** contenedor o proceso aislado con límites de CPU, RAM, tiempo y red, sin privilegios de administrador (Herramientas §9).
- **Equipos físicos:** sus protecciones (eléctricas, térmicas, mecánicas) son **hardware o firmware independientes**. Hope no puede anularlas (Herramientas §9, HCRI §12).

### 8.3 Conflicto detectado: auto-consolidación vs. Carta

- **ASG §14** permite `AUTO_CONSOLIDATION_WITHIN_LIMIT`.
- **Carta art. 6.7** exige aprobación humana para "modificaciones importantes".
- **Resolución (decidida por el usuario):** la Carta prevalece. Un cambio es «importante», y **siempre** requiere aprobación humana, si cumple cualquiera de estas condiciones:
  - agrega o retira **más del 10%** de los parámetros totales en un solo cambio;
  - es un cambio de nivel 5 (regional o global, §6.4);
  - toca una región protegida.

  La auto-consolidación solo aplica por debajo de esos límites. Está implementado en `configs/growth_budget.yaml` y en el guardián.

### 8.3.1 Acceso a internet (pedido del usuario; `configs/web.yaml`, `guardian/web.py`)

- Cada acceso a la red es `network.access` (nivel 3). El guardián lo decide con los permisos que fija el usuario en `configs/web.yaml`, releído en cada pedido:
  - `preguntar` (por defecto): solicitud pendiente hasta la aprobación humana (una aprobación vale para un pedido);
  - `lista_permitida`: para los dominios de la lista, la aprobación humana queda **delegada** en esa configuración del usuario, y cada acceso se registra en la auditoría con ese motivo; fuera de la lista, solicitud pendiente;
  - `apagado`: denegado sin solicitud.
- **Invariantes:** máximo de consultas por hora y de bytes por día (el guardián lleva el contador), dominios bloqueados, solo http/https, redirecciones re-autorizadas. Al alcanzar un límite se deniega; no se pregunta.
- `configs/web.yaml` y el contador están en rutas protegidas: Hope no puede modificarlos.
- **Solo lectura** en todos los modos: únicamente GET/HEAD; otros métodos, cuerpos, credenciales y cookies se deniegan sin solicitud; `network.write` es una acción prohibida; `external.publish` pedida por la herramienta web se deniega. El navegador de capturas e inspección aborta toda petición que no sea de lectura, las navegaciones y los sitios no permitidos.
- El bot se identifica (User-Agent con el contacto del usuario) y nunca se hace pasar por una persona; google.com está en la lista `prohibido`.
- El contenido web es **no confiable**: se marca como dato con su URL, se neutralizan las instrucciones dirigidas al modelo y nunca se ejecuta. Detalle en docs/HERRAMIENTAS_HN1.md §7.
- Adquisición de datos: solo fuentes legales registradas en `configs/fuentes_web.yaml` (también del usuario), con licencia por documento y cuota diaria (docs/resultados/fuentes_legales.md).

### 8.4 Pruebas de regresión de seguridad (art. 14)

Hay un conjunto fijo de pruebas que **toda** versión candidata debe pasar antes de consolidarse:

- instrucciones maliciosas;
- intentos de eliminar restricciones;
- honestidad sobre la incertidumbre;
- respeto de permisos;
- comportamiento durante el sueño y la autoevolución.

Ningún resultado de benchmark compensa un fallo en estas pruebas.

---

## 9. Estados epistémicos y de ciclo de vida (unificación)

Las especificaciones definen al menos cinco listas de estados distintas: SCA §2, HCRI §9, HED §16, Herramientas etapa 8 y HWM §6. **[Propuesta]** Usar dos enumeraciones compartidas en todo el sistema:

**`EstadoEpistemico`** (para conocimientos, ideas, restricciones, modelos, relaciones):

| Estado | Significado | Equivalencias en las especificaciones |
|---|---|---|
| `OBSERVADO` | Dato tomado de una fuente, con procedencia | — |
| `INFERIDO` | Derivado por razonamiento, sin verificar | — |
| `GENERADO_EN_SUENO` | Producido por SCA | DREAM_GENERATED, ASSOCIATIVE_DISCOVERY |
| `SIMULADO` | Resultado de un modelo interno | INTERNAL_SIMULATION |
| `HIPOTESIS` | Propuesta pendiente de prueba | HYPOTHESIS, TECHNICAL_TEST_PENDING, "Hipótesis prometedora" |
| `PROBADO` | Sometido a pruebas, sin veredicto suficiente | TECHNICALLY_TESTED, "No verificado" |
| `RESPALDADO` | Evidencia favorable, falta validación independiente | "Respaldada provisionalmente" |
| `VALIDADO_CONDICIONAL` | Válido bajo condiciones o aproximaciones registradas | PARTIALLY_VALIDATED, Aproximada, Condicional |
| `VALIDADO` | Superó pruebas independientes dentro de un dominio definido | VALIDATED, Confirmada |
| `REFUTADO` | Contradicho por evidencia | REJECTED, Refutada |
| `INCONCLUSO` | No hay información suficiente | INCONCLUSIVE, Desconocida |

**`EstadoCicloVida`** (para herramientas, arquitecturas, especialistas, estrategias): `EXPERIMENTAL`, `EN_VALIDACION`, `APROBADO_CON_LIMITACIONES`, `ESTABLE`, `DEPRECADO`, `BLOQUEADO`.

**Regla:** un estado solo sube por evidencia registrada. Nunca sube por repetición: SCA §2 dice que "imaginar algo muchas veces no aumenta su confianza".

---

## 10. Sistemas de aprendizaje y descubrimiento

### 10.1 Entrenamiento del núcleo (HLA, Language Core §10–11)

| Fase | Datos | Objetivo |
|---|---|---|
| P0. Preentrenamiento | Corpus reales (español + inglés + código + texto técnico y científico) con procedencia y licencias registradas | Predicción del siguiente byte |
| P1. Episodios educativos | Generados por Qwen con la estructura de HLA §12 (concepto, explicación, principios, ejemplo, variaciones, casos límite, contraejemplos, problemas de aplicación y transferencia) | Aprendizaje con pocos ejemplos y transferencia |
| P2. Razonamiento y uso de herramientas | Problemas verificables (matemática, código, circuitos) con verificadores deterministas | Entrenar la parada adaptativa, el router y las llamadas a herramientas |
| P3. Familias ARS §15 | Problemas incompletos, contradictorios, engañosos, sin solución, etc. | Detectar carencias, saber decir "no sé" y cambiar de estrategia |

#### 10.1.1 Corpus de preentrenamiento

**Fuentes confirmadas:** papers, diccionarios, Wikipedia, libros y manuales. **Idiomas: español e inglés.** **Volumen:** ~1 TB al inicio y 3–4 TB a largo plazo, más los episodios generados por Qwen.

- **Dos idiomas con 200M:** es una carga razonable. Se mide el rendimiento por idioma y se ajusta la mezcla (propuesta inicial: alrededor de 50/50, a revisar con los resultados).
- El procesamiento por bytes trata igual a ambos idiomas; la línea base BPE se entrena con un tokenizador propio ES+EN.
- **1 TB en la A770:** funciona como **reserva de la que se seleccionan** los subconjuntos de entrenamiento (§3.9). Procesarlo entero requiere más cómputo.

Es un corpus curado de alta calidad que encaja bien con el enfoque técnico y científico de Hope.

**Proceso de datos [Propuesta]:**
1. **Extracción:** papers y manuales suelen venir en PDF. Hay que extraer texto conservando fórmulas (LaTeX), tablas y la estructura de secciones, con herramientas tipo Marker, Nougat o GROBID. Es la etapa más costosa en tiempo de CPU.
2. **Normalización y deduplicación:** a nivel de documento (hash) y de fragmentos (MinHash). Wikipedia y los libros comparten mucho texto con otras fuentes.
3. **Filtrado de calidad:** descartar texto mal extraído, páginas de índice y basura de OCR.
4. **Registro de procedencia y licencia** por documento. Wikipedia es CC BY-SA, los papers de arXiv tienen licencias variables y muchos libros tienen derechos de autor. Para uso experimental privado alcanza con el registro; para publicar pesos o datos hay que revisarlo.
5. **Separación de conjuntos sellados** antes de entrenar.

**Doble uso:** diccionarios, manuales y Wikipedia sirven también como **archivo externo de AHMC**. El modelo puede recuperar la definición o el dato exacto con su fuente, en vez de depender solo de haberlo memorizado. Eso cumple ASG §12.

**Brechas detectadas:**
- **Código: se agrega en cantidad (decisión del usuario).** Fuente propuesta: repositorios con licencia permisiva y deduplicados (por ejemplo, subconjuntos de The Stack v2 filtrados por licencia). Lenguajes prioritarios: Python, C y C++, más los que pide Hope Code. Compite por la capacidad del modelo con ES/EN; propuesta inicial de 20–30% del corpus de entrenamiento, ajustable con resultados.
- **Diálogo e instrucciones:** falta. Se cubre después con los episodios educativos de Qwen (P1).

**Reglas para el maestro:**
- **Destilación a nivel de secuencia, no de logits.** Hope trabaja en bytes y Qwen usa su propio tokenizador, así que la destilación de logits no es directa.
- **Filtrado contra errores del maestro:** todo dato verificable de Qwen se comprueba con herramientas (ejecutar el código, resolver la matemática) antes de entrenar. Lo no verificable se marca y se le da menos peso (Language Core §10).
- **Enseñanza explicada, no solo destilación (decisión del usuario):** Qwen explica el razonamiento completo: datos, supuestos, procedimiento, pasos intermedios y cómo llegó al resultado. Esto **reemplaza** la regla de Language Core §10 que prohibía pedir cadenas de razonamiento. **Confirmado por el usuario.**
  - Se pide como **explicación didáctica estructurada**, no como el "pensamiento" interno crudo del modelo.
  - **Riesgo de racionalización:** una explicación armada "desde el resultado hacia atrás" puede ser una justificación verosímil que no refleja cómo se resuelve realmente. Por eso cada paso se verifica, no solo la respuesta final.
  - **Riesgo de imitación:** un modelo chico puede copiar el estilo de una explicación larga sin entenderla. Por eso los episodios se combinan con problemas que Hope resuelve sin explicación (niveles ART de HLA §6).
- **Formato del episodio de enseñanza [Propuesta]:**
  ```yaml
  Episodio:
    idioma: …
    problema: …
    datos: [{valor, unidad, fuente}]
    supuestos: […]
    plan: "…"                      # qué estrategia y por qué
    pasos: [{n, operacion, resultado_intermedio, justificacion, verificable: bool}]
    resultado: …
    verificacion_inversa: "…"      # desde el resultado: ¿cumple datos, unidades y restricciones?
    alternativas_descartadas: […]
    errores_frecuentes: […]
    variaciones: […]               # para transferencia (HLA §12)
    verificacion_herramientas: {estado, detalle}   # completado por HTE, no por Qwen
    estado_epistemico: …
  ```
- **Costo de generación [Estimación]:**
  - Un Qwen de 7–8B a 4 bits en la A770 produce del orden de decenas de tokens por segundo.
  - Mil millones de tokens de explicaciones llevarían cerca de un año. **La generación del maestro es el cuello de botella real.**
  - Hay que priorizar **decenas de miles de episodios de alto valor**, elegidos por el diagnóstico de carencias (§7). Se puede usar un Qwen más chico para episodios simples, o una API si se acepta.
- **Licencia:** verificar la licencia del modelo Qwen elegido para generar datos de entrenamiento. La mayoría de los modelos Qwen2.5 son Apache 2.0, pero hay excepciones por tamaño.

#### 10.1.2 Ciclo maestro–alumno (implementado: `hope/teacher/cycle.py`)

**Decisión del usuario:** al principio Qwen enseña todo lo que pueda, hasta que Hope razone de forma semi-funcional. Después, Hope pregunta por sus carencias y Qwen responde. El ciclo tiene dos fases:

1. **Arranque (currículo).** Qwen enseña por iniciativa propia, nivel por nivel (`configs/curriculum.yaml`, español e inglés). Un nivel se habilita cuando Hope supera los temas del anterior.
2. **Preguntas.** Cuando se cumple el criterio de «razonamiento semi-funcional», Hope intenta, detecta qué falla y pregunta de forma dirigida: qué intentó, qué tiene en memoria y qué salió mal.

Cada vuelta, en las dos fases:
- Qwen explica con un episodio.
- Las herramientas verifican cada paso (aritmética exacta, SymPy en sandbox, unidades, código). Lo refutado no se entrena y queda como ejemplo de error; lo no verificable pesa menos.
- Lo verificado entra en la memoria, y Hope lo usa enseguida como ejemplo resuelto al responder.
- Se consolida en los pesos con repaso del corpus.
- Hope razona paso a paso sobre las variaciones, problemas que nunca entrenó, y se mide la exactitud por tema.

**Criterio de «razonamiento semi-funcional» (provisional; lo fija el usuario):** exactitud ≥ 60% en variaciones no vistas, en ≥ 80% de los temas con al menos 5 problemas, habiendo pasado por todos los niveles del currículo.

#### 10.1.3 Generadores verificables y autoentrenamiento (propuestas 1 y 2, aprobadas)

- **Generadores** (`hope/data/generators.py`): nueve familias de problemas con respuesta exacta, en el mismo formato de episodio que Qwen y con cada paso verificable por herramientas. Cada ronda del ciclo agrega episodios generados a los temas que tienen generador.
- **Limitación actual:** varias familias producen solo enunciados en español.
- **STaR** (`hope/train/star.py`): Hope resuelve problemas de práctica (generados, nunca los de evaluación) con varias muestras. Se entrena solo con las soluciones que aciertan **y** pasan la crítica; un acierto con pasos incorrectos se descarta.
- **Racionalización:** si no resuelve un problema, se le da la respuesta como pista y se guarda la explicación verificada sin la pista.
- **Efecto esperado:** cada consulta a Qwen (lenta en la A770) rinde más, y la dependencia del maestro baja con el tiempo.

**Contaminación:** ningún ejemplo de enseñanza aparece idéntico en la evaluación (HLA §12). Se deduplica por n-gramas contra los conjuntos reservados.

#### 10.1.3.1 Calidad de datos, fuentes y entrenamiento de largo plazo (pedido del usuario)

- **Filtro de calidad** (`hope/data/quality.py`):
  - reglas estilo Gopher/C4 que descartan basura sin costo;
  - clasificador estilo FineWeb-Edu: Qwen puntúa de 0 a 5 una muestra y un regresor sobre n-gramas de bytes (≈ 4 ms por documento en CPU) puntúa el resto. Se entrena con lo mejor puntuado.
- **Fuentes** (`hope/data/sources.py`): Wikipedia en español e inglés desde Hugging Face, con procedencia y licencia (CC BY-SA) por documento, y código de carpetas locales. Cada fuente se construye con un presupuesto de bytes.
- **Repositorios propios** (`hope/data/repos.py`):
  - código actual + historial de commits como episodios «problema → archivos → corrección», con los arreglos marcados;
  - los commits con coautoría de una IA se excluyen por defecto (`--incluir-coautoria-ia` para incluirlos): los términos de uso de varios proveedores restringen entrenar otros modelos con sus salidas. En los repos del usuario, 1126 de 1128 commits tienen esa coautoría.
- **WSD** (`TrainConfig.schedule="wsd"`, `Trainer.anneal`): tasa constante durante meses y enfriamiento corto al final con los mejores datos. Se puede enfriar una copia en cualquier momento sin cortar la rama estable.
- **Muon** (`hope/train/muon.py`): opción a comparar con AdamW. Se adopta solo si gana la ablación.
- **Programa de largo plazo** (`python -m hope.train.run --plan configs/plan_6m.yaml`):
  - etapas por horas, checkpoint cada 30 minutos y retoma automática;
  - lote automático;
  - crecimiento con aprobación humana (se detiene y espera);
  - enfriamiento final guardando antes la rama estable.

#### 10.1.4 Qwen describe imágenes (decisión del usuario; `hope/teacher/vision.py`, `hope/data/vision.py`)

- **Una consulta densa por imagen:** Qwen multimodal devuelve un JSON con descripción en español e inglés, texto visible, objetos con cantidad y posición, datos extraídos (gráficos, tablas, circuitos) y 3–5 preguntas con razonamiento. De cada anotación salen 5–9 muestras de entrenamiento (describir, transcribir, contar, extraer datos, responder con razonamiento). La pérdida se aplica solo sobre la respuesta.
- **Analizar más con menos consultas:**
  1. deduplicación exacta (hash) y casi exacta (hash perceptual);
  2. caché por contenido: se puede cortar y retomar sin repetir consultas;
  3. con un presupuesto de N consultas, se eligen las N imágenes **más diversas** (k-centro voraz), no las primeras N;
  4. consultas en paralelo al servidor local;
  5. las imágenes se guardan una vez, redimensionadas, en JPEG (≈ 20 KB a 224 px).
- **Imágenes procedurales verificables:** formas (contar, color, posición), texto renderizado (lectura) y gráficos de barras (valores). Son infinitas, gratis en CPU y con respuesta exacta.
- **Verificación del maestro:** `audit_teacher` le muestra a Qwen imágenes procedurales y compara con la verdad. Así se mide su tasa de error antes de confiar en sus descripciones.
- **Entrenamiento:** `Trainer.train(..., vision=fuente, vision_frac=0.3)` mezcla pasos de imagen y de texto. El texto hace de repaso para que no se olvide mientras aprende a ver.
- **Cuello de botella esperado:** la velocidad de Qwen, no la de Hope. Un modelo de visión de ~8B en la A770 anota del orden de 1 imagen cada pocos segundos (≈ 0,5–1M de imágenes por mes si corriera todo el tiempo). Por eso conviene:
  - priorizar diversidad;
  - repetir épocas sobre lo anotado;
  - usar las procedurales y los conjuntos públicos para el volumen.
- **Uso:** `python -m hope.teacher.vision --images carpeta --store data/vision --budget 5000 --audit 50`.

### 10.2 SCA — sueño cognitivo

- Es un **planificador** que corre cuando no hay tareas, con presupuesto de AIS. Es suspendible y guarda su estado (SCA §11).
- **Modos** (SCA §4):
  - *ligero* = consolidación de memoria y replay;
  - *asociativo* = proponer relaciones en el índice;
  - *analítico* = reexaminar fallos de PESAM;
  - *experimental / evolutivo* = generar entradas para HED/ASG.
- **Salida obligatoria:** registros de idea (esquema de SCA §6) con estado `GENERADO_EN_SUENO` o `HIPOTESIS`. **El sueño nunca modifica pesos consolidados ni estructura.** Solo propone (SCA §7).

### 10.2.1 Implementación v0 (`hope/sleep/`)

- **Modos:**
  - ligero: consolidación de memoria;
  - asociativo: relaciones entre temas distintos, marcadas como generadas en sueño;
  - analítico: errores recurrentes según la crítica, y reintento de problemas fallados;
  - creativo: problemas compuestos;
  - evolutivo: propuestas para ASG/AANS a partir del diagnóstico LTH (iteraciones sin ganancia, especialistas sin uso, neuronas ociosas).
- **Registro de ideas** con los campos de SCA §6. Las ideas repetidas se descartan y hay un tope por sesión.
- **Interrumpible:** guarda en qué modo quedó y retoma desde ahí.
- **Vigilia** (`wake_review`): prueba las ideas con las pruebas disponibles. Una idea sube a RESPALDADO o baja a REFUTADO solo con evidencia; las refutadas se conservan.
- **Lo que el sueño nunca hace:** modificar pesos o estructura, o ejecutar acciones externas.

### 10.3 HWM + HED

- **HWM v0:** modelos explícitos (grafo + ecuaciones SymPy + simuladores de HTE) construidos con representaciones MRC. No hay un "modelo del mundo neuronal" separado.
- **HED:** el ciclo de investigación de HED §9 sobre el mismo esquema de diagnóstico, con priorización IGM por valor de información / costo.
- **Integración con AANS para estudiar la arquitectura propia:** es la fase 5 de HED. Requiere que F5–F6 (§11) ya funcionen.

### 10.4 Ecosistema de herramientas (HTE)

- **Catálogo:** cada herramienta usa el formato de Herramientas §16 (YAML) con `EstadoCicloVida`.
- **Interfaz estándar:** entrada y salida tipadas (JSON Schema), unidades explícitas y errores explícitos.
- **Herramientas iniciales deterministas** (Herramientas §14, fase 1):
  - aritmética de precisión (`decimal` / `mpmath`);
  - SymPy;
  - análisis dimensional (`pint`);
  - ejecución de Python en sandbox;
  - z3;
  - búsqueda en la memoria propia.
- **Principio "verificación independiente":** una herramienta creada por TCG no se valida solo con las pruebas que generó ella misma (Herramientas §18).
- **Usar bien las herramientas** (docs/HERRAMIENTAS_HN1.md §8): banco sellado (`hope/eval/tool_bench.py`), currículo con llamadas ejecutadas (`hope/data/tool_curriculum.py`), sugerencias de la TLM por temas parecidos y refuerzo (`hope/train/tool_rl.py`).
- **Internet** (`hope/tools/web.py`): `⟦buscar_web: consulta⟧` (Wikipedia, arXiv, DuckDuckGo, SearxNG; sin claves) y `⟦leer_web: url⟧`, con permiso del usuario (§8.3.1), caché, límites, robots.txt y contenido marcado como no confiable. Lo leído entra a la memoria como OBSERVADO con su URL y las contradicciones entre fuentes quedan registradas. `hope/data/web_corpus.py` convierte búsquedas en corpus filtrado (calidad, deduplicación, licencia) y `hope/data/web_episodes.py` genera episodios de búsqueda sin red para el currículo de herramientas, también desde trayectorias reales. `⟦inspeccionar⟧` (`hope/tools/web_visual.py`) ve las páginas en un navegador aislado de solo lectura: capturas para la zona visual y DOM + CSS + manejadores de eventos para aprender desarrollo web. `hope/data/web_sources.py` trae cada día datos de fuentes legales y `⟦buscar_wiki⟧` consulta la Wikipedia local sin red (docs/HERRAMIENTAS_HN1.md §7).

---

## 11. Hoja de ruta con criterios de salida

Unifica ASG §17, Herramientas §14, AANS §14, HNHS §21, MDD §16, HED §17 y HLA §19. **Ninguna fase avanza por haber implementado funciones, sino por pasar sus pruebas** (ASG §17).

| Fase | Contenido | Criterio de salida (medible) |
|---|---|---|
| **F0. Infraestructura** | Prueba del entorno XPU (benchmark bf16, SDPA, torch.compile) + registro de topología, contador de parámetros, `GrowthBudget`, linaje y rollback, guardián + auditoría, sandbox, harness de evaluación con conjuntos sellados, esquemas de memoria y estados epistémicos | Pruebas unitarias. Una operación estructural sobre un modelo de juguete **preserva la salida** (ε < 1e-5) y se revierte. El guardián bloquea una operación que excede el techo |
| **F1. Núcleo pequeño (~25M)** | Toda la arquitectura de la §3 a escala reducida + línea base BPE equivalente | Entrena establemente. Pérdida comparable o mejor que la línea base con el mismo cómputo. Ablación HNHS medida |
| **F1.b Núcleo 200M** | Escalado a ~175M asignados, con un subconjunto filtrado de 20–100 GB | Curvas de pérdida y benchmarks iniciales de Language Core §13 en ES/EN |
| **F2. Memoria** | AHMC/MDD/PESAM + atención de recuperación + continuidad | Precisión de recuperación y recuperación exacta de números desde la fuente |
| **F3. Razonamiento adaptativo** | Núcleo recurrente + parada + ARS v0 + CRE + HCRI | Mejor exactitud por unidad de cómputo que con iteraciones fijas. Detección de contradicciones y de datos faltantes |
| **F4. Herramientas** | HTE + TLM + selección de herramientas + TCG/TVE con Qwen | % de herramienta correcta. Mejora frente a "sin herramientas" (comparaciones A–D de Herramientas §11) |
| **F5. Plasticidad controlada** | Operaciones de §6.2 + replay/EWC + LTH como diagnóstico | Retención ≥ umbral tras aprender tareas nuevas. Mejor que una red sin plasticidad estructural (ASG §17 F3) |
| **F6. ASG** | Diagnóstico (§7) + ciclo experimental completo | Supera el experimento de validación de §12.2 |
| **F7. SCA + HED + MLE** | Sueño, descubrimiento y metaaprendizaje | Comparaciones A–D de SCA §12 y A–E de ARS §16 |
| **F8. Escalado** | Techos de 500M, 1B, … (por autorización) | Comparaciones obligatorias de ASG §18 |
| **F9. Harness de trabajo (pedido del usuario) — en progreso (§4.2.5)** | Interfaz tipo Claude Code: el usuario escribe y Hope trabaja sobre una **carpeta local fija** (no un repositorio de GitHub). Hope y Qwen corren **en paralelo** sobre la misma tarea para comparar respuestas y, cuando corresponda, usar a Qwen como maestro. Herramientas: leer, buscar, editar y ejecutar dentro de la carpeta, siempre a través del guardián. Las sesiones se registran en PESAM y pueden convertirse en datos de entrenamiento (con verificación, §10.1) | Requiere el modelo de 200M entrenado. Criterios: ninguna acción fuera de la carpeta sin aprobación; registro completo; comparación Hope vs. Qwen por tarea. **Estado:** la infraestructura (`hope/harness/`) cumple los tres criterios en las pruebas (`tests/test_harness.py`) con agentes simulados, `MockTeacher` y `hn1-tiny`; falta usarlo con el modelo de 200M entrenado y un Qwen real |

**Recomendación fuerte:** hacer F1 a **~25M** antes de 200M. Cada error de diseño cuesta horas en lugar de días.

### 11.1 Plan de 6 meses en la A770 (decisión del usuario: hasta 500M, con visión)

Cálculo con `python -m hope.bench.plan --tflops 12`. **12 TFLOPS efectivos es una suposición:** el valor real sale de `env_check` y todas las cifras escalan con él.

| etapa | modelo | meses | cómputo en imágenes | texto | imágenes vistas | tokens equivalentes |
|---|---|---|---|---|---|---|
| 1. texto y código | hn1-200m | 2 | 0% | ~220 GB | — | ~56B |
| 2. crecer a 500M; visión con procedurales | hn1-500m | 1 | 25% | ~60 GB | ~36M | ~18B |
| 3. multimodal con Qwen + ciclo maestro–alumno | hn1-500m | 3 | 30% | ~170 GB | ~130M | ~53B |

- **Por qué crecer y no empezar en 500M:** los primeros 2 meses cuestan un 27% menos por byte, y el crecimiento que preserva la función conserva lo aprendido.
- **Pasos que requieren aprobación humana:** subir el techo a 500M (`python -m guardian.cli limite 500M --by <usuario>`) y cada paso de crecimiento de más del 10%.
- **Imágenes vistas** cuenta pasadas, no imágenes distintas: las anotadas por Qwen se repiten en varias épocas y las procedurales completan el volumen.
- **Memoria:** 464M en `bf16-8bit` ocupan ≈ 2,8 GB más las activaciones. Un Qwen de visión de ~8B a 4 bits (≈ 6 GB) entra al mismo tiempo en los 16 GB.


---

## 12. Evaluación

### 12.1 Reglas generales

- **Conjuntos reservados sellados:** se guardan con hash. El harness registra cada acceso. No se usan para entrenar ni para decidir cambios repetidamente, y se renuevan periódicamente (ASG §8).
- **Todas las comparaciones con datos y cómputo equivalentes**, con varias semillas e intervalos de confianza.
- **Multidimensional** (AANS §8): exactitud, generalización, retención, latencia, VRAM, operaciones por tarea, parámetros totales y activos, y costo de mantenimiento.
- **Ablaciones obligatorias** para cada mecanismo nuevo.

### 12.2 Experimento de validación de la hipótesis central **[Propuesta]**

Antes de confiar en ASG con lenguaje real, conviene probarlo en un **banco sintético donde la respuesta correcta se conoce**:

- Varias familias de tareas (aritmética modular, copia larga, lógica proposicional, lenguaje simple) con **cuellos de botella inducidos de causa conocida**:
  - una región demasiado pequeña (causa real: capacidad);
  - datos con etiquetas corruptas (causa: datos);
  - una tarea que una calculadora resuelve (causa: herramienta);
  - una tarea que necesita recuperar algo (causa: memoria).
- **Éxito:**
  1. el diagnóstico (§7) identifica la causa correcta por encima de un umbral;
  2. ASG solo crece cuando la causa es capacidad, y en la región correcta;
  3. ASG vence a la expansión uniforme con los mismos parámetros y cómputo (ASG §18).

Si esto falla a pequeña escala, no tiene sentido escalarlo.

### 12.3 Comparaciones obligatorias (ASG §18)

Red fija de 200M · ampliada uniformemente · ampliada manualmente · ASG · ASG + LTH + AACA. Todas con datos, entrenamiento y presupuesto equivalentes.

---

## 13. Riesgos principales

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Bytes con parches poco probados a 200M | Peor lenguaje por parámetro | Línea base BPE en paralelo. Decidir con datos en F1 |
| Restricciones HNHS (Dale, inhibición competitiva) empeoran el rendimiento | Pérdida de calidad | Ablación en F1. Mantener solo lo que mejore |
| Inestabilidad del bucle recurrente y del router | Entrenamiento divergente | Normalización, límites de activación, pérdida de balanceo del router, `n_max` acotado |
| Datos insuficientes para 200M | Modelo débil en lenguaje general | Expectativas explícitas (§3.9). Priorizar dominios técnicos |
| El maestro introduce errores | Aprender errores | Verificación por herramientas y pesos por confianza |
| Sobreajuste al evaluador (ASG §13, AANS §10) | Falsas mejoras | Conjuntos sellados, rotación y evaluadores independientes |
| Explosión de alcance (40 módulos) | Nada termina | Fases con criterios de salida. v0 mínima de cada módulo |

---

## 14. Estructura del repositorio propuesta

```
hope/
  core/        # capa A: codificador de bytes, backbone, núcleo recurrente, especialistas, HNHS
  registry/    # topología, IDs estables, conteo de parámetros
  growth/      # ASG, operaciones estructurales, GrowthBudget, diagnóstico
  lineage/     # checkpoints, ramas, rollback
  runtime/     # ARS/ARC, Loop Controller, AIS, HTO, CRE
  constraints/ # HCRI: solvers, IRM, verificación dimensional
  repr/        # MRC: sobre de representación, traductores
  memory/      # AHMC, MDD (n-gramas, engramas, índice), PESAM
  tools/       # HTE, TLM, TCG, TVE, sandbox
  teacher/     # ATI: cliente Qwen, generación de episodios
  sleep/       # SCA
  discovery/   # HWM, HED
  eval/        # harness de evaluación, conjuntos sellados, comparaciones
  harness/     # F9: consola de trabajo sobre una carpeta local, Hope vs. Qwen en paralelo (§4.2.5)
guardian/      # capa E: proceso separado (permisos, invariantes, auditoría)
configs/       # GrowthBudget, modelos, entrenamiento
docs/
tests/
```

Stack: Python 3.11+, PyTorch con backend `xpu` (Intel Arc), código portable a `cuda` y `cpu`, SSM implementado en PyTorch/Triton (sin kernels CUDA), SQLite, FAISS (CPU), SymPy, z3, pint.

---

## 15. Decisiones que necesito del usuario

1. ~~Hardware~~ → Intel Arc A770 16 GB + 64 GB RAM + **1 TB de disco** (§3.9).
   - **Problema:** 1 TB de corpus crudo no entra en 1 TB de disco junto con los datos procesados, los índices y los checkpoints.
   - **Mitigación [Propuesta]:** guardar el corpus comprimido (zstd reduce el texto 3–4×) y procesar en streaming. Mantener el crudo en un disco externo. Guardar solo los últimos checkpoints más los de la rama estable (~2,4 GB cada uno con el estado del optimizador).
   - Recomendación: sumar 2–4 TB de disco antes de llegar a 3–4 TB de corpus.
2. ~~Modelo maestro~~ → la intención es usar **Qwen 4**, que todavía no salió. **Decisión [Propuesta]:**
   - ATI se diseña **independiente del modelo**: interfaz común para modelos locales (llama.cpp / OpenVINO) y API.
   - Mientras tanto se desarrolla con el Qwen más reciente disponible, o con uno chico para las pruebas del proceso.
   - Cuando salga Qwen 4 se cambia por configuración.
   - Riesgo: no se sabe si su tamaño útil entrará en los 16 GB de la A770 cuantizado.
3. ~~Umbral de "modificación importante"~~ → más del 10% de los parámetros (§8.3).
4. ~~Composición del corpus~~ → papers, diccionarios, Wikipedia, libros, manuales y código; español e inglés (§10.1.1).
5. **¿Aceptás empezar por F0 + F1 a ~25M** antes de comprometer el diseño a 200M?
