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
- **Resultados en CPU** (`docs/resultados/bucle_grpo.md`; hn1-cpu, 1.500 pasos, 2 semillas): en texto general el bucle estable no mejora el BPB (+0,04, dentro del ruido), pero no colapsa (usa ~5 iteraciones contra 1,0 de ACT) y en cadenas de 2–4 pasos sube la exactitud de 0,26 a 0,32. **No generaliza a más pasos que los entrenados, e iterar más en inferencia (hasta 32) no cambia la exactitud.** Se mantiene como opción hasta repetirlo a 200M.

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
- **Crecimiento** (`hope/growth/hn1_ops.py`; presupuesto como objetivo en §6.5): `AddVisualZone`, `AppendBackboneBlock` (bloque nuevo = identidad) y `plan_growth_to(modelo, "hn1-500m")`, que arma toda la secuencia desde `hn1-200m` con operaciones que preservan la función. Cada paso pasa por el presupuesto y el guardián. Los especialistas nuevos quedan enmascarados hasta habilitarlos (`unmask_expert`).
- **Mosaicos para documentos (`vis_tiling`, sin parámetros nuevos):** una imagen de 2S×2S entra como miniatura S×S **más** 4 mosaicos a resolución completa. Cada mosaico pasa solo por el codificador, así que el costo crece con la cantidad de mosaicos y no con el cuadrado de la imagen. Sus parches llevan la posición fila/columna **de la imagen entera** (interpolada), así la red sabe dónde está cada mosaico. Fichas: (1 + 4)·V. Con `vis_tiling=False` (valor por defecto, compatible con los checkpoints anteriores) todo queda como antes.
- **Pérdida contrastiva auxiliar (opcional, `Trainer.train(..., vision_contrastive=0.1)`):** sigmoide estilo SigLIP (Zhai et al. 2023), sin parámetros nuevos:
  - imagen = promedio de las fichas visuales;
  - texto = promedio de los estados globales de Hope sobre la respuesta, calculados **sin imagen y sin gradiente** (como LiT, pero con el texto fijo): el codificador visual aprende a llevar cada imagen a donde Hope ya representa su descripción;
  - t = 10 y b = −10 fijos; las respuestas idénticas del lote cuentan como positivas.
  - Cuesta un pase hacia adelante extra (sin gradiente) en los pasos con imagen.
- **Currículo (`ImageTextSource.curriculum(total)`):** cada generador tiene un nivel (1 formas y barras … 4 circuitos y OCR con ruido). El nivel máximo sube de 1 a 4 durante el primer 60 % de los lotes de imagen.
- **Respuestas que llaman herramientas:** las cuentas de las respuestas procedurales se escriben como llamadas: `⟦calc: 1/(1/12 + 1/4) + 6⟧ → 9` o `⟦circuito: V1 a 0 5; R1 a n1 100; …⟧ → {…}`. El resultado lo calcula **la herramienta real** (el mismo `ToolRunner` del runtime) y tiene peso 0 en la pérdida, porque en uso lo escribe el runtime. Hope aprende a leer la imagen, armar la llamada y usar el resultado.
- **Banco sellado (`hope/eval/vision_bench.py`):** ver §10.1.4 y `docs/resultados/vision.md`.
- **Pendiente:** medir en la A770 con `hn1-500m` (224 px, documentos a 448 px).

#### Preset `hn1-500m` (464M)

| parte | hn1-200m | hn1-500m |
|---|---|---|
| backbone | 8 bloques (109M) | 12 bloques (164M) |
| zonas | 8 especialistas de 2048 (43M) | 16 especialistas de 4096 (128M), top-2 |
| visión | — | ViT 768×12 + atención cruzada (101M) |
| total | 181M | **464M** (deja ~36M de margen bajo un techo de 500M) |
| cómputo de texto por byte | 1× | **1,36×** |

Presets intermedios (mismo d=1024 y misma parte local; `plan_growth_to` llega a ellos desde `hn1-200m` o `hn1-500m`):

| preset | backbone | zonas | visión | total | cómputo de texto por byte |
|---|---|---|---|---|---|
| `hn1-700m` | 16 bloques | 32 especialistas de 4096, top-2 | igual que 500m (bloques 3, 7 y 10) | **689M** | 1,62× |
| `hn1-1b` | 20 bloques | 56 especialistas de 4096, top-2 | igual que 500m | **998M** | 1,88× |

Crecer desde hn1-200m da unos M menos que el preset construido de cero: los especialistas «relacional» no se ensanchan (quedan en 2048).

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
- ampliarlo al equipo de 6 agentes (maestro, verificador y 4 especialistas): ver §11.4.

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
2. **Explicaciones de Qwen** (destilación de razonamiento), **tutoría on-policy** (Qwen revisa los intentos de Hope, da pistas y puntúa; §10.1.5) y **datos de alta calidad** (§10.1.3.1).
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

### 6.5 Presupuesto de parámetros como objetivo (pedido del usuario; `hope/growth/budget_target.py`)

El usuario dice «tu presupuesto ahora es X» (p. ej. de 200M a 1B, o bajar) y Hope se adapta **sin perder lo aprendido**.

- **Quién fija el techo:** solo el usuario, con `python -m guardian.cli limite X --by <nombre>`. Hope no puede (acción `budget.change` prohibida). El plan nunca supera `min(X, techo)`.
- **Ver el plan antes de aplicar:** `python -m hope.growth.budget_target --checkpoint runs/hope/... --objetivo 1B --perfil a770` muestra las operaciones, los parámetros, la memoria de entrenamiento y la velocidad antes y después. `--aplicar --salida <carpeta>` lo ejecuta.
- **En el plan de entrenamiento:** una etapa `presupuesto: 1000M` en `hope.train.run`. Cada paso de más del 10 % de los parámetros pide aprobación (`structural.consolidate`). Si queda pendiente, el programa guarda lo ya aplicado, se detiene y al relanzarlo replanifica desde ahí (con el mismo cómputo de referencia).

**Crecer** (`plan_to_budget`) con operaciones que preservan la función (§6.2). Política, en este orden:

| orden | familia | límite | parámetros por FLOP/byte de entrenamiento (medido, hn1-200m) |
|---|---|---|---|
| 1 | especialistas (`AddExpert`, del mismo tipo y ancho que el ciclo de la zona) | hasta 64 por zona | ~660 (dispersos: top-2 de n) |
| 2 | bloques al final del backbone (`AppendBackboneBlock`, ritmo ssm, ssm, attn, ssm) | cómputo por byte ≤ 1,4× el de partida | 0,8 |
| 3 | ensanchar especialistas (`WidenExpert` `zero_out`, tandas de 1024, 256 y 64) | hasta 2× el ancho de la configuración y cómputo ≤ 1,6× | 24 |
| 4 | zona visual (`AddVisualZone`) si falta y entra | — | se paga por imagen |

- Las cuentas se hacen sobre una copia «meta» del modelo (sin memoria), así que son exactas y no tocan el original.
- Cada operación tiene que entrar en la memoria del perfil de hardware (`hope.bench.hardware.training_fit`), con el optimizador en la GPU o en la RAM (`offload`).
- La velocidad estimada sale de `hope.bench.plan.speed`.
- Los especialistas nuevos quedan enmascarados hasta habilitarlos: en `hope.train.run` se habilitan al terminar, como en `crecer_a`.

**Resultados del plan desde hn1-200m (181M):**

| objetivo | perfil | resultado | pasos | entrenamiento (texto por día) |
|---|---|---|---|---|
| 500M | A770 | 499,9M | 56 especialistas + 4 ensanches | ×0,997 |
| 700M | A770 | 700,0M | 56 especialistas, 6 bloques, 53 ensanches | 3,7 → 2,6 GB (×0,71) |
| 1B | A770 | 946M | lo anterior + 110 ensanches + visión; el límite de cómputo deja 54M sin usar | 3,7 → 2,4 GB (×0,67) |
| 1B | GTX 1060 | 487M | solo especialistas: más no entra ni con el optimizador en la RAM | ×0,997 |

Para pasar el límite de cómputo hay `max_flops_ratio` (`--max-computo` en la CLI). Para llegar a la estructura exacta de un preset se usa `plan_growth_to` (`hn1-700m`, `hn1-1b`).

**Reducir**, en dos pasos (§6.2: enmascarar → evaluar → eliminar):

- **Candidatos:**
  - bloques **agregados** (posteriores a `base_blocks`, retirados desde el final) cuya contribución relativa `‖bloque(x) − x‖ / ‖x‖` es menor que 0,02, y que no tienen atención cruzada visual ni conexiones;
  - después, los especialistas **menos usados**: fracción de elecciones del router medida con `lth.expert_usage` sobre datos de validación (`--datos`). Quedan al menos top-k + 2 activos por zona y los protegidos no se tocan.
- **Elección:** se prueba cada cantidad de bloques completando con especialistas y gana la que queda más cerca por debajo de X.
- **Pérdida informada:** al planificar se enmascara lo elegido sobre el modelo real, se mide la pérdida (bits por byte) y se deshace. El plan informa esa pérdida.
- **Al aplicar:** se vuelve a enmascarar y medir. Si se aleja de lo informado (más de 10⁻³ bpb), se deshace. Si no, se elimina; eliminar es exacto respecto del estado enmascarado.

**Pruebas** (`tests/test_budget_target.py`, con `hn1-tiny`):
- crecer preserva la salida (ε = 10⁻⁵), nunca se pasa del objetivo ni del techo y respeta el orden de la política y el límite de cómputo;
- la memoria de la 1060 limita el crecimiento;
- sin aprobación no se aplica nada;
- reducir saca lo menos usado y la pérdida medida después es la informada;
- ida y vuelta 1× → 2× → 1× con checkpoints da la misma salida y el entrenamiento la retoma;
- la etapa `presupuesto:` de `hope.train.run` se detiene y sigue con aprobaciones.

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
- **Imágenes procedurales verificables** (`hope/data/vision.py`). Son infinitas, gratis en CPU y con respuesta exacta. Cada pregunta trae su razonamiento paso a paso y una tarea del banco:

  | generador | qué pregunta | nivel |
  |---|---|---|
  | `formas` | contar, color de una forma, dónde está | 1 |
  | `barras` | valor de una barra (escala con marcas), la más alta, suma con ⟦calc⟧ | 1 |
  | `texto` | transcribir, cuenta escrita con ⟦calc⟧ | 2 |
  | `escena` | «¿qué está a la izquierda del círculo rojo?», cuántas a la derecha, la más alta, color, posición | 2 |
  | `reloj` | qué hora marca (agujas), qué hora será dentro de N minutos | 2 |
  | `lineas` | valor en un x, dónde está el máximo, cuánto cambia (⟦calc⟧) | 3 |
  | `torta` | porcentaje de una porción (tolerancia 5), la mayor, color en la leyenda | 3 |
  | `tabla` | leer una celda, sumar una columna (⟦calc⟧) | 3 |
  | `geometria` | áreas de rectángulo, triángulo y círculo, perímetro, tercer ángulo (⟦calc⟧) | 3 |
  | `ocr` | varias líneas, 15 fuentes TrueType (las instaladas), tamaños, colores y ruido | 4 |
  | `circuito` | resistencias en serie/paralelo: R equivalente (⟦calc⟧, verificada con el simulador de circuitos), corriente con una fuente (⟦circuito⟧) | 4 |

  - Los documentos (`tabla`, `ocr`, `circuito`) se generan a `doc_scale`× la resolución, para los mosaicos.
  - Los aumentos `augment` (rotación ±3°, ruido gaussiano, JPEG 40–90) no cambian la respuesta.
  - Las semillas `RESERVED_SEEDS` quedan para el banco: `procedural` se niega a usarlas.
- **Banco visual sellado** (`python -m hope.eval.vision_bench construir … --sellar …`):
  - el mismo número de preguntas en cada una de las 6 tareas: contar, color/forma, posición, lectura de texto (exactitud por carácter), valores de gráficos/tablas/relojes y razonamiento;
  - se mide la exactitud de la respuesta generada (con herramientas, opcional) y el bpb de la respuesta de referencia con la imagen correcta y con una cambiada (cuánto se usa la imagen);
  - se sella con `SealedRegistry`, controla que ninguna imagen del banco aparezca en el entrenamiento y corre igual en CPU (`hn1-cpu` + visión) y en GPU (`hn1-500m`).
- **Verificación del maestro:** `audit_teacher` le muestra a Qwen imágenes procedurales y compara con la verdad. Así se mide su tasa de error antes de confiar en sus descripciones.
- **Entrenamiento:** `Trainer.train(..., vision=fuente, vision_frac=0.3)` mezcla pasos de imagen y de texto. El texto hace de repaso para que no se olvide mientras aprende a ver.
- **Cuello de botella esperado:** la velocidad de Qwen, no la de Hope. Un modelo de visión de ~8B en la A770 anota del orden de 1 imagen cada pocos segundos (≈ 0,5–1M de imágenes por mes si corriera todo el tiempo). Por eso conviene:
  - priorizar diversidad;
  - repetir épocas sobre lo anotado;
  - usar las procedurales y los conjuntos públicos para el volumen.
- **Uso:** `python -m hope.teacher.vision --images carpeta --store data/vision --budget 5000 --audit 50`.
- **Tutores locales con verificación cruzada** (`hope/teacher/vision_tutores.py`, `configs/tutores_vision.yaml`; pedido del usuario). Google Lens no tiene API oficial y automatizarlo viola sus términos, así que se usan modelos abiertos que corren en casa y se controlan entre sí:

  | rol | modelo | licencia | qué confirma |
  |---|---|---|---|
  | principal | Qwen2.5-VL-7B-Instruct (llama.cpp/LM Studio con mmproj) | Apache 2.0 | anota: descripción, texto, objetos (con nombre en inglés), datos, preguntas |
  | segundo | SmolVLM2-2.2B o Moondream2 | Apache 2.0 | contesta las mismas preguntas por su cuenta |
  | florence | Florence-2 large | MIT | OCR y detección con cajas (objetos y cantidades) |
  | ocr | Tesseract o PaddleOCR | Apache 2.0 | el texto |
  | verificador | SigLIP (o CLIP, MIT) | Apache 2.0 | la descripción: tiene que ser la que mejor corresponde a la imagen dentro del lote |

  - La verificación es **campo por campo**. Solo los campos confirmados entran al `ImageStore` (fuente `tutores`, estado `verificado` o `parcial`). Lo dudoso queda en `dudosas.jsonl` con el motivo y **no se entrena**. Los valores de un gráfico sin texto no los puede confirmar el OCR, así que quedan afuera.
  - Licencias: cada tutor declara `entrenar_con_salidas`. Uno con licencia que restringe entrenar con sus salidas (Qwen2.5-VL-3B, PaliGemma, YOLO AGPL; listados en `no_usar_para_entrenar`) solo puede verificar. Si fuera el principal, todo queda como `solo_verificacion`.
  - **Turnos** (GTX 1060): el panel trabaja por etapas, un modelo de visión por turno. Cada etapa guarda su resultado en disco y `--etapa siguiente` corre la que falta; en la A770 entran juntos (`--etapa todas`).
  - **Modo tutor** (`VisionTutorSession`, como `tutor.py`): Hope contesta; si falla, el tutor mira la imagen y el intento y da una pista sin la respuesta (`safe_hint`, que descarta pistas que la filtran); Hope reintenta. La respuesta verificada y el reintento corregido quedan como datos. La verdad sale de imágenes procedurales o de preguntas confirmadas por dos tutores.
  - Pruebas con tutores simulados: `tests/test_vision_tutores.py`.

#### 10.1.5 Tutoría on-policy: Qwen como tutor completo (pedido del usuario; `hope/teacher/tutor.py`)

**Decisión del usuario:** un Qwen grande no solo destila (escribe explicaciones), sino que es **tutor completo** de Hope. La diferencia con §10.1.2 es que el tutor trabaja sobre lo que **Hope** produce: es destilación on-policy en el sentido de «Razonamiento por parámetro para Hope» (la señal se calcula sobre las muestras del alumno), pero con revisión, pistas y problemas a medida en lugar de log-probabilidades por token (Hope es de bytes y Qwen no; ver el informe).

**Ciclo de una ronda** (`TutorSession.round`):

1. **Diagnóstico** (primera ronda, `diagnose`):
   - problemas de cada tema: de los generadores verificables (§10.1.3) y propuestos por el tutor (verificados con herramientas);
   - si el modelo tiene visión, preguntas sobre imágenes procedurales (respuesta exacta por construcción; tema «imagen»);
   - Hope responde sin ayuda; el tutor clasifica cada fallo;
   - resultado: un **perfil** por tema con exactitud, tipos de error (conceptual, procedimiento, cálculo, herramienta, enunciado, formato), llamadas a herramientas y si funcionaron, notas del tutor, pistas que sirvieron y explicaciones necesarias. `Profile.strengths()` y `Profile.weaknesses()` lo resumen.
2. **Intento → crítica → pista → reintento** (`tutor_problem`):
   - Hope resuelve (con su contexto de la base: soluciones y lecciones parecidas);
   - el tutor revisa **el intento de Hope**, no su propia solución: recibe el razonamiento, la respuesta, la revisión automática de CRE y las herramientas que usó;
   - devuelve el error, su tipo y paso, una **pista**, la idea clave, el error a evitar y notas 0–1 (pasos, herramienta, eficiencia);
   - **la pista no puede contener la respuesta** (`leaks_answer`: el número exacto, también con coma decimal, o la frase). Si la contiene, se reemplaza por una genérica a partir del tipo y del paso del error (`safe_hint`);
   - Hope reintenta con la pista;
   - si vuelve a fallar, el tutor **explica completo** (episodio verificado, §10.1.2) y Hope **anota en su cuaderno** (`Notebook.write`): «<tema>: idea clave» y «<tema>: error a evitar». La lección queda también en la base (`KnowledgeBase.learn_lesson`), así que la próxima vez aparece en «Errores a evitar».
3. **Variaciones dirigidas** (`targeted_variations`): el tutor genera problemas nuevos que atacan exactamente el error encontrado. Los verificados entran al **repaso espaciado** (`ReviewQueue`: 1, 2, 4… días; fallar vuelve a 1 día). Hay un tope por tema.
4. **Evaluación del razonamiento y de las herramientas:** las notas del tutor alimentan la recompensa del refuerzo con `combined_reward`:
   - la verificación automática (`verified_reward`: 1, 0,2 o 0) **siempre manda**: si es 0, el tutor no puede rescatarla;
   - entre respuestas correctas, el promedio de las notas mueve la recompensa dentro de [v·(1 − w), v], con w = 0,3;
   - cada intento queda en `recompensas.jsonl` (texto, verificación, notas, recompensa).
5. **Preguntas de Hope** (`maybe_ask`):
   - después de cada intento se lee la **cabeza de incertidumbre** de Hope (pérdida que predice sobre su propia respuesta);
   - si supera `ask_threshold`, Hope le pregunta al tutor por la línea más dudosa;
   - la respuesta se verifica si trae un valor; si es refutada o filtra la respuesta final, no se usa;
   - si sirve, Hope reintenta con esa ayuda antes de la revisión;
   - todo queda en `preguntas.jsonl` y en PESAM (`pregunta_de_hope`).
6. **Currículo adaptativo** (`next_topics`):
   - prioridad = debilidad (1 − exactitud reciente) + repasos vencidos + estancamiento (tasa de mejora ≤ 0) − penalización por nivel (0,5 por cada nivel por encima del más bajo sin dominar);
   - un tema está **dominado** con exactitud ≥ 90% en las últimas 10 respuestas (mínimo 8) y sin repasos vencidos;
   - un tema dominado sale de la rotación hasta que le vence un repaso; si todo está dominado, la sesión termina.
7. **Datos para entrenar** (`TutorData`):
   - **intentos corregidos** con máscara de pérdida (`corrected_segments`): el enunciado y los pasos equivocados de Hope quedan como contexto con peso 0; los pasos correctos anteriores al error, la línea «Revisión: …» y el reintento correcto (o la explicación del tutor) tienen peso 1. Así Hope aprende a notar el error y corregirlo sin aprender el error;
   - **soluciones propias** correctas, **episodios** del tutor verificados y **lecciones** del cuaderno vencidas;
   - se escriben como `train.bin` + `train.mask` (y `val.*`). `hope.train.run` lee la máscara si existe (`Source.mask`, `Mixture.batch_weighted`, `HN1Model.loss(weights=…)`);
   - cada ronda se consolida en los pesos con repaso del corpus.
8. **SFT antes que RL** (MobileLLM-R1): con `rl: true`, después de consolidar corre GRPO (`hope/train/rlvr.py`) con la recompensa combinada, solo en temas que ya superan `rl_min_accuracy` (con exactitud cero los grupos no enseñan nada). El tutor solo puntúa las muestras correctas.

**Guardián y verificación del tutor:**

- todo el tráfico pasa por `Teacher.ask`: un tutor local es acción de nivel 1 y uno remoto requiere aprobación humana. Si se deniega, la sesión sigue con los generadores y lo registra (`tutor_denegado`);
- lo que dice el tutor se verifica con herramientas cuando se puede:
  - problemas y variaciones: `check_step` sobre su `verificacion`, y la respuesta tiene que coincidir con lo verificado;
  - explicaciones: `verify_episode`;
  - respuestas a preguntas: `check_step`;
  - revisiones: el veredicto de «correcto» lo decide la verificación automática, no el tutor;
- lo refutado se registra como `error_del_tutor` y **no entra al entrenamiento**. Los problemas sin verificación posible sirven para el diagnóstico, pero no se entrenan salvo `train_unverified`.

**Dónde corre:**

- **A770, Qwen grande** (`configs/tutor_a770.yaml`, Qwen3-14B Q4 ≈ 9 GB junto a Hope de 500M): etapa `3b_tutoria` de `configs/plan_a770_continuacion.yaml`, 72 h, con RL. Las etapas siguientes mezclan `tutor: data/tutor` (con máscara);
- **1060, Qwen chico, opcional** (`configs/tutor_1060.yaml`, Qwen3-1.7B ≈ 1,2 GB): etapa `2b_tutoria_opcional` de `configs/plan_gtx1060.yaml`. Se omite sola si no hay servidor;
- **a mano:** `python -m hope.teacher.tutor --resume <checkpoint> --teacher configs/tutor_a770.yaml --out runs/tutoria --rounds 20 [--rl]`. Es reanudable (`tutor_estado.json`: perfil, repaso, problemas vistos y datos).

**Pruebas** (`tests/test_tutor.py`, con `MockTutor`, que extiende `MockTeacher`, y hn1-tiny): ciclo completo, pista sin la respuesta (también con un tutor que la filtra), anotación en el cuaderno, variaciones dirigidas al repaso, errores del tutor descartados (problemas, explicaciones y revisiones equivocadas), currículo adaptativo, recompensa combinada, máscara de pérdida, RL después del SFT, guardián con tutor remoto y la etapa del plan.

**Pendiente:**

- medir con Qwen real cuánto mejora Hope por consulta frente al ciclo de §10.1.2;
- calibrar `ask_threshold` con la cabeza de incertidumbre entrenada;
- probar la destilación on-policy por prefill con alineación por fragmentos (ALM), del informe.

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


### 11.2 Hoja de ruta de rendimiento por parámetro (decisión del usuario, 2026-11-10)

Lista de optimizaciones aprobadas para probar. **Ninguna está medida en Hope todavía:** las ganancias son rangos tomados de la literatura o de nuestras mediciones a escala chica, y pueden no trasladarse. Cada una entra solo si pasa la ablación (§12.1): **2 semillas como mínimo**, el banco sellado y la misma cantidad de cómputo que la línea base. Las que no mejoran quedan en el código, apagadas por defecto, y se vuelven a probar en cada salto de tamaño.

La ganancia se expresa como **multiplicador de cómputo efectivo** (×): «1,2×» quiere decir que se llega a la misma pérdida con un 1/1,2 del cómputo. Esa es la unidad que permite sumar mejoras de distinto tipo.

#### Prioridad 0 — lo que el análisis del 2026-11-10 marcó como más importante

| # | Optimización | Por qué | Ganancia esperada | Estado |
|---|---|---|---|---|
| P0.1 | **Revivir el núcleo recurrente**: entrenar con una cantidad de iteraciones al azar en cada paso (1–8, como Huginn, Geiping et al. 2025) y agregar la parada adaptativa recién al final, con un mínimo | La parada ACT colapsó a 1 iteración en los dos checkpoints: hoy el núcleo no aporta | Si funciona, es la mayor palanca de cómputo sin parámetros (más cómputo en lo difícil); sin medir | pendiente |
| P0.2 | **Recuperación como pilar**: almacén kNN-LM permanente en la app (por idioma y código) y fragmentos recuperados como contexto durante el entrenamiento (estilo RETRO) | kNN-LM ya midió −16/17 % de BPB sin parámetros | La parte de entrenamiento está sin medir; a mayor escala el efecto del kNN suele achicarse | kNN hecho; RETRO pendiente |
| P0.3 | **Destilación por secuencias en lugar de probabilidades** | tokens+KL empeoró (2,706 contra 2,431 BPB) | Evita una pérdida medida; el tutor y el harvest ya generan secuencias verificadas | en uso; sacar el KL por tokens de los planes |
| P0.4 | **Podar lo que no rinde**: contrastiva, ejemplos de estilo, bucle estable y autodepuración, apagados por defecto | Empeoran o no ayudan a esta escala | Mediciones más limpias, menos errores | pendiente |
| P0.5 | **Medir cuántos parámetros «ahorra» cada herramienta**: exactitud con y sin la herramienta | Decide qué capacidad sacar de la red | — | pendiente |

#### Ola A — baratas, se miden en la ablación de CPU

| # | Optimización | Ganancia esperada | Cuando crezca | Estado |
|---|---|---|---|---|
| A1 | **Embeddings de n-gramas de bytes con hash** (BLT, Pagnoni et al. 2024): cada byte suma vectores de los n-gramas de 3 a 8 bytes que terminan en él, buscados en tablas con hash | 1,1–1,3× | se mantiene | pendiente |
| A2 | **Trucos del speedrun de nanoGPT**: ReLU² en el MLP, recorte suave de logits, QK-norm, conexiones tipo U-net entre bloques, value embeddings (Muon ya está) | 1,2–1,5× en conjunto | algunos se achican (las U-net); QK-norm y softcap dan sobre todo estabilidad, más útil al crecer | pendiente |
| A3 | **Promedio de pesos**: EMA durante el entrenamiento y promedio de checkpoints en el enfriamiento WSD | 1,02–1,1× | se mantiene | EMA existe en `hope/train/continual.py`; falta en `run.py` |
| A4 | **Lote que crece** (chico al principio, grande después) | 1,0–1,1× | se mantiene | pendiente |
| A5 | **Deduplicación MinHash** de todas las fuentes, no solo código y web diaria | 1,1–1,3× (depende de cuánto se repita cada fuente) | crece: con más datos hay más duplicados | código abierto y datos diarios ya deduplican; falta Wikipedia, maestros y la deduplicación cruzada entre fuentes |
| A6 | **Empaquetado con máscara por documento** (un documento no atiende al anterior en la misma secuencia) | 1,0–1,1× (más en datos cortos: herramientas, diálogos) | se mantiene | verificar qué hace hoy el cargador |

#### Ola B — necesitan diseño o corridas más largas

| # | Optimización | Ganancia esperada | Cuando crezca | Estado |
|---|---|---|---|---|
| B1 | **µP** (parametrización para transferir hiperparámetros): los ajustes se buscan en 25M y se transfieren a 200M, 500M y 1B | 1,0–1,3×; sobre todo evita un 200M mal ajustado | crece: Hope crece varias veces | pendiente; **antes de crecer a 500M** |
| B2 | **Mezcla de fuentes optimizada** con un modelo chico de prueba (estilo DoReMi), más repetir hasta ~4 épocas los datos mejores (Muennighoff et al. 2023) | 1,1–1,5× | se mantiene | los pesos de los planes hoy están puestos a mano |
| B3 | **Capas compartidas + LoRA por capa** (Relaxed Recursive Transformers, Bae et al. 2024) | más calidad por parámetro, no por cómputo: misma calidad con menos parámetros únicos | complementa a P0.1 | pendiente |
| B4 | **Mixture-of-Depths**: cada parche decide si pasa por un bloque o lo saltea | hasta 1,5× menos cómputo con la misma calidad | se mantiene | pendiente |

#### Ola C — generación y hardware real (GTX 1060 / A770)

| # | Optimización | Ganancia esperada | Estado |
|---|---|---|---|
| C1 | **Sin fp16 para cálculo en Pascal**: en la 1060 el fp16 corre a una fracción mínima de la velocidad de fp32. Verificar que `fp32-8bit` nunca caiga en operaciones fp16 | evita una caída grande de velocidad | verificar con `env_check` |
| C2 | **La GPU nunca espera**: datos preparados en el i5 con varios hilos, memoria fijada y lotes adelantados | 1,0–1,3× en tiempo de pared | verificar |
| C3 | **Turnos para el maestro**: si Qwen corre en la misma GPU, generar sus datos en bloques (por ejemplo, de noche) y no mezclarlos con el entrenamiento | evita la competencia por la GPU | integrar con `hope/app/turnos` |
| C4 | **CUDA graphs** (o XPU graphs en la A770) en la generación | 1,2–2× en generación (con modelos chicos domina el lanzamiento de operaciones) | pendiente |
| C5 | **Autoespeculación con salida temprana** (LayerSkip, Elhoushi et al. 2024): las primeras capas proponen y el modelo completo verifica | 1,3–2× en generación; se superpone con la especulativa por n-gramas | pendiente |

#### Ola D — después del entrenamiento

| # | Optimización | Ganancia esperada | Estado |
|---|---|---|---|
| D1 | **Mejor de N con verificador** (tests, calculadora, gramática) en el tiempo sobrante | grande en tareas verificables, una vez que pass@1 sea mayor que 0 | idea de «reanálisis» existente; falta el juez |
| D2 | **Fusionar modelos**: ajustar copias especializadas (código, lengua, herramientas) y promediar sus pesos | junta habilidades sin parámetros nuevos | pendiente |
| D3 | **RL con recompensas verificables más largo** (GRPO ya implementado) | de lo que más rinde en código y matemática, con un modelo que ya resuelva algo | pendiente |

#### Estimación sumada (a confirmar con mediciones)

Las ganancias **no se suman: se multiplican y se superponen**. Por ejemplo, la deduplicación y la mezcla de fuentes atacan en parte el mismo problema, y lo mismo pasa con los trucos de nanoGPT y µP.

| Grupo | Producto de las medianas | Con superposición (estimación) |
|---|---|---|
| Entrenamiento (A1–A6, B1, B2) | ≈ 1,15 · 1,3 · 1,05 · 1,05 · 1,15 · 1,05 · 1,1 · 1,2 ≈ 2,6× | **1,5–2,5×** de cómputo efectivo |
| Generación (C4, C5 sobre lo actual) | ≈ 1,5 · 1,6 ≈ 2,4× | **1,5–2×** más sobre el 2–5× actual |
| Hardware (C1–C3) | — | no suben el techo; evitan perder hasta 2× en la 1060 |
| Tareas (D1–D3, P0.1, P0.2) | — | no se puede estimar mientras pass@k = 0; es donde está el mayor potencial |

Traducido a tamaño: con un entrenamiento óptimo en cómputo, los parámetros escalan como C^0,5 (Hoffmann et al. 2022). Entonces **1,5–2,5× de cómputo efectivo equivale a un modelo ≈ 1,2–1,6× más grande**: un hn1-200m con todo esto se comportaría en texto como uno de ~250–320M entrenado sin estas mejoras. Si P0.1 (recurrencia) y P0.2 (recuperación) funcionan, el efecto sobre conocimiento y razonamiento se suma por fuera de esa cuenta.

**Cuando crezca:** las mejoras de datos (A5, B2), µP y el almacén de recuperación conservan o aumentan su valor. Los trucos de arquitectura a escala chica (A2) suelen achicarse. Lo que hoy da 0 a 25M (código, razonamiento visual, autodepuración, contrastiva) se vuelve a probar en cada salto: 200M, 500M y 1B.

**Orden de trabajo:** P0.4 y Ola A (CPU, 2 semillas) → P0.1 y B1 antes del 200M en la 1060 → C1–C3 el primer día en la GPU real → B2–B4 durante la etapa 1 → C4–C5 y Ola D con el 200M entrenado.

### 11.3 Investigar, verificar y revisar antes de responder (decisión del usuario, 2026-11-10)

Objetivo: que en texto Hope se comporte como una red más grande, **compensando con datos externos lo que no cabe en sus parámetros**. Busca como lo haría una persona: primero Wikipedia, después varias fuentes verificadas y seguras. Confirma los datos antes de afirmarlos y revisa varias veces lo que va a decir, todo dentro del presupuesto de velocidad de `BudgetedResponder`.

**Idea central:** a esta escala, los parámetros van al lenguaje y al razonamiento, y los **hechos se buscan**. En preguntas de conocimiento, la recuperación permite que modelos chicos se acerquen a otros mucho más grandes (Atlas, Izacard et al. 2022: 11B con recuperación igualó o superó a modelos de 540B en preguntas de dominio abierto). En redacción y razonamiento no compensa: ahí ayudan P0.1 (recurrencia) y la revisión.

#### Qué ya existe

- `hope/tools/web.py`: búsqueda (Wikipedia, arXiv, DuckDuckGo, SearXNG, Brave, CSE) y lectura de páginas, solo GET/HEAD, a través del guardián. Trayectorias consulta → resultados → páginas → conclusión. Neutralización de instrucciones inyectadas en las páginas.
- `⟦buscar_wiki⟧`: Wikipedia local sin red (`hope/memory/wiki_index.py`).
- `hope/data/web_episodes.py`: episodios de entrenamiento de buscar → leer → citar, contradicciones entre fuentes, acceso denegado sin inventar fuentes.
- `BudgetedResponder` (`hope/runtime/budget.py`): borrador rápido y escalado en calidad con el tiempo sobrante, crítico CRE/HCRI y consenso.
- KnowledgeBase, Notebook y KnowledgeGraph (`hope/memory/`), para guardar lo aprendido.

#### Qué falta (orden de implementación)

| # | Pieza | Qué hace | Estado |
|---|---|---|---|
| V1 | **Fuentes por nivel de confianza** (`configs/web.yaml`) | Nivel 1: Wikipedia local y fuentes primarias (organismos oficiales, documentación, arXiv, CORE). Nivel 2: enciclopedias y medios de referencia. Nivel 3: el resto, solo como pista. El guardián sigue decidiendo el acceso; el nivel solo pesa en la verificación | pendiente |
| V2 | **Extraer afirmaciones verificables** del borrador | Separa las frases con datos comprobables (fechas, cifras, nombres, relaciones) de las opiniones y del razonamiento | pendiente |
| V3 | **Buscar como una persona, del más barato al más caro** | 1) memoria propia (KnowledgeBase y grafo, ya verificados); 2) Wikipedia local; 3) búsqueda web en fuentes de nivel 1 y 2; 4) seguir enlaces dentro de sitios permitidos. Corta apenas hay confirmación suficiente | parcial (las herramientas existen; falta la estrategia) |
| V4 | **Verificación cruzada** | Un dato se marca *confirmado* con 2 fuentes independientes, o con 1 de nivel 1. Si las fuentes no coinciden, se informan las dos con su procedencia. Si no se encuentra, se dice «no verificado»: nunca se inventa una fuente (Carta art. 4) | parcial (los episodios de contradicción existen) |
| V5 | **Revisar antes de decir** (Chain-of-Verification, Dhuliawala et al. 2023) | Borrador → preguntas de verificación para cada afirmación → responderlas **sin mirar el borrador** (con fuentes) → corregir el borrador → crítico CRE/HCRI → repetir mientras haya tiempo y queden afirmaciones sin confirmar | pendiente; se integra como rondas de `BudgetedResponder` |
| V6 | **Citas en la respuesta** | Cada dato confirmado lleva su fuente (título y URL); lo no verificado se marca | parcial |
| V7 | **Caché de verificados** | Lo confirmado va a KnowledgeBase y al grafo, con fuente, fecha y nivel. La próxima vez no se busca, y el Notebook lo repasa (Leitner). Los datos que cambian (cargos, precios, versiones) vencen y se vuelven a verificar | pendiente |
| V8 | **Entrenarlo, no solo armarlo** | Episodios V2–V6 generados con Qwen como maestro y verificados contra las fuentes: Hope aprende *cuándo* buscar (incertidumbre alta o dato concreto), *qué* buscar y a citar. Recompensa en GRPO: exactitud verificada − costo de búsquedas (recompensa con costo, ya implementada) | parcial (curriculum de herramientas y `web_episodes` existen) |

#### Cuidado: el modelo chico no es buen juez de sí mismo

La autocrítica pura (que el modelo relea y «se dé cuenta») rinde poco en modelos chicos: tienden a confirmar su propio error. Por eso la revisión de V5 se apoya en **verificadores externos**: fuentes, calculadora, unidades, tests, gramática y CRE/HCRI. La opinión del propio modelo es la última señal, no la primera.

#### Eficiencia

- **Dificultad adaptativa:** se busca solo si hay datos concretos en juego o si la incertidumbre del modelo es alta. Las preguntas de razonamiento o charla no disparan búsquedas.
- **Del más barato al más caro (V3):** la memoria propia y la Wikipedia local cuestan milisegundos; la web, segundos y cuota diaria.
- **Caché (V7):** con el uso, la mayoría de las verificaciones salen de la memoria propia.
- **Presupuesto:** todo corre dentro de `BudgetedResponder`; si no alcanza el tiempo, la respuesta sale con los datos no confirmados marcados, nunca tarde ni inventada.

#### Ganancia esperada (a medir)

- **Preguntas de conocimiento con fuentes disponibles:** la mejora más grande; puede acercarse a modelos varias veces más grandes, porque el dato viene de la fuente y no de los parámetros.
- **Alucinaciones:** CoVe redujo claramente los datos inventados en listas y respuestas largas; con fuentes externas el efecto debería ser mayor.
- **Redacción y razonamiento:** poca ganancia por esta vía; dependen de P0.1 y de la Ola A.

**Medición:** un banco sellado de preguntas factuales en es/en con respuesta verificable, en cuatro modos: sin herramientas, solo Wikipedia local, web, y web + revisión (V5). Medir exactitud, % de afirmaciones con cita correcta, % de alucinaciones y tiempo por respuesta.

### 11.4 Equipo de 6 agentes: maestro, verificador y 4 especialistas (decisión del usuario, 2026-11-10)

Objetivo: usar el excedente de velocidad (§11.3; ≈ 10× en la 1060 y ≈ 30× en la A770 con 4 bits y especulativa) para que **varios roles trabajen sobre la misma respuesta**, en lugar de un solo pase. Amplía el consejo actual (§4.2.2), que tiene 4 personalidades (innovador, conservador, optimizado, veloz) y un coordinador–verificador fijo.

**Sigue siendo un solo modelo en una sola GPU:** cada agente es un adaptador LoRA (rango 4–8, pocos MB) sobre el modelo compartido (`attach_named_adapters`, `use_adapter`). No se duplican redes (Language Core §7).

#### Los 6 agentes

| Agente | Rol | Herramientas y datos propios | Se entrena con |
|---|---|---|---|
| **Maestro** (director) | Entiende el pedido, decide si es simple o complejo, lo divide en subtareas, elige especialistas y presupuesto, integra la respuesta final y aprende qué funciona en cada tema (`AgentUsageMemory`, UCB1) | memoria, KnowledgeGraph, `BudgetedResponder` | episodios de planificación y reparto generados por Qwen y verificados; premio por resultado (bandido) |
| **Verificador** | Revisa todo lo que van a decir los demás antes de que salga. Nunca produce la respuesta que revisa | fuentes (§11.3 V3–V4), calculadora, unidades, tests en sandbox, gramática, CRE/HCRI, Chain-of-Verification | detección de errores: mutaciones (la sesión de lenguaje ya mide 90 % es / 78 % en), soluciones erróneas de Hope y juicios de Qwen confirmados por herramientas |
| **Lenguaje** | Redacción es/en, resúmenes, traducción, estilo, antirrepetición | herramientas de lengua (19), MBR, bucle de escritura | Wikipedia, escritura, ejercicios de lengua |
| **Código** | Escribir, leer, depurar y editar código | gramática, FIM, editor, ejecución en sandbox, autodepuración | corpus de código abierto, trazas, FIM, RL con tests |
| **Investigador** | Buscar, leer y confirmar datos como una persona (§11.3) | Wikipedia local, web solo lectura vía guardián, citas, caché de verificados | `web_episodes`, trayectorias reales |
| **Razonamiento** | Matemática, lógica, ciencia, planificación con números | calculadora, unidades, z3, HCRI, más iteraciones del núcleo (P0.1) | currículo de herramientas, GRPO con recompensa verificable |

Las 4 **personalidades** actuales no se pierden: pasan a ser **modos** (temperatura, revisiones, iteraciones del núcleo, largo) que el maestro aplica a cualquier especialista. Por ejemplo, «código + innovador» para probar otro enfoque cuando los tests fallan. La visión no es un agente: es una zona del modelo que cualquier agente usa cuando hay imagen.

#### Cómo trabajan juntos

1. **Maestro:** clasifica el pedido.
   - **Simple** (charla, dato conocido): responde un solo especialista y el verificador hace un control liviano. Es el camino rápido; la mayoría de los pedidos debería ir por acá.
   - **Complejo:** el maestro lo divide en subtareas y asigna especialista, modo y presupuesto a cada una.
2. **Especialistas:** resuelven sus subtareas. Si son independientes, corren **en el mismo lote** (`hope/core/batched.py`): en una sola GPU, el lote es la forma de usar la velocidad en paralelo.
3. **Verificador:** revisa cada resultado con herramientas y fuentes. Devuelve un veredicto (verificado, sospechoso o refutado) con la falla concreta.
4. **Corrección:** lo que falla vuelve al especialista con la crítica, mientras haya tiempo. Si un especialista falla dos veces, el maestro prueba otro modo u otro especialista.
5. **Maestro:** integra, y el verificador hace un último control de coherencia de la respuesta completa (que las partes no se contradigan).
6. **Aprendizaje:** se registra qué especialista y qué modo funcionaron en cada tema, y los episodios verificados vuelven como datos de entrenamiento de cada adaptador.

**Reparto orientativo del presupuesto** en un pedido complejo: maestro ~10 %, especialistas ~40 %, verificador ~35 %, integración ~15 %. Si el tiempo no alcanza, sale la mejor respuesta verificada hasta ese momento, con lo no confirmado marcado (algoritmo *anytime*, como `BudgetedResponder`).

#### Reglas que no cambian

- **Ningún agente se aprueba a sí mismo.** El verificador nunca revisa algo que escribió él, y su veredicto sale de herramientas y fuentes antes que de su propia opinión: un modelo chico es mal juez de sí mismo (§11.3).
- Todo acceso a red, sandbox o herramientas pasa por el guardián, sea cual sea el agente. Ningún agente puede cambiar límites ni configuración de seguridad.
- Qwen sigue siendo el maestro externo durante el entrenamiento. El agente maestro de Hope solo puede consultarlo por la herramienta `ask_teacher`, con las aprobaciones que ya exige la Carta: un maestro local está permitido, uno remoto requiere aprobación humana.

#### Implementación (sobre lo que ya existe)

| # | Paso | Archivos |
|---|---|---|
| E1 | Perfiles por área (4 especialistas + maestro + verificador) y modos separados de las áreas | `hope/council/agents.py` |
| E2 | Maestro: clasificar simple/complejo, dividir, asignar y presupuestar | `hope/council/coordinator.py` → nuevo `hope/council/maestro.py` |
| E3 | Verificador como agente con adaptador propio, que usa CRE/HCRI, fuentes y tests | `hope/council/verificador.py`, `hope/constraints/critic.py`, §11.3 V4–V5 |
| E4 | Subtareas en lote | `hope/core/batched.py` |
| E5 | Integración con el presupuesto de velocidad | `hope/runtime/budget.py` |
| E6 | Datos y entrenamiento por adaptador; el verificador primero, porque es el que más rinde | `hope/train/`, `hope/teacher/` |
| E7 | Ver en la app qué agente trabajó y qué verificó (pestaña de charla) | `hope/app/` |

#### Ganancia esperada y cómo medirla

- **El verificador con herramientas es la pieza que más rinde:** convierte velocidad en exactitud (mejor de N, CoVe, tests). Va primero.
- **El reparto por especialistas** rinde en pedidos mixtos (por ejemplo, investigar un dato y escribir código con él). En pedidos simples no debería agregar costo: por eso existe el camino rápido.
- **Los adaptadores por área a 200M** probablemente dan una mejora moderada: las zonas MoE ya especializan por dentro. Hay que medir que cada adaptador le gane al modelo base en su área; si no gana, ese especialista usa el modelo base con sus herramientas y su modo.

**Comparación obligatoria, a igual tiempo de respuesta:** 1) un agente solo, 2) el consejo actual de 4 personalidades, 3) el equipo de 6. Banco sellado mixto (lengua, código, datos, razonamiento). Medir exactitud, alucinaciones, tiempo y la frecuencia con que el maestro elige bien. El equipo entra en la app solo si le gana al agente solo con el mismo tiempo.

### 11.5 Razonamiento, lenguaje y «hablar como un modelo más grande» (decisión del usuario, 2026-11-10)

Complementa §11.2–11.4: esto **no** repite la recurrencia (P0.1), el kNN (P0.2), la verificación (§11.3) ni el equipo de agentes (§11.4).

**Diagnóstico:** lo que hace que un modelo «suene grande» se aprende sobre todo **después** del preentrenamiento: ajuste con conversaciones de calidad, preferencias, calibración y formato. Modelos chicos con buen postentrenamiento conversan mucho mejor que su versión base del mismo tamaño. Hope todavía no tiene esta etapa: el tutor y GRPO enseñan tareas, no cómo conversar.

#### L — Lenguaje y conversación

| # | Qué | Por qué | Estado |
|---|---|---|---|
| L1 | **Datos «de libro» sintéticos** generados por Qwen (y los otros maestros por turnos): explicaciones claras, diálogos, cuentos y textos didácticos en es/en, con temas y estilos variados. Filtrados por calidad y deduplicados | Los modelos chicos aprenden lenguaje fluido mucho más rápido con texto limpio y didáctico (TinyStories, Eldan y Li 2023; Phi-1, Gunasekar et al. 2023) | pendiente; reutiliza `harvest` y los turnos de maestros |
| L2 | **SFT conversacional**: miles de conversaciones de varios turnos generadas por los maestros y revisadas, con español rioplatense y neutro, preguntas de seguimiento, pedidos ambiguos y correcciones del usuario | Es lo que convierte un modelo base en un interlocutor | pendiente |
| L3 | **Optimización por preferencias** (DPO, Rafailov et al. 2023, o SimPO): para cada pregunta, dos respuestas de Hope; el maestro y los verificadores eligen la mejor (claridad, exactitud, tono, largo justo). Los datos verificables se juzgan con herramientas, no con opinión | Mejora mucho la calidad percibida por poco cómputo; no hace falta un modelo de recompensa aparte | pendiente |
| L4 | **Planificar antes de escribir**: un esquema oculto (ideas en orden) y después el texto. En respuestas largas, el resumen continuo (`escritura.py`) mantiene el hilo | Los modelos chicos pierden coherencia en textos largos: el plan la sostiene | resumen continuo hecho; plan pendiente |
| L5 | **Pase de pulido** por el especialista de lenguaje (§11.4): concordancia (spaCy), repeticiones, conectores y registro, usando el excedente de velocidad | Es barato y saca los errores que más delatan a un modelo chico | herramientas de lengua hechas; falta el pase |
| L6 | **Largo justo y formato**: respuestas breves cuando alcanza, listas y tablas cuando ayudan; se entrena con L2–L3 | Un modelo chico se nota más cuando se extiende de más | pendiente |
| L7 | **Calibración**: decir «no lo sé», «no lo pude verificar» o pedir una aclaración en lugar de inventar | Una respuesta honesta parece más inteligente que una inventada; se entrena con L3 y §11.3 | parcial (episodios de acceso denegado) |
| L8 | **Muestreo min-p** (Nguyen et al. 2024) en lugar de temperatura sola o top-p | Texto más variado sin perder coherencia; ayuda más en modelos chicos | pendiente (MBR y antirrepetición ya están) |
| L9 | **Memoria del usuario**: preferencias, nombre, temas en curso y lo ya explicado (Notebook y grafo) | Recordar la conversación da sensación de un modelo mucho más capaz | parcial (memorias existen; falta usarlas en la charla) |
| L10 | **Persona estable**: tono y forma de hablar consistentes (adaptador o datos L2), sin perder la honestidad sobre lo que es | Coherencia de voz entre respuestas | pendiente |

#### R — Razonamiento

| # | Qué | Por qué | Estado |
|---|---|---|---|
| R1 | **Razonamientos destilados, cortos y verificados**: cadenas del maestro (Qwen3 con pensamiento) que solo entran si la respuesta final se verifica; mezclar cadenas cortas y largas | Los modelos chicos aprenden peor de cadenas muy largas de un maestro fuerte; rinde mejor una mezcla con cadenas cortas (Li et al. 2025) | parcial (tutor y harvest generan; falta el filtro por largo y verificación final) |
| R2 | **Razonar con programas** (PAL, Gao et al. 2022): para cuentas y lógica, escribir un pequeño programa o una llamada a ⟦calc⟧ y ejecutarlo en el sandbox | Saca la aritmética de los parámetros; es de lo que más rinde en modelos chicos | parcial (calc y sandbox existen; falta entrenarlo como hábito) |
| R3 | **Tareas procedurales infinitas** con respuesta exacta: aritmética, lógica, ordenar, contar, deducción, problemas de texto generados por plantilla, de fácil a difícil | Datos verificables gratis y sin licencia; base para GRPO | parcial (currículo de herramientas) |
| R4 | **Descomponer de menor a mayor** (least-to-most, Zhou et al. 2023): partir el problema en subproblemas y resolverlos en orden | Encaja con el maestro de §11.4 | pendiente |
| R5 | **Verificación por paso**: CRE/HCRI revisa cada paso del razonamiento, no solo el resultado, y premia en GRPO los pasos correctos | Evita aciertos por casualidad y razonamientos rotos que llegan bien | parcial (CRE revisa pasos; falta el premio por paso) |
| R6 | **STaR**: el modelo genera razonamientos, se quedan los que llegan a la respuesta correcta y se reentrena; si falla, se le da la respuesta y se le pide que la justifique | Mejora sola con cada vuelta, sin un maestro externo | hecho (`hope/train/star.py`); falta incorporarlo al plan |
| R7 | **GRPO más largo** sobre R1–R5, con recompensa con costo | Es la etapa que más mejoró el razonamiento en modelos recientes | hecho (`rlvr.py`); falta ejecutarlo a escala |

#### Orden propuesto

Esto va **después** del preentrenamiento base, como etapa de postentrenamiento en la 1060 y la A770:

1. L1 y R3: datos desde ya, en paralelo con el preentrenamiento (la etapa 2 del plan de la 1060 ya trae al maestro).
2. L2 y R1: SFT conversacional y de razonamiento.
3. L3 (DPO) y R7 (GRPO), alternados.
4. L4–L10, R2, R4 y R5 en el runtime, con el excedente de velocidad y el equipo de §11.4.

#### Ganancia esperada (a medir)

- **Calidad percibida en conversación:** es la mejora más grande y más visible. El postentrenamiento (L2–L3) es lo que más cambia cómo «suena» un modelo de este tamaño.
- **Razonamiento:** R2 y R3 dan las mejoras más seguras en cuentas y lógica. R1 y R7 mejoran el razonamiento general, en forma moderada a esta escala.
- **Límite honesto:** esto hace que Hope **hable** como un modelo más grande y razone mejor en lo que puede verificar. No le da el conocimiento ni la profundidad de razonamiento de un modelo 10× más grande. El conocimiento se compensa con §11.3 y la profundidad, si funciona, con P0.1.

**Medición:**
- Comparaciones a ciegas: Qwen y otro maestro como jueces, más el usuario, entre Hope antes y después de L2–L3.
- Banco sellado de conversación en es/en.
- Frecuencia de errores gramaticales y repeticiones.
- Exactitud en R3, más un banco de razonamiento verbal, con y sin R2.

### 11.6 Código (K): cómo mejorarlo y qué esperar por tamaño (decisión del usuario, 2026-11-10)

**Punto de partida medido** (`docs/resultados/codigo.md`, hn1-25m, 40 problemas propios estilo HumanEval): pass@1 = 0 % en todos los modos. La gramática sube la sintaxis válida del 31 % al 99,5 % (con cuerpo, del 26 % al 62 %). La autodepuración no ayuda todavía. El ajuste con trazas + FIM bajó el BPB en trazas de 3,66 a 0,23, pero no logró predicciones exactas.

**Ya existe:** corpus de código abierto (licencias permisivas, MinHash), repos del usuario (sin coautoría de IA), trazas de ejecución, FIM, gramática Python/JSON, editor y buscar/reemplazar, autodepuración, sandbox, banco con pass@k y GRPO con tests.

#### Qué agregar

| # | Qué | Por qué | Estado |
|---|---|---|---|
| K1 | **Código «de libro» y ejercicios sintéticos**: Qwen genera explicaciones de conceptos con código y ejercicios con solución, en es/en | Phi-1 (Gunasekar et al. 2023) llegó a 50,6 % en HumanEval con 1,3B y ~7B tokens de datos curados; su versión de 350M, a 45 %. La calidad de datos es la palanca n.º 1 en código chico | pendiente; reutiliza `harvest` |
| K2 | **Problemas verificados por ejecución** (estilo OSS-Instruct, Wei et al. 2023): Qwen escribe problema + tests + solución a partir de fragmentos reales del corpus; solo entran los que pasan en el sandbox | Datos infinitos, correctos y variados; también sirven de base para GRPO | parcial (sandbox y banco existen) |
| K3 | **Filtro educativo del corpus**: clasificador de calidad entrenado con juicios del maestro (estilo Stack-Edu de SmolLM2); se priorizan el código claro, documentado y didáctico | El código de GitHub es muy desparejo; a 200M sobra volumen y falta calidad | parcial (clasificador de calidad general existe) |
| K4 | **Foco en Python primero** (más JS/TS, bash, SQL y JSON de apoyo) | A 200M no alcanza la capacidad para muchos lenguajes; ampliar al crecer | ajustar pesos de los planes |
| K5 | **Tests primero y reordenar con tests** (CodeT, Chen et al. 2022): Hope genera tests, después N soluciones, y elige la que pasa más tests acordados | Convierte el excedente de velocidad en exactitud, incluso sin tests del usuario | pendiente; va en el verificador (§11.4) |
| K6 | **Linters y tipos como verificadores**: ruff, mypy o pyright y el compilador en el ciclo de corrección, además de la ejecución | Errores baratos de detectar antes de correr | pendiente |
| K7 | **Contexto de repositorio**: recuperar los archivos y funciones relevantes (índice de símbolos, kNN sobre el repo) antes de escribir | Las tareas reales dependen del resto del proyecto; usa §11.3 y el harness F9 | parcial (harness y contexto largo) |
| K8 | **Ediciones como diffs**: entrenar a responder con buscar/reemplazar o diffs y no reescribir archivos enteros | Menos bytes generados, menos errores, más velocidad | parcial (herramientas de editor hechas; falta el entrenamiento) |
| K9 | **GRPO con tests a escala** (`rlvr.py`), con premio por pasar tests y castigo por costo; currículo de fácil a difícil | Es lo que más mejoró el código en modelos recientes, una vez que pass@1 > 0 | hecho; falta correrlo |
| K10 | **Trazas + FIM sobre un checkpoint más grande**: se vuelven a medir en el 200M | A 25M no alcanzaron predicciones exactas | pendiente (sesión de lenguaje) |
| K11 | **Explicar y revisar código**, no solo escribirlo: leer código ajeno, encontrar el error y explicarlo en es/en | Es la mitad del uso real y usa el lenguaje que ya se entrena | pendiente; datos con K1–K2 |

**Orden:** K3–K4 (datos, desde ya) → K1–K2 (generación con los maestros, en paralelo con el preentrenamiento) → K5–K6 en el verificador → K9 cuando pass@1 > 0 → K7, K8 y K11 con el harness.

#### Qué resultados esperar por tamaño (estimación)

**Referencias reales** en HumanEval (Python, pass@1):
- Codex (Chen et al. 2021): 12M 2,0 %, 85M 8,2 %, 300M 13,2 %, 679M 16,2 %, 2,5B 21,4 %, 12B 28,8 %.
- Phi-1: 350M 45 %, 1,3B 50,6 %.

Los modelos de código actuales de 0,5–1,5B, entrenados con billones de tokens y buen postentrenamiento, están muy por encima de Codex a igual tamaño.

**Datos de código que vería Hope** (`hope.bench.plan`, hn1-200m, ~33 % de código en los planes; los TFLOPS de la 1060 son una suposición):
- 1060, 1 semana: ≈ 2B tokens equivalentes en total (≈ 0,7B de código).
- 1060, 2 semanas: ≈ 4B en total.
- A770, 2 meses: ≈ 56B en total (≈ 15B de código).

| Etapa | pass@1 estimado (HumanEval o similar) | Se compara con | Con mejor de N + tests en tiempo de respuesta (K5) |
|---|---|---|---|
| hn1-25m, CPU (medido) | 0 % | menos que Codex 12M | 0 % |
| hn1-200m, 1060, 1–2 semanas | 2–8 % | Codex 85M–300M | 5–15 % |
| hn1-200m, A770, 2 meses + K1–K3 + K9 | 10–25 % | Codex 300M–12B | 20–40 % |
| hn1-500m, A770, 6 meses + todo §11.6 | 20–40 % | Codex 12B; por debajo de Phi-1 si los datos no llegan a su calidad | 35–55 % |
| hn1-1b (si se autoriza) | 30–50 % | Phi-1 | 45–65 % |

**Cómo leer la tabla:**
- Son estimaciones, no mediciones: el rango es amplio porque todo depende de la calidad de los datos (K1–K3). Con datos comunes de GitHub, Hope quedaría en la mitad baja de cada rango; con datos al estilo Phi-1, en la alta.
- El «mejor de N» usa el excedente de velocidad (§11.3): a igual tiempo de respuesta, un 200M con tests puede rendir como un modelo varias veces mayor **en funciones aisladas y verificables**.
- **Límite:** en tareas de repositorio completo (varios archivos, contexto largo, estilo SWE-bench), a estos tamaños el resultado esperado es cercano a 0 %, con o sin herramientas. El harness F9 sirve para tareas acotadas: una función, un script, un error concreto.
- **Dónde se satura:** hasta ~500M, el cuello de botella son los datos y la verificación. A partir de ~1B empieza a pesar el tamaño: comprensión de proyectos grandes, APIs poco comunes y lenguajes poco representados.

**Medición:**
- El banco propio (`hope/bench/codigo.py`) en cada checkpoint.
- HumanEval y MBPP solo para evaluar, nunca para entrenar: hay que verificar que no se filtren en el corpus.
- Errores de sintaxis, tiempo por problema y aciertos con y sin K5.

### 11.7 Hoja de ruta integrada (§11.2–11.6 por etapa de hardware)

| Etapa | Entrenamiento | Datos que se generan en paralelo | Runtime y app |
|---|---|---|---|
| **Ahora, CPU** (ablaciones con 2 semillas) | P0.4, Ola A (A1–A6), P0.1 (recurrencia), B1 (µP) | — | §11.3 V1–V7, §11.4 E1–E5, min-p (L8) |
| **1060, etapa 1 (1–2 semanas, 200M)** | Preentrenamiento con la mejor configuración de la ablación, deduplicación (A5), K3–K4 en los pesos | L1, R3, K1, K2 con los maestros por turnos (§11.2 C3), de noche | §11.2 C1–C3 (hardware); medir TFLOPS reales el primer día |
| **1060, etapa 2 (maestro + herramientas)** | Episodios de herramientas, web (§11.3 V8), R1, K10 | sigue L1, R3, K1, K2 | banco factual y banco de código en cada checkpoint |
| **A770, etapa 1 (texto y código, 200M)** | Más datos con B2 (mezcla optimizada), B3–B4 si ganaron | L2 (conversaciones), R1 | equipo de 6 agentes en la app (E7) |
| **A770, crecer a 500M** (aprobación humana) | Crecimiento en profundidad primero; µP | — | — |
| **A770, postentrenamiento** | L2 (SFT) → L3 (DPO) y R7/K9 (GRPO) alternados; QAT 4/8 bits en el enfriamiento | datos de preferencia del equipo de agentes | Ola C (CUDA graphs, LayerSkip), Ola D, L4–L10 |
| **Cada salto de tamaño** | Volver a probar lo que dio 0 o empeoró: contrastiva, autodepuración, razonamiento visual, ACT | — | comparación obligatoria a igual tiempo: agente solo contra el equipo |

**Regla:** nada pasa de etapa por estar implementado, sino por ganar en el banco sellado con 2 semillas y el mismo cómputo (§11, §12.1).

### 11.8 Libros abiertos: leerlos, razonarlos y aplicarlos (decisión del usuario, 2026-11-10)

Objetivo: que Hope aprenda de libros de ciencia, programación y otras áreas cuya licencia lo permite, y que los **lea como una persona** (capítulo por capítulo, con notas, ejercicios y repaso), no solo como bytes de preentrenamiento.

#### Regla de licencias (sin cambios)

- **Entrenar:** solo dominio público, CC0, CC BY, CC BY-SA, PSF y licencias de software permisivas (MIT, Apache-2.0), libro por libro (`licencias_entrenamiento`, `license_allows`).
- **NC y ND:** nunca entran al entrenamiento. Se pueden **leer y citar en tiempo de respuesta** (§11.3) si el sitio lo permite.
- **Gratis para leer no es lo mismo que libre para entrenar.** Muchos libros de programación conocidos se leen gratis en la web, pero tienen licencia NC o todos los derechos reservados: se tratan como solo lectura.
- **Nunca:** bibliotecas piratas o «sombra» (Library Genesis, Z-Library, Anna's Archive y similares), PDFs subidos sin permiso ni préstamo digital de obras con derechos. Los libros sin licencia clara no entran.

#### Fuentes (`configs/fuentes_web.yaml`)

| Fuente | Qué tiene | Licencia | Acceso | Estado |
|---|---|---|---|---|
| Wikilibros es / en | Libros de texto colaborativos: programación, matemática, ciencias, idiomas | CC BY-SA 4.0 | API MediaWiki | es activo; **en agregado y activo** |
| Wikisource es / en | Obras clásicas, muchas de dominio público | CC BY-SA / dominio público | API MediaWiki | en agregado (inactivo, como es) |
| Project Gutenberg | Libros de dominio público, incluidos clásicos de ciencia y matemática | dominio público (EE. UU.) | mirror oficial (no el sitio principal) | ya estaba (inactivo) |
| **Standard Ebooks** | Ediciones cuidadas de libros de dominio público | CC0 + dominio público | repos git | agregado; falta lector |
| **OpenStax** | Libros de texto universitarios de física, química, biología, cálculo, estadística y computación; varios en español | mayoría CC BY 4.0; algunos BY-NC-SA (solo lectura) | fuentes publicadas | agregado; falta lector |
| **DOAB** y **OAPEN** | Libros académicos de acceso abierto de todas las áreas | por libro (muchos CC BY) | OAI-PMH | agregado; falta lector |
| **PubMed Central OA** | Artículos de biomedicina | subconjunto de uso comercial (CC0, CC BY) | descarga masiva | agregado; falta lector |
| **SciELO** | Ciencia en español y portugués | por revista o artículo (muchas CC BY) | OAI-PMH | agregado; falta lector |
| arXiv, CORE | Artículos científicos | por artículo | API | ya estaban |
| Documentación de Python, MDN | Documentación de programación | PSF / CC BY-SA 2.5 | páginas | ya estaban (inactivas) |
| Libros de código con licencia libre | Por ejemplo: SICP (CC BY-SA 4.0), el libro de Rust (MIT/Apache-2.0) | según cada libro | repos o páginas | a curar uno por uno (lista `libros_codigo`) |
| LibreTexts, Open Textbook Library | Muchos libros de texto | **mayoría NC** → solo lectura; algunos CC BY | por página | sin agregar: filtrar libro por libro |

**Para activar una fuente:** sus dominios tienen que estar en `configs/web.yaml` (`sitios_permitidos`) o aprobarse acceso por acceso. Esa decisión es del usuario, no de Hope.

#### Lectura activa: cómo «lee» Hope un libro

| # | Paso | Qué produce | Usa |
|---|---|---|---|
| B1 | **Leer en orden**: índice, capítulo por capítulo, con resumen continuo | texto limpio con capítulo, sección y licencia | `web_sources`, `escritura.py` (resumen continuo) |
| B2 | **Tomar notas**: definiciones, fórmulas, ideas clave y relaciones entre conceptos | notas en Notebook (repaso Leitner) y nodos en KnowledgeGraph, con cita al libro | `hope/memory/` |
| B3 | **Preguntarse y responder**: el maestro arma preguntas de comprensión por sección y Hope responde con el libro a la vista; las respuestas se verifican contra el texto | pares pregunta–respuesta con fuente | maestros por turnos, verificador §11.4 |
| B4 | **Hacer los ejercicios**: muchos libros de texto (OpenStax, por ejemplo) traen ejercicios con respuesta. Las respuestas numéricas y el código se verifican con herramientas | datos de razonamiento **verificables** (se suman a R3 y K2) | calculadora, unidades, sandbox, CRE |
| B5 | **Aplicarlo**: problemas nuevos que combinan varios capítulos o libros (por ejemplo, física + programación) | ejercicios de transferencia | maestro + verificador |
| B6 | **Repasar**: el Notebook vuelve sobre lo que Hope falla, con preguntas nuevas sobre el mismo tema | repetición espaciada | Notebook |
| B7 | **Consultar y citar** en tiempo de respuesta: ⟦buscar_libro⟧ busca en el índice local de libros (incluidos los de solo lectura) y cita capítulo y sección | respuestas con fuente | §11.3 V3, V6 |

**Qué entra al entrenamiento:** el texto de los libros con licencia permitida (B1) y los datos derivados verificados (B3–B5), con la licencia del libro de origen registrada. De los libros de solo lectura no se entrena nada, ni el texto ni lo derivado: solo se consultan y citan (B7).

#### Implementación

1. Lectores nuevos en `hope/data/web_sources.py`: `openstax`, `git_libros`, `oai_pmh` y `pmc_oa`.
2. Índice local de libros, con capítulo, sección y licencia, y la herramienta ⟦buscar_libro⟧.
3. Ciclo de lectura activa B1–B6 (nuevo `hope/data/libros.py`), en los turnos nocturnos de los maestros.
4. Pesos en los planes: libros de texto y ejercicios verificados dentro de L1, R3 y K1.

#### Ganancia esperada

- **Calidad de datos:** los libros de texto son exactamente el tipo de dato «de libro» que más rinde en modelos chicos (Phi-1, TinyStories; §11.5 L1, §11.6 K1).
- **Ejercicios con respuesta:** es la mejor fuente de razonamiento verificable gratis que hay. Se suman a R3 y K2.
- **Conocimiento aplicable:** las notas, el grafo y la cita a capítulos convierten lo leído en algo que Hope puede **consultar** cuando responde, no solo algo que recuerda a medias.

**Medición:** antes y después de leer un libro, comparar ejercicios del libro que no se usaron para entrenar y preguntas del banco factual sobre su tema.

#### 11.8.1 Libros comunes con Qwen como tutor de lectura (decisión del usuario, 2026-11-10)

Además de los libros de texto, Hope lee **libros comunes**: novelas, cuentos, ensayos, historia y divulgación. Toma notas, vuelve a contar la historia con sus palabras, y Qwen (u otro maestro de los turnos) revisa sus notas y le explica **dónde malinterpretó algo**. Es el mismo método que usa el usuario para estudiar y es exactamente lo que hace `hope/teacher/tutor.py` (tutoría *on-policy*): el maestro corrige lo que Hope produjo, no un tema genérico. Por eso cada consulta rinde más en una red chica (destilación on-policy, Agarwal et al. 2024): ataca los errores reales de Hope.

**Qué libros:** los mismos criterios de §11.8.
- **Literatura de dominio público:** cubre una parte enorme de los libros «comunes» (Gutenberg, Standard Ebooks y Wikisource en es/en): Cervantes, Galdós, Darío, Quiroga, Lugones, Arlt, Austen, Dickens, Verne, Conan Doyle, Twain, Tolstói o Chéjov en traducciones libres, entre muchos otros. El dominio público depende del país: se usa la marca de cada fuente y la ley argentina (70 años desde la muerte del autor) como control adicional.
- **Libros con derechos** (aunque el usuario los tenga comprados): **no entran por defecto**, ni el texto ni las notas o resúmenes derivados. Como con el código GPL, es una decisión del usuario, con una opción explícita y apagada (`libros_propios`), sabiendo que entrenar con obras protegidas tiene riesgo legal.

**Ciclo de lectura con tutor** (amplía B1–B6):

| # | Paso | Quién | Qué queda |
|---|---|---|---|
| B8 | **Leer** un capítulo y escribir **notas**: qué pasó, quién es quién, por qué hace lo que hace, qué cambió, ideas y temas | Hope | borrador de notas |
| B9 | **Volver a contar** el capítulo con sus palabras (resumen narrativo) y escribir **qué espera que pase** y qué preguntas le quedan | Hope | relato + predicciones |
| B10 | **Revisión del tutor**: con el capítulo a la vista, el maestro compara las notas y el relato de Hope con el texto y marca cada error: hecho mal leído, personaje confundido, causa o motivo equivocado, idea principal omitida, detalle inventado o ironía no captada. Explica **por qué** está mal y **dónde** está en el texto (capítulo y escena, sin copiar fragmentos largos). No da notas nuevas: corrige las de Hope | maestro | lista de errores con explicación |
| B11 | **Verificar al tutor**: cada corrección se contrasta con el texto (que el pasaje exista y diga eso) y, si hay dudas, con un segundo maestro de los turnos. Las correcciones que no se confirman se descartan: el tutor también se equivoca | verificador (§11.4) + segundo maestro | correcciones confirmadas |
| B12 | **Reescribir**: Hope corrige sus notas y su relato con las explicaciones | Hope | notas corregidas → Notebook y grafo (personajes, relaciones, línea de tiempo, causas) |
| B13 | **Repasar**: cada error confirmado se vuelve una tarjeta de repaso espaciado (Leitner) con preguntas nuevas sobre lo mismo; al terminar el libro, preguntas sobre la obra completa | Notebook + maestro | repaso |

**Ejercicios propios de la narrativa** (generados por el maestro y verificados contra el texto):
- **Línea de tiempo y causas:** qué pasó antes y qué provocó qué.
- **Mente de los personajes:** qué sabe uno que otro no sabe y por qué actúa así.
- **Volver a contar** desde la mirada de otro personaje.
- **Inferir** lo que el texto no dice explícitamente, justificándolo con lo que sí dice.
- **Continuar la historia** o escribir una propia con el mismo estilo (escritura creativa; el texto lo produce Hope, no se copia el libro).

**Cómo se usa para entrenar:**
- **SFT:** las notas y relatos corregidos (B12) y las explicaciones del tutor (B10) confirmadas (B11).
- **Preferencias (DPO, §11.5 L3):** cada par «nota original con error» contra «nota corregida» es un ejemplo de preferencia que viene de los errores reales de Hope.
- **Repaso:** las tarjetas de B13 se vuelven a preguntar con el tiempo; si Hope vuelve a fallar, el ejemplo vuelve al entrenamiento.
- **Licencia:** todo lo derivado lleva la licencia del libro de origen; de libros de solo lectura no se entrena nada.

**Por qué ayuda a una red chica:**
- Comprender un relato largo obliga a seguir personajes, causas y tiempo: es razonamiento sobre lenguaje.
- Las correcciones atacan los errores que Hope de verdad comete.
- Escribir con sus palabras mejora la redacción (§11.5).

**Medición:** en libros que no se usaron para entrenar, comparar preguntas de comprensión (hechos, causas, personajes, inferencias) y la calidad de los resúmenes juzgada a ciegas, antes y después de varios libros con tutor y frente a la misma lectura sin tutor.

**Implementación:** modo `lectura` en `hope/teacher/tutor.py` (B8–B13), índice de libros de §11.8 para que el tutor ubique el pasaje, y turnos nocturnos de los maestros.

### 11.9 Caché de 1 GB para textos largos y más datos (decisión del usuario, 2026-11-10)

El usuario fija un presupuesto de **1 GB de caché KV** (configurable en la app: `cache_mb`, 1024 por defecto) para que Hope lea textos largos y guarde más datos en memoria de trabajo.

**Cuánto texto entra en 1 GB** (pendiente marginal entre 32 y 128 KB, según `docs/resultados/contexto_memoria.json` de la rama `-contexto`; la ventana local fija ocupa unos pocos MB):

| Preset | Caché | Bytes de caché por byte de texto | Texto en 1 GB |
|---|---|---|---|
| hn1-200m (actual) | int4 | ~4.900 | ~200 KB, pero el pico al leer el prompt lo limita a ~30 KB en la 1060 |
| hn1-200m-largo | bf16 | ~230 | ~4,5 MB |
| hn1-200m-largo | int8 | ~120 | ~8,5 MB |
| hn1-200m-largo | int4 | ~70 | **~14 MB** (≈ 3,5M tokens equivalentes, varios libros) |

Con 1 GB hace falta el preset **largo**. En la 1060, 1 GB de caché más los pesos a 4 bits entra en los 6 GB si Qwen no está usando la GPU al mismo tiempo. En la A770 sobra.

**Que entre no significa que lo use bien.** El currículo actual llega a 128 KB y 14 MB es unas 100 veces más. Por eso:

| # | Pieza | Qué hace | Estado |
|---|---|---|---|
| M1 | **Caché por niveles** | Lo reciente a 8 bits y lo viejo a 4 bits, dentro de `cache_mb`. Al llenarse, se descartan primero las páginas menos consultadas (el puntaje de Quest), no las más viejas | pendiente; KV int8/int4 y Quest hechos |
| M2 | **Quest al generar** | Con 14 MB en caché, en cada paso solo se leen las páginas relevantes: la velocidad no cae en proporción al largo | hecho |
| M3 | **Currículo extendido en la A770** | 128 KB → 512 KB → 2 MB, con datos largos reales (libros completos, repos) y la mitad de los datos cortos | pendiente; ampliar etapas 5–5c |
| M4 | **Cachés guardadas en disco** | Hope lee un libro o un repo una vez y guarda su caché comprimida en el disco de datos. Volver a abrirlo es instantáneo, sin releer | pendiente; el prefill de 14 MB tarda minutos (≈ 0,1 GFLOP por byte: ~5 min en la 1060, ~2 min en la A770, estimado) |
| M5 | **Recuperación para lo que no entra o no se usa bien** | Más allá de lo que la atención aprovecha, el kNN, ⟦buscar_libro⟧ y las notas del Notebook traen el pasaje exacto. La caché es la memoria de trabajo; la recuperación es la biblioteca | parcial (kNN hecho; índice de libros en §11.8) |
| M6 | **Medir antes de confiar** | Banco de contexto (dato escondido, varias claves, combinar datos lejanos) a 128 KB, 1 MB y 14 MB, con y sin M5 | banco hecho; falta correrlo |

**Expectativa:**
- **Encontrar un dato puntual** en varios MB: debería andar después de M3, con ayuda de Quest.
- **Combinar muchos datos lejanos:** difícil para un 200M; ahí rinde más M5 (recuperación) que la caché sola.
- **Más datos de verdad** (todos los libros, Wikipedia): eso va por recuperación y memoria en disco, no por la caché.

#### 11.9.1 Equivalencia en tokens

Un token de un BPE equivale a ≈ 4 bytes (en español, algo más por las tildes y las palabras largas). Con 1 GB de caché:

| Caché (preset largo) | Texto | Tokens equivalentes |
|---|---|---|
| bf16 | ~4,5 MB | ~1,1M |
| int8 | ~8,5 MB | ~2,1M |
| int4 | ~14 MB | **~3,5M** |

Como referencia, los modelos grandes actuales trabajan con contextos de 128K a 1M de tokens. **Que entre no es lo mismo que usarlo bien:** la precisión real se mide con M6.

**Cómo mejorar el manejo** (además de M1–M6):
- **Índice jerárquico del contexto:** mientras lee, Hope arma resúmenes por sección y por capítulo (resumen continuo de `escritura.py`) que quedan **al frente** de la caché. Para responder, primero mira el índice y después baja a las páginas con Quest, como una persona que usa el índice de un libro.
- **Entrenar a buscar en el contexto:** ejercicios de dato escondido, varias claves y combinar datos lejanos a largos crecientes (los genera `contexto_bench`) dentro del currículo M3, para que la atención aprenda a recuperar y no solo a leer.
- **Notas en lugar de releer:** lo importante se anota en los scratchpads (§11.10), que son cortos y quedan siempre a mano.

### 11.10 Scratchpads: notas por agente y una pizarra general (decisión del usuario, 2026-11-10)

Cada agente del equipo (§11.4) tiene un **scratchpad propio** con su caché, y hay una **pizarra general** compartida por todo el modelo. Funcionan como la memoria de trabajo de una persona: anotar lo importante para no tener que releer todo.

**Reparto del presupuesto de 1 GB** (`cache_mb`; ajustable en la app):

| Parte | Presupuesto | Qué guarda |
|---|---|---|
| Documentos (contexto compartido) | ~768 MB | lo que Hope está leyendo: libros, repos, conversación larga |
| Pizarra general | ~64 MB | plan del maestro, subtareas, hechos confirmados con su fuente, veredictos del verificador, decisiones |
| 6 scratchpads (uno por agente) | ~32 MB cada uno (~192 MB) | notas, borradores, cálculos intermedios, dudas y pendientes de ese agente |

A ~120 bytes de caché por byte de texto (int8), 32 MB son ~270 KB de notas por agente, y 64 MB son ~530 KB de pizarra: mucho más de lo que hace falta para notas.

**Herramientas:**
- `⟦anotar: …⟧`, `⟦leer_notas⟧` y `⟦borrar_nota: id⟧` sobre el scratchpad propio.
- `⟦pizarra: …⟧` y `⟦leer_pizarra⟧` sobre la general.
- Las notas son texto corto con identificador, autor, tipo (hecho, duda, plan, cálculo) y fuente si la hay.

**Reglas:**
- Cada agente escribe solo en su scratchpad. La pizarra admite escritura de todos, y cada entrada queda firmada por su autor.
- **El verificador no lee el scratchpad del agente que revisa** antes de formar su propio juicio: lee solo las afirmaciones publicadas en la pizarra y las comprueba por su cuenta (Chain-of-Verification, §11.3 V5). Así no hereda el error.
- En la pizarra, solo el verificador puede marcar un hecho como *confirmado*.
- Los scratchpads duran lo que dura la tarea. Al terminar, el maestro consolida lo útil en el Notebook y en el grafo (memoria de largo plazo, con repaso). El resto se descarta.
- Todo queda dentro del sandbox y del registro de auditoría. Ningún agente puede usar los scratchpads para cambiar configuración ni límites.

**Detalle técnico importante: compartir la caché de documentos.** Hoy `attach_named_adapters` pone LoRA en **todas** las capas lineales de `regions` y `backbone`, incluidas las proyecciones K/V. Entonces la caché de un documento calculada con el adaptador de un agente **no sirve** para otro, y habría que leer el documento una vez por agente (6× memoria y 6× tiempo). La solución:
- Los adaptadores de agente **no tocan K ni V** (solo Q, la salida de la atención, los MLP y las zonas), o se limitan a las capas superiores.
- Así la caché de documentos se calcula **una sola vez** con el modelo base y la comparten los 6 agentes. Lo único propio de cada uno es su scratchpad.
- Hay que medir que restringir los adaptadores no les quite capacidad de especializarse.

**Implementación:**

| # | Paso | Archivos |
|---|---|---|
| S1 | Adaptadores sin K/V (opción `excluir_kv=True` por defecto para agentes) | `hope/plasticity/adapters.py` |
| S2 | Caché segmentada: documentos + pizarra + scratchpad del agente activo, cada segmento con su presupuesto y su política de descarte (M1) | `hope/core/kvcache.py`, `hope/core/batched.py` (caché de conversación) |
| S3 | Herramientas de notas y pizarra, con autor y permisos | nuevo `hope/council/scratchpad.py`, `hope/runtime/toolcalls.py` |
| S4 | Consolidación al terminar la tarea → Notebook y grafo | `hope/council/maestro.py`, `hope/memory/` |
| S5 | Ver la pizarra y las notas en la app (pestaña de charla) | `hope/app/` |
| S6 | Entrenar el hábito: episodios donde anotar a tiempo es lo que permite responder bien (tareas largas, varias subtareas) | datos del tutor y del currículo |

**Medición:** tareas largas de varias subtareas (leer un libro y responder preguntas que cruzan capítulos; arreglar un error que toca varios archivos), con y sin scratchpads, a igual presupuesto de caché y tiempo.

#### 11.10.1 Registro completo de notas, scratchpads y caché (decisión del usuario, 2026-11-10)

**Objetivo:** cada nota, cada scratchpad y cada movimiento de caché quedan documentados, para que el usuario y Hope puedan **releer, retomar o rehacer** cualquier tarea más adelante sin volver a leer todo. El registro es también lo que hace eficiente el trabajo con textos largos: las notas apuntan a la posición exacta de cada fuente, así se recarga solo lo que hace falta.

**Dónde se guarda** (en el disco de datos de la app, nunca en el SSD del sistema):

```
memoria/
  indice.sqlite                    # índice de búsqueda (texto completo) de todo lo de abajo
  documentos.jsonl                 # catálogo de documentos leídos
  tareas/<tarea_id>/
    notas.jsonl                    # cada evento de nota (scratchpads y pizarra)
    cache.jsonl                    # cada evento de caché
    resumen.md                     # resumen legible al cerrar la tarea
  cache_disco/
    <doc_id>__<modelo>__<bits>.kv  # caché guardada (M4)
    <doc_id>__<modelo>__<bits>.json  # su manifiesto
```

**1. Notas (`notas.jsonl`): solo se agregan, nunca se borran ni se reescriben.** Editar o borrar crea un evento nuevo que referencia al anterior, así la historia completa queda.

| Campo | Contenido |
|---|---|
| `id`, `tarea`, `agente`, `espacio` | identificador; tarea; quién la escribió (maestro, verificador, lenguaje, código, investigador, razonamiento); `scratch:<agente>` o `pizarra` |
| `evento` | `crear`, `editar` (con `reemplaza`), `borrar` (marca, no borra), `confirmar` o `refutar` (solo el verificador), `consolidar` (pasó al Notebook o al grafo) |
| `tipo` | hecho, duda, plan, cálculo, borrador, decisión, pendiente |
| `texto` | la nota |
| `fuente` | `doc_id` + rango de bytes + capítulo/sección/página, o URL, o herramienta y entrada usada |
| `deriva_de` | ids de las notas o pasajes de los que sale (cadena de razonamiento reconstruible) |
| `estado` | borrador / confirmado / refutado / obsoleto, con el veredicto y el motivo del verificador |
| `licencia` | la del documento de origen: decide si la nota puede usarse para entrenar (§11.8) |
| `t`, `hash`, `hash_prev` | hora; encadenado por hash como el registro de auditoría (`guardian/audit.py`): no se puede alterar sin que se note |

**2. Caché (`cache.jsonl`)**: cada operación con su motivo.

| Evento | Qué se registra |
|---|---|
| `cargar` | `doc_id`, sha256 del contenido, rango de bytes, segmento (documentos / pizarra / scratch:agente), preset, bits, adaptador (base), MB y tiempo de lectura |
| `comprimir` | qué páginas pasaron de 8 a 4 bits y por qué (antigüedad, presupuesto) |
| `descartar` | qué páginas salieron, su puntaje de Quest, cuántas veces se consultaron y el motivo (presupuesto lleno) |
| `consultar` | qué páginas leyó Quest para cada respuesta (muestreado, para no inflar el registro) |
| `guardar_disco` / `restaurar` | archivo, tamaño y si la restauración fue válida |
| `invalidar` | caché descartada porque cambió el modelo o el documento |

**3. Documentos (`documentos.jsonl`):** `doc_id`, título, fuente, licencia, sha256, tamaño, fecha, y el **índice jerárquico** (resumen por sección y capítulo con su rango de bytes; §11.9.1). Si el contenido cambia, cambia el sha256 y se registra como documento nuevo.

**4. Caché en disco y su manifiesto:** una caché solo vale para **el mismo modelo** que la calculó. El manifiesto guarda el hash del checkpoint, el preset, los bits, el rango de bytes, el sha256 del documento y la fecha. Al restaurar se comprueba todo. Si el modelo cambió (más entrenamiento, crecimiento), la caché se invalida y se rehace, y se registra el motivo.

**5. Resumen de la tarea (`resumen.md`), al cerrarla:** pedido original, plan, decisiones, hechos confirmados con su fuente, dudas abiertas, qué se consolidó en el Notebook, documentos usados y enlaces a las notas. Es lo primero que lee el usuario o Hope para retomar.

**Cómo se retoma una tarea** (`⟦retomar: tarea⟧`, también desde la app):
1. Se lee `resumen.md` y las notas vigentes (la última versión de cada una, sin las borradas).
2. Por cada documento: si hay caché en disco válida, se restaura; si no, **se recargan solo los rangos de bytes que citan las notas** más el índice jerárquico, no el documento entero.
3. La pizarra y los scratchpads se reconstruyen desde las notas.
4. Para profundizar, Quest y el índice bajan a las páginas que falten.

**Búsqueda:** `⟦buscar_notas: consulta⟧` busca por texto, tarea, agente, documento, tipo, estado y fecha en `indice.sqlite`. La app tiene una vista **«Memoria de trabajo»** con la línea de tiempo de cada tarea, filtros, la historia de cada nota y exportación a Markdown.

**Espacio y retención:**
- Las notas y los registros son texto chico: se guardan siempre y se comprimen al cerrar la tarea.
- Las cachés en disco son grandes: tienen una cuota (`cache_disco_gb`, 20 GB por defecto). Al llenarse, se borran las menos usadas, y se registra cuáles. Se pueden rehacer desde el documento.
- Hope no puede borrar ni modificar la historia. El usuario sí puede purgarla, por tarea o completa, con `guardian.cli`, y la purga queda en el registro de auditoría.

**Implementación:**

| # | Paso | Archivos |
|---|---|---|
| S7 | Registro de notas y de caché, append-only y encadenado por hash | `hope/council/scratchpad.py`, `hope/core/kvcache.py`, reutiliza `guardian/audit.py` |
| S8 | Catálogo de documentos con índice jerárquico | `hope/memory/documentos.py` |
| S9 | Caché en disco con manifiesto y validación | `hope/core/kvcache.py` (M4) |
| S10 | `resumen.md`, ⟦retomar⟧ y ⟦buscar_notas⟧ | `hope/council/maestro.py`, `hope/runtime/toolcalls.py` |
| S11 | Vista «Memoria de trabajo» y purga | `hope/app/`, `guardian/cli.py` |

**Medición:** retomar una tarea larga (por ejemplo, un libro leído a medias) desde el registro, comparado con empezar de cero. Medir tiempo, MB recargados y si las respuestas siguen igual de bien.

### 11.11 Red adaptativa: escalera de maestros, autodiagnóstico de capacidad y evolución de arquitectura (decisión del usuario, 2026-11-10)

Objetivo del usuario: hacer a Hope lo más inteligente posible y **gradual**. Hope detecta sola dónde le falta potencia, crece en esa parte y cambia su arquitectura si aparece un método más eficiente. El maestro también sube de nivel: primero un Qwen chico y después la generación siguiente, más rápida y capaz.

**Lo que ya existe:**
- `hope/growth/diagnosis.py`: diagnóstico por reglas que descarta primero las causas baratas (datos → herramienta → memoria → optimización → capacidad) y distingue capacidad de optimización con una **sonda** (copia con más capacidad contra un control con el mismo entrenamiento extra).
- `hope/growth/ops.py` y `hn1_ops.py`: crecimiento que preserva la función (expertos, ancho, bloques, zona visual).
- `budget_target`, `asg.py`, el guardián, aprobación humana por encima del 10 % y techo fijado por el usuario.

#### A1 — Escalera de maestros

| Paso | Qué |
|---|---|
| A1.1 | **Maestro a la medida del alumno.** Con una brecha muy grande entre maestro y alumno, el alumno aprende peor; un maestro intermedio ayuda (*teacher assistant*, Mirzadeh et al. 2020). Hope empieza con un Qwen chico (o Nemotron 3 Nano 4B, por turnos) y sube a modelos más grandes o a la generación siguiente a medida que crece |
| A1.2 | **Cambiar de maestro es barato:** Hope aprende con secuencias (textos, correcciones, ejercicios verificados), no con las probabilidades del tokenizador del maestro (§11.2 P0.3). Cualquier modelo nuevo sirve sin tocar la red, a través de `maestros.py` y los turnos |
| A1.3 | **Cuándo subir de maestro** (señales medidas, no fechas): (a) el tutor ya casi no encuentra errores que sepa corregir; (b) Hope alcanza al maestro actual en el banco sellado de un área; (c) el maestro se equivoca seguido en lo que Hope necesita (sus correcciones se refutan en §11.8.1 B11) |
| A1.4 | **Maestros especializados por área:** un Qwen de código para código, uno de visión para imágenes, uno general para lengua. Cada uno sube de nivel por separado |
| A1.5 | **Cada maestro nuevo pasa por el control de licencias** antes de usarse: su licencia tiene que permitir entrenar con sus salidas (Carta; los modelos de Llama, Gemma y similares siguen apagados por defecto) |

#### A2 — Autodiagnóstico de capacidad (ampliar `diagnosis.py`)

**Señales nuevas por área y por región:**
- **Curva de aprendizaje:** si más datos y más tiempo todavía bajan la pérdida, el problema **no** es de capacidad.
- **Sonda de capacidad** (ya existe): ganancia de una copia ensanchada contra un control.
- **Uso de expertos MoE:** expertos saturados (mucha carga, pérdida alta) o muertos (nunca elegidos).
- **Iteraciones del núcleo:** si llegan seguido al máximo, el problema pide más profundidad; si colapsan a 1, no aprovecha el núcleo (P0.1).
- **Errores del tutor por categoría** (§11.8.1) y del verificador (§11.4).
- **Rango efectivo** de las activaciones y de las actualizaciones de cada bloque.

**Decisión → acción** (la expansión es la última opción):
- datos → más datos o datos mejores del área;
- herramienta o memoria → herramienta, recuperación o notas;
- estrategia → razonamiento o RL;
- capacidad local → crecer **solo esa parte**: experto nuevo en la región, ensanchar un experto, un bloque más o más iteraciones;
- capacidad general → crecer el modelo dentro del techo.

**Cada propuesta lleva su evidencia:** señales medidas, sonda, costo en parámetros y velocidad, y la ganancia esperada. Se registra en el linaje con vuelta atrás.

#### A3 — Evolución de arquitectura

| Paso | Qué |
|---|---|
| A3.1 | **Catálogo de variantes:** las mejoras de §11.2 (Ola A/B), las que proponga la literatura y las que proponga Hope. Cada una se prueba primero en **modelos chicos de 25M con µP**, así el resultado se traslada a 200M o 500M (B1) |
| A3.2 | **Si la variante preserva la función** (por ejemplo, embeddings de n-gramas con inicialización en cero, un bloque nuevo con salida en cero), se agrega en caliente con `hope/growth/` |
| A3.3 | **Si no la preserva** (otra activación, otra atención, otro patcheo): **Hope se destila a sí misma.** El modelo nuevo aprende de la Hope actual, además de los maestros y los datos. Entre versiones de Hope comparten el vocabulario de bytes, así que se pueden copiar las **probabilidades completas**, que es la destilación más eficiente y la que no funcionaba con Qwen por los tokenizadores distintos. Así se cambia la arquitectura sin perder lo aprendido |
| A3.4 | **Promoción solo con evidencia:** la variante nueva tiene que ganar en el banco sellado con 2 semillas y el mismo cómputo, sin empeorar ninguna área ni las pruebas de seguridad. Si no gana, se descarta y queda en el registro |

#### A4 — Resolver de forma no tradicional: generar y verificar

| Paso | Qué |
|---|---|
| A4.1 | **Búsqueda evolutiva de programas** (estilo AlphaEvolve, 2025): para problemas difíciles con evaluador automático (código con tests, optimización, matemática), Hope y el maestro proponen muchas variantes de una solución. El evaluador las puntúa, las mejores se mutan y se combinan, y así varias generaciones. Encuentra soluciones que ninguno escribiría de una vez, y usa el excedente de velocidad |
| A4.2 | Las soluciones nuevas y verificadas vuelven como datos de entrenamiento (K2, R3) y como herramientas nuevas (`hope/tools/creation.py`) |
| A4.3 | Para código, combina con K5 (tests primero), K9 (GRPO) y los lectores de libros de programación (§11.8) |

#### A5 — Más datos con menos cómputo: elegir qué aprender

| Paso | Qué |
|---|---|
| A5.1 | **Pérdida selectiva por byte** (Rho-1, Lin et al. 2024): un modelo de referencia entrenado con datos de alta calidad marca qué partes del texto valen la pena. Hope entrena con más peso en esas y no gasta capacidad en ruido. Con bytes es directo, porque Hope y su referencia comparten vocabulario |
| A5.2 | **Currículo por aprendibilidad:** priorizar los ejemplos que Hope todavía no sabe pero puede aprender (pérdida alta y que baja con el entrenamiento), no los imposibles ni los ya sabidos |
| A5.3 | **La biblioteca crece por fuera de los parámetros:** libros, notas, cachés y recuperación (§11.8–11.10). Los parámetros se reservan para razonar |

#### Lo que no cambia (Carta Fundamental)

Hope **propone** y no se aprueba sola:
- el guardián evalúa cada operación contra el presupuesto;
- más del 10 % de los parámetros requiere aprobación humana;
- el techo de parámetros lo sube **solo** el usuario (`guardian.cli limite`);
- `budget.change`, `safety.modify` y `audit.modify` siguen prohibidos;
- **ningún cambio de arquitectura puede tocar el guardián, la auditoría ni la configuración de seguridad;**
- el banco sellado queda oculto para Hope, para que no pueda optimizar el examen en lugar de la habilidad;
- todo cambio queda en el linaje y se puede revertir.

#### Ganancia esperada

- **Escalera de maestros:** mejor aprovechamiento de cada etapa. Es la forma de que la calidad de Hope siga a la de los maestros nuevos sin rehacer nada.
- **Crecimiento dirigido:** más rendimiento por parámetro que crecer en bloque, porque cada parámetro nuevo va donde la sonda mostró que falta.
- **Autodestilación entre arquitecturas:** permite adoptar métodos nuevos sin empezar de cero. Es lo que hace a Hope realmente adaptativa a lo largo de los años.
- **Generar y verificar:** es la vía realista para soluciones «no tradicionales». No sale de la red sola, sino de la búsqueda con un evaluador estricto.

**Orden:** A5.1 y A1.1–A1.3 con el entrenamiento de la 1060 → A2 (señales nuevas) en cada checkpoint → A3 cuando la Ola A deje variantes ganadoras → A4 con el 200M entrenado.

### 11.12 Entrenar en el equipo grande, ejecutar en el chico (Lenovo A275) (decisión del usuario, 2026-11-10)

**Plan del usuario:** entrenar en la PC grande (GTX 1060 y después A770) y **probar y usar** en la Lenovo A275 (AMD A12-8830B, 4 núcleos, AVX2, 16 GB de RAM, iGPU sin soporte de PyTorch: **solo CPU**). La meta es llevar todas las capacidades al máximo y elegir el modelo de código más grande que la A275 pueda ejecutar bien.

**Qué limita en la A275:** no la RAM (1B a 4 bits ocupa ~0,5 GB, más 1 GB de caché), sino la **velocidad de la CPU y de la memoria** (≈ 15–30 GB/s). Gracias al patcheo y al MoE, los parámetros **activos por byte** crecen mucho menos que el total. Cálculo con `hope.bench.plan`; la columna «realista» supone un runtime optimizado (R1–R2), y sin él la velocidad es varias veces menor:

| Preset | Total | Activos por byte | Pesos int4 | Techo teórico (30–50 GFLOPS) | Realista con R1–R2 |
|---|---|---|---|---|---|
| hn1-200m | 181M | 47M | ~90 MB | 80–130 tok/s | ~35–60 tok/s |
| hn1-500m | 464M | 62M | ~230 MB | 60–100 tok/s | ~25–45 tok/s |
| hn1-700m | 689M | 74M | ~340 MB | 50–85 tok/s | ~20–40 tok/s |
| **hn1-1b** | 998M | 86M | ~500 MB | 45–75 tok/s | **~20–35 tok/s** |

**Conclusión:**
- **El techo práctico lo pone el equipo de entrenamiento, no la A275.** La A770 entrena hasta ~1B (`budget_target`: 946M), y la A275 ejecuta ese 1B a una velocidad usable para código (~20–35 tok/s estimado).
- **Más allá de 1B:** un MoE más ancho (más expertos, mismos activos) de 2–3B totales y ~100M activos por byte entraría en RAM y correría a ~15–25 tok/s en la A275. Pero necesita una GPU más grande para entrenarlo (4090/5090; §11.1).
- **Versión de código para la A275:** hn1-1b con zonas MoE cargadas hacia código (más expertos de código; §11.6), QAT a 4 bits, y el resto de §11.6. Estimación del §11.6: **30–50 % pass@1**.
- **Contra qué balancear:** a ~25 tok/s queda poco tiempo sobrante para varios intentos con tests (K5) y para el equipo de agentes, y los tests también consumen CPU. Hay que medir en la A275 qué rinde más con el mismo tiempo de espera: **1B con 1–2 intentos o 500M con 4–8 intentos**. Para código verificable puede ganar el 500M.
- **Destilar para el equipo chico:** si la A770 entrena un modelo más grande del que la A275 corre cómodo, se destila a uno más chico con el mismo vocabulario de bytes (Hope → Hope, §11.11 A3.3), sin perder casi nada.

**Para llevar todo al máximo en la A275:**

| # | Pieza | Qué hace | Estado |
|---|---|---|---|
| R1 | **Exportar a un runtime compilado** (`torch.compile` en CPU u ONNX Runtime) | Elimina el costo de Python y de lanzar operaciones; hoy un modelo chico en CPU usa ~6 % de su techo | pendiente |
| R2 | **Kernels propios int4/int8 con AVX2** (estilo llama.cpp/ggml) para lineales, atención con caché cuantizada y SSM | Acerca la velocidad al techo de ancho de banda | pendiente; es el trabajo más grande |
| R3 | **QAT a 4 bits** en el enfriamiento (ya en el plan de la A770) y exportación verificada a int4 | El modelo se entrena para funcionar cuantizado | hecho; verificar la exportación en CPU |
| R4 | **Cachés guardadas en disco** (§11.9 M4, §11.10.1): la PC grande lee los libros y repos, y la A275 solo restaura la caché | Leer 14 MB en la A275 tardaría decenas de minutos; restaurar, segundos | pendiente; requiere el **mismo** checkpoint en las dos máquinas |
| R5 | **Perfil `a275` en `hope/bench/hardware.py`** (4 núcleos, AVX2, 16 GB, ~15–30 GB/s, solo CPU) y presupuesto de velocidad propio en `BudgetedResponder` (menos rondas, más camino rápido) | La app se ajusta sola a la máquina | pendiente |
| R6 | **Paquete de despliegue:** checkpoint int4 + cachés + índices de libros y Wikipedia + notas, copiado de la PC grande a la A275 con el hash de cada archivo | Llevar el modelo entrenado de una máquina a la otra | pendiente |
| R7 | **Medición en la A275 real:** velocidad por preset, consumo, banco de código con 1–8 intentos y banco de conversación | Decide el preset final de la A275 | pendiente; necesita la máquina |

**Orden:** R5 y R3 (baratos) → R1 → R7 con el 200M entrenado → R2 si R1 no alcanza → R4 y R6 cuando estén las cachés en disco.

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
