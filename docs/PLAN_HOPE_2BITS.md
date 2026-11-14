# Plan unificado: Hope en 2 bits que crece con redes especializadas

Decisión del usuario, 2026-11-14. Junta en un solo camino lo decidido en `ARQUITECTURA_HN1.md` §11.13–11.29, `SISTEMA_HOPE.md` y las cuatro investigaciones de `docs/investigacion/`.

**Idea:** no hace falta un modelo enorme. Hope arranca con un **núcleo en 2 bits** que entra en la PC de la 1060 y **crece por partes**: expertos nuevos, redes chicas especializadas, adaptadores, herramientas exactas, memoria y búsqueda. Cada parte se agrega solo si mejora lo que mide el banco, sin rehacer lo anterior.

---

## 1. Las tres capas de Hope

| Capa | Qué es | Dónde vive |
|---|---|---|
| **1. Núcleo 2 bits** | MoE ~30–35B totales / ~3B activos (Qwen3-Coder-30B-A3B primero; Qwen3.5-35B-A3B como destino; Qwen 4 si sale), ajustado para rendir en 2 bits (§11.28) | atención, router, compartido y caché caliente de expertos en la 1060; resto de expertos en RAM, calculados por la CPU |
| **2. Especialistas que se agregan** | expertos nuevos en 2 bits dentro del MoE, adaptadores LoRA por habilidad y redes chicas separadas (verificador, recuperación, router, visión, OCR, TRM de rompecabezas) | GPU o CPU según tamaño y uso |
| **3. Lo que no son pesos** | herramientas de cálculo exacto certificado e ingeniería, memoria (notas, grafo, caché de 1 GB, registro), búsqueda y libros, modo recursivo (RLM), equipo de agentes con verificador | sandbox, disco, CPU |

La **Hope de bytes** (`HOPE_V1.md`) sigue como **línea de investigación** de largo plazo, con el proxy en CPU. El producto se construye sobre el núcleo 2 bits.

---

## 2. Cómo crece (sin rehacer)

| Tipo de crecimiento | Cómo | Costo | Evidencia |
|---|---|---|---|
| **Expertos nuevos de un área** (código de un lenguaje, ingeniería, ciberseguridad defensiva, firmware) | se agregan expertos en 2 bits copiados de los existentes y reiniciados en parte (Drop-Upcycling), con **los expertos y las filas del router viejos congelados** y destilación hacia el modelo anterior | horas a días en la A770, por rama | MoTE, Drop-Upcycling, Lifelong-MoE (`arquitectura_crecimiento.md`, `ternario_1_58.md`) |
| **Habilidad puntual** (un formato, una API, el repo del usuario) | adaptador LoRA sobre la base congelada; el router elige cuáles activar | minutos a horas | LoRA aprende menos y olvida menos (`arquitectura_crecimiento.md`) |
| **Función distinta** (verificar, buscar, decidir, ver, OCR, grillas) | red chica separada (`SISTEMA_HOPE.md` N2–N8) | días, se entrena aparte | — |
| **Cálculo y datos fijos** | herramienta determinista con certificado (§11.16 T3) | sin entrenamiento | — |
| **Conocimiento** | libros, Wikidata, notas y búsqueda (§11.3, §11.8) | sin entrenamiento | Atlas, kNN-LM |
| **Contexto muy largo** | modo recursivo RLM (§11.29) + caché y notas | sin entrenamiento | RLM (MIT, 2025) |

**Regla de entrada:** cada pieza nueva entra solo si **mejora su área en el banco sellado y no empeora ninguna otra más de un 1 %** (alarma nocturna con vuelta atrás, §11.22 F3.3). Cada agregado queda en el linaje y se puede sacar.

**Regla de limpieza:** cada tanto se corre el barrido de poda (§11.18.1): lo que no se usa se saca y su lugar va a donde más falta.

---

## 3. Presupuesto de memoria (PC de la 1060)

| Recurso | Total | Núcleo 2 bits | Lugar para crecer |
|---|---|---|---|
| VRAM | 6 GB (~0,5 GB para el escritorio) | ~2–3 GB (atención, router, compartido, caché KV en 4 bits) | ~2–3 GB para la **caché caliente de expertos** y las redes chicas |
| RAM | 16 GB (~4 GB para el sistema y la app) | ~9–10 GB de expertos en 2 bits | **~2 GB**: muy poco |
| Disco de datos | — | modelo + cachés + índices | amplio |

**Cuello de botella para crecer: la RAM.** Con 16 GB, el núcleo ya ocupa casi todo. **Pasar a 32 GB de RAM** (DDR4, una de las mejoras más baratas del equipo) habilita ~16 GB más de expertos en 2 bits, del orden de **50B parámetros más** para áreas nuevas, sin tocar la velocidad por byte: los expertos se leen solo cuando el router los elige.

Sin más RAM, crecer pasa por:
- reemplazar expertos poco usados (barrido de poda);
- adaptadores LoRA, que pesan poco;
- redes chicas en la VRAM que sobra;
- herramientas, memoria y búsqueda, que no ocupan RAM de pesos.

---

## 4. Orden de trabajo

| # | Etapa | Resultado |
|---|---|---|
| 1 | Núcleo 2 bits con Qwen3-Coder-30B-A3B: ajuste capa por capa en la A770, kernels W2A8 `dp4a` (1060) y AVX2 (CPU), expertos en RAM con caché caliente, `hope serve` para TISMA Code | Hope Local funcionando en la 1060 |
| 2 | Herramientas de cálculo exacto e ingeniería + verificador con tests | calidad alta en lo verificable |
| 3 | Modo RLM + recuperación entrenada + memoria | repos y documentos largos |
| 4 | Primer crecimiento: expertos de **ingeniería** (firmware, electrónica, 3D) y de **ciberseguridad defensiva** en 2 bits, con destilación de maestros más grandes | áreas del usuario mejor cubiertas |
| 5 | Comparar con Qwen3.5-35B-A3B como base (kernel Gated DeltaNet sin Triton) | elegir el núcleo definitivo |
| 6 | Visión (SigLIP 2 + proyector) para planos, capturas y fotos de piezas | multimodal básico |
| 7 | Red distribuida (`RED_DISTRIBUIDA.md`) | más cómputo sin modelo más grande |

---

## 5. Cómo se mide el progreso

En cada etapa y en la 1060 real:
- **tareas resueltas por minuto** en el banco de código e ingeniería y en tareas reales del usuario;
- **tok/s** y latencia al primer token;
- **VRAM y RAM** usadas;
- comparación con un modelo denso de igual memoria en 4 bits y con Nemotron Nano 4B como referencia;
- regresión por área: ninguna baja de más de 1 % al agregar piezas.
