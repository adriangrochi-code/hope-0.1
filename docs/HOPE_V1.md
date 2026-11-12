# Hope v1: arquitectura exacta para empezar a entrenar

Decisión del usuario, 2026-11-12: definir la arquitectura concreta de Hope y empezar a entrenarla con todos los datos abiertos que se pueda (Hugging Face y documentación libre), para después darle de a poco más ciberseguridad, programación y el resto.

Fija lo que se entrena **primero**. Las mejoras que todavía no están medidas (núcleo TRM, tronco donado, capas de memoria, poda) entran como **variantes a comparar**, no como base, porque cada una pasa por la ablación con 2 semillas (MAPA_REDES §5).

**Presets en el código:** `hope-v1-proxy` (29,5M, CPU) y `hope-v1-semilla` (233,5M, GTX 1060), en `hope/core/hn1.py`. Las dos tienen las mismas proporciones, así lo que gana en el proxy se traslada a la semilla.

---

## 1. Qué la hace más flexible y potente por parámetro que una red tradicional

| Rasgo | Red tradicional | Hope v1 | Por qué rinde más por parámetro |
|---|---|---|---|
| Entrada | tokens de un vocabulario fijo (la tabla ocupa muchos parámetros) | **bytes** con **parches por entropía**: cortes donde el texto es difícil | no gasta parámetros en vocabulario; más cómputo donde hace falta. Medido: 2,43 BPB contra 2,56 con tokens de Qwen |
| Capas | todas iguales (atención + MLP) | **mezcla de SSM y atención** (3 a 1) | los SSM son baratos para el contexto largo; la atención, precisa donde hace falta |
| Neuronas | un solo tipo de MLP | **7 tipos de especialistas**: MLP, GLU, excitatorio/inhibitorio, recurrente, convolucional, asociativo y relacional, elegidos por un router (top-4 de 16) más un **experto compartido** | cada tipo resuelve mejor una clase de problema; solo corre una parte por parche |
| Profundidad | fija | **núcleo recurrente** con reinyección de la entrada y 1–8 vueltas (profundidad log-normal) | pensar más sin más parámetros |
| Salida | un token por paso | byte siguiente + **3 cabezas multi-byte** | aprende a prever más adelante y sirve de borrador para generar 2–5× más rápido |
| Memoria | solo los pesos | **atención cruzada a memoria** + kNN, notas, libros y caché de 1 GB por fuera | el conocimiento vive en parte fuera de los parámetros |
| Cálculos | los aprende «de memoria» | **herramientas** para todo lo determinista (§11.16) | cero parámetros gastados en contar o calcular |
| Crecimiento | se entrena de nuevo más grande | **crece y se poda conservando lo aprendido** (§6, §11.18) | el tamaño se ajusta a lo que hace falta |

---

## 2. Especificación de la semilla (`hope-v1-semilla`, GTX 1060)

| Parte | Valor | Parámetros |
|---|---|---|
| Parte local de bytes | `d_local` 512, 8 cabezas, codificador de 2 capas, decodificador de 3 | ~13M |
| Patcheo | por entropía (`patching: entropia`, orden 3, 65.536 cubos) | — |
| Tronco global | `d` 1024, 16 cabezas con **GQA** (4 de K/V), MLP 2048, **12 bloques**: SSM-SSM-SSM-atención × 3 | ~159M |
| Memoria | atención cruzada | ~4M |
| Núcleo recurrente | reinyección convexa (Ā = 0,5), paso senoidal, profundidad log-normal de media 4 y tope 8 | ~9M |
| Zona de especialistas | 16 especialistas (7 tipos), oculta 1024, **top-4**, **experto compartido** | ~46M |
| Cabezas multi-byte | 3 | ~1M |
| **Total** | | **233,5M** |

**Entrenamiento en la 1060:**
- precisión `fp32-8bit` con el optimizador en RAM (Pascal no tiene bf16);
- **Muon + WSD** (medido: −0,17 BPB juntos);
- **QAT a 4/8 bits** en el enfriamiento;
- secuencias de 4–8 KB.

**Proxy (`hope-v1-proxy`, 29,5M):** las mismas piezas con `d` 384 y 6 bloques. Es el que corre en CPU para comparar variantes.

---

## 3. Variantes que compiten contra la base (en el proxy, 2 semillas, mismo cómputo)

| Variante | Qué cambia | Si gana… |
|---|---|---|
| **V-TRM** | el núcleo pasa a la receta TRM (recursión de latente y respuesta, supervisión por vuelta) + Mixture-of-Recursions | reemplaza al núcleo de la semilla |
| **V-Ola A** | n-gramas de bytes con hash, ReLU², QK-norm, recorte de logits, promedio de pesos | entra en la semilla |
| **V-Memoria** | capas de memoria con claves de producto en lugar de parte de la zona | entra y abre el camino a §11.19 |
| **V-Donante** | tronco bytificado de un modelo abierto chico (licencia a verificar) + expertos reciclados | cambia el tronco de la semilla (y su `d`) |
| **V-µP** | parametrización para transferir hiperparámetros | se usa para pasar del proxy a la semilla |
| **V-Poda** | barrido de poda con ≤5 % de pérdida por área y redistribución | se aplica al cerrar la semilla |

---

## 4. Datos para la base

**Criterio:** solo conjuntos que pasan la revisión D0 (licencia, origen rastreable, cómo se generó y contenido; ARQUITECTURA_HN1 §11.21). La 1060 lee ~1–2B tokens equivalentes por semana, así que se prioriza calidad sobre volumen.

| Área | Peso inicial | Fuentes (verificar la ficha antes de activar) |
|---|---|---|
| Texto general de calidad, es/en | 35 % | Wikipedia es/en (ya integrada), FineWeb-Edu, FineWeb-2 (español), Wikilibros, libros de dominio público |
| Código | 25 % | código abierto con licencia permisiva (ya integrado, The Stack v2 con filtro por archivo), repos del usuario sin coautoría de IA, documentación de Python y MDN |
| Matemática y ciencia | 10 % | OpenWebMath, OpenStax (CC BY), arXiv (resúmenes) |
| **Ciberseguridad defensiva** | 5 % (creciendo de a poco) | NVD, CISA KEV, MITRE ATT&CK/CWE/CAPEC/D3FEND, OWASP, NIST CSF, reglas Sigma |
| Herramientas y razonamiento verificable | 15 % | currículo de herramientas, ejercicios procedurales (R3), episodios del maestro verificados |
| Maestros (textos «de libro», conversaciones, correcciones) | 10 % | Qwen3-4B, Qwen2.5-Coder-1.5B, Phi-4-mini, por turnos |

**Gradualidad:** los pesos de ciberseguridad, código y conversación suben por etapas. Las ramas especialistas de §11.13 los llevan más lejos sin desequilibrar la base.

---

## 5. Orden para arrancar

1. **Proxy en CPU:** base contra las variantes de §3 (2 semillas). Se puede correr ya con `hope-v1-proxy`; V-TRM, V-Memoria y V-Donante requieren código nuevo.
2. **Preparar datos:** revisión D0 de cada conjunto, filtro de calidad, deduplicación y el filtro de seguridad.
3. **Semilla en la 1060:** `hope-v1-semilla` con las variantes ganadoras y el plan `configs/plan_gtx1060.yaml` adaptado a este preset. Primer día: medir TFLOPS reales.
4. **Medir en cada checkpoint:** banco sellado por área y comparación con Nemotron Nano 4B como referencia.
5. **Seguir la escalera:** ramas especialistas → A770 → clúster → ejecutar en la 1060 (ARQUITECTURA_HN1 §11.13–11.20).
