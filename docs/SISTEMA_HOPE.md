# Sistema Hope: una IA completa hecha de redes especializadas

Decisión del usuario, 2026-11-12: Hope no tiene que ser «un LLM más». Es un **sistema completo** donde cada función la hace la red (o el programa) **optimizada para esa función**, y el modelo de lenguaje es una pieza entre varias.

Complementa a `HOPE_V1.md` (el modelo de lenguaje) y a `MAPA_REDES.md` (qué maestro y qué módulo se usa).

---

## 1. Por qué no un solo LLM

Un LLM hace todo con la misma red: percibe, recuerda, decide, calcula, verifica y habla. Cada una de esas funciones tiene una forma de red que la hace mejor y más barato:

- **Buscar en la memoria:** un codificador chico y rápido, no una pasada del modelo grande.
- **Decidir qué hacer:** un clasificador de pocos millones de parámetros.
- **Verificar:** una red **separada** del que escribe. Si es la misma red, tiende a aprobar su propio error.
- **Calcular:** un programa (herramienta), no una red.
- **Seguridad:** reglas deterministas e independientes, nunca una red.

Separar las funciones da tres ventajas:
- **más rendimiento por parámetro:** cada red es chica y hace una sola cosa;
- **más velocidad:** las redes chicas se llaman muchas veces sin costo;
- **más confiabilidad:** el verificador y el guardián no dependen del que genera.

---

## 2. Las redes del sistema

| # | Función | Red | Tamaño (1060) | Se entrena con | Estado |
|---|---|---|---|---|---|
| N1 | **Lenguaje y código** (entender, razonar en palabras, redactar, programar) | Hope v1: bytes + SSM/atención + especialistas de 7 tipos + núcleo con vueltas (`HOPE_V1.md`) | 230M (semilla) → crece | texto, código y maestros | presets listos; falta entrenar |
| N2 | **Ver** (imágenes, documentos, capturas, diagramas) | codificador ViT preentrenado (SigLIP 2 base, licencia a verificar) + proyector + atención cruzada | ~90M + ~5M | descripciones verificadas de los tutores de visión | zona visual existe; falta el codificador preentrenado |
| N3 | **Leer texto en imágenes** (OCR) | Tesseract como herramienta; red propia chica destilada de Florence-2 (MIT) si hace falta | — / ~20M | OCR de Florence-2 verificado | pendiente |
| N4 | **Recordar y buscar** (notas, libros, kNN, conversaciones) | codificador de recuperación chico (bi-encoder) entrenado por contraste; mientras tanto, `HashingEmbedder` sin entrenar | ~20–50M | pares pregunta–pasaje de libros, Wikipedia y notas | `hope/memory/embed.py` (hash) existe; `ModelEmbedder` pendiente |
| N5 | **Decidir qué hacer** (tipo de pedido, zona, agente, herramienta, presupuesto de tiempo) | router / orquestador: clasificador chico sobre el pedido; la TLM elige herramientas | ~5–20M | decisiones que funcionaron (registro del maestro de §11.4) | TLM existe; router por zonas pendiente |
| N6 | **Verificar** (¿está bien esto?) | **verificador separado**: modelo de recompensa por paso + verificadores deterministas (tests, calculadora, unidades, fuentes, CRE/HCRI) | ~50–100M | correcciones confirmadas del tutor, errores detectados, pares DPO | CRE/HCRI existen; red verificadora pendiente |
| N7 | **Saber cuándo no sabe** (incertidumbre y calibración) | cabeza de incertidumbre de Hope (predice su propia pérdida) + calibración | ~0 (cabeza) | la pérdida real | cabeza existe; falta usarla para decidir buscar o preguntar |
| N8 | **Resolver problemas estructurados** (grillas, restricciones, rompecabezas) | red recursiva estilo TRM entrenada por nosotros, como herramienta | ~7M | rompecabezas generados con solución | pendiente |
| N9 | **Recordar a largo plazo** | grafo de conocimiento (Hebbiano), Notebook con repaso espaciado, base de conocimiento, caché de 1 GB y scratchpads | — (no son pesos) | uso | existen; scratchpads y registro pendientes (§11.10) |
| N10 | **Mejorarse** (diagnóstico de capacidad, crecer y podar) | reglas de diagnóstico + sondas; con experiencia, un clasificador aprendido | — / chico | registro de diagnósticos y resultados | reglas y operaciones existen; aprendizaje pendiente |
| N11 | **Seguridad y límites** | **guardián determinista**: reglas, aprobaciones humanas y auditoría encadenada. **No es una red, a propósito**: no se puede «convencer» ni entrenar en contra | — | — | existe |
| N12 | **Voz** (opcional, para la app) | reconocimiento de voz y síntesis con modelos abiertos chicos (licencia a verificar en cada uno) | ~40–250M | — | opcional, a decidir |

**Herramientas deterministas** (§11.16): todo lo que tiene una única respuesta correcta lo hace un programa, no una red.

---

## 3. Cómo trabajan juntas en una respuesta

1. **N5 (router)** lee el pedido y decide: ¿es simple o complejo?, ¿qué zona, agente y herramientas?, ¿cuánto tiempo hay?
2. **N7 (incertidumbre)** marca si hace falta buscar o preguntar.
3. **N4 (recuperación)** trae notas, pasajes de libros y datos del kNN. **N2/N3** procesan las imágenes si las hay.
4. **N1 (Hope)** razona y escribe, con las zonas y las vueltas del núcleo que hagan falta. Llama a herramientas y a **N8** para los problemas estructurados.
5. **N6 (verificador)** revisa con tests, fuentes y su red propia. Lo que falla vuelve a N1 con la crítica.
6. **N11 (guardián)** controla cada acción con efecto: red, archivos, ejecución.
7. **N9** guarda lo útil en la memoria de largo plazo. **N10** registra qué funcionó para mejorar.

El equipo de 6 agentes (§11.4) es la forma de organizar estos pasos cuando el pedido es complejo.

---

## 4. Cuánto ocupa en la 1060

| Pieza | VRAM aproximada |
|---|---|
| N1 a 4 bits (230M → hasta ~1–2B) | 0,15–1 GB |
| N2 (visión) | ~0,2 GB |
| N4 + N5 + N6 + N8 | ~0,2–0,3 GB |
| Caché de 1 GB + scratchpads | ~1 GB |
| Contexto de CUDA y buffers | ~0,8 GB |
| **Total** | **~2,5–3,5 GB de 6 GB** |

Sobra lugar para que N1 crezca. Las redes chicas responden en milisegundos.

---

## 5. Orden de construcción

1. **N1:** proxy y semilla de Hope v1 (`HOPE_V1.md` §5).
2. **N6 (verificador):** primero los deterministas que ya existen; después la red propia con los datos del tutor. Es la pieza que más rinde (§11.4).
3. **N4 (recuperación entrenada):** hace útiles la memoria, los libros y las notas.
4. **N5 (router)** y **N7 (incertidumbre)** con los registros de uso.
5. **N2 con el codificador preentrenado** y **N3**.
6. **N8 (TRM de rompecabezas)** y **N10 aprendido**.
7. **N12 (voz)**, si se decide.

Cada red sigue el ciclo de `MAPA_REDES.md` §5: medir, probar en chico con 2 semillas, integrar solo si gana, documentar.

**Reglas que no cambian:**
- **El verificador (N6) se entrena por separado de N1:** ninguna red se aprueba a sí misma.
- **El guardián (N11) no es una red** y ninguna red puede modificarlo.
- Los pesos ajenos entran solo con licencia permisiva y quedan registrados (§11.15 J1).
