# Mapa de redes de Hope: qué red hace cada cosa y qué modelo se usa

Decisión del usuario, 2026-11-10. Resume en un solo lugar las decisiones de `ARQUITECTURA_HN1.md` §11.2–11.16.

**Objetivo inmediato:** el mejor rendimiento **ejecutado en la GTX 1060**. El tamaño final queda abierto y se elige midiendo.

**Regla general:**
- **La red** decide, razona y redacta.
- **Las herramientas** hacen todo lo determinista (§11.16).
- **Los maestros** enseñan por datos (textos, correcciones y ejercicios verificados), nunca fusionados con Hope.
- **Los módulos donados** (pesos ajenos) entran solo con licencia permisiva y quedan registrados (§11.15 J1).

Lo marcado **«a decidir midiendo»** se resuelve en la ablación de CPU con 2 semillas y el mismo cómputo antes de la semilla de la 1060.

---

## 1. Dentro de Hope: una red para cada función

| Función | Componente | Arquitectura | Punto de partida | Dónde se entrena |
|---|---|---|---|---|
| Leer y escribir bytes | **Parte local** (codificador y decodificador de bytes) | parches por entropía (medido: mejor que tokens, 2,43 contra 2,56 BPB) + embeddings de n-gramas con hash (§11.2 A1) | desde cero: no hay donantes de bytes con licencia y forma compatibles | 1060 (semilla) |
| Lenguaje y estructura comunes | **Tronco global** compartido | **a decidir midiendo**: (a) SSM + atención propio, o (b) tronco «bytificado» de un donante (§11.15 I2) | (a) desde cero; (b) **SmolLM2-360M** o **Qwen2.5-0.5B** (Apache-2.0, verificar en la ficha antes de usar) | 1060 (semilla); con (b), primero congelado |
| Pensar varias vueltas | **Núcleo recurrente** | **receta TRM** (recursión de latente y respuesta, supervisión en cada vuelta) + profundidad por parche (Mixture-of-Recursions) | desde cero: se usa la receta, no los pesos | 1060 (semilla) |
| Especialidades | **Zonas**: lenguaje, código, razonamiento, conocimiento y visión-lenguaje | MoE fino con experto compartido y router jerárquico (§11.13) | expertos reciclados de los MLP del donante (I3) o desde cero, **a decidir midiendo** | ramas en 1060 + A770, mezcla en A770 |
| Conocimiento memorizado | **Capas de memoria** (claves de producto) | muchas filas, pocas leídas por parche (§11.14 X1.1) | desde cero | ramas |
| Conocimiento consultable | **Memoria externa** (no son parámetros) | kNN-LM (medido: −17 % de BPB), índice de libros, Wikidata local, notas, caché de 1 GB | — | — |
| Ver imágenes | **Codificador visual** + proyector + atención cruzada | ViT preentrenado congelado al principio | **SigLIP 2 base** (~90M, Apache-2.0, verificar en la ficha); SigLIP so400m (Apache-2.0, ya en `tutores_vision.yaml`) es más fuerte pero pesa ~4× | proyector y atención cruzada en las ramas |
| Roles de trabajo | **Adaptadores LoRA** de los 6 agentes (§11.4) | rango 4–8, sin tocar K/V (§11.10) | desde cero | postentrenamiento |
| Elegir zona y herramienta | **Router de zonas** y **TLM** (selector de herramientas) | propios | desde cero | mezcla y currículo de herramientas |

**Escalera de hardware (§11.20):** 1060 → A770 → clúster → **ejecutar en la 1060**, con poda y destilación al final. **Destino (§11.19):** ~4B con bucles: núcleo TRM y Mixture-of-Recursions para la profundidad, capas de memoria para el conocimiento, y módulos donados y zonas. Solo si el proxy de 25M rinde como un denso de ~100M. **Tamaño:** se elige midiendo. En la 1060 se puede **ejecutar** hasta ~1B a 4 bits y **entrenar** hasta ~490M.

---

## 2. Maestros: quién enseña cada área

Todos se descargan y alternan desde la app (`configs/catalogo_maestros.yaml`, `hope/app/maestros.py`). Solo se entrena con salidas de modelos del grupo **permitido**.

| Área | En la 1060 (ahora) | Siguiente escalón (A770) | Licencia |
|---|---|---|---|
| Lengua es/en, conversación, tutor de lectura | **Qwen3-4B**, Phi-4-mini | Qwen2.5-7B, OLMo 2 7B | Apache-2.0 / MIT |
| Código | **Qwen2.5-Coder-1.5B**, Qwen3-4B | **Qwen2.5-Coder-7B** | Apache-2.0 (el Coder **3B no** es Apache) |
| Razonamiento y matemática | DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-4B (con pensamiento) | Qwen3 más grande (verificar licencia de la versión) | MIT / Apache-2.0 |
| Visión (describir, OCR, preguntas) | **Florence-2-large** (OCR y descripciones), SmolVLM2-2.2B | **Qwen2.5-VL-7B** | MIT / Apache-2.0 |
| Verificación cruzada de un maestro | el otro maestro del turno | — | — |

**Notas:**
- **Nemotron Nano 4B** (el que usa el usuario) es base Llama 3.1 con la licencia de NVIDIA: queda en **«confirmar»**. Se puede usar como **referencia de comparación** en el banco. Para entrenar con sus salidas hay que leer y aceptar su licencia primero.
- **En la 1060, un maestro de 4B a 4 bits (~2,5 GB) no convive con el entrenamiento** (6 GB). Se alternan por turnos, por ejemplo con el maestro de noche (`hope/app/turnos.py`), o el maestro corre en la CPU (más lento).
- **Escalera** (§11.11 A1): se sube de escalón cuando el tutor casi no encuentra errores que sepa corregir, cuando Hope alcanza al maestro en un área, o cuando las correcciones del maestro se refutan seguido.

---

## 3. Fuera de la red: herramientas y módulos sin entrenar

| Tipo | Qué | Licencia |
|---|---|---|
| Cálculo y matemática | `calc`, `unidades`, `resolver`, `derivar`, `integrar`, `matriz`, `estad`, `ode`, `ajuste` (existen) + primos, combinatoria, probabilidad, geometría, programación lineal y grafos (§11.16) | SymPy, SciPy, NetworkX: BSD |
| Lógica | `logica` + SAT/SMT y grillas | z3: MIT |
| Código | `python`, `pruebas`, `lint`, `compilar`, `java`, `regex`, `validar`, `diff`, `git_leer`, editor (existen) + formatear, tipos, AST, cobertura, parches, índice de símbolos | ruff (MIT), mypy (MIT), coverage (Apache-2.0) |
| Texto y fechas | herramientas de lengua es/en (19, en la rama de lenguaje) + contar, ordenar, sílabas, números ↔ letras, fechas, husos y feriados | biblioteca estándar |
| Documentos | PDF, DOCX, XLSX y CSV, conversión de formatos, OCR | pypdf (BSD), openpyxl (MIT), Tesseract (Apache-2.0) |
| Datos fijos | constantes CODATA (existe), **Wikidata local**, tabla periódica | Wikidata: CC0 |
| Web y libros | `buscar_web`, `leer_web`, `buscar_wiki`, ⟦buscar_libro⟧ | según la fuente (§11.8) |
| Rompecabezas | **TRM entrenado por nosotros** para grillas, Sudoku y ARC, como herramienta (§11.15 I5) | propio |
| Electrónica, 3D y binarios | `circuito`, `resistencias`, `gcode`, `stl`, `marlin`, `material`, `hex`, `desensamblar`, … (existen) | — |

---

## 4. Lo que no se usa

- **Pesos o salidas con condiciones:** Llama y Gemma, apagados por defecto; Qwen2.5-3B, Qwen2.5-Coder-3B y Qwen2.5-VL-3B (licencia de investigación); Qwen2.5-VL-72B (licencia propia); YOLO de Ultralytics (AGPL).
- **Servicios sin API o con términos que prohíben automatizar:** Google Lens.
- **Datos:** bibliotecas piratas; libros con derechos (salvo que el usuario active `libros_propios`); código GPL salvo decisión del usuario; commits con coautoría de IA.
- **Unir modelos de lenguaje completos con otros tokenizadores por estados internos:** es frágil. Para aprender de ellos están los maestros; para usarlos, las herramientas.

---

## 5. Ciclo de desarrollo (cómo se trabaja en Hope desde ahora)

Cada mejora sigue el mismo ciclo:

1. **Medir dónde está:** banco sellado (código, conversación, factual, herramientas, visión, contexto) contra el checkpoint anterior y contra Nemotron Nano 4B como referencia, con el mismo tiempo de respuesta.
2. **Diagnosticar:** `hope/growth/diagnosis.py` decide si el problema es de datos, herramienta, memoria, estrategia o capacidad (§11.11 A2).
3. **Proponer** la mejora más barata que ataque esa causa.
4. **Probar en chico:** proxy de 25M en CPU, 2 semillas, mismo cómputo.
5. **Integrar si gana:** si no gana, queda apagada en el código y registrada.
6. **Documentar:** `docs/resultados/` y esta hoja de ruta.
7. **Escalar:** semilla en la 1060 → ramas → mezcla. Los cambios estructurales de más del 10 % requieren aprobación del usuario.
8. **Podar y recrecer** al cerrar cada fase (§11.18): LTH para encontrar lo redundante, poda **estructurada** para sacarlo, autodestilación para recuperar, y el presupuesto liberado va a donde el diagnóstico muestra falta de capacidad.

**Primer ciclo (desde el miércoles):** núcleo TRM y Mixture-of-Recursions, tronco desde cero contra bytificado, Ola A y µP, todo en el proxy de 25M. En paralelo: elegir el codificador visual, las herramientas T1 y el banco contra Nemotron.
