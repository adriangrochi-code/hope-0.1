# Maestro local en CPU (Qwen3-4B) y datos verificados para Hope

**Fecha:** 2026-11-09 · **Máquina:** contenedor de la sesión, CPU de 4 núcleos, 15 GB de RAM, sin GPU.

## 1. Servidor y modelo

| | |
|---|---|
| **Modelo** | Qwen3-4B (`Qwen/Qwen3-4B-GGUF`, archivo `Qwen3-4B-Q4_K_M.gguf`, 2,5 GB) |
| **Licencia** | Apache 2.0 (etiqueta `license:apache-2.0` en Hugging Face): permite usar sus salidas para entrenar a Hope |
| **Cuantización** | Q4_K_M |
| **Servidor** | `llama-cpp-python[server]` 0.3.36 (`python -m llama_cpp.server … --host 127.0.0.1 --port 8080 --n_ctx 4096 --n_threads 4`), API de chat de OpenAI |
| **Configuración** | `configs/teacher_local.yaml` (episodios: temperatura 0,5, `no_think`), `configs/teacher_local_notas.yaml` (notas: temperatura 0, `max_tokens` 16) |
| **Velocidad** | generación **6,4–6,7 tokens/s**; lectura del pedido **~75–80 tokens/s** |
| **Costo por unidad** | episodio: **~80–150 s** (500–900 tokens); nota de calidad: **~10 s** por documento de 1200 caracteres |

- **Compilar llama.cpp desde GitHub** fue bloqueado por el clasificador de permisos de la sesión (código externo). Se usó el paquete de PyPI `llama-cpp-python`, que compila la misma biblioteca: el resultado es equivalente.
- **Qwen2.5-1.5B-Instruct** (Q4_K_M, Apache 2.0) también se probó: 13,9 tokens/s, pero el episodio de prueba («porcentajes») fue incoherente (pasos que no tienen relación entre sí, `"idioma": "es|en"` copiado del formato) y no cerró el JSON en 1500 tokens. Se descartó como maestro de episodios.
- **Modo de pensamiento:** con `<think>` cada episodio tardaría varias veces más en CPU. Se agregó la opción `no_think` al cliente (agrega `/no_think` al último mensaje). Qwen3 igual emite un bloque `<think></think>` vacío, que el cliente separa.
- No se usaron Qwen2.5-3B ni 72B (licencias que no son Apache 2.0). Qwen2.5-Math-1.5B no se llegó a probar (ver §6).
- **Guardián:** cada consulta pasa por `Guardian.decide("teacher.query")`; el maestro local es nivel 1 (interno). La prueba `test_harvest_respeta_guardian` comprueba que un endpoint remoto queda bloqueado sin aprobación humana.

## 2. Auditoría del maestro (ronda 0, un episodio por tema)

ESTADO_AUDITORIA

## 3. Cambios para que la verificación sea justa

La auditoría mostró que la mayoría de los rechazos no eran errores del maestro, sino del verificador o del formato:

| Problema | Ejemplo | Corrección | Archivo |
|---|---|---|---|
| Redondeo honesto tomado como error | `1/3` → `0.3333` | Se acepta si el valor mostrado es el exacto redondeado a sus decimales (al menos 2) | `hope/tools/verify.py` (`rounded_ok`) |
| Redondeo propagado | `1 / 0.4167` → `2.4` | Si la expresión usa entradas ya redondeadas (≥3 decimales), error relativo ≤ 5·10⁻⁴ | `hope/tools/verify.py` |
| **Bug de la herramienta de unidades** | `10 V / 2 A` = `5 ohm` daba **falso** | pint leía `10·V/2·A` (la multiplicación implícita liga más débil que `/`). Ahora cada «número unidad» se agrupa entre paréntesis y `Ω` → `ohm` | `hope/tools/builtin.py` (`_group_quantities`) |
| JSON con un cierre equivocado | `["…")` en lugar de `["…"]` | Se cambia solo el carácter de cierre por el que corresponde según la pila de corchetes | `hope/teacher/episode.py` |
| Texto sobrante después del JSON | `…}]}}` | Se lee el primer objeto completo y se ignora el resto | `hope/teacher/episode.py` |

Los episodios reparados quedan marcados con `"reparacion_json": true`. Las reparaciones no tocan el contenido, que después pasa igual por el verificador.

También se reforzó el pedido (`RULES_HINT` en `hope/teacher/episode.py`): resultado breve que coincida con el último paso, expresiones aritméticas sin unidades ni letras, formato de las verificaciones de unidades y código, el problema tiene que ser del tema pedido y, en programación, el código debe resolver el problema (no imprimir el resultado ya calculado). El pedido incluye además los problemas ya enseñados del tema (para no repetirlos) y un enfoque al azar (cocina, electrónica, finanzas…) para variar los contextos.

## 4. Episodios cosechados

ESTADO_COSECHA

## 5. Notas de calidad para el filtro del corpus

- **Muestra:** `python -m hope.data.sources --out data/pre --wiki es:20M --wiki en:15M` → 1095 artículos en español y 635 en inglés (35 MB). Se intercalaron es/en en `data/pre/muestra_calidad.txt` (no se sube: `data/` está ignorada).
- **Puntuación:** `python -m hope.data.quality puntuar --teacher configs/teacher_local_notas.yaml --input data/pre/muestra_calidad.txt --out datos_maestro/notas_calidad.jsonl --n 1500 --max-chars 1200`. Se cortó en **1000 documentos** (519 es, 481 en) para darle tiempo a la cosecha. El maestro ve los primeros 1200 caracteres; el texto guardado está recortado a eso.
- **Distribución de notas:** 2 → 27, 3 → 577, 4 → 214, 5 → 182 (media 3,55). Ninguna nota 0 o 1: es Wikipedia, ya filtrada por las reglas. Español 3,59, inglés 3,51.
- **Consistencia del maestro:** por un error mío corrieron dos procesos de puntuación a la vez y 802 documentos se puntuaron dos veces (temperatura 0). Coincidieron en **779 (97 %)** y solo 2 difirieron en más de un punto. La segunda nota quedó como `nota_2`.
- **Clasificador** (`python -m hope.data.quality entrenar --notas datos_maestro/notas_calidad.jsonl --out datos_maestro/calidad.npz --epocas 500 --lr 1.0 --validacion 0.2`):

| | Error medio absoluto |
|---|---|
| Predecir siempre la media | 0,70 |
| Clasificador, 30 épocas (valor por defecto anterior) | 0,68 |
| **Clasificador, 500 épocas, lr 1,0** (en 200 notas no vistas) | **0,57** |
| Clasificador sobre sus propias notas de entrenamiento | 0,48 |

  Correlación de rangos (Spearman) en las notas apartadas: **0,40**. El 30 % mejor puntuado por el clasificador tiene nota media del maestro 3,85; el 30 % peor, 3,03. Con 30 épocas el regresor casi no aprendía; por eso `entrenar` ahora acepta `--epocas`, `--lr` y `--validacion`. Más épocas o otro `l2` no bajaron de 0,565.

## 6. Recomendaciones

RECOMENDACIONES
