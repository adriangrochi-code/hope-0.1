# Control de calidad (rama `ccr-976b883c-06ef2m-calidad`)

Fecha: 2026-11-09. Máquina: contenedor en la nube, 4 núcleos, solo CPU, PyTorch 2.15 (CPU), Python 3.13.

## Duración de la batería

| Corrida | Pruebas | Tiempo de reloj |
|---|---|---|
| Antes, `pytest -q` (sobre `6025523`, con `hope/data` ya subido) | 219 pasan | **3 min 59 s** |
| Después, `pytest -q` | 228 pasan (219 + 8 de regresión + 1 que salió de partir una prueba) | **3 min 10 s** |
| Después, `pytest -q -m "not slow"` | 224 pasan, 4 excluidas | **1 min 55 s** (el objetivo era < 3 min) |

`pytest -q` sigue corriendo todo. El marcador `slow` está registrado en `pyproject.toml`
(`[tool.pytest.ini_options] markers`).

### Pruebas lentas: qué se hizo con cada una

| Prueba | Antes | Después | Cambio |
|---|---|---|---|
| `test_council.py::test_routing_learns_which_agents_work_per_topic` | 13,2 s | < 0,5 s | Sin cambiar la prueba. Casi todo el tiempo era `analyze()` lanzando un proceso sympy (~0,7 s) por cada crítica del **mismo** problema. Ahora el resultado se recuerda (ver error 6). Ayuda también al consejo, al presupuesto de velocidad y al ciclo de enseñanza. |
| `test_rlvr.py::test_grpo_increases_reward` | 15,4 s | 8,3 s | 10 pasos intermedios en vez de 25, con `lr` 1e-2 en vez de 3e-3. Misma semilla (0) y mismo umbral (+0,1). La mejora medida es +0,52, contra +0,29 antes: queda más margen. |
| `test_cycle.py::test_teacher_budget_is_respected` | 14,5 s | 10,2 s | 3 pasos de consolidación en vez de 30. La prueba revisa el límite de llamadas al maestro, no lo que se aprende. |
| `test_cycle.py::test_strict_criterion_keeps_teaching` | 13,0 s | 10,6 s | Igual: 3 pasos. Con 3 pasos el modelo tampoco llega al 99 %. |
| `test_cycle.py::test_answers_are_reviewed_and_failures_carry_the_critique` | 6,9 s | 4,1 s | Igual: 3 pasos. |
| `test_vision.py::test_trainer_learns_to_use_the_image` | 24,7 s | 27,7 s | `@pytest.mark.slow`. Con 200 o 150 pasos la diferencia baja de 0,27 a 0,01, así que necesita los 400. La parte del checkpoint de un modelo que creció pasó a otra prueba, `test_grown_vision_checkpoint_resumes`, que no está marcada y corre siempre. |
| `test_cycle.py::test_full_cycle_with_mock_teacher` | 22,1 s | 22,1 s | `@pytest.mark.slow`. Comprueba que la consolidación baja la pérdida y que hay cambio de fase: necesita los 30 pasos. |
| `test_continual.py::test_replay_limits_forgetting` | 15,4 s | 14,7 s | `@pytest.mark.slow`. Comprueba que hay olvido y que el repaso lo reduce, así que necesita entrenar. |
| `test_lowprec.py::test_low_precision_learns_like_fp32_with_much_less_state` | 11,4 s | 12,4 s | `@pytest.mark.slow`. Entrena dos veces (fp32 y bf16-8bit) para compararlas. |

Bajar los hilos de PyTorch (`OMP_NUM_THREADS=1`) no cambia el tiempo: lo que pesa es la generación byte a byte en Python.

## Errores encontrados y cómo se corrigieron

Cada corrección tiene su prueba en `tests/test_calidad.py`. Todas esas pruebas fallaban antes del arreglo.

0. **`hope/data/` no estaba en git** (bloqueante). La regla `data/` de `.gitignore` también ignoraba
   `hope/data/`. En un clon limpio, 12 módulos de prueba no se podían ni recolectar, y fallaban otras 6
   pruebas (`ModuleNotFoundError: hope.data`). Avisé a la sesión principal. Subió el paquete y cambió
   la regla a `/data/` y `/runs/` (`6025523`). Esta rama parte de ese commit.

1. **GRPO con `group=1` escribía NaN en los pesos** (`hope/train/rlvr.py`). `torch.std` de un solo
   elemento da NaN, y `NaN < 1e-6` es falso, así que el grupo no se salteaba. La ventaja quedaba NaN y
   el paso de optimización llenaba los pesos de NaN. Pasaba lo mismo en `grpo_step_tools`. Ahora un
   grupo de menos de 2 muestras se saltea.
   Prueba: `test_grpo_group_of_one_is_skipped_instead_of_nan`.

2. **`GRPO.step` dejaba el modelo en modo evaluación** (`rlvr.py`). Pasaba lo mismo en
   `grpo_step_tools`. Se llamaba a `m.eval()` para generar y nunca se volvía al modo anterior. Un
   entrenamiento normal que siguiera después (p. ej. en el ciclo) corría con el núcleo convergente en
   modo inferencia: profundidad fija y sin muestreo de iteraciones. Ahora se restaura el modo original.
   El paso de gradiente sigue en modo evaluación, igual que el muestreo y que la referencia.
   Prueba: `test_grpo_keeps_the_training_mode_of_the_model`.

3. **`completion_logprob` con prompt vacío solo contaba el último byte** (`rlvr.py`). Pasaba
   `[:, -1:]` en vez de `[:, 0:]`. Prueba: `test_completion_logprob_with_empty_prompt_counts_every_predicted_byte`.
   Además, `float(kl)` sobre un tensor con gradiente daba un aviso de PyTorch. Ahora se usa `kl.detach()`.

4. **`long_loss` daba el mismo peso a todos los segmentos** (`hope/core/hn1.py`). Si el último
   segmento era corto (p. ej. 16 bytes contra 1024), pesaba lo mismo que uno completo. Además, las
   estadísticas `ce`/`bpb` eran solo las del último segmento. Ahora la pérdida y `ce`/`bpb` son el
   promedio por byte de todo el texto. También, si se pasa `weights`, se corta por segmento: antes
   llegaba entero a cada segmento y fallaba por forma.
   Prueba: `test_long_loss_weights_each_segment_by_its_bytes`.

5. **`Notebook.write` no actualizaba el vector de búsqueda al cambiar una nota**
   (`hope/memory/knowledge.py`). La nota se seguía encontrando por lo que decía antes. Eso afecta a
   `lookup` aproximado, al contexto de la base y a los enlaces del grafo. Ahora se recalcula el vector
   en SQLite y en la matriz en memoria.
   Prueba: `test_notebook_new_version_updates_the_search_vector`.

6. **El enfriamiento WSD de `hope/train/run.py` era un diente de sierra.** La etapa con `enfriar: true`
   llamaba a `Trainer.train(..., decay=True)` en cada tramo de `checkpoint_min` minutos. Cada llamada
   bajaba la tasa de 1 a 0 dentro del tramo, y el tramo siguiente volvía a la tasa máxima. El código
   decía que la tasa bajaba «según la fracción de horas cumplidas», pero no lo hacía. Ahora
   `Trainer.train` acepta `decay_span=(desde, hasta)`. `run.py` lo calcula con `cooldown_span(horas
   hechas, minutos, horas)`, así que la tasa baja una sola vez a lo largo de toda la etapa. Sin
   `decay_span` el comportamiento es el de antes (`hope/train/continual.py`).
   Prueba: `test_cooldown_lr_follows_the_hours_not_each_chunk`.

7. **`find_batch_size` devolvía 1 sin probarlo** (`hope/core/fast.py`). Pasaba cuando ni el lote
   inicial entraba en memoria. Ahora prueba lotes cada vez más chicos (la mitad cada vez) hasta
   encontrar uno que entre. Prueba: `test_find_batch_size_goes_below_the_start_when_it_does_not_fit`.

8. **Rendimiento: `analyze()` relanzaba el sandbox de sympy para el mismo problema**
   (`hope/constraints/solver.py`). La crítica analiza el mismo enunciado para cada candidato y cada
   revisión, a ~0,7 s por proceso. Esto también gastaba el presupuesto de tiempo de
   `hope/runtime/budget.py`. Ahora hay un caché (hasta 1024 entradas) con el resultado según los datos
   y las leyes. Los fallos no se guardan, y cada uso lee el JSON de nuevo, así que nadie comparte listas
   que después se modifican. El resultado es el mismo y solo cambia el tiempo.
   Prueba: `test_constraint_analysis_reuses_the_sandbox_result`.

## Revisado sin cambios

- `hope/core/speculative.py`: la máquina de parches del borrador, la aceptación, `truncate_cache`, y
  `snapshot`/`restore` de `GlobalCache` coinciden con `generate`. Las pruebas de igualdad de
  `tests/test_speed.py` lo cubren. `ngram_draft` no puede devolver el propio sufijo.
- `hope/core/hn1.py`: `GlobalCache` (los estados de los bloques se reemplazan y nunca se modifican en
  el lugar, así que copiar los diccionarios alcanza), `generate_long` y `set_context_scale` (fórmula
  NTK correcta para la `rope` de `layers.py`, cuyas frecuencias son `base^(-i/half)`).
- `hope/train/muon.py`: Newton–Schulz, momento de Nesterov, escala de Moonlight y AdamW para lo que no
  es matriz.
- `hope/runtime/budget.py`: estimaciones por agente y nivel, y desempates. Se beneficia del caché del
  punto 8.
- `hope/memory/graph.py`: `expand` ya excluye lo refutado y las contradicciones.

## Estático

`ruff check --select F,E9 hope tests guardian` → sin errores.

## Segunda ronda (sobre `0f32b33`: lote-cache, fp32-8bit, optimizador en RAM, hardware)

Batería sobre el resultado integrado más estas correcciones: **252 pasan en 3 min 16 s**;
`-m "not slow"`: **248 pasan en 2 min 07 s**. `ruff check --select F,E9` sin errores.

9. **Retomar un checkpoint `fp32` en `bf16-8bit` o `fp32-8bit` fallaba con `KeyError: 'm_q'`**
   (`hope/train/lowprec.py`). Es el caso de entrenar en CPU con `fp32` y seguir en la GTX 1060 o
   la A770. `_load_moments` aceptaba `m`/`v` y `m_q`/`v_q`, pero no `exp_avg`/`exp_avg_sq`, que es
   el formato de `torch.optim.AdamW`, y `step` venía como tensor. Ahora acepta los tres formatos y
   los momentos siguen en lugar de volver a cero.
   Prueba: `test_low_precision_resumes_a_fp32_adamw_checkpoint`.

10. **Muon perdía precisión al retomar** (`hope/train/muon.py`). `Optimizer.load_state_dict`
    convierte el estado al tipo del parámetro, así que el momento fp32 volvía como bf16.
    `LowPrecisionAdamW` lo evitaba con su propio `load_state_dict`, pero Muon no lo tenía. Ahora
    ambos usan `load_state_exact`, que copia el estado sin convertirlo. Con `offload`, lo deja en
    la CPU en vez de subirlo entero a la GPU al cargar: esa parte no se pudo probar en una GPU real
    en este contenedor, porque solo tiene CPU.
    Prueba: `test_muon_resume_keeps_its_state_in_fp32`.

11. **`calibrate` nunca reconocía la placa** (`hope/bench/hardware.py`). Comparaba palabras del
    nombre del perfil con las del controlador («NVIDIA GeForce GTX 1060 6GB», «Intel(R) Arc(TM)
    A770 Graphics») y no coincidía ninguna. Toda máquina calibrada heredaba el perfil de CPU: 40 GB/s
    de ancho de banda en vez de 192 o 560, y `compile_ok` falso, así que la velocidad de generación
    estimada salía de 5 a 14 veces más baja. Ahora `match_profile` busca el número de modelo.
    Prueba: `test_calibration_recognizes_real_device_names`.

Revisado sin cambios:
- `hope/core/batched.py`: la máquina de parches por secuencia, cómo se cierra una secuencia que se
  detiene, `_stack`/`_split`/`_take` con relleno y posición por fila, y la agrupación de la parte
  global (los estados vacíos van aparte).
- `fetch_state`/`park_state`.
- El offload de Muon (el estado se trae antes de usarlo y se devuelve después).

Queda abierta una cosa que no corregí: pasar de un modo de 8 bits a `fp32` (AdamW de PyTorch) al
retomar tampoco funcionaría, porque AdamW no entiende `m_q`. No es el camino del plan (casa → A770
usa `fp32-8bit` → `bf16-8bit`). Si hiciera falta, `Trainer.resume` podría descartar el estado del
optimizador cuando el formato no coincide.
