# Traspaso de Hope a la sesión local (2026-11-25)

Todo lo que veníamos planeando, para seguir desde tu máquina. Hope está **en pausa** hasta que vos digas. El recordatorio automático quedó desactivado y no hay nada corriendo.

---

## 1. Dónde está todo

- **Repositorio:** `adriangrochi-code/hope-0.1`
- **Rama principal de trabajo:** `ccr-976b883c-06ef2m` (último commit `213f2e9`, más este traspaso)

```bash
git clone https://github.com/adriangrochi-code/hope-0.1.git
cd hope-0.1
git checkout ccr-976b883c-06ef2m
pip install -e .[herramientas]        # extras opcionales: scipy, matplotlib, capstone, Pillow
python -m pytest -q tests/test_api.py # 25 pruebas de hope.api
```

**Ramas de las sesiones en la nube** (trabajo subido y en pausa; integrarlas a la rama principal con `git merge`):

| Rama | Último commit | Contenido | Falta |
|---|---|---|---|
| `ccr-976b883c-06ef2m-lenguaje` | `74b7cda` | escritura, MBR, kNN-LM (−17 % BPB), herramientas de lengua es/en | bancos de lengua y repetición, informe |
| `ccr-976b883c-06ef2m-codigo` | `f76f75c` | trazas, FIM, gramática Python/JSON, autodepuración, editor, banco de código | ajuste trazas + FIM sobre checkpoint nuevo |
| `ccr-976b883c-06ef2m-contexto` | `2dcf9b6` | YaRN, KV int8/int4, GQA + ventana, Quest, currículo largo, banco de contexto | experimento en CPU con modelo entrenado |
| `ccr-976b883c-06ef2m-tokens` | `906417a` | comparación bytes contra tokens (parcial, 1 semilla: bytes con entropía 2,43 BPB, tokens Qwen 2,56) | BPE, Nemotron y segunda semilla |
| `ccr-976b883c-06ef2m-vision` | `90fecab` | banco visual, mosaicos, contrastiva, tutores de imágenes | 5 de 12 corridas de la ablación |
| `ccr-976b883c-06ef2m-app` | `9eee24f` | app (Hope.exe, automático, maestros, charla), pantalla de tutores de visión | conectar `distill.py` («Qwen lee junto») |

---

## 2. Qué leer, en este orden

1. **`docs/PLAN_HOPE_2BITS.md`** — el plan actual completo (núcleo en 2 bits que crece, post-entrenamiento, el conjunto de 1 TB).
2. **`docs/API_HOPE.md`** — `hope.api` v0.1.0: las herramientas que funcionan con cualquier modelo base.
3. **`docs/ARQUITECTURA_HN1.md`**, sección 11 (§11.2–11.30): la hoja de ruta con todas las decisiones.
4. **`docs/investigacion/`**:
   - `arquitecturas_por_tarea.md`: qué red rinde más en cada tarea de código;
   - `arquitectura_crecimiento.md`: cómo crecer sin rehacer;
   - `optimizacion_practica.md`: A770, 1060 y CPU, y la red base al detalle;
   - `ternario_1_58.md`: ternario contra 2 bits.
5. **`docs/SISTEMA_HOPE.md`** (redes por función), **`docs/MAPA_REDES.md`** (qué modelo para qué) y **`docs/RED_DISTRIBUIDA.md`** (proyecto aparte, posterior).

---

## 3. Decisiones tomadas (resumen)

**Objetivo:** ejecución local, centrada en **código e ingeniería** (electrónica, impresión 3D, firmware, mallas, BIOS), útil en varias áreas, usando herramientas para todo lo exacto.

**Núcleo «Hope Local»:**
- **Modelo base:** MoE de ~30–35B parámetros totales con ~3B activos.
  - Empezar con **Qwen3-Coder-30B-A3B** (Apache-2.0).
  - Destino probable: **Qwen3.5-35B-A3B** (Apache-2.0; híbrido Gated DeltaNet + atención, 256 expertos finos, 8 + 1 compartido).
  - Qwen 4, si sale. Releer la licencia en la tarjeta oficial antes de descargar cualquiera.
- **Cuantización:** **2 bits de 4 niveles** (no ternario; el ternario queda solo como variante para CPU).
- **Ajuste:** en la **Arc A770**, capa por capa, con destilación desde el original. Usar PyTorch XPU oficial (IPEX quedó archivado) y sin GradScaler.
- **Ejecución en la GTX 1060:**
  - en la VRAM: atención, router, experto compartido, caché KV a 4 bits y una **caché caliente** con los expertos más usados;
  - en la RAM: el resto de los expertos, calculados por la CPU (AVX2);
  - kernels: W2A8 con `dp4a`; nunca fp16;
  - software: CUDA 12.9, driver 580, PyTorch 2.14+cu126.
- **Cuello de botella: la RAM.** Pasar a 32 GB permite crecer ~50B parámetros más de expertos en 2 bits.

**Post-entrenamiento:**
1. **Recuperar lo perdido al cuantizar:** destilación con datos amplios y tu conjunto.
2. **Especializar:** SFT y RL con tests, en adaptadores LoRA o en expertos nuevos.
3. **Usar herramientas:** entrenar con trazas de uso verificadas.

En los pasos 2 y 3 va un 5–25 % de datos de repaso. Si un área cae más de 1 %, se vuelve atrás.

**Tu conjunto de ~1 TB** (~250B tokens, mucho más de lo que procesa la A770):
1. inventario;
2. revisión de licencia, origen y si fue generado por un modelo, más quitar secretos y datos personales;
3. deduplicación;
4. clasificación en niveles de calidad;
5. ~5–10B tokens del mejor nivel para recuperar;
6. lo mejor de código e ingeniería para especializar;
7. el resto, a un índice de búsqueda.

**Crecimiento sin rehacer:**
- expertos nuevos en 2 bits, con los viejos congelados;
- adaptadores LoRA por habilidad;
- redes chicas separadas: verificador, recuperación, router, visión, OCR y TRM de rompecabezas;
- herramientas certificadas, memoria, búsqueda y el modo recursivo RLM para material largo.

Cada pieza entra solo si mejora sin empeorar otra área más de 1 %.

**La Hope de bytes** (presets `hope-v1-proxy` y `hope-v1-semilla` en `hope/core/hn1.py`) queda como línea de investigación de largo plazo.

---

## 4. Lo que ya está hecho

- **`hope.api` v0.1.0** (`hope/api/`, `tests/test_api.py`):
  - **siete herramientas con certificado:** intervalos, equivalencia simbólica, raíces, integrales, LP con dual, SMT con modelo comprobado y grafos;
  - **el catálogo completo de Hope** bajo la misma interfaz, pasando por el guardián;
  - **llamadas en cualquier sintaxis:** `⟦⟧`, `<tool_call>` en JSON, el XML de Qwen3-Coder, bloques JSON y `tool_calls` de OpenAI;
  - esquemas OpenAI, búsqueda de herramientas, un bucle que funciona con cualquier modelo, y el crítico;
  - **sin `eval`**: lista blanca sobre el AST.
- **Todo lo anterior del proyecto:** runtime, memoria, guardián, herramientas, app y harness F9.

**Mediciones en CPU:**
- Muon + WSD: −0,17 BPB juntos.
- kNN-LM: −17 % de BPB.
- Hope-mini: ~1,88 BPB en español.
- Modelo de 25M: 1,975 BPB en español y 1,296 en código.
- Generación: entre 2 y 5,1 veces más rápida con caché incremental más especulativa.

---

## 5. Próximos pasos (en orden)

1. **Etapa 1, núcleo 2 bits** con Qwen3-Coder-30B-A3B (§11.28, M1–M5):
   - ruteo GPU/CPU por token;
   - caché caliente de expertos;
   - kernels W2A8 `dp4a` (1060) y AVX2 de 2 bits (CPU), probados primero en CPU;
   - ajuste capa por capa en la A770;
   - medición contra un 14B denso y un 8B en 2 bits.

   Necesita la A770 y la 1060.
2. **Etapa 2, herramientas y verificador:**
   - el resto del cálculo certificado (uncertainties, holidays con Argentina, periodictable, fluids, shapely, statsmodels, CoolProp, python-control, scikit-rf, symengine, python-flint);
   - integrar `exactas_worker.py` de TISMA (repo `adriangrochi-code/antivirus`, rama `ccr-8ab6bd04-4px8f0`, `userspace/tisma-code/tisma_code/exactas_worker.py`, con 36 pruebas en `userspace/tisma-code/tests/test_exactas.py`);
   - herramientas de ingeniería: compilar firmware, ngspice, CadQuery, lector de hojas de datos, Klipper;
   - banco de ingeniería.
3. **Integración con TISMA Code** (§11.24). Ya hecho: T1 (`hope.api`). Falta:
   - T2: `hope_modelo.json`;
   - T3: `hope serve`;
   - T4: doble sintaxis de llamadas y trazas;
   - T5: caché por prefijo y sesión;
   - T6: medición A-B-B-A.

   **Interfaz que TISMA ya implementó de su lado:**
   - arranque: `python -m hope serve --host 127.0.0.1 --port N --api-key K --modelo DIR --dispositivo {cuda|xpu|cpu} --contexto-bytes C`;
   - `Bearer` obligatorio en `/v1/*`; `/health` sin clave;
   - el id del modelo empieza con «hope»;
   - la sesión llega en `user` y en `X-Hope-Session`, como 32 caracteres hex;
   - `usage` en tokens equivalentes (bytes / 4);
   - cachés solo en `XDG_CACHE_HOME`;
   - `hope_modelo.json` en versión 1, con `licencia_pesos`.
4. **Etapa 3:** modo RLM (§11.29), recuperación entrenada (granite-embedding-small-r2; verificar licencia) y memoria.
5. **Etapa 4:** expertos nuevos de ingeniería y de ciberseguridad defensiva en 2 bits.
6. **Etapa 5:** probar Qwen3.5-35B-A3B como núcleo (kernel Gated DeltaNet sin Triton).
7. **Tu conjunto de 1 TB:** inventario y primera selección (§4.2 de `PLAN_HOPE_2BITS.md`). Para eso hace falta saber qué contiene, en qué proporción, qué parte es tuya y si hay algo generado por modelos y de cuáles.

---

## 6. Reglas que no cambian

- **Guardián:**
  - es independiente y determinista;
  - Hope no puede cambiar sus límites, la seguridad ni la auditoría;
  - las acciones desconocidas se deniegan;
  - la red, los maestros remotos y la publicación requieren tu aprobación;
  - el registro de auditoría está encadenado por hash;
  - los cambios de más del 10 % de los parámetros y el techo de parámetros los aprobás vos (`guardian.cli`).
- **Web:** solo lectura (GET y HEAD). Nada de formularios ni de logins, y nunca hacerse pasar por una persona.
- **Licencias:**
  - entrenar solo con datos y pesos de licencia permisiva;
  - nada de NC ni ND;
  - nada generado por modelos cuyos términos lo prohíben (GPT, Claude);
  - commits con coautoría de IA fuera de los datos de entrenamiento;
  - Llama y Gemma, apagados por defecto;
  - Nemotron solo como referencia, salvo que aceptes su licencia;
  - GPL apagado por defecto.
- **Ciberseguridad:** solo defensiva. Nada de malware funcional, exploits armados, bases filtradas ni kits de phishing.
- **Claves:** solo en variables de entorno, nunca en el repo.

---

## 7. Para pegar en la sesión local

> Retomo Hope desde la rama `ccr-976b883c-06ef2m` de `adriangrochi-code/hope-0.1`. Leé primero `docs/TRASPASO_SESION_LOCAL.md` y `docs/PLAN_HOPE_2BITS.md`, y después las secciones 11.24–11.30 de `docs/ARQUITECTURA_HN1.md` y `docs/API_HOPE.md`. Prioridad: etapa 1 del plan (núcleo MoE en 2 bits con Qwen3-Coder-30B-A3B, expertos en RAM con caché caliente en la GTX 1060, ajuste en la Arc A770). En paralelo: herramientas certificadas restantes y `hope serve` según la interfaz acordada con TISMA. Respetá las reglas de la sección 6. Antes de descargar cualquier modelo, verificá su licencia en la tarjeta oficial.
