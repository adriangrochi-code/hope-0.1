# Trabajo en paralelo (sesiones y agentes)

Pedido del usuario: usar varias sesiones y agentes para acelerar el desarrollo de Hope.

**Cómo funciona:**
- Cada frente corre en una **sesión en la nube independiente**, con su propia máquina, sobre la rama `ccr-976b883c-06ef2m`.
- Cada sesión publica en su propia rama.
- La sesión principal revisa cada rama, corre las pruebas e integra en `ccr-976b883c-06ef2m`.

| Frente | Rama | Qué entrega |
|---|---|---|
| Preentrenamiento F1 (hn1-25m) en CPU | `ccr-976b883c-06ef2m-preentreno-25m` | Corpus real (Wikipedia es/en + código), corrida de varias horas con WSD, pesos livianos `checkpoints/hn1-25m-cpu.pt`, curvas en `docs/resultados/preentreno_25m.md` |
| Generación en lote y caché de conversación | `ccr-976b883c-06ef2m-lote-cache` | `hope/core/batched.py`: varias muestras por lectura de pesos (consejo, STaR, GRPO, mejor-de-N) y conversación sin releer el historial; salida idéntica a `generate` |
| Harness F9 (Hope + Qwen sobre carpeta local) | `ccr-976b883c-06ef2m-harness` | `hope/harness/`: consola tipo Claude Code, herramientas de carpeta a través del guardián, Hope y Qwen en paralelo, episodios verificados |
| Herramientas creadas por Qwen (TCG/TVE) y composición (HTO) | `ccr-976b883c-06ef2m-herramientas` | `hope/tools/creation.py`, `hope/tools/compose.py`: validación en sandbox, revisión de seguridad, aprobación humana para consolidar |
| Control de calidad | `ccr-976b883c-06ef2m-calidad` | Batería completa, pruebas lentas más rápidas (`-m "not slow"` en menos de 3 min), revisión de los módulos nuevos, `docs/resultados/calidad.md` |
| Maestro open source local (Qwen en CPU, Apache 2.0) | `ccr-976b883c-06ef2m-maestro-local` | llama.cpp + Qwen3/Qwen2.5 en GGUF, episodios verificados del currículo (`datos_maestro/`), notas de calidad y clasificador del corpus, `docs/resultados/maestro_local.md` |
| Aplicación con interfaz (1060 en casa → A770) | `ccr-976b883c-06ef2m-app` | `python -m hope.app`: calibración, datos, aprender el lenguaje, aprender del maestro, pasar a la A770, guardián; instaladores para Windows y Linux; `docs/APP.md` |
| Núcleo en bucle estable y GRPO corregido | `ccr-976b883c-06ef2m-bucle-grpo` | Recomendaciones 2 y 4 del informe de investigación: reinyección de la entrada, profundidad sin tope fijo, EMA; GRPO sin sesgos de normalización y con respuestas más largas; ablación con 2 semillas |
| Calidad en imágenes | `ccr-976b883c-06ef2m-vision` | Banco sellado de visión, datos verificables nuevos (circuitos, tablas, gráficos, OCR, relojes, geometría, relaciones espaciales), mosaicos, contrastiva, currículo; ablación con 2 semillas |
| Calidad en uso de herramientas | `ccr-976b883c-06ef2m-uso-herramientas` | Banco sellado de uso de herramientas, currículo con llamadas reales (elegir, encadenar, no usar, recuperarse de errores), selección guiada por TLM, refuerzo con costo |
| Búsqueda web y corpus de internet | `ccr-976b883c-06ef2m-web` | `⟦buscar_web⟧`/`⟦leer_web⟧` con permisos del usuario (`configs/web.yaml`), contenido no confiable neutralizado, caché, procedencia; corpus filtrado de la web |
| Qwen como tutor completo | `ccr-976b883c-06ef2m-tutor` | Tutoría on-policy: diagnóstico, intento → crítica → pista → reintento, variaciones dirigidas, notas en el cuaderno, evaluación de herramientas, currículo adaptativo |
| Investigación: razonamiento por parámetro (terminada) | (agente local) | `reports/Razonamiento por parámetro para Hope.md`: técnicas 2024–2026 con evidencia y un top 8 para Hope |
| Ablación en CPU y Hope-mini | rama principal | `docs/resultados/ablacion_cpu.md` y preentrenamiento de `hn1-cpu` con la configuración ganadora |

**Integración:**
- Cada rama se integra con `git merge`.
- Se corren las pruebas y se registra acá la fecha y el commit de integración.

## Registro de integración

| Fecha (UTC) | Rama | Commit de integración | Estado |
|---|---|---|---|
| 2026-11-09 19:00 | calidad (1.ª ronda) | 7f5b470 | 7 errores corregidos; batería de 219 a 228 pruebas |
| 2026-11-09 19:00 | lote-cache | 0f32b33 | generación en lote y conversación sin releer el historial |
| 2026-11-09 19:30 | calidad (2.ª ronda) | fddd5ee | retomar entre precisiones, Muon al retomar, calibración |
| 2026-11-09 19:37 | app | 129fdf9 | 7 pasos, instaladores, exportación cuantizada |
| 2026-11-09 19:45 | harness | (merge harness) | F9: consola sobre carpeta local, Hope y Qwen en paralelo, guardián |
| 2026-11-09 19:45 | herramientas | 11131ad | creación de herramientas con Qwen (validación en sandbox) y composición |
| 2026-11-09 19:50 | calidad (3.ª ronda) | 415c176 | la QAT entrena la misma grilla que se exporta; se apaga en etapas sin QAT |
| 2026-11-09 21:00 | maestro-local (parcial) | 53dcf33 | Qwen3-4B en CPU: 25 de 27 episodios verificados (92,6 %), 716 notas de calidad, `hope/teacher/harvest.py` |
| 2026-11-09 21:45 | bucle-grpo | 71e9ec7 | bucle estable del núcleo y GRPO corregido (ablación en curso) |
| 2026-11-09 21:50 | web | 3dfefa9 | búsqueda e inspección web solo lectura, fuentes legales con cuota diaria, Wikipedia local; 360 pruebas |
| 2026-11-09 22:10 | app (completa) | 43b664b | modo automático, disco aparte, celular, chat de control, Chatear; 387 pruebas |
| 2026-11-09 22:15 | preentreno-25m (parcial) | — | 2,35 h: 2,21 BPB es, 1,62 código; pesos en checkpoints/hn1-25m-cpu.pt |
| 2026-11-09 23:30 | presupuesto | b952c46 | crecer o reducir a X parámetros sin perder lo aprendido; presets hn1-700m y hn1-1b |
| 2026-11-09 23:30 | uso-herramientas | b952c46 | banco sellado, currículo verificado con llamadas reales, TLM en el contexto |
| — | tutor, código abierto | — | resolviendo conflictos de planes en sus ramas |
| — | visión, catálogo, lenguaje, código, contexto, tokens, maestro-local | — | en curso |

