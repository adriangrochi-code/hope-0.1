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
| Investigación: razonamiento por parámetro | (agente local) | `reports/Razonamiento por parámetro para Hope.md`: técnicas 2024–2026 con evidencia y un top 8 para Hope |
| Ablación en CPU y Hope-mini | rama principal | `docs/resultados/ablacion_cpu.md` y preentrenamiento de `hn1-cpu` con la configuración ganadora |

**Integración:**
- Cada rama se integra con `git merge`.
- Se corren las pruebas y se registra acá la fecha y el commit de integración.
