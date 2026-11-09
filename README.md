# Hope HN-1

Arquitectura neuronal híbrida, modular y estructuralmente plástica. **Proyecto experimental:** ninguna capacidad está demostrada todavía.

- Diseño: [`docs/ARQUITECTURA_HN1.md`](docs/ARQUITECTURA_HN1.md)
- Especificaciones originales: [`docs/especificaciones/`](docs/especificaciones/)

## Estado: fase F0 (infraestructura)

| Pieza | Ubicación | Qué hace |
|---|---|---|
| Prueba del entorno | `hope/bench/env_check.py` | Mide matmul, atención y un paso de entrenamiento en la GPU real |
| Estados epistémicos | `hope/epistemic.py` | Enumeraciones compartidas; subir de estado exige evidencia |
| Registro de topología | `hope/registry/topology.py` | IDs estables; parámetros totales, activos y entrenables |
| Presupuesto de crecimiento | `hope/growth/budget.py`, `configs/growth_budget.yaml` | Techo autorizado, modos de crecimiento, qué cambio es «importante» |
| Operaciones estructurales | `hope/growth/ops.py` | Agregar especialista, ensanchar, conectar regiones, retirar en dos pasos; preservan la función |
| Versiones y linaje | `hope/lineage/store.py` | Ramas `stable` / `experimental/*`, promoción con aprobación, vuelta atrás |
| Guardián de seguridad | `guardian/` | Niveles de autorización, invariantes, aprobaciones humanas, auditoría encadenada |
| Evaluación sellada | `hope/eval/sealed.py` | Conjuntos con hash, registro de accesos, detección de contaminación |
| Modelo de juguete | `hope/core/toy.py` | Regiones con especialistas para probar todo lo anterior |
| Diagnóstico de carencias | `hope/growth/diagnosis.py` | Reglas en orden: datos → herramienta → memoria → optimización / capacidad (con sonda) |
| Ciclo ASG v0 | `hope/growth/asg.py` | Plan de crecimiento dirigido o uniforme, permiso del guardián, decisión contra un control |
| PESAM v0 | `hope/memory/pesam.py` | Registro de experiencias de solo agregado |
| Memoria AHMC/MDD v0 | `hope/memory/store.py`, `retrieval.py`, `working.py`, `embed.py` | Archivo de originales, engramas con procedencia y estado, hechos con detección de contradicciones, n-gramas, recuperación híbrida, memoria de trabajo, continuidad de proyectos, consolidación |
| Núcleo HN-1 | `hope/core/hn1.py`, `hope/core/layers.py` | Bytes con parches, backbone SSM + atención, núcleo recurrente con parada adaptativa, especialistas, memoria; presets de 25M y 200M |
| Línea base BPE | `hope/core/baseline.py` | Transformer convencional para comparar en bits por byte |
| Datos | `hope/data/corpus.py` | Normalización, deduplicación, partición determinista, registro de procedencia |
| Entrenamiento | `hope/train/lm.py`, `hope/train/compare.py` | Entrena HN-1 o la línea base y compara corridas |
| Diagnóstico LTH | `hope/growth/lth.py`, `hope/growth/usage_report.py` | Uso de especialistas, neuronas e iteraciones internas; lottery tickets; reasignación de unidades ociosas |
| Baja precisión | `hope/train/lowprec.py` | Base `bf16-8bit`: pesos bf16 con redondeo estocástico y Adam en 8 bits (~6 bytes/parámetro) |
| Entrenamiento continuo | `hope/train/continual.py` | Retomar desde checkpoint, mezcla de fuentes con pesos, repaso contra el olvido, BPB por fuente, registro en el linaje |
| Maestro (ATI) | `hope/teacher/client.py`, `episode.py`, `configs/teacher.yaml` | Cliente intercambiable (llama.cpp, Ollama, OpenVINO, API) con permiso del guardián; formato de episodio; maestro simulado para pruebas |
| Herramientas (HTE) | `hope/tools/` | Catálogo con fichas, sandbox, aritmética exacta, SymPy, unidades, Python; verificación de episodios |
| Ciclo maestro–alumno | `hope/teacher/cycle.py`, `configs/curriculum.yaml` | Fase de arranque por currículo hasta razonamiento semi-funcional; después Hope pregunta por sus carencias |
| HCRI (restricciones) | `hope/constraints/model.py`, `solver.py`, `dynamic.py` | Magnitudes e incógnita, invariantes con dominio, datos contradictorios/faltantes/fuera de rango, resultado esperado con solver en sandbox; restricciones nuevas como hipótesis |
| CRE (crítica) y Loop Controller | `hope/constraints/critic.py`, `hope/runtime/loop.py` | Revisa cada paso (aritmética, unidades, dimensión, invariantes, repeticiones) y hace reintentar con la crítica |
| Banco sintético | `hope/bench/synthetic.py` | Cuellos de botella con causa conocida; resultados en [`docs/resultados/banco_sintetico_v0.md`](docs/resultados/banco_sintetico_v0.md) |

## Instalación en la Intel Arc A770

1. Instalar el driver de GPU de Intel para tu sistema (Windows o Linux).
2. Crear un entorno e instalar PyTorch con soporte XPU:

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # en Windows: .venv\Scripts\activate
   pip install torch --index-url https://download.pytorch.org/whl/xpu
   pip install -e ".[all]"
   ```

3. Comprobar que PyTorch ve la GPU:

   ```bash
   python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
   ```

## Primeros pasos

```bash
# Pruebas de la infraestructura (no necesitan GPU)
python -m pytest

# Banco sintético de ASG (~3 minutos en CPU)
python -m hope.bench.synthetic --seeds 3 --first-seed 100

# Medir la A770 y guardar el resultado
python -m hope.bench.env_check --json runs/env_a770.json

# Preparar un corpus (carpetas con .txt, .md, código o .pdf)
python -m hope.data.corpus --input /ruta/a/textos --out data/f1

# F1: HN-1 de 25M contra la línea base, con el mismo tiempo de cómputo
python -m hope.train.lm --arch hn1 --preset hn1-25m --data data/f1 --out runs/hn1_25m --max-minutes 240
python -m hope.train.lm --arch baseline --data data/f1 --out runs/bpe_25m --max-minutes 240
python -m hope.train.compare runs/hn1_25m runs/bpe_25m

# Seguir entrenando con datos nuevos y repaso de lo anterior
python -m hope.train.continual --resume runs/hn1_25m --out runs/hn1_25m_b \
    --source libros=data/libros:0.6 --source repaso=data/f1:0.4 --steps 5000

# Ciclo con Qwen: arrancar el servidor (p. ej. llama-server -m qwen.gguf --port 8080),
# revisar configs/teacher.yaml y configs/curriculum.yaml, y después:
python -m hope.teacher.cycle --resume runs/hn1_25m --replay data/f1 --out runs/ensenanza --rounds 20

# Diagnóstico de uso (especialistas, neuronas, iteraciones internas)
python -m hope.growth.usage_report --run runs/hn1_25m --data data/f1

# Techo de parámetros: ver el estado y autorizar crecer (p. ej. hasta 1.000 millones)
python -m guardian.cli estado
python -m guardian.cli limite 1B --by <tu-nombre>
python -m guardian.cli limite 500M --by <tu-nombre> --modo USER_APPROVAL_REQUIRED

# Guardián: ver solicitudes pendientes, aprobar, verificar la auditoría
python -m guardian.cli pending
python -m guardian.cli approve <id> --by <tu-nombre>
python -m guardian.cli verify
```

Compartí el contenido de `runs/env_a770.json`. Con esas cifras se recalculan los tiempos de entrenamiento del diseño.
