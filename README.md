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
| Banco sintético | `hope/bench/synthetic.py` | Cuellos de botella con causa conocida; resultados en [`docs/resultados/banco_sintetico_v0.md`](docs/resultados/banco_sintetico_v0.md) |

## Instalación en la Intel Arc A770

1. Instalar el driver de GPU de Intel para tu sistema (Windows o Linux).
2. Crear un entorno e instalar PyTorch con soporte XPU:

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # en Windows: .venv\Scripts\activate
   pip install torch --index-url https://download.pytorch.org/whl/xpu
   pip install -e ".[dev]"
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

# Guardián: ver solicitudes pendientes, aprobar, verificar la auditoría
python -m guardian.cli pending
python -m guardian.cli approve <id> --by <tu-nombre>
python -m guardian.cli verify
```

Compartí el contenido de `runs/env_a770.json`. Con esas cifras se recalculan los tiempos de entrenamiento del diseño.
