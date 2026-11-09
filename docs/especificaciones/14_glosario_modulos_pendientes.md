# HOPE HN-1 — GLOSARIO Y DEFINICIÓN DE MÓDULOS PENDIENTES

Versión: 1.0 (documento del autor, transcrito sin cambios de contenido)

| Módulo | Nombre | Responsabilidad central | Integraciones |
|---|---|---|---|
| PESAM | Persistent Experience & Self-Analysis Memory | Recordar experiencias y autoanálisis; continuidad entre sesiones | MDD, AHMC, RRA, HLA, ARS, SCA, CRE |
| AHMC | Adaptive Hierarchical Memory & Context (nombre propuesto) | Organizar y recuperar memoria contextual en varias escalas (inmediata, trabajo, episódica, asociativa), con fuentes originales y procedencia | MDD, PESAM, MRC, RRA, HLA, HWM, SCA |
| AIS | Adaptive Inference Scheduler | Estimar dificultad, seleccionar módulos, paralelizar, distribuir CPU/GPU/memoria, fijar profundidad y presupuestos, detener por rendimiento marginal | ARS, Loop Controller, AACA, CRE, HTO, SCA |
| UAC | Uncertainty and Active Curiosity | Reconocer lo desconocido, detectar datos faltantes, distinguir desconocimiento de contradicción, formular preguntas, decidir cuándo consultar al maestro. *No inventar respuestas para ocultar desconocimiento.* | ATI, ARS, HED, HWM, CRE, PESAM, HLA |
| ATI | Autonomous Teacher Interaction | Detectar carencias y solicitar enseñanza a Qwen; practicar y demostrar comprensión independiente | UAC, HLA, PESAM, RRA, ARS, AANS, SCA |
| RRA | Retrospective Reasoning Analysis | Analizar razonamientos pasados: errores recurrentes, premisas incorrectas, estrategias a conservar; generar ejemplos de error para entrenamiento | PESAM, MDD, CRE, ARS, HLA, SCA |
| CRE | Critical Reasoning Engine (nombre propuesto, módulo general) | Evaluación crítica durante el procesamiento: consistencia, contradicciones, premisas dudosas, contraejemplos, hechos vs. inferencias vs. hipótesis. Coordina y reutiliza HCRI, ARS, HED, UAC y RRA sin duplicarlos | HCRI, ARS, HED, UAC, RRA |
| Loop Controller | Controlador de ciclos de procesamiento | Decidir repetir, revisar, cambiar profundidad o estrategia, detectar estancamiento y ciclos improductivos, respetar presupuestos. Ciclos no fijos | ARS, AIS, CRE, SCA, AACA |
| Hope Code | Sistema de programación y desarrollo de software | Comprender, diseñar, programar, depurar, probar y mantener software (incl. sus propias herramientas); ejecutar solo con autorización; evaluar regresiones | HTE, HTO, TCG, TVE, TLM, TEO, PESAM, AHMC, ARS, CRE, AANS |

**Diferencia CRE / RRA:** CRE evalúa durante el procesamiento; RRA analiza retrospectivamente.

## Principio de arquitectura

Estos módulos no deben convertirse necesariamente en redes neuronales independientes. Pueden implementarse mediante componentes neuronales compartidos, controladores especializados, memorias persistentes, algoritmos deterministas, herramientas externas locales, operadores y circuitos dinámicos, y sistemas de evaluación y verificación. Se debe evitar duplicar funcionalidades, parámetros y procesos. Cada módulo debe tener responsabilidades, interfaces, entradas, salidas y criterios de evaluación claramente definidos.

**Objetivo final:** construir un sistema cognitivo integrado en el que cada módulo contribuya a una capacidad general superior, sin que la complejidad de la arquitectura se convierta en un coste innecesario.
