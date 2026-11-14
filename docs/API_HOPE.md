# `hope.api` — herramientas de Hope para cualquier modelo base (v0.1.0)

Fachada estable (§11.24 T1). Sirve para Hope, Qwen3-Coder, Qwen3.5, Qwen 4 o cualquier modelo que hable por un servidor compatible con OpenAI. **Lo que no está en `hope.api` es interno y puede cambiar.** Versionado semántico: un cambio que rompe la interfaz sube el primer número.

## Uso rápido

```python
from hope import api

h = api.Herramientas(guardian=mi_guardian)    # sin guardián, las herramientas de archivos/red se niegan

# 1. Esquemas para la API compatible con OpenAI (campo "tools"), o texto para modelos sin herramientas nativas
tools = h.especificaciones()                     # todas
tools = h.especificaciones([f.nombre for f in h.buscar("maximizar ganancia con restricciones", k=8)])
sistema = h.instrucciones("hermes")              # o "hope", "qwen_xml", "json"

# 2. Ejecutar lo que pidió el modelo, en la sintaxis que sea
for llamada, resultado, respuesta in h.ejecutar_texto(salida_del_modelo):
    ...   # `respuesta` ya está en la misma sintaxis: devolvésela al modelo

# Mensajes OpenAI con tool_calls
for ll in api.desde_openai(mensaje):
    r = h.ejecutar(ll.nombre, ll.argumentos)
    mensajes.append(api.formatear_resultado(ll, r))   # {"role": "tool", ...}

# 3. Bucle completo con cualquier modelo: generar(texto) -> texto
resp = api.responder(generar, "¿Cuál es el máximo de 3x+2y con x+y≤4, x+3y≤6?", h, sintaxis="hermes")
resp.texto, resp.llamadas, resp.todo_verificado

# 4. Crítico CRE/HCRI
api.criticar(problema, razonamiento, respuesta)  # {"veredicto": ..., "problemas": [...]}
```

## Sintaxis de llamada que entiende

| Sintaxis | Cómo llama el modelo | Cómo se le responde |
|---|---|---|
| `hope` | `⟦nombre: entrada⟧` | ` → resultado` |
| `hermes` (Qwen2.5/3 y muchos) | `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` | `<tool_response>{...}</tool_response>` |
| `qwen_xml` (Qwen3-Coder) | `<tool_call><function=nombre><parameter=p>valor</parameter></function></tool_call>` | `<tool_response>{...}</tool_response>` |
| `json` | bloque ```json {"name": ..., "arguments": {...}}``` | `<tool_response>{...}</tool_response>` |
| `openai` | `tool_calls` en el mensaje | `{"role": "tool", "tool_call_id": ..., "content": ...}` |

## Resultado

Toda herramienta devuelve `Resultado`:
- `ok`: si funcionó.
- `salida`: texto corto para el modelo.
- `valor`: dato estructurado.
- `certificado`: prueba o segunda verificación.
- `verificado`: `True` / `False` / `None` (`None` = la herramienta no certifica).
- `segundos`, `avisos`.

`Resultado.texto()` agrega `[verificado]` o `[NO verificado]`. Las instrucciones le piden al modelo que no dé por cierto lo no verificado.

## Herramientas certificadas (v0.1.0)

| Nombre | Qué hace | Certificado |
|---|---|---|
| `calc_cert` | expresión con aritmética de intervalos (mpmath.iv) | intervalo [a, b] que **contiene** el valor real |
| `equivalencia` | ¿dos expresiones son iguales? | SymPy + evaluación en puntos al azar (tienen que coincidir) |
| `raiz` | raíz de f(x)=0 en [a, b] | bisección con cambio de signo comprobado por intervalos |
| `integral` | integral definida | tanh-sinh y Gauss-Legendre tienen que coincidir |
| `lp` | programación lineal (HiGHS vía SciPy) | duales, brecha de dualidad y factibilidad primal |
| `smt` | restricciones en SMT-LIB2 (z3; cvc5 si está instalado) | el modelo se reevalúa contra cada aserción; cvc5 tiene que coincidir |
| `grafo` | camino mínimo y flujo máximo (networkx) | potenciales que cumplen la desigualdad triangular / corte mínimo = flujo |

Además, todo el **catálogo de Hope** (calc, unidades, resolver, derivar, integrar, matriz, logica, python en sandbox, pruebas, lint, compilar, circuito, filtro, sql, convertir, gcode, stl, marlin, material, regex, validar, hex, desensamblar, búsqueda web y Wikipedia…), con un argumento `entrada`. Pasa por el guardián.

**Seguridad:** sin `eval`. Las expresiones se analizan con `ast` y solo se aceptan números, variables declaradas, operadores y funciones matemáticas de una lista. Los exponentes tienen que ser números (≤ 10 000) o variables. Hay límites de tamaño para expresiones, problemas lineales y grafos.

**Dependencias:** mpmath, sympy, networkx y z3 (ya están en la lista de TISMA). Para `lp`, SciPy (`python3-scipy`). Para la doble verificación en `smt`, cvc5 (`python3-cvc5`), opcional.

## Próximo (v0.2)

uncertainties, holidays (con Argentina), periodictable, fluids, shapely, statsmodels, CoolProp y el resto de §11.16 T3, el worker `exactas_worker.py` de TISMA integrado, y `hope_modelo.json` + `hope serve` (§11.24 T2–T3).
