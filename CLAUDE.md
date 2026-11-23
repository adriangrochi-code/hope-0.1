# Reglas del proyecto (las lee toda sesión que trabaje en este repositorio)

## Estilo del código: «al estilo Marlin», máximo detalle y fácil de leer

Pedido del dueño del proyecto: **todo el código se escribe y documenta como el firmware Marlin**,
para que cualquiera, sin conocer el proyecto, lo entienda al leerlo. Vale para código nuevo y para
todo código que se toque (al modificar una función, dejarla documentada así).

1. **Encabezado de archivo**: cada archivo empieza con un bloque que dice qué es, para qué sirve,
   cómo encaja con el resto (qué lo usa y qué usa), unidades y convenciones, y un ejemplo de uso.
2. **Secciones con separadores**, como Marlin:

   ```python
   # ==============================================================================================
   # ===================================== PLANIFICADOR ============================================
   # ==============================================================================================
   ```

   y subsecciones con `# --- nombre ----------------------------------------------------------`.
3. **Cada parámetro de configuración se explica** donde se define: qué hace, unidad, rango
   razonable, valor por defecto y qué pasa si se exagera (como `Configuration.h`). Ejemplo:

   ```python
   # Desviación de unión (mm). Cuánto puede «redondear» el planificador una esquina sin frenar del
   # todo. Más alto = más rápido en curvas hechas de tramos cortos, menos preciso en esquinas.
   # Típico 0.01–0.05. grbl usa 0.01 por defecto.
   ```
4. **Cada función/clase** tiene docstring: qué hace, parámetros (con unidades), qué devuelve,
   errores que puede lanzar y, si no es obvio, **por qué** se hace así.
5. **Comentarios en línea para lo que no es evidente**: fórmulas (con la física o la geometría
   explicada), atajos de rendimiento, compatibilidad con equipos viejos, compatibilidad con Mach3,
   decisiones de seguridad. No comentar lo obvio (`i += 1  # suma 1`).
6. **Nombres descriptivos**; sin abreviaturas crípticas salvo las del G-code (G0, F, S…).
7. **Idioma**: comentarios y documentación en español (rioplatense neutro), como el resto del
   proyecto. Los identificadores pueden quedar en inglés.
8. En JavaScript y CSS rige lo mismo (`/* === Sección === */`, cada función con su comentario).

## Rendimiento y compatibilidad (TISMA CNC)

* Tiene que andar en PCs viejas: Windows XP (Python 3.4 traducido, ver
  `empaquetado/tisma_cnc/py34/`), Vista (Python 3.7), Firefox 52 / Chrome 57, 1024×768.
* El código fuente es Python 3.7+ (`hope/cnc/ruff.toml`); no usar `:=`, `async`, parámetros
  solo posicionales ni `typing` fuera de anotaciones (el traductor a 3.4 los rechaza).
* No depender del orden de los `dict` para nada que se vea o se envíe (en 3.4 no se conserva).
* Antes de optimizar, medir. Comentar cada optimización con lo que mide y por qué es segura.

## Seguridad

* Mover una máquina real pide aprobación de una persona (guardián, `hardware.control`).
* La parada de emergencia es física: ningún programa la reemplaza.

## Pruebas

* `python -m pytest -q tests/test_cnc*.py` (también en Python 3.7; con `TISMA_PY34=/ruta/python3.4`
  corre la suite traducida a Python 3.4).
* `ruff check hope/cnc empaquetado/tisma_cnc tests/test_cnc*.py`.
