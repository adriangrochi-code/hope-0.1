"""Aplicación con interfaz para entrenar a Hope en casa (pedido del usuario).

``python -m hope.app`` abre ``http://127.0.0.1:8765``: una página con pasos guiados
(calibración, datos, lenguaje, maestro, pasar a la A770, guardián). Ver ``docs/APP.md``.

Sin dependencias nuevas: servidor HTTP de la biblioteca estándar y una página propia
(HTML/JS/CSS sin CDN, gráficos SVG hechos a mano). Escucha solo en 127.0.0.1 y solo
ejecuta acciones de una lista fija; los procesos pesados (calibración, entrenamiento,
muestras) corren como subprocesos de módulos fijos de Hope.
"""
