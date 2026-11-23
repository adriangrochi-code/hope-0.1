"""App de PC de Hope CNC: pantalla al estilo Mach3 con estética moderna (docs/CNC_HOPE.md §7).

``python -m hope.cnc app`` levanta un servidor local (solo 127.0.0.1) y abre la pantalla en una
ventana propia (Chrome/Edge/Chromium en modo app; si no hay, el navegador por defecto). Sin
dependencias nuevas ni Internet: funciona igual en una PC vieja que en una moderna.

Contenido del paquete
---------------------
* :mod:`hope.cnc.app.server`: servidor HTTP local, lista blanca de acciones, apertura de la ventana y
  arranque completo de la app (:func:`hope.cnc.app.server.run_app`).
* :mod:`hope.cnc.app.compat_css`: genera ``static/compat.css`` para navegadores viejos.
* ``static/``: la página. ``index.html`` (distribución de las pantallas de Mach3Mill), ``cnc.js``
  (lógica de la pantalla), ``pathview.js`` (dibujo de la trayectoria), ``compat.js`` y ``viejo.js``
  (Firefox 52 / Chrome 57 en adelante, y aviso claro para los más viejos), hojas de estilo y logo.

Ejemplo::

    python -m hope.cnc app                    # abre la ventana en http://127.0.0.1:8766/
    python -m hope.cnc app --sin-ventana      # solo el servidor (la pantalla se abre a mano)
"""
