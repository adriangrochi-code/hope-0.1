"""Hope CNC PC app: a Mach3-style screen with a modern look (docs/CNC_HOPE.md §7).

``python -m hope.cnc app`` starts a local server (127.0.0.1 only) and opens the screen in its own
window (Chrome/Edge/Chromium in app mode; if none is found, the default browser). No new
dependencies and no Internet: it works the same on an old PC as on a modern one.

Package contents
----------------
* :mod:`hope.cnc.app.server`: local HTTP server, action whitelist, window opening and full app
  startup (:func:`hope.cnc.app.server.run_app`).
* :mod:`hope.cnc.app.compat_css`: generates ``static/compat.css`` for old browsers.
* ``static/``: the page. ``index.html`` (layout of the Mach3Mill screens), ``cnc.js``
  (screen logic), ``pathview.js`` (toolpath drawing), ``compat.js`` and ``viejo.js``
  (Firefox 52 / Chrome 57 onwards, and a clear warning for older ones), stylesheets and logo.

Example::

    python -m hope.cnc app                    # opens the window at http://127.0.0.1:8766/
    python -m hope.cnc app --sin-ventana      # server only (open the screen by hand)
"""
