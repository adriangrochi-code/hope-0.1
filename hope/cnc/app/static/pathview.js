/* ================================================================================================
 * ======================================== pathview.js ===========================================
 * ================================================================================================
 *
 * Vista de trayectoria en canvas 2D con proyección ortográfica (sin WebGL: corre en PCs viejas).
 * Dos capas: «base» (trayectoria, se redibuja al cambiar la vista o el avance) y «marca» (herramienta).
 *
 * QUÉ ES
 *   Es el equivalente a la ventana «ToolPath» de Mach3: dibuja el recorrido del programa G-code
 *   cargado (rápidos, cortes, avances sin cortar, perforaciones de plasma), la grilla de la mesa, la
 *   caja del recorrido de la máquina, los ejes en el origen de trabajo y la herramienta.
 *   Se puede girar (arrastrar), mover (Mayús + arrastrar o botón derecho), hacer zoom (rueda) y
 *   ajustar a la ventana (doble clic). También tiene vistas fijas: arriba, ISO, frente y lado.
 *
 * CÓMO ENCAJA
 *   * index.html carga, en este orden: viejo.js → compat.js → pathview.js → cnc.js.
 *   * Usa de compat.js: observeResize() y window.TISMA_LITE (modo liviano).
 *   * cnc.js crea un PathView por cada <div class="view" data-view="…"> (pantallas Ejecución, MDI y
 *     Trayectoria), le pasa los datos de /api/trayectoria con setData(), cambia la vista con
 *     setView() y, en cada consulta de estado (cada 150 ms, o 400 ms en modo liviano), llama a
 *     setProgress() con la línea en curso y la posición de la herramienta. También escribe directo
 *     las propiedades `follow` (botón «Seguir») y `hot` (antorcha de plasma encendida).
 *   * Los colores salen de las variables CSS de cnc.css / tema_tisma.css (--rapid, --accent, --done,
 *     --torch, --dro-machine, --mono): cambiar el tema cambia el dibujo sin tocar este archivo.
 *
 * CONVENCIONES
 *   * Coordenadas en unidades del programa (mm o pulgadas), relativas al origen de trabajo G54.
 *   * Cada punto es un arreglo [x, y, z, tipo, línea]:
 *       tipo -1 = inicio de un movimiento (se «levanta el lápiz»: no se dibuja tramo hasta acá),
 *       tipo  0 = rápido (G0), línea punteada roja,
 *       tipo  1 = corte (G1/G2/G3 con husillo o antorcha encendidos),
 *       tipo  3 = avance sin cortar (sensado, bajadas en vacío), gris.
 *     El estilo «2» (hecho, ya cortado) no viene en los datos: se decide al dibujar comparando la
 *     línea del punto con la línea en curso del programa.
 *     `línea` es el número de línea del G-code que generó ese punto (sirve para pintar lo ya hecho).
 *   * Coordenadas de pantalla en píxeles CSS, (0, 0) arriba a la izquierda, Y hacia abajo.
 *   * Ángulos en radianes: `yaw` = giro alrededor de Z (visto desde arriba), `elev` = inclinación de
 *     la cámara (0 = de frente, π/2 = desde arriba).
 *
 * POR QUÉ DOS CANVAS
 *   La trayectoria puede tener 100 000 puntos o más; redibujarla entera cada vez que la herramienta
 *   se mueve (varias veces por segundo) ahoga a una PC vieja. Por eso hay dos canvas apilados:
 *     * `base`: grilla + caja + ejes + trayectoria. Solo se redibuja entera cuando cambia la vista
 *       (giro, zoom, tamaño, «Seguir») o cuando el programa retrocede. Si el programa avanza, se le
 *       pinta encima SOLO el tramo nuevo ya cortado (_drawDone, pintado incremental).
 *     * `over`: la herramienta (marca chica). Se borra y redibuja en cada actualización: es barato.
 *       Además recibe el ratón/táctil, porque está arriba.
 *
 * EJEMPLO
 *     const v = new PathView(document.querySelector('[data-view="path"]'));
 *     v.setData([[0,0,5,-1,1], [10,0,5,0,1], [10,0,-1,1,2], [20,10,-1,1,3]],
 *               { X: [0, 300], Y: [0, 200], Z: [-50, 0] }, []);
 *     v.setView("iso");
 *     v.setProgress(2, { X: 10, Y: 0, Z: -1 });   // línea 2 en curso, herramienta en (10, 0, -1)
 */
"use strict";

/* ================================================================================================
 * ========================================== PathView ============================================
 * ================================================================================================ */
class PathView {
  /* Crea la vista dentro de `host` (un <div class="view">, que cnc.css posiciona relativo para que
     los dos canvas queden apilados ocupando todo el div).
       host: elemento contenedor; su tamaño define el del dibujo.
     Deja la vista ISO por defecto y se redibuja sola cuando el contenedor cambia de tamaño. */
  constructor(host) {
    this.host = host;
    this.base = document.createElement("canvas");   // capa de fondo: trayectoria (cara de redibujar)
    this.over = document.createElement("canvas");   // capa de arriba: herramienta (barata) + ratón
    host.append(this.base, this.over);
    this.pts = [];                 // puntos [x, y, z, tipo, línea] (ver CONVENCIONES)
    this.limits = null;            // recorrido de la máquina { X: [mín, máx], … } en coords de trabajo
    this.yaw = -Math.PI / 4;       // giro alrededor de Z: −45° = mirar en diagonal (vista ISO)
    this.elev = 0.62;              // inclinación (rad): 0.62 ≈ 35.5°, casi la isométrica (35.26°)
    this.zoom = 1;                 // zoom del usuario (rueda), multiplica a `scale`
    this.pan = [0, 0];             // desplazamiento del usuario en píxeles
    this.scale = 1;                // píxeles por unidad que calcula fit() para que todo entre
    this.center = [0, 0];          // punto proyectado (u, v) que va al centro de la pantalla
    this.doneLine = 0;             // línea del programa en curso: lo anterior se pinta como «hecho»
    this.tool = null;              // posición de la herramienta { X, Y, Z } o null si no se conoce
    this.dirty = true;             // true = hay que redibujar la capa base entera
    this._bind();
    observeResize(host, () => { this.dirty = true; this.draw(); });
  }

  /* --- datos y vistas ------------------------------------------------------------------------- */

  /* Carga una trayectoria nueva y la ajusta a la ventana.
       pts:     [[x, y, z, tipo, línea], …] (puede ser null/vacío: sin programa).
       limits:  { X: [mín, máx], Y: […], Z: […] } recorrido de la máquina; un valor puede ser null.
       pierces: [[x, y, z], …] puntos de perforación del plasma (se dibujan como circulitos). */
  setData(pts, limits, pierces) {
    this.pts = pts || [];
    this.limits = limits;
    this.pierces = pierces || [];
    this.fit();
  }

  /* Cambia a una vista fija (botón «Vista» / Display Mode de Mach3) y reajusta el zoom.
       v: "top" (planta), "iso", "front" (frente, plano XZ), "side" (lado, plano YZ) o "fit"
          (solo reajusta, sin cambiar el ángulo).
     Cada vista es un par [yaw, elev] en radianes. */
  setView(v) {
    const views = { top: [0, Math.PI / 2], iso: [-Math.PI / 4, 0.62], front: [0, 0], side: [-Math.PI / 2, 0] };
    if (v === "fit") { this.fit(); return; }
    [this.yaw, this.elev] = views[v];
    this.fit();
  }

  /* --- geometría: proyección y encuadre ------------------------------------------------------- */

  /* Proyección ortográfica de un punto 3D de la máquina al plano de la pantalla.
       x, y, z: coordenadas de trabajo.
     Devuelve [u, v] en unidades del programa (todavía sin escala ni centro), v hacia ARRIBA.
     Por qué así: primero se gira el punto alrededor de Z un ángulo `yaw` (como girar la mesa), y
     después se inclina la cámara un ángulo `elev`: la altura en pantalla mezcla la profundidad (yr)
     y la altura real (z). Con elev = π/2 se ve desde arriba (v = yr) y con elev = 0 de frente
     (v = z). Es ortográfica (sin perspectiva) porque así las medidas no se deforman con la
     distancia y alcanza con senos/cosenos: no hace falta WebGL ni matrices 4×4. */
  project(x, y, z) {
    const c = Math.cos(this.yaw), s = Math.sin(this.yaw);
    const xr = x * c - y * s, yr = x * s + y * c;           // giro 2D alrededor de Z
    return [xr, yr * Math.sin(this.elev) + z * Math.cos(this.elev)];
  }

  /* Caja que ocupa el dibujo ya proyectado, para encuadrarlo.
     Devuelve [uMín, uMáx, vMín, vMáx] en unidades proyectadas. Incluye la herramienta, si se conoce.
     Sin programa, usa las 8 esquinas del recorrido de la máquina (o una caja de 100×100×10 si no
     hay límites) para que la vista vacía muestre la mesa. */
  bounds() {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    const add = (x, y, z) => {
      const [u, v] = this.project(x, y, z);
      if (u < x0) x0 = u; if (u > x1) x1 = u; if (v < y0) y0 = v; if (v > y1) y1 = v;
    };
    // Rendimiento: para encuadrar alcanza con una muestra de ~20 000 puntos (salteando de a `step`);
    // recorrer los 100 000+ en cada cambio de vista se nota en una PC vieja y el encuadre casi no cambia.
    const step = Math.max(1, Math.floor(this.pts.length / 20000));
    for (let i = 0; i < this.pts.length; i += step) add(this.pts[i][0], this.pts[i][1], this.pts[i][2]);
    if (!this.pts.length) {
      const L = this.limits || {};
      // r(eje, 0 = mín / 1 = máx, valor por defecto si ese límite no está definido)
      const r = (a, i, d) => (L[a] && L[a][i] != null ? L[a][i] : d);
      for (const x of [r("X", 0, 0), r("X", 1, 100)]) for (const y of [r("Y", 0, 0), r("Y", 1, 100)])
        for (const z of [r("Z", 0, -10), r("Z", 1, 0)]) add(x, y, z);
    }
    if (this.tool) add(this.tool.X || 0, this.tool.Y || 0, this.tool.Z || 0);
    return [x0, x1, y0, y1];
  }

  /* Ajusta la escala para que todo el dibujo entre en la ventana (doble clic / «fit»), centra y
     anula el zoom y el desplazamiento del usuario. Deja un margen (86 % del ancho, 82 % del alto)
     para que nada quede pegado al borde ni tapado por las etiquetas de los ejes. La escala se limita
     a 400 px por unidad para que un programa de un solo punto no dé un zoom infinito. */
  fit() {
    const [x0, x1, y0, y1] = this.bounds();
    const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1;
    const sx = (w * 0.86) / Math.max(1e-6, x1 - x0), sy = (h * 0.82) / Math.max(1e-6, y1 - y0);
    this.scale = Math.min(sx, sy, 400);
    this.center = [(x0 + x1) / 2, (y0 + y1) / 2];
    this.zoom = 1;
    this.pan = [0, 0];
    this.dirty = true;
    this.draw();
  }

  /* Punto 3D de trabajo → píxel en pantalla.
     Devuelve [px, py] en píxeles CSS. Se invierte v (pantalla con Y hacia abajo, máquina con Y/Z
     hacia arriba). Usa this.w / this.h, que calcula _size() en cada dibujo. */
  toScreen(x, y, z) {
    const [u, v] = this.project(x, y, z);
    const k = this.scale * this.zoom;
    return [this.w / 2 + (u - this.center[0]) * k + this.pan[0], this.h / 2 - (v - this.center[1]) * k + this.pan[1]];
  }

  /* --- dibujo ---------------------------------------------------------------------------------- */

  /* Ajusta el tamaño interno del canvas `c` al del contenedor y devuelve su contexto 2D, ya escalado
     para dibujar en píxeles CSS.
     En pantallas de alta densidad (devicePixelRatio 2) el canvas se hace el doble de grande para que
     las líneas salgan nítidas; en modo liviano se usa siempre 1, porque 4 veces más píxeles es 4
     veces más trabajo para una PC vieja. Solo se cambia width/height si hace falta: asignarlos
     borra el canvas y reserva memoria nueva, aunque el valor sea el mismo. */
  _size(c) {
    const r = window.TISMA_LITE ? 1 : (window.devicePixelRatio || 1);
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (c.width !== Math.round(w * r) || c.height !== Math.round(h * r)) { c.width = Math.round(w * r); c.height = Math.round(h * r); }
    const g = c.getContext("2d");
    g.setTransform(r, 0, 0, r, 0, 0);
    this.w = w; this.h = h;
    return g;
  }

  /* Avance del programa: lo llama cnc.js en cada consulta de estado (varias veces por segundo).
       line: línea del G-code en curso (0 si no hay trabajo activo); lo anterior se pinta «hecho».
       tool: posición de la herramienta { X, Y, Z } en coordenadas de trabajo.
     Decide qué redibujar: la capa base entera solo si hace falta (retroceso, «Seguir»), si no
     apenas lo nuevo; la capa de la herramienta siempre (es barata). */
  setProgress(line, tool) {
    // ¿se movió la herramienta? (más de una millonésima en total; con «Seguir» eso obliga a recentrar)
    const moved = !this.tool || !tool || Math.abs(this.tool.X - tool.X) + Math.abs(this.tool.Y - tool.Y) + Math.abs(this.tool.Z - tool.Z) > 1e-6;
    this.tool = tool;
    const now = performance.now();
    if (line !== this.doneLine) {
      this.doneLine = line;
      // Rendimiento: si el programa avanzó, se pintan encima SOLO los tramos nuevos ya cortados
      // (antes se redibujaban todos los puntos cada 300 ms: con 100 000 puntos, una PC vieja no daba
      // abasto justo mientras la máquina trabaja). Si retrocedió (rebobinar), se redibuja todo.
      if (!this.dirty && line > (this._doneDrawn || 0)) this._drawDone(line);
      else this.dirty = true;
    }
    // «Seguir» (Jog Follow de Mach3): la vista acompaña a la herramienta
    // (como recentrar obliga a redibujar todo, se hace como mucho cada 120 ms: ~8 veces por segundo)
    if (this.follow && tool && moved && (!this._lastBase || now - this._lastBase > 120)) this.dirty = true;
    this.draw();
  }

  /* Redibuja lo que haga falta: la capa base solo si está marcada `dirty`, la de la herramienta
     siempre. No hace nada si el contenedor está oculto (ancho 0: pantalla no visible), así las
     vistas de las pantallas que no se ven no gastan CPU. Con «Seguir», recentra en la herramienta
     antes de redibujar la base. */
  draw() {
    if (!this.host.clientWidth) return;
    if (this.follow && this.tool && this.dirty) {
      this.center = this.project(this.tool.X || 0, this.tool.Y || 0, this.tool.Z || 0);
      this.pan = [0, 0];
    }
    if (this.dirty) { this._drawBase(); this.dirty = false; this._lastBase = performance.now(); }
    this._drawOver();
  }

  /* Dibuja en el contexto `g` las referencias de la mesa:
       1. grilla en Z = 0 sobre el recorrido X/Y de la máquina (o 100×100 si no hay límites), con un
          paso «redondo» (1, 2, 5 × 10^n) que da unas 12 divisiones, como en papel milimetrado;
       2. la caja punteada del recorrido de la máquina, si se conocen los límites de Z;
       3. los ejes X (rojo), Y (verde) y Z (azul) en el origen de trabajo, como en CAD. */
  _grid(g) {
    const L = this.limits || {};
    // lim(eje, 0 = mín / 1 = máx) o null si ese límite no está definido
    const lim = (a, i) => (L[a] && L[a][i] != null ? L[a][i] : null);
    let gx0 = lim("X", 0), gx1 = lim("X", 1), gy0 = lim("Y", 0), gy1 = lim("Y", 1);
    if (gx0 == null || gx1 == null || gy0 == null || gy1 == null) { gx0 = 0; gx1 = 100; gy0 = 0; gy1 = 100; }
    // paso de grilla: la potencia de 10 por debajo de span/12, multiplicada por 1, 2, 5 o 10 (el
    // primero que alcance). Ej.: 300 mm → 25 → p = 10 → paso 50 mm.
    const span = Math.max(gx1 - gx0, gy1 - gy0);
    const stepRaw = span / 12;
    const p = Math.pow(10, Math.floor(Math.log10(stepRaw)));
    const step = [1, 2, 5, 10].map(m => m * p).find(s => s >= stepRaw) || p * 10;
    g.lineWidth = 1;
    g.strokeStyle = "rgba(180,170,195,0.07)";                // casi invisible: no compite con la trayectoria
    g.beginPath();
    for (let x = Math.ceil(gx0 / step) * step; x <= gx1 + 1e-9; x += step) {
      const a = this.toScreen(x, gy0, 0), b = this.toScreen(x, gy1, 0); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]);
    }
    for (let y = Math.ceil(gy0 / step) * step; y <= gy1 + 1e-9; y += step) {
      const a = this.toScreen(gx0, y, 0), b = this.toScreen(gx1, y, 0); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]);
    }
    g.stroke();
    // recorrido de la máquina (caja)
    const z0 = lim("Z", 0), z1 = lim("Z", 1);
    if (z0 != null && z1 != null) {
      // 8 esquinas, numeradas en binario: índice = 4·(x máx) + 2·(y máx) + (z máx).
      // Las 12 aristas unen esquinas que difieren en un solo bit (de a 1: Z, de a 2: Y, de a 4: X).
      const c = [];
      for (const x of [gx0, gx1]) for (const y of [gy0, gy1]) for (const z of [z0, z1]) c.push(this.toScreen(x, y, z));
      const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
      g.setLineDash([4, 5]);
      g.strokeStyle = "rgba(170,160,185,0.35)";
      g.beginPath();
      for (const [i, j] of edges) { g.moveTo(c[i][0], c[i][1]); g.lineTo(c[j][0], c[j][1]); }
      g.stroke();
      g.setLineDash([]);
    }
    // ejes en el origen de trabajo
    // (largo = un paso de grilla, como mínimo 10 unidades, para que se vean a cualquier escala)
    const o = this.toScreen(0, 0, 0), len = Math.max(10, step);
    const axes = [["X", [len, 0, 0], "#f87171"], ["Y", [0, len, 0], "#4ade80"], ["Z", [0, 0, len], "#60a5fa"]];
    g.lineWidth = 2;
    g.font = "11px " + getComputedStyle(document.body).getPropertyValue("--mono");
    for (const [n, v, col] of axes) {
      const e = this.toScreen(...v);
      g.strokeStyle = col; g.fillStyle = col;
      g.beginPath(); g.moveTo(o[0], o[1]); g.lineTo(e[0], e[1]); g.stroke();
      g.fillText(n, e[0] + 4, e[1] - 3);
    }
  }

  /* Redibuja entera la capa base: grilla, trayectoria completa y perforaciones.
     Es la operación cara (recorre todos los puntos): se llama solo cuando `dirty` está activo.
     Truco de rendimiento: los tramos consecutivos del mismo estilo se juntan en un solo
     beginPath()/stroke(); cambiar de color o de punteado cuesta, y trazar miles de caminos chicos
     es mucho más lento que uno largo. Al terminar anota hasta dónde quedó pintado lo «hecho»
     (_doneDrawn, _doneIdx) para que _drawDone() siga desde ahí. */
  _drawBase() {
    const g = this._size(this.base);
    g.clearRect(0, 0, this.w, this.h);
    this._grid(g);
    const P = this.pts;
    if (!P.length) return;
    const css = getComputedStyle(document.documentElement);
    // v(variable CSS, color por defecto si el tema no la define)
    const v = (n, d) => (css.getPropertyValue(n) || d).trim() || d;
    // estilo de cada tipo de tramo: [color, grosor en px, punteado, opacidad]
    const styles = {
      0: [v("--rapid", "#f87171"), 1, [4, 4], 0.6],    // rápido (G0): punteado, semitransparente
      1: [v("--accent", "#38bdf8"), 1.4, [], 1],      // corte pendiente
      2: [v("--done", "#34d399"), 1.8, [], 1],        // corte ya hecho (antes de la línea en curso)
      3: ["#5b6b7d", 1, [], 0.8],                     // avance sin cortar (sensado, bajada en vacío)
    };
    let cur = null;        // estilo del camino abierto ahora (null = ninguno)
    let prev = null;       // punto anterior ya pasado a pantalla
    const flush = () => { if (cur !== null) g.stroke(); };
    // modo liviano: dibujar como mucho ~40 000 puntos (los extremos de cada tramo se conservan)
    const stride = window.TISMA_LITE && P.length > 40000 ? Math.ceil(P.length / 40000) : 1;
    for (let i = 0; i < P.length; i++) {
      const p = P[i];
      // salteo en modo liviano: solo puntos intermedios de una racha del mismo tipo (nunca el
      // inicio de un movimiento ni el último antes de un cambio de tipo, para no deformar la forma)
      if (stride > 1 && i % stride && p[3] > 0 && i + 1 < P.length && P[i + 1][3] === p[3]) continue;
      const s = this.toScreen(p[0], p[1], p[2]);
      if (p[3] >= 0 && prev) {
        const st = p[3] === 0 ? 0 : p[3] === 3 ? 3 : (this.doneLine && p[4] < this.doneLine ? 2 : 1);
        if (st !== cur) {
          // cambio de estilo: se cierra el camino anterior y se abre otro desde el punto previo
          flush();
          const [col, lw, dash, alpha] = styles[st];
          g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash); g.globalAlpha = alpha;
          g.beginPath(); g.moveTo(prev[0], prev[1]);
          cur = st;
        }
        g.lineTo(s[0], s[1]);
      } else if (cur !== null) {
        // tipo -1 (inicio de movimiento): se salta hasta acá sin trazar
        g.moveTo(s[0], s[1]);
      }
      prev = s;
    }
    flush();
    g.setLineDash([]);
    g.globalAlpha = 1;
    // hasta dónde quedó pintado lo «ya cortado» (para seguir desde ahí con _drawDone)
    this._doneDrawn = this.doneLine || 0;
    let k = 0;
    while (k < P.length && P[k][4] < this._doneDrawn) k++;
    this._doneIdx = k;                                   // primer punto todavía no pintado como hecho
    // perforaciones del plasma: circulito naranja de 4 px en Z = 0 (la superficie de la chapa)
    if (this.pierces && this.pierces.length) {
      g.strokeStyle = v("--torch", "#fb923c"); g.lineWidth = 1.5;
      g.beginPath();
      for (const q of this.pierces) { const s = this.toScreen(q[0], q[1], 0); g.moveTo(s[0] + 4, s[1]); g.arc(s[0], s[1], 4, 0, Math.PI * 2); }
      g.stroke();
    }
  }

  /* Pinta como «hecho» (verde) los tramos de corte entre lo ya pintado y la línea `line`, sobre el
     dibujo de fondo que ya está. Mismo estilo que el tipo 2 de _drawBase. */
  /* Pintado incremental: en vez de recorrer los 100 000 puntos en cada actualización, empieza en
     _doneIdx (donde quedó la vez anterior) y avanza solo hasta `line`. Así el costo es proporcional
     a lo que se cortó desde la última consulta (unos pocos puntos), no al tamaño del programa: es
     lo que permite que una PC vieja muestre el avance sin trabarse mientras la máquina trabaja.
     Es seguro porque solo se pinta ENCIMA (verde sobre azul, mismo trazo y más grueso), nunca se
     borra; si el programa retrocede, setProgress() pide un redibujo completo.
       line: línea en curso; se pintan los puntos con línea < line. */
  _drawDone(line) {
    const P = this.pts;
    let i = this._doneIdx || 0;
    if (!P.length || i >= P.length) { this._doneDrawn = line; return; }
    // contexto sin _size(): el canvas ya tiene el tamaño y la escala del último _drawBase (y cambiar
    // el tamaño lo borraría). Hereda la transformación por devicePixelRatio que dejó setTransform.
    const g = this.base.getContext("2d");
    const css = getComputedStyle(document.documentElement);
    g.strokeStyle = (css.getPropertyValue("--done") || "#34d399").trim() || "#34d399";
    g.lineWidth = 1.8; g.setLineDash([]); g.globalAlpha = 1;
    g.beginPath();
    let prev = i > 0 ? this.toScreen(P[i - 1][0], P[i - 1][1], P[i - 1][2]) : null;
    for (; i < P.length && P[i][4] < line; i++) {
      const p = P[i];
      const s = this.toScreen(p[0], p[1], p[2]);
      // solo tramos de corte (los rápidos y los avances en vacío no se «hacen»)
      if (prev && (p[3] === 1 || p[3] === 2)) { g.moveTo(prev[0], prev[1]); g.lineTo(s[0], s[1]); }
      prev = s;
    }
    g.stroke();
    this._doneIdx = i;
    this._doneDrawn = line;
  }

  /* Dibuja la herramienta en la capa de arriba (se borra entera cada vez: es poco dibujo).
     Marca: triángulo que apunta a la punta (como una fresa vista de costado: 36 unidades de pantalla
     en Z, o sea ~36 px de frente y menos cuanto más desde arriba se mire, sea cual sea el zoom),
     círculo de 7 px en la punta y una cruz tenue de 16 px.
     Color: --dro-machine normalmente; --torch con la antorcha de plasma encendida (`hot`), más un
     halo, para que el operador vea de lejos que está cortando. */
  _drawOver() {
    const g = this._size(this.over);
    g.clearRect(0, 0, this.w, this.h);
    if (!this.tool) return;
    const t = this.tool;
    const [x, y] = this.toScreen(t.X || 0, t.Y || 0, t.Z || 0);
    // punto 36 px «más arriba» en Z (de frente): se divide por la escala para que no dependa del zoom
    const [xu, yu] = this.toScreen(t.X || 0, t.Y || 0, (t.Z || 0) + 12 / (this.scale * this.zoom) * 3);
    const tc = (getComputedStyle(document.documentElement).getPropertyValue(this.hot ? "--torch" : "--dro-machine") || "#fbbf24").trim();
    g.strokeStyle = tc; g.fillStyle = tc;
    g.lineWidth = 2;
    g.beginPath(); g.moveTo(x, y); g.lineTo(xu - 5, yu); g.lineTo(xu + 5, yu); g.closePath(); g.fill();
    g.beginPath(); g.arc(x, y, 7, 0, Math.PI * 2); g.stroke();
    g.globalAlpha = 0.35; g.lineWidth = 1;
    g.beginPath(); g.moveTo(x - 16, y); g.lineTo(x + 16, y); g.moveTo(x, y - 16); g.lineTo(x, y + 16); g.stroke();
    g.globalAlpha = 1;
    if (this.hot) { g.fillStyle = tc; g.globalAlpha = 0.25; g.beginPath(); g.arc(x, y, 14, 0, Math.PI * 2); g.fill(); g.globalAlpha = 1; }
  }

  /* --- ratón, táctil y rueda ------------------------------------------------------------------ */

  /* Conecta los gestos sobre la capa de arriba:
       arrastrar                         → girar (yaw con el movimiento horizontal, elev con el vertical)
       Mayús + arrastrar / botón derecho
       / dos dedos                       → mover (pan)
       rueda                             → zoom centrado en el puntero
       doble clic                        → ajustar a la ventana (fit)
     Usa Pointer Events si existen (Chrome 55+, Firefox 59+, Edge); en Firefox 52 (Windows XP/Vista)
     no están, y se usan ratón + táctil por separado. */
  _bind() {
    let drag = null;     // arrastre en curso: { x, y (última posición), pan (true = mover, false = girar) }
    const start = e => {
      const p = e.touches ? e.touches[0] : e;
      drag = { x: p.clientX, y: p.clientY, pan: e.shiftKey || (e.button !== undefined && e.button !== 0) || (e.touches && e.touches.length > 1) };
      // captura: el arrastre sigue aunque el puntero salga del canvas
      if (e.pointerId !== undefined && this.over.setPointerCapture) this.over.setPointerCapture(e.pointerId);
    };
    const move = e => {
      if (!drag) return;
      const p = e.touches ? e.touches[0] : e;
      const dx = p.clientX - drag.x, dy = p.clientY - drag.y;
      drag.x = p.clientX; drag.y = p.clientY;
      if (drag.pan) { this.pan[0] += dx; this.pan[1] += dy; }
      // giro: 0.01 rad por píxel (~100 px = 57°). La inclinación se limita entre −0.2 rad (un poco
      // por debajo de la mesa) y π/2 (desde arriba), para no dar vuelta la vista patas arriba.
      else { this.yaw += dx * 0.01; this.elev = Math.max(-0.2, Math.min(Math.PI / 2, this.elev + dy * 0.01)); }
      this.dirty = true; this.draw();
      if (e.touches) e.preventDefault();                // que el dedo no desplace la página
    };
    const end = () => { drag = null; };
    if (window.PointerEvent) {
      this.over.addEventListener("pointerdown", start);
      this.over.addEventListener("pointermove", move);
      this.over.addEventListener("pointerup", end);
    } else {                                         // navegadores viejos: ratón y táctil
      // (mousemove/mouseup en la ventana: sin captura de puntero, es la forma de no perder el arrastre)
      this.over.addEventListener("mousedown", start);
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", end);
      this.over.addEventListener("touchstart", start);
      this.over.addEventListener("touchmove", move);
      this.over.addEventListener("touchend", end);
    }
    this.over.addEventListener("contextmenu", e => e.preventDefault());   // botón derecho = mover
    // Zoom con la rueda, centrado en el puntero: factor exponencial (cada «clic» de rueda, deltaY ≈
    // 100, cambia el zoom ~15 %; igual de suave para acercar que para alejar). Se corrige el pan para que el
    // punto bajo el puntero quede quieto. passive: false porque se llama a preventDefault() (si no,
    // la rueda además desplazaría la página).
    this.over.addEventListener("wheel", e => {
      e.preventDefault();
      const f = Math.exp(-e.deltaY * 0.0015);
      const r = this.over.getBoundingClientRect();
      const mx = e.clientX - r.left - this.w / 2 - this.pan[0], my = e.clientY - r.top - this.h / 2 - this.pan[1];
      this.pan[0] -= mx * (f - 1); this.pan[1] -= my * (f - 1);
      this.zoom *= f;
      this.dirty = true; this.draw();
    }, { passive: false });
    this.over.addEventListener("dblclick", () => this.fit());
  }
}
window.PathView = PathView;      // global: cnc.js no usa módulos (Firefox 52 no los soporta)
