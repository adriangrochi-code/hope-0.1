/* ================================================================================================
 * ========================================= pathview.js ==========================================
 * ================================================================================================
 *
 * Toolpath view on a 2D canvas with orthographic projection (no WebGL: it runs on old PCs).
 * Two layers: «base» (toolpath, redrawn when the view or the progress changes) and «marker» (tool).
 *
 * WHAT IT IS
 *   It is the equivalent of Mach3's «ToolPath» window: it draws the path of the loaded G-code
 *   program (rapids, cuts, non-cutting feeds, plasma pierces), the table grid, the machine travel
 *   box, the axes at the work origin and the tool.
 *   It can be rotated (drag), panned (Shift + drag or right button), zoomed (wheel) and fitted to
 *   the window (double click). It also has fixed views: top, ISO, front and side.
 *
 * HOW IT FITS
 *   * index.html loads, in this order: viejo.js → compat.js → pathview.js → cnc.js.
 *   * Uses from compat.js: observeResize() and window.TISMA_LITE (light mode).
 *   * cnc.js creates one PathView per <div class="view" data-view="…"> (Ejecución, MDI and
 *     Trayectoria screens), passes it the /api/trayectoria data with setData(), changes the view
 *     with setView() and, on every status poll (every 150 ms, or 400 ms in light mode), calls
 *     setProgress() with the current line and the tool position. It also writes the `follow`
 *     («Seguir» (Follow) button) and `hot` (plasma torch on) properties directly.
 *   * Colors come from the CSS variables of cnc.css / tema_tisma.css (--rapid, --accent, --done,
 *     --torch, --dro-machine, --mono): changing the theme changes the drawing without touching this file.
 *
 * CONVENTIONS
 *   * Coordinates in program units (mm or inches), relative to the G54 work origin.
 *   * Each point is an array [x, y, z, type, line]:
 *       type -1 = start of a move («pen up»: no segment is drawn up to here),
 *       type  0 = rapid (G0), red dashed line,
 *       type  1 = cut (G1/G2/G3 with the spindle or torch on),
 *       type  3 = non-cutting feed (probing, plunges in the air), gray.
 *     Style «2» (done, already cut) is not in the data: it is decided when drawing by comparing the
 *     point's line with the program's current line.
 *     `line` is the number of the G-code line that produced that point (used to paint what is done).
 *   * Screen coordinates in CSS pixels, (0, 0) at the top left, Y downwards.
 *   * Angles in radians: `yaw` = rotation around Z (seen from above), `elev` = camera tilt
 *     (0 = from the front, π/2 = from above).
 *
 * WHY TWO CANVASES
 *   The toolpath may have 100 000 points or more; redrawing all of it every time the tool moves
 *   (several times per second) chokes an old PC. That is why there are two stacked canvases:
 *     * `base`: grid + box + axes + toolpath. It is only fully redrawn when the view changes
 *       (rotation, zoom, size, «Seguir») or when the program goes back. If the program advances,
 *       ONLY the newly cut stretch is painted on top (_drawDone, incremental painting).
 *     * `over`: the tool (small marker). It is cleared and redrawn on every update: it is cheap.
 *       It also receives mouse/touch input, because it is on top.
 *
 * EXAMPLE
 *     const v = new PathView(document.querySelector('[data-view="path"]'));
 *     v.setData([[0,0,5,-1,1], [10,0,5,0,1], [10,0,-1,1,2], [20,10,-1,1,3]],
 *               { X: [0, 300], Y: [0, 200], Z: [-50, 0] }, []);
 *     v.setView("iso");
 *     v.setProgress(2, { X: 10, Y: 0, Z: -1 });   // line 2 running, tool at (10, 0, -1)
 */
"use strict";

/* ================================================================================================
 * =========================================== PathView ===========================================
 * ================================================================================================ */
class PathView {
  /* Creates the view inside `host` (a <div class="view">, which cnc.css positions as relative so
     the two canvases stay stacked filling the whole div).
       host: container element; its size defines the drawing's size.
     Leaves the ISO view as the default and redraws itself when the container changes size. */
  constructor(host) {
    this.host = host;
    this.base = document.createElement("canvas");   // background layer: toolpath (expensive to redraw)
    this.over = document.createElement("canvas");   // top layer: tool (cheap) + mouse
    host.append(this.base, this.over);
    this.pts = [];                 // points [x, y, z, type, line] (see CONVENTIONS)
    this.limits = null;            // machine travel { X: [min, max], … } in work coordinates
    this.yaw = -Math.PI / 4;       // rotation around Z: −45° = diagonal view (ISO view)
    this.elev = 0.62;              // tilt (rad): 0.62 ≈ 35.5°, almost isometric (35.26°)
    this.zoom = 1;                 // user zoom (wheel), multiplies `scale`
    this.pan = [0, 0];             // user pan in pixels
    this.scale = 1;                // pixels per unit, computed by fit() so everything fits
    this.center = [0, 0];          // projected point (u, v) that goes to the screen center
    this.doneLine = 0;             // current program line: everything before it is painted as «done»
    this.tool = null;              // tool position { X, Y, Z } or null if unknown
    this.dirty = true;             // true = the whole base layer must be redrawn
    this._bind();
    observeResize(host, () => { this.dirty = true; this.draw(); });
  }

  /* --- data and views ------------------------------------------------------------------------- */

  /* Loads a new toolpath and fits it to the window.
       pts:     [[x, y, z, type, line], …] (may be null/empty: no program).
       limits:  { X: [min, max], Y: […], Z: […] } machine travel; a value may be null.
       pierces: [[x, y, z], …] plasma pierce points (drawn as small circles). */
  setData(pts, limits, pierces) {
    this.pts = pts || [];
    this.limits = limits;
    this.pierces = pierces || [];
    this.fit();
  }

  /* Switches to a fixed view (the «Vista» button / Mach3's Display Mode) and refits the zoom.
       v: "top" (plan), "iso", "front" (XZ plane), "side" (YZ plane) or "fit"
          (only refits, without changing the angle).
     Each view is a [yaw, elev] pair in radians. */
  setView(v) {
    const views = { top: [0, Math.PI / 2], iso: [-Math.PI / 4, 0.62], front: [0, 0], side: [-Math.PI / 2, 0] };
    if (v === "fit") { this.fit(); return; }
    [this.yaw, this.elev] = views[v];
    this.fit();
  }

  /* --- geometry: projection and framing ------------------------------------------------------- */

  /* Orthographic projection of a 3D machine point onto the screen plane.
       x, y, z: work coordinates.
     Returns [u, v] in program units (still without scale or centering), v pointing UP.
     Why this way: first the point is rotated around Z by `yaw` (like turning the table), and then
     the camera is tilted by `elev`: the screen height mixes depth (yr) and real height (z). With
     elev = π/2 it is seen from above (v = yr) and with elev = 0 from the front (v = z). It is
     orthographic (no perspective) so measurements are not distorted by distance and sines/cosines
     are enough: no WebGL or 4×4 matrices needed. */
  project(x, y, z) {
    const c = Math.cos(this.yaw), s = Math.sin(this.yaw);
    const xr = x * c - y * s, yr = x * s + y * c;           // 2D rotation around Z
    return [xr, yr * Math.sin(this.elev) + z * Math.cos(this.elev)];
  }

  /* Bounding box of the already projected drawing, for framing.
     Returns [uMin, uMax, vMin, vMax] in projected units. Includes the tool, if known.
     Without a program, it uses the 8 corners of the machine travel (or a 100×100×10 box if there
     are no limits) so the empty view shows the table. */
  bounds() {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    const add = (x, y, z) => {
      const [u, v] = this.project(x, y, z);
      if (u < x0) x0 = u; if (u > x1) x1 = u; if (v < y0) y0 = v; if (v > y1) y1 = v;
    };
    // Performance: for framing, a sample of ~20 000 points is enough (skipping by `step`);
    // walking all 100 000+ on every view change is noticeable on an old PC and the framing barely changes.
    const step = Math.max(1, Math.floor(this.pts.length / 20000));
    for (let i = 0; i < this.pts.length; i += step) add(this.pts[i][0], this.pts[i][1], this.pts[i][2]);
    if (!this.pts.length) {
      const L = this.limits || {};
      // r(axis, 0 = min / 1 = max, default value if that limit is not defined)
      const r = (a, i, d) => (L[a] && L[a][i] != null ? L[a][i] : d);
      for (const x of [r("X", 0, 0), r("X", 1, 100)]) for (const y of [r("Y", 0, 0), r("Y", 1, 100)])
        for (const z of [r("Z", 0, -10), r("Z", 1, 0)]) add(x, y, z);
    }
    if (this.tool) add(this.tool.X || 0, this.tool.Y || 0, this.tool.Z || 0);
    return [x0, x1, y0, y1];
  }

  /* Adjusts the scale so the whole drawing fits in the window (double click / «fit»), centers it and
     resets the user's zoom and pan. Leaves a margin (86 % of the width, 82 % of the height) so
     nothing touches the edge or is hidden by the axis labels. The scale is capped at 400 px per
     unit so a single-point program does not produce infinite zoom. */
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

  /* 3D work point → screen pixel.
     Returns [px, py] in CSS pixels. v is inverted (screen with Y downwards, machine with Y/Z
     upwards). Uses this.w / this.h, which _size() computes on every draw. */
  toScreen(x, y, z) {
    const [u, v] = this.project(x, y, z);
    const k = this.scale * this.zoom;
    return [this.w / 2 + (u - this.center[0]) * k + this.pan[0], this.h / 2 - (v - this.center[1]) * k + this.pan[1]];
  }

  /* --- drawing --------------------------------------------------------------------------------- */

  /* Sets the internal size of canvas `c` to the container's and returns its 2D context, already
     scaled to draw in CSS pixels.
     On high-density screens (devicePixelRatio 2) the canvas is made twice as large so lines come
     out sharp; in light mode 1 is always used, because 4 times more pixels is 4 times more work
     for an old PC. width/height are only changed when needed: assigning them clears the canvas
     and allocates new memory, even if the value is the same. */
  _size(c) {
    const r = window.TISMA_LITE ? 1 : (window.devicePixelRatio || 1);
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (c.width !== Math.round(w * r) || c.height !== Math.round(h * r)) { c.width = Math.round(w * r); c.height = Math.round(h * r); }
    const g = c.getContext("2d");
    g.setTransform(r, 0, 0, r, 0, 0);
    this.w = w; this.h = h;
    return g;
  }

  /* Program progress: cnc.js calls it on every status poll (several times per second).
       line: current G-code line (0 if no job is active); everything before is painted «done».
       tool: tool position { X, Y, Z } in work coordinates.
     Decides what to redraw: the whole base layer only when needed (going back, «Seguir»),
     otherwise just the new part; the tool layer always (it is cheap). */
  setProgress(line, tool) {
    // did the tool move? (more than a millionth in total; with «Seguir» that forces recentering)
    const moved = !this.tool || !tool || Math.abs(this.tool.X - tool.X) + Math.abs(this.tool.Y - tool.Y) + Math.abs(this.tool.Z - tool.Z) > 1e-6;
    this.tool = tool;
    const now = performance.now();
    if (line !== this.doneLine) {
      this.doneLine = line;
      // Performance: if the program advanced, ONLY the newly cut segments are painted on top
      // (previously every point was redrawn every 300 ms: with 100 000 points, an old PC could not
      // keep up exactly while the machine is working). If it went back (rewind), everything is redrawn.
      if (!this.dirty && line > (this._doneDrawn || 0)) this._drawDone(line);
      else this.dirty = true;
    }
    // «Seguir» (Mach3's Jog Follow): the view follows the tool
    // (since recentering forces a full redraw, it is done at most every 120 ms: ~8 times per second)
    if (this.follow && tool && moved && (!this._lastBase || now - this._lastBase > 120)) this.dirty = true;
    this.draw();
  }

  /* Redraws what is needed: the base layer only if it is marked `dirty`, the tool layer always.
     Does nothing if the container is hidden (width 0: screen not visible), so the views of
     screens that are not shown do not spend CPU. With «Seguir», it recenters on the tool before
     redrawing the base. */
  draw() {
    if (!this.host.clientWidth) return;
    if (this.follow && this.tool && this.dirty) {
      this.center = this.project(this.tool.X || 0, this.tool.Y || 0, this.tool.Z || 0);
      this.pan = [0, 0];
    }
    if (this.dirty) { this._drawBase(); this.dirty = false; this._lastBase = performance.now(); }
    this._drawOver();
  }

  /* Draws the table references on context `g`:
       1. grid at Z = 0 over the machine's X/Y travel (or 100×100 if there are no limits), with a
          «round» step (1, 2, 5 × 10^n) giving about 12 divisions, like graph paper;
       2. the dashed box of the machine travel, if the Z limits are known;
       3. the X (red), Y (green) and Z (blue) axes at the work origin, as in CAD. */
  _grid(g) {
    const L = this.limits || {};
    // lim(axis, 0 = min / 1 = max) or null if that limit is not defined
    const lim = (a, i) => (L[a] && L[a][i] != null ? L[a][i] : null);
    let gx0 = lim("X", 0), gx1 = lim("X", 1), gy0 = lim("Y", 0), gy1 = lim("Y", 1);
    if (gx0 == null || gx1 == null || gy0 == null || gy1 == null) { gx0 = 0; gx1 = 100; gy0 = 0; gy1 = 100; }
    // grid step: the power of 10 below span/12, multiplied by 1, 2, 5 or 10 (the
    // first one that is enough). E.g.: 300 mm → 25 → p = 10 → step 50 mm.
    const span = Math.max(gx1 - gx0, gy1 - gy0);
    const stepRaw = span / 12;
    const p = Math.pow(10, Math.floor(Math.log10(stepRaw)));
    const step = [1, 2, 5, 10].map(m => m * p).find(s => s >= stepRaw) || p * 10;
    g.lineWidth = 1;
    g.strokeStyle = "rgba(180,170,195,0.07)";                // almost invisible: does not compete with the toolpath
    g.beginPath();
    for (let x = Math.ceil(gx0 / step) * step; x <= gx1 + 1e-9; x += step) {
      const a = this.toScreen(x, gy0, 0), b = this.toScreen(x, gy1, 0); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]);
    }
    for (let y = Math.ceil(gy0 / step) * step; y <= gy1 + 1e-9; y += step) {
      const a = this.toScreen(gx0, y, 0), b = this.toScreen(gx1, y, 0); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]);
    }
    g.stroke();
    // machine travel (box)
    const z0 = lim("Z", 0), z1 = lim("Z", 1);
    if (z0 != null && z1 != null) {
      // 8 corners, numbered in binary: index = 4·(x max) + 2·(y max) + (z max).
      // The 12 edges join corners that differ in a single bit (by 1: Z, by 2: Y, by 4: X).
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
    // axes at the work origin
    // (length = one grid step, at least 10 units, so they are visible at any scale)
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

  /* Fully redraws the base layer: grid, complete toolpath and pierces.
     It is the expensive operation (walks all the points): it is only called when `dirty` is set.
     Performance trick: consecutive segments of the same style are joined in a single
     beginPath()/stroke(); changing color or dash pattern is costly, and stroking thousands of
     small paths is much slower than one long one. When finished it records how far the «done»
     part was painted (_doneDrawn, _doneIdx) so _drawDone() continues from there. */
  _drawBase() {
    const g = this._size(this.base);
    g.clearRect(0, 0, this.w, this.h);
    this._grid(g);
    const P = this.pts;
    if (!P.length) return;
    const css = getComputedStyle(document.documentElement);
    // v(CSS variable, default color if the theme does not define it)
    const v = (n, d) => (css.getPropertyValue(n) || d).trim() || d;
    // style of each segment type: [color, width in px, dash pattern, opacity]
    const styles = {
      0: [v("--rapid", "#f87171"), 1, [4, 4], 0.6],    // rapid (G0): dashed, semi-transparent
      1: [v("--accent", "#38bdf8"), 1.4, [], 1],      // pending cut
      2: [v("--done", "#34d399"), 1.8, [], 1],        // cut already done (before the current line)
      3: ["#5b6b7d", 1, [], 0.8],                     // non-cutting feed (probing, plunge in the air)
    };
    let cur = null;        // style of the currently open path (null = none)
    let prev = null;       // previous point already converted to screen
    const flush = () => { if (cur !== null) g.stroke(); };
    // light mode: draw at most ~40 000 points (the ends of each segment are kept)
    const stride = window.TISMA_LITE && P.length > 40000 ? Math.ceil(P.length / 40000) : 1;
    for (let i = 0; i < P.length; i++) {
      const p = P[i];
      // skipping in light mode: only intermediate points of a run of the same type (never the
      // start of a move nor the last one before a type change, so the shape is not distorted)
      if (stride > 1 && i % stride && p[3] > 0 && i + 1 < P.length && P[i + 1][3] === p[3]) continue;
      const s = this.toScreen(p[0], p[1], p[2]);
      if (p[3] >= 0 && prev) {
        const st = p[3] === 0 ? 0 : p[3] === 3 ? 3 : (this.doneLine && p[4] < this.doneLine ? 2 : 1);
        if (st !== cur) {
          // style change: the previous path is closed and a new one is opened from the previous point
          flush();
          const [col, lw, dash, alpha] = styles[st];
          g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash); g.globalAlpha = alpha;
          g.beginPath(); g.moveTo(prev[0], prev[1]);
          cur = st;
        }
        g.lineTo(s[0], s[1]);
      } else if (cur !== null) {
        // type -1 (start of a move): jump here without stroking
        g.moveTo(s[0], s[1]);
      }
      prev = s;
    }
    flush();
    g.setLineDash([]);
    g.globalAlpha = 1;
    // how far the «already cut» part was painted (to continue from there with _drawDone)
    this._doneDrawn = this.doneLine || 0;
    let k = 0;
    while (k < P.length && P[k][4] < this._doneDrawn) k++;
    this._doneIdx = k;                                   // first point not yet painted as done
    // plasma pierces: small 4 px orange circle at Z = 0 (the sheet surface)
    if (this.pierces && this.pierces.length) {
      g.strokeStyle = v("--torch", "#fb923c"); g.lineWidth = 1.5;
      g.beginPath();
      for (const q of this.pierces) { const s = this.toScreen(q[0], q[1], 0); g.moveTo(s[0] + 4, s[1]); g.arc(s[0], s[1], 4, 0, Math.PI * 2); }
      g.stroke();
    }
  }

  /* Paints as «done» (green) the cut segments between what is already painted and line `line`, over
     the existing background drawing. Same style as type 2 in _drawBase. */
  /* Incremental painting: instead of walking the 100 000 points on every update, it starts at
     _doneIdx (where it stopped last time) and only advances up to `line`. So the cost is
     proportional to what was cut since the last poll (a few points), not to the program size: this
     is what lets an old PC show the progress without stalling while the machine works.
     It is safe because it only paints ON TOP (green over blue, same stroke and thicker), it never
     erases; if the program goes back, setProgress() asks for a full redraw.
       line: current line; points with line < line are painted. */
  _drawDone(line) {
    const P = this.pts;
    let i = this._doneIdx || 0;
    if (!P.length || i >= P.length) { this._doneDrawn = line; return; }
    // context without _size(): the canvas already has the size and scale of the last _drawBase (and
    // changing the size would clear it). It inherits the devicePixelRatio transform setTransform left.
    const g = this.base.getContext("2d");
    const css = getComputedStyle(document.documentElement);
    g.strokeStyle = (css.getPropertyValue("--done") || "#34d399").trim() || "#34d399";
    g.lineWidth = 1.8; g.setLineDash([]); g.globalAlpha = 1;
    g.beginPath();
    let prev = i > 0 ? this.toScreen(P[i - 1][0], P[i - 1][1], P[i - 1][2]) : null;
    for (; i < P.length && P[i][4] < line; i++) {
      const p = P[i];
      const s = this.toScreen(p[0], p[1], p[2]);
      // cut segments only (rapids and non-cutting feeds are not «done»)
      if (prev && (p[3] === 1 || p[3] === 2)) { g.moveTo(prev[0], prev[1]); g.lineTo(s[0], s[1]); }
      prev = s;
    }
    g.stroke();
    this._doneIdx = i;
    this._doneDrawn = line;
  }

  /* Draws the tool on the top layer (cleared entirely each time: it is little drawing).
     Marker: triangle pointing at the tip (like an end mill seen from the side: 36 screen units in
     Z, i.e. ~36 px from the front and less the more it is viewed from above, whatever the zoom),
     a 7 px circle at the tip and a faint 16 px cross.
     Color: --dro-machine normally; --torch with the plasma torch on (`hot`), plus a halo, so the
     operator can see from afar that it is cutting. */
  _drawOver() {
    const g = this._size(this.over);
    g.clearRect(0, 0, this.w, this.h);
    if (!this.tool) return;
    const t = this.tool;
    const [x, y] = this.toScreen(t.X || 0, t.Y || 0, t.Z || 0);
    // point 36 px «higher» in Z (from the front): divided by the scale so it does not depend on zoom
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

  /* --- mouse, touch and wheel ----------------------------------------------------------------- */

  /* Wires up the gestures on the top layer:
       drag                              → rotate (yaw with horizontal motion, elev with vertical)
       Shift + drag / right button
       / two fingers                     → move (pan)
       wheel                             → zoom centered on the pointer
       double click                      → fit to the window (fit)
     Uses Pointer Events if they exist (Chrome 55+, Firefox 59+, Edge); in Firefox 52 (Windows
     XP/Vista) they are missing, and mouse + touch are used separately. */
  _bind() {
    let drag = null;     // drag in progress: { x, y (last position), pan (true = move, false = rotate) }
    const start = e => {
      const p = e.touches ? e.touches[0] : e;
      drag = { x: p.clientX, y: p.clientY, pan: e.shiftKey || (e.button !== undefined && e.button !== 0) || (e.touches && e.touches.length > 1) };
      // capture: the drag continues even if the pointer leaves the canvas
      if (e.pointerId !== undefined && this.over.setPointerCapture) this.over.setPointerCapture(e.pointerId);
    };
    const move = e => {
      if (!drag) return;
      const p = e.touches ? e.touches[0] : e;
      const dx = p.clientX - drag.x, dy = p.clientY - drag.y;
      drag.x = p.clientX; drag.y = p.clientY;
      if (drag.pan) { this.pan[0] += dx; this.pan[1] += dy; }
      // rotation: 0.01 rad per pixel (~100 px = 57°). Tilt is limited between −0.2 rad (slightly
      // below the table) and π/2 (from above), so the view does not flip upside down.
      else { this.yaw += dx * 0.01; this.elev = Math.max(-0.2, Math.min(Math.PI / 2, this.elev + dy * 0.01)); }
      this.dirty = true; this.draw();
      if (e.touches) e.preventDefault();                // keep the finger from scrolling the page
    };
    const end = () => { drag = null; };
    if (window.PointerEvent) {
      this.over.addEventListener("pointerdown", start);
      this.over.addEventListener("pointermove", move);
      this.over.addEventListener("pointerup", end);
    } else {                                         // old browsers: mouse and touch
      // (mousemove/mouseup on the window: without pointer capture, this is how the drag is not lost)
      this.over.addEventListener("mousedown", start);
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", end);
      this.over.addEventListener("touchstart", start);
      this.over.addEventListener("touchmove", move);
      this.over.addEventListener("touchend", end);
    }
    this.over.addEventListener("contextmenu", e => e.preventDefault());   // right button = move
    // Wheel zoom, centered on the pointer: exponential factor (each wheel «click», deltaY ≈
    // 100, changes zoom by ~15 %; equally smooth zooming in and out). The pan is corrected so the
    // point under the pointer stays still. passive: false because preventDefault() is called (otherwise
    // the wheel would also scroll the page).
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
window.PathView = PathView;      // global: cnc.js does not use modules (Firefox 52 does not support them)
