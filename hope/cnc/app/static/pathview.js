/* Vista de trayectoria en canvas 2D con proyección ortográfica (sin WebGL: corre en PCs viejas).
   Dos capas: «base» (trayectoria, se redibuja al cambiar la vista o el avance) y «marca» (herramienta). */
"use strict";

class PathView {
  constructor(host) {
    this.host = host;
    this.base = document.createElement("canvas");
    this.over = document.createElement("canvas");
    host.append(this.base, this.over);
    this.pts = [];
    this.limits = null;
    this.yaw = -Math.PI / 4;
    this.elev = 0.62;
    this.zoom = 1;
    this.pan = [0, 0];
    this.scale = 1;
    this.center = [0, 0];
    this.doneLine = 0;
    this.tool = null;
    this.dirty = true;
    this._bind();
    observeResize(host, () => { this.dirty = true; this.draw(); });
  }

  setData(pts, limits, pierces) {
    this.pts = pts || [];
    this.limits = limits;
    this.pierces = pierces || [];
    this.fit();
  }

  setView(v) {
    const views = { top: [0, Math.PI / 2], iso: [-Math.PI / 4, 0.62], front: [0, 0], side: [-Math.PI / 2, 0] };
    if (v === "fit") { this.fit(); return; }
    [this.yaw, this.elev] = views[v];
    this.fit();
  }

  project(x, y, z) {
    const c = Math.cos(this.yaw), s = Math.sin(this.yaw);
    const xr = x * c - y * s, yr = x * s + y * c;
    return [xr, yr * Math.sin(this.elev) + z * Math.cos(this.elev)];
  }

  bounds() {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    const add = (x, y, z) => {
      const [u, v] = this.project(x, y, z);
      if (u < x0) x0 = u; if (u > x1) x1 = u; if (v < y0) y0 = v; if (v > y1) y1 = v;
    };
    const step = Math.max(1, Math.floor(this.pts.length / 20000));
    for (let i = 0; i < this.pts.length; i += step) add(this.pts[i][0], this.pts[i][1], this.pts[i][2]);
    if (!this.pts.length) {
      const L = this.limits || {};
      const r = (a, i, d) => (L[a] && L[a][i] != null ? L[a][i] : d);
      for (const x of [r("X", 0, 0), r("X", 1, 100)]) for (const y of [r("Y", 0, 0), r("Y", 1, 100)])
        for (const z of [r("Z", 0, -10), r("Z", 1, 0)]) add(x, y, z);
    }
    if (this.tool) add(this.tool.X || 0, this.tool.Y || 0, this.tool.Z || 0);
    return [x0, x1, y0, y1];
  }

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

  toScreen(x, y, z) {
    const [u, v] = this.project(x, y, z);
    const k = this.scale * this.zoom;
    return [this.w / 2 + (u - this.center[0]) * k + this.pan[0], this.h / 2 - (v - this.center[1]) * k + this.pan[1]];
  }

  _size(c) {
    const r = window.TISMA_LITE ? 1 : (window.devicePixelRatio || 1);
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (c.width !== Math.round(w * r) || c.height !== Math.round(h * r)) { c.width = Math.round(w * r); c.height = Math.round(h * r); }
    const g = c.getContext("2d");
    g.setTransform(r, 0, 0, r, 0, 0);
    this.w = w; this.h = h;
    return g;
  }

  setProgress(line, tool) {
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
    if (this.follow && tool && moved && (!this._lastBase || now - this._lastBase > 120)) this.dirty = true;
    this.draw();
  }

  draw() {
    if (!this.host.clientWidth) return;
    if (this.follow && this.tool && this.dirty) {
      this.center = this.project(this.tool.X || 0, this.tool.Y || 0, this.tool.Z || 0);
      this.pan = [0, 0];
    }
    if (this.dirty) { this._drawBase(); this.dirty = false; this._lastBase = performance.now(); }
    this._drawOver();
  }

  _grid(g) {
    const L = this.limits || {};
    const lim = (a, i) => (L[a] && L[a][i] != null ? L[a][i] : null);
    let gx0 = lim("X", 0), gx1 = lim("X", 1), gy0 = lim("Y", 0), gy1 = lim("Y", 1);
    if (gx0 == null || gx1 == null || gy0 == null || gy1 == null) { gx0 = 0; gx1 = 100; gy0 = 0; gy1 = 100; }
    const span = Math.max(gx1 - gx0, gy1 - gy0);
    const stepRaw = span / 12;
    const p = Math.pow(10, Math.floor(Math.log10(stepRaw)));
    const step = [1, 2, 5, 10].map(m => m * p).find(s => s >= stepRaw) || p * 10;
    g.lineWidth = 1;
    g.strokeStyle = "rgba(180,170,195,0.07)";
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

  _drawBase() {
    const g = this._size(this.base);
    g.clearRect(0, 0, this.w, this.h);
    this._grid(g);
    const P = this.pts;
    if (!P.length) return;
    const css = getComputedStyle(document.documentElement);
    const v = (n, d) => (css.getPropertyValue(n) || d).trim() || d;
    const styles = {
      0: [v("--rapid", "#f87171"), 1, [4, 4], 0.6],
      1: [v("--accent", "#38bdf8"), 1.4, [], 1],
      2: [v("--done", "#34d399"), 1.8, [], 1],
      3: ["#5b6b7d", 1, [], 0.8],
    };
    let cur = null;
    let prev = null;
    const flush = () => { if (cur !== null) g.stroke(); };
    // modo liviano: dibujar como mucho ~40 000 puntos (los extremos de cada tramo se conservan)
    const stride = window.TISMA_LITE && P.length > 40000 ? Math.ceil(P.length / 40000) : 1;
    for (let i = 0; i < P.length; i++) {
      const p = P[i];
      if (stride > 1 && i % stride && p[3] > 0 && i + 1 < P.length && P[i + 1][3] === p[3]) continue;
      const s = this.toScreen(p[0], p[1], p[2]);
      if (p[3] >= 0 && prev) {
        const st = p[3] === 0 ? 0 : p[3] === 3 ? 3 : (this.doneLine && p[4] < this.doneLine ? 2 : 1);
        if (st !== cur) {
          flush();
          const [col, lw, dash, alpha] = styles[st];
          g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash); g.globalAlpha = alpha;
          g.beginPath(); g.moveTo(prev[0], prev[1]);
          cur = st;
        }
        g.lineTo(s[0], s[1]);
      } else if (cur !== null) {
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
    this._doneIdx = k;
    if (this.pierces && this.pierces.length) {
      g.strokeStyle = v("--torch", "#fb923c"); g.lineWidth = 1.5;
      g.beginPath();
      for (const q of this.pierces) { const s = this.toScreen(q[0], q[1], 0); g.moveTo(s[0] + 4, s[1]); g.arc(s[0], s[1], 4, 0, Math.PI * 2); }
      g.stroke();
    }
  }

  /* Pinta como «hecho» (verde) los tramos de corte entre lo ya pintado y la línea `line`, sobre el
     dibujo de fondo que ya está. Mismo estilo que el tipo 2 de _drawBase. */
  _drawDone(line) {
    const P = this.pts;
    let i = this._doneIdx || 0;
    if (!P.length || i >= P.length) { this._doneDrawn = line; return; }
    const g = this.base.getContext("2d");
    const css = getComputedStyle(document.documentElement);
    g.strokeStyle = (css.getPropertyValue("--done") || "#34d399").trim() || "#34d399";
    g.lineWidth = 1.8; g.setLineDash([]); g.globalAlpha = 1;
    g.beginPath();
    let prev = i > 0 ? this.toScreen(P[i - 1][0], P[i - 1][1], P[i - 1][2]) : null;
    for (; i < P.length && P[i][4] < line; i++) {
      const p = P[i];
      const s = this.toScreen(p[0], p[1], p[2]);
      if (prev && (p[3] === 1 || p[3] === 2)) { g.moveTo(prev[0], prev[1]); g.lineTo(s[0], s[1]); }
      prev = s;
    }
    g.stroke();
    this._doneIdx = i;
    this._doneDrawn = line;
  }

  _drawOver() {
    const g = this._size(this.over);
    g.clearRect(0, 0, this.w, this.h);
    if (!this.tool) return;
    const t = this.tool;
    const [x, y] = this.toScreen(t.X || 0, t.Y || 0, t.Z || 0);
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

  _bind() {
    let drag = null;
    const start = e => {
      const p = e.touches ? e.touches[0] : e;
      drag = { x: p.clientX, y: p.clientY, pan: e.shiftKey || (e.button !== undefined && e.button !== 0) || (e.touches && e.touches.length > 1) };
      if (e.pointerId !== undefined && this.over.setPointerCapture) this.over.setPointerCapture(e.pointerId);
    };
    const move = e => {
      if (!drag) return;
      const p = e.touches ? e.touches[0] : e;
      const dx = p.clientX - drag.x, dy = p.clientY - drag.y;
      drag.x = p.clientX; drag.y = p.clientY;
      if (drag.pan) { this.pan[0] += dx; this.pan[1] += dy; }
      else { this.yaw += dx * 0.01; this.elev = Math.max(-0.2, Math.min(Math.PI / 2, this.elev + dy * 0.01)); }
      this.dirty = true; this.draw();
      if (e.touches) e.preventDefault();
    };
    const end = () => { drag = null; };
    if (window.PointerEvent) {
      this.over.addEventListener("pointerdown", start);
      this.over.addEventListener("pointermove", move);
      this.over.addEventListener("pointerup", end);
    } else {                                         // navegadores viejos: ratón y táctil
      this.over.addEventListener("mousedown", start);
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", end);
      this.over.addEventListener("touchstart", start);
      this.over.addEventListener("touchmove", move);
      this.over.addEventListener("touchend", end);
    }
    this.over.addEventListener("contextmenu", e => e.preventDefault());
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
window.PathView = PathView;
