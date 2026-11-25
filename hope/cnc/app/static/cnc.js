/* ==================================================================================================
   TISMA CNC (Hope CNC) — screens with the Mach3Mill layout. No dependencies: runs in old and new
   browsers. The repeated groups (DRO, tool, feed, spindle) come from templates and are updated
   through the data-b attribute, so each screen has its own, as in Mach3.
   ==================================================================================================

   WHAT IT IS
     The control screen logic: it reads the machine state, shows it (DRO, LEDs, G-code listing,
     toolpath) and sends the server what the operator asks for (buttons, keyboard, MDI, jog).
     It decides nothing about the machine: every action is validated and executed by the server
     (hope/cnc/app/server.py → hope/cnc/session.py), and moving a real machine requires the guardian's
     approval (the «Aprobar» (Approve) dialog).

   HOW IT FITS
     * index.html loads, in this order: viejo.js (notice if the browser is too old), compat.js,
       pathview.js and this file. All are scripts served by the server itself (the CSP allows no inline scripts).
     * From compat.js it uses: onPress(el, press, release) for buttons that are held down (jog),
       readText(file) to read files, normKey(event) for the key name (old browsers say "Left",
       "Esc"…), setLite(on) and window.TISMA_LITE (light mode for old PCs).
     * From pathview.js it uses the PathView class (one toolpath drawing per [data-view]).
     * From the server it uses: GET /api/estado (every 150 ms; 400 ms in light mode), /api/config,
       /api/programa, /api/trayectoria, /api/origenes, /api/recientes, /api/perfiles, /api/perfil and
       POST /api/accion (see act()).

   HTML CONVENTIONS
     * data-b="key"     element showing a value; setB/clsB/led update it on ALL screens.
     * data-act="x"     button that runs ACTIONS.x (a single click handler for the whole page).
     * data-ov="t:a"    override button (feed/spindle/rapid): sends act("override", {tipo, accion}).
     * data-mdi="G54"   button that sends that line through MDI.
     * data-jog="X:1"   jog button: axis and direction.
     * data-tpl="dro"   box filled with the <template id="tpl-dro"> template.
     * data-view="run"  toolpath canvas of the «run» screen.
     The texts the operator sees and the API names are in Spanish; units in mm, mm/min, rpm and
     seconds, as in the server.

   COMPATIBILITY
     Firefox 52 ESR (Windows XP/Vista) and Chrome 57 onwards: ES2017 syntax (async/await, arrows,
     template literals, default parameters). Do NOT use optional chaining, nullish coalescing,
     replaceAll, at, flatMap, Object.fromEntries, structuredClone, Response.text or ResizeObserver
     (tests/test_cnc_compat.py checks it); for pointer events, use onPress() from compat.js.

   EXAMPLE
     <button class="mbtn" data-act="ciclo">Cycle Start</button>   → on click: act("ciclo")
     <span data-b="fro"></span>%                                   → render() writes the feed override
   ================================================================================================== */
"use strict";

// =================================================================================================
// ================================== GLOBAL STATE AND CONSTANTS ===================================
// =================================================================================================

// Shortcut: $("mdi") = document.getElementById("mdi"). No jQuery: no dependencies.
const $ = id => document.getElementById(id);
// S: the whole screen state in a single object (easy to inspect from the browser console).
//   st        last /api/estado response (null until the first poll)
//   cfg       effective machine configuration (/api/config)
//   lines     lines of the loaded program, as shown in the listing
//   sha       sha256 of the shown program: undefined = not queried yet, null = no program
//   errLines  line numbers with errors (painted red); errSha = which program they belong to
//   coord     "work" or "machine": which coordinates the big DRO shows (Mach3's Machine Coords)
//   tab       visible screen (run, mdi, profiles, path, offsets, config, diag)
//   jogOn     jog enabled (Ctrl-Alt-J in Mach3)
//   jogStep   false = continuous jog while held down; true = step jog (Mach3's Step)
//   steps     available jog steps in mm; stepIdx = the chosen one (0.1 mm at start)
//   jogPct    jog speed as % of the axis maximum (Mach3's Slow Jog Rate), between 5 and 100
//   jogKeys   jog keys currently held (to ignore keyboard auto-repeat)
//   lastLine  last running line seen (to follow it in the listing without redrawing too much)
//   follow    the toolpath view follows the tool
//   axes      axes the DRO is built for; viewIdx = index in VIEWS (1 = ISO)
//   programLoadedAt  reserved (not used for now)
// Added later: followCode/followT (follow the current line in the listing), lastNext,
// clearedAt (when the status line was cleared) and pk/prof/presets (profiles).
const S = { st: null, cfg: null, lines: [], sha: undefined, errLines: new Set(), coord: "work", tab: "run",
            jogOn: true, jogStep: false, steps: [0.001, 0.01, 0.1, 1, 10], stepIdx: 2, jogPct: 30,
            jogKeys: new Set(), lastLine: -1, follow: false, axes: [], viewIdx: 1, programLoadedAt: 0 };
// Toolpath views, in the order the «Vista» button (Mach3's Display Mode) cycles through them.
const VIEWS = ["top", "iso", "front", "side"];
// Display name of each view.
const VIEW_NAMES = { top: "Arriba", iso: "ISO", front: "Frente", side: "Lado" };
// Height of each G-code listing line in px. It must equal «.code-row { height: 21px }» in
// cnc.css: the virtual list uses it to compute which lines are in view.
const ROW = 21;

// =================================================================================================
// ================================= COMMUNICATION WITH THE SERVER =================================
// =================================================================================================

// --- communication ---------------------------------------------------------------
// api(path): GET of a server query (/api/...); returns the already parsed JSON.
// No cache, so a stale state is never shown. If the network fails, the promise rejects (the
// caller handles it; poll() shows it as «Sin conexión» (No connection)).
async function api(path) { const r = await fetch(path, { cache: "no-store" }); return r.json(); }
// act(action, params, quiet): runs a server action (ACTIONS whitelist in server.py).
//   action  name ("ciclo", "jog", "mdi"…); params: its parameters; quiet: do not show errors.
// Returns the response ({ok, mensaje, …}); without a connection, {ok: false}. Never throws.
// WHY the X-Hope header and JSON: the server requires them so that another web page cannot
// send actions (CSRF); a form from another site cannot set either of them.
// If the guardian asks for approval (j.aprobar), it opens the dialog; and it always refreshes the
// state right away, so the button responds without waiting for the next periodic poll.
async function act(accion, params = {}, quiet = false) {
  try {
    const r = await fetch("/api/accion", { method: "POST", headers: { "Content-Type": "application/json", "X-Hope": "1" },
                                           body: JSON.stringify({ accion, params }) });
    const j = await r.json();
    if (!j.ok && !quiet) toast(j.mensaje || "error");
    if (j.aprobar) showApproval(j.aprobar);
    poll(true);
    return j;
  } catch (e) { if (!quiet) toast("sin conexión con el control CNC"); return { ok: false }; }
}
// toast(msg, kind): floating notice at the bottom of the screen. kind "info" = green, lasts 2.5 s;
// no kind = error, lasts 5 s. textContent (not innerHTML): the message may carry text from the file.
function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = "toast " + kind; t.textContent = msg;
  $("toasts").append(t);
  setTimeout(() => t.remove(), kind === "info" ? 2500 : 5000);
}
// esc(s): escapes & < > " to put text into hand-built HTML (file names, G-code comments,
// profiles). Every piece of data that is not ours goes through here before reaching innerHTML.
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// =================================================================================================
// ====================================== SCREEN CONSTRUCTION ======================================
// =================================================================================================

// --- data bindings (data-b) ----------------------------------------------------------
// setB(key, text): puts the text in every [data-b=key]. It only touches the DOM if it changed: it
// is called dozens of times every 150 ms and always writing would make the page re-render (slow on old PCs).
function setB(key, text) { document.querySelectorAll(`[data-b="${key}"]`).forEach(e => { if (e.textContent !== String(text)) e.textContent = text; }); }
// clsB(key, cls, on): adds or removes a CSS class on every [data-b=key].
function clsB(key, cls, on) { document.querySelectorAll(`[data-b="${key}"]`).forEach(e => e.classList.toggle(cls, !!on)); }
// led(key, on, blink): Mach3 LED (the little square next to each button): lit and/or blinking.
function led(key, on, blink = false) { document.querySelectorAll(`[data-b="${key}"]`).forEach(e => { e.classList.toggle("on", !!on); e.classList.toggle("blink", !!blink); }); }

// --- templates (Mach3 groups repeat on several screens) -----------------------
// Copies each template into its boxes when the page loads (Mach3 repeats the DRO, the tool, etc. on
// several screens; here they are written only once in index.html).
document.querySelectorAll("[data-tpl]").forEach(box => {
  box.append($("tpl-" + box.dataset.tpl).content.cloneNode(true));
  if (box.dataset.tpl === "dro") box.classList.add("dro-panel");
});
// buildDro(axes): builds the DRO rows (Zero button, referenced LED and value) for the machine's
// axes, e.g. ["X","Y","Z"] or with "A". Rebuilt when the machine changes; the 4th-axis buttons
// (.ax-a) are hidden if there is no A. «Cero» = G10 L20 P1 (Mach3's Zero X/Y/Z).
function buildDro(axes) {
  S.axes = axes;
  document.querySelectorAll("[data-dro-rows]").forEach(rows => {
    rows.innerHTML = axes.map(a => `
      <div class="dro-row">
        <button class="mbtn zero" data-act="cero" data-eje="${a}" title="Poner ${a} en cero aquí (G10 L20 P1)">Cero<b>${a}</b></button>
        <span class="ref-led" data-b="ref-${a}" title="referenciado"></span>
        <div class="dro-val"><div class="v" data-b="dro-${a}">+0.0000</div><div class="s" data-b="dro-sub-${a}"></div></div>
      </div>`).join("");
  });
  // Rotary jog buttons: each one only if the machine has that axis; the B/C row only if it has B or C.
  for (const a of ["A", "B", "C"]) {
    document.querySelectorAll(".ax-" + a.toLowerCase()).forEach(b => b.style.visibility = axes.includes(a) ? "" : "hidden");
  }
  document.querySelectorAll(".ax-bc").forEach(e => e.hidden = !(axes.includes("B") || axes.includes("C")));
}

// --- screens (Alt-1, 2, 4, 5, 6, 7 as in Mach3) --------------------------------------------
// Key (with Alt) → screen. Mach3: Alt-1 Program Run, Alt-2 MDI, Alt-4 ToolPath, Alt-5 Offsets,
// Alt-6 Settings, Alt-7 Diagnostics. Alt-3 (old Toolpath in Mach3) is used here for profiles.
const TAB_KEYS = { "1": "run", "2": "mdi", "3": "profiles", "4": "path", "5": "offsets", "6": "config", "7": "diag" };
// setTab(name): shows one screen and hides the others. On entering it loads what that screen
// needs (offsets, configuration, profiles) and refits the toolpath view to the new size
// (with setTimeout, after the browser has shown the screen and knows its size).
function setTab(name) {
  S.tab = name;
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".screen").forEach(t => t.classList.toggle("on", t.id === "screen-" + name));
  if (name === "mdi") setTimeout(() => $("mdi").focus(), 0);
  if (name === "offsets") loadOffsets();
  if (name === "config") loadConfig();
  if (name === "run") renderCode();
  if (name === "profiles") openProfileTab();
  setTimeout(() => { const v = curView(); if (v) { v.dirty = true; v.fit(); } }, 0);
}
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => setTab(b.dataset.tab));

// =================================================================================================
// ===================================== PROGRAM AND TOOLPATH ======================================
// =================================================================================================

// --- toolpath ----------------------------------------------------------------------------------
// One PathView (pathview.js) per [data-view] canvas; the key is the screen it is on.
const views = {};
document.querySelectorAll("[data-view]").forEach(el => { views[el.dataset.view] = new PathView(el); });
// The toolpath view of the visible screen (or null if it has none).
const curView = () => views[S.tab] || null;
// Last toolpath received: points [x, y, z, line…], limits and pierces (plasma).
let pathData = { puntos: [], limites: null, perforaciones: [] };
// loadPath(): requests the toolpath from the server and passes it to every view. If the program is
// very large the server trims it (recortado) so drawing does not stall an old PC.
async function loadPath() {
  pathData = await api("/api/trayectoria");
  for (const v of Object.values(views)) { v.follow = S.follow; v.setData(pathData.puntos, pathData.limites, pathData.perforaciones); v.setView(VIEWS[S.viewIdx]); }
  if (pathData.recortado) toast("trayectoria recortada para dibujar (programa muy grande)", "info");
  renderLimits();
}
// renderLimits(): table of the program's X, Y, Z minimums and maximums (like Mach3's «Program
// Limits»), to see before running whether it fits on the table. With a plain for: there can be
// hundreds of thousands of points and spreading them into Math.min would overflow the stack.
// wrapTool(pos): tool position for the view. With rotary axes, the toolpath is drawn on the
// workpiece (wrapped, see _view_point_function in session.py): the tool is rotated the same way,
// opposite to each axis angle (A around X, B around Y, C around Z), so that they match.
function wrapTool(pos) {
  // Same math as _view_point_function in session.py: for each table rotary axis [axis, cu, cv]
  // (centre in G54 on the axis's plane), rotate the tool point by −angle around that centre.
  const PLANE = { A: ["Y", "Z"], B: ["Z", "X"], C: ["X", "Y"] };
  const p = { X: pos.X || 0, Y: pos.Y || 0, Z: pos.Z || 0 };
  for (const r of pathData.rotativos || []) {
    const a = r[0], cu = r[1], cv = r[2], u = PLANE[a][0], v = PLANE[a][1];
    const ang = -(pos[a] || 0) * Math.PI / 180, c = Math.cos(ang), s = Math.sin(ang);
    const du = p[u] - cu, dv = p[v] - cv;
    p[u] = cu + du * c - dv * s; p[v] = cv + du * s + dv * c;
  }
  return p;
}
function renderLimits() {
  const P = pathData.puntos || [];
  if (!P.length) { $("limits-table").innerHTML = '<tr><td class="muted">sin programa</td></tr>'; return; }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of P) for (let i = 0; i < 3; i++) { if (p[i] < lo[i]) lo[i] = p[i]; if (p[i] > hi[i]) hi[i] = p[i]; }
  $("limits-table").innerHTML = "<tr><th>Eje</th><th>Mín</th><th>Máx</th></tr>" +
    ["X", "Y", "Z"].map((a, i) => `<tr><td>${a}</td><td>${lo[i].toFixed(3)}</td><td>${hi[i].toFixed(3)}</td></tr>`).join("");
}

// --- G-code (virtual list: performs well with hundreds of thousands of lines) ---------------------
// hl(line): highlights a G-code line for the listing: comments (in parentheses or after «;»),
// G codes and M/T codes. Returns HTML already escaped with esc().
function hl(line) {
  return line.split(/(\(.*?\)|;.*$)/).map(p => {
    if (!p) return "";
    if (p.startsWith("(") || p.startsWith(";")) return `<span class="c">${esc(p)}</span>`;
    return esc(p).replace(/\b([Gg]\d+(?:\.\d+)?)/g, '<span class="g">$1</span>').replace(/\b([MmTt]\d+)/g, '<span class="m">$1</span>');
  }).join("");
}
// renderCode(): draws the G-code listing as a «virtual list»: a spacer with the total height
// (lines × ROW) gives the real scrollbar, and only the visible rows are created plus a
// margin (10 above, 20 below). So it works the same with 300 000 lines as with 30.
// Marks the current line (cur), the done ones (done), the chosen next one (next) and the errors (err).
function renderCode() {
  const box = $("code"), rows = $("code-rows");
  $("code-empty").hidden = S.lines.length > 0;
  $("code-spacer").style.height = S.lines.length * ROW + "px";
  const p = S.st && S.st.programa;
  const cur = p ? p.linea_actual : 0, next = S.st ? S.st.linea_siguiente : 0;
  const first = Math.max(0, Math.floor(box.scrollTop / ROW) - 10);
  const last = Math.min(S.lines.length, first + Math.ceil(box.clientHeight / ROW) + 20);
  rows.style.top = first * ROW + "px";
  let html = "";
  for (let i = first; i < last; i++) {
    const n = i + 1;
    let cls = n === cur ? "cur" : (cur && n < cur ? "done" : "");
    if (next && n === next) cls += " next";
    if (S.errLines.has(n)) cls += " err";
    html += `<div class="code-row ${cls}" data-line="${n}"><span class="n">${n}</span><span>${hl(S.lines[i])}</span></div>`;
  }
  rows.innerHTML = html;
}
$("code").addEventListener("scroll", renderCode);
// If the operator moves the wheel, it stops following the current line for 4 s to let them look around.
$("code").addEventListener("wheel", () => { S.followCode = false; clearTimeout(S.followT); S.followT = setTimeout(() => S.followCode = true, 4000); });
// Clicking a line writes it into the «Set Next Line» box (as in Mach3).
$("code").addEventListener("click", e => { const r = e.target.closest(".code-row"); if (r) $("line-in").value = r.dataset.line; });
S.followCode = true;
// followLine(n): scrolls the listing so line n is in view (at one third of the height),
// only if it is leaving the visible margin, so the listing does not jump on every line.
function followLine(n) {
  if (!n || !S.followCode) return;
  const box = $("code"), y = (n - 1) * ROW;
  if (y < box.scrollTop + ROW * 2 || y > box.scrollTop + box.clientHeight - ROW * 4) box.scrollTop = y - box.clientHeight / 3;
}
// loadProgram(): fetches the loaded program's lines and its toolpath, and goes back to the start.
async function loadProgram() {
  const p = await api("/api/programa");
  S.lines = p.lineas || [];
  $("code").scrollTop = 0;
  renderCode();
  await loadPath();
}
// loadFile(file): sends a chosen (or dropped) file to the server to be checked and loaded.
// The 32 MB limit is the same as the server's MAX_BODY: the user is warned before sending, to avoid waiting in vain.
async function loadFile(file) {
  if (!file) return;
  if (file.size > 32 * 1024 * 1024) return toast("archivo demasiado grande (máx. 32 MB)");
  const r = await act("cargar", { nombre: file.name, texto: await readText(file) });
  if (r.ok) { afterLoad(r, file.name); }
}
// afterLoad(response, name): after loading a program (file, editor or recent files) reports
// whether it is ready or has errors, and shows it.
async function afterLoad(r, name) {
  S.sha = r.informe.sha256;                 // we already loaded it: poll() must not request it again
  (r.notas || []).forEach(n => toast("DXF: " + n));   // warnings of a DXF (open contours, tiny holes…)
  toast(r.informe.runnable ? `«${name}» revisado: listo` : `«${name}» tiene errores: mirá Diagnóstico (Alt-7)`, r.informe.runnable ? "info" : "");
  await loadProgram();
}
// Choose file (Load G-Code). The value is cleared so the same file can be chosen again.
$("file").onchange = e => { loadFile(e.target.files[0]); e.target.value = ""; };
// Dragging and dropping a file anywhere in the window also loads it.
document.addEventListener("dragover", e => { e.preventDefault(); document.body.classList.add("drop"); });
document.addEventListener("dragleave", e => { if (!e.relatedTarget) document.body.classList.remove("drop"); });
document.addEventListener("drop", e => { e.preventDefault(); document.body.classList.remove("drop"); loadFile(e.dataTransfer.files[0]); });

// =================================================================================================
// =============================== MACHINE STATE (PERIODIC POLLING) ================================
// =================================================================================================

// --- state ----------------------------------------------------------------------------------
// fmt4(v): DRO number with sign and 4 decimals, as in Mach3 (+12.3400 / −0.5000). The «−» is the
// typographic minus sign, as wide as «+», so the digits do not dance.
const fmt4 = v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(4);
// hms(seconds): «hh:mm:ss» for elapsed and remaining time; null → 00:00:00.
const hms = s => { if (s == null) return "00:00:00"; s = Math.max(0, Math.round(s)); return [s / 3600, s % 3600 / 60, s % 60].map(x => String(Math.floor(x)).padStart(2, "0")).join(":"); };
// Controller state (grbl names) → text shown in the state «pill».
const STATE_NAMES = { Idle: "En reposo", Run: "Ejecutando", Hold: "En pausa", Jog: "Jog", Home: "Referenciando", Alarm: "ALARMA",
                      Desconectado: "Desconectado", Reset: "RESET", Door: "Puerta" };
// A status poll is in progress: on a slow PC polls do not pile up one after another.
let polling = false;
// poll(force): requests /api/estado and shows it with render(). If a poll is already in progress
// it does nothing, except with force (after an action, so the effect shows right away).
async function poll(force) {
  if (polling && !force) return;
  polling = true;
  try { render(await api("/api/estado")); }
  catch (e) { setB("pill", "Sin conexión"); clsB("pill", "Desconectado", true); }
  finally { polling = false; }
}
// render(st): dumps the machine state onto the whole screen (DRO, tool, times,
// overrides, LEDs, plasma, RESET, program, messages). Called on every poll (every 150 ms), which
// is why everything goes through setB/clsB/led, which only touch the DOM if something changed.
// st: /api/estado response (see CncSession.state in hope/cnc/session.py).
function render(st) {
  S.st = st;
  const c = st.estado, job = c.trabajo, p = st.programa;
  watchProbe(st.palpado);
  document.querySelectorAll(".mill-only").forEach(e => { e.hidden = !!st.plasma; });
  if (S.axes.join() !== st.ejes.join()) buildDro(st.ejes);
  const state = st.habilitada ? c.estado : "Reset";
  document.querySelectorAll('[data-b="pill"]').forEach(e => { e.className = "pill " + state; e.textContent = STATE_NAMES[state] || state; });
  setB("modal", st.modal || "—");
  setB("tipo-label", { plasma: "Plasma", laser: "Láser", router: "Router", fresadora: "Fresadora" }[st.tipo] || st.tipo);
  // DRO
  for (const a of st.ejes) {
    const w = st.trabajo_pos[a], m = c.mpos[a];
    const main = S.coord === "work" ? w : m, sub = S.coord === "work" ? m : w;
    setB("dro-" + a, fmt4(main));
    setB("dro-sub-" + a, (S.coord === "work" ? "máquina " : "trabajo ") + fmt4(sub));
    clsB("ref-" + a, "on", c.referenciado);
  }
  document.querySelectorAll(".dro-panel").forEach(e => e.classList.toggle("machine", S.coord === "machine"));
  led("led-coord", S.coord === "machine", S.coord === "machine");
  // tool
  const t = st.herramienta || {};
  setB("tool-n", "T" + (t.n || 0)); setB("tool-num", t.n || 0);
  setB("tool-dia", (t.diametro || 0).toFixed(3)); setB("tool-len", (t.largo || 0).toFixed(3));
  led("led-cambio", t.cambio, t.cambio);
  setB("elapsed", hms(p ? p.transcurrido : null));
  setB("left", p && job.activo ? hms(p.tiempo_restante) : p ? "estimado " + hms(p.tiempo_total) : "—");
  led("led-jog", S.jogOn);
  // feed and spindle
  const fprog = (st.modal || "").match(/F([\d.]+)/), sprog = (st.modal || "").match(/S([\d.]+)/);
  setB("fro", c.ov.avance); setB("sro", c.ov.husillo);
  const fp = c.avance_prog != null ? c.avance_prog : (fprog ? +fprog[1] : 0);
  setB("fro-val", Math.round(fp * c.ov.avance / 100));
  setB("f-prog", Math.round(fp)); setB("s-prog", sprog ? sprog[1] : 0);
  setB("feed-now", (+c.avance).toFixed(2)); setB("rpm-now", Math.round(c.husillo));
  document.querySelectorAll('[data-b="rapid-seg"] button').forEach(b => b.classList.toggle("on", b.dataset.ov === "rapido:" + c.ov.rapido));
  led("led-husillo", c.husillo_dir !== "off");
  led("led-refrigerante", c.refrigerante);
  led("led-g4", c.pausa_g4);
  for (const k of ["bloque_a_bloque", "borrado_bloque", "parada_opcional"]) led("led-" + k, st.opciones && st.opciones[k]);
  // plasma
  if (st.plasma) {
    clsB("torch-state", "on", st.plasma.antorcha);
    setB("torch-label", st.plasma.antorcha ? "ENCENDIDA" : "APAGADA");
    setB("pierce-count", `${st.plasma.perforaciones_hechas} / ${st.plasma.perforaciones_total}`);
  }
  for (const v of Object.values(views)) v.hot = !!(st.plasma && st.plasma.antorcha);
  // RESET (as in Mach3: red/green LED that blinks until enabled)
  clsB("reset", "on", st.habilitada);
  document.querySelectorAll('[data-b="reset-led"]').forEach(e => { e.classList.toggle("on", st.habilitada); e.classList.toggle("blink", !st.habilitada); });
  document.body.classList.toggle("running", job.activo && c.estado === "Run");
  // program
  setB("file", p ? p.nombre : "sin programa");
  // Profiles chosen from another window (or tablet): rebuild the bar, the machine and the program.
  const sk = selectionKey(st.perfiles && st.perfiles.seleccion);
  if (S.selKey === undefined) S.selKey = sk;          // first poll: startup already loaded them
  else if (sk !== S.selKey) { S.selKey = sk; applySelection(); }
  // First poll: the program that was already loaded is fetched by startup (loadProgram); just note it.
  if (S.sha === undefined) S.sha = p ? p.sha256 : null;
  if (p) {
    if (p.sha256 !== S.sha) {
      // New program loaded from another window (or tablet): fetch the listing and the toolpath.
      // If this window loaded it, afterLoad() already set S.sha and it is not requested twice.
      S.sha = p.sha256;
      loadProgram();
    }
    if (S.errSha !== p.sha256) { S.errSha = p.sha256; S.errLines = new Set(p.errores.map(e => +((e.match(/línea (\d+)/) || [])[1] || 0))); }
    document.querySelectorAll('[data-b="progress"]').forEach(e => e.style.width = (job.terminado ? 100 : p.progreso * 100) + "%");
    setB("line-info", `Línea ${p.linea_actual} / ${p.lineas_total}` + (job.activo ? ` · faltan ${hms(p.tiempo_restante)}` : ""));
    setB("report", p.runnable ? `✓ Listo · ${p.avisos.length} avisos · ${hms(p.tiempo_total)} estimado` : `✕ ${p.errores.length} errores — ${p.errores[0] || ""}`);
    clsB("report", "ok", p.runnable); clsB("report", "bad", !p.runnable);
    if (p.linea_actual !== S.lastLine) { S.lastLine = p.linea_actual; followLine(p.linea_actual); if (S.tab === "run") renderCode(); }
    $("diag-report").textContent = p.informe;
  } else {
    document.querySelectorAll('[data-b="progress"]').forEach(e => e.style.width = "0");
    setB("line-info", "Línea 0 / 0"); setB("report", "Cargá un programa para revisarlo."); clsB("report", "ok", false); clsB("report", "bad", false);
    $("diag-report").textContent = "Sin programa.";
    if (S.sha) { S.sha = null; S.lines = []; renderCode(); loadPath(); }
  }
  setB("next-info", st.linea_siguiente ? `Próximo Cycle Start: línea ${st.linea_siguiente}${st.desde_aqui ? " (correr desde aquí)" : ""}` : "");
  if (st.linea_siguiente !== S.lastNext) { S.lastNext = st.linea_siguiente; renderCode(); }
  const v = curView();
  if (v) { const tool = wrapTool(st.trabajo_pos); v.setProgress(job.activo || job.terminado ? (p ? p.linea_actual : 0) : 0, tool); }
  // status (bottom line), profile
  const last = st.mensajes[st.mensajes.length - 1];
  const msg = c.alarma ? "ALARMA: " + c.alarma : c.mensaje ? "Mensaje: " + c.mensaje : job.error ? "Detenido: " + job.error : last ? last.m : "—";
  setB("status", S.clearedAt && last && last.t <= S.clearedAt && !c.alarma && !c.mensaje ? "" : msg);
  setB("machine", st.maquina); setB("conn", "· " + c.conexion + (st.fisica ? "" : " (sin máquina)"));
  if (S.tab === "diag") {
    $("diag-state").textContent = JSON.stringify(c, null, 2);
    $("diag-log").innerHTML = st.mensajes.slice().reverse().map(x => `<li><span class="t">${new Date(x.t * 1000).toLocaleTimeString()}</span>${esc(x.m)}</li>`).join("");
  }
  if (S.tab === "mdi") {
    const html = st.mdi_historial.slice().reverse().map(x => `<li data-cmd="${esc(x)}">${esc(x)}</li>`).join("") || '<li class="muted">vacío</li>';
    if ($("mdi-history").innerHTML !== html) $("mdi-history").innerHTML = html;
  }
  if (S.tab === "offsets") setB("wcs-cur", c.wcs || "G54");
  if (st.pendiente && $("modal").hidden) showApproval(st.pendiente);
}
// startPolling(): (re)starts the periodic state polling. Every 150 ms (≈7 per second: the DRO looks
// smooth) or every 400 ms in light mode, to leave CPU to an old PC. Called again when
// light mode changes.
let pollTimer = null;
function startPolling() { clearInterval(pollTimer); pollTimer = setInterval(() => poll(), window.TISMA_LITE ? 400 : 150); }
startPolling();

// =================================================================================================
// ===================================== BUTTONS AND CONTROLS ======================================
// =================================================================================================

// --- actions (delegation: the buttons are repeated on several screens) --------------------------
// ACTIONS: what each [data-act] button does. Most send the action of the same name to the server;
// some are screen-only (coord, vista, seguir, jog_*). Each function receives the button (b)
// to read its data-*. Mach3 equivalents: ciclo = Cycle Start, pausa = Feed Hold, parar = Stop,
// rebobinar = Rewind, cerrar = Close G-Code, editar = Edit G-Code, recientes = Recent File,
// fijar_linea / desde_aqui = Set Next Line / Run From Here, referenciar = Ref All Home,
// ir_a_cero = Goto Zero, coord = Machine Coords, regen = Regen. Toolpath, vista = Display Mode.
const ACTIONS = {
  ciclo: () => act("ciclo"), pausa: () => act("pausa"), parar: () => act("parar"), reset: () => act("reset"),
  rebobinar: () => { act("rebobinar"); $("code").scrollTop = 0; },
  cargar: () => $("file").click(),
  cerrar: async () => { const r = await act("cerrar_programa"); if (r.ok) loadProgram(); },
  editar: editProgram, recientes: showRecent,
  fijar_linea: () => setNext(false), desde_aqui: () => setNext(true),
  opcion: b => act("opcion", { nombre: b.dataset.name }),
  refrigerante: () => act("refrigerante"),
  husillo: () => act("husillo", { on: !(S.st && S.st.estado.husillo_dir !== "off") }),
  referenciar: () => act("referenciar"), ir_a_cero: () => act("ir_a_cero"),
  cero: b => act("cero", { eje: b.dataset.eje }), cero_todo: () => act("cero", { eje: "todo" }),
  coord: () => { S.coord = S.coord === "work" ? "machine" : "work"; if (S.st) render(S.st); },
  regen: () => loadPath(),
  vista: b => {
    S.viewIdx = b.dataset.v ? VIEWS.indexOf(b.dataset.v) : (S.viewIdx + 1) % VIEWS.length;
    setB("vista", VIEW_NAMES[VIEWS[S.viewIdx]]);
    for (const v of Object.values(views)) v.setView(VIEWS[S.viewIdx]);
  },
  seguir: () => { S.follow = !S.follow; led("led-seguir", S.follow); for (const v of Object.values(views)) { v.follow = S.follow; v.dirty = true; v.draw(); } },
  sensar: () => act("sensar_altura"),
  disparo: async () => {
    const ok = await openModal("Disparo de prueba", "La antorcha se enciende 0,3 s en la posición actual.",
      "Antes: careta puesta, nadie mirando el arco, chapa debajo de la antorcha y aire a presión.", "Disparar");
    if (ok) act("disparo_prueba");
  },
  jog_onoff: () => { S.jogOn = !S.jogOn; led("led-jog", S.jogOn); toast(S.jogOn ? "Jog habilitado" : "Jog deshabilitado", "info"); },
  jog_cerrar: () => $("jog").hidden = true,
  jog_modo: () => { S.jogStep = !S.jogStep; renderJog(); },
  jog_paso: () => { S.stepIdx = (S.stepIdx + 1) % S.steps.length; renderJog(); },
  jog_pct: b => { S.jogPct = Math.max(5, Math.min(100, S.jogPct + +b.dataset.d)); renderJog(); },
  fijar_z: () => {
    const g = parseFloat($("gage").value.replace(",", "."));
    if (!isFinite(g)) return toast("escribí el espesor del calibre");
    act("mdi", { linea: `G10 L20 P${wcsP()} Z${g}` }).then(r => r.ok && toast(`Z fijado: la punta está a ${g} sobre la pieza`, "info"));
  },
  aplicar_origen: applyOffset, origenes_refrescar: () => loadOffsets(),
  ayuda_g: () => helpCodes("G"), ayuda_m: () => helpCodes("M"), asistentes: () => openWizards(), palpado: () => openProbing(),
  historial: () => setTab("diag"),
  perfiles_editar: () => setTab("profiles"),
  perfil_usar: () => S.prof && chooseProfile(S.prof.tipo, S.prof.nombre),
  perfil_guardar: () => saveProfile(false), perfil_guardar_como: () => saveProfile(true),
  perfil_exportar: exportProfile, perfil_borrar: deleteProfile,
  perfil_importar: () => $("prof-file").click(),
  borrar_estado: () => { S.clearedAt = Date.now() / 1000; setB("status", ""); },
};
// A single click handler for the whole page («delegation»): the buttons are repeated on several
// screens and some are created later (DRO, offsets), and this way all work without registering each.
document.addEventListener("click", e => {
  const b = e.target.closest("[data-act]");
  if (b && ACTIONS[b.dataset.act]) { ACTIONS[b.dataset.act](b); return; }
  const ov = e.target.closest("[data-ov]");
  if (ov) { const [tipo, accion] = ov.dataset.ov.split(":"); act("override", { tipo, accion }); return; }
  const mdi = e.target.closest("[data-mdi]");
  if (mdi) { act("mdi", { linea: mdi.dataset.mdi }); return; }
  const h = e.target.closest("#mdi-history li[data-cmd]");
  if (h) { $("mdi").value = h.dataset.cmd; $("mdi").focus(); }
});
// setNext(fromHere): Set Next Line (false) or Run From Here (true) with the line typed in the box.
// Run From Here also brings the machine to that line's modal state before continuing.
function setNext(fromHere) {
  const n = parseInt($("line-in").value, 10) || 0;
  act("linea_siguiente", { linea: n, desde_aqui: fromHere });
}

// --- jog ---------------------------------------------------------------------------------------
// renderJog(): shows the mode (continuous/step), the chosen step and the jog speed %.
function renderJog() {
  led("led-jog-cont", !S.jogStep); led("led-jog-step", S.jogStep);
  setB("jog-step", S.steps[S.stepIdx].toFixed(4)); setB("jog-pct", S.jogPct);
}
// jogStart(axis, direction, fast): starts moving an axis. direction: +1 or −1; fast: with Shift, at
// maximum speed (as in Mach3). In step mode it moves S.steps[S.stepIdx] mm and stops by itself.
function jogStart(axis, dir, fast) {
  if (!S.jogOn) return toast("Jog deshabilitado (Ctrl-Alt-J)");
  if (!S.axes.includes(axis)) return;
  act("jog", { eje: axis, dir, paso: S.jogStep ? S.steps[S.stepIdx] : null, rapido: fast, porcentaje: S.jogPct });
}
// jogEnd(): the key or button was released: in continuous jog, stop now (no error messages).
function jogEnd() { if (!S.jogStep) act("jog_parar", {}, true); }
// On-screen jog buttons: they move while held down (onPress from compat.js works with
// mouse, touch or pointer events, depending on what the browser has).
document.querySelectorAll("[data-jog]").forEach(b => {
  const [a, d] = b.dataset.jog.split(":");
  const up = () => { if (b.classList.contains("act")) { b.classList.remove("act"); jogEnd(); } };
  onPress(b, e => { b.classList.add("act"); jogStart(a, +d, e.shiftKey); }, up);
});

// --- MDI ---------------------------------------------------------------------------------------
// Position in the MDI history while browsing it with ↑/↓ (−1 = new, empty line).
let mdiIdx = -1;
// MDI input: Enter sends the line (cleared if it succeeded), ↑/↓ browse the history, Esc clears.
// stopPropagation: keys typed here are not shortcuts (the space bar does not pause).
$("mdi").addEventListener("keydown", async e => {
  if (e.altKey) return;                           // Alt-1…7, Alt-R, Alt-S keep working from the input
  const h = S.st ? S.st.mdi_historial : [];
  if (normKey(e) === "Enter") {
    const v = $("mdi").value.trim();
    if (v) { const r = await act("mdi", { linea: v }); if (r.ok) { $("mdi").value = ""; mdiIdx = -1; } }
  } else if (normKey(e) === "ArrowUp" && h.length) { mdiIdx = Math.min(h.length - 1, mdiIdx + 1); $("mdi").value = h[h.length - 1 - mdiIdx]; e.preventDefault(); }
  else if (normKey(e) === "ArrowDown") { mdiIdx = Math.max(-1, mdiIdx - 1); $("mdi").value = mdiIdx < 0 ? "" : h[h.length - 1 - mdiIdx]; e.preventDefault(); }
  else if (normKey(e) === "Escape") { $("mdi").value = ""; $("mdi").blur(); }
  e.stopPropagation();
});

// =================================================================================================
// ============================ DATA SCREENS: OFFSETS AND CONFIGURATION ============================
// =================================================================================================

// --- offsets ----------------------------------------------------------------------------------
// Work offsets (Mach3's Fixtures): G54 = P1 … G59 = P6.
const WCS = ["G54", "G55", "G56", "G57", "G58", "G59"];
// P number (1–6) of the active offset, for G10 L2/L20 P…
const wcsP = () => WCS.indexOf((S.st && S.st.estado.wcs) || "G54") + 1 || 1;
// Last offsets table received.
let offsetsData = null;
// Python 3.4 (Windows XP) does not keep dict order: sort here, do not rely on the JSON order
const AXIS_ORDER = "XYZABCUVW";
function axisKeys(o) { return Object.keys(o).sort((a, b) => AXIS_ORDER.indexOf(a) - AXIS_ORDER.indexOf(b)); }
// sortedEntries(obj, cmp): [key, value] pairs sorted by key (with cmp if given).
function sortedEntries(o, cmp) {
  return Object.keys(o).sort(cmp || ((a, b) => (a < b ? -1 : a > b ? 1 : 0))).map(k => [k, o[k]]);
}

// loadOffsets(): Offsets screen (Alt-5): G54–G59 offsets table, editor of the active offset,
// buttons to change offset and the tool table (length H and diameter).
async function loadOffsets() {
  const r = await api("/api/origenes");
  const axes = S.axes.length ? S.axes : ["X", "Y", "Z"];
  if (!r.ok) { $("offsets-table").innerHTML = `<tr><td>${esc(r.mensaje)}</td></tr>`; return; }
  offsetsData = r.origenes;
  const cur = (S.st && S.st.estado.wcs) || "G54";
  $("offsets-table").innerHTML = `<tr><th>Origen</th>${axes.map(a => `<th>${a}</th>`).join("")}</tr>` +
    sortedEntries(r.origenes).map(([w, v]) => `<tr><td>${w === cur ? "▶ " : ""}${w}</td>${axes.map(a => `<td>${(v[a] || 0).toFixed(4)}</td>`).join("")}</tr>`).join("");
  $("wcs-edit").innerHTML = `<tr><th>Eje</th><th>Valor (máquina)</th></tr>` +
    axes.map(a => `<tr><td>${a}</td><td><input data-off="${a}" value="${((r.origenes[cur] || {})[a] || 0).toFixed(4)}"></td></tr>`).join("");
  $("fixtures").innerHTML = WCS.map((w, i) => `<button class="mbtn ${w === cur ? "on" : ""}" data-mdi="${w}">Origen ${i + 1} (${w})</button>`).join("");
  const cfg = S.cfg || (S.cfg = await api("/api/config"));
  const tools = sortedEntries(cfg.herramientas, (a, b) => a - b);
  $("tools-table").innerHTML = `<tr><th>Herramienta</th><th>Largo (H)</th><th>Diámetro</th><th>Nombre</th></tr>` +
    (tools.length ? tools.map(([n, t]) => `<tr><td>T${n}</td><td>${t.largo.toFixed(3)}</td><td>${t.diametro.toFixed(3)}</td><td>${esc(t.nombre || "")}</td></tr>`).join("")
                  : `<tr><td class="muted">sin tabla de herramientas</td></tr>`);
}
// applyOffset(): saves the edited values to the active offset with G10 L2 (machine values).
// Accepts a decimal comma («12,5»), as written in Spanish.
function applyOffset() {
  const vals = [...document.querySelectorAll("[data-off]")].map(i => `${i.dataset.off}${parseFloat(i.value.replace(",", ".")) || 0}`).join(" ");
  act("mdi", { linea: `G10 L2 P${wcsP()} ${vals}` }).then(r => { if (r.ok) { toast("Origen actualizado", "info"); loadOffsets(); } });
}

// --- configuration ------------------------------------------------------------------------------
// loadConfig(): Settings screen (Alt-6): axes (steps/mm, speed, acceleration, limits) and machine
// data; with plasma, also heights and delays. Read only: it is changed from Profiles.
async function loadConfig() {
  const c = S.cfg || (S.cfg = await api("/api/config"));
  setB("cfg-file", c.archivo);
  const n = v => v == null ? "—" : (+v).toFixed(3);
  $("axes-table").innerHTML = `<tr><th>Eje</th><th>Pasos/mm</th><th>Vel. máx (mm/min)</th><th>Acel. (mm/s²)</th><th>Mín</th><th>Máx</th></tr>` +
    axisKeys(c.ejes).map(a => [a, c.ejes[a]]).map(([a, x]) => `<tr><td>${a}</td><td>${x.pasos_mm}</td><td>${x.vel_max}</td><td>${x.acel}</td><td>${n(x.min)}</td><td>${n(x.max)}</td></tr>`).join("");
  const pl = c.plasma;
  $("cfg-extra").innerHTML = `<table class="table">
    <tr><td>Nombre</td><td>${esc(c.nombre)}</td></tr><tr><td>Tipo</td><td>${esc(c.tipo)}</td></tr>
    <tr><td>Desviación de unión</td><td>${c.desviacion_union} mm</td></tr><tr><td>Tolerancia de arco</td><td>${c.tolerancia_arco} mm</td></tr>
    <tr><td>Husillo</td><td>${c.husillo.rpm_min}–${c.husillo.rpm_max} rpm</td></tr>
    <tr><td>Controlador</td><td>${esc(c.controlador.tipo || "sim")}${c.controlador.puerto ? " (" + esc(c.controlador.puerto) + ")" : ""}</td></tr>
    ${pl ? `<tr><td>Plasma: perforación / corte</td><td>${pl.altura_perforacion} / ${pl.altura_corte} mm</td></tr>
    <tr><td>Plasma: retardo / traslado</td><td>${pl.retardo_perforacion} s / ${pl.altura_traslado} mm</td></tr>
    <tr><td>Plasma: perforación auto / IHS</td><td>${pl.perforacion_auto ? "sí" : "no"} / ${pl.sensado ? "sí" : "no"}</td></tr>` : ""}
  </table>`;
}
// applyMachine(cfg): adapts the screen to the machine type: with plasma it shows the torch and hides
// spindle and coolant; it also puts the brand in the title and the plasma data.
// The probing (IHS) button is disabled if the machine has no sensor.
function applyMachine(cfg) {
  const plasma = cfg.tipo === "plasma" && cfg.plasma;
  document.querySelectorAll(".torch-part").forEach(e => e.hidden = !plasma);
  document.querySelectorAll(".spindle-part").forEach(e => e.hidden = !!plasma);
  document.querySelectorAll(".coolant-only").forEach(e => e.hidden = !!plasma);
  document.title = cfg.marca || "TISMA CNC";
  if (plasma) {
    const p = cfg.plasma, mm = v => v.toFixed(1) + " mm";
    setB("pl-pierce", mm(p.altura_perforacion)); setB("pl-cut", mm(p.altura_corte));
    setB("pl-delay", p.retardo_perforacion.toFixed(2) + " s"); setB("pl-travel", mm(p.altura_traslado));
    setB("pl-auto", p.perforacion_auto ? "sí" : "no"); setB("pl-ihs", p.sensado ? "sí" : "no");
    document.querySelectorAll('[data-act="sensar"]').forEach(b => b.disabled = !p.sensado);
  }
}

// =================================================================================================
// ============================================ DIALOGS ============================================
// =================================================================================================

// --- dialogs ------------------------------------------------------------------------------------
// pendingId: guardian request currently shown (so it is not opened twice).
// modalResolve: function that resolves the open dialog's promise.
let pendingId = null, modalResolve = null;
// openModal(title, text, report, okText, opts): opens the dialog and returns a promise that gives
// true (Aceptar) or false (Cancelar / Esc). opts: html (extra content, already escaped), guardian
// (shows the guardian's note), noCancel (a single button), focus (id of the field to focus).
// There is a single dialog on the page: opening another replaces the previous content.
function openModal(title, text, report, okLabel, opts = {}) {
  $("modal-title").textContent = title;
  $("modal-text").textContent = text || "";
  $("modal-report").textContent = report || "";
  $("modal-report").hidden = !report;
  $("modal-body").innerHTML = opts.html || "";
  $("modal-note").hidden = !opts.guardian;
  $("modal-ok").textContent = okLabel || "Aceptar";
  $("modal-cancel").hidden = !!opts.noCancel;
  $("modal").hidden = false;
  (opts.focus ? $(opts.focus) : $("modal-ok")).focus();
  return new Promise(res => { modalResolve = res; });
}
// closeModal(ok): closes the dialog and hands the answer to whoever opened it.
function closeModal(ok) { $("modal").hidden = true; if (modalResolve) { const r = modalResolve; modalResolve = null; r(ok); } }
$("modal-ok").onclick = () => closeModal(true);
$("modal-cancel").onclick = () => closeModal(false);
// showApproval(p): asks the operator to approve what the guardian held back: enabling the machine
// (RESET, type "manual") or running a program (with its check report). Answers the server with
// «aprobar» or «cancelar». p: {id, tipo, resumen, informe} sent by the server.
async function showApproval(p) {
  if (pendingId === p.id) return;
  pendingId = p.id;
  const ok = await openModal(p.tipo === "manual" ? "Habilitar la máquina" : "Aprobar el programa", p.resumen,
    p.informe ? (S.st && S.st.programa ? S.st.programa.informe : "") : "Jog, MDI, referenciado y ceros quedan habilitados hasta pulsar RESET.",
    "Aprobar y seguir", { guardian: true });
  await act(ok ? "aprobar" : "cancelar", ok ? { id: p.id } : {});
  pendingId = null;
}
// openWizards(): «Asistentes» (Mach3's Wizards): pick a wizard, fill its few numbers and the server
// generates the program with the tool and speeds of the selected profiles, then loads it like a
// file (checked, simulated and approved the same way). The form comes from /api/asistentes.
async function openWizards() {
  const w = await api("/api/asistentes");
  const list = w.asistentes || [];
  if (!list.length) return toast("no hay asistentes");
  const opts = list.map(a => `<option value="${a.tipo}">${esc(a.titulo)}</option>`).join("");
  const fields = a => a.campos.map(c => `<label class="wiz-row"><span>${esc(c.etiqueta)}</span>` +
    `<input data-wk="${c.clave}" inputmode="decimal" value="${c.valor}"><span class="muted">${esc(c.unidad)}</span></label>`).join("");
  const html = `<select id="wiz-kind">${opts}</select><div id="wiz-fields" class="wiz-form">${fields(list[0])}</div>` +
    `<p class="muted small">${esc(w.herramienta || "")}</p>`;
  const done = openModal("Asistentes", "Elegí qué hacer y completá las medidas (mm, cero de pieza G54, Z0 arriba del material).",
    "", "Generar y cargar", { html: html, focus: "wiz-kind" });
  $("wiz-kind").onchange = () => {
    const a = list.filter(x => x.tipo === $("wiz-kind").value)[0];
    $("wiz-fields").innerHTML = fields(a);
  };
  if (!(await done)) return;
  const datos = {};
  document.querySelectorAll("[data-wk]").forEach(i => { datos[i.dataset.wk] = parseFloat(i.value.replace(",", ".")); });
  const kind = $("wiz-kind").value;
  const r = await act("asistente", { tipo: kind, datos: datos });
  if (r.ok) afterLoad(r, "asistente_" + kind + ".nc");
}
// --- probing («Palpado») ---------------------------------------------------------------------------
// probeField(c, saved): one form row of the probing dialog: a number box, or a list when the field
// has «opciones». saved = the value the operator used last time (if any), else the default.
function probeField(c, saved) {
  const v = saved !== undefined ? saved : c.valor;
  const input = c.opciones
    ? `<select data-pk="${c.clave}">` + c.opciones.map(o => `<option value="${esc(o[0])}"${String(o[0]) === String(v) ? " selected" : ""}>${esc(o[1])}</option>`).join("") + "</select>"
    : `<input data-pk="${c.clave}" data-num="1" inputmode="decimal" value="${esc(String(v))}">`;
  // A list has no unit and takes the unit's column too (its options are long sentences).
  return `<label class="wiz-row"><span>${esc(c.etiqueta)}</span>${input}` + (c.opciones ? "" : `<span class="muted">${esc(c.unidad)}</span>`) + "</label>";
}
// Last values per routine, remembered in this browser only (a convenience: e.g. where the tool
// setter is). Wrapped in try: private windows or blocked storage throw, and then nothing is kept.
function probeSaved() { try { return JSON.parse(localStorage.getItem("tisma-palpado") || "{}"); } catch (e) { return {}; } }
function probeSave(all) { try { localStorage.setItem("tisma-palpado", JSON.stringify(all)); } catch (e) { /* not kept */ } }
// openProbing(): «Palpado» dialog: pick a routine (edge, corner, bore / boss centre, surface, tool
// length, height map), fill its numbers and the probe's, and run it. The server moves, touches and
// sets the zero in a background thread; the result comes back in the state («palpado») and is
// shown as messages by watchProbe(). With a height map measured, it also offers to apply it to the
// loaded program. The forms come from /api/palpado.
async function openProbing() {
  const w = await api("/api/palpado");
  const list = w.rutinas || [];
  if (!list.length) return toast("no hay rutinas de palpado");
  const saved = probeSaved();
  const last = saved._rutina && list.some(r => r.tipo === saved._rutina) ? saved._rutina : list[0].tipo;
  const opts = list.map(r => `<option value="${r.tipo}"${r.tipo === last ? " selected" : ""}>${esc(r.titulo)}</option>`).join("");
  const fieldsOf = tipo => {
    const r = list.filter(x => x.tipo === tipo)[0], sv = saved[tipo] || {};
    return r.campos.map(c => probeField(c, sv[c.clave])).join("");
  };
  const probeSv = saved._sonda || {};
  const mapa = w.mapa
    ? `<p class="muted small">Mapa medido: ${w.mapa.z[0].length}×${w.mapa.z.length} puntos, Z de ${w.mapa.min.toFixed(3)} a ${w.mapa.max.toFixed(3)} mm.</p>` +
      '<button class="mbtn sm" id="pk-apply" type="button">Aplicar el mapa al programa cargado</button>'
    : "";
  const html = `<select id="pk-kind">${opts}</select><div id="pk-fields" class="wiz-form">${fieldsOf(last)}</div>` +
    '<p class="muted small">Sonda</p><div class="wiz-form">' + w.sonda.map(c => probeField(c, probeSv[c.clave])).join("") + "</div>" + mapa;
  const done = openModal("Palpado", "Acercá la sonda a la pieza con el jog (por debajo del borde, o dentro del agujero, o arriba del saliente) y elegí la rutina. Conectá la sonda antes de empezar.",
    "", "Palpar", { html: html, focus: "pk-kind" });
  $("pk-kind").onchange = () => { $("pk-fields").innerHTML = fieldsOf($("pk-kind").value); };
  if ($("pk-apply")) $("pk-apply").onclick = async () => {
    closeModal(false);
    const r = await act("aplicar_mapa", { tramo: 5 });
    if (r.ok) afterLoad(r, (S.st && S.st.programa ? S.st.programa.nombre : "programa") + " (con mapa)");
  };
  if (!(await done)) return;
  const kind = $("pk-kind").value, datos = {}, sonda = {}, mine = {};
  document.querySelectorAll("[data-pk]").forEach(i => {
    const v = i.dataset.num ? parseFloat(i.value.replace(",", ".")) : i.value;
    datos[i.dataset.pk] = v;
    (w.sonda.some(c => c.clave === i.dataset.pk) ? sonda : mine)[i.dataset.pk] = v;
  });
  saved[kind] = mine; saved._sonda = sonda; saved._rutina = kind;
  probeSave(saved);
  const r = await act("palpado", { tipo: kind, datos: datos });
  if (r.ok) toast("palpando… (Stop lo cancela)", "info");
}
// watchProbe(p): called on every state; when a probing routine ends, shows its result once.
// p = state «palpado»: {tipo, estado: corriendo|listo|error, mensajes}.
function watchProbe(p) {
  const key = p ? p.tipo + ":" + p.estado + ":" + p.mensajes.join("|") : "";
  if (key === S.probeKey) return;
  const first = S.probeKey === undefined;
  S.probeKey = key;
  if (first || !p || p.estado === "corriendo") return;      // not on page load, not while running
  p.mensajes.forEach(m => toast(m, p.estado === "listo" ? "info" : ""));
}
// editProgram(): Edit G-Code: edits the loaded program in a text box and loads it again
// (the server checks it again, like a new file).
async function editProgram() {
  if (!S.lines.length) return toast("no hay programa cargado");
  let name = (S.st && S.st.programa && S.st.programa.nombre) || "programa.nc";
  // A program generated from a DXF is edited as G-code: save it under a .nc name, or the server
  // would read the edited G-code as a drawing.
  if (/\.dxf$/i.test(name)) name += ".nc";
  const ok = await openModal("Editar G-code", name, "", "Guardar y revisar",
    { html: `<textarea id="editor" spellcheck="false">${esc(S.lines.join("\n"))}</textarea>`, focus: "editor" });
  if (!ok) return;
  const r = await act("cargar", { nombre: name, texto: $("editor").value });
  if (r.ok) afterLoad(r, name);
}
// showRecent(): Recent File: list of the latest programs; choosing one loads it by its sha256.
async function showRecent() {
  const r = await api("/api/recientes");
  if (!r.recientes.length) return toast("todavía no hay archivos recientes", "info");
  const html = '<ul class="list">' + r.recientes.map(x => `<li data-sha="${x.sha}">${esc(x.nombre)} <span class="muted">· ${x.lineas} líneas · ${new Date(x.fecha * 1000).toLocaleString()}</span></li>`).join("") + "</ul>";
  const p = openModal("Archivos recientes", "Elegí uno para cargarlo.", "", "Cerrar", { html, noCancel: true });
  document.querySelectorAll("#modal-body li[data-sha]").forEach(li => li.onclick = async () => {
    closeModal(false);
    const res = await act("abrir_reciente", { sha: li.dataset.sha });
    if (res.ok) afterLoad(res, li.textContent.split(" · ")[0].trim());
  });
  await p;
}
// Quick help for the G and M codes TISMA CNC understands («G-codes» and «M-codes» buttons, as in Mach3).
const HELP = {
  G: [["G0", "rápido"], ["G1", "avance lineal"], ["G2 / G3", "arco horario / antihorario (I J K o R, P vueltas)"], ["G4 P", "pausa en segundos"],
      ["G10 L2 / L20", "fijar origen de trabajo"], ["G17 / G18 / G19", "plano XY / ZX / YZ"], ["G20 / G21", "pulgadas / mm"],
      ["G28 / G30", "ir a posición de inicio"], ["G31 / G38.2 / G38.3", "sensado (G31 de Mach3)"], ["G43 Hn / G49", "largo de herramienta"],
      ["G53", "coordenadas de máquina"], ["G54–G59", "orígenes de trabajo"], ["G61 / G64", "parada exacta / velocidad constante"],
      ["G73 / G81 / G82 / G83", "ciclos de taladrado (G80 cancela)"], ["G90 / G91", "absoluto / incremental"], ["G90.1 / G91.1", "centros de arco absolutos / incrementales"],
      ["G92 / G92.1", "desplazamiento de origen"], ["G93 / G94", "avance en tiempo inverso (1/min, 4 y 5 ejes) / normal (mm/min)"],
      ["G68.2 / G53.1 / G69", "plano inclinado «3+2»: definirlo (X Y Z = origen, I J K = ángulos), girar la mesa hacia la herramienta, cancelar"],
      ["G98 / G99", "retracción al inicial / a R"]],
  M: [["M0 / M1", "pausa / pausa opcional"], ["M2 / M30", "fin de programa"], ["M3 / M4 / M5", "husillo (o antorcha en plasma) horario / antihorario / apagado"],
      ["M6 Tn", "cambio de herramienta (manual)"], ["M7 / M8 / M9", "niebla / refrigerante / apagado"], ["M100+", "macros de usuario de Mach3: se ignoran con aviso"]],
};
// helpCodes("G" or "M"): shows the help table and what is not supported.
function helpCodes(kind) {
  const html = '<table class="table">' + HELP[kind].map(([c, d]) => `<tr><td>${c}</td><td style="text-align:left">${d}</td></tr>`).join("") + "</table>" +
    '<p class="muted small">No soportados (dan error claro): parámetros #, subprogramas M98/M99, G41/G42, G93, G12/G13, G15/G16, G50/G51, G52.</p>';
  openModal(kind === "G" ? "G-codes soportados" : "M-codes soportados", "", "", "Cerrar", { html, noCancel: true });
}

// =================================================================================================
// ===================================== SLICER-STYLE PROFILES =====================================
// =================================================================================================

// As in PrusaSlicer/OrcaSlicer: the effective machine comes from four profiles (machine, material,
// consumable and process). They are chosen at the top, in the bar (#presets); on the Profiles
// screen (Alt-3) they are edited. System ones are never overwritten: «Guardar como…» (Save as)
// creates an own one that «inherits» from it and stores only what changed (own values).

// --- slicer-style profiles (TISMA Engine / PrusaSlicer / OrcaSlicer) ----------------------------------------
// Profile types, in the order of the selectors.
const PKINDS = ["maquina", "material", "consumible", "proceso"];
S.pk = "maquina"; S.prof = null; S.presets = null;
// loadPresets(): fills the profile selectors (system / mine), the thickness (plasma) and the
// summary of the chosen combination, with its warnings.
async function loadPresets() {
  const r = await api("/api/perfiles");
  S.presets = r;
  $("presets").hidden = !r.activos;
  if (!r.activos) return;
  const plasma = r.resumen && r.resumen.tipo === "plasma";
  for (const k of PKINDS) {
    const sel = document.querySelector(`#presets select[data-kind="${k}"]`);
    const list = r.tipos[k] || [];
    const group = (orig, label) => {
      const items = list.filter(p => p.origen === orig);
      return items.length ? `<optgroup label="${label}">` + items.map(p => `<option value="${esc(p.nombre)}">${esc(p.nombre)}</option>`).join("") + "</optgroup>" : "";
    };
    sel.innerHTML = group("sistema", "Del sistema") + group("usuario", "Míos");
    sel.value = r.seleccion[k] || "";
  }
  document.querySelectorAll(".plasma-only").forEach(e => e.hidden = !plasma);
  $("espesor").innerHTML = (r.espesores || []).map(e => `<option value="${e}">${e} mm</option>`).join("");
  if (r.seleccion.espesor != null) $("espesor").value = String(r.seleccion.espesor);
  setB("cons-label", plasma ? "Antorcha" : "Herramienta");
  setB("perfil-resumen", r.resumen ? r.resumen.texto : "—");
  setB("perfil-avisos", r.resumen && r.resumen.avisos.length ? "⚠ " + r.resumen.avisos.join(" · ") : "");
}
// chooseProfile(type, name, thickness): chooses a profile (or a sheet thickness) and rebuilds all
// that depends on the machine: configuration, DRO, selectors, program (checked again with the new
// machine). Without arguments, re-applies the current selection (after saving a profile in use).
async function chooseProfile(kind, name, espesor) {
  const params = espesor != null ? { espesor } : kind ? { tipo: kind, nombre: name } : {};
  const r = await act("perfil_elegir", params);
  if (!r.ok) { loadPresets(); return; }
  S.selKey = selectionKey(r.seleccion);     // we already applied it here: poll() must not repeat it
  await applySelection();
  toast("Perfil aplicado: " + (r.resumen ? r.resumen.texto : ""), "info");
}
// applySelection(): rebuilds everything that depends on the machine and the chosen profiles
// (configuration, DRO, bar selectors, program checked again, list on the Profiles screen).
async function applySelection() {
  S.cfg = await api("/api/config");
  applyMachine(S.cfg);
  buildDro(axisKeys(S.cfg.ejes));
  await loadPresets();
  await loadProgram();
  if (S.tab === "profiles") renderProfList();
}
// selectionKey(sel): text identifying a profile selection, with sorted keys (the order of a
// JSON built by Python 3.4 is not fixed). Used to notice a change made in another window.
function selectionKey(sel) {
  return sel ? Object.keys(sel).sort().map(k => k + "=" + sel[k]).join("|") : "";
}
document.querySelectorAll("#presets select[data-kind]").forEach(sel => sel.addEventListener("change", () => chooseProfile(sel.dataset.kind, sel.value)));
$("espesor").addEventListener("change", () => chooseProfile(null, null, parseFloat($("espesor").value)));

// editor
document.querySelectorAll("#prof-kinds button").forEach(b => b.onclick = () => { S.pk = b.dataset.pk; renderProfList(); openFirst(); });
// openProfileTab(): on entering the Profiles screen.
async function openProfileTab() { await loadPresets(); renderProfList(); if (!S.prof || S.prof.tipo !== S.pk) openFirst(); }
// openFirst(): opens the in-use profile of the chosen type (or the first in the list).
function openFirst() {
  const cur = S.presets && S.presets.seleccion[S.pk];
  const list = (S.presets && S.presets.tipos[S.pk]) || [];
  const name = cur || (list[0] && list[0].nombre);
  if (name) openProfile(S.pk, name);
}
// renderProfList(): list of profiles of the chosen type, split into «Del sistema» and «Míos».
function renderProfList() {
  document.querySelectorAll("#prof-kinds button").forEach(b => b.classList.toggle("on", b.dataset.pk === S.pk));
  if (!S.presets || !S.presets.activos) { $("prof-items").innerHTML = '<p class="muted">Perfiles no activos.</p>'; return; }
  const list = S.presets.tipos[S.pk] || [];
  const inuse = S.presets.seleccion[S.pk];
  const item = p => `<div class="prof-item ${S.prof && S.prof.nombre === p.nombre && S.prof.tipo === S.pk ? "on" : ""} ${p.nombre === inuse ? "inuse" : ""}" data-pn="${esc(p.nombre)}">${esc(p.nombre)}${p.hereda ? `<small>hereda de ${esc(p.hereda)}</small>` : ""}</div>`;
  const sys = list.filter(p => p.origen === "sistema"), usr = list.filter(p => p.origen === "usuario");
  $("prof-items").innerHTML = (sys.length ? "<h5>Del sistema</h5>" + sys.map(item).join("") : "") +
    "<h5>Míos</h5>" + (usr.length ? usr.map(item).join("") : '<p class="muted small">Todavía no hay: abrí uno del sistema, cambiá lo que quieras y «Guardar como…».</p>');
  $("prof-items").querySelectorAll("[data-pn]").forEach(e => e.onclick = () => openProfile(S.pk, e.dataset.pn));
}
// openProfile(type, name): fetches a profile (effective data, parent's, raw and the field
// schema) and shows it in the form.
async function openProfile(kind, name) {
  const r = await fetch(`/api/perfil?tipo=${encodeURIComponent(kind)}&nombre=${encodeURIComponent(name)}`, { cache: "no-store" }).then(x => x.json());
  if (r.ok === false) return toast(r.mensaje);
  S.prof = { tipo: kind, nombre: name, origen: r.origen, hereda: r.hereda, datos: r.datos, padre: r.padre, crudo: r.crudo,
             esquema: r.esquema, propios: new Set(r.propios), sucio: false };
  renderProfList();
  renderProfForm();
}
// getP(obj, "a.b.c") / setP(obj, "a.b.c", v): read and write by dotted key (the schema keys
// look like that). setP creates any missing intermediate objects.
const getP = (o, k) => k.split(".").reduce((a, p) => (a && typeof a === "object" ? a[p] : undefined), o);
function setP(o, k, v) { const ps = k.split("."); let c = o; for (const p of ps.slice(0, -1)) { if (typeof c[p] !== "object" || c[p] === null) c[p] = {}; c = c[p]; } c[ps[ps.length - 1]] = v; }
// fieldApplies(field, data): does the field apply to this machine type (e.g. plasma fields are not
// shown on a mill)? For material/consumable/process the profile's «tipos» list is checked.
function fieldApplies(f, d) {
  if (!f.tipos) return true;
  const kinds = S.prof.tipo === "maquina" ? [d.tipo] : (d.tipos || []);
  return f.tipos.some(t => kinds.indexOf(t) >= 0);
}
// isOwn(key): is the value this profile's own (or something inside that key)? If not, it is
// inherited from the parent and shown dimmed.
function isOwn(key) {
  if (!S.prof.hereda) return true;
  for (const k of S.prof.propios) if (k === key || k.indexOf(key + ".") === 0) return true;
  return false;
}
// renderProfForm(): builds the form of the open profile, grouped as the schema says. System
// ones cannot be saved or deleted (only «Guardar como…»).
function renderProfForm() {
  const P = S.prof, d = P.datos;
  $("prof-name").textContent = P.nombre + (P.sucio ? " (modificado)" : "");
  $("prof-origin").textContent = P.origen === "sistema" ? "del sistema" : "mío";
  $("prof-origin").className = "badge" + (P.origen === "usuario" ? " user" : "");
  $("prof-parent").textContent = P.hereda ? "· hereda de " + P.hereda : "";
  $("prof-note").hidden = !d.nota; $("prof-note").textContent = d.nota || "";
  document.querySelector('[data-act="perfil_guardar"]').disabled = P.origen === "sistema";
  document.querySelector('[data-act="perfil_borrar"]').disabled = P.origen === "sistema";
  const groups = {};
  for (const f of P.esquema) if (fieldApplies(f, d)) (groups[f.grupo] = groups[f.grupo] || []).push(f);
  let html = "";
  for (const g of Object.keys(groups)) {
    html += `<div class="prof-group"><h4>${esc(g)}</h4>`;
    for (const f of groups[g]) html += fieldHtml(f, getP(d, f.clave));
    html += "</div>";
  }
  $("prof-form").innerHTML = html;
}
// fieldHtml(field, value): HTML of a field by its type (bool, opcion, multiple, tabla, num, texto),
// with unit, help and the ↺ button to go back to the inherited value.
function fieldHtml(f, v) {
  const own = isOwn(f.clave), cls = "prof-field " + (P_hasParent() ? (own ? "own" : "inh") : "");
  const rev = P_hasParent() && own ? `<button class="revert" data-rev="${f.clave}" title="Volver al valor de «${esc(S.prof.hereda)}»">↺</button>` : "<span></span>";
  const help = f.ayuda ? `<span class="help">${esc(f.ayuda)}</span>` : "<span></span>";
  let input;
  if (f.tipo === "bool") input = `<input type="checkbox" data-k="${f.clave}" data-t="bool" ${v ? "checked" : ""}>`;
  else if (f.tipo === "opcion") input = `<select data-k="${f.clave}" data-t="texto">${f.opciones.map(o => `<option ${o === v ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>`;
  else if (f.tipo === "multiple") input = `<span class="checks">${f.opciones.map(o => `<label><input type="checkbox" data-k="${f.clave}" data-t="multi" value="${o}" ${(v || []).indexOf(o) >= 0 ? "checked" : ""}>${o}</label>`).join("")}</span>`;
  else if (f.tipo === "tabla") return tableHtml(f, v || [], own, rev);
  else input = `<input type="text" data-k="${f.clave}" data-t="${f.tipo === "num" ? "num" : "texto"}" value="${v == null ? "" : esc(v)}" ${f.tipo === "num" ? 'inputmode="decimal"' : ""}>`;
  return `<div class="${cls}"><label>${esc(f.etiqueta)}</label>${input}<span class="unit">${esc(f.unidad || "")}</span>${rev}${help}</div>`;
}
// Does the open profile inherit from another?
const P_hasParent = () => !!(S.prof && S.prof.hereda);
// tableHtml(field, rows, own, revertButton): table-type field (e.g. cuts by thickness), with
// editable rows, an ✕ button to delete one and «+ fila» (copies the last one).
function tableHtml(f, rows, own, rev) {
  const cols = f.columnas;
  const head = "<tr>" + cols.map(c => `<th>${esc(c.etiqueta)}${c.unidad ? ` <span class="muted">(${esc(c.unidad)})</span>` : ""}</th>`).join("") + "<th></th></tr>";
  const body = rows.map((r, i) => "<tr>" + cols.map(c => `<td><input data-tk="${f.clave}" data-row="${i}" data-col="${c.clave}" data-tt="${c.texto ? "texto" : "num"}" value="${r[c.clave] == null ? "" : esc(r[c.clave])}"></td>`).join("") +
    `<td><button class="del" data-tdel="${f.clave}" data-row="${i}" title="Borrar fila">✕</button></td></tr>`).join("");
  return `<div class="prof-field ${P_hasParent() ? (own ? "own" : "inh") : ""}"><label>${esc(f.etiqueta)}</label><span></span><span></span>${rev}<span class="help">${esc(f.ayuda || "")}</span>
    <div class="ptable-wrap"><table class="ptable ${P_hasParent() && !own ? "inh" : ""}">${head}${body}</table>
    <button class="mbtn sm" data-tadd="${f.clave}">+ fila</button></div></div>`;
}
// num(v): number with decimal comma or point; null if it is not a number.
const num = v => { const x = parseFloat(String(v).replace(",", ".")); return isFinite(x) ? x : null; };
// markOwn(key): the value becomes own and the profile is marked «(modificado)».
function markOwn(key) { S.prof.propios.add(key); S.prof.sucio = true; $("prof-name").textContent = S.prof.nombre + " (modificado)"; }
// A field changed: it is stored in S.prof.datos (not on the server yet: that is «Guardar»).
// If the machine type changed, the form is rebuilt because the applicable fields change.
$("prof-form").addEventListener("change", e => {
  const t = e.target, P = S.prof;
  if (!P) return;
  if (t.dataset.k) {
    let v;
    if (t.dataset.t === "bool") v = t.checked;
    else if (t.dataset.t === "multi") v = [...$("prof-form").querySelectorAll(`input[data-k="${t.dataset.k}"]`)].filter(x => x.checked).map(x => x.value);
    else if (t.dataset.t === "num") { v = t.value.trim() === "" ? null : num(t.value); if (t.value.trim() !== "" && v === null) { toast("tiene que ser un número"); return; } }
    else v = t.value;
    setP(P.datos, t.dataset.k, v);
    markOwn(t.dataset.k);
    if (t.dataset.k === "tipo" || t.dataset.k === "tipos") renderProfForm(); else t.closest(".prof-field").className = "prof-field " + (P.hereda ? "own" : "");
  } else if (t.dataset.tk) {
    const rows = getP(P.datos, t.dataset.tk) || [];
    const row = rows[+t.dataset.row];
    row[t.dataset.col] = t.dataset.tt === "texto" ? t.value : num(t.value);
    setP(P.datos, t.dataset.tk, rows);
    markOwn(t.dataset.tk);
  }
});
// Buttons inside the form: ↺ (back to the parent's value), + fila and ✕ in the tables.
// JSON.parse(JSON.stringify(x)): deep copy, compatible with old browsers.
$("prof-form").addEventListener("click", e => {
  const t = e.target, P = S.prof;
  if (!P) return;
  if (t.dataset.rev) {
    const pv = getP(P.padre || {}, t.dataset.rev);
    setP(P.datos, t.dataset.rev, pv === undefined ? null : JSON.parse(JSON.stringify(pv)));
    for (const k of [...P.propios]) if (k === t.dataset.rev || k.indexOf(t.dataset.rev + ".") === 0) P.propios.delete(k);
    P.sucio = true; renderProfForm();
  } else if (t.dataset.tadd) {
    const rows = (getP(P.datos, t.dataset.tadd) || []).slice();
    rows.push(rows.length ? JSON.parse(JSON.stringify(rows[rows.length - 1])) : {});
    setP(P.datos, t.dataset.tadd, rows); markOwn(t.dataset.tadd); renderProfForm();
  } else if (t.dataset.tdel) {
    const rows = (getP(P.datos, t.dataset.tdel) || []).slice();
    rows.splice(+t.dataset.row, 1);
    setP(P.datos, t.dataset.tdel, rows); markOwn(t.dataset.tdel); renderProfForm();
  }
});
// saveProfile(asNew): saves the profile. A system one is always saved as a new one that inherits from
// it. If the saved profile is in use, it is re-applied so the machine picks up the changes.
async function saveProfile(asNew) {
  const P = S.prof;
  if (!P) return;
  let name = P.nombre, inherits = P.hereda;
  if (asNew || P.origen === "sistema") {
    const sugg = P.origen === "sistema" ? P.nombre + " (mío)" : P.nombre + " (copia)";
    const ok = await openModal("Guardar como…", "Nombre del perfil nuevo:", "", "Guardar",
      { html: `<input id="new-name" class="wide-in" value="${esc(sugg)}">`, focus: "new-name" });
    if (!ok) return;
    name = $("new-name").value.trim();
    inherits = P.origen === "sistema" ? P.nombre : (P.hereda || null);
  }
  const datos = JSON.parse(JSON.stringify(P.datos));
  delete datos.nombre;
  const r = await act("perfil_guardar", { tipo: P.tipo, nombre: name, datos, hereda: inherits });
  if (r.ok) { toast(`Perfil «${name}» guardado`, "info"); await loadPresets(); await openProfile(P.tipo, name); if (S.presets.seleccion[P.tipo] === name || P.tipo === "maquina" && S.presets.seleccion.maquina === name) chooseProfile(); }
}
// exportProfile(): downloads the profile as .json (to move it to another PC; imported with «Importar»).
function exportProfile() {
  const P = S.prof;
  if (!P) return;
  const blob = new Blob([JSON.stringify(P.crudo, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = P.nombre.replace(/[^\w\-]+/g, "_") + ".json";
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}
// deleteProfile(): deletes an own profile, with confirmation.
async function deleteProfile() {
  const P = S.prof;
  if (!P || P.origen === "sistema") return;
  const ok = await openModal("Borrar perfil", `¿Borrar «${P.nombre}»?`, "", "Borrar");
  if (!ok) return;
  const r = await act("perfil_borrar", { tipo: P.tipo, nombre: P.nombre });
  if (r.ok) { S.prof = null; await loadPresets(); renderProfList(); openFirst(); }
}
// Import a .json profile of the chosen type.
$("prof-file").onchange = async e => {
  const f = e.target.files[0]; e.target.value = "";
  if (!f) return;
  const r = await act("perfil_importar", { tipo: S.pk, texto: await readText(f) });
  if (r.ok) { toast(`Importado: ${r.perfil.nombre}`, "info"); await loadPresets(); openProfile(S.pk, r.perfil.nombre); }
};

// =================================================================================================
// ===================================== KEYBOARD AND STARTUP ======================================
// =================================================================================================

// --- keyboard: Mach3 shortcuts ---------------------------------------------------------------------
// Mach3 jog keys: arrows = X/Y, PgUp/PgDn = Z. Value: [axis, direction].
const JOGKEYS = { ArrowLeft: ["X", -1], ArrowRight: ["X", 1], ArrowUp: ["Y", 1], ArrowDown: ["Y", -1], PageUp: ["Z", 1], PageDown: ["Z", -1] };
// Mach3 keyboard shortcuts (with focus outside a text field):
//   Alt-1…7 screens · Alt-R Cycle Start · Alt-S Stop · Esc Stop · Space Feed Hold
//   Alt-N single block (Single BLK) · Ctrl-W Rewind · Ctrl-O load · Ctrl-F coolant
//   F5 spindle · Ctrl-Alt-J jog on/off · Tab shows or hides the jog panel · arrows/PgUp/PgDn jog
// Screens (Alt-number) switch even while typing in a field, except with a dialog open.
// With a dialog open only Esc works (cancel).
document.addEventListener("keydown", e => {
  const tag = e.target.tagName;
  if (e.altKey && TAB_KEYS[normKey(e)] && $("modal").hidden) { e.target.blur && e.target.blur(); setTab(TAB_KEYS[normKey(e)]); e.preventDefault(); return; }
  if ((tag === "INPUT" && e.target.type !== "range") || tag === "TEXTAREA") {
    if (normKey(e) === "Escape" && tag === "INPUT") e.target.blur();
    return;
  }
  if (!$("modal").hidden) { if (normKey(e) === "Escape") closeModal(false); return; }
  const k = normKey(e).toLowerCase();
  if (e.altKey && TAB_KEYS[normKey(e)]) { setTab(TAB_KEYS[normKey(e)]); e.preventDefault(); return; }
  if (e.ctrlKey && e.altKey && k === "j") { ACTIONS.jog_onoff(); e.preventDefault(); return; }
  if (e.altKey && k === "r") { act("ciclo"); e.preventDefault(); return; }
  if (e.altKey && k === "s") { act("parar"); e.preventDefault(); return; }
  if (e.altKey && k === "n") { act("opcion", { nombre: "bloque_a_bloque" }); e.preventDefault(); return; }
  if (e.ctrlKey && k === "w") { ACTIONS.rebobinar(); e.preventDefault(); return; }
  if (e.ctrlKey && k === "o") { $("file").click(); e.preventDefault(); return; }
  if (e.ctrlKey && k === "f") { act("refrigerante"); e.preventDefault(); return; }
  if (normKey(e) === "F5") { ACTIONS.husillo(); e.preventDefault(); return; }
  if (normKey(e) === " ") { act("pausa"); e.preventDefault(); return; }
  if (normKey(e) === "Escape") { act("parar"); e.preventDefault(); return; }
  if (normKey(e) === "Enter" && S.tab === "mdi") { $("mdi").focus(); e.preventDefault(); return; }
  if (normKey(e) === "Tab") { $("jog").hidden = !$("jog").hidden; e.preventDefault(); return; }
  if (JOGKEYS[normKey(e)]) {
    e.preventDefault();
    if (e.repeat || S.jogKeys.has(normKey(e))) return;
    S.jogKeys.add(normKey(e));
    const [a, d] = JOGKEYS[normKey(e)];
    jogStart(a, d, e.shiftKey);
  }
});
// Releasing a jog key stops it; and if the window loses focus with a key held down (Alt-Tab),
// too: otherwise the axis would keep moving because the keyup never arrives.
document.addEventListener("keyup", e => { if (JOGKEYS[normKey(e)] && S.jogKeys.has(normKey(e))) { S.jogKeys.delete(normKey(e)); jogEnd(); } });
window.addEventListener("blur", () => { if (S.jogKeys.size) { S.jogKeys.clear(); jogEnd(); } });

// --- startup -----------------------------------------------------------------------------------------
// Light mode checkbox: changes the polling interval and redraws the views with the new mode.
$("lite").checked = !!window.TISMA_LITE;
$("lite").addEventListener("change", () => { setLite($("lite").checked); startPolling(); for (const v of Object.values(views)) { v.dirty = true; v.draw(); } });

// Startup: configuration → screen for the machine → DRO → profiles → controls → program →
// first status poll. In order, because each step uses what the previous one loaded.
(async () => {
  S.cfg = await api("/api/config");
  applyMachine(S.cfg);
  buildDro(axisKeys(S.cfg.ejes));
  await loadPresets();
  renderJog(); led("led-seguir", S.follow); led("led-jog", S.jogOn); setB("vista", VIEW_NAMES[VIEWS[S.viewIdx]]);
  await loadProgram();
  poll(true);
})();
