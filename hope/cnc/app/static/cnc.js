/* TISMA CNC (Hope CNC) — pantallas con la distribución de Mach3Mill. Sin dependencias: corre en
   navegadores viejos y nuevos. Los grupos que se repiten (DRO, herramienta, avance, husillo) salen de
   plantillas y se actualizan por atributo data-b, así cada pantalla tiene los suyos como en Mach3. */
"use strict";

const $ = id => document.getElementById(id);
const S = { st: null, cfg: null, lines: [], sha: null, errLines: new Set(), coord: "work", tab: "run",
            jogOn: true, jogStep: false, steps: [0.001, 0.01, 0.1, 1, 10], stepIdx: 2, jogPct: 30,
            jogKeys: new Set(), lastLine: -1, follow: false, axes: [], viewIdx: 1, programLoadedAt: 0 };
const VIEWS = ["top", "iso", "front", "side"];
const VIEW_NAMES = { top: "Arriba", iso: "ISO", front: "Frente", side: "Lado" };
const ROW = 21;

// --- comunicación ---------------------------------------------------------------
async function api(path) { const r = await fetch(path, { cache: "no-store" }); return r.json(); }
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
function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = "toast " + kind; t.textContent = msg;
  $("toasts").append(t);
  setTimeout(() => t.remove(), kind === "info" ? 2500 : 5000);
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// --- enlaces de datos (data-b) ----------------------------------------------------------
function setB(key, text) { document.querySelectorAll(`[data-b="${key}"]`).forEach(e => { if (e.textContent !== String(text)) e.textContent = text; }); }
function clsB(key, cls, on) { document.querySelectorAll(`[data-b="${key}"]`).forEach(e => e.classList.toggle(cls, !!on)); }
function led(key, on, blink = false) { document.querySelectorAll(`[data-b="${key}"]`).forEach(e => { e.classList.toggle("on", !!on); e.classList.toggle("blink", !!blink); }); }

// --- plantillas (los grupos de Mach3 se repiten en varias pantallas) -----------------------
document.querySelectorAll("[data-tpl]").forEach(box => {
  box.append($("tpl-" + box.dataset.tpl).content.cloneNode(true));
  if (box.dataset.tpl === "dro") box.classList.add("dro-panel");
});
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
  document.querySelectorAll(".ax-a").forEach(b => b.style.visibility = axes.includes("A") ? "" : "hidden");
}

// --- pantallas (Alt-1, 2, 4, 5, 6, 7 como Mach3) --------------------------------------------
const TAB_KEYS = { "1": "run", "2": "mdi", "4": "path", "5": "offsets", "6": "config", "7": "diag" };
function setTab(name) {
  S.tab = name;
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".screen").forEach(t => t.classList.toggle("on", t.id === "screen-" + name));
  if (name === "mdi") setTimeout(() => $("mdi").focus(), 0);
  if (name === "offsets") loadOffsets();
  if (name === "config") loadConfig();
  if (name === "run") renderCode();
  setTimeout(() => { const v = curView(); if (v) { v.dirty = true; v.fit(); } }, 0);
}
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => setTab(b.dataset.tab));

// --- trayectoria -------------------------------------------------------------------------------
const views = {};
document.querySelectorAll("[data-view]").forEach(el => { views[el.dataset.view] = new PathView(el); });
const curView = () => views[S.tab] || null;
let pathData = { puntos: [], limites: null, perforaciones: [] };
async function loadPath() {
  pathData = await api("/api/trayectoria");
  for (const v of Object.values(views)) { v.follow = S.follow; v.setData(pathData.puntos, pathData.limites, pathData.perforaciones); v.setView(VIEWS[S.viewIdx]); }
  if (pathData.recortado) toast("trayectoria recortada para dibujar (programa muy grande)", "info");
  renderLimits();
}
function renderLimits() {
  const P = pathData.puntos || [];
  if (!P.length) { $("limits-table").innerHTML = '<tr><td class="muted">sin programa</td></tr>'; return; }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of P) for (let i = 0; i < 3; i++) { if (p[i] < lo[i]) lo[i] = p[i]; if (p[i] > hi[i]) hi[i] = p[i]; }
  $("limits-table").innerHTML = "<tr><th>Eje</th><th>Mín</th><th>Máx</th></tr>" +
    ["X", "Y", "Z"].map((a, i) => `<tr><td>${a}</td><td>${lo[i].toFixed(3)}</td><td>${hi[i].toFixed(3)}</td></tr>`).join("");
}

// --- G-code (lista virtual: rinde con cientos de miles de líneas) ---------------------------------
function hl(line) {
  return line.split(/(\(.*?\)|;.*$)/).map(p => {
    if (!p) return "";
    if (p.startsWith("(") || p.startsWith(";")) return `<span class="c">${esc(p)}</span>`;
    return esc(p).replace(/\b([Gg]\d+(?:\.\d+)?)/g, '<span class="g">$1</span>').replace(/\b([MmTt]\d+)/g, '<span class="m">$1</span>');
  }).join("");
}
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
$("code").addEventListener("wheel", () => { S.followCode = false; clearTimeout(S.followT); S.followT = setTimeout(() => S.followCode = true, 4000); });
$("code").addEventListener("click", e => { const r = e.target.closest(".code-row"); if (r) $("line-in").value = r.dataset.line; });
S.followCode = true;
function followLine(n) {
  if (!n || !S.followCode) return;
  const box = $("code"), y = (n - 1) * ROW;
  if (y < box.scrollTop + ROW * 2 || y > box.scrollTop + box.clientHeight - ROW * 4) box.scrollTop = y - box.clientHeight / 3;
}
async function loadProgram() {
  const p = await api("/api/programa");
  S.lines = p.lineas || [];
  $("code").scrollTop = 0;
  renderCode();
  await loadPath();
}
async function loadFile(file) {
  if (!file) return;
  if (file.size > 32 * 1024 * 1024) return toast("archivo demasiado grande (máx. 32 MB)");
  const r = await act("cargar", { nombre: file.name, texto: await file.text() });
  if (r.ok) { afterLoad(r, file.name); }
}
async function afterLoad(r, name) {
  toast(r.informe.runnable ? `«${name}» revisado: listo` : `«${name}» tiene errores: mirá Diagnóstico (Alt-7)`, r.informe.runnable ? "info" : "");
  await loadProgram();
}
$("file").onchange = e => { loadFile(e.target.files[0]); e.target.value = ""; };
document.addEventListener("dragover", e => { e.preventDefault(); document.body.classList.add("drop"); });
document.addEventListener("dragleave", e => { if (!e.relatedTarget) document.body.classList.remove("drop"); });
document.addEventListener("drop", e => { e.preventDefault(); document.body.classList.remove("drop"); loadFile(e.dataTransfer.files[0]); });

// --- estado ----------------------------------------------------------------------------------
const fmt4 = v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(4);
const hms = s => { if (s == null) return "00:00:00"; s = Math.max(0, Math.round(s)); return [s / 3600, s % 3600 / 60, s % 60].map(x => String(Math.floor(x)).padStart(2, "0")).join(":"); };
const STATE_NAMES = { Idle: "En reposo", Run: "Ejecutando", Hold: "En pausa", Jog: "Jog", Home: "Referenciando", Alarm: "ALARMA",
                      Desconectado: "Desconectado", Reset: "RESET", Door: "Puerta" };
let polling = false;
async function poll(force) {
  if (polling && !force) return;
  polling = true;
  try { render(await api("/api/estado")); }
  catch (e) { setB("pill", "Sin conexión"); clsB("pill", "Desconectado", true); }
  finally { polling = false; }
}
function render(st) {
  S.st = st;
  const c = st.estado, job = c.trabajo, p = st.programa;
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
  // herramienta
  const t = st.herramienta || {};
  setB("tool-n", "T" + (t.n || 0)); setB("tool-num", t.n || 0);
  setB("tool-dia", (t.diametro || 0).toFixed(3)); setB("tool-len", (t.largo || 0).toFixed(3));
  led("led-cambio", t.cambio, t.cambio);
  setB("elapsed", hms(p ? p.transcurrido : null));
  setB("left", p && job.activo ? hms(p.tiempo_restante) : p ? "estimado " + hms(p.tiempo_total) : "—");
  led("led-jog", S.jogOn);
  // avance y husillo
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
  // RESET (como Mach3: LED rojo/verde que parpadea hasta habilitar)
  clsB("reset", "on", st.habilitada);
  document.querySelectorAll('[data-b="reset-led"]').forEach(e => { e.classList.toggle("on", st.habilitada); e.classList.toggle("blink", !st.habilitada); });
  document.body.classList.toggle("running", job.activo && c.estado === "Run");
  // programa
  setB("file", p ? p.nombre : "sin programa");
  if (p) {
    if (p.sha256 !== S.sha) { S.sha = p.sha256; S.errLines = new Set(p.errores.map(e => +((e.match(/línea (\d+)/) || [])[1] || 0))); }
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
  if (v) { const tool = { X: st.trabajo_pos.X, Y: st.trabajo_pos.Y, Z: st.trabajo_pos.Z }; v.setProgress(job.activo || job.terminado ? (p ? p.linea_actual : 0) : 0, tool); }
  // estado (línea inferior), perfil
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
setInterval(() => poll(), 150);

// --- acciones (delegación: los botones están repetidos en varias pantallas) --------------------------
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
  ayuda_g: () => helpCodes("G"), ayuda_m: () => helpCodes("M"),
  historial: () => setTab("diag"),
  borrar_estado: () => { S.clearedAt = Date.now() / 1000; setB("status", ""); },
};
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
function setNext(fromHere) {
  const n = parseInt($("line-in").value, 10) || 0;
  act("linea_siguiente", { linea: n, desde_aqui: fromHere });
}

// --- jog ---------------------------------------------------------------------------------------
function renderJog() {
  led("led-jog-cont", !S.jogStep); led("led-jog-step", S.jogStep);
  setB("jog-step", S.steps[S.stepIdx].toFixed(4)); setB("jog-pct", S.jogPct);
}
function jogStart(axis, dir, fast) {
  if (!S.jogOn) return toast("Jog deshabilitado (Ctrl-Alt-J)");
  if (!S.axes.includes(axis)) return;
  act("jog", { eje: axis, dir, paso: S.jogStep ? S.steps[S.stepIdx] : null, rapido: fast, porcentaje: S.jogPct });
}
function jogEnd() { if (!S.jogStep) act("jog_parar", {}, true); }
document.querySelectorAll("[data-jog]").forEach(b => {
  const [a, d] = b.dataset.jog.split(":");
  b.addEventListener("pointerdown", e => { b.classList.add("act"); jogStart(a, +d, e.shiftKey); });
  const up = () => { if (b.classList.contains("act")) { b.classList.remove("act"); jogEnd(); } };
  b.addEventListener("pointerup", up); b.addEventListener("pointerleave", up);
});

// --- MDI ---------------------------------------------------------------------------------------
let mdiIdx = -1;
$("mdi").addEventListener("keydown", async e => {
  if (e.altKey) return;                           // Alt-1…7, Alt-R, Alt-S siguen funcionando desde la entrada
  const h = S.st ? S.st.mdi_historial : [];
  if (e.key === "Enter") {
    const v = $("mdi").value.trim();
    if (v) { const r = await act("mdi", { linea: v }); if (r.ok) { $("mdi").value = ""; mdiIdx = -1; } }
  } else if (e.key === "ArrowUp" && h.length) { mdiIdx = Math.min(h.length - 1, mdiIdx + 1); $("mdi").value = h[h.length - 1 - mdiIdx]; e.preventDefault(); }
  else if (e.key === "ArrowDown") { mdiIdx = Math.max(-1, mdiIdx - 1); $("mdi").value = mdiIdx < 0 ? "" : h[h.length - 1 - mdiIdx]; e.preventDefault(); }
  else if (e.key === "Escape") { $("mdi").value = ""; $("mdi").blur(); }
  e.stopPropagation();
});

// --- orígenes ----------------------------------------------------------------------------------
const WCS = ["G54", "G55", "G56", "G57", "G58", "G59"];
const wcsP = () => WCS.indexOf((S.st && S.st.estado.wcs) || "G54") + 1 || 1;
let offsetsData = null;
async function loadOffsets() {
  const r = await api("/api/origenes");
  const axes = S.axes.length ? S.axes : ["X", "Y", "Z"];
  if (!r.ok) { $("offsets-table").innerHTML = `<tr><td>${esc(r.mensaje)}</td></tr>`; return; }
  offsetsData = r.origenes;
  const cur = (S.st && S.st.estado.wcs) || "G54";
  $("offsets-table").innerHTML = `<tr><th>Origen</th>${axes.map(a => `<th>${a}</th>`).join("")}</tr>` +
    Object.entries(r.origenes).map(([w, v]) => `<tr><td>${w === cur ? "▶ " : ""}${w}</td>${axes.map(a => `<td>${(v[a] || 0).toFixed(4)}</td>`).join("")}</tr>`).join("");
  $("wcs-edit").innerHTML = `<tr><th>Eje</th><th>Valor (máquina)</th></tr>` +
    axes.map(a => `<tr><td>${a}</td><td><input data-off="${a}" value="${((r.origenes[cur] || {})[a] || 0).toFixed(4)}"></td></tr>`).join("");
  $("fixtures").innerHTML = WCS.map((w, i) => `<button class="mbtn ${w === cur ? "on" : ""}" data-mdi="${w}">Origen ${i + 1} (${w})</button>`).join("");
  const cfg = S.cfg || (S.cfg = await api("/api/config"));
  const tools = Object.entries(cfg.herramientas);
  $("tools-table").innerHTML = `<tr><th>Herramienta</th><th>Largo (H)</th><th>Diámetro</th><th>Nombre</th></tr>` +
    (tools.length ? tools.map(([n, t]) => `<tr><td>T${n}</td><td>${t.largo.toFixed(3)}</td><td>${t.diametro.toFixed(3)}</td><td>${esc(t.nombre || "")}</td></tr>`).join("")
                  : `<tr><td class="muted">sin tabla de herramientas</td></tr>`);
}
function applyOffset() {
  const vals = [...document.querySelectorAll("[data-off]")].map(i => `${i.dataset.off}${parseFloat(i.value.replace(",", ".")) || 0}`).join(" ");
  act("mdi", { linea: `G10 L2 P${wcsP()} ${vals}` }).then(r => { if (r.ok) { toast("Origen actualizado", "info"); loadOffsets(); } });
}

// --- configuración ------------------------------------------------------------------------------
async function loadConfig() {
  const c = S.cfg || (S.cfg = await api("/api/config"));
  setB("cfg-file", c.archivo);
  const n = v => v == null ? "—" : (+v).toFixed(3);
  $("axes-table").innerHTML = `<tr><th>Eje</th><th>Pasos/mm</th><th>Vel. máx (mm/min)</th><th>Acel. (mm/s²)</th><th>Mín</th><th>Máx</th></tr>` +
    Object.entries(c.ejes).map(([a, x]) => `<tr><td>${a}</td><td>${x.pasos_mm}</td><td>${x.vel_max}</td><td>${x.acel}</td><td>${n(x.min)}</td><td>${n(x.max)}</td></tr>`).join("");
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

// --- diálogos ------------------------------------------------------------------------------------
let pendingId = null, modalResolve = null;
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
function closeModal(ok) { $("modal").hidden = true; if (modalResolve) { const r = modalResolve; modalResolve = null; r(ok); } }
$("modal-ok").onclick = () => closeModal(true);
$("modal-cancel").onclick = () => closeModal(false);
async function showApproval(p) {
  if (pendingId === p.id) return;
  pendingId = p.id;
  const ok = await openModal(p.tipo === "manual" ? "Habilitar la máquina" : "Aprobar el programa", p.resumen,
    p.informe ? (S.st && S.st.programa ? S.st.programa.informe : "") : "Jog, MDI, referenciado y ceros quedan habilitados hasta pulsar RESET.",
    "Aprobar y seguir", { guardian: true });
  await act(ok ? "aprobar" : "cancelar", ok ? { id: p.id } : {});
  pendingId = null;
}
async function editProgram() {
  if (!S.lines.length) return toast("no hay programa cargado");
  const name = (S.st && S.st.programa && S.st.programa.nombre) || "programa.nc";
  const ok = await openModal("Editar G-code", name, "", "Guardar y revisar",
    { html: `<textarea id="editor" spellcheck="false">${esc(S.lines.join("\n"))}</textarea>`, focus: "editor" });
  if (!ok) return;
  const r = await act("cargar", { nombre: name, texto: $("editor").value });
  if (r.ok) afterLoad(r, name);
}
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
const HELP = {
  G: [["G0", "rápido"], ["G1", "avance lineal"], ["G2 / G3", "arco horario / antihorario (I J K o R, P vueltas)"], ["G4 P", "pausa en segundos"],
      ["G10 L2 / L20", "fijar origen de trabajo"], ["G17 / G18 / G19", "plano XY / ZX / YZ"], ["G20 / G21", "pulgadas / mm"],
      ["G28 / G30", "ir a posición de inicio"], ["G31 / G38.2 / G38.3", "sensado (G31 de Mach3)"], ["G43 Hn / G49", "largo de herramienta"],
      ["G53", "coordenadas de máquina"], ["G54–G59", "orígenes de trabajo"], ["G61 / G64", "parada exacta / velocidad constante"],
      ["G73 / G81 / G82 / G83", "ciclos de taladrado (G80 cancela)"], ["G90 / G91", "absoluto / incremental"], ["G90.1 / G91.1", "centros de arco absolutos / incrementales"],
      ["G92 / G92.1", "desplazamiento de origen"], ["G98 / G99", "retracción al inicial / a R"]],
  M: [["M0 / M1", "pausa / pausa opcional"], ["M2 / M30", "fin de programa"], ["M3 / M4 / M5", "husillo (o antorcha en plasma) horario / antihorario / apagado"],
      ["M6 Tn", "cambio de herramienta (manual)"], ["M7 / M8 / M9", "niebla / refrigerante / apagado"], ["M100+", "macros de usuario de Mach3: se ignoran con aviso"]],
};
function helpCodes(kind) {
  const html = '<table class="table">' + HELP[kind].map(([c, d]) => `<tr><td>${c}</td><td style="text-align:left">${d}</td></tr>`).join("") + "</table>" +
    '<p class="muted small">No soportados (dan error claro): parámetros #, subprogramas M98/M99, G41/G42, G93, G12/G13, G15/G16, G50/G51, G52.</p>';
  openModal(kind === "G" ? "G-codes soportados" : "M-codes soportados", "", "", "Cerrar", { html, noCancel: true });
}

// --- teclado: atajos de Mach3 ---------------------------------------------------------------------
const JOGKEYS = { ArrowLeft: ["X", -1], ArrowRight: ["X", 1], ArrowUp: ["Y", 1], ArrowDown: ["Y", -1], PageUp: ["Z", 1], PageDown: ["Z", -1] };
document.addEventListener("keydown", e => {
  const tag = e.target.tagName;
  if (e.altKey && TAB_KEYS[e.key] && $("modal").hidden) { e.target.blur && e.target.blur(); setTab(TAB_KEYS[e.key]); e.preventDefault(); return; }
  if ((tag === "INPUT" && e.target.type !== "range") || tag === "TEXTAREA") {
    if (e.key === "Escape" && tag === "INPUT") e.target.blur();
    return;
  }
  if (!$("modal").hidden) { if (e.key === "Escape") closeModal(false); return; }
  const k = e.key.toLowerCase();
  if (e.altKey && TAB_KEYS[e.key]) { setTab(TAB_KEYS[e.key]); e.preventDefault(); return; }
  if (e.ctrlKey && e.altKey && k === "j") { ACTIONS.jog_onoff(); e.preventDefault(); return; }
  if (e.altKey && k === "r") { act("ciclo"); e.preventDefault(); return; }
  if (e.altKey && k === "s") { act("parar"); e.preventDefault(); return; }
  if (e.altKey && k === "n") { act("opcion", { nombre: "bloque_a_bloque" }); e.preventDefault(); return; }
  if (e.ctrlKey && k === "w") { ACTIONS.rebobinar(); e.preventDefault(); return; }
  if (e.ctrlKey && k === "o") { $("file").click(); e.preventDefault(); return; }
  if (e.ctrlKey && k === "f") { act("refrigerante"); e.preventDefault(); return; }
  if (e.key === "F5") { ACTIONS.husillo(); e.preventDefault(); return; }
  if (e.key === " ") { act("pausa"); e.preventDefault(); return; }
  if (e.key === "Escape") { act("parar"); e.preventDefault(); return; }
  if (e.key === "Enter" && S.tab === "mdi") { $("mdi").focus(); e.preventDefault(); return; }
  if (e.key === "Tab") { $("jog").hidden = !$("jog").hidden; e.preventDefault(); return; }
  if (JOGKEYS[e.key]) {
    e.preventDefault();
    if (e.repeat || S.jogKeys.has(e.key)) return;
    S.jogKeys.add(e.key);
    const [a, d] = JOGKEYS[e.key];
    jogStart(a, d, e.shiftKey);
  }
});
document.addEventListener("keyup", e => { if (JOGKEYS[e.key] && S.jogKeys.has(e.key)) { S.jogKeys.delete(e.key); jogEnd(); } });
window.addEventListener("blur", () => { if (S.jogKeys.size) { S.jogKeys.clear(); jogEnd(); } });

// --- inicio -----------------------------------------------------------------------------------------
(async () => {
  S.cfg = await api("/api/config");
  applyMachine(S.cfg);
  buildDro(Object.keys(S.cfg.ejes));
  renderJog(); led("led-seguir", S.follow); led("led-jog", S.jogOn); setB("vista", VIEW_NAMES[VIEWS[S.viewIdx]]);
  await loadProgram();
  poll(true);
})();
