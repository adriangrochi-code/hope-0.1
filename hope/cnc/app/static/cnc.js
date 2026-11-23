/* Hope CNC — pantalla al estilo Mach3. Sin dependencias: corre en navegadores viejos y nuevos. */
"use strict";

const $ = id => document.getElementById(id);
const S = { st: null, lines: [], sha: null, errLines: new Set(), coord: "work", step: 0, tab: "run",
            jogKeys: new Set(), lastLine: -1, follow: true, axes: [], cfg: null };
const ROW = 22;

// --- comunicación ---------------------------------------------------------------
async function api(path) {
  const r = await fetch(path, { cache: "no-store" });
  return r.json();
}
async function act(accion, params = {}, quiet = false) {
  try {
    const r = await fetch("/api/accion", { method: "POST", headers: { "Content-Type": "application/json", "X-Hope": "1" },
                                           body: JSON.stringify({ accion, params }) });
    const j = await r.json();
    if (!j.ok && !quiet) toast(j.mensaje || "error");
    if (j.aprobar) showApproval(j.aprobar);
    poll(true);
    return j;
  } catch (e) { if (!quiet) toast("sin conexión con Hope CNC"); return { ok: false }; }
}
function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = "toast " + kind; t.textContent = msg;
  $("toasts").append(t);
  setTimeout(() => t.remove(), kind === "info" ? 2500 : 5000);
}

// --- pestañas ------------------------------------------------------------------
function setTab(name) {
  S.tab = name;
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("on", t.id === "tab-" + name));
  if (name === "mdi") setTimeout(() => $("mdi").focus(), 0);
  if (name === "offsets") loadOffsets();
  if (name === "config") loadConfig();
  if (name === "path") { views.path.dirty = true; setTimeout(() => views.path.fit(), 0); }
  if (name === "run") { views.run.dirty = true; setTimeout(() => views.run.draw(), 0); renderCode(); }
}
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => setTab(b.dataset.tab));

// --- vistas de trayectoria ---------------------------------------------------------
const views = { run: new PathView($("view-run")), path: new PathView($("view-path")) };
document.querySelectorAll("[data-view-for]").forEach(seg => {
  const v = views[seg.dataset.viewFor];
  seg.querySelectorAll("button").forEach(b => b.onclick = () => {
    if (b.dataset.view !== "fit") seg.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    v.setView(b.dataset.view);
  });
});
async function loadPath() {
  const t = await api("/api/trayectoria");
  for (const v of Object.values(views)) v.setData(t.puntos, t.limites);
  if (t.recortado) toast("trayectoria recortada para dibujar (programa muy grande)", "info");
}

// --- G-code (lista virtual: rinde con cientos de miles de líneas) ------------------------
const esc = s => s.replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
function hl(line) {
  const parts = line.split(/(\(.*?\)|;.*$)/);
  return parts.map(p => {
    if (!p) return "";
    if (p.startsWith("(") || p.startsWith(";")) return `<span class="c">${esc(p)}</span>`;
    return esc(p).replace(/\b([Gg]\d+(?:\.\d+)?)/g, '<span class="g">$1</span>').replace(/\b([MmTt]\d+)/g, '<span class="m">$1</span>');
  }).join("");
}
function renderCode() {
  const box = $("code"), rows = $("code-rows");
  $("code-empty").hidden = S.lines.length > 0;
  $("code-spacer").style.height = S.lines.length * ROW + "px";
  const cur = S.st && S.st.programa ? S.st.programa.linea_actual : 0;
  const first = Math.max(0, Math.floor(box.scrollTop / ROW) - 10);
  const last = Math.min(S.lines.length, first + Math.ceil(box.clientHeight / ROW) + 20);
  rows.style.top = first * ROW + "px";
  let html = "";
  for (let i = first; i < last; i++) {
    const n = i + 1;
    const cls = n === cur ? "cur" : (cur && n < cur ? "done" : "") + (S.errLines.has(n) ? " err" : "");
    html += `<div class="code-row ${cls}"><span class="n">${n}</span><span>${hl(S.lines[i])}</span></div>`;
  }
  rows.innerHTML = html;
}
$("code").addEventListener("scroll", () => { renderCode(); });
$("code").addEventListener("wheel", () => { S.follow = false; clearTimeout(S.followT); S.followT = setTimeout(() => S.follow = true, 4000); });
function followLine(n) {
  if (!n || !S.follow) return;
  const box = $("code");
  const y = (n - 1) * ROW;
  if (y < box.scrollTop + ROW * 2 || y > box.scrollTop + box.clientHeight - ROW * 4) box.scrollTop = y - box.clientHeight / 3;
}

async function loadProgram() {
  const p = await api("/api/programa");
  S.lines = p.lineas || [];
  $("file-name").textContent = p.nombre || "sin programa";
  $("code").scrollTop = 0;
  renderCode();
  await loadPath();
}
async function loadFile(file) {
  if (!file) return;
  if (file.size > 32 * 1024 * 1024) return toast("archivo demasiado grande (máx. 32 MB)");
  const text = await file.text();
  const r = await act("cargar", { nombre: file.name, texto: text });
  if (r.ok) {
    toast(r.informe.runnable ? `«${file.name}» revisado: listo para enviar` : `«${file.name}» tiene errores: mirá Diagnóstico`,
          r.informe.runnable ? "info" : "");
    await loadProgram();
  }
}
$("btn-load").onclick = () => $("file").click();
$("file").onchange = e => { loadFile(e.target.files[0]); e.target.value = ""; };
$("btn-close").onclick = async () => { const r = await act("cerrar_programa"); if (r.ok) loadProgram(); };
document.addEventListener("dragover", e => { e.preventDefault(); document.body.classList.add("drop"); });
document.addEventListener("dragleave", e => { if (!e.relatedTarget) document.body.classList.remove("drop"); });
document.addEventListener("drop", e => { e.preventDefault(); document.body.classList.remove("drop"); loadFile(e.dataTransfer.files[0]); });

// --- DRO --------------------------------------------------------------------------
function buildDro(axes) {
  S.axes = axes;
  $("dro-rows").innerHTML = axes.map(a => `
    <div class="dro-row"><span class="dro-axis">${a}</span><span class="led" id="led-${a}" title="referenciado"></span>
      <span class="dro-val" id="dro-${a}">+0.000</span><button class="btn" data-zero="${a}">Cero ${a}</button>
      <span class="dro-sub" id="dro-sub-${a}"></span></div>`).join("");
  document.querySelectorAll("[data-zero]").forEach(b => b.onclick = () => act("cero", { eje: b.dataset.zero }));
}
const fmt = v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3);
const hms = s => { if (s == null) return "—"; s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), x = s % 60;
                   return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(x).padStart(2, "0"); };
document.querySelectorAll("#coord-mode button").forEach(b => b.onclick = () => {
  S.coord = b.dataset.coord;
  document.querySelectorAll("#coord-mode button").forEach(x => x.classList.toggle("on", x === b));
  document.querySelector(".dro").classList.toggle("machine", S.coord === "machine");
});

// --- estado -------------------------------------------------------------------------
let polling = false;
async function poll(force) {
  if (polling && !force) return;
  polling = true;
  try { render(await api("/api/estado")); }
  catch (e) { $("state-pill").textContent = "Sin conexión"; $("state-pill").className = "pill Desconectado"; }
  finally { polling = false; }
}
function render(st) {
  S.st = st;
  const c = st.estado, job = c.trabajo;
  if (S.axes.join() !== st.ejes.join()) buildDro(st.ejes);
  const state = st.habilitada ? c.estado : "Reset";
  const names = { Idle: "En reposo", Run: "Ejecutando", Hold: "En pausa", Jog: "Jog", Home: "Referenciando", Alarm: "ALARMA",
                  Desconectado: "Desconectado", Reset: "RESET", Door: "Puerta" };
  $("state-pill").textContent = names[state] || state;
  $("state-pill").className = "pill " + state;
  $("conn-name").textContent = c.conexion + (st.fisica ? "" : " · sin máquina");
  for (const a of st.ejes) {
    const w = st.trabajo_pos[a], m = c.mpos[a];
    const main = S.coord === "work" ? w : m, sub = S.coord === "work" ? m : w;
    $("dro-" + a).textContent = fmt(main);
    $("dro-sub-" + a).textContent = (S.coord === "work" ? "máq " : "trab ") + fmt(sub);
    $("led-" + a).classList.toggle("on", !!c.referenciado);
  }
  $("wcs").textContent = c.wcs || "G54";
  $("fro").textContent = c.ov.avance; $("sro").textContent = c.ov.husillo;
  document.querySelectorAll("#rapid-ov button").forEach(b => b.classList.toggle("on", b.dataset.ov === "rapido:" + c.ov.rapido));
  $("feed-now").textContent = Math.round(c.avance);
  $("rpm-now").textContent = Math.round(c.husillo);
  $("btn-spindle").classList.toggle("on", c.husillo_dir !== "off");
  $("btn-coolant").classList.toggle("on", !!c.refrigerante);
  const reset = $("btn-reset");
  reset.classList.toggle("off", !st.habilitada); reset.classList.toggle("on", st.habilitada);
  $("reset-sub").textContent = st.habilitada ? "máquina habilitada · Esc para deshabilitar" : "máquina deshabilitada · pulsar para habilitar";
  document.body.classList.toggle("running", job.activo && c.estado === "Run");

  const p = st.programa;
  if (p) {
    if (p.sha256 !== S.sha) { S.sha = p.sha256; S.errLines = new Set(p.errores.map(e => +((e.match(/línea (\d+)/) || [])[1] || 0))); }
    $("progress-bar").style.width = (job.terminado ? 100 : p.progreso * 100) + "%";
    $("line-info").textContent = `Línea ${p.linea_actual} / ${p.lineas_total}`;
    $("time-info").textContent = job.activo ? `${hms(p.transcurrido)} · faltan ${hms(p.tiempo_restante)}` : `estimado ${hms(p.tiempo_total)}`;
    const rl = $("report-line");
    rl.textContent = p.runnable ? `✓ Listo · ${p.avisos.length} avisos · ${hms(p.tiempo_total)} estimado`
                                : `✕ ${p.errores.length} errores — ${p.errores[0] || ""}`;
    rl.className = "report " + (p.runnable ? "ok" : "bad");
    if (p.linea_actual !== S.lastLine) { S.lastLine = p.linea_actual; followLine(p.linea_actual); if (S.tab === "run") renderCode(); }
    $("st-elapsed").textContent = hms(p.transcurrido); $("st-left").textContent = job.activo ? hms(p.tiempo_restante) : "—";
    $("diag-report").textContent = p.informe;
  } else {
    $("progress-bar").style.width = "0"; $("line-info").textContent = "Línea 0 / 0"; $("time-info").textContent = "—";
    $("report-line").textContent = "Cargá un programa para revisarlo."; $("report-line").className = "report";
    $("diag-report").textContent = "Sin programa.";
    if (S.sha) { S.sha = null; S.lines = []; renderCode(); }
  }
  const tool = { X: st.trabajo_pos.X, Y: st.trabajo_pos.Y, Z: st.trabajo_pos.Z };
  const line = job.activo || job.terminado ? (p ? p.linea_actual : 0) : 0;
  views[S.tab === "path" ? "path" : "run"].setProgress(line, tool);

  const last = st.mensajes[st.mensajes.length - 1];
  $("st-msg").textContent = c.alarma ? "ALARMA: " + c.alarma : c.mensaje ? "Mensaje: " + c.mensaje
                           : job.error ? "Detenido: " + job.error : last ? last.m : "—";
  $("st-machine").textContent = st.maquina;
  if (S.tab === "diag") {
    $("diag-state").textContent = JSON.stringify(c, null, 2);
    $("diag-log").innerHTML = st.mensajes.slice().reverse().map(x =>
      `<li><span class="t">${new Date(x.t * 1000).toLocaleTimeString()}</span>${esc(x.m)}</li>`).join("");
  }
  if (S.tab === "mdi") {
    const h = st.mdi_historial.slice().reverse();
    const html = h.map(x => `<li data-cmd="${esc(x)}">${esc(x)}</li>`).join("") || '<li class="muted">vacío</li>';
    if ($("mdi-history").innerHTML !== html) {
      $("mdi-history").innerHTML = html;
      $("mdi-history").querySelectorAll("li[data-cmd]").forEach(li => li.onclick = () => { $("mdi").value = li.dataset.cmd; $("mdi").focus(); });
    }
  }
  if (st.pendiente && $("modal").hidden) showApproval(st.pendiente);
}
setInterval(() => poll(), 150);

// --- botones ----------------------------------------------------------------------------
$("btn-start").onclick = () => act("ciclo");
$("btn-hold").onclick = () => act("pausa");
$("btn-stop").onclick = () => act("parar");
$("btn-rewind").onclick = () => { act("rebobinar"); $("code").scrollTop = 0; };
$("btn-reset").onclick = () => act("reset");
$("btn-home").onclick = () => act("referenciar");
$("btn-zero-all").onclick = () => act("cero", { eje: "todo" });
$("btn-goto-zero").onclick = () => act("ir_a_cero");
$("btn-spindle").onclick = () => act("husillo", { on: !$("btn-spindle").classList.contains("on") });
$("btn-coolant").onclick = () => act("refrigerante");
document.querySelectorAll("[data-ov]").forEach(b => b.onclick = () => {
  const [tipo, accion] = b.dataset.ov.split(":"); act("override", { tipo, accion });
});

// --- MDI ----------------------------------------------------------------------------------
let mdiIdx = -1;
$("mdi").addEventListener("keydown", async e => {
  const h = S.st ? S.st.mdi_historial : [];
  if (e.key === "Enter") {
    const v = $("mdi").value.trim();
    if (!v) return;
    const r = await act("mdi", { linea: v });
    if (r.ok) { $("mdi").value = ""; mdiIdx = -1; }
  } else if (e.key === "ArrowUp" && h.length) {
    mdiIdx = Math.min(h.length - 1, mdiIdx + 1); $("mdi").value = h[h.length - 1 - mdiIdx]; e.preventDefault();
  } else if (e.key === "ArrowDown") {
    mdiIdx = Math.max(-1, mdiIdx - 1); $("mdi").value = mdiIdx < 0 ? "" : h[h.length - 1 - mdiIdx]; e.preventDefault();
  } else if (e.key === "Escape") { $("mdi").blur(); }
  e.stopPropagation();
});
document.querySelectorAll("[data-mdi]").forEach(b => b.onclick = () => act("mdi", { linea: b.dataset.mdi }));

// --- orígenes y máquina -------------------------------------------------------------------
async function loadOffsets() {
  const r = await api("/api/origenes");
  const axes = S.axes.length ? S.axes : ["X", "Y", "Z"];
  if (!r.ok) { $("offsets-table").innerHTML = `<tr><td>${esc(r.mensaje)}</td></tr>`; return; }
  $("offsets-table").innerHTML = `<tr><th>Origen</th>${axes.map(a => `<th>${a}</th>`).join("")}</tr>` +
    Object.entries(r.origenes).map(([w, v]) => `<tr><td>${w}</td>${axes.map(a => `<td>${(v[a] || 0).toFixed(3)}</td>`).join("")}</tr>`).join("");
  const cfg = S.cfg || (S.cfg = await api("/api/config"));
  const tools = Object.entries(cfg.herramientas);
  $("tools-table").innerHTML = `<tr><th>Herramienta</th><th>Largo</th><th>Diámetro</th><th>Nombre</th></tr>` +
    (tools.length ? tools.map(([n, t]) => `<tr><td>T${n}</td><td>${t.largo.toFixed(3)}</td><td>${t.diametro.toFixed(3)}</td><td>${esc(t.nombre || "")}</td></tr>`).join("")
                  : `<tr><td class="muted">sin tabla de herramientas</td></tr>`);
}
$("btn-offsets").onclick = loadOffsets;
async function loadConfig() {
  const c = S.cfg || (S.cfg = await api("/api/config"));
  $("cfg-file").textContent = c.archivo;
  const n = v => v == null ? "—" : (+v).toFixed(3);
  $("axes-table").innerHTML = `<tr><th>Eje</th><th>Pasos/mm</th><th>Vel. máx (mm/min)</th><th>Acel. (mm/s²)</th><th>Mín</th><th>Máx</th></tr>` +
    Object.entries(c.ejes).map(([a, x]) => `<tr><td>${a}</td><td>${x.pasos_mm}</td><td>${x.vel_max}</td><td>${x.acel}</td><td>${n(x.min)}</td><td>${n(x.max)}</td></tr>`).join("");
  $("cfg-extra").innerHTML = `<p><span class="muted">Tipo</span> ${esc(c.tipo)} · <span class="muted">Desviación de unión</span> ${c.desviacion_union} mm ·
     <span class="muted">Tolerancia de arco</span> ${c.tolerancia_arco} mm · <span class="muted">Husillo</span> ${c.husillo.rpm_min}–${c.husillo.rpm_max} rpm ·
     <span class="muted">Controlador</span> ${esc(c.controlador.tipo || "sim")}${c.controlador.puerto ? " (" + esc(c.controlador.puerto) + ")" : ""}</p>`;
}

// --- aprobación -------------------------------------------------------------------------------
let pendingId = null;
function showApproval(p) {
  pendingId = p.id;
  $("modal-title").textContent = p.tipo === "manual" ? "Habilitar la máquina" : "Aprobar el programa";
  $("modal-text").textContent = p.resumen;
  $("modal-report").textContent = p.informe ? (S.st && S.st.programa ? S.st.programa.informe : "") : "Jog, MDI, referenciado y ceros quedan habilitados hasta pulsar RESET.";
  $("modal").hidden = false;
  $("modal-ok").focus();
}
$("modal-ok").onclick = async () => { $("modal").hidden = true; const r = await act("aprobar", { id: pendingId }); pendingId = null; return r; };
$("modal-cancel").onclick = async () => { $("modal").hidden = true; await act("cancelar"); pendingId = null; };

// --- jog -------------------------------------------------------------------------------------
function setStep(v) {
  S.step = +v;
  document.querySelectorAll("#jog-mode button").forEach(b => b.classList.toggle("on", +b.dataset.step === S.step));
  $("jog-mode-label").textContent = S.step ? `Paso ${S.step}` : "Continuo";
}
document.querySelectorAll("#jog-mode button").forEach(b => b.onclick = () => setStep(b.dataset.step));
$("jog-pct").oninput = () => $("jog-pct-v").textContent = $("jog-pct").value + "%";
$("jog-close").onclick = () => $("jog").hidden = true;
function jogStart(axis, dir, fast) {
  if (!S.axes.includes(axis)) return;
  act("jog", { eje: axis, dir, paso: S.step || null, rapido: fast, porcentaje: +$("jog-pct").value });
}
function jogEnd() { if (!S.step) act("jog_parar", {}, true); }
document.querySelectorAll("[data-jog]").forEach(b => {
  const [a, d] = b.dataset.jog.split(":");
  b.addEventListener("pointerdown", e => { b.classList.add("act"); jogStart(a, +d, e.shiftKey); });
  const up = () => { if (b.classList.contains("act")) { b.classList.remove("act"); jogEnd(); } };
  b.addEventListener("pointerup", up); b.addEventListener("pointerleave", up);
});

// --- teclado (atajos de Mach3) -------------------------------------------------------------------
const JOGKEYS = { ArrowLeft: ["X", -1], ArrowRight: ["X", 1], ArrowUp: ["Y", 1], ArrowDown: ["Y", -1], PageUp: ["Z", 1], PageDown: ["Z", -1] };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type !== "range") return;
  if (!$("modal").hidden) {
    if (e.key === "Escape") $("modal-cancel").click();
    return;
  }
  if (e.altKey && /^[1-6]$/.test(e.key)) { setTab(["run", "mdi", "path", "offsets", "diag", "config"][+e.key - 1]); e.preventDefault(); return; }
  if (e.altKey && e.key.toLowerCase() === "r") { act("ciclo"); e.preventDefault(); return; }
  if (e.altKey && e.key.toLowerCase() === "s") { act("parar"); e.preventDefault(); return; }
  if (e.ctrlKey && e.key.toLowerCase() === "w") { $("btn-rewind").click(); e.preventDefault(); return; }
  if (e.ctrlKey && e.key.toLowerCase() === "o") { $("file").click(); e.preventDefault(); return; }
  if (e.key === " ") { act("pausa"); e.preventDefault(); return; }
  if (e.key === "Escape") { if (S.st && S.st.habilitada) act("reset"); e.preventDefault(); return; }
  if (e.key === "Tab") { $("jog").hidden = !$("jog").hidden; e.preventDefault(); return; }
  if (JOGKEYS[e.key]) {
    e.preventDefault();
    if (e.repeat || S.jogKeys.has(e.key)) return;
    S.jogKeys.add(e.key);
    const [a, d] = JOGKEYS[e.key];
    jogStart(a, d, e.shiftKey);
  }
});
document.addEventListener("keyup", e => {
  if (JOGKEYS[e.key] && S.jogKeys.has(e.key)) { S.jogKeys.delete(e.key); jogEnd(); }
});
window.addEventListener("blur", () => { if (S.jogKeys.size) { S.jogKeys.clear(); jogEnd(); } });

// --- inicio ---------------------------------------------------------------------------------------
(async () => {
  S.cfg = await api("/api/config");
  buildDro(Object.keys(S.cfg.ejes));
  await loadProgram();
  poll(true);
})();
