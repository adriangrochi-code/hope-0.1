/* ==================================================================================================
   TISMA CNC (Hope CNC) — pantallas con la distribución de Mach3Mill. Sin dependencias: corre en
   navegadores viejos y nuevos. Los grupos que se repiten (DRO, herramienta, avance, husillo) salen de
   plantillas y se actualizan por atributo data-b, así cada pantalla tiene los suyos como en Mach3.
   ==================================================================================================

   QUÉ ES
     La lógica de la pantalla de control: lee el estado de la máquina, lo muestra (DRO, LEDs, listado
     de G-code, trayectoria) y manda al servidor lo que pide el operador (botones, teclado, MDI, jog).
     No decide nada sobre la máquina: toda acción la valida y ejecuta el servidor (hope/cnc/app/server.py
     → hope/cnc/session.py), y mover una máquina real pide aprobación del guardián (diálogo «Aprobar»).

   CÓMO ENCAJA
     * index.html carga, en este orden: viejo.js (aviso si el navegador es demasiado viejo), compat.js,
       pathview.js y este archivo. Todos son scripts propios del servidor (la CSP no deja scripts en línea).
     * De compat.js usa: onPress(el, bajar, subir) para botones que se mantienen apretados (jog),
       readText(archivo) para leer archivos, normKey(evento) para el nombre de la tecla (los navegadores
       viejos dicen "Left", "Esc"…), setLite(on) y window.TISMA_LITE (modo liviano para PCs viejas).
     * De pathview.js usa la clase PathView (un dibujo de la trayectoria por cada [data-view]).
     * Del servidor usa: GET /api/estado (cada 150 ms; 400 ms en modo liviano), /api/config,
       /api/programa, /api/trayectoria, /api/origenes, /api/recientes, /api/perfiles, /api/perfil y
       POST /api/accion (ver act()).

   CONVENCIONES DEL HTML
     * data-b="clave"   elemento que muestra un dato; setB/clsB/led lo actualizan en TODAS las pantallas.
     * data-act="x"     botón que ejecuta ACTIONS.x (un solo manejador de clics para toda la página).
     * data-ov="t:a"    botón de override (avance/husillo/rápido): manda act("override", {tipo, accion}).
     * data-mdi="G54"   botón que manda esa línea por MDI.
     * data-jog="X:1"   botón de jog: eje y sentido.
     * data-tpl="dro"   caja que se llena con la plantilla <template id="tpl-dro">.
     * data-view="run"  lienzo de trayectoria de la pantalla «run».
     Los textos que ve el operador y los nombres de la API están en español; unidades en mm, mm/min,
     rpm y segundos, como en el servidor.

   COMPATIBILIDAD
     Firefox 52 ESR (Windows XP/Vista) y Chrome 57 en adelante: sintaxis ES2017 (async/await, flechas,
     plantillas de texto, parámetros por defecto). NO usar encadenamiento opcional, coalescencia nula,
     replaceAll, at, flatMap, Object.fromEntries, structuredClone, Response.text ni ResizeObserver
     (tests/test_cnc_compat.py lo revisa); para eventos de puntero, onPress() de compat.js.

   EJEMPLO
     <button class="mbtn" data-act="ciclo">Cycle Start</button>   → al hacer clic: act("ciclo")
     <span data-b="fro"></span>%                                   → render() pone el override de avance
   ================================================================================================== */
"use strict";

// =================================================================================================
// =================================== ESTADO GLOBAL Y CONSTANTES ==================================
// =================================================================================================

// Atajo: $("mdi") = document.getElementById("mdi"). Sin jQuery: no hay dependencias.
const $ = id => document.getElementById(id);
// S: todo el estado de la pantalla en un solo objeto (fácil de mirar desde la consola del navegador).
//   st        última respuesta de /api/estado (null hasta la primera consulta)
//   cfg       configuración de la máquina efectiva (/api/config)
//   lines     líneas del programa cargado, tal como se muestran en el listado
//   sha       sha256 del programa mostrado: undefined = todavía no se consultó, null = sin programa
//   errLines  números de línea con error (se pintan en rojo); errSha = de qué programa son
//   coord     "work" o "machine": qué coordenadas muestra el DRO grande (Machine Coords de Mach3)
//   tab       pantalla visible (run, mdi, profiles, path, offsets, config, diag)
//   jogOn     jog habilitado (Ctrl-Alt-J en Mach3)
//   jogStep   false = jog continuo mientras se mantiene apretado; true = por pasos (Step de Mach3)
//   steps     pasos de jog disponibles en mm; stepIdx = el elegido (0.1 mm al empezar)
//   jogPct    velocidad del jog en % de la máxima del eje (Slow Jog Rate de Mach3), entre 5 y 100
//   jogKeys   teclas de jog apretadas ahora (para ignorar la autorrepetición del teclado)
//   lastLine  última línea en ejecución vista (para seguirla en el listado sin redibujar de más)
//   follow    la vista de trayectoria sigue a la herramienta
//   axes      ejes que tiene armado el DRO; viewIdx = índice en VIEWS (1 = ISO)
//   programLoadedAt  reservado (no se usa por ahora)
// Más adelante se agregan: followCode/followT (seguir la línea actual en el listado), lastNext,
// clearedAt (cuándo se borró la línea de estado) y pk/prof/presets (perfiles).
const S = { st: null, cfg: null, lines: [], sha: undefined, errLines: new Set(), coord: "work", tab: "run",
            jogOn: true, jogStep: false, steps: [0.001, 0.01, 0.1, 1, 10], stepIdx: 2, jogPct: 30,
            jogKeys: new Set(), lastLine: -1, follow: false, axes: [], viewIdx: 1, programLoadedAt: 0 };
// Vistas de la trayectoria, en el orden en que las recorre el botón «Vista» (Display Mode de Mach3).
const VIEWS = ["top", "iso", "front", "side"];
// Nombre que se muestra de cada vista.
const VIEW_NAMES = { top: "Arriba", iso: "ISO", front: "Frente", side: "Lado" };
// Alto de cada línea del listado de G-code en px. Tiene que ser igual a «.code-row { height: 21px }»
// de cnc.css: la lista virtual calcula con esto qué líneas están a la vista.
const ROW = 21;

// =================================================================================================
// ================================== COMUNICACIÓN CON EL SERVIDOR =================================
// =================================================================================================

// --- comunicación ---------------------------------------------------------------
// api(ruta): GET de una consulta del servidor (/api/...); devuelve el JSON ya interpretado.
// Sin caché, para no mostrar un estado viejo. Si falla la red, la promesa se rechaza (lo maneja
// quien llama; poll() lo muestra como «Sin conexión»).
async function api(path) { const r = await fetch(path, { cache: "no-store" }); return r.json(); }
// act(accion, params, quiet): ejecuta una acción del servidor (lista blanca ACTIONS de server.py).
//   accion  nombre ("ciclo", "jog", "mdi"…); params: sus parámetros; quiet: no mostrar errores.
// Devuelve la respuesta ({ok, mensaje, …}); si no hay conexión, {ok: false}. Nunca lanza error.
// POR QUÉ el encabezado X-Hope y JSON: el servidor los exige para que otra página web no pueda
// mandar acciones (CSRF); un formulario de otro sitio no puede poner ninguno de los dos.
// Si el guardián pide aprobación (j.aprobar), abre el diálogo; y siempre refresca el estado ya, así
// el botón se ve responder sin esperar a la próxima consulta periódica.
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
// toast(msg, kind): aviso flotante abajo de la pantalla. kind "info" = verde, dura 2,5 s;
// sin kind = error, dura 5 s. textContent (no innerHTML): el mensaje puede traer texto del archivo.
function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = "toast " + kind; t.textContent = msg;
  $("toasts").append(t);
  setTimeout(() => t.remove(), kind === "info" ? 2500 : 5000);
}
// esc(s): escapa & < > " para meter texto en HTML armado a mano (nombres de archivo, comentarios
// del G-code, perfiles). Todo dato que no sea nuestro pasa por acá antes de ir a innerHTML.
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// =================================================================================================
// ===================================== ARMADO DE LA PANTALLA =====================================
// =================================================================================================

// --- enlaces de datos (data-b) ----------------------------------------------------------
// setB(clave, texto): pone el texto en todos los [data-b=clave]. Solo toca el DOM si cambió: se
// llama decenas de veces cada 150 ms y escribir siempre haría redibujar la página (lento en PCs viejas).
function setB(key, text) { document.querySelectorAll(`[data-b="${key}"]`).forEach(e => { if (e.textContent !== String(text)) e.textContent = text; }); }
// clsB(clave, clase, on): pone o saca una clase CSS en todos los [data-b=clave].
function clsB(key, cls, on) { document.querySelectorAll(`[data-b="${key}"]`).forEach(e => e.classList.toggle(cls, !!on)); }
// led(clave, on, blink): LED de Mach3 (el cuadradito al lado de cada botón): prendido y/o titilando.
function led(key, on, blink = false) { document.querySelectorAll(`[data-b="${key}"]`).forEach(e => { e.classList.toggle("on", !!on); e.classList.toggle("blink", !!blink); }); }

// --- plantillas (los grupos de Mach3 se repiten en varias pantallas) -----------------------
// Copia cada plantilla en sus cajas al cargar la página (Mach3 repite el DRO, la herramienta, etc. en
// varias pantallas; acá se escriben una sola vez en index.html).
document.querySelectorAll("[data-tpl]").forEach(box => {
  box.append($("tpl-" + box.dataset.tpl).content.cloneNode(true));
  if (box.dataset.tpl === "dro") box.classList.add("dro-panel");
});
// buildDro(ejes): arma las filas del DRO (botón Cero, LED de referenciado y valor) para los ejes de la
// máquina, p. ej. ["X","Y","Z"] o con "A". Se rehace al cambiar de máquina; los botones de 4.º eje
// (.ax-a) se ocultan si no hay A. «Cero» = G10 L20 P1 (Zero X/Y/Z de Mach3).
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
// Tecla (con Alt) → pantalla. Mach3: Alt-1 Program Run, Alt-2 MDI, Alt-4 ToolPath, Alt-5 Offsets,
// Alt-6 Settings, Alt-7 Diagnostics. Alt-3 (en Mach3, Toolpath viejo) se usa acá para los perfiles.
const TAB_KEYS = { "1": "run", "2": "mdi", "3": "profiles", "4": "path", "5": "offsets", "6": "config", "7": "diag" };
// setTab(nombre): muestra una pantalla y oculta las otras. Al entrar carga lo que esa pantalla
// necesita (orígenes, configuración, perfiles) y reajusta la vista de trayectoria al nuevo tamaño
// (con setTimeout, después de que el navegador haya mostrado la pantalla y conozca su tamaño).
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
// ===================================== PROGRAMA Y TRAYECTORIA ====================================
// =================================================================================================

// --- trayectoria -------------------------------------------------------------------------------
// Un PathView (pathview.js) por cada lienzo [data-view]; la clave es la pantalla donde está.
const views = {};
document.querySelectorAll("[data-view]").forEach(el => { views[el.dataset.view] = new PathView(el); });
// La vista de trayectoria de la pantalla visible (o null si no tiene).
const curView = () => views[S.tab] || null;
// Última trayectoria recibida: puntos [x, y, z, línea…], límites y perforaciones (plasma).
let pathData = { puntos: [], limites: null, perforaciones: [] };
// loadPath(): pide la trayectoria al servidor y se la pasa a todas las vistas. Si el programa es muy
// grande el servidor la recorta (recortado) para que el dibujo no trabe una PC vieja.
async function loadPath() {
  pathData = await api("/api/trayectoria");
  for (const v of Object.values(views)) { v.follow = S.follow; v.setData(pathData.puntos, pathData.limites, pathData.perforaciones); v.setView(VIEWS[S.viewIdx]); }
  if (pathData.recortado) toast("trayectoria recortada para dibujar (programa muy grande)", "info");
  renderLimits();
}
// renderLimits(): tabla de mínimos y máximos de X, Y, Z del programa (como el «Program Limits» de
// Mach3), para ver antes de correr si entra en la mesa. Con un for simple: puede haber cientos de
// miles de puntos y Math.min(...puntos) desbordaría la pila.
// wrapTool(pos): posición de la herramienta para la vista. Con ejes rotativos, la trayectoria se
// dibuja sobre la pieza (enrollada, ver _view_point_function en session.py): la herramienta se gira
// igual, al revés del ángulo de cada eje (A alrededor de X, B de Y, C de Z), para que coincida.
function wrapTool(pos) {
  let x = pos.X || 0, y = pos.Y || 0, z = pos.Z || 0;
  for (const a of pathData.rotativos || []) {
    const ang = -(pos[a] || 0) * Math.PI / 180, c = Math.cos(ang), s = Math.sin(ang);
    if (a === "A") { const y2 = y * c - z * s; z = y * s + z * c; y = y2; }
    else if (a === "B") { const z2 = z * c - x * s; x = z * s + x * c; z = z2; }
    else { const x2 = x * c - y * s; y = x * s + y * c; x = x2; }
  }
  return { X: x, Y: y, Z: z };
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
// hl(línea): resalta una línea de G-code para el listado: comentarios (entre paréntesis o tras «;»),
// códigos G y códigos M/T. Devuelve HTML ya escapado con esc().
function hl(line) {
  return line.split(/(\(.*?\)|;.*$)/).map(p => {
    if (!p) return "";
    if (p.startsWith("(") || p.startsWith(";")) return `<span class="c">${esc(p)}</span>`;
    return esc(p).replace(/\b([Gg]\d+(?:\.\d+)?)/g, '<span class="g">$1</span>').replace(/\b([MmTt]\d+)/g, '<span class="m">$1</span>');
  }).join("");
}
// renderCode(): dibuja el listado de G-code como «lista virtual»: un espaciador con el alto total
// (líneas × ROW) da la barra de desplazamiento real, y solo se crean las filas visibles más un
// margen (10 arriba, 20 abajo). Así anda igual con 300 000 líneas que con 30.
// Marca la línea actual (cur), las ya hechas (done), la próxima elegida (next) y las con error (err).
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
// Si el operador mueve la rueda, deja de seguir la línea actual por 4 s para poder mirar el programa.
$("code").addEventListener("wheel", () => { S.followCode = false; clearTimeout(S.followT); S.followT = setTimeout(() => S.followCode = true, 4000); });
// Clic en una línea: queda escrita en la casilla de «Set Next Line» (como en Mach3).
$("code").addEventListener("click", e => { const r = e.target.closest(".code-row"); if (r) $("line-in").value = r.dataset.line; });
S.followCode = true;
// followLine(n): desplaza el listado para que la línea n quede a la vista (a un tercio del alto),
// solo si se está yendo del margen visible, para que el listado no salte en cada línea.
function followLine(n) {
  if (!n || !S.followCode) return;
  const box = $("code"), y = (n - 1) * ROW;
  if (y < box.scrollTop + ROW * 2 || y > box.scrollTop + box.clientHeight - ROW * 4) box.scrollTop = y - box.clientHeight / 3;
}
// loadProgram(): trae las líneas del programa cargado y su trayectoria, y vuelve al principio.
async function loadProgram() {
  const p = await api("/api/programa");
  S.lines = p.lineas || [];
  $("code").scrollTop = 0;
  renderCode();
  await loadPath();
}
// loadFile(archivo): manda un archivo elegido (o arrastrado) al servidor para revisarlo y cargarlo.
// El límite de 32 MB es el mismo MAX_BODY del servidor: se avisa antes de mandarlo para no esperar en vano.
async function loadFile(file) {
  if (!file) return;
  if (file.size > 32 * 1024 * 1024) return toast("archivo demasiado grande (máx. 32 MB)");
  const r = await act("cargar", { nombre: file.name, texto: await readText(file) });
  if (r.ok) { afterLoad(r, file.name); }
}
// afterLoad(respuesta, nombre): después de cargar un programa (archivo, editor o recientes) avisa si
// quedó listo o tiene errores y lo muestra.
async function afterLoad(r, name) {
  S.sha = r.informe.sha256;                 // ya lo cargamos nosotros: que poll() no lo vuelva a pedir
  toast(r.informe.runnable ? `«${name}» revisado: listo` : `«${name}» tiene errores: mirá Diagnóstico (Alt-7)`, r.informe.runnable ? "info" : "");
  await loadProgram();
}
// Elegir archivo (Load G-Code). Se vacía el valor para poder volver a elegir el mismo archivo.
$("file").onchange = e => { loadFile(e.target.files[0]); e.target.value = ""; };
// Arrastrar y soltar un archivo en cualquier parte de la ventana también lo carga.
document.addEventListener("dragover", e => { e.preventDefault(); document.body.classList.add("drop"); });
document.addEventListener("dragleave", e => { if (!e.relatedTarget) document.body.classList.remove("drop"); });
document.addEventListener("drop", e => { e.preventDefault(); document.body.classList.remove("drop"); loadFile(e.dataTransfer.files[0]); });

// =================================================================================================
// =========================== ESTADO DE LA MÁQUINA (CONSULTA PERIÓDICA) ===========================
// =================================================================================================

// --- estado ----------------------------------------------------------------------------------
// fmt4(v): número del DRO con signo y 4 decimales, como Mach3 (+12.3400 / −0.5000). El «−» es el
// signo menos tipográfico, del mismo ancho que «+», para que los dígitos no bailen.
const fmt4 = v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(4);
// hms(segundos): «hh:mm:ss» para tiempo transcurrido y restante; null → 00:00:00.
const hms = s => { if (s == null) return "00:00:00"; s = Math.max(0, Math.round(s)); return [s / 3600, s % 3600 / 60, s % 60].map(x => String(Math.floor(x)).padStart(2, "0")).join(":"); };
// Estado del controlador (nombres de grbl) → texto que se muestra en la «píldora» de estado.
const STATE_NAMES = { Idle: "En reposo", Run: "Ejecutando", Hold: "En pausa", Jog: "Jog", Home: "Referenciando", Alarm: "ALARMA",
                      Desconectado: "Desconectado", Reset: "RESET", Door: "Puerta" };
// Hay una consulta de estado en curso: en una PC lenta no se apilan consultas una atrás de otra.
let polling = false;
// poll(force): pide /api/estado y lo muestra con render(). Si ya hay una consulta en curso no hace
// nada, salvo con force (después de una acción, para que se vea el efecto enseguida).
async function poll(force) {
  if (polling && !force) return;
  polling = true;
  try { render(await api("/api/estado")); }
  catch (e) { setB("pill", "Sin conexión"); clsB("pill", "Desconectado", true); }
  finally { polling = false; }
}
// render(st): vuelca el estado de la máquina en toda la pantalla (DRO, herramienta, tiempos,
// overrides, LEDs, plasma, RESET, programa, mensajes). Se llama en cada consulta (cada 150 ms), por
// eso todo pasa por setB/clsB/led, que solo tocan el DOM si algo cambió.
// st: respuesta de /api/estado (ver CncSession.state en hope/cnc/session.py).
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
  // Perfiles elegidos desde otra ventana (o tablet): rearmar la barra, la máquina y el programa.
  const sk = selectionKey(st.perfiles && st.perfiles.seleccion);
  if (S.selKey === undefined) S.selKey = sk;          // primera consulta: el arranque ya los cargó
  else if (sk !== S.selKey) { S.selKey = sk; applySelection(); }
  // Primera consulta: el programa que ya estaba lo trae el arranque (loadProgram); solo se anota.
  if (S.sha === undefined) S.sha = p ? p.sha256 : null;
  if (p) {
    if (p.sha256 !== S.sha) {
      // Programa nuevo cargado desde otra ventana (o tablet): traer el listado y la trayectoria.
      // Si lo cargó esta ventana, afterLoad() ya puso S.sha y no se pide dos veces.
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
// startPolling(): (re)arranca la consulta periódica del estado. Cada 150 ms (≈7 por segundo: el DRO se
// ve fluido) o cada 400 ms en modo liviano, para dejarle CPU a una PC vieja. Se llama de nuevo al
// cambiar el modo liviano.
let pollTimer = null;
function startPolling() { clearInterval(pollTimer); pollTimer = setInterval(() => poll(), window.TISMA_LITE ? 400 : 150); }
startPolling();

// =================================================================================================
// ====================================== BOTONES Y CONTROLES ======================================
// =================================================================================================

// --- acciones (delegación: los botones están repetidos en varias pantallas) --------------------------
// ACTIONS: qué hace cada botón [data-act]. La mayoría manda la acción del mismo nombre al servidor;
// algunas son solo de la pantalla (coord, vista, seguir, jog_*). Cada función recibe el botón (b)
// para leer sus data-*. Equivalencias con Mach3: ciclo = Cycle Start, pausa = Feed Hold, parar = Stop,
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
  ayuda_g: () => helpCodes("G"), ayuda_m: () => helpCodes("M"),
  historial: () => setTab("diag"),
  perfiles_editar: () => setTab("profiles"),
  perfil_usar: () => S.prof && chooseProfile(S.prof.tipo, S.prof.nombre),
  perfil_guardar: () => saveProfile(false), perfil_guardar_como: () => saveProfile(true),
  perfil_exportar: exportProfile, perfil_borrar: deleteProfile,
  perfil_importar: () => $("prof-file").click(),
  borrar_estado: () => { S.clearedAt = Date.now() / 1000; setB("status", ""); },
};
// Un solo manejador de clics para toda la página («delegación»): los botones se repiten en varias
// pantallas y algunos se crean después (DRO, orígenes), y así todos funcionan sin registrar cada uno.
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
// setNext(desdeAqui): Set Next Line (false) o Run From Here (true) con la línea escrita en la casilla.
// Run From Here, además, lleva la máquina al estado modal de esa línea antes de seguir.
function setNext(fromHere) {
  const n = parseInt($("line-in").value, 10) || 0;
  act("linea_siguiente", { linea: n, desde_aqui: fromHere });
}

// --- jog ---------------------------------------------------------------------------------------
// renderJog(): muestra el modo (continuo/por pasos), el paso elegido y el % de velocidad del jog.
function renderJog() {
  led("led-jog-cont", !S.jogStep); led("led-jog-step", S.jogStep);
  setB("jog-step", S.steps[S.stepIdx].toFixed(4)); setB("jog-pct", S.jogPct);
}
// jogStart(eje, sentido, rápido): empieza a mover un eje. sentido: +1 o −1; rápido: con Shift, a la
// velocidad máxima (como Mach3). En modo por pasos mueve S.steps[S.stepIdx] mm y para solo.
function jogStart(axis, dir, fast) {
  if (!S.jogOn) return toast("Jog deshabilitado (Ctrl-Alt-J)");
  if (!S.axes.includes(axis)) return;
  act("jog", { eje: axis, dir, paso: S.jogStep ? S.steps[S.stepIdx] : null, rapido: fast, porcentaje: S.jogPct });
}
// jogEnd(): se soltó la tecla o el botón: en jog continuo, parar ya (sin mensajes de error).
function jogEnd() { if (!S.jogStep) act("jog_parar", {}, true); }
// Botones de jog en pantalla: se mueven mientras están apretados (onPress de compat.js funciona con
// ratón, táctil o pointer events, según lo que tenga el navegador).
document.querySelectorAll("[data-jog]").forEach(b => {
  const [a, d] = b.dataset.jog.split(":");
  const up = () => { if (b.classList.contains("act")) { b.classList.remove("act"); jogEnd(); } };
  onPress(b, e => { b.classList.add("act"); jogStart(a, +d, e.shiftKey); }, up);
});

// --- MDI ---------------------------------------------------------------------------------------
// Posición en el historial de MDI al recorrerlo con ↑/↓ (−1 = línea nueva, vacía).
let mdiIdx = -1;
// Entrada MDI: Enter manda la línea (se borra si salió bien), ↑/↓ recorren el historial, Esc borra.
// stopPropagation: las teclas escritas acá no son atajos (la barra espaciadora no pausa).
$("mdi").addEventListener("keydown", async e => {
  if (e.altKey) return;                           // Alt-1…7, Alt-R, Alt-S siguen funcionando desde la entrada
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
// ========================== PANTALLAS DE DATOS: ORÍGENES Y CONFIGURACIÓN =========================
// =================================================================================================

// --- orígenes ----------------------------------------------------------------------------------
// Orígenes de trabajo (Fixtures de Mach3): G54 = P1 … G59 = P6.
const WCS = ["G54", "G55", "G56", "G57", "G58", "G59"];
// Número P (1–6) del origen activo, para G10 L2/L20 P…
const wcsP = () => WCS.indexOf((S.st && S.st.estado.wcs) || "G54") + 1 || 1;
// Última tabla de orígenes recibida.
let offsetsData = null;
// Python 3.4 (Windows XP) no conserva el orden de los diccionarios: ordenar acá, no confiar en el JSON
const AXIS_ORDER = "XYZABCUVW";
function axisKeys(o) { return Object.keys(o).sort((a, b) => AXIS_ORDER.indexOf(a) - AXIS_ORDER.indexOf(b)); }
// sortedEntries(obj, cmp): pares [clave, valor] ordenados por clave (con cmp si se da).
function sortedEntries(o, cmp) {
  return Object.keys(o).sort(cmp || ((a, b) => (a < b ? -1 : a > b ? 1 : 0))).map(k => [k, o[k]]);
}

// loadOffsets(): pantalla Offsets (Alt-5): tabla de orígenes G54–G59, editor del origen activo,
// botones para cambiar de origen y tabla de herramientas (largo H y diámetro).
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
// applyOffset(): guarda los valores editados en el origen activo con G10 L2 (valores de máquina).
// Acepta coma decimal («12,5»), como se escribe en castellano.
function applyOffset() {
  const vals = [...document.querySelectorAll("[data-off]")].map(i => `${i.dataset.off}${parseFloat(i.value.replace(",", ".")) || 0}`).join(" ");
  act("mdi", { linea: `G10 L2 P${wcsP()} ${vals}` }).then(r => { if (r.ok) { toast("Origen actualizado", "info"); loadOffsets(); } });
}

// --- configuración ------------------------------------------------------------------------------
// loadConfig(): pantalla Settings (Alt-6): ejes (pasos/mm, velocidad, aceleración, límites) y datos
// de la máquina; con plasma, también alturas y retardos. Solo lectura: se cambia desde Perfiles.
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
// applyMachine(cfg): adapta la pantalla al tipo de máquina: con plasma muestra la antorcha y oculta
// husillo y refrigerante; también pone la marca en el título y los datos de plasma.
// El botón de sensado (IHS) se deshabilita si la máquina no tiene sensor.
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
// ============================================ DIÁLOGOS ===========================================
// =================================================================================================

// --- diálogos ------------------------------------------------------------------------------------
// pendingId: solicitud del guardián que se está mostrando (para no abrirla dos veces).
// modalResolve: función que resuelve la promesa del diálogo abierto.
let pendingId = null, modalResolve = null;
// openModal(título, texto, informe, textoOk, opts): abre el diálogo y devuelve una promesa que da
// true (Aceptar) o false (Cancelar / Esc). opts: html (contenido extra, ya escapado), guardian
// (muestra la nota del guardián), noCancel (solo un botón), focus (id del campo a enfocar).
// Hay un solo diálogo en la página: abrir otro reemplaza el contenido del anterior.
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
// closeModal(ok): cierra el diálogo y entrega la respuesta a quien lo abrió.
function closeModal(ok) { $("modal").hidden = true; if (modalResolve) { const r = modalResolve; modalResolve = null; r(ok); } }
$("modal-ok").onclick = () => closeModal(true);
$("modal-cancel").onclick = () => closeModal(false);
// showApproval(p): pide al operador aprobar lo que retuvo el guardián: habilitar la máquina (RESET,
// tipo "manual") o correr un programa (con su informe de revisión). Contesta al servidor con
// «aprobar» o «cancelar». p: {id, tipo, resumen, informe} que manda el servidor.
async function showApproval(p) {
  if (pendingId === p.id) return;
  pendingId = p.id;
  const ok = await openModal(p.tipo === "manual" ? "Habilitar la máquina" : "Aprobar el programa", p.resumen,
    p.informe ? (S.st && S.st.programa ? S.st.programa.informe : "") : "Jog, MDI, referenciado y ceros quedan habilitados hasta pulsar RESET.",
    "Aprobar y seguir", { guardian: true });
  await act(ok ? "aprobar" : "cancelar", ok ? { id: p.id } : {});
  pendingId = null;
}
// editProgram(): Edit G-Code: edita el programa cargado en un cuadro de texto y lo vuelve a cargar
// (el servidor lo revisa otra vez, como un archivo nuevo).
async function editProgram() {
  if (!S.lines.length) return toast("no hay programa cargado");
  const name = (S.st && S.st.programa && S.st.programa.nombre) || "programa.nc";
  const ok = await openModal("Editar G-code", name, "", "Guardar y revisar",
    { html: `<textarea id="editor" spellcheck="false">${esc(S.lines.join("\n"))}</textarea>`, focus: "editor" });
  if (!ok) return;
  const r = await act("cargar", { nombre: name, texto: $("editor").value });
  if (r.ok) afterLoad(r, name);
}
// showRecent(): Recent File: lista de los últimos programas; al elegir uno se carga por su sha256.
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
// Ayuda rápida de los códigos G y M que entiende TISMA CNC (botones «G-codes» y «M-codes», como Mach3).
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
// helpCodes("G" o "M"): muestra la tabla de ayuda y lo que no se soporta.
function helpCodes(kind) {
  const html = '<table class="table">' + HELP[kind].map(([c, d]) => `<tr><td>${c}</td><td style="text-align:left">${d}</td></tr>`).join("") + "</table>" +
    '<p class="muted small">No soportados (dan error claro): parámetros #, subprogramas M98/M99, G41/G42, G93, G12/G13, G15/G16, G50/G51, G52.</p>';
  openModal(kind === "G" ? "G-codes soportados" : "M-codes soportados", "", "", "Cerrar", { html, noCancel: true });
}

// =================================================================================================
// ====================================== PERFILES TIPO SLICER =====================================
// =================================================================================================

// Como en PrusaSlicer/OrcaSlicer: la máquina efectiva sale de cuatro perfiles (máquina, material,
// consumible y proceso). Arriba, en la barra, se eligen (#presets); en la pantalla Perfiles (Alt-3)
// se editan. Los del sistema no se pisan: «Guardar como…» crea uno propio que «hereda» de él y
// guarda solo lo cambiado (propios).

// --- perfiles tipo slicer (TISMA Engine / PrusaSlicer / OrcaSlicer) ----------------------------------------
// Tipos de perfil, en el orden de los selectores.
const PKINDS = ["maquina", "material", "consumible", "proceso"];
S.pk = "maquina"; S.prof = null; S.presets = null;
// loadPresets(): llena los selectores de perfiles (del sistema / míos), el espesor (plasma) y el
// resumen de la combinación elegida, con sus avisos.
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
// chooseProfile(tipo, nombre, espesor): elige un perfil (o un espesor de chapa) y rearma todo lo que
// depende de la máquina: configuración, DRO, selectores, programa (se vuelve a revisar con la nueva
// máquina). Sin argumentos, vuelve a aplicar la selección actual (después de guardar un perfil en uso).
async function chooseProfile(kind, name, espesor) {
  const params = espesor != null ? { espesor } : kind ? { tipo: kind, nombre: name } : {};
  const r = await act("perfil_elegir", params);
  if (!r.ok) { loadPresets(); return; }
  S.selKey = selectionKey(r.seleccion);     // ya lo aplicamos acá: que poll() no lo repita
  await applySelection();
  toast("Perfil aplicado: " + (r.resumen ? r.resumen.texto : ""), "info");
}
// applySelection(): rearma todo lo que depende de la máquina y los perfiles elegidos (configuración,
// DRO, selectores de la barra, programa revisado de nuevo, lista de la pantalla Perfiles).
async function applySelection() {
  S.cfg = await api("/api/config");
  applyMachine(S.cfg);
  buildDro(axisKeys(S.cfg.ejes));
  await loadPresets();
  await loadProgram();
  if (S.tab === "profiles") renderProfList();
}
// selectionKey(sel): texto que identifica una selección de perfiles, con las claves ordenadas (el
// orden de un JSON que arma Python 3.4 no es fijo). Sirve para notar un cambio hecho en otra ventana.
function selectionKey(sel) {
  return sel ? Object.keys(sel).sort().map(k => k + "=" + sel[k]).join("|") : "";
}
document.querySelectorAll("#presets select[data-kind]").forEach(sel => sel.addEventListener("change", () => chooseProfile(sel.dataset.kind, sel.value)));
$("espesor").addEventListener("change", () => chooseProfile(null, null, parseFloat($("espesor").value)));

// editor
document.querySelectorAll("#prof-kinds button").forEach(b => b.onclick = () => { S.pk = b.dataset.pk; renderProfList(); openFirst(); });
// openProfileTab(): al entrar a la pantalla Perfiles.
async function openProfileTab() { await loadPresets(); renderProfList(); if (!S.prof || S.prof.tipo !== S.pk) openFirst(); }
// openFirst(): abre el perfil en uso del tipo elegido (o el primero de la lista).
function openFirst() {
  const cur = S.presets && S.presets.seleccion[S.pk];
  const list = (S.presets && S.presets.tipos[S.pk]) || [];
  const name = cur || (list[0] && list[0].nombre);
  if (name) openProfile(S.pk, name);
}
// renderProfList(): lista de perfiles del tipo elegido, separados en «Del sistema» y «Míos».
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
// openProfile(tipo, nombre): trae un perfil (datos efectivos, del padre, crudos y el esquema de
// campos) y lo muestra en el formulario.
async function openProfile(kind, name) {
  const r = await fetch(`/api/perfil?tipo=${encodeURIComponent(kind)}&nombre=${encodeURIComponent(name)}`, { cache: "no-store" }).then(x => x.json());
  if (r.ok === false) return toast(r.mensaje);
  S.prof = { tipo: kind, nombre: name, origen: r.origen, hereda: r.hereda, datos: r.datos, padre: r.padre, crudo: r.crudo,
             esquema: r.esquema, propios: new Set(r.propios), sucio: false };
  renderProfList();
  renderProfForm();
}
// getP(obj, "a.b.c") / setP(obj, "a.b.c", v): leer y escribir por clave con puntos (las claves del
// esquema son así). setP crea los objetos intermedios que falten.
const getP = (o, k) => k.split(".").reduce((a, p) => (a && typeof a === "object" ? a[p] : undefined), o);
function setP(o, k, v) { const ps = k.split("."); let c = o; for (const p of ps.slice(0, -1)) { if (typeof c[p] !== "object" || c[p] === null) c[p] = {}; c = c[p]; } c[ps[ps.length - 1]] = v; }
// fieldApplies(campo, datos): ¿el campo vale para este tipo de máquina? (p. ej. los de plasma no se
// muestran en una fresadora). En material/consumible/proceso se mira la lista «tipos» del perfil.
function fieldApplies(f, d) {
  if (!f.tipos) return true;
  const kinds = S.prof.tipo === "maquina" ? [d.tipo] : (d.tipos || []);
  return f.tipos.some(t => kinds.indexOf(t) >= 0);
}
// isOwn(clave): ¿el valor es propio de este perfil (o de algo adentro de esa clave)? Si no, se hereda
// del padre y se muestra atenuado.
function isOwn(key) {
  if (!S.prof.hereda) return true;
  for (const k of S.prof.propios) if (k === key || k.indexOf(key + ".") === 0) return true;
  return false;
}
// renderProfForm(): arma el formulario del perfil abierto, agrupado como dice el esquema. Los del
// sistema no se pueden guardar ni borrar (solo «Guardar como…»).
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
// fieldHtml(campo, valor): HTML de un campo según su tipo (bool, opcion, multiple, tabla, num, texto),
// con unidad, ayuda y el botón ↺ para volver al valor heredado.
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
// ¿El perfil abierto hereda de otro?
const P_hasParent = () => !!(S.prof && S.prof.hereda);
// tableHtml(campo, filas, propio, botónRevertir): campo tipo tabla (p. ej. cortes por espesor), con
// filas editables, botón ✕ para borrar una y «+ fila» (copia la última).
function tableHtml(f, rows, own, rev) {
  const cols = f.columnas;
  const head = "<tr>" + cols.map(c => `<th>${esc(c.etiqueta)}${c.unidad ? ` <span class="muted">(${esc(c.unidad)})</span>` : ""}</th>`).join("") + "<th></th></tr>";
  const body = rows.map((r, i) => "<tr>" + cols.map(c => `<td><input data-tk="${f.clave}" data-row="${i}" data-col="${c.clave}" data-tt="${c.texto ? "texto" : "num"}" value="${r[c.clave] == null ? "" : esc(r[c.clave])}"></td>`).join("") +
    `<td><button class="del" data-tdel="${f.clave}" data-row="${i}" title="Borrar fila">✕</button></td></tr>`).join("");
  return `<div class="prof-field ${P_hasParent() ? (own ? "own" : "inh") : ""}"><label>${esc(f.etiqueta)}</label><span></span><span></span>${rev}<span class="help">${esc(f.ayuda || "")}</span>
    <div class="ptable-wrap"><table class="ptable ${P_hasParent() && !own ? "inh" : ""}">${head}${body}</table>
    <button class="mbtn sm" data-tadd="${f.clave}">+ fila</button></div></div>`;
}
// num(v): número con coma o punto decimal; null si no es un número.
const num = v => { const x = parseFloat(String(v).replace(",", ".")); return isFinite(x) ? x : null; };
// markOwn(clave): el valor pasa a ser propio y el perfil queda «(modificado)».
function markOwn(key) { S.prof.propios.add(key); S.prof.sucio = true; $("prof-name").textContent = S.prof.nombre + " (modificado)"; }
// Un campo cambió: se guarda en S.prof.datos (todavía no en el servidor: eso es «Guardar»).
// Si cambió el tipo de máquina, se rearma el formulario porque cambian los campos que aplican.
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
// Botones dentro del formulario: ↺ (volver al valor del padre), + fila y ✕ en las tablas.
// JSON.parse(JSON.stringify(x)): copia profunda, compatible con navegadores viejos.
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
// saveProfile(comoNuevo): guarda el perfil. Uno del sistema siempre se guarda como nuevo, que hereda de
// él. Si el perfil guardado está en uso, se vuelve a aplicar para que la máquina tome los cambios.
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
// exportProfile(): descarga el perfil como .json (para pasarlo a otra PC; se importa con «Importar»).
function exportProfile() {
  const P = S.prof;
  if (!P) return;
  const blob = new Blob([JSON.stringify(P.crudo, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = P.nombre.replace(/[^\w\-]+/g, "_") + ".json";
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}
// deleteProfile(): borra un perfil propio, con confirmación.
async function deleteProfile() {
  const P = S.prof;
  if (!P || P.origen === "sistema") return;
  const ok = await openModal("Borrar perfil", `¿Borrar «${P.nombre}»?`, "", "Borrar");
  if (!ok) return;
  const r = await act("perfil_borrar", { tipo: P.tipo, nombre: P.nombre });
  if (r.ok) { S.prof = null; await loadPresets(); renderProfList(); openFirst(); }
}
// Importar un perfil .json del tipo elegido.
$("prof-file").onchange = async e => {
  const f = e.target.files[0]; e.target.value = "";
  if (!f) return;
  const r = await act("perfil_importar", { tipo: S.pk, texto: await readText(f) });
  if (r.ok) { toast(`Importado: ${r.perfil.nombre}`, "info"); await loadPresets(); openProfile(S.pk, r.perfil.nombre); }
};

// =================================================================================================
// ======================================== TECLADO E INICIO =======================================
// =================================================================================================

// --- teclado: atajos de Mach3 ---------------------------------------------------------------------
// Teclas de jog de Mach3: flechas = X/Y, RePág/AvPág = Z. Valor: [eje, sentido].
const JOGKEYS = { ArrowLeft: ["X", -1], ArrowRight: ["X", 1], ArrowUp: ["Y", 1], ArrowDown: ["Y", -1], PageUp: ["Z", 1], PageDown: ["Z", -1] };
// Atajos de teclado de Mach3 (con el foco fuera de un campo de texto):
//   Alt-1…7 pantallas · Alt-R Cycle Start · Alt-S Stop · Esc Stop · Espacio Feed Hold
//   Alt-N bloque a bloque (Single BLK) · Ctrl-W Rewind · Ctrl-O cargar · Ctrl-F refrigerante
//   F5 husillo · Ctrl-Alt-J jog sí/no · Tab muestra u oculta el panel de jog · flechas/RePág/AvPág jog
// Las pantallas (Alt-número) cambian incluso escribiendo en un campo, salvo con un diálogo abierto.
// Con un diálogo abierto solo vale Esc (cancelar).
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
// Al soltar una tecla de jog se para; y si la ventana pierde el foco con una tecla apretada (Alt-Tab),
// también: si no, el eje seguiría moviéndose porque el keyup nunca llega.
document.addEventListener("keyup", e => { if (JOGKEYS[normKey(e)] && S.jogKeys.has(normKey(e))) { S.jogKeys.delete(normKey(e)); jogEnd(); } });
window.addEventListener("blur", () => { if (S.jogKeys.size) { S.jogKeys.clear(); jogEnd(); } });

// --- inicio -----------------------------------------------------------------------------------------
// Casilla de modo liviano: cambia el intervalo de consulta y redibuja las vistas con el nuevo modo.
$("lite").checked = !!window.TISMA_LITE;
$("lite").addEventListener("change", () => { setLite($("lite").checked); startPolling(); for (const v of Object.values(views)) { v.dirty = true; v.draw(); } });

// Arranque: configuración → pantalla según la máquina → DRO → perfiles → controles → programa →
// primera consulta de estado. En orden, porque cada paso usa lo que cargó el anterior.
(async () => {
  S.cfg = await api("/api/config");
  applyMachine(S.cfg);
  buildDro(axisKeys(S.cfg.ejes));
  await loadPresets();
  renderJog(); led("led-seguir", S.follow); led("led-jog", S.jogOn); setB("vista", VIEW_NAMES[VIEWS[S.viewIdx]]);
  await loadProgram();
  poll(true);
})();
