"use strict";
// Interfaz de Hope: consulta /api/estado cada 2 s y manda acciones de una lista fija a /api/accion.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 2) => (v === null || v === undefined || isNaN(v)) ? "—" : Number(v).toLocaleString("es-AR", { maximumFractionDigits: d, minimumFractionDigits: 0 });
const SERIE = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7"];
const ESTADOS = {
  inactiva: "sin iniciar", en_curso: "en curso", pausando: "pausando…", pausada: "en pausa", hecha: "terminada", hecho: "hecho",
  error: "error", detenido: "detenido", pausado: "en pausa", esperando_aprobacion: "espera aprobación",
  reintentando: "reintentando", terminado: "terminado", pendientes: "pendientes", al_dia: "al día",
  listo: "listo", falta: "falta algo",
};
let S = null;            // último estado
let formListo = false;   // los formularios se llenan una vez (y después de guardar)

function guardarLocal(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* sin almacenamiento */ } }
function leerLocal(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }

function aviso(texto, mal = false) {
  const a = $("#aviso");
  a.textContent = texto;
  a.className = "aviso" + (mal ? " mal" : "");
  clearTimeout(aviso.t);
  aviso.t = setTimeout(() => a.classList.add("oculto"), mal ? 9000 : 4000);
}

async function api(accion, params = {}) {
  const r = await fetch("/api/accion", {
    method: "POST", headers: { "Content-Type": "application/json", "X-Hope": "1" },
    body: JSON.stringify({ accion, params }),
  });
  let j = {};
  try { j = await r.json(); } catch (e) { j = { ok: false, mensaje: "respuesta inválida" }; }
  if (r.status === 401) { mostrarLogin(true); return j; }
  if (!r.ok || j.ok === false) aviso(j.mensaje || "no se pudo", true);
  else if (j.mensaje) aviso(j.mensaje);
  setTimeout(actualizar, 300);
  return j;
}

// --- navegación ------------------------------------------------------------
function mostrar(paso) {
  $$("nav button").forEach((b) => b.classList.toggle("activo", b.dataset.paso === paso));
  $$(".panel").forEach((p) => p.classList.toggle("activo", p.id === "p-" + paso));
  guardarLocal("hope-paso", paso);
}
$$("nav button").forEach((b) => b.addEventListener("click", () => mostrar(b.dataset.paso)));
mostrar(leerLocal("hope-paso") || "ajustes");

// --- formularios ---------------------------------------------------------------
function valorDe(obj, clave) { return clave.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj); }

function llenarFormularios(cfg) {
  $$("[data-campo]").forEach((el) => {
    const v = valorDe(cfg, el.dataset.campo);
    if (el.type === "checkbox") el.checked = !!v;
    else if (el.tagName === "TEXTAREA") el.value = Array.isArray(v) ? v.join("\n") : (v ?? "");
    else el.value = v ?? "";
  });
  $$("[data-reparto]").forEach((el) => { el.value = (cfg.datos.reparto || {})[el.dataset.reparto] ?? 0; });
  $("#usuario").value = cfg.usuario || "";
  formListo = true;
}

function leerCampo(el) {
  if (el.type === "checkbox") return el.checked;
  if (el.tagName === "TEXTAREA") return el.value.split("\n").map((x) => x.trim()).filter(Boolean);
  return el.value;
}

async function guardarGrupo(grupo) {
  const cambios = {};
  $$(`[data-campo^="${grupo}."]`).forEach((el) => { if (!el.hasAttribute("data-auto")) cambios[el.dataset.campo] = leerCampo(el); });
  if (grupo === "datos") {
    const rep = {};
    $$("[data-reparto]").forEach((el) => { rep[el.dataset.reparto] = parseFloat(el.value || "0"); });
    cambios["datos.reparto"] = rep;
  }
  const r = await api("guardar_config", { cambios });
  if (r.ok) { aviso("guardado"); formListo = false; }
}
$$("[data-guardar-grupo]").forEach((b) => b.addEventListener("click", () => guardarGrupo(b.dataset.guardarGrupo)));
$$("[data-auto]").forEach((el) => el.addEventListener("change", () => api("guardar_config", { cambios: { [el.dataset.campo]: leerCampo(el) } })));
$("#usuario").addEventListener("change", (e) => api("guardar_config", { cambios: { usuario: e.target.value } }));
$('[data-guardar="plan"]').addEventListener("click", async () => {
  const r = await api("guardar_config", { cambios: { plan: $("#plan").value } });
  if (r.ok) aviso("plan elegido: " + $("#plan").value);
});

// --- botones de acción ------------------------------------------------------------
$$("[data-accion]").forEach((b) => b.addEventListener("click", () => {
  if (b.dataset.confirmar && !confirm(b.dataset.confirmar)) return;
  let params = {};
  if (b.dataset.params) params = JSON.parse(b.dataset.params);
  if (b.dataset.params && b.dataset.confirmar) params.confirmar = true;
  if (b.dataset.accion === "maestro_probar") {
    $("#mae-prueba").textContent = "probando…";
    api("maestro_probar").then((r) => {
      $("#mae-prueba").innerHTML = (r.ok ? '<span class="ok">✔ ' : '<span class="mal">✘ ') + esc(r.mensaje || "") + "</span>" +
        (r.aviso ? ' <span class="av">' + esc(r.aviso) + "</span>" : "") +
        (r.respuesta ? " · respuesta: <code>" + esc(r.respuesta) + "</code>" : "");
    });
    return;
  }
  api(b.dataset.accion, params);
}));
$("#cua-iniciar").addEventListener("click", () => api("cuantizar_iniciar", { bits: parseInt($('input[name="cua-bits"]:checked').value, 10) }));
$("#mig-exportar").addEventListener("click", () => api("exportar_iniciar", { destino: $("#mig-destino").value }));
$("#mig-importar").addEventListener("click", () => api("importar_iniciar", { zip: $("#mig-zip").value }));
$("#gua-valor").addEventListener("input", () => { $("#gua-frase").textContent = "AUTORIZO " + humano($("#gua-valor").value); });
$("#gua-techo").addEventListener("click", async () => {
  const valor = $("#gua-valor").value.trim();
  if (!valor) return aviso("escribí el nuevo techo", true);
  if (!confirm(`¿Cambiar el techo de parámetros a ${valor}? Queda en la auditoría con tu nombre.`)) return;
  const r = await api("guardian_techo", { valor, confirmacion: $("#gua-confirmacion").value, modo: $("#gua-modo").value || null });
  $("#gua-resultado").innerHTML = r.ok
    ? '<span class="ok">' + esc(r.mensaje) + "</span>" + (r.avisos || []).map((a) => '<div class="av">aviso: ' + esc(a) + "</div>").join("")
    : '<span class="mal">' + esc(r.mensaje) + "</span>";
  if (r.ok) $("#gua-confirmacion").value = "";
});

function humano(texto) {   // aproximación de guardian.cli.human para mostrar la frase de confirmación
  const t = (texto || "").trim().toLowerCase().replace(/_/g, "");
  const m = t.match(/^([\d.,]+)\s*(k|m|b|g|t|millones|mil millones)?$/);
  if (!m) return "…";
  const mult = { k: 1e3, m: 1e6, millones: 1e6, b: 1e9, g: 1e9, "mil millones": 1e9, t: 1e12 }[m[2]] || 1;
  const n = m[2] ? parseFloat(m[1].replace(",", ".")) * mult : parseInt(m[1].replace(/[.,]/g, ""), 10);
  for (const [s, u] of [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "k"]]) {
    if (n >= s) return (n / s).toFixed(2).replace(/0+$/, "").replace(/\.$/, "") + u;
  }
  return String(Math.round(n));
}

// --- utilidades de render ------------------------------------------------------------
function chip(id, estado, hecho) {
  const el = $("#chip-" + id);
  if (!el) return;
  // Un paso está «hecho» cuando cumplió su objetivo; una tarea «terminada» puede no alcanzar (p. ej. el maestro).
  const activo = ["en_curso", "pausando", "reintentando", "error", "esperando_aprobacion", "pendientes"].includes(estado);
  const e = hecho && !activo && id !== "guardian" ? "hecho" : estado;
  el.textContent = ESTADOS[e] || e;
  el.className = "chip " + (e === "hecho" ? "hecha" : e === "hecha" ? "" : e);
  el.closest("button").classList.toggle("hecho", !!hecho);
}
function barra(id, p) { $(id).style.width = Math.round(Math.max(0, Math.min(1, p || 0)) * 100) + "%"; }
function tarjeta(k, v, s = "") { return `<div class="tarjeta"><div class="k">${esc(k)}</div><div class="v">${v}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`; }
function logEn(id, lineas) {
  const el = $(id);
  const abajo = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
  el.textContent = (lineas || []).join("\n");
  if (abajo) el.scrollTop = el.scrollHeight;
}
function tarea(prefijo, t) {
  $(`#${prefijo}-estado`).textContent = ESTADOS[t.estado] || t.estado;
  barra(`#${prefijo}-barra`, t.progreso);
  $(`#${prefijo}-detalle`).innerHTML = esc(t.detalle) + (t.error ? ` <span class="mal">${esc(t.error)}</span>` : "");
}
function horas(h) {
  if (h === null || h === undefined) return "—";
  if (h < 1) return Math.round(h * 60) + " min";
  if (h < 48) return num(h, 1) + " h";
  return num(h / 24, 1) + " días";
}

// --- 1 · calibración ------------------------------------------------------------------
function renderCalibracion(s) {
  const si = s.sistema;
  const nv = si.nvidia && si.nvidia[0];
  const gpu = nv ? `${esc(nv.nombre)}` : (si.placas[0] ? esc(si.placas[0]) : "no detectada");
  $("#cal-sistema").innerHTML =
    tarjeta("Placa de video", gpu, nv ? `${num(nv.memoria_total_mb / 1024, 1)} GB · driver ${esc(nv.driver)}` : "") +
    tarjeta("Procesador", esc(si.cpu), `${si.nucleos} núcleos`) +
    tarjeta("Memoria RAM", num(si.ram_gb, 1) + " GB") +
    tarjeta("Disco (datos)", si.disco.libre_gb === null ? "—" : num(si.disco.libre_gb, 0) + " GB libres", esc(si.disco.ruta)) +
    tarjeta("Sistema", esc(si.sistema), "Python " + esc(si.python));
  tarea("cal", s.calibracion);
  logEn("#cal-log", s.calibracion.registro);
  const r = s.calibracion.resultado;
  let pr = "";
  if (r && r.torch) {
    (r.torch.problemas || []).forEach((p, i) => {
      pr += `<div class="problema"><b>${esc(p)}</b>${r.torch.comandos[i] ? `<div>Cerrá la aplicación y, con el entorno activado, corré:</div><pre>${esc(r.torch.comandos[i])}</pre>` : ""}</div>`;
    });
  }
  $("#cal-problemas").innerHTML = pr;
  if (r && r.recomendacion) {
    const filas = r.recomendacion.map((x) => `<tr class="${x.preset === (r.lote || {}).preset ? "si" : ""}"><td>${esc(x.preset)}</td><td>${num(x.parametros_M, 0)}M</td>
      <td>${num(x.memoria_entrenamiento_GB, 1)} GB</td><td>${x.entra ? '<span class="ok">sí</span>' : "no"}</td>
      <td>${x.entra_con_offload ? '<span class="ok">sí</span>' : "no"}</td><td>${num(x.texto_GB_en_meses, 0)} GB</td>
      <td>${num(x.tokens_por_parametro, 0)}</td><td>${num(x.generacion_tokens_s_aprox, 0)}</td></tr>`).join("");
    const lote = r.lote ? (r.lote.lote ? `lote máximo de ${esc(r.lote.preset)}: <b>${r.lote.lote}</b> (secuencia ${r.lote.secuencia}${r.lote.memoria_pico_gb ? `, pico ${num(r.lote.memoria_pico_gb, 1)} GB` : ""})` : esc(r.lote.nota || r.lote.error || "no entra ni con lote 1")) : "lote: se busca al calibrar con GPU";
    $("#cal-resultado").innerHTML = `<div class="caja"><h2>Resultado <small>(${esc(r.fecha)})</small></h2>
      <div class="tarjetas">${tarjeta("Dispositivo", esc(r.dispositivo || "cpu"), esc(r.perfil.name))}
      ${tarjeta("Cómputo efectivo", num(r.perfil.tflops_eff, 2) + " TFLOPS", "≈ 40 % del matmul medido")}
      ${tarjeta("Precisión", esc(r.perfil.precision), r.perfil.bf16 ? "con bf16" : "sin bf16 (Pascal)")}
      ${tarjeta("Plan recomendado", esc(r.plan_recomendado), esc(r.motivo))}</div>
      <div class="detalle">${lote}</div>
      <table class="tabla"><tr><th>preset</th><th>parámetros</th><th>memoria</th><th>entra</th><th>con optimizador en RAM</th><th>texto en 2 meses</th><th>tokens/parámetro</th><th>generación tok/s</th></tr>${filas}</table>
      ${r.plan_recomendado && r.plan_recomendado !== s.config.plan ? `<div class="fila"><button class="prim" id="usar-rec">Usar ${esc(r.plan_recomendado)}</button></div>` : ""}</div>`;
    const b = $("#usar-rec");
    if (b) b.addEventListener("click", () => api("guardar_config", { cambios: { plan: r.plan_recomendado } }));
  } else $("#cal-resultado").innerHTML = "";
  const sel = $("#plan");
  if (sel.options.length !== Object.keys(s.planes).length || sel.dataset.v !== s.config.plan) {
    sel.innerHTML = Object.entries(s.planes).map(([k, v]) => `<option value="${esc(k)}">${esc(v)} — ${esc(k)}</option>`).join("");
    sel.value = s.config.plan; sel.dataset.v = s.config.plan;
  }
  const p = s.plan || {};
  $("#plan-resumen").innerHTML = p.etapas ? `salida <code>${esc(p.salida)}</code> · precisión ${esc(p.precision)} · ` +
    p.etapas.map((e) => `${esc(e.nombre)}${e.preset ? " (" + esc(e.preset) + ")" : ""}${e.horas ? " " + horas(e.horas) : ""}`).join(" → ") : esc(p.error || "");
}

// --- 2 · datos -------------------------------------------------------------------------
function renderDatos(s) {
  const d = s.datos;
  tarea("datos", d);
  logEn("#datos-log", d.registro);
  const filas = Object.entries(d.fuentes).map(([k, f]) => `<tr><td>${esc(k)}</td><td>${num(f.bytes_gb, 2)} / ${num(f.presupuesto_gb, 2)} GB</td>
    <td><div class="mini"><div style="width:${Math.round(f.progreso * 100)}%"></div></div></td>
    <td>${f.hecha ? '<span class="ok">lista</span>' : (f.bytes_gb > 0 ? "a medias" : "—")}</td>
    <td>${f.resumen ? `${num(f.resumen.docs, 0)} docs · ${num(f.resumen.descartados_reglas, 0)} descartados · ${num(f.resumen.duplicados, 0)} duplicados` : ""}</td></tr>`).join("");
  $("#datos-fuentes").innerHTML = `<tr><th>fuente</th><th>tamaño</th><th></th><th>estado</th><th>resumen</th></tr>` + filas;
  const di = s.sistema.disco;
  $("#datos-disco").textContent = di.libre_gb === null ? "" : `${num(di.libre_gb, 0)} GB libres en ${di.ruta}`;
}

// --- 3 · lenguaje --------------------------------------------------------------------------
function grafico(serie, leyenda = "#ent-leyenda", ancho = 760, alto = 280) {
  if (!serie.length) return '<div class="vacio">Todavía no hay mediciones: la primera llega con el primer checkpoint.</div>';
  const fuentes = [...new Set(serie.flatMap((p) => Object.keys(p.bpb)))];
  const W = ancho, H = alto, L = 48, R = 30, T = 12, B = 32;
  const xs = serie.map((p) => p.h);
  const ys = serie.flatMap((p) => Object.values(p.bpb)).filter((v) => isFinite(v));
  let x0 = Math.min(...xs), x1 = Math.max(...xs); if (x1 - x0 < 1e-6) { x1 = x0 + 1; }
  let y0 = Math.min(...ys), y1 = Math.max(...ys); const pad = (y1 - y0) * 0.08 || 0.1; y0 = Math.max(0, y0 - pad); y1 += pad;
  const X = (v) => L + (v - x0) / (x1 - x0) * (W - L - R), Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  let g = "";
  for (let i = 0; i <= 4; i++) {
    const v = y0 + (y1 - y0) * i / 4;
    g += `<line class="rejilla" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text class="etq" x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${num(v, 2)}</text>`;
  }
  const nx = W < 500 ? 3 : 5;
  for (let i = 0; i <= nx; i++) {
    const v = x0 + (x1 - x0) * i / nx;
    const etq = x1 - x0 < 0.5 ? num(v * 60, 1) + " min" : horas(v);
    g += `<text class="etq" x="${X(v)}" y="${H - 10}" text-anchor="middle">${etq}</text>`;
  }
  g += `<line class="eje" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/><line class="eje" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}"/>`;
  // separadores de etapa
  let prev = null;
  serie.forEach((p) => { if (prev && p.etapa !== prev) g += `<line class="rejilla" x1="${X(p.h)}" x2="${X(p.h)}" y1="${T}" y2="${H - B}"/>`; prev = p.etapa; });
  fuentes.forEach((f, i) => {
    const pts = serie.filter((p) => isFinite(p.bpb[f])).map((p) => `${X(p.h).toFixed(1)},${Y(p.bpb[f]).toFixed(1)}`);
    const color = `var(${SERIE[i % SERIE.length]})`;
    if (pts.length > 1) g += `<polyline class="linea" stroke="${color}" points="${pts.join(" ")}"/>`;
    const last = pts[pts.length - 1];
    if (last) { const [cx, cy] = last.split(","); g += `<circle cx="${cx}" cy="${cy}" r="3" fill="${color}"/>`; }
  });
  $(leyenda).innerHTML = fuentes.map((f, i) => {
    const ult = [...serie].reverse().find((p) => isFinite(p.bpb[f]));
    return `<span><i style="background:var(${SERIE[i % SERIE.length]})"></i>${esc(f)} ${ult ? num(ult.bpb[f], 3) : ""}</span>`;
  }).join("");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="BPB de validación por fuente">${g}</svg>`;
}

function renderLenguaje(s) {
  const e = s.entrenamiento;
  $("#ent-estado").textContent = (ESTADOS[e.estado] || e.estado) + (e.pid ? ` · pid ${e.pid}` : "");
  $("#ent-mensaje").innerHTML = [e.mensaje ? esc(e.mensaje) : "", e.plan ? `plan <code>${esc(e.plan)}</code>` : ""].filter(Boolean).join(" · ");
  $("#ent-etapas").innerHTML = e.etapas.map((x) => {
    const frac = x.horas_plan ? Math.min(x.horas / x.horas_plan, 1) : (x.hecha ? 1 : 0);
    return `<div class="etapa ${x.hecha ? "hecha" : ""} ${x.nombre === e.etapa_actual ? "actual" : ""}"><b>${esc(x.nombre)}</b>
      ${x.preset ? esc(x.preset) + " · " : ""}${x.crecer_a ? "crecer a " + esc(x.crecer_a) + " · " : ""}${x.horas_plan ? horas(x.horas) + " de " + horas(x.horas_plan) : ""}
      ${x.hecha ? ' · <span class="ok">hecha</span>' : ""}${x.solicitud && !x.aprobacion ? ' · <span class="av">espera aprobación</span>' : ""}
      <div class="mini"><div style="width:${Math.round(frac * 100)}%"></div></div></div>`;
  }).join("");
  const gpu = e.gpu;
  $("#ent-tarjetas").innerHTML =
    tarjeta("Etapa", esc(e.etapa_actual || "—"), e.eta_horas !== null ? "faltan " + horas(e.eta_horas) : "") +
    tarjeta("Horas entrenadas", horas(e.horas_totales)) +
    tarjeta("Velocidad", e.bytes_por_s ? num(e.bytes_por_s / 1000, 1) + " kB/s" : "—", e.lote ? "lote " + e.lote : "") +
    tarjeta("Paso", num(e.paso, 0)) +
    tarjeta("Memoria GPU", gpu ? `${num(gpu.memoria_usada_mb / 1024, 1)} / ${num(gpu.memoria_total_mb / 1024, 1)} GB` : "—", gpu ? `uso ${num(gpu.uso_pct, 0)} %` : "sin nvidia-smi") +
    tarjeta("Temperatura", gpu && gpu.temperatura_c !== null ? num(gpu.temperatura_c, 0) + " °C" : "—", gpu && gpu.temperatura_c > 83 ? '<span class="mal">alta: revisá la ventilación</span>' : "");
  $("#ent-grafico").innerHTML = grafico(e.serie || []);
  if (!(e.serie || []).length) $("#ent-leyenda").innerHTML = "";
  logEn("#ent-log", e.registro);
  const tm = e.tarea_muestras;
  $("#ent-muestra-estado").textContent = tm.estado === "en_curso" ? "generando… " + esc(tm.detalle) : (tm.error ? "error: " + tm.error : "");
  const m = e.muestras;
  $("#ent-muestras").innerHTML = m ? `<div class="detalle">${esc(m.fecha)}</div>` + m.muestras.map((x) =>
    `<div class="muestra"><small>${esc(x.idioma)} · ${num(x.segundos, 1)} s</small><pre><span class="prompt">${esc(x.prompt)}</span>${esc(x.texto)}</pre></div>`).join("")
    : '<div class="detalle">Sin muestras todavía.</div>';
}

// --- 4 · maestro -----------------------------------------------------------------------------
function renderMaestro(s) {
  const m = s.maestro;
  tarea("mae", m);
  logEn("#mae-log", m.registro);
  const c = m.conteos;
  $("#mae-conteos").innerHTML = tarjeta("Aceptados", `<span class="ok">${num(c.aceptados, 0)}</span>`, `${num(c.respaldados, 0)} verificados · ${num(c.inferidos, 0)} sin verificar`) +
    tarjeta("Rechazados", `<span class="mal">${num(c.rechazados, 0)}</span>`, "algún paso no pasó la verificación") +
    tarjeta("Inválidos", num(c.invalidos, 0), "respuesta sin el formato pedido") +
    tarjeta("Del maestro", num(c.del_maestro, 0), `${num(c.generados, 0)} de generadores`) +
    tarjeta("Fuente «maestro»", num((c.bytes_train || 0) / 1e6, 2) + " MB", "train.bin");
  const temas = Object.entries(c.por_tema || {});
  $("#mae-temas").innerHTML = temas.length ? "<tr><th>tema</th><th>aceptados</th><th>rechazados</th></tr>" +
    temas.map(([t, v]) => `<tr><td>${esc(t)}</td><td>${v.aceptados}</td><td>${v.rechazados}</td></tr>`).join("") : "";
  const et = m.etapa;
  $("#mae-etapa").innerHTML = et ? `${esc(et.nombre)}: ${et.hecha ? '<span class="ok">hecha</span>' : horas(et.horas) + " de " + horas(et.horas_plan)}` +
    (s.entrenamiento.etapa_actual === et.nombre ? " · <b>en curso</b>" : "") : "el plan elegido no tiene etapa del maestro";
}

// --- 5 · migrar --------------------------------------------------------------------------------
function renderMigrar(s) {
  const mg = s.migrar;
  ["exp", "imp"].forEach((k) => {
    const t = k === "exp" ? mg.exportar : mg.importar;
    $(`#mig-${k}-estado`).textContent = ESTADOS[t.estado] || t.estado;
    barra(`#mig-${k}-barra`, t.progreso);
    $(`#mig-${k}-detalle`).innerHTML = esc(t.detalle) + (t.error ? ` <span class="mal">${esc(t.error)}</span>` : "") +
      (t.datos && t.datos.zip ? ` · <code>${esc(t.datos.zip)}</code> (${num(t.datos.gb, 2)} GB)` : "");
  });
  logEn("#mig-log", [...mg.exportar.registro, ...mg.importar.registro]);
  const c = mg.continuacion;
  const box = $("#mig-cont");
  box.classList.toggle("oculto", !c.es_continuacion);
  if (c.es_continuacion) box.innerHTML = `<b>Esta PC continúa el entrenamiento de otra máquina.</b> Último checkpoint
    <code>${esc(c.ultimo_checkpoint || "")}</code>${c.horas !== undefined ? ` · ${horas(c.horas)} entrenadas` : ""} · plan
    <code>${esc(s.config.plan)}</code>${c.xpu ? "" : ' · <span class="av">calibrá en el paso 1 para confirmar que PyTorch ve la A770 (xpu)</span>'}`;
}

// --- 6 · guardián ------------------------------------------------------------------------------
function renderGuardian(s) {
  const g = s.guardian;
  if (g.error) { $("#gua-estado").innerHTML = `<div class="problema">${esc(g.error)}</div>`; return; }
  const e = g.estado;
  $("#gua-estado").innerHTML = tarjeta("Techo de parámetros", esc(g.techo_humano), esc(e.techo_de_parametros)) +
    tarjeta("Modo", esc(e.modo)) + tarjeta("Aprobación si el cambio supera", esc(e.cambio_importante_requiere_aprobacion_si_supera)) +
    tarjeta("Auditoría", g.auditoria_integra ? '<span class="ok">íntegra</span>' : `<span class="mal">alterada (${g.auditoria_alterada_en})</span>`);
  const pend = Object.entries(g.pendientes || {});
  const cont = $("#gua-pendientes");
  const firma = pend.map(([k]) => k).join(",");
  if (cont.dataset.firma === firma) return;
  cont.dataset.firma = firma;
  cont.innerHTML = pend.length ? pend.map(([id, r]) => `<div class="pend"><div class="fila"><b>${esc(id)}</b>
    <span class="detalle">${esc((r.action || {}).kind || "")}</span></div>
    <div class="detalle">${esc((r.reasons || []).join("; "))}</div>
    <pre>${esc(JSON.stringify(r.action || r, null, 2))}</pre>
    <div class="fila"><input placeholder="nota (opcional)" data-nota="${esc(id)}" class="largo">
    <button class="prim" data-aprobar="${esc(id)}">Aprobar</button><button class="peligro" data-denegar="${esc(id)}">Denegar</button></div></div>`).join("")
    : '<div class="detalle">No hay solicitudes pendientes.</div>';
  $$("[data-aprobar]", cont).forEach((b) => b.addEventListener("click", () => {
    if (confirm(`¿Aprobar ${b.dataset.aprobar}?`)) api("guardian_aprobar", { id: b.dataset.aprobar, nota: $(`[data-nota="${b.dataset.aprobar}"]`).value });
  }));
  $$("[data-denegar]", cont).forEach((b) => b.addEventListener("click", () => {
    if (confirm(`¿Denegar ${b.dataset.denegar}?`)) api("guardian_denegar", { id: b.dataset.denegar, nota: $(`[data-nota="${b.dataset.denegar}"]`).value });
  }));
}

// --- 7 · cuantizar ------------------------------------------------------------------------------
function renderCuantizar(s) {
  const c = s.cuantizar;
  tarea("cua", c);
  logEn("#cua-log", c.registro);
  $("#cua-modelo").innerHTML = c.modelo ? `Modelo: <code>${esc(c.modelo)}</code> (${num(c.mb_modelo, 1)} MB en bf16)`
    : "Todavía no hay modelo ligero: se guarda con el primer checkpoint del paso 3.";
  const firma = c.exportados.map((x) => x.archivo).join(",");
  const box = $("#cua-resultado");
  if (box.dataset.firma === firma) return;
  box.dataset.firma = firma;
  box.innerHTML = c.exportados.length ? `<div class="caja"><h2>Exportados</h2><table class="tabla">
    <tr><th>archivo</th><th>bits</th><th>tamaño</th><th>modelo ligero</th><th>pérdida original → cuantizado</th><th>fecha</th></tr>` +
    c.exportados.map((x) => `<tr><td><code>${esc(x.archivo)}</code></td><td>${x.bits}</td><td><b>${num(x.mb_archivo, 1)} MB</b></td>
      <td>${num(x.mb_ligero, 1)} MB</td><td>${num(x.perdida_original, 3)} → ${num(x.perdida_cuantizado, 3)}
      <span class="${x.perdida_cuantizado - x.perdida_original > 0.05 ? "av" : "ok"}">(${x.perdida_cuantizado >= x.perdida_original ? "+" : ""}${num(x.perdida_cuantizado - x.perdida_original, 3)})</span></td>
      <td>${esc(x.fecha)}</td></tr>`).join("") + `</table>` +
    (c.exportados[0].muestra ? `<div class="muestra"><small>muestra del último exportado</small><pre><span class="prompt">La ciudad de Buenos Aires</span>${esc(c.exportados[0].muestra)}</pre></div>` : "") +
    `<div class="detalle">Para usarlo desde Python: <code>from hope.app.cuantizar import cargar; m = cargar("archivo.pt")</code></div></div>` : "";
}

// --- 8 · datos diarios ---------------------------------------------------------------------------
function renderDiarios(s) {
  const d = s.diarios;
  if (!d) return;
  tarea("dia", d);
  logEn("#dia-log", d.registro);
  const inf = d.ultimo_informe ? d.ultimo_informe.fuentes || {} : {};
  const firma = JSON.stringify([d.fuentes.map((f) => f.fuente + f.activa), d.ultimo_informe && d.ultimo_informe.fecha]);
  const box = $("#dia-fuentes");
  if (box.dataset.firma === firma) return;
  box.dataset.firma = firma;
  box.innerHTML = `<div class="caja"><h2>Fuentes${d.cuota ? ` · cuota ${num(d.cuota.gb_dia, 2)} GB/día` : ""}</h2>
    <table class="tabla"><tr><th>fuente</th><th>método</th><th>licencia</th><th>activa</th><th>último día</th></tr>` +
    d.fuentes.map((f) => { const r = inf[f.fuente];
      return `<tr><td>${esc(f.fuente)}</td><td>${esc(f.metodo)}</td><td>${esc(f.licencia)}</td><td>${f.activa ? "sí" : "no"}</td>
      <td>${r ? `${r.guardados} guardados, ${r.solo_indice || 0} solo índice${r.aviso ? " · " + esc(r.aviso) : ""}` : "—"}</td></tr>`; }).join("") +
    `</table></div>`;
}

// --- ⚙ configuración ------------------------------------------------------------------------------
let AJ = null;            // estado de ajustes con el que se llenó el formulario
let ajInicial = null;     // cambios «vacíos» (para mandar solo las secciones tocadas)
const lineas = (t) => t.split("\n").map((x) => x.trim()).filter(Boolean);

function llenarAjustes(s) {
  const a = s.ajustes;
  if (!a || a.error) { $("#aj-error").textContent = a ? a.error : ""; $("#aj-error").classList.toggle("oculto", !a); return; }
  AJ = a;
  const sel = $("#aj-plan");
  sel.innerHTML = Object.entries(s.planes).map(([k, v]) => `<option value="${esc(k)}">${esc(v)} — ${esc(k)}</option>`).join("");
  sel.value = a.plan.nombre;
  $("#aj-secuencia").value = a.plan.secuencia ?? "";
  $("#aj-optimizador").innerHTML = a.optimizadores.map((o) => `<option>${esc(o)}</option>`).join("");
  $("#aj-optimizador").value = a.plan.optimizador;
  $("#aj-precision").innerHTML = a.precisiones.map((p) => `<option ${p.startsWith("bf16") && a.bf16 === false ? "disabled" : ""}>${esc(p)}</option>`).join("");
  $("#aj-precision").value = a.plan.precision;
  $("#aj-bf16").textContent = a.bf16 === false ? "bf16 bloqueado: esta GPU no lo soporta (Pascal)" :
    a.bf16 === true ? "la GPU soporta bf16" : "calibrá para saber si la GPU soporta bf16";
  $("#aj-etapas").innerHTML = a.plan.etapas.map((e) => e.crecer
    ? `<div class="etapa-aj"><b>${esc(e.nombre)}</b> · etapa de crecimiento (sin horas ni datos)</div>`
    : `<div class="etapa-aj" data-etapa="${esc(e.nombre)}"><div class="fila"><b>${esc(e.nombre)}</b>${e.preset ? " · " + esc(e.preset) : ""}
      <label>horas <input type="number" min="0.01" step="any" data-k="horas" value="${e.horas ?? ""}"></label>
      <label>lote máximo <input type="number" min="1" step="1" data-k="lote_max" value="${e.lote_max ?? ""}"></label>
      <span class="detalle">${e.horas ? "≈ " + num(e.horas / 24, 1) + " días" : ""}</span></div>
      <div class="pesos">${Object.entries(e.fuentes).map(([f, w]) => `<label title="${esc(e.rutas[f])}">${esc(f)}
        <input type="number" min="0" max="1" step="0.01" data-fuente="${esc(f)}" value="${w}"></label>`).join("")}
        <span class="suma"></span></div></div>`).join("");
  $$("#aj-etapas .etapa-aj").forEach((el) => { el.addEventListener("input", () => sumaPesos(el)); sumaPesos(el); });
  $("#aj-carpeta").value = s.config.datos.carpeta;
  $("#aj-endpoint").value = s.config.maestro.endpoint;
  $("#aj-modelo").value = s.config.maestro.model;
  const w = a.web;
  $("#aj-web-modo").innerHTML = a.modos_web.map((m) => `<option>${esc(m)}</option>`).join("");
  $("#aj-web-modo").value = w.modo;
  $("#aj-web-robots").checked = !!w.respetar_robots;
  $("#aj-web-permitidos").value = (w.sitios_permitidos || []).join("\n");
  $("#aj-web-bloqueados").value = (w.sitios_bloqueados || []).join("\n");
  const l = w.limites || {};
  $("#aj-web-cph").value = l.consultas_por_hora; $("#aj-web-ppd").value = l.paginas_por_dia; $("#aj-web-mbd").value = l.mb_por_dia;
  $("#aj-web-mbp").value = l.mb_por_pagina; $("#aj-web-spp").value = l.segundos_por_pedido;
  $("#aj-web-existe").textContent = a.web_existe ? "" : "(todavía no existe: se crea al aplicar)";
  const f = a.fuentes_web;
  $("#aj-fw-mb").value = f.cuota_diaria_mb; $("#aj-fw-pag").value = f.paginas_por_dia; $("#aj-fw-carpeta").value = f.carpeta;
  $("#aj-fw-desde").value = (f.horario || {}).desde ?? 0; $("#aj-fw-hasta").value = (f.horario || {}).hasta ?? 24;
  $("#aj-fw-existe").textContent = a.fuentes_web_existe ? "" : "(todavía no existe: se crea al aplicar)";
  $("#aj-techo").value = ""; $("#aj-techo-conf").value = ""; $("#aj-techo-modo").value = "";
  $("#aj-respaldos").textContent = a.respaldos.length ? `${a.respaldos.length} copia(s) en runs/respaldos (última: ${a.respaldos[0]})` : "";
  $("#aj-revision").innerHTML = "";
  ajInicial = JSON.stringify(cambiosAjustes(false));
}

function sumaPesos(el) {
  const t = $$("[data-fuente]", el).reduce((acc, i) => acc + (parseFloat(i.value) || 0), 0);
  const sp = $(".suma", el);
  const ok = Math.abs(t - 1) <= 1e-3;
  sp.innerHTML = `suma <b class="${ok ? "ok" : "mal"}">${num(t, 3)}</b>${ok ? "" : " (tiene que dar 1)"}`;
  $$("[data-fuente]", el).forEach((i) => i.classList.toggle("invalido", !ok));
}

function cambiosAjustes(soloTocados = true) {
  const etapas = {};
  $$("#aj-etapas [data-etapa]").forEach((el) => {
    const e = { fuentes: {} };
    $$("[data-k]", el).forEach((i) => { if (i.value !== "") e[i.dataset.k] = Number(i.value); });
    $$("[data-fuente]", el).forEach((i) => { e.fuentes[i.dataset.fuente] = Number(i.value); });
    etapas[el.dataset.etapa] = e;
  });
  const c = {
    plan: { secuencia: Number($("#aj-secuencia").value), optimizador: $("#aj-optimizador").value,
      precision: $("#aj-precision").value, etapas },
    app: { "datos.carpeta": $("#aj-carpeta").value, "maestro.endpoint": $("#aj-endpoint").value, "maestro.model": $("#aj-modelo").value },
    web: { modo: $("#aj-web-modo").value, respetar_robots: $("#aj-web-robots").checked,
      sitios_permitidos: lineas($("#aj-web-permitidos").value), sitios_bloqueados: lineas($("#aj-web-bloqueados").value),
      limites: { consultas_por_hora: Number($("#aj-web-cph").value), paginas_por_dia: Number($("#aj-web-ppd").value),
        mb_por_dia: Number($("#aj-web-mbd").value),
        mb_por_pagina: Number($("#aj-web-mbp").value), segundos_por_pedido: Number($("#aj-web-spp").value) } },
    fuentes_web: { cuota_diaria_mb: Number($("#aj-fw-mb").value), paginas_por_dia: Number($("#aj-fw-pag").value),
      carpeta: $("#aj-fw-carpeta").value, horario: { desde: Number($("#aj-fw-desde").value), hasta: Number($("#aj-fw-hasta").value) } },
  };
  if ($("#aj-techo").value.trim()) c.techo = { valor: $("#aj-techo").value.trim(), modo: $("#aj-techo-modo").value || null, confirmacion: $("#aj-techo-conf").value };
  if (soloTocados && ajInicial) {
    const ini = JSON.parse(ajInicial);
    for (const k of ["web", "fuentes_web", "app"]) if (JSON.stringify(ini[k]) === JSON.stringify(c[k])) delete c[k];
  }
  return c;
}

function pintarDiff(t) {
  return esc(t).split("\n").map((l) => l.startsWith("+") && !l.startsWith("+++") ? `<span class="mas">${l}</span>`
    : l.startsWith("-") && !l.startsWith("---") ? `<span class="menos">${l}</span>` : l.startsWith("@@") ? `<span class="arr">${l}</span>` : l).join("\n");
}

$("#aj-revisar").addEventListener("click", async () => {
  const cambios = cambiosAjustes();
  const r = await api("config_previsualizar", { cambios });
  const box = $("#aj-revision");
  let h = "";
  (r.errores || []).forEach((e) => { h += `<div class="mal">✘ ${esc(e)}</div>`; });
  (r.avisos || []).forEach((e) => { h += `<div class="av">⚠ ${esc(e)}</div>`; });
  if (r.ok && !(r.diffs || []).length) h += '<div class="detalle">No hay cambios.</div>';
  (r.diffs || []).forEach((d) => { h += `<h2>${esc(d.archivo)}${d.nuevo ? " <small>(archivo nuevo)</small>" : ""}</h2><pre class="diff">${pintarDiff(d.diff)}</pre>`; });
  if (r.frase_techo) $("#aj-techo-frase").textContent = r.frase_techo;
  if (r.ok && (r.diffs || []).length) h += `<div class="fila"><button class="prim" id="aj-aplicar">Aplicar estos cambios</button>
    <span class="detalle">Se guarda una copia de cada archivo en runs/respaldos.</span></div>`;
  box.innerHTML = h;
  const b = $("#aj-aplicar");
  if (b) b.addEventListener("click", async () => {
    if (!confirm("¿Aplicar los cambios mostrados?")) return;
    const r2 = await api("config_aplicar", { cambios, confirmar: true });
    if (r2.ok) {
      box.innerHTML = `<div class="ok">✔ ${esc(r2.mensaje)}</div>` + (r2.avisos || []).map((x) => `<div class="av">⚠ ${esc(x)}</div>`).join("");
      AJ = null;           // recargar el formulario con lo guardado
      formListo = false;
    } else box.innerHTML += (r2.errores || [r2.mensaje]).map((e) => `<div class="mal">✘ ${esc(e)}</div>`).join("");
  });
});
$("#aj-descartar").addEventListener("click", () => { AJ = null; actualizar(); });
$("#aj-cambiar-plan").addEventListener("click", async () => {
  const r = await api("guardar_config", { cambios: { plan: $("#aj-plan").value } });
  if (r.ok) { AJ = null; formListo = false; }
});
$("#aj-probar").addEventListener("click", async () => {
  $("#aj-prueba").textContent = "probando…";
  const r = await api("maestro_probar", { endpoint: $("#aj-endpoint").value, model: $("#aj-modelo").value });
  $("#aj-prueba").innerHTML = (r.ok ? '<span class="ok">✔ ' : '<span class="mal">✘ ') + esc(r.mensaje || "") + "</span>" +
    (r.aviso ? ' <span class="av">' + esc(r.aviso) + "</span>" : "");
});
$("#aj-techo").addEventListener("input", () => { $("#aj-techo-frase").textContent = "AUTORIZO " + humano($("#aj-techo").value); });
$("#aj-empezar").addEventListener("click", async () => {
  const r = await api("empezar_entrenar");
  if (r.requisitos) pintarRequisitos(r.requisitos);
  if (r.ok) mostrar("lenguaje");
});
$("#cerrar-app").addEventListener("click", async () => {
  if (!confirm("¿Cerrar la aplicación? El entrenamiento en curso sigue; para volver, abrí Hope de nuevo.")) return;
  await api("cerrar_app", { confirmar: true });
  document.body.innerHTML = '<main><h1>Hope se cerró</h1><p class="ayuda">El entrenamiento en curso sigue. Para volver, abrí Hope de nuevo.</p></main>';
});

function pintarRequisitos(req) {
  $("#aj-requisitos").innerHTML = req.map((r) => `<li>${r.ok ? '<span class="ok">✔</span>' : '<span class="mal">✘</span>'} <b>${esc(r.nombre)}</b> — ${esc(r.detalle)}</li>`).join("");
}

function renderAjustes(s) {
  if (!s.ajustes) return;
  if (!AJ || AJ.plan.nombre !== s.ajustes.plan.nombre) llenarAjustes(s);
  if (s.ajustes.listo) pintarRequisitos(s.ajustes.listo);
  const listo = (s.ajustes.listo || []).every((r) => r.ok);
  const corriendo = ["en_curso", "pausando", "reintentando"].includes(s.entrenamiento.estado);
  $("#aj-empezar").disabled = corriendo;
  $("#aj-empezar").textContent = corriendo ? "Entrenando…" : "▶ Empezar a entrenar";
  chip("ajustes", corriendo ? "en_curso" : (listo ? "listo" : "falta"), false);
}

// --- vista: inicio (simple) / más opciones -----------------------------------------------------
function vista(cual) {
  const avanzada = cual === "avanzado";
  $("#avanzado").classList.toggle("oculto", !avanzada);
  $("#inicio").classList.toggle("oculto", avanzada);
  $("#btn-vista").textContent = avanzada ? "← Inicio" : "Más opciones";
  guardarLocal("hope-vista", cual);
}
$("#btn-vista").addEventListener("click", () => vista($("#avanzado").classList.contains("oculto") ? "avanzado" : "inicio"));
vista(leerLocal("hope-vista") || "inicio");

function mostrarLogin(si) {
  $("#login").classList.toggle("oculto", !si);
  if (si) { $("#inicio").classList.add("oculto"); $("#avanzado").classList.add("oculto"); }
  else if ($("#inicio").classList.contains("oculto") && $("#avanzado").classList.contains("oculto")) vista(leerLocal("hope-vista") || "inicio");
}
$("#login-entrar").addEventListener("click", async () => {
  const r = await fetch("/api/login", { method: "POST", headers: { "Content-Type": "application/json", "X-Hope": "1" },
    body: JSON.stringify({ pin: $("#login-pin").value }) });
  const j = await r.json().catch(() => ({}));
  if (r.ok) { $("#login-pin").value = ""; mostrarLogin(false); actualizar(); }
  else $("#login-msg").textContent = j.mensaje || "no se pudo";
});

// --- inicio -------------------------------------------------------------------------------------
let discoTocado = false;
function renderInicio(s) {
  const e = s.entrenamiento, ini = s.inicio || {}, auto = (ini.automatico || {});
  $("#frase").textContent = ini.frase && typeof ini.frase === "string" ? ini.frase : "Hope";
  $("#estado-corto").textContent = ESTADOS[e.estado] || e.estado;
  const et = (e.etapas || []).find((x) => x.nombre === e.etapa_actual);
  const autoCorriendo = auto.estado === "en_curso" || auto.estado === "pausando";
  barra("#inicio-barra", autoCorriendo && auto.progreso < 1 && (!et || !et.horas) ? auto.progreso : (et && et.horas_plan ? et.horas / et.horas_plan : 0));
  const partes = [];
  if (autoCorriendo) partes.push(auto.detalle);
  if (auto.error) partes.push("⚠ " + auto.error);
  if (e.mensaje && !autoCorriendo) partes.push(e.mensaje);
  const h = ini.horario || {};
  if (h.activo) partes.push(`Horario: pausa a las ${String(h.pausa).padStart(2, "0")}:00, sigue a las ${String(h.seguir).padStart(2, "0")}:00`);
  $("#inicio-detalle").textContent = partes.filter(Boolean).join(" · ");
  const pasos = auto.pasos || [];
  const hechos = pasos.filter((p) => p.hecho).length;
  const mostrarPasos = autoCorriendo || (hechos > 0 && hechos < pasos.length);
  $("#inicio-pasos").classList.toggle("oculto", !mostrarPasos);
  $("#inicio-pasos").innerHTML = pasos.map((p) => `<li class="${p.hecho ? "hecho" : ""} ${p.actual ? "actual" : ""}">${p.hecho ? "✔" : p.actual ? "▶" : "○"} ${esc(p.nombre)}${p.info ? ` <small>(${esc(p.info)})</small>` : ""}</li>`).join("");
  const entrenando = ["en_curso", "pausando", "reintentando"].includes(e.estado);
  const empezo = hechos > 0 || (e.etapas || []).some((x) => x.horas > 0);
  $("#btn-pausar").classList.toggle("oculto", !(entrenando || autoCorriendo));
  $("#btn-pausar").disabled = e.estado === "pausando" || auto.estado === "pausando";
  $("#btn-continuar").classList.toggle("oculto", entrenando || autoCorriendo || !empezo || e.estado === "terminado");
  $("#btn-empezar").classList.toggle("oculto", entrenando || autoCorriendo || empezo);
  $("#inicio-grafico").innerHTML = grafico(e.serie || [], "#inicio-leyenda", 420, 230);
  if (!(e.serie || []).length) $("#inicio-leyenda").innerHTML = "";
  const m = e.muestras;
  $("#inicio-muestra").innerHTML = m && m.muestras && m.muestras.length
    ? `<div class="muestra"><pre><span class="prompt">${esc(m.muestras[0].prompt)}</span>${esc(m.muestras[0].texto)}</pre></div><small>${esc(m.fecha)}</small>`
    : "Todavía no hay muestras: la primera llega con el primer checkpoint.";
  // disco
  const d = s.discos || {};
  const card = $("#disco-card");
  card.classList.toggle("oculto", !!d.elegida || !d.propuesta);
  if (!d.elegida && d.propuesta && !discoTocado) {
    const p = d.propuesta;
    $("#disco-propuesta").innerHTML = p.carpeta ? `Voy a usar <b>${esc(p.carpeta)}</b> (${num(p.disco.libre_gb, 0)} GB libres, ${esc(p.disco.tipo)}). Ahí van los textos, los checkpoints, el entorno de Python y las cachés.` : "No encontré discos.";
    $("#disco-aviso").textContent = p.aviso || "";
    $("#disco-aviso").classList.toggle("oculto", !p.aviso);
    $("#disco-lista").innerHTML = (p.alternativas || []).map((x) => `<li><button class="link" data-disco="${esc(x.ruta)}">${esc(x.ruta)}</button> — ${num(x.libre_gb, 0)} GB libres · ${esc(x.tipo)}${x.sistema ? " · <b>disco del sistema</b>" : ""}</li>`).join("");
    $$("[data-disco]").forEach((b) => b.addEventListener("click", () => {
      const sep = b.dataset.disco.includes("\\") ? "\\" : "/";
      $("#disco-carpeta").value = b.dataset.disco.replace(/[\\/]+$/, "") + sep + "Hope";
    }));
  }
}
$("#btn-empezar").addEventListener("click", () => api("auto_iniciar"));
$("#btn-continuar").addEventListener("click", () => {
  const pasos = (((S || {}).inicio || {}).automatico || {}).pasos || [];
  api(pasos.every((p) => p.hecho) ? "entrenar_iniciar" : "auto_iniciar");
});
$("#btn-pausar").addEventListener("click", () => api("auto_pausar"));
$("#disco-usar").addEventListener("click", () => api("disco_usar", { confirmar: true }));
$("#disco-cambiar").addEventListener("click", () => { discoTocado = true; $("#disco-otro").classList.toggle("oculto"); });
$("#disco-usar-otro").addEventListener("click", () => {
  if (confirm(`¿Usar ${$("#disco-carpeta").value}?`)) api("disco_usar", { carpeta: $("#disco-carpeta").value, confirmar: true });
});

// --- chat de control ----------------------------------------------------------------------------
function mensaje(quien, html) {
  const box = $("#chat-mensajes");
  const div = document.createElement("div");
  div.className = "msg " + quien;
  div.innerHTML = html;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
}
async function enviarChat(texto) {
  if (!texto.trim()) return;
  mensaje("yo", esc(texto));
  const r = await fetch("/api/accion", { method: "POST", headers: { "Content-Type": "application/json", "X-Hope": "1" },
    body: JSON.stringify({ accion: "chat", params: { texto } }) });
  if (r.status === 401) { mostrarLogin(true); return; }
  const j = await r.json();
  if (j.tipo !== "propuesta") { mensaje("hope", esc(j.texto || j.mensaje || "")); return; }
  const p = j.propuesta;
  const div = mensaje("hope", esc(p.resumen) +
    (p.diffs || []).map((d) => `<pre class="diff">${pintarDiff(d.diff)}</pre>`).join("") +
    (p.sensible ? `<label>Para confirmar escribí <code>${esc(p.frase)}</code><input class="frase-conf" autocomplete="off"></label>` : "") +
    '<div class="botones"><button class="prim">Confirmar</button><button>Cancelar</button></div>');
  const [ok, no] = $$(".botones button", div);
  ok.addEventListener("click", async () => {
    ok.disabled = no.disabled = true;
    const res = await api("chat_confirmar", { id: p.id, frase: ($(".frase-conf", div) || {}).value || "" });
    if (res && res.ok === false && p.sensible) { ok.disabled = no.disabled = false; return; }
    mensaje("hope", res && res.ok !== false ? "✔ " + esc(res.mensaje || "hecho") : "✘ " + esc((res || {}).mensaje || "no se pudo"));
  });
  no.addEventListener("click", () => { ok.disabled = no.disabled = true; mensaje("hope", "Listo, no cambio nada."); });
}
$("#chat-form").addEventListener("submit", (ev) => { ev.preventDefault(); const t = $("#chat-texto").value; $("#chat-texto").value = ""; enviarChat(t); });
$$("[data-sugerencia]").forEach((b) => b.addEventListener("click", () => enviarChat(b.dataset.sugerencia)));
mensaje("hope", "Hola. Escribime qué querés cambiar o preguntame cómo va. Siempre te muestro qué voy a hacer antes de hacerlo.");

// --- extras de configuración: parámetros, disco, celular ---------------------------------------------
let redLleno = false;
function renderExtras(s) {
  const pr = s.presupuesto || {};
  $("#pd-actual").textContent = pr.techo_humano || "—";
  const d = s.discos || {};
  const el = d.elegida;
  $("#aj-disco").innerHTML = el ? `Hope usa <code>${esc(el.carpeta)}</code>${el.disco ? ` (${esc(el.disco.tipo)})` : ""}.` +
    ((d.en_sistema || []).length ? ` <span class="av">⚠ todavía en el disco del sistema: ${esc(d.en_sistema.join(", "))}</span>` : "")
    : (d.propuesta ? `Sin elegir todavía. Propuesta: <code>${esc(d.propuesta.carpeta)}</code>` : "");
  const rd = s.red || {};
  if (!redLleno) { $("#red-lan").checked = !!rd.lan; redLleno = true; }
  $("#red-estado").innerHTML = (rd.escuchando_en_red ? "Escuchando en la red local: " + (rd.urls || []).map((u) => `<code>${esc(u)}</code>`).join(" ") :
    "Solo en esta PC (127.0.0.1).") + (rd.pin_configurado ? " · PIN configurado" : "");
}
$("#pd-ver").addEventListener("click", async () => {
  const valor = $("#pd-valor").value.trim();
  if (!valor) return aviso("escribí cuántos parámetros (p. ej. 500M)", true);
  const r = await api("presupuesto_plan", { valor });
  const box = $("#pd-plan");
  if (!r || r.ok === false) { box.innerHTML = `<div class="mal">${esc((r || {}).mensaje || "")}</div>`; return; }
  box.innerHTML = (r.mensaje ? `<div class="av">${esc(r.mensaje)}</div>` : "") + (r.plan ? `<pre class="diff">${esc(r.plan)}</pre>` : "") +
    `<div class="fila"><label class="largo"><span>Para confirmar escribí <code>${esc(r.frase)}</code></span><input id="pd-frase" autocomplete="off"></label>
     <button class="peligro" id="pd-aplicar">${r.disponible ? "Subir el techo y agregar la etapa" : "Solo cambiar el techo"}</button></div>`;
  $("#pd-aplicar").addEventListener("click", async () => {
    if (!confirm(`¿Parámetros disponibles: ${r.humano}? Queda en la auditoría.`)) return;
    const a = await api("presupuesto_aplicar", { valor, frase: $("#pd-frase").value, confirmar: true });
    if (a && a.ok) box.innerHTML = `<div class="ok">✔ ${esc(a.mensaje)}</div>`;
  });
});
$("#aj-disco-usar").addEventListener("click", () => {
  const c = $("#aj-disco-carpeta").value.trim();
  if (c && confirm(`¿Usar ${c}? Lo que ya hay en runs/ y data/ se mueve ahí.`)) api("disco_usar", { carpeta: c, confirmar: true });
});
$("#red-guardar").addEventListener("click", async () => {
  if (!confirm($("#red-lan").checked ? "¿Permitir el acceso desde el celular (misma red, con PIN)?" : "¿Usar solo en esta PC?")) return;
  const r = await api("red_configurar", { lan: $("#red-lan").checked, pin: $("#red-pin").value, confirmar: true });
  if (r && r.ok) { $("#red-pin").value = ""; }
});

// --- ciclo -------------------------------------------------------------------------------------
function render(s) {
  if (!formListo) llenarFormularios(s.config);
  Object.entries(s.pasos).forEach(([k, p]) => chip(k, p.estado, p.hecho));
  const c = s.migrar.continuacion;
  const b = $("#banner");
  const pend = Object.keys(s.guardian.pendientes || {}).length;
  const partes = [];
  if (c.es_continuacion) partes.push("Continuación del entrenamiento de la GTX 1060 en esta PC");
  if (pend) partes.push(`${pend} solicitud(es) del guardián esperan tu decisión`);
  if (s.entrenamiento.estado === "en_curso") partes.push("entrenando · " + (s.entrenamiento.etapa_actual || ""));
  b.textContent = partes.join(" · ");
  b.classList.toggle("oculto", !partes.length);
  $("#nav-pie").textContent = "actualizado " + s.hora;
  renderCalibracion(s); renderDatos(s); renderLenguaje(s); renderMaestro(s); renderMigrar(s); renderGuardian(s); renderCuantizar(s); renderDiarios(s); renderAjustes(s); renderInicio(s); renderExtras(s);
}

async function actualizar() {
  try {
    const r = await fetch("/api/estado", { cache: "no-store" });
    if (r.status === 401) { mostrarLogin(true); return; }
    mostrarLogin(false);
    S = await r.json();
    render(S);
  } catch (e) {
    console.error(e);
    $("#nav-pie").textContent = "sin conexión con la aplicación";
    $("#frase").textContent = "Sin conexión con Hope. ¿Está abierta en la PC?";
  }
}
actualizar();
setInterval(actualizar, 2000);
