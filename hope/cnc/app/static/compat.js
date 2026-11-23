/* Compatibilidad con equipos viejos (TISMA CNC tiene que andar en PCs de hace muchos años).
   Objetivo: Firefox 52 ESR (el último de Windows XP/Vista) y Chrome 57 (2017) en adelante, con la misma
   sintaxis (ES2017) que el resto. Este archivo se carga primero y da:
   * clase «no-flexgap» en <html> si el navegador no sabe «gap» en flexbox (usa compat.css con márgenes);
   * modo liviano (menos refrescos, sin animaciones ni sombras) automático en equipos de 1–2 núcleos o
     con ?liviano=1, y recordado por el navegador;
   * ayudas: onPress (pointer, ratón o táctil), observeResize, readText (FileReader), normKey (teclas). */
"use strict";

(function () {
  const root = document.documentElement;
  // ¿flexbox con gap? (Chrome 84 / Firefox 63 en adelante)
  const d = document.createElement("div");
  d.style.display = "flex"; d.style.flexDirection = "column"; d.style.rowGap = "1px";
  d.appendChild(document.createElement("div")); d.appendChild(document.createElement("div"));
  document.body ? document.body.appendChild(d) : root.appendChild(d);
  const ok = d.scrollHeight === 1;
  d.parentNode.removeChild(d);
  if (!ok) root.classList.add("no-flexgap");

  // modo liviano
  let lite = null;
  try { lite = window.localStorage.getItem("tisma-cnc-liviano"); } catch (e) { lite = null; }
  const q = /[?&]liviano=(\d)/.exec(window.location.search);
  if (q) lite = q[1];
  if (lite === null) lite = (navigator.hardwareConcurrency || 2) <= 2 ? "1" : "0";
  window.TISMA_LITE = lite === "1";
  if (window.TISMA_LITE) root.classList.add("lite");
})();

function setLite(on) {
  window.TISMA_LITE = !!on;
  document.documentElement.classList.toggle("lite", !!on);
  try { window.localStorage.setItem("tisma-cnc-liviano", on ? "1" : "0"); } catch (e) { /* sin almacenamiento */ }
}

/** Presionar y soltar con pointer events, o ratón + táctil en navegadores sin ellos. */
function onPress(el, down, up) {
  if (window.PointerEvent) {
    el.addEventListener("pointerdown", down);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointerleave", up);
    el.addEventListener("pointercancel", up);
    return;
  }
  el.addEventListener("mousedown", down);
  el.addEventListener("mouseup", up);
  el.addEventListener("mouseleave", up);
  el.addEventListener("touchstart", e => { e.preventDefault(); down(e); });
  el.addEventListener("touchend", up);
  el.addEventListener("touchcancel", up);
}

/** Avisar cuando cambia el tamaño de un elemento (ResizeObserver o, si no hay, la ventana). */
function observeResize(el, cb) {
  if (window.ResizeObserver) { new ResizeObserver(cb).observe(el); return; }
  window.addEventListener("resize", cb);
  setInterval(() => {                       // por si cambia sin que cambie la ventana (pestañas)
    const w = el.clientWidth, h = el.clientHeight;
    if (w !== el._w || h !== el._h) { el._w = w; el._h = h; cb(); }
  }, 700);
}

/** Texto de un archivo elegido (File.text() es de 2019: se usa FileReader). */
function readText(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(r.error);
    r.readAsText(file);
  });
}

/** Nombre de tecla normalizado (Edge/IE viejos usan «Esc», «Left», «Spacebar»…). */
const KEY_ALIASES = { Esc: "Escape", Left: "ArrowLeft", Right: "ArrowRight", Up: "ArrowUp", Down: "ArrowDown",
                      Spacebar: " ", Del: "Delete" };
function normKey(e) { const k = e.key || ""; return KEY_ALIASES[k] || k; }
