/* ================================================================================================
 * ========================================= compat.js ============================================
 * ================================================================================================
 *
 * Compatibilidad con equipos viejos (TISMA CNC tiene que andar en PCs de hace muchos años).
 * Objetivo: Firefox 52 ESR (el último de Windows XP/Vista) y Chrome 57 (2017) en adelante, con la misma
 * sintaxis (ES2017) que el resto. Este archivo se carga primero y da:
 * * clase «no-flexgap» en <html> si el navegador no sabe «gap» en flexbox (usa compat.css con márgenes);
 * * modo liviano (menos refrescos, sin animaciones ni sombras) automático en equipos de 1–2 núcleos o
 *   con ?liviano=1, y recordado por el navegador;
 * * ayudas: onPress (pointer, ratón o táctil), observeResize, readText (FileReader), normKey (teclas).
 *
 * CÓMO ENCAJA
 *   * index.html carga: viejo.js → compat.js → pathview.js → cnc.js. viejo.js ya descartó los
 *     navegadores que ni siquiera entienden ES2017 (IE, Firefox < 52, Chrome < 57); acá se cubren
 *     las diferencias que quedan ENTRE los navegadores que sí sirven.
 *   * Lo que se define acá son funciones globales (no hay módulos: Firefox 52 no los soporta y el
 *     proyecto no usa empaquetador). Las usan cnc.js y pathview.js.
 *   * La parte de CSS está en compat.css, que GENERA hope/cnc/app/compat_css.py a partir de cnc.css
 *     y tema_tisma.css (no editarlo a mano). Este archivo solo pone las clases que lo activan:
 *       .no-flexgap → márgenes en lugar de «gap» en los contenedores flex;
 *       .lite       → sin animaciones, transiciones ni sombras.
 *
 * QUÉ FALTA EN CADA NAVEGADOR VIEJO (y qué alternativa se usa)
 *   | Función                 | Desde (Chrome / Firefox)  | Alternativa acá                       |
 *   |-------------------------|---------------------------|---------------------------------------|
 *   | gap en flexbox          | 84 / 63                   | clase no-flexgap + compat.css         |
 *   | Pointer Events          | 55 / 59                   | mouse* + touch* (onPress)             |
 *   | ResizeObserver          | 64 / 69                   | resize de ventana + sondeo (700 ms)   |
 *   | File.text()             | 76 / 69                   | FileReader (readText)                 |
 *   | KeyboardEvent.key («Escape», «ArrowLeft»…) | Edge 18 / IE usan «Esc», «Left» | normKey  |
 *
 * EJEMPLO
 *     onPress(boton, () => jogStart("X", 1), () => jogStop());
 *     observeResize(div, () => vista.draw());
 *     readText(input.files[0]).then(texto => cargar(texto));
 *     document.addEventListener("keydown", e => { if (normKey(e) === "Escape") parar(); });
 */
"use strict";

/* ================================================================================================
 * ============================ DETECCIÓN AL CARGAR (se ejecuta una vez) ==========================
 * ================================================================================================ */
(function () {
  const root = document.documentElement;
  // ¿flexbox con gap? (Chrome 84 / Firefox 63 en adelante)
  // No hay forma directa de preguntarlo (CSS.supports("gap", …) da true por el gap de grid aunque
  // flex no lo use), así que se mide: una columna flex con dos hijos vacíos y 1 px de separación
  // mide 1 px de alto solo si el navegador aplica el gap en flex; si no, mide 0.
  const d = document.createElement("div");
  d.style.display = "flex"; d.style.flexDirection = "column"; d.style.rowGap = "1px";
  d.appendChild(document.createElement("div")); d.appendChild(document.createElement("div"));
  document.body ? document.body.appendChild(d) : root.appendChild(d);   // el <script> puede correr antes de <body>
  const ok = d.scrollHeight === 1;
  d.parentNode.removeChild(d);
  if (!ok) root.classList.add("no-flexgap");

  // modo liviano
  // Prioridad: ?liviano=0/1 en la dirección > lo que se eligió antes (localStorage) > automático
  // según núcleos de CPU (1–2 núcleos = PC vieja). Sin hardwareConcurrency (Safari viejo) se
  // asume 2, o sea liviano: ante la duda, que ande.
  let lite = null;
  // localStorage puede tirar excepción (archivo local, almacenamiento bloqueado): se trata como vacío
  try { lite = window.localStorage.getItem("tisma-cnc-liviano"); } catch (e) { lite = null; }
  const q = /[?&]liviano=(\d)/.exec(window.location.search);
  if (q) lite = q[1];
  if (lite === null) lite = (navigator.hardwareConcurrency || 2) <= 2 ? "1" : "0";
  window.TISMA_LITE = lite === "1";          // lo leen cnc.js (frecuencia de consulta) y pathview.js
  if (window.TISMA_LITE) root.classList.add("lite");
})();

/* ================================================================================================
 * ======================================== AYUDAS GLOBALES =======================================
 * ================================================================================================ */

/* Activa o desactiva el modo liviano (casilla «Modo liviano» de Configuración) y lo recuerda en el
   navegador para la próxima vez.
     on: true = liviano (consultas más espaciadas, sin animaciones ni sombras, dibujo simplificado).
   No devuelve nada. Si no hay almacenamiento, el cambio vale solo hasta recargar la página. */
function setLite(on) {
  window.TISMA_LITE = !!on;
  document.documentElement.classList.toggle("lite", !!on);
  try { window.localStorage.setItem("tisma-cnc-liviano", on ? "1" : "0"); } catch (e) { /* sin almacenamiento */ }
}

/** Presionar y soltar con pointer events, o ratón + táctil en navegadores sin ellos. */
/* Para botones que actúan MIENTRAS están apretados (jog: la máquina se mueve hasta soltar).
     el:   elemento;
     down: función al apretar;
     up:   función al soltar. Se llama también al salir del botón o si el sistema cancela el toque:
           por seguridad, un jog nunca debe quedar andando porque se perdió el «soltar».
   Pointer Events: Chrome 55+, Firefox 59+. Firefox 52 (Windows XP/Vista) no los tiene: se usan
   mouse* y touch*. En touchstart se llama a preventDefault() para que el navegador no genere además
   un mousedown «de compatibilidad» (sería un segundo down) ni haga zoom/desplazamiento. */
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
/*   el: elemento a vigilar;  cb: función sin argumentos útiles que se llama en cada cambio.
   ResizeObserver: Chrome 64+, Firefox 69+. Sin él, el evento resize de la ventana no alcanza: un
   elemento cambia de tamaño también al mostrar una pantalla oculta (pestañas Alt-1…7) sin que la
   ventana cambie. Por eso además se sondea el tamaño cada 700 ms (barato: dos lecturas) y se
   guarda el último visto en el propio elemento (el._w, el._h). */
function observeResize(el, cb) {
  if (window.ResizeObserver) { new ResizeObserver(cb).observe(el); return; }
  window.addEventListener("resize", cb);
  setInterval(() => {                       // por si cambia sin que cambie la ventana (pestañas)
    const w = el.clientWidth, h = el.clientHeight;
    if (w !== el._w || h !== el._h) { el._w = w; el._h = h; cb(); }
  }, 700);
}

/** Texto de un archivo elegido (File.text() es de 2019: se usa FileReader). */
/*   file: un File (de <input type="file"> o de arrastrar y soltar).
   Devuelve una Promise con el texto (UTF-8), o que falla con el error del lector.
   File.text() existe desde Chrome 76 / Firefox 69; FileReader anda en todos. */
function readText(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(r.error);
    r.readAsText(file);
  });
}

/** Nombre de tecla normalizado (Edge/IE viejos usan «Esc», «Left», «Spacebar»…). */
/* KEY_ALIASES: nombre viejo → nombre estándar de KeyboardEvent.key. Edge hasta la versión 18
   (EdgeHTML, el de Windows 10 hasta 2020) e IE usan los nombres del borrador viejo de la norma.
   Importa porque los atajos de Mach3 (Esc = Stop, Espacio = Feed Hold, flechas = jog) tienen que
   responder igual en todos. */
const KEY_ALIASES = { Esc: "Escape", Left: "ArrowLeft", Right: "ArrowRight", Up: "ArrowUp", Down: "ArrowDown",
                      Spacebar: " ", Del: "Delete" };
/* normKey(e): devuelve e.key con el nombre estándar ("" si el navegador no da e.key). */
function normKey(e) { const k = e.key || ""; return KEY_ALIASES[k] || k; }
