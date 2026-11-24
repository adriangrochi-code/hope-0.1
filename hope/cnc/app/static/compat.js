/* ================================================================================================
 * ========================================== compat.js ===========================================
 * ================================================================================================
 *
 * Compatibility with old computers (TISMA CNC has to run on PCs that are many years old).
 * Target: Firefox 52 ESR (the last one for Windows XP/Vista) and Chrome 57 (2017) onwards, with the
 * same syntax (ES2017) as the rest. This file is loaded first and provides:
 * * the «no-flexgap» class on <html> if the browser does not know «gap» in flexbox (compat.css then
 *   uses margins);
 * * light mode (fewer refreshes, no animations or shadows), automatic on 1–2 core machines or with
 *   ?liviano=1, and remembered by the browser;
 * * helpers: onPress (pointer, mouse or touch), observeResize, readText (FileReader), normKey (keys).
 *
 * HOW IT FITS
 *   * index.html loads: viejo.js → compat.js → pathview.js → cnc.js. viejo.js has already rejected
 *     the browsers that do not even understand ES2017 (IE, Firefox < 52, Chrome < 57); this file
 *     covers the differences that remain AMONG the browsers that do work.
 *   * What is defined here are global functions (there are no modules: Firefox 52 does not support
 *     them and the project uses no bundler). cnc.js and pathview.js use them.
 *   * The CSS part is in compat.css, which hope/cnc/app/compat_css.py GENERATES from cnc.css and
 *     tema_tisma.css (do not edit it by hand). This file only sets the classes that enable it:
 *       .no-flexgap → margins instead of «gap» in flex containers;
 *       .lite       → no animations, transitions or shadows.
 *
 * WHAT EACH OLD BROWSER LACKS (and which alternative is used)
 *   | Feature                 | Since (Chrome / Firefox)  | Alternative here                      |
 *   |-------------------------|---------------------------|---------------------------------------|
 *   | gap in flexbox          | 84 / 63                   | no-flexgap class + compat.css         |
 *   | Pointer Events          | 55 / 59                   | mouse* + touch* (onPress)             |
 *   | ResizeObserver          | 64 / 69                   | window resize + polling (700 ms)      |
 *   | File.text()             | 76 / 69                   | FileReader (readText)                 |
 *   | KeyboardEvent.key («Escape», «ArrowLeft»…) | Edge 18 / IE use «Esc», «Left» | normKey    |
 *
 * EXAMPLE
 *     onPress(boton, () => jogStart("X", 1), () => jogStop());
 *     observeResize(div, () => vista.draw());
 *     readText(input.files[0]).then(texto => cargar(texto));
 *     document.addEventListener("keydown", e => { if (normKey(e) === "Escape") parar(); });
 */
"use strict";

/* ================================================================================================
 * ================================ DETECTION ON LOAD (runs once) =================================
 * ================================================================================================ */
(function () {
  const root = document.documentElement;
  // Flexbox with gap? (Chrome 84 / Firefox 63 onwards)
  // There is no direct way to ask (CSS.supports("gap", …) returns true because of grid gap even if
  // flex does not use it), so it is measured: a flex column with two empty children and 1 px gap
  // is 1 px tall only if the browser applies gap in flex; otherwise it is 0.
  const d = document.createElement("div");
  d.style.display = "flex"; d.style.flexDirection = "column"; d.style.rowGap = "1px";
  d.appendChild(document.createElement("div")); d.appendChild(document.createElement("div"));
  document.body ? document.body.appendChild(d) : root.appendChild(d);   // the <script> may run before <body>
  const ok = d.scrollHeight === 1;
  d.parentNode.removeChild(d);
  if (!ok) root.classList.add("no-flexgap");

  // light mode
  // Priority: ?liviano=0/1 in the address > what was chosen before (localStorage) > automatic
  // by CPU core count (1–2 cores = old PC). Without hardwareConcurrency (old Safari) 2 is
  // assumed, i.e. light: when in doubt, make it work.
  let lite = null;
  // localStorage may throw (local file, blocked storage): treated as empty
  try { lite = window.localStorage.getItem("tisma-cnc-liviano"); } catch (e) { lite = null; }
  const q = /[?&]liviano=(\d)/.exec(window.location.search);
  if (q) lite = q[1];
  if (lite === null) lite = (navigator.hardwareConcurrency || 2) <= 2 ? "1" : "0";
  window.TISMA_LITE = lite === "1";          // read by cnc.js (polling frequency) and pathview.js
  if (window.TISMA_LITE) root.classList.add("lite");
})();

/* ================================================================================================
 * ======================================== GLOBAL HELPERS ========================================
 * ================================================================================================ */

/* Turns light mode on or off (the «Modo liviano» (Light mode) checkbox in Configuración) and
   remembers it in the browser for next time.
     on: true = light (less frequent polling, no animations or shadows, simplified drawing).
   Returns nothing. Without storage, the change only lasts until the page is reloaded. */
function setLite(on) {
  window.TISMA_LITE = !!on;
  document.documentElement.classList.toggle("lite", !!on);
  try { window.localStorage.setItem("tisma-cnc-liviano", on ? "1" : "0"); } catch (e) { /* no storage */ }
}

/** Press and release with pointer events, or mouse + touch in browsers without them. */
/* For buttons that act WHILE they are held down (jog: the machine moves until released).
     el:   element;
     down: function called on press;
     up:   function called on release. Also called when leaving the button or when the system
           cancels the touch: for safety, a jog must never keep running because a «release» was lost.
   Pointer Events: Chrome 55+, Firefox 59+. Firefox 52 (Windows XP/Vista) does not have them, so
   mouse* and touch* are used. On touchstart preventDefault() is called so the browser does not also
   generate a «compatibility» mousedown (it would be a second down) nor zoom/scroll. */
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

/** Notify when an element changes size (ResizeObserver or, if missing, the window). */
/*   el: element to watch;  cb: function (no useful arguments) called on every change.
   ResizeObserver: Chrome 64+, Firefox 69+. Without it, the window resize event is not enough: an
   element also changes size when a hidden screen is shown (Alt-1…7 tabs) without the window
   changing. That is why the size is also polled every 700 ms (cheap: two reads) and the last seen
   size is stored on the element itself (el._w, el._h). */
function observeResize(el, cb) {
  if (window.ResizeObserver) { new ResizeObserver(cb).observe(el); return; }
  window.addEventListener("resize", cb);
  setInterval(() => {                       // in case it changes without the window changing (tabs)
    const w = el.clientWidth, h = el.clientHeight;
    if (w !== el._w || h !== el._h) { el._w = w; el._h = h; cb(); }
  }, 700);
}

/** Text of a chosen file (File.text() is from 2019: FileReader is used). */
/*   file: a File (from <input type="file"> or from drag and drop).
   Returns a Promise with the text (UTF-8), or one that rejects with the reader's error.
   File.text() exists since Chrome 76 / Firefox 69; FileReader works everywhere. */
function readText(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(r.error);
    r.readAsText(file);
  });
}

/** Normalized key name (old Edge/IE use «Esc», «Left», «Spacebar»…). */
/* KEY_ALIASES: old name → standard KeyboardEvent.key name. Edge up to version 18 (EdgeHTML, the
   one in Windows 10 until 2020) and IE use the names from the old draft of the standard.
   It matters because the Mach3 shortcuts (Esc = Stop, Space = Feed Hold, arrows = jog) must
   respond the same in all of them. */
const KEY_ALIASES = { Esc: "Escape", Left: "ArrowLeft", Right: "ArrowRight", Up: "ArrowUp", Down: "ArrowDown",
                      Spacebar: " ", Del: "Delete" };
/* normKey(e): returns e.key with the standard name ("" if the browser gives no e.key). */
function normKey(e) { const k = e.key || ""; return KEY_ALIASES[k] || k; }
