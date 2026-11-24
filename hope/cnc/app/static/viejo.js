/* ================================================================================================
 * =========================================== viejo.js ===========================================
 * ================================================================================================
 *
 * Notice for browsers that are too old (Internet Explorer, Firefox < 52, Chrome < 57).
 * Deliberately written in 1999-era JavaScript (ES3: var, function, no arrows) so that even IE runs
 * it and shows what to install, instead of a blank screen.
 *
 * HOW IT FITS
 *   * It is the FIRST script index.html loads (before compat.js, pathview.js and cnc.js).
 *   * If the browser is good enough, it does nothing and the rest starts up normally.
 *   * If it is not, it replaces the page content with a notice in Spanish and sets
 *     window.TISMA_VIEJO = true. The following scripts are still loaded, but in IE they fail with
 *     a syntax error (const, class, =>) without breaking the notice, which is already static HTML.
 *
 * RULE FOR THIS FILE
 *   ES3 only: no let/const, arrow functions, `…` templates, classes, or ES5+ methods
 *   (forEach, JSON, addEventListener…). A single syntax error and IE shows nothing.
 *   No eval or new Function: the page's security policy (CSP) forbids them.
 *
 * EXAMPLE
 *   Open the page in IE 11 → instead of the Mach3 screen, «TISMA CNC necesita un navegador más
 *   nuevo» (TISMA CNC needs a newer browser) appears, with the list of browsers that work on
 *   Windows XP/Vista/7.
 */
(function () {
  // CSS grid + ES2017 functions = Firefox 52 / Chrome 57 / Edge 16 and later: the same ones that
  // understand async/await (without eval, so the page's security policy stays strict).
  // Functions are tested (not syntax: testing syntax would need eval). Each one marks a minimum:
  //   Object.entries, Object.getOwnPropertyDescriptors, String.prototype.padStart → ES2017
  //     (Chrome 54-57, Firefox 47-52);
  //   CSS.supports("display", "grid") → the screen is laid out with grid (Chrome 57, Firefox 52,
  //     Edge 16; IE 10/11 has an old -ms- prefixed grid that is not enough);
  //   Promise, fetch, Map, Set, Object.assign → cnc.js uses them everywhere.
  var ok = !!(window.Promise && window.fetch && window.Map && window.Set && Object.assign && Object.entries &&
              Object.getOwnPropertyDescriptors && String.prototype.padStart &&
              window.CSS && window.CSS.supports && window.CSS.supports("display", "grid"));
  if (ok) { return; }
  window.TISMA_VIEJO = true;
  // Builds the notice. Inline styles and fixed colors (TISMA violet #7525a7, black #1b1b1b) because
  // cnc.css uses CSS variables, which IE does not understand. Light background: readable on any monitor.
  var show = function () {
    document.body.innerHTML =
      '<div style="font-family: Arial, sans-serif; max-width: 640px; margin: 60px auto; padding: 24px; ' +
      'background: #ffffff; color: #1b1b1b; border: 3px solid #7525a7;">' +
      '<h2 style="color: #7525a7; margin-top: 0;">TISMA CNC necesita un navegador más nuevo</h2>' +
      '<p>Este navegador (probablemente Internet Explorer o un Firefox/Chrome muy viejo) no puede mostrar la pantalla.</p>' +
      '<p>Instalá uno de estos, que funcionan en PCs viejas:</p><ul>' +
      '<li><b>Firefox ESR</b>: en Windows XP y Vista, <b>Firefox 52 ESR</b> (el último para ellos)</li>' +
      '<li><b>Supermium</b>: Chromium actual que funciona en Windows XP, Vista y 7</li>' +
      '<li><b>Chrome 109</b> (el último para Windows 7) o más nuevo</li></ul>' +
      '<p>Después volvé a abrir TISMA CNC: la ventana se abre sola con el navegador nuevo. ' +
      'También podés copiar esta dirección en él: <b>' + window.location.href + '</b></p></div>';
    document.body.style.background = "#eeeeee";
  };
  // This script is at the end of <body>, so document.body normally exists already; if not,
  // wait for onload (direct assignment: addEventListener does not exist in IE 8 and earlier).
  if (document.body) { show(); } else { window.onload = show; }
})();
