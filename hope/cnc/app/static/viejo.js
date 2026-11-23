/* ================================================================================================
 * ========================================== viejo.js ============================================
 * ================================================================================================
 *
 * Aviso para navegadores demasiado viejos (Internet Explorer, Firefox < 52, Chrome < 57).
 * Escrito a propósito en JavaScript de 1999 (ES3: var, function, sin flechas) para que hasta IE lo
 * ejecute y muestre qué instalar, en lugar de una pantalla en blanco.
 *
 * CÓMO ENCAJA
 *   * Es el PRIMER script que carga index.html (antes de compat.js, pathview.js y cnc.js).
 *   * Si el navegador sirve, no hace nada y el resto arranca normal.
 *   * Si no sirve, reemplaza el contenido de la página por un aviso en español y deja
 *     window.TISMA_VIEJO = true. Los scripts siguientes igual se intentan cargar, pero en IE fallan
 *     con error de sintaxis (const, class, =>) sin romper el aviso, que ya es HTML fijo.
 *
 * REGLA DE ESTE ARCHIVO
 *   Solo ES3: nada de let/const, funciones flecha, plantillas `…`, clases, ni métodos de ES5+
 *   (forEach, JSON, addEventListener…). Un solo error de sintaxis y IE no muestra nada.
 *   Sin eval ni new Function: la política de seguridad de la página (CSP) los prohíbe.
 *
 * EJEMPLO
 *   Abrir la página en IE 11 → en lugar de la pantalla de Mach3 aparece «TISMA CNC necesita un
 *   navegador más nuevo» con la lista de navegadores que andan en Windows XP/Vista/7.
 */
(function () {
  // CSS grid + funciones de ES2017 = Firefox 52 / Chrome 57 / Edge 16 en adelante: los mismos que
  // entienden async/await (sin eval, así la política de seguridad de la página sigue estricta)
  // Se prueban funciones (no la sintaxis: probar sintaxis pediría eval). Cada una marca un piso:
  //   Object.entries, Object.getOwnPropertyDescriptors, String.prototype.padStart → ES2017
  //     (Chrome 54-57, Firefox 47-52);
  //   CSS.supports("display", "grid") → la pantalla está armada con grid (Chrome 57, Firefox 52,
  //     Edge 16; IE 10/11 tiene un grid viejo con prefijo -ms- que no alcanza);
  //   Promise, fetch, Map, Set, Object.assign → los usa cnc.js en todos lados.
  var ok = !!(window.Promise && window.fetch && window.Map && window.Set && Object.assign && Object.entries &&
              Object.getOwnPropertyDescriptors && String.prototype.padStart &&
              window.CSS && window.CSS.supports && window.CSS.supports("display", "grid"));
  if (ok) { return; }
  window.TISMA_VIEJO = true;
  // Arma el aviso. Estilos en línea y colores fijos (violeta TISMA #7525a7, negro #1b1b1b) porque
  // cnc.css usa variables CSS, que IE no entiende. Fondo claro: se lee bien en cualquier monitor.
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
  // Este script está al final de <body>, así que normalmente document.body ya existe; si no,
  // se espera a onload (asignación directa: addEventListener no existe en IE 8 y anteriores).
  if (document.body) { show(); } else { window.onload = show; }
})();
