/* Aviso para navegadores demasiado viejos (Internet Explorer, Firefox < 52, Chrome < 57).
   Escrito a propósito en JavaScript de 1999 (ES3: var, function, sin flechas) para que hasta IE lo
   ejecute y muestre qué instalar, en lugar de una pantalla en blanco. */
(function () {
  // CSS grid + funciones de ES2017 = Firefox 52 / Chrome 57 / Edge 16 en adelante: los mismos que
  // entienden async/await (sin eval, así la política de seguridad de la página sigue estricta)
  var ok = !!(window.Promise && window.fetch && window.Map && window.Set && Object.assign && Object.entries &&
              Object.getOwnPropertyDescriptors && String.prototype.padStart &&
              window.CSS && window.CSS.supports && window.CSS.supports("display", "grid"));
  if (ok) { return; }
  window.TISMA_VIEJO = true;
  var show = function () {
    document.body.innerHTML =
      '<div style="font-family: Arial, sans-serif; max-width: 640px; margin: 60px auto; padding: 24px; ' +
      'background: #ffffff; color: #1b1b1b; border: 3px solid #7525a7;">' +
      '<h2 style="color: #7525a7; margin-top: 0;">TISMA CNC necesita un navegador más nuevo</h2>' +
      '<p>Este navegador (probablemente Internet Explorer o un Firefox/Chrome muy viejo) no puede mostrar la pantalla.</p>' +
      '<p>Instalá uno de estos, que funcionan en PCs viejas:</p><ul>' +
      '<li><b>Firefox ESR</b> (Windows 7 en adelante, y Linux)</li>' +
      '<li><b>Chrome 109</b> (el último para Windows 7) o más nuevo</li></ul>' +
      '<p>Después volvé a abrir TISMA CNC: la ventana se abre sola con el navegador nuevo. ' +
      'También podés copiar esta dirección en él: <b>' + window.location.href + '</b></p></div>';
    document.body.style.background = "#eeeeee";
  };
  if (document.body) { show(); } else { window.onload = show; }
})();
