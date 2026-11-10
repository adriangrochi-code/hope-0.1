# Código open source para entrenar a Hope (hope/data/codigo_abierto.py)

Pedido del usuario: entrenar a Hope con **código open source de GitHub**, de forma legal y filtrada.
Complementa `fuentes_legales.md` (texto web) y `hope/data/repos.py` (repos propios).

## Resumen

| | |
|---|---|
| Módulo | `hope/data/codigo_abierto.py` |
| Configuración | `configs/fuentes_web.yaml`, sección `codigo_abierto` (del usuario) |
| Corre | dentro de `python -m hope.data.web_sources --diario` (misma corrida diaria y horario, **cuota propia**) o sola con `python -m hope.data.codigo_abierto --diario` |
| Salida del día | `data/codigo/<fecha>/` → `train.bin`, `val.bin`, `provenance.jsonl`, `ATRIBUCIONES.md`, `informe.json` |
| Acumulado (para entrenar) | `data/pre/codigo_abierto/` → fuente `codigo_abierto` en `plan_gtx1060.yaml` (0,10 y 0,08) y `plan_a770_continuacion.yaml` (0,10 y 0,10) |
| Pruebas | `tests/test_codigo_abierto.py` (10 pruebas, sin red) + redirecciones de descarga en `tests/test_web_sources.py` |

## Fuentes, en orden de preferencia

1. **The Stack v2 dedup** (`bigcode/the-stack-v2-dedup`) — inactiva por defecto. Licencia detectada por
   archivo y exclusión voluntaria de autores («Am I in The Stack?») ya aplicada. Solo trae
   identificadores: el contenido se baja de Software Heritage (`contenido_desde: swh_s3`), un archivo por
   pedido, mucho más lento. Útil si se quiere lo más nuevo.
2. **The Stack dedup v1** (`bigcode/the-stack-dedup`) — inactiva hasta que el usuario la habilite. Trae el
   código dentro del Parquet, por carpetas de lenguaje (`data/python`, `data/c++`, …).
3. **codeparrot/github-code-clean** — **activa**, sin acceso restringido. Campo `license` por archivo.
4. **Repos de GitHub** por tema y lenguaje — **activa**. Lo que aporta y los conjuntos no tienen: el
   **historial de commits** → episodios «problema → corrección».

### Habilitar The Stack (acceso restringido)

1. Entrar con la cuenta de Hugging Face a la ficha del conjunto
   (https://huggingface.co/datasets/bigcode/the-stack-dedup o …/the-stack-v2-dedup), leer los términos
   de uso y aceptarlos (botón «Agree and access repository»).
2. Crear un token de **solo lectura** en https://huggingface.co/settings/tokens.
3. Ponerlo en la variable de entorno `HF_TOKEN` (nunca en un archivo del repo):
   Linux `export HF_TOKEN=hf_…` (en `~/.profile`); Windows `setx HF_TOKEN hf_…`.
4. En `configs/fuentes_web.yaml` → `codigo_abierto.fuentes`, `activo: true` en `the_stack_dedup`.
5. Autorizar en `configs/web.yaml` los dominios `huggingface.co`, `cas-bridge.xethub.hf.co`,
   `cdn-lfs.hf.co`, `cdn-lfs-us-1.hf.co` (los Parquet redirigen a la CDN) y, para GitHub,
   `api.github.com` y `github.com`.

Sin el token, la fuente queda en pausa con un aviso que explica estos pasos y no toca la red. El token
viaja solo en la cabecera `Authorization`, nunca en la URL, y **no** se reenvía si una redirección
lleva a otro dominio (arreglo nuevo en `WebClient.download`, que antes no seguía redirecciones: las
descargas de Hugging Face —incluida la Wikipedia de `hf_parquet`— fallaban con HTTP 302).

### GitHub

* Búsqueda `GET /search/repositories?q=topic:… language:… stars:>=50 fork:false archived:false`,
  ordenada por estrellas. Token opcional en `HOPE_GITHUB_TOKEN` (un token sin permisos alcanza). La API
  permite 10 búsquedas por minuto sin token y 30 con token: `ritmo_s: 7` deja margen en ambos casos.
  Todo pasa por el guardián (`network.access`), igual que el resto de la red.
* Licencia del repo desde `license.spdx_id`; si el archivo `LICENSE` del clon dice otra cosa no permitida,
  manda la más restrictiva. `NOASSERTION` o sin licencia → no se clona.
* `git clone --depth 300 --single-branch --no-tags --filter=blob:limit=256k`, con
  `GIT_NO_LAZY_FETCH=1` al procesar (git no baja blobs por su cuenta fuera de la cuota). El tamaño del
  clon se descuenta de la cuota y del límite diario del guardián. El clon se borra al terminar.
* Temas configurados según los proyectos del usuario: impresión 3D, Arduino, embebidos, firmware, CNC,
  robótica, Android, CLI, aprendizaje automático, electrónica (editable).
* Los repos ya procesados quedan en `vistos.sqlite` y no se vuelven a clonar.

## Licencias

| Grupo | Licencias | Por defecto |
|---|---|---|
| Permisivas | MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Unlicense, CC0-1.0, Zlib | **sí** |
| Copyleft débil | MPL-2.0 (`incluir_mpl`) | no (opcional) |
| Copyleft | GPL-2.0/3.0, LGPL-2.1/3.0, AGPL-3.0 (`incluir_copyleft`) | **no** |
| Sin licencia, `NOASSERTION`, propietarias o desconocidas | — | **nunca** |

*Nota sobre copyleft:* si entrenar un modelo con código GPL obliga a algo (por ejemplo, a publicar el
modelo bajo la misma licencia) es una cuestión discutida y sin resolver en lo legal. Por eso queda
apagado; activarlo es decisión del usuario. Mientras esté apagado, **Marlin (GPL-3.0) no entra**: el
código propio del usuario sobre Marlin sigue entrando por `hope/data/repos.py`.

Reglas: las expresiones SPDX se respetan (`MIT OR GPL-3.0` pasa, `MIT AND GPL-3.0` no); si un conjunto
lista varias licencias para un archivo, **todas** deben estar permitidas; un encabezado
`SPDX-License-Identifier` dentro del archivo manda sobre la licencia del repo.

**Procedencia**: cada documento guarda en `provenance.jsonl` fuente, conjunto, repo, ruta, commit,
licencia, lenguaje, nota de calidad, secretos quitados, hash y partición. **Atribuciones**:
`ATRIBUCIONES.md` (y `atribuciones.json`) en la carpeta del día y en el acumulado: una fila por repo con
licencia, cantidad de archivos y de episodios, commit y origen. Regenerar a mano:
`python -m hope.data.codigo_abierto --atribuciones data/pre/codigo_abierto`.

## Filtros (en este orden)

1. **Lenguaje** por extensión (o etiqueta del conjunto) y **licencia**.
2. **Archivo**: vendorizado (`node_modules/`, `vendor/`, `third_party/`, `dist/`, `build/`…), generado
   (nombres `*.min.js`, `*_pb2.py`, `*.pb.h`, lockfiles…, o marcas `@generated`, `DO NOT EDIT`,
   `auto-generated` y «generated by» solo en comentarios), binario, > 100 KB, > 5000 líneas, línea
   > 1000 caracteres, largo medio > 120 (minificado), < 25 % alfanumérico (datos) y arreglos hex
   embebidos (bitmaps de pantallas de arranque en firmware).
3. **Muestreo por lenguaje** determinista (hash), con los pesos de `pesos_lenguajes`: Python, C, C++,
   Arduino y G-code 1,0; Rust y shell 0,8; Kotlin y TypeScript 0,7; Java y JavaScript 0,6; otros 0.
4. **Secretos y datos personales** → marcadores: claves privadas (`<CLAVE_PRIVADA>`), AWS, Google,
   OpenAI, Anthropic, Stripe (`<CLAVE_API>`), tokens de GitHub, GitLab, Slack, Hugging Face, npm, JWT,
   Telegram, webhooks (`<TOKEN>`), credenciales en URL y asignaciones `password=… / api_key=… /
   token=…` (`<CONTRASEÑA>`, `<CLAVE_API>`, `<TOKEN>`), cadenas de alta entropía (≥ 4,3 bits/carácter,
   mayúsculas + minúsculas + dígitos, ≥ 32 caracteres: `<SECRETO>`), emails (`<EMAIL>`; se conservan
   `example.com`, `noreply` de GitHub y `git@`) e IP públicas (`<IP>`; las privadas y de bucle quedan).
   Los marcadores de posición (`YOUR_API_KEY`, `os.environ[...]`) y los hashes hex no se tocan.
5. **Calidad** (nota 0–1, mínimo 0,6): **que parsee** (`ast.parse` en Python, `bash -n` en shell si hay
   bash, llaves/corchetes/paréntesis balanceados salteando cadenas y comentarios en C, C++, Arduino,
   Java, Kotlin, JS/TS, Rust…, líneas de G-code válidas); comentarios (ni 0 ni casi todo), estructura
   (funciones/clases), largo de línea razonable, repetición.
6. **Deduplicación** exacta (SHA-256 sin espacios finales) y aproximada (MinHash de 128 permutaciones
   sobre 5-gramas de tokens, LSH 16×8, Jaccard estimado ≥ 0,8), persistente en
   `data/codigo/vistos.sqlite` y **también contra el corpus existente** (`existente`: `data/pre/codigo`,
   `repos_codigo`, `repos_historial`; se indexa de forma incremental, sin el encabezado `⟦repo: …⟧`).

Formato de cada documento: `⟦repo: dueño/nombre⟧ ⟦archivo: ruta⟧ ⟦lenguaje: python⟧ ⟦licencia: MIT⟧`
y el código (mismo estilo que `repos.py`).

## Episodios de corrección

Reusa `hope.data.repos.history_docs`: commits con *fix, bug, arregl…, corrig…, error, crash…* →
`⟦repo: dueño/nombre⟧ ⟦licencia: MIT⟧ ⟦arreglo⟧ Problema: <mensaje> Archivos: … Corrección: <diff>`.
**Se excluyen los commits con coautoría de IA** (`Co-Authored-By: Claude`, `noreply@anthropic.com`,
`Claude-Session`, «Generated with Claude Code») y los archivos cuyo historial es todo de commits así,
igual que en `repos.py`. También se descartan los commits del borde del clon superficial (su diff sería
el repo entero), los que no tocan código (o solo vendorizado/generado) y los que no son arreglos
(`solo_arreglos: true`). Los diffs también pasan por el filtro de secretos y la deduplicación.

## Volúmenes estimados

Medición propia del filtro (este contenedor, un núcleo, Python 3.13 + paquetes instalados, 20 421
archivos, 215 MB): **1,5 MB/s por núcleo**, se conservó el **66 %**. Motivos de descarte: generados
7284 (casi todos encabezados `@generated` de PyTorch), muy cortos 567, muy largos 270, casi duplicados
223, no parsean 74, minificados 69, duplicados 57. Secretos/datos personales reemplazados: 494 emails,
114 IP, 110 cadenas de alta entropía, 89 tokens, 49 contraseñas, 9 claves de API, 1 clave privada.
El cuello de botella es la red, no la CPU (400 MB/día de salida ≈ 5 minutos de CPU).

Tamaño de las fuentes, **orden de magnitud** según las fichas de los conjuntos (verificar en cada
ficha antes de planificar; el texto plano es ~3–4× el Parquet comprimido):

| Lenguaje | The Stack dedup v1 (texto) | Quedaría tras licencia permisiva + filtros (≈ 45–60 %) |
|---|---:|---:|
| Python | ~60–80 GB | ~30–45 GB |
| C | ~50 GB | ~20–30 GB |
| C++ (incl. Arduino) | ~50–70 GB | ~20–35 GB |
| Java | ~90–110 GB | ~45–60 GB (peso 0,6 → ~30 GB) |
| JavaScript | ~90–130 GB | ~35–60 GB (peso 0,6) |
| TypeScript | ~25–40 GB | ~12–20 GB |
| Rust | ~10–15 GB | ~6–9 GB |
| Kotlin | ~5–8 GB | ~3–5 GB |
| Shell | ~5–10 GB | ~2–5 GB |
| G-code | < 1 GB (casi todo salida de rebanadores, se descarta por tamaño) | ~0,1 GB |

`codeparrot/github-code-clean` es del orden de 100 millones de archivos (~1 TB de texto entre ~30
lenguajes, mezclados en los mismos Parquet): con los pesos de arriba aprovecha ~40–50 % de lo bajado.

**Por día, con la configuración por defecto** (`red_mb_por_dia: 1500`, `salida_mb_por_dia: 400`):

| Fuente | Red/día | Código filtrado/día |
|---|---:|---:|
| github-code-clean (1500 MB de Parquet ≈ 4–5 GB de texto) | ~1,4 GB | tope de **0,4 GB** (la salida limita antes que la red) |
| GitHub, 20 repos/día | ~50–150 MB | ~20–60 MB de archivos + ~3–10 MB de episodios (≈ 50–300 arreglos) |
| **Total** | ≈ 1,5 GB | ≈ **0,4 GB/día ≈ 12 GB/mes** |

`codigo_abierto` pesa 0,08–0,10 en los planes, así que un flujo de ~12 GB/mes alcanza para que el
modelo vea código nuevo sin repetir durante meses; subir `salida_mb_por_dia` sirve para juntar reserva
antes de la A770. Los episodios de GitHub son la parte valiosa y escasa: para
más, subir `max_repos_por_dia` (con `HOPE_GITHUB_TOKEN`).

## Uso

```bash
python -m hope.data.web_sources --diario            # web + código, la cuota de hoy
python -m hope.data.codigo_abierto                  # fuentes de código y licencias permitidas
python -m hope.data.codigo_abierto --diario         # solo código
python -m hope.data.codigo_abierto --revisar archivo.py   # qué haría el filtro con un archivo
python -m hope.train.run --plan configs/plan_gtx1060.yaml # usa data/pre/codigo_abierto si existe
```

## Límites conocidos

* El chequeo de llaves es aproximado (literales de regex en JS, macros raras de C): puede descartar algún
  archivo válido; no deja pasar archivos truncados.
* La lista de archivos de Hugging Face se pide una vez por fuente (`/api/datasets/…/tree`); si una
  carpeta tuviera más de 1000 archivos habría que paginar.
* Detección de secretos por patrones y entropía: reduce mucho, no garantiza el 100 %.
* The Stack v2 vía Software Heritage puede pedir credenciales de AWS según la política vigente del
  depósito; por eso está apagada y la v1 dedup es la opción recomendada con acceso restringido.
