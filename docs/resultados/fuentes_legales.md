# Fuentes legales de datos web para Hope

Decisión del usuario: adquirir datos de internet **solo por vías legales, identificado y controlado**, con un flujo diario constante. Implementación: `hope/data/web_sources.py`, registro en `configs/fuentes_web.yaml` (propiedad del usuario), permisos de red en `configs/web.yaml`, paso 8 de la app («Datos diarios»).

## Reglas comunes

- **Método preferido:** dump oficial > API oficial > lectura de páginas. La lectura de páginas queda para aprender a navegar, la zona visual y la inspección (unos miles de páginas), no para el volumen.
- **Identificación honesta:** User-Agent `HopeHN1/0.1 (...)` más el contacto que el usuario ponga en `configs/web.yaml` (`contacto:`; vacío por defecto, sin datos personales en el repositorio). Nunca se simula ser una persona ni se esquivan bloqueos.
- **Solo lectura:** GET/HEAD; sin formularios, sesiones, cookies, publicaciones ni subidas (guardián + código).
- **robots.txt, términos y ritmo:** `ritmo_s` por fuente (pausa mínima entre pedidos al mismo sitio), pausa entre fuentes y límites por hora y por día del guardián.
- **Claves:** solo en variables de entorno del usuario (`HOPE_CORE_API_KEY`, `HOPE_BRAVE_API_KEY`, `HOPE_GOOGLE_CSE_KEY`/`_CX`); se mandan en cabeceras o se agregan a la URL solo al pedir, y nunca quedan en la auditoría ni en la caché.
- **Licencia por documento:** cada documento guarda licencia, términos, URL y fecha en `provenance.jsonl`. Solo entra a `train.bin` lo que tiene una licencia de `licencias_entrenamiento` (por defecto CC BY, CC BY-SA, CC0, dominio público, PSF-2.0); **NC y ND nunca** salvo que el usuario las agregue. Lo demás queda como ficha consultable en `indice.jsonl`.
- **Deduplicación:** hash exacto (también contra lo ya consolidado), casi duplicados (Jaccard de 5-gramas) en las API y páginas, y entre agregadores de papers por DOI y título normalizado (`data/web/vistos.sqlite`: CORE contra arXiv).

## Tabla de fuentes

GB/día: cuota configurada por defecto (el máximo que se baja; el texto útil es menor). Ritmo: pausa mínima entre pedidos.

| Fuente | Licencia | Método | Ritmo | GB/día | Entrenamiento | Notas |
|---|---|---|---|---|---|---|
| Wikipedia es (`wikipedia_es_hf`) | CC BY-SA 4.0 | dump: Parquet `wikimedia/wikipedia` (HF), 13 archivos | 5 s | 1,0 | sí | también al índice local `⟦buscar_wiki⟧`; ≈ 3 GB de texto en total |
| Wikipedia en (`wikipedia_en_hf`) | CC BY-SA 4.0 | dump: Parquet (HF), 41 archivos | 5 s | 1,0 | sí | desactivada hasta terminar la de español; ≈ 20 GB de texto |
| Wikipedia es (`wikipedia_es_dump`) | CC BY-SA 4.0 | dump oficial XML de dumps.wikimedia.org | 5 s | 1,0 | sí | alternativa más actual; descarga reanudable, wikitexto → texto |
| Wikilibros / Wikisource / Wikcionario | CC BY-SA 4.0 (Wikisource: muchas obras de dominio público) | API MediaWiki (`allpages` + `extracts`) | 2 s | 0,02–0,05 | sí | 300 páginas/día cada una |
| Proyecto Gutenberg | dominio público (EE. UU.) | mirror oficial (no el sitio principal, como pide su política de robots) | 3 s | 0,2 | sí | se quita el encabezado/pie de la licencia; los libros marcados con copyright se omiten |
| arXiv | CC0 (metadatos y resúmenes) | API oficial | 3 s (arXiv pide ≥ 3 s) | 0,02 | resúmenes sí | el texto completo no se baja: depende de la licencia de cada artículo |
| CORE (core.ac.uk) | **por artículo** | API v3 con clave gratuita (`HOPE_CORE_API_KEY`) | 6 s | 0,2 | solo con licencia apta | el resto, solo ficha en `indice.jsonl`; dedup contra arXiv por DOI/título |
| Stack Exchange | CC BY-SA 4.0 | dump oficial (archive.org, 7z) | 5 s | 0,5 | sí | pregunta + respuesta aceptada; procesar requiere `py7zr` |
| docs.python.org | PSF-2.0 | páginas (siguiendo enlaces del tutorial) | 2 s | 0,02 | sí | 200 páginas/día |
| MDN | CC BY-SA 2.5 | páginas | 2 s | 0,02 | sí | 200 páginas/día |
| Conjuntos de Hugging Face | la de cada conjunto (verificar la ficha) | dump: Parquet | 5 s | 0,5 | si la licencia es apta | plantilla `hf_ejemplo`, desactivada |

Cuota total por defecto: **2 GB y 3000 documentos por día** (`diario:` en `configs/fuentes_web.yaml`).

## Atribución (CC BY-SA)

CC BY-SA exige reconocer la autoría, indicar la licencia y enlazar la fuente. Por eso:

- cada documento conserva **título, URL y licencia** en `provenance.jsonl` (Wikipedia: la URL del artículo, cuyo historial lista a los autores);
- `⟦buscar_wiki⟧` devuelve siempre el título, la URL y la licencia junto al fragmento;
- si se publica un modelo o un conjunto derivado, hay que acompañarlo de esa lista de fuentes y licencias. Si un texto generado reproduce un fragmento sustancial de una fuente CC BY-SA, se cita con su URL.

## Uso

```
python -m hope.data.web_sources --fuentes            # tabla de fuentes, licencias y estado
python -m hope.data.web_sources --diario              # la cuota de hoy
python -m hope.data.web_sources --diario --repetir    # queda corriendo, una vez por día (03:17)
python -m hope.memory.wiki_index buscar --db data/wiki.sqlite "ley de Ohm"
```

Salida: `data/web/<fecha>/<fuente>/` (`train.bin`, `val.bin`, `provenance.jsonl`, `indice.jsonl`, `manifest.json`, `informe.json` del día) y `data/pre/web_diario/<fuente>/` con la suma de todos los días, para agregar al plan (`web_wiki: data/pre/web_diario/wikipedia_es_hf:0.1`). Las descargas grandes se reanudan con `Range`: cada día baja un tramo dentro de la cuota.

Para que funcione: poner los dominios de las fuentes activas en `dominios_permitidos` de `configs/web.yaml` (huggingface.co, dumps.wikimedia.org, es.wikibooks.org, export.arxiv.org, api.core.ac.uk, el mirror de Gutenberg, archive.org…) con `modo: lista_permitida`, o aprobar cada acceso en el paso 6.

## Límites conocidos

- La conversión de wikitexto a texto es aproximada (sin dependencias): plantillas, referencias y tablas se quitan.
- arXiv: solo metadatos y resúmenes (CC0). Para texto completo habría que leer la licencia de cada artículo (OAI-PMH `arXivRaw`) y bajar solo CC BY/CC0; no está implementado.
- CORE: el nombre del campo de licencia puede variar entre versiones de la API (se leen `license`, `licence` o `rights`); sin licencia reconocible, el artículo queda solo como ficha.
- Las pruebas (`tests/test_web_sources.py`) usan un servidor HTTP local y respuestas simuladas: no validan contra los servicios reales.
