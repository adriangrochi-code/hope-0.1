# Red distribuida de cómputo para Hope (diseño, 2026-11-12)

**Objetivo del usuario:** una aplicación tipo Claude Code donde la gente se conecta, trabaja con un agente de código e ingeniería, y la potencia viene de una red de computadoras de los propios usuarios. Cada usuario recibe capacidad **según el cómputo que aporta**, más planes pagos mensuales. **Todo pasa por un servidor central** que coordina, verifica y cobra: una red tipo peer-to-peer, pero con coordinación central. Hope hoy es **cerrado** (pesos privados); más adelante se abrirá. Prioridad: **seguridad máxima**.

Es un proyecto aparte de la red neuronal, que se apoya en `hope serve` (§11.24). Las decisiones de red, servicios externos, pagos y publicación son del usuario (Carta Fundamental).

---

## 1. Principios de diseño

1. **Las herramientas corren en la máquina del usuario, no en la red.** Como en Claude Code y TISMA Code, el agente lee, edita, ejecuta y prueba en la computadora de quien hace el pedido. A la red solo van las **llamadas al modelo** (texto entra, texto sale). Así los repos y los archivos nunca salen enteros: solo lo que entra en cada llamada.
2. **Cada nodo ejecuta el modelo entero.** Partir un modelo entre máquinas de distintas redes suma la latencia de internet a cada byte generado y lo vuelve muy lento (Petals). Hope entra en una GPU de 6 GB, así que cada llamada va completa a un solo nodo.
3. **Todo pasa por el servidor central:** autenticación, ruteo, verificación, contabilidad y pagos. Usuarios y proveedores **nunca se conectan entre sí**: no ven las IP del otro, y el servidor puede aplicar políticas y filtros.
4. **Desconfianza por defecto:** se asume que cualquier proveedor puede mentir, copiar o mirar. Todo se diseña para detectarlo, limitar el daño o evitar que tenga algo valioso para robar.

---

## 2. El problema central: modelo cerrado en máquinas ajenas

**Con GPUs de consumo no hay forma técnica de impedir que un proveedor copie los pesos que ejecuta.** Cifrarlos no alcanza (tienen que estar descifrados en su memoria para correr), y las 1060, 3060 o A770 no tienen cómputo confidencial (enclaves en GPU). Partir el modelo entre proveedores tampoco lo protege: cada uno se queda con su parte, y además se vuelve muy lento.

**Propuesta por etapas:**

| Etapa | Qué corre en los proveedores | Dónde corre Hope |
|---|---|---|
| **A. Hope cerrado** | **modelos abiertos** con licencia que permita servirlos (Qwen Apache-2.0, etc.) para las tareas que los usen, y **verificadores, recuperación y herramientas** (redes chicas, poco valiosas de robar) | **solo en nodos del dueño** o de **proveedores de confianza con contrato** (acuerdo de confidencialidad, máquinas revisadas) |
| **B. Hope parcialmente abierto** | una versión **chica destilada y abierta** de Hope | la versión grande sigue en nodos de confianza |
| **C. Hope abierto** | Hope completo | en toda la red |

Así la red crece desde el día uno sin poner en riesgo los pesos de Hope mientras sean cerrados.

---

## 3. Privacidad de los prompts

**Fragmentar los prompts protege poco con modelos de lenguaje.** Para responder bien, el nodo necesita ver el contexto completo de esa llamada. Si se parte el modelo por capas, el que corre las primeras ve la entrada, y las activaciones intermedias se pueden invertir para reconstruir el texto. Lo que sí sirve:

| Medida | Qué protege |
|---|---|
| **Redacción en el cliente** antes de enviar: claves, tokens, contraseñas, rutas personales, mails y números se reemplazan por marcadores y se restauran al volver la respuesta | secretos y datos personales, que son lo más dañino si se filtran |
| **Contexto mínimo:** el agente local manda solo los fragmentos necesarios para cada llamada (recuperación y notas de §11.9–11.10), no el repo entero | reduce lo expuesto en cada llamada |
| **Partir la tarea, no el prompt:** subtareas independientes (por ejemplo, «escribí esta función» y «revisá este test») van a nodos distintos, sin que ninguno vea el proyecto completo | dificulta reconstruir el proyecto entero desde un solo nodo |
| **Niveles de privacidad** elegidos por el usuario: «abierto» (cualquier nodo verificado), «confiable» (solo nodos con contrato), «privado» (solo su propia máquina o nodos del dueño) | el usuario decide según la sensibilidad |
| **Modo local:** si el usuario tiene GPU, sus pedidos privados corren en su máquina y su cómputo sobrante va a la red | privacidad total para lo propio |
| **Política clara:** qué ve un proveedor, cuánto tiempo se guarda (los proveedores no guardan nada y el servidor central guarda lo mínimo), cumplimiento de la Ley 25.326 de datos personales | transparencia y legalidad |

---

## 4. Componentes

| Componente | Qué hace |
|---|---|
| **Cliente** (app de escritorio, basada en el harness de TISMA Code / F9) | agente con herramientas locales, redacción de secretos, elección de nivel de privacidad, cuenta y créditos |
| **Servidor central** | autenticación, cola y ruteo de llamadas, verificación, reputación, contabilidad de créditos, pagos, filtros de abuso, registro de auditoría encadenado (como `guardian/audit.py`) |
| **Agente de proveedor** | instala y actualiza el motor (binarios firmados), corre la inferencia aislada (sin red salvo el túnel al servidor, sin acceso a archivos, sin ejecutar código ajeno), mide su capacidad y reporta el trabajo |
| **Motor de inferencia** | `hope serve` (§11.24) o un servidor de modelos abiertos, con la API compatible con OpenAI |
| **Túnel** | conexión **saliente** del proveedor al servidor (WebSocket/QUIC con TLS mutuo): atraviesa NAT sin abrir puertos ni exponer IPs |

---

## 5. Verificar que el trabajo es real

1. **Repetición al azar:** un porcentaje de las llamadas (por ejemplo 2–5 %, más al principio o con poca reputación) se ejecuta también en otro nodo o en uno del dueño, con decodificación determinista o semilla fija, y se comparan los resultados.
2. **Pedidos trampa** con respuesta conocida, mezclados con los reales e indistinguibles.
3. **Pruebas de capacidad:** el nodo declara su GPU, y el servidor lo verifica con tareas medidas (velocidad, memoria).
4. **Reputación** por proveedor: los fallos de verificación bajan su nivel, retienen pagos y llegan hasta el bloqueo. Los nuevos empiezan con pedidos «abiertos» y poco valor.
5. **Versiones firmadas:** el servidor solo acepta motores y modelos con hash conocido. Sin hardware confiable no se puede garantizar al 100 %, pero las verificaciones 1–2 detectan trampas.

---

## 6. Créditos según el cómputo aportado

- **Unidad de trabajo:** bytes generados y procesados por modelo, normalizados por su costo (activos por byte, de `hope_modelo.json`).
- **Ganás créditos** con trabajo **verificado**, no con tiempo conectado.
- **Gastás créditos** con tus llamadas. Los planes pagos mensuales suman créditos y prioridad.
- **Prioridad justa:** quien aporta más tiene prioridad en horas pico (como los créditos de BOINC o el *tit-for-tat* de BitTorrent), con un mínimo garantizado para todos.
- **Pago a proveedores:** opcional, en dinero, por trabajo verificado y menos la comisión de la plataforma. Requiere identificación del proveedor, facturación e impuestos (en Argentina, ARCA). Se puede hacer con pasarelas comunes que permiten pagar a terceros (Mercado Pago, Stripe); no hace falta cripto.

---

## 7. Seguridad de los proveedores

El proveedor también corre riesgos al prestar su máquina:
- **Solo inferencia:** el agente de proveedor nunca ejecuta código del usuario. Las herramientas corren en la máquina del usuario.
- **Aislamiento:** sandbox (bubblewrap en Linux, AppContainer o similar en Windows), sin acceso a sus archivos, red solo al servidor central, límites de VRAM, temperatura y horario elegidos por el proveedor.
- **Actualizaciones firmadas** y verificadas antes de instalarse.

---

## 8. Abuso y responsabilidad legal

- **Filtros en el servidor central** (pedidos y respuestas) con políticas de uso: nada de malware, fraude ni contenido ilegal (la ciberseguridad de Hope es solo defensiva, §11.21).
- **Términos de uso** para usuarios y proveedores, que definan quién responde por qué.
- **Límites de uso, cuentas verificadas** para los planes pagos y registro de auditoría mínimo.
- **Asesoramiento legal antes de cobrar** (comercio electrónico, datos personales, pagos a terceros, impuestos).

---

## 8.1 Los modelos se descargan solo de nosotros (decisión del usuario, 2026-11-12)

**Regla:** la app solo ejecuta modelos **distribuidos por la plataforma**. El proveedor los descarga desde nuestros servidores, y si no tiene el modelo exacto que asignamos, no recibe trabajo.

**Cómo se hace cumplir:**

| Pieza | Qué hace |
|---|---|
| **Catálogo firmado** | el servidor publica qué modelos existen (id, versión, hash SHA-256 de cada archivo, licencia, VRAM mínima). El catálogo va firmado con la clave de la plataforma |
| **Descarga solo desde la plataforma** | el agente de proveedor baja los modelos de nuestro CDN, verifica hash y firma, y se niega a cargar cualquier archivo que no esté en el catálogo. No acepta rutas locales ni modelos de otras fuentes |
| **Asignación por modelo** | el servidor solo manda llamadas de un modelo a nodos que declararon tenerlo, con el hash correcto |
| **Prueba de que corre ese modelo** | como conocemos el modelo exacto, los pedidos trampa y la repetición con decodificación determinista (§5) **detectan** a un nodo que corre otro modelo, una versión cuantizada distinta o un atajo: su salida no coincide byte a byte (o en la distribución, si se muestrea). Si no coincide, no cobra y pierde reputación |
| **Desafíos de identidad del modelo** | de vez en cuando, el servidor pide la salida para entradas secretas y la compara con la salida de referencia de ese hash |
| **Cifrado en disco atado al nodo** (opcional) | los pesos se guardan cifrados con una clave por nodo que entrega el servidor al iniciar la sesión. Copiar la carpeta a otra máquina no sirve directamente |

**Lo que esto logra y lo que no:**
- ✅ **Que nadie calcule en la red con un modelo que no sea el nuestro:** se puede garantizar en la práctica, por la verificación de salidas.
- ✅ **Control de versiones y licencias:** todos corren la misma versión, actualizada y firmada.
- ❌ **Que un proveedor no copie los pesos que descargó:** **no** se puede garantizar. Si el modelo corre en su máquina, en algún momento está descifrado en su memoria, y alguien con conocimientos puede extraerlo. El cifrado atado al nodo solo lo hace más difícil.
- **Consecuencia para Hope mientras sea cerrado:** se mantiene la etapa A de §2. Los modelos que se distribuyen a cualquier proveedor son **abiertos**: se pueden redistribuir respetando su licencia (por ejemplo, Apache-2.0 exige incluir la licencia y los avisos). **Hope se distribuye solo a nodos de confianza con contrato** hasta que se abra. El contrato es lo que protege legalmente lo que la técnica no puede proteger.

## 9. Entrenamiento distribuido (más adelante)

Con métodos de poca comunicación (DiLoCo, Google 2023; OpenDiLoCo; INTELLECT-1, Prime Intellect 2024, que entrenó 10B con nodos en varios continentes), los proveedores podrían entrenar **ramas** de Hope (§11.13). Es más difícil de verificar que la inferencia y, con Hope cerrado, expone los pesos. Por eso queda para la etapa C o para nodos de confianza.

---

## 10. Plan de construcción

| Paso | Qué | Con quién |
|---|---|---|
| R1 | Prototipo con **tus propias máquinas** (1060, A770, notebook): servidor central, agente de proveedor, túnel, ruteo, verificación por repetición. Sin cobrar | solo el dueño |
| R2 | Cliente tipo Claude Code: el harness de TISMA Code apuntando al servidor central, con redacción de secretos y niveles de privacidad | dueño + sesión de TISMA |
| R3 | Créditos por cómputo verificado y reputación | grupo cerrado de conocidos, modelos abiertos |
| R4 | Revisión de seguridad externa (sandbox, túnel, firmas, filtros) | antes de abrir |
| R5 | Términos legales, pagos y facturación | con asesoramiento |
| R6 | Apertura gradual al público (etapa A: Hope solo en nodos de confianza) | — |
| R7 | Etapas B y C cuando Hope se abra | — |

**Métricas:** latencia al primer byte y bytes por segundo por nivel de privacidad, tasa de trampas detectadas, disponibilidad de la red, y créditos ganados frente a consumidos por usuario.
