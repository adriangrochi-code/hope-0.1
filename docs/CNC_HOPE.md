# Hope CNC: control CNC moderno como herramienta interna de Hope

**Objetivo:** reemplazar Mach3 con un programa que funcione igual en una PC vieja y en una
moderna, que **revise el programa antes de mover la máquina** y en el que Hope pueda ayudar
(revisar, simular, explicar errores) pero **nunca mover la máquina sin permiso humano**.

Código: `hope/cnc/`. Pruebas: `tests/test_cnc.py`. Máquina de ejemplo: `configs/cnc_maquina.yaml`.

## 1. Por qué Mach3 envejeció y qué se cambia

| Mach3 | Hope CNC |
|---|---|
| Genera los pulsos desde Windows por el puerto paralelo: depende de la latencia de la PC y de un driver de 32 bits | La PC **no genera pulsos**. Los genera un controlador de tiempo real (grblHAL, FluidNC, LinuxCNC con Mesa o con puerto paralelo) |
| Lo que se revisa no es exactamente lo que se ejecuta (macros, modos de Mach3) | Un **solo intérprete** (el de Hope) decide la trayectoria; al controlador le llega G-code normalizado y simple |
| Los errores aparecen con la máquina en marcha | **Revisión previa**: límites de recorrido (arcos incluidos), husillo, F faltante, códigos no soportados, tiempo real con aceleración |
| Solo Windows | Python en Windows, Linux o Raspberry Pi; el controlador puede ser USB o Ethernet |

## 2. Arquitectura

```
 G-code (CAM, Mach3)        configs/cnc_maquina.yaml
        │                            │
        ▼                            ▼
  gcode.py: intérprete ──► Program (ops en coordenadas de máquina, diagnósticos por línea)
        │
        ├─► planner.py: lookahead + trapecios ──► tiempo exacto, velocidades por tramo
        ├─► check.py:   límites, husillo, final ──► informe («LISTO» / «NO se puede enviar»)
        └─► dialect.py: G-code normalizado (grbl | linuxcnc)
                    │
        controller.py ── guardián: hardware.control (aprobación humana, ligada al hash)
                    │
        backends/: sim │ grbl (serie/TCP, conteo de caracteres) │ linuxcnc (API Python)
                    │
          tiempo real: grblHAL / FluidNC / LinuxCNC ──► drivers ──► motores
```

La **seta de emergencia** corta físicamente la potencia de los drivers. Ningún software, tampoco
este, la reemplaza.

## 3. Equipos viejos y modernos

| Equipo | Camino recomendado | Configuración (`controlador:`) |
|---|---|---|
| PC vieja con **puerto paralelo** y la placa de conexión de Mach3 | LinuxCNC (ISO de linuxcnc.org, StepConf); mismos cables | `tipo: linuxcnc` |
| PC vieja **sin** buen tiempo real, o laptop | grblHAL/FluidNC por USB: la PC solo envía líneas, así que sirve casi cualquier equipo | `tipo: grbl`, `puerto: /dev/ttyUSB0` o `COM3` |
| PC moderna o mini PC | grblHAL por Ethernet (Teensy 4.1, placas BTT/PicoCNC), o LinuxCNC con Mesa 7i96S/7i76E | `tipo: grbl`, `puerto: tcp://IP:23`, `bufer: 1024` |
| Raspberry Pi 4/5 | LinuxCNC (imagen para Pi) o grblHAL por USB | igual que arriba |
| Sin hardware | Simulador | `tipo: sim` |

Para pasar los datos de Mach3: en *Config → Motor Tuning*, «Steps per» → `pasos_mm`,
«Velocity» → `vel_max` (mm/min) y «Acceleration» → `acel` (mm/s²). Los límites de *Homing/Limits*
van en `min`/`max`.

## 4. Qué hay en la v0

* **Intérprete** (`gcode.py`): G0/G1/G2/G3 (IJK, R, vueltas con P, helicoidales), G4, G17–G19,
  G20/G21, G90/G91, G90.1/G91.1, G28/G30, G53, G54–G59, G10 L2/L20, G92/G92.1, G43 Hn con tabla de
  herramientas, G43.1, G49, G61/G64, G98/G99 y los ciclos de taladrado G81/G82/G83/G73 (expandidos,
  así funcionan también en grbl). M0/M1, M2/M30, M3/M4/M5, M6, M7/M8/M9, comentarios, `(MSG, …)` y
  borrado de bloque.
  Los códigos de Mach3 que no son estándar (G12/G13, G15/G16, G50/G51, G52, M98/M99) dan un
  **error claro** con la alternativa. Las macros de usuario (M100 en adelante) dan un aviso.
* **Planificador** (`planner.py`): el mismo modelo que grbl (desviación de unión, pasada hacia
  atrás y hacia adelante, trapecios), con tiempos exactos. Comprobado contra cálculos a mano.
* **Revisión** (`check.py`): envolvente en máquina y G54, límites de recorrido por línea, corte
  con husillo apagado, programa sin M2/M30, tiempo, distancias y herramientas.
* **Traducción** (`dialect.py`): G-code normalizado por dialecto. Al volver a interpretar la
  traducción salen **los mismos movimientos y el mismo tiempo** (hay una prueba de ida y vuelta).
* **Controladores**:
  * `sim`;
  * `grbl` (serie o TCP, conteo de caracteres, estados, `$#`, errores y alarmas explicados, `!`/`~`/reinicio);
  * `linuxcnc` (modo automático por la API).
* **Guardián**: ejecutar en una máquina real es `hardware.control` (nivel humano). La aprobación
  queda ligada a:
  * el hash del programa;
  * la máquina y el controlador;
  * la envolvente calculada con los **orígenes reales** leídos del controlador.

  Cada aprobación se usa una vez. Un programa con errores nunca llega al controlador.
* **Pausas**: M0 y el cambio manual de herramienta en grbl esperan al operador (Enter para
  seguir, «a» para abortar con reinicio).
* **Herramienta interna de Hope**: `⟦cnc: pieza.nc | maquina=… dialecto=grbl⟧` revisa en el
  sandbox y **no mueve nada** (`hte_cnc_v1`).

### Uso

```bash
python -m hope.cnc revisar pieza.nc --maquina configs/cnc_maquina.yaml
python -m hope.cnc traducir pieza.nc --dialecto grbl --salida pieza_grbl.nc
python -m hope.cnc estado
python -m hope.cnc enviar pieza.nc                       # crea la solicitud en el guardián
python -m guardian.cli approve <id> --by <tu-nombre>
python -m hope.cnc enviar pieza.nc --aprobacion <id>
```

Para grbl por puerto serie hace falta `pip install pyserial`. Por TCP no hace falta nada más.

## 5. Límites honestos de la v0

* **Sin probar en una máquina real.** El protocolo de grbl está probado contra un grbl simulado y
  LinuxCNC solo en el caso «no instalado». Primera prueba real: en el aire, con Z arriba y sin
  herramienta.
* Sin parámetros ni expresiones (`#1`, `[ ]`), subprogramas, compensación de radio G41/G42 ni
  tiempo inverso G93.
* Ciclos fijos solo en G17 y G90.
* Sin sondeo (G38.x), referenciado (`$H`) ni jog desde Hope CNC. Por ahora se hacen con el
  controlador o con un enviador como gSender.
* El tiempo usa el modelo trapezoidal de grbl. Con curva S (grblHAL, LinuxCNC 2.9 o posterior) la
  máquina real puede tardar un poco más.

## 6. Próximos pasos

1. **Interfaz web en `hope/app/`** (ya hay servidor local con PIN y móvil):
   * vista 3D de la trayectoria coloreada por velocidad;
   * posición en vivo (DRO), jog, referenciado y puesta a cero;
   * botón de emergencia de software, que es un `!` más reinicio y no reemplaza a la seta;
   * aprobación del guardián desde la misma pantalla.
2. **Importar el perfil XML de Mach3.** Necesita un `.xml` de ejemplo de tu máquina para no
   adivinar los nombres de los campos.
3. Sondeo G38.2 y herramienta de medición de largo; G41/G42; parámetros `#` y subprogramas
   `O…sub`.
4. Probar en hardware: grblHAL por USB y Ethernet, y LinuxCNC con puerto paralelo.
5. Opcional: firmware propio en RP2040/RP2350 (PIO para los pulsos, curva S). Solo vale la pena si
   grblHAL se queda corto.
6. Hope como asistente: explicar los errores del informe, proponer correcciones al programa o al
   postprocesador del CAM, y estimar el desgaste por herramienta. Siempre como propuesta para que
   la persona la revise.
