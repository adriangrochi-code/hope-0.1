%
(TISMA CNC - ejemplo G68.2: el mismo cubo de 40 mm, con plano de trabajo inclinado)
(Cada cara se programa como en 3 ejes: G68.2 define el plano, G53.1 gira la mesa hacia la herramienta)
(MSG, ejemplo G68.2: probarlo con el perfil Fresadora 5 ejes mesa A+C)
G21 G90 G17 G94 G54
T1 M6
S6000 M3
G0 Z20
(cara de arriba, sin inclinar)
G0 X-10 Y-10
G1 Z-0.5 F150
G1 X10 F400
G1 Y10
G1 X-10
G1 Y-10
G0 Z20
(cara lateral: plano girado 0 grados alrededor de Z e inclinado 90)
G69
G0 Z60
G68.2 X0 Y-20 Z-20 I0 J90 K0
G53.1
G0 X-10 Y-10 Z20
G1 Z-0.5 F150
G1 X10 F400
G1 Y10
G1 X-10
G1 Y-10
G0 Z20
(cara lateral: plano girado 90 grados alrededor de Z e inclinado 90)
G69
G0 Z60
G68.2 X20 Y0 Z-20 I90 J90 K0
G53.1
G0 X-10 Y-10 Z20
G1 Z-0.5 F150
G1 X10 F400
G1 Y10
G1 X-10
G1 Y-10
G0 Z20
(y un circulo de 10 mm en esta cara)
G0 X0 Y0
G0 X5
G1 Z-0.5 F150
G2 X5 Y0 I-5 J0 F300
G0 Z20
(cara lateral: plano girado 180 grados alrededor de Z e inclinado 90)
G69
G0 Z60
G68.2 X0 Y20 Z-20 I180 J90 K0
G53.1
G0 X-10 Y-10 Z20
G1 Z-0.5 F150
G1 X10 F400
G1 Y10
G1 X-10
G1 Y-10
G0 Z20
(cara lateral: plano girado 270 grados alrededor de Z e inclinado 90)
G69
G0 Z60
G68.2 X-20 Y0 Z-20 I270 J90 K0
G53.1
G0 X-10 Y-10 Z20
G1 Z-0.5 F150
G1 X10 F400
G1 Y10
G1 X-10
G1 Y-10
G0 Z20
G69
G0 Z60
G0 A0 C0
M5
M30
%
