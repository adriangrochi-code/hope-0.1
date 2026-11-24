%
(TISMA CNC - ejemplo 3+2: cubo de 40 mm en una mesa basculante A+C)
(Cero de pieza G54 en el centro de la cara de arriba; el centro de giro de A esta 20 mm abajo)
(Graba un cuadrado de 20 mm en la cara de arriba y en las 4 caras laterales)
(MSG, ejemplo 3+2: probarlo con el perfil Fresadora 5 ejes mesa A+C)
G21 G90 G17 G94 G54
T1 M6
S6000 M3
G0 Z20
(cara de arriba)
G0 A0 C0
G0 X-10 Y-10
G1 Z-0.5 F150
G1 X10 F400
G1 Y10
G1 X-10
G1 Y-10
G0 Z20
(cara lateral: C0 y la cuna inclinada A-90)
G0 C0
G0 A-90
G0 X-10 Y-10
G1 Z-0.5 F150
G1 X10 F400
G1 Y10
G1 X-10
G1 Y-10
G0 Z20
(cara lateral: C90 y la cuna inclinada A-90)
G0 C90
G0 A-90
G0 X-10 Y-10
G1 Z-0.5 F150
G1 X10 F400
G1 Y10
G1 X-10
G1 Y-10
G0 Z20
(cara lateral: C180 y la cuna inclinada A-90)
G0 C180
G0 A-90
G0 X-10 Y-10
G1 Z-0.5 F150
G1 X10 F400
G1 Y10
G1 X-10
G1 Y-10
G0 Z20
(cara lateral: C270 y la cuna inclinada A-90)
G0 C270
G0 A-90
G0 X-10 Y-10
G1 Z-0.5 F150
G1 X10 F400
G1 Y10
G1 X-10
G1 Y-10
G0 Z20
G0 A0 C0
M5
M30
%
