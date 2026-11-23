%
(TISMA CNC - ejemplo de 4o eje: cilindro de 40 mm en un divisor A, alrededor de X)
(Z0 en el eje de giro: Z es el radio. Desbaste en helice a radio 19 y 8 ranuras a lo largo)
(MSG, ejemplo de 4o eje: probarlo con el perfil Router con 4o eje)
G21 G90 G17 G94 G54
T1 M6
S16000 M3
G0 Z30
G0 X0 Y0 A0
G1 Z19 F300
(desbaste: 20 vueltas mientras avanza 80 mm en X = paso de 4 mm por vuelta)
G1 X80 A7200 F1500
G0 Z30
G0 A0
(ranuras: una cada 45 grados)
G0 X5 A0
G1 Z17 F300
G1 X75 F900
G0 Z30
G0 X5 A45
G1 Z17 F300
G1 X75 F900
G0 Z30
G0 X5 A90
G1 Z17 F300
G1 X75 F900
G0 Z30
G0 X5 A135
G1 Z17 F300
G1 X75 F900
G0 Z30
G0 X5 A180
G1 Z17 F300
G1 X75 F900
G0 Z30
G0 X5 A225
G1 Z17 F300
G1 X75 F900
G0 Z30
G0 X5 A270
G1 Z17 F300
G1 X75 F900
G0 Z30
G0 X5 A315
G1 Z17 F300
G1 X75 F900
G0 Z30
G0 X0 A0
M5
M30
%
