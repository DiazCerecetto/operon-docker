
from math import exp


#C_A(C_Ain, t) = 2 C_Ain / (1 + - exp(-5.7t))
def ferreira1(Cain,t):
    return 2 * Cain / (1 + exp(-5.7 * t))

#C_B(C_Ain, t) = C_Ain(4.1104 exp(-6t) + 0.2163 -4.3267exp(-5.7t))
def ferreira2(Cain,t):
    return Cain * (4.1104 * exp(-6 * t) + 0.2163 - 4.3267 * exp(-5.7 * t))

#C_C(C_Ain, t) = C_Ain(-8.455 exp(-5t) - 16.442e(-6t) + 24.724 e(-5.7t) + 0.173)
def ferreira3(Cain,t):
    return Cain * (-8.455 * exp(-5 * t) - 16.442 * exp(-6 * t) + 24.724 * exp(-5.7 * t) + 0.173)

#C_D(C_Ain, t) = C_Ain(-1e(-2t) + 8.455e(-5t) + 12.3315(-6t)- 20.045 e(-5.7t) + 0.26)
def ferreira4(Cain,t):
    return Cain * (-1 * exp(-2 * t) + 8.455 * exp(-5 * t) + 12.3315 * exp(-6 * t) - 20.045 * exp(-5.7 * t) + 0.26)
lista_funciones = [ferreira1,ferreira2,ferreira3,ferreira4]
    