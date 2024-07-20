import inspect
from math import exp, cos, sin


# vladislavleva1: (e^((-x0-1)^2)) / (1.2 + (x1 - 2.5)^2)
def vladislavleva0(x0, x1):
    return exp(-((x0 + 1) ** 2)) / (1.2 + (x1 - 2.5) ** 2)

# vladislavleva2: (e^(-x1)) * x1^3 * cos(x1) * sin(x1) (cos(x1) * (sin(x1))^2 -1)
def vladislavleva1(x1):
    return (exp(-x1) * x1 ** 3 * cos(x1) * sin(x1) * (cos(x1) * (sin(x1) ** 2) - 1))

# vladislavleva3: (e^(-x1)) * x1^3 * cos(x1) * sin(x1) (cos(x1) * (sin(x1))^2 -1) (x2-5)
def vladislavleva2(x1, x2):
    return (exp(-x1) * x1 ** 3 * cos(x1) * sin(x1) * (cos(x1) * (sin(x1) ** 2) - 1) * (x2 - 5))

# vladislavleva4: 10/(5 + (x1-3)^2 + (x2-3)^2 + (x3-3)^2 + (x4-3)^2 + (x5-3)^2)
def vladislavleva3(x1, x2, x3, x4, x5):
    return 10 / (5 + (x1 - 3) ** 2 + (x2 - 3) ** 2 + (x3 - 3) ** 2 + (x4 - 3) ** 2 + (x5 - 3) ** 2)

def vladislavleva4(x1,x2,x3):
    return 30 * ((x1-1)*(x3-1) / ((x2**2) * (x1 - 10)))

# vladislavleva5: 6 * sin (x1) * cos(x2)
def vladislavleva5(x1, x2):
    return 6 * sin(x1) * cos(x2)

# vladislavleva6: (x1 - 3) * (x2-3) + 2 * sin((x1-4) * (x2-4))
def vladislavleva6(x1, x2):
    return (x1 - 3) * (x2 - 3) + 2 * sin((x1 - 4) * (x2 - 4))

# vladislavleva7: ((x1-3)^4 + (x2-3)^3 - (x2-3)) / ((x2-2)^4 + 10)
def vladislavleva7(x1, x2):
    return ((x1 - 3) ** 4 + (x2 - 3) ** 3 - (x2 - 3)) / ((x2 - 2) ** 4 + 10)


lista_funciones = [vladislavleva0,vladislavleva1, vladislavleva2, vladislavleva3, vladislavleva4, vladislavleva5, vladislavleva6, vladislavleva7] 

