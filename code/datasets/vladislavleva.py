from math import exp, cos, sin


# vladislavleva1: (e^((-x0-1)^2)) / (1.2 + (x1 - 2.5)^2)
def vladislavleva1(x0, x1):
    return exp(-((x0 + 1) ** 2)) / (1.2 + (x1 - 2.5) ** 2)

# vladislavleva2: (e^(-x1)) * x1^3 * cos(x1) * sin(x1) (cos(x1) * (sin(x1))^2 -1)
def vladislavleva2(x1):
    return (exp(-x1) * x1 ** 3 * cos(x1) * sin(x1) * (cos(x1) * (sin(x1) ** 2) - 1))

# vladislavleva3: (e^(-x1)) * x1^3 * cos(x1) * sin(x1) (cos(x1) * (sin(x1))^2 -1) (x2-5)
def vladislavleva3(x1, x2):
    return (exp(-x1) * x1 ** 3 * cos(x1) * sin(x1) * (cos(x1) * (sin(x1) ** 2) - 1) * (x2 - 5))

# vladislavleva4: 10/(5 + (x1-3)^2 + (x2-3)^2 + (x3-3)^2 + (x4-3)^2 + (x5-3)^2)
def vladislavleva4(x1, x2, x3, x4, x5):
    return 10 / (5 + (x1 - 3) ** 2 + (x2 - 3) ** 2 + (x3 - 3) ** 2 + (x4 - 3) ** 2 + (x5 - 3) ** 2)

def vladislavleva5(x1,x2,x3):
    return 30 * ((x1-1)*(x3-1) / ((x2**2) * (x1 - 10)))

# vladislavleva5: 6 * sin (x1) * cos(x2)
def vladislavleva6(x1, x2):
    return 6 * sin(x1) * cos(x2)

# vladislavleva6: (x1 - 3) * (x2-3) + 2 * sin((x1-4) * (x2-4))
def vladislavleva7(x1, x2):
    return (x1 - 3) * (x2 - 3) + 2 * sin((x1 - 4) * (x2 - 4))

# vladislavleva7: ((x1-3)^4 + (x2-3)^3 - (x2-3)) / ((x2-2)^4 + 10)
def vladislavleva8(x1, x2):
    return ((x1 - 3) ** 4 + (x2 - 3) ** 3 - (x2 - 3)) / ((x2 - 2) ** 4 + 10)


lista_funciones = [vladislavleva1, vladislavleva2, vladislavleva3, vladislavleva4, vladislavleva5, vladislavleva6, vladislavleva7, vladislavleva8] 
vladislavlevaA = "add,sub,mul,square,constant,variable,div"
vladislavlevaB = "add,sub,mul,square,exp,constant,variable,div"
vladislavlevaC = "add,sub,mul,square,exp,cos,sin,constant,variable,div"
lista_funciones_function_sets = [
    {"funcion":vladislavleva1, "fset":vladislavlevaB},
    {"funcion":vladislavleva2, "fset":vladislavlevaC},
    {"funcion":vladislavleva3, "fset":vladislavlevaC},
    {"funcion":vladislavleva4, "fset":vladislavlevaA},
    {"funcion":vladislavleva5, "fset":vladislavlevaA},
    {"funcion":vladislavleva6, "fset":vladislavlevaB},
    {"funcion":vladislavleva7, "fset":vladislavlevaC},
    {"funcion":vladislavleva8, "fset":vladislavlevaA}
]