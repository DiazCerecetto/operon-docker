import os
import numpy as np
import pandas as pd
from .config import PATH_FERREIRA_TEST, PATH_FERREIRA_TRAIN, PATH_FERREIRA, generar_datos_modelo
from math import exp
########################################################
############## FUNCIONES DE FERREIRA ###################
########################################################

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

def ferreira5(Cain,t):
    return Cain * (-1 * exp(-2 * t) + 8.455 * exp(-5 * t) + 12.3315 * exp(-6 * t) - 20.045 * exp(-5.7 * t) + 0.26)

def ferreira5(x1):
    return x1 / (x1 + 3.7)

def ferreira6(x1):
    return 2 * x1 / (x1 + 3.7)

def ferreira7(x1):
    return 3 * x1 / (x1 + 3.7)

def ferreira8(x1, x2, x3):
    return ((x1 / (x1 + 3.7)) * x1 + (2 * x2 / (x2 + 3.7)) * x2 + (3 * x3 / (x3 + 3.7)) * x3) / (x1 + x2 + x3)

ferreira="add,sub,mul,div,exp,constant,variable"

lista_funciones_function_sets = [
    {"funcion":ferreira1, "fset":ferreira},
    {"funcion":ferreira2, "fset":ferreira},
    {"funcion":ferreira3, "fset":ferreira},
    {"funcion":ferreira4, "fset":ferreira},
    {"funcion":ferreira5, "fset":ferreira},
    {"funcion":ferreira6, "fset":ferreira},
    {"funcion":ferreira7, "fset":ferreira},
    {"funcion":ferreira8, "fset":ferreira},
]
########################################################
########################################################
########################################################
  
def generar_ecuaciones_ferreira():
    # Para garantizar reproducibilidad
    SEED_TRAIN = 1234
    SEED_TEST = 5678    
    
    # Generar puntos para train y test
    df = pd.read_csv(os.path.join(PATH_FERREIRA, 'FerreiraEquations.csv'))

    cantidad_train = int(input('Ingrese la cantidad de puntos para train: '))
    cantidad_test = int(input('Ingrese la cantidad de puntos para test: '))
    generar_datos_modelo(cantidad_train, PATH_FERREIRA_TRAIN, SEED_TRAIN,df, lista_funciones, 'ferreira')
    generar_datos_modelo(cantidad_test, PATH_FERREIRA_TEST, SEED_TEST,df, lista_funciones, 'ferreira')

    print('Listo!')  