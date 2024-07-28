
import sys
import pandas as pd
import numpy as np
import os
from SRutils import entrenar_evaluar_modelo, PATH_VLADISLAVLEVA, PATH_VLADISLAVLEVA_TEST, PATH_RESULTADOS_VLADISLAVLEVA
from vladislavleva_functions import lista_funciones as funciones_vladislavleva

if __name__ == '__main__':
    iteraciones = int(input("Ingrese la cantidad de ejecuciones independientes: "))
    funciones = []
    lista_funciones = str(input("Ingrese las funciones a utilizar separadas por coma: ")).split(",")
    for f in lista_funciones:
        # las funciones van de 1 a 8, pero en el código se manejan de 0 a 7
        # seleccionar los archivos de funciones de acuerdo a la entrada del usuario
        funciones.append(funciones_vladislavleva[int(f)-1])
    print("Funciones a utilizar: ", funciones)
    print("Código de funciones: ", [f.__code__ for f in funciones])
    entrenar_evaluar_modelo(iteraciones,PATH_VLADISLAVLEVA,PATH_VLADISLAVLEVA_TEST,PATH_RESULTADOS_VLADISLAVLEVA,funciones,"vladislavleva")
