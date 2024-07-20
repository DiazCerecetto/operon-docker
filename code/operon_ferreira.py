
import sys
import pandas as pd
import numpy as np
import os
from SRutils import entrenar_evaluar_modelo, PATH_FERREIRA, PATH_FERREIRA_TEST, PATH_RESULTADOS_FERREIRA
from ferreira_functions import lista_funciones




if __name__ == '__main__':
    iteraciones = int(input("Ingrese la cantidad de ejecuciones independientes: "))
    # Si la carpeta resultados no existe, crearla
    os.makedirs(PATH_RESULTADOS_FERREIRA, exist_ok=True)
    entrenar_evaluar_modelo(iteraciones, PATH_FERREIRA, PATH_FERREIRA_TEST, PATH_RESULTADOS_FERREIRA, lista_funciones, "Ferreira")