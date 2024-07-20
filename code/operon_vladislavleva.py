
import sys
import pandas as pd
import numpy as np
import os
from SRutils import entrenar_evaluar_modelo, PATH_VLADISLAVLEVA, PATH_VLADISLAVLEVA_TEST, PATH_RESULTADOS_VLADISLAVLEVA
from vladislavleva_functions import lista_funciones



if __name__ == '__main__':
    iteraciones = int(input("Ingrese la cantidad de ejecuciones independientes: "))
    entrenar_evaluar_modelo(iteraciones,PATH_VLADISLAVLEVA,PATH_VLADISLAVLEVA_TEST,PATH_RESULTADOS_VLADISLAVLEVA,lista_funciones,"vladislavleva")
