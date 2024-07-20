
import sys
import pandas as pd
import numpy as np
import os
from SRutils import entrenar_evaluar_modelo, PATH_FEYNMAN, PATH_FEYNMAN_TEST, PATH_RESULTADOS_FEYNMAN
from feynman_functions import lista_funciones




if __name__ == '__main__':
    iteraciones = int(input("Ingrese la cantidad de ejecuciones independientes: "))
    entrenar_evaluar_modelo(iteraciones,PATH_FEYNMAN,PATH_FEYNMAN_TEST,PATH_RESULTADOS_FEYNMAN,lista_funciones,"feynman")
    
  