
import sys
import pandas as pd
import numpy as np
import os
from SRutils import entrenar_evaluar_modelo, PATH_FEYNMAN, PATH_FEYNMAN_TEST, PATH_RESULTADOS_FEYNMAN
from feynman_functions import lista_funciones




if __name__ == '__main__':
    # si el directorio de resultados no existe, lo creo
    if not os.path.exists(PATH_RESULTADOS_FEYNMAN):
        os.makedirs(PATH_RESULTADOS_FEYNMAN)
    iteraciones = int(input("Ingrese la cantidad de ejecuciones independientes: "))
    # Medir el tiempo de ejecución
    tiempo_total, tiempos_iteraciones = entrenar_evaluar_modelo(iteraciones,PATH_FEYNMAN,PATH_FEYNMAN_TEST,PATH_RESULTADOS_FEYNMAN,lista_funciones,"feynman")
    
    print(f"Tiempo total de ejecución: {tiempo_total}")
    print(f"Tiempo promedio de ejecución: {np.mean(tiempos_iteraciones)}")
    
    # Local iterations 30:
    # Tiempo total de ejecución: 334.12499237060547
    # Tiempo promedio de ejecución: 333.39086651802063
    
    # Local iterations indefinido con 30 ejecuciones manuales:
    
    # Tiempo promedio de ejecución: 139.14308404922485
    # Tiempo total de ejecucion: 4110.426569219589
  