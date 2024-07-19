
import sys
import pandas as pd
import numpy as np
import os
from SRutils import *
from feynman_functions import lista_funciones



def main(iteraciones):
    print("****************************")
    print("* Operon desde archivo csv *")
    print("****************************")
    
    predicciones = []
    # Leer los archivos csv, en el directorio Feynman
    # para cada archivo CSV, predecir y guardar el modelo
    # Luego, comparar los modelos obtenidos con los modelos reales
    # indicar por consola la cantidad de iteraciones
    
    for file in os.listdir(PATH_FEYNMAN):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(PATH_FEYNMAN, file))
            est = SymbolicRegressor(**default_params)
            est, m = predecir_desde_csv(est, df,file,iteraciones)
            predicciones.append(est)
    print("***************************")
    
    
    # Comparar los modelos obtenidos con los modelos reales
    i = 0
    for i, f in enumerate(lista_funciones):
        print("***************************")
        print("Original: " + f.__name__)
        print("Modelo: ")
        print(predicciones[i].get_model_string())
        
        # Para el i correspondiente, obtener del directorio Feynman_test el archivo CSV
        # y predecir el modelo
        lista_csvs = os.listdir(PATH_FEYNMAN_TEST)
        nombre_csv = lista_csvs[i] 
        df = pd.read_csv(os.path.join(PATH_FEYNMAN_TEST,nombre_csv))
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        r2 = predicciones[i].score(X, y)
        print("R2: ", r2)
        print("***************************")
        i += 1
        
    print("***************************")
    


if __name__ == '__main__':
    #capturar la salida estandar y ponerla en un txt salida.txt
    
    # preguntar si se quiere guardar la salida en un archivo, si es asi, preguntar el nombre
    nombre_archivo = input("Ingrese el nombre del archivo donde se guardara la salida: ")
    iteraciones = int(input("Ingrese la cantidad de iteraciones: "))
    if nombre_archivo:
        sys.stdout = open(os.path.join(PATH_BASE,nombre_archivo), 'w')
        
    main(iteraciones)
    
    if nombre_archivo:
        sys.stdout.close()

    
  