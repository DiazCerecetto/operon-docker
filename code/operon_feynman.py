
import sys
import pandas as pd
import numpy as np
import os
from SRutils import entrenar_desde_csv, PATH_FEYNMAN, PATH_FEYNMAN_TEST, PATH_RESULTADOS, obtener_modelo
from feynman_functions import lista_funciones, obtener_funcion



def main(iteraciones,path_resultados = "resultados.csv"):
    print("****************************")
    print("****** Operon Feynman ******")
    print("****************************")
    print("** Iteraciones: ", iteraciones ," **")
    print()
    predicciones = []
    # Leer los archivos csv, en el directorio Feynman
    # para cada archivo CSV, predecir y guardar el modelo
    # Luego, comparar los modelos obtenidos con los modelos reales
    # indicar por consola la cantidad de iteraciones
    
    maximo = len(os.listdir(PATH_FEYNMAN))
    i = 1
    for file in os.listdir(PATH_FEYNMAN):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(PATH_FEYNMAN, file))
            est = obtener_modelo()
            est, m = entrenar_desde_csv(est, df,iteraciones)
            predicciones.append(est)

            # Para imprimir el progreso
            sys.stdout.write("\033[F")
            print("Progreso: ",i, " de ", maximo, " entrenamientos")
            i += 1
            
    print("\nEvaluando resultados...")
    
    
    # Comparar los modelos obtenidos con los modelos reales
    i = 0
    df_salida = pd.DataFrame(columns=["Original","R2", "Modelo"])
    for i, f in enumerate(lista_funciones):
        

        
        # Para el i correspondiente, obtener del directorio Feynman_test el archivo CSV
        # y predecir el modelo
        lista_csvs = os.listdir(PATH_FEYNMAN_TEST)
        nombre_csv = lista_csvs[i] 
        df = pd.read_csv(os.path.join(PATH_FEYNMAN_TEST,nombre_csv))
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        r2 = predicciones[i].score(X, y)
        
        df_salida.loc[i] = [obtener_funcion(f), r2, predicciones[i].get_model_string()]
        i += 1
        
    print("Resultados obtenidos en el archivo: ", path_resultados)
    
    df_salida.to_csv(path_resultados,index=False)
    


if __name__ == '__main__':
    iteraciones = int(input("Ingrese la cantidad de iteraciones: "))
    # Obtener path del archivo
    os.makedirs(PATH_RESULTADOS, exist_ok=True)
    resultados_feynman = os.path.join(PATH_RESULTADOS, "resultados_feynman.csv")
    main(iteraciones,resultados_feynman)

    
  