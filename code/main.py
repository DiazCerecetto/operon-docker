# Archivo con funciones de utilidad 
import inspect
import os
import sys
import time
import numpy as np
import pandas as pd
from sympy import preview, sympify
from operon.sklearn import SymbolicRegressor # type: ignore
from datasets.config import PATH_FERREIRA_TRAIN, PATH_FEYNMAN_TRAIN, PATH_RESULTADOS_FEYNMAN, PATH_RESULTADOS_FERREIRA, PATH_RESULTADOS_VLADISLAVLEVA, PATH_FEYNMAN, PATH_FEYNMAN_TEST, PATH_FERREIRA, PATH_FERREIRA_TEST, PATH_VLADISLAVLEVA, PATH_VLADISLAVLEVA_TEST, PATH_VLADISLAVLEVA_TRAIN
from datasets.vladislavleva import lista_funciones as funciones_vladislavleva
from datasets.ferreira import lista_funciones as funciones_ferreira
from datasets.feynman import lista_funciones as funciones_feynman
PATH_IMAGENES_LATEX = "imagenes_latex"


num_threads = int(os.environ['OMP_NUM_THREADS']) if 'OMP_NUM_THREADS' in os.environ else 1


def entrenar_desde_csv(est, df):
    # Leer el archivo csv y entrenar el modelo    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    est = est.fit(X, y)
    m = est.model_
    return est, m

def obtener_modelo(i):
    default_params = {
        'allowed_symbols':"add,sub,mul,div,square,variable,sin,cos,exp", # Vlad-2: add,sub,mul,div,n2,exp,expneg,sin,cos 
        'offspring_generator': 'basic',
        'initialization_method': 'btc',
        'n_threads': 5,
        'objectives':  ['rmse'],
        'epsilon': 1e-5,
        'random_state' : np.random.default_rng(i),
        'reinserter': 'keep-best',
        'max_evaluations': int(1e6),   # 5000 generaciones * 1000 population size = 5M > 1M
        'tournament_size': 3,
        'pool_size': None,
        'time_limit': 600,
        #'local_iterations': 30,
        'max_length': 13,
        'max_depth': 5,
        'generations': 5000
        }
    return SymbolicRegressor(**default_params)
    
def obtener_funcion(f):
    source = inspect.getsource(f)
    funcion = source.split("return")[1]
    funcion = funcion.strip()
    return funcion

def entrenar_evaluar_modelo(iteraciones, path_train, path_test, path_resultados, lista_funciones, nombre):
    print("****************************")
    print(f"*** Operon {nombre} ***")
    print("****************************")
    print()
    
        
    tiempo_inicio = time.time()
    tiempos_iteraciones = []
    tiempo_total = 0
    for iter in range(iteraciones):
        tiempo_iteracion = time.time()
        resultados = os.path.join(path_resultados, f"resultados_{nombre}{iter}.csv")
        predicciones = []
        maximo = len(os.listdir(path_train))
        i = 0
        for file in os.listdir(path_train):
            
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(path_train, file))
                est = obtener_modelo(iter)
                est, m = entrenar_desde_csv(est, df)
                
                predicciones.append(est)

                sys.stdout.write("\033[F")
                print("Iteración: ", str(iter+1), " de ", iteraciones, " || ", i+1, " de ", maximo, " funciones")
                i += 1
        tiempo_iteracion = time.time() - tiempo_iteracion
        tiempos_iteraciones.append(tiempo_iteracion)
        
        
        df_salida = pd.DataFrame(columns=["Original", "R2", "Modelo", "RMSE"])
        for j, f in enumerate(lista_funciones):
            lista_csvs = os.listdir(path_test)
            nombre_csv = lista_csvs[j]
            df = pd.read_csv(os.path.join(path_test, nombre_csv))
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
            if np.isnan(X).any() or np.isnan(y).any():
                print("Valores NaN en el archivo ", nombre_csv)
                continue
            y_pred = predicciones[j].predict(X)
            #print type
            
            rmse = calcular_rmse(y, y_pred)

            #print("y_pred: ", y_pred)
            #print("expr: ", simplify_expression(predicciones[j].get_model_string()))
            #print("\n")
            

            # Si hay valores NaN en las predicciones, se descarta
            if np.isnan(y_pred).any():
                print("Habia nans")
                df_salida.loc[j] = [obtener_funcion(f), 0, predicciones[j].get_model_string(), np.nan]
                df_salida.to_csv(resultados, index=False)
                tiempo_total = time.time() - tiempo_inicio
                continue
        
            r2 = 0
            try:
                r2 = predicciones[j].score(X, y)
            except:
                r2 = np.nan
    
            df_salida.loc[j] = [obtener_funcion(f), r2, simplify_expression(predicciones[j].get_model_string()), rmse]

            
            df_salida.to_csv(resultados, index=False)
            tiempo_total = time.time() - tiempo_inicio
    return tiempo_total, tiempos_iteraciones


def calcular_rmse(y_true, y_pred):
    # Convertir a ndarray y aplicar redondeo a 4 decimales
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_true = np.round(y_true, 4)
    y_pred = np.round(y_pred, 4)
    
    return np.sqrt(np.mean((y_true - y_pred)**2))


def python_to_latex(expression_str,nombre="imagen.png"):
    # si el nombre ya existe, agregar un número al final en orden
    # es decir, si ya existe imagen.png, se crea imagen1.png, si ya existe imagen1.png, se crea imagen2.png, etc.
    i = 1
    nombre_aux = nombre
    
    while os.path.exists(PATH_IMAGENES_LATEX + "/" + nombre_aux):
        nombre_aux = nombre.split(".")[0] + str(i) + ".png"
        i += 1
    nombre = nombre_aux
    expression = sympify(expression_str)
    # si la carpeta no existe, la crea
    if not os.path.exists(PATH_IMAGENES_LATEX):
        os.makedirs(PATH_IMAGENES_LATEX)
    preview(expression, viewer='file', filename=PATH_IMAGENES_LATEX + "/" + nombre, euler=False)

def simplify_expression(expression_str):
    expression = sympify(expression_str)
    return expression.simplify()


def summary(carpeta_archivos):
    # Diccionarios para almacenar los datos
    r2_mejores = {}
    rmse_mejores = {}

    # Iterar sobre cada archivo en la carpeta
    for archivo in os.listdir(carpeta_archivos):
        if archivo.endswith('.csv'):
            # Leer el CSV
            ruta_archivo = os.path.join(carpeta_archivos, archivo)
            df = pd.read_csv(ruta_archivo)
            
            for _, row in df.iterrows():
                if row['Original'] not in r2_mejores:
                    r2_mejores[row['Original']] = []
                    rmse_mejores[row['Original']] = []
                r2_mejores[row['Original']].append(( row['R2'],  row['Modelo']))
                rmse_mejores[row['Original']].append((row['RMSE'],  row['Modelo']))


    r2_mejores = {k: sorted(v, key=lambda x: x[0], reverse=True)[:5] for k, v in r2_mejores.items()}
    rmse_mejores = {k: sorted(v, key=lambda x: x[0])[:5] for k, v in rmse_mejores.items()}

    df_mejores_r2 = pd.DataFrame([(k, v[0], v[1]) for k, lst in r2_mejores.items() for v in lst], columns=['Original', 'R2', 'Modelo'])
    df_mejores_rmse = pd.DataFrame([(k, v[0], v[1]) for k, lst in rmse_mejores.items() for v in lst], columns=['Original', 'RMSE', 'Modelo'])

    # Guardar los resultados
    df_mejores_r2.to_csv(os.path.join(carpeta_archivos, 'summary_r2.csv'), index=False)
    df_mejores_rmse.to_csv(os.path.join(carpeta_archivos, 'summary_rmse.csv'), index=False)          
    print("Resumen guardado exitosamente en " + carpeta_archivos)

def ferreira_train_test():
    iteraciones = int(input("Ingrese la cantidad de ejecuciones independientes: "))
    # Si la carpeta resultados no existe, crearla
    os.makedirs(PATH_RESULTADOS_FERREIRA, exist_ok=True)
    # si la carpeta de resultados tiene archivos, eliminarlos
    for file in os.listdir(PATH_RESULTADOS_FERREIRA):
        os.remove(os.path.join(PATH_RESULTADOS_FERREIRA, file))
    entrenar_evaluar_modelo(iteraciones, PATH_FERREIRA_TRAIN, PATH_FERREIRA_TEST, PATH_RESULTADOS_FERREIRA, funciones_ferreira, "Ferreira")
    
def feynman_train_test():
    # si el directorio de resultados no existe, lo creo
    if not os.path.exists(PATH_RESULTADOS_FEYNMAN):
        os.makedirs(PATH_RESULTADOS_FEYNMAN)
    iteraciones = int(input("Ingrese la cantidad de ejecuciones independientes: "))
    # Si la carpeta de resultados tiene archivos, eliminarlos
    for file in os.listdir(PATH_RESULTADOS_FEYNMAN):
        os.remove(os.path.join(PATH_RESULTADOS_FEYNMAN, file))
    # Medir el tiempo de ejecución
    tiempo_total, tiempos_iteraciones = entrenar_evaluar_modelo(iteraciones,PATH_FEYNMAN_TRAIN,PATH_FEYNMAN_TEST,PATH_RESULTADOS_FEYNMAN,funciones_feynman,"feynman")
    
    print(f"Tiempo total de ejecución: {tiempo_total}")
    print(f"Tiempo promedio de ejecución: {np.mean(tiempos_iteraciones)}")
    
def vladislavleva_train_test():
    iteraciones = int(input("Ingrese la cantidad de ejecuciones independientes: "))
    if not os.path.exists(PATH_RESULTADOS_VLADISLAVLEVA):
        os.makedirs(PATH_RESULTADOS_VLADISLAVLEVA)
    # Si la carpeta de resultados tiene archivos, eliminarlos
    for file in os.listdir(PATH_RESULTADOS_VLADISLAVLEVA):
        os.remove(os.path.join(PATH_RESULTADOS_VLADISLAVLEVA, file))
    
    entrenar_evaluar_modelo(iteraciones,PATH_VLADISLAVLEVA_TRAIN,PATH_VLADISLAVLEVA_TEST,PATH_RESULTADOS_VLADISLAVLEVA,funciones_vladislavleva,"vladislavleva")

def main():
    print("Generar datos o cargar datos existentes?")
    print("1. Generar datos")
    print("2. Cargar datos existentes")
    opcion = int(input("Ingrese la opción: "))
    if opcion == 1:
        print("Cargar datos existentes")
        print("Seleccione el dataset a utilizar:")
        print("1. Feynman")
        print("2. Ferreira")
        opcion = int(input("Ingrese la opción: "))
        if opcion == 1:
            from datasets.feynman import generar_ecuaciones_feynman
            generar_ecuaciones_feynman()
            print("Datos cargados exitosamente")
        elif opcion == 2:
            from datasets.ferreira import generar_ecuaciones_ferreira
            generar_ecuaciones_ferreira()
            print("Datos cargados exitosamente")
        else:
            print("Opción inválida")
    elif opcion == 2:
        print("Seleccione el dataset a utilizar:")
        print("1. Feynman")
        print("2. Ferreira")
        print("3. Vladislavleva")
        opcion = int(input("Ingrese la opción: "))
        if opcion == 1:
            feynman_train_test()
            summary(PATH_RESULTADOS_FEYNMAN)
        elif opcion == 2:
            ferreira_train_test()
            summary(PATH_RESULTADOS_FERREIRA)
        elif opcion == 3:
            vladislavleva_train_test()
            summary(PATH_RESULTADOS_VLADISLAVLEVA)
        else:
            print("Opción inválida")
            return
    else:
        print("Opción inválida")


if __name__ == "__main__":
    main()