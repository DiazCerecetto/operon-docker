# Archivo con funciones de utilidad 
import inspect
import os
import sys
import time
import numpy as np
import pandas as pd
from sympy import preview, sympify
from operon.sklearn import SymbolicRegressor # type: ignore

# Direcciones de los archivos
PATH_FEYNMAN = "Feynman_train"
PATH_FEYNMAN_TEST = "Feynman_test"
PATH_VLADISLAVLEVA = "Vladislavleva_train"
PATH_VLADISLAVLEVA_TEST = "Vladislavleva_test"
PATH_FERREIRA = "Ferreira_train"
PATH_FERREIRA_TEST = "Ferreira_test"
PATH_IMAGENES_LATEX = "Imagenes_Latex"

PATH_RESULTADOS_FEYNMAN = "Resultados_Feynman"
PATH_RESULTADOS_VLADISLAVLEVA = "Resultados_Vladislavleva"
PATH_RESULTADOS_FERREIRA = "Resultados_Ferreira"



# Direcciones del proyecto
PATH_CODE = os.path.dirname(os.path.abspath(__file__))
PATH_BASE = os.path.dirname(PATH_CODE)

# Direcciones completas
PATH_RESULTADOS_FEYNMAN = os.path.join(PATH_BASE, PATH_RESULTADOS_FEYNMAN)
PATH_RESULTADOS_VLADISLAVLEVA = os.path.join(PATH_BASE, PATH_RESULTADOS_VLADISLAVLEVA)
PATH_RESULTADOS_FERREIRA = os.path.join(PATH_BASE, PATH_RESULTADOS_FERREIRA)

PATH_FEYNMAN = os.path.join(PATH_BASE, PATH_FEYNMAN)
PATH_FEYNMAN_TEST = os.path.join(PATH_BASE, PATH_FEYNMAN_TEST)
PATH_VLADISLAVLEVA = os.path.join(PATH_BASE, PATH_VLADISLAVLEVA)
PATH_VLADISLAVLEVA_TEST = os.path.join(PATH_BASE, PATH_VLADISLAVLEVA_TEST)
PATH_FERREIRA = os.path.join(PATH_BASE, PATH_FERREIRA)
PATH_FERREIRA_TEST = os.path.join(PATH_BASE, PATH_FERREIRA_TEST)

PATH_IMAGENES_LATEX = os.path.join(PATH_BASE, PATH_IMAGENES_LATEX)
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
        'allowed_symbols':"add,sub,mul,div,constant,variable,sin,cos,exp,log",
        'offspring_generator': 'basic',
        'initialization_method': 'btc',
        'n_threads': 8,
        'objectives':  ['r2'],
        'epsilon': 1e-5,
        'random_state' : np.random.default_rng(i),
        'reinserter': 'keep-best',
        'max_evaluations': int(1e6),
        'tournament_size': 3,
        'pool_size': None,
        'time_limit': 600,
        'local_iterations': 30
        }
    return SymbolicRegressor(**default_params)
    
def obtener_funcion(f):
    source = inspect.getsource(f)
    funcion = source.split("return")[1]
    funcion = funcion.strip()
    return funcion
import os
import pandas as pd
import numpy as np

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
            

            # Si hay valores NaN en las predicciones, se descarta
            if np.isnan(y_pred).any():
                print("Predicciones con valores NaN en la iteración ", iter+1, " y función ", obtener_funcion(f))
                print()
                df_salida.loc[j] = [obtener_funcion(f), 0, predicciones[j].get_model_string(), np.nan]
                continue
        
            r2 = 0
            try:
                r2 = predicciones[j].score(X, y)
            except:
                r2 = np.nan
    
            rmse = calcular_rmse(y, y_pred)
            df_salida.loc[j] = [obtener_funcion(f), r2, simplify_expression(predicciones[j].get_model_string()), rmse]

            
            df_salida.to_csv(resultados, index=False)
            tiempo_total = time.time() - tiempo_inicio
    return tiempo_total, tiempos_iteraciones


def calcular_rmse(y_true, y_pred):
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