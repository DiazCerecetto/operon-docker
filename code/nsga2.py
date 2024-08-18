import numpy as np
import os
import time
import pandas as pd
import sys
from sympy import simplify as simplify_expression
from sklearn.metrics import r2_score
from operon.sklearn import SymbolicRegressor # type: ignore
from main import entrenar_desde_csv, calcular_rmse, obtener_funcion, obtener_numero_funcion_archivo
from datasets.ferreira import lista_funciones as funciones_ferreira
from datasets.feynman import lista_funciones as funciones_feynman
from datasets.vladislavleva import lista_funciones as funciones_vladislavleva

def obtener_modelo(i):
    default_params = {
        'allowed_symbols':"add,sub,mul,div,square,variable,sin,cos,exp", # Vlad-2: add,sub,mul,div,n2,exp,expneg,sin,cos 
        'offspring_generator': 'basic',
        'initialization_method': 'btc',
        'n_threads': 5,
        'objectives':  ['rmse','undefined_count'],
        'epsilon': 1e-5,
        'random_state' : np.random.default_rng(i),
        'reinserter': 'keep-best',
        'max_evaluations': int(5e6),   # 5000 generaciones * 1000 population size = 5M > 1M
        'tournament_size': 3,
        'pool_size': None,
        'time_limit': 600,
        #'local_iterations': 30,
        'max_length': 13,
        'max_depth': 5,
        'generations': 5000
        }
    return SymbolicRegressor(**default_params)
    
from datasets.config import PATH_FERREIRA_TRAIN, PATH_FEYNMAN_TRAIN, PATH_RESULTADOS_FEYNMAN, PATH_RESULTADOS_FERREIRA, PATH_RESULTADOS_VLADISLAVLEVA, PATH_FEYNMAN, PATH_FEYNMAN_TEST, PATH_FERREIRA, PATH_FERREIRA_TEST, PATH_VLADISLAVLEVA, PATH_VLADISLAVLEVA_TEST, PATH_VLADISLAVLEVA_TRAIN
from datasets.config import feynman, ferreira, vladislavleva


def train_nsga(iteraciones, path_train, path_test, path_resultados, lista_funciones, nombre, lista_admitidos=None):
    print("****************************")
    print(f"*** OPNSGA {nombre} ***")
    print("****************************")
    print()
    
    # Crear una carpeta para los resultados dentro de la carpeta de resultados
    # con el timestamp actual
    path_resultados = os.path.join(path_resultados, str(int(time.time())))
    os.makedirs(path_resultados, exist_ok=True)
    
    tiempo_inicio = time.time()
    tiempos_iteraciones = []
    tiempo_total = 0
    
    print("****************************")
    print(" * Parámetros del modelo: * ")
    print("****************************")
    est = obtener_modelo(0)
    params = est.get_params()
    
    for param, value in params.items():
        print(f"{param}: {value}")
    print("****************************")
    print()
    lista_enteros_funcion_archivo = []
    for entero in lista_admitidos:
        # Agregar a un json el entero y la funcion correspondiente
        funcion = None
        archivo = None
        archivo_test = None
        for f in lista_funciones:
            if obtener_numero_funcion_archivo(f.__name__) == entero:
                funcion = f
        for arch in os.listdir(path_train):
            if arch.endswith(".csv") and obtener_numero_funcion_archivo(arch) == entero:
                archivo = arch
        for archi in os.listdir(path_test):
            if archi.endswith(".csv") and obtener_numero_funcion_archivo(archi)== entero:
                archivo_test = archi
        if funcion is not None and archivo is not None and archivo_test is not None:
            lista_enteros_funcion_archivo.append({
                "numero":entero,
                "funcion":funcion,
                "archivo":archivo,
                "archivo_test":archivo_test
                })
        else:
            for f in lista_enteros_funcion_archivo:
                print("Entero: ", entero)
                print("Funcion: ", f["funcion"])
                print("Archivo: ", f["archivo"])
                print("Archivo test: ", f["archivo_test"])
            return 0, []
    cantidad_funciones = len(lista_enteros_funcion_archivo)
    cantidad_actual = 0
    for objetoJson in lista_enteros_funcion_archivo:
        cantidad_actual += 1
        predicciones = []
        resultados = os.path.join(path_resultados, f"resultados_{nombre}{objetoJson["numero"]}.csv")
        lista_filas = []
        for iter in range(iteraciones):
            tiempo_iteracion = time.time()
            
            
            df = pd.read_csv(os.path.join(path_train, objetoJson["archivo"]))
            est = obtener_modelo(iter)
            est, m = entrenar_desde_csv(est, df)
            if "modelo" not in objetoJson or objetoJson["modelo"] is None:
                objetoJson["modelo"] = [m]
            else:
                objetoJson["modelo"].append(m)
                
            sys.stdout.write("\033[F")
            print("Funcion: ", cantidad_actual, " de ", cantidad_funciones, " || ", iter+1, " de ", iteraciones, " iteraciones")
            tiempo_iteracion = time.time() - tiempo_iteracion
            tiempos_iteraciones.append(tiempo_iteracion)
            
            df_test = pd.read_csv(os.path.join(path_test, objetoJson["archivo_test"]))
            # imprimir longitud de test
            X_test = df_test.iloc[:, :-1].values
            y_test = df_test.iloc[:, -1].values
            y_pred = est.predict(X_test)
            predicciones.append(y_pred)
            
            # Calcular RMSE
            # Si y_pred contiene nan, no se calcula el RMSE
            if np.isnan(y_pred).any():
                rmse = np.nan
            else:
                rmse = calcular_rmse(y_test, y_pred)
            try:
                r2 = r2_score(X_test, y_test)
            except:
                r2 = np.nan
            # Guardar los resultados en un archivo
            simplificada = ""
            try:
                simplificada = simplify_expression(est.get_model_string())
            except Exception as e:
                simplificada = "Error al simplificar " + str(e)
                
            # Evaluar sobre X_train y X_test y para cada tupla de valores fijarse si da Nan
            # si da Nan, se incrementa el contador de undefined_count
            undefined_count_train = 0
            undefined_count_test = 0
            X_train = df.iloc[:, :-1].values
            y_train = df.iloc[:, -1].values
            for i in range(X_test.shape[0]):
                if np.isnan(y_pred[i]):
                    undefined_count_test += 1
            for i in range(X_train.shape[0]):
                try:
                    y_pred_train = est.predict(X_train[i].reshape(1, -1))
                    if np.isnan(y_pred_train):
                        undefined_count_train += 1
                except:
                    undefined_count_train += 1
                    
                        
            

            lista_filas.append({
                'Numero_iteracion':iter,
                'Funcion': obtener_funcion(objetoJson["funcion"]),
                'Original': est.get_model_string(),
                'Simplificado': simplificada,
                'RMSE': rmse,
                'R2': r2,
                'tiempo': tiempo_iteracion,
                'undefined_count_train': undefined_count_train,
                'undefined_count_test': undefined_count_test,
                #'pareto_front': est.get_pareto_front(1)
            })
        # ordenar lista_filas por RMSE
        lista_filas = sorted(lista_filas, key=lambda x: x['RMSE'])
        
        
        df_resultados = pd.DataFrame(lista_filas)
            
        # Guardar los resultados en un archivo
        df_resultados.to_csv(resultados, index=False)
        # Calcular el tiempo total
        tiempo_total = time.time() - tiempo_inicio
    
    return tiempo_total, tiempos_iteraciones





def main():
    print("Generar datos o cargar datos existentes?")
    print("1. Ejecutar NSGA-II")
    opcion = int(input("Ingrese la opción: "))
    if opcion == 1:
        print("Seleccione el dataset a utilizar:")
        print("1. Feynman")
        print("2. Ferreira")
        print("3. Vladislavleva")
        opcion = int(input("Ingrese la opción: "))
        cantidad_iteraciones = int(input("Ingrese la cantidad de iteraciones: "))
        lista_funciones = input("Ingrese la lista de funciones a utilizar separadas por comas: ")
        if lista_funciones != None and lista_funciones != "":
            lista_funciones = lista_funciones.split(",")
            for i in range(len(lista_funciones)):
                lista_funciones[i] = int(lista_funciones[i])
        else:
            lista_funciones = None
        print("Ejecutando NSGA-II")
        print("Cantidad de iteraciones: ", cantidad_iteraciones)
        print("Lista de funciones: ", lista_funciones)
        print("Dataset seleccionado: ", opcion)
        if opcion == 1:
            if lista_funciones is None:
                lista_funciones = [i for i in range(len(funciones_feynman))]
            train_nsga(cantidad_iteraciones, PATH_FEYNMAN_TRAIN, PATH_FEYNMAN_TEST, PATH_RESULTADOS_FEYNMAN, funciones_feynman,"feynman",lista_funciones)
        elif opcion == 2:
            if lista_funciones is None:
                lista_funciones = [i for i in range(len(funciones_ferreira))]
            train_nsga(cantidad_iteraciones, PATH_FERREIRA_TRAIN, PATH_FERREIRA_TEST, PATH_RESULTADOS_FERREIRA, funciones_ferreira,"Ferreira",lista_funciones)
        elif opcion == 3:
            if lista_funciones is None:
                lista_funciones = [i for i in range(len(funciones_vladislavleva))]
            train_nsga(cantidad_iteraciones, PATH_VLADISLAVLEVA_TRAIN, PATH_VLADISLAVLEVA_TEST, PATH_RESULTADOS_VLADISLAVLEVA, funciones_vladislavleva,"Vladislavleva",lista_funciones)
        else:
            print("Opción inválida")
            return

if __name__ == "__main__":
    main()
    