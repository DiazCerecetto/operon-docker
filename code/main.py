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
from sklearn.metrics import r2_score
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
    
def obtener_funcion(f):
    source = inspect.getsource(f)
    funcion = source.split("return")[1]
    funcion = funcion.strip()
    return funcion

# Ejemplos:
# obtener_numero_funcion_archivo("resultados_feynmann_1.csv") -> 1
# obtener_numero_funcion_archivo("resultados_feynmann15.csv") -> 15
# obtener_numero_funcion_archivo("resultados_Vladislavleva2.csv") -> 2
def obtener_numero_funcion_archivo(nombre_archivo):
    # Obtener el número que viene antes de .csv
    # Contemplar el caso en que el número sea de más de un dígito
    numero = ""
    for c in nombre_archivo:
        try:
            int(c)
            numero += c
        except:
            pass
    return int(numero)


def entrenar_evaluar_modelo(iteraciones, path_train, path_test, path_resultados, lista_funciones, nombre, lista_admitidos=None):
    print("****************************")
    print(f"*** Operon {nombre} ***")
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
                
            lista_filas.append({
                'Numero_iteracion':iter,
                'Funcion': obtener_funcion(objetoJson["funcion"]),
                'Original': est.get_model_string(),
                'Simplificado': simplificada,
                'RMSE': rmse,
                'R2': r2,
                'tiempo': tiempo_iteracion,
            })
        # ordenar lista_filas por RMSE
        lista_filas = sorted(lista_filas, key=lambda x: x['RMSE'])
        
        
        df_resultados = pd.DataFrame(lista_filas)
            
        # Guardar los resultados en un archivo
        df_resultados.to_csv(resultados, index=False)
        # Calcular el tiempo total
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

# Para todos los archivos en la carpeta, combina los resultados en un solo archivo

def combine_csv_files(folder_path,prefix):
    

    lista_ejecuciones = os.listdir(folder_path)
    lista_ejecuciones = [x for x in lista_ejecuciones if x.startswith(f"resultados_{prefix}")]
    data_frames = []
    #Concatenar todos los dataframes y guardar el archivo
    for i in range(len(lista_ejecuciones)):
        file_name = f"resultados_{prefix}{i}.csv"
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Agregar la columna Iteración_independiente al dataframe
            df['Iteración_independiente'] = i
            data_frames.append(df)
        else:
            print(f"El archivo {file_name} no existe en la ruta {folder_path}")
    
            
    # Concatenar todos los dataframes y guardar el archivo
    summary_df = pd.concat(data_frames, ignore_index=True)
    summary_file_path = os.path.join(folder_path, "concatenated.csv")
    summary_df.to_csv(summary_file_path, index=False)
    print(f"Archivo concatenated.csv creado en {folder_path}")

def train_test(path_resultados,path_train,path_test,funciones,prefix,cantidad_funciones):
    iteraciones = int(input("Ingrese la cantidad de ejecuciones independientes: "))
    os.makedirs(path_resultados, exist_ok=True)
    lista_admitidos = input("Ingrese los números de las funciones a cargar separados por comas (ejemplo: 1,2,3) Dejar blanco para cargar todas: ")
    lista_admitidos = [int(x) for x in lista_admitidos.split(",")] if lista_admitidos != "" else None
    if lista_admitidos is None:
            lista_admitidos = list(range(1, cantidad_funciones+1))
    entrenar_evaluar_modelo(iteraciones, path_train, path_test, path_resultados, funciones, prefix, lista_admitidos)
    
    
    
def main():
    print("Generar datos o cargar datos existentes?")
    print("1. Generar datos")
    print("2. Cargar datos existentes")
    print("3. Concatenar resultados de ejecuciones independientes")
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
            train_test(PATH_RESULTADOS_FEYNMAN,PATH_FEYNMAN_TRAIN,PATH_FEYNMAN_TEST,funciones_feynman,"feynman",100)
        elif opcion == 2:
            train_test(PATH_RESULTADOS_FERREIRA,PATH_FERREIRA_TRAIN,PATH_FERREIRA_TEST,funciones_ferreira,"Ferreira",4)
        elif opcion == 3:
            train_test(PATH_RESULTADOS_VLADISLAVLEVA,PATH_VLADISLAVLEVA_TRAIN,PATH_VLADISLAVLEVA_TEST,funciones_vladislavleva,"vladislavleva",8)
        else:
            print("Opción inválida")
            return
    elif opcion == 3:
        print("Seleccione el dataset a concatenar:")
        print("1. Feynman")
        print("2. Ferreira")
        print("3. Vladislavleva")
        opcion = int(input("Ingrese la opción: "))
        path = ""
        prefix = ""
        if opcion == 1:
            path = PATH_RESULTADOS_FEYNMAN
            prefix = "feynman"
        elif opcion == 2:
            path =  PATH_RESULTADOS_FERREIRA
            prefix = "Ferreira"
        elif opcion == 3:
            path =  PATH_RESULTADOS_VLADISLAVLEVA
            prefix = "vladislavleva"
        else :
            print("Opción inválida")
            return 
        # obtener la ultima carpeta creada
        lista_carpetas = os.listdir(path)
        lista_carpetas = [x for x in lista_carpetas if x.isdigit()]
        lista_carpetas.sort()
        if len(lista_carpetas) == 0:
            print("No hay carpetas en el directorio")
            return
        
        # como estan creadas por timestamp, la ultima es la mas reciente
        # y esta tiene el mayor valor, por eso se toma el ultimo elemento (lista_carpetas[-1])
        path = os.path.join(path,lista_carpetas[-1])
        try:
            print("Creando archivo concatenated.csv en la carpeta ",path) 
            combine_csv_files(path,prefix)
        except Exception as e:
            print("Error al concatenar los archivos: ",e)

    else:
        print("Opción inválida")


if __name__ == "__main__":
    main()