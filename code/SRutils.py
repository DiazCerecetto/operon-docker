# Archivo con funciones de utilidad 
import os
import numpy as np
from operon.sklearn import SymbolicRegressor

PATH_FEYNMAN = "Feynman"
PATH_FEYNMAN_TEST = "Feynman_test"
PATH_VLADISLAVLEVA = "Vladislavleva"
# salir de la carpeta code 
os.chdir("..")
# obtener el path actual
PATH_BASE = os.getcwd()
# volver a la carpeta code
os.chdir("code")

PATH_FEYNMAN = os.path.join(PATH_BASE, PATH_FEYNMAN)
PATH_FEYNMAN_TEST = os.path.join(PATH_BASE, PATH_FEYNMAN_TEST)
PATH_VLADISLAVLEVA = os.path.join(PATH_BASE, PATH_VLADISLAVLEVA)

num_threads = int(os.environ['OMP_NUM_THREADS']) if 'OMP_NUM_THREADS' in os.environ else 1
rng = np.random.default_rng(1234)
default_params = {
        'allowed_symbols':"add,sub,mul,div,constant,variable,sin,cos",
        'offspring_generator': 'basic',
        'initialization_method': 'btc',
        'n_threads': 8,
        'objectives':  ['r2', 'length'],
        'epsilon': 1e-5,
        'random_state': rng,
        'reinserter': 'keep-best',
        'max_evaluations': int(1e6),
        'tournament_size': 3,
        'pool_size': None,
        'time_limit': 600
            }
def predecir_desde_csv(est, df,nombre,iteraciones = 30):
    # Leer el archivo csv
    # 30 iteraciones
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    print(X.shape, y.shape)
    for i in range(int(iteraciones)):
        est = est.fit(X, y)
    m = est.model_
    #print("***************************")
    #print("Original: " + nombre)
    #print("Modelo: " + str(m))
    #print("***************************")
    return est, m

def obtener_modelo():
    return SymbolicRegressor(**default_params)
    
