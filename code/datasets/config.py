import os

import numpy as np
import pandas as pd
# Definimos el path raíz del proyecto

PATH_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DATASETS = os.path.join(PATH_BASE, 'datasets')


############# FEYNMAN ############
PATH_FEYNMAN = os.path.join(PATH_DATASETS, 'Feynman')
PATH_FEYNMAN_TEST = os.path.join(PATH_FEYNMAN, 'test')
PATH_FEYNMAN_TRAIN = os.path.join(PATH_FEYNMAN, 'train')
PATH_RESULTADOS_FEYNMAN = os.path.join(PATH_FEYNMAN, 'resultados')

############# VLADISLAVLEVA ############
PATH_VLADISLAVLEVA = os.path.join(PATH_DATASETS, 'Vladislavleva')
PATH_VLADISLAVLEVA_TEST = os.path.join(PATH_VLADISLAVLEVA, 'test')
PATH_VLADISLAVLEVA_TRAIN = os.path.join(PATH_VLADISLAVLEVA, 'train')
PATH_RESULTADOS_VLADISLAVLEVA = os.path.join(PATH_VLADISLAVLEVA, 'resultados')

############# FERREIRA ############
PATH_FERREIRA = os.path.join(PATH_DATASETS, 'Ferreira')
PATH_FERREIRA_TEST = os.path.join(PATH_FERREIRA, 'test')
PATH_FERREIRA_TRAIN = os.path.join(PATH_FERREIRA, 'train')
PATH_RESULTADOS_FERREIRA = os.path.join(PATH_FERREIRA, 'resultados')

feynman = {
    'name': 'feynman',
    'base': PATH_FEYNMAN,
    'train': PATH_FEYNMAN,
    'test': PATH_FEYNMAN_TEST,
    'results': PATH_RESULTADOS_FEYNMAN,
}

ferreira = {
    'name' : 'ferreira',
    'base': PATH_FERREIRA,
    'train': PATH_FERREIRA_TRAIN,
    'test': PATH_FERREIRA_TEST,
    'results': PATH_RESULTADOS_FERREIRA,
}

vladislavleva = {
    'name': 'vladislavleva',
    'base': PATH_VLADISLAVLEVA,
    'train': PATH_VLADISLAVLEVA_TRAIN,
    'test': PATH_VLADISLAVLEVA_TEST,
    'results': PATH_RESULTADOS_VLADISLAVLEVA,
}

def generar_datos(low, high, num_points=100, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low, high, num_points)


def generar_datos_modelo(cantidad, nombre_directorio, seed, df, lista_funciones, prefix):
    os.makedirs(nombre_directorio, exist_ok=True)
    
    # Si el directorio ya existe, borrar los archivos que haya dentro
    for file in os.listdir(nombre_directorio):
        print('Atención: Hay archivos en el directorio de salida. Se borrarán todos los archivos existentes.')
        os.remove(os.path.join(nombre_directorio, file))
    
    if seed is not None:
        np.random.seed(seed)
        
    for index, row in df.iterrows():
        variables = []
        resultado = []
        for i in range(1, 11):
            v_low = row.get(f'v{i}_low', None)
            v_high = row.get(f'v{i}_high', None)
            if pd.isna(v_low) or pd.isna(v_high):
                break
            variables.append(generar_datos(v_low, v_high, cantidad))

        for i in range(cantidad):
            resultado.append(lista_funciones[index](*[v[i] for v in variables]))
        
        df_resultado = pd.DataFrame({f'v{i+1}': variables[i] for i in range(len(variables))})
        df_resultado['resultado'] = resultado
        df_resultado.to_csv(os.path.join(nombre_directorio, f'{prefix}{index+1}.csv'), index=False)