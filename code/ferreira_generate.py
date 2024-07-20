import os
import numpy as np
import pandas as pd
from ferreira_functions import lista_funciones

# Para garantizar reproducibilidad
SEED_TRAIN = 1234
SEED_TEST = 5678
PATH_CODE = os.path.dirname(os.path.abspath(__file__))
os.chdir(PATH_CODE)

def generar_datos(low, high, num_points=100, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low, high, num_points)

df = pd.read_csv(os.path.join(PATH_CODE, 'FerreiraEquations.csv'))

# Para cada funcion en la lista de funciones
# generar variando de forma uniforme Cain [0.5–3] y q [0.5–4] en  puntos
def generar_ecuaciones_ferreira(cantidad, nombre_directorio,seed):
    os.makedirs(f'Ferreira_{nombre_directorio}', exist_ok=True)
    os.chdir(f'Ferreira_{nombre_directorio}')
    # Si el directorio ya existe, borrar los archivos que haya dentro
    for file in os.listdir():
        os.remove(file)
    
    if seed is not None:
        np.random.seed(seed)
        
    for index, row in df.iterrows():

        # 1. Generar los valores de las variables respetando los intervalos
        variables = []
        resultado = []
        for i in range(2):
            v_low = row[f'v{i}_low']
            v_high = row[f'v{i}_high']
            # si son NaN, ya no hay más variables, break
            if pd.isna(v_low) or pd.isna(v_high):
                break
            variables.append(generar_datos(v_low, v_high, cantidad))

        # 2. Evaluar la función en los valores de las variables
        for i in range(cantidad):
            resultado.append(lista_funciones[index](*[v[i] for v in variables]))
        
        # 3. Armar un DataFrame que tenga variables y resultado y guardarlo en un archivo CSV
        df_resultado = pd.DataFrame({f'v{i+1}': variables[i] for i in range(len(variables))})
        df_resultado['resultado'] = resultado
        df_resultado.to_csv(f'ferreira{index+1}.csv', index=False)
    
    os.chdir('..')

if __name__ == '__main__':
    os.chdir('..')
    # Generar puntos para train y test
    cantidad_train = int(input('Ingrese la cantidad de puntos para train: '))
    cantidad_test = int(input('Ingrese la cantidad de puntos para test: '))
    generar_ecuaciones_ferreira(cantidad_train, 'train', seed=SEED_TRAIN)
    generar_ecuaciones_ferreira(cantidad_test, 'test', seed=SEED_TEST)
    print('Listo!')