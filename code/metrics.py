import pandas as pd
import numpy as np
import os
from datasets.config import PATH_RESULTADOS_VLADISLAVLEVA, PATH_RESULTADOS_FEYNMAN, PATH_RESULTADOS_FERREIRA
# Para cada archivo en el directorio indicado

def tomar_metricas(path_archivos,nombre):


    path_archivos = os.path.join(path_archivos, str(max([int(x) for x in os.listdir(path_archivos)]))   )
    lista_datos = []
    for archivo_csv in os.listdir(path_archivos):
        # Si el archivo no es un CSV, se salta
        if not archivo_csv.endswith('.csv'):
            continue
        # Leer el archivo CSV
        data = pd.read_csv(os.path.join(path_archivos, archivo_csv))
        # Si contiene inf o nan en alguna columna, se eliminan las filas
        
        
        # Si el archivo tiene nan o valores infinitos en 'RMSE' o R2, se eliminan las filas
        data = data.dropna(subset=['RMSE', 'R2'])
        data = data[~data['RMSE'].isin([np.nan, np.inf, -np.inf])]
        data = data[~data['R2'].isin([np.nan, np.inf, -np.inf])]
        
        
        media_rmse = data['RMSE'].mean()
        iqr_rmse = np.percentile(data['RMSE'], 75) - np.percentile(data['RMSE'], 25)
        max_rmse = data['RMSE'].max()
        promedio_tiempo = data['tiempo'].mean()

        #Guardar en un txt
        with open(os.path.join(path_archivos,f"{archivo_csv}_metrics.txt"), 'w') as f:
            f.write(f'RMSE\n')
            f.write(f'Media: {media_rmse}\n')
            f.write(f'IQR: {iqr_rmse}\n')
            f.write(f'Máximo: {max_rmse}\n') 
            f.write(f'Tiempo promedio: {promedio_tiempo}\n')
        # leer datasets\Feynman\FeynmanEquations.csv y funcion_feynman es 
        # la ecuacion de la columna Filename
        #feynman_source = pd.read_csv(os.path.join(PATH_FEYNMAN, 'FeynmanEquations.csv'))
        # obtener numero de resultados_feynman{numero}.csv
        #numero = archivo_csv.split('feynman')[1].split('.')[0]
        #numero = int(numero)
        # aqui se tiene feynman{numero}
        #
        #funcion_feynman = feynman_source.iloc[numero-1]['Filename']
        funcion_feynman = archivo_csv.split(nombre)[1].split('.')[0]
        lista_datos.append([archivo_csv,funcion_feynman, media_rmse, iqr_rmse, max_rmse, promedio_tiempo])
        
        
            
        
    # Para cada elemento  en lista_datos, escribir:
    # {funcion_feynman} & & & & &{media_rmse} &{iqr_rmse} &{max_rmse} &{promedio_tiempo} & & & &
    # imprimir los 4 datos en formato 4 decimales con e-00 

        
    # hacer sort de lista_datos por dato[0].split('feynman')[1].split('.')[0]
    lista_datos = sorted(lista_datos, key=lambda x: int(x[0].split('vladislavleva')[1].split('.')[0]))

    for dato in lista_datos:
        print(dato[0])
        print(dato[1])
    for datos in lista_datos:
        print(f'{datos[1]} & & & & &{datos[2]:.2e} &{datos[3]:.2e} &{datos[4]:.2e} &{datos[5]:.2e} & & & & \\\\') 
        
    return lista_datos

if __name__ == '__main__':
    print('Indicar dataset:')
    print('1. Feynman')
    print('2. Vladislavleva')
    print('3. Ferreira')
    
    dataset = input()
    
    if dataset == '1':
        tomar_metricas(PATH_RESULTADOS_FEYNMAN, 'feynman')
    elif dataset == '2':
        tomar_metricas(PATH_RESULTADOS_VLADISLAVLEVA, 'vladislavleva')
    elif dataset == '3':
        tomar_metricas(PATH_RESULTADOS_FERREIRA, 'ferreira')
    else:
        print('Opción no válida')
    
