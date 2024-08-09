import pandas as pd
import numpy as np
import os
from datasets.config import  PATH_RESULTADOS_VLADISLAVLEVA
# Para cada archivo en el directorio indicado
path_archivos = PATH_RESULTADOS_VLADISLAVLEVA

# Entrar en la carpeta con el numero mas alto, es decir
# el ultimo experimento

path_archivos = os.path.join(path_archivos, str(max([int(x) for x in os.listdir(path_archivos)]))   )

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

    #Guardar en un txt
    with open(os.path.join(path_archivos,f"{archivo_csv}_metrics.txt"), 'w') as f:
        f.write(f'RMSE\n')
        f.write(f'Media: {media_rmse}\n')
        f.write(f'IQR: {iqr_rmse}\n')
        f.write(f'Máximo: {max_rmse}\n') 
