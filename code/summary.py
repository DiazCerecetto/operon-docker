import os
import pandas as pd
from generate_summary import combine_csv_files

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
if __name__ == '__main__':
    PATH_ARCHIVOS = "Resultados_Vladislavleva"
    summary(PATH_ARCHIVOS)
    combine_csv_files(PATH_ARCHIVOS)