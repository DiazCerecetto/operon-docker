import os
import pandas as pd

def combine_csv_files(folder_path):
    # Lista para almacenar los DataFrames
    data_frames = []

    # Recorre cada archivo en el rango esperado
    for i in range(30):
        file_name = f"resultados_vladislavleva{i}.csv"
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.exists(file_path):
            # Lee el archivo CSV y añádelo a la lista
            df = pd.read_csv(file_path)
            data_frames.append(df)
        else:
            print(f"El archivo {file_name} no existe en la ruta {folder_path}")

    # Combina todos los DataFrames en uno solo
    summary_df = pd.concat(data_frames, ignore_index=True)

    # Guarda el DataFrame combinado en un archivo summary.csv
    summary_file_path = os.path.join(folder_path, "summary.csv")
    summary_df.to_csv(summary_file_path, index=False)
    print(f"Archivo summary.csv creado en {folder_path}")

if __name__ == '__main__':
    folder_path = "Resultados_Vladislavleva" 
    combine_csv_files(folder_path)
