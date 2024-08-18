import pandas as pd
import os

## Leer el archivo benchmarks.csv
# change workdir to datasets/Feynman
os.chdir("datasets/Feynman")
benchmarks_df = pd.read_csv("FeynmanEquations.csv")

# Iterar sobre cada fila del DataFrame
for index, row in benchmarks_df.iterrows():
    # Crear el nombre del archivo a comprobar
    filename = row['Filename']  # Asegúrate de que 'Filename' es el nombre correcto de la columna
    # Replace . with _ on filename
    filename = filename.replace('.', '_') 
    old_filename = f'data_Feynman_{filename}_n0.00_s0.csv'
    
    # Verificar si el archivo existe
    if os.path.exists(old_filename):
        # Nuevo nombre del archivo
        new_filename = f'feynman{index +1}.csv'
        
        # Renombrar el archivo
        os.rename(old_filename, new_filename)
        print(f'Renamed {old_filename} to {new_filename}')
    else:
        print(f'{old_filename} does not exist')

print('Renaming process completed.')
