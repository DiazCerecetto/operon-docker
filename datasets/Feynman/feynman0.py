import pandas as pd
import os

# Directorio de entrada y salida
input_dir = "."
train_dir = "train"
test_dir = "test"
os.chdir("datasets/Feynman")
# Crear directorios de salida si no existen
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Iterar sobre los archivos en el directorio de entrada
for filename in os.listdir(input_dir):
    print(f'Processing {filename}...')
    if filename.startswith("feynman") and filename.endswith(".csv"):
        # Leer el archivo CSV
        file_path = os.path.join(input_dir, filename)
        df = pd.read_csv(file_path, header=None)

        # Dividir el DataFrame a la mitad
        half = len(df) // 2
        train = df.iloc[:half]
        test = df.iloc[half:]
        
        # Guardar las mitades en los directorios correspondientes
        train.to_csv(os.path.join(train_dir, filename), index=False, header=False)
        test.to_csv(os.path.join(test_dir, filename), index=False, header=False)
        
        print(f'Processed {filename}: Train and test files saved.')

print('All files have been processed.')
