# Operon

Crear la imagen

    1. docker build -t operon .



Correr con el volumen, cambiar "ruta" por el path directorio local, la siguiente línea
debería imprimir el directorio local (powershell)

    2. echo $(pwd)

    3. docker run -v C:\repos\operon:/home/app -it operon /bin/bash


Luego, en la terminal de docker:

    Activar el environment de conda
    4. conda activate operon

    Instalar operon (demora aprox 30-50 minutos)
    5. bash install.sh

    Instalar librerías adicionales
    6. pip install matplotlib

    Correr el archivo de python con el hola mundo
    7. python operon_from_csv.py

## Una vez instalado, solo necesitamos levantar el container

