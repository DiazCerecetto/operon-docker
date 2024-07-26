
# Operon - docker
## Instalación
### 1. Crear la imagen:

```
    docker build -t operon .
```

### 1. Imágen de Docker Hub
También es posible descargar la imágen desde Docker Hub, mediante el siguiente comando

```
    docker pull diazcerecetto/operon-docker
```

### 2. Ejecutar con el volumen apropiado para tener los archivos más cómodamente para programar
Cambiar "ruta" por el path directorio local, la siguiente línea debería imprimir el directorio local (powershell)

```
    echo $(pwd)
    docker run -v <ruta_al_directorio_local>/home/app -it operon-docker /bin/bash
```

### 3. Luego, en la terminal de docker:

Activar el environment de conda

```
    conda activate operon
```

Instalar operon (demora aprox 30-50 minutos)

```
    bash install.sh
```

Instalar librerías adicionales

```
    pip install matplotlib
```

Luego, ejecutar un archivo de python

```
    python test_operon.py
```

## Réplica del experimento
Primero se deben generar los puntos de las funciones de Feynman, para ello se debe ejecutar primero el siguiente comando, este preguntará cuantos elementos se necesitan en el conjunto de entrenamiento y test, se debe indicar un entero en cada caso.

```
    python code/feynman_generate.py
```

Luego, se puede replicar el experimento mediante el siguiente comando, este solicitará el número de iteraciones a ejecutar.

```
    python code/operon_feynman.py 
```
Finalmente, los resultados se encuentran en el directorio Resultados/resultados_feynman.csv


## Bibliografía
El archivo install.sh fue extraído de:
https://github.com/cavalab/srbench

Código de pyoperon disponible en:
https://github.com/heal-research/pyoperon
