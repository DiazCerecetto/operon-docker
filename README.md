
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
Cambiar "ruta_al_directorio_local" por el path directorio local, la siguiente línea debería imprimir el directorio local (powershell)

```
    echo $(pwd)
    docker run -v <ruta_al_directorio_local>/home/app -it operon-docker /bin/bash
```

### 3. Luego, en la terminal de docker:

Activar el environment de conda

```
    conda activate operon
```

Instalar operon (demora aprox 30-50 minutos). NOTA: Este paso y el siguiente no son necesarios si se instaló la imágen mediante docker pull

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

Se pueden ejecutar ambas fases desde el archivo principal 

```
    python code/main.py
```

Cada ejecución del experimento creará una carpeta nueva con nombre único que contendrá los archivos con las predicciones y su evaluación.

## Bibliografía
El archivo install.sh fue extraído de:
https://github.com/cavalab/srbench

Código de pyoperon disponible en:
https://github.com/heal-research/pyoperon