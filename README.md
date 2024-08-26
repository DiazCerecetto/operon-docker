
# Operon - docker
## Instalación
### 1. Mediante instalación local

Para instalar localmente, se debe descargar la imagen partiendo de una instalación de miniconda3 y posteriormente instalando el software necesario

```
    docker pull continuumio/miniconda3 
```

```
    docker run -v C:\Users\admin\Desktop\clone\operon-docker:/home/app -it continuumio/miniconda3 /bin/bash
```

Cambiar "ruta_al_directorio_local" por el path directorio local, la siguiente línea debería imprimir el directorio local (powershell)

```
    echo $(pwd)
    docker run -v <ruta_al_directorio_local>:/home/app -it continuumio/miniconda3 /bin/bash
```

Luego, instalar las librerías necesarias
```
    cd /home/app
    bash createdocker.sh
    conda activate operon
```
Instalar operon y pyoperon junto con sus dependencias y los cambios de este repositorio

```
    bash install.sh
```

Instalar librerías adicionales

```
    pip install matplotlib
    pip install jmetalpy
```


### 1.1 Ejecutar
Si ya se tiene creado el contenedor con los pasos previos, inicializar el contenedor y ejecutar el siguiente comando:

```
    docker exec -it <id_del_contenedor> /bin/bash
    cd /home/app
    conda activate operon
```

### 1. Imágen de Docker Hub
También es posible descargar la imágen desde Docker Hub, mediante el siguiente comando

```
    docker pull diazcerecetto/operon-docker
```

Se debe crear un contenedor y utilizar el comando de la sección 1.1


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