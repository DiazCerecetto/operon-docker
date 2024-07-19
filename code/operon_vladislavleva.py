import pandas as pd
import numpy as np
from SRutils import *


def main():
    nombre_archivo = PATH_VLADISLAVLEVA + "/vladislavleva1.csv"
    df = pd.read_csv(nombre_archivo)
    est = obtener_modelo()
    est, m = predecir_desde_csv(est, df, "vladislavleva1")
    # Probar en el conjunto de test
    nombre_archivo = PATH_VLADISLAVLEVA + "/vladislavleva2.csv"
    df = pd.read_csv(nombre_archivo)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    r2 = est.score(X, y)
    print(X.shape, y.shape)
    print("R2: ", r2)
    print("Modelo: ", est.get_model_string())

if __name__ == "__main__":
    main()
