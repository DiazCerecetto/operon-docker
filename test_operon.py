from operon.sklearn import SymbolicRegressor
import numpy as np
modelo = SymbolicRegressor()
# crear caso de prueba con (1,1) y (2,2)
X = np.array([[1,1],[2,2]])
y = np.array([2,4])
modelo.fit(X,y)
# obtener el modelo en string
print(modelo.get_model_string())
# obtener el score en test
X = np.array([[3,3],[4,4]])
y = np.array([6,9]) 
print(modelo.score(X,y))