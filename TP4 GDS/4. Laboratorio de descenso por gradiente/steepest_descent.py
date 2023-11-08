"""Filtrado óptimo Wiener con descenso por gradiente.

22.46 Procesamiento adaptativo de Señales Aleatorias
"""

import numpy as np

def steepest_descent(R, p, w0, mu, N):
	"""Implementa el filtrado óptimo Wiener con descenso por gradiente.

	Argumentos:
		R: matriz de autocorrelación
		p: matriz de correlación cruzada
		w0: valor inicial de los coeficientes del filtro
		mu: tamaño de paso
		N: número máximo de iteraciones

	Devuelve:
		Una matriz de tipo np.array en cuyas filas están
		los coeficientes w para cada paso.
	"""
	Wt = np.zeros((N, len(w0)))
	Wt[0] = w0
	for n, w_n_1 in enumerate(Wt):
		if n != len(Wt)-1:
			grad_J_2 = np.dot(R, w_n_1) - p
			Wt[n+1] = w_n_1 + mu * grad_J_2
	return Wt


R = [[2,1],[1,2]]
p = [6,4]
mu = 0.1
N = 1000
wo = [0,0]
sigma_d = np.sqrt(20)

matrix = steepest_descent(R, p, wo, mu, N)