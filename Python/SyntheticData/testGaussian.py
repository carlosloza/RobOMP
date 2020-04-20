from RobOMP import gOMP
import numpy as np

m = 100
n = 500
K = 5
D = np.random.normal(0, 1, m*n).reshape((m, n))
D = np.divide(D, np.linalg.norm(D, axis=0))
x0 = np.zeros((n, 1))
#x0 = np.zeros((n))
#x0[2] = np.random.normal(0, 1, K).reshape((K, 1))
x0[np.random.choice(n, K, replace=False)] = np.random.normal(0, 1, K).reshape((K, 1))
y = np.matmul(D, x0)
#x0[np.random.choice(n, K, replace=False)] = 10
#print(y.shape)

omp = gOMP(nnonzero = K, N0 = 1)
x, e, X, E = omp.fit(y, D)

print(np.linalg.norm(x0 - x))

