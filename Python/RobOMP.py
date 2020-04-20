import numpy as np


class gOMP:

    def __init__(self, nnonzero=None, tol=None, N0=1):
        self.nnonzero = nnonzero
        self.tol = tol
        self.N0 = N0

    def fit(self, y, D):
        m, n = D.shape
        # Check inputs and set defaults
        if self.nnonzero is not None and self.tol is None:
            flcase = 1
        elif self.nnonzero is None and self.tol is not None:
            flcase = 2
        elif self.nnonzero is None and self.tol is None:
            flcase = 3

        if flcase == 1:
            self.tol = -1
        elif flcase == 2:
            self.nnonzero = n
        elif flcase == 3:
            self.nnonzero = n
            self.tol = 0.1

        if self.nnonzero % self.N0 < 0:
            print("nnonzero is not multiple of N0")
        elif self.nnonzero % self.N0 > 0:
            print("N0 is larger than nnonzero")
            self.N0 = self.nnonzero

        maxiter = int(self.nnonzero / self.N0)
        normy = np.linalg.norm(y)
        r = y
        i = -1
        X = np.zeros((n, maxiter))
        E = np.zeros((m, maxiter))
        idx_spcode = np.empty((0,0), dtype=int)
        while np.linalg.norm(r) / normy > self.tol and i < (maxiter - 1):
            i = i + 1
            abscorr = np.absolute(np.dot(r.T, D))
            idx = np.argsort(-abscorr, axis = None)
            idx_spcode = np.append(idx_spcode, idx[0:self.N0])
            b, _, _, _ = np.linalg.lstsq(D[:, idx_spcode], y, rcond=None)
            X[idx_spcode, i] = b.flatten()
            r = y - np.matmul(D[:, idx_spcode], b)
            E[:, i] = r.flatten()

        X = X[:, 0:i + 1]
        E = E[:, 0:i + 1]
        x = X[:, -1].reshape((n, 1))
        e = E[:, -1].reshape((m, 1))

        return x, e, X, E