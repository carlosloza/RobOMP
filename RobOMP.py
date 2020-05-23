import numpy as np
"""
Robust Orthogonal Matching Pursuit (RobOMP) algorithms 
Author: Carlos Loza
Part of RobOMP package. DOI: 10.7717/peerj-cs.192 (open access)
https://github.carlosloza/RobOMP
"""


class gOMP:
    """ Generalized Orthogonal Matching Pursuit (gOMP).

    Parameters
    ----------
    nnonzero :  int, optional
                Number of non-zero coefficients in sparse code
                Default: n, i.e. extreme non-sparse case.
    tol :       float, optional
                Residual norm tolerance. Dispersion/power rate not explained by
                the sparse code with respect to the norm of y
                Default: 0.1, i.e. 10% of the L2 norm of input y.
    N0 :        int, optional
                Number of atoms chosen per iteration
                Default: 1, i.e. regular orthogonal matching pursuit (OMP).
    verbose :   bool, optional
                Enable verbose output.

    If nnonzero is not a multiple of N0, a warning flag is displayed
    and the actual number of non-zero coefficients in the sparse code
    is set to N0*floor(nnonzero/N0)
    nnonzero is equal to the number of iterations (i.e. OMP case),
    only when N0 = 1
    If neither nnonozero nor tol are set, then tol is set to default
    If both nnonzero and tol are set, then the algorithm stops when both conditions
    are met

    Attributes
    ----------
    coef_ :         array, shape (n_atoms,)
                    Sparse code (X in formula).
    n_iter_ :       int
                    Number of sequential ordinary least squares (OLS) estimations.
    error :         array, shape (n_features, 1)
                    Residue/error after after sparse coding of y with sparsity
                    level nnonzero.
    coef_iter :     array, shape (n_atoms, n_iter_)
                    Same as coef_, but each column corresponds to decreasingly
                    sparser solutions according to n_iter_.
    error_iter :    array, shape (n_features, n_iter_)
                    Same as error, but each column corresponds to residue after
                    decreasingly sparser solutions, i.e. likewise X.
    coef_iter_ext : array, shape (n_atoms, nnonzero)
                    Same as coef_iter, but each column corresponds to decreasingly
                    sparse solutions according to nnonzero. This array will have
                    repeated inputs/columns if N0 ~= 0. It is mainly used for
                    comparisons with classic OMP encoders where N0 = 1.
    error_iter_ext: array, shape (n_features, nnonzero)
                    Same as error, but each column corresponds to residue after
                    decreasingly sparser solutions according to nnonzero,
                    i.e. likewise coef_iter_ext.

    Examples
    --------
    >>> from RobOMP import gOMP
    >>> from sklearn.datasets import make_sparse_coded_signal
    >>> n_features, n_components = 100, 500
    >>> n_nonzero_coefs = 10
    >>> y, X, w = make_sparse_coded_signal(n_samples=1,
                                   n_components=n_components,
                                   n_features=n_features,
                                   n_nonzero_coefs=n_nonzero_coefs,
                                   random_state=0)
    >>> scgOMP = gOMP(nnonzero=n_nonzero_coefs, N0 = 1).fit(X, y)

    Notes
    -----
    Generalized Orthogonal Matching Pursuit (gOMP) was introduced
    by Wang, Kwon, and Shim 2011 (DOI: 10.1109/TSP.2012.2218810)
    """
    def __init__(self, nnonzero=None, tol=None, N0=1, verbose=False):
        self.nnonzero = nnonzero
        self.tol = tol
        self.N0 = N0
        self.verbose = verbose

    def fit(self, D, y):
        """ Fit the sparse model using D, y as training data, i.e. sparse coding.

        Parameters
        ----------
        D :     array, shape (n_features/dimensions, n_atoms)
                Dictionary/measurement matrix made up of atoms.
        y :     array, shape (n_features, 1) or (n_features,)
                Signal to be sparsely encoded.

        Returns
        -------
        self :  object
                returns an instance of self.
        """
        m, n = D.shape
        # Check inputs
        if self.nnonzero is not None and self.tol is None:
            flcase = 1
        elif self.nnonzero is None and self.tol is not None:
            flcase = 2
        elif self.nnonzero is None and self.tol is None:
            flcase = 3

        # Set defaults if variables were not set
        if flcase == 1:
            self.tol = -1
        elif flcase == 2:
            self.nnonzero = n           # Extreme case
        elif flcase == 3:
            self.nnonzero = n           # Extreme case
            self.tol = 0.1

        nnonzero_ini = self.nnonzero
        # Check if N0 is larger than nnonzero and if nnonzero is multiple of N0
        if self.N0 > self.nnonzero:
            if self.verbose is True:
                print("N0 is larger than nnonzero. N0 will be set to nnonzero")
            self.N0 = self.nnonzero
            nnonzero_ini = self.N0
        if self.nnonzero % self.N0 != 0:
            if self.verbose is True:
                print("nnonzero is not multiple of N0. Actual support of sparse code will be decreased")
            self.nnonzero = int(self.N0*np.floor(self.nnonzero/self.N0))

        n_iter_ = int(self.nnonzero / self.N0)
        y = y.flatten()
        normy = np.linalg.norm(y)
        r = y
        i = -1
        X = np.zeros((n, n_iter_))
        E = np.zeros((m, n_iter_))
        idx_spcode = np.empty((0, 0), dtype=int)
        while np.linalg.norm(r) / normy > self.tol and i < (n_iter_ - 1):
            i += 1
            abscorr = np.absolute(np.dot(r.T, D))
            idx = np.argsort(-abscorr, axis = None)
            if len(np.intersect1d(idx_spcode, idx[0:self.N0])) > 0:
                # Case for repeated atoms
                X[:, i:] = np.tile(X[:, i - 1], (n_iter_ - i, 1)).T
                E[:, i:] = np.tile(E[:, i - 1], (n_iter_ - i, 1)).T
                i = n_iter_ - 1
                if self.verbose is True:
                    print("Repeated atom detected. Algorithm stops.")
                break

            idx_spcode = np.append(idx_spcode, idx[0:self.N0])
            # Ordinary least squares (OLS) regression
            b = np.linalg.lstsq(D[:, idx_spcode], y, rcond=None)[0]
            X[idx_spcode, i] = b.flatten()
            r = y - np.matmul(D[:, idx_spcode], b)      # Residue
            E[:, i] = r.flatten()

        # Extended versions for comparisons with classic OMP encoders
        Xext = np.zeros((n, self.nnonzero))
        Eext = np.zeros((m, self.nnonzero))
        ct = 0
        for i in range(0, self.nnonzero, self.N0):
            Xext[:, i:i + self.N0] = np.tile(X[:, ct], (self.N0, 1)).T
            Eext[:, i:i + self.N0] = np.tile(E[:, ct], (self.N0, 1)).T
            ct += 1
        if nnonzero_ini > self.nnonzero:
            Xext = np.concatenate((Xext, np.tile(X[:, -1], (nnonzero_ini - self.nnonzero, 1)).T), axis=1)
            Eext = np.concatenate((Eext, np.tile(E[:, -1], (nnonzero_ini - self.nnonzero, 1)).T), axis=1)

        self.coef_ = X[:, -1]
        self.n_iter_ = n_iter_
        self.error = E[:, -1].reshape((m, 1))
        self.coef_iter = X[:, 0:i + 1]
        self.error_iter = E[:, 0:i + 1]
        self.coef_iter_ext = Xext
        self.error_iter_ext = Eext

        return self


class CMP:
    """ Correntropy Matching Pursuit (CMP).

    Parameters
    ----------
    nnonzero :  int, optional
                Number of non-zero coefficients in sparse code
                Default: n, i.e. extreme non-sparse case.
    tol :       float, optional
                Residual norm tolerance. Dispersion/power rate not explained by
                the sparse code with respect to the norm of y
                Default: 0.1, i.e. 10% of the L2 norm of input y.
    verbose :   bool, optional
                Enable verbose output.

    If neither nnonozero nor tol are set, then tol is set to default
    If both nnonzero and tol are set, then the algorithm stops when both conditions
    are met

    Attributes
    ----------
    coef_ :         array, shape (n_atoms,)
                    Sparse code (X in formula).
    n_iter_ :       int
                    Number of sequential ordinary least squares (OLS) estimations.
    weights_ :      array, shape (n_features,)
                    Weights associated to each entry/feature of input array.
    error :         array, shape (n_features, 1)
                    Residue/error after after sparse coding of y with sparsity
                    level nnonzero.
    coef_iter :     array, shape (n_atoms, n_iter_)
                    Same as coef_, but each column corresponds to decreasingly
                    sparser solutions according to n_iter_.
    error_iter :    array, shape (n_features, n_iter_)
                    Same as error, but each column corresponds to residue after
                    decreasingly sparser solutions, i.e. likewise X.

    Examples
    --------
    >>> from RobOMP import CMP
    >>> from sklearn.datasets import make_sparse_coded_signal
    >>> n_features, n_components = 100, 500
    >>> n_nonzero_coefs = 10
    >>> y, X, w = make_sparse_coded_signal(n_samples=1,
                                   n_components=n_components,
                                   n_features=n_features,
                                   n_nonzero_coefs=n_nonzero_coefs,
                                   random_state=0)
    >>> scCMP = CMP(nnonzero=n_nonzero_coefs).fit(X, y)

    Notes
    -----
    Correntropy Matching Pursuit (CMP) was introduced
    by Wang et al. 2017 (DOI: 10.1109/TCYB.2016.2544852)
    """
    def __init__(self, nnonzero=None, tol=None, verbose=False):
        self.nnonzero = nnonzero
        self.tol = tol
        self.verbose = verbose

    def CorrentropyReg(self, X, y):
        """ Fit the linear model using X, y as training data, i.e. robust regression.

        Parameters
        ----------
        X :     array, shape (n_features/dimensions, K)
                K different regressors, input/independent variables in a linear
                regression framework. For our problem, X is made up of K atoms
                from the dictionary D.
        y :     array, shape (n_features, 1) or (n_features,)
                Dependent/response/measured variable in a linear regression
                framework.

        Returns
        -------
        b :     array, shape (K,)
                Effects or regression coefficients in a linear regression
                framework.
        w :     array, shape (n_features,)
                Weights associated to each entry/feature of input array.
        """

        max_it = 100                # Maximum number of iterations
        th = 0.01                   # Stopping threshold for IRLS
        inv_const = 0.00001         # To avoid matrix-inversion-related errors
        m = y.size
        d, n = X.shape
        X2 = X.T @ X                # Compute to accelerate computations
        # Initial estimate: OLS
        b = np.linalg.lstsq(X2 + inv_const*np.identity(n), X.T @ y, rcond=None)[0]
        e = y - X @ b
        # Estimate sigma of gaussian kernel of correntropy
        sig = np.sqrt((np.sum(np.square(e)))/(2*m))
        # Fast calculation of matrix multiplications
        JM = np.zeros((n, n, d))
        for k in np.arange(d):
            JM[:, :, k] = np.outer(X[k, :], X[k, :])
        JM = np.reshape(JM, (n**2, d))
        w = np.exp(-np.square(e)/(2 * sig**2))
        bprev = b
        it = 1
        fl = 1
        # IRLS - Iteratively Reweighted Least Squares
        while fl:
            Xmul = np.reshape(JM @ w, (n, n))
            #b = np.linalg.lstsq(Xmul + inv_const*np.identity(n), (X * w).T @ y, rcond=None)[0]
            b = np.linalg.lstsq(Xmul + inv_const * np.identity(n), (X.T * w) @ y, rcond=None)[0]
            if np.sqrt(np.sum(np.square(b - bprev)))/np.sqrt(np.sum(np.square(bprev))) <= th:
                fl = 0
            else:
                # Compute values for weight array
                e = y - X @ b
                sig = np.sqrt((np.sum(np.square(e))) / (2 * m))
                w = np.exp(-np.square(e) / (2 * sig ** 2))
                bprev = b
                it += 1
            if it == max_it:
                fl = 0
                if self.verbose is True:
                    print("Solution did not converge in maximum number of iterations allowed")

        return b, w

    def fit(self, D, y):
        """ Fit the sparse model using D, y as training data, i.e. robust sparse coding.

        Parameters
        ----------
        D :     array, shape (n_features/dimensions, n_atoms)
                Dictionary/measurement matrix made up of atoms.
        y :     array, shape (n_features, 1) or (n_features,)
                Signal to be sparsely encoded.

        Returns
        -------
        self :  object
                returns an instance of self.
        """
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

        n_iter_ = self.nnonzero
        y = y.flatten()
        normy = np.linalg.norm(y)
        r = y
        i = -1
        X = np.zeros((n, n_iter_))
        E = np.zeros((m, n_iter_))
        idx_spcode = np.empty((0, 0), dtype=int)
        while np.linalg.norm(r) / normy > self.tol and i < (n_iter_ - 1):
            i = i + 1
            abscorr = np.absolute(np.dot(r.T, D))
            idx = np.argsort(-abscorr, axis=None)
            if len(np.intersect1d(idx_spcode, idx[0])) > 0:
                # repeated atoms
                X[:, i:] = np.tile(X[:, i - 1], (n_iter_ - i, 1)).T
                E[:, i:] = np.tile(E[:, i - 1], (n_iter_ - i, 1)).T
                i = n_iter_ - 1
                if self.verbose is True:
                    print("Repeated atom detected. Algorithm stops.")
                break

            idx_spcode = np.append(idx_spcode, idx[0])
            # Robust, correntropy-based regression
            b, w = self.CorrentropyReg(D[:, idx_spcode], y)
            X[idx_spcode, i] = b.flatten()
            r = y - np.matmul(D[:, idx_spcode], b)
            E[:, i] = r.flatten()

        self.coef_ = X[:, -1]
        self.n_iter_ = n_iter_
        self.weights_ = w
        self.error = E[:, -1].reshape((m, 1))
        self.coef_iter = X[:, 0:i + 1]
        self.error_iter = E[:, 0:i + 1]

        return self


class RobustOMP:
    """ Robust Orthogonal Matching Pursuit (RobOMP).

    Parameters
    ----------
    nnonzero :      int, optional
                    Number of non-zero coefficients in sparse code
                    Default: n, i.e. extreme non-sparse case.
    tol :           float, optional
                    Residual norm tolerance. Dispersion/power rate not explained by
                    the sparse code with respect to the norm of y
                    Default: 0.1, i.e. 10% of the L2 norm of input y.
    warmst :        bool, optional
                    Flag that indicates if Huber variant is used as initial solution.
                    Default: True
    m_est :         string, optional
                    M-estimator to use. Options: Cauchy, Fair, Huber, Tukey, Welsch.
                    Default: Tukey
    m_est_hyperp :  float, optional
                    Hyperparameter of m-estimators. Defaults are set in the "fit"
                    method according to 95% asymptotic efficiency on the standard
                    Normal distribution, see Table 2 of Loza 2019 for specifics.
    verbose :       bool, optional
                    Enable verbose output.

    If neither nnonozero nor tol are set, then tol is set to default
    If both nnonzero and tol are set, then the algorithm stops when both conditions
    are met

    Attributes
    ----------
    coef_ :         array, shape (n_atoms,)
                    Sparse code (X in formula).
    n_iter_ :       int
                    Number of sequential ordinary least squares (OLS) estimations.
    weights_ :      array, shape (n_features,)
                    Weights associated to each entry/feature of input array.
    error :         array, shape (n_features, 1)
                    Residue/error after after sparse coding of y with sparsity
                    level nnonzero.
    coef_iter :     array, shape (n_atoms, n_iter_)
                    Same as coef_, but each column corresponds to decreasingly
                    sparser solutions according to n_iter_.
    error_iter :    array, shape (n_features, n_iter_)
                    Same as error, but each column corresponds to residue after
                    decreasingly sparser solutions, i.e. likewise X.

    Examples
    --------
    >>> from RobOMP import RobustOMP
    >>> from sklearn.datasets import make_sparse_coded_signal
    >>> n_features, n_components = 100, 500
    >>> n_nonzero_coefs = 10
    >>> y, X, w = make_sparse_coded_signal(n_samples=1,
                                   n_components=n_components,
                                   n_features=n_features,
                                   n_nonzero_coefs=n_nonzero_coefs,
                                   random_state=0)
    >>> scCMP = RobustOMP(nnonzero=n_nonzero_coefs, m_est='Tukey').fit(X, y)

    Notes
    -----
    Robust Orthogonal Matching Pursuit (RobOMP) was introduced
    by Loza 2019 (DOI: 10.7717/peerj-cs.192)
    """
    def __init__(self, nnonzero=None, tol=None, warmst=True, m_est='Tukey', m_est_hyperp=None, verbose=False):
        self.nnonzero = nnonzero
        self.tol = tol
        self.warmst = warmst
        self.m_est = m_est
        self.m_est_hyperp = m_est_hyperp
        self.verbose = verbose

    # Cauchy m-estimator weighting
    def Cauchy(self, e, s, m_est_hyperp):
        res = e/(m_est_hyperp * s)
        w = 1/(1 + res**2)
        return w
    # Fair m-estimator weighting
    def Fair(self, e, s, m_est_hyperp):
        res = e/(m_est_hyperp * s)
        w = 1/(1 + abs(res))
        return w
    #  Huber m-estimator weighting
    def Huber(self, e, s, m_est_hyperp):
        res = e/s
        w = np.ones(e.size)
        idx = abs(res) >= m_est_hyperp
        w[idx] = m_est_hyperp/abs(res[idx])
        return w
    # Tukey m-estimator weighting
    def Tukey(self, e, s, m_est_hyperp):
        res = e/s
        w = np.zeros(e.size)
        idx = abs(res) < m_est_hyperp
        w[idx] = (1 - (res[idx]/m_est_hyperp)**2)**2
        return w
    # Welsch m-estimator weighting
    def Welsch(self, e, s, m_est_hyperp):
        res = e/(m_est_hyperp * s)
        w = np.exp(-(res**2))
        return w

    def MEstimator(self, X, y, m_est='Huber', m_est_hyperp=1.345, b=None):
        """ M-estimator-based regression.
        Fit the linear model using X, y as training data.

        Parameters
        ----------
        X :             array, shape (n_features/dimensions, K)
                        K different regressors, input/independent variables in a linear
                        regression framework. For our problem, X is made up of K atoms
                        from the dictionary D.
        y :             array, shape (n_features, 1) or (n_features,)
                        Dependent/response/measured variable in a linear regression
                        framework.
        m_est :         string, optional
                        M-estimator to use. Options: Cauchy, Fair, Huber, Tukey, Welsch.
                        Default: Huber (for warmstart)
        m_est_hyperp:   float, optional
                        Hyperparameter of m-estimators.
                        Default: 1.345 correspoding to Huber variant (for warmstart)
        b :             array, shape (K,)
                        Initial solution of linear regression problem (for warmstart)

        Returns
        -------
        b :             array, shape (K,)
                        Effects or regression coefficients in a linear regression
                        framework.
        w :             array, shape (n_features,)
                        Weights associated to each entry/feature of input array.
        """
        # Handles for m-estimator methods
        if m_est.lower() == "Cauchy".lower():
            method_m_est = getattr(RobustOMP, 'Cauchy')
        elif m_est.lower() == "Fair".lower():
            method_m_est = getattr(RobustOMP, 'Fair')
        elif m_est.lower() == "Huber".lower():
            method_m_est = getattr(RobustOMP, 'Huber')
        elif m_est.lower() == "Tukey".lower():
            method_m_est = getattr(RobustOMP, 'Tukey')
        elif m_est.lower() == "Welsch".lower():
            method_m_est = getattr(RobustOMP, 'Welsch')

        max_it = 100                # Maximum number of iterations
        th = 0.01                   # Stopping threshold for IRLS
        inv_const = 0.00001         # To avoid matrix-inversion-related errors
        d, n = X.shape
        # If no initial solution is provided, use OLS
        if b is None:
            X2 = X.T @ X            # Compute to accelerate computations
            b = np.linalg.lstsq(X2 + inv_const * np.identity(n), X.T @ y, rcond=None)[0]
        e = y - X @ b
        # Estimate scale
        s = 1.4824 * np.median(abs(e - np.median(e)))
        w = method_m_est(self, e, s, m_est_hyperp)
        # Fast calculation of matrix multiplications
        JM = np.zeros((n, n, d))
        for k in np.arange(d):
            JM[:, :, k] = np.outer(X[k, :], X[k, :])
        JM = np.reshape(JM, (n**2, d))
        bprev = b
        it = 1
        fl = 1
        # IRLS - Iteratively Reweighted Least Squares
        while fl:
            Xmul = np.reshape(JM @ w, (n, n))
            b = np.linalg.lstsq(Xmul + inv_const * np.identity(n), (X.T * w) @ y, rcond=None)[0]
            if np.sqrt(np.sum(np.square(b - bprev)))/np.sqrt(np.sum(np.square(bprev))) <= th:
                fl = 0
            else:
                # Compute values for weight array
                e = y - X @ b
                s = 1.4824 * np.median(abs(e - np.median(e)))
                w = method_m_est(self, e, s, m_est_hyperp)
                bprev = b
                it += 1
            if it == max_it:
                fl = 0
                if self.verbose is True:
                    print("Solution did not converge in maximum number of iterations allowed")
        return b, w

    def RobustReg(self, X, y):
        """ Wrapper for M-estimator-based regression.
        Fit the linear model using X, y as training data.

        Parameters
        ----------
        X :     array, shape (n_features/dimensions, K)
                K different regressors, input/independent variables in a linear
                regression framework. For our problem, X is made up of K atoms
                from the dictionary D.
        y :     array, shape (n_features, 1) or (n_features,)
                Dependent/response/measured variable in a linear regression
                framework.

        Returns
        -------
        b :     array, shape (K,)
                Effects or regression coefficients in a linear regression
                framework.
        w :     array, shape (n_features,)
                Weights associated to each entry/feature of input array.
        """
        if self.warmst is True:
            # Warm start
            # Initial robust solution: Huber variant
            b_ini = self.MEstimator(X, y, m_est = 'Huber', m_est_hyperp = 1.345)[0]
            # Solve for selected m-estimator
            b, w = self.MEstimator(X, y, m_est = self.m_est, m_est_hyperp = self.m_est_hyperp, b = b_ini)
        else:
            # Cold start - initial solution is usual OLS
            b, w = self.MEstimator(X, y, m_est=self.m_est, m_est_hyperp=self.m_est_hyperp)
        return b, w

    def fit(self, D, y):
        """ Fit the sparse model using D, y as training data, i.e. robust sparse coding.

        Parameters
        ----------
        D :     array, shape (n_features/dimensions, n_atoms)
                Dictionary/measurement matrix made up of atoms.
        y :     array, shape (n_features, 1) or (n_features,)
                Signal to be sparsely encoded.

        Returns
        -------
        self :  object
                returns an instance of self.
        """
        m, n = D.shape
        # Check inputs and set defaults
        if self.nnonzero is not None and self.tol is None:
            flcase = 1
        elif self.nnonzero is None and self.tol is not None:
            flcase = 2
        elif self.nnonzero is None and self.tol is None:
            flcase = 3

        # Optimal default hyperparameters
        if self.m_est.lower() == "Cauchy".lower():
            self.m_est_hyperp = 2.385
        elif self.m_est.lower() == "Fair".lower():
            self.m_est_hyperp = 1.4
        elif self.m_est.lower() == "Huber".lower():
            self.m_est_hyperp = 1.345
        elif self.m_est.lower() == "Tukey".lower():
            self.m_est_hyperp = 4.685
        elif self.m_est.lower() == "Welsch".lower():
            self.m_est_hyperp = 2.985

        if flcase == 1:
            self.tol = -1
        elif flcase == 2:
            self.nnonzero = n
        elif flcase == 3:
            self.nnonzero = n
            self.tol = 0.1

        n_iter_ = self.nnonzero
        y = y.flatten()
        normy = np.linalg.norm(y)
        r = y
        i = -1
        X = np.zeros((n, n_iter_))
        E = np.zeros((m, n_iter_))
        idx_spcode = np.empty((0, 0), dtype=int)
        while np.linalg.norm(r) / normy > self.tol and i < (n_iter_ - 1):
            i = i + 1
            abscorr = np.absolute(np.dot(r.T, D))
            idx = np.argsort(-abscorr, axis=None)
            if len(np.intersect1d(idx_spcode, idx[0])) > 0:
                # repeated atoms
                X[:, i:] = np.tile(X[:, i - 1], (n_iter_ - i, 1)).T
                E[:, i:] = np.tile(E[:, i - 1], (n_iter_ - i, 1)).T
                i = n_iter_ - 1
                if self.verbose is True:
                    print("Repeated atom detected. Algorithm stops.")
                break

            idx_spcode = np.append(idx_spcode, idx[0])
            # Robust, m-estimator based regression
            b, w = self.RobustReg(D[:, idx_spcode], y)
            X[idx_spcode, i] = b.flatten()
            r = y - np.matmul(D[:, idx_spcode], b)
            E[:, i] = r.flatten()

        self.coef_ = X[:, -1]
        self.n_iter_ = n_iter_
        self.weights_ = w
        self.error = E[:, -1].reshape((m, 1))
        self.coef_iter = X[:, 0:i + 1]
        self.error_iter = E[:, 0:i + 1]

        return self

