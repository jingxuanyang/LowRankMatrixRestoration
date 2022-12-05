import numpy as np


class RobustPCA:
    """ Robust principal component analysis.

    Problems
    --------
    $\min \|L\|_{*} + \lambda \|S\|_{m_1}, s.t. L+S=M.$
    """

    def __init__(self, M : np.ndarray, method="ALM", delta=None, mu=None, mu_ratio=1, lmbda=None):
        self.M = M
        self.L = np.zeros_like(self.M)
        self.S = np.zeros_like(self.M)
        self.Y = np.zeros_like(self.M)
        self.method = method
        self.delta = delta if delta else 0.1
        self.mu = mu if mu else mu_ratio * np.prod(self.M.shape) / (4 * np.linalg.norm(self.M, ord=1))
        self.mu_inv = 1 / self.mu
        self.min_mu = self.mu * 0.01
        self.lmbda = lmbda if lmbda else 1 / np.sqrt(np.max(self.M.shape))
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8

    @staticmethod
    def frobenius_norm(M):
        """ Frobenius norm of matrices.
        """
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        """ Shrink operator of matrices.
        """
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros_like(M))

    def svd_threshold(self, M, tau):
        """ Thresholding operator of matrices with SVD.
        """
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def l1_entry(self, L):
        """ Entrywise l1 norm of a matrix.
        """
        return np.abs(self.M - L).sum()

    def objective(self, L):
        """ Objective function.
        """
        f = np.linalg.norm(L, ord='nuc') + self.lmbda * self.l1_entry(L)
        return f

    def subgrad(self, L):
        """ Subgradient of the objective function.
        """
        U, _, V_T = np.linalg.svd(L)
        r = np.min(self.M.shape)
        sg = U[:, :r] @ V_T[:r, :] - self.lmbda * np.sign(self.M - L)
        return sg
    
    def lr(self, iter_ratio):
        """ Learning rate adjust.
        """
        return self.mu - iter_ratio * (self.mu - self.min_mu)

    def _crude_gradient_descent(self, tol=0.001, max_iter=1000, iter_print=100, Adam=False):
        """ Crude gradient descent.
        """
        iter = 0
        err = 1
        L = self.L
        f = [self.objective(L)]
        m, v = 0, 0
        terminate = False
        while not terminate and iter < max_iter:
            G = self.subgrad(L)
            r = iter / max_iter
            a = self.mu if r < 0.1 else self.lr(r)
            if Adam:
                m = self.beta_1 * m + (1 - self.beta_1) * G
                v = self.beta_2 * v + (1 - self.beta_2) * G * G
                m_h = m / (1 - self.beta_1**(iter + 1))
                v_h = v / (1 - self.beta_2**(iter + 1))
                L_ = L - a * m_h / (np.sqrt(v_h) + self.epsilon)
            else:
                L_ = L - a * G
            f.append(self.objective(L_))
            iter += 1
            if iter > 10:
                err = f[iter-10] - f[iter]
                terminate = abs(err) < tol
            if (iter % iter_print) == 0 or iter == 1 or terminate:
                print('iteration: {0}, rank: {1}, error: {2}'.format(
                      iter, np.linalg.matrix_rank(L_), err))
            L = L_
        S = self.M - L
        return L, S

    def _singular_value_thresholding(self, tol=None, max_iter=1000, iter_print=100):
        """ Singular value thresholding.

        References:
        -----------
        Jian-Feng Cai, Emmanuel J. Candes, Zuowei Shen. (2008). A Singular Value Thresholding Algorithm for Matrix Completion.
        """
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.M.shape)
        _tol = tol if tol else 1E-7 * self.frobenius_norm(self.M)
        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(self.mu_inv * Yk, self.mu_inv * self.lmbda)
            Yk += self.delta * (self.M - Lk - Sk)
            err = self.frobenius_norm(self.M - Lk - Sk)
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print('iteration: {0}, rank: {1}, error: {2}'.format(
                      iter, np.linalg.matrix_rank(Lk), err))
        return Lk, Sk

    def _accelerated_proximal_gradient(self, tol=None, max_iter=1000, iter_print=100):
        """ Accelerated proximal gradient.

        References:
        -----------
        Zhouchen Lin, Arvind Ganesh, John Wright, Leqin Wu, Minming Chen and Yi Ma. (2009). Fast Convex Optimization Algorithms for Exact Recovery of a Corrupted Low-Rank Matrix.
        """
        iter = 0
        t = 1
        eta = 0.9
        mu = 0.99 * np.linalg.norm(self.M, ord=2)
        mu_hat = 1e-6 * mu
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Yk_ = self.Y
        Lk = np.zeros(self.M.shape)
        _tol = tol if tol else 1E-7 * self.frobenius_norm(self.M)
        while (err > _tol) and iter < max_iter:
            Lk_ = self.svd_threshold(Yk + (self.M - Yk - Yk_) / 2, mu / 2)
            Sk_ = self.shrink(Yk_ + (self.M - Yk - Yk_) / 2, self.lmbda * mu / 2)
            t_ = (1 + np.sqrt(1 + 4 * t**2)) / 2
            Yk = Lk_ + (t - 1) * (Lk_ - Lk) / t_
            Yk_ = Sk_ + (t - 1) * (Sk_ - Sk) / t_
            t, Lk, Sk = t_, Lk_, Sk_
            mu = max(eta * mu, mu_hat)
            err = self.frobenius_norm(self.M - Lk - Sk)
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print('iteration: {0}, rank: {1}, error: {2}'.format(
                      iter, np.linalg.matrix_rank(Lk), err))
        return Lk, Sk

    def _augmented_Lagrange_multipliers(self, tol=None, max_iter=1000, iter_print=100):
        """ Augmented Lagrange multipliers.

        References:
        -----------
        Emmanuel J. Candes, Xiaodong Li, Yi Ma, and John Wright. (2009). Robust Principal Component Analysis?
        """
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.M.shape)
        _tol = tol if tol else 1E-7 * self.frobenius_norm(self.M)
        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(self.M - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(self.M - Lk + self.mu_inv * Yk, self.mu_inv * self.lmbda)
            Yk += self.mu * (self.M - Lk - Sk)
            err = self.frobenius_norm(self.M - Lk - Sk)
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print('iteration: {0}, rank: {1}, error: {2}'.format(
                      iter, np.linalg.matrix_rank(Lk), err))
        return Lk, Sk

    def fit(self, tol=None, max_iter=1000, iter_print=100, Adam=False):
        """ Call specific method to solve the robust PCA.
        """
        if self.method == "ALM":
            return self._augmented_Lagrange_multipliers(tol=tol, max_iter=max_iter, iter_print=iter_print)
        elif self.method == "SVT":
            return self._singular_value_thresholding(tol=tol, max_iter=max_iter, iter_print=iter_print)
        elif self.method == "APG":
            return self._accelerated_proximal_gradient(tol=tol, max_iter=max_iter, iter_print=iter_print)
        elif self.method == "CGD":
            return self._crude_gradient_descent(tol=tol, max_iter=max_iter, iter_print=iter_print, Adam=Adam)
        else:
            print("Error! Input method is illegal.")
            return self.M, self.S

