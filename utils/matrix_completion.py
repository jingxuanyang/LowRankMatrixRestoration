import numpy as np
import cvxpy as cp
import logging
from sklearn.utils.extmath import svd_flip
from scipy.sparse.linalg import svds


class MatrixCompletion:
    """ Matrix completion.
    """

    def __init__(self, M : np.ndarray, missing_idx : np.ndarray, method="ADMM", mu=None, mu_ratio=1, lmbda=None):
        self.M = M
        self.missing_idx = missing_idx
        self.known_idx = ~ missing_idx
        self.S = np.zeros(self.M.shape)
        self.Y = np.zeros(self.M.shape)
        self.method = method
        self.mask = np.ones_like(self.M)
        self.mask[self.missing_idx] = 0
        self.mu = mu if mu else mu_ratio * np.prod(self.M.shape) / (4 * np.linalg.norm(self.M, ord=1))
        self.mu_inv = 1 / self.mu
        self.lmbda = lmbda if lmbda else 1 / np.sqrt(np.max(self.M.shape))

    @staticmethod
    def frobenius_norm(M):
        """ Frobenius norm of matrices.
        """
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M : np.ndarray, tau):
        """ Shrink operator of matrices.
        """
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        """ Thresholding operator of matrices with SVD.
        """
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def proj(self, M):
        """ Projection operator of matrices.
        """
        S = np.zeros_like(M)
        S[self.missing_idx] = M[self.missing_idx]
        return S

    def _nuclear_norm_solve(self):
        """ Solve directly the nulear norm with CVXPY.
        """
        X = cp.Variable(shape=self.M.shape)
        objective = cp.Minimize(self.mu * cp.norm(X, "nuc") +
                            cp.sum_squares(cp.multiply(self.mask, X - self.M)))
        problem = cp.Problem(objective, [])
        problem.solve(solver=cp.SCS)
        return X.value, self.M - X.value

    def _alternating_direction(self, tol=None, max_iter=1000, iter_print=100):
        """ Alternating direction method (ADM).
        """
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.M.shape)
        _tol = tol if tol else 1E-7 * self.frobenius_norm(self.M)
        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(self.M - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.proj(self.M - Lk - self.mu_inv * Yk)
            Yk = Yk + self.mu * (self.M - Lk - Sk)
            err = self.frobenius_norm(self.M - Lk - Sk)
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print('iteration: {0}, rank: {1}, error: {2}'.format(
                      iter, np.linalg.matrix_rank(Lk), err))
        return Lk, Sk

    @staticmethod
    def _svd(M : np.ndarray, k):
        (U, S, V) = svds(M, k=min(k, min(M.shape)-1))
        S = S[::-1]
        U, V = svd_flip(U[:, ::-1], V[::-1])
        return U, S, V

    def _singular_value_thresholding(self, epsilon=1e-2, max_iter=1000, iter_print=100):
        """ Iterative singular value thresholding (SVT).
        """
        Y = np.zeros_like(self.M)
        tau = 5 * np.sum(self.M.shape) / 2
        delta = 1.2 * np.prod(self.M.shape) / np.sum(self.mask)
        r_previous = 0
        for k in range(max_iter):
            if k == 0:
                X = np.zeros_like(self.M)
            else:
                sk = r_previous + 1
                U, S, V = self._svd(Y, sk)
                while np.min(S) >= tau:
                    sk = sk + 5
                    U, S, V = self._svd(Y, sk)
                shrink_S = np.maximum(S - tau, 0)
                r_previous = np.count_nonzero(shrink_S)
                diag_shrink_S = np.diag(shrink_S)
                X = np.linalg.multi_dot([U, diag_shrink_S, V])
            Y += delta * self.mask * (self.M - X)
            recon_error = np.linalg.norm(self.mask * (X - self.M)) / np.linalg.norm(self.mask * self.M)
            if (k % iter_print == 0) or (k == 1) or (k > max_iter) or (recon_error < epsilon):
                print("Iteration: %i; Rank: %d; Rel error: %.4f" % (k + 1, np.linalg.matrix_rank(X), recon_error))
            if recon_error < epsilon:
                break
        return X, self.M - X

    def _probabilistic_matrix_factorization(self, k=10, mu=1e-2, epsilon=1e-3, max_iter=1000, iter_print=100):
        """ Solve probabilistic matrix factorization using alternating least squares.

        Parameters:
        -----------
        k : integer
            how many factors to use

        mu : float
            hyper-parameter penalizing norm of factored U, V

        epsilon : float
            convergence condition on the difference between iterative results

        max_iterations: int
            hard limit on maximum number of iterations

        Returns:
        --------
        X: m x n array
            completed matrix

        References:
        ---------- 
        This code is taken from https://github.com/tonyduan/matrix-completion
        """
        logger = logging.getLogger(__name__)
        m, n = self.M.shape
        U = np.random.randn(m, k)
        V = np.random.randn(n, k)
        C_u = [np.diag(row) for row in self.mask]
        C_v = [np.diag(col) for col in self.mask.T]
        prev_X = np.dot(U, V.T)
        for _ in range(max_iter):
            for i in range(m):
                U[i] = np.linalg.solve(np.linalg.multi_dot([V.T, C_u[i], V]) +
                                    mu * np.eye(k),
                                    np.linalg.multi_dot([V.T, C_u[i], self.M[i,:]]))
            for j in range(n):
                V[j] = np.linalg.solve(np.linalg.multi_dot([U.T, C_v[j], U]) +
                                    mu * np.eye(k),
                                    np.linalg.multi_dot([U.T, C_v[j], self.M[:,j]]))
            X = np.dot(U, V.T)
            mean_diff = np.linalg.norm(X - prev_X) / m / n
            if (k % iter_print == 0) or (k == 1) or (k > max_iter) or (mean_diff < epsilon):
                print("Iteration: %i; Rank: %d; Mean diff: %.4f" % (k + 1, np.linalg.matrix_rank(X), mean_diff))
            if mean_diff < epsilon:
                break
            prev_X = X
        return X, self.M - X

    def _biased_probabilistic_matrix_factorization(self, k=10, mu=1e-2, epsilon=1e-3, max_iter=1000, iter_print=100):
        """ Solve biased probabilistic matrix factorization via alternating least
        squares.

        Parameters:
        -----------
        A : m x n array
            matrix to complete

        mask : m x n array
            matrix with entries zero (if missing) or one (if present)

        k : integer
            how many factors to use

        mu : float
            hyper-parameter penalizing norm of factored U, V and biases beta, gamma

        epsilon : float
            convergence condition on the difference between iterative results

        max_iterations: int
            hard limit on maximum number of iterations

        Returns:
        --------
        X: m x n array
            completed matrix

        References:
        ---------- 
        This code is taken from https://github.com/tonyduan/matrix-completion
        """
        logger = logging.getLogger(__name__)
        m, n = self.M.shape
        U = np.random.randn(m, k)
        V = np.random.randn(n, k)
        beta = np.random.randn(m)
        gamma = np.random.randn(n)
        C_u = [np.diag(row) for row in self.mask]
        C_v = [np.diag(col) for col in self.mask.T]
        prev_X = np.dot(U, V.T) + \
                        np.outer(beta, np.ones(n)) + \
                        np.outer(np.ones(m), gamma)
        for _ in range(max_iter):
            # iteration for U
            A_tilde = self.M - np.outer(np.ones(m), gamma)
            V_tilde = np.c_[np.ones(n), V]
            for i in range(m):
                U_tilde = np.linalg.solve(np.linalg.multi_dot([V_tilde.T, C_u[i],
                                                            V_tilde]) +
                                        mu * np.eye(k + 1),
                                        np.linalg.multi_dot([V_tilde.T, C_u[i],
                                                            A_tilde[i,:]]))
                beta[i] = U_tilde[0]
                U[i] = U_tilde[1:]
            # iteration for V
            A_tilde = self.M - np.outer(beta, np.ones(n))
            U_tilde = np.c_[np.ones(m), U]
            for j in range(n):
                V_tilde = np.linalg.solve(np.linalg.multi_dot([U_tilde.T, C_v[j],
                                                            U_tilde]) +
                                                            mu * np.eye(k + 1),
                                        np.linalg.multi_dot([U_tilde.T, C_v[j],
                                                            A_tilde[:,j]]))
                gamma[j] = V_tilde[0]
                V[j] = V_tilde[1:]
            X = np.dot(U, V.T) + np.outer(beta, np.ones(n)) + np.outer(np.ones(m), gamma)
            mean_diff = np.linalg.norm(X - prev_X) / m / n
            if (k % iter_print == 0) or (k == 1) or (k > max_iter) or (mean_diff < epsilon):
                print("Iteration: %i; Rank: %d; Mean diff: %.4f" % (k + 1, np.linalg.matrix_rank(X), mean_diff))
            if mean_diff < epsilon:
                break
            prev_X = X
        return X, self.M - X

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        """ Call specific method to solve the matrix completion.
        """
        if self.method == "NUC":
            return self._nuclear_norm_solve()
        elif self.method == "ADMM":
            return self._alternating_direction(tol=tol, max_iter=max_iter, iter_print=iter_print)
        elif self.method == "SVT":
            return self._singular_value_thresholding(max_iter=max_iter)
        elif self.method == "PMF":
            return self._probabilistic_matrix_factorization(max_iter=max_iter)
        elif self.method == "BPMF":
            return self._biased_probabilistic_matrix_factorization(max_iter=max_iter)
        else:
            print("Error! Input method is illegal.")
            return self.M, self.S

