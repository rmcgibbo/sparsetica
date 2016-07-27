import scipy.stats
import scipy.linalg
import numpy as np
import time
import msmbuilder.version
import cvxpy as cp
from msmbuilder.decomposition._speigh import solve_admm, speigh
print(msmbuilder.version.full_version)
print(cp.installed_solvers())

class CVXPYQCQP(object):
    """Solve the problem from line 31 of Algorithm 1

    x = argmin_x ||x-y||_2^2 + c||diag(w)*x||_1
    s.t. x^T B x <= 1
    """
    def __init__(self, B):
        assert B.ndim == 2
        self.B = B
        self.n = B.shape[0]

        self.b = cp.Parameter(self.n)
        self.w = cp.Parameter(self.n)
        self.x = cp.Variable(self.n)

        term1 = 0.5 * cp.square(cp.norm((self.x - self.b)))
        term2 = cp.norm1(cp.diag(self.w) * self.x)

        objective = cp.Minimize(term1 + term2)
        constraints = [cp.quad_form(self.x, self.B) <= 1]
        self.problem = cp.Problem(objective, constraints)


    def solve(self, b, w):
        assert b.ndim == 1 and w.ndim == 1 and b.shape == w.shape
        assert w.shape[0] == self.n
        self.b.value = b
        self.w.value = w
        self.problem.solve(solver='SCS')
        x = np.asarray(self.x.value).flatten()
        return x


# for N in map(int, np.linspace(10, 1000, 20)):
#     for _ in range(5):
#         sigma = scipy.stats.wishart(N, scale=np.eye(N)).rvs()
#         B_eigenvalues, B_eigenvectors = map(np.ascontiguousarray, scipy.linalg.eigh(sigma))
#
#         problem = CVXPYQCQP(sigma)
#         b, w = np.random.randn(N), np.random.randn(N)
#
#         start = time.time()
#         value1 = problem.solve(b, w)
#         cvxpy_time = time.time() - start
#
#
#         value2 = np.random.randn(N)
#         start = time.time()
#         solve_admm(b, w, B_eigenvalues, B_eigenvectors, value2, verbose=0, tol=1e-10, maxiter=1000)
#         admm_time = time.time() - start
#
#         print(N, cvxpy_time, admm_time)


for N in [50, 100, 200, 500]:
    for _ in range(5):

        A = scipy.stats.wishart(N, scale=np.eye(N)).rvs()
        B = scipy.stats.wishart(N, scale=np.eye(N)).rvs()
        rho = 0.1
        eps = 1e-6

        failed = False
        try:
            start = time.time()
            speigh(A, B, rho=rho, eps=eps, method=1)
            interval = time.time()
            speigh(A, B, rho=rho, eps=eps, method=2)
            end = time.time()
        except (cp.error.SolverError, ValueError) as e:
            print(e)
            failed = True
            print(N, '-', '-')

        if not failed:
            time_admm = interval - start
            time_cvxpy = end - interval
            print(N, time_cvxpy, time_admm)



