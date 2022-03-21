import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from ..features import Features
from ..targets import Targets
from ..laplacian import Laplacian
from ..utils import make_B1, make_B2, make_ST
from .base import Model, filter_functions

class CovarianceEstimator:
    """
    Helper class to group routines related to estimating the covariance
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def get_rho(self, SN_MLE):
        """
        Rao-Blackwell Ledoit-Wolf(RBLW) estimator
        """
        st = (SN_MLE ** 2).sum()
        ts = np.trace(SN_MLE) ** 2
        u = ts + st * (self.T - 2) / self.T
        d = (self.T + 2) * (st - ts / self.N)
        return min(u / d, 1)

    def estimate_SN(self, theta: float):
        SN_MLE = (self.EE - theta * self.EB1E + theta ** 2 * self.EB2E) / (self.T * (1 - theta ** 2))
        rho = self.get_rho(SN_MLE)
        return (1 - rho) * SN_MLE + rho * np.trace(SN_MLE) * np.eye(self.N) / self.T

    @staticmethod
    def real_root(roots):
        return np.real(roots[np.argmin(np.abs(roots.imag))])

    def estimate_theta(self, SN):
        SNiE = np.linalg.solve(SN, self.E)
        a = (self.E * SNiE).sum()
        b = (self.E * (SNiE @ self.B1)).sum()
        c = (self.E * (SNiE @ self.B2)).sum()
        roots = np.roots([self.N * (1 - self.T), b / 2, self.N * (self.T - 1) - a - c - self.N * self.T * self.alpha, b / 2])
        return self.real_root(roots)

    def estimate(self, E, tol=1e-3, theta0: float = 0):
        self.E = E

        self.N, self.T = E.shape
        self.B1 = make_B1(self.T, sparse=True)
        self.B2 = make_B2(self.T, sparse=True)

        self.EE = E @ E.T
        self.EB1E = E @ self.B1 @ E.T
        self.EB2E = E @ self.B2 @ E.T

        dtheta = 1
        theta = theta0
        SN = None

        while abs(dtheta) > tol:
            SN = self.estimate_SN(theta)
            dtheta = self.estimate_theta(SN) - theta
            theta += dtheta

        return SN, theta


class GLSKGR(Model):

    def __init__(self, gamma: float = 1, K_std: float = 20, filter_func: str = 'sigmoid', beta: float = 1, alpha=10):
        MAX_ITERS = 60
        FTOL = 1e-4
        super().__init__(gamma=gamma, K_std=K_std, filter_func=filter_func, beta=beta, alpha=alpha, max_iters=MAX_ITERS, ftol=FTOL)
        self.covariace_estimator = CovarianceEstimator(alpha)

    def set_laplacian(self, laplacian: Laplacian):
        self.lap = laplacian
        return self

    def set_data(self, features: Features, targets: Targets):
        super().set_data(features, targets)
        return self.set_K().decompose_L().set_Hs()

    def set_K(self):
        self.K = self.features.get_K(self.params['K_std'])
        return self

    def decompose_L(self):
        self.lamL, self.U = np.linalg.eigh(self.lap.L)
        return self

    def set_Hs(self):
        self.Hs = (self.U * filter_functions[self.params['filter_func']](self.lamL, self.params['beta']) ** 2) @ self.U.T
        return self

    def get_F(self, SN: np.ndarray, ST: np.ndarray):

        lamT, Psi = np.linalg.eigh(ST)
        lamN, Phi = np.linalg.eigh(SN)

        PsiLam = Psi * (lamT ** -0.5)
        K_ = self.targets.ST @ self.K @ self.targets.ST.T
        lamK_, V_ = np.linalg.eigh(PsiLam.T @ K_ @ PsiLam)

        PhiLam = Phi * (lamN ** -0.5)
        H_ = self.targets.SN @ self.Hs @ self.targets.SN.T
        lamH_, U_ = np.linalg.eigh(PhiLam.T @ H_ @ PhiLam)

        J = 1 / (self.params['gamma'] + lamK_[None, :] * lamH_[:, None])

        B = self.Hs @ self.targets.SN.T @ PhiLam @ U_
        C = self.K @ self.targets.ST.T @ PsiLam @ V_
        Y_ = U_.T @ PhiLam.T @ self.targets.Y.values @ PsiLam @ V_

        return B @ (J * Y_) @ C.T

    def solve_GLS(self):

        F = self.get_F(np.eye(self.targets.N_), np.eye(self.targets.T_))
        SN, theta = self.covariace_estimator.estimate(self.targets.Y.values - self.targets.SN @ F @ self.targets.ST.T)

        dtheta = 1
        it = 1
        theta = 0

        while abs(dtheta) > self.params['ftol']:

            F = self.get_F(SN, make_ST(self.targets.T_, theta))
            SN, theta_ = self.covariace_estimator.estimate(self.targets.Y.values - self.targets.SN @ F @ self.targets.ST.T, theta0=theta)

            dtheta = theta - theta_
            theta = theta_
            it += 1
            if it >= self.params['max_iters']:
                print('max iterations reached')
                break

        self.F = pd.DataFrame(self.get_F(SN, make_ST(self.targets.T_, theta)), index=self.targets.sites.index, columns=self.targets.Y0.columns)
        self.theta = theta
        self.SN = SN

        return self


    def update_gamma(self, gamma):
        self.params['gamma'] = gamma
        return self.solve_GLS()

    def update_alpha(self, alpha):
        self.params['alpha'] = alpha
        self.covariace_estimator = CovarianceEstimator(alpha)
        return self.solve_GLS()

    def update_beta(self, beta):
        self.params['beta'] = beta
        return self.set_Hs().solve_GLS()

    def update_Kstd(self, K_std):
        self.params['K_std'] = K_std
        return self.set_K().solve_GLS()

    def optimize_alpha(self):
        alphas = np.logspace(-2, 0.8, 10)
        error = [self.update_alpha(alpha).RMSE_unlabelled_full() for alpha in tqdm(alphas, leave=False)]
        return alphas[np.argmin(error)]

    def optimize_gamma(self):
        gammas = np.logspace(-6, 1, 50)
        error = [self.update_gamma(gamma).RMSE_unlabelled_full() for gamma in tqdm(gammas, leave=False)]
        return gammas[np.argmin(error)]

    def optimize_beta(self):
        betas = np.logspace(0, 2, 50)
        error = [self.update_beta(beta).RMSE_unlabelled_full() for beta in tqdm(betas, leave=False)]
        return betas[np.argmin(error)]

    def optimize_Kstd(self):
        stds = np.logspace(0, 3, 50)
        error = [self.update_Kstd(std).RMSE_unlabelled_full() for std in tqdm(stds, leave=False)]
        return stds[np.argmin(error)]

    def optimize(self):

        for i in tqdm(range(9)):
            if i % 3 == 0:
                self.update_gamma(self.optimize_gamma())
            if i % 3 == 1:
                self.update_beta(self.optimize_beta())
            if i % 3 == 2:
                self.update_Kstd(self.optimize_Kstd())

        return self.params, self.RMSE_unlabelled_full()