import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from ..features import Features
from ..targets import Targets
from ..laplacian import Laplacian

from .base import Model
from . import filter_functions


class KGR(Model):

    def __init__(self, gamma: float = 1, K_std: float = 20, filter_func: str = 'sigmoid', beta: float = 1, **kwargs):
        super().__init__(gamma=gamma, K_std=K_std, filter_func=filter_func, beta=beta, **kwargs)

    def set_laplacian(self, laplacian: Laplacian):
        self.lap = laplacian
        return self

    def set_data(self, features: Features, targets: Targets):
        super().set_data(features, targets)
        return self.set_K().decompose_L().set_Hs().decompose_K().decompose_H().set_J().set_F()

    def set_K(self):
        self.K = self.features.get_K(self.params['K_std'])
        return self

    def decompose_L(self):
        self.lamL, self.U = np.linalg.eigh(self.lap.L)
        return self

    def set_Hs(self):
        self.Hs = (self.U * filter_functions[self.params['filter_func']](self.lamL, self.params['beta']) ** 2) @ self.U.T
        return self

    def decompose_K(self):
        self.lamK_, self.V_ = np.linalg.eigh(self.targets.ST @ self.K @ self.targets.ST.T)
        return self

    def decompose_H(self):
        self.lamH_, self.U_ = np.linalg.eigh(self.targets.SN @ self.Hs @ self.targets.SN.T)
        return self

    def set_J(self):
        self.J = 1 / (self.params['gamma'] + np.outer(self.lamH_, self.lamK_))
        return self

    def set_F(self):
        F = self.Hs @ self.targets.SN.T @ self.U_ @ (self.J * (self.U_.T @ self.targets.Y.values @ self.V_)) @ self.V_.T @ (self.targets.ST @ self.K)
        self.F = pd.DataFrame(F, index=self.targets.sites.index, columns=self.targets.Y0.columns)
        return self

    def set_Fvar(self):
        lamK_, V_ = np.linalg.eig(self.targets.ST.T.to_sparse_array() @ (self.targets.ST @ self.K))
        lamH_, U_ = np.linalg.eig(self.targets.SN.T.to_sparse_array() @ (self.targets.SN @ self.Hs))
        J_ = 1 / (np.outer(lamH_, lamK_) + self.params['gamma'])
        Fvar = (np.linalg.inv(U_).T * (self.Hs @ U_)) @ J_ @ ((V_.T @ self.K) * np.linalg.inv(V_))
        self.Fvar = pd.DataFrame(Fvar, index=self.targets.sites.index, columns=self.targets.Y0.columns)
        return self

    def update_gamma(self, gamma):
        self.params['gamma'] = gamma
        return self.set_J().set_F()

    def update_beta(self, beta):
        self.params['beta'] = beta
        return self.set_Hs().decompose_H().set_J().set_F()

    def update_Kstd(self, K_std):
        self.params['K_std'] = K_std
        return self.set_K().decompose_K().set_J().set_F()

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

        for i in tqdm(range(9), leave=False):
            if i % 3 == 0:
                self.update_gamma(self.optimize_gamma())
            if i % 3 == 1:
                self.update_beta(self.optimize_beta())
            if i % 3 == 2:
                self.update_Kstd(self.optimize_Kstd())

        return self.params, self.RMSE_unlabelled_full()


class GraphFeaturesKGR(KGR):

    def __init__(self, gamma: float = 1, beta: float = 1, beta_f: float=1, filter_func: str = 'exponential', **kwargs):
        super().__init__(gamma=gamma, beta=beta, beta_f=beta_f, filter_func=filter_func, **kwargs)


    def set_K(self):
        self.K = self.features.get_Hs(self.params['filter_func'], self.params['beta_f'])
        return self

    def set_data(self, features: Features, targets: Targets):
        features.get_L()
        return super().set_data(features, targets)

    def update_beta_f(self, beta_f):
        self.params['beta_f'] = beta_f
        return self.set_K().decompose_K().set_J().set_F()

    def optimize_beta_f(self):
        betas = np.logspace(0, 2, 50)
        error = [self.update_beta_f(beta).RMSE_unlabelled_full() for beta in tqdm(betas, leave=False)]
        return betas[np.argmin(error)]

    def optimize(self):

        for i in tqdm(range(9), leave=False):
            if i % 3 == 0:
                self.update_gamma(self.optimize_gamma())
            if i % 3 == 1:
                self.update_beta(self.optimize_beta())
            if i % 3 == 2:
                self.update_beta_f(self.optimize_beta_f())

        return self.params, self.RMSE_unlabelled_full()