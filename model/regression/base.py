import numpy as np
from ..features import Features
from ..targets import Targets

filter_functions = {'inverse': lambda lamL, beta: (1 + beta * lamL) ** -1,
                    'exponential': lambda lamL, beta: np.exp(-beta * lamL),
                    'ReLu': lambda lamL, beta: np.maximum(1 - beta * lamL, 0),
                    'sigmoid': lambda lamL, beta: 2 * np.exp(-beta * lamL) * (1 + np.exp(-beta * lamL)) ** -1,
                    'cosine': lambda lamL, beta: np.cos(lamL * np.pi / (2 * lamL.max())) ** beta,
                    'cut-off': lambda lamL, beta: (lamL <= 1 / beta).astype(int) if beta != 0 else np.ones_like(lamL)}

class Model:

    def __init__(self, **params):
        self.params = params

    def set_data(self, features: Features, targets: Targets):
        self.features = features
        self.targets = targets

    def optimize(self):
        raise NotImplementedError

    @staticmethod
    def SE(F1: np.ndarray, F2: np.ndarray):
        """
        Simple square difference summed across two matrices
        """
        return ((F1 - F2) ** 2).sum()

    def RMSE_labelled(self):
        """
        RMSE calculated for all labelled data
        """

        Y_ = self.targets.Y0.loc[self.targets.train_sites, self.targets.train_dates].values
        F_ = self.F.loc[self.targets.train_sites, self.targets.train_dates].values

        return (Model.SE(Y_, F_) / Y_.size) ** 0.5

    def RMSE_missing_nodes(self):
        """
        RMSE calculated for missing nodes at labeled time points
        """

        Y_ = self.targets.Y0.loc[self.targets.test_sites, self.targets.train_dates].values
        F_ = self.F.loc[self.targets.test_sites, self.targets.train_dates].values

        return (Model.SE(Y_, F_) / Y_.size) ** 0.5

    def RMSE_missing_times(self):
        """
        RMSE for missing time points at labeled nodes
        """

        Y_ = self.targets.Y0.loc[self.targets.train_sites, self.targets.test_dates].values
        F_ = self.F.loc[self.targets.train_sites, self.targets.test_dates].values

        return (Model.SE(Y_, F_) / Y_.size) ** 0.5

    def RMSE_unlabelled_full(self):
        """
        RMSE for all unlabeled points
        """

        Y1_ = self.targets.Y0.loc[self.targets.train_sites, self.targets.test_dates].values
        F1_ = self.F.loc[self.targets.train_sites, self.targets.test_dates].values

        Y2_ = self.targets.Y0.loc[self.targets.test_sites, self.targets.train_dates].values
        F2_ = self.F.loc[self.targets.test_sites, self.targets.train_dates].values

        Y3_ = self.targets.Y0.loc[self.targets.test_sites, self.targets.test_dates].values
        F3_ = self.F.loc[self.targets.test_sites, self.targets.test_dates].values

        return ((Model.SE(Y1_, F1_) + Model.SE(Y2_, F2_) + Model.SE(Y3_, F3_)) / (Y1_.size + Y2_.size + Y3_.size)) ** 0.5