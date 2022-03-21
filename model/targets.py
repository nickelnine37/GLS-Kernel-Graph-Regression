import os
import numpy as np
from scipy.sparse import csr_matrix

from geo.monitors import Monitors
from .features import Features
from .selection import SelectionMatrix

CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))


class Targets:

    def __init__(self, metric: str, transform: str='log', seed=0, n_: float = 0.8, t_: float = 0.8, seq_t: bool = True):
        """

        Parameters
        ----------
        metric
        transform
        eed
        n_
        t_
        seq_t
        """

        np.random.seed(seed)

        # all labeled data
        self.Y0 = Features.get_data(metric, transform).iloc[1:, :].T
        self.labeled_sites = self.Y0.index.values

        self.labled_times = self.Y0.columns.values
        self.T = len(self.labled_times)

        self.sites = Monitors().data
        self.N = len(self.sites)

        self.select_training_data(n_, t_, seq_t)

        self.SN = SelectionMatrix(self.N, self.train_n)       # SN selects train data from full unlabelled+labeled node set
        self.SN_test = SelectionMatrix(self.N, self.test_n)   # SN_test selcts test data from full unlabelled+labeled node set
        self.ST = SelectionMatrix(self.T, self.train_t)       # ST selects train data from labeled set
        self.ST_test = SelectionMatrix(self.T, self.test_t)   # ST_test selects test data from labelled set

        # test data only
        self.Y = self.Y0.loc[self.train_sites, self.train_dates]


    def select_training_data(self, n_: float = 0.8, t_: float = 0.8, seq_t: bool = True):

        # number of training examples
        self.N_ = int(np.ceil(n_ * len(self.labeled_sites)))
        self.T_ = int(np.ceil(t_ * len(self.labled_times)))

        def train_test_split(n_train: int, data: np.ndarray):
            """
            Helper function to split array into train and test, returning both the chosen ]
            indices and the coressponding items in the data array.
            """
            total = len(data)
            train_i = np.random.choice(total, size=n_train, replace=False)
            test_i = np.array([i for i in range(total) if i not in train_i])
            return train_i, test_i, data[train_i], data[test_i]

        # choose random training nodes
        self.train_n, self.test_n, self.train_sites, self.test_sites = train_test_split(self.N_, self.labeled_sites)

        # overwrite train_n/test_n to now refer to the all_sites data
        self.train_n = np.argwhere(self.train_sites.reshape(-1, 1) == self.sites.index.values)[:, 1]
        self.test_n = np.argwhere(self.test_sites.reshape(-1, 1) == self.sites.index.values)[:, 1]

        if seq_t:
            # choose sequential training times
            self.train_t = np.arange(self.T_)
            self.test_t = np.arange(self.T_, self.T)
            self.train_dates = self.labled_times[:self.T_]
            self.test_dates = self.labled_times[self.T_:]
        else:
            # choose random training times
            self.train_t, self.test_t, self.train_dates, self.test_dates = train_test_split(self.T_, self.labled_times)


    @staticmethod
    def SE(F1: np.ndarray, F2: np.ndarray):
        """
        Simple square difference summed across two matrices
        """
        return ((F1 - F2) ** 2).sum()


