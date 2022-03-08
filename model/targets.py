import os
import numpy as np
from scipy.sparse import csr_matrix

from geo.monitors import Monitors
from .features import Features
from .selection import SelectionMatrix

CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))


class Targets:

    def __init__(self, metric: str, seed=0, n_: float = 0.8, t_: float = 0.8, seq_t: bool = True):

        np.random.seed(seed)

        # all labeled data
        self.Y0 = Features.get_data(metric).iloc[1:, :].T
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


#
# class Targets:
#
#     def __init__(self, metric: str, seed=0):
#
#         self.Y0 = Features.get_data(metric).T
#
#         self.times = self.Y0.columns.values
#         self.T = len(self.times)
#
#         self.nodes = self.Y0.index.values
#         self.sites = AllSites().data.loc[self.nodes, :]
#
#         self.N = len(self.nodes)
#
#         self.SN, self.ST = self.get_SN_ST(seed=seed)
#         self.Y0 = self.Y0.values
#         self.Y = self.select(self.Y0, n_label=True, t_label=True)
#
#     def select(self, F: np.ndarray, n_label: bool = True, t_label: bool = True) -> np.ndarray:
#         """
#         Select a spcific section of a (N, T) matrix.
#
#                tL     tU
#             | F_11 | F_12 | nL
#         F =   ---- | ----
#             | F_21 | F_22 | nU
#
#         F_11 -> (N_, T_)
#         F_12 -> (N_, T - T_)
#         F_21 -> (N - N_, T_)
#         F_22 -> (N - N_, T - T_)
#
#         """
#         return F[self.n_ if n_label else self.n__, :][:, self.t_ if t_label else self.t__]
#
#     def get_SN_ST(self,
#                   n_: float = 0.8,
#                   t_: float = 0.8,
#                   seq_t: bool = True,
#                   seed: int = 0,
#                   sparse: bool = True) -> Union[Tuple[csr_matrix, csr_matrix], Tuple[np.ndarray, np.ndarray]]:
#         """
#         Get objects representing the matrices SN and ST as defined in the paper to map between
#         labelled and unlabelled nodes and time points respecitvely.
#
#         Parameters
#         ----------
#         n_          The percentage of the nodes to be in the labeled set
#         t_          The percentage of time points to be in the labeled set
#         seq_t       Whether the labeled time points should all be sequential
#         seed        Random seed
#         sparse      Whether to return sparse matrix objects
#
#         Returns
#         -------
#         SN          Matrix representing labeled node set
#         ST          Matrix representing labeled time point set
#         """
#
#         np.random.seed(seed)
#
#         self.N_ = int(np.ceil(n_ * self.N))
#         self.T_ = int(np.ceil(t_ * self.T))
#
#         self.n_ = np.random.choice(self.N, size=self.N_, replace=False)
#         self.n_.sort()
#         self.n__ = np.array([n for n in range(self.N) if n not in self.n_])
#
#         if seq_t:
#             self.t_ = np.asarray(list(range(self.T_)))
#         else:
#             self.t_ = np.random.choice(self.T, size=self.T_, replace=False)
#             self.t_.sort()
#         self.t__ = np.array([t for t in range(self.T) if t not in self.t_])
#
#         SN = np.zeros((self.N_, self.N))
#         ST = np.zeros((self.T_, self.T))
#
#         SN[range(self.N_), self.n_] = 1
#         ST[range(self.T_), self.t_] = 1
#
#         if sparse:
#             return csr_matrix(SN), csr_matrix(ST)
#         else:
#             return SN, ST
#
#     def SE(self, F1: np.ndarray, F2: np.ndarray):
#         """
#         Simple square difference summed across two matrices
#         """
#         return ((F1 - F2) ** 2).sum()
#
#     def RMSE_missing_nodes(self, F: np.ndarray):
#         """
#         RMSE calculated for missing nodes at labeled time points
#         """
#
#         Y0_ = self.select(self.Y0, False, True)
#         F_ = self.select(F, False, True)
#         tot = Y0_.shape[0] * Y0_.shape[1]
#
#         return (self.SE(Y0_, F_) / tot) ** 0.5
#
#     def RMSE_missing_times(self, F: np.ndarray):
#         """
#         RMSE for missing time points at labeled nodes
#         """
#
#         Y0_ = self.select(self.Y0, True, False)
#         F_ = self.select(F, True, False)
#         tot = Y0_.shape[0] * Y0_.shape[1]
#
#         return (self.SE(Y0_, F_) / tot) ** 0.5
#
#     def RMSE_unlabelled_full(self, F: np.ndarray):
#         """
#         RMSE for all unlabeled points
#         """
#
#         Y0_1 = self.select(self.Y0, False, True)
#         F_1 = self.select(F, False, True)
#
#         Y0_2 = self.select(self.Y0, True, False)
#         F_2 = self.select(F, True, False)
#
#         Y0_3 = self.select(self.Y0, False, False)
#         F_3 = self.select(F, False, False)
#
#         tot = Y0_1.shape[0] * Y0_1.shape[1] + Y0_2.shape[0] * Y0_2.shape[1] + Y0_3.shape[0] * Y0_3.shape[1]
#         SE = self.SE(Y0_1, F_1) + self.SE(Y0_2, F_2) + self.SE(Y0_3, F_3)
#
#         return (SE / tot) ** 0.5
#
#     def RMSE_full(self, F: np.ndarray):
#         """
#         RMSE for all points labeled and unlabeled
#         """
#         return (self.SE(self.Y0, F) / (self.T * self.N)) ** 0.5
#
#
