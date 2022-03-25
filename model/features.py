import pandas as pd
import os
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

from model.regression.base import filter_functions

CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))

class Features:

    def __init__(self, transform='log'):
        """

        Parameters
        ----------
        transform     one of 'log', 'norm', 'quantile', None
        """
        self.transform = transform
        self.metrics = ['Ozone', 'SO2', 'CO', 'NO2', 'PM25', 'PM10', 'Wind', 'Pressure', 'Temperature', 'Humidity']
        self.X = self.get_X()
        self.diff = squareform(pdist(self.X.values))
        self.edges = self.get_edges()

    def get_X(self) -> pd.DataFrame:
        """
        Get a (T, M) pandas Dataframe containing all the features
        """
        X = [self.get_data_PCA(metric,  n_components=10, transform=self.transform) for metric in self.metrics]
        X.append(self.get_data('Fire', transform='log'))
        X = pd.concat(X, axis=1)
        X = pd.DataFrame(X.values[:-1, :], columns=X.columns, index=X.index[1:]) # offset shift by 1 day

        return X

    def get_K(self, ss: float = 20) -> np.ndarray:
        """
        Get a (T, T) numpy array representing the exponential kernel matrix
        """
        return np.exp(-self.diff / ss)

    def get_L(self):
        """
        Get the features Laplacian
        """

        xi, yi = np.array(self.edges).T
        A = np.zeros_like(self.diff)
        A[xi, yi] = 1
        A[yi, xi] = 1

        self.L = np.diag(A.sum(0)) - A

        self.lamL, self.U = np.linalg.eigh(self.L)

        return self.L

    def get_Hs(self, filter_function: str='exponential', beta: float=1):

        self.Hs =  (self.U * filter_functions[filter_function](self.lamL, beta) ** 2) @ self.U.T
        return self.Hs

    def get_edges(self, n_retries: int=1, sigma_D: float=5, seed: int=0) -> list:
        """
        Use perturbed MST algorithm to find sparse graph. Return edges as list of (i, j) tuples
        """

        np.random.seed(seed)

        mst = minimum_spanning_tree(self.diff)
        edges = set(tuple(i) for i in np.argwhere(mst))

        for i in range(n_retries):
            mst = minimum_spanning_tree(self.diff + np.random.normal(loc=0, scale=sigma_D, size=self.diff.shape))
            edges = edges.union(set(tuple(i) for i in np.argwhere(mst)))

        return list(edges)

    @staticmethod
    def get_data_PCA(metric: str, transform: str, n_components=10 ) -> pd.DataFrame:
        """
        Get n_components of the PCA compressed features for a certain metric
        """

        data = Features.get_data(metric, transform)
        pca = PCA(n_components=n_components)

        return pd.DataFrame(pca.fit_transform(data), index=data.index, columns=[f'{metric}_PCA_{n + 1}' for n in range(n_components)])

    @staticmethod
    def get_data(metric: str, transform: str) -> pd.DataFrame:
        """
        Get the raw data for a certain metric
        """

        if transform == 'log':
            folder = 'LogNormalize'
        elif transform == 'quantile':
            folder = 'Quantile'
        elif transform == 'norm':
            folder = 'Normalize'
        else:
            folder = 'NoTransform'

        data = pd.read_csv(f'{CURRENT_FOLDER}/../data/processed/{folder}/{metric}.csv', parse_dates=True, index_col=0)

        return data