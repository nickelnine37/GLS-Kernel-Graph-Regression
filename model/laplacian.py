import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from geo.distance import Distance
import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 50)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

class Laplacian:

    def __init__(self, sites: pd.DataFrame=None, h_multiple: float=100):

        self.distance = Distance(sites)
        self.D = self.distance.get_distance_matrix(h_multiple)
        self.edges = self.get_edges()
        self.L = self.get_L()

    def get_edges(self, n_retries: int=1, sigma_D: float=5, seed: int=0) -> list:
        """
        Use perturbed MST algorithm to find sparse graph. Return edges as list of (i, j) tuples
        """

        np.random.seed(seed)

        mst = minimum_spanning_tree(self.D)
        edges = set(tuple(i) for i in np.argwhere(mst))

        for i in range(n_retries):
            mst = minimum_spanning_tree(self.D + np.random.normal(loc=0, scale=sigma_D, size=self.D.shape))
            edges = edges.union(set(tuple(i) for i in np.argwhere(mst)))

        return list(edges)

    def get_L(self) -> np.ndarray:
        """
        Use the edges calucated to constuct the adjacency matrix
        """

        xi, yi = np.array(self.edges).T
        A = np.zeros_like(self.D)
        A[xi, yi] = 1
        A[yi, xi] = 1

        return np.diag(A.sum(0)) - A

    def set_h_multiple(self, h_multiple: float):
        """
        Reset the h_multiple for use in edge calculation
        """

        self.D = self.distance.get_distance_matrix(h_multiple)
        self.edges = self.get_edges()
        self.L = self.get_L()





if __name__ == '__main__':

    from model.targets import Targets

    targets = Targets('Ozone')

    L = Laplacian(targets.nodes)



