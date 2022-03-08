from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from .california import California
from .monitors import Monitors

class Distance:
    """
    Class to help measure geodesic distances efficiently and help constuct adjacency matrices
    """

    def __init__(self, sites: pd.DataFrame=None):

        self.cali = California()

        # if no sites are provided, use the coords of all available monitors
        if sites is None:
            sites = Monitors()
            sites = sites.clean(sites.get_raw_data())
            self.coords = sites[['Latitude', 'Longitude']].values

        else:
            self.coords = sites[['Latitude', 'Longitude']].values

        self.N = len(self.coords)
        self.h = self.get_relief_matrix()
        self.d = self.get_geodesic_matrix()

    @staticmethod
    def distance_between(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """
        Distance in km between coordinates (lat1, lon1), (lat2, lon2) which are
        numpy arrays of the same shape (or floats).
        """

        phi1 = np.radians(lat1)
        theta1 = np.radians(lon1)
        phi2 = np.radians(lat2)
        theta2 = np.radians(lon2)

        return 2 * 6371 * np.arcsin((np.sin((phi2 - phi1) / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin((theta2 - theta1) / 2) ** 2) ** 0.5)

    def get_geodesic_matrix(self) -> np.ndarray:
        """
        For a (N, 2) numpy array of lat/lon coords, get the NxN distance matrix
        """

        return Distance.distance_between(np.repeat(self.coords[:, 0][None, :], self.N, axis=0),
                                         np.repeat(self.coords[:, 1][:, None], self.N, axis=1),
                                         np.repeat(self.coords[:, 0][:, None], self.N, axis=1),
                                         np.repeat(self.coords[:, 1][None, :], self.N, axis=0))


    def get_relief_matrix(self) -> np.ndarray:
        """
        For a (N, 2) numpy array of lat/lon coords, find the difference between the highest and
        lowest point on the path between each pair, as an NxN matrix.
        """
        a = []

        for i in range(self.N):
            for j in range(i + 1, self.N):

                # linear lat/lon path for simplicity
                path = np.linspace(self.coords[i], self.coords[j], 100)

                ix = np.round(120 * (self.cali.lat_mesh[0, 0] - path[:, 0].flatten()), 0).astype(int)
                iy = np.round(120 * (path[:, 1].flatten() - self.cali.lon_mesh[0, 0]), 0).astype(int)

                profile = self.cali.elevation[ix, iy]
                a.append(profile.max() - profile.min())

        return squareform(np.array(a))


    def get_distance_matrix(self, h_multiple: float=100):
        """
        Simple helper function to construct a penalty matrix by combining the distance and height matrices
        """
        D = (self.d ** 2 + (h_multiple * self.h / 1000) ** 2) ** 0.5
        return D



if __name__ == '__main__':

    pass