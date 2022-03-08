import os
import json
import numpy as np
from .utils import get_transform
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt

CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = '/media/ed/DATA/Datasets/GLSKGR'


class California:

    def __init__(self, lat_min: float=30, lat_max: float=44, lon_min: float=-125.5, lon_max: float=-113, projection: str='lambert conformal conic'):

        # get transform function for converting lat/lon to x/y via projection
        self.projection = projection
        self.transform = get_transform(self.projection)

        # get min and max boundaries for lat/lon and i/j
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = lat_min, lat_max, lon_min, lon_max
        self.imin, self.imax, self.jmin, self.jmax = self.get_mesh_indices()

        # get full mesh grids for lat/lon
        self.lat_mesh, self.lon_mesh = self.get_lat_lon_mesh()

        # get the elevation map
        self.elevation = self.get_elevation(DATA_PATH + '/elev')

        # get boundaries
        self.state_boundaries = self.get_boundaries(include_counties=False)
        self.county_boundaries = self.get_boundaries(include_counties=True)
        self.boundary_mask = self.get_boundary_mask()

    def get_mesh_indices(self):
        """
        Transfrom of lat/lon min/max, to represent the indices of the mesh
        """
        imin, imax = int((50 - self.lat_max) * 120), int((50 - self.lat_min) * 120)
        jmin, jmax = int((180 - abs(self.lon_min)) * 120), int((180 - abs(self.lon_max)) * 120)
        return imin, imax, jmin, jmax

    def get_lat_lon_mesh(self):
        """
        Given the lat/lon rectangle defined in __init__, get two numpy arrays representing a full meshgrid
        of lat/lon values. The spacing is defined by the e10g elevation map, which is 1/120 deg.
        """

        lat_mesh, lon_mesh = np.meshgrid(np.arange(0, 50 - 1e-8, 1 / 120), np.arange(-180, -90 - 1e-8, 1 / 120), indexing='ij')
        lat_mesh = np.flipud(lat_mesh)

        return lat_mesh[self.imin:self.imax, self.jmin:self.jmax], lon_mesh[self.imin:self.imax, self.jmin:self.jmax]

    @staticmethod
    def get_boundaries(include_counties=False):
        """
        Return a list of numpy arrays. Each one is an (N, 2) array of lat/lon paths defining the boundaries of California.
        If include_counties, include individual county outlines.
        """

        if include_counties:
            fname = 'CA_Counties.geo.json'
        else:
            fname = 'US_states.geo.json'

        with open(f'{CURRENT_FOLDER}/../data/geoJson/{fname}') as json_file:
            data = json.load(json_file)

        outlines = []

        for feature in data['features']:

            if not include_counties and feature['properties']['NAME'] != 'California':
                continue

            for f in feature['geometry']['coordinates']:

                if len(f) == 1:
                    f = np.array(f[0])
                else:
                    f = np.array(f)

                outlines.append(f)

        return outlines

    def get_boundary_mask(self):
        """
        Given a lat and lon grid (first two outputs from get_elevation_map), return a boolean array of the
        same size indicating whether each cell is containined within the boundary of California
        """

        mask_points = np.array([self.lon_mesh.reshape(-1), self.lat_mesh.reshape(-1)]).T
        masks = [Path(outline, closed=True).contains_points(mask_points) for outline in self.state_boundaries]

        return np.logical_or.reduce(masks).reshape(*self.lat_mesh.shape)

    def get_elevation(self, location: str):
        """
        Pass the location of the folder contaianing the e10g elevation file downloaded in download.py. Return three
        arrays representing a grid of latitudes, londitues and elevations respectively.
        """
        elevation = np.fromfile(f'{location}/e10g', dtype=np.int16).reshape(6000, 10800)
        return elevation[self.imin:self.imax, self.jmin:self.jmax]

    def plot_outline(self, ax, line=None):
        """
        Simple function to plot the outline of california onto a axis
        """

        for boundary in self.state_boundaries:
            out_x, out_y = self.transform(boundary[:, 0], boundary[:, 1])
            if isinstance(line, dict):
                ax.plot(out_x, out_y, **line)
            else:
                ax.plot(out_x, out_y, lw=1.5, color='white', ls='--')

        self.fix_ax(ax)

    def fix_ax(self, ax):
        """
        Helper function to fix the aspect ratio and limits for map notebooks
        """
        ax.set_xlim(-2.4e6, -1.6e6)
        ax.set_ylim(-5.5e5, 7.2e5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    def plot_gridlines(self, ax, line=None):
        """
        Function to plot gridlines onto a cali map
        """

        if isinstance(line, dict):
            kwargs = line
        else:
            kwargs = dict(color='white', ls='--', lw=0.5)

        for lat in range(30, 45, 2):
            x, y = self.transform(np.linspace(-125.5, -113, 101), lat * np.ones(101))
            ax.plot(x, y, **kwargs)

        for lon in range(-125, -113, 2):
            x, y = self.transform(lon * np.ones(101), np.linspace(30, 45, 101))
            ax.plot(x, y, **kwargs)


    def plot_elevation(self, ax=None, gridlines=True, outline=True, back_alpha=0.1):

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 8))

        x, y = self.transform(self.lon_mesh, self.lat_mesh)
        plt.pcolormesh(x, y, np.ma.masked_array(self.elevation, ~self.boundary_mask), cmap='terrain', vmin=-50, vmax=2.4e3)
        plt.pcolormesh(x, y, np.ma.masked_array(self.elevation, self.boundary_mask), cmap='terrain', alpha=back_alpha, vmin=-50, vmax=2.4e3)

        if outline:
            if isinstance(outline, dict):
                self.plot_outline(ax, outline)
            else:
                self.plot_outline(ax)

        if gridlines:
            self.plot_gridlines(ax, gridlines)

        self.fix_ax(ax)

        return ax


    def plot_regions(self, ax=None):

        region_county_map = {'Shasta': [0, 20, 42, 59, 61],
                             'N Coast': [12, 24, 25, 37, 41, 58],
                             'N Basin': [22, 46, 50, 63],
                             'Central Basin': [1, 5, 15, 19, 28, 33, 34, 35, 40, 44, 47, 49, 51, 53, 56, 57],
                             'S Basin': [13, 39, 43, 52],
                             'Bay': [16, 17, 18, 21, 29, 31, 45, 48, 60, 62],
                             'Sierra Navada': [26, 27, 32],
                             'Central Coast': [4, 23, 36, 64],
                             'S Califonia': [8, 11, 14, 54],
                             'Desert': [30, 38, 55],
                             'Offshore': [2, 3, 6, 7, 9, 10]}

        colours = {'Shasta': '#1f77b4', 'N Coast': '#ff7f0e', 'N Basin': '#2ca02c', 'Central Basin': '#d62728', 'S Basin': '#9467bd', 'Bay': '#8c564b',
                   'Sierra Navada': '#e377c2', 'Central Coast': '#7f7f7f', 'S Califonia': '#bcbd22', 'Desert': '#17becf', 'Offshore': 'black'}
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 8))

        patch_labels = []
        region_labels = []

        for region, counties in region_county_map.items():

            for county in counties:
                coords = self.county_boundaries[county]
                path = Path(np.array(self.transform(coords[:, 0], coords[:, 1])).T)
                patch = PathPatch(path, facecolor=colours[region], label=region, alpha=0.5, lw=0)
                ax.add_patch(patch)

            patch_labels.append(patch)
            region_labels.append(region)

        ax.legend(patch_labels, region_labels, fontsize='small')
        self.fix_ax(ax)

        return ax


if __name__ == '__main__':

    cali = California()

    cali.plot_regions()

    plt.show()
