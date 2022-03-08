import pandas as pd
import os

from geo.california import California
from geo.utils import get_transform

CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))

pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 50)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

import warnings



class Monitors:

    def __init__(self, lat_min: float = 30, lat_max: float = 44, lon_min: float = -125.5, lon_max: float = -113):

        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

        self.data = self.process()

    def get_raw_data(self):
        """
        Get the raw sites data as saved
        """

        uri = f"{CURRENT_FOLDER}/../data/sites/aqs_sites.csv"
        sites = pd.read_csv(uri)

        return sites

    def clean(self, data: pd.DataFrame):
        """
        Remove sites not in designated lat/lon rectangle, add ID column, and remove unecessary columns
        """

        keep_cols = ['State Code', 'County Code', 'Site Number', 'Latitude', 'Longitude', 'Elevation', 'Land Use', 'Location Setting']

        data = data[keep_cols]

        # can't get SettingWithCopyWarning to go away!!
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data['State Code'] = pd.to_numeric(data['State Code'], errors='coerce')

        data = data.dropna()
        data['State Code'] = data.loc[:, 'State Code'].astype(int)
        data['id'] = data['State Code'].apply(str) + '_' + data['County Code'].apply(str) + '_' + data['Site Number'].apply(str)

        data = data[(data['Latitude'] > self.lat_min) &
                    (data['Latitude'] < self.lat_max) &
                    (data['Longitude'] > self.lon_min) &
                    (data['Longitude'] < self.lon_max)]

        return data.drop(['State Code', 'County Code', 'Site Number'], axis=1).set_index('id')

    def transform(self, data: pd.DataFrame):
        """
        Transform site data into normalised numerical features
        """

        one_hot = pd.get_dummies(data['Location Setting'], prefix='Location')
        data = pd.concat([data.drop(['Land Use', 'Location Setting'], axis=1), one_hot], axis=1)

        transform = get_transform()
        x, y = transform(data['Longitude'].values, data['Latitude'].values)
        data['x'] = x
        data['y'] = y
        data = data.rename({'Elevation': 'h'}, axis=1)

        data[['x_normed', 'y_normed']] = data[['x', 'y']] - data[['x', 'y']].mean()
        data[['x_normed', 'y_normed']] /= data[['x', 'y']].values.std()

        data['h_normed'] = data['h'] - data['h'].mean()
        data['h_normed'] /= data['h'].std()

        return data[['Latitude','Longitude', 'h', 'x', 'y', 'x_normed', 'y_normed', 'h_normed', 'Location_RURAL', 'Location_SUBURBAN', 'Location_UNKNOWN', 'Location_URBAN AND CENTER CITY']]

    def process(self):
        """
        Process all site data
        """
        data = self.get_raw_data()
        data = self.clean(data)
        data = self.transform(data)

        return data


if __name__ == '__main__':

    sites = Monitors()

    print(sites.data)


