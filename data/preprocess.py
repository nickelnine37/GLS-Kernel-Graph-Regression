import pandas as pd
import os
from matplotlib.path import Path
from sklearn.preprocessing import QuantileTransformer
from tqdm.autonotebook import tqdm

from geo.monitors import Monitors
import numpy as np
from geo.california import California

from data import get_data_dir

pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 50)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))


class FirePipeline:

    def __init__(self, data_dir: str, start_date: str = '2017-01-01', end_date: str = '2021-04-20', transform: str=None):
        """

        Parameters
        ----------
        data_dir        location of raw data
        start_date
        end_date
        transform       one of 'log', 'quantile', 'norm', None
        """

        self.data_dir = data_dir
        self.start_date = pd.to_datetime(start_date + ' 12:00:00')
        self.end_date = pd.to_datetime(end_date + ' 12:00:00')
        self.date_range = pd.date_range(start=start_date, end=end_date)
        self.transform = transform

    def read_data_raw(self):
        """
        Read in the data as downloaded, drop and rename some columns, and sort out timestamp columns
        """

        cols = ['incident_name', 'incident_date_created', 'incident_date_extinguished', 'incident_acres_burned', 'incident_latitude', 'incident_longitude']
        data = pd.read_csv(os.path.join(self.data_dir, 'fires', 'fires.csv'), parse_dates=['incident_date_created', 'incident_date_extinguished'], usecols=cols).dropna()
        data = data.rename({'incident_name': 'Name',
                            'incident_date_created': 'Started',
                            'incident_acres_burned': 'Acres',
                            'incident_longitude': 'Longitude',
                            'incident_latitude': 'Latitude',
                            'incident_date_extinguished': 'Extinguished'}, axis=1)

        data['Started'] = data['Started'].dt.tz_localize(None).round("D")
        data['Extinguished'] = data['Extinguished'].dt.tz_localize(None).round("D")

        return data

    def add_cols(self, data: pd.DataFrame):
        """
        Add in a few new columns to calculate the acres burned per day. Also filter out
        events that fall outside the considered window.
        """

        data = data[(data['Started'] > self.start_date) & (data['Extinguished'] < self.end_date)]
        data['Length'] = (data['Extinguished'] - data['Started']).dt.days
        data = data[data['Length'] > 0].reset_index(drop=True)
        data['Acres per day'] = data['Acres'] / data['Length']

        return data

    def add_region(self, data: pd.DataFrame):
        """
        Use the California geoJson county data to find out which county each fire happened in. Then use
        the predefined county to region map to turn this into a region variable.
        """

        boundaries = California.get_boundaries(include_counties=True)

        A = np.argwhere(np.array([Path(boundary).contains_points(data[['Longitude', 'Latitude']]) for boundary in boundaries]))
        data['Region'] = np.nan

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

        county_region_map = {county: region for region, counties in region_county_map.items() for county in counties}

        for i in range(65):
            data.loc[A[A[:, 0] == i][:, 1], 'Region'] = county_region_map[i]

        return data.dropna()

    def clean(self, data: pd.DataFrame):
        """
        Transform the event list into a time series of acres burning per region. Drop a Gaussian shaped burn
        curve here.
        """

        regions = ['Shasta', 'N Coast', 'N Basin', 'Central Basin', 'S Basin', 'Bay', 'Sierra Navada', 'Central Coast', 'S Califonia', 'Desert', 'Offshore']
        clean_data = {region: np.zeros(len(self.date_range), dtype=float) for region in regions}

        for region, df in data.groupby('Region'):
            for i, row in df.iterrows():
                n1 = np.argwhere(self.date_range == row['Started'])[0, 0]
                n2 = np.argwhere(self.date_range == row['Extinguished'])[0, 0]
                n = n2 - n1 + 1
                x = np.linspace(-1, 1, n)
                y = np.exp(- x ** 2 / (2 * 0.3 ** 2)) / (0.3 * (2 * np.pi) ** 0.5)
                y /= y.sum()
                clean_data[region][n1:n2 + 1] += y * row['Acres']

        return pd.DataFrame(clean_data, index=self.date_range)

    def process(self):
        """
        Tie together the whole processing pipeline
        """

        data = self.read_data_raw()
        data = self.add_cols(data)
        data = self.add_region(data)
        data = self.clean(data)

        if self.transform == 'log':
            self.data = self.transform_log(data)
        elif self.transform == 'quantile':
            self.data = self.transform_quantile(data)
        elif self.transform == 'norm':
            self.data = self.transform_normalize(data)
        elif not self.transform:
            self.data = data
        else:
            raise ValueError

        return self

    def transform_log(self, data: pd.DataFrame):

        data = np.log(1 + data)
        data = self.transform_normalize(data)

        return data

    def transform_normalize(self, data: pd.DataFrame):
        return (data - data.values.mean()) / data.values.std()

    def transform_quantile(self, data: pd.DataFrame):
        """
        Do a quntile transform so that each dataframe is normal
        """
        transformer = QuantileTransformer(output_distribution='normal')
        data.loc[:, :] = transformer.fit_transform(data)
        return data

    def to_csv(self):
        """
        Save the data to a csv
        """

        if self.transform == 'log':
            folder = 'LogNormalize'
        elif self.transform == 'quantile':
            folder = 'Quantile'
        elif self.transform == 'norm':
            folder = 'Normalize'
        else:
            folder = 'NoTransform'

        dir = os.path.join(CURRENT_FOLDER, 'processed', folder)

        if not os.path.exists(dir):
            os.makedirs(dir)

        self.data.to_csv(os.path.join(dir, 'Fire.csv'))


class MetPipeline:

    def __init__(self, data_dir: str, metric: str, start_date: str = '2017-01-01', end_date: str = '2021-04-20', null_tol: float = 0.1, transform: str='log'):
        """

        Parameters
        ----------
        data_dir        location of raw data
        metric          One of 'Ozone', 'SO2', 'CO', 'NO2', 'PM25', 'PM10', 'Wind','Pressure', 'Temperature', 'Humidity'
        start_date
        end_date
        null_tol        The fraction of nulls tolerated per monitor before discarding
        transform       Which transform to perform ('log', 'quantile', 'norm', or None)
        """

        self.data_dir = data_dir
        self.metric = metric
        self.date_range = pd.date_range(start=start_date, end=end_date)
        self.null_tol = null_tol
        self.transform = transform

    def read_data_raw(self):
        """
        Get the fully raw data. Only operation is to drop certain columns and concatenate
        all the years under consideration
        """

        cols = ['State Code', 'County Code', 'Site Num', 'POC', 'Date Local', 'Arithmetic Mean',
                'Parameter Name', 'Sample Duration', 'Latitude', 'Longitude']

        years = [2017, 2018, 2019, 2020, 2021]

        def read_csv(year):
            return pd.read_csv(os.path.join(self.data_dir, 'met', f'{self.metric}_{year}.csv'), usecols=cols, parse_dates=['Date Local'])

        return pd.concat([read_csv(year) for year in years])

    def strip_parameter_and_duration(self, raw_data: pd.DataFrame):
        """
        Some metrics have multiple parameters and/or sample durations. This function defines which
        ones we want to keep, and strips away the rest. It also drops these columns as they are no
        longer necessary
        """

        params_durations = {'Ozone': ['Ozone', '8-HR RUN AVG BEGIN HOUR'],
                            'SO2': ['Sulfur dioxide', '3-HR BLK AVG'],
                            'CO': ['Carbon monoxide', '8-HR RUN AVG END HOUR'],
                            'NO2': ['Nitrogen dioxide (NO2)', '1 HOUR'],
                            'PM25': ['PM2.5 - Local Conditions', '24-HR BLK AVG'],
                            'PM10': ['PM10 Total 0-10um STP', '24-HR BLK AVG'],
                            'Wind': ['Wind Speed - Resultant', '1 HOUR'],
                            'Pressure': ['Barometric pressure', '1 HOUR'],
                            'Temperature': ['Outdoor Temperature', '1 HOUR'],
                            'Humidity': ['Relative Humidity ', '1 HOUR']}

        parameter_name, sample_duration = params_durations[self.metric]

        return raw_data[(raw_data['Parameter Name'] == parameter_name) &
                        (raw_data['Sample Duration'] == sample_duration)].drop(['Parameter Name', 'Sample Duration'], axis=1)

    def add_id_column(self, raw_data: pd.DataFrame):
        """
        Create an ID column by combining state code, county code and site number. Then drop these columns
        """

        # create an ID column by concatenating the state code, county code, and site num
        raw_data['id'] = raw_data['State Code'].apply(str) + '_' + raw_data['County Code'].apply(str) + '_' + raw_data['Site Num'].apply(str)

        return raw_data.drop(['State Code', 'County Code', 'Site Num'], axis=1)

    def clean_group(self, id: str, raw_group: pd.DataFrame):
        """
        For all the data on one specific site, which is a section of the raw data,
        transform it to have a reindexed date index, with a single POC, with only
        the measured value as the column
        """

        # if there are multiple POCs at this site, get the one with the most data
        df = max(raw_group.groupby('POC'), key=lambda group: len(group[1]))[1]
        df = df.set_index('Date Local')
        df = df.rename({'Arithmetic Mean': id}, axis=1)
        df = df[~df.index.duplicated()]
        return df.reindex(self.date_range)[id]

    def concat_sites(self, raw_data: pd.DataFrame):
        """
        Clean each group, which is a specific site, and concatenate horizontally. This makes a full cleaned df, but with plenty of nulls
        """

        sites = Monitors().data
        return pd.concat([self.clean_group(id, df) for (id, df) in raw_data.groupby('id') if id in sites.index], axis=1)

    def remove_nulls(self, cleaned_data: pd.DataFrame):
        """
        Take a cleaned dataframe and remove the nulls according to the following rules:
            1. Drop any columns that have more than null_tol% nulls
            2. Interloplate linearly for all nulls in the middle somewherr
            3. Fill remaining nulls on the edges with the row mean
        """

        cleaned_data = cleaned_data.loc[:, cleaned_data.isnull().sum(0) / len(self.date_range) < self.null_tol]
        cleaned_data = cleaned_data.interpolate(method='linear')
        cleaned_data = cleaned_data.apply(lambda row: row.fillna(row.mean()), axis=1)

        return cleaned_data

    def transform_quantile(self, data: pd.DataFrame):
        """
        Do a quntile transform so that each dataframe is normal
        """
        transformer = QuantileTransformer(output_distribution='normal')
        data.loc[:, :] = transformer.fit_transform(data)
        return data


    def transform_log(self, data: pd.DataFrame):
        """
        Do a quntile transform so that each dataframe is normal
        """

        transforms = {'CO': {'zero-max': True, 'log': True, 'log const': 0.1},
                      'SO2': {'zero-max': True, 'log': True, 'log const': 0.1},
                      'NO2': {'zero-max': True, 'log': True, 'log const': 1},
                      'Ozone': {'zero-max': True, 'log': False, 'log const': None},
                      'PM25': {'zero-max': True, 'log': True, 'log const': 1},
                      'PM10': {'zero-max': True, 'log': True, 'log const': 1},
                      'Pressure': {'zero-max': False, 'log': False, 'log const': None},
                      'Humidity': {'zero-max': False, 'log': False, 'log const': None},
                      'Temperature': {'zero-max': False, 'log': False, 'log const': None},
                      'Wind': {'zero-max': False, 'log': True, 'log const': 0.1}}

        if transforms[self.metric]['zero-max']:
            data = np.maximum(data, 0)
        if transforms[self.metric]['log']:
            data = np.log(data + transforms[self.metric]['log const'])

        data = self.transform_normalize(data)

        return data

    def transform_normalize(self, data: pd.DataFrame):
        return (data - data.values.mean()) / data.values.std()

    def process(self):
        """
        Complete the whole preprocessing pipeline by stringing together all the methods
        """

        data = self.read_data_raw()
        data = self.strip_parameter_and_duration(data)
        data = self.add_id_column(data)
        data = self.concat_sites(data)
        data = self.remove_nulls(data)

        if self.transform == 'log':
            self.data = self.transform_log(data)
        elif self.transform == 'quantile':
            self.data = self.transform_quantile(data)
        elif self.transform == 'norm':
            self.data = self.transform_normalize(data)
        elif not self.transform:
            self.data = data
        else:
            raise ValueError

        return self

    def to_csv(self):
        """
        Save the data to a csv
        """

        if self.transform == 'log':
            folder = 'LogNormalize'
        elif self.transform == 'quantile':
            folder = 'Quantile'
        elif self.transform == 'norm':
            folder = 'Normalize'
        else:
            folder = 'NoTransform'

        dir = os.path.join(CURRENT_FOLDER, 'processed', folder)

        if not os.path.exists(dir):
            os.makedirs(dir)

        self.data.to_csv(os.path.join(dir, f'{self.metric}.csv'))


if __name__ == '__main__':

    DATA_DIR = get_data_dir()

    # Preprocess the fire data for each transform type

    transform_pbar = tqdm(['log', 'quantile', 'norm', None], leave=False)

    for transform in transform_pbar:
        transform_pbar.set_description(f'Transform: {transform}')
        FirePipeline(DATA_DIR, transform=transform).process().to_csv()

    # Preprocess each metric for each transform type

    transform_pbar = tqdm(['log', 'quantile', 'norm', None], leave=False)
    for transform in transform_pbar:

        transform_pbar.set_description(f'Transform: {transform}')
        metric_pbar = tqdm(['Ozone', 'SO2', 'CO', 'NO2', 'PM25', 'PM10', 'Wind', 'Pressure', 'Temperature', 'Humidity'], leave=False)

        for metric in metric_pbar:
            metric_pbar.set_description(f'Metric: {metric}')
            MetPipeline(DATA_DIR, metric, transform=transform).process().to_csv()

    