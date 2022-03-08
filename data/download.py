import requests
import os
from zipfile import ZipFile
from tqdm import tqdm


def download_met_data(dest_folder: str = '.'):
    """
    Download pollutant data directly from the EPA website. Downloads as zip files, then
    extracts and renames CSVs in the specified folder.

    https://aqs.epa.gov/aqsweb/airdata/download_files.html#Meta

    Parameters
    ----------
    dest_folder: directory to save CSV files to
    """

    items = {'Ozone': '44201', 'SO2': '42401', 'CO': '42101', 'NO2': '42602', 'PM25': '88101',
             'PM10': '81102', 'Wind': 'WIND', 'Pressure': 'PRESS', 'Temperature': 'TEMP', 'Humidity': 'RH_DP'}
    years = [2021, 2020, 2019, 2018, 2017]

    base_url = 'https://aqs.epa.gov/aqsweb/airdata/daily_{}_{}.zip'

    for item in tqdm(items):
        for year in years:
            url = base_url.format(items[item], year)
            data = requests.get(url)

            with open(f'{dest_folder}/{item}_{year}.zip', 'wb') as f:
                f.write(data.content)

            with ZipFile(f'{dest_folder}/{item}_{year}.zip', 'r') as zipped:
                zipped.extractall(dest_folder)

            os.rename(f'{dest_folder}/daily_{items[item]}_{year}.csv', f'{dest_folder}/{item}_{year}.csv')
            os.remove(f'{dest_folder}/{item}_{year}.zip')


def download_GLOBE_data(dest_folder: str = '.'):
    """
    Download elevation data from the GLOBE project

    https://www.ngdc.noaa.gov/mgg/topo/globe.html

    Parameters
    ----------
    dest_folder: directory to save data to
    """

    url = "https://www.ngdc.noaa.gov/mgg/topo/DATATILES/elev/e10g.zip"
    data = requests.get(url)

    with open(f'{dest_folder}/e10g.zip', 'wb') as f:
        f.write(data.content)

    with ZipFile(f'{dest_folder}/e10g.zip', 'r') as zipped:
        zipped.extractall(dest_folder)

    os.remove(f'{dest_folder}/e10g.zip')


def download_fire_data(dest_folder: str = '.'):
    """
    Download data about wildfires in California

    https://www.fire.ca.gov/incidents/
    """

    url = 'https://www.fire.ca.gov/imapdata/mapdataall.csv'
    data = requests.get(url)

    with open(f'{dest_folder}/fires.csv', 'wb') as f:
        f.write(data.content)


if __name__ == '__main__':

    dest_folder = '/media/ed/DATA/Datasets/GLSKGR'

    download_met_data(dest_folder + '/met')
    download_GLOBE_data(dest_folder + '/elev')
    download_fire_data(dest_folder + '/fires')
