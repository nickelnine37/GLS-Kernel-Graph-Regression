import requests
import os
from zipfile import ZipFile
from tqdm.autonotebook import tqdm
from data import get_data_dir
import time

def download_met_data(data_dir: str):
    """
    Download pollutant data directly from the EPA website. Downloads as zip files, then
    extracts and renames CSVs in the specified folder.

    https://aqs.epa.gov/aqsweb/airdata/download_files.html#Meta

    Parameters
    ----------
    dest_folder: directory to save CSV files to
    """

    t0 = time.time()

    met_folder = os.path.join(data_dir, 'met')

    if not os.path.exists(met_folder):
        os.mkdir(met_folder)

    items = {'Ozone': '44201', 'SO2': '42401', 'CO': '42101', 'NO2': '42602', 'PM25': '88101',
             'PM10': '81102', 'Wind': 'WIND', 'Pressure': 'PRESS', 'Temperature': 'TEMP', 'Humidity': 'RH_DP'}
    years = [2017, 2018, 2019, 2020, 2021]

    base_url = 'https://aqs.epa.gov/aqsweb/airdata/daily_{}_{}.zip'

    metric_pbar = tqdm(items, leave=False)
    for item in metric_pbar:
        metric_pbar.set_description(f'Metric: {item}')
        years_pbar = tqdm(years, leave=False)
        for year in years_pbar:
            years_pbar.set_description(f'Year: {year}')
            url = base_url.format(items[item], year)
            data = requests.get(url)

            zip_fname = os.path.join(data_dir, 'met', f'{item}_{year}.zip')
            csv_fname1 = os.path.join(data_dir, 'met', f'daily_{items[item]}_{year}.csv')
            csv_fname2 = os.path.join(data_dir, 'met', f'{item}_{year}.csv')

            with open(zip_fname, 'wb') as f:
                f.write(data.content)

            with ZipFile(zip_fname, 'r') as zipped:
                zipped.extractall(os.path.join(data_dir, 'met'))

            os.rename(csv_fname1, csv_fname2)
            os.remove(zip_fname)

    print(f'Completed in {time.time()-t0:.2f} seconds')
    print(f'Success: Meteorological data saved to {met_folder}')

def download_GLOBE_data(data_dir: str):
    """
    Download elevation data from the GLOBE project

    https://www.ngdc.noaa.gov/mgg/topo/globe.html

    Parameters
    ----------
    data_dir: directory to save data to
    """

    t0 = time.time()

    url = "https://www.ngdc.noaa.gov/mgg/topo/DATATILES/elev/e10g.zip"
    data = requests.get(url)

    elev_folder = os.path.join(data_dir, 'elev')

    if not os.path.exists(elev_folder):
        os.mkdir(elev_folder)

    with open(os.path.join(data_dir, 'elev', 'e10g.zip'), 'wb') as f:
        f.write(data.content)

    with ZipFile(os.path.join(data_dir, 'elev', 'e10g.zip'), 'r') as zipped:
        zipped.extractall(os.path.join(data_dir, 'elev'))

    os.remove(os.path.join(data_dir, 'elev', 'e10g.zip'))

    print(f'Completed in {time.time()-t0:.2f} seconds')
    print(f'Success: GLOBE data saved to {elev_folder}/e10g')



def download_fire_data(data_dir: str):
    """
    Download data about wildfires in California

    https://www.fire.ca.gov/incidents/
    """

    t0 = time.time()

    url = 'https://www.fire.ca.gov/imapdata/mapdataall.csv'
    data = requests.get(url)

    fire_folder = os.path.join(data_dir, 'fires')

    if not os.path.exists(fire_folder):
        os.mkdir(fire_folder)

    csv_fname = os.path.join(data_dir, 'fires', 'fires.csv')

    with open(csv_fname, 'wb') as f:
        f.write(data.content)

    print(f'Completed in {time.time() - t0:.2f} seconds')
    print(f'Success: fire data saved to {csv_fname}')


if __name__ == '__main__':

    DATA_DIR = get_data_dir()

    download_met_data(DATA_DIR)
    download_GLOBE_data(DATA_DIR)
    download_fire_data(DATA_DIR)
