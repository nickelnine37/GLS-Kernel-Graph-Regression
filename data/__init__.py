import json
import os

def get_data_dir() -> str:
    """
    Read data directory from config.json and check that it is valid
    """

    current_folder = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(current_folder, '..', 'config.json')) as f:
        data_folder = json.load(f)['data_dir']

    if not os.path.exists(data_folder):
        raise OSError(f'Cannot find directory {data_folder}. Check config.json file.')

    return data_folder