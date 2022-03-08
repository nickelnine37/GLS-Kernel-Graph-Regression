from pyproj import Proj, transform, CRS
import warnings

def get_transform(projection: str='lambert conformal conic') -> callable:
    """
    For a given projection, return a function that takes in two arguments, latitude and londitude,
    and outputs the coressponding projected coordinates

    Parameters
    ----------
    projection   One of: ['mercator', 'lambert azimuthal equal area', 'lambert conformal conic',
                         'albers equal area conic', 'equidistant conic']

    Returns
    -------

    projeciton_transformer  A function taking two arguments, lat and lon, returning projected coords

    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat_proj = CRS(get_CRS(projection))
        lat_long = Proj(init='epsg:4326')

    return lambda lon, lat: transform(lat_long, flat_proj, lon, lat)

def get_CRS(projection):

    crs = {'mercator':                      "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +no_defs",
           'lambert azimuthal equal area':  "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs",
           'lambert conformal conic':       "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs",
           'albers equal area conic':       "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs",
           'equidistant conic':             "+proj=eqdc +lat_0=39 +lon_0=-96 +lat_1=33 +lat_2=45 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"}

    assert projection in crs, f'projection must be one of {list(crs.keys())} but it is {projection}'

    return crs[projection]