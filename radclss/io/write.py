import json
import urllib.request
import warnings

def write_radclss_output(ds, output_filename, process, version=None):
    """Write the RadCLSS output dataset to a NetCDF file.

    Parameters
    ----------
    ds : xarray.Dataset
        The RadCLSS output dataset.
    output_filename : str
        The path to the output NetCDF file.
    process : str
        The process name (i.e. radclss)
    version : str
        The version of the process used. Set to None to use the latest version.
    """
    # Write the dataset to a NetCDF file
    base_url = 'https://pcm.arm.gov/pcm/api/dods/'

    # Get data from DOD api
    with urllib.request.urlopen(base_url + process) as url:
        data = json.loads(url.read().decode())
    keys = list(data['versions'].keys())
    
    version = keys[-1]
    variables = data['versions'][version]['vars']
    encoding = {}
    for v in variables:
        type_str = v['type']
        if v['name'] in ds.variables:
            if type_str == 'float':
                encoding[v['name']] = {'dtype': 'float32'}
            elif type_str == 'double':
                encoding[v['name']] = {'dtype': 'float64'}
            elif type_str == 'short':
                encoding[v['name']] = {'dtype': 'int16'}
            elif type_str == 'int':
                encoding[v['name']] = {'dtype': 'int32'}
            elif type_str == "char":
                encoding[v['name']] = {'dtype': 'S1'}
            elif type_str == "byte":
                encoding[v['name']] = {'dtype': 'int8'}

    ds.to_netcdf(output_filename, format='NETCDF4_CLASSIC', encoding=encoding)