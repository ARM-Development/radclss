import logging
import time
import xarray as xr
import act
import xradar as xd 

from ..util.column_utils import subset_points, match_datasets_act
from ..util.dod import adjust_radclss_dod
from ..config.default_config import DEFAULT_DISCARD_VAR
from ..config.output_config import get_output_config
from dask.distributed import Client, as_completed

def radclss(volumes, input_site_dict, serial=True, dod_version='', discard_var={}, verbose=False,
            base_station="M1"):
    """
    Extracted Radar Columns and In-Situ Sensors

    Utilizing Py-ART and ACT, extract radar columns above various sites and 
    collocate with in-situ ground based sensors.

    Within this verison of RadCLss, supported sensors are:
        - Pluvio Weighing Bucket Rain Gauge (pluvio)
        - Surface Meteorological Sensors (met)
        - Laser Disdrometer (mutliple sites) (ld/vdisquants)
        - Radiosonde (sondewnpn)
        - Ceilometer (ceil)

    Parameters
    ----------
    volumes : Dictionary
        Dictionary contianing files for each of the instruments, including
        all CMAC processed radar files per day. Each key is formatted as follows:
            'date': 'YYYYMMDD'
            'instrument_site': file/list of files
        where instrument is one of the supported instruments (radar, met, sonde,
        pluvio, ld, vd, wxt) and site is the site name (e.g., M1, SGP, TWP, etc).
    input_site_dict : Dictionary
        Dictionary containing site information for each site being processed.
        Each key is the site name (e.g., M1, SGP, TWP, etc) and the value is a 3-tuple
        containing (latitude, longitude, altitude in meters).
    serial : Boolean, Default = False
        Option to denote serial processing; used to start dask cluster for
        subsetting columns in parallel.
    dod_version : str, Default = ''
        Option to supply a Data Object Description version to verify standards.
        If this is an empty string, then the latest version will be used.
    discard_var : Dictionary, Default = {}
        Dictionary containing variables to drop from each datastream.
    verbose : Boolean, Default = False
        Option to print additional information during processing.
    base_station : str, Default = "M1"
        The base station name to use for time variables.

    Returns
    -------
    ds : Xarray Dataset
        Daily time-series of extracted columns saved into ARM formatted netCDF files. 
    """
    
    if discard_var == {}:
        discard_var = DEFAULT_DISCARD_VAR
    if verbose:
        print(volumes['date'] + " start subset-points: ", time.strftime("%H:%M:%S"))
    
    # Call Subset Points
    columns = []
    if serial == False:
        current_client = Client.current()
        if current_client is None:
            raise RuntimeError("No Dask client found. Please start a Dask client before running in parallel mode.")
        results = current_client.map(subset_points, volumes["radar"], input_site_dict=input_site_dict)
        for done_work in as_completed(results, with_results=False):
            try:
                columns.append(done_work.result())
            except Exception as error:
                logging.log.exception(error)
    else:
        for rad in volumes['radar']:
            if verbose:
                print(f"Processing file: {rad}")
            columns.append(subset_points(rad, input_site_dict=input_site_dict))
            if verbose:
                print("Processed file: ", rad)
                print("Current number of successful columns: ", len(columns))
                print("Last processed file results: ")
                print(columns[-1])

    # Assemble individual columns into single DataSet
    try:
        # Concatenate all extracted columns across time dimension to form daily timeseries
        output_config = get_output_config()
        output_platform = output_config['platform']
        output_level = output_config['level']
        ds_concat = xr.concat([data for data in columns if data], dim="time")
        ds = act.io.create_ds_from_arm_dod(f'{output_platform}-{output_level}', 
                                               {'time': ds_concat.sizes['time'], 
                                                'height': ds_concat.sizes['height'], 
                                                'station': ds_concat.sizes['station']},
                                               version=dod_version)
        
        
        ds['time'] = ds_concat.sel(station=base_station).base_time
        ds['time_offset'] = ds_concat.sel(station=base_station).base_time
        ds['base_time'] = ds_concat.sel(station=base_station).isel(time=0).base_time
        ds['lat'] = ds_concat.isel(time=0).lat
        ds['lon'] = ds_concat.isel(time=0).lon
        ds['alt'] = ds_concat.isel(time=0).alt
        for var in ds_concat.data_vars:
            if var not in ['time', 'time_offset', 'base_time', 'lat', 'lon', 'alt']:
                ds[var][:] = ds_concat[var][:]

        # Remove all the unused CMAC variables
        ds = ds.drop_vars(discard_var["radar"])
        # Drop duplicate latitude and longitude
        ds = ds.drop_vars(['latitude', 'longitude']) 
        del ds_concat       
    except ValueError as e:
        print(f"Error concatenating columns: {e}")
        ds = None

    # Free up Memory
    del columns 

    # If successful column extraction, apply in-situ
    if ds:
        # Depending on how Dask is behaving, may be to resort time
        ds = ds.sortby("time")
        if verbose:
            print(volumes['date'] + " finish subset-points: ", time.strftime("%H:%M:%S"))
        # Find all of the met stations and match to columns
        vol_keys = list(volumes.keys())
        for k in vol_keys:
            instrument, site = k.split("_", 1)

            if instrument == "met":
                ds = match_datasets_act(ds, 
                                        volumes[k][0], 
                                        site.upper(), 
                                        resample="mean",
                                        discard=discard_var['met'])
        
            # Radiosonde
            if instrument == "sonde":
                # Read in the file using ACT
                grd_ds = act.io.read_arm_netcdf(volumes[k], 
                                                cleanup_qc=True,
                                                drop_variables=discard_var['sonde'])
                # Default are Lazy Arrays; convert for matching with column
                grd_ds = grd_ds.compute()
                # check if a list containing new variable names exists.
                prefix = "sonde_"
                grd_ds = grd_ds.rename_vars({v: f"{prefix}{v}" for v in grd_ds.data_vars})
                # Match to the columns
                ds = match_datasets_act(ds, 
                                        grd_ds, 
                                        site.upper(),
                                        discard=discard_var[instrument],
                                        DataSet=True,
                                        resample="mean")
                # clean up
                del grd_ds

            if instrument == "pluvio":
                # Weighing Bucket Rain Gauge
                ds = match_datasets_act(ds, 
                                        volumes[k][0], 
                                        site.upper(), 
                                        discard=discard_var["pluvio"])

            if instrument == "ld":
                ds = match_datasets_act(ds, 
                                        volumes[k][0], 
                                        site.upper(), 
                                        discard=discard_var['ldquants'],
                                        resample="mean",
                                        prefix="ldquants_")

        
            if instrument == "vd":
                # Laser Disdrometer - Supplemental Site
                ds = match_datasets_act(ds, 
                                        volumes[k][0], 
                                        site.upper(), 
                                        discard=discard_var['vdisquants'],
                                        resample="mean",
                                        prefix="vdisquants_")
        
            if instrument == "wxt":
                # Laser Disdrometer - Supplemental Site
                ds = match_datasets_act(ds, 
                                        volumes[k][0], 
                                        site.upper(), 
                                        discard=discard_var['wxt'],
                                        resample="mean")
        
    else:
        # There is no column extraction
        raise RuntimeError(": RadCLss FAILURE (All Columns Failed to Extract): ")

    return ds