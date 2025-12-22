import logging
import time
import xarray as xr
import act

from ..util.column_utils import subset_points, match_datasets_act
from ..config.default_config import DEFAULT_DISCARD_VAR
from ..config.output_config import OUTPUT_SITE, OUTPUT_FACILITY, OUTPUT_PLATFORM, OUTPUT_LEVEL
from dask.distributed import Client, as_completed

def radclss(volumes, serial=True, dod_file=None, discard_var={}, verbose=False):
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
            'instrument_site': 
    serial : Boolean, Default = False
        Option to denote serial processing; used to start dask cluster for
        subsetting columns in parallel.
    dod_file : str, Default = None
        Option to supply a Data Object Description file to verify standards
    discard_var : Dictionary, Default = {}
        Dictionary containing variables to drop from each datastream.
    verbose : Boolean, Default = False
        Option to print additional information during processing.

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
            results = current_client.map(subset_points, volumes["radar"])
            for done_work in as_completed(results, with_results=False):
                try:
                    columns.append(done_work.result())
                except Exception as error:
                    logging.log.exception(error)
    else:
        for rad in volumes['radar']:
            columns.append(subset_points(rad))

    # Assemble individual columns into single DataSet
    try:
        # Concatenate all extracted columns across time dimension to form daily timeseries
        ds = xr.concat([data for data in columns if data], dim="time")
        ds['time'] = ds.sel(station="M1").base_time
        ds['time_offset'] = ds.sel(station="M1").base_time
        ds['base_time'] = ds.sel(station="M1").isel(time=0).base_time
        ds['lat'] = ds.isel(time=0).lat
        ds['lon'] = ds.isel(time=0).lon
        ds['alt'] = ds.isel(time=0).alt
        # Remove all the unused CMAC variables
        ds = ds.drop_vars(discard_var["radar"])
        # Drop duplicate latitude and longitude
        ds = ds.drop_vars(['latitude', 'longitude'])
    except ValueError:
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

        # ----------------
        # Check DOD - TBD
        # ----------------
        # verify the correct dimension order
        ds = ds.transpose("time", "height", "station")
        if dod_file:
            try:
                dod = xr.open_dataset(dod_file)
                # verify the correct dimension order
                ds = adjust_radclss_dod(ds, dod)
            except ValueError as e:
                print(f"Error: {e}")
                print(f"Error type: {type(e).__name__}")
                print("WARNING: Unable to Verify DOD")
        
    else:
        # There is no column extraction
        raise RuntimeError(": RadCLss FAILURE (All Columns Failed to Extract): ")

    return ds