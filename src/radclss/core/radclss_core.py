import logging
import time
import xarray as xr
import act
import numpy as np

from ..util.column_utils import subset_points, match_datasets_act
from ..config.default_config import DEFAULT_DISCARD_VAR
from ..config.output_config import get_output_config
from dask.distributed import Client, as_completed


def radclss(
    volumes,
    input_site_dict,
    serial=True,
    dod_version="",
    discard_var={},
    verbose=False,
    base_station="M1",
    current_client=None,
):
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
    current_client : Dask Client, Default = None
        Option to supply an existing Dask client for parallel processing.
        Set to None to use the current active client.

    Returns
    -------
    ds : Xarray Dataset
        Daily time-series of extracted columns saved into ARM formatted netCDF files.
    """

    if discard_var == {}:
        discard_var = DEFAULT_DISCARD_VAR
    if verbose:
        print(volumes["date"] + " start subset-points: ", time.strftime("%H:%M:%S"))

    # Call Subset Points
    columns = []
    if not serial:
        if current_client is None:
            try:
                current_client = Client.current()
            except ValueError:
                raise RuntimeError(
                    "No Dask client found. Please start a Dask client before running in parallel mode."
                )
        results = current_client.map(
            subset_points,
            volumes["radar"],
            sonde=volumes["sonde"],
            input_site_dict=input_site_dict,
        )
        for done_work in as_completed(results, with_results=False):
            try:
                columns.append(done_work.result())
            except Exception as error:
                logging.log.exception(error)
    else:
        for rad in volumes["radar"]:
            if verbose:
                print(f"Processing file: {rad}")
            columns.append(
                subset_points(
                    rad, sonde=volumes["sonde"], input_site_dict=input_site_dict
                )
            )
            if verbose:
                print("Processed file: ", rad)
                print("Current number of successful columns: ", len(columns))
                print("Last processed file results: ")
                print(columns[-1])

    # Assemble individual columns into single DataSet
    # try:
    # Concatenate all extracted columns across time dimension to form daily timeseries
    output_config = get_output_config()
    output_platform = output_config["platform"]
    output_level = output_config["level"]
    ds_concat = xr.concat([data for data in columns if data], dim="time")
    if verbose:
        print("Grabbing DOD for platform/level: ", f"{output_platform}.{output_level}")
    ds = act.io.create_ds_from_arm_dod(
        f"{output_platform}.{output_level}",
        {
            "time": ds_concat.sizes["time"],
            "height": ds_concat.sizes["height"],
            "station": ds_concat.sizes["station"],
        },
        version=dod_version,
    )

    ds["time"] = ds_concat.sel(station=base_station).base_time
    ds["time_offset"] = ds_concat.sel(station=base_station).base_time
    ds["base_time"] = ds_concat.sel(station=base_station).isel(time=0).base_time
    ds["station"] = ds_concat["station"]
    ds["height"] = ds_concat["height"]
    ds["lat"][:] = ds_concat.isel(time=0)["lat"][:]
    ds["lon"][:] = ds_concat.isel(time=0)["lon"][:]
    ds["alt"][:] = ds_concat.isel(time=0)["alt"][:]

    for var in ds_concat.data_vars:
        if var not in ["time", "time_offset", "base_time", "lat", "lon", "alt"]:
            if var in ds.data_vars:
                if verbose:
                    print(f"Adding variable to output dataset: {var}")
                    print(
                        f"Original dtype: {ds[var].dtype}, New dtype: {ds_concat[var].dtype}"
                    )
                old_type = ds[var].dtype

                # Assign data and convert to original dtype
                ds[var][:] = ds_concat[var][:]
                ds[var] = ds[var].astype(old_type)
                if "_FillValue" in ds[var].attrs:
                    if isinstance(ds[var].attrs["_FillValue"], str):
                        if ds[var].dtype == "float32":
                            ds[var].attrs["_FillValue"] = np.float32(
                                ds[var].attrs["_FillValue"]
                            )
                        elif ds[var].dtype == "float64":
                            ds[var].attrs["_FillValue"] = np.float64(
                                ds[var].attrs["_FillValue"]
                            )
                        elif ds[var].dtype == "int32":
                            ds[var].attrs["_FillValue"] = np.int32(
                                ds[var].attrs["_FillValue"]
                            )
                        elif ds[var].dtype == "int64":
                            ds[var].attrs["_FillValue"] = np.int64(
                                ds[var].attrs["_FillValue"]
                            )
                    ds[var] = ds[var].fillna(ds[var].attrs["_FillValue"]).astype(float)
                if "missing_value" in ds[var].attrs:
                    if isinstance(ds[var].attrs["missing_value"], str):
                        if ds[var].dtype == "float32":
                            ds[var].attrs["missing_value"] = np.float32(
                                ds[var].attrs["missing_value"]
                            )
                        elif ds[var].dtype == "float64":
                            ds[var].attrs["missing_value"] = np.float64(
                                ds[var].attrs["missing_value"]
                            )
                        elif ds[var].dtype == "int32":
                            ds[var].attrs["missing_value"] = np.int32(
                                ds[var].attrs["missing_value"]
                            )
                        elif ds[var].dtype == "int64":
                            ds[var].attrs["missing_value"] = np.int64(
                                ds[var].attrs["missing_value"]
                            )
                    ds[var] = (
                        ds[var].fillna(ds[var].attrs["missing_value"]).astype(float)
                    )

    # Remove all the unused CMAC variables
    # Drop duplicate latitude and longitude
    del ds_concat

    # Free up Memory
    del columns

    # If successful column extraction, apply in-situ
    if ds:
        # Depending on how Dask is behaving, may be to resort time
        ds = ds.sortby("time")
        if verbose:
            print(
                volumes["date"] + " finish subset-points: ", time.strftime("%H:%M:%S")
            )
        # Find all of the met stations and match to columns
        vol_keys = list(volumes.keys())
        for k in vol_keys:
            if len(volumes[k]) == 0:
                if verbose:
                    print(f"No files found for instrument/site: {k}")
                continue
            if "_" in k:
                instrument, site = k.split("_", 1)
            else:
                instrument = k
                site = base_station
            if instrument == "met":
                if verbose:
                    print(f"Matching MET data for site: {site}")
                ds = match_datasets_act(
                    ds,
                    volumes[k][0],
                    site.upper(),
                    resample="mean",
                    discard=discard_var["met"],
                    verbose=verbose,
                )

            if instrument == "pluvio":
                # Weighing Bucket Rain Gauge
                ds = match_datasets_act(
                    ds,
                    volumes[k],
                    site.upper(),
                    resample="sum",
                    discard=discard_var["pluvio"],
                    verbose=verbose,
                )

            if instrument == "ld":
                ds = match_datasets_act(
                    ds,
                    volumes[k],
                    site.upper(),
                    discard=discard_var["ldquants"],
                    resample="mean",
                    prefix="ldquants_",
                    verbose=verbose,
                )

            if instrument == "vd":
                # Laser Disdrometer - Supplemental Site
                ds = match_datasets_act(
                    ds,
                    volumes[k],
                    site.upper(),
                    discard=discard_var["vdisquants"],
                    resample="mean",
                    prefix="vdisquants_",
                    verbose=verbose,
                )

            if instrument == "wxt":
                # Laser Disdrometer - Supplemental Site
                ds = match_datasets_act(
                    ds,
                    volumes[k],
                    site.upper(),
                    discard=discard_var["wxt"],
                    resample="mean",
                    verbose=verbose,
                )

    else:
        # There is no column extraction
        raise RuntimeError(": RadCLss FAILURE (All Columns Failed to Extract): ")
    del ds["base_time"].attrs["units"]
    del ds["time_offset"].attrs["units"]
    del ds["time"].attrs["units"]
    return ds
