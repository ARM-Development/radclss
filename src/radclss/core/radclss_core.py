import logging
import time
import xarray as xr
import act
import numpy as np
import pandas as pd

from ..util.column_utils import subset_points, match_datasets_act, get_nexrad_column
from ..config.default_config import DEFAULT_DISCARD_VAR
from ..config.output_config import get_output_config
from dask.distributed import Client, as_completed


def radclss(
    volumes,
    input_site_dict,
    time_coords,
    serial=True,
    dod_version="1.0",
    discard_var={},
    verbose=False,
    base_station="M1",
    current_client=None,
    nexrad=True,
    nexrad_site=None,
    height_bins=np.arange(500, 8500, 250),
):
    """
    Extracted Radar Columns and In-Situ Sensors

    Utilizing Py-ART and ACT, extract radar columns above various sites and
    collocate with in-situ ground based sensors.

    Within this version of RadCLss, supported sensors are:
        - Pluvio Weighing Bucket Rain Gauge (pluvio)
        - Surface Meteorological Sensors (met)
        - Laser Disdrometer (multiple sites) (ld/vdisquants)
        - Radiosonde (sondewnpn)
        - Ceilometer (ceil)
        - NEXRAD radar

    Parameters
    ----------
    volumes : dict
        Dictionary containing files for each of the instruments, including
        all CMAC processed radar files per day. Each key is formatted as follows:
        'date': 'YYYYMMDD'
        'instrument_site': file/list of files
        where instrument is one of the supported instruments (radar, met, sonde,
        pluvio, ld, vd, wxt) and site is the site name (e.g., M1, S2, S3, etc).
    input_site_dict : dict
        Dictionary containing site information for each site being processed.
        Each key is the site name (e.g., M1, S2, S3, etc) and the value is a 3-tuple
        containing (latitude, longitude, altitude in meters).
    time_coords : str
        The instrument to base the time coordinates off of, or an averaging interval
        in minutes or seconds. For example "radar_csapr2cmac" will use the CSAPR2 times as
        the time coordinate for all of the data. NEXRAD is currently not supported as a time coordinate, but can be used as a reference for reindexing.
         If the specified time coordinate is not found in the volumes, then an error will be raised.
    serial : bool, optional
        Option to denote serial processing. Set to False to use dask cluster for
        subsetting columns in parallel. Default is True.
    dod_version : str, optional
        Option to supply a Data Object Description version to verify standards.
        If this is an empty string, then the latest version will be used. Default is '1.2'.
    discard_var : dict, optional
        Dictionary containing variables to drop from each datastream. Default is {}.
    verbose : bool, optional
        Option to print additional information during processing. Default is False.
    base_station : str, optional
        The base station name to use for time variables. Default is "M1".
    current_client : dask.distributed.Client, optional
        Option to supply an existing Dask client for parallel processing.
        Set to None to use the current active client. Default is None.
    nexrad : bool, optional
        Set to True to pull from the nearest NEXRAD from the ARM site. Default is True.
    nexrad_site : str or None, optional
        If the nexrad flag is True, then use this NEXRAD radar to get the data.
        Set to None to use the default settings in RadCLss. Default is None.
    height_bins : numpy.ndarray, optional
        The height bins in meters to provide the column over.
        Default is np.arange(500, 8500, 250).

    Returns
    -------
    ds : xarray.Dataset
        Daily time-series of extracted columns saved into ARM formatted netCDF files.
    """

    if discard_var == {}:
        discard_var = DEFAULT_DISCARD_VAR

    if "sonde" not in volumes.keys():
        volumes["sonde"] = None

    if verbose:
        print("=" * 80)
        print(f"RadCLss Processing for {volumes['date']}")
        print("=" * 80)
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Serial mode: {serial}")
        print(f"NEXRAD enabled: {nexrad}")
        print(f"Time coordinates: {time_coords}")
        print(f"Number of sites: {len(input_site_dict)}")
        print(f"Sites: {list(input_site_dict.keys())}")
        print(
            f"Height bins: {len(height_bins)} levels from {height_bins[0]}m to {height_bins[-1]}m"
        )
        print("-" * 80)

    if "radar" in time_coords:
        if time_coords not in volumes.keys():
            raise IndexError(
                f"{time_coords} is not a valid time basis! Please choose a radar or an "
                + "Interval"
            )
        if verbose:
            print(f"Using {time_coords} as time basis")
            print(f"Number of {time_coords} files: {len(volumes[time_coords])}")
    else:
        raise NotImplementedError(
            "Currently, only radar-based time coordinates are supported. Please specify a radar key from the volumes dictionary as the time_coords argument."
        )
    # Call Subset Points
    columns = {}
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 1: Extracting radar columns")
        print("=" * 80)

    if not serial:
        if current_client is None:
            try:
                current_client = Client.current()
            except ValueError:
                raise RuntimeError(
                    "No Dask client found. Please start a Dask client before running in parallel mode."
                )
        for k in volumes.keys():
            if "radar" in k:
                if verbose:
                    print(f"\nProcessing radar: {k}")
                    print(f"  Number of files: {len(volumes[k])}")
                    print(f"  Submitting {len(volumes[k])} tasks to dask cluster...")
                columns[k] = []
                results = current_client.map(
                    subset_points,
                    volumes[k],
                    sonde=volumes["sonde"],
                    input_site_dict=input_site_dict,
                    height_bins=height_bins,
                    rad_key=k,
                )

                successful_count = 0
                failed_count = 0
                for done_work in as_completed(results, with_results=False):
                    try:
                        columns[k].append(done_work.result())
                        successful_count += 1
                        if verbose and successful_count % 10 == 0:
                            print(
                                f"  Completed {successful_count}/{len(volumes[k])} files..."
                            )
                    except Exception:
                        failed_count += 1
                        if verbose:
                            print(
                                f"  ERROR processing file (total failures: {failed_count})"
                            )

                if verbose:
                    print(
                        f"  Finished {k}: {successful_count} successful, {failed_count} failed"
                    )
    else:
        for k in volumes.keys():
            if "radar" in k:
                if verbose:
                    print(f"\nProcessing radar: {k}")
                    print(f"  Number of files: {len(volumes[k])}")
                columns[k] = []
                file_count = 0
                for rad in volumes[k]:
                    file_count += 1
                    if verbose:
                        print(
                            f"  [{file_count}/{len(volumes[k])}] Processing: {rad.split('/')[-1]}"
                        )
                    result = subset_points(
                        rad,
                        sonde=volumes["sonde"],
                        input_site_dict=input_site_dict,
                        height_bins=height_bins,
                        rad_key=k,
                    )
                    columns[k].append(result)
                    if verbose:
                        if result is not None:
                            print(
                                f"    ✓ Success - extracted {result.dims.get('time', 0)} time steps"
                            )
                        else:
                            print("    ✗ Failed - no data extracted")

                if verbose:
                    successful = sum(1 for c in columns[k] if c is not None)
                    print(
                        f"  Finished {k}: {successful}/{len(columns[k])} successful extractions"
                    )

    # Assemble individual columns into single DataSet
    # try:
    # Concatenate all extracted columns across time dimension to form daily timeseries

    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2: Assembling columns and determining time range")
        print("=" * 80)

    output_config = get_output_config()
    nexrad_columns = []
    min_times = {}
    max_times = {}
    for k in columns.keys():
        if "radar" in k and len(columns[k]) > 0:
            times = np.array([x["base_time"].values[0] for x in columns[k]])
            min_times[k] = np.min(times)
            max_times[k] = np.max(times)
            if verbose:
                print(f"  {k}: {len(columns[k])} columns")
                print(f"    Time range: {min_times[k]} to {max_times[k]}")

    min_time = min(np.array([x for x in min_times.values()]))
    max_time = max(np.array([x for x in max_times.values()]))

    if verbose:
        print(f"\nOverall time range: {min_time} to {max_time}")

    if nexrad:
        if verbose:
            print("\n" + "=" * 80)
            print("STEP 3: Fetching NEXRAD data")
            print("=" * 80)
            print(
                f"  NEXRAD site: {nexrad_site if nexrad_site else 'auto-detect from ARM site'}"
            )

        if "radar" in time_coords:
            time_list = sorted(
                [
                    str(x["base_time"].dt.strftime("%Y-%m-%dT%H:%M:%S").values[0])
                    for x in columns[time_coords]
                ]
            )

        if verbose:
            print(f"  Number of NEXRAD time steps to fetch: {len(time_list)}")
            print(f"  Time list: {time_list[0]} to {time_list[-1]}")

        if not serial:
            if current_client is None:
                try:
                    current_client = Client.current()
                except ValueError:
                    raise RuntimeError(
                        "No Dask client found. Please start a Dask client before running in parallel mode."
                    )
            if verbose:
                print(f"  Submitting {len(time_list)} NEXRAD tasks to dask cluster...")

            def _get_nexrad_wrapper(time_str):
                return get_nexrad_column(
                    time_str,
                    output_config["site"],
                    input_site_dict,
                    nexrad_radar=nexrad_site,
                )

            results = current_client.map(_get_nexrad_wrapper, time_list)

            successful_count = 0
            failed_count = 0
            for done_work in as_completed(results, with_results=False):
                try:
                    nexrad_columns.append(done_work.result())
                    successful_count += 1
                    if verbose and successful_count % 5 == 0:
                        print(
                            f"  Completed {successful_count}/{len(time_list)} NEXRAD columns..."
                        )
                except Exception as error:
                    failed_count += 1
                    if verbose:
                        print(
                            f"  ERROR fetching NEXRAD data (total failures: {failed_count})"
                        )
                    logging.exception(error)

            if verbose:
                print(
                    f"  Finished NEXRAD: {successful_count} successful, {failed_count} failed"
                )
        else:
            if verbose:
                print("  Processing NEXRAD columns in serial mode...")
            for i, time_str in enumerate(time_list, 1):
                if verbose and i % 5 == 0:
                    print(f"  [{i}/{len(time_list)}] Fetching NEXRAD for {time_str}")
                nexrad_columns.append(
                    get_nexrad_column(
                        time_str,
                        output_config["site"],
                        input_site_dict,
                    )
                )

        if verbose:
            valid_nexrad = sum(1 for x in nexrad_columns if x is not None)
            print(f"  Concatenating {valid_nexrad} valid NEXRAD columns...")

        nexrad_columns = xr.concat(
            [data for data in nexrad_columns if data], dim="time"
        )
    else:
        nexrad_columns = None

    if verbose:
        print("\n" + "=" * 80)
        print("STEP 4: Concatenating and processing time coordinates")
        print("=" * 80)

    output_platform = output_config["platform"]
    output_level = output_config["level"]

    # Convert time variables to something xarray understands
    ds_concat = {}
    for k in columns.keys():
        if verbose:
            print(f"  Processing {k}...")
        ds_concat[k] = xr.concat([data for data in columns[k] if data], dim="time")
        if verbose:
            print(
                f"    Concatenated dimensions: time={ds_concat[k].dims['time']}, station={ds_concat[k].dims['station']}, height={ds_concat[k].dims['height']}"
            )
        ds_concat[k]["time"] = ds_concat[k].sel(station=base_station).base_time
        ds_concat[k]["base_time"] = (
            ds_concat[k].sel(station=base_station).isel(time=0).base_time
        )
        ds_concat[k] = ds_concat[k].sortby("time")

    if nexrad:
        if verbose:
            print("  Processing NEXRAD columns...")
            print(
                f"    NEXRAD dimensions: time={nexrad_columns.dims['time']}, station={nexrad_columns.dims['station']}, height={nexrad_columns.dims['height']}"
            )
        nexrad_columns["time"] = nexrad_columns.sel(station=base_station).base_time
        nexrad_columns["base_time"] = (
            nexrad_columns.sel(station=base_station).isel(time=0).base_time
        )
        nexrad_columns = nexrad_columns.sortby("time")
        nexrad_columns = nexrad_columns.drop_duplicates(dim="time")
        if verbose:
            print(
                f"    After removing duplicates: {nexrad_columns.dims['time']} time steps"
            )

    # Do the time resampling
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 5: Time resampling and alignment")
        print("=" * 80)
        print(f"  Time coordinate method: {time_coords}")

    if "radar" in time_coords:
        if verbose:
            print(f"  Reindexing all datasets to {time_coords} time coordinates")
            print(f"    Reference time steps: {len(ds_concat[time_coords]['time'])}")
        for k in ds_concat.keys():
            if not k == time_coords:
                if verbose:
                    print(f"    Reindexing {k}...")
                ds_concat[k] = ds_concat[k].reindex(
                    time=ds_concat[time_coords]["time"], method="nearest"
                )

        if nexrad:
            if verbose:
                print("    Reindexing NEXRAD columns...")
            nexrad_columns = nexrad_columns.reindex(
                time=ds_concat[time_coords]["time"], method="nearest"
            )
    elif time_coords.lower() == "nexrad":
        if verbose:
            print("  Reindexing all datasets to NEXRAD time coordinates")
            print(f"    Reference time steps: {len(nexrad_columns['time'])}")
        for k in ds_concat.keys():
            if verbose:
                print(f"    Reindexing {k}...")
            ds_concat[k] = ds_concat[k].reindex(
                time=nexrad_columns["time"], method="nearest"
            )
    else:
        if verbose:
            print(f"  Resampling to {time_coords} intervals")
        for k in ds_concat.keys():
            ds_concat[k] = ds_concat[k].resample(time=time_coords)
        if nexrad:
            nexrad_columns = nexrad_columns.resample(time=time_coords)

        # Then, reindex to the largest of the time arrays
        new_coordinates = pd.date_range(min_time, max_time, time_coords)
        if verbose:
            print(f"    Creating new time grid: {len(new_coordinates)} time steps")
        for k in ds_concat.keys():
            ds_concat[k] = ds_concat[k].reindex(time=new_coordinates)
        if nexrad:
            nexrad_columns = nexrad_columns.reindex(time=new_coordinates)

    # Rename all variables according to their radar name
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 6: Renaming variables and merging datasets")
        print("=" * 80)

    for k in ds_concat.keys():
        radar_name = k.split("_")[1:]
        if verbose:
            print(f"  Renaming {k} variables with prefix: {'_'.join(radar_name)}_")
        for var in ds_concat[k].data_vars:
            if var not in [
                "time",
                "time_offset",
                "base_time",
                "height",
                "lat",
                "lon",
                "alt",
                "latitude",
                "longitude",
            ]:
                if "sonde_" not in var:
                    ds_concat[k] = ds_concat[k].rename_vars(
                        {var: f"{radar_name[0]}_{var}"}
                    )

    if nexrad_columns is not None:
        if verbose:
            print("  Renaming NEXRAD variables with prefix: nexrad_")
        for var in nexrad_columns.data_vars:
            if var not in [
                "time",
                "time_offset",
                "base_time",
                "height",
                "lat",
                "lon",
                "alt",
                "latitude",
                "longitude",
            ]:
                if "sonde_" not in var:
                    nexrad_columns = nexrad_columns.rename_vars({var: f"nexrad_{var}"})

    if verbose:
        print(f"  Merging {len(ds_concat)} radar datasets...")

    # Drop time_offset since we won't need it until we write the final dataset
    for k in ds_concat.keys():
        if verbose:
            print(f" Time arrays from {k}:")
            print(ds_concat[k]["base_time"])
        ds_concat[k] = ds_concat[k].drop(["time_offset", "base_time"])
    nexrad_columns = nexrad_columns.drop(["time_offset", "base_time"])
    first_key = list(ds_concat.keys())[0]
    for k in list(ds_concat.keys())[1:]:
        for var in ds_concat[k].data_vars:
            if var in ds_concat[first_key].data_vars:
                if verbose:
                    print(f"Dropping {var} from {k}")
                ds_concat[k] = ds_concat[k].drop(var)

    for var in nexrad_columns.data_vars:
        for k in ds_concat.keys():
            if var in ds_concat[k].data_vars:
                if verbose:
                    print(f"Dropping {var} from nexrad_columns")
                nexrad_columns = nexrad_columns.drop(var)

    ds_concat = xr.merge([x for x in ds_concat.values()])
    if verbose:
        print("Output xarray dataset so far:")
        print(ds_concat)

    if nexrad_columns is not None:
        if verbose:
            print("  Merging NEXRAD data into combined dataset...")
        ds_concat = xr.merge([ds_concat, nexrad_columns])

    if verbose:
        print(f"  Total variables in merged dataset: {len(ds_concat.data_vars)}")
        print("\n" + "=" * 80)
        print("STEP 7: Creating output dataset from ARM DOD")
        print("=" * 80)
        print(f"  Platform/Level: {output_platform}.{output_level}")
        print(f"  DOD version: {dod_version}")
        print("Variables in merged dataset:")
        for vars in ds_concat.data_vars:
            print(vars)

    ds = act.io.create_ds_from_arm_dod(
        f"{output_platform}.{output_level}",
        {
            "time": ds_concat.sizes["time"],
            "height": ds_concat.sizes["height"],
            "station": ds_concat.sizes["station"],
        },
        version=dod_version,
    )

    if verbose:
        print("  Created output dataset with dimensions:")
        print(f"    time: {ds.sizes['time']}")
        print(f"    height: {ds.sizes['height']}")
        print(f"    station: {ds.sizes['station']}")
        print("\n  Assigning coordinate variables...")

    # Calculate base_time as the first timestamp
    ds["time"] = ds_concat["time"]
    ds["base_time"] = ds_concat.time[0]

    # Calculate time as seconds since base_time
    ds["time_offset"] = ds["time"]
    ds["station"] = ds_concat["station"]
    ds["height"] = ds_concat["height"]
    ds["lat"][:] = ds_concat.isel(time=0)["lat"][:]
    ds["lon"][:] = ds_concat.isel(time=0)["lon"][:]
    ds["alt"][:] = ds_concat.isel(time=0)["alt"][:]

    if verbose:
        print("\n" + "=" * 80)
        print("STEP 8: Populating output dataset with radar variables")
        print("=" * 80)

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
    if verbose:
        print("\n  Freeing memory: deleting intermediate datasets...")
    del ds_concat

    # Free up Memory
    del columns

    # If successful column extraction, apply in-situ
    if ds:
        # Depending on how Dask is behaving, may be to resort time
        ds = ds.sortby("time")

        if verbose:
            print("\n" + "=" * 80)
            print("STEP 9: Matching in-situ ground instruments")
            print("=" * 80)
            print(f"  Radar processing completed at: {time.strftime('%H:%M:%S')}")

        # Find all of the met stations and match to columns
        vol_keys = list(volumes.keys())
        for k in vol_keys:
            if k == "sonde":
                continue
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

    if verbose:
        print("\n" + "=" * 80)
        print("STEP 10: Finalizing dataset")
        print("=" * 80)
        print("  Removing time unit attributes...")

    del ds["base_time"].attrs["units"]
    del ds["time_offset"].attrs["units"]
    del ds["time"].attrs["units"]

    if verbose:
        print("\n" + "=" * 80)
        print(f"RadCLss Processing Complete for {volumes['date']}")
        print("=" * 80)
        print(f"  End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("  Final dataset dimensions:")
        print(f"    time: {ds.dims['time']}")
        print(f"    height: {ds.dims['height']}")
        print(f"    station: {ds.dims['station']}")
        print(f"  Total variables: {len(ds.data_vars)}")
        print(f"  Total size: {ds.nbytes / 1e6:.2f} MB")
        print("=" * 80)

    return ds
