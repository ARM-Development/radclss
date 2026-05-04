import boto3
import pyart
import act
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import logging

from datetime import timedelta
from botocore.config import Config
from botocore import UNSIGNED

from ..config import DEFAULT_DISCARD_VAR, DEFAULT_NEXRAD_RADARS
from ..config import get_output_config

_sonde_cache = {}
_nexrad_cache = {}


def _read_sonde_cached(path, exclude):
    """Return a fully-loaded sonde dataset, reading from disk only on first call."""
    if path not in _sonde_cache:
        raw = act.io.read_arm_netcdf(path, cleanup_qc=True, drop_variables=exclude)
        _sonde_cache[path] = raw.compute()
        raw.close()
    return _sonde_cache[path]


def _read_nexrad_cache(path):
    if path not in _nexrad_cache.keys():
        raw = pyart.io.read_nexrad_archive(path)
        _nexrad_cache[path] = raw
        # Only maintain 15 files in the cache to save memory
        if len(_nexrad_cache.keys()) > 15:
            first_key = next(iter(_nexrad_cache))
            del _nexrad_cache[first_key]
    return _nexrad_cache[path]


def _grab_90_degree_rays(radar):
    """Special case for column right over the radar in an RHI"""
    # Get the rays within 0.5 degrees of 90 degrees
    ray = np.argmin(radar.elevation["data"] - 90.0)
    moment = {key: [] for key in radar.fields.keys()}
    # Determine the center of each gate for the subsetted rays.
    rhi_z = radar.range["data"]

    for key in moment:
        moment[key] = radar.fields[key]["data"][ray, :].squeeze()
    # Add radar elevation to height gates
    # to define height as center of each gate above sea level
    zgate = rhi_z + radar.altitude["data"][0]
    # Determine the time at the center of each ray within the column
    # Define the start of the radar volume as a numpy datetime object for xr
    base_time = np.datetime64(pyart.util.datetime_from_radar(radar).isoformat(), "ns")
    # Convert Py-ART radar object time (time since volume start) to time delta
    # Add to base time to have sequential time within the xr Dataset
    # for easier future merging/work
    combined_time = pd.to_timedelta(radar.time["data"][ray], unit="s")

    # Create a blank list to hold the xarray DataArrays
    ds_container = []
    da_meta = [
        "units",
        "standard_name",
        "long_name",
        "valid_max",
        "valid_min",
        "coordinates",
    ]
    # Convert the moment dictionary to xarray DataArray.
    # Apply radar object meta data to DataArray attribute
    for key in moment:
        if key != "height":
            da = xr.DataArray(
                moment[key], coords=dict(height=zgate), name=key, dims=["height"]
            )
            for tag in da_meta:
                if tag in radar.fields[key]:
                    da.attrs[tag] = radar.fields[key][tag]
            # Append to ds container
            ds_container.append(da.to_dataset(name=key))

    # Add additional DataArrays 'base_time' and 'time_offset'
    # if not present within the radar object.
    da_base = xr.DataArray(base_time, name="base_time")
    da_offset = xr.DataArray(
        combined_time, coords=dict(height=zgate), name="time_offset", dims=["height"]
    )
    ds_container.append(da_base.to_dataset(name="base_time"))
    ds_container.append(da_offset.to_dataset(name="time_offset"))

    # Create a xarray DataSet from the DataArrays
    column = xr.merge(ds_container)

    # Assign Attributes for the Height and Times
    height_des = (
        "Height Above Sea Level [in meters] for the Center of Each"
        + " Radar Gate Above the Target Location"
    )
    column.height.attrs.update(
        long_name="Height of Radar Beam",
        units="m",
        standard_name="height",
        description=height_des,
    )

    column.base_time.attrs.update(long_name="UTC Reference Time", units="seconds")

    time_long = "Time in Seconds Since Volume Start"
    time_des = (
        "Time in Seconds Since Volume Start that Cooresponds"
        + " to the Center of Each Height Gate"
        + " Above the Target Location"
    )
    column.time_offset.attrs.update(
        long_name=time_long, units="seconds", description=time_des
    )

    # Assign Global Attributes to the DataSet
    column.attrs["distance_from_radar"] = "0 km"
    column.attrs["azimuth"] = "0 degrees"
    column.attrs["latitude_of_location"] = str(radar.latitude["data"][0]) + " degrees"
    column.attrs["longitude_of_location"] = str(radar.longitude["data"][0]) + " degrees"
    return column


def _vpt_to_column_timeseries(radar, height_bins):
    """Convert all rays of a VPT scan to a time series of columns.

    For VPT scans elevation ≈ 90°, so gate range ≈ height above radar.
    Each ray becomes one time step; all rays are stacked into a (time, height) dataset.
    """
    zgate = radar.range["data"] + radar.altitude["data"][0]

    base_vol_time = np.datetime64(
        pyart.util.datetime_from_radar(radar).isoformat(), "ns"
    )
    ray_time_offsets = pd.to_timedelta(radar.time["data"], unit="s")
    abs_times = (base_vol_time + ray_time_offsets).astype("datetime64[ns]")

    da_meta = [
        "units",
        "standard_name",
        "long_name",
        "valid_max",
        "valid_min",
        "coordinates",
    ]
    data_vars = {}
    for key in radar.fields:
        arr = np.ma.filled(radar.fields[key]["data"], np.nan).astype(float)
        attrs = {
            tag: radar.fields[key][tag] for tag in da_meta if tag in radar.fields[key]
        }
        data_vars[key] = xr.DataArray(arr, dims=["time", "height"], attrs=attrs)

    ds = xr.Dataset(data_vars, coords={"height": zgate, "time": abs_times})

    valid_h = np.isfinite(ds["height"])
    if int(valid_h.sum()) > 0:
        try:
            ds = ds.dropna("height").sortby("height").interp(height=height_bins)
        except pd.errors.InvalidIndexError:
            ds = (
                ds.drop_duplicates("height", keep="first")
                .dropna("height")
                .sortby("height")
                .interp(height=height_bins)
            )
    else:
        ds = ds.reindex(height=xr.DataArray(height_bins, dims="height", name="height"))

    abs_times_s = abs_times.astype("datetime64[s]")
    ds["base_time"] = xr.DataArray(abs_times_s, dims="time")
    ds["time_offset"] = xr.DataArray(
        ray_time_offsets.values.astype("timedelta64[s]"), dims="time"
    )
    ds["gate_time"] = xr.DataArray(abs_times_s, dims="time")

    height_des = (
        "Height Above Sea Level [in meters] for the Center of Each"
        + " Radar Gate Above the Target Location"
    )
    ds.height.attrs.update(
        long_name="Height of Radar Beam",
        units="m",
        standard_name="height",
        description=height_des,
    )
    ds["base_time"].attrs.update(long_name="UTC Reference Time", units="seconds")
    ds["time_offset"].attrs.update(
        long_name="Time in Seconds Since Volume Start", units="seconds"
    )
    ds.attrs["distance_from_radar"] = "0 km"
    ds.attrs["latitude_of_location"] = str(radar.latitude["data"][0]) + " degrees"
    ds.attrs["longitude_of_location"] = str(radar.longitude["data"][0]) + " degrees"
    return ds


def _vpt_nan_fill(radar, height_bins):
    """Return a NaN-filled dataset matching the VPT column time-series shape.

    Used for stations that are not co-located with the VPT radar site so that
    the time axis still aligns with the radar-site columns.
    """
    base_vol_time = np.datetime64(
        pyart.util.datetime_from_radar(radar).isoformat(), "ns"
    )
    ray_time_offsets = pd.to_timedelta(radar.time["data"], unit="s")
    abs_times = (base_vol_time + ray_time_offsets).astype("datetime64[ns]")
    abs_times_s = abs_times.astype("datetime64[s]")

    nrays = radar.nrays
    n_heights = len(height_bins)
    da_meta = [
        "units",
        "standard_name",
        "long_name",
        "valid_max",
        "valid_min",
        "coordinates",
    ]
    data_vars = {}
    for key in radar.fields:
        attrs = {
            tag: radar.fields[key][tag] for tag in da_meta if tag in radar.fields[key]
        }
        data_vars[key] = xr.DataArray(
            np.full((nrays, n_heights), np.nan),
            dims=["time", "height"],
            attrs=attrs,
        )

    ds = xr.Dataset(data_vars, coords={"height": height_bins, "time": abs_times})
    ds["base_time"] = xr.DataArray(abs_times_s, dims="time")
    ds["time_offset"] = xr.DataArray(
        ray_time_offsets.values.astype("timedelta64[s]"), dims="time"
    )
    ds["gate_time"] = xr.DataArray(abs_times_s, dims="time")

    height_des = (
        "Height Above Sea Level [in meters] for the Center of Each"
        + " Radar Gate Above the Target Location"
    )
    ds.height.attrs.update(
        long_name="Height of Radar Beam",
        units="m",
        standard_name="height",
        description=height_des,
    )
    ds["base_time"].attrs.update(long_name="UTC Reference Time", units="seconds")
    ds["time_offset"].attrs.update(
        long_name="Time in Seconds Since Volume Start", units="seconds"
    )
    return ds


def get_nexrad_column(
    rad_time,
    site,
    input_site_dict,
    height_bins=np.arange(500, 8500, 250),
    nexrad_radar=None,
):
    """
    This file will add data from the specified NEXRAD column to RadCLss if it is
    available.

    Parameters
    ----------
    rad_time: str
        The radar time in format "%Y-%m-%dT%H:%M:%S"
    site: str
        The ARM site code (i.e. BNF, SGP) to use.
    input_site_dict : dict
        Dictionary containing the site names as keys and their
        lat/lon coordinates as values in a list format:
        {'site1': [lat1, lon1, alt1],
        'site2': [lat2, lon2, alt2],
        ...}
    height_bins: numpy array
        The height bins in meters to provide the column over.
    nexrad_radar: str or None
        The NEXRAD radar to obtain the column from. Setting to None will use
        the default setting for the ARM site.

    Returns
    -------
    da: xr.Dataset
        An xarray dataset containing the matched columns from the NEXRAD data.

    """
    if nexrad_radar is None:
        if site.lower() in DEFAULT_NEXRAD_RADARS.keys():
            nexrad_radar = DEFAULT_NEXRAD_RADARS[site.lower()]
        else:
            raise UserWarning(
                f"There are no NEXRAD radars within 100 km of {site}. Returning None."
            )
            return None

    lats = list([x[0] for x in input_site_dict.values()])
    lons = list([x[1] for x in input_site_dict.values()])
    site_alt = list([x[2] for x in input_site_dict.values()])
    sites = list(input_site_dict.keys())
    right_now = datetime.datetime.strptime(rad_time, "%Y-%m-%dT%H:%M:%S")
    yesterday = right_now - timedelta(days=1)
    year = right_now.year
    month = right_now.month
    day = right_now.day

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket_name = "unidata-nexrad-level2"
    prefix = f"{year}/{month:02d}/{day:02d}/{nexrad_radar}"
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    file_list = [x["Key"] for x in response["Contents"]]

    # Find yesterday's scans
    prefix = (
        f"{yesterday.year}/{yesterday.month:02d}/{yesterday.day:02d}/{nexrad_radar}"
    )
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    file_list = file_list + [x["Key"] for x in response["Contents"]]
    time_list = []
    for filepath in file_list:
        name = filepath.split("/")[-1]
        if name[-3:] == "MDM":
            time_list.append(
                datetime.datetime.strptime(name, f"{nexrad_radar}%Y%m%d_%H%M%S_V06_MDM")
            )
        else:
            time_list.append(
                datetime.datetime.strptime(name, f"{nexrad_radar}%Y%m%d_%H%M%S_V06")
            )

    time_list = np.array(time_list)
    path = f"s3://{bucket_name}/" + file_list[np.argmin(np.abs(time_list - right_now))]
    radar_obj = _read_nexrad_cache(path)
    try:
        column_list = []
        for lat, lon in zip(lats, lons):
            # Make sure we are interpolating from the radar's location above sea level
            # NOTE: interpolating throughout Troposphere to match sonde to in the future

            da = pyart.util.columnsect.column_vertical_profile(radar_obj, lat, lon)
            da = da.sortby("height")
            # check for valid heights
            valid = np.isfinite(da["height"])
            n_valid = int(valid.sum())
            if n_valid > 0:
                da = da.sortby("height").interp(height=height_bins)
            else:
                target_height = xr.DataArray(height_bins, dims="height", name="height")
                da = da.reindex(height=target_height)

            # Add the latitude and longitude of the extracted column
            da["lat"], da["lon"] = lat, lon
            # Convert timeoffsets to timedelta object and precision on datetime64
            da.time_offset.data = da.time_offset.values.astype("timedelta64[s]")
            da.base_time.data = da.base_time.values.astype("datetime64[s]")
            # Time is based off the start of the radar volume
            da["gate_time"] = da.base_time.values + da.isel(height=0).time_offset.values
            column_list.append(da)
    finally:
        del radar_obj

    # Concatenate the extracted radar columns for this scan across all sites
    ds = xr.concat([data for data in column_list if data], dim="station")
    ds = _add_station_vars(ds, sites, site_alt)
    ds.attrs["nexrad_radar"] = nexrad_radar

    del column_list, da
    return ds


def subset_points(
    nfile,
    input_site_dict,
    sonde=None,
    height_bins=np.arange(500, 8500, 250),
    rad_key="radar_csapr2",
    **kwargs,
):
    """
    Subset a radar file for a set of latitudes and longitudes
    utilizing Py-ART's column-vertical-profile functionality.

    Parameters
    ----------
    file : str
        Path to the radar file to extract columns from
    input_site_dict : dict
        Dictionary containing the site names as keys and their
        lat/lon coordinates as values in a list format:
        {'site1': [lat1, lon1, alt1],
        'site2': [lat2, lon2, alt2],
        ...}
    sonde : list, optional
        List of radiosonde file paths to be merged into the radar
        prior to column extraction. The nearest sonde file to the
        radar start time will be used. Default is None.
    height_bins : numpy array, optional
        Numpy array containing the desired height bins to interpolate
        the extracted radar columns to. Default is np.arange(500, 8500, 250).
    rad_key: str
        The radar key to use for dropping select variables from the column
        statistics.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    ds : xarray DataSet
        Xarray Dataset containing the radar column above a give set of locations

    """
    ds = None

    # Define the splash locations [lon,lat]

    lats = list([x[0] for x in input_site_dict.values()])
    lons = list([x[1] for x in input_site_dict.values()])
    site_alt = list([x[2] for x in input_site_dict.values()])

    sites = list(input_site_dict.keys())
    try:
        radar = pyart.io.read(nfile, exclude_fields=DEFAULT_DISCARD_VAR[rad_key])
    except OSError:
        logging.warning(
            f"{nfile} failed to open and is possibly corrupt."
            + "RadCLss will not generate a column for this file."
        )
        return ds

    try:
        # Check for RHI and reduce to first sweep if > 1 sweep
        if "rhi" in radar.scan_type:
            radar = radar.extract_sweeps([0])

        # Check for single sweep scans
        if np.ma.is_masked(radar.sweep_start_ray_index["data"][1:]):
            radar.sweep_start_ray_index["data"] = np.ma.array([0])
            radar.sweep_end_ray_index["data"] = np.ma.array([radar.nrays])

        if radar.time["data"].size > 0:
            # Easier to map the nearest sonde file to radar gates before extraction
            if sonde is not None:
                # variables to discard when reading in the sonde file
                exclude_sonde = DEFAULT_DISCARD_VAR["sonde"]

                # find the nearest sonde file to the radar start time
                radar_start = datetime.datetime.strptime(
                    nfile.split("/")[-1].split(".")[-3]
                    + "."
                    + nfile.split("/")[-1].split(".")[-2],
                    "%Y%m%d.%H%M%S",
                )
                sonde_start = [
                    datetime.datetime.strptime(
                        xfile.split("/")[-1].split(".")[2]
                        + "-"
                        + xfile.split("/")[-1].split(".")[3],
                        "%Y%m%d-%H%M%S",
                    )
                    for xfile in sonde
                ]
                # difference in time between radar file and each sonde file
                start_diff = np.array([radar_start - sonde for sonde in sonde_start])

                # merge the sonde file into the radar object (cached across radar files)
                sonde_path = sonde[np.argmin(np.abs(start_diff))]
                ds_sonde = _read_sonde_cached(sonde_path, exclude_sonde)

                # create list of variables within sonde dataset to add to the radar file
                for var in list(ds_sonde.keys()):
                    if var != "alt":
                        z_dict, sonde_dict = pyart.retrieve.map_profile_to_gates(
                            ds_sonde.variables[var], ds_sonde.variables["alt"], radar
                        )
                    field_name = list(radar.fields.keys())[0]
                    # add the field to the radar file
                    radar.add_field_like(
                        field_name,
                        "sonde_" + var,
                        sonde_dict["data"],
                        replace_existing=True,
                    )
                    radar.fields["sonde_" + var]["units"] = sonde_dict["units"]
                    radar.fields["sonde_" + var]["long_name"] = sonde_dict["long_name"]
                    radar.fields["sonde_" + var]["standard_name"] = sonde_dict[
                        "standard_name"
                    ]
                    radar.fields["sonde_" + var][
                        "input_datastream"
                    ] = ds_sonde.datastream

                del radar_start, sonde_start, ds_sonde
                del z_dict, sonde_dict

            column_list = []
            for lat, lon, site in zip(lats, lons, sites):
                # Make sure we are interpolating from the radar's location above sea level
                # NOTE: interpolating throughout Troposphere to match sonde to in the future

                if "vpt" in radar.scan_type:
                    if radar.metadata.get("facility_id", "") == site:
                        da = _vpt_to_column_timeseries(radar, height_bins)
                    else:
                        da = _vpt_nan_fill(radar, height_bins)
                    da["lat"] = lat
                    da["lon"] = lon
                    column_list.append(da)
                    continue

                if "rhi" not in radar.scan_type:
                    da = pyart.util.columnsect.column_vertical_profile(radar, lat, lon)
                else:
                    try:
                        da = pyart.util.get_field_location(radar, lat, lon)
                    except ValueError:
                        # If the columnsect fails, try adding 180 to the azimuths to account for potential mislabeling of radar location
                        if np.all(radar.azimuth["data"] < 180):
                            radar.azimuth["data"] = radar.azimuth["data"] + 180
                            # Need to adjust elevation as well to maintain the same relative geometry between radar and column locations
                            radar.elevation["data"] = 180 - radar.elevation["data"]
                        try:
                            da = pyart.util.get_field_location(radar, lat, lon)
                        except ValueError:
                            # Grab the vertically pointing ray(s) if the radar site == site
                            if radar.metadata["facility_id"] == site:
                                try:
                                    da = _grab_90_degree_rays(radar)
                                except Exception:
                                    logging.warning(
                                        f"Failed to grab 90 degree rays for {site} from {nfile}."
                                        + "NaNs will be returned for this column."
                                    )
                            else:
                                # NaNs will be returned if the columnsect fails again after adjusting the azimuths
                                da = pyart.util.columnsect.column_vertical_profile(
                                    radar, lat, lon
                                )

                    time_offset = da["time_offset"]
                # check for valid heights
                da = da.sortby("height")
                valid = np.isfinite(da["height"])
                n_valid = int(valid.sum())
                if n_valid > 0:
                    try:
                        # Drop all NaNs
                        da = (
                            da.dropna("height")
                            .sortby("height")
                            .interp(height=height_bins)
                        )
                    except pd.errors.InvalidIndexError:
                        da = da.drop_duplicates("height", keep="first")

                        valid = np.isfinite(da["height"])
                        da = (
                            da.dropna("height")
                            .sortby("height")
                            .interp(height=height_bins)
                        )
                        time_offset = time_offset.drop_duplicates(
                            "height", keep="first"
                        )
                else:
                    target_height = xr.DataArray(
                        height_bins, dims="height", name="height"
                    )
                    da = da.reindex(height=target_height)
                if "rhi" in radar.scan_type:
                    da["time_offset"] = time_offset

                # Add the latitude and longitude of the extracted column
                da["lat"], da["lon"] = lat, lon
                # Convert timeoffsets to timedelta object and precision on datetime64
                da["time_offset"].data = da["time_offset"].values.astype(
                    "timedelta64[s]"
                )
                da.base_time.data = da.base_time.values.astype("datetime64[s]")
                # Time is based off the start of the radar volume
                da["gate_time"] = (
                    da.base_time.values + da.isel(height=0).time_offset.values
                )
                column_list.append(da)

            # Concatenate the extracted radar columns for this scan across all sites
            ds = xr.concat([data for data in column_list if data], dim="station")
            ds = _add_station_vars(ds, sites, site_alt)

            del column_list, da
    finally:
        del radar
    return ds


def _prepare_match(
    ground,
    site,
    discard,
    column_time,
    column_height,
    resample="sum",
    resample_time="5Min",
    DataSet=False,
    prefix=None,
):
    """
    Load a ground instrument file, resample it to the column time/height grid,
    and return ``(site, matched_dataset)`` without modifying the column in-place.

    Safe to call concurrently from multiple threads.
    """
    if DataSet:
        grd_ds = ground
    else:
        _grd_raw = act.io.read_arm_netcdf(
            ground,
            cleanup_qc=True,
            drop_variables=discard,
            parallel=False,
        )
        grd_ds = _grd_raw
        if prefix:
            if prefix == "wxt_":
                rename_dict = {
                    v: f"{prefix}{v}" for v in grd_ds.data_vars if "wxt_" not in v
                }
            else:
                rename_dict = {v: f"{prefix}{v}" for v in grd_ds.data_vars}

            grd_ds = grd_ds.rename_vars(rename_dict)

    if "base_time" in grd_ds.data_vars:
        del grd_ds["base_time"]

    if "height" in grd_ds.dims:
        if grd_ds["height"].attrs["units"] == "km":
            grd_ds["height"] = grd_ds["height"] * 1000
            grd_ds["height"].attrs["units"] = "m"
        grd_ds = grd_ds.interp(height=column_height, method="linear")

    if "range" in grd_ds.dims:
        grd_ds = grd_ds.interp(range=column_height, method="linear")
        grd_ds = grd_ds.drop_vars("height")
        grd_ds = grd_ds.rename({"range": "height"})

    non_numeric_vars = [
        var
        for var in grd_ds.data_vars
        if not np.issubdtype(grd_ds[var].dtype, np.number)
    ]
    grd_ds = grd_ds.drop_vars(non_numeric_vars)

    if resample == "mean":
        matched = (
            grd_ds.resample(time=resample_time, closed="right")
            .mean(keep_attrs=True)
            .interp(time=column_time, method="linear")
        )
    elif resample == "skip":
        matched = grd_ds.interp(time=column_time, method="linear")
    elif resample == "sum":
        matched = (
            grd_ds.resample(time=resample_time, closed="right")
            .sum(keep_attrs=True)
            .interp(time=column_time, method="linear")
        )
    else:
        raise ValueError(
            "Invalid resample method. Please choose 'mean', 'sum', or 'skip'."
        )

    matched = matched.assign_coords(coords=dict(station=site))
    matched = matched.expand_dims("station")

    for attr in ("lat", "lon", "alt"):
        if attr in matched.data_vars:
            del matched[attr]

    for var in matched.data_vars:
        matched[var].attrs.update(source=matched.datastream)
    grd_ds.close()
    _grd_raw.close()
    return site, matched


def _apply_match(column, site, matched):
    """Merge a prepared match result into ``column`` in-place and return it."""
    for k in matched.data_vars:
        if k in column.data_vars:
            column[k].sel(station=site)[:] = matched.sel(station=site)[k][:].astype(
                column[k].dtype
            )
            if "_FillValue" in column[k].attrs:
                if isinstance(column[k].attrs["_FillValue"], str):
                    column[k].attrs["_FillValue"] = float(column[k].attrs["_FillValue"])
                column[k] = (
                    column[k].fillna(column[k].attrs["_FillValue"]).astype(float)
                )
            if "missing_value" in column[k].attrs:
                if isinstance(column[k].attrs["missing_value"], str):
                    column[k].attrs["missing_value"] = float(
                        column[k].attrs["missing_value"]
                    )
                column[k] = (
                    column[k].fillna(column[k].attrs["missing_value"]).astype(float)
                )
    return column


def match_datasets_act(
    column,
    ground,
    site,
    discard,
    resample="sum",
    resample_time="5Min",
    DataSet=False,
    prefix=None,
    verbose=False,
):
    """
    Time synchronization of a Ground Instrumentation Dataset to
    a Radar Column for Specific Locations using the ARM ACT package.
    This module also supports vertically pointing radars such as the KAZR.

    Parameters
    ----------
    column : Xarray DataSet
        Xarray DataSet containing the extracted radar column above multiple locations.
        Dimensions should include Time, Height, Site

    ground : str; Xarray DataSet
        String containing the path of the ground instrumentation file that is desired
        to be included within the extracted radar column dataset.
        If DataSet is set to True, ground is Xarray Dataset and will skip I/O.

    site : str
        Location of the ground instrument. Should be included within the filename.

    discard : list
        List containing the desired input ground instrumentation variables to be
        removed from the xarray DataSet.

    resample : str
        Mathematical operational for resampling ground instrumentation to the radar time.
        'sum' will sum the ground data within the resample_time window,
        'mean' will average the ground data within the resample_time window,
        and 'skip' will not resample the data and will only interpolate to the radar time.
        Default is 'sum'.

    resample_time : str
        Time resolution for resampling ground instrumentation data before mapping to radar time.
        Default is "5Min".

    DataSet : boolean
        Boolean flag to determine if ground input is an Xarray Dataset.
        Set to True if ground input is Xarray DataSet.

    prefix : str
        prefix for the desired spelling of variable names for the input
        datastream (to fix duplicate variable names between instruments)

    verbose : boolean
        Boolean flag to set verbose output during processing. Default is False.

    Returns
    -------
    ds : Xarray DataSet
        Xarray Dataset containing the time-synced in-situ ground observations with
        the inputed radar column
    """
    _, matched = _prepare_match(
        ground,
        site,
        discard,
        column.time,
        column["height"],
        resample=resample,
        resample_time=resample_time,
        DataSet=DataSet,
        prefix=prefix,
    )
    return _apply_match(column, site, matched)


def _add_station_vars(ds, sites, site_alt):
    ds["station"] = sites
    # Assign the Main and Supplemental Site altitudes
    ds = ds.assign(alt=("station", site_alt))
    # Add attributes for Time, Latitude, Longitude, and Sites
    output_config = get_output_config()

    ds.gate_time.attrs.update(output_config["gate_time_attrs"])
    ds.time_offset.attrs.update(output_config["time_offset_attrs"])
    ds.station.attrs.update(output_config["station_attrs"])
    ds.lat.attrs.update(output_config["lat_attrs"])
    ds.lon.attrs.update(output_config["lon_attrs"])
    ds.alt.attrs.update(output_config["alt_attrs"])
    return ds
