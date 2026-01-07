import pyart
import act  
import numpy as np
import xarray as xr
import datetime

from ..config import DEFAULT_DISCARD_VAR 
from ..config import get_output_config

def subset_points(nfile, input_site_dict, sonde=None, height_bins=np.arange(500, 8500, 250), **kwargs):
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
    #try:
    # Read in the file
   
    #xradar_ds = xd.io.open_cfradial1_datatree(nfile)
    #for var in DEFAULT_DISCARD_VAR['radar']:
    #    if var in xradar_ds.data_vars:
    #        xradar_ds = xradar_ds.drop_vars(var)
    #if xradar_ds["/sweep_0"]["sweep_mode"] == "rhi":
    #    xradar_ds.close()
    #    return None 
    #radar = xradar_ds.pyart.to_radar()
    #xradar_ds.close()
    radar = pyart.io.read(nfile, exclude_fields=DEFAULT_DISCARD_VAR['radar'])
    # Check for single sweep scans
    if np.ma.is_masked(radar.sweep_start_ray_index["data"][1:]):
        radar.sweep_start_ray_index["data"] = np.ma.array([0])
        radar.sweep_end_ray_index["data"] = np.ma.array([radar.nrays])
    #except Exception as e:
    #    print(f"Error reading radar file {nfile}: {e}")
    #    radar = None

    if radar:
        if radar.time['data'].size > 0:
            # Easier to map the nearest sonde file to radar gates before extraction
            if sonde is not None:
                # variables to discard when reading in the sonde file
                exclude_sonde = DEFAULT_DISCARD_VAR['sonde'] 
    
                # find the nearest sonde file to the radar start time
                radar_start = datetime.datetime.strptime(nfile.split('/')[-1].split('.')[-3] + '.' + nfile.split('/')[-1].split('.')[-2], 
                                                        '%Y%m%d.%H%M%S'
                )
                sonde_start = [datetime.datetime.strptime(xfile.split('/')[-1].split('.')[2] + 
                                                            '-' + 
                                                            xfile.split('/')[-1].split('.')[3], 
                                                            '%Y%m%d-%H%M%S') for xfile in sonde
                ]
                # difference in time between radar file and each sonde file
                start_diff = [radar_start - sonde for sonde in sonde_start]

                # merge the sonde file into the radar object
                ds_sonde = act.io.read_arm_netcdf(sonde[start_diff.index(min(start_diff))], 
                                                    cleanup_qc=True, 
                                                    drop_variables=exclude_sonde)

                # create list of variables within sonde dataset to add to the radar file
                for var in list(ds_sonde.keys()):
                    if var != "alt":
                        z_dict, sonde_dict = pyart.retrieve.map_profile_to_gates(ds_sonde.variables[var],
                                                                                ds_sonde.variables['alt'],
                                                                                radar)
                    # add the field to the radar file
                    radar.add_field_like('corrected_reflectivity', "sonde_" + var,  sonde_dict['data'], replace_existing=True)
                    radar.fields["sonde_" + var]["units"] = sonde_dict["units"]
                    radar.fields["sonde_" + var]["long_name"] = sonde_dict["long_name"]
                    radar.fields["sonde_" + var]["standard_name"] = sonde_dict["standard_name"]
                    radar.fields["sonde_" + var]["datastream"] = ds_sonde.datastream

                del radar_start, sonde_start, ds_sonde
                del z_dict, sonde_dict
    
            column_list = []
            for lat, lon in zip(lats, lons):
                # Make sure we are interpolating from the radar's location above sea level
                # NOTE: interpolating throughout Troposphere to match sonde to in the future
                
                da = pyart.util.columnsect.column_vertical_profile(radar, lat, lon)
                # check for valid heights
                valid = np.isfinite(da["height"])
                n_valid = int(valid.sum())
                if n_valid > 0:
                    da = da.sel(height=valid).sortby("height").interp(height=height_bins)
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
    
            # Concatenate the extracted radar columns for this scan across all sites    
            ds = xr.concat([data for data in column_list if data], dim='station')
            ds["station"] = sites
            # Assign the Main and Supplemental Site altitudes
            ds = ds.assign(alt=("station", site_alt))
            # Add attributes for Time, Latitude, Longitude, and Sites
            output_config = get_output_config()

            ds.gate_time.attrs.update(output_config['gate_time_attrs'])
            ds.time_offset.attrs.update(output_config['time_offset_attrs'])
            ds.station.attrs.update(output_config['station_attrs'])
            ds.lat.attrs.update(output_config['lat_attrs'])
            ds.lon.attrs.update(output_config['lon_attrs'])
            ds.alt.attrs.update(output_config['alt_attrs'])
                                        
            # delete the radar to free up memory
            del radar, column_list, da
        else:
            # delete the rhi file
            del radar
    return ds

def match_datasets_act(column, 
                       ground, 
                       site, 
                       discard, 
                       resample='sum', 
                       DataSet=False,
                       prefix=None,
                       verbose=False):
    """
    Time synchronization of a Ground Instrumentation Dataset to 
    a Radar Column for Specific Locations using the ARM ACT package
    
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
        Default is to sum the data across the resampling period. Checks for 'mean' or 
        to 'skip' altogether. 
    
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
     # Check to see if input is xarray DataSet or a file path
    if DataSet == True:
        grd_ds = ground
    else:
        # Read in the file using ACT
        grd_ds = act.io.read_arm_netcdf(ground, cleanup_qc=True, drop_variables=discard)
        # Default are Lazy Arrays; convert for matching with column
        grd_ds = grd_ds.compute()
        # check if a list containing new variable names exists. 
        if prefix:
            grd_ds = grd_ds.rename_vars({v: f"{prefix}{v}" for v in grd_ds.data_vars})
        
    # Remove Base_Time before Resampling Data since you can't force 1 datapoint to 5 min sum
    if 'base_time' in grd_ds.data_vars:
        del grd_ds['base_time']
        
    # Check to see if height is a dimension within the ground instrumentation. 
    # If so, first interpolate heights to match radar, before interpolating time.
    if 'height' in grd_ds.dims:
        grd_ds = grd_ds.interp(height=np.arange(3150, 10050, 50), method='linear')
        
    # Resample the ground data to 5 min and interpolate to the CSU X-Band time. 
    # Keep data variable attributes to help distingish between instruments/locations
    if resample.split('=')[-1] == 'mean':
        matched = grd_ds.resample(time='5Min', 
                                  closed='right').mean(keep_attrs=True).interp(time=column.time, 
                                                                               method='linear')
    elif resample.split('=')[-1] == 'skip':
        matched = grd_ds.interp(time=column.time, method='linear')
    else:
        matched = grd_ds.resample(time='5Min', 
                                  closed='right').sum(keep_attrs=True).interp(time=column.time, 
                                                                              method='linear')
    
    # Add SAIL site location as a dimension for the Pluvio data
    matched = matched.assign_coords(coords=dict(station=site))
    matched = matched.expand_dims('station')
   
    # Remove Lat/Lon Data variables as it is included within the Matched Dataset with Site Identfiers
    if 'lat' in matched.data_vars:
        del matched['lat']
    if 'lon' in matched.data_vars:
        del matched['lon']
    if 'alt' in matched.data_vars:
        del matched['alt']
        
    # Update the individual Variables to Hold Global Attributes
    # global attributes will be lost on merging into the matched dataset.
    # Need to keep as many references and descriptors as possible
    for var in matched.data_vars:
        matched[var].attrs.update(source=matched.datastream)
        
    # Merge the two DataSets
    for k in matched.data_vars:
        if k in column.data_vars:
            column[k].sel(station=site)[:] = matched.sel(station=site)[k][:].astype(column[k].dtype)
            if "_FillValue" in column[k].attrs:
                if isinstance(column[k].attrs["_FillValue"], str):
                    column[k].attrs["_FillValue"] = float(column[k].attrs["_FillValue"])
                column[k] = column[k].fillna(column[k].attrs["_FillValue"]).astype(float)
            if "missing_value" in column[k].attrs:
                if isinstance(column[k].attrs["missing_value"], str):
                    column[k].attrs["missing_value"] = float(column[k].attrs["missing_value"])
                column[k] = column[k].fillna(column[k].attrs["missing_value"]).astype(float)
    return column
