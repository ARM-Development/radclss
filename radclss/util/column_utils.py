def subset_points(nfile, **kwargs):
    """
    Subset a radar file for a set of latitudes and longitudes
    utilizing Py-ART's column-vertical-profile functionality.

    Parameters
    ----------
    file : str
        Path to the radar file to extract columns from
    nsonde : list
        List containing file paths to the desired sonde file to merge

    Calls
    -----
    radar_start_time
    merge_sonde

    Returns
    -------
    ds : xarray DataSet
        Xarray Dataset containing the radar column above a give set of locations
    
    """
    ds = None
    
    # Define the splash locations [lon,lat]
    M1 = [34.34525, -87.33842]
    S4 = [34.46451,	-87.23598]
    S20 = [34.65401, -87.29264]
    S30	= [34.38501, -86.92757]
    S40	= [34.17932, -87.45349]
    S13 = [34.343889, -87.350556]

    sites    = ["M1", "S4", "S20", "S30", "S40", "S13"]
    site_alt = [293, 197, 178, 183, 236, 286]

    # Zip these together!
    lats, lons = list(zip(M1,
                          S4,
                          S20,
                          S30,
                          S40,
                          S13))
    try:
        # Read in the file
        radar = pyart.io.read(nfile)
        # Check for single sweep scans
        if np.ma.is_masked(radar.sweep_start_ray_index["data"][1:]):
            radar.sweep_start_ray_index["data"] = np.ma.array([0])
            radar.sweep_end_ray_index["data"] = np.ma.array([radar.nrays])
    except:
        radar = None

    if radar:
        if radar.scan_type != "rhi":
            if radar.time['data'].size > 0:
                # Easier to map the nearest sonde file to radar gates before extraction
                if 'sonde' in kwargs:
                    # variables to discard when reading in the sonde file
                    exclude_sonde = ['base_time', 'time_offset', 'lat', 'lon', 'qc_pres',
                                    'qc_tdry', 'qc_dp', 'qc_wspd', 'qc_deg', 'qc_rh',
                                    'qc_u_wind', 'qc_v_wind', 'qc_asc']
        
                    # find the nearest sonde file to the radar start time
                    radar_start = datetime.datetime.strptime(nfile.split('/')[-1].split('.')[-3] + '.' + nfile.split('/')[-1].split('.')[-2], 
                                                            '%Y%m%d.%H%M%S'
                    )
                    sonde_start = [datetime.datetime.strptime(xfile.split('/')[-1].split('.')[2] + 
                                                              '-' + 
                                                              xfile.split('/')[-1].split('.')[3], 
                                                              '%Y%m%d-%H%M%S') for xfile in kwargs['sonde']
                    ]
                    # difference in time between radar file and each sonde file
                    start_diff = [radar_start - sonde for sonde in sonde_start]

                    # merge the sonde file into the radar object
                    ds_sonde = act.io.read_arm_netcdf(kwargs['sonde'][start_diff.index(min(start_diff))], 
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
                    try:
                        da = (
                            pyart.util.columnsect.column_vertical_profile(radar, lat, lon)
                            .interp(height=np.arange(500, 8500, 250))
                        )
                    except ValueError:
                        da = pyart.util.columnsect.column_vertical_profile(radar, lat, lon)
                        # check for valid heights
                        valid = np.isfinite(da["height"])
                        n_valid = int(valid.sum())
                        if n_valid > 0:
                            da = da.sel(height=valid).sortby("height").interp(height=np.arange(500, 8500, 250))
                        else:
                            target_height = xr.DataArray(np.arange(500, 8500, 250), dims="height", name="height")
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
                ds.gate_time.attrs.update(long_name=('Time in Seconds that Cooresponds to the Start'
                                                    + " of each Individual Radar Volume Scan before"
                                                    + " Concatenation"),
                                          description=('Time in Seconds that Cooresponds to the Minimum'
                                                    + ' Height Gate'))
                ds.time_offset.attrs.update(long_name=("Time in Seconds Since Midnight"),
                                            description=("Time in Seconds Since Midnight that Cooresponds"
                                                        + "to the Center of Each Height Gate"
                                                        + "Above the Target Location ")
                                            )
                ds.station.attrs.update(long_name="Bankhead National Forest AMF-3 In-Situ Ground Observation Station Identifers")
                ds.lat.attrs.update(long_name='Latitude of BNF AMF-3 Ground Observation Site',
                                         units='Degrees North')
                ds.lon.attrs.update(long_name='Longitude of BNF AMF-3 Ground Observation Site',
                                          units='Degrees East')
                ds.alt.attrs.update(long_name="Altitude above mean sea level for each station",
                                          units="m")
                # delete the radar to free up memory
                del radar, column_list, da
            else:
                # delete the rhi file
                del radar
        else:
            del radar
    return ds

def match_datasets_act(column, 
                       ground, 
                       site, 
                       discard, 
                       resample='sum', 
                       DataSet=False,
                       prefix=None):
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
    column = xr.merge([column, matched])
   
    return column
