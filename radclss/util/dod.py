import numpy as np
import xarray as xr
import datetime
import getpass
import socket 

from ..config import get_output_config

def adjust_radclss_dod(radclss, dod):
    """
    Adjust the RadCLss DataSet to include missing datastreams

    Parameters
    ----------
    radclss : Xarray DataSet
        extracted columns and in-situ sensor file
    dod : Xarray DataSet
        expected datastreams and data standards for RadCLss

    returns
    -------
    radclss : Xarray DataSet
        Corrected RadCLss that has all expected parmeters and attributes
    """
    # Supplied DOD has correct data attributes and all required parameters. 
    # Update the RadCLss dataset variable values with the DOD dataset. 
    for var in dod.variables:
        # Make sure the variable isn't a dimension
        if var not in dod.dims:
            # check to see if variable is within RadCLss
            # note: it may not be if file is missing.
            if var not in radclss.variables:
                new_size = []
                for dim in dod[var].dims:
                    if dim == "time":
                        new_size.append(radclss.sizes['time'])
                    else:
                        new_size.append(dod.sizes[dim])
                    #new_data = np.full(())
                # create a new array to hold the updated values
                new_data = np.full(new_size, dod[var].data[0])
                # create a new DataArray and add back into RadCLss
                new_array = xr.DataArray(new_data, dims=dod[var].dims)
                new_array.attrs = dod[var].attrs
                radclss[var] = new_array
                
                # clean up some saved values
                del new_size, new_data, new_array

    # Adjust the radclss time attributes
    if hasattr(radclss['time'], "units"):
        del radclss["time"].attrs["units"]
    if hasattr(radclss['time_offset'], "units"):
        del radclss["time_offset"].attrs["units"]
    if hasattr(radclss['base_time'], "units"):
        del radclss["base_time"].attrs["units"]

    # reorder the DataArrays to match the ARM Data Object Identifier 
    first = ["base_time", "time_offset", "time", "height", "station", "gate_time"]           
    last  = ["lat", "lon", "alt"]   # the three you want last

    # Keep only data variables, preserve order, and drop the ones already in first/last
    middle = [v for v in radclss.data_vars if v not in first + last]

    ordered = first + middle + last
    radclss = radclss[ordered]

    # Update the global attributes (some) to match the DOD

    config = get_output_config()
    OUTPUT_SITE = config['site']
    OUTPUT_FACILITY = config['facility']
    OUTPUT_PLATFORM = config['platform']
    OUTPUT_LEVEL = config['level']
    radclss.drop_attrs()
    radclss.attrs = dod.attrs
    radclss.attrs['vap_name'] = ""
    radclss.attrs['command_line'] = "python bnf_radclss.py --serial True --array True"
    radclss.attrs['site_id'] = OUTPUT_SITE
    radclss.attrs['platform_id'] = OUTPUT_PLATFORM
    radclss.attrs['facility_id'] = OUTPUT_FACILITY
    radclss.attrs['data_level'] = OUTPUT_LEVEL
    radclss.attrs['location_description'] = "Southeast U.S. in Bankhead National Forest (BNF), Decatur, Alabama"
    radclss.attrs['datastream'] = "bnfcsapr2radclssS3.c2"
    radclss.attrs['input_datastreams'] = ["bnfcsapr2cmacS3.c1",
                                          'bnfmetM1.b1',
                                          'bnfmetS20.b1',
                                          "bnfmetS30.b1",
                                          "bnfmetS40.b1",
                                          "bnfsondewnpnM1.b1",
                                          "bnfwbpluvio2M1.a1",
                                          "bnfldquantsM1.c1",
                                          "bnfldquantsS30.c1",
                                          "bnfvdisquantsM1.c1",
                                          "bnfmetwxtS13.b1"
    ]
    username = getpass.getuser()
    hostname = socket.gethostname()
    radclss.attrs['history'] = ("created by user " + username + " on machine " + hostname + " at " +
                                 str(datetime.datetime.now()) +
                                " using Py-ART and ACT"
    )
    # return radclss, close DOD file
    del dod

    return radclss