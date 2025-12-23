OUTPUT_SITE = "BNF"
OUTPUT_FACILITY = "S2"
OUTPUT_PLATFORM = "csapr2radclss"
OUTPUT_LEVEL = "c2"
OUTPUT_GATE_TIME_ATTRS = {'long_name': ('Time in Seconds that Cooresponds to the Start'
                                        + " of each Individual Radar Volume Scan before"
                                        + " Concatenation"),
                            'description': 'Time in Seconds that Cooresponds to the Minimum Height Gate'}
OUTPUT_TIME_OFFSET_ATTRS = dict(long_name=("Time in Seconds Since Midnight"),
                                            description=("Time in Seconds Since Midnight that Cooresponds"
                                                        + "to the Center of Each Height Gate"
                                                        + "Above the Target Location "))
OUTPUT_STATION_ATTRS = dict(long_name="Bankhead National Forest AMF-3 In-Situ Ground Observation Station Identifers")
OUTPUT_LAT_ATTRS = dict(long_name=f'Latitude of BNF AMF-3 Ground Observation Site', units='Degrees North')
OUTPUT_LON_ATTRS = dict(long_name='Longitude of BNF AMF-3 Ground Observation Site', units='Degrees East')
OUTPUT_ALT_ATTRS = dict(long_name="Altitude above mean sea level for each station", units="m")

def set_output_site(site):
    """
    Set the output site for RadCLss files.

    Parameters
    ----------
    site : str
        The output site name.
    """
    global OUTPUT_SITE
    OUTPUT_SITE = site

def set_output_facility(facility):
    """
    Set the output facility for RadCLss files.

    Parameters
    ----------
    facility : str
        The output facility name.
    """
    global OUTPUT_FACILITY
    OUTPUT_FACILITY = facility

def set_output_platform(platform):
    """
    Set the output platform for RadCLss files.

    Parameters
    ----------
    platform : str
        The output platform name.
    """
    global OUTPUT_PLATFORM
    OUTPUT_PLATFORM = platform

def set_output_level(level):
    """
    Set the output level for RadCLss files.

    Parameters
    ----------
    level : str
        The output level name.
    """
    global OUTPUT_LEVEL
    OUTPUT_LEVEL = level

def set_output_gate_time_attrs(attrs):
    """
    Set the attributes for the gate_time variable.  

    Parameters
    ----------
    attrs : dict
        Dictionary of attributes for gate_time.
    """
    global OUTPUT_GATE_TIME_ATTRS
    OUTPUT_GATE_TIME_ATTRS = attrs  

def set_output_time_offset_attrs(attrs):
    """
    Set the attributes for the time_offset variable.    

    Parameters
    ----------
    attrs : dict
        Dictionary of attributes for time_offset.
    """
    global OUTPUT_TIME_OFFSET_ATTRS
    OUTPUT_TIME_OFFSET_ATTRS = attrs

def set_output_station_attrs(attrs):
    """
    Set the attributes for the station variable.
    
    Parameters
    ----------
    attrs : dict
        Dictionary of attributes for station.
    """
    global OUTPUT_STATION_ATTRS
    OUTPUT_STATION_ATTRS = attrs    

def set_output_lat_attrs(attrs):
    """
    Set the attributes for the lat variable.

    Parameters
    ----------
    attrs : dict
        Dictionary of attributes for lat.
    """
    global OUTPUT_LAT_ATTRS
    OUTPUT_LAT_ATTRS = attrs

def set_output_lon_attrs(attrs):
    """
    Set the attributes for the lon variable.
    
    Parameters
    ----------
    attrs : dict
        Dictionary of attributes for lon.
    """
    global OUTPUT_LON_ATTRS
    OUTPUT_LON_ATTRS = attrs    

def set_output_alt_attrs(attrs):
    """
    Set the attributes for the alt variable.

    Parameters
    ----------
    attrs : dict
        Dictionary of attributes for alt.
    """
    global OUTPUT_ALT_ATTRS
    OUTPUT_ALT_ATTRS = attrs   

def get_output_config():
    """
    Get the current output configuration as a dictionary.

    Returns
    -------
    dict
        Dictionary containing the current output configuration.
    """
    return {
        "site": OUTPUT_SITE,
        "facility": OUTPUT_FACILITY,
        "platform": OUTPUT_PLATFORM,
        "level": OUTPUT_LEVEL,
        "gate_time_attrs": OUTPUT_GATE_TIME_ATTRS,
        "time_offset_attrs": OUTPUT_TIME_OFFSET_ATTRS,
        "station_attrs": OUTPUT_STATION_ATTRS,
        "lat_attrs": OUTPUT_LAT_ATTRS,
        "lon_attrs": OUTPUT_LON_ATTRS,
        "alt_attrs": OUTPUT_ALT_ATTRS
    } 

