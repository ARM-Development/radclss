OUTPUT_SITE = "BNF"
OUTPUT_FACILITY = "S2"
OUTPUT_PLATFORM = "csapr2radclss"
OUTPUT_LEVEL = "c2"

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