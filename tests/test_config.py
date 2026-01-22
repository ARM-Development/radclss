import radclss


def test_radclss_set_discarded_variables():
    # Define a sample variable list to discard
    sample_vars = [
        "reflectivity",
        "specific_attenuation",
    ]

    # Set discarded variables for radar
    radclss.config.set_discarded_variables("radar", sample_vars)

    # Verify that the variables are set correctly
    discarded_radar_vars = radclss.config.DEFAULT_DISCARD_VAR["radar"]
    assert "reflectivity" in discarded_radar_vars
    assert "specific_attenuation" in discarded_radar_vars

    # Set discarded variables for met
    sample_vars = [
        "qc_temp_mean",
        "lat",
    ]
    radclss.config.set_discarded_variables("met", sample_vars)

    # Verify that the variables are set correctly
    discarded_met_vars = radclss.config.DEFAULT_DISCARD_VAR["met"]
    assert "qc_temp_mean" in discarded_met_vars
    assert "lat" in discarded_met_vars
    assert "reflectivity" not in discarded_met_vars  # Should not be in met


def test_set_output_site():
    site = "bnf"
    radclss.config.set_output_site(site)
    config = radclss.config.get_output_config()
    assert config["site"] == site


def test_set_output_facility():
    facility = "S3"
    radclss.config.set_output_facility(facility)
    config = radclss.config.get_output_config()
    assert config["facility"] == facility


def test_set_output_platform():
    platform = "csapr2radclss"
    radclss.config.set_output_platform(platform)
    config = radclss.config.get_output_config()
    assert config["platform"] == platform


def test_set_output_level():
    level = "c2"
    radclss.config.set_output_level(level)
    config = radclss.config.get_output_config()
    assert config["level"] == level


def test_set_output_gate_time_attrs():
    attrs = {
        "long_name": "Gate Time",
        "units": "seconds since 1970-01-01 00:00:00",
        "standard_name": "time",
    }
    radclss.config.set_output_gate_time_attrs(attrs)
    config = radclss.config.get_output_config()
    assert config["gate_time_attrs"] == attrs


def test_set_output_time_offset_attrs():
    attrs = {
        "long_name": "Time Offset",
        "units": "seconds since gate_time",
    }
    radclss.config.set_output_time_offset_attrs(attrs)
    config = radclss.config.get_output_config()
    assert config["time_offset_attrs"] == attrs


def test_get_output_config():
    config = radclss.config.get_output_config()
    assert isinstance(config, dict)
    # Check for default keys
    assert "site" in config
    assert "facility" in config
    assert "platform" in config
    assert "level" in config
    assert "gate_time_attrs" in config
    assert "time_offset_attrs" in config

    # Check default values
    assert config["site"] == "bnf"
    assert config["facility"] == "S3"
    assert config["platform"] == "csapr2radclss"
    assert config["level"] == "c2"


def test_set_output_station_attrs():
    attrs = {
        "long_name": "Station Name",
        "description": "Name of the observation station",
    }
    radclss.config.set_output_station_attrs(attrs)
    config = radclss.config.get_output_config()
    assert config["station_attrs"] == attrs


def test_set_output_alt_attrs():
    attrs = {
        "long_name": "Altitude",
        "units": "meters",
        "standard_name": "altitude",
    }
    radclss.config.set_output_alt_attrs(attrs)
    config = radclss.config.get_output_config()
    assert config["alt_attrs"] == attrs


def test_set_output_lat_lon_attrs():
    lat_attrs = {
        "long_name": "Latitude",
        "units": "degrees_north",
        "standard_name": "latitude",
    }
    lon_attrs = {
        "long_name": "Longitude",
        "units": "degrees_east",
        "standard_name": "longitude",
    }
    radclss.config.set_output_lat_attrs(lat_attrs)
    radclss.config.set_output_lon_attrs(lon_attrs)
    config = radclss.config.get_output_config()
    assert config["lat_attrs"] == lat_attrs
    assert config["lon_attrs"] == lon_attrs
