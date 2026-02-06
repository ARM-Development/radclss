import radclss
import arm_test_data
import os
import glob
import xarray as xr
import act
import numpy as np

from distributed import Client, LocalCluster


def test_radclss_serial():
    test_data_path = arm_test_data.DATASETS.abspath

    # Before testing, ensure that ARM credentials are set in environment variables
    username = os.getenv("ARM_USERNAME")
    token = os.getenv("ARM_PASSWORD")
    if not username or not token:
        return  # Skip test if credentials are not set

    act.discovery.download_arm_data(
        username,
        token,
        "bnfcsapr2cmacS3.c1",
        "2025-06-19T12:00:00",
        "2025-06-19T12:30:00",
        output=test_data_path,
    )

    act.discovery.download_arm_data(
        username,
        token,
        "bnfsondewnpnM1.b1",
        "2025-06-18T00:00:00",
        "2025-06-20T00:00:00",
        output=test_data_path,
    )
    for files in arm_test_data.DATASETS.registry.keys():
        if "bnf" in files:
            if not os.path.exists(os.path.join(test_data_path, files)):
                arm_test_data.DATASETS.fetch(files)

    rad_path = os.path.join(test_data_path, "*bnfcsapr2cmacS3.c1*.nc")
    radar_files = sorted(glob.glob(rad_path))
    sonde_files = sorted(
        glob.glob(os.path.join(test_data_path, "*bnfsondewnpnM1.b1*cdf"))
    )

    vd_M1_files = glob.glob(os.path.join(test_data_path, "*bnfvdisquantsM1.c1*nc"))
    met_M1_files = glob.glob(os.path.join(test_data_path, "*bnfmetM1.b1*"))
    met_S20_files = glob.glob(os.path.join(test_data_path, "*bnfmetS20.b1*"))
    met_S30_files = glob.glob(os.path.join(test_data_path, "*bnfmetS30.b1*"))
    met_S40_files = glob.glob(os.path.join(test_data_path, "*bnfmetS40.b1*"))
    wxt_S13_files = glob.glob(os.path.join(test_data_path, "*bnfmetwxtS13.b1*nc"))
    pluvio_M1_files = glob.glob(os.path.join(test_data_path, "*bnfwbpluvio2M1.a1*nc"))
    ld_M1_files = glob.glob(os.path.join(test_data_path, "*bnfldquantsM1.c1*nc"))
    ld_S30_files = glob.glob(os.path.join(test_data_path, "*bnfldquantsS30.c1*nc"))
    volumes = {
        "date": "20250619",
        "radar": radar_files,
        "sonde": sonde_files,
        "vd_M1": vd_M1_files,
        "met_M1": met_M1_files,
        "met_S20": met_S20_files,
        "met_S30": met_S30_files,
        "met_S40": met_S40_files,
        "wxt_S13": wxt_S13_files,
        "pluvio_M1": pluvio_M1_files,
        "ld_M1": ld_M1_files,
        "ld_S30": ld_S30_files,
    }

    input_site_dict = {
        "M1": (34.34525, -87.33842, 293),
        "S4": (34.46451, -87.23598, 197),
        "S20": (34.65401, -87.29264, 178),
        "S30": (34.38501, -86.92757, 183),
        "S40": (34.17932, -87.45349, 236),
        "S13": (34.343889, -87.350556, 286),
    }

    my_columns = radclss.core.radclss(
        volumes, input_site_dict, serial=True, verbose=False
    )
    assert isinstance(my_columns, xr.Dataset)
    assert "corrected_reflectivity" in my_columns.data_vars
    assert my_columns.dims["time"] == 6
    assert my_columns.dims["height"] == 32
    assert my_columns.dims["station"] == 6
    assert np.array_equal(
        my_columns["station"].values, ["M1", "S4", "S20", "S30", "S40", "S13"]
    )

    # Radar and sonde data check
    for site in my_columns["station"].values:
        missing_value = (
            my_columns["corrected_reflectivity"]
            .sel(station=site)
            .attrs.get("missing_value", None)
        )
        assert not (
            my_columns["corrected_reflectivity"].sel(station=site) == missing_value
        ).all()
        missing_value = (
            my_columns["attenuation_corrected_differential_reflectivity_lag_1"]
            .sel(station=site)
            .attrs.get("missing_value", None)
        )
        assert not (
            my_columns["attenuation_corrected_differential_reflectivity_lag_1"].sel(
                station=site
            )
            == missing_value
        ).all()
        missing_value = (
            my_columns["corrected_specific_diff_phase"]
            .sel(station=site)
            .attrs.get("missing_value", None)
        )
        assert not (
            my_columns["corrected_specific_diff_phase"].sel(station=site)
            == missing_value
        ).all()

    for site in ["M1", "S4", "S20", "S30", "S40"]:
        # Sonde data
        missing_value = (
            my_columns["sonde_u_wind"]
            .sel(station=site)
            .attrs.get("missing_value", None)
        )
        assert not (my_columns["sonde_u_wind"].sel(station=site) == missing_value).all()
        missing_value = (
            my_columns["sonde_v_wind"]
            .sel(station=site)
            .attrs.get("missing_value", None)
        )
        assert not (my_columns["sonde_v_wind"].sel(station=site) == missing_value).all()
        missing_value = (
            my_columns["sonde_tdry"].sel(station=site).attrs.get("missing_value", None)
        )
        assert not (my_columns["sonde_tdry"].sel(station=site) == missing_value).all()
        missing_value = (
            my_columns["sonde_rh"].sel(station=site).attrs.get("missing_value", None)
        )
        assert not (my_columns["sonde_rh"].sel(station=site) == missing_value).all()
        missing_value = (
            my_columns["sonde_pres"].sel(station=site).attrs.get("missing_value", None)
        )
        assert not (my_columns["sonde_pres"].sel(station=site) == missing_value).all()

    # Met data check
    for site in ["M1", "S20", "S30", "S40", "S13"]:
        missing_value = (
            my_columns["temp_mean"].sel(station=site).attrs.get("missing_value", None)
        )
        assert not (my_columns["temp_mean"].sel(station=site) == missing_value).all()

    for site in ["S4"]:
        missing_value = (
            my_columns["temp_mean"].sel(station=site).attrs.get("missing_value", None)
        )
        assert (my_columns["temp_mean"].sel(station=site) == missing_value).all()

    # Pluvio data check
    missing_value = (
        my_columns["accum_nrt"].sel(station="M1").attrs.get("missing_value", None)
    )
    assert not (my_columns["accum_nrt"].sel(station="M1") == missing_value).all()
    missing_value = (
        my_columns["bucket_nrt"].sel(station="M1").attrs.get("missing_value", None)
    )
    assert not (my_columns["bucket_nrt"].sel(station="M1") == missing_value).all()

    for site in ["S20", "S30", "S40", "S13", "S4"]:
        missing_value = (
            my_columns["accum_nrt"].sel(station=site).attrs.get("missing_value", None)
        )
        assert (my_columns["accum_nrt"].sel(station=site) == missing_value).all()
        missing_value = (
            my_columns["bucket_nrt"].sel(station=site).attrs.get("missing_value", None)
        )
        assert (my_columns["bucket_nrt"].sel(station=site) == missing_value).all()

    # LD data check
    for site in ["M1", "S30"]:
        missing_value = (
            my_columns["ldquants_rain_rate"]
            .sel(station=site)
            .attrs.get("missing_value", None)
        )
        assert not (
            my_columns["ldquants_rain_rate"].sel(station=site) == missing_value
        ).all()
        missing_value = (
            my_columns["ldquants_med_diameter"]
            .sel(station=site)
            .attrs.get("missing_value", None)
        )
        assert not (
            my_columns["ldquants_med_diameter"].sel(station=site) == missing_value
        ).all()


def test_radclss_parallel():
    test_data_path = arm_test_data.DATASETS.abspath

    # Before testing, ensure that ARM credentials are set in environment variables
    username = os.getenv("ARM_USERNAME")
    token = os.getenv("ARM_PASSWORD")
    if not username or not token:
        return  # Skip test if credentials are not set

    act.discovery.download_arm_data(
        username,
        token,
        "bnfcsapr2cmacS3.c1",
        "2025-06-19T12:00:00",
        "2025-06-19T12:30:00",
        output=test_data_path,
    )

    act.discovery.download_arm_data(
        username,
        token,
        "bnfsondewnpnM1.b1",
        "2025-06-18T00:00:00",
        "2025-06-20T00:00:00",
        output=test_data_path,
    )
    for files in arm_test_data.DATASETS.registry.keys():
        if "bnf" in files:
            if not os.path.exists(os.path.join(test_data_path, files)):
                arm_test_data.DATASETS.fetch(files)

    rad_path = os.path.join(test_data_path, "*bnfcsapr2cmacS3.c1*.nc")
    radar_files = sorted(glob.glob(rad_path))
    sonde_files = sorted(
        glob.glob(os.path.join(test_data_path, "*bnfsondewnpnM1.b1*cdf"))
    )
    print(sonde_files)
    vd_M1_files = glob.glob(os.path.join(test_data_path, "*bnfvdisquantsM1.c1*nc"))
    met_M1_files = glob.glob(os.path.join(test_data_path, "*bnfmetM1.b1*"))
    met_S20_files = glob.glob(os.path.join(test_data_path, "*bnfmetS20.b1*"))
    met_S30_files = glob.glob(os.path.join(test_data_path, "*bnfmetS30.b1*"))
    met_S40_files = glob.glob(os.path.join(test_data_path, "*bnfmetS40.b1*"))
    wxt_S13_files = glob.glob(os.path.join(test_data_path, "*bnfmetwxtS13.b1*nc"))
    pluvio_M1_files = glob.glob(os.path.join(test_data_path, "*bnfwbpluvio2M1.a1*nc"))
    ld_M1_files = glob.glob(os.path.join(test_data_path, "*bnfldquantsM1.c1*nc"))
    ld_S30_files = glob.glob(os.path.join(test_data_path, "*bnfldquantsS30.c1*nc"))
    volumes = {
        "date": "20250619",
        "radar": radar_files,
        "sonde": sonde_files,
        "vd_M1": vd_M1_files,
        "met_M1": met_M1_files,
        "met_S20": met_S20_files,
        "met_S30": met_S30_files,
        "met_S40": met_S40_files,
        "wxt_S13": wxt_S13_files,
        "pluvio_M1": pluvio_M1_files,
        "ld_M1": ld_M1_files,
        "ld_S30": ld_S30_files,
    }

    input_site_dict = {
        "M1": (34.34525, -87.33842, 293),
        "S4": (34.46451, -87.23598, 197),
        "S20": (34.65401, -87.29264, 178),
        "S30": (34.38501, -86.92757, 183),
        "S40": (34.17932, -87.45349, 236),
        "S13": (34.343889, -87.350556, 286),
    }
    with Client(LocalCluster(n_workers=2, threads_per_worker=1)) as client:  # noqa
        my_columns = radclss.core.radclss(
            volumes, input_site_dict, serial=False, verbose=False
        )
    assert isinstance(my_columns, xr.Dataset)
    assert "corrected_reflectivity" in my_columns.data_vars
    assert my_columns.dims["time"] == 6
    assert my_columns.dims["height"] == 32
    assert my_columns.dims["station"] == 6
    assert np.array_equal(
        my_columns["station"].values, ["M1", "S4", "S20", "S30", "S40", "S13"]
    )

    # Radar and sonde data check
    for site in my_columns["station"].values:
        missing_value = (
            my_columns["corrected_reflectivity"]
            .sel(station=site)
            .attrs.get("missing_value", None)
        )
        assert not (
            my_columns["corrected_reflectivity"].sel(station=site) == missing_value
        ).all()
        missing_value = (
            my_columns["attenuation_corrected_differential_reflectivity_lag_1"]
            .sel(station=site)
            .attrs.get("missing_value", None)
        )
        assert not (
            my_columns["attenuation_corrected_differential_reflectivity_lag_1"].sel(
                station=site
            )
            == missing_value
        ).all()
        missing_value = (
            my_columns["corrected_specific_diff_phase"]
            .sel(station=site)
            .attrs.get("missing_value", None)
        )
        assert not (
            my_columns["corrected_specific_diff_phase"].sel(station=site)
            == missing_value
        ).all()

    for site in ["M1", "S4", "S20", "S30", "S40"]:
        # Sonde data
        missing_value = (
            my_columns["sonde_u_wind"]
            .sel(station=site)
            .attrs.get("missing_value", None)
        )
        assert not (my_columns["sonde_u_wind"].sel(station=site) == missing_value).all()
        missing_value = (
            my_columns["sonde_v_wind"]
            .sel(station=site)
            .attrs.get("missing_value", None)
        )
        assert not (my_columns["sonde_v_wind"].sel(station=site) == missing_value).all()
        missing_value = (
            my_columns["sonde_tdry"].sel(station=site).attrs.get("missing_value", None)
        )
        assert not (my_columns["sonde_tdry"].sel(station=site) == missing_value).all()
        missing_value = (
            my_columns["sonde_rh"].sel(station=site).attrs.get("missing_value", None)
        )
        assert not (my_columns["sonde_rh"].sel(station=site) == missing_value).all()
        missing_value = (
            my_columns["sonde_pres"].sel(station=site).attrs.get("missing_value", None)
        )
        assert not (my_columns["sonde_pres"].sel(station=site) == missing_value).all()

    # Met data check
    for site in ["M1", "S20", "S30", "S40", "S13"]:
        missing_value = (
            my_columns["temp_mean"].sel(station=site).attrs.get("missing_value", None)
        )
        assert not (my_columns["temp_mean"].sel(station=site) == missing_value).all()

    for site in ["S4"]:
        missing_value = (
            my_columns["temp_mean"].sel(station=site).attrs.get("missing_value", None)
        )
        assert (my_columns["temp_mean"].sel(station=site) == missing_value).all()

    # Pluvio data check
    missing_value = (
        my_columns["accum_nrt"].sel(station="M1").attrs.get("missing_value", None)
    )
    assert not (my_columns["accum_nrt"].sel(station="M1") == missing_value).all()
    missing_value = (
        my_columns["bucket_nrt"].sel(station="M1").attrs.get("missing_value", None)
    )
    assert not (my_columns["bucket_nrt"].sel(station="M1") == missing_value).all()

    for site in ["S20", "S30", "S40", "S13", "S4"]:
        missing_value = (
            my_columns["accum_nrt"].sel(station=site).attrs.get("missing_value", None)
        )
        assert (my_columns["accum_nrt"].sel(station=site) == missing_value).all()
        missing_value = (
            my_columns["bucket_nrt"].sel(station=site).attrs.get("missing_value", None)
        )
        assert (my_columns["bucket_nrt"].sel(station=site) == missing_value).all()

    # LD data check
    for site in ["M1", "S30"]:
        missing_value = (
            my_columns["ldquants_rain_rate"]
            .sel(station=site)
            .attrs.get("missing_value", None)
        )
        assert not (
            my_columns["ldquants_rain_rate"].sel(station=site) == missing_value
        ).all()
        missing_value = (
            my_columns["ldquants_med_diameter"]
            .sel(station=site)
            .attrs.get("missing_value", None)
        )
        assert not (
            my_columns["ldquants_med_diameter"].sel(station=site) == missing_value
        ).all()


def test_subset_points():
    test_data_path = arm_test_data.DATASETS.abspath
    # Before testing, ensure that ARM credentials are set in environment variables
    username = os.getenv("ARM_USERNAME")
    token = os.getenv("ARM_PASSWORD")
    if not username or not token:
        return  # Skip test if credentials are not set

    act.discovery.download_arm_data(
        username,
        token,
        "bnfcsapr2cmacS3.c1",
        "2025-06-19T12:00:00",
        "2025-06-19T12:30:00",
        output=test_data_path,
    )

    act.discovery.download_arm_data(
        username,
        token,
        "bnfsondewnpnM1.b1",
        "2025-06-18T00:00:00",
        "2025-06-20T00:00:00",
        output=test_data_path,
    )

    rad_path = os.path.join(test_data_path, "*bnfcsapr2cmacS3.c1*.nc")
    radar_files = sorted(glob.glob(rad_path))
    input_site_dict = {
        "M1": (34.34525, -87.33842, 293),
        "S30": (34.38501, -86.92757, 183),
    }
    subset_ds = radclss.util.subset_points(radar_files[0], input_site_dict, sonde=None)
    assert set(subset_ds["station"].values) == {"M1", "S30"}
    assert "corrected_reflectivity" in subset_ds.data_vars
    assert subset_ds.dims["station"] == 2
    assert np.array_equal(subset_ds["height"].values, np.arange(500, 8500, 250))
    assert "sonde_u_wind" not in subset_ds.data_vars
    assert "sonde_v_wind" not in subset_ds.data_vars
    assert "sonde_tdry" not in subset_ds.data_vars
    assert "sonde_rh" not in subset_ds.data_vars

    # Test with rawinsonde input instead of sonde=False
    sonde_files = sorted(
        glob.glob(os.path.join(test_data_path, "*bnfsondewnpnM1.b1*cdf"))
    )

    subset_ds = radclss.util.subset_points(
        radar_files[0], input_site_dict, sonde=sonde_files
    )
    assert set(subset_ds["station"].values) == {"M1", "S30"}
    assert np.array_equal(subset_ds["height"].values, np.arange(500, 8500, 250))
    assert "corrected_reflectivity" in subset_ds.data_vars
    assert "sonde_u_wind" in subset_ds.data_vars
    assert "sonde_v_wind" in subset_ds.data_vars
    assert "sonde_tdry" in subset_ds.data_vars
    assert "sonde_rh" in subset_ds.data_vars
    assert "sonde_u_wind" in subset_ds.data_vars
    assert "sonde_v_wind" in subset_ds.data_vars
    assert "sonde_tdry" in subset_ds.data_vars
    assert "sonde_rh" in subset_ds.data_vars


def test_match_datasets_act():
    test_data_path = arm_test_data.DATASETS.abspath
    radclss_file = arm_test_data.DATASETS.fetch(
        "bnfcsapr2radclss.c2.20250619.000000.nc"
    )
    for files in arm_test_data.DATASETS.registry.keys():
        if "bnf" in files:
            if not os.path.exists(os.path.join(test_data_path, files)):
                arm_test_data.DATASETS.fetch(files)

    met_M1_files = glob.glob(os.path.join(test_data_path, "*bnfmetM1.b1*"))
    print(met_M1_files)
    radclss_ds = xr.open_dataset(radclss_file)
    radclss_ds = radclss_ds.drop_vars(
        [var for var in radclss_ds.data_vars if "temp" in var]
    )  # Remove temp variables for testing
    matched_ds_mean = radclss.util.match_datasets_act(
        radclss_ds,
        met_M1_files,
        "M1",
        resample="mean",
        discard=radclss.config.DEFAULT_DISCARD_VAR["met"],
    )
    matched_ds_skip = radclss.util.match_datasets_act(
        radclss_ds,
        met_M1_files,
        "M1",
        resample="skip",
        discard=radclss.config.DEFAULT_DISCARD_VAR["met"],
    )
    matched_ds_sum = radclss.util.match_datasets_act(
        radclss_ds,
        met_M1_files,
        "M1",
        resample="sum",
        discard=radclss.config.DEFAULT_DISCARD_VAR["met"],
    )

    assert not np.array_equal(
        matched_ds_mean["rh_mean"].values, matched_ds_skip["rh_mean"].values
    )
    assert not np.array_equal(
        matched_ds_mean["rh_mean"].values, matched_ds_sum["rh_mean"].values
    )
    assert not np.array_equal(
        matched_ds_skip["rh_mean"].values, matched_ds_sum["rh_mean"].values
    )
    assert matched_ds_mean.dims["time"] == radclss_ds.dims["time"]
    assert matched_ds_skip.dims["time"] == radclss_ds.dims["time"]
    assert matched_ds_sum.dims["time"] == radclss_ds.dims["time"]
