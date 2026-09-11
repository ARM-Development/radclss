from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from radclss.util.column_utils import (
    _accumulate_to_grid,
    _column_time_step,
    get_nexrad_column,
)


def test_get_nexrad_column():
    """
    Test get_nexrad_column function to verify it outputs columns of reflectivity
    data over the specified input sites.

    This test mocks the S3 and PyART calls to avoid network dependencies.
    """
    # Define input site dictionary with lat, lon, alt
    input_site_dict = {
        "M1": (34.34525, -87.33842, 293),
        "S4": (34.46451, -87.23598, 197),
        "S20": (34.65401, -87.29264, 178),
        "S30": (34.38501, -86.92757, 183),
        "S40": (34.17932, -87.45349, 236),
        "S13": (34.343889, -87.350556, 286),
    }

    # Test parameters
    site = "BNF"
    rad_time = "2025-06-19T00:00:00"
    height_bins = np.arange(500, 8500, 250)
    nexrad_radar = "KHTX"

    # Create mock radar object with expected structure
    mock_radar = MagicMock()

    # Create mock column data that would be returned by column_vertical_profile
    mock_heights = np.arange(500, 8500, 100)
    n_heights = len(mock_heights)

    # Mock DataArray for a single column
    mock_column = xr.Dataset(
        {
            "corrected_reflectivity": (
                ["height"],
                np.random.randn(n_heights) * 10 + 30,
            ),
            "height": (["height"], mock_heights),
            "time_offset": (["height"], np.zeros(n_heights)),
            "base_time": np.datetime64("2025-06-19T00:00:00"),
        }
    )
    mock_column = mock_column.set_coords(["height", "base_time"])

    # Mock S3 client and responses
    with patch("radclss.util.column_utils.boto3.client") as mock_boto3:
        mock_s3_client = MagicMock()
        mock_boto3.return_value = mock_s3_client

        # Mock S3 list_objects_v2 response
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "2025/06/19/KHTX/KHTX20250619_000000_V06"},
                {"Key": "2025/06/19/KHTX/KHTX20250619_010000_V06"},
            ]
        }

        # Mock PyART read_nexrad_archive
        with patch(
            "radclss.util.column_utils.pyart.io.read_nexrad_archive"
        ) as mock_read:
            mock_read.return_value = mock_radar

            # Mock PyART column_vertical_profile to return our mock column
            with patch(
                "radclss.util.column_utils.pyart.util.columnsect.column_vertical_profile"
            ) as mock_cvp:
                mock_cvp.return_value = mock_column

                # Call the function
                result = get_nexrad_column(
                    rad_time=rad_time,
                    site=site,
                    input_site_dict=input_site_dict,
                    height_bins=height_bins,
                    nexrad_radar=nexrad_radar,
                )

    # Assertions
    assert result is not None, "Function should return a dataset"
    assert isinstance(result, xr.Dataset), "Result should be an xarray Dataset"

    # Check dimensions
    assert "station" in result.dims, "Result should have 'station' dimension"
    assert "height" in result.dims, "Result should have 'height' dimension"

    # Check that we have the correct number of stations
    assert result.dims["station"] == len(
        input_site_dict
    ), f"Should have {len(input_site_dict)} stations"

    # Check that height bins match
    assert result.dims["height"] == len(
        height_bins
    ), f"Should have {len(height_bins)} height levels"

    # Check that reflectivity data exists
    assert (
        "corrected_reflectivity" in result.data_vars
    ), "Result should contain reflectivity data"

    # Check that coordinate variables exist
    assert (
        "lat" in result.data_vars or "lat" in result.coords
    ), "Result should contain latitude data"
    assert (
        "lon" in result.data_vars or "lon" in result.coords
    ), "Result should contain longitude data"

    # Check that gate_time exists
    assert "gate_time" in result.data_vars, "Result should contain gate_time"

    # Check data types
    assert result["corrected_reflectivity"].dtype in [
        np.float32,
        np.float64,
    ], "Reflectivity should be float type"


def test_get_nexrad_column_integration():
    """
    Integration test for get_nexrad_column using real S3 data.
    """
    input_site_dict = {
        "M1": (34.34525, -87.33842, 293),
        "S4": (34.46451, -87.23598, 197),
        "S20": (34.65401, -87.29264, 178),
        "S30": (34.38501, -86.92757, 183),
        "S40": (34.17932, -87.45349, 236),
        "S13": (34.343889, -87.350556, 286),
    }

    site = "BNF"
    rad_time = "2025-06-19T00:00:00"
    height_bins = np.arange(500, 8500, 250)

    # This would actually download from S3
    result = get_nexrad_column(
        rad_time=rad_time,
        site=site,
        input_site_dict=input_site_dict,
        height_bins=height_bins,
    )

    # Verify structure
    assert result is not None
    assert isinstance(result, xr.Dataset)
    assert "station" in result.dims
    assert "height" in result.dims
    assert result.dims["station"] == len(input_site_dict)
    assert "reflectivity" in result.data_vars


def _synthetic_gauge(n_minutes=120):
    """A 1-minute gauge record with a rain burst, a dry spell and a data gap."""
    time = pd.date_range("2025-06-19T00:01", periods=n_minutes, freq="1min")
    accum = np.zeros(n_minutes)
    accum[20:60] = np.linspace(0.05, 0.9, 40)  # burst
    accum[70:80] = 0.2  # second, flatter burst
    accum[90:100] = np.nan  # instrument gap
    return xr.Dataset(
        {"accum_nrt": ("time", accum, {"units": "mm", "long_name": "Accum"})},
        coords={"time": time},
    )


@pytest.mark.parametrize("freq", ["1min", "5min", "15min"])
def test_accumulate_to_grid_conserves_total_on_regular_grids(freq):
    """
    Re-binning an accumulation must move rain between time steps, never create
    it. Summing into fixed bins and interpolating back onto a finer grid used to
    inflate the daily total by the ratio of the two steps.
    """
    gauge = _synthetic_gauge()
    column_time = xr.DataArray(
        pd.date_range(gauge.time.values[0], gauge.time.values[-1], freq=freq),
        dims="time",
        name="time",
    )

    regridded = _accumulate_to_grid(gauge, column_time, "5Min")

    assert np.isclose(
        np.nansum(regridded["accum_nrt"].values),
        np.nansum(gauge["accum_nrt"].values),
    )


def test_accumulate_to_grid_is_identity_at_native_resolution():
    """On a grid matching the source, the values must come back untouched."""
    gauge = _synthetic_gauge()

    regridded = _accumulate_to_grid(gauge, gauge.time, "5Min")

    np.testing.assert_allclose(
        np.nan_to_num(regridded["accum_nrt"].values),
        np.nan_to_num(gauge["accum_nrt"].values),
        atol=1e-9,
    )
    assert regridded["accum_nrt"].attrs["units"] == "mm"


def test_accumulate_to_grid_conserves_total_on_irregular_grid():
    """Radar-based time coordinates are irregular; the integral must survive."""
    gauge = _synthetic_gauge()
    rng = np.random.default_rng(0)
    picks = np.sort(rng.choice(np.arange(1, gauge.sizes["time"]), 40, replace=False))
    column_time = gauge.time.isel(time=picks)

    regridded = _accumulate_to_grid(gauge, column_time, "5Min")

    assert np.isclose(
        np.nansum(regridded["accum_nrt"].values),
        np.nansum(gauge["accum_nrt"].values),
    )


def test_accumulate_to_grid_keeps_gaps_missing():
    """A stretch with no valid samples must stay missing, not report zero rain."""
    gauge = _synthetic_gauge()
    column_time = xr.DataArray(
        pd.date_range(gauge.time.values[0], gauge.time.values[-1], freq="5min"),
        dims="time",
        name="time",
    )

    regridded = _accumulate_to_grid(gauge, column_time, "5Min")

    assert np.isnan(regridded["accum_nrt"].values).any()


def test_column_time_step_falls_back_when_unmeasurable():
    """Degenerate grids fall back to the supplied default rather than raising."""
    single = xr.DataArray(pd.to_datetime(["2025-06-19T00:00"]), dims="time")
    duplicated = xr.DataArray(pd.to_datetime(["2025-06-19T00:00"] * 4), dims="time")

    assert _column_time_step(single, "5Min") == pd.Timedelta("5min")
    assert _column_time_step(duplicated, "5Min") == pd.Timedelta("5min")


def test_column_time_step_measures_regular_and_irregular_grids():
    regular = xr.DataArray(
        pd.date_range("2025-06-19", periods=10, freq="1min"), dims="time"
    )
    jittered = xr.DataArray(
        pd.to_datetime(
            ["2025-06-19T00:00", "2025-06-19T00:05", "2025-06-19T00:11"]
            + ["2025-06-19T00:16", "2025-06-19T00:21", "2025-06-19T01:30"]
        ),
        dims="time",
    )

    assert _column_time_step(regular, "5Min") == pd.Timedelta("1min")
    # Median, so the one long outlying gap does not set the step.
    assert _column_time_step(jittered, "5Min") == pd.Timedelta("5min")
