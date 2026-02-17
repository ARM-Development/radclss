import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock
from radclss.util.column_utils import get_nexrad_column


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
                    site=site,
                    input_site_dict=input_site_dict,
                    rad_time=rad_time,
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
        site=site,
        input_site_dict=input_site_dict,
        rad_time=rad_time,
        height_bins=height_bins,
    )

    # Verify structure
    assert result is not None
    assert isinstance(result, xr.Dataset)
    assert "station" in result.dims
    assert "height" in result.dims
    assert result.dims["station"] == len(input_site_dict)
    assert "reflectivity" in result.data_vars
