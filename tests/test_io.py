import radclss
import xarray as xr
import arm_test_data


def test_write():
    radclss_file = arm_test_data.DATASETS.fetch(
        "bnfcsapr2radclss.c2.20250619.000000.nc"
    )
    ds = xr.open_dataset(radclss_file)
    radclss.io.write_radclss_output(ds, "test_output.nc", "csapr2radclss.c2")
    ds_out = xr.open_dataset("test_output.nc")
    assert ds_out.dims == ds.dims
    assert set(ds_out.data_vars) == set(ds.data_vars)
    for var in ds.data_vars:
        assert ds_out[var].dtype == ds[var].dtype
    ds_out.close()
