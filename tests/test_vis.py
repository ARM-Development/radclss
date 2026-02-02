import pytest
import radclss
import arm_test_data
import xarray as xr


@pytest.mark.mpl_image_compare
def test_create_radclss_columns():
    radclss_file = arm_test_data.DATASETS.fetch(
        "bnfcsapr2radclss.c2.20250619.000000.nc"
    )
    fig, axarr = radclss.vis.create_radclss_columns(radclss_file)
    fig.tight_layout()
    assert fig is not None
    assert axarr is not None
    return fig


@pytest.mark.mpl_image_compare
def test_create_radclss_columns_subset():
    radclss_file = arm_test_data.DATASETS.fetch(
        "bnfcsapr2radclss.c2.20250619.000000.nc"
    )
    fig, axarr = radclss.vis.create_radclss_columns(
        radclss_file, stations=["M1", "S30"]
    )
    fig.tight_layout()
    assert fig is not None
    assert axarr is not None
    return fig


@pytest.mark.mpl_image_compare
def test_create_radclss_columns_custom_vmin_vmax():
    radclss_file = arm_test_data.DATASETS.fetch(
        "bnfcsapr2radclss.c2.20250619.000000.nc"
    )
    fig, axarr = radclss.vis.create_radclss_columns(radclss_file, vmin=10, vmax=50)
    fig.tight_layout()
    assert fig is not None
    assert axarr is not None
    return fig


@pytest.mark.mpl_image_compare
def test_create_radclss_columns_different_field():
    radclss_file = arm_test_data.DATASETS.fetch(
        "bnfcsapr2radclss.c2.20250619.000000.nc"
    )
    fig, axarr = radclss.vis.create_radclss_columns(
        radclss_file, field="corrected_velocity", cmap="balance", vmin=-20, vmax=20
    )
    fig.tight_layout()
    assert fig is not None
    assert axarr is not None
    return fig


@pytest.mark.mpl_image_compare
def test_create_radclss_columns_load_data_first():
    radclss_file = arm_test_data.DATASETS.fetch(
        "bnfcsapr2radclss.c2.20250619.000000.nc"
    )
    radclss_file = xr.open_dataset(radclss_file)
    fig, axarr = radclss.vis.create_radclss_columns(
        radclss_file, field="corrected_velocity", cmap="balance", vmin=-20, vmax=20
    )
    fig.tight_layout()
    radclss_file.close()
    assert fig is not None
    assert axarr is not None
    return fig


@pytest.mark.mpl_image_compare
def test_create_radclss_timeseries():
    radclss_file = arm_test_data.DATASETS.fetch(
        "bnfcsapr2radclss.c2.20250619.000000.nc"
    )
    fig, axarr = radclss.vis.create_radclss_rainfall_timeseries(
        radclss_file, field="corrected_reflectivity"
    )
    fig.tight_layout()
    assert fig is not None
    assert axarr is not None
    return fig


@pytest.mark.mpl_image_compare
def test_create_radclss_timeseries_different_field():
    radclss_file = arm_test_data.DATASETS.fetch(
        "bnfcsapr2radclss.c2.20250619.000000.nc"
    )
    fig, axarr = radclss.vis.create_radclss_rainfall_timeseries(
        radclss_file,
        field="corrected_velocity",
        cmap="balance",
        vmin=-20,
        vmax=20,
        rheight=1200,
    )
    fig.tight_layout()
    assert fig is not None
    assert axarr is not None
    return fig


@pytest.mark.mpl_image_compare
def test_create_radclss_timeseries_load_data_first():
    radclss_file = arm_test_data.DATASETS.fetch(
        "bnfcsapr2radclss.c2.20250619.000000.nc"
    )
    radclss_file = xr.open_dataset(radclss_file)
    fig, axarr = radclss.vis.create_radclss_rainfall_timeseries(
        radclss_file,
        field="corrected_velocity",
        cmap="balance",
        vmin=-20,
        vmax=20,
        rheight=1200,
    )
    fig.tight_layout()
    radclss_file.close()
    assert fig is not None
    assert axarr is not None
    return fig


@pytest.mark.mpl_image_compare
def test_create_radclss_timeseries_custom_dpi():
    radclss_file = arm_test_data.DATASETS.fetch(
        "bnfcsapr2radclss.c2.20250619.000000.nc"
    )
    fig, axarr = radclss.vis.create_radclss_rainfall_timeseries(
        radclss_file,
        field="corrected_reflectivity",
        figure_dpi=200,
        rheight=1200,
    )
    fig.tight_layout()
    assert fig is not None
    assert axarr is not None
    return fig


@pytest.mark.mpl_image_compare
def test_create_radclss_timeseries_no_title():
    radclss_file = arm_test_data.DATASETS.fetch(
        "bnfcsapr2radclss.c2.20250619.000000.nc"
    )
    fig, axarr = radclss.vis.create_radclss_rainfall_timeseries(
        radclss_file, field="corrected_reflectivity", title_flag=False, rheight=1200
    )
    fig.tight_layout()
    assert fig is not None
    assert axarr is not None
    return fig
