import pytest
import radclss
import arm_test_data


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
