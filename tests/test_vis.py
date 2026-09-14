import datetime

import arm_test_data
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import radclss
from radclss.vis import quicklooks


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


def _wraparound_dataset(start="2026-07-13T23:58:00", periods=1440):
    """
    Build a RadCLss-style dataset whose first volume starts before midnight.

    Mirrors ``nsaradclssC1.c0.20260713.235803.nc``: the file is the 2026-07-14
    product, but because the first radar volume of the day starts at 23:58 the
    first timestamp falls on 2026-07-13.
    """
    time = np.datetime64(start) + np.arange(periods) * np.timedelta64(1, "m")
    height = np.arange(0.0, 3000.0, 500.0)
    station = ["M1", "S30"]

    ramp = np.linspace(0.0, 1.0, time.size)
    field = (
        ramp[:, None, None]
        * np.ones((1, height.size, 1))
        * np.ones((1, 1, len(station)))
    ) * 40.0 - 10.0
    rain = np.tile(ramp[:, None], (1, len(station)))

    return xr.Dataset(
        {
            "corrected_reflectivity": (
                ("time", "height", "station"),
                field,
                {"long_name": "Equivalent reflectivity factor", "units": "dBZ"},
            ),
            "rain_rate_A": (
                ("time", "height", "station"),
                field * 0.0 + 1.0,
                {"long_name": "Rain rate", "units": "mm/hr"},
            ),
            "intensity_rtnrt": (
                ("time", "station"),
                rain,
                {"long_name": "Pluvio2 rain rate", "units": "mm/hr"},
            ),
            "ldquants_rain_rate": (
                ("time", "station"),
                rain,
                {"long_name": "LDQUANTS rain rate", "units": "mm/hr"},
            ),
        },
        coords={"time": time, "height": height, "station": station},
    )


def test_daily_time_window_wraparound():
    """The day is taken from the bulk of the samples, not from time[0]."""
    ds = _wraparound_dataset()
    start, end, ref_day = quicklooks._daily_time_window(ds["time"].data)

    # time[0] is on 07-13, but the file is the 07-14 product
    assert ref_day == np.datetime64("2026-07-14", "D")
    # the pre-midnight leader and the last sample are both kept
    assert start == np.datetime64("2026-07-13T23:58:00", "s")
    assert end == np.datetime64("2026-07-15T00:00:00", "s")


def test_daily_time_window_no_wraparound():
    """A file starting exactly at midnight is anchored on its own day."""
    ds = _wraparound_dataset(start="2026-07-14T00:00:00")
    start, end, ref_day = quicklooks._daily_time_window(ds["time"].data)

    assert ref_day == np.datetime64("2026-07-14", "D")
    assert start == np.datetime64("2026-07-14T00:00:00", "s")
    assert end == np.datetime64("2026-07-15T00:00:00", "s")


def test_daily_time_window_partial_day():
    """A short file that wraps is still anchored on the majority day."""
    ds = _wraparound_dataset(start="2026-07-17T23:54:00", periods=184)
    _start, _end, ref_day = quicklooks._daily_time_window(ds["time"].data)

    assert ref_day == np.datetime64("2026-07-18", "D")


def test_daily_time_window_empty():
    with pytest.raises(ValueError, match="empty time coordinate"):
        quicklooks._daily_time_window(np.array([], dtype="datetime64[s]"))


def test_create_radclss_columns_wraparound_keeps_all_times():
    """Columns spanning midnight must not be discarded by the day slice."""
    ds = _wraparound_dataset()
    fig, axarr = radclss.vis.create_radclss_columns(
        ds, field="corrected_reflectivity", vmin=-10, vmax=30
    )

    # every sample in the file is drawn, not just the two before midnight
    mesh = axarr[0, 0].collections[0]
    n_plotted = mesh.get_coordinates().shape[1] - 1
    assert n_plotted == ds.sizes["time"]

    # and the axis covers the 07-14 product day rather than 07-13
    x_min, x_max = axarr[0, 0].get_xlim()
    span = mdates.num2date(x_max) - mdates.num2date(x_min)
    assert span > datetime.timedelta(hours=23)
    assert (
        mdates.num2date(x_min)
        < datetime.datetime(2026, 7, 14, 12, tzinfo=datetime.UTC)
        < mdates.num2date(x_max)
    )
    plt.close(fig)


def test_create_radclss_timeseries_wraparound_panels_aligned():
    """All three timeseries panels share the corrected day window."""
    ds = _wraparound_dataset()
    fig, axarr = radclss.vis.create_radclss_rainfall_timeseries(
        ds, field="corrected_reflectivity", rheight=1000
    )

    limits = [ax.get_xlim() for ax in axarr]
    assert limits[0] == limits[1] == limits[2]

    x_min, x_max = limits[0]
    assert mdates.num2date(x_min) == datetime.datetime(
        2026, 7, 13, 23, 58, tzinfo=datetime.UTC
    )
    assert mdates.num2date(x_max) == datetime.datetime(2026, 7, 15, tzinfo=datetime.UTC)

    # the suptitle reports the product day, not the day of time[0]
    assert fig._suptitle.get_text().endswith("2026-07-14")
    plt.close(fig)
