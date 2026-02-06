import act
import sys
import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from datetime import timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_radclss_columns(
    radclss,
    field="corrected_reflectivity",
    vmin=-5,
    vmax=65,
    stations=None,
    **kwargs,
):
    """
    With the RadCLss product, generate a figure of all extracted columns

    Input
    -----
    radclss : str or xarray dataset
        Filepath to the RadCLss file or xarray dataset with the columns.
    field : str
        Specific CMAC field to display extracted column of
    vmin : int
        Minimum value to display between all subplots for the specific radar
        parameter
    vmax : int
        Maximum value to display between all subplots for the specific radar
        parameter
    stations : list of str
        List of station identifiers to plot. If None, defaults to
        all of them.

    Output
    ------
    fig: matplotlib figure
        Figure containing the extracted columns.
    axarr : matplotlib axes array
        Array of matplotlib axes containing the extracted columns.

    """

    if "cmap" not in kwargs:
        kwargs["cmap"] = "ChaseSpectral"

    if stations == []:
        raise ValueError(
            "\nERROR - (create_radclss_columns):"
            + " \n\tStations list is empty. Please provide at least one station.\n"
        )
    # read the RadCLss file
    if isinstance(radclss, str):
        try:
            ds = xr.open_dataset(radclss, decode_timedelta=False)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            # 'e' will contain the error object
            print(
                "\nERROR - (create_radclss_timeseries):"
                + f" \n\tOccured When Reading in RadCLss File: \n\t{e}"
            )
            print(f"\tError type: {type(e)}")
            print("\tLine Number: ", exc_tb.tb_lineno)
            print("\tFile Name: ", exc_tb.tb_frame.f_code.co_filename)
            print("\n")
            return
    elif isinstance(radclss, xr.Dataset):
        ds = radclss
    else:
        raise TypeError(
            "\nERROR - (create_radclss_timeseries):"
            + " \n\tRadCLss Input is Not a String or xarray Dataset\n"
        )
    # Create the figure
    stations = ds["station"].data if stations is None else stations
    nrows = int(np.ceil(len(stations) / 2))
    ncols = 2
    height = 5 * nrows
    width = 14
    fig, axarr = plt.subplots(nrows, ncols, figsize=(width, height))
    plt.subplots_adjust(hspace=0.8)

    # Define the time of the radar file we are plotting against
    radar_time = datetime.datetime.strptime(
        np.datetime_as_string(ds["time"].data[0], unit="s"), "%Y-%m-%dT%H:%M:%S"
    )
    final_time = radar_time + timedelta(days=1)
    for i, station in enumerate(stations):
        row = i // 2
        col = i % 2
        if len(axarr.shape) == 1:
            axarr = np.expand_dims(axarr, axis=0)
        ds[field].sel(station=station).sel(
            time=slice(
                radar_time.strftime("%Y-%m-%dT00:00:00"),
                final_time.strftime("%Y-%m-%dT00:00:00"),
            )
        ).plot(y="height", ax=axarr[row, col], vmin=vmin, vmax=vmax, **kwargs)

    return fig, axarr


def create_radclss_rainfall_timeseries(
    radclss,
    field="corrected_reflectivity",
    vmin=-5,
    vmax=65,
    cmap="ChaseSpectral",
    rr_min=0,
    rr_max=250,
    cum_min=0,
    cum_max=500,
    dis_site="M1",
    rheight=750,
    title_flag=True,
    figure_dpi=300,
):
    """
    With the RadCLss product, generate a timeseries of radar reflectivity
    factor, particle size distribution and cumuluative precipitaiton
    for the ARM SAIL M1 Site.

    This timeseries quick is to serve as a means for evaluating the RadCLss
    product.

    Input
    -----
    radclss : str
        Filepath to the RadCLss file.
    field : str
        Specific CMAC field to display extracted column of
    vmin : int
        Minimum value to display between all subplots for the specific radar
        parameter
    vmax : int
        Maximum value to display between all subplots for the specific radar
        parameter
    cmap : str
        Colormap to use for the radar field.
    rr_min: float
        Minimum value for the precipitation rate axis.
    rr_max: float
        Maximum value for the precipitation rate axis.
    cum_min: float
        Minimum value for the cumulative precipitation axis.
    cum_max: float
        Maximum value for the cumulative precipitation axis.
    dis_site : str
        Identifer of the supported location for lat/lon slices
    height : int
        Column height to compare against in-situ sensors for precipitation
        accumulation.
    title_flag : bool
        Flag to add the title to the figure.
    figure_dpi : int
        DPI to set for the figure.

    Output
    ------
    timeseries : png
        Saved image of the RadCLss timeseris
    """
    # Create the figure
    fig = plt.figure(figsize=(14, 10))
    plt.subplots_adjust(wspace=0.05)

    # read the RadCLss file
    try:
        if isinstance(radclss, str):
            ds = xr.open_dataset(radclss, decode_timedelta=False)
        elif isinstance(radclss, xr.Dataset):
            ds = radclss
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        # 'e' will contain the error object
        print(
            "\nERROR - (create_radclss_timeseries):"
            + f" \n\tOccured When Reading in RadCLss File: \n\t{e}"
        )
        print(f"\tError type: {type(e)}")
        print("\tLine Number: ", exc_tb.tb_lineno)
        print("\tFile Name: ", exc_tb.tb_frame.f_code.co_filename)
        print("\n")
        return

    # Define the time of the radar file we are plotting against
    radar_time = datetime.datetime.strptime(
        np.datetime_as_string(ds["time"].data[0], unit="s"), "%Y-%m-%dT%H:%M:%S"
    )
    final_time = radar_time + timedelta(days=1)

    # -----------------------------------------------
    # Side Plot A - Display the RadClss Radar Field
    # -----------------------------------------------
    # Top right hand subplot - Radar TimeSeries
    ax2 = fig.add_subplot(311)

    ds[field].sel(station=dis_site).plot(
        x="time", ax=ax2, cmap=cmap, vmin=vmin, vmax=vmax
    )

    ax2.set_title(
        "Extracted Radar Columns and In-Situ Sensors (RadCLss), BNF Site: " + dis_site
    )
    ax2.set_ylabel("Height [m]")
    ax2.set_xlabel("Time [UTC]")

    # --------------------------------------
    # Side Plot B - Display the Rain Rates
    # --------------------------------------
    # Top right hand subplot - Radar TimeSeries
    ax3 = fig.add_subplot(312)

    # CMAC derived rain rate
    ds["rain_rate_A"].sel(station=dis_site).sel(height=rheight, method="nearest").plot(
        x="time", ax=ax3, label="CMAC"
    )

    # Pluvio2 Weighing Bucket Rain Gauge
    ds["intensity_rtnrt"].sel(station=dis_site).plot(x="time", ax=ax3, label="PLUVIO2")

    # LDQUANTS derived rain rate
    ds["ldquants_rain_rate"].sel(station=dis_site).plot(
        x="time", ax=ax3, label="LDQUANTS"
    )

    ax3.set_title(" ")
    ax3.set_ylabel("Precipitation Rate \n[mm/hr]")
    ax3.set_xlabel("Time [UTC]")
    ax3.set_xlim(
        [
            radar_time.strftime("%Y-%m-%dT00:00:00"),
            final_time.strftime("%Y-%m-%dT00:00:00"),
        ]
    )
    ax3.legend(loc="upper right")
    ax3.grid(True)
    ax3.set_ylim(rr_min, rr_max)
    # Add a blank space next to the subplot to shape it as the above plot
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="3%", pad=1.9)
    cax.set_visible(False)

    # ------------------------------------------
    # Side Plot C - Precipitation Accumulation
    # ------------------------------------------
    ax4 = fig.add_subplot(313)

    # CMAC Accumulated Rain Rates
    radar_accum = act.utils.accumulate_precip(
        ds.sel(station=dis_site).sel(height=rheight, method="nearest"), "rain_rate_A"
    ).compute()
    # CMAC Accumulated Rain Rates
    radar_accum["rain_rate_A_accumulated"].plot(x="time", ax=ax4, label="CMAC")

    # PLUVIO2 Accumulation
    if dis_site == "M1":
        gauge_precip_accum = act.utils.accumulate_precip(
            ds.sel(station=dis_site), "intensity_rtnrt"
        ).intensity_rtnrt_accumulated.compute()
        gauge_precip_accum.plot(x="time", ax=ax4, label="PLUVIO2")

    # LDQUANTS Accumulation
    if dis_site == "M1" or dis_site == "S30":
        ld_precip_accum = act.utils.accumulate_precip(
            ds.sel(station=dis_site), "ldquants_rain_rate"
        ).ldquants_rain_rate_accumulated.compute()
        ld_precip_accum.plot(x="time", ax=ax4, label="LDQUANTS")

    ax4.set_title(" ")
    ax4.set_ylabel("Accumulated Precipitation \n[mm]")
    ax4.set_xlabel("Time [UTC]")
    ax4.legend(loc="upper left")
    ax4.grid(True)
    ax4.set_xlim(
        [
            radar_time.strftime("%Y-%m-%dT00:00:00"),
            final_time.strftime("%Y-%m-%dT00:00:00"),
        ]
    )
    ax4.set_ylim(cum_min, cum_max)
    # Add a blank space next to the subplot to shape it as the above plot
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="3%", pad=1.9)
    cax.set_visible(False)

    # Set the DPI to a higher value (e.g., 300)
    plt.rcParams["figure.dpi"] = figure_dpi
    plt.rcParams["savefig.dpi"] = figure_dpi

    # Add the title
    if title_flag is True:
        plt.suptitle(
            "BNF Extracted Radar Columns and In-Situ Sensors (RadCLss) \n"
            + radar_time.strftime("%Y-%m-%d")
        )

    # Clean up this function
    ax = np.array([ax2, ax3, ax4])
    del radar_accum
    if dis_site == "M1" or dis_site == "S30":
        del ld_precip_accum
    if dis_site == "M1":
        del gauge_precip_accum
    if isinstance(radclss, str):
        ds.close()

    return fig, ax
