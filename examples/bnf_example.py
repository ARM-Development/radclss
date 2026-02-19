"""
Example script demonstrating radclss processing with BNF site data.

This example shows how to:
- Process multiple radar systems (CSAPR2, KASACR, XSACR) with radclss
- Integrate sonde and ground-based meteorological instruments
- Use parallel processing with Dask distributed computing
- Enable NEXRAD data integration for additional radar coverage
- Generate and visualize radclss column output

The script processes data from the BNF (Bankhead National Forest) site for a single date,
combining radar volumes with ground stations at multiple locations (M1, S4, S20, S30, S40, S13).
"""

import radclss
import glob
import matplotlib.pyplot as plt
import os

from dask.distributed import Client, LocalCluster


def main():
    date = "20250520"
    base_path = os.environ["DATA_HOME"]

    radar_files = glob.glob(f"{base_path}/bnf/bnfcsapr2cmacS3.c1/*{date}*.nc")
    kasacr_files = glob.glob(f"{base_path}/bnf/bnfkasacrcfrS4.a1/*{date}*.nc")
    xsacr_files = glob.glob(f"{base_path}/bnf/bnfxsacrcfrS4.a1/*{date}*.nc")
    volumes = {
        "date": date,
        "radar_csapr2cmac": radar_files[:30],  # Limit to first 2 files for testing
        "radar_kasacr": kasacr_files,
        "radar_xsacr": xsacr_files,
        "sonde": glob.glob(f"{base_path}/bnf/bnfsondewnpnM1.b1/*{date}*.cdf"),
        "vd_M1": glob.glob(f"{base_path}/bnf/bnfvdisquantsM1.c1/*{date}*.nc"),
        "met_M1": glob.glob(f"{base_path}/bnf/bnfmetM1.b1/*{date}*"),
        "met_S20": glob.glob(f"{base_path}/bnf/bnfmetS20.b1/*{date}*"),
        "met_S30": glob.glob(f"{base_path}/bnf/bnfmetS30.b1/*{date}*"),
        "met_S40": glob.glob(f"{base_path}/bnf/bnfmetS40.b1/*{date}*"),
        "wxt_S13": glob.glob(f"{base_path}/bnf/bnfmetwxtS13.b1/*{date}*.nc"),
        "pluvio_M1": glob.glob(f"{base_path}/bnf/bnfwbpluvio2M1.a1/*{date}*.nc"),
        "ld_M1": glob.glob(f"{base_path}/bnf/bnfldquantsM1.c1/*{date}*.nc"),
        "ld_S30": glob.glob(f"{base_path}/bnf/bnfldquantsS30.c1/*{date}*.nc"),
    }

    input_site_dict = {
        "M1": (34.34525, -87.33842, 293),
        "S4": (34.46451, -87.23598, 197),
        "S20": (34.65401, -87.29264, 178),
        "S30": (34.38501, -86.92757, 183),
        "S40": (34.17932, -87.45349, 236),
        "S13": (34.343889, -87.350556, 286),
    }

    with Client(LocalCluster(n_workers=4, threads_per_worker=1)) as client:  # noqa
        my_columns = radclss.core.radclss(
            volumes,
            input_site_dict,
            "radar_csapr2cmac",
            serial=False,
            verbose=True,
            nexrad=True,
        )
    my_columns.to_netcdf("nexrad_radclss_example.nc")
    radclss.io.write_radclss_output(my_columns, "radclss_example.nc", "radclss.c2")

    for vars in my_columns.data_vars:
        print(vars, my_columns[vars].dtype)
    fig, ax = radclss.vis.create_radclss_columns("radclss_example.nc")
    print(fig)
    plt.show()


if __name__ == "__main__":
    main()
