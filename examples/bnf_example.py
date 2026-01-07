import radclss
import glob
import xradar as xd
import xarray as xr
import matplotlib.pyplot as plt

date = '20250619'
radar_files = glob.glob(f'/Volumes/Untitled/bnf/bnfcsapr2cmacS3.c1/*{date}*.nc')
volumes = {'date': date,
           'radar': radar_files[0:2],  # Limit to first 2 files for testing
           'sonde': glob.glob(f'/Volumes/Untitled/bnf/in_situ/bnfsondewnpnM1.b1/*{date}*.cdf'),
           'vd_M1': glob.glob(f'/Volumes/Untitled/bnf/in_situ/bnfvdisquantsM1.c1/*{date}*.nc'),
           'met_M1': glob.glob(f'/Volumes/Untitled/bnf/in_situ/bnfmetM1.b1/*{date}*'),
           'met_S20': glob.glob(f'/Volumes/Untitled/bnf/in_situ/bnfmetS20.b1/*{date}*'),
           'met_S30': glob.glob(f'/Volumes/Untitled/bnf/in_situ/bnfmetS30.b1/*{date}*'),
           'met_S40': glob.glob(f'/Volumes/Untitled/bnf/in_situ/bnfmetS40.b1/*{date}*'),
           'wxt_S13': glob.glob(f'/Volumes/Untitled/bnf/in_situ/bnfmetwxtS13.b1/*{date}*.nc'),
           'pluvio_M1': glob.glob(f'/Volumes/Untitled/bnf/in_situ/bnfwbpluvio2M1.a1/*{date}*.nc'),
           'ld_M1': glob.glob(f'/Volumes/Untitled/bnf/in_situ/bnfldquantsM1.c1/*{date}*.nc'),
           'ld_S30': glob.glob(f'/Volumes/Untitled/bnf/in_situ/bnfldquantsS30.c1/*{date}*.nc')}

input_site_dict = {'M1': (34.34525, -87.33842, 293),
                   'S4': (34.46451, -87.23598, 197),
                   'S20': (34.65401, -87.29264, 178),
                   'S30': (34.38501, -86.92757, 183),
                   'S40': (34.17932, -87.45349, 236),
                   'S13': (34.343889, -87.350556, 286)}
my_columns = radclss.core.radclss(volumes, input_site_dict, serial=True, verbose=True)
radclss.io.write_radclss_output(my_columns, 'radclss_example.nc', 'csapr2radclss.c2')

for vars in my_columns.data_vars:
    print(vars, my_columns[vars].dtype)
fig, ax = radclss.vis.create_radclss_columns('radclss_example.nc')
print(fig)
plt.show()
