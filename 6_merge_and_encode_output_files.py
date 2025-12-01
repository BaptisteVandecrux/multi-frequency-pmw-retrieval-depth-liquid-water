import xarray as xr
import os
import lib.smrt_functions as sf
import lib.load as ll

input_dir = "./output/rf_depth_model_ensemble_20250619_max_depth-15_min_samples_leaf-20_min_samples_split-2_n_estimators-100/"
output_file = "./output/rf_depth_model_ensemble_20250619_max_depth-15_min_samples_leaf-20_min_samples_split-2_n_estimators-100/depth_subsurface_water_2010_2023.nc"
years = range(2010, 2024)

def load_year(year):
    ds = xr.open_dataset(
        os.path.join(input_dir, f"rf_output_{year}.nc"), decode_cf=True
    )
    return ds.rename({
        "predicted_depth": "udlw",
        "predicted_std": "udlw_std"
    })
ds_list = [load_year(year) for year in years]
ds_merged = xr.concat(ds_list, dim="time")
ds_merged = ds_merged.drop_vars("iterations")
ds_merged = ds_merged.resample(time='D').first()

ds_merged = ds_merged.where(ds_merged.udlw_std<1)
ds_merged = ds_merged.where(ds_merged.udlw<5)

years = range(2010, 2024)
ds_melt = xr.concat(
    [ll.clip_firn_area(ll.load_melt_xr(y)) for y in years],
    dim='time'
)
ds_melt_daily = (ds_melt.snow_status_wet_dry_01V == 1).resample(time="1D").any()
ds_merged = ds_merged.sel(time=slice(ds_melt_daily.time[0],ds_melt_daily.time[-1]))

ds_melt_daily = ds_melt_daily.sel(time=slice(ds_merged.time[0],ds_merged.time[-1]))
ds_merged = ds_merged.where(ds_melt_daily == 1)
ds_merged = ds_merged.drop_vars("iterations")
ds_merged = ds_merged.drop_vars("band")

ds_merged["udlw"].attrs.update({
    "standard_name": "depth_liquid_water",
    "long_name": "Upper depth of liquid water",
    "units": "m"
})

ds_merged["udlw_std"].attrs.update({
    "standard_name": "standard_deviation_of_depth_liquid_water",
    "long_name": "Standard deviation in estimated upper depth of liquid water among the retrieval's model ensemble",
    "units": "m"
})

ds_merged["x"].attrs.update({
    "standard_name": "projection_x_coordinate",
    "long_name": "x coordinate of projection",
    "units": "m"
})

ds_merged["y"].attrs.update({
    "standard_name": "projection_y_coordinate",
    "long_name": "y coordinate of projection",
    "units": "m"
})

for key in ["units", "calendar"]:
    ds_merged["time"].attrs.pop(key, None)

ds_merged["time"].attrs.update({
    "standard_name": "time",
    "long_name": "time",
    # "units": "hours since 2010-01-01 08:00:00",
    # "calendar": "proleptic_gregorian"
})

ds_merged["udlw"].attrs["grid_mapping"] = "spatial_ref"
ds_merged["udlw_std"].attrs["grid_mapping"] = "spatial_ref"

ds_merged.attrs["title"] = "Depth of subsurface water on the Greenland ice sheet from machine learning and multifrequency passive microwave measurements (2010-2023)"
ds_merged.attrs["DOI"] = "10.22008/FK2/69CRRT"
ds_merged.attrs["version"] = "V2"
ds_merged.attrs["URL"] = "https://doi.org/10.22008/FK2/69CRRT"
ds_merged.attrs["author"] = "Baptiste Vandecrux"
ds_merged.attrs["email"] = "bav@geus.dk"
ds_merged.attrs["Funding"] = "European Space Agency, Climate Change Initiative Research Fellowship"
ds_merged.attrs["Citation"] = "Vandecrux, B.: Depth of subsurface water on the Greenland ice sheet from machine learning and multifrequency passive microwave measurements (2010-2023). GEUS Dataverse, https://doi.org/10.22008/FK2/69CRRT, 2025."
ds_merged.attrs["Conventions"] = "CF-1.8"

valid_enc_keys = {
    'contiguous', 'szip_coding', 'dtype', 'fletcher32', 'shuffle',
    'least_significant_digit', 'szip_pixels_per_block', 'blosc_shuffle',
    'significant_digits', 'complevel', '_FillValue', 'compression',
    'chunksizes', 'quantize_mode', 'endian', 'zlib', "scale_factor", "add_offset"
}

encoding = {}
for var in ds_merged.data_vars:
    if var == 'time':
        var_encoding={}  #"units": "hours since 2010-01-01 08:00:00"}
    else:
        var_encoding = ds_list[0][var].encoding
    print(var_encoding,'\n')
    encoding[var] = {k: v for k, v in var_encoding.items() if k in valid_enc_keys}

ds_merged.to_netcdf(output_file, format="NETCDF4", encoding=encoding)

# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Load merged dataset
ds = xr.open_dataset(output_file)

# Prepare the map data
udlw_masked = ds["udlw"].where(ds["udlw"] > 0)
udlw_count = udlw_masked.count(dim="time") / 14

# Create figure with two panels
fig, (ax_map, ax_ts) = plt.subplots(1, 2, figsize=(12, 5))
im = ax_map.imshow(udlw_count, origin="upper", cmap="viridis")
fig.colorbar(im, ax=ax_map, label="Mean annual detection (2010–2023)")
ax_map.set_title("Click on map to show time series")

line, = ax_ts.plot([], [], marker='o', c='k', label='retrieved udlw')
line1, = ax_ts.plot([], [], ls='--', c='k')
line2, = ax_ts.plot([], [], ls='--', c='k', label='+/- 1 std')
ax_ts.set_title("udlw time series")
ax_ts.set_xlabel("Time")
ax_ts.set_ylabel("Depth (m)")
ax_ts.invert_yaxis()
ax_ts.legend()
# Click handler
def onclick(event):
    if event.inaxes != ax_map:
        return
    x_pixel = int(round(event.xdata))
    y_pixel = int(round(event.ydata))
    if x_pixel < 0 or y_pixel < 0:
        return
    try:
        series = ds["udlw"][:, y_pixel, x_pixel]
        std = ds["udlw_std"][:, y_pixel, x_pixel]
        if series.isnull().all():
            return
        line.set_data(ds["time"].values, series.values)
        line1.set_data(ds["time"].values, series.values+std.values)
        line2.set_data(ds["time"].values, series.values-std.values)
        ax_ts.set_xlim(ds["time"].values[0], ds["time"].values[-1])
        ax_ts.set_ylim(np.nanmax(series), np.nanmin(series))
        ax_ts.set_title(f"udlw time series at (x={x_pixel}, y={y_pixel})")
        fig.canvas.draw_idle()
    except IndexError:
        pass

fig.canvas.mpl_connect("button_press_event", onclick)
plt.tight_layout()
plt.show()

