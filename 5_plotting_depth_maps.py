# -*- coding: utf-8 -*-
"""
Visualization and Analysis of Predicted Subsurface Water Depth over Greenland

This script visualizes and analyzes the spatial and temporal patterns of 
subsurface liquid water (DLW) predicted by a trained Random Forest model applied 
to satellite brightness temperature observations over the Greenland ice sheet.

It includes:
- Extracting DLW time series at specific AWS sites
- Creating maps of DLW mean and uncertainty for selected dates and years
- Mapping the number of days with water depth > 1 m
- Comparing uncertainty vs. depth
- Transect-based visualization
- Generating animated GIFs of DLW evolution

Author: Baptiste Vandecrux  
Contact: bav@geus.dk  
License: CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/)  
Please cite:  
Vandecrux, B., Picard, G., Zeiger, P., Leduc-Leballeur, M., Colliander, A.,  
Hossan, A., & Ahlstrøm, A. (submitted). Estimating the depth of subsurface  
water on the Greenland Ice Sheet using multi-frequency passive microwave  
remote sensing, radiative transfer modeling, and machine learning.  
*Remote Sensing of Environment*.
"""

import xarray as xr
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray
import lib.smrt_functions as sf
import lib.load as ll
import matplotlib.patches as mpatches
from matplotlib.animation import PillowWriter

land = gpd.read_file('data/GIS/Land_3413.shp')

greenland_ice = gpd.read_file('data/GIS/Greenland_ice_shape.shp')
if greenland_ice.crs != "EPSG:3413":
    greenland_ice = greenland_ice.to_crs("EPSG:3413")

# loading data
year = 2021
# ds_pmw = ll.load_pmw_all(year=year)
# final_dataset = ll.prepare_ds_pmw(ds_pmw, filter=True)
# final_dataset = xr.open_dataset(
#     f'data/PMW grids/pre_processed/pmw_data_formatted_{year}.nc'
#     ).load()

years = range(2010, 2024)
ds_melt = xr.concat(
    [ll.clip_firn_area(ll.load_melt_xr(y)) for y in years],
    dim='time'
)

# rf_model_output = ll.clip_firn_area(xr.open_dataset(
    # f'output/rf_depth_model_ensemble_20250513_max_depth-15_min_samples_leaf-20_min_samples_split-2_n_estimators-100/rf_output_{year}.nc'))
rf_model_output = ll.clip_firn_area(xr.open_mfdataset(
    'output/rf_depth_model_hold_out_2021_20250327/rf_output_*.nc'))
rf_model_output = rf_model_output.rename({"predicted_depth": "depth_mean"})
# rf_model_output['predicted_depth'] = rf_model_output.predicted_depth.where(rf_model_output.predicted_depth<5)

# %% test at given sites
import lib.plot as lp
site = 'CP1'
year_data = ll.load_snow_model_output(site).sel(time=str(year))
df_pmw = ll.extract_pixel(final_dataset, site, year_data.attrs['latitude'], year_data.attrs['longitude']).to_dataframe()
ds_pred_site  = ll.extract_pixel(rf_model_output,
                                 site,
                                 year_data.attrs['latitude'],
                                 year_data.attrs['longitude'])
df_pred = ds_pred_site.to_dataframe()[['depth_mean', 'depth_std']]

lp.plot_result(year_data, df_pmw, df_pmw, df_pred,
                None, slope_threshold = -1.5, mode='pmw', year_cv='XXXX')

# %% Plotting water depth and TLWC maps
def plot_base_layers(ax, date, transect_y):
    land.plot(ax=ax, color='saddlebrown')
    greenland_ice.plot(ax=ax, color='lightblue')

    # ax.axhline(y=transect_y, color='black', linestyle='--', linewidth=1)

    ax.set_axis_off()
abc='abcd'
def plot_figure(data_array, cmap, vmin, vmax, colorbar_label, dates, filename, alpha=1, invert_cbar=True):
    fig, axs = plt.subplots(1,4, figsize=(10, 4), constrained_layout=True)
    depth_plot = None
    count=0
    for ax, date in zip(axs.flatten(), dates):
        plot_base_layers(ax, date, transect_y)
        data = data_array.sel(time=date, method='nearest')
        data = data.where(data >= vmin) if vmin > 0 else data
        depth_plot = data.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
        binary_data = ds_melt.snow_status_wet_dry_01V.sel(time=date, method='nearest')
        binary_data = xr.where(binary_data == -10, 0, binary_data)
        mask = np.isnan(binary_data) | (binary_data != 1)
        binary_data_masked = binary_data.where(mask)
        binary_data_masked.plot(ax=ax, cmap='Greys', add_colorbar=False, alpha=1)
        ax.set_title(f"({abc[count]}) {date.date()}")
        count+=1
    no_melt_patch = mpatches.Patch(color=0.6 * np.array([1, 1, 1]), label='No melt')
    land_patch = mpatches.Patch(color='saddlebrown', label='Land')
    ice_patch = mpatches.Patch(color='lightblue', label='Bare ice')
    fig.legend(handles=[no_melt_patch, land_patch, ice_patch], loc='lower left',
               fontsize=9, bbox_to_anchor=(0.15,0.05))
    cbar = fig.colorbar(depth_plot, ax=axs, orientation='vertical', shrink=0.7, pad=0.05)
    # if invert_cbar:
    #     cbar.ax.invert_yaxis()
    cbar.set_label(colorbar_label)
    plt.show()
    fig.savefig(filename, dpi=300)

plt.close('all')
transect_y = rf_model_output.y.min().item() + (rf_model_output.y.max().item() \
                               - rf_model_output.y.min().item()) * 0.87 / 3
if year == 2021:
    dates = pd.to_datetime([f'{year}-08-16', f'{year}-08-20', f'{year}-08-25', f'{year}-08-30'])
else:
    dates = pd.to_datetime([f'{year}-08-16', f'{year}-08-20', f'{year}-08-25', f'{year}-08-30'])

plot_figure(
    rf_model_output.depth_mean, 'gnuplot_r', 0, 5,
    'Estimated depth of subsurface water (m)',
    dates, f'figures/depth maps/{year}_depth_map.png'
)

cmap = plt.get_cmap('bwr')
cmap.set_under(color=(0, 0, 0, 0))
abc='efgh'
plot_figure(
    rf_model_output.depth_std, cmap, 0, 1,
    'Uncertainty on the depth of subsurface water (m))',
    dates, f'figures/depth maps/{year}_std_map.png', alpha=0.5
)

# %% Maps of days with water below 1 mplt.close('all')

years = range(2010, 2024)
abc = 'abcdefghijklmnopqrstuvwxy'
count_data_by_year = {}

for year in years:
    print(year)
    dates = rf_model_output.time.sel(time=str(year)).values
    data = rf_model_output.depth_mean.sel(time=dates)
    binary_mask = ds_melt.snow_status_wet_dry_01V.sel(time=dates, method='nearest')
    binary_mask = xr.where(binary_mask == -10, 0, binary_mask)
    data = data.sel(time=~data.time.to_index().duplicated())
    binary_mask = binary_mask.sel(time=~binary_mask.time.to_index().duplicated(), method='nearest')
    mask = np.isnan(binary_mask) | (binary_mask != 1)
    valid_data = data.where(~mask)
    count_days = (valid_data > 1).sum(dim='time')
    count_days = ll.clip_firn_area(count_days)
    count_data_by_year[year] = count_days
# %%
fig, axs = plt.subplots(3,5, figsize=(10, 8))
fig.subplots_adjust(hspace=0.08, wspace=0.01, left=0.05,top=0.95, right=0.9,bottom=0.05)
axs = axs.flatten()

for idx, (year, ax) in enumerate(zip(years, axs)):
    print(year)
    plot_base_layers(ax, None, transect_y)
    img = count_data_by_year[year].plot(ax=ax, cmap='viridis', vmin=0, vmax=100, add_colorbar=False)
    ax.set_title("")
    ax.text(0, 0.98, f"({abc[idx]}) {year}", transform=ax.transAxes,
        ha='left', va='bottom', fontsize=plt.rcParams['axes.titlesize'])

for ax in axs[len(years):]:
    ax.remove()

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cax = fig.add_axes([0.9, 0.25, 0.03, 0.5])  # [left, bottom, width, height]
cbar = fig.colorbar(img, cax=cax)
cbar.set_label("Days with DLW >1 m")

fig.savefig("figures/depth maps/water_depth_days_above_1m.png", dpi=300)
plt.show()


# %% Uncertainty vs. depth
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

x, y = [], []
for d in dates:
    mean = rf_model_output.depth_mean.sel(time=d, method='nearest')
    std = rf_model_output.depth_std.sel(time=d, method='nearest')
    mask = ds_melt.snow_status_wet_dry_01V.sel(time=d, method='nearest') == 1
    mean = mean.where(mask)
    std = std.where(mask)
    x.append(mean.data.ravel())
    y.append(std.data.ravel())
x, y = np.concatenate(x), np.concatenate(y)
m = ~np.isnan(x) & ~np.isnan(y)
x, y = x[m], y[m]

z = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
i = z.argsort()
plt.scatter(x[i], y[i], c=z[i], s=5, cmap='viridis', alpha=0.7)
plt.xlabel("DLW Mean (m)"); plt.ylabel("DLW Uncertainty (m)")
plt.colorbar(label="Density"); plt.grid(True); plt.tight_layout(); plt.show()


# %% transect plot

# Define a color map to ensure each date has a unique color
colors = plt.cm.tab10(np.linspace(0, 1, len(dates)))

# Loop through the timestamps to plot all depths as a line, and melting depths as markers
for i, (date, color) in enumerate(zip(dates, colors)):
    fig, ax = plt.subplots(figsize=(10, 5))
    # Select depth data at the specific date and y-coordinate using .sel
    depth_data = (rf_model_output.predicted_depth
                  .sel(y=transect_y, method='nearest')
                  .sel(
                      time=slice(date, date + pd.to_timedelta('2 days'))
                      ).mean(dim='time').to_dataframe().reset_index()
                  )
    ax.plot(depth_data.x, depth_data.predicted_depth, color=color, label=f'{date.date()}')

    # Select binary melt data to highlight melting points
    melt_transect = ds_melt.snow_status_wet_dry_01V.sel(time=date, y=transect_y, method='nearest')
    melt_transect = xr.where(melt_transect == 1, 1, np.nan)  # Keep only melt points
    melt_depth_data = depth_data[melt_transect.values == 1]  # Filter only melting depths

    # Plot melting depths as round markers with the same color
    ax.plot(melt_depth_data.x, melt_depth_data.predicted_depth, 'o',markersize=10, color=color, alpha=0.6)

    # Customize plot
    ax.set_title(f'Predicted Depth and Melt-Affected Depths Along Transect at y = {transect_y:.2f}')
    ax.set_xlabel('x-coordinate')
    ax.set_ylabel('Predicted Depth (m)')
    ax.set_ylim(5,0)
    ax.legend(title='Date')
    plt.grid()
    plt.show()


# %% plot GIF
for year in range(2010,2024):
    fig, ax = plt.subplots(figsize=(8, 8))
    dates = pd.date_range(start=f'{year}-07-01', end=f'{year}-10-01', freq='D')
    writer = PillowWriter(fps=5)  # Set frames per second (adjust as needed)
    
    with writer.saving(fig, f'figures/depth maps/{year}_depth_map.gif', dpi=100):
    
        for date in dates:
            print(date)
            ax.clear()
    
            land.plot(ax=ax, color='saddlebrown', label='land')
            greenland_ice.plot(ax=ax, color='lightblue', label='bare ice area')
    
            depth_plot = rf_model_output.predicted_depth.sel(
                time=slice(date, date + pd.to_timedelta('2 days'))
            ).mean(dim='time').plot(
                ax=ax, cmap='gnuplot_r', add_colorbar=False,
                vmin=0,vmax=5   , label='predicted depth'
            )
    
            binary_data = ds_melt.snow_status_wet_dry_01V.sel(time=date, method='nearest')
            binary_data = xr.where(binary_data == -10, 0, binary_data)
    
            # Mask areas where melt_01 is NaN or not equal to 1
            mask = np.isnan(binary_data) | (binary_data != 1)
            binary_data_masked = binary_data.where(mask)
    
            binary_data_masked.plot(ax=ax, cmap='Greys', add_colorbar=False, alpha=1)
    
            no_melt_patch = mpatches.Patch(color=0.6 * np.array([1, 1, 1]), label='No melt detected')
            land_patch = mpatches.Patch(color='saddlebrown', label='land')
            ice_patch = mpatches.Patch(color='lightblue', label='bare ice area')
    
            fig.legend(handles=[no_melt_patch, land_patch, ice_patch], loc='upper right')
            ax.set_title(f"{date.strftime('%Y-%m-%d')}")
            ax.set_axis_off()
    
            if date == dates[0]:  # Add colorbar only on the first frame
                cbar = fig.colorbar(depth_plot, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
                cbar.set_label('depth of subsurface water (m)')
                cbar.ax.invert_yaxis()
    
            writer.grab_frame()

# %% click and plot tool
# from timeseries_explorer import DataExplorer
# import xarray as xr
# predicted_depth_da = xr.open_dataset('output/rf_depth_model_hold_out_2021_20250313/predicted_depth_2021.nc')
# app = DataExplorer([-predicted_depth_da.predicted_depth]) #, final_dataset.melt_01])
# app.mainloop()
