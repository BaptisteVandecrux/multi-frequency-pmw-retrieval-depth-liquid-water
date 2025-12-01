# -*- coding: utf-8 -*-
"""
Water Depth and Brightness Temperature Analysis for Greenland AWS Sites

This script loads SMRT simulation outputs and GEUS snow model outputs for a set
of AWS sites on the Greenland ice sheet. It computes frequency ratios, derives
dry-season averages, resamples snow model data to fixed depths, and extracts
the upper depth of liquid water.

The results are merged into a unified dataframe per site and saved for further
analysis and visualization. Optionally, diagnostic plots of modeled TB and water
depths are generated.

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.load as ll
import lib.smrt_functions as sf
import lib.plot as lp
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import xarray as xr
from tqdm import tqdm

df_list=[]
station_list = [ 'H2','H3', 'T1old','T2_09','T3', 'T4',
             'FA-13','FA-15-1','FA-15-2','SIGMA-A',
             'KAN_U', 'DYE-2','CP1', 'South Dome', 'NASA-SE',
             'NASA-U','Saddle', 'CEN2','Humboldt'
             # 'NEEM','NGRIP', 'Summit', 'Tunu-N', 'NASA-E','EastGRIP'
             ]
# for site in ['KAN_U','FA-13','DYE-2','CP1',]:
for site in station_list:
# for site in ['DYE-2']:
    print(site)
    folder = 'smrt_output/with_gs_and_n_ice_tuning/'
    files_new_format = glob.glob(f"{folder}/{site}_*_Tb_smrt.nc")

    if files_new_format:
        df = ll.load_smrt_simulations(site, folder=folder)
    else:
        print("No SMRT data files found for",site)
        continue

    df = sf.calc_ratio(df)

    features_start = ['01_V', '06_V', '10_V', '19_V',
                      '01_ratio', '06_ratio', '10_ratio', '19_ratio']

    features_dry = [f+'_dry' for f in features_start]
    df[features_dry] = np.nan
    for yr in np.unique(df.index.year):
        df.loc[str(yr),features_dry] = \
            df.loc[str(yr)+'-02-01':str(yr)+'-03-01', features_start].mean().values

    df = df.dropna()

    data = ll.load_snow_model_output(site)

    # for year in range(2010,2024):
    # for year in [2021]:
    # print(year)
    # year_data = data.sel(time=str(year))
    year_data = data.copy()
    slwc_sum = year_data['slwc'].sum(dim='level')
    year_data = year_data.where(slwc_sum>0,drop=True)
    slwc_sum = year_data['slwc'].sum(dim='level')

    # list_da = []
    # for i, time in tqdm(enumerate(year_data.time)):
    #     list_da.append(sf.resample_firn_model_output_at_fixed_depths(
    #                                     year_data[['depth','slwc']].sel(time=time).copy(),
    #                                     plot=False))
    # data_resample=xr.concat(list_da, dim='time')


    def process_time(t):
        return sf.resample_firn_model_output_at_fixed_depths(
            year_data[['depth', 'slwc']].sel(time=t).copy(), plot=False
        )

    list_da = Parallel(n_jobs=-1)(delayed(process_time)(t) for t in tqdm(year_data.time.values))

    data_resample = xr.concat(list_da, dim='time')

    upper_depth_df =  ll.get_upper_water_depth(data_resample)

    slwc_sum_df = slwc_sum.to_dataframe()
    slwc_sum_df.columns=['total_liquid_water']

    merged_df = df.merge(upper_depth_df, left_index=True, right_index=True, how='inner')
    merged_df = merged_df.merge(slwc_sum_df, left_index=True, right_index=True, how='inner')
    merged_df['site'] = site
    merged_df['year'] = merged_df.index.year

    merged_df['latitude'] = data.attrs['latitude']
    merged_df['longitude'] = data.attrs['longitude']
    merged_df['altitude'] = data.attrs['altitude']

    merged_df['T2m_avg'] = data.attrs['T2m_avg']
    merged_df['SF_avg'] = data.attrs['SF_avg']
    merged_df['melt_avg'] = data.attrs['melt_avg']

    # Display the result
    df_list.append(merged_df)
    plot = False
    if plot:
        for year in range(2010,2024):
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 12))
            plt.subplots_adjust(top=0.7)
            # Adjust the first subplot to leave space for the colorbar
            box1 = axs[0].get_position()  # Get original position of the first axis
            box2 = axs[1].get_position()  # Get original position of the first axis
            axs[0].set_position([box1.x0, box1.y0, box2.width, box1.height])  # Shrink the first axis width by 10%


            for freq in [ '01', '06','10','19',]:
                if freq != '01':
                    tmp = df.resample('D').mean()
                    axs[0].plot(tmp.index, tmp[f'{freq}_V'],marker='o', label=f'{freq}_V')
                else:
                    axs[0].plot(df.index, df[f'{freq}_V'],alpha=0.9,marker='o', label=f'{freq}_V')

            axs[0].set_title(site)
            axs[0].set_ylabel('BT (K)')
            axs[0].legend(title='From SMRT:', ncols=1, loc='upper left',)

            # Panel 2: LWC and DWT
            im, cbar_ax  = lp.plot_LWC(fig, axs[1], year_data)

            axs[1].plot(upper_depth_df.index, upper_depth_df,
                        label='target depth', color='tab:purple',
                        linestyle='None', marker='o', markersize=8)

            axs[1].legend(loc='lower left',markerscale=1)
            axs[1].set_ylabel('Depth (m)')
            axs[0].grid(True)
            axs[1].grid(True)
            axs[1].set_ylim(10, 0)
            axs[1].set_xlim(pd.to_datetime(f'{year}-01-01'), pd.to_datetime(f'{year}-12-31'))

            if site in ['H2', 'FA-13']:
                axs[1].set_ylim(10, 0)
                axs[1].set_xlim(pd.to_datetime(f'{year}-05-01'), pd.to_datetime(f'{year}-12-31'))

            # Final plot settings and save
            plt.show()
            fig.savefig(f'figures/water_depth_Tb_gs_and_n_ice_tuning/{site}_{year}_model_output.png', dpi=300, bbox_inches='tight')

df_all = pd.concat(df_list)
df_all[['site','latitude','longitude','T2m_avg','SF_avg', 'melt_avg' ]].drop_duplicates().to_csv('site_overview.tsv',sep='\t',index=None)
df_all.to_csv('output/water_depth_Tb_gs_and_n_ice_tuning_after_resample_new.csv')
