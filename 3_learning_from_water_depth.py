# -*- coding: utf-8 -*-
"""
Depth of Liquid Water Prediction using Random Forest Regression

This script trains and applies a Random Forest regression model to predict the 
depth of subsurface liquid water (DLW) on the Greenland ice sheet using SMRT 
simulated brightness temperatures and normalized features. It performs leave-one-year-out
cross-validation and saves one model per test year. It also provides inference of
mean and standard deviation of DLW on new time series.

The workflow includes:
- Preprocessing and normalization of multi-frequency TB
- Weighting of training samples to balance depth classes
- Model training and evaluation per year
- Application to new PMW time series data
- Visualization of predictions alongside brightness temperature observations

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

import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import joblib
from datetime import datetime  # To get the current date
import lib.smrt_functions as sf
import lib.plot as lp
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm
import os
# Load the data
df = pd.read_csv('output/water_depth_Tb_gs_and_n_ice_tuning_after_resample.csv')
# df = pd.read_csv('output/water_depth_Tb_gs_and_n_ice_tuning_wrong_time.csv')

df = sf.calc_ratio(df)

# Prepare the dataset
features_start = ['01_V', '06_V', '10_V', '19_V']

if 'time' in df.columns:
    df['time'] = pd.to_datetime(df.time)
    df = df.set_index('time')
    df = sf.calc_dry_norm_BT(df, features_start)
    df = df.reset_index()
else:
    df = sf.calc_dry_norm_BT_no_time(df, features_start)

features = ['01_V', '06_V', '10_V', '19_V',
            '01_V_norm', '06_V_norm', '10_V_norm', '19_V_norm']

year_cv = 2021

# df = df.loc[df.total_liquid_water > 0, :]
df.loc[df.total_liquid_water==0, 'depth_water'] = 0
df = df[['depth_water', 'site','year', 'total_liquid_water']+features].dropna()

if True:
    current_date = datetime.now().strftime("%Y%m%d")
    # best_params
    params = {'max_depth': 15,
      'min_samples_leaf': 20,
      'min_samples_split': 2,
      'n_estimators': 100}

    suffix = "_".join(f"{k}-{v}" for k, v in params.items())
    print(suffix)
    ensemble_name = f"rf_depth_model_ensemble_{current_date}_{suffix}"
    os.makedirs(f'output/{ensemble_name}', exist_ok=True)
    os.makedirs(f'figures/RF/{ensemble_name}', exist_ok=True)

    # Calculate the percentage of data for each site
    site_counts = df['site'].value_counts()
    total_count = len(df)
    site_percentages = (site_counts / total_count) * 100

    print('Features:')
    print(features)
    X = df[features]
    y_depth = df['depth_water']
    y_tlw = df['total_liquid_water']

    df_training = df.copy()

    # Split the data into training and testing based on year
    # for year in [year_cv]:
    for year in range(2010,2024):
        print(year)
        test = df[df['year'] == year]
        train = df[df['year'] != year]
        msk_train = (train['total_liquid_water']>0)
        train = train.loc[msk_train]
        X_depth_train = train[features]
        y_depth_train = train['depth_water']

        msk_test = (test['total_liquid_water']>0)
        test = test.loc[msk_test]
        X_depth_test = test[features]
        y_depth_test = test['depth_water']

        # % defining the weights
        bins = np.concatenate([np.arange(np.floor(y_depth_train.min()), 20, 1), [50]])
        bin_labels = pd.cut(y_depth_train, bins=bins, include_lowest=True)

        bin_counts = bin_labels.value_counts()
        bin_weights = 1.0 / len(bin_counts) / bin_counts
        weights_depth = bin_labels.map(bin_weights).values

        # fig, ax2 = plt.subplots(1, 1, figsize=(10, 4))
        # ax2.plot(y_depth_train, weights_depth, marker='o', ls='None', label='Sample weights')
        # ax2.set_yscale('log')
        # ax2.set_xlabel('y_depth_train')
        # ax2.set_ylabel('Weight')
        # ax2_twin = ax2.twinx()
        # ax2_twin.hist(y_depth_train, bins=50, alpha=0.3, color='gray', label='Histogram', log=True)
        # ax2_twin.set_ylabel('Count')

        # hyperparameters that have been tested
        # param_grid = {
        #     'n_estimators': [100, 200, 300, 500, 700, 1000],
        #     'max_depth': [10, 15, 20, 30, 50, 100, 150, 500, 1000],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 5, 10, 20, 50, 100]
        # }

        # from lib.rf_functions import run_hyperparameter_search
        # best_params = run_hyperparameter_search(param_grid, X_depth_train, y_depth_train, weights_depth, test, train, features, year, lp)

        rf_depth = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        rf_depth.fit(X_depth_train, y_depth_train, sample_weight=weights_depth)

        X_test = test[features]
        y_depth_test = test['depth_water']
        y_depth_pred = rf_depth.predict(X_test)
        y_depth_pred_train = rf_depth.predict(X_depth_train)

        model_path = f"output/{ensemble_name}/rf_depth_model_hold_out_{year}.joblib"
        joblib.dump(rf_depth, model_path)

        mse_depth = mean_squared_error(y_depth_test, y_depth_pred)
        print(f'Mean Squared Error for depth prediction: {mse_depth}')

        test_results = test.copy()
        test_results['depth_water_pred'] = y_depth_pred

        train_results = train.copy()
        train_results['depth_water_pred'] = np.nan
        train_results.loc[X_depth_train.index, 'depth_water_pred'] = y_depth_pred_train

        lp.plot_evaluation_scatter(train_results, test_results, year, lp, f'figures/RF/{ensemble_name}/')

# %% applying to a time series
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lib.load as ll
import lib.plot as lp
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import joblib
from datetime import datetime  # To get the current date
import lib.smrt_functions as sf
import lib.plot as lp


features_start = ['01_V', '06_V', '10_V', '19_V']
features = ['01_V', '06_V', '10_V', '19_V',
            '01_V_norm', '06_V_norm', '10_V_norm', '19_V_norm']

#  run for all years
# for site in ['H2','KAN_U','FA-13','DYE-2']:
df_list=[]

df_sumup, df_meta = ll.load_sumup_temperature()

# %%
def predict_mean_std(df):
    preds = []
    for year_cv in range(2010, 2024):
        model_filename = f'output/rf_depth_model_ensemble_20250619_max_depth-15_min_samples_leaf-20_min_samples_split-2_n_estimators-100/rf_depth_model_hold_out_{year_cv}.joblib'
        model = joblib.load(model_filename)
        preds.append(model.predict(df))
    pred_array = np.stack(preds)

    df_out =pd.DataFrame({
        "depth_mean": pred_array.mean(axis=0),
        "depth_std": pred_array.std(axis=0)
    }, index=df.index)

    return df_out

for site in ['DYE-2']:
    # years = range(2013, 2024)
    if site == 'FA-13':
        years = [2013]
    # elif site == 'CP1':
    #     years = [2021]
    # elif site == 'DYE-2':
    #     years = [2021]
    elif site == 'FA-15-1':
        years = [2015]
    else:
        years = [2016,2019,2021,2022,2023]

    data = ll.load_snow_model_output(site)
    df_sumup_selec, df_meta_selec = ll.select_sumup(df_sumup,
                                                     df_meta,
                                                     data.latitude,
                                                     data.longitude)
    df_pmw = ll.load_pmw_site(site, data.attrs['latitude'], data.attrs['longitude'])

    df_pmw = sf.calc_dry_norm_BT(df_pmw, features_start)

    df_pmw = df_pmw[features].dropna()
    df_pmw_smooth = df_pmw[features].copy()

    for var in ['01_V','01_V_norm', '06_V','06_V_norm',
            '10_V','10_V_norm', '19_V','19_V_norm',]:
        window = 7
        threshold = 10
        smoothed = df_pmw[var].rolling(window=window, center=True,
                         win_type='gaussian').mean(std=3)
        peaks = df_pmw[var].where(abs(df_pmw[var]-smoothed) > threshold)
        df_pmw_smooth[var]  = peaks.combine_first(smoothed)

    df_pred = predict_mean_std(df_pmw_smooth)
    # %
    count = 0
    for i, year in enumerate(years):
        print(year)
        year_data = data.sel(time=slice(f'{year}-01-01', f'{year+1}-05-01'))
        # %
        lp.plot_result(year_data, df_pmw,
                       df_pmw_smooth,
                        df_pred, df_sumup_selec,
                        slope_threshold = -1.5, mode='pmw')

    # %%
    df_2021 = df_pmw.loc['2021']
    plt.close('all')

    mask_8 = df_2021.index.hour == 8
    mask_17 = df_2021.index.hour == 17

    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green']
    plt.figure()
    df_2021[['01_V','06_V','10_V','19_V']][mask_8].plot(ax=plt.gca(), marker='o', linestyle='', label='08:00', color=colors)
    df_2021[['01_V','06_V','10_V','19_V']][mask_17].plot(ax=plt.gca(), marker='x', linestyle='', label='17:00', color=colors)


    plt.legend()

    may_mask = df_2021.index.month == 1

    std_all = df_2021[['01_V','06_V','10_V','19_V']].loc[may_mask].std()
    std_8 = df_2021[['01_V','06_V','10_V','19_V']].loc[may_mask & mask_8].std()
    std_17 = df_2021[['01_V','06_V','10_V','19_V']].loc[may_mask & mask_17].std()

(df_2021[['01_V','06_V','10_V','19_V']].loc[may_mask & mask_8].resample('D').first() -\
    df_2021[['01_V','06_V','10_V','19_V']].loc[may_mask & mask_17].resample('D').first()).mean()
# %% Corrected and uncorrected in single figure
plot = False
if plot == True:
    # Create a single figure with two panels
    for i, freq in enumerate(['01', '06', '10', '19', '37']):
        fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        fig.suptitle(f'TB at {freq} GHz (V polarization) modelled and observed at {site}')

        # Panel 1: Raw TB data
        ax_raw = axs[0]
        if freq+'_V' in df.columns:
            ax_raw.plot(df.index, df[freq+'_V'], c='tab:red', alpha=0.7, label='Modelled')
        if freq+'_V' in df_pmw.columns:
            ax_raw.plot(df_pmw.index, df_pmw[freq+'_V'], c='tab:green', alpha=0.7, label='Observed')
        ax_raw.set_ylabel(freq+' GHz (Raw)')
        ax_raw.legend(loc='upper right')

        # Panel 2: Normalized TB data
        ax_norm = axs[1]
        if freq+'_V_norm' in df.columns:
            ax_norm.plot(df.index, df[freq+'_V_norm'], c='tab:red', alpha=0.7, label='Modelled')
        if freq+'_V_norm' in df_pmw.columns:
            ax_norm.plot(df_pmw.index, df_pmw[freq+'_V_norm'], c='tab:green', alpha=0.7, label='Observed')
        ax_norm.set_ylabel(freq+' GHz (Normalized)')
        ax_norm.set_xlabel('Time')

        # Adjust layout for clarity
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Show the figure
        plt.show()
