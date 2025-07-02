# -*- coding: utf-8 -*-
"""
Gridded DLW Prediction using Trained Random Forest Ensemble

This script applies an ensemble of trained Random Forest models to preprocessed 
passive microwave brightness temperature data (SMOS/AMSR) across the Greenland 
ice sheet to estimate the depth of subsurface liquid water (DLW). It performs 
per-pixel inference in parallel using multiprocessing and writes yearly gridded 
outputs of mean and standard deviation.

Workflow includes:
- Loading preprocessed PMW features for each year
- Parallel batch prediction using all trained models (2010–2023)
- Outputting NetCDF files with compressed encoded predictions

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
import lib.load as ll
from multiprocessing import Process, Queue
from tqdm import tqdm

features = ['01_V', '06_V', '10_V', '19_V',
            '01_V_norm', '06_V_norm', '10_V_norm', '19_V_norm']

read_from_scratch = False
if read_from_scratch:
    for year in range(2010,2024):
        print('######', year)
        ds_pmw = ll.load_pmw_all(year=year)
        final_dataset = ll.prepare_ds_pmw(ds_pmw, filter=True)
        final_dataset = final_dataset.sel(time=final_dataset.time.dt.hour == 9)
        final_dataset['time'] = final_dataset.time.dt.floor('D')
        ds_melt = ll.clip_firn_area(ll.load_melt_xr(year, only_smos=True))
        ds_melt_daily = (ds_melt.snow_status_wet_dry_01V == 1).resample(time="1D").any()
        # ds_merged = ds_merged.sel(time=slice(ds_melt_daily.time[0],ds_melt_daily.time[-1]))

        ds_melt_daily = ds_melt_daily.sel(time=slice(final_dataset.time[0],final_dataset.time[-1]))
        final_dataset = final_dataset.where(ds_melt_daily == 1)
        encoding = {var: {'dtype': 'int16', 'scale_factor': 1e-4 if "_norm" in var else 0.01,
                          'zlib': True, 'complevel': 4, '_FillValue': -32768} for var in final_dataset.data_vars}
        final_dataset.to_netcdf(f'data/PMW grids/pre_processed/pmw_data_formatted_{year}.nc', encoding=encoding)
# else:
#     final_dataset = xr.read_dataset('data/PMW grids/pmw_data_formatted.nc')


# %%
models = [joblib.load(f'output/rf_depth_model_ensemble_20250619_max_depth-15_min_samples_leaf-20_min_samples_split-2_n_estimators-100/rf_depth_model_hold_out_{year_cv}.joblib')
          for year_cv in range(2010, 2024)]

def predict_mean_std(df, models):
    preds = [model.predict(df) for model in models]
    pred_array = np.stack(preds)

    df_out =pd.DataFrame({
        "depth_mean": pred_array.mean(axis=0),
        "depth_std": pred_array.std(axis=0)
    }, index=df.index)

    return df_out


def process_batch(batch, queue, worker_id, feature_data):
    results = []
    with tqdm(total=len(batch), desc=f"Worker {worker_id}", position=worker_id) as pbar:
        for idx in batch:
            pixel_data = feature_data[:, idx[0], idx[1], idx[2]]
            pixel_df = pd.DataFrame([pixel_data], columns=features)
            df_pred = predict_mean_std(pixel_df, models)
            results.append((tuple(idx), df_pred.depth_mean[0], df_pred.depth_std[0]))
            pbar.update(1)
    queue.put(results)

for year in range(2010, 2024):
    print(year)
    final_dataset = xr.open_dataset(
        f'data/PMW grids/pre_processed/pmw_data_formatted_{year}.nc'
    )

    # Extract feature data directly from the xarray dataset
    feature_data = final_dataset[features].to_array(dim="feature").data

    coords = np.array(list(np.ndindex(final_dataset.sizes['time'], final_dataset.sizes['y'], final_dataset.sizes['x'])))
    valid_mask = ~np.all(np.isnan(feature_data), axis=0)
    valid_coords = coords[valid_mask.ravel()]

    # Split into 7 batches
    num_workers = 23
    batches = np.array_split(valid_coords, num_workers)

    # Multiprocessing setup
    queue = Queue()
    processes = []

    for i, batch in enumerate(batches):
        p = Process(target=process_batch, args=(batch, queue, i, feature_data))
        processes.append(p)
        p.start()

    # Initialize output arrays
    predicted_depth = np.full(final_dataset[features[0]].shape, np.nan, dtype=np.float32)
    predicted_std = np.full(final_dataset[features[0]].shape, np.nan, dtype=np.float32)

    # Collect worker results
    for _ in range(num_workers):
        worker_results = queue.get()
        for idx, mean_val, std_val in worker_results:
            predicted_depth[idx] = mean_val
            predicted_std[idx] = std_val


    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Convert to xarray DataSet
    rf_output = xr.Dataset({
        'predicted_depth': (['time', 'y', 'x'], predicted_depth),
        'predicted_std': (['time', 'y', 'x'], predicted_std)
    }, coords={
        'time': final_dataset.coords['time'],
        'y': final_dataset.coords['y'],
        'x': final_dataset.coords['x']
    })

    encoding = {
                "predicted_depth": {
                    "dtype": "uint16",
                    "scale_factor": 0.001,
                    "add_offset": 0,
                    "zlib": True,
                    "complevel": 9,
                    "_FillValue": 65535
                },
                "predicted_std": {
                    "dtype": "uint16",
                    "scale_factor": 0.001,
                    "add_offset": 0,
                    "zlib": True,
                    "complevel": 9,
                    "_FillValue": 65535
                }
    }

    # predicted_depth_da = predicted_depth_da.where(final_dataset.melt_01 == 1, 0)
    import os
    output_folder = 'output/rf_depth_model_ensemble_20250619_max_depth-15_min_samples_leaf-20_min_samples_split-2_n_estimators-100'
    os.makedirs(output_folder, exist_ok=True)
    filename = f'{output_folder}/rf_output_{year}.nc'
    rf_output.to_netcdf(filename, encoding=encoding)
    print('Done. Wrote results to:', f'{filename}')
