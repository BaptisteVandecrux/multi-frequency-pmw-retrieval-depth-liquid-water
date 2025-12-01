# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:37:11 2025

@author: bav
"""

from smrt import sensor_list, make_model
from smrt.inputs.make_medium import make_medium
from smrt.emmodel.iba import derived_IBA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime

from smrt.core.globalconstants import DENSITY_OF_ICE, DENSITY_OF_WATER


def calc_ratio(df):
    for frq in ['01', '06', '10', '19', '37']:
        # NPR = (TBV-TBH)/(TBV+TBH)
        if f'{frq}_V' in df.columns:
            df[f'{frq}_ratio'] = (df[f'{frq}_H']-df[f'{frq}_V'] )/(df[f'{frq}_H'] + df[f'{frq}_V'])
        # df[f'{frq}_ratio'] = df[f'{frq}_H'] / df[f'{frq}_V']
    return df

def calc_dry_norm_BT(df, features=['01_V', '06_V', '10_V', '19_V']):
    """
    Compute dry reference values and normalized brightness temperatures (BTs)
    for each site or grid cell, based on the median of the first 60 observations
    of each year. The normalization is done as:

        BT_norm = (BT - BT_dry) / (273.15 - BT_dry)

    For gridded data (MultiIndex with 'x' and 'y'), each (x, y) pair is treated
    as an individual site. For tabular site-based data, it uses the 'site'
    column if available or assigns a dummy one.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame indexed by time or a MultiIndex (time, y, x) containing
        brightness temperature features to normalize.
    features : list of str
        List of column names for the brightness temperatures to normalize.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional columns: <feature>_dry and <feature>_norm
        for each feature in the input list.
    """
    df_out = df.copy()

    if any([f + '_dry' not in df.columns for f in features]):
        if isinstance(df.index, pd.MultiIndex) and {'x', 'y'}.issubset(df.index.names):
            df_out['year'] = df_out.index.get_level_values('time').year
            medians = (
                df_out.groupby([pd.Grouper(level='y'), pd.Grouper(level='x'), 'year'])
                .head(60)
                .groupby(['y', 'x', 'year'])[features]
                .median()
                .rename(columns={f: f + '_dry' for f in features})
            )
            df_out = df_out.join(medians, on=['y', 'x', 'year'])
            df_out = df_out.drop(columns='year')
        else:
            drop_site = False
            if 'site' not in df_out.columns:
                df_out['site'] = 'site'
                drop_site = True

            df_out['year'] = df_out.index.year
            df_out= df_out.sort_values(["site", "year", "time"])
            medians = (
                df_out.groupby(['site', 'year'])
                .head(60)
                .groupby(['site', 'year'])[features]
                .median()
                .rename(columns={f: f + '_dry' for f in features})
            )
            medians_tail = (
                df_out.groupby(['site', 'year'])
                .tail(60)
                .groupby(['site', 'year'])[features]
                .median()
                .rename(columns={f: f + '_dry' for f in features})
            )
            medians = medians.combine_first(medians_tail)
            df_out = df_out.join(medians, on=['site', 'year'])
            df_out = df_out.drop(columns='year')

            if drop_site:
                df_out = df_out.drop(columns='site')

    for freq in ['01', '06', '10', '19', '37']:
        if f'{freq}_V_dry' in df_out.columns and f'{freq}_V' in df_out.columns:
            df_out[f'{freq}_V_norm'] = (
                (df_out[f'{freq}_V'] - df_out[f'{freq}_V_dry']) /
                (273.15 - df_out[f'{freq}_V_dry'])
            )

    return df_out




def calc_dry_norm_BT_no_time(df, features=['01_V', '06_V', '10_V', '19_V',]):
    features_max =  [v+'_max' for v in features]
    features_dry = [f+'_dry' for f in features]

    df[features_dry] = np.nan
    df[features_max] = np.nan
    dummy_site = False

    if 'site' not in df.columns:
        dummy_site=True
        df['site'] = 'site'
    for site in np.unique(df.site):
        for yr in np.unique(df.loc[(df.site==site)].year):
            df.loc[(df.year==yr)  & (df.site==site), features_dry] = \
                df.loc[(df.site==site)&(df.total_liquid_water==0),  features].mean().values
            # df.loc[(df.index.year==yr)  & (df.site==site),features_max] = \
            #     df.loc[(df.site==site)].loc[str(yr), features].max().values
    for freq in ['01', '06', '10', '19']:
        df[f'{freq}_V_norm'] = ((df[f'{freq}_V'] - df[f'{freq}_V_dry']) \
            / (273.15 - df[f'{freq}_V_dry'])).values
    if dummy_site:
        df = df.drop(columns=['site'])

    return df


def resample_firn_model_output_at_fixed_depths(data,
                                               layers = None,
                                               plot=False,
                                               num_ice_layers=None):
    if layers is None:
        layers= np.concatenate([
            np.arange(0, 10, 0.5),   # 0.5 m bins up to 10 m
            np.arange(10, 30, 1),    # 1 m bins from 10 to 30 m
            np.arange(30, 121, 10),   # 10 m bins from 30 m onward
        ])

    ds = (data.copy().set_index(level='depth')
          .rename_dims({'level': 'depth'})
          .rename_vars({'level': 'depth'}))

    depth_bins = np.concatenate([layers, [ds.depth.data[-2]],[ds.depth.data[-1]]])
    depth_bins = np.sort(depth_bins)

    # Compute layer thicknesses
    layer_thickness = np.diff(ds.depth, prepend=0)  # Prepend 0 for first element
    # Variables that need to be normalized before interpolation
    absolute_vars = ['slwc', 'snowc']
    for var in absolute_vars:
        if var in ds:
            ds[var+'_coef'] = ('depth', layer_thickness)


    # Add bin edges to ds.depth with NaN values
    depth_combined = np.union1d(ds.depth.values, depth_bins[1:])
    thickness_upper_bin = np.diff(depth_bins)[np.searchsorted(depth_bins[1:], depth_combined, side="left")]
    ds_interp = ds.reindex(depth=depth_combined)

    # Backward fill missing values
    ds_interp = ds_interp.bfill(dim='depth')

    # Compute new layer thickness after binning
    new_layer_thickness = np.diff(ds_interp.depth, prepend=0)  # Prepend 0 for first element

    # Multiply back to get absolute quantities
    for var in ds_interp:
        if var in absolute_vars:
            ds_interp[var] = ds_interp[var] * new_layer_thickness / ds_interp[var+'_coef'].values  # Convert back to absolute quantities
        else:
            ds_interp[var] = ds_interp[var] * new_layer_thickness / thickness_upper_bin

    # Reaverage based on depth bins
    ds_avg = ds_interp.groupby_bins('depth', depth_bins).sum()

    # Assign final depth values
    ds_avg['depth'] = ('depth_bins', [bin_item.right for bin_item in ds_avg['depth_bins'].values])

    for var in ['slwc', 'snic', 'snowc']:
        if var+'_coef' in ds:
            ds_avg = ds_avg.drop_vars(var+'_coef')

    # Rename dimensions and variables for clarity
    ds_avg = (ds_avg.set_index(depth_bins='depth')
                    .rename_dims({'depth_bins': 'depth'})
                    .rename_vars({'depth_bins': 'depth'}))

    # special case for snic
    # ds_avg['snic'] = 0 * ds_avg['snowc']
    # the ice thickness is allocated to the closest layer in the new depth scale
    # this ensures that the number of ice layer is approximately the same
    # before and after resampling
    # for depth, snic in ds[['depth', 'snic']].to_dataframe().iterrows():
    #     nearest_depth = ds_avg['snic'].depth.sel(depth=depth, method='nearest')
    #     ds_avg['snic'].loc[dict(depth=nearest_depth)] += snic.values[0]

    # print('creating profile with',len(ds_avg['snic']),'layers')
    # print('originally with',(ds['snic']>0).sum().item(),'ice layers')
    # print('resampled into with',(ds_avg['snic']>0).sum().item(),'ice layers')

    if num_ice_layers is not None:
        if (ds_avg['snic']>0).sum() > num_ice_layers:
            # print('too many ice layers')
            # print('assigning ice content to only',num_ice_layers,'layers')
            # Identify the top N layers with the highest snic values
            sorted_indices = ds_avg['snic'].argsort()[::-1].data  # Reverse to get descending order
            top_indices = sorted_indices[:num_ice_layers]
            icy_layers = ds_avg['depth'].isel(depth=top_indices).sortby('depth')
            non_icy_mask = ~ds_avg['depth'].isin(icy_layers['depth'])
            remaining_snic = ds_avg['snic'].where(non_icy_mask, 0)
            ds_avg['snic'] = ds_avg['snic'].where(~non_icy_mask, 0)

            # Redistribute snic values to nearest icy layer
            for depth, snic_value in remaining_snic.drop_vars('time', errors='ignore').to_dataframe().dropna().itertuples():
                nearest_icy_layer = icy_layers['depth'].sel(depth=depth, method='nearest')
                ds_avg['snic'].loc[dict(depth=nearest_icy_layer)] += snic_value

            # print('Final number of layers with ice:',(ds_avg['snic']>0).sum().item())
        elif (ds_avg['snic']>0).sum() < num_ice_layers:
            # print('not enough ice layers')
            # print('missing', num_ice_layers - (ds_avg['snic']>0).sum().item())

            ds_avg['num_ice_layers'] = ('depth', (ds_avg['snic'].data * 0) + 1)
            ds_avg['num_ice_layers'] = ds_avg['num_ice_layers'].where(ds_avg['snic'].data > 0, 0)
            ds_avg['num_ice_layers'].data = ds_avg['num_ice_layers'].data.astype(np.int32)

            # Loop to increment ice layers until the target is reached
            while ds_avg['num_ice_layers'].sum().item() < num_ice_layers:
                ice_concentration = ds_avg.snic / ds_avg.snowc / ds_avg.num_ice_layers
                ice_concentration = ice_concentration.where(np.isfinite(ice_concentration),0)
                idx = np.argmax(ice_concentration.data)  # Find index of max value
                ds_avg['num_ice_layers'].data[idx] += 1  # Increment the layer at that index

            # print('now asking SMRT to use', (ds_avg['num_ice_layers'].sum()).item(), 'ice layers')

    # special case for deep temperature: we overwrite the last temperature with the original deepest temperature
    if 'T_ice' in ds_avg.data_vars:
        ds_avg.T_ice.loc[dict(depth=ds_avg.depth[-1])] = data.T_ice.isel(level=-1)

    def plot_ds(ds,var,label, ax):
        if var in ['slwc', 'snic', 'snowc']:
            density = np.hstack(([0, ds[var].cumsum()]))
        else:
            density= np.hstack(([ds[var].data[0]], ds[var]))
        depth_edges = np.hstack(([0], ds.depth))  # Adding initial 0 depth

        ax.step(density, -depth_edges, where='post', lw=2, label=label)  # where='pre' ensures correct step alignment

    if plot:
        var_list = [v for v in ds.data_vars if not v.endswith('coef')]
        fig, axs = plt.subplots(1,len(var_list), sharey=True)
        if len(var_list) == 1:
            axs=[axs]
        for var, ax in zip(var_list,axs):
            if var in ds.data_vars:
                plot_ds(ds,var,var+' original',ax)
                plot_ds(ds_avg,var,var+' '+str(len(layers)),ax)
                ax.legend(loc='lower center')
                ax.grid()


    ds_avg['thickness'] = ds_avg.depth.copy()
    ds_avg['thickness'].attrs['long_name'] = 'Layer thickness'
    ds_avg['thickness'].attrs['units'] = 'm'
    ds_avg['thickness'].data[1:] = ds_avg['depth'].data[1:] - ds_avg['depth'].data[0:-1]

    # reintroducing level as an dimension
    ds_avg = ds_avg.rename({'depth': 'level'})
    ds_avg['depth'] = ds_avg['level']  # Keep depth as a variable
    ds_avg = ds_avg.assign_coords(level=range(0, ds_avg.sizes['level']))
    return ds_avg


def secant_method(f, x0, x1, tol=1e-6, max_iter=100, min_value=0.01, max_value=float('inf'),
                  integer=False, min_gradient=0.1):
    fx0, fx1 = f(x0), f(x1)
    iter_vals = [x0, x1]
    f_vals = [fx0, fx1]

    for _ in range(max_iter):
        gradient = fx1 - fx0
        if abs(gradient) < min_gradient:
            gradient = min_gradient * (1 if gradient >= 0 else -1)
        x_new = x1 - fx1 * (x1 - x0) / gradient
        x_new = max(min_value, min(x_new, max_value))
        if integer:
            if round(x_new) == x1:
                x_new = round(x_new) + 1
            else:
                x_new = round(x_new)
        fxnew = f(x_new)
        iter_vals.append(x_new)
        f_vals.append(fxnew)
        if abs(x_new - x1) < tol:
            # return int(x_new) if integer else x_new, iter_vals, f_vals
            break

        x0, x1, fx0, fx1 = x1, x_new, fx1, fxnew

    best_idx = min(range(len(f_vals)), key=lambda i: abs(f_vals[i]))
    best_x = iter_vals[best_idx]

    return int(best_x) if integer else best_x, iter_vals, f_vals


def find_opt_num_ice_layers(data_jan_feb, BT_obs_V, frq_list=['01']):
    # import pdb; pdb.set_trace()
    year = data_jan_feb.time.isel(time=0).dt.year.item()
    station = data_jan_feb.attrs["station"]

    def f(num_ice_layers):
        print(f'Running num_ice_layers: {num_ice_layers}')
        num_ice_layers = int(round(num_ice_layers))  # Ensure integer value

        data_in = resample_firn_model_output_at_fixed_depths(
            data_jan_feb, layers=None, num_ice_layers=num_ice_layers, plot=False)

        ds_result = run_smrt(data_in, frq_list,
                             time=data_in.time.isel(time=0))

        BT_smrt_V = ds_result.sel(polarization='V').to_dataframe()[frq_list].iloc[0]
        return np.mean(BT_smrt_V - BT_obs_V)

    try:
        num_ice_layers, iter_vals, f_vals = secant_method(f, 1, 50, tol=5e-2,
                                                min_value=1, max_value=200,
                                                max_iter=10, integer=True)
    except Exception as e:
        print(station, year, e)
        return data_jan_feb, np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    best_idx = min(range(len(f_vals)), key=lambda i: abs(f_vals[i]))
    ax.plot(range(len(f_vals)), f_vals, marker='o', linestyle='-', color='r', label='num_ice_layers')
    ax.scatter(best_idx, f_vals[best_idx], color='b', marker='*', s=150, label='Best solution')
    for i, txt in enumerate(iter_vals):
        ax.annotate(f"{txt}", (i, f_vals[i]), textcoords="offset points", xytext=(5,5), ha='right')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Difference between SMRT and observed BT (K)")
    ax.legend()
    ax.grid()
    fig.savefig(f'figures/opt/opt_{station}_num_ice_layer_{year}.png')

    num_ice_layers = int(round(num_ice_layers))  # Ensure integer value
    data_opt = resample_firn_model_output_at_fixed_depths(
        data_jan_feb, layers=None, num_ice_layers=num_ice_layers, plot=False)

    return data_opt, num_ice_layers


def find_opt_gs_factor(data_jan_feb, BT_obs_V, frq_list=['06', '10', '19'], alpha=0.3, min_step=0.05, max_step=1.0, max_gs=5.0, tol=1.0):
    year = data_jan_feb.time.isel(time=0).dt.year.item()
    station = data_jan_feb.attrs["station"]
    def f(gs_factor):
        print('running gs_factor',round(gs_factor, 2))
        data_in = data_jan_feb.copy()
        data_in['dgrain'] = data_in.dgrain * gs_factor

        ds_result = run_smrt(data_in, frq_list,
                             time=data_jan_feb.time.isel(time=0))

        BT_smrt_V = ds_result.sel(polarization='V').to_dataframe()[frq_list].iloc[0]
        return np.mean(BT_smrt_V - BT_obs_V)

    try:
        gs_factor, iter_vals, f_vals = secant_method(f, 0.1, 2, tol=5e-2, min_value=0.2, max_value=5)
    except Exception as e:
        print(station, year, e)
        return np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(f_vals)), f_vals, marker='o', linestyle='-', color='r', label='annotation: gs_factor')
    for i, gs_factor in enumerate(iter_vals):
        ax.annotate(f"{round(gs_factor, 2)}", (i, f_vals[i]), textcoords="offset points", xytext=(5,5), ha='right')

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Difference between SMRT and observed BT (K)")
    ax.legend()
    ax.grid()
    fig.savefig(f'figures/opt/opt_{station}_gs_factor_{year}.png')
    return round(gs_factor, 2)

# def find_opt_num_ice_layers(data_jan_feb, BT_obs_V, frq_list=['01'], initial_num=1, max_num=100, alpha=0.5, min_step=1):
#     year = data_jan_feb.time.isel(time=0).dt.year.item()
#     fig = plt.figure()
#     plt.axhline(BT_obs_V.iloc[0], ls='--', label='Observed BT')

#     num_ice_layers = initial_num
#     last_perf = float('inf')  # Start with a large error
#     last_step = 0  # Track last step size
#     last_data = data_jan_feb.copy()
#     while num_ice_layers <= max_num:
#         # Generate input data with the current number of ice layers
#         data_in = resample_firn_model_output_at_fixed_depths(
#             data_jan_feb, layers=None, num_ice_layers=num_ice_layers, plot=False)

#         # Run SMRT model
#         ds_result = run_smrt(data_in, frq_list, time=data_in.time.isel(time=0))
#         BT_smrt_V = ds_result.sel(polarization='V').to_dataframe()[frq_list].iloc[0]

#         # Compute error
#         perf = abs(np.mean(BT_smrt_V - BT_obs_V))

#         print(f"num_ice_layers: {num_ice_layers}, performance: {perf}")
#         if perf > last_perf:
#             # If performance worsens, take a half-step back before stopping
#             num_ice_layers -= max(min_step, last_step // 2)
#             break

#         # Euler update step: adjust num_ice_layers based on performance difference
#         step = max(min_step, int(alpha * abs(BT_obs_V.iloc[0] - BT_smrt_V.iloc[0])))
#         last_step = step  # Store step size for potential rollback
#         num_ice_layers += step

#         plt.plot(num_ice_layers, BT_smrt_V, marker='o', label=f'SMRT num_ice_layers: {num_ice_layers}')
#         last_perf = perf  # Update best performance
#         last_data = data_in.copy()

#     fig.savefig(f'figures/opt_{data_jan_feb.attrs["station"]}_num_ice_layer_{year}.png')

#     return last_data, num_ice_layers


def make_snowpack_from_model_output(data, depth_lim,
                                    num_ice_layers_per_layer=1,
                                    inject_water=False,
                                    layer_with_water=None,
                                    total_liquid_water=None):
    data = data.where(data.depth<depth_lim, drop=True)
    sp = pd.DataFrame(dict(medium='snow',
                      thickness=data.thickness.values.astype(np.float64),
                      microstructure_model="sticky_hard_spheres",
                      stickiness=0.17,
                      radius=data.dgrain.values.astype(np.float64) / 2 * 1e-3,
                      dry_density=data.rho.values.astype(np.float64),
                      temperature=data.T_ice.values.astype(np.float64)))
    sp.loc[sp.index[-1], 'thickness'] = 10000
    sp.loc[sp.index[-1], 'radius'] = 0.3e-3

    initial_thickness = sp['thickness'].sum()

    ice_thickness_in_layer = data.snic * 1000 / 917  # convert from m water eq to m ice eq

    # # method #1: ice layers are 0.02 cm and their number are calcuated based on ice_thickness
    # # thickness of the ice layers
    # ice_layer_thickness = 0.02
    # # number of ice layers in each layer
    # n_ice_layer = ice_thickness / ice_layer_thickness

    # method #2: we only create N extra layer of ice to receive the content of ice_layer
    if 'num_ice_layers' in data.data_vars:
        n_ice_layer = data.num_ice_layers.data
    else:
        n_ice_layer = np.ones_like(ice_thickness_in_layer) * num_ice_layers_per_layer
        n_ice_layer[ice_thickness_in_layer==0] = 0

    sp['n_ice_layer'] = n_ice_layer
    if inject_water:
        sp['density'] = sp.dry_density
        # convert water from m to kg/m2 then to kg/m3  then to m3 of water than devide by volume of ice
        # liquid_water is the volume of water per volume of ice
        sp['volumetric_liquid_water'] = data.slwc.values.astype(float) * 0
        if layer_with_water<sp.index.max():
            water_density = total_liquid_water / sp.loc[layer_with_water, 'thickness']  # kg/m3
            water_volume = water_density / DENSITY_OF_WATER
            dry_density = sp.loc[layer_with_water, 'dry_density']
            ice_volume = sp.loc[layer_with_water, 'dry_density'] / DENSITY_OF_ICE
            if water_volume + ice_volume > 1:
                # the layer is saturated, we remove some ice...
                ice_volume = 1 - water_volume
                dry_density = ice_volume * DENSITY_OF_ICE
            sp.loc[layer_with_water, 'density'] = dry_density + water_density
            sp.loc[layer_with_water, 'volumetric_liquid_water'] = water_volume

        sp.loc[sp.volumetric_liquid_water>0,'temperature'] = 273.15
        sp.loc[sp.volumetric_liquid_water==0,'temperature'] = np.minimum(273.15, sp.loc[sp.volumetric_liquid_water==0,'temperature'] )
    else:
        # convert water from m to kg/m2 then to kg/m3  then to m3 of water than devide by volume of ice
        # liquid_water is the volume of water per volume of ice
        sp['liquid_water'] = data.slwc * 1000 / sp.thickness / 1000 / (data.rho / 917)
        sp.loc[sp.liquid_water<0,'liquid_water'] = 0
        sp.loc[sp.liquid_water>0,'temperature'] = 273.15
        sp.loc[sp.liquid_water==0,'temperature'] = np.minimum(273.15, sp.loc[sp.liquid_water==0,'temperature'] )
        sp['density'] = sp.dry_density + data.slwc * 1000 / sp.thickness

    for i, n in enumerate(n_ice_layer):
        n = round(float(n))
        # guess the size of the snow vs ice layers
        snow_thickness = sp.loc[i].thickness - ice_thickness_in_layer.loc[i]
        if snow_thickness <= 0:  # too many layers...
            snow_thickness = sp.loc[i].thickness / 4  # snow is 1/4 of the ice

        # insert the layers
        for j in range(n):

            iice = i + 1e-4 * (j + 1)
            isnow = iice + 0.5e-4

            sp.loc[iice] = sp.loc[i].copy()  # the ice layer
            sp.loc[iice, 'density'] = 912
            if inject_water:
                sp.loc[iice, 'volumetric_liquid_water'] = 0
            else:
                sp.loc[iice, 'liquid_water'] = 0
            sp.loc[iice, 'radius'] = 0.3e-3
            sp.loc[iice, 'thickness'] = ice_thickness_in_layer.loc[i] / n

            sp.loc[isnow] = sp.loc[i].copy() # the same as the inital snow layer above
            sp.loc[isnow, 'thickness'] = snow_thickness / (n + 1)
        sp.loc[i, 'thickness'] = snow_thickness / (n + 1)
    sp = sp.sort_index()

    # import pdb; pdb.set_trace()
    # assert np.allclose(sp['thickness'].sum(), initial_thickness)
    # print((sp.density==912).sum().item(),'ice layers')
    sp['depth'] =  sp.thickness.cumsum()

    return sp, make_medium(sp)


def plot_sp(sp, var, title, label=[]):
    label_list = dict(density='density (kg m-3)',
                      radius='radius (m)')
    if len(label)==0:
        label = label_list[var]
    fig = plt.figure()
    plt.stairs(sp[var],
               np.insert(sp.depth.values,0,0))
    plt.xlabel('depth (m)')
    plt.ylabel(label)
    plt.grid()
    plt.xlim(0,30)
    plt.title(title)
    return fig


def run_smrt(data_in,
                frq_list,
                time,
                num_ice_layers_per_layer=1,
                parallel_computation=False,
                inject_water=False,
                layer_with_water=None,
                total_liquid_water=None):
    # frq_list = ['01', '06', '10', '19']
    depth_dic = {'01':1000, '06':30, '10':20, '19':10}
    list_da = []
    for frq in frq_list:
        sp, snowpack = make_snowpack_from_model_output(data_in,
                                                        depth_dic[frq],
                                                        num_ice_layers_per_layer,
                                                        inject_water=inject_water,
                                                        layer_with_water=layer_with_water,
                                                        total_liquid_water=total_liquid_water)

        if frq == '01':
            radiometer = sensor_list.smos(theta=40)
        else:
            radiometer = sensor_list.amsre([frq+'V',frq+'H'])

        # create the model including the scattering model (IBA) and the radiative transfer solver (DORT)
        m = make_model(derived_IBA(), "dort",
                       emmodel_options=dict(dense_snow_correction="auto"),
                       rtsolver_options=dict(
                           diagonalization_method="shur_forcedtriu",
                           phase_normalization='forced',
                           prune_deep_snowpack=6))

        result = m.run(radiometer, snowpack,
                     parallel_computation=parallel_computation)

        da = xr.Dataset()
        da[frq] = result.Tb()
        da = da.assign_coords(time = time)
        da = da.expand_dims(dim="time")
        # da['ka'] = (('frq', 'time'), result.other_data['ka'].rename('ka')
        # da['ks'] = result.other_data['ks']
        da['site'] = data_in.attrs['station']
        da['latitude'] = data_in.attrs['latitude']
        da['longitude'] = data_in.attrs['longitude']
        if "theta" in da.data_vars:
            da = da.drop_vars("theta")
        list_da.append(da)

    return xr.merge(list_da, compat='override')
