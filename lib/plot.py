# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import warnings
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import lib.load as ll

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Create a custom colormap
cmap = plt.get_cmap('winter')
cmap_colors = cmap(np.arange(cmap.N))
# Set the first color (for slwc == 0) to white
cmap_colors[0] = np.array([1, 1, 1, 1])
custom_cmap = mpl.colors.ListedColormap(cmap_colors)


def plot_result(year_data, df_BT, df_BT_smooth,
                df_pred,  df_sumup_selec,
                mode='pmw', tag=''):
    year = year_data.time.isel(time=0).dt.year.item()
    site= year_data.attrs['station']

    # loading PMW melt indexes
    ds_smos = ll.load_smos_site(year_data.attrs['station'],
                                year_data.attrs['latitude'],
                                year_data.attrs['longitude'])
    ds_smos = ds_smos.where(ds_smos>0).loc[str(year)]
    ds_smos = ds_smos.resample('1D').max()
    ds_smos.index = ds_smos.index + pd.to_timedelta(9, unit='h')

    ds_amsr = ll.load_amsr_site(year_data.attrs['station'],
                                year_data.attrs['latitude'],
                                year_data.attrs['longitude'])
    # ds_amsr=ds_amsr.where(ds_amsr>0).loc[str(year)]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(11, 5), gridspec_kw={'height_ratios': [1, 2]})
    fig.subplots_adjust(hspace=0.08,bottom=0.05,top=0.95)

    labels = ['1.4 GHz', '6.9 GHz', '10.7 GHz', '18.7 GHz']
    colors = ['tab:blue', 'tab:orange','tab:green', 'tab:red']
    features = []
    for i, freq in enumerate(['01', '06', '10', '19']):
        if freq+'_V' in df_BT.columns:
            axs[0].plot(df_BT.index, df_BT[freq+'_V'], alpha=0.3, label='_'+freq,
                        c=colors[i])
            axs[0].plot(df_BT.index, df_BT_smooth[freq+'_V'],
                        c=colors[i], label=labels[i],marker='.',
                        )
            features.append(freq+'_V')
    mask = (df_BT.index >= f'{year}-04-01') & (df_BT.index <= f'{year}-11-30')
    ymin = df_BT.loc[mask, features].min().min()
    max_tb = 340
    if np.isnan(ymin): ymin = 160
    axs[0].set_ylim(ymin, max_tb)

    import matplotlib.patheffects as pe

    def add_text(x, y, t, c, fontsize=10):
        axs[0].text(
            x, y, t, color=c, fontsize=9, ha='left', va='top', transform=axs[0].transAxes,
            path_effects=[pe.withStroke(linewidth=3, foreground='white')]
        )

    add_text(0.07, 0.97, "wet surface at 18.7 GHz:", 'tab:red')
    axs[0].plot(ds_amsr.snow_status_wet_dry_19_V*(ymin + (max_tb-ymin)*0.94),
                marker='o',ls='None',c='tab:red')

    if (ds_amsr.snow_status_wet_dry_06_V.loc[
            slice(f'{year}-04-01', f'{year+1}-01-01')]==1).any():
        add_text(0.07, 0.88, "wet surface at 6.9 GHz:", 'tab:orange')
        axs[0].plot(
            ds_amsr.snow_status_wet_dry_06_V*(ymin + (max_tb-ymin)*0.85),
            marker='o',ls='None',c='tab:orange')

    add_text(0.07, 0.77, "wet surface at 1.4 GHz:", 'tab:blue')
    axs[0].plot(
        ds_smos.snow_status_wet_dry_smos*(ymin + (max_tb-ymin)*0.74),
        marker='o',ls='None',c='tab:blue')


    if site =='FA-15-1':
        axs[0].text(0.01, 0.93, '(c)', transform=axs[0].transAxes, fontsize=12, weight='bold', va='top', ha='left')
    else:
        axs[0].text(0.01, 0.93, '(a)', transform=axs[0].transAxes, fontsize=12, weight='bold', va='top', ha='left')
    axs[0].set_ylabel('Observed V-pol $\mathrm{T_B}$ (K)')
    axs[0].legend(loc='upper right',
                      fontsize=9)
    # ========== second pannel ============
    ax1 = axs[1]

    # plotting subsurface temp
    if isinstance(df_sumup_selec, pd.DataFrame):
        count = df_sumup_selec[(df_sumup_selec.timestamp >= pd.to_datetime(f'{year}-04-01')) & \
                               (df_sumup_selec.timestamp <= pd.to_datetime(f'{year+1}-01-01'))].shape[0]

        if count>1:
            plot_obs_temperature(fig, axs[1], df_sumup_selec, box2=None, alpha=0.6)

    # UDLW from RF(TB_obs)
    flag = df_pred.depth_mean.copy().astype(bool)

    flag.iloc[:] = False
    if ds_smos.snow_status_wet_dry_smos.first_valid_index() and \
            ds_smos.snow_status_wet_dry_smos.last_valid_index():
        valid_idx = ds_smos.snow_status_wet_dry_smos.loc[
            ds_smos.snow_status_wet_dry_smos.notnull()].index
        common_idx = flag.index.intersection(valid_idx)
        flag.loc[common_idx] = True

    # Inflate and shrink by one pixel
    inflated = flag.copy()
    inflated.iloc[1:] |= flag.iloc[:-1].values
    inflated.iloc[:-1] |= flag.iloc[1:].values
    shrunk = inflated.copy()
    shrunk.iloc[1:] &= inflated.iloc[:-1].values
    shrunk.iloc[:-1] &= inflated.iloc[1:].values
    shrunk.loc[df_pred.depth_std>1] = False

    # Plot
    axs[1].plot(df_pred.index, df_pred.depth_mean,
                color='gray',
                marker='o',markersize=5,ls='None',
                # linewidth=2,
                zorder=3, label='1.4 GHz dry or st. dev. > 1')
    tmp = df_pred.depth_mean.copy(deep=True)
    tmp.loc[~shrunk] = np.NaN
    tmp.loc[df_pred.depth_std > 0.5] = np.NaN
    tmp.loc[df_pred.depth_mean > 5] = np.NaN
    axs[1].plot(tmp.index, tmp.values,
                marker='o',markersize=10,ls='None',alpha=0.7,
                # linewidth=3,
                color='w',zorder=4, label='__nolegend__')
    axs[1].plot(tmp.index, tmp.values,
                marker='o',markersize=6,ls='None',
                # linewidth=3,
                color='k',zorder=4, label='1.4 GHz wet and st. dev. < 1')

    axs[1].fill_between(df_pred.index,
                        df_pred.depth_mean - df_pred.depth_std,
                        df_pred.depth_mean + df_pred.depth_std,
                        color='gray', alpha=0.4, zorder=0,
                        label='± 1 standard deviation')

    axs[1].plot([], [], 'o', color='w', label='   ')

    # UDLW in GEUS model
    thickness = year_data.depth.copy()
    thickness.data[:,1:] = year_data.depth.diff(dim='level', label='upper')
    top_depth = year_data.depth.copy()
    top_depth.data[:, :-1] = year_data.depth.data[:, :-1] - thickness.data[:, :-1]

    # Get top depth where slwc > 0
    # uppermost_top_depth = top_depth.where(year_data.slwc > 0).min(dim='level').to_series()
    # uppermost_top_depth.plot(ax=axs[1],
    #                          color='tab:pink',
    #                          ls='None',
    #                              label='UDLW in the GEUS snow model',
    #                               lw=3, zorder=1)

    axs[1].plot([], [], 'o', color='tab:red', label='thermistors with temperature >-0.2 °C')

    axs[1].set_ylabel('Depth (m)')

    for ax in axs:
        ax.grid(True)

    legend = axs[1].legend(title='Mean UDLW from the RF model ensemble:', loc='lower left',
                  fontsize=9, title_fontsize=9, framealpha=1)
    legend._legend_box.align = "left"
    axs[1].set_ylim(5, -0.2)
    axs[1].set_xlim(pd.to_datetime(f'{year}-04-01'), pd.to_datetime(f'{year}-11-30'))
    if site in ['H2', 'FA-13', 'T2_09','FA-15-1', 'KAN_U']:
        axs[1].set_ylim(10, 0)
        axs[1].set_xlim(pd.to_datetime(f'{year}-02-01'), pd.to_datetime(f'{year+1}-03-01'))
    if site in ['QAS_U']:
        axs[1].set_ylim(5, 0)
        axs[1].set_xlim(pd.to_datetime(f'{year}-02-01'), pd.to_datetime(f'{year}-08-01'))
    if site in ['CP1']:
        axs[1].set_ylim(10, 0)
        axs[1].set_xlim(pd.to_datetime(f'{year}-05-15'), pd.to_datetime(f'{year}-11-01'))

    plt.suptitle(site)

    if site =='FA-15-1':
        axs[1].text(0.01, 0.96, '(d)', transform=ax.transAxes, fontsize=12, weight='bold', va='top', ha='left')
    else:
        axs[1].text(0.01, 0.96, '(b)', transform=ax.transAxes, fontsize=12, weight='bold', va='top', ha='left')
    ax.tick_params(axis='x', rotation=45)

    fig.savefig(f'figures/RF_evaluation/{site}_{tag}{year}_dlw'\
                + ('_obs' if mode=='pmw' else '_mod') \
                        +'.jpeg', dpi=300, bbox_inches='tight')


def plot_LWC(fig, ax, year_data, box1=None):
    if box1 == None:
        box1 = ax.get_position()
    im = ax.pcolormesh(
        year_data.time.expand_dims(
            dim={"level": year_data.level.shape[0] + 1}).transpose(),
        np.hstack([year_data.surface_height.values.reshape([-1, 1]),
                   year_data.depth.values]),
        year_data.slwc.isel(time=slice(1, None)),
        shading='flat',
        cmap=custom_cmap, vmin=0, vmax=5,
        zorder=0
    )

    # Add colorbar outside the first subplot without affecting alignment
    cbar_ax = fig.add_axes([box1.x0 + box1.width * 1.01, box1.y0, 0.02, box1.height])
    plt.colorbar(im, cax=cbar_ax, label='LWC ($\mathrm{kg m^{-3}}$)')
    return im, cbar_ax


def plot_obs_temperature(fig, ax, df_sumup_selec, box2=None, alpha=1):
    if box2 is None:
        box2 = ax.get_position()

    # Temperature bounds with 0.2°C increments
    bounds = np.arange(-10, 0.2, 0.2)

    # Base colormap (excluding the red section)
    base_cmap = plt.get_cmap('summer', len(bounds) - 1)
    colors = base_cmap(np.linspace(0, 1, len(bounds) - 1))

    # Replace color for -0.2 to 0°C with red
    colors[-1] = mcolors.to_rgba('red')

    # Create the custom colormap
    cmap = ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

    # Scatter plot with the custom colormap
    sc = ax.scatter(df_sumup_selec.timestamp,
                    df_sumup_selec['depth'],
                    25,
                    df_sumup_selec['temperature'],
                    cmap=cmap, norm=norm,
                    alpha=alpha,
                    zorder=1, label='_thermistor')

    # Highlight temperatures > -0.2°C
    # T_crit = -0.2
    # ax.plot(df_sumup_selec.loc[df_sumup_selec['temperature'] > T_crit, 'timestamp'],
    #         df_sumup_selec.loc[df_sumup_selec['temperature'] > T_crit, 'depth'],
    #         marker='d', ls='None', markersize=5, alpha=0.7,
    #         color='tab:red', zorder=2, label=f'thermistor > {T_crit} °C')

    # Custom colorbar
    cbar_ax = fig.add_axes([box2.x0 + box2.width * 1.01, box2.y0, 0.02, box2.height])
    cbar = plt.colorbar(sc, cax=cbar_ax, label='Temperature (°C)', boundaries=bounds, ticks=[])

    # Major ticks at -10, -8, -6, -4, -2, 0
    major_ticks = np.arange(-10, 1, 2)
    cbar.ax.set_yticks(major_ticks)
    cbar.ax.set_yticklabels(major_ticks, fontsize=8)

    # Minor ticks for each 0.2°C increment
    cbar.ax.set_yticks(bounds, minor=True)
    cbar.ax.tick_params(which='minor', length=3)  # Smaller ticks for minor grid
    cbar.ax.tick_params(which='major', length=6)  # Larger ticks for major grid

    return sc, cbar_ax


def plot_feature_importances(model, features, year=None,
                             title_prefix="Feature Importances",
                             suffix=""):
    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([features[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    title = f"{title_prefix}{' ' + str(year) if year else ''}"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'figures/RF/RF_{year}_importance_{suffix}.png', dpi=300)



def plot_evaluation_scatter(train_results, test_results, year, lp, save_path, suffix=""):
    fig, axs = plt.subplots(1, 2, figsize=(9, 5))

    for ax, results, title in zip(axs, [train_results, test_results],
                                  ['(a) Training set', '(b) Hold-out set']):
        handles, labels = plot_evaluation_panel(
            ax, results, 'total_liquid_water',
            title, 'Actual UDLW (m)', 'Predicted UDLW (m)', size=6
        )
        ax.set_xlim(-0.2, 4)
        ax.set_ylim(-0.2, 4)

        # inset = ax.inset_axes([0.6, 0.08, 0.35, 0.25])
        # plot_evaluation_panel(
        #     inset, results, 'total_liquid_water',
        #     '', '', '', annotation=False,
        # )
        # inset.set_xlim(-0.2, 32)
        # inset.set_ylim(-0.2, 32)
        # inset.tick_params(labelsize=8)
        # inset.set_xticks([0, 10, 20, 30])
        # inset.set_yticks([0, 10, 20, 30])

    fig.legend(handles, labels, loc='center left',
               bbox_to_anchor=(0.92, 0.5), markerscale=2, title="Sites")

    filename = f'{save_path}/RF_{year}_{suffix}.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    fig.show()
    # plt.close(fig)

# Example call:
# plot_evaluation_panels(train_results, test_results, 2015, lp, 'figures/RF', suffix='_v2')

# Plot predicted vs actual values for depth
def compute_rmse_bias(df):
    errors = df['depth_water_pred'] - df['depth_water']
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)
    return rmse, bias

import matplotlib.cm as cm

def plot_evaluation_panel(ax, df, size_var, title,
                          xlabel, ylabel, annotation=True, size=3):

    unique_sites = df['site'].unique()
    cmap = cm.get_cmap('tab20', len(unique_sites))  # Get tab20 with as many unique colors as sites
    site_colors = {site: cmap(i) for i, site in enumerate(unique_sites)}  # Assign colors



    df = df.loc[df[size_var] > 0.1, :]
    handles, labels = [], []

    for site in df['site'].unique():
        mask = (df['site'] == site)
        sizes = (df.loc[mask, size_var] - df[size_var].min()) / \
            (df[size_var].max() - df[size_var].min()) * 100

        ax.scatter(df.loc[mask, 'depth_water'],
                        df.loc[mask, 'depth_water_pred'],
                        alpha=0.5,
                        color=site_colors[site],
                        s=size, label="__nolegend__")

        sc_legend = ax.scatter(np.nan, np.nan,
                               color=site_colors[site],
                               alpha=1, s=20, label=site)
        handles.append(sc_legend)
        labels.append(site)

    ax.plot([df['depth_water'].min(), df['depth_water'].max()],
            [df['depth_water'].min(), df['depth_water'].max()], 'r--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()

    # Compute RMSE and bias
    if annotation:
        rmse, bias = compute_rmse_bias(df)
        msk = (df['depth_water_pred']<5) # & (df[y_var]>0.1)
        rmse_5m, bias_5m = compute_rmse_bias(df.loc[msk,:])
        ax.text(0.06, 0.98,(f'RMSE: {rmse:.2f}\nBias: {bias:.2f}'
                            f'\nfor UDLW < 5 m:'
                            f'\nRMSE: {rmse_5m:.2f}\nBias: {bias_5m:.2f}'),
                transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5))

    return handles, labels
