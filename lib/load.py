# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import xarray as xr
import numpy as np
import pandas as pd
from scipy.spatial import distance
from math import sin, cos, sqrt, atan2, radians
import geopandas as gpd
import glob
import rasterio as rio
import lib.smrt_functions as sf
from shapely.geometry import mapping

# %% Passive microwave & SMRT loading & manipulation functions
def load_smrt_simulations(site, folder = 'smrt_output', tag='', frq_list=['01', '06','10','19']):
    list_file_old_format = []
    missing_frq = []
    for frq in frq_list:
        try:
            list_file_old_format.append(
                xr.open_mfdataset(glob.glob(
                    folder+'/'+site+'_'+tag+frq+'_*_Tb_smrt.nc')
                    ))
        except:
            missing_frq.append(frq)

    if len(list_file_old_format)>1:
        print('loading', folder+'/'+site+tag+'_<frq>_<yr>_Tb_smrt.nc')
        ds_smrt = xr.merge(list_file_old_format, compat='override')
        if len(missing_frq)>0:
            print('Missing SMRT data for frequencies:', missing_frq)
    else:
        print('loading', folder+'/'+site+tag+'_<yr>_Tb_smrt.nc')
        ds_smrt = xr.open_mfdataset(glob.glob(folder+'/'+site+tag+'_*_Tb_smrt.nc'),
                                    coords='all',
                                    compat='override')

    df = ds_smrt.sel(polarization='H')[frq_list].to_dataframe()[frq_list]
    df.columns=df.columns+'_H'
    df_v = ds_smrt.sel(polarization='V')[frq_list].to_dataframe()[frq_list]
    df_v.columns=df_v.columns+'_V'
    df[df_v.columns] = df_v
    return df


def load_pmw_all(year=None):
    ds_pmw = xr.open_dataset('data/PMW grids/PMW-Greenland-12km_2010_2024.nc')[
        ['06V_ASC','06V_DSC','10V_ASC','10V_DSC','19V_ASC','19V_DSC',
          '06H_ASC','06H_DSC','10H_ASC','10H_DSC','19H_ASC','19H_DSC',
         ]]

    # ds_smos = xr.open_zarr('data/PMW grids/SMOS_old.zarr')
    ds_smos = xr.open_zarr('data/PMW grids/SMOS_rSIR-enhanced-40_Greenland_2010-2024.zarr')[
                               ['TB_V_morning','TB_V_afternoon',
                                'TB_H_morning','TB_H_afternoon'
                                ]]

    ds_pmw['01H_ASC'] = ds_smos.TB_H_morning
    ds_pmw['01H_DSC'] = ds_smos.TB_H_afternoon
    ds_pmw['01V_ASC'] = ds_smos.TB_V_morning
    ds_pmw['01V_DSC'] = ds_smos.TB_V_afternoon

    for v in ds_pmw.data_vars:
        ds_pmw[v].attrs['grid_mapping'] = 'spatial_ref'
    print('PMW data loaded')
    if year:
        return ds_pmw.sel(time=str(year)).load()
    else:
        return ds_pmw

def load_amsr_site(site, latitude, longitude):
    ds_amsr = xr.open_dataset('data/PMW grids/melt-4D-Greenland-12km.nc')

    # loading PMW data
    df_points = pd.DataFrame(dict(latitude=latitude, longitude=longitude),
                             index=[site])
    df_points = gpd.GeoDataFrame(
        df_points, geometry=gpd.points_from_xy(df_points.longitude,
                                               df_points.latitude), crs="EPSG:4326"
                                ).to_crs(3413)
    df_points['x_3413'] =  df_points.geometry.x
    df_points['y_3413'] =  df_points.geometry.y
    df_points = pd.DataFrame(df_points)
    ds_amsr_site =  ds_amsr.sel(x=df_points['x_3413'].item(), y=df_points['y_3413'].item(), method='nearest')

    # Extract data and adjust timestamps
    # Frequencies and polarizations to process
    frequencies = ['snow_status_wet_dry_'+f for f in ['06', '10', '19', '37']]
    polarizations = ['V','H']
    merged_df = pd.DataFrame()

    # Process each frequency and polarization
    for freq in frequencies:
        for pol in polarizations:
            combined_data = merge_and_adjust(ds_amsr_site, freq, pol)
            if merged_df.empty:
                merged_df = combined_data
            else:
                merged_df = merged_df.merge(combined_data, on='time', how='outer')

    # Sort the dataframe by time
    return merged_df.sort_values(by='time').reset_index(drop=True).set_index('time')

def calc_dry_norm_BT_xr(ds, features=['01_V', '06_V', '10_V', '19_V']):
    features_max = [v+'_max' for v in features]
    features_dry = [f+'_dry' for f in features]

    ds = ds.copy()
    for f in features_dry + features_max:
        ds[f] = xr.full_like(ds[features[0]], np.nan)

    years = np.unique(ds['time.year'])

    for yr in years:
        yearly_ds = ds.sel(time=ds.time.dt.year == yr)
        dry_values = yearly_ds.isel(time=slice(0, 60))[features].median(dim='time')

        for d in features_dry:
            ds[d] = xr.where(ds.time.dt.year == yr, dry_values[d.split('_dry')[0]], ds[d])

    for freq in ['01', '06', '10', '19', '37']:
        dry_key = f'{freq}_V_dry'
        if dry_key in ds:
            ds[f'{freq}_V_norm'] = (ds[f'{freq}_V'] - ds[dry_key]) / (273.15 - ds[dry_key])

    return ds[features+[v+'_norm' for v in features]]

def clip_firn_area(ds_pmw):
    firn_line = gpd.read_file('data/FirnLayer2000-2017.shp')
    firn_line = firn_line.to_crs('EPSG:3413')

    ds_pmw = ds_pmw.rio.write_crs('EPSG:3413')
    transform = ds_pmw.rio.transform(recalc=True)

    mask = rio.features.geometry_mask(
        # [mapping(g) for g in firn_line.geometry if not g.is_empty],
        [geom for geom in firn_line.geometry],
        out_shape=(len(ds_pmw.y), len(ds_pmw.x)),
        transform=transform,
        invert=True,
    )
    mask = xr.DataArray(mask, dims=("y", "x"), coords={"y": ds_pmw.y, "x": ds_pmw.x})

    return ds_pmw.where(mask)

def prepare_ds_pmw(ds_pmw, filter=True):
    # only keeping firn area
    print('Filtering non-firn areas')
    if filter:
        ds_pmw = clip_firn_area(ds_pmw)
    print('done')
    print('Assigning timestamps to ASC and DSC')

    # giving timestamps to ascending and descending pass
    list_df_freq = []
    # Process each frequency and polarization
    for freq in ['01', '06', '10', '19']:
        print(freq)
        var_name_asc = f"{freq}V_ASC"
        var_name_dsc = f"{freq}V_DSC"
        ds_freq = merge_and_adjust_xr(ds_pmw,
                                         var_name_asc,
                                         var_name_dsc,
                                         f"{freq}_V")
        list_df_freq.append(ds_freq.to_dataframe()[f"{freq}_V"])

    # Merge the sorted datasets
    df_pmw = pd.concat(list_df_freq, axis=1)
    print('done')

    print('Applying smoothing')
    df_pmw = sf.calc_dry_norm_BT(df_pmw)

    features = ['01_V','01_V_norm', '06_V','06_V_norm',
                '10_V','10_V_norm', '19_V','19_V_norm']
    df_pmw_smooth = df_pmw.copy()

    for var in features:
        print(var)
        window = 7
        threshold = 10
        smoothed = df_pmw[var].rolling(window=window, center=True, win_type='gaussian').mean(std=3)
        peaks = df_pmw[var].where(abs(df_pmw[var] - smoothed) > threshold)
        df_pmw_smooth[var] = peaks.combine_first(smoothed)

    final_dataset = df_pmw_smooth.to_xarray()
    print('done')

    return final_dataset


def load_smos_site(site, latitude, longitude):
    ds_smos = xr.open_dataset('data/PMW grids/SMOS-melt-12km-Greenland-2010_2024.nc')

    # loading PMW data
    df_points = pd.DataFrame(dict(latitude=latitude, longitude=longitude),
                             index=[site])
    df_points = gpd.GeoDataFrame(
        df_points, geometry=gpd.points_from_xy(df_points.longitude, df_points.latitude), crs="EPSG:4326"
    ).to_crs(3413)
    df_points['x_3413'] =  df_points.geometry.x
    df_points['y_3413'] =  df_points.geometry.y
    df_points = pd.DataFrame(df_points)
    ds_smos_site =  ds_smos.sel(x=df_points['x_3413'].item(), y=df_points['y_3413'].item(), method='nearest')

    # Extract data and adjust timestamps
    var_name_asc = 'snow_status_wet_dry_smos_morning'
    var_name_dsc = 'snow_status_wet_dry_smos_afternoon'
    asc_data = ds_smos_site[var_name_asc].to_dataframe().reset_index()
    asc_data['time'] = (asc_data['time'] + pd.to_timedelta(9, unit='h')).values
    asc_data = asc_data.rename(columns={var_name_asc: 'snow_status_wet_dry_smos'})

    dsc_data = ds_smos_site[var_name_dsc].to_dataframe().reset_index()
    dsc_data['time'] = (dsc_data['time'] + pd.to_timedelta(17, unit='h')).values
    dsc_data = dsc_data.rename(columns={var_name_dsc:'snow_status_wet_dry_smos'})

    # Combine the dataframes
    combined_data = pd.concat([asc_data[['time', "snow_status_wet_dry_smos"]],
                               dsc_data[['time', "snow_status_wet_dry_smos"]]],
                              ignore_index=True)

    # Sort the dataframe by time
    return combined_data.sort_values(by='time').reset_index(drop=True).set_index('time')


def load_melt_xr(year = None, only_smos=False):
    ds_smos = xr.open_dataset('data/PMW grids/SMOS-melt-12km-Greenland-2010_2024.nc')
    if year:
        ds_smos = ds_smos.sel(time=str(year))
    # Extract data and adjust timestamps
    var_name_asc = 'snow_status_wet_dry_smos_morning'
    var_name_dsc = 'snow_status_wet_dry_smos_afternoon'
    ds_melt = merge_and_adjust_xr(ds_smos, var_name_asc, var_name_dsc, 'snow_status_wet_dry_01V').isel(band=0).to_dataset()

    if only_smos: return ds_melt

    ds_amsr = xr.open_dataset('data/PMW grids/melt-4D-Greenland-12km.nc')
    if year:
        ds_amsr = ds_amsr.sel(time=str(year))

    # Process each frequency and polarization
    for freq in  ['06', '10', '19', '37']:
        var_name_asc =  f'snow_status_wet_dry_{freq}V_ASC'
        var_name_dsc = f'snow_status_wet_dry_{freq}V_DSC'
        ds_melt[f'snow_status_wet_dry_{freq}V'] = merge_and_adjust_xr(ds_amsr, var_name_asc, var_name_dsc, f'snow_status_wet_dry_{freq}V')

    return ds_melt

def extract_pixel(ds_pmw, site, latitude, longitude):
    df_points = pd.DataFrame(dict(latitude=latitude, longitude=longitude),
                             index=[site])
    df_points = gpd.GeoDataFrame(
        df_points, geometry=gpd.points_from_xy(df_points.longitude, df_points.latitude), crs="EPSG:4326"
    ).to_crs(3413)
    df_points['x_3413'] =  df_points.geometry.x
    df_points['y_3413'] =  df_points.geometry.y
    df_points = pd.DataFrame(df_points)
    return ds_pmw.sel(x=df_points['x_3413'].item(), y=df_points['y_3413'].item(), method='nearest')


def load_pmw_site(site, latitude, longitude, ds_pmw=None):
    if ds_pmw is None:
        ds_pmw = load_pmw_all()

    # loading PMW data
    ds_pmw_site = extract_pixel(ds_pmw, site, latitude, longitude)

    # Initialize an empty DataFrame to store the merged results
    merged_df = pd.DataFrame()

    # Frequencies and polarizations to process
    frequencies = ['01', '06', '10', '19']
    polarizations = ['V', 'H']

    # Process each frequency and polarization
    for freq in frequencies:
        for pol in polarizations:
            if freq+pol+'_ASC' not in ds_pmw.data_vars:
                continue
            combined_data = merge_and_adjust(ds_pmw_site, freq, pol)
            if merged_df.empty:
                merged_df = combined_data
            else:
                merged_df = merged_df.merge(combined_data, on='time', how='outer')

    # Sort the dataframe by time
    return merged_df.sort_values(by='time').reset_index(drop=True).set_index('time')

def merge_and_adjust(ds, freq, pol):
    var_name_asc = f"{freq}{pol}_ASC"
    var_name_dsc = f"{freq}{pol}_DSC"
    var_out = f"{freq}_{pol}"

    use_asc = freq.startswith("01")
    use_dsc = not use_asc

    frames = []

    if use_asc and var_name_asc in ds:
        asc_data = ds[var_name_asc].to_dataframe().reset_index()
        asc_data['time'] += pd.to_timedelta(9, unit='h')
        asc_data = asc_data.rename(columns={var_name_asc: var_out})
        frames.append(asc_data[['time', var_out]])

    if use_dsc and var_name_dsc in ds:
        dsc_data = ds[var_name_dsc].to_dataframe().reset_index()
        dsc_data['time'] += pd.to_timedelta(9, unit='h')
        dsc_data = dsc_data.rename(columns={var_name_dsc: var_out})
        frames.append(dsc_data[['time', var_out]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['time', var_out])


def merge_and_adjust_xr(ds, var_name_asc, var_name_dsc, new_name):
    asc_data = ds[var_name_asc]
    dsc_data = ds[var_name_dsc]

    use_asc = var_name_asc.startswith("01")
    use_dsc = not use_asc

    adjusted = []

    if use_asc:
        asc_data = asc_data.assign_coords(time=asc_data.time + pd.to_timedelta(9, unit='h'))
        adjusted.append(asc_data)

    if use_dsc:
        dsc_data = dsc_data.assign_coords(time=dsc_data.time + pd.to_timedelta(9, unit='h'))
        adjusted.append(dsc_data)

    if not adjusted:
        return xr.DataArray(name=new_name)

    combined_data = xr.concat(adjusted, dim='time')
    return combined_data.rename(new_name).sortby("time")


# %% Snow model output load & manipulation

def load_firn_model(site):
    path_to_firn_model = '../GEUS-SEB-firn-model/output/new/'+site+'_100_layers_3h/'
    data = xr.load_dataset(path_to_firn_model+site+'_rhofirn.nc').rename({'rhofirn':'rho'})
    data = data.where(data.time.dt.hour.isin([9]), drop=True).sel(time=slice('2010','2024'))
    for var in ['T_ice', 'dgrain', 'slwc','snic','snowc']:
        data[var] = xr.load_dataset(path_to_firn_model+site+'_'+var+'.nc').sel(time=data.time, method='nearest')[var]

    data['thickness'] = data.depth.copy()
    data['thickness'].attrs['long_name'] = 'Layer thickness'
    data['thickness'].attrs['units'] = 'm'
    data['thickness'].data[1:,:] = data['depth'].data[1:,:] - data['depth'].data[0:-1,:]
    return data

def load_snow_model_output(site, var_list=['slwc','T_ice']):
        path_to_firn_model = '../GEUS-SEB-firn-model/output/new/' + site + '_100_layers_3h/'
        data = xr.load_dataset(path_to_firn_model + site + '_rhofirn.nc').rename({'rhofirn': 'rho'})

        for var in var_list:
            data[var] = xr.load_dataset(path_to_firn_model + site + '_' + var + '.nc').sel(time=data.time, method='nearest')[var]
        data = data.sel(time=slice('2010', None))
        data = data.transpose()
        data['surface_height'] = 0 * data.depth.isel(level=-1)

        data = data.where(data.time.dt.hour.isin([9]), drop=True)

        # Change unit to mm / m3
        data['slwc'] = data.slwc * 1000 / data.depth

        # some additional info:
        df_surface = xr.load_dataset(path_to_firn_model + site + '_surface.nc')[['theta_2m','snowfall_mweq','melt_mweq']].to_dataframe()
        data.attrs['T2m_avg'] = df_surface.theta_2m.mean()
        data.attrs['SF_avg'] = df_surface.snowfall_mweq.resample('YE').sum().mean()
        data.attrs['melt_avg'] = df_surface.melt_mweq.resample('YE').sum().mean()
        return data


def get_upper_water_depth(year_data):
    mask = year_data['slwc'] > 0
    upper_depth = (year_data['depth']
                   .shift(level=1,
                          fill_value=0)
                   .isel(level=mask.argmax(dim='level'))
                   .where(mask.any(dim='level'), np.nan))

    # Step 2: Convert other dataframes to pandas DataFrames
    upper_depth_df = upper_depth.to_dataframe()[['depth']]
    upper_depth_df.columns = ['depth_water']
    return upper_depth_df

def get_model_water_content(year_data):
    tlwc = year_data['slwc'].sum(dim='level')*1000

    # Step 2: Convert other dataframes to pandas DataFrames
    tlwc = tlwc.to_dataframe()[['slwc']]
    tlwc.columns = ['slwc']
    return tlwc

# %% SUMup load % select functions

def get_distance(point1, point2):
    R = 6370
    lat1 = radians(point1[0])  #insert value
    lon1 = radians(point1[1])
    lat2 = radians(point2[0])
    lon2 = radians(point2[1])

    dlon = lon2 - lon1
    dlat = lat2- lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance


def load_sumup_temperature():
    # Evaluating temperature with SUMup 2024
    df_sumup = xr.open_dataset(
        '../SUMup-data/SUMup_2025_temperature_greenland.nc',
        group='DATA', lock=False, decode_timedelta=False).to_dataframe()

    ds_meta = xr.open_dataset(
        '../SUMup-data/SUMup_2025_temperature_greenland.nc',
        group='METADATA', lock=False, decode_timedelta=False)
    # decoding utf-8 bytes
    decode_utf8 = np.vectorize(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    for v in ['name','reference','reference_short','method']:
        ds_meta[v] = ds_meta[v].str.decode('utf-8')


    df_sumup.method_key = df_sumup.method_key.replace(np.nan,-9999)
    # df_sumup['method'] = ds_meta.method.sel(method_key = df_sumup.method_key.values)
    df_sumup['name'] = ds_meta.name.sel(name_key = df_sumup.name_key.values)
    df_sumup['reference'] = ds_meta.reference.sel(reference_key=df_sumup.reference_key.values)
    df_sumup['reference_short'] = ds_meta.reference_short.sel(reference_key=df_sumup.reference_key.values)

    # df_ref = ds_meta.reference.to_dataframe()
    df_sumup = df_sumup.loc[df_sumup.timestamp>pd.to_datetime('2005')]
    # selecting Greenland metadata measurements
    df_meta = df_sumup.loc[df_sumup.latitude>0,
                      ['latitude', 'longitude', 'name_key', 'name', 'method_key',
                       'reference_short','reference', 'reference_key']
                      ].drop_duplicates()
    return df_sumup, df_meta



def select_sumup(df_sumup, df_meta, latitude, longitude):
    query_point = [[latitude, longitude]]
    all_points = df_meta[['latitude', 'longitude']].values
    df_meta['distance_from_query_point'] = distance.cdist(all_points, query_point, get_distance)
    min_dist = 5 # in km
    df_meta_selec = df_meta.loc[df_meta.distance_from_query_point<min_dist, :]

    return (df_sumup.loc[
                df_sumup.latitude.isin(df_meta_selec.latitude) \
                    & df_sumup.longitude.isin(df_meta_selec.longitude),:].copy(),
        df_meta_selec.copy())
