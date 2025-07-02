import numpy as np
import xarray as xr
import multiprocessing
from datetime import datetime
import os
import lib.smrt_functions as sf
import lib.load as ll
import pandas as pd
site='FA-13'
outfolder=''
filesuffix=''
opt_gs=True
opt_ice_layers=True
gs_factor=1
num_ice_layers=1
# %%
def main(site='DYE-2', outfolder='', filesuffix='', opt_gs=True,opt_ice_layers=True, gs_factor=1, num_ice_layers=1):
    # %%
    print(site, '; outfolder:',outfolder, '; filesuffix:',filesuffix,
          '; opt_gs:', opt_gs, '; gs_factor:',gs_factor, '; num_ice_layers:',num_ice_layers)
    firn_data = ll.load_firn_model(site)

    df_pmw = ll.load_pmw_site(firn_data.attrs['station'],
                          firn_data.attrs['latitude'],
                          firn_data.attrs['longitude'])
    df_pmw = df_pmw.loc[df_pmw.index.hour==9,:]

    gs_factor_ini =1
    num_ice_layers_ini = 50
    for year in np.unique(firn_data.time.dt.year):

        print(site,year,year)
        filename = f'smrt_output/{outfolder}/{site}_{str(year)}_Tb_smrt{filesuffix}.nc'
        
        # if os.path.exists(filename):
            # print(filename, 'already exists, skipping')
            # continue

        data_yr = firn_data.sel(time=str(year))
        if year == 2010:
            data_jan_feb = data_yr.sel(time=slice(str(year)+'-04-01', str(year)+'-05-01')).median(dim='time')
        elif year == 2012:
            data_jan_feb = firn_data.sel(time=slice(str(year-1)+'-02-01', str(year-1)+'-03-01')).median(dim='time')
        else:
            data_jan_feb = data_yr.sel(time=slice(str(year)+'-02-01', str(year)+'-03-01')).median(dim='time')

        data_jan_feb.attrs = data_yr.attrs
        time = pd.to_datetime(
            data_yr.sel(time=str(year) + '-02-01', method='nearest').time.item()
        )
        data_jan_feb = data_jan_feb.assign_coords(time=("time", [time]))

        frq_list = ['01','06','10','19']
        if year == 2010:
            BT_obs_V = df_pmw.loc[str(year)+'-04-01': str(year)+'-05-01'].median()[[f+'_V' for f in frq_list]]
        elif year == 2012:
            BT_obs_V = df_pmw.loc[str(year-1)+'-01-01': str(year-1)+'-03-01'].median()[[f+'_V' for f in frq_list]]
        else:
            BT_obs_V = df_pmw.loc[str(year)+'-01-01': str(year)+'-03-01'].median()[[f+'_V' for f in frq_list]]
        BT_obs_V.index = frq_list

        if opt_ice_layers:
            data_resampled, num_ice_layers = sf.find_opt_num_ice_layers(data_jan_feb, BT_obs_V[['01']])
            # data_resampled, num_ice_layers = sf.scan_num_ice_layers(data_jan_feb, BT_obs_V[['01']])
            print(site,year,'done with num_ice_layers tuning:',num_ice_layers)
            if np.isnan(num_ice_layers):
                num_ice_layers=num_ice_layers_ini
            else:
                num_ice_layers_ini=num_ice_layers

        if opt_gs:
            gs_factor = sf.find_opt_gs_factor(data_resampled, BT_obs_V[['06','10','19']])
            print(site,year,'done with gs tuning',gs_factor)
            if np.isnan(gs_factor):
                gs_factor=gs_factor_ini
            else:
                gs_factor_ini=gs_factor

        list_da = []
        for i, time in enumerate(data_yr.time):
            t1 = datetime.now()
            # data_selec = data_yr.sel(time=time).copy()
            data_selec = sf.resample_firn_model_output_at_fixed_depths(
                                            data_yr.sel(time=time).copy(),
                                            layers=None,
                                            num_ice_layers=num_ice_layers,
                                            plot=False)
            data_selec['dgrain'] = data_selec.dgrain * gs_factor

            da = sf.run_smrt(data_selec,
                             ['01', '06', '10', '19'],
                                time = time,
                                num_ice_layers_per_layer = 1,
                                parallel_computation=False)
            print(site, year, f'{i}/{len(data_yr.time)}', datetime.now()-t1)

            list_da.append(da)

        ds_result = xr.merge(list_da)

        ds_result.attrs['gs_factor'] = gs_factor
        ds_result.attrs['num_ice_layers'] = num_ice_layers

        comp = dict(zlib=True, complevel=9)
        encoding = {var: comp for var in ds_result.data_vars}

        print(site,year,'saving in',filename)
        ds_result.to_netcdf(filename,
                            encoding=encoding)

        # plt.figure()
        # ds_result['01'].isel(polarization=0).plot(marker='o', ax=plt.gca())
        # df_pmw['01_V'].plot(marker='o', ax=plt.gca())
        # plt.xlim(ds_result.time[[0, -1]].data)
# %%
    # désactiver multithreading de mkl
    # result.ks + result.ka = ke  e folding depth: 1/Ke
    # tracer exp(-Ke z) pour chaque couche
    # pruning à 4?

def run_on_core(station, core):
    """Set CPU affinity and run the main function for the given station."""
    print(f"Core {core} processing station: {station}")
    os.system(
        f"taskset -c {core} python3 -c 'import A1_smrt_run; "
        f"A1_smrt_run.main(\"{station}\", "
        f"outfolder=\"with_gs_and_n_ice_tuning\", "
        f"filesuffix=\"\", "
        f"opt_gs=True, "
        f"num_ice_layers=1)'"
    )
    print(f"Core {core} finished station: {station}")

def worker(task_queue):
    """Worker process that executes tasks sequentially on its assigned core."""
    core = task_queue.get()  # Get the assigned core
    while True:
        station = task_queue.get()  # Get the next station to process
        if station is None:
            break  # Stop worker when None is received
        run_on_core(station, core)

def standard_run_parallel(station_list):
    # max_core_usage =  multiprocessing.cpu_count()-1 # all cores except one
    max_core_usage =  23 # all cores except one
    num_cores = min(len(station_list), max_core_usage)
    task_queues = [multiprocessing.Queue() for _ in range(num_cores)]
    processes = []

    # Start workers with dedicated cores
    for core, task_queue in enumerate(task_queues):
        task_queue.put(core)  # Assign core number to worker
        p = multiprocessing.Process(target=worker, args=(task_queue,))
        p.start()
        processes.append((p, task_queue))

    # Distribute tasks in a round-robin fashion
    for i, station in enumerate(station_list):
        task_queues[i % num_cores].put(station)

    # Send termination signal (None) to workers
    for _, task_queue in processes:
        task_queue.put(None)

    # Wait for all workers to finish
    for p, _ in processes:
        p.join()

if __name__ == "__main__":
    # run_SMRT(('01',50))
    # with xr.open_dataset("./data/CARRA_at_AWS.nc") as ds:
        # station_list = ds.stid.load().values
    # unwanted= ['NUK_K', 'MIT', 'ZAK_A', 'ZAK_L', 'ZAK_Lv3', 'ZAK_U', 'ZAK_Uv3', # local glaciers
                # 'LYN_L', 'LYN_T', 'FRE',  # local glaciers
                # 'KAN_B', 'NUK_B','WEG_B', # off-ice AWS
                # 'DY2','NSE','SDL','NAU','NAE','SDM','TUN','HUM','SWC', 'JAR', 'NEM',  # redundant
                # 'T2_08','S10','Swiss Camp 10m','SW2','SW4','Crawford Point 1', 'G1','EGP', # redundant
                # 'CEN1','THU_L2'
                # ]
    # station_list = [s for s in station_list if s not in unwanted]
    # station_list = [s for s in station_list if 'v3' not in s]

    station_list = [ 'H2','H3', 'T1old','T2_09','T3', 'T4',
                 'FA-13','FA-15-1','FA-15-2','SIGMA-A',
                 'KAN_U', 'DYE-2','CP1', 'South Dome', 'NASA-SE',
                 'NASA-U','Saddle', 'Humboldt', 'CEN2']
    # station_list = [ 'NEEM','NGRIP', 'Summit', 'Tunu-N', 'NASA-E','EastGRIP','Humboldt']
    standard_run_parallel(station_list)

    # standard run in series
    # for site in station_list:
    # for site in ['DYE-2','KAN_U','CP1','FA-13']:
        # main(site, outfolder='with_gs_tuning',
                    # opt_gs=True,
                    # num_ice_layers=1)



    # sensitivity run on num_ice_layer
    # for site in ['DYE-2','KAN_U','CP1','FA-13']:
    #     for num_ice_layers in [2,1, 3, 4, 5, 10]:
    #         main(site,
    #              outfolder='sensitivity_num_ice_layer',
    #              filesuffix=f'_num_ice_layer_{num_ice_layers}',
    #              opt_gs=False,
    #              num_ice_layers=num_ice_layers)
