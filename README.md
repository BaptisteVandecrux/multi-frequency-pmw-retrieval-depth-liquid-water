# Estimating the Depth of Subsurface Liquid Water on the Greenland Ice Sheet

<p align="center">
  <span style="display:inline-flex; align-items:center; height:120px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/ESA_logo.png/330px-ESA_logo.png" alt="ESA" height="55"/>
  </span>
  <span style="display:inline-flex; align-items:center; height:120px; margin: 0 20px;">
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSZwPTj_X0dIN8FSEQnsLUbwacJzAZgRkAIeKnUXYsZYSE0RrAo5spEoFvToIdHrwE7Azo&usqp=CAU" alt="CCI" height="60"/>
  </span>
  <span style="display:inline-flex; align-items:center; height:120px;">
    <img src="https://github.com/SUMup-database/.github/blob/main/profile/doc/misc/Promice_GC-Net_colour.jpg?raw=true" alt="PROMICE/GC-Net" height="100"/>
  </span>
</p>

Multi-frequency passive microwave emissions from the Greenland Ice Sheet are known to be sensitive to the presence of liquid water at different depths. Here we derive the upper depth of the wet layer on the ice sheet—henceforth referred to as the depth of liquid water (DLW)—from brightness temperature (BT) observations at 1.4, 6.9, 10.7, and 18.7 GHz from the Soil Moisture and Ocean Salinity (SMOS) satellite and Advanced Microwave Scanning Radiometers (AMSR-E, AMSR2), combined with snow and radiative transfer modeling and machine learning.

First, to understand the response of multi-frequency TB to the presence of liquid water in the snow, we build the following simulation catalogue. The GEUS snow model (Vandecrux et al., [2018](https://doi.org/10.1029/2017JF004597), [2020a](https://doi.org/10.1017/jog.2020.30), [2020b](https://doi.org/10.5194/tc-14-3785-2020)) was run at 19 sites in the accumulation area of the Greenland Ice Sheet using the Copernicus Arctic Regional Reanalysis (CARRA) as forcing. The Snow Microwave Radiative Transfer (SMRT) model from [Picard et al. (2018)](https://doi.org/10.5194/gmd-11-2763-2018) calculated the daily (6 AM) brightness temperature (TB) at four frequencies using as input the simulated profiles of snow temperature, density, and grain size from the GEUS snow model. The coupling between these two models was optimized through the adjustment of two parameters for each site and each year. First, the number of pure ice layers to be considered in the SMRT input—based on the ice content simulated by the GEUS snow model—was optimized to maximize the match between observed and simulated winter vertically polarized TB at 1.4 GHz. Then, a multiplicative correction factor applied to the GEUS snow model's simulated grain diameter was also optimized each year and at each site to maximize the match between observed and simulated vertically polarized TB at 6.9, 10.7, and 18.7 GHz.

Then we train an ensemble of 14 random forest (RF) models on this simulation catalogue, each RF model leaving one year out of the data. As input, the RF models take the vertically polarized BT at our four frequencies of interest, along with the same TB normalized between the preceding winter TB and 273.15 K (i.e., TB_norm = 0 if TB = TB_winter and TB_norm = 1 if TB = 273.15). Once trained, the RF model ensemble provides a prediction of the DLW and an evaluation of its uncertainty: DLW_std, the ensemble's standard deviation for a given prediction. We recommend discarding retrievals with DLW_std > 1. To ensure that values are only retrieved when water is detected on the ice sheet, we only consider times and pixels when the surface wetness maps from Zeiger et al. ([2024](https://doi.org/10.1016/j.rse.2024.114469)) derived from both vertically and horizontally polarized TB at 1.4 GHz, indicate water at or below the surface.

Please refer to, and cite:

Vandecrux, B., Picard, G., Zeiger, P., Leduc-Leballeur, M., Colliander, A., Hossan, A., & Ahlstrøm, A. (submitted). *Estimating the depth of subsurface water on the Greenland Ice Sheet using multi-frequency passive microwave remote sensing, radiative transfer modeling, and machine learning*. Remote Sensing of Environment.

This dataset was produced under the European Space Agency Climate Change Initiative research fellowship [Water Under Snow Cover](https://climate.esa.int/en/about-us-new/fellowships/esa-cci-research-fellowship-Baptiste-Vandecrux/).

---

## Script Overview

**1. `1_smrt_run.py`**  
Runs the SMRT model at 19 Greenland sites using GEUS snow model output. Optimizes the number of pure ice layers and grain size correction factor to match observed brightness temperatures. Produces site-year SMRT simulations saved to NetCDF.

**2. `2_building_training_data_on_site_runs.py`**  
Loads SMRT simulation output and derives daily snow model characteristics. Computes features like brightness temperature ratios and normalizations, and prepares the training dataset used for machine learning.

**3. `3_learning_from_water_depth.py`**  
Trains a random forest model ensemble on the prepared SMRT simulation data using leave-one-year-out cross-validation. Predicts DLW and its uncertainty and evaluates model performance per year.

**4. `4_applying_rf_to_map.py`**  
Applies the trained random forest models to brightness temperature grids. Uses multiprocessing to perform per-pixel DLW and uncertainty predictions and writes yearly gridded NetCDF outputs.

**5. `5_plotting_depth_maps.py`**  
Visualizes DLW predictions as static maps, time series, transects, and animated GIFs. Also produces diagnostic plots and maps of prediction uncertainty and frequency of deep water occurrence.


## Data Products and Dependencies

The scripts in this repository produce daily maps of the depth of subsurface liquid water (DLW) across the Greenland Ice Sheet for the years 2010–2023, available at [https://doi.org/10.22008/FK2/69CRRT](https://doi.org/10.22008/FK2/69CRRT) (Vandecrux, 2025a). They also generate simulated brightness temperatures at 19 sites using the coupled and optimized GEUS snow model and SMRT radiative transfer model, available at [https://doi.org/10.22008/FK2/KKIOZZ](https://doi.org/10.22008/FK2/KKIOZZ) (Vandecrux et al., 2025b).

To run the scripts, several external datasets are required:
- Enhanced-resolution SMOS brightness temperature and snow wetness grids from Zeiger and Picard ([2024b](https://doi.org/10.57932/f72f9515-3699-4fae-92fc-a350075d042f), [2024c](https://doi.org/10.57932/1970fb7c-cdb4-4ddf-9891-1e6836a46f25))
- AMSR2 and AMSR-E observations from [JAXA G-Portal](https://gportal.jaxa.jp/gpr/) and [NSIDC](https://doi.org/10.5067/AMSR-E/AE_L2A.003) (Ashcroft and Wentz, 2013), or the re-gridded zarr archive at [https://snow.univ-grenoble-alpes.fr/opendata/PMW-Greenland-12km.zarr](https://snow.univ-grenoble-alpes.fr/opendata/PMW-Greenland-12km.zarr) (Picard, 2024), which can be accessed via `wget` or `xarray.open_zarr`
- The SUMup collaborative firn observation database ([https://doi.org/10.18739/A2M61BR5M](https://doi.org/10.18739/A2M61BR5M); Vandecrux et al., 2024)
- The Copernicus Arctic Regional Reanalysis (CARRA) dataset ([https://doi.org/10.24381/cds.713858f6](https://doi.org/10.24381/cds.713858f6); Schyberg et al., 2020)

---

**Author:** Baptiste Vandecrux  
**Contact:** bav@geus.dk  
**License:** CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
