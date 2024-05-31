# TITAN : Training Inputs & Targets from Arome for Neural networks

Titan is a dataset made to train an AI NWP emulator on France.

![Domaines](figs/titan_veryverylight.png)

## Data

* 3 data sources: Analyses from AROME, ANTILOPE (rain data calibrated with rain gauges) and analyses and forecasts from ARPEGE (coupling model)
* 1 hour timestep
* Depth: 2 years
* Format GRIB2 (conversion to npy possible)

3 days of data stored on [HuggingFace](https://huggingface.co/datasets/meteofrance/titan)

To download :

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/datasets/meteofrance/titan
```

Then adapt the `PY4CAST_TITAN_PATH` env variable to put your titan path (cf [doc](../README.md/#setting-environment-variables)).

![Domaines](figs/titan_domaines.png)


### Available parameters

GRIB2 files are grouped in folders per hour. Each folder contains 14 grib files. Each grib file contains several parameters, grouped per model, grid, leadtime (0h ou 1h for flux) and level.

The parameters in each grib are described here:

| File name  | Model | Grid | Level  | Parameters |
| :---:   | :---: | :---: | :---: | :---: |
| ANTJP7CLIM_1S100_60_SOL | ANTILOPE | FRANXL1S100  (0.01°)  | Ground |  Cumulated Rainfall on next 1h |
| PA_01D_10M | ARPEGE | EURAT01 (0.1°)   | 10m |  U, V |
| PA_01D_2M | ARPEGE  | EURAT01 (0.1°) | 2m |  T, HU |
| PA_01D_ISOBARE | ARPEGE | EURAT01 (0.1°) | 24 Isobaric levels |  Z, T, U, V, HU |
| PA_01D_MER | ARPEGE | EURAT01 (0.1°) | Sea |  P |
| PAAROME_1S100_ECH0_10M | AROME | EURW1S100 (1.3 km) | 10m |  U, V |
| PAAROME_1S100_ECH0_2M | AROME | EURW1S100 (1.3 km) | 2m |  T, HU |
| PAAROME_1S100_ECH0_SOL | AROME | EURW1S100 (1.3 km) | Ground |  RESR_SNOW |
| PAAROME_1S100_ECH1_10M | AROME | EURW1S100 (1.3 km) | 10m |  U_RAF, V_RAF |
| PAAROME_1S100_ECH1_SOL | AROME | EURW1S100 (1.3 km) | Ground |  PRECIP, WATER, SNOW |
| PAAROME_1S40_ECH0_ISOBARE | AROME | EURW1S40 (2.5 km) | 24 Isobaric levels |  Z, T, U, V, VV2, HU, CIWC, CLD_WATER, CLD_RAIN, CLD_SNOW, CLD_GRAUPL |
| PAAROME_1S40_ECH0_MER | AROME | EURW1S40 (2.5 km) | Sea |  P |
| PAAROME_1S40_ECH0_SOL | AROME | EURW1S40 (2.5 km) | Ground |  COLUMN_VAPO |
| PAAROME_1S40_ECH1_SOL | AROME | EURW1S40 (2.5 km) | Ground |  FLTHERM, FLSOLAR |



## Storage

* Size of GRIB2 files for 1 hour : ~ 480 Mo

* Size of **compressed** file for one day : ~ 6 Go

* Size of **compressed** dataset for 1 year : ~ 2.2 To

## Usage

* All interfaces to use the dataset for training are in `__init__.py`

* Use the script `plot_data.py` to plot all the parameters for one time step.

* The Titan default configuration uses only a very small subdomain on britany and only one parameter 2m temperature, on the 3 day sample dataset (2023-01-01 to 2023-01-03). It should work for training out of the box. The `titan_full` configuration uses 2 years of data, more parameters and a bigger domain.

## Notes

* 2024/05/16 : At the moment, the dataloader only works with data from AROME, with one-level parameters. Options for multi-level and multi-grid data will be added later.

* ANTILOPE data are commercial, so they will not be included in the public sample of the dataset.