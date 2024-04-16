from pathlib import Path

SCRATCH_PATH = Path("/scratch/shared/Titan")

GRIDS = {
    "ANTJP7CLIM_1S100": {
        "size": [1051, 1651],
        "resolution": 0.01,
        "extent": [51.5, 41.0, -6.0, 10.5],
        "prefix": "ant",
    },
    "PAAROME_1S100": {
        "size": [1791, 2801],
        "resolution": 0.01,
        "extent": [55.4, 37.5, -12.0, 16.0],
        "prefix": "aro",
    },
    "PAAROME_1S40": {
        "size": [717, 1121],
        "resolution": 0.25,
        "extent": [55.4, 37.5, -12.0, 16.0],
        "prefix": "aro",
    },
    "PA_01D": {
        "size": [521, 741],
        "resolution": 0.1,
        "extent": [72.0, 20.0, -32.0, 42.0],
        "prefix": "arp",
    },
}

GRIB_PARAMS = {
    "ANTJP7CLIM_1S100_60_SOL.grib": ["ant_prec"],
    "PAAROME_1S100_ECH0_10M.grib": ["aro_u10", "aro_v10"],
    "PAAROME_1S100_ECH0_2M.grib": ["aro_t2m", "aro_r2"],
    "PAAROME_1S100_ECH0_SOL.grib": ["aro_sd"],
    "PAAROME_1S100_ECH1_10M.grib": ["aro_ugust", "aro_vgust"],
    "PAAROME_1S100_ECH1_SOL.grib": ["aro_tp", "aro_tirf", "aro_sprate"],
    "PAAROME_1S40_ECH0_ISOBARE.grib": [
        "aro_z",
        "aro_t",
        "aro_u",
        "aro_v",
        "aro_wz",
        "aro_r",
        "aro_ciwc",
        "aro_clwc",
        "aro_crwc",
        "aro_cswc",
        "aro_unknown",
    ],
    "PAAROME_1S40_ECH0_MER.grib": ["aro_prmsl"],
    "PAAROME_1S40_ECH0_SOL.grib": ["aro_tciwv"],
    "PAAROME_1S40_ECH1_SOL.grib": ["aro_str", "aro_ssr"],
    "PA_01D_10M.grib": ["arp_u10", "arp_v10"],
    "PA_01D_2M.grib": ["arp_t2m", "arp_r2"],
    "PA_01D_ISOBARE.grib": ["arp_z", "arp_t", "arp_u", "arp_v", "arp_r"],
    "PA_01D_MER.grib": ["arp_prmsl"],
}

ISOBARIC_LEVELS_HPA = sorted(
    list(range(100, 1050, 50)) + [125, 175, 225, 275, 925], reverse=True
)
