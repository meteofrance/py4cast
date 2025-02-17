import glob
import os
import os.path as osp
import re
import time

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc

# General paths
titan_path = "<npy-dataset-root>/AROME/TITAN"
data_path = osp.join(
    titan_path, "subdatasets/titan_full_PAAROME_1S40_100-612-240-880/data"
)
grid_conf_file = osp.join(titan_path, "conf_PAAROME_1S40.grib")
landsea_mask_file = osp.join(titan_path, "PEARO_EURW1S40_Orography_crop.npy")


def get_version(string):
    """Extract the version of a zarr dataset path name, and convert to int."""
    return int(re.findall(r"-v(\d+).zarr", string)[0])


# Set the zarr directory
zarr_path = "<zarr-dataset-root>/AROME/titan_2021-2023_PAAROME_1S40_4_1h_100-612-240-880-0p025deg-chunk-1-v0.zarr"
zarr_datasets = sorted(glob.glob(zarr_path.replace("v0", "v*")), key=get_version)
if zarr_datasets:
    version = get_version(zarr_datasets[-1]) + 1
    zarr_path = zarr_path.replace("v0", f"v{version}")

# Set the subdomain range
subdomain = [100, 612, 240, 880]

# Define the variables metadata
var_info = {
    "latitude": {
        "long_name": "latitude",
        "units": "degrees_north",
    },
    "longitude": {
        "long_name": "longitude",
        "units": "degrees_east",
    },
    "level": {},
    "time": {},
    "10m_u_component_of_wind": {
        "long_name": "10 metre U wind component",
        "short_name": "u10",
        "units": "m s**-1",
        "type_level": "heightAboveGround",
    },
    "10m_v_component_of_wind": {
        "long_name": "10 metre V wind component",
        "short_name": "v10",
        "units": "m s**-1",
        "type_level": "heightAboveGround",
    },
    "2m_relative_humidity": {
        "long_name": "2 metre relative humidity",
        "short_name": "r2",
        "units": "%",
        "type_level": "heightAboveGround",
    },
    "2m_temperature": {
        "long_name": "2 metre temperature",
        "short_name": "t2m",
        "units": "K",
        "type_level": "heightAboveGround",
    },
    "geopotential": {
        "long_name": "Geopotential",
        "short_name": "z",
        "units": "m**2 s**-2",
        "type_level": "isobaricInhPa",
    },
    "geopotential_at_surface": {
        "long_name": "Geopotential at surface",
        "short_name": "z_surf",
        "units": "m**2 s**-2",
        "type_level": "isobaricInhPa",
    },
    "land_sea_mask": {
        "long_name": "Land-sea mask",
        "short_name": "lsm",
        "units": "(0-1)",
        "type_level": "surface",
    },
    "temperature": {
        "long_name": "Temperature",
        "short_name": "t",
        "units": "K",
        "type_level": "isobaricInhPa",
    },
    "total_precipitation": {
        "long_name": "Total precipitation",
        "short_name": "tp",
        "units": "m",
        "type_level": "surface",
    },
    "u_component_of_wind": {
        "long_name": "U component of wind",
        "short_name": "u",
        "units": "m s**-1",
        "type_level": "isobaricInhPa",
    },
    "v_component_of_wind": {
        "long_name": "V component of wind",
        "short_name": "v",
        "units": "m s**-1",
        "type_level": "isobaricInhPa",
    },
}

# Surface variables names
surf_var_names = [
    "r2",
    "t2m",
    "u10",
    "v10",
    "tp",
]
# Pressure level variables names
p_var_names = ["t", "u", "v", "z"]

# The list of delta of time used to compute the std_diff statistics on each variables
time_deltas = [1, 3, 6, 12]  # in hours


def main(batch: int = 1):
    """
    Convert the numpy version of TITAN data into a Zarr version.
    This function reads TITAN numpy data for each date, stores them in a batch array, and creates a Zarr
    dataset with 4D data (time, latitude, longitude, level). Additional data such as land/sea mask
    are also included in the new dataset.
    The process involves:
    - Listing all subdirectories in the TITAN directory corresponding to dates.
    - Loading the grid configuration file.
    - Cropping the domain based on the specified subdomain.
    - Extracting pressure levels from the file names.
    - Defining common arguments for creating Zarr arrays.
    - Reading and stacking data for surface and pressure level variables.
    - Handling missing dates by skipping them.
    - Adding land-sea mask data.
    - Creating an xarray Dataset with the collected data and saving it as a Zarr file.
    - Adding the missing_dates as a dataset attribute.
    - Compute the variable statistics and add them as variable attributes.
    The resulting Zarr dataset includes variables such as wind components, temperature, relative
    humidity, geopotential, total precipitation, and land-sea mask, with appropriate metadata and
    compression.

    Args:
        batch (int): the number of dates added to the dataset at once.
    """

    # List all the subdirectories in the TITAN directory, corresponding to dates
    dates = sorted(glob.glob("**", root_dir=data_path))[0:20]

    # List the files in the last directory
    # Doing this because I am sure that this last directory contains all the files
    # TODO: check pressure levels are correctly sorted (string sorting != int sorting)
    files = sorted(os.listdir(osp.join(data_path, dates[-1])))

    # Load the grid configuration file
    grid_conf = xr.open_dataset(
        grid_conf_file,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )

    # Crop the domain
    lat = subdomain[1] - subdomain[0]
    lon = subdomain[3] - subdomain[2]
    latitude = grid_conf.latitude.values[subdomain[0] : subdomain[1]]
    longitude = grid_conf.longitude.values[subdomain[2] : subdomain[3]]

    # Get the pressure levels
    levels = list(
        set([file.split("_")[2].split(".")[0] for file in files if "hpa" in file])
    )
    levels = sorted([int(level.split("hpa")[0]) for level in levels])

    # Define common arguments when creating a zarr array
    surface_chunks = (lat, lon)  # {"time": 1, "latitude": lat, "longitude": lon}
    plevel_chunks = (
        lat,
        lon,
        len(levels),
    )  # {"time": 1, "latitude": lat, "longitude": lon, "level": len(levels)}
    compressor = Blosc(blocksize=0, clevel=5, cname="lz4", shuffle=Blosc.SHUFFLE)

    # Compute land-sea mask
    z_surf = np.load(landsea_mask_file, mmap_mode="r")
    z_surf = da.from_array(
        z_surf[subdomain[0] : subdomain[1], subdomain[2] : subdomain[3]],
        chunks=surface_chunks,
    )
    landsea_mask = da.where(z_surf > 0.0, 1.0, 0.0)

    # Init missing date list
    missing_dates = []
    # Start the processing loop iterating on batch of dates
    start = time.time()

    for i in range(0, len(dates), batch):
        # The list of dates (as strings) included in the batch
        batch_dates = dates[i : i + batch]

        # Format the string dates to np.datetime64
        times = np.array(
            [
                np.datetime64(d.replace("_", "T").replace("h", ":") + ":00")
                for d in batch_dates
            ],
            dtype="datetime64[ns]",
        )
        print(
            f"Treating date {times}"
            if len(times) == 1
            else f"Treating date range {(str(times[0]),str(times[-1]))}"
        )

        # Init the batch dicts of data
        batch_surf_var = {}
        batch_p_var = {}

        # Init missing date list for the batch
        missing_batch_dates = []
        # Iterate on each date in the batch
        for d in batch_dates:

            # Temporary pressure level variables
            date_p_level = {}

            # Check if the date is valid, if not skip it
            if len(os.listdir(osp.join(data_path, d))) != len(files):
                miss_date = np.datetime64(d.replace("_", "T").replace("h", ":") + ":00")
                missing_batch_dates.append(miss_date)
                times = np.delete(times, np.where(times == miss_date))
                continue

            # Fill the variables with the data from the right file
            for f in files:
                # Load the numpy file of the varname variable
                varname = f.split("_")[1]
                data = da.from_array(
                    np.load(osp.join(data_path, d, f)), chunks=surface_chunks
                )

                # Put the data in the proper dict
                if "hpa" in f:
                    date_p_level.setdefault(f"{varname}_level", []).append(data)
                else:
                    batch_surf_var.setdefault(varname, []).append(data)

            # Stack the pressure level data
            for k_level, v in date_p_level.items():
                var_key = k_level.split("_level")[0]
                batch_p_var.setdefault(var_key, []).append(
                    da.stack(v, axis=-1).rechunk(plevel_chunks)
                )

        # Store the missing dates
        if missing_batch_dates:
            print(f"There are {len(set(missing_batch_dates))} missing dates.")
            missing_dates.extend(missing_batch_dates)

        # Check if there is
        if batch_surf_var and batch_p_var:
            # Convert each batch data to dask.array
            for var_key, var in batch_surf_var.items():
                batch_surf_var[var_key] = da.stack(var, axis=0).rechunk(
                    (1,) + surface_chunks
                )
            for var_key, var in batch_p_var.items():
                batch_p_var[var_key] = da.stack(var, axis=0).rechunk(
                    (1,) + plevel_chunks
                )

            # Merge the two dicts
            batch_dict = {**batch_p_var, **batch_surf_var}

            # Create the xarray dataset
            ds = xr.Dataset(
                {
                    "10m_u_component_of_wind": (
                        ["time", "latitude", "longitude"],
                        batch_dict[var_info["10m_u_component_of_wind"]["short_name"]],
                    ),
                    "10m_v_component_of_wind": (
                        ["time", "latitude", "longitude"],
                        batch_dict[var_info["10m_v_component_of_wind"]["short_name"]],
                    ),
                    "2m_relative_humidity": (
                        ["time", "latitude", "longitude"],
                        batch_dict[var_info["2m_relative_humidity"]["short_name"]],
                    ),
                    "2m_temperature": (
                        ["time", "latitude", "longitude"],
                        batch_dict[var_info["2m_temperature"]["short_name"]],
                    ),
                    "geopotential": (
                        ["time", "latitude", "longitude", "level"],
                        batch_dict[var_info["geopotential"]["short_name"]],
                    ),
                    "geopotential_at_surface": (
                        ["latitude", "longitude"],
                        z_surf,
                    ),
                    "land_sea_mask": (["latitude", "longitude"], landsea_mask),
                    "temperature": (
                        ["time", "latitude", "longitude", "level"],
                        batch_dict[var_info["temperature"]["short_name"]],
                    ),
                    "total_precipitation": (
                        ["time", "latitude", "longitude"],
                        batch_dict[var_info["total_precipitation"]["short_name"]],
                    ),
                    "u_component_of_wind": (
                        ["time", "latitude", "longitude", "level"],
                        batch_dict[var_info["u_component_of_wind"]["short_name"]],
                    ),
                    "v_component_of_wind": (
                        ["time", "latitude", "longitude", "level"],
                        batch_dict[var_info["v_component_of_wind"]["short_name"]],
                    ),
                },
                coords={
                    "time": times,
                    "latitude": np.flip(latitude),
                    "longitude": longitude,
                    "level": levels,
                },
            )

            # Store the dataset in zarr, either creating it or updating an existing one
            if not osp.exists(zarr_path):
                first_date = str(times[0]).split("T")[0]
                encoding = {var: {"compressor": compressor} for var in ds.data_vars}
                encoding.update({"time": {"units": f"hours since {first_date}"}})
                ds.to_zarr(
                    store=zarr_path,
                    mode="w",
                    group="/",
                    encoding=encoding,
                    consolidated=True,
                    zarr_format=2,
                )
            else:
                ds.to_zarr(
                    store=zarr_path,
                    mode="a-",
                    group="/",
                    consolidated=True,
                    zarr_format=2,
                    append_dim="time",
                )

    print(f"Conversion done in : {time.time() - start}s")

    # Finalize the dataset, adding the missing dates and statistics
    start = time.time()

    zarr_ds = zarr.open_group(zarr_path, zarr_version=2)
    with xr.open_zarr(zarr_path, zarr_format=2, consolidated=True) as xr_ds:

        zarr_ds.attrs.put({"missing_dates": sorted([str(m) for m in missing_dates])})
        for var in xr_ds.variables:
            zarr_ds.get(var).attrs.update(var_info[var])

        # Prepare the time diff dataarray to filter in the next step the invalid time steps
        time_length = xr_ds.time.size
        time_diff = {}
        for delta_t in time_deltas:
            if delta_t < time_length:
                d1, d2 = xr.align(
                    xr_ds.time[delta_t:], xr_ds.time[:-delta_t], join="override"
                )
                time_diff[delta_t] = d1 - d2
            else:
                print(
                    f"Warning: delta_t' value '{delta_t}' skipped. The 'delta_t' value is larger "
                    f"than the dataset size: {delta_t} > {time_length}"
                )
                time_deltas.remove(delta_t)

        # Compute the statistics
        for var in xr_ds.data_vars:
            print(f"Computing statistics for {var}")
            op_dims = (
                list(xr_ds[var].dims[:-1])
                if "level" in xr_ds[var].dims
                else list(xr_ds[var].dims)
            )
            zarr_ds.get(var).attrs.update(
                {
                    "mean": xr_ds[var].mean(dim=op_dims, skipna=True).values.tolist(),
                    "std": xr_ds[var]
                    .std(dim=op_dims, skipna=True, ddof=1)
                    .values.tolist(),
                    "min": xr_ds[var].min(dim=op_dims, skipna=True).values.tolist(),
                    "max": xr_ds[var].max(dim=op_dims, skipna=True).values.tolist(),
                }
            )
            if "time" in xr_ds[var].coords:
                for delta_t in time_deltas:
                    valid_diff = (
                        xr_ds[var]
                        .diff(dim="time")
                        .where(
                            time_diff[delta_t] == np.timedelta64(delta_t, "h"),
                            drop=True,
                        )
                    )
                    zarr_ds.get(var).attrs.update(
                        {
                            f"std_diff_{delta_t}": valid_diff.std(
                                dim=op_dims, skipna=True, ddof=1
                            ).values.tolist()
                        }
                    )

    # Sync zarr .zmetadata
    zarr.consolidate_metadata(zarr_path)

    print(f"Missing dates and statistics computed in : {time.time() - start}s")


if __name__ == "__main__":
    # The number of dates added to the zarr dataset at once
    batch = 5000

    main(batch)
    print("Converting Numpy Titan to Zarr Titan is done ! 🥳 ")
