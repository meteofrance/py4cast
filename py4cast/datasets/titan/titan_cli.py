import datetime as dt
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import tqdm
from typer import Typer

from py4cast.datasets import compute_dataset_stats as cds
from py4cast.datasets.access import WeatherParam
from py4cast.datasets.base import DatasetABC, get_param_list
from py4cast.datasets.titan import TitanAccessor
from py4cast.datasets.titan.settings import DEFAULT_CONFIG

app = Typer()


def process_sample_dataset(dataset: DatasetABC, date: dt.datetime, params: List[Param]):
    """Saves each 2D parameter data of the given date as one NPY file."""
    for param in params:
        dest_file = dataset.get_filepath(dataset.name, param, date, file_format="npy")
        dest_file.parent.mkdir(exist_ok=True)
        if not dest_file.exists():
            try:
                arr = dataset.accessor.load_data_from_disk(
                    dataset.name, param, date, file_format="grib"
                )
                np.save(dest_file, arr)
            except Exception as e:
                print(e)
                print(
                    f"WARNING: Could not load grib data {param.name} {param.level} {date}. Skipping sample."
                )
                break


@app.command()
def prepare(
    path_config: Path = DEFAULT_CONFIG,
    num_input_steps: int = 1,
    num_pred_steps_train: int = 1,
    num_pred_steps_val_test: int = 1,
    convert_grib2npy: bool = False,
    compute_stats: bool = True,
    write_valid_samples_list: bool = True,
):
    """Prepares Titan dataset for training.
    This command will:
        - create all needed folders
        - convert gribs to npy and rescale data to the wanted grid
        - establish a list of valid samples for each set
        - computes statistics on all weather parameters."""
    print("--> Preparing Titan Dataset...")

    print("Load dataset configuration...")
    with open(path_config, "r") as fp:
        conf = json.load(fp)

    print("Creating folders...")
    train_ds, valid_ds, test_ds = DatasetABC.from_dict(
        path_config.stem,
        conf,
        num_input_steps,
        num_pred_steps_train,
        num_pred_steps_val_test,
    )
    train_ds.cache_dir.mkdir(exist_ok=True)
    data_dir = train_ds.cache_dir / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"Dataset will be saved in {train_ds.cache_dir}")

    if convert_grib2npy:
        print("Converting gribs to npy...")
        param_list = get_param_list(
            conf,
            train_ds.grid,
            TitanAccessor.load_param_info,
            TitanAccessor.get_weight_per_level,
        )
        sum_dates = (
            list(train_ds.period.date_list)
            + list(valid_ds.period.date_list)
            + list(test_ds.period.date_list)
        )
        dates = sorted(list(set(sum_dates)))
        for date in tqdm.tqdm(dates):
            process_sample_dataset(train_ds, date, param_list)
        print("Done!")

    if write_valid_samples_list:
        train_ds.write_list_valid_samples()
        valid_ds.write_list_valid_samples()
        test_ds.write_list_valid_samples()

    if compute_stats:
        conf["settings"]["standardize"] = False
        train_ds, valid_ds, test_ds = DatasetABC.from_dict(
            path_config.stem,
            conf,
            num_input_steps,
            num_pred_steps_train,
            num_pred_steps_val_test,
        )
        print("Computing stats on each parameter...")
        cds.compute_parameters_stats(train_ds)

        conf["settings"]["standardize"] = True
        train_ds, valid_ds, test_ds = DatasetABC.from_dict(
            path_config.stem,
            conf,
            num_input_steps,
            num_pred_steps_train,
            num_pred_steps_val_test,
        )
        print("Computing time stats on each parameters, between 2 timesteps...")
        cds.compute_time_step_stats(train_ds)


@app.command()
def describe(path_config: Path = DEFAULT_CONFIG):
    """Describes Titan."""
    train_ds, _, _ = DatasetABC.from_json(path_config, 2, 1, 5)
    train_ds.dataset_info.summary()
    print("Len dataset : ", len(train_ds))
    print("First Item description :")
    print(train_ds[0])


@app.command()
def plot(path_config: Path = DEFAULT_CONFIG):
    """Plots a png and a gif for one sample."""
    train_ds, _, _ = DatasetABC.from_json(path_config, 2, 1, 5)
    print("Plot gif of one sample...")
    sample = train_ds.sample_list[0]
    sample.plot_gif("test.gif")
    print("Plot png for one step of sample...")
    item = sample.load(no_standardize=True)
    sample.plot(item, 0, "test.png")


@app.command()
def speedtest(path_config: Path = DEFAULT_CONFIG, n_iter: int = 5):
    """Makes a loading speed test."""
    train_ds, _, _ = DatasetABC.from_json(path_config, 2, 1, 5)
    data_iter = iter(train_ds.torch_dataloader())
    print("Dataset file_format: ", train_ds.settings.file_format)
    print("Speed test:")
    start_time = time.time()
    for _ in tqdm.trange(n_iter, desc="Loading samples"):
        next(data_iter)
    delta = time.time() - start_time
    print("Elapsed time : ", delta)
    speed = n_iter / delta
    print(f"Loading speed: {round(speed, 3)} batch(s)/sec")


if __name__ == "__main__":
    app()
