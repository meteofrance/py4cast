import json
import time
from pathlib import Path

import numpy as np
import tqdm
from typer import Typer

from py4cast.datasets import compute_dataset_stats as cds
from py4cast.datasets.access import Timestamps
from py4cast.datasets.base import DatasetABC
from py4cast.datasets.titan import TitanAccessor
from py4cast.datasets.titan.settings import DEFAULT_CONFIG, FORMATSTR

app = Typer()


def convert_sample_grib2_numpy(dataset: DatasetABC):
    """Saves each 2D parameter data of the given date as one NPY file."""
    dataset.settings.file_format = "grib"
    sample_list = dataset.sample_list
    for sample in sample_list:
        dest_folder = sample.timestamps.validity_times[0].strftime(FORMATSTR)
        d, v, t = (
            sample.timestamps.datetime,
            sample.timestamps.validity_times[0],
            sample.timestamps.terms[0],
        )
        t = Timestamps(datetime=d, terms=np.array(t), validity_times=[v])
        path = (
            dataset.accessor.get_dataset_path(dataset.name, dataset.grid)
            / "data"
            / dest_folder
        )
        path.mkdir(exist_ok=True)
        for p in sample.params:
            dest_file = dataset.accessor.get_filepath(
                dataset.name, p, v, file_format="npy"
            )
            if not dest_file.exists():
                try:
                    arr = dataset.accessor.load_data_from_disk(
                        dataset.name, p, t, file_format="grib"
                    ).squeeze()
                    np.save(
                        dest_file,
                        arr[
                            dataset.grid.subdomain[0] : dataset.grid.subdomain[1],
                            dataset.grid.subdomain[2] : dataset.grid.subdomain[3],
                        ].astype(np.float32),
                    )
                except Exception as e:
                    print(e)
                    print(
                        f"WARNING: Could not load grib {dataset.accessor.parameter_namer(p)} {p.level} {v}. Skipping."
                    )
                    break
    dataset.settings.file_format = "npy"


@app.command()
def prepare(
    path_config: Path = DEFAULT_CONFIG,
    num_input_steps: int = 1,
    num_pred_steps_train: int = 1,
    num_pred_steps_val_test: int = 1,
    convert_grib2npy: bool = False,
    compute_stats: bool = True,
):
    """Prepares Titan dataset for training.
    This command will:
        - create all needed folders
        - convert gribs to npy and rescale data to the wanted grid
        - computes statistics on all weather parameters."""
    print("--> Preparing Titan Dataset...")

    print("Load dataset configuration...")
    with open(path_config, "r") as fp:
        conf = json.load(fp)

    print("Creating folders...")
    train_ds, valid_ds, test_ds = DatasetABC.from_dict(
        TitanAccessor,
        name=path_config.stem,
        conf=conf,
        num_input_steps=num_input_steps,
        num_pred_steps_train=num_pred_steps_train,
        num_pred_steps_val_test=num_pred_steps_val_test,
    )
    train_ds.cache_dir.mkdir(exist_ok=True)
    data_dir = train_ds.cache_dir / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"Dataset will be saved in {train_ds.cache_dir}")

    if convert_grib2npy:
        train_ds.settings.standardize = False
        valid_ds.settings.standardize = False
        test_ds.settings.standardize = False

        print("Converting gribs to npy...")
        print("train")
        convert_sample_grib2_numpy(train_ds)
        print("validation")
        convert_sample_grib2_numpy(valid_ds)
        print("test")
        convert_sample_grib2_numpy(test_ds)
        print("Done!")

        train_ds.settings.standardize = True
        valid_ds.settings.standardize = True
        test_ds.settings.standardize = True

    if compute_stats:
        del train_ds.sample_list
        train_ds.settings.standardize = False
        print("Computing stats on each parameter...")
        cds.compute_parameters_stats(train_ds)
        del train_ds.sample_list
        train_ds.settings.standardize = True
        print("Computing time stats on each parameters, between 2 timesteps...")
        cds.compute_time_step_stats(train_ds)


@app.command()
def describe(path_config: Path = DEFAULT_CONFIG, dataset_name: str = "titan"):
    """Describes Titan."""
    train_ds, _, _ = DatasetABC.from_json(
        TitanAccessor,
        dataset_name=dataset_name,
        fname=path_config,
        num_input_steps=2,
        num_pred_steps_train=1,
        num_pred_steps_val_tests=5,
    )
    train_ds.dataset_info.summary()
    print("Len dataset : ", len(train_ds))
    print("First Item description :")
    print(train_ds[0])


@app.command()
def plot(path_config: Path = DEFAULT_CONFIG, dataset_name: str = "titan"):
    """Plots a png and a gif for one sample."""
    train_ds, _, _ = DatasetABC.from_json(
        TitanAccessor,
        dataset_name=dataset_name,
        fname=path_config,
        num_input_steps=2,
        num_pred_steps_train=1,
        num_pred_steps_val_tests=5,
    )
    print("Plot gif of one sample...")
    sample = train_ds.sample_list[0]
    sample.plot_gif("test.gif")
    print("Plot png for one step of sample...")
    item = sample.load(no_standardize=True)
    sample.plot(item, 0, "test.png")


@app.command()
def speedtest(
    path_config: Path = DEFAULT_CONFIG, n_iter: int = 5, dataset_name: str = "titan"
):
    """Makes a loading speed test."""
    train_ds, _, _ = DatasetABC.from_json(
        TitanAccessor,
        dataset_name=dataset_name,
        fname=path_config,
        num_input_steps=2,
        num_pred_steps_train=1,
        num_pred_steps_val_tests=5,
    )
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
