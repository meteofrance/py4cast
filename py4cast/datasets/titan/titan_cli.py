import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm, trange
from typer import Typer

from py4cast.datasets import compute_dataset_stats as cds
from py4cast.datasets.base import DatasetABC
from py4cast.datasets.titan import TitanAccessor
from py4cast.datasets.titan.settings import DEFAULT_CONFIG

app = Typer()


def convert_samples_grib2_numpy(dataset: DatasetABC):
    """Saves each 2D parameter data of the given date as one NPY file."""
    dataset.settings.file_format = "grib"  # Use grib files to define valid samples
    domain = dataset.grid.subdomain
    for sample in tqdm(dataset.sample_list, desc="Converting samples"):
        for date in sample.timestamps.validity_times:
            for p in sample.params:
                dest_file = TitanAccessor.get_filepath(
                    dataset.name, p, date, file_format="npy"
                )

                # Create date folder if needed
                Path(dest_file).parent.mkdir(exist_ok=True)
                if dest_file.exists():
                    continue  # already converted
                try:
                    arr = TitanAccessor.load_data_for_date(
                        dataset.name, p, date, file_format="grib"
                    )
                    arr = arr[domain[0] : domain[1], domain[2] : domain[3]]
                    np.save(dest_file, arr.astype(np.float32))
                except Exception as e:
                    print(e)
                    print(
                        f"WARNING: Could not load grib {TitanAccessor.parameter_namer(p)} {p.level} {date}. Skipping."
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

    print("Loading dataset configuration:", path_config)
    with open(path_config, "r") as fp:
        conf = yaml.safe_load(fp)["data"]

    print("Creating folders...")
    train_ds, valid_ds, test_ds = DatasetABC.from_dict(
        TitanAccessor,
        name=conf["dataset_name"],
        conf=conf["dataset_conf"],
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
        train_ds.settings.file_format = "grib"
        valid_ds.settings.file_format = "grib"
        test_ds.settings.file_format = "grib"

        print("Converting gribs to npy...")
        print("train")
        convert_samples_grib2_numpy(train_ds)
        print("validation")
        convert_samples_grib2_numpy(valid_ds)
        print("test")
        convert_samples_grib2_numpy(test_ds)
        print("Done!")

        train_ds.settings.standardize = True
        valid_ds.settings.standardize = True
        test_ds.settings.standardize = True

    if compute_stats:
        if hasattr(train_ds, "sample_list"):
            del train_ds.sample_list
        train_ds.settings.standardize = False
        print("Computing stats on each parameter...")
        cds.compute_parameters_stats(train_ds)
        if hasattr(train_ds, "sample_list"):
            del train_ds.sample_list
        train_ds.settings.standardize = True
        print("Computing time stats on each parameters, between 2 timesteps...")
        cds.compute_time_step_stats(train_ds)


def load_simple_train_ds(path_config: Path):
    print("Using config", path_config)
    with open(path_config, "r") as fp:
        conf = yaml.safe_load(fp)["data"]
    train_ds, _, _ = DatasetABC.from_dict(
        TitanAccessor,
        name=conf["dataset_name"],
        conf=conf["dataset_conf"],
        num_input_steps=2,
        num_pred_steps_train=1,
        num_pred_steps_val_test=5,
    )
    return train_ds


@app.command()
def describe(path_config: Path = DEFAULT_CONFIG):
    """Describes Titan."""
    train_ds = load_simple_train_ds(path_config)
    train_ds.dataset_info.summary()
    print("Len dataset : ", len(train_ds))
    print("First Item description :")
    print(train_ds[0])


@app.command()
def plot(path_config: Path = DEFAULT_CONFIG):
    """Plots a png and a gif for one sample."""
    train_ds = load_simple_train_ds(path_config)
    print("Plot gif of one sample...")
    sample = train_ds.sample_list[0]
    print(sample)
    sample.plot_gif("test.gif")
    print("Plot png for one step of sample...")
    item = sample.load(no_standardize=True)
    sample.plot(item, 0, "test.png")


@app.command()
def speedtest(path_config: Path = DEFAULT_CONFIG, n_iter: int = 5):
    """Makes a loading speed test."""
    train_ds = load_simple_train_ds(path_config)
    data_iter = iter(train_ds.torch_dataloader())
    print("Dataset file_format: ", train_ds.settings.file_format)
    print("Speed test:")
    start_time = time.time()
    for _ in trange(n_iter, desc="Loading samples"):
        next(data_iter)
    delta = time.time() - start_time
    print("Elapsed time : ", delta)
    speed = n_iter / delta
    print(f"Loading speed: {round(speed, 3)} batch(s)/sec")


if __name__ == "__main__":
    app()
