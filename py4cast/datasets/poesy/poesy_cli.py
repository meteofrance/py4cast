import json
import time
from pathlib import Path

import tqdm
from typer import Typer

from py4cast.datasets import compute_dataset_stats as cds
from py4cast.datasets.base import DatasetABC
from py4cast.datasets.poesy import PoesyAccessor
from py4cast.datasets.poesy.settings import DEFAULT_CONFIG

app = Typer()


@app.command()
def prepare(
    path_config: Path = DEFAULT_CONFIG,
    num_input_steps: int = 1,
    num_pred_steps_train: int = 1,
    num_pred_steps_val_test: int = 1,
    compute_stats: bool = True,
):
    """
    Prepares Poesy dataset for training.
    This command will:
        - create all needed folders
        - computes statistics on all weather parameters.
    """

    print("--> Preparing Poesy Dataset...")

    print("Load train dataset configuration...")
    with open(path_config, "r") as fp:
        conf = json.load(fp)

    print("instantiate train dataset configuration...")
    train_ds, _, _ = DatasetABC.from_dict(
        PoesyAccessor,
        name=path_config.stem,
        conf=conf,
        num_input_steps=num_input_steps,
        num_pred_steps_train=num_pred_steps_train,
        num_pred_steps_val_test=num_pred_steps_val_test,
    )

    print("Creating cache folder")
    print(train_ds.cache_dir)
    train_ds.cache_dir.mkdir(exist_ok=True)

    if compute_stats:
        print(f"Dataset stats will be saved in {train_ds.cache_dir}")

        print("Computing stats on each parameter...")
        train_ds.settings.standardize = False

        cds.compute_parameters_stats(train_ds)

        print("Computing time stats on each parameters, between 2 timesteps...")
        train_ds.settings.standardize = True
        cds.compute_time_step_stats(train_ds)

    return train_ds


@app.command()
def describe(path_config: Path = DEFAULT_CONFIG, dataset_name: str = "poesy"):
    """Describes Poesy."""
    train_ds, _, _ = DatasetABC.from_json(
        PoesyAccessor,
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
def plot(path_config: Path = DEFAULT_CONFIG, dataset_name: str = "poesy"):
    """Plots a png and a gif for one sample."""
    train_ds, _, _ = DatasetABC.from_json(
        PoesyAccessor,
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
    path_config: Path = DEFAULT_CONFIG, dataset_name: str = "poesy", n_iter: int = 5
):
    print("Speed test:")
    train_ds, _, _ = DatasetABC.from_json(
        PoesyAccessor,
        dataset_name=dataset_name,
        fname=path_config,
        num_input_steps=2,
        num_pred_steps_train=1,
        num_pred_steps_val_tests=5,
    )
    data_iter = iter(train_ds.torch_dataloader())
    start_time = time.time()
    for i in tqdm.trange(n_iter, desc="Loading samples"):
        _ = next(data_iter)
    delta = time.time() - start_time
    speed = n_iter / delta
    print(f"Loading speed: {round(speed, 3)} sample(s)/sec")


if __name__ == "__main__":
    app()
