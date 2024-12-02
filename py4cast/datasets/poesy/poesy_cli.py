import datetime as dt
from pathlib import Path
from typing import List

from typer import Typer

from py4cast.datasets import compute_dataset_stats as cds
from py4cast.datasets.access import Param
from py4cast.datasets.base import get_param_list
from py4cast.datasets.poesy import TitanDataset
from py4cast.datasets.poesy.settings import DEFAULT_CONFIG

app = Typer()


@app.command()
def prepare(dataset: PoesyDataset, path_config: Path):
    print("--> Preparing Poesy Dataset...")

    print("Load train dataset configuration...")
    with open(path_config, "r") as fp:
        conf = json.load(fp)

    print("Computing stats on each parameter...")
    conf["settings"]["standardize"] = True
    train_ds, _, _ = PoesyDataset.from_json(
        fname=path_config,
        num_input_steps=2,
        num_pred_steps_train=1,
        num_pred_steps_val_test=1,
    )
    compute_parameters_stats(PoesyDataset)

    print("Computing time stats on each parameters, between 2 timesteps...")
    conf["settings"]["standardize"] = True
    train_ds, _, _ = PoesyDataset.from_json(
        fname=path_config,
        num_input_steps=2,
        num_pred_steps_train=1,
        num_pred_steps_val_test=1,
    )
    compute_time_step_stats(PoesyDataset)

    return train_ds


@app.command()
def describe(path_config: Path = DEFAULT_CONFIG):
    """Describes Titan."""
    train_ds, _, _ = PoesyDataset.from_json(path_config, 2, 1, 5)
    train_ds.dataset_info.summary()
    print("Len dataset : ", len(train_ds))
    print("First Item description :")
    print(train_ds[0])


@app.command()
def speedtest(path_config: Path = DEFAULT_CONFIG, n_iter: int = 5):
    print("Speed test:")
    start_time = time.time()
    for i in tqdm.trange(n_iter, desc="Loading samples"):
        _ = next(data_iter)
    delta = time.time() - start_time
    speed = args.n_iter / delta
    print(f"Loading speed: {round(speed, 3)} sample(s)/sec")


if __name__ == "__main__":
    app()
