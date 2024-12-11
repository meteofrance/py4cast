import json
import time
from pathlib import Path
import tqdm
from typer import Typer

from py4cast.datasets import compute_dataset_stats as cds
from py4cast.datasets.poesy import PoesyDataset
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
    cds.compute_parameters_stats(PoesyDataset)

    print("Computing time stats on each parameters, between 2 timesteps...")
    conf["settings"]["standardize"] = True
    train_ds, _, _ = PoesyDataset.from_json(
        fname=path_config,
        num_input_steps=2,
        num_pred_steps_train=1,
        num_pred_steps_val_test=1,
    )
    cds.compute_time_step_stats(PoesyDataset)

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
def plot(path_config: Path = DEFAULT_CONFIG):
    """Plots a png and a gif for one sample."""
    train_ds, _, _ = PoesyDataset.from_json(path_config, 2, 1, 5)
    print("Plot gif of one sample...")
    sample = train_ds.sample_list[0]
    sample.plot_gif("test.gif")
    print("Plot png for one step of sample...")
    item = sample.load(no_standardize=True)
    sample.plot(item, 0, "test.png")


@app.command()
def speedtest(path_config: Path = DEFAULT_CONFIG, n_iter: int = 5):
    print("Speed test:")
    train_ds, _, _ = PoesyDataset.from_json(path_config, 2, 1, 5)
    data_iter = iter(train_ds.torch_dataloader())
    start_time = time.time()
    for i in tqdm.trange(n_iter, desc="Loading samples"):
        _ = next(data_iter)
    delta = time.time() - start_time
    speed = n_iter / delta
    print(f"Loading speed: {round(speed, 3)} sample(s)/sec")


if __name__ == "__main__":
    app()
