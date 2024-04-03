from pathlib import Path

import torch
import typer
from tqdm import tqdm

from pnia.datasets import SmeagolDataset, TitanDataset
from pnia.datasets.titan import TitanHyperParams

torch.set_num_threads(8)
path = Path(__file__)
app = typer.Typer()
smeagol_app = typer.Typer()
titan_app = typer.Typer()
app.add_typer(smeagol_app, name="smeagol")
app.add_typer(titan_app, name="titan")


@smeagol_app.command("stats")
def smeagol_stats(dataconf=path.parent.parent / "config/smeagolstats.json"):
    train_ds, _, _ = SmeagolDataset.from_json(
        dataconf, {"train": {"standardize": False}}
    )
    train_ds.compute_parameters_stats()
    train_ds.settings.standardize = True
    train_ds.compute_time_step_stats()


@smeagol_app.command("test")
def testsmeagol(dataconf=path.parent.parent / "pnia/xp_conf/smeagol.json"):
    """
    Used to test a particular function of smeagol.
    This function could change...
    """
    train_ds, _, _ = SmeagolDataset.from_json(
        dataconf, {"train": {"standardize": True, "nb_pred_steps": 20}}
    )
    print("Forcing", train_ds.shortnames("forcing"))
    print("Weather", train_ds.shortnames("input_output"))
    print("Weather", train_ds.shortnames("diagnostic"))
    print("Weather", train_ds.shortnames("input"))
    print("Weather", train_ds.shortnames("all"))

    for index in tqdm(range(len(train_ds.sample_list))):
        try:
            train_ds.test_sample(index)
        except KeyError:
            print(f"Problem in sample {train_ds.sample_list[index]}")
        except OSError:
            print(f"Problem in sample {train_ds.sample_list[index]}")


@titan_app.command("all")
def titan_all():
    train_ds = TitanDataset(TitanHyperParams(**{"nb_pred_steps": 1}))
    train_ds.compute_parameters_stats()


if __name__ == "__main__":
    app()
