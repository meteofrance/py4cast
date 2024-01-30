from pathlib import Path

import torch
import typer
from pnia.datasets import SmeagolDataset, TitanDataset, create_grid_features
from pnia.datasets.titan import TitanHyperParams
from pnia.models.nlam import create_mesh

torch.set_num_threads(8)
path = Path(__file__)
app = typer.Typer()
smeagol_app = typer.Typer()
titan_app = typer.Typer()
app.add_typer(smeagol_app, name="smeagol")
app.add_typer(titan_app, name="titan")


@smeagol_app.command("grid")
def smeagol_grid(dataconf=path.parent.parent / "pnia/xp_conf/smeagol.json"):
    """
    Compute parameter stats for smeagol dataset using the configuration file.
    This stats are stored and load from the training dataset and used for standardisation.
    Alse prepare grid_features.
    """
    train_ds, _, _ = SmeagolDataset.from_json(dataconf)
    create_grid_features.prepare(dataset=train_ds)


@smeagol_app.command("stats")
def smeagol_stats(dataconf=path.parent.parent / "pnia/xp_conf/smeagol.json"):
    train_ds, _, _ = SmeagolDataset.from_json(dataconf)
    train_ds.compute_parameters_stats()


@smeagol_app.command("diffstats")
def smeagol_diffstats(dataconf=path.parent.parent / "pnia/xp_conf/smeagol.json"):
    train_ds, _, _ = SmeagolDataset.from_json(dataconf)
    train_ds.standardize = True
    train_ds.compute_timestep_stats()


@smeagol_app.command("all")
def smeagol_all(dataconf=path.parent.parent / "pnia/xp_conf/smeagol.json"):
    train_ds, _, _ = SmeagolDataset.from_json(dataconf)
    create_grid_features.prepare(dataset=train_ds)
    train_ds.compute_parameters_stats()
    train_ds.standardize = True
    train_ds.compute_timestep_stats()


@smeagol_app.command("test")
def testsmeagol(dataconf=path.parent.parent / "pnia/xp_conf/smeagol.json"):
    """
    Used to test a particular function of smeagol.
    This function could change...
    """
    train_ds, _, _ = SmeagolDataset.from_json(
        dataconf, {"train": {"standardize": True}}
    )
    train_ds.standardize = True
    print("Forcing", train_ds.shortnames("forcing"))
    print("Weather", train_ds.shortnames("input_output"))
    print("Weather", train_ds.shortnames("diagnostic"))
    print("Weather", train_ds.shortnames("input"))
    print("Weather", train_ds.shortnames("all"))
    print("Weather", train_ds.shortnames("Toto"))
    # datas = train_ds.load_dataset_stats()


@titan_app.command("all")
def titan_all():
    train_ds = TitanDataset(TitanHyperParams(**{"nb_pred_steps": 1}))
    create_grid_features.prepare(dataset=train_ds)
    train_ds.compute_parameters_stats()
    # train_ds.standardize = True  # TODO : standardisation de Titan
    train_ds.compute_timestep_stats()


@app.command()
def nlam(
    dataset: str = "smeagol",
    dataconf: str = path.parent.parent / "pnia/xp_conf/smeagol.json",
    hierarchical: bool = True,
):
    """
    Create NeuralLam mesh and grid feature.
    Need to know for which dataset we are working (as some info on the grid is needed)
    """
    if dataset == "smeagol":
        train_ds, _, _ = SmeagolDataset.from_json(dataconf)
    elif dataset == "titan":
        train_ds = TitanDataset(TitanHyperParams())
    else:
        raise NotImplementedError(
            f"Nothing implemented for this dataset rightnow {dataset}"
        )
    create_mesh.prepare(dataset=train_ds, hierarchical=hierarchical)


if __name__ == "__main__":
    app()
