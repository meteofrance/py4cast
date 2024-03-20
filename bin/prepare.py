from pathlib import Path

import torch
import typer
from tqdm import tqdm

from pnia.datasets import SmeagolDataset, TitanDataset, create_grid_features
from pnia.datasets.titan import TitanHyperParams
from pnia.models.nlam import create_mesh
from pnia.settings import CACHE_DIR

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
    train_ds.compute_timestep_stats()


@smeagol_app.command("all")
def smeagol_all(dataconf=path.parent.parent / "pnia/xp_conf/smeagol.json"):
    train_ds, _, _ = SmeagolDataset.from_json(dataconf)
    create_grid_features.prepare(dataset=train_ds)
    train_ds.compute_parameters_stats()
    train_ds.compute_timestep_stats()


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
    cache_dir_path = CACHE_DIR / "neural_lam" / str(dataset)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    create_mesh.build_graph_for_grid(
        train_ds.statics.grid_info, cache_dir_path, hierarchical=hierarchical
    )


if __name__ == "__main__":
    app()
