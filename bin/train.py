import random
import time
from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as pl

# Torch import
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning_fabric.utilities import seed
from torch.utils.data import Dataset

from pnia.datasets import SmeagolDataset, TitanDataset
from pnia.datasets.titan import TitanHyperParams
from pnia.models import GraphLAM, HiLAM, HiLAMParallel

# From the package
from pnia.models.ar_model import Graph, HyperParam

DATASETS = {
    "titan": {"dataset": TitanDataset},
    "smeagol": {"dataset": SmeagolDataset},
}

MODELS = {
    "graph_lam": GraphLAM,
    "hi_lam": HiLAM,
    "hi_lam_parallel": HiLAMParallel,
}


@dataclass
class TrainingParams:
    evaluation: str = None  # None (train model)
    precision: str = "32"
    val_interval: int = 1
    epochs: int = 200
    devices: int = 1
    profiler: str = "None"
    run_id: int = 1
    no_log: bool = False


def main(
    model,
    training_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    tp: TrainingParams,
    hp: HyperParam,
):

    train_loader = training_dataset.loader
    val_loader = val_dataset.loader
    test_loader = test_dataset.loader

    # Instatiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # torch.set_float32_matmul_precision("high") # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    prefix = ""
    if tp.evaluation:
        prefix += f"eval-{tp.evaluation}-"

    run_name = (
        f"{prefix}{str(hp.dataset)}{hp.graph.model}-{hp.graph.processor_layers}x{hp.graph.hidden_dim}-"
        f"{time.strftime('%m%d%H:%M')}-{tp.run_id}"
    )

    callback_list = []
    if not tp.no_log:
        print(f"Model and checkpoint will be saved in saved_models/{run_name}")
        callback_list.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=f"saved_models/{run_name}",
                filename="min_val_loss",
                monitor="val_mean_loss",
                mode="min",
                save_last=True,
            )
        )

    # Logger
    if tp.no_log:
        logger = None
    else:
        logger = TensorBoardLogger(
            save_dir="/scratch/shared/pnia/logs",
            name=f"{args.model}/{args.dataset}",
            default_hp_metric=False,
        )

    if tp.profiler == "pytorch":
        profiler = PyTorchProfiler(
            dirpath=f"/scratch/shared/pnia/logs/{args.model}/{args.dataset}",
            filename=f"profile_{tp.run_id}",
            export_to_chrome=True,
        )
        print("Initiate pytorchProfiler")
    else:
        profiler = None
        print(f"No profiler set {tp.profiler}")

    trainer = pl.Trainer(
        devices=tp.devices,
        num_nodes=1,
        max_epochs=tp.epochs,
        deterministic=True,
        strategy="ddp",
        accelerator=device_name,
        logger=logger,
        profiler=profiler,
        log_every_n_steps=1,
        callbacks=callback_list,
        check_val_every_n_epoch=tp.val_interval,
        precision=tp.precision,
    )

    if tp.evaluation:
        if tp.evaluation == "val":
            eval_loader = val_loader
        else:
            eval_loader = test_loader
        trainer.test(model=model, dataloaders=eval_loader)
    else:
        # Train model
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )


if __name__ == "__main__":

    path = Path(__file__)

    parser = ArgumentParser(description="Train or evaluate models for LAM")
    # For our implementation
    parser.add_argument(
        "--dataset",
        type=str,
        default="titan",
        help="Dataset to use",
        choices=["titan", "smeagol"],
    )
    parser.add_argument(
        "--data_conf",
        type=str,
        default=path.parent.parent / "pnia/xp_conf/smeagol.json",
        help="Configuration file for this dataset. Used only for smeagol dataset right now.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="graph_lam",
        help="Model architecture to train/evaluate (default: graph_lam)",
    )

    # Old arguments from nlam
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )

    # Model architecture
    parser.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Graph to load and use in graph-based model (default: multiscale)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensionality of all hidden representations (default: 64)",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs (default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=4,
        help="Number of GNN layers in processor GNN (default: 4)",
    )
    parser.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean) (default: sum)",
    )

    # Training options
    parser.add_argument(
        "--loss", type=str, default="mse", help="Loss function to use (default: mse)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run (default: 1)",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during evaluation (default: 1)",
    )

    # General options
    parser.add_argument(
        "--epochs", type=int, default=200, help="upper epoch limit (default: 200)"
    )
    parser.add_argument("--gpus", type=int, default=1, help="Gpus asked.")
    parser.add_argument(
        "--profiler",
        type=str,
        default="None",
        help="Profiler required. Possibilities are ['simple', 'pytorch', 'None']",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--load", type=str, help="Path to load model parameters from (default: None)"
    )
    parser.add_argument(
        "--restore_opt",
        type=int,
        default=0,
        help="If optimizer state shoudl be restored with model (default: 0 (false))",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )

    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test) (default: None (train model))",
    )

    # New arguments
    parser.add_argument(
        "--standardize",
        action=BooleanOptionalAction,
        default=False,
        help="Do we need to standardize",
    )
    parser.add_argument(
        "--diagnose",
        action=BooleanOptionalAction,
        default=False,
        help="Do we need to show extra print ?",
    )
    parser.add_argument(
        "--memory",
        action=BooleanOptionalAction,
        default=False,
        help="Do we need to save memory ?",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=0,
        help="Do we select a subset ? 0 means Full dataset set is used. Otherwise, number of sample to use.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="How many ARSteps do we use ? ",
    )
    parser.add_argument(
        "--no_log",
        action=BooleanOptionalAction,
        default=False,
        help="When activated, log are not stored and models are not saved. Use in dev mode.",
    )
    args, other = parser.parse_known_args()

    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.eval in (None, "val", "test"), f"Unknown eval setting: {args.eval}"

    args, others = parser.parse_known_args()

    random_run_id = random.randint(0, 9999)
    seed.seed_everything(args.seed)

    # Initialisation du dataset
    if args.dataset == "smeagol":
        training_dataset, validation_dataset, test_dataset = SmeagolDataset.from_json(
            args.data_conf,
            args={
                "train": {
                    "nb_pred_steps": args.steps,
                    "diagnose": args.diagnose,
                    "standardize": args.standardize,
                    "subset": args.subset,
                },
                "valid": {
                    "nb_pred_steps": 5,
                    "standardize": args.standardize,
                    "subset": args.subset,
                },
                "test": {
                    "nb_pred_steps": 5,
                    "standardize": args.standardize,
                    "subset": args.subset,
                },
            },
        )

    elif args.dataset == "titan":
        hparams_train_dataset = TitanHyperParams(**{"nb_pred_steps": 1})
        training_dataset = DATASETS["titan"]["dataset"](hparams_train_dataset)
        hparams_val_dataset = TitanHyperParams(
            **{"nb_pred_steps": 19, "split": "valid"}
        )
        validation_dataset = DATASETS["titan"]["dataset"](hparams_val_dataset)
        hparams_test_dataset = TitanHyperParams(
            **{"nb_pred_steps": 19, "split": "test"}
        )
        test_dataset = DATASETS["titan"]["dataset"](hparams_test_dataset)

    print(f"Activation of memory saving option {args.memory}")
    # Lie uniquement au modele choisi
    graph = Graph(
        model=args.model,
        name=args.graph,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        processor_layers=args.processor_layers,
        mesh_aggr=args.mesh_aggr,
        checkpoint=args.memory,
        offload=args.memory,
    )
    # On rajoute le modele dedans
    hp = HyperParam(
        dataset=training_dataset,
        graph=graph,
        lr=args.lr,
        loss=args.loss,
        n_example_pred=args.n_example_pred,
    )
    # Parametre pour le training uniquement
    tp = TrainingParams(
        evaluation=args.eval,
        precision=args.precision,
        val_interval=args.val_interval,
        epochs=args.epochs,
        devices=args.gpus,
        profiler=args.profiler,
        no_log=args.no_log,
        run_id=random_run_id,
    )

    model_class = MODELS[graph.model]

    if args.load:
        # Appel pas correct actuelement
        model = model_class.load_from_checkpoint(args.load, args=args)
        if args.restore_opt:
            # Save for later
            # Unclear if this works for multi-GPU
            model.opt_state = torch.load(args.load)["optimizer_states"][0]
    else:
        model = model_class(hp)

    main(model, training_dataset, validation_dataset, test_dataset, tp, hp)
