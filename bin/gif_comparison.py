import argparse

from pytorch_lightning import Trainer

from py4cast.datasets import get_datasets
from py4cast.datasets.base import TorchDataloaderSettings, collate_fn, Item
from py4cast.lightning import AutoRegressiveLightning, ArLightningHyperParam
from typing import List, Tuple
import torch
from py4cast.plots import DomainInfo
from tqdm import trange
import einops
from pathlib import Path


import matplotlib.pyplot as plt
import gif


def get_model_and_hparams(ckpt: Path, num_pred_steps:int) -> Tuple[AutoRegressiveLightning, ArLightningHyperParam]:
    model = AutoRegressiveLightning.load_from_checkpoint(ckpt)
    hparams = model.hparams["hparams"]
    hparams.num_pred_steps_val_test = num_pred_steps
    model.eval()
    return model, hparams


def get_item_for_date(date:str, hparams: ArLightningHyperParam) -> Item:
    """ Returns Item containing sample of chosen date.
    Date should be in format YYYYMMDDHH.
    """
    config_override = {"periods": {"test": {"start": args.date, "end": args.date}}}
    _, _, test_ds = get_datasets(
            hparams.dataset_name,
            hparams.num_input_steps,
            hparams.num_pred_steps_train,
            hparams.num_pred_steps_val_test,
            hparams.dataset_conf,
            config_override=config_override,
        )
    item = test_ds.__getitem__(0)
    return item


def make_forecast(model: AutoRegressiveLightning, item: Item) -> torch.tensor:
    batch_item = collate_fn([item])
    preds = model(batch_item)
    num_spatial_dims = preds.num_spatial_dims
    forecast = preds.tensor
    # Here we reshape output from GNNS to be on the grid
    if preds.num_spatial_dims == 1:
        forecast = einops.rearrange(
            forecast, "b t (x y) n -> b t x y n", x=model.grid_shape[0]
        )
    return forecast[0]


@gif.frame
def plot_frame(t:int, idx_feature:int, target: torch.tensor, predictions: List[torch.tensor], domain_info: DomainInfo,
    title=None, models_names=None)-> None:
    # Get common scale for values
    vmin = target[:, :, :, idx_feature].min().cpu().item()
    vmax = target[:, :, :, idx_feature].max().cpu().item()

    nb_preds = len(predictions) + 1
    fig, axes = plt.subplots(
        1, nb_preds, figsize=(15, 7), subplot_kw={"projection": domain_info.projection}
    )

    for i, data in enumerate([target] + predictions):
        axes[i].coastlines()
        array = data.cpu().detach().numpy()

        im = axes[i].imshow(
            data.cpu().detach().numpy()[t, :, :, idx_feature],
            origin="lower",
            extent=domain_info.grid_limits,
            vmin=vmin,
            vmax=vmax,
            cmap="plasma",
        )
        if models_names:
            axes[i].set_title(models_names[i], size=15)

    if title:
        fig.suptitle(title, size=20)
    plt.tight_layout()


if __name__ == "__main__":

    # Parse arguments: model_path, dataset name and config file and finally date for inference
    parser = argparse.ArgumentParser("py4cast Inference script")
    parser.add_argument("--ckpt", type=str, action='append', help="Paths to the model checkpoint", required=True)
    parser.add_argument("--date", type=str, help="Date for inference. Format YYYYMMDDHH.",  required=True)
    parser.add_argument(
        "--num_pred_steps",
        type=int,
        default=12,
        help="Number of auto-regressive steps/prediction steps.",
    )
    args = parser.parse_args()

    y_preds = []
    models_names = []
    for ckpt in args.ckpt:
        model, hparams = get_model_and_hparams(ckpt, args.num_pred_steps)
        item = get_item_for_date(args.date, hparams)
        forecast = make_forecast(model, item)
        models_names.append(f"{hparams.model_name}\n{hparams.save_path.name}")
        y_preds.append(forecast)
    
    y_true = item.outputs.tensor
    domain_info = hparams.dataset_info.domain_info
    models_names = ["AROME Analysis"] + models_names

    for preds in y_preds:
        print(preds.shape)
    print("Models: ", models_names)

    for feature_name in item.inputs.feature_names[:5]:
        print(feature_name)
        idx_feature = item.inputs.feature_names_to_idx[feature_name]
        frames = []
        for t in trange(args.num_pred_steps):
            title = f"{feature_name} - {args.date} +{t+1}h"
            frame = plot_frame(t, idx_feature, y_true, y_preds, domain_info, title, models_names)
            frames.append(frame)
        gif.save(frames, f"{args.date}_{feature_name}.gif", duration=250)
