import json
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import cartopy
import einops
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchmetrics import Metric
from tueplots import bundles, figsizes

if TYPE_CHECKING:
    from py4cast.datasets.base import ItemBatch
    from py4cast.lightning import AutoRegressiveLightning

from mfai.torch.namedtensor import NamedTensor


@dataclass(slots=True)
class DomainInfo:
    """
    Information required for plotting.
    """

    grid_limits: Tuple[float, float, float, float]
    projection: cartopy.crs


def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of the page width.
    """
    bundle = bundles.neurips2023(usetex=False, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (original_figsize[0] / fraction, original_figsize[1])
    return bundle


@matplotlib.rc_context(fractional_plot_bundle(1))
def plot_error_map(errors, shortnames, units, title=None, step_duration=3):
    """
    Plot a heatmap of errors of different variables at different predictions horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape

    # Normalize all errors to [0,1] for color map
    max_errors = errors_np.max(axis=1)  # d_f
    errors_norm = errors_np / np.expand_dims(max_errors, axis=1)

    fig, ax = plt.subplots(figsize=(15, 10))

    ax.imshow(
        errors_norm,
        cmap="OrRd",
        vmin=0,
        vmax=1.0,
        interpolation="none",
        aspect="auto",
        alpha=0.8,
    )

    # ax and labels
    for (j, i), error in np.ndenumerate(errors_np):
        # Numbers > 9999 will be too large to fit
        formatted_error = f"{error:.3f}" if error < 9999 else f"{error:.2E}"
        ax.text(i, j, formatted_error, ha="center", va="center", usetex=False)

    # Ticks and labels
    label_size = 15
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
    pred_hor_h = step_duration * pred_hor_i  # Prediction horiz. in hours
    ax.set_xticklabels(pred_hor_h, size=label_size)
    ax.set_xlabel("Lead time (h)", size=label_size)

    ax.set_yticks(np.arange(d_f))
    y_ticklabels = [f"{name} ({unit})" for name, unit in zip(shortnames, units)]
    ax.set_yticklabels(y_ticklabels, rotation=30, size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig


# Plot psd_pred and psd_target in function of k
def plot_log_psd(
    k: np.ndarray, psd_pred: np.ndarray, psd_target: np.ndarray, title: str = ""
):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(k, psd_pred, label="pred")
    ax.plot(k, psd_target, label="target")
    ax.set_xlabel("k")
    ax.set_ylabel("psd_k")
    ax.legend()
    ax.set_title(title)
    plt.yscale("log")
    plt.close()
    return fig


@matplotlib.rc_context(fractional_plot_bundle(1))
def plot_prediction(
    pred,
    target,
    interior_mask,
    domain_info: DomainInfo,
    title=None,
    vrange=None,
):
    """
    Plot example prediction and grond truth.
    Each has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange

    # Set up masking of border region
    # mask_reshaped = obs_mask.reshape(*grid_shape)
    pixel_alpha = interior_mask.clamp(0.7, 1).cpu().numpy()  # Faded border region

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 7), subplot_kw={"projection": domain_info.projection}
    )

    # Plot pred and target
    for ax, data in zip(axes, (target, pred)):
        ax.coastlines()  # Add coastline outlines
        # data_grid = data.reshape(*grid_shape).cpu().numpy()
        im = ax.imshow(
            data.cpu().numpy(),
            origin="lower",
            extent=domain_info.grid_limits,
            alpha=pixel_alpha,
            vmin=vmin,
            vmax=vmax,
            cmap="plasma",
        )

    # Ticks and labels
    axes[0].set_title("Ground Truth", size=15)
    axes[1].set_title("Prediction", size=15)
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig


@matplotlib.rc_context(fractional_plot_bundle(1))
def plot_spatial_error(
    error,
    obs_mask,
    domain_info: DomainInfo,
    title=None,
    vrange=None,
):
    """
    Plot errors over spatial map
    Error and obs_mask has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange

    # Set up masking of border region

    pixel_alpha = obs_mask.clamp(0.7, 1).cpu().numpy()  # Faded border region

    fig, ax = plt.subplots(
        figsize=(5, 4.8), subplot_kw={"projection": domain_info.projection}
    )

    ax.coastlines()  # Add coastline outlines
    im = ax.imshow(
        error.cpu().numpy(),
        origin="lower",
        extent=domain_info.grid_limits,
        alpha=pixel_alpha,
        vmin=vmin,
        vmax=vmax,
        cmap="OrRd",
    )
    # Ticks and labels
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)
    return fig


class Plotter(ABC):
    """
    Abstract class to plot errors between prediction and target.
    Prediction and target had already been computed.
    """

    @abstractmethod
    def update(
        self,
        obj: "AutoRegressiveLightning",
        batch: "ItemBatch",
        prediction: NamedTensor,
        target: NamedTensor,
        mask: torch.Tensor,
    ) -> None:
        """
        Do an action when "step" is trigger
        """
        pass

    @abstractmethod
    def on_step_end(self, obj: "AutoRegressiveLightning", label: str = "") -> None:
        """
        Do an action when "end" is trigger
        """
        pass


class MapPlot(Plotter):
    """
    Abstract Observer used to plot maps during training
    """

    def __init__(
        self,
        num_samples_to_plot: int,
        num_features_to_plot: Union[None, int] = None,
        prefix: str = "Test",
        save_path: Path = None,
    ):
        self.num_samples_to_plot = num_samples_to_plot
        self.plotted_examples = 0
        self.prefix = prefix
        self.num_features_to_plot = num_features_to_plot
        self.save_path = save_path

    def update(
        self,
        obj: "AutoRegressiveLightning",
        batch: "ItemBatch",
        prediction: NamedTensor,
        target: NamedTensor,
        mask: torch.Tensor,
    ) -> None:
        """
        Update. Should be call by "on_{training/validation/test}_step
        """
        pred = deepcopy(prediction).tensor  # In order to not modify the input
        targ = deepcopy(target).tensor  # In order to not modify the input
        batch_copy = deepcopy(batch)

        # Here we reshape output from GNNS to be on the grid
        if prediction.num_spatial_dims == 1:
            pred = einops.rearrange(
                pred, "b t (x y) n -> b t x y n", x=obj.grid_shape[0]
            )
            targ = einops.rearrange(
                targ, "b t (x y) n -> b t x y n", x=obj.grid_shape[0]
            )

        if (
            obj.trainer.is_global_zero
            and self.plotted_examples < self.num_samples_to_plot
        ):
            # Need to plot more example predictions
            n_additional_examples = min(
                pred.shape[0], self.num_samples_to_plot - self.plotted_examples
            )
            # Rescale to original data scale
            std = obj.stats.to_list("std", prediction.feature_names).to(
                pred, non_blocking=True
            )
            mean = obj.stats.to_list("mean", prediction.feature_names).to(
                pred, non_blocking=True
            )
            prediction_rescaled = pred * std + mean
            target_rescaled = targ * std + mean
            batch_copy.inputs.tensor = batch_copy.inputs.tensor * std + mean
            batch_copy.forcing.tensor[:, :, :, :, :-5] = (
                batch_copy.forcing.tensor[:, :, :, :, :-5] * std + mean
            )  # not rescalled cos_hour, ect

            # Iterate over the examples
            # We assume examples are already on grid
            for pred_slice, target_slice in zip(
                prediction_rescaled[:n_additional_examples],
                target_rescaled[:n_additional_examples],
            ):
                # Each slice is (pred_steps, Nlat, Nlon, features)
                self.plotted_examples += 1  # Increment already here

                var_vmin = target_slice.flatten(0, 2).min(dim=0)[0].cpu().numpy()
                var_vmax = target_slice.flatten(0, 2).max(dim=0)[0].cpu().numpy()
                var_vranges = list(zip(var_vmin, var_vmax))

                feature_names = (
                    prediction.feature_names[: self.num_features_to_plot]
                    if self.num_features_to_plot
                    else prediction.feature_names
                )

                self.plot_map(
                    obj,
                    batch_copy,
                    pred_slice,
                    target_slice,
                    feature_names,
                    var_vranges,
                )

    @abstractmethod
    def plot_map(
        self,
        obj: "AutoRegressiveLightning",
        batch: "ItemBatch",
        prediction: torch.tensor,
        target: torch.tensor,
        feature_names: List[str],
        var_vranges: List,
    ) -> None:
        pass

    def on_step_end(self, obj: "AutoRegressiveLightning", label: str = "") -> None:
        """
        Do an action when "end" is trigger
        """
        pass


def make_gif(paths: List[Path], dest: Path):
    frames = [Image.open(path) for path in paths]
    frame_one = frames[0]
    frame_one.save(
        dest,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=250,
        loop=0,
    )


class PredictionTimestepPlot(MapPlot):
    """
    Observer used to plot prediction and target for each timestep.
    """

    def plot_map(
        self,
        obj: "AutoRegressiveLightning",
        batch: "ItemBatch",
        prediction: torch.tensor,
        target: torch.tensor,
        feature_names: List[str],
        var_vranges: List,
    ) -> None:
        # Prediction and target: (pred_steps, Nlat, Nlon, features)
        # Iterate over prediction horizon time steps
        paths_dict = defaultdict(list)
        for t_i, (pred_t, target_t) in enumerate(zip(prediction, target), start=1):
            # Create one figure per variable at this time step
            # This generate a matplotlib warning as more than 20 figures are plotted.
            units = [obj.dataset_info.units[name] for name in feature_names]
            var_figs = [
                plot_prediction(
                    pred_t[:, :, var_i],
                    target_t[:, :, var_i],
                    obj.interior_2d[:, :, 0],
                    title=f"{var_name} ({var_unit}), "
                    f"t={t_i} ({obj.dataset_info.pred_step*t_i} h)",
                    vrange=var_vrange,
                    domain_info=obj.dataset_info.domain_info,
                )
                for var_i, (var_name, var_unit, var_vrange) in enumerate(
                    zip(feature_names, units, var_vranges)
                )
            ]
            # TODO : don't create all figs at once to avoid matplotlib warning
            tensorboard = obj.logger.experiment
            for var_name, fig in zip(feature_names, var_figs):
                fig_name = f"timestep_evol_per_param/{var_name}_example_{self.plotted_examples}"
                tensorboard.add_figure(fig_name, fig, t_i)
                fig_full_name = f"{fig_name}_{t_i}.png"
                if self.save_path is not None and self.save_path.exists():
                    dest_file = self.save_path / fig_full_name
                    paths_dict[var_name].append(dest_file)
                    dest_file.parent.mkdir(exist_ok=True)
                    fig.savefig(dest_file)

                if obj.mlflow_logger:
                    run_id = obj.mlflow_logger.version
                    obj.mlflow_logger.experiment.log_figure(
                        run_id=run_id,
                        figure=fig,
                        artifact_file=f"figures/{fig_full_name}",
                    )

                plt.close(fig)

        for var_name, paths in paths_dict.items():
            if len(paths) > 1:
                make_gif(
                    paths, self.save_path / f"timestep_evol_per_param/{var_name}.gif"
                )


class PredictionEpochPlot(MapPlot):
    """
    Observer used to plot prediction and target for max timestep at each epoch.
    Used to visualize improvement of model over epochs
    """

    def plot_map(
        self,
        obj: "AutoRegressiveLightning",
        batch: "ItemBatch",
        prediction: torch.tensor,
        target: torch.tensor,
        feature_names: List[str],
        var_vranges: List,
    ) -> None:
        # We keep only the max timestep of the pred and target: shape (Nlat, Nlon, features)
        max_step = prediction.shape[0]
        pred_t, target_t = prediction[max_step - 1], target[max_step - 1]

        # Create one figure per variable at this time step
        # This generate a matplotlib warning as more than 20 figures are plotted.
        leadtime = obj.dataset_info.pred_step * max_step
        var_figs = [
            plot_prediction(
                pred_t[:, :, var_i],
                target_t[:, :, var_i],
                obj.interior_2d[:, :, 0],
                title=f"{var_name} ({var_unit}), "
                f"t={max_step} ({leadtime} h) - epoch {obj.current_epoch}",
                vrange=var_vrange,
                domain_info=obj.dataset_info.domain_info,
            )
            for var_i, (var_name, var_unit, var_vrange) in enumerate(
                zip(
                    feature_names,
                    [obj.dataset_info.units[name] for name in feature_names],
                    var_vranges,
                )
            )
        ]

        tensorboard = obj.logger.experiment
        for var_name, fig in zip(feature_names, var_figs):
            fig_name = (
                f"epoch_evol_per_param/{var_name}_example_{self.plotted_examples}"
            )
            tensorboard.add_figure(fig_name, fig, obj.current_epoch)
            fig_full_name = f"{fig_name}_{obj.current_epoch}.png"
            if self.save_path is not None:
                dest_file = self.save_path / fig_full_name
                dest_file.parent.mkdir(exist_ok=True)
                fig.savefig(dest_file)

            if obj.mlflow_logger:
                run_id = obj.mlflow_logger.version
                obj.mlflow_logger.experiment.log_figure(
                    run_id=run_id, figure=fig, artifact_file=f"figures/{fig_full_name}"
                )

        plt.close("all")  # Close all figs for this time step, saves memory


class StateErrorPlot(Plotter):
    """
    Produce a figure where the error for each variable is shown
    with respect to step
    """

    def __init__(
        self,
        metrics: Dict[str, Metric],
        prefix: str = "Test",
        save_path: Path = None,
    ):
        self.metrics = metrics
        self.prefix = prefix
        self.losses = {}
        self.shortnames = []
        self.units = []
        self.initialized = False
        self.save_path = save_path
        for metric in self.metrics:
            self.losses[metric] = []

    def update(
        self,
        obj: "AutoRegressiveLightning",
        batch: "ItemBatch",
        prediction: NamedTensor,
        target: NamedTensor,
        mask: torch.Tensor,
    ) -> None:
        """
        Compute the metric. Append to a dictionnary
        """
        for name in self.metrics:
            self.losses[name].append(
                obj.trainer.strategy.reduce(
                    self.metrics[name](prediction, target, mask), reduce_op="mean"
                ).cpu()
            )
        if not self.initialized:
            self.shortnames = prediction.feature_names
            self.units = [
                obj.dataset_info.units[name] for name in prediction.feature_names
            ]
            self.initialized = True

    def on_step_end(self, obj: "AutoRegressiveLightning", label: str = "") -> None:
        """
        Make the summary figure
        """
        tensorboard = obj.logger.experiment
        if obj.trainer.is_global_zero:
            for name in self.metrics:
                loss_tensor = torch.cat(self.losses[name], dim=0)
                loss = torch.mean(loss_tensor, dim=0)

                # Log metrics in tensorboard, with x axis as forecast timestep
                loss_dict = {self.shortnames[k]: [] for k in range(loss.shape[1])}
                for t in range(loss.shape[0]):
                    for k in range(loss.shape[1]):
                        scalar_name = f"{label}_{name}/timestep_{self.shortnames[k]}"
                        tensorboard.add_scalar(scalar_name, loss[t][k], t + 1)
                        loss_dict[self.shortnames[k]].append(loss[t][k].item())

                # Plot the score card
                if not obj.trainer.sanity_checking:
                    fig = plot_error_map(
                        loss,
                        self.shortnames,
                        self.units,
                        step_duration=obj.dataset_info.pred_step,
                    )

                    # log it in tensorboard
                    fig_name = f"score_cards/{self.prefix}_{name}"
                    tensorboard.add_figure(fig_name, fig, obj.current_epoch)
                    fig_full_name = f"{fig_name}.png"
                    if self.save_path is not None:
                        dest_file = self.save_path / fig_full_name
                        dest_file.parent.mkdir(exist_ok=True)
                        fig.savefig(dest_file)

                    if obj.mlflow_logger:
                        run_id = obj.mlflow_logger.version
                        obj.mlflow_logger.experiment.log_figure(
                            run_id=run_id,
                            figure=fig,
                            artifact_file=f"figures/{fig_full_name}",
                        )
                    plt.close(fig)

                    if self.save_path is not None:
                        # Save metrics in json file
                        with open(
                            self.save_path / f"{label}_{name}_scores.json", "w"
                        ) as json_file:
                            json.dump(loss_dict, json_file)
            # Free memory
            [self.losses[name].clear() for name in self.metrics]


class SpatialErrorPlot(Plotter):
    """
    Produce a map which shows where the error are accumulating (all variables together).
    """

    def __init__(self, prefix: str = "Test"):
        self.spatial_loss_maps = []
        self.prefix = prefix

    def update(
        self,
        obj: "AutoRegressiveLightning",
        batch: "ItemBatch",
        prediction: NamedTensor,
        target: NamedTensor,
        mask: torch.Tensor,
    ) -> None:
        spatial_loss = obj.loss(prediction, target, mask, reduce_spatial_dim=False)
        # Getting only spatial loss for the required val_step_errors
        if prediction.num_spatial_dims == 1:
            spatial_loss = einops.rearrange(
                spatial_loss, "b t (x y) -> b t x y ", x=obj.grid_shape[0]
            )
        self.spatial_loss_maps.append(
            obj.trainer.strategy.reduce(spatial_loss, reduce_op="mean").cpu()
        )  # (B, N_log, N_lat, N_lon)

    def on_step_end(self, obj: "AutoRegressiveLightning", label: str = "") -> None:
        """
        Make the summary figure
        """
        # (N_test, N_log, N_lat, N_lon)
        if obj.trainer.is_global_zero:
            spatial_loss_tensor = torch.cat(self.spatial_loss_maps, dim=0)
            mean_spatial_loss = torch.mean(
                spatial_loss_tensor, dim=0
            )  # (N_log, N_lat, N_lon)

            loss_map_figs = [
                plot_spatial_error(
                    loss_map,
                    obj.interior_2d[:, :, 0],
                    title=f"{self.prefix} loss, t={t_i} ({obj.dataset_info.pred_step*t_i} h)",
                    domain_info=obj.dataset_info.domain_info,
                )
                for t_i, loss_map in enumerate(mean_spatial_loss)
            ]
            tensorboard = obj.logger.experiment

            for t_i, fig in enumerate(loss_map_figs):
                fig_full_name = f"spatial_error_{label}/{self.prefix}_loss"
                tensorboard.add_figure(fig_full_name, fig, t_i)

                if obj.mlflow_logger:
                    run_id = obj.mlflow_logger.version
                    obj.mlflow_logger.experiment.log_figure(
                        run_id=run_id,
                        figure=fig,
                        artifact_file=f"figures/{fig_full_name}.png",
                    )
            plt.close()

        self.spatial_loss_maps.clear()
