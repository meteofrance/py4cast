from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Union

import einops
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pytorch_lightning.strategies import (  # Use for homogeneity in gathering operation for plot.
    ParallelStrategy,
    SingleDeviceStrategy,
)

if TYPE_CHECKING:
    from py4cast.lightning import AutoRegressiveLightning

from py4cast.datasets.base import NamedTensor
from py4cast.losses import Py4CastLoss
from py4cast.plots import plot_error_map, plot_prediction, plot_spatial_error


def gather(
    obj: "AutoRegressiveLightning", tensor_to_gather: torch.Tensor
) -> torch.Tensor:
    """
    Send a tensor with the same dimension whether we are in Paralll or SingleDevice strategy.
    Be careful if you are doing something else than plotting results.
    """
    if isinstance(obj.trainer.strategy, SingleDeviceStrategy):
        loss_tensor = obj.trainer.strategy.all_gather(tensor_to_gather)
    elif isinstance(obj.trainer.strategy, ParallelStrategy):
        loss_tensor = obj.trainer.strategy.all_gather(tensor_to_gather).flatten(0, 1)
    else:
        raise TypeError(
            f"Unknwon type {obj.trainer.strategy}. Know {SingleDeviceStrategy} and {ParallelStrategy}"
        )

    return loss_tensor


class ErrorObserver(ABC):
    """
    Abstract class to observe Errors between prediction and target.
    Prediction and target had already been computed.
    """

    @abstractmethod
    def update(
        self,
        obj: "AutoRegressiveLightning",
        prediction: NamedTensor,
        target: NamedTensor,
    ) -> None:
        """
        Do an action when "step" is trigger
        """
        pass

    @abstractmethod
    def on_step_end(self, obj: "AutoRegressiveLightning") -> None:
        """
        Do an action when "end" is trigger
        """
        pass


class MapPlot(ErrorObserver):
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
        prediction: NamedTensor,
        target: NamedTensor,
    ) -> None:
        """
        Update. Should be call by "on_{training/validation/test}_step
        """
        pred = deepcopy(prediction).tensor  # In order to not modify the input
        targ = deepcopy(target).tensor  # In order to not modify the input

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
                    pred_slice,
                    target_slice,
                    feature_names,
                    var_vranges,
                )

    @abstractmethod
    def plot_map(
        self,
        obj: "AutoRegressiveLightning",
        prediction: torch.tensor,
        target: torch.tensor,
        feature_names: List[str],
        var_vranges: List,
    ) -> None:
        pass

    def on_step_end(self, obj: "AutoRegressiveLightning") -> None:
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
            units = [
                obj.hparams["hparams"].dataset_info.units[name]
                for name in feature_names
            ]
            var_figs = [
                plot_prediction(
                    pred_t[:, :, var_i],
                    target_t[:, :, var_i],
                    obj.interior_2d[:, :, 0],
                    title=f"{var_name} ({var_unit}), "
                    f"t={t_i} ({obj.hparams['hparams'].dataset_info.step_duration*t_i} h)",
                    vrange=var_vrange,
                    domain_info=obj.hparams["hparams"].dataset_info.domain_info,
                )
                for var_i, (var_name, var_unit, var_vrange) in enumerate(
                    zip(feature_names, units, var_vranges)
                )
            ]
            tensorboard = obj.logger.experiment
            for var_name, fig in zip(feature_names, var_figs):
                fig_name = f"timestep_evol_per_param/{var_name}_example_{self.plotted_examples}"
                tensorboard.add_figure(fig_name, fig, t_i)
                if self.save_path is not None and self.save_path.exists():
                    dest_file = self.save_path / f"{fig_name}_{t_i}.png"
                    paths_dict[var_name].append(dest_file)
                    dest_file.parent.mkdir(exist_ok=True)
                    fig.savefig(dest_file)

            plt.close("all")  # Close all figs for this time step, saves memory

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
        leadtime = obj.hparams["hparams"].dataset_info.step_duration * max_step
        var_figs = [
            plot_prediction(
                pred_t[:, :, var_i],
                target_t[:, :, var_i],
                obj.interior_2d[:, :, 0],
                title=f"{var_name} ({var_unit}), "
                f"t={max_step} ({leadtime} h) - epoch {obj.current_epoch}",
                vrange=var_vrange,
                domain_info=obj.hparams["hparams"].dataset_info.domain_info,
            )
            for var_i, (var_name, var_unit, var_vrange) in enumerate(
                zip(
                    feature_names,
                    [
                        obj.hparams["hparams"].dataset_info.units[name]
                        for name in feature_names
                    ],
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
            if self.save_path is not None:
                dest_file = self.save_path / f"{fig_name}_{obj.current_epoch}.png"
                dest_file.parent.mkdir(exist_ok=True)
                fig.savefig(dest_file)

        plt.close("all")  # Close all figs for this time step, saves memory


class StateErrorPlot(ErrorObserver):
    """
    Produce a figure where the error for each variable is shown
    with respect to step
    """

    def __init__(self, metrics: Dict[str, Py4CastLoss], prefix: str = "Test"):
        self.metrics = metrics
        self.prefix = prefix
        self.losses = {}
        self.shortnames = []
        self.units = []
        self.initialized = False
        for metric in self.metrics:
            self.losses[metric] = []

    def update(
        self,
        obj: "AutoRegressiveLightning",
        prediction: NamedTensor,
        target: NamedTensor,
    ) -> None:
        """
        Compute the metric. Append to a dictionnary
        """
        for name in self.metrics:
            self.losses[name].append(self.metrics[name](prediction, target))
        if not self.initialized:
            self.shortnames = prediction.feature_names
            self.units = [
                obj.hparams["hparams"].dataset_info.units[name]
                for name in prediction.feature_names
            ]
            self.initialized = True

    def on_step_end(self, obj: "AutoRegressiveLightning") -> None:
        """
        Make the summary figure
        """
        for name in self.metrics:
            loss_tensor = gather(obj, torch.cat(self.losses[name], dim=0))
            if obj.trainer.is_global_zero:
                loss = torch.mean(loss_tensor, dim=0)
                if not obj.trainer.sanity_checking:
                    fig = plot_error_map(
                        loss,
                        self.shortnames,
                        self.units,
                        step_duration=obj.hparams["hparams"].dataset_info.step_duration,
                    )

                    tensorboard = obj.logger.experiment
                    fig_name = f"score_cards/{self.prefix}_{name}"
                    tensorboard.add_figure(fig_name, fig, obj.current_epoch)
        # Free memory
        [self.losses[name].clear() for name in self.metrics]


class SpatialErrorPlot(ErrorObserver):
    """
    Produce a map which shows where the error are accumulating (all variables together).
    """

    def __init__(self, prefix: str = "Test"):
        self.spatial_loss_maps = []
        self.prefix = prefix

    def update(
        self,
        obj: "AutoRegressiveLightning",
        prediction: NamedTensor,
        target: NamedTensor,
    ) -> None:
        spatial_loss = obj.loss(prediction, target, reduce_spatial_dim=False)
        # Getting only spatial loss for the required val_step_errors
        if prediction.num_spatial_dims == 1:
            spatial_loss = einops.rearrange(
                spatial_loss, "b t (x y) -> b t x y ", x=obj.grid_shape[0]
            )
        self.spatial_loss_maps.append(spatial_loss)  # (B, N_log, N_lat, N_lon)

    def on_step_end(self, obj: "AutoRegressiveLightning") -> None:
        """
        Make the summary figure
        """
        spatial_loss_tensor = gather(
            obj, torch.cat(self.spatial_loss_maps, dim=0)
        )  # (N_test, N_log, N_lat, N_lon)

        if obj.trainer.is_global_zero:
            mean_spatial_loss = torch.mean(
                spatial_loss_tensor, dim=0
            )  # (N_log, N_lat, N_lon)
            loss_map_figs = [
                plot_spatial_error(
                    loss_map,
                    obj.interior_2d[:, :, 0],
                    title=f"{self.prefix} loss, t={t_i} ({obj.hparams['hparams'].dataset_info.step_duration*t_i} h)",
                    domain_info=obj.hparams["hparams"].dataset_info.domain_info,
                )
                for t_i, loss_map in enumerate(mean_spatial_loss)
            ]
            tensorboard = obj.logger.experiment
            for t_i, fig in enumerate(loss_map_figs):
                tensorboard.add_figure(f"spatial_error/{self.prefix}_loss", fig, t_i)
            plt.close()

        self.spatial_loss_maps.clear()
