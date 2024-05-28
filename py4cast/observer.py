from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Union

import einops
import matplotlib.pyplot as plt
import torch
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


class PredictionPlot(ErrorObserver):
    """
    Observer used to plot prediction and target
    """

    def __init__(
        self,
        num_samples_to_plot: int,
        num_features_to_plot: Union[None, int] = None,
        prefix: str = "Test",
    ):
        self.num_samples_to_plot = num_samples_to_plot
        self.plotted_examples = 0
        self.prefix = prefix
        self.num_features_to_plot = num_features_to_plot

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
        if obj.model.info.output_dim == 1:
            pred = einops.rearrange(
                pred, "b t (x y) n -> b t x y n", x=obj.grid_shape[0]
            )
            targ = einops.rearrange(
                targ, "b t (x y) n -> b t x y n", x=obj.grid_shape[0]
            )

        # Starting by plotting images
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

                var_vmin = (
                    torch.minimum(
                        pred_slice.flatten(0, 2).min(dim=0)[0],
                        target_slice.flatten(0, 2).min(dim=0)[0],
                    )
                    .cpu()
                    .numpy()
                )  # (d_f,)
                var_vmax = (
                    torch.maximum(
                        pred_slice.flatten(0, 2).max(dim=0)[0],
                        target_slice.flatten(0, 2).max(dim=0)[0],
                    )
                    .cpu()
                    .numpy()
                )  # (d_f,)
                var_vranges = list(zip(var_vmin, var_vmax))

                feature_names = (
                    prediction.feature_names[: self.num_features_to_plot]
                    if self.num_features_to_plot
                    else prediction.feature_names
                )

                # Iterate over prediction horizon time steps
                for t_i, (pred_t, target_t) in enumerate(
                    zip(pred_slice, target_slice), start=1
                ):
                    # Create one figure per variable at this time step
                    # This generate a matplotlib warning as more than 20 figures are plotted.
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
                    [
                        tensorboard.add_figure(
                            f"{var_name}_example_{self.plotted_examples}", fig, t_i
                        )
                        for var_name, fig in zip(
                            feature_names,
                            var_figs,
                        )
                    ]

                    plt.close("all")  # Close all figs for this time step, saves memory

    def on_step_end(self, obj: "AutoRegressiveLightning") -> None:
        """
        Do an action when "end" is trigger
        """
        pass


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
                    tensorboard.add_figure(
                        f"{self.prefix}_{name}", fig, obj.current_epoch
                    )
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
        if obj.model.info.output_dim == 1:
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
            print(mean_spatial_loss.shape)
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
            [
                tensorboard.add_figure(f"{self.prefix}_spatial_error", fig, t_i)
                for t_i, fig in enumerate(loss_map_figs)
            ]
            plt.close()

        self.spatial_loss_maps.clear()
