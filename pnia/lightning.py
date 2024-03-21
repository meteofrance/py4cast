from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from lightning.pytorch.utilities import rank_zero_only
from torchinfo import summary

# A noter que vis depend de constant ... qui n'a donc pas les bonnes choses (car portées par le dataset).
from pnia.datasets.base import AbstractDataset
from pnia.losses import WeightedL1Loss, WeightedMSELoss
from pnia.models.conv import ConvModel, ConvSettings
from pnia.models.nlam import vis
from pnia.models.nlam.models import BaseGraphModel, GraphModelSettings
from pnia.models.nlam.utils import val_step_log_errors
from pnia.models.transformers import TransformerModel, TransformerSettings
from pnia.models.utils import expand_to_batch


@dataclass
class ArLightningHyperParam:
    """
    Settings and hyperparameters for the lightning AR model.
    """

    dataset: AbstractDataset
    batch_size: int

    model_conf: Union[Path, None] = None

    model_name: str = "graph"
    lr: float = 0.1
    loss: str = "mse"
    n_example_pred: int = 2
    step_length: float = 0.25

    # guess some of the model parameters using the dataset
    shape_from_dataset: bool = True


# each model type must provide a factory function to build the model and a settings class
# as well as a defaut configuration file.
MODELS = {
    "graph": (
        BaseGraphModel,
        GraphModelSettings,
        Path(__file__).parents[1] / "config" / "graph.json",
    ),
    "conv": (
        ConvModel,
        ConvSettings,
        Path(__file__).parents[1] / "config" / "conv.json",
    ),
    "transformer": (
        TransformerModel,
        TransformerSettings,
        Path(__file__).parents[1] / "config" / "segformer.json",
    ),
}


@rank_zero_only
def rank_zero_init(model_kls, model_settings, statics):
    if hasattr(model_kls, "rank_zero_setup"):
        model_kls.rank_zero_setup(model_settings, statics)


class AutoRegressiveLightning(pl.LightningModule):
    """
    Auto-regressive lightning module for predicting meteorological fields.
    """

    def __init__(self, hparams: ArLightningHyperParam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(asdict(hparams))

        # Load static features for grid/data
        statics = self.hparams.dataset.statics

        self.N_interior = statics.N_interior

        self.val_maes = []
        self.test_maes = []
        self.test_mses = []

        # For making restoring of optimizer state optional (slight hack)
        self.opt_state = None

        # For example plotting
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []

        # instantiate the model with dataclass json schema validation for settings
        model_kls, settings_class, default_settings = MODELS[hparams.model_name]

        model_conf = hparams.model_conf if hparams.model_conf else default_settings

        with open(model_conf, "r") as f:
            model_settings = settings_class.schema().loads(f.read())

        if hparams.shape_from_dataset:
            # Set model input/output grid features based on dataset tensor shapes
            grid_static_dim = statics.grid_static_features.values.shape[-1]

            input_features = (
                # we inject the previous and the penultimate states hence the 2
                2 * self.hparams.dataset.weather_dim
                + grid_static_dim
                + self.hparams.dataset.forcing_dim
            )
            model_settings.input_features = input_features
            model_settings.output_features = self.hparams.dataset.weather_dim

            # todo fix this it should decoupled
            if hasattr(model_settings, "graph_dir"):
                model_settings.graph_dir = self.hparams.dataset.cache_dir
                print(f"Model settings: {model_settings.graph_dir}")

        print(f"Distributed ranks: {self.local_rank} {self.global_rank}")

        rank_zero_init(model_kls, model_settings, statics)

        self.model = model_kls.build_from_settings(model_settings, statics)
        summary(self.model)

        # We transform and register the statics after the model has been set up

        statics = self.model.transform_statics(statics)
        statics.register_buffers(self)
        # On enregistre manuellement les valueurs pour grid_static_features.
        # On l'etend a la taille attendu du batch.
        # De cette manière on évite les expands dans chaque module et l'operation est faite une unique fois.
        self.register_buffer(
            "grid_static_features",
            expand_to_batch(
                statics.grid_static_features.values, self.hparams.batch_size
            ),
            persistent=False,
        )
        # We need to instantiate the loss after statics had been transformed.
        # Indeed, the statics used should be in the right dimensions.
        # MSE loss, need to do reduction ourselves to get proper weighting
        if hparams.loss == "mse":
            self.loss = WeightedMSELoss(reduction="none")
        elif hparams.loss == "mae":
            self.loss = WeightedL1Loss(reduction="none")
        else:
            assert False, f"Unknown loss function: {hparams.loss}"

        self.loss.prepare(statics)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.95)
        )
        if self.opt_state:
            opt.load_state_dict(self.opt_state)

        return opt

    def unroll_prediction(self, init_states, forcing_features, true_states):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, N_grid, d_f)
        forcing_features: (B, pred_steps, N_grid, d_static_f)
        true_states: (B, pred_steps, N_grid, d_f)
        """
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_steps = forcing_features.shape[1]

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            border_state = true_states[:, i]
            #
            # TODO c'est ici qu'il faudra charger le forceur sur les côtés.
            #
            x = torch.cat(
                [
                    prev_state,
                    prev_prev_state,
                    self.grid_static_features,
                    forcing,
                ],
                dim=-1,
            )

            # Graph (B, N_grid, d_f) or Conv (B, N_lat,N_lon d_f)
            y = self.model(x)

            predicted_state = prev_state + y * self.step_diff_std + self.step_diff_mean

            # Overwrite border with true state
            new_state = (
                self.border_mask * border_state + self.interior_mask * predicted_state
            )
            prediction_list.append(new_state)

            # Upate conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        return torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, N_grid, d_f) or (B, pred_steps, N_lat, N_lon, d_f)

    def common_step(self, batch):
        """
        Predict on single batch

        init_states: (B, 2, N_grid, d_features) or (B, 2, Nlat,Nlon, d_features) depend on the grid
        target_states: (B, pred_steps, N_grid, d_features) or (B, pred_steps, Nlat,Nlon,, dfeatures)
        forcing_features: (B, pred_steps, N_grid, d_forcing), where index 0
            corresponds to index 1 of init_states -> Ngrid could also be (Nlat,Nlon)
        """
        (
            init_states,
            target_states,
            forcing_features,
        ) = self.model.transform_batch(batch)

        prediction = self.unroll_prediction(
            init_states, forcing_features, target_states
        )  # (B, pred_steps, N_grid, d_f) or (B, pred_steps, Nlat, Nlon, df)

        return prediction, target_states

    def training_step(self, batch):
        """
        Train on single batch
        """
        prediction, target = self.common_step(batch)

        # Compute loss
        batch_loss = torch.mean(
            self.loss(prediction, target)
        )  # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss}
        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return batch_loss

    def per_var_error(self, prediction, target, error="mae"):
        """
        Computed MAE/MSE per variable and time step
        prediction/target: (B, pred_steps, N_grid, d_f)
        returns (B, pred_steps)
        """
        if error == "mse":
            loss_func = torch.nn.functional.mse_loss
        else:
            loss_func = torch.nn.functional.l1_loss

        entry_loss = loss_func(
            prediction, target, reduction="none"
        )  # (B, pred_steps, N_grid, d_f) or (B, pred_steps, Nlat, Nlon, df)

        # Dimension on which to aggregate (to get rid of the grid)
        # Aggregate_dims is negative (the last dimensions correspond at the grid in the loss)
        dims = [len(prediction.shape) - 1 + x for x in self.loss.aggregate_dims]

        mean_error = (
            torch.sum(entry_loss * self.interior_mask, dim=dims) / self.N_interior
        )  # (B, pred_steps, d_f)

        return mean_error

    def all_gather_cat(self, tensor_to_gather):
        """
        Gather tensors across all ranks, and concatenate across dim. 0 (instead of
        stacking in new dim. 0)

        tensor_to_gather: (d1, d2, ...), distributed over K ranks

        returns: (K*d1, d2, ...)
        """
        # print("In all_gather_cat",tensor_to_gather.shape)
        return self.all_gather(tensor_to_gather).flatten(0, 1)

    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target = self.common_step(batch)

        time_step_loss = torch.mean(
            self.loss(prediction, target), dim=0
        )  # (time_steps-1)

        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in val_step_log_errors
        }
        val_log_dict["val_mean_loss"] = mean_loss

        maes = self.per_var_error(prediction, target)  # (B, pred_steps, d_f)
        self.val_maes.append(maes)

        self.log_dict(val_log_dict, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """

        val_mae_tensor = self.all_gather_cat(
            torch.cat(self.val_maes, dim=0)
        )  # (N_val, pred_steps, d_f)

        if self.trainer.is_global_zero:
            # pass
            val_mae_total = torch.mean(val_mae_tensor, dim=0)  # (pred_steps, d_f)
            val_mae_rescaled = val_mae_total * self.data_std  # (pred_steps, d_f)

            if not self.trainer.sanity_checking:
                # Don't log this during sanity checking
                mae_fig = vis.plot_error_map(
                    val_mae_rescaled,
                    self.hparams.dataset,
                    title="Validation MAE",
                    step_length=self.hparams.step_length,
                )
                tensorboard = self.logger.experiment
                tensorboard.add_figure("val_mae", mae_fig, self.current_epoch)
                #    wandb.log({"val_mae": wandb.Image(mae_fig)})
                plt.close("all")  # Close all figs

        self.val_maes.clear()  # Free memory

    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        prediction, target = self.common_step(batch)

        time_step_loss = torch.mean(
            self.loss(prediction, target), dim=0
        )  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in val_step_log_errors
        }
        test_log_dict["test_mean_loss"] = mean_loss

        self.log_dict(test_log_dict, on_step=False, on_epoch=True, sync_dist=True)

        # For error maps
        maes = self.per_var_error(prediction, target)  # (B, pred_steps, d_f)
        self.test_maes.append(maes)
        mses = self.per_var_error(
            prediction, target, error="mse"
        )  # (B, pred_steps, d_f)
        self.test_mses.append(mses)

        # Save per-sample spatial loss for specific times
        spatial_loss = self.loss(
            prediction, target, reduce_spatial_dim=False
        )  # (B, pred_steps, N_grid)
        log_spatial_losses = spatial_loss[:, val_step_log_errors - 1]

        self.spatial_loss_maps.append(log_spatial_losses)  # (B, N_log, N_grid)

        # Plot example predictions (on rank 0 only)
        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.hparams.n_example_pred
        ):
            self.plot_pred(prediction, target)

    def plot_pred(self, prediction, target):
        # Need to plot more example predictions
        n_additional_examples = min(
            prediction.shape[0],
            self.hparams.n_example_pred - self.plotted_examples,
        )
        # Rescale to original data scale
        prediction_rescaled = prediction * self.data_std + self.data_mean
        target_rescaled = target * self.data_std + self.data_mean

        # Iterate over the examples
        for pred_slice, target_slice in zip(
            prediction_rescaled[:n_additional_examples],
            target_rescaled[:n_additional_examples],
        ):
            # Each slice is (pred_steps, N_grid, d_f)
            self.plotted_examples += 1  # Increment already here

            var_vmin = (
                torch.minimum(
                    pred_slice.flatten(0, 1).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vmax = (
                torch.maximum(
                    pred_slice.flatten(0, 1).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vranges = list(zip(var_vmin, var_vmax))

            # Iterate over prediction horizon time steps
            for t_i, (pred_t, target_t) in enumerate(
                zip(pred_slice, target_slice), start=1
            ):
                # Create one figure per variable at this time step
                var_figs = [
                    vis.plot_prediction(
                        pred_t[:, var_i],
                        target_t[:, var_i],
                        self.interior_mask[:, 0],
                        grid_shape=self.hparams.dataset.grid_shape,
                        title=f"{var_name} ({var_unit}), "
                        f"t={t_i} ({self.hparams.step_length*t_i} h)",
                        vrange=var_vrange,
                        projection=self.hparams.dataset.projection,
                        grid_limits=self.hparams.dataset.grid_limits,
                    )
                    for var_i, (var_name, var_unit, var_vrange) in enumerate(
                        zip(
                            self.hparams.dataset.shortnames(kind="output"),
                            self.hparams.dataset.units(kind="output"),
                            var_vranges,
                        )
                    )
                ]
                tensorboard = self.logger.experiment
                [
                    tensorboard.add_figure(
                        f"{var_name}_example_{self.plotted_examples}", fig, t_i
                    )
                    for var_name, fig in zip(
                        self.hparams.dataset.shortnames(kind="output"), var_figs
                    )
                ]

                plt.close("all")  # Close all figs for this time step, saves memory

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        Will gather stored tensors and perform plotting and logging on rank 0.
        """
        # Create error maps for RMSE and MAE
        print("Mae cat", torch.cat(self.test_maes, dim=0).shape)

        test_mae_tensor = self.all_gather(torch.cat(self.test_maes, dim=0))
        test_mse_tensor = self.all_gather(torch.cat(self.test_mses, dim=0))

        print("test_mae_tensor ", test_mae_tensor.shape, "should be (Ntest, preds, df)")

        if self.trainer.is_global_zero:
            test_mae_rescaled = (
                torch.mean(test_mae_tensor, dim=0) * self.data_std
            )  # (pred_steps, d_f)
            # print(test_mae_rescaled.shape)
            test_rmse_rescaled = (
                torch.sqrt(torch.mean(test_mse_tensor, dim=0)) * self.data_std
            )  # (pred_steps, d_f)
            # print(test_rmse_rescaled.shape)
            mae_fig = vis.plot_error_map(
                test_mae_rescaled,
                self.hparams.dataset,
                step_length=self.hparams.step_length,
            )
            rmse_fig = vis.plot_error_map(
                test_rmse_rescaled,
                self.hparams.dataset,
                step_length=self.hparams.step_length,
            )

            # Save pdf:s
            tensorboard = self.logger.experiment
            tensorboard.add_figure("test_mae", mae_fig)
            tensorboard.add_figure("test_rmse", rmse_fig)

        self.test_maes.clear()  # Free memory
        self.test_mses.clear()

        # Plot spatial loss maps
        spatial_loss_tensor = self.all_gather(
            torch.cat(self.spatial_loss_maps, dim=0)
        )  # (N_test, N_log, N_grid)
        if self.trainer.is_global_zero:
            mean_spatial_loss = torch.mean(
                spatial_loss_tensor, dim=0
            )  # (N_log, N_grid)

            loss_map_figs = [
                vis.plot_spatial_error(
                    loss_map,
                    self.interior_mask[:, 0],
                    grid_shape=self.hparams.dataset.grid_shape,
                    title=f"Test loss, t={t_i} ({self.hparams.step_length*t_i} h)",
                    projection=self.hparams.dataset.projection,
                    grid_limits=self.hparams.dataset.grid_limits,
                )
                for t_i, loss_map in zip(val_step_log_errors, mean_spatial_loss)
            ]
            tensorboard = self.logger.experiment
            [
                tensorboard.add_figure("spatial_error", fig, t_i)
                for t_i, fig in zip(val_step_log_errors, loss_map_figs)
            ]

        self.spatial_loss_maps.clear()
