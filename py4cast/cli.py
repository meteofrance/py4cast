from lightning.pytorch.cli import LightningCLI

from py4cast.lightning import AutoRegressiveLightning, PlDataModule
from lightning.pytorch.callbacks import BasePredictionWriter

from py4cast.io.outputs import GribSavingSettings, save_named_tensors_to_grib


class GribWriter(BasePredictionWriter):
    """
    Callback class implementing how predictions are stored after the inference.
    Args:
        active: (bool) Use this callback or not
        write_interval (str): When to write
        others : used to instantiate GribSavingSettings

    """
    def __init__(
        self,
        active,
        write_interval,
        output_dir,
        template_grib,
        output_kwargs,
        sample_identifiers,
        output_fmt,
    ):
        super().__init__(write_interval)
        self.active = active
        self.save_settings = GribSavingSettings(
            template_grib=template_grib,
            output_dir=output_dir,
            output_kwargs=output_kwargs,
            sample_identifiers=sample_identifiers,
            output_fmt=output_fmt,
        )

        ph = len(self.save_settings.output_fmt.split("{}")) - 1
        kw = len(self.save_settings.output_kwargs)
        fi = len(self.save_settings.sample_identifiers)
        try:
            assert ph == (fi + kw)
        except AssertionError:
            raise ValueError(
                f"Filename fmt has {ph} placeholders,\
                but {kw} output_kwargs and {fi} sample identifiers."
            )

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        """
        Write at the end of the inference, the predictions into gribs
        """
        if self.active:
            model_ds = trainer.datamodule.infer_ds

            for sample, pred in zip(model_ds.sample_list, predictions):
                save_named_tensors_to_grib(pred, model_ds, sample, self.save_settings)


class LCli(LightningCLI):
    """
    CLI - Command Line Interface from lightning

    Args:
        A model which inherits from LightningModule
        A datamodule which inherits from LightningDataModule
        save_config_kwargs define if checkpoint should be store even if one is already 
        present in the folder, useful for development.
    """

    def __init__(self, model_class, datamodule_class):
        super().__init__(
            model_class, datamodule_class, save_config_kwargs={"overwrite": True}
        )

    def add_arguments_to_parser(self, parser):
        
        parser.link_arguments(
            "data.dataset",
            "model.dataset_name",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.dataset_conf",
            "model.dataset_conf",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.batch_size",
            "model.batch_size",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.num_input_steps",
            "model.num_input_steps",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.num_pred_steps_train",
            "model.num_pred_steps_train",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.num_pred_steps_val_test",
            "model.num_pred_steps_val_test",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.train_dataset_info",
            "model.dataset_info",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "data.len_train_dl",
            "model.len_train_loader",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "model.save_path",
            "trainer.logger.init_args.save_dir",
            apply_on="instantiate",
        )

        parser.link_arguments(
            "model.save_path",
            "trainer.callbacks.init_args.dirpath",
            apply_on="instantiate",
        )


def cli_main():
    LCli(AutoRegressiveLightning, PlDataModule)
