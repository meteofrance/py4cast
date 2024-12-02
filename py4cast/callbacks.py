from pathlib import Path

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
        active: bool,
        write_interval: str,
        output_dir: str,
        template_grib: str,
        output_kwargs: list,
        sample_identifiers: list,
        output_fmt: str,
    ):
        super().__init__(write_interval)
        self.active = active
        output_dir = Path(output_dir)
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
        Write at the end, of the inference, the predictions into gribs
        """
        if self.active:
            model_ds = trainer.datamodule.infer_ds

            for sample, pred in zip(model_ds.sample_list, predictions):
                save_named_tensors_to_grib(pred, model_ds, sample, self.save_settings)
