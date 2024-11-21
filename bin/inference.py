import argparse
from pathlib import Path

from pytorch_lightning import Trainer

from py4cast.datasets.base import TorchDataloaderSettings
from py4cast.io.outputs import GribSavingSettings, save_named_tensors_to_grib
from py4cast.lightning import AutoRegressiveLightning, PlDataModule

default_config_root = Path(__file__).parents[1] / "config/IO/"

if __name__ == "__main__":
    # Parse arguments: model_path, dataset name and config file and finally date for inference
    parser = argparse.ArgumentParser("py4cast Inference script")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--date", type=str, help="Date for inference", default=None)
    parser.add_argument(
        "--dataset", type=str, help="Dataset used in inference", default="poesy_infer"
    )
    parser.add_argument(
        "--dataset_conf",
        type=str,  # Union[str, None] # Union does not work from CLI.
        default=None,
        help="Configuration file for the dataset. If None, default configuration is used.",
    )
    parser.add_argument(
        "--infer_steps", type=int, help="Number of inference steps", default=1
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="floating point precision for the inference",
        default="32",
    )
    parser.add_argument(
        "--grib",
        action="store_true",
        help="Whether the outputs should be saved as grib, needs saving conf",
    )
    parser.add_argument(
        "--saving_conf",
        type=str,
        help="name of the config file for write settings (json)",
        default="poesy_grib_settings.json",
    )

    args = parser.parse_args()

    # Load checkpoint
    if Path(args.model_path).is_dir():
        # Find the first checkpoint in the directory and load it
        for file in Path(args.model_path).iterdir():
            if file.suffix == ".ckpt":
                args.model_path = file
                break
    lightning_module = AutoRegressiveLightning.load_from_checkpoint(args.model_path)
    hparams = lightning_module.hparams["hparams"]
    lightning_module.hparams["precision"] = args.precision

    if args.date is not None:
        config_override = {
            "periods": {"test": {"start": args.date, "end": args.date, "step": 1}},
            "num_inference_pred_steps": args.infer_steps,
        }
    else:
        config_override = {"num_inference_pred_steps": args.infer_steps}

    dl_settings = TorchDataloaderSettings(batch_size=hparams.batch_size)

    dm = PlDataModule(
        dataset=args.dataset,
        num_input_steps=hparams.num_input_steps,
        num_pred_steps_train=hparams.num_pred_steps_train,
        num_pred_steps_val_test=hparams.num_pred_steps_val_test,
        dl_settings=dl_settings,
        dataset_conf=args.dataset_conf,
        config_override=config_override,
    )

    trainer = Trainer(devices="auto")
    preds = trainer.predict(lightning_module, dm)

    infer_ds = dm.infer_ds

    if args.grib:
        with open(default_config_root / args.saving_conf, "r") as f:
            save_settings = GribSavingSettings.schema().loads((f.read()))
            ph = len(save_settings.output_fmt.split("{}")) - 1
            kw = len(save_settings.output_kwargs)
            fi = len(save_settings.sample_identifiers)
            try:
                assert ph == (fi + kw)
            except AssertionError:
                raise ValueError(
                    f"Filename fmt has {ph} placeholders,\
                    but {kw} output_kwargs and {fi} sample identifiers."
                )

        for sample, pred in zip(infer_ds.sample_list, preds):
            save_named_tensors_to_grib(pred, infer_ds, sample, save_settings)
