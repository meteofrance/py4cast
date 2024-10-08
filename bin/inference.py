import argparse
from pathlib import Path

from pytorch_lightning import Trainer

from py4cast.datasets.base import TorchDataloaderSettings
from py4cast.lightning import AutoRegressiveLightning, PlDataModule


if __name__ == "__main__":

    # Parse arguments: model_path, dataset name and config file and finally date for inference
    parser = argparse.ArgumentParser("py4cast Inference script")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--date", type=str, help="Date for inference", default=None)
    parser.add_argument(
        "--dataset", type=str, help="Dataset used in inference", default="poesy_infer"
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
            "periods": {"test": {"start": args.date, "end": args.date}},
            "num_inference_pred_steps": args.infer_steps,
        }
    else:
        config_override = {"num_inference_pred_steps": args.infer_steps}

    dl_settings = TorchDataloaderSettings(batch_size=hparams.batch_size)
    dm = PlDataModule(
        dataset = args.dataset,
        num_input_steps = hparams.num_input_steps,
        num_pred_steps_train= hparams.num_pred_steps_train,
        num_pred_steps_val_test= hparams.num_pred_steps_val_test,
        dl_settings = dl_settings,
        dataset_conf = hparams.dataset_conf,
        config_override = config_override
    )
    trainer = Trainer(devices="auto")
    preds = trainer.predict(lightning_module, dm)
    print(preds.shape)
