import argparse
import json

from pytorch_lightning import Trainer

from py4cast.datasets import get_datasets
from py4cast.datasets.base import TorchDataloaderSettings
from py4cast.lightning import AutoRegressiveLightning
from py4cast.writing_outputs import saveNamedTensorToGrib

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
        "--saving_config",
        type=str,
        help="config path for file writing settings ",
        default=1,
    )

    args = parser.parse_args()

    # Load checkpoint
    lightning_module = AutoRegressiveLightning.load_from_checkpoint(args.model_path)
    hparams = lightning_module.hparams["hparams"]

    if args.date is not None:
        config_override = {
            "periods": {"test": {"start": args.date, "end": args.date}},
            "num_inference_pred_steps": args.infer_steps,
        }
    else:
        config_override = {"num_inference_pred_steps": args.infer_steps}

    # Get dataset for inference
    _, _, infer_ds = get_datasets(
        args.dataset,
        hparams.num_input_steps,
        hparams.num_pred_steps_train,
        hparams.num_pred_steps_val_test,
        hparams.dataset_conf,
        config_override=config_override,
    )

    # Transform in dataloader

    dl_settings = TorchDataloaderSettings(batch_size=1)
    infer_loader = infer_ds.torch_dataloader(dl_settings)
    trainer = Trainer(devices="auto")
    preds = trainer.predict(lightning_module, infer_loader)

    params = infer_ds.params
    leadtimes = infer_ds
    output_fmt = "grid.ai_{}_{}_forecast_{}_ech{}.grib"
    output_kwargs = [lightning_module.hparams["name"], args.dataset, args.date]
    
    # TODO : add json schema validation
    with open(args.saving_config, 'r') as f:
        save_settings = json.load(f)
    saveNamedTensorToGrib(preds, params, leadtimes, save_settings)
