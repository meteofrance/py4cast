import argparse

from pytorch_lightning import Trainer

from py4cast.datasets import get_datasets
from py4cast.datasets.base import TorchDataloaderSettings
from py4cast.lightning import AutoRegressiveLightning

if __name__ == "__main__":

    # Parse arguments: model_path, dataset name and config file and finally date for inference
    parser = argparse.ArgumentParser("py4cast Inference script")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--date", type=str, help="Date for inference", default=None)
    parser.add_argument("--dataset", type=str, help="Dataset used in inference", default="poesy_infer")

    args = parser.parse_args()

    # Load checkpoint
    lightning_module = AutoRegressiveLightning.load_from_checkpoint(args.model_path)
    hparams = lightning_module.hparams["hparams"]

    if args.date is not None:
        config_override = {"periods": {"test": {"start": args.date, "end": args.date}}}
    else:
        config_override = None

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

