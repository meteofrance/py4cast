"""
Test our pure PyTorch models to make sure they can be :
1. Instanciated
2. Trained
3. onnx exported
4. onnx loaded and used for inference
"""
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pytorch_lightning as pl
import torch

from py4cast.datasets import get_datasets
from py4cast.datasets.base import TorchDataloaderSettings, collate_fn
from py4cast.lightning import ArLightningHyperParam, AutoRegressiveLightning
from py4cast.models import get_model_kls_and_settings


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


class FakeSumDataset(torch.utils.data.Dataset):
    def __init__(self, grid_height: int, grid_width: int, num_inputs: int):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_inputs = num_inputs
        super().__init__()

    def __len__(self):
        return 4

    def __getitem__(self, idx: int):
        x = torch.rand((self.grid_height, self.grid_width, self.num_inputs))
        y = torch.sum(x, -1).unsqueeze(-1)
        return x, y


@dataclass
class FakeStatics:
    """
    The GNNs need a description of the input grid in meshgrid format to
    buid the first and last layers of the graph respectively from input_grid =>
    graph_mesh and from graph_mesh => input_grid.
    """

    grid_height: int
    grid_width: int

    @property
    def meshgrid(self):
        x = np.arange(0, self.grid_width, 1)
        y = np.arange(0, self.grid_height, 1)
        xx, yy = np.meshgrid(x, y)
        return torch.from_numpy(np.asarray([xx, yy]))


def onnx_export_load_infer(model, filepath, sample):
    onnx_program = torch.onnx.dynamo_export(model, sample)
    onnx_program.save(filepath)

    # Check the model with onnx
    onnx_model = onnx.load(filepath)
    onnx.checker.check_model(onnx_model)

    # Perform an inference with onnx
    onnx_input = onnx_model.adapt_torch_inputs_to_onnx(sample)
    ort_session = onnxruntime.InferenceSession(
        filepath, providers=["CPUExecutionProvider"]
    )

    onnxruntime_input = {
        k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)
    }

    ort_session.run(None, onnxruntime_input)


def test_torch_training_loop():
    """
    Checks that our models are trainable on a toy problem (sum).
    """
    GRID_WIDTH = 64
    GRID_HEIGHT = 64
    NUM_INPUTS = 2
    NUM_OUTPUTS = 1

    for model_name in (
        "swinunetr",
        "hilam",
        "graphlam",
        "halfunet",
        "unet",
        "segformer",
        "identity",
        "unetrpp",
    ):
        model_kls, model_settings = get_model_kls_and_settings(model_name)

        # GNNs build the graph here, once at rank zero
        if hasattr(model_kls, "rank_zero_setup"):
            model_kls.rank_zero_setup(
                model_settings, FakeStatics(GRID_HEIGHT, GRID_WIDTH)
            )

        model = model_kls(
            num_input_features=NUM_INPUTS,
            num_output_features=NUM_OUTPUTS,
            settings=model_settings,
            input_shape=(GRID_HEIGHT, GRID_WIDTH),
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss_fn = torch.nn.MSELoss()
        ds = FakeSumDataset(GRID_HEIGHT, GRID_WIDTH, NUM_INPUTS)
        training_loader = torch.utils.data.DataLoader(ds, batch_size=2)

        for _, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, targets = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Flatten (H, W) -> ngrid for GNNs
            if len(model.input_dims) == 3:
                inputs = inputs.flatten(1, 2)
                targets = targets.flatten(1, 2)

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, targets)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

    # Make a prediction in eval mode
    model.eval()
    sample = ds[0][0].unsqueeze(0)
    sample = sample.flatten(1, 2) if len(model.input_dims) == 3 else sample
    model(sample)

    # We test if models claiming to be onnx exportable really are post training.
    # See https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
    if model.onnx_supported:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".onnx") as dst:
            sample = torch.rand((1, GRID_HEIGHT, GRID_WIDTH, NUM_INPUTS))
            if len(model.input_dims) == 3:
                sample = sample.flatten(1, 2)
            onnx_export_load_infer(model, dst.name, sample)


def test_lightning_fit_inference():
    """Checks that our Lightning module and training loop is working and
    that we can make a simple inference with the trained model."""
    NUM_INPUTS = 2
    NUM_OUTPUTS = 1
    DATASET = "dummy"
    MODEL = "halfunet"
    BATCH_SIZE = 2
    datasets = get_datasets(
        DATASET,
        NUM_INPUTS,
        NUM_OUTPUTS,
        NUM_OUTPUTS,
        None,
    )
    dl_settings = TorchDataloaderSettings(
        batch_size=BATCH_SIZE,
        num_workers=2,
    )
    train_ds, val_ds, test_ds = datasets
    train_loader = train_ds.torch_dataloader(dl_settings)
    val_loader = val_ds.torch_dataloader(dl_settings)
    test_loader = test_ds.torch_dataloader(dl_settings)
    trainer = pl.Trainer(
        max_epochs=3,
        limit_train_batches=3,
        limit_val_batches=3,
        limit_test_batches=3,
    )
    save_path = Path("lightning_logs/")
    hp = ArLightningHyperParam(
        dataset_info=datasets[0].dataset_info,
        dataset_name=DATASET,
        dataset_conf=None,
        batch_size=BATCH_SIZE,
        model_name=MODEL,
        model_conf=None,
        num_input_steps=NUM_INPUTS,
        num_pred_steps_train=NUM_OUTPUTS,
        num_pred_steps_val_test=NUM_OUTPUTS,
        save_path=save_path,
    )
    lightning_module = AutoRegressiveLightning(hp)
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    trainer.test(ckpt_path="best", dataloaders=test_loader)

    # Load model for simple inference
    log_dir = sorted(list(save_path.glob("version_*")))[-1]
    ckpt_path = log_dir / "checkpoints/epoch=2-step=9.ckpt"
    print(ckpt_path)
    model = AutoRegressiveLightning.load_from_checkpoint(ckpt_path)
    hparams = model.hparams["hparams"]
    hparams.num_pred_steps_val_test = NUM_OUTPUTS
    model.eval()

    item = test_ds[0]  # Load data directly from dataset (no dataloader)
    batch_item = collate_fn([item])  # Transform to BatchItem
    model(batch_item)
    print("All good!")


def test_model_registry():
    """
    Imports the registry and checks that all models are available
    and also that there is no 'intruder' detected by our plugin system.
    """
    from py4cast.models import registry

    assert set(registry.keys()) == {
        "hilam",
        "graphlam",
        "halfunet",
        "unet",
        "segformer",
        "identity",
        "hilamparallel",
        "swinunetr",
        "unetrpp",
    }
