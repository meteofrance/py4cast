"""
Test our pure PyTorch models to make sure they can be :
1. Instanciated
2. Trained
3. onnx exported
4. onnx loaded and used for inference
"""

import tempfile
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from mfai.torch import export_to_onnx, onnx_load_and_infer
from mfai.torch.models.base import ModelType
from mfai.torch.models.utils import features_last_to_second, features_second_to_last

from py4cast.models import all_nn_architectures, get_model_kls_and_settings


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


@pytest.mark.parametrize("model_name", all_nn_architectures)
def test_torch_training_loop(model_name):
    """
    Checks that our models are trainable on a toy problem (sum).
    """
    GRID_WIDTH = 64
    GRID_HEIGHT = 64
    NUM_INPUTS = 2
    NUM_OUTPUTS = 1

    model_kls, model_settings = get_model_kls_and_settings(model_name)

    # GNNs build the graph here, once at rank zero
    if hasattr(model_kls, "rank_zero_setup"):
        model_kls.rank_zero_setup(
            model_settings, FakeStatics(GRID_HEIGHT, GRID_WIDTH).meshgrid
        )

    model = model_kls(
        in_channels=NUM_INPUTS,
        out_channels=NUM_OUTPUTS,
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
        if model.model_type == ModelType.GRAPH:
            inputs = inputs.flatten(1, 2)
            targets = targets.flatten(1, 2)

        # Make predictions for this batch
        if model.features_second:
            inputs = features_last_to_second(inputs)
            outputs = model(inputs)
            outputs = features_second_to_last(outputs)
        else:
            outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, targets)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

    # Make a prediction in eval mode
    model.eval()
    sample = ds[0][0].unsqueeze(0)
    sample = sample.flatten(1, 2) if model.model_type == ModelType.GRAPH else sample

    if model.features_second:
        sample = features_last_to_second(sample)
        model(sample)
        sample = features_second_to_last(sample)
    else:
        model(sample)

    # We test if models claiming to be onnx exportable really are post training.
    # See https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
    if model.onnx_supported:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".onnx") as dst:
            sample = torch.rand((1, GRID_HEIGHT, GRID_WIDTH, NUM_INPUTS))
            if model.model_type == ModelType.GRAPH:
                sample = sample.flatten(1, 2)
            if model.features_second:
                sample = features_last_to_second(sample)
            export_to_onnx(model, sample, dst.name)
            onnx_load_and_infer(dst.name, sample)


def test_model_registry():
    """
    Imports the registry and checks that all models are available
    and also that there is no 'intruder' detected by our plugin system.
    """
    from py4cast.models import registry

    assert set(registry.keys()) == {
        "DeepLabV3",
        "DeepLabV3Plus",
        "HalfUNet",
        "Segformer",
        "SwinUNETR",
        "UNet",
        "CustomUnet",
        "UNETRPP",
        "Identity",
        "HiLAM",
        "GraphLAM",
        "HiLAMParallel",
    }
