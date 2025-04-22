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

from py4cast.models import (
    all_nn_architectures,
    get_model_kls_and_settings,
    nn_architectures,
)


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


class FakePanguDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        surface_variables: int,
        plevel_variables: int,
        plevels: int,
        static_length: int,
    ):
        self.surface_shape = (surface_variables, *input_shape)
        self.plevel_shape = (plevel_variables, plevels, *input_shape)
        self.static_shape = (static_length, *input_shape)
        super().__init__()

    def __len__(self):
        return 4

    def __getitem__(self, idx: int):
        input_surface = torch.rand(*self.surface_shape)
        input_plevel = torch.rand(*self.plevel_shape)
        input_static = torch.rand(*self.static_shape)
        target_surface = torch.rand(*self.surface_shape)
        target_plevel = torch.rand(*self.plevel_shape)
        return {
            "input_surface": input_surface,
            "input_plevel": input_plevel,
            "input_static": input_static,
            "target_surface": target_surface,
            "target_plevel": target_plevel,
        }


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


@pytest.mark.parametrize(
    "model_kls",
    nn_architectures[ModelType.GRAPH]
    + nn_architectures[ModelType.CONVOLUTIONAL]
    + nn_architectures[ModelType.VISION_TRANSFORMER],
)
def test_torch_training_loop(model_kls):
    """
    Checks that our models are trainable on a toy problem (sum).
    """
    GRID_WIDTH = 64
    GRID_HEIGHT = 64
    NUM_INPUTS = 2
    NUM_OUTPUTS = 1

    model_settings = model_kls.settings_kls()

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


@pytest.mark.parametrize("model_kls", nn_architectures[ModelType.PANGU])
def test_torch_pangu_training_loop(model_kls):
    """
    Checks that our models are trainable on a toy problem (sum).
    """
    INPUT_SHAPE = (64, 64, 64)
    NUM_INPUTS = 7
    NUM_OUTPUTS = 6
    SURFACE_VARIABLES = 2
    PLEVEL_VARIABLES = 2
    PLEVELS = 2
    STATIC_LENGTH = 1

    settings = model_kls.settings_kls(
        surface_variables=SURFACE_VARIABLES,
        plevel_variables=PLEVEL_VARIABLES,
        plevels=PLEVELS,
        static_length=STATIC_LENGTH,
    )

    # We test the model for all supported input spatial dimensions
    for spatial_dims in model_kls.supported_num_spatial_dims:
        if hasattr(settings, "spatial_dims"):
            settings.spatial_dims = spatial_dims

        model = model_kls(
            in_channels=NUM_INPUTS,
            out_channels=NUM_OUTPUTS,
            input_shape=INPUT_SHAPE[:spatial_dims],
            settings=settings,
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss_fn = torch.nn.MSELoss()

        ds = FakePanguDataset(
            input_shape=INPUT_SHAPE[:spatial_dims],
            surface_variables=SURFACE_VARIABLES,
            plevel_variables=PLEVEL_VARIABLES,
            plevels=PLEVELS,
            static_length=STATIC_LENGTH,
        )

        training_loader = torch.utils.data.DataLoader(ds, batch_size=2)

        # Simulate 2 EPOCHS of training
        for _ in range(2):
            for data in training_loader:
                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                output_plevel, output_surface = model(
                    data["input_plevel"], data["input_surface"], data["input_static"]
                )

                # Compute the loss and its gradients
                loss = loss_fn(output_plevel, data["target_plevel"]) + loss_fn(
                    output_surface, data["target_surface"]
                )
                loss.backward()

                # Adjust learning weights
                optimizer.step()

        # Make a prediction in eval mode
        model.eval()
        sample = ds[0]
        model(
            sample["input_plevel"].unsqueeze(0),
            sample["input_surface"].unsqueeze(0),
            sample["input_static"].unsqueeze(0),
        )

        # We test if models claiming to be onnx exportable really are post training.
        # See https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
        if model.onnx_supported:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".onnx") as dst:
                sample_surface = torch.rand(
                    1, SURFACE_VARIABLES, *INPUT_SHAPE[:spatial_dims]
                )
                sample_plevel = torch.rand(
                    1, PLEVEL_VARIABLES, PLEVELS, *INPUT_SHAPE[:spatial_dims]
                )
                sample_static = torch.rand(
                    1, STATIC_LENGTH, *INPUT_SHAPE[:spatial_dims]
                )
                sample = (sample_plevel, sample_surface, sample_static)
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
        "PanguWeather",
    }
