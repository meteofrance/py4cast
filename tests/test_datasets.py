"""
Unit tests for datasets and NamedTensor.
"""

import datetime

import numpy as np
import pytest
import torch

from py4cast.datasets.base import Item, NamedTensor, collate_fn
from py4cast.forcingutils import generate_toa_radiation_forcing, get_year_hour_forcing


def test_item():
    """
    Creates an item with NamedTensors and test it
    """
    tensor = torch.rand(256, 256, 5)
    inputs = NamedTensor(
        tensor,
        names=["lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(5)],
    )
    outputs = NamedTensor(
        torch.rand(256, 256, 5),
        names=["lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(5)],
    )

    forcing = NamedTensor(
        torch.rand(256, 256, 2),
        names=["lat", "lon", "features"],
        feature_names=[f"forcing_{i}" for i in range(2)],
    )

    validity_times = [datetime.datetime(year=2023, month=1, day=1, hour=18)]
    item = Item(inputs=inputs, outputs=outputs, forcing=forcing, validity_times=validity_times)
    print(item)

    # test collate_fn
    items = [item, item, item]
    batch = collate_fn(items)
    print(batch)

    # Assert a New Batch dimension has been added with size 3
    assert batch.inputs.tensor.shape == (3, 256, 256, 5)
    assert batch.outputs.tensor.shape == (3, 256, 256, 5)
    assert batch.forcing.tensor.shape == (3, 256, 256, 2)

    # Input and Output must have the same number of features
    with pytest.raises(ValueError):
        inputs = NamedTensor(
            tensor,
            names=["lat", "lon", "features"],
            feature_names=[f"feature_{i}" for i in range(5)],
        )
        outputs = NamedTensor(
            torch.rand(256, 256, 5),
            names=["lat", "lon", "features"],
            feature_names=[f"feature_{i}" for i in range(4)],
        )

        item = Item(inputs=inputs, outputs=outputs, forcing=forcing)

    # Input and Output must have the same feature names
    with pytest.raises(ValueError):
        inputs = NamedTensor(
            tensor,
            names=["lat", "lon", "features"],
            feature_names=[f"feature_{i}" for i in range(5)],
        )
        outputs = NamedTensor(
            torch.rand(256, 256, 5),
            names=["lat", "lon", "features"],
            feature_names=[f"f_{i}" for i in range(5)],
        )

        item = Item(inputs=inputs, outputs=outputs, forcing=forcing)

    # Input and Output must have the same dim names
    with pytest.raises(ValueError):
        inputs = NamedTensor(
            tensor,
            names=["lat", "lon", "features"],
            feature_names=[f"feature_{i}" for i in range(5)],
        )
        outputs = NamedTensor(
            torch.rand(256, 256, 5),
            names=["lat", "lon", "wrong_name"],
            feature_names=[f"feature_{i}" for i in range(5)],
        )

        item = Item(inputs=inputs, outputs=outputs, forcing=forcing)


def test_date_forcing():
    """
    Testing the date forcing.
    """
    date = datetime.datetime(year=2023, month=12, day=31, hour=23)
    relativ_terms = [
        datetime.timedelta(hours=1),
        datetime.timedelta(hours=2),
        datetime.timedelta(hours=3),
    ]

    nb_terms = len(relativ_terms)
    nb_forcing = 4

    forcing = get_year_hour_forcing(date, relativ_terms)

    # Check the dimension of the output
    assert forcing.shape == torch.Size((nb_terms, nb_forcing))
    # Check the result of the specific value : midnight of the new year 2024
    assert torch.allclose(forcing[0], torch.tensor([0.5, 1, 0.5, 1]))


def test_solar_forcing():
    """
    Testing the solar forcing.
    """
    # Testing the specific value of the exercice 1.6.2.a of Solar Engineering of Thermal Processes,
    # Photovoltaics and Wind 5th ed.

    # Input data of the exercice
    lat = torch.tensor(43)
    lon = torch.tensor(-89)
    relativ_terms = [datetime.timedelta(hours=0)]
    # solar_date = datetime.datetime(year=2023, month=2, day=13, hour=9, minute = 30), add 5h56 to convert into utc hour
    utc_date = datetime.datetime(year=2023, month=2, day=13, hour=15, minute=26)
    E0 = 1366

    # Tolerance to imprecision
    error_margin = 0.01

    # Solution
    cos_solution = np.cos(np.radians(66.5))
    solution = E0 * cos_solution

    forcing = generate_toa_radiation_forcing(lat, lon, utc_date, relativ_terms)

    # Testing if result is far from solution
    assert np.abs(forcing - solution) < error_margin

    # Testing the output shape
    lat = torch.rand(16, 16)
    lon = torch.rand(16, 16)
    utc_date = datetime.datetime(year=2023, month=5, day=20, hour=13)
    relativ_terms = [
        datetime.timedelta(hours=1),
        datetime.timedelta(hours=2),
        datetime.timedelta(hours=3),
    ]

    forcing = generate_toa_radiation_forcing(lat, lon, utc_date, relativ_terms)

    # Test the shape
    assert forcing.shape == torch.Size((3, 16, 16, 1))
