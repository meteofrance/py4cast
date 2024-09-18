"""
Unit tests for datasets and NamedTensor.
"""
import pytest
import torch

import datetime 
from py4cast.datasets.base import Item, NamedTensor, collate_fn, DatasetABC

import json
from py4cast.datasets.poesy import Grid as PoesyGrib

def test_named_tensor():
    """
    Test NamedTensor class.
    """
    # Create a tensor
    tensor = torch.rand(3, 256, 256, 50)
    # Create a NamedTensor
    nt = NamedTensor(
        tensor,
        names=["batch", "lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(50)],
    )
    # Test
    assert nt.names == ["batch", "lat", "lon", "features"]
    assert nt.tensor.shape == (3, 256, 256, 50)

    # Test dim sizes
    assert nt.dim_size("batch") == 3
    assert nt.dim_size("lat") == 256
    assert nt.dim_size("lon") == 256
    assert nt.dim_size("features") == 50

    nt2 = nt.clone()

    # Concat should raise because of feature names collision
    with pytest.raises(ValueError):
        nt | nt2

    nt3 = NamedTensor(
        torch.rand(3, 256, 256, 50),
        names=["batch", "lat", "lon", "features"],
        feature_names=[f"levels_{i}" for i in range(50)],
    )

    # Test | operator (concatenation)
    # the number of features must match the sum of the feature names of the two tensors
    nt_cat = nt | nt3
    assert nt_cat.tensor.shape == (3, 256, 256, 100)

    # Test | operator (concatenation)
    # last dim name does not mach => ValueError
    nt4 = NamedTensor(
        torch.rand(3, 256, 256, 50),
        names=["batch", "lat", "lon", "levels"],
        feature_names=[f"feature_{i}" for i in range(50)],
    )
    assert nt4.spatial_dim_idx == [1, 2]
    with pytest.raises(ValueError):
        nt | nt4

    # different number of dims => ValueError
    nt5 = NamedTensor(
        torch.rand(3, 256, 50),
        names=["batch", "lat", "lon"],
        feature_names=[f"feature_{i}" for i in range(50)],
    )
    with pytest.raises(ValueError):
        nt | nt5

    # missing feature with __getitem__ lookup => ValueError
    with pytest.raises(ValueError):
        nt["feature_50"]

    # valid feature name should return tensor of the right shape (unsqueezed on feature dim)
    f = nt["feature_0"]
    assert f.shape == (3, 256, 256, 1)

    # test expanding a lower dim NamedTensor to a higher dim NamedTensor
    nt6 = NamedTensor(
        torch.rand(3, 256),
        names=["batch", "features"],
        feature_names=[f"f_{i}" for i in range(256)],
    )
    nt6.unsqueeze_and_expand_from_(nt)
    assert nt6.tensor.shape == (3, 256, 256, 256)
    assert nt6.names == ["batch", "lat", "lon", "features"]

    # test flattening lat,lon to ndims to simulate gridded data with 2D spatial dims into a GNN
    nt6.flatten_("ngrid", 1, 2)
    assert nt6.tensor.shape == (3, 65536, 256)
    assert nt6.names == ["batch", "ngrid", "features"]
    assert nt6.spatial_dim_idx == [1]

    # test creating a NamedTensor from another NamedTensor
    new_nt = NamedTensor.new_like(torch.rand(3, 256, 256, 50), nt)

    # make sure our __str__ works
    print(new_nt)

    # it must raise ValueError if wrong number of dims
    with pytest.raises(ValueError):
        NamedTensor.new_like(torch.rand(3, 256, 256), nt)

    # it must raise ValueError if wrong number of feature names versus last dim size
    with pytest.raises(ValueError):
        NamedTensor(
            torch.rand(3, 256, 256, 2),
            names=["batch", "lat", "lon", "features"],
            feature_names=[f"f_{i}" for i in range(5)],
        )

    # test one shot concat of multiple NamedTensors
    nt7 = NamedTensor(
        torch.rand(3, 256, 256, 10),
        names=["batch", "lat", "lon", "features"],
        feature_names=[f"feature_{i}" for i in range(10)],
    )
    nt8 = NamedTensor(
        torch.rand(3, 256, 256, 10),
        names=["batch", "lat", "lon", "features"],
        feature_names=[f"v_{i}" for i in range(10)],
    )
    nt9 = NamedTensor(
        torch.rand(3, 256, 256, 10),
        names=["batch", "lat", "lon", "features"],
        feature_names=[f"u_{i}" for i in range(10)],
    )

    nt_cat = NamedTensor.concat([nt7, nt8, nt9])

    assert nt_cat.tensor.shape == (3, 256, 256, 30)
    assert nt_cat.feature_names == [f"feature_{i}" for i in range(10)] + [
        f"v_{i}" for i in range(10)
    ] + [f"u_{i}" for i in range(10)]
    assert nt_cat.names == ["batch", "lat", "lon", "features"]


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
    item = Item(inputs=inputs, outputs=outputs, forcing=forcing)
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


class FakeDataset(DatasetABC):

    def __init__(self):
        pass
     
    def torch_dataloader(self):
        pass

    def dataset_info(self):
        pass

    def meshgrid(self):
        pass

    def geopotential_info(self):
        pass

    def cache_dir(self):
        pass

    def from_json(self, path_to_json):
        with open(path_to_json, "r") as fp:
            conf = json.load(fp)

        grid = PoesyGrib(**conf["grid"])
        return grid

def test_date_forcing():

    fk_ds = FakeDataset()
    date  = datetime.datetime(year = 2023, month = 12, day = 31, hour=23)

    relativ_terms = [1, 2, 3]

    nb_terms = len(relativ_terms)
    nb_forcing = 4
    
    forcing = fk_ds.get_year_hour_forcing(date, relativ_terms)

    assert forcing.shape == torch.Size((nb_terms, nb_forcing))
    assert torch.allclose(forcing[0], torch.tensor([0.5, 1, 0.5, 1]))

    return

def test_solar_forcing():

    fk_ds = FakeDataset()

    date  = datetime.datetime(year = 2023, month = 6, day = 21, hour=0)
    relativ_terms = range(24*5)
    path_to_json = "/home/mrmn/seznecc/repository/py4cast/config/datasets/poesy.json"
    grid  = fk_ds.from_json(path_to_json)

    # forcing = fk_ds.generate_toa_radiation_forcing(grid, date, relativ_terms).sum((1,2,3))


    # import matplotlib.pyplot as plt
    # x_values = range(len(forcing))
    # plt.plot(x_values, forcing, marker='o')
    # plt.savefig("test.png")
    
    return


