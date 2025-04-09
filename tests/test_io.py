"""
Unit tests for datasets and NamedTensor.
"""

import datetime
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import yaml

from py4cast.datasets.access import Timestamps
from py4cast.datasets.base import DatasetABC
from py4cast.datasets.dummy import DummyAccessor
from py4cast.io import outputs as out

DUMMY_CONFIG = Path(__file__).parents[1] / "config/CLI/dataset/dummy.yaml"

with open(DUMMY_CONFIG, "r") as fp:
    conf = yaml.safe_load(fp)["data"]["dataset_conf"]


def test_match_latlon():
    """Test the match_latlon function"""

    _, _, dummy_ds = DatasetABC.from_dict(
        accessor_kls=DummyAccessor,
        name="dummy",
        conf=conf,
        num_input_steps=1,
        num_pred_steps_train=2,
        num_pred_steps_val_test=2,
    )

    exact_lat = (np.arange(64) - 16) * 0.5
    exact_lon = (np.arange(64) + 30) * 0.5

    # the function should throw no error and return an exact fit
    idxs_latlon_grib = out.match_latlon(dummy_ds, exact_lat, exact_lon)

    # checking values for latitudes, then longitudes
    assert idxs_latlon_grib[:2] == (0, 63)
    assert idxs_latlon_grib[2:] == (0, 63)

    fitting_lat = (np.arange(70) - 16) * 0.5
    fitting_lon = (np.arange(70) + 30) * 0.5

    # the function should throw no error and have correct results
    idxs_latlon_grib = out.match_latlon(dummy_ds, fitting_lat, fitting_lon)

    # checking values for latitudes, then longitudes
    assert idxs_latlon_grib[:2] == (0, 63)
    assert idxs_latlon_grib[2:] == (0, 63)

    # wrong latitude grid
    with pytest.raises(ValueError):
        unfit_lat = (np.arange(64) - 20) * 0.5
        fit_lon = (np.arange(64) + 30) * 0.5

        idxs_latlon_grib = out.match_latlon(dummy_ds, unfit_lat, fit_lon)

    # wrong longitude grid
    with pytest.raises(ValueError):
        fit_lat = (np.arange(64) - 16) * 0.5
        unfit_lon = (np.arange(64) + 25) * 0.5

        idxs_latlon_grib = out.match_latlon(dummy_ds, fit_lat, unfit_lon)


@dataclass
class FakeSample:
    timestamps: Timestamps
    fancy_ident: str
    output_timestamps: Timestamps = field(default=None)
    member: int = 1


def test_fill_tensor_with():

    shape_grib = (1000, 500)
    embedded_idxs = (500, 600, 250, 350)
    embedded_data = False
    default_v = True
    _dtype = bool

    tensor = out.fill_tensor_with(
        embedded_data=embedded_data,
        embedded_idxs=embedded_idxs,
        shape=shape_grib,
        default_v=default_v,
        _dtype=_dtype,
    )

    assert np.all(~tensor[500:600, 250:350])
    mask = np.ones_like(tensor, dtype=bool)
    mask[500:601, 250:351] = False
    assert np.all(tensor[mask])


def test_get_output_filename():
    saving_settings = out.GribSavingSettings(
        template_grib="fake_grib.grib",
        directory="fakedirectory",
        sample_identifiers=["runtime", "member", "leadtime"],
        output_fmt="no_dataset/{}/mb{}/forecast/grid.emul_aro_ai_ech_{}.grib",
    )
    sample = FakeSample(
        timestamps=Timestamps(
            datetime.datetime(1911, 10, 30),
            [datetime.timedelta(days=i) for i in range(1, 5)],
        ),
        fancy_ident="first solvay congress",
        output_timestamps=Timestamps(
            datetime.datetime(1911, 10, 30),
            [datetime.timedelta(days=i) for i in range(3, 5)],
        ),
    )
    leadtime = 1.0

    filename = out.get_output_filename(saving_settings, sample, leadtime)
    assert (
        filename
        == "no_dataset/19111101T0000P/mb002/forecast/grid.emul_aro_ai_ech_1.0.grib"
    )

    saving_settings = out.GribSavingSettings(
        template_grib="fake_grib.grib",
        directory="fakedirectory",
        output_kwargs=("congress",),
        sample_identifiers=["runtime", "member", "leadtime"],
        output_fmt="dataset_{}/{}/mb{}/forecast/grid.emul_aro_ai_ech_{}.grib",
    )

    filename = out.get_output_filename(saving_settings, sample, leadtime)
