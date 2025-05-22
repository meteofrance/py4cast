"""
Unit tests for datasets and NamedTensor.
"""

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


def test_output_saving_settings():
    """
    The function evaluates the output of the class's methods.
    """
    settings = out.OutputSavingSettings(
        template_grib="./template/test.grib",
        dir_grib="./path/to/gribdir",
        dir_gif="./path/to/gifdir",
        path_to_runtime="Rocky_{}/runtime_{}",
        output_kwargs=["Balboa"],
        grib_fmt="mb_{}/leadtime_{}.grib",
        grib_identifiers=["member", "leadtime"],
        gif_fmt="runtime_{}_feature_{}.gif",
        gif_identifiers=["runtime", "feature"],
    )
    assert (
        str(settings.get_gif_path(3, "feature"))
        == "path/to/gifdir/Rocky_Balboa/runtime_3/runtime_3_feature_feature.gif"
    )
    assert (
        str(settings.get_grib_path(3, 5, 2))
        == "path/to/gribdir/Rocky_Balboa/runtime_3/mb_005/leadtime_2.grib"
    )


@pytest.mark.parametrize(
    "path_to_runtime, output_kwargs, gif_fmt, gif_identifiers",
    [
        (
            "Rocky_{}/Rocky_{}",
            ["Balboa", "Marciano"],
            "runtime_{}_feature_{}.gif",
            ["runtime", "feature"],
        ),
        (
            "Rocky_{}/Rocky_{}",
            ["Balboa"],
            "runtime_{}_feature_{}.gif",
            ["runtime"],
        ),
    ],
)
def test_get_gif_path(path_to_runtime, output_kwargs, gif_fmt, gif_identifiers):
    """
    This test function iterates over get_gif_path to raise the value error of mismatch
    between the number of placeholders and identifiers.
    """
    with pytest.raises(ValueError):
        settings = out.OutputSavingSettings(
            template_grib="./template/test.grib",
            dir_grib="./path/to/gribdir",
            dir_gif="./path/to/gifdir",
            path_to_runtime=path_to_runtime,
            output_kwargs=output_kwargs,
            grib_fmt="mb_{}/leadtime_{}.grib",
            grib_identifiers=["member", "leadtime"],
            gif_fmt=gif_fmt,
            gif_identifiers=gif_identifiers,
        )
        settings.get_gif_path(runtime="2024052000", feature="feature")


@pytest.mark.parametrize(
    "path_to_runtime, output_kwargs, grib_fmt, grib_identifiers",
    [
        (
            "Rocky_{}/Rocky_{}",
            ["Balboa", "Marciano"],
            "mb_{}/leadtime_{}.grib",
            ["member", "leadtime"],
        ),
        (
            "Rocky_{}/Rocky_{}",
            ["Balboa"],
            "mb_{}/leadtime.grib",
            ["member", "leadtime"],
        ),
    ],
)
def test_get_grib_path(path_to_runtime, output_kwargs, grib_fmt, grib_identifiers):
    """
    This test function iterates over get_grib_path to raise the value error of mismatch
    between the number of placeholders and identifiers.
    """
    with pytest.raises(ValueError):
        settings = out.OutputSavingSettings(
            template_grib="/template/test.grib",
            dir_grib="./path/to/gribdir",
            dir_gif="./path/to/gifdir",
            path_to_runtime=path_to_runtime,
            output_kwargs=output_kwargs,
            grib_fmt=grib_fmt,
            grib_identifiers=grib_identifiers,
            gif_fmt="runtime_{}_feature_{}.gif",
            gif_identifiers=["runtime", "feature"],
        )
        settings.get_grib_path(runtime="2024052000", member=3, leadtime=1)
