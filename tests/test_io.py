"""
Unit tests for datasets and NamedTensor.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from py4cast.datasets.base import NamedTensor
from py4cast.datasets.dummy import DummyDataset
from py4cast.io import outputs as out


class FakeXarrayLatLon(xr.Dataset):
    def __init__(self, lat, lon):
        super().__init__()
        self.Shape = (len(lat), len(lon))
        self.template_ds = xr.Dataset(
            data_vars={
                "fakedata": (
                    ("latitude", "longitude"),
                    np.ones(self.Shape).astype(np.float32),
                )
            },
            coords={"latitude": lat, "longitude": lon},
        )


def test_nan_mask():
    """Test the make_nan_mask function"""

    _, _, dummy_ds = DummyDataset.from_json("fakepath.json", 1, 2, 4)

    exact_lat = (np.arange(64) - 16) * 0.5
    exact_lon = (np.arange(64) + 30) * 0.5
    dummy_template = FakeXarrayLatLon(exact_lat, exact_lon)

    # the function should throw no error and return an exact fit
    nanmask, latlons_idx = out.make_nan_mask(dummy_ds, dummy_template.template_ds)

    # returning all None
    # nanmask definition
    assert nanmask.shape == dummy_template.Shape

    # checking values for latitudes, then longitudes

    assert latlons_idx[:2] == (0, 63)
    assert latlons_idx[2:] == (0, 63)

    fitting_lat = (np.arange(70) - 16) * 0.5
    fitting_lon = (np.arange(70) + 30) * 0.5

    dummy_template = FakeXarrayLatLon(fitting_lat, fitting_lon)

    # the function should throw no error and have correct results
    nanmask, latlons_idx = out.make_nan_mask(dummy_ds, dummy_template.template_ds)

    # nanmask definition
    assert nanmask.shape == dummy_template.Shape

    # checking values for latitudes, then longitudes
    assert latlons_idx[:2] == (0, 63)
    assert latlons_idx[2:] == (0, 63)

    # wrong latitude grid
    with pytest.raises(ValueError):
        unfit_lat = (np.arange(64) - 20) * 0.5
        fit_lon = (np.arange(64) + 30) * 0.5
        dummy_template = FakeXarrayLatLon(unfit_lat, fit_lon)

        nanmask, latlons_idx = out.make_nan_mask(dummy_ds, dummy_template.template_ds)

    # wrong longitude grid
    with pytest.raises(ValueError):
        fit_lat = (np.arange(64) - 16) * 0.5
        unfit_lon = (np.arange(64) + 25) * 0.5
        dummy_template = FakeXarrayLatLon(fit_lat, unfit_lon)

        nanmask, latlons_idx = out.make_nan_mask(dummy_ds, dummy_template.template_ds)


@dataclass
class FakeParam:
    shortname: str = "u"
    levels: tuple[float, ...] = (850.0,)
    level_type: str = "isobaricInhPa"

    @cached_property
    def parameter_short_name(self):
        return [f"{self.shortname}_{lvl}" for lvl in self.levels]


def test_get_grib_param_dataframe():
    names = ["u", "v", "t2m", "u10"]
    lvl_types = ["isobaricInhPa", "heightAboveGround"]
    params = [
        FakeParam(names[i], (900.0, 50.0), lvl_types[0]) for i in range(len(names[:2]))
    ] + [
        FakeParam(names[2], (2,), lvl_types[1]),
        FakeParam(names[3], (10,), lvl_types[1]),
    ]
    plist = []
    for p in params:
        plist.extend(p.parameter_short_name)
    tensor = torch.ones(4, 32, 32, 6)
    pred = NamedTensor(
        tensor, names=["batch", "lat", "lon", "features"], feature_names=plist
    )

    dataframe = out.get_grib_param_dataframe(pred, params)
    reference = pd.DataFrame(
        {
            "feature_name": [
                "u_900.0",
                "u_50.0",
                "v_900.0",
                "v_50.0",
                "t2m_2",
                "u10_10",
            ],
            "level": [900.0, 50.0, 900.0, 50.0, 2, 10],
            "name": ["u", "u", "v", "v", "t2m", "u10"],
            "typeOfLevel": [
                "isobaricInhPa",
                "isobaricInhPa",
                "isobaricInhPa",
                "isobaricInhPa",
                "heightAboveGround",
                "heightAboveGround",
            ],
        },
        index=["u_900.0", "u_50.0", "v_900.0", "v_50.0", "t2m_2", "u10_10"],
    )
    assert reference.equals(dataframe)


def test_get_grib_groups():
    reference = pd.DataFrame(
        {
            "feature_name": [
                "u_900.0",
                "u_50.0",
                "v_900.0",
                "v_50.0",
                "u_900.0",
                "u_50.0",
                "v_900.0",
                "v_50.0",
            ],
            "level": [900.0, 50.0, 900.0, 50.0, 900.0, 50.0, 900.0, 50.0],
            "name": ["u", "u", "v", "v", "u", "u", "v", "v"],
            "typeOfLevel": [
                "heightAboveGround",
                "heightAboveGround",
                "heightAboveGround",
                "heightAboveGround",
                "isobaricInhPa",
                "isobaricInhPa",
                "isobaricInhPa",
                "isobaricInhPa",
            ],
        }
    )

    groups = out.get_grib_groups(reference)

    assert set(groups.keys()) == {
        "u_heightAboveGround",
        "v_heightAboveGround",
        "u_isobaricInhPa",
        "v_isobaricInhPa",
    }

    assert groups["u_heightAboveGround"]["level"] == [900.0, 50.0]
    assert groups["u_heightAboveGround"]["cfVarName"] == "u"
    assert groups["u_heightAboveGround"]["typeOfLevel"] == "heightAboveGround"

    assert groups["v_heightAboveGround"]["level"] == [900.0, 50.0]
    assert groups["v_heightAboveGround"]["cfVarName"] == "v"
    assert groups["v_heightAboveGround"]["typeOfLevel"] == "heightAboveGround"

    assert groups["u_isobaricInhPa"]["level"] == [900.0, 50.0]
    assert groups["u_isobaricInhPa"]["cfVarName"] == "u"
    assert groups["u_isobaricInhPa"]["typeOfLevel"] == "isobaricInhPa"

    assert groups["v_isobaricInhPa"]["level"] == [900.0, 50.0]
    assert groups["v_isobaricInhPa"]["cfVarName"] == "v"
    assert groups["v_isobaricInhPa"]["typeOfLevel"] == "isobaricInhPa"


@dataclass
class FakeSample:
    date: str
    fancy_ident: str


def test_get_output_filename():

    saving_settings = out.GribSavingSettings(
        template_grib="fake_grib.grib",
        directory="fakedirectory",
        sample_identifiers=("date", "leadtime"),
        output_fmt="grid.forecast_ai_date_{}_ech_{}.json",
    )
    sample = FakeSample(date="1911-10-30", fancy_ident="first solvay congress")
    leadtime = 1.0

    filename = out.get_output_filename(saving_settings, sample, leadtime)
    assert filename == f"grid.forecast_ai_date_1911-10-30_ech_{leadtime}.json"

    saving_settings = out.GribSavingSettings(
        template_grib="fake_grib.grib",
        directory="fakedirectory",
        output_kwargs=("congress",),
        sample_identifiers=("date", "leadtime"),
        output_fmt="grid.forecast_{}_date_{}_ech_{}.json",
    )

    filename = out.get_output_filename(saving_settings, sample, leadtime)
