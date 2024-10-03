"""
Unit tests for datasets and NamedTensor.
"""
import datetime

import numpy as np
import pytest
import torch
import xarray as xr
from py4cast.datasets.base import Item, NamedTensor, collate_fn
from py4cast.datasets.dummy import DummyDataset
from py4cast.io import writing_outputs as wr


class FakeXarrayLatLon(xr.Dataset):
    def __init__(self,lat, lon):
        super().__init__()
        self.shape = (len(lat), len(lon))
        self.template_ds = xr.Dataset(
            data_vars={"fakedata" : np.ones(self.shape).astype(np.float32)},
            coords={"latitude" : lat, "longitude" : lon}
        )

def test_nan_mask():
    """Test the make_nan_mask function
    """
    
    _,_, dummy_ds = DummyDataset.from_json("fakepath.json",1,2,4)
    
    fitting_lat = ((np.arange(64) - 16) * 0.5)[4:-4]
    fitting_lon = ((np.arange(64) + 30) * 0.5)[4:-4]
    dummy_template = FakeXarrayLatLon(fitting_lat,fitting_lon)
    
    # the function should throw no error and have correct results
    nanmask, latlons_idx = wr.make_nan_mask(dummy_ds, dummy_template)
    
    #nanmask definition
    assert (nanmask.shape==dummy_template.shape)
    assert (~np.isnan(nanmask)).any()
    with pytest.raises(ValueError):
    # checking values for latitudes, then longitudes
    assert latlons_idx[:2]==(4,60)
    assert latlons_idx[2:]==(4,60)
    
    exact_lat = ((np.arange(64) - 16) * 0.5)
    exact_lon = ((np.arange(64) - 16) * 0.5)
    dummy_template = FakeXarrayLatLon(exact_lat,exact_lon)
    
    # the function should throw no error and return None's
    nanmask, latlons_idx = wr.make_nan_mask(dummy_ds, dummy_template)
    
    # returning all None
    assert (nanmask is None)
    assert latlons_idx==(None,None,None,None)
    
    
