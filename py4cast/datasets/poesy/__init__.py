import datetime as dt
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch

from py4cast.datasets.access import (
    DataAccessor,
    Grid,
    GridConfig,
    ParamConfig,
    SamplePreprocSettings,
    Stats,
    WeatherParam,
)
from py4cast.datasets.base import DatasetABC, Period, Sample, Timestamps
from py4cast.datasets.poesy.settings import (
    LATLON_FNAME,
    METADATA,
    OROGRAPHY_FNAME,
    SCRATCH_PATH,
)
from py4cast.settings import CACHE_DIR


@dataclass
class PoesyAccessor(DataAccessor):

    @staticmethod
    def get_dataset_path(name: str, grid: Grid) -> Path:
        complete_name = str(name) + "_" + grid.name
        return CACHE_DIR / complete_name

    @staticmethod
    def get_weight_per_level(
        level: float,
        level_type: Literal["isobaricInhPa", "heightAboveGround", "surface", "meanSea"],
    ) -> float:
        if level_type == "isobaricInHpa":
            return 1.0 + (level) / (90)
        elif level_type == "heightAboveGround":
            return 2.0
        elif level_type == "surface":
            return 1.0
        else:
            raise Exception(f"unknown level_type:{level_type}")

    @staticmethod
    def load_grid_info(grid: Grid) -> GridConfig:
        geopotential = np.load(SCRATCH_PATH / OROGRAPHY_FNAME)
        latlon = np.load(SCRATCH_PATH / LATLON_FNAME)
        full_size = geopotential.shape
        latitude = latlon[1, :, 0]
        longitude = latlon[0, 0]
        landsea_mask = np.where(geopotential > 0, 1.0, 0.0).astype(np.float32)
        return GridConfig(full_size, latitude, longitude, geopotential, landsea_mask)

    @staticmethod
    def load_param_info(name: str) -> ParamConfig:
        info = METADATA["WEATHER_PARAMS"][name]
        unit = info["unit"]
        long_name = info["long_name"]
        grid = info["grid"]
        level_type = info["level_type"]
        grib_name = None
        grib_param = None
        return ParamConfig(unit, level_type, long_name, grid, grib_name, grib_param)

    @staticmethod
    def get_grid_coords(param: WeatherParam) -> List[float]:
        raise NotImplementedError("Poesy does not require get_grid_coords")

    @staticmethod
    def get_filepath(
        ds_name: str,
        param: WeatherParam,
        date: dt.datetime,
        file_format: str = "npy",
    ) -> str:
        """
        Return the filename.
        """
        var_file_name = METADATA["WEATHER_PARAMS"][param.name]["file_name"]
        return (
            SCRATCH_PATH
            / f"{date.strftime('%Y-%m-%dT%H:%M:%SZ')}_{var_file_name}_lt1-45_crop.npy"
        )

    def load_data_from_disk(
        self,
        ds_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        term: np.array,
        member: int,
        file_format: str = "npy",
    ) -> np.array:
        """
        date : Date of file.
        term : Position of leadtimes in file.
        """
        data_array = np.load(self.get_filepath(ds_name, param, timestamps.datetime), mmap_mode="r")
        return data_array[
            param.grid.subdomain[0] : param.grid.subdomain[1],
            param.grid.subdomain[2] : param.grid.subdomain[3],
            (timestamps.term / dt.timedelta(hours=1)).astype(int) - 1,
            member,
        ]

    def exists(
        self,
        ds_name: str,
        param: WeatherParam,
        timestamps: Timestamps,
        file_format: str = "npy",
    ) -> bool:

        filepath = self.get_filepath(ds_name, param, timestamps.datetime, file_format)
        if not filepath.exists():
            return False
        return True

    def valid_timestamp(self, n_inputs: int, timestamps: Timestamps) -> bool:
        """
        Verification function called after the creation of each timestamps.
        Check if computed terms respect the dataset convention.
        Reminder:
        Poesy terms are between +1h lead time and +45h lead time.
        """
        limits = METADATA["TERMS"]
        for t in timestamps.terms:

            if (t > dt.timedelta(hours=int(limits["end"]))) or (
                t < dt.timedelta(hours=int(limits["start"]))
            ):
                return False
        return True

class InferSample(Sample):
    """
    Sample dedicated to inference. No outputs terms, only inputs.
    """

    def __post_init__(self):
        self.terms = self.input_terms


class PoesyDataset(DatasetABC):
    def __init__(
        self,
        name: str,
        grid: Grid,
        period: Period,
        params: List[WeatherParam],
        settings: SamplePreprocSettings,
        accessor_kls: PoesyAccessor,
    ):
        super().__init__(name, grid, period, params, settings, accessor=accessor_kls())

    def __len__(self):
        return len(self.sample_list)


# class InferPoesyDataset(PoesyDataset):
#     """
#     Inherite from the PoesyDataset class.
#     This class is used for inference, the class overrides methods sample_list and from_json.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     @cached_property
#     def sample_list(self):
#         """
#         Create a list of sample from information.
#         Outputs terms are computed from the number of prediction steps in argument.
#         """
#         print("Start forming samples")
#         terms = list(
#             np.arange(
#                 METADATA["TERMS"]["start"],
#                 METADATA["TERMS"]["end"],
#                 METADATA["TERMS"]["timestep"],
#             )
#         )

#         num_total_steps = self.settings.num_input_steps + self.settings.num_pred_steps
#         sample_by_date = len(terms) // num_total_steps
#         samples = []
#         number = 0
#         for date in self.period.date_list:
#             for member in self.settings.members:
#                 for sample in range(0, sample_by_date):
#                     input_terms = terms[
#                         sample * num_total_steps : sample * num_total_steps
#                         + self.settings.num_input_steps
#                     ]

#                     output_terms = [
#                         input_terms[-1] + METADATA["TERMS"]["timestep"] * (step + 1)
#                         for step in range(self.settings.num_pred_steps)
#                     ]

#                     samp = InferSample(
#                         date=date,
#                         member=member,
#                         settings=self.settings,
#                         input_terms=input_terms,
#                         output_terms=output_terms,
#                     )

#                     if samp.is_valid():
#                         samples.append(samp)
#                         number += 1
#         print(f"All {len(samples)} samples are now defined")

#         return samples

#     @classmethod
#     def from_json(
#         cls,
#         fname: Path,
#         num_input_steps: int,
#         num_pred_steps_train: int,
#         num_pred_steps_val_tests: int,
#         config_override: Union[Dict, None] = None,
#     ) -> Tuple[None, None, "InferPoesyDataset"]:
#         """
#         Return 1 InferPoesyDataset.
#         Override configuration file if needed.
#         """

#         with open(fname, "r") as fp:
#             conf = json.load(fp)
#             if config_override is not None:
#                 conf = merge_dicts(conf, config_override)
#                 print(conf["periods"]["test"])
#         conf["grid"]["load_grid_info_func"] = load_grid_info
#         grid = Grid(**conf["grid"])
#         param_list = get_param_list(conf, grid, load_param_info, get_weight_per_level)
#         inference_period = Period(**conf["periods"]["test"], name="infer")

#         ds = InferPoesyDataset(
#             grid,
#             inference_period,
#             param_list,
#             SamplePreprocSettings(
#                 num_pred_steps=0,
#                 num_input_steps=num_input_steps,
#                 members=conf["members"],
#                 **conf["settings"],
#             ),
#         )

#         return None, None, ds


if __name__ == "__main__":
    path_config = "config/datasets/poesy.json"

    parser = ArgumentParser(description="Prepare Poesy dataset and test loading speed.")
    parser.add_argument(
        "--path_config",
        default=path_config,
        type=Path,
        help="Configuration file for the dataset.",
    )
    parser.add_argument(
        "--n_iter",
        default=10,
        type=int,
        help="Number of samples to test loading speed.",
    )
    args = parser.parse_args()

    PoesyDataset.prepare(args.path_config)

    print("Dataset info : ")
    train_ds, _, _ = PoesyDataset.from_json(args.path_config, 2, 3, 3)
    train_ds.dataset_info.summary()

    print("Test __get_item__")
    print("Len dataset : ", len(train_ds))

    print("First Item description :")
    data_iter = iter(train_ds.torch_dataloader())
    print(next(data_iter))
