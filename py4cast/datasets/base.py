"""
Base classes defining our software components
and their interfaces
"""

import datetime as dt
import json
from copy import deepcopy
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Type, Union

import einops
import gif
import matplotlib.pyplot as plt
import numpy as np
import torch
from mfai.torch.namedtensor import NamedTensor
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import collate_tensor_fn
from tqdm import tqdm

from py4cast.datasets.access import (
    DataAccessor,
    Grid,
    Period,
    SamplePreprocSettings,
    Stats,
    Timestamps,
    WeatherParam,
    grid_static_features,
)
from py4cast.forcingutils import generate_toa_radiation_forcing, get_year_hour_forcing
from py4cast.plots import DomainInfo
from py4cast.utils import RegisterFieldsMixin, merge_dicts


@dataclass(slots=True)
class Item:
    """
    Dataclass holding one Item.
    inputs has shape (timestep, lat, lon, features)
    outputs has shape (timestep, lat, lon, features)
    forcing has shape (timestep, lat, lon, features)
    """

    inputs: NamedTensor | None
    forcing: NamedTensor | None
    outputs: NamedTensor

    def unsqueeze_(self, dim_name: str, dim_index: int):
        """
        Insert a new dimension dim_name at dim_index of size 1
        """
        self.outputs.unsqueeze_(dim_name, dim_index)
        if self.inputs:
            self.inputs.unsqueeze_(dim_name, dim_index)
        if self.forcing:
            self.forcing.unsqueeze_(dim_name, dim_index)

    def squeeze_(self, dim_name: Union[List[str], str]):
        """
        Squeeze the underlying tensor along the dimension(s)
        given its/their name(s).
        """
        self.outputs.squeeze_(dim_name)
        if self.inputs:
            self.inputs.squeeze_(dim_name)
        if self.forcing:
            self.forcing.squeeze_(dim_name)

    def to_(self, *args, **kwargs):
        """
        'In place' operation to call torch's 'to' method on the underlying NamedTensors.
        """
        
        self.outputs.to_(*args, **kwargs)
        if self.inputs:
            self.inputs.squeeze_(dim_name)
        if self.forcing:
            self.forcing.squeeze_(dim_name)

    def pin_memory(self):
        """
        Custom Item must implement this method to pin the underlying tensors to memory.
        See https://pytorch.org/docs/stable/data.html#memory-pinning
        """
        self.outputs.pin_memory_()
        if self.inputs:
            self.inputs.pin_memory_()
        if self.forcing:
            self.forcing.pin_memory_()
        return self

    def __post_init__(self):
        """
        Checks that the dimensions of the inputs, outputs are consistent.
        This is necessary for our auto-regressive training.
        """
        if self.inputs.names != self.outputs.names:
            raise ValueError(
                f"Inputs and outputs must have the same dim names, got {self.inputs.names} and {self.outputs.names}"
            )

        # Also check feature names
        if self.inputs.feature_names != self.outputs.feature_names:
            raise ValueError(
                f"Inputs and outputs must have the same feature names, "
                f"got {self.inputs.feature_names} and {self.outputs.feature_names}"
            )

    def __str__(self) -> str:
        """
        Utility method to explore a batch/item shapes and names.
        """
        table = []
        for attr in (f.name for f in fields(self)):
            nt: NamedTensor = getattr(self, attr)
            if nt is not None:
                for feature_name in nt.feature_names:
                    tensor = nt[feature_name]
                    table.append(
                        [
                            attr,
                            nt.names,
                            list(nt[feature_name].shape),
                            feature_name,
                            tensor.min(),
                            tensor.max(),
                        ]
                    )
        headers = [
            "Type",
            "Dimension Names",
            "Torch Shape",
            "feature name",
            "Min",
            "Max",
        ]
        return str(tabulate(table, headers=headers, tablefmt="simple_outline"))


@dataclass
class ItemBatch(Item):
    """
    Dataclass holding a batch of items.
    input has shape (batch, timestep, lat, lon, features)
    output has shape (batch, timestep, lat, lon, features)
    forcing has shape (batch, timestep, lat, lon, features)
    """

    @cached_property
    def batch_size(self):
        return self.outputs.dim_size("batch")

    @cached_property
    def num_input_steps(self):
        return self.outputs.dim_size("timestep")

    @cached_property
    def num_pred_steps(self):
        return self.outputs.dim_size("timestep")


def collate_fn(items: List[Item]) -> ItemBatch:
    """
    Collate a list of item. Add one dimension at index zero to each NamedTensor.
    Necessary to form a batch from a list of items.
    See https://pytorch.org/docs/stable/data.html#working-with-collate-fn
    """
    # Here we postpone that for each batch the same dimension should be present.
    batch_of_items = {}

    # Iterate over inputs, outputs and forcing fields
    for field_name in (f.name for f in fields(Item)):
        batched_tensor = collate_tensor_fn(
            [getattr(item, field_name).tensor for item in items]
        ).type(torch.float32)

        batch_of_items[field_name] = NamedTensor.expand_to_batch_like(
            batched_tensor, getattr(items[0], field_name)
        )

    return ItemBatch(**batch_of_items)


@dataclass
class Statics(RegisterFieldsMixin):
    """
    Static fields of the dataset.
    Tensor can be registered as buffer in a lightning module
    using the register_buffers method.
    """

    # border_mask: torch.Tensor
    grid_statics: NamedTensor
    grid_shape: Tuple[int, int]
    border_mask: torch.Tensor = field(init=False)
    interior_mask: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.border_mask = self.grid_statics["border_mask"]
        self.interior_mask = 1.0 - self.border_mask

    @cached_property
    def meshgrid(self) -> torch.Tensor:
        """
        Return a tensor concatening X,Y
        """
        return einops.rearrange(
            torch.cat(
                [
                    self.grid_statics["x"],
                    self.grid_statics["y"],
                ],
                dim=-1,
            ),
            ("x y n -> n x y"),
        )


def generate_forcings(
    date: dt.datetime, timedeltas: List[dt.timedelta], grid: Grid
) -> List[NamedTensor]:
    """
    Generate all the forcing in this function.
    Return a list of NamedTensor.
    """
    # Datetime Forcing
    datetime_forcing = get_year_hour_forcing(date, timedeltas).type(torch.float32)

    # Solar forcing, dim : [num_pred_steps, Lat, Lon, feature = 1]
    solar_forcing = generate_toa_radiation_forcing(
        grid.lat, grid.lon, date, timedeltas
    ).type(torch.float32)

    lforcings = [
        NamedTensor(
            feature_names=[
                "cos_hour",
                "sin_hour",
            ],  # doy : day_of_year
            tensor=datetime_forcing[:, :2],
            names=["timestep", "features"],
        ),
        NamedTensor(
            feature_names=[
                "cos_doy",
                "sin_doy",
            ],  # doy : day_of_year
            tensor=datetime_forcing[:, 2:],
            names=["timestep", "features"],
        ),
        NamedTensor(
            feature_names=[
                "toa_radiation",
            ],
            tensor=solar_forcing,
            names=["timestep", "lat", "lon", "features"],
        ),
    ]

    return lforcings


@dataclass(slots=True)
class DatasetInfo:
    """
    This dataclass holds all the informations
    about the dataset that other classes
    and functions need to interact with it.
    """

    name: str  # Name of the dataset
    domain_info: DomainInfo  # Information used for plotting
    units: Dict[str, str]  # d[shortname] = unit (str)
    weather_dim: int
    forcing_dim: int
    pred_step: dt.timedelta  # Duration of one step in the dataset.
    statics: Statics  # A lot of static variables
    stats: Stats
    diff_stats: Stats
    state_weights: Dict[str, float]
    shortnames: Dict[str, List[str]] = None

    def summary(self):
        """
        Print a table summarizing variables present in the dataset (and their role)
        """
        print(f"\n Summarizing {self.name} \n")
        print(f"Step_duration {self.pred_step}")
        print(f"Static fields {self.statics.grid_statics.feature_names}")
        print(f"Grid static features {self.statics.grid_statics}")
        print(f"Features shortnames {self.shortnames}")
        for p in ["input", "input_output", "output"]:
            names = self.shortnames[p]
            mean = self.stats.to_list("mean", names)
            std = self.stats.to_list("std", names)
            mini = self.stats.to_list("min", names)
            maxi = self.stats.to_list("max", names)
            units = [self.units[name] for name in names]
            if p != "input":
                diff_mean = self.diff_stats.to_list("mean", names)
                diff_std = self.diff_stats.to_list("std", names)
                weight = [self.state_weights[name] for name in names]

                data = list(
                    zip(
                        names, units, mean, std, mini, maxi, diff_mean, diff_std, weight
                    )
                )
                table = tabulate(
                    data,
                    headers=[
                        "Name",
                        "Unit",
                        "Mean",
                        "Std",
                        "Minimum",
                        "Maximum",
                        "DiffMean",
                        "DiffStd",
                        "Weight in Loss",
                    ],
                    tablefmt="simple_outline",
                )
            else:
                data = list(zip(names, units, mean, std, mini, maxi))
                table = tabulate(
                    data,
                    headers=["Name", "Unit", "Mean", "Std", "Minimun", "Maximum"],
                    tablefmt="simple_outline",
                )
            if data:
                print(p.upper())  # Print the kind of variable
                print(table)  # Print the table


def get_param_list(
    conf: dict,
    grid: Grid,
    # function to retrieve all parameters information about the dataset
    accessor: DataAccessor,
) -> List[WeatherParam]:
    param_list = []
    for name, values in conf["params"].items():
        for lvl in values["levels"]:
            param = WeatherParam(
                name=name,
                level=lvl,
                grid=grid,
                load_param_info=accessor.load_param_info,
                kind=values["kind"],
                get_weight_per_level=accessor.get_weight_per_level,
            )
            param_list.append(param)
    return param_list


#############################################################
#                            SAMPLE                         #
#############################################################


@dataclass(slots=True)
class Sample:
    """
    Describes a sample from a given dataset.
    The description is a "light" collection of objects
    and manipulation functions.
    Provide "autonomous" functionalities for a Sample
     -> load data from the description and return an Item
     -> plot each timestep in the sample
     -> plot a gif from the whole sample
    """

    timestamps: Timestamps
    settings: SamplePreprocSettings
    params: List[WeatherParam]
    stats: Stats
    grid: Grid
    accessor: DataAccessor
    member: int = 0

    output_timestamps: Timestamps = field(default=None)

    def __post_init__(self):
        """Setups time variables to be able to define a sample.
        For example for n_inputs = 2, n_preds = 3, step_duration = 3h:
        all_steps = [-1, 0, 1, 2, 3]
        all_timesteps = [-3h, 0h, 3h, 6h, 9h]
        pred_timesteps = [3h, 6h, 9h]
        all_dates = [24/10/22 21:00,  24/10/23 00:00, 24/10/23 03:00, 24/10/23 06:00, 24/10/23 09:00]
        """

        if self.settings.num_input_steps + self.settings.num_pred_steps != len(
            self.timestamps.validity_times
        ):
            raise Exception("Length of validity times does not match inputs + outputs")

        self.output_timestamps = Timestamps(
            datetime=self.timestamps.datetime,
            timedeltas=self.timestamps.timedeltas[self.settings.num_input_steps :],
        )

    def __repr__(self):
        return f"Date {self.timestamps.datetime}"

    def is_valid(self) -> bool:
        for param in self.params:
            if not self.accessor.exists(
                ds_name=self.settings.dataset_name,
                param=param,
                timestamps=self.timestamps,
                file_format=self.settings.file_format,
            ):
                return False
        return True

    def get_param_tensor(
        self, param: WeatherParam, timestamps: Timestamps, standardize: bool
    ) -> torch.tensor:
        """
        Fetch data on disk fo the given parameter and all involved dates
        Unless specified, normalize the samples with parameter-specific constants
        returns a tensor
        """

        arr = self.accessor.load_data_from_disk(
            self.settings.dataset_name,
            param,
            timestamps,
            self.member,
            self.settings.file_format,
        )

        if standardize:
            name = self.accessor.parameter_namer(param)
            means = np.asarray(self.stats[name]["mean"])
            std = np.asarray(self.stats[name]["std"])
            arr = (arr - means) / std
        return torch.from_numpy(arr)

    def load(self, no_standardize: bool = False) -> Item:
        """
        Return inputs, outputs, forcings as tensors concatenated into an Item.
        """
        linputs, loutputs, lforcings = [], [], []

        # Reading parameters from files
        for param in self.params:
            state_kwargs = {
                "feature_names": [self.accessor.parameter_namer(param)],
                "names": ["timestep", "lat", "lon", "features"],
            }

            stamps = (
                self.timestamps
                if param.kind == "input_output"
                else self.output_timestamps
            )
            tensor = self.get_param_tensor(
                param=param,
                timestamps=stamps,
                standardize=(self.settings.standardize and not no_standardize),
            )
            tmp_state = NamedTensor(tensor=tensor, **deepcopy(state_kwargs))

            if param.kind == "input":
                # forcing is an input, taken for every predicted step (= output timestamps)
                lforcings.append(tmp_state)

            elif param.kind == "output":
                # output-only params count in the loss
                loutputs.append(tmp_state)

            else:  # input_output, separated along the steps
                loutputs.append(
                    NamedTensor(
                        tensor=tensor[-self.settings.num_pred_steps :],
                        **deepcopy(state_kwargs),
                    )
                )

                linputs.append(
                    NamedTensor(
                        tensor=tensor[: self.settings.num_input_steps],
                        **deepcopy(state_kwargs),
                    )
                )

        external_forcings = generate_forcings(
            date=self.timestamps.datetime,
            timedeltas=self.output_timestamps.timedeltas,
            grid=self.grid,
        )

        for forcing in external_forcings:
            forcing.unsqueeze_and_expand_from_(loutputs[0])
        lforcings = lforcings + external_forcings

        inputs = NamedTensor.concat(linputs) if linputs else None
        outputs = NamedTensor.concat(loutputs) if loutputs else None
        forcing = NamedTensor.concat(lforcings) if lforcings else None

        if outputs is None:
            raise ValueError("Can't train anything without target data: list of outputs is empty.")


        return Item(
            inputs=inputs,
            outputs=outputs,
            forcing=forcing,
        )

    def plot(self, item: Item, step: int, save_path: Path = None) -> None:
        # Retrieve the named tensor
        ntensor = item.inputs if step <= 0 else item.outputs

        # Retrieve the timestep data index
        if step <= 0:  # input step
            index_tensor = step + self.settings.num_input_steps - 1
        else:  # output step
            index_tensor = step - 1

        # Sort parameters by level, to plot each level on one line
        levels = sorted(list(set([p.level for p in self.params])))
        dict_params = {level: [] for level in levels}
        for param in self.params:
            name = self.accessor.parameter_namer(param)
            if name in ntensor.feature_names:
                dict_params[param.level].append(param)

        # Groups levels 0m, 2m and 10m on one "surf" level
        dict_params["surf"] = []
        for lvl in [0, 2, 10]:
            if lvl in levels:
                dict_params["surf"] += dict_params.pop(lvl)

        # Plot settings
        kwargs = {"projection": self.grid.projection}
        nrows = len(dict_params.keys())
        ncols = max([len(param_list) for param_list in dict_params.values()])
        fig, axs = plt.subplots(nrows, ncols, figsize=(20, 15), subplot_kw=kwargs)

        for i, level in enumerate(dict_params.keys()):
            for j, param in enumerate(dict_params[level]):
                pname = self.accessor.parameter_namer(param)
                tensor = ntensor[pname][index_tensor, :, :, 0]
                arr = tensor.numpy()[::-1]  # invert latitude
                vmin, vmax = self.stats[pname]["min"], self.stats[pname]["max"]
                img = axs[i, j].imshow(
                    arr, vmin=vmin, vmax=vmax, extent=self.grid.grid_limits
                )
                axs[i, j].set_title(pname)
                axs[i, j].coastlines(resolution="50m")
                cbar = fig.colorbar(img, ax=axs[i, j], fraction=0.04, pad=0.04)
                cbar.set_label(param.unit)

        plt.suptitle(
            f"Run: {self.timestamps.datetime} - Valid time: {self.timestamps.validity_times[step]}"
        )
        plt.tight_layout()

        # this function can be a interm. step for gif plotting
        # hence the plt.fig is not closed (or saved) by default ;
        # this is a desired behavior
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()

    @gif.frame
    def plot_frame(self, item: Item, step: int) -> None:
        """
        Intermediary step, using plotting without saving, to be used in gif
        """
        self.plot(item, step)

    def plot_gif(self, save_path: Path):
        """
        Making a gif starting from the first input step to the last output step
        Using the functionalities of the Sample (ability to load and plot a single frame)
        """
        # We don't want to standardize data for plots
        item = self.load(no_standardize=True)
        frames = []
        n_inputs, n_preds = self.settings.num_input_steps, self.settings.num_pred_steps
        steps = list(range(-n_inputs + 1, n_preds + 1))
        for step in tqdm(steps, desc="Making gif"):
            frame = self.plot_frame(item, step)
            frames.append(frame)
        gif.save(frames, str(save_path), duration=250)


class DatasetABC(Dataset):
    """
    Base class for gridded datasets used in weather forecasts
    """

    def __init__(
        self,
        name: str,
        grid: Grid,
        period: Period,
        params: List[WeatherParam],
        settings: SamplePreprocSettings,
        accessor: Type[DataAccessor],
    ):
        self.name = name
        self.grid = grid
        self.period = period
        self.params = params
        self.settings = settings
        self.accessor = accessor
        self.shuffle = self.period.name == "train"
        self.cache_dir = accessor.cache_dir(name, grid)

    def __str__(self) -> str:
        return f"{self.name}_{self.grid.name}"

    def __getitem__(self, index):
        """
        Return an item from an index of the sample_list
        """
        sample = self.sample_list[index]
        item = sample.load()
        return item

    def __len__(self):
        return len(self.sample_list)

    @cached_property
    def dataset_info(self) -> DatasetInfo:
        """Returns a DatasetInfo object describing the dataset.

        Returns:
            DatasetInfo: _description_
        """
        shortnames = {
            "input": self.shortnames("input"),
            "input_output": self.shortnames("input_output"),
            "output": self.shortnames("output"),
        }
        return DatasetInfo(
            name=str(self),
            domain_info=self.domain_info,
            shortnames=shortnames,
            units=self.units,
            weather_dim=self.input_output_dim,
            forcing_dim=self.input_dim,
            pred_step=self.period.forecast_step,
            statics=self.statics,
            stats=self.stats,
            diff_stats=self.diff_stats,
            state_weights=self.state_weights,
        )

    @cached_property
    def sample_list(self) -> List[Sample]:
        """Creates the list of samples."""
        print("Start creating samples...")
        stats = self.stats if self.settings.standardize else None

        timestamps = []
        for t0, leadtime in self.period.available_t0_and_leadtimes:
            if self.accessor.optional_check_before_exists(
                t0,
                self.settings.num_input_steps,
                self.settings.num_pred_steps,
                self.period.forecast_step,
                leadtime,
            ):
                timesteps = [
                    delta * self.period.forecast_step + leadtime
                    for delta in range(
                        -self.settings.num_input_steps + 1,
                        self.settings.num_pred_steps + 1,
                    )
                ]
                timestamps.append(Timestamps(datetime=t0, timedeltas=timesteps))

        samples = []
        invalid_samples = 0
        for ts in tqdm(
            timestamps, desc=f"Checking samples of '{self.period.name}' period"
        ):
            for member in self.settings.members:
                sample = Sample(
                    ts,
                    self.settings,
                    self.params,
                    stats,
                    self.grid,
                    self.accessor,
                    member,
                )
                if sample.is_valid():
                    samples.append(sample)
                else:
                    invalid_samples += 1
        print(
            f"--> {len(samples)} {self.period.name} samples are now defined, with {invalid_samples} invalid samples."
        )
        return samples

    def torch_dataloader(
        self,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        prefetch_factor: Union[int, None],
        pin_memory: bool,
    ) -> DataLoader:
        """
        Builds a torch dataloader from self.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            prefetch_factor=prefetch_factor,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    @cached_property
    def input_dim(self) -> int:
        """
        Return the number of forcings.
        """
        res = 4  # For date
        res += 1  # For solar forcing

        for param in self.params:
            if param.kind == "input":
                res += 1
        return res

    @cached_property
    def input_output_dim(self) -> int:
        """
        Return the dimension of pronostic variable.
        """
        res = 0
        for param in self.params:
            if param.kind == "input_output":
                res += 1
        return res

    @cached_property
    def output_dim(self):
        """
        Return dimensions of output variable only
        Not used yet
        """
        res = 0
        for param in self.params:
            if param.kind == "output":
                res += 1
        return res

    @property
    def dataset_extra_statics(self) -> List[NamedTensor]:
        """
        Datasets can override this method to add
        more static data.
        Optionally, add the LandSea Mask to the statics."""

        if self.settings.add_landsea_mask:
            return [
                NamedTensor(
                    feature_names=["LandSeaMask"],
                    tensor=torch.from_numpy(self.grid.landsea_mask)
                    .type(torch.float32)
                    .unsqueeze(2),
                    names=["lat", "lon", "features"],
                )
            ]
        return []

    @cached_property
    def grid_shape(self) -> tuple:
        x, _ = self.grid.meshgrid
        return x.shape

    @cached_property
    def statics(self) -> Statics:
        return Statics(
            **{
                "grid_statics": grid_static_features(
                    self.grid, self.dataset_extra_statics
                ),
                "grid_shape": self.grid_shape,
            }
        )

    @cached_property
    def stats(self) -> Stats:
        return Stats(fname=self.cache_dir / "parameters_stats.pt")

    @cached_property
    def diff_stats(self) -> Stats:
        return Stats(fname=self.cache_dir / "diff_stats.pt")

    def shortnames(
        self,
        kind: List[Literal["input", "output", "input_output"]] = [
            "input",
            "output",
            "input_output",
        ],
    ) -> List[str]:
        """
        List of readable names for the parameters in the dataset.
        Does not include grid information (such as geopotentiel and LandSeaMask).
        Make the difference between inputs, outputs.
        """
        return [self.accessor.parameter_namer(p) for p in self.params if p.kind == kind]

    @cached_property
    def units(self) -> Dict[str, str]:
        """
        Return a dictionnary with name and units
        """
        return {self.accessor.parameter_namer(p): p.unit for p in self.params}

    @cached_property
    def state_weights(self):
        """Weights used in the loss function."""
        kinds = ["output", "input_output"]
        return {
            self.accessor.parameter_namer(p): p.state_weight
            for p in self.params
            if p.kind in kinds
        }

    @cached_property
    def domain_info(self) -> DomainInfo:
        """Information on the domain considered. Usefull information for plotting."""
        return DomainInfo(
            grid_limits=self.grid.grid_limits, projection=self.grid.projection
        )

    @classmethod
    def from_dict(
        cls,
        accessor_kls: Type[DataAccessor],
        name: str,
        conf: dict,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_test: int,
    ) -> Tuple[Type["DatasetABC"], Type["DatasetABC"], Type["DatasetABC"]]:
        grid = Grid(load_grid_info_func=accessor_kls.load_grid_info, **conf["grid"])

        try:
            members = conf["members"]
        except KeyError:
            members = [0]

        param_list = get_param_list(conf, grid, accessor_kls)

        train_settings = SamplePreprocSettings(
            dataset_name=name,
            num_input_steps=num_input_steps,
            num_pred_steps=num_pred_steps_train,
            members=members,
            **conf["settings"],
        )
        train_period = Period(**conf["periods"]["train"], name="train")
        train_ds = cls(
            name, grid, train_period, param_list, train_settings, accessor_kls()
        )

        valid_settings = SamplePreprocSettings(
            dataset_name=name,
            num_input_steps=num_input_steps,
            num_pred_steps=num_pred_steps_val_test,
            members=members,
            **conf["settings"],
        )
        valid_period = Period(**conf["periods"]["valid"], name="valid")
        valid_ds = cls(
            name, grid, valid_period, param_list, valid_settings, accessor_kls()
        )

        test_period = Period(**conf["periods"]["test"], name="test")
        test_ds = cls(
            name, grid, test_period, param_list, valid_settings, accessor_kls()
        )

        return train_ds, valid_ds, test_ds

    @classmethod
    def from_json(
        cls,
        accessor_kls: Type[DataAccessor],
        fname: Path,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_tests: int,
        config_override: Union[Dict, None] = None,
    ) -> Tuple[Type["DatasetABC"], Type["DatasetABC"], Type["DatasetABC"]]:
        """
        Load a dataset from a json file + the number of expected timesteps
        taken as inputs (num_input_steps) and to predict (num_pred_steps)
        Return the train, valid and test datasets, in that order
        config_override is a dictionary that can be used to override
        some keys of the config file.
        """
        with open(fname, "r") as fp:
            conf = json.load(fp)
            if config_override is not None:
                conf = merge_dicts(conf, config_override)
        return cls.from_dict(
            accessor_kls,
            fname.stem,
            conf,
            num_input_steps,
            num_pred_steps_train,
            num_pred_steps_val_tests,
        )
