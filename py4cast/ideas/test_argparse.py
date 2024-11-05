import getpass
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Tuple, Union

import yaml

DATA_SPLIT = {
    "train": {"start": datetime(2023, 3, 1, 6), "end": datetime(2023, 3, 9, 18)},
    "valid": {"start": datetime(2023, 3, 10, 6), "end": datetime(2023, 3, 19, 18)},
    "test": {"start": datetime(2023, 3, 20, 6), "end": datetime(2023, 3, 31, 18)},
}


@dataclass
class WeatherParam:
    name: str
    long_name: str
    param: str
    model: str
    prefix_model: str
    unit: str
    cumulative: bool
    type_level: str
    levels: Tuple[int]
    grib: str
    grid: str
    shape: Tuple[int]
    extend: Tuple[float]


@dataclass
class Split:
    split_name: Literal["train", "valid", "test"] = "train"
    date_start: datetime = DATA_SPLIT["train"]["start"]
    date_end: datetime = DATA_SPLIT["train"]["end"]
    shuffle: bool = True

    def __post_init__(self):
        self.date_start = DATA_SPLIT[self.split_name]["start"]
        self.date_end = DATA_SPLIT[self.split_name]["end"]
        self.shuffle = self.split_name == "train"


@dataclass
class Grid:
    border_size: int = 10
    # Subgrid selection. If (0,0,0,0) the whole grid is kept.
    subgrid: Tuple[int] = (0, 0, 0, 0)


@dataclass
class TrainingHyperParams:
    experiment_name: str = field(
        default="exp0", metadata=dict(help="Name of folder that regroups runs.")
    )
    run_name: str = field(default="my_run", metadata=dict(help="Name of the run."))
    username: str = getpass.getuser()
    date: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_size: int = 4
    num_workers: int = 2
    learning_rate: float = 0.001
    max_epochs: int = 5


@dataclass
class TitanHyperParams:
    path: Path = Path("/scratch/shared/Titan/")
    weather_params: Tuple[Union[str, WeatherParam]] = ("aro_t2m", "aro_r2")
    isobaric_levels: Tuple[int] = (1000, 850)  # hPa
    timestep: int = 1  # hours
    nb_input_steps: int = 2
    nb_pred_steps: int = 4
    step_btw_samples: int = 6  # hours

    def __post_init__(self):
        with open(self.path / "metadata.yaml", "r") as file:
            self.metadata = yaml.safe_load(file)
        params = []
        for param in self.weather_params:
            if isinstance(param, str):
                wp = WeatherParam(**self.metadata["WEATHER_PARAMS"][param])
                if wp.grid != "PAAROME_1S100":
                    raise NotImplementedError(
                        "Can't load Arpege or Antilope data for now"
                    )
                params.append(wp)
            else:
                params.append(param)
        self.weather_params = tuple(params)


class MergeDataclassMixin:
    """
    A mixin to create one attribute per inherited
    dataclass with the field populated based
    on name matching
    Always inherit FIRST from this Mixin (and other mixins)

    @dataclass
    class(MergeDataclassMixin, Dataclass1, Dataclass2, ...):
        pass

    """

    def __post_init__(self):
        super().__post_init__()
        list_vars = []
        for parent in self.__class__.__bases__:
            if is_dataclass(parent):
                list_vars += [f.name for f in fields(parent)]
                attr_name = str(parent.__name__).lower()
                attrs = {f.name: getattr(self, f.name) for f in fields(parent)}
                setattr(self, attr_name, parent(**attrs))

        duplicate_fields = set([x for x in list_vars if list_vars.count(x) > 1])
        if len(duplicate_fields) != 0:
            raise AttributeError(
                f"Duplicate fields (same names) in inherited dataclasses : {duplicate_fields}"
            )


@dataclass
class CLIArgs(MergeDataclassMixin, Split, Grid, TitanHyperParams, TrainingHyperParams):
    pass


if __name__ == "__main__":
    from mfai.argparse_dataclass import MfaiArgumentParser

    print("MISSING : ", MISSING)
    parser = MfaiArgumentParser(CLIArgs)
    params, _ = parser.parse_known_args()

    print("params : ", params)
    print("params.grid : ", params.grid)
    print("params.split : ", params.split)
    print("params.titanhyperparams : ", params.titanhyperparams)
    print("params.traininghyperparams : ", params.traininghyperparams)

    # OUTPUT

    # runai python test_argparse.py --split-name valid
    # /scratch/labia/berthomierl/.cache/torch
    # /scratch/labia/berthomierl/.cache/pip
    # --> PARSE ARGS
    # params :  CLIArgs(experiment_name='exp0', run_name='my_run', username='berthomierl',
    #                   date='2024-01-18_12-29-03', batch_size=4, num_workers=2,
    #                   learning_rate=0.001, max_epochs=5,
    #                   path=PosixPath('/scratch/shared/Titan'),
    #                   weather_params=('aro_t2m', 'aro_r2'),
    #                   isobaric_levels=(1000, 850), timestep=1, nb_input_steps=2,
    #                   nb_pred_steps=4, step_btw_samples=6, border_size=10,
    #                   subgrid=(0, 0, 0, 0), split_name='valid',
    #                   date_start=datetime.datetime(2023, 3, 1, 6, 0),
    #                   date_end=datetime.datetime(2023, 3, 9, 18, 0), shuffle=True)
    # params.grid :  Grid(border_size=10, subgrid=(0, 0, 0, 0))
    # params.split :  Split(split_name='valid', date_start=datetime.datetime(2023, 3, 10, 6, 0),
    #                       date_end=datetime.datetime(2023, 3, 19, 18, 0), shuffle=False)
    # params.titanhparams :  TitanHyperParams(path=PosixPath('/scratch/shared/Titan'), ...
    # params.traininghparams :  TrainingHyperParams(experiment_name='exp0', run_name='my_run',
    #                              username='berthomierl', date='2024-01-18_12-29-03',
    #                              batch_size=4, num_workers=2, learning_rate=0.001, max_epochs=5)
