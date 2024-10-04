import traceback
import warnings
from pathlib import Path
from typing import Dict, Tuple, Union

from .base import DatasetABC  # noqa: F401

registry = {}


# we try to import and register the datasets
# with loose coupling
# missing dependencies for a dataset should not
# break the code
# NEW DATASETS MUST BE REGISTERED HERE


default_config_root = Path(__file__).parents[2] / "config/datasets/"


try:
    from .smeagol import SmeagolDataset

    registry["smeagol"] = (SmeagolDataset, default_config_root / "smeagol.json")
except ImportError:
    warnings.warn(f"Could not import SmeagolDataset. {traceback.format_exc()}")

try:
    from .smeagol import InferSmeagolDataset

    registry["smeagol_infer"] = (
        InferSmeagolDataset,
        default_config_root / "smeagol.json",
    )
except ImportError:
    warnings.warn(f"Could not import SmeagolDataset. {traceback.format_exc()}")


try:
    from .titan import TitanDataset

    registry["titan"] = (TitanDataset, default_config_root / "titan.json")
except (ImportError, FileNotFoundError, ModuleNotFoundError):
    warnings.warn(f"Could not import TitanDataset. {traceback.format_exc()}")

try:
    from .poesy import PoesyDataset

    registry["poesy"] = (PoesyDataset, default_config_root / "poesy.json")
except ImportError:
    warnings.warn(f"Could not import PoesyDataset. {traceback.format_exc()}")

try:
    from .poesy import InferPoesyDataset

    registry["poesy_infer"] = (
        InferPoesyDataset,
        default_config_root / "poesy_infer.json",
    )
except ImportError:
    warnings.warn(f"Could not import InferPoesyDataset. {traceback.format_exc()}")

try:
    from .dummy import DummyDataset

    registry["dummy"] = (DummyDataset, "")
except ImportError:
    warnings.warn(f"Could not import DummyDataset. {traceback.format_exc()}")


def get_datasets(
    name: str,
    num_input_steps: int,
    num_pred_steps_train: int,
    num_pred_steps_val_test: int,
    config_file: Union[Path, None] = None,
    config_override: Union[Dict, None] = None,
) -> Tuple[DatasetABC, DatasetABC, DatasetABC]:
    """
    Lookup dataset by name in our registry and uses either
    the specified config file or the default one.

    Returns 3 instances of the dataset: train, val, test
    """
    try:
        dataset_kls, default_config = registry[name]
    except KeyError as ke:
        raise ValueError(
            f"Dataset {name} not found in registry, available datasets are :{registry.keys()}"
        ) from ke

    config_file = default_config if config_file is None else config_file

    return dataset_kls.from_json(
        config_file,
        num_input_steps,
        num_pred_steps_train,
        num_pred_steps_val_test,
        config_override,
    )
