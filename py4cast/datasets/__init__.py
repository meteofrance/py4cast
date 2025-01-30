import traceback
import warnings
from typing import Dict, Tuple

from py4cast.settings import DEFAULT_CONFIG_DIR

from .base import DatasetABC  # noqa: F401

registry = {}


# we try to import and register the datasets
# with loose coupling
# missing dependencies for a single dataset should not
# break the code
# NEW ACCESSORS MUST BE REGISTERED HERE

try:
    from .titan import TitanAccessor

    registry["titan"] = (
        TitanAccessor,
        DEFAULT_CONFIG_DIR / "datasets" / "titan_refacto.json",
    )

except (ImportError, FileNotFoundError, ModuleNotFoundError):
    warnings.warn(f"Could not import TitanAccessor. {traceback.format_exc()}")

try:
    from .poesy import PoesyAccessor

    registry["poesy"] = (
        PoesyAccessor,
        DEFAULT_CONFIG_DIR / "datasets" / "poesy_refacto.json",
    )

except ImportError:
    warnings.warn(f"Could not import PoesyAccessor. {traceback.format_exc()}")

try:
    from .dummy import DummyAccessor

    registry["dummy"] = (
        DummyAccessor,
        DEFAULT_CONFIG_DIR / "datasets" / "dummy_config.json",
    )
except ImportError:
    warnings.warn(f"Could not import DummyAccessor. {traceback.format_exc()}")

try:
    from .rainfall import RainfallAccessor

    registry["rainfall"] = (
        RainfallAccessor,
        DEFAULT_CONFIG_DIR / "datasets" / "rainfall.json",
    )
except ImportError:
    warnings.warn(f"Could not import RainfallAccessor. {traceback.format_exc()}")


def get_datasets(
    name: str,
    num_input_steps: int,
    num_pred_steps_train: int,
    num_pred_steps_val_test: int,
    dataset_conf: Dict | None = None,
) -> Tuple[DatasetABC, DatasetABC, DatasetABC]:
    """
    Lookup dataset by name in our registry and uses either
    the specified config file or the default one.

    Returns 3 instances of the dataset: train, val, test
    """

    # checks if name has a registry key as component (substring)
    registered_name = ""
    for k in registry:
        if k in name.lower():
            registered_name = k
            break
    try:
        accessor_kls, _ = registry[registered_name]
    except KeyError as ke:
        raise ValueError(
            f"Dataset {name} doesn't match a registry substring, available datasets are :{registry.keys()}"
        ) from ke

    return DatasetABC.from_dict(
        accessor_kls,
        name,
        dataset_conf,
        num_input_steps,
        num_pred_steps_train,
        num_pred_steps_val_test,
    )
