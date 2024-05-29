import importlib
import pkgutil
from pathlib import Path
from typing import Union

from py4cast.models.base import ModelABC

from .nlam.models import GraphLAM, HiLAM, HiLAMParallel
from .vision.conv import HalfUnet, Unet
from .vision.transformers import Segformer

# Models MUST be added to the registry
# in order to be used by the training script.
registry = {}
for kls in (HalfUnet, Unet, GraphLAM, HiLAM, HiLAMParallel, Segformer):
    registry[kls.__name__.lower()] = kls


PLUGIN_PREFIX = "py4cast_plugin_"
discovered_modules = {
    name: importlib.import_module(name)
    for _, name, _ in pkgutil.iter_modules()
    if name.startswith(PLUGIN_PREFIX)
}

# We now add to the registry all the models found in the plugins.
# Valid models are ModelABC subclasses contained in a module prefixed by PLUGIN_PREFIX.
# We explore all modules accessible to the Python interpreter added using pip install or PYTHONPATH.
# Inspired from: https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins
for module_name, module in discovered_modules.items():
    for name, kls in module.__dict__.items():
        if isinstance(kls, type) and issubclass(kls, ModelABC) and kls != ModelABC:
            if kls.__name__.lower() in registry:
                raise ValueError(
                    f"Model {kls.__name__} from plugin {module_name} already exists in the registry."
                )
            registry[kls.__name__.lower()] = kls


def get_model_kls_and_settings(
    model_name: str, settings_path: Union[Path, None] = None
):
    """
    Returns the classes for a model and its settings instance.
    """
    try:
        model_kls = registry[model_name]
    except KeyError as e:
        raise KeyError(
            f"Model {model_name} not found in registry of {__file__}. Did you add it ? Names are {registry.keys()}"
        ) from e
    settings_kls = model_kls.settings_kls

    if settings_path is None:
        model_settings = settings_kls()
    else:
        with open(settings_path, "r") as f:
            model_settings = settings_kls.schema().loads(f.read())
    return model_kls, model_settings


def build_model_from_settings(
    network_name: str,
    num_input_features: int,
    num_output_features: int,
    settings_path: Union[Path, None],
    *args,
    **kwargs,
):
    """
    Instanciates a model based on its name and an optional settings file.
    """
    model_kls, model_settings = get_model_kls_and_settings(network_name, settings_path)
    return (
        model_kls(
            num_input_features, num_output_features, model_settings, *args, **kwargs
        ),
        model_settings,
    )
