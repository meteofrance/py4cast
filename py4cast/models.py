"""
Python module to plug mfai models into py4cast
and also register any plugin found in the PYTHONPATH.
"""

import importlib
import pkgutil
from typing import Any, Tuple

from mfai.pytorch.models import registry as mfai_registry
from mfai.pytorch.models.base import ModelABC

# Models MUST be added to the registry
# in order to be used by the training script.
# We init the registry with the models from mfai.
registry = {}
registry.update(mfai_registry)


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
        if (
            isinstance(kls, type)
            and issubclass(kls, ModelABC)
            and kls != ModelABC
            and kls.register
        ):
            if kls.__name__ in registry:
                raise ValueError(
                    f"Model {kls.__name__} from plugin {module_name} already exists in the registry."
                )
            registry[kls.__name__] = kls
all_nn_architectures = list(registry)


def get_model_kls_and_settings(model_name: str, settings: dict):
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

    model_settings = settings_kls(**settings)
    return model_kls, model_settings


def build_model_from_settings(
    network_name: str,
    num_input_features: int,
    num_output_features: int,
    settings: dict,
    input_shape: tuple,
    *args,
    **kwargs,
) -> Tuple[ModelABC, Any]:
    """
    Instanciates a model based on its name and an optional settings file.
    """
    model_kls, model_settings = get_model_kls_and_settings(network_name, settings)
    return (
        model_kls(
            num_input_features,
            num_output_features,
            input_shape,
            model_settings,
            *args,
            **kwargs,
        ),
        model_settings,
    )
