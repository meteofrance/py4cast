from pathlib import Path
from typing import Union

from .nlam.models import GraphLAM, HiLAM, HiLAMParallel
from .vision.conv import HalfUnet
from .vision.transformers import Segformer

# Models MUST be added to the registry
# in order to be used by the training script.
registry = {}
registry["halfunet"] = HalfUnet
registry["graphlam"] = GraphLAM
registry["hilam"] = HiLAM
registry["hilamparallel"] = HiLAMParallel
registry["segformer"] = Segformer


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
    no_input_features: int,
    no_output_features: int,
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
            no_input_features, no_output_features, model_settings, *args, **kwargs
        ),
        model_settings,
    )
