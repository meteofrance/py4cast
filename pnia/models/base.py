from dataclasses import dataclass


@dataclass(slots=True)
class ModelInfo:
    """
    Information specific to a model
    """

    output_dim: int  # Spatial dimension of the output
