from dataclasses import fields
from pathlib import Path
from typing import Dict

import lightning.pytorch as pl
import torch


def nullable_string(val: str):
    if val == "None":
        return None
    return val


def torch_save(data, path: Path):
    """Saving files with torch to be writeable by anyone"""
    if path.exists():
        path.unlink()
    torch.save(data, path)
    path.chmod(0o666)


def torch_load(path: Path, device: str):
    return torch.load(path, map_location=device)


class RegisterFieldsMixin:
    """
    Mixin class to register
    a dataclass fields
    as a lightning buffer.
    See https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html
    """

    def register_buffers(self, lm: type[pl.LightningModule], persistent: bool = False):
        """
        Register the dataclass fields which are torch.Tensor as lightning buffers.
        """
        for field in fields(self):
            field_instance = getattr(self, field.name)
            if isinstance(field_instance, torch.Tensor):
                lm.register_buffer(field.name, field_instance, persistent=persistent)


def merge_dicts(d1: Dict, d2: Dict) -> Dict:
    """
    Recursively merge two nested dictionaries.
    """
    for key in d2:
        if key in d1 and isinstance(d1[key], dict) and isinstance(d2[key], dict):
            merge_dicts(d1[key], d2[key])
        else:
            d1[key] = d2[key]
    return d1


str_to_dtype = {
    "bf16-true": torch.bfloat16,
    "16-true": torch.float16,
    "32-true": torch.float32,
    "64-true": torch.float64,
}
