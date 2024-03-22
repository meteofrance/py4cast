from dataclasses import fields
from pathlib import Path

import pytorch_lightning as pl
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
