from dataclasses import fields

import pytorch_lightning as pl
import torch


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
