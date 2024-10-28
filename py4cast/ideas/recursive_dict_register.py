from typing import List

import torch


class RegisterDictMixin:
    """
    Register dictionnaries.
    Enable to recursively register dictionnary (or other object with a getitem method).
    """

    def register_dict(self, name: str, data):
        """
        Add a dict to the register

        Args:
            name (str): The register name
            data (_type_): The dictionnary like object to register.
        """
        if not hasattr(self, "_registered_dicts"):
            self._registered_dicts = {}
        if not isinstance(data, dict) and not hasattr(data, "__getitem__"):
            raise TypeError(f"Data must be a dictionary. Get {type(data)}")
        self._registered_dicts[name] = data

    def register_dict_as_buffers(self, prefix=""):
        """Register all dictionnary as buffers"""
        for name, data in self._registered_dicts.items():
            self._register_tensors_recursive(data, prefix + name + "__")

    def _register_tensors_recursive(self, data, prefix=""):
        """
        Args:
            data (dict): Dictionnary like object
        """
        for key, value in data.items():
            if isinstance(value, dict):
                self._register_tensors_recursive(value, prefix + key + "__")
            elif isinstance(value, torch.Tensor):
                self.register_buffer(prefix + key, value)

    def __getitem__(self, key: str):
        if key not in self._registered_dicts:
            raise KeyError(f"Key '{key}' not found.")
        value = self._registered_dicts[key]

        if isinstance(value, dict) or hasattr(value, "__getitem__"):
            return self._get_nested_buffers(key)
        else:
            raise TypeError("Cannot use __getitem__ on non-dictionary values.")

    def _get_nested_buffers(self, name: str):
        nested_buffers = {}
        for buffer_name, buffer_value in self._buffers.items():
            if buffer_name.startswith(name + "__"):
                split_name = buffer_name.split("__")
                d = build_nested_dict(split_name[1:], buffer_value)
                recursive_update(nested_buffers, d)  # [split_name[1]] = buffer_value
        return nested_buffers


def build_nested_dict(keys: List[str], value):
    """
    Build a dictionnary with different recursive key.

    Args:
        keys (str): List of key
        value (_type_): value to set
    Ex :
    d = build_nested_dict(["k1","k2"],"toto")
    d = {"k1":{"k2":"toto"}}
    """
    result = {}
    current_dict = result
    for key in keys[:-1]:
        current_dict.setdefault(key, {})
        current_dict = current_dict[key]
    current_dict[keys[-1]] = value
    return result


def recursive_update(old: dict, new: dict):
    """
    Recursively update input with new_dict
    Args:
        old (dict): Dictionnary to update
        new (dict): New dictionnary to integrate
    """
    for k, v in new.items():
        if isinstance(v, dict) and k in old and isinstance(old[k], dict):
            recursive_update(old[k], v)
        else:
            old[k] = v
