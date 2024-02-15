from pathlib import Path

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
