"""
This script demonstrates how memory leaks
occur when doing  standardisation
of Torch tensor on CPU.
Using numpy seems to work fine.
"""

import gc
import json
import os
import subprocess
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
import typer
import xarray as xr

app = typer.Typer()
res_dir = Path("/scratch/shared/py4cast/cache/tmp")
res_dir.mkdir(parents=True, exist_ok=True)


def current_memory_usage():
    """Returns current memory usage (in MB) of a current process"""
    out = (
        subprocess.Popen(
            ["ps", "-p", str(os.getpid()), "-o", "rss"], stdout=subprocess.PIPE
        )
        .communicate()[0]
        .split(b"\n")
    )
    mem = float(out[1].strip()) / 1024
    return mem  # , str(os.getpid())


def json_write(file, list):
    with open(res_dir / file, "w") as fp:
        json.dump(list, fp)


@dataclass
class Sample:
    # Describe a sample
    member: int

    @cached_property
    def size(self):
        return self.member

    def open_ds(self):
        return xr.open_dataset(
            "/scratch/shared/smeagol/nc/france/arome_franmgsp32/20221012H00/mb_001_surface.nc",
            mask_and_scale=False,
            decode_cf=False,
            decode_times=False,
            cache=False,
        )


class dataloader:
    def __init__(self, size):
        self.size = size
        self.shape = 100

    @cached_property
    def sample_list(self):
        samples = []
        for i in range(self.size):
            samples.append(
                Sample(
                    member=np.random.randint(1, 7),
                )
            )
        return samples


legend = {
    "1": "Basic",
    "2": "No xarray",
    "3": "Only xarray (no torch)",
    "5": "Start with xarray",
    "6": "No normalisation",
    "7": "Numpy",
    "8": "+Mean",
    "10": "Negative mean before going to torch",
    "4": "Opening context (but no as)",
    "9": "Opening context (with as) ",
}


@app.command()
def test1(samples: int, pmem: bool = True):
    """
    Open a netcdf. No reading
    """
    loader = dataloader(samples)
    lmem = []
    for i in range(loader.size):
        mem = current_memory_usage()
        lmem.append(mem)
        if pmem:
            print(f"\t {mem}")
        sample = loader.sample_list[i]
        ds = sample.open_ds()
        mean = torch.from_numpy(np.random.randn(1) * np.ones((sample.size)))
        tmp = np.random.randn(3, sample.size, loader.shape, loader.shape)
        tmp = torch.from_numpy(tmp).permute([0, 2, 3, 1])
        tmp = tmp - mean
        ds.close()
        gc.collect()
    json_write("file1.json", lmem)
    return tmp


@app.command()
def test2(samples: int, pmem: bool = True):
    """
    No xarray opening
    """
    loader = dataloader(samples)
    lmem = []
    print("Test2")
    for i in range(loader.size):
        mem = current_memory_usage()
        lmem.append(mem)
        if pmem:
            print(f"\t {mem}")
        sample = loader.sample_list[i]
        mean = torch.from_numpy(np.random.randn(1) * np.ones((sample.size)))
        tmp = np.random.randn(3, sample.size, loader.shape, loader.shape)
        tmp = torch.from_numpy(tmp).permute([0, 2, 3, 1])
        tmp = tmp - mean
        gc.collect()
    json_write("file2.json", lmem)
    return tmp


@app.command()
def test3(samples: int, pmem: bool = True):
    """
    Only xarray opening
    """

    loader = dataloader(samples)
    lmem = []
    print("Test3")
    for i in range(loader.size):
        mem = current_memory_usage()
        lmem.append(mem)
        if pmem:
            print(f"\t {mem}")
        sample = loader.sample_list[i]
        ds = sample.open_ds()
        ds.close()
        gc.collect()
    json_write("file3.json", lmem)
    return None


@app.command()
def test5(samples: int, pmem: bool = True):
    """
    Xarray opening at the begining of the loop
    """
    loader = dataloader(samples)
    print("Test5")
    lmem = []
    for i in range(loader.size):
        mem = current_memory_usage()
        lmem.append(mem)
        if pmem:
            print(f"\t {mem}")
        sample = loader.sample_list[i]
        ds = sample.open_ds()
        ds.close()
        mean = torch.from_numpy(np.random.randn(1) * np.ones((sample.size)))
        tmp = np.random.randn(3, sample.size, loader.shape, loader.shape)
        tmp = torch.from_numpy(tmp).permute([0, 2, 3, 1])
        tmp = tmp - mean
        gc.collect()
    json_write("file5.json", lmem)
    return tmp


@app.command()
def test6(samples: int, pmem: bool = True):
    """
    Open a netcdf. No normalisation
    """
    loader = dataloader(samples)
    print("Test6")

    lmem = []
    for i in range(loader.size):
        mem = current_memory_usage()
        lmem.append(mem)
        if pmem:
            print(f"\t {mem}")
        sample = loader.sample_list[i]
        ds = sample.open_ds()
        tmp = np.random.randn(3, sample.size, loader.shape, loader.shape)
        tmp = torch.from_numpy(tmp).permute([0, 2, 3, 1])
        ds.close()
        gc.collect()
    json_write("file6.json", lmem)
    return tmp


@app.command()
def test7(samples: int, pmem: bool = True):
    """
    Like 1. Normalisation in numpy and then convert to torch.
    """
    loader = dataloader(samples)
    lmem = []
    print("Test7")
    for i in range(loader.size):
        mem = current_memory_usage()
        lmem.append(mem)
        if pmem:
            print(f"\t {mem}")
        sample = loader.sample_list[i]
        ds = sample.open_ds()
        mean = np.random.randn(1) * np.ones((sample.size))
        tmp = np.random.randn(3, sample.size, loader.shape, loader.shape)
        tmp = np.transpose(tmp, axes=[0, 2, 3, 1])
        tmp = tmp - mean
        tmp = torch.from_numpy(tmp)
        ds.close()
        gc.collect()
    json_write("file7.json", lmem)
    return tmp


@app.command()
def test8(samples: int, pmem: bool = True):
    """
    Open a netcdf. No reading
    """
    loader = dataloader(samples)
    lmem = []
    for i in range(loader.size):
        mem = current_memory_usage()
        lmem.append(mem)
        if pmem:
            print(f"\t {mem}")
        sample = loader.sample_list[i]
        ds = sample.open_ds()
        mean = torch.from_numpy(np.random.randn(1) * np.ones((sample.size)))
        tmp = np.random.randn(3, sample.size, loader.shape, loader.shape)
        tmp = torch.from_numpy(tmp).permute([0, 2, 3, 1])
        _ = tmp + mean
        ds.close()
        gc.collect()
    json_write("file8.json", lmem)
    return tmp


@app.command()
def test10(samples: int, pmem: bool = True):
    """
    Open a netcdf. No reading
    """
    loader = dataloader(samples)
    lmem = []
    for i in range(loader.size):
        mem = current_memory_usage()
        lmem.append(mem)
        if pmem:
            print(f"\t {mem}")
        sample = loader.sample_list[i]
        ds = sample.open_ds()
        mean = torch.from_numpy(-np.random.randn(1) * np.ones((sample.size)))
        tmp = np.random.randn(3, sample.size, loader.shape, loader.shape)
        tmp = torch.from_numpy(tmp).permute([0, 2, 3, 1])
        _ = tmp + mean
        ds.close()
        gc.collect()
    json_write("file10.json", lmem)
    return tmp


@app.command()
def test4(samples: int, pmem: bool = True):
    """
    No samples
    """
    loader = dataloader(samples)
    lmem = []
    for i in range(loader.size):
        mem = current_memory_usage()
        lmem.append(mem)
        if pmem:
            print(f"\t {mem}")

        with xr.open_dataset(
            "/scratch/shared/smeagol/nc/france/arome_franmgsp32/20221012H00/mb_001_surface.nc"
        ):
            mean = torch.from_numpy(np.random.randn(1) * np.ones((5)))
            tmp = np.random.randn(3, 5, loader.shape, loader.shape)
            tmp = torch.from_numpy(tmp).permute([0, 2, 3, 1])
            _ = tmp - mean

        gc.collect()
    json_write("file4.json", lmem)
    return tmp


@app.command()
def test9(samples: int, pmem: bool = True):
    """
    In context
    """
    loader = dataloader(samples)
    lmem = []
    for i in range(loader.size):
        mem = current_memory_usage()
        lmem.append(mem)
        if pmem:
            print(f"\t {mem}")

        with xr.open_dataset(
            "/scratch/shared/smeagol/nc/france/arome_franmgsp32/20221012H00/mb_001_surface.nc"
        ) as ds:
            mean = torch.from_numpy(np.random.randn(1) * np.ones((5)))
            tmp = np.random.randn(3, 5, loader.shape, loader.shape)
            tmp = torch.from_numpy(tmp).permute([0, 2, 3, 1])
            _ = tmp - mean
            ds.close()
        gc.collect()
    json_write("file9.json", lmem)
    return tmp


@app.command()
def plotting(nbtest: int):
    import matplotlib.pyplot as plt

    for i in range(1, nbtest + 1):
        data = {}
        with open(res_dir / f"file{i}.json") as fp:
            data["i"] = json.load(fp)
            print(type(data))
        plt.plot(np.asarray(data["i"])[2:] - data["i"][1], label=legend[f"{i}"])
    plt.xlabel("Iteration")
    plt.ylabel("Supplementary memory")
    plt.yscale("log")
    plt.legend()
    plt.savefig(res_dir / "plot.png")


if __name__ == "__main__":
    app()
