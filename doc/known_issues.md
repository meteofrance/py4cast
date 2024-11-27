# Known issues

## Installation

### Onnxruntime > 1.17 with GLIBC < 2.28

#### Issue

Since the [version 1.17](https://github.com/microsoft/onnxruntime/releases/tag/v1.17.0), `onnxruntime` only support `GLIBC >= 2.28`, but some old OS release (e.g. CentOS7, RHEL7) could only embed `GLIBC < 2.28`.

In this case, installing an environment with conda and `onnxruntime >= 1.17`, as it is documented in [Install with conda](../README.md#install-with-conda), will fail with the following `pip` error :

```
ERROR: Could not find a version that satisfies the requirement onnxruntime==1.18.0 (from versions: 1.12.0, 1.12.1, 1.13.1, 1.14.0, 1.14.1, 1.15.0, 1.15.1, 1.16.0, 1.16.1, 1.16.2, 1.16.3)
ERROR: No matching distribution found for onnxruntime==1.18.0
```

Pip detect the conflict between the the `GLIBC` and `onnxruntime`, but is not able to resolve it.

#### Workaroud

The first alternative is to use the an [containerizer environment](../README.md#build-docker-image) (docker/podman/singularity). But, in case no one of these solution can be used, conda can resolve the issue.

Conda doen't use the pypi `onnxruntime` wheels, while it publishes its own packages. The conda `onnxruntime` libs are statically linked (in contrary to the pypi ones) and that seems to solve the issue (maybe they compile in their own for a glibc 2.27, and/or they embed many stuff that solves the issue ...).

It is then recommended to manually edit the `requirements.txt` file to move the `onnxruntime` dependency to the `env.yaml` file, such that conda install it instead of pip:

```
name: py4cast
channels:
  - pytorch
  - nvidia
dependencies:
  - python=3.11.9           # The reference python version for the project
  - pytorch-cuda=12.1       # The reference cuda version for the project
  - pytorch==2.4.1          # The reference pytorch version for the project
  - onnxruntime-gpu==1.19.2
  - pip
  - pip:
     - -r requirements.txt
     - -e .                 # install py4cast in editable mode
```
