# PY4CAST

This project built using **PyTorch** and **PyTorch-lightning** is designed to train a variety of Neural Network architectures (GNNs, CNNs, Vision Transformers, ...) on weather forecasting datasets. This is a **Work in Progress**, intented to share ideas and design concepts with partners.

Developped at Météo-France by **DSM/Lab IA** and **CNRM/GMAP/PREV**.

Contributions are welcome.

This project is licensed under the [APACHE 2.0 license.](LICENSE-2.0.txt)

# Acknowledgements

This project started as a fork of neural-lam, a project by Joel Oskarsson, see [here](https://github.com/mllam/neural-lam). Thanks to Joel for his work.

# Table of contents

1. [Environnment variables](#Setting-environment-variables)
2. [Running at MF](#running-using-runai-météo-france-internal)
3. [Running using Conda and friends](#running-without-runai )
    1. [Micromamba](#running-using-micromamba)
    2. [Conda](#running-using-micromamba)
    3. [Specifying your sbatch card](#specifying-your-sbatch-card)
4. [Available PyTorch's architecture](#available-pytorchs-architecture)
    1. [Adding a new neural network architecture to the project.](#adding-a-new-neural-network-architecture-to-the-project)
5. [Available Training strategies](#available-training-strategies)
6. [NamedTensors](#namedtensors)
7. [Plots](#plots)
8. [Experiment tracking](#experiment-tracking)
9. [Unit tests](#unit-tests)
10. [Continuous Integration](#continuous-integration)
11. [TLDR design choices](#tldr-design-choices)


## Setting environment variables

In order to be able to run the code on different machines, some environment variables can be set.
You may add them in your `.bashrc` or modify them just before launching an experiment.

- `PY4CAST_ROOTDIR` : Specify the ROOT DIR for your experiment. It also modifies the CACHE_DIR.
- `PY4CAST_SMEAGOL_PATH`: Specify where the smeagol dataset is stored. Only needed if you want to use the smeagol dataset. **Should be moved in the configuration file**.
- `PY4CAST_TITAN_PATH`: Specify where the titan dataset is stored. Only needed if you want to use the titan dataset.
If you plan to use micromamba or conda you should also add `py4cast` to your **PYTHONPATH** by expanding it (Export or change your `PYTHONPATH`).

## Running using **runai** (Météo-France internal)

**runai** is our docker wrapper for training neural networks. See our [repository](https://git.meteo.fr/dsm-labia/monorepo4ai) for installation instructions.

`runai` commands must be issued at the root directory of the `py4cast` project:

1. Run an interactive training session

```bash
runai gpu_play 4
runai build
runai exec_gpu python bin/train.py --dataset smeagol --model hilam
```

2. Train using sbatch single node multi-GPUs

```bash
export RUNAI_GRES="gpu:v100:4"
runai sbatch python bin/train.py --dataset smeagol --model hilam
```

3. Train using sbatch multi nodes multi GPUs

Here we use 2 nodes with 4 GPUs each.

```bash
export RUNAI_SLURM_NNODES=2
export RUNAI_SLURM_NTASKS_PER_NODE=4
export RUNAI_GRES="gpu:v100:4"
runai sbatch_multi_node python bin/train.py --dataset smeagol --model hilam
```

You can find more details about our **train.py** [here](./bin/Readme.md)


## Running without runai
### Running using micromamba
Please install the environment using :
```sh
micromamba create -f env.yaml
```
Then run your classic srun command to get an interactive shell on gpu :
```sh
srun --job-name="MySuperJob" --cpus-per-task=15 --partition=node3 --gres=gpu:v100:1 --time=10:00:00 --ntasks-per-node=1 --pty bash
```
Activate your environnment :
```sh
micromamba activate nlam
```

Go to pnia directory and do :
```sh
export PYTHONPATH=`pwd`
```
You can also add directly the directory in your `.bashrc`.

This is done in order to register pnia package in the python path.
Then you can do a classical training such as
```sh
python3.10 bin/train.py  --dataset smeagol --model halfunet --dataset_conf config/datasets/smeagoldev.json --limit_train_batches 10 --epochs 2 --strategy scaled_ar
```
### Running using conda
You can install a conda environment using
```sh
conda env create --file env_conda.yaml
```
Note that this environment is a bit different from the micromamba one. Both should be merged in a near future.

### Specifying your sbatch card
To do so, you will need a small `sh` script.

```sh
#!/usr/bin/bash
#SBATCH --partition=ndl
#SBATCH --nodes=1 # Specify the number of GPU node you required
#SBATCH --gres=gpu:1 # Specify the number of GPU required per Node
#SBATCH --time=05:00:00 # Specify your experiment Time limit
#SBATCH --ntasks-per-node=1 # Specify the number of task per node. This should match the number of GPU Required per Node

# Note that other variable could be set (according to you machine). For example you may need to set the number of CPU or the memory used by your experiment.
# On our hpc, this is proportional to the number of GPU required per node. This is not the case on other machine (e.g MétéoFrance AILab machine).

source ~/.bashrc  # Be sure that all your environment variables are set
conda activate py4cast # Activate your environment (installed by micromamba or conda)
cd $PY4CAST_PATH # Go to Py4CAST (you can either add an environment variable or hard code it here)
# Launch your favorite command.
srun python bin/train.py --model halfunet --dataset smeagol --dataset_conf config/smeagolsmall.json --num_pred_steps_val_test 4 --strategy scaled_ar --num_inter_steps 4 --num_input_steps 1 --batch_size 10
```
Then just launch this script using
```sh
sbatch my_tiny_script.sh
```


## Available datasets

| Dataset  | Domain  | Description    | Documentation  | Maintainer(s) |
| :---:   | :---: | :---: | :---: | :---: |
| titan | France | AROME Analyses + ARPEGE Analyses and forecasts + 1h Rainfall; Timestep 1h; 2022-2023; [download link](https://huggingface.co/datasets/meteofrance/titan)  | [link](pnia/datasets/titan/README.md) | Léa Berthomier |
| smeagol | France | WIP  | WIP |  Vincent Chabot |
| dummy | WIP  | WIP | WIP |  WIP |

### Adding a new dataset
A dataset should expose a few methods to be used in py4cast.
It should have


## Available PyTorch's architecture

Currently we support the following neural network architectures:

| Model  | Research Paper  | Input Shape    | Notes  | Maintainer(s) |
| :---:   | :---: | :---: | :---: | :---: |
| halfunet | https://www.researchgate.net/publication/361186968_Half-UNet_A_Simplified_U-Net_Architecture_for_Medical_Image_Segmentation | (Batch, Height, Width, features)   | In prod/oper on espresso v2 with 128 filters and standard conv blocks instead of ghost |  Frank Guibert |
| unet | https://arxiv.org/pdf/1505.04597.pdf| (Batch, Height, Width, features)   | Vanilla U-Net |  Sara Akodad / Frank Guibert |
| segformer | https://arxiv.org/abs/2105.15203   | (Batch, Height, Width, features) | on par with u-net like on deepsyg, added an upsampling stage. Adapted from [Lucidrains' github](https://github.com/lucidrains/segformer-pytorch) |  Frank Guibert |
| hilam, graphlam | https://arxiv.org/abs/2309.17370  | (Batch, graph_node_id, features)   | Imported and adapted from [Joel's github](https://github.com/joeloskarsson/neural-lam) |  Vincent Chabot/Frank Guibert |

To train on a dataset using a network with its default settings just pass the name of the architecture (all lowercase) as shown below:

```bash
runai exec_gpu python bin/train.py --dataset smeagol --model hilam

runai exec_gpu python bin/train.py --dataset smeagol --model halfunet

runai exec_gpu python bin/train.py --dataset smeagol --model segformer
```

You can override some settings of the model using a json config file (here we increase the number of filter to 128 and use ghost modules):

```bash
runai exec_gpu python bin/train.py --dataset smeagol --model halfunet --model_conf config/models/halfunet128_ghost.json
```

You can also override the dataset default configuration file:

```bash
runai exec_gpu python bin/train.py --dataset smeagol --model halfunet --dataset_conf config/datasets/smeagol.json
```

The figure below illustrates the principal components of the Py4cast architecture.

![py4cast](doc/figs/py4cast_diag.jpg)

### Adding a new neural network architecture to the project.

1. Neural network architectures MUST be Python classes and inherit from both **ModelABC** and  **nn.Module**, in that order.

```python
class NewModel(ModelABC, nn.Module):
    settings_kls = NewModelSettings
    onnx_supported = True
```

2. The **onnx_supported** attribute MUST be present and set to **True** if the architecture is onnx exportable, see [our unit tests](tests/test_models.py).

3. The **settings_kls** attribute MUST be present and set to the **dataclass_json** setting class of the architecture.

```python
@dataclass_json
@dataclass(slots=True)
class NewModelSettings:
    num_filters: int = 64
    dilation: int = 1
    bias: bool = False
    use_ghost: bool = False
    last_activation: str = "Identity"
```

4. The constructor of the architecture MUST have the following signature:

```python
    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        settings: HalfUnetSettings,
        *args,
        **kwargs,
    ):
```
**num_input_features** is the number of features/channels of the input tensor. **num_output_features** is the number of features/channels of the output tensor, in our case the number of weather parameters to predict. **settings** is a **dataclass** instance with the settings of the model.

5. Finally, the **ModelABC** is a Python **ABC** which forces you to implement (or be Exceptioned upon) 2 methods **transform_statics** and **transform_batch**.

Below is a default working implementation for CNNs and VTs (no transformation):

```python
def transform_statics(self, statics: Statics) -> Statics:
    """
    Statics are used 'as-is' by vision models.
    """
    return statics

def transform_batch(self, batch: ItemBatch) -> ItemBatch:
    """
    Batch are used 'as-is' by vision models.
    """
    return batch
```

For GNNs we need to flatten the width and height dimensions to graph_node_ids, since our datasets produce gridded shape data.

```python

def transform_statics(self, statics: Statics) -> Statics:
    """
    Take the statics in inputs.
    Return the statics as expected by the model.
    """
    statics.grid_static_features.flatten_("ngrid", 0, 1)
    statics.border_mask = statics.border_mask.flatten(0, 1)
    statics.interior_mask = statics.interior_mask.flatten(0, 1)
    return statics

def transform_batch(self, batch: ItemBatch) -> ItemBatch:
    """
    Transform the batch for our GNNS
    Our grided datasets produce tensor of shape (B, T, W, H, F)
    so we flatten (W,H) => (num_graph_nodes) for GNNs
    """
    batch.inputs.flatten_("ngrid", 2, 3)
    batch.outputs.flatten_("ngrid", 2, 3)
    batch.forcing.flatten_("ngrid", 2, 3)

    return batch
```

Now your model can be either registered explicitely in the system (in case the code is in this repository) or injected in the system as a plugin (in case the code is hosted on a third party repository).

1. Model in the same git repository

Add your **NewModel** class to the registry explicitly in the models package [__init__.py](py4cast/models/__init__.py)

```python
registry = {}
for kls in (HalfUnet, Unet, GraphLAM, HiLAM, HiLAMParallel, Segformer, NewModel):
    registry[kls.__name__.lower()] = kls
```

2. Model as a third party plugin

In order to be discovered, your model Python class MUST:

* be contained in a python module prefixed with **py4cast_plugin_**
* inherit from **ModelABC** and **nn.Module**
* have a different name than the models already present in the system
* be discoverable by the system (e.g. in the PYTHONPATH or pip installed)

We provide an example module [here](py4cast_plugin_example.py) to help you create your own plugin. This approach is based on the [official python packaging guidelines](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/).

## Available Training strategies

You can choose a training strategy using the **--strategy STRATEGY_NAME** cli argument. The strategy determines how the next timestep is computed
in the forward pass. **x** are the neural network inputs and **model(x)** is the returned value by the neural network when fed **x** as input.

| Strategy Name | Reference | Update Rule | Boundary forcing |  Intermediary Steps |
| :---:   | :---: | :---: | :---: | :---: |
| scaled_ar |  | next_state = previous_state + model(x)*diff_std + diff_mean | y_true  | Yes |
|  diff_ar | | next_state = previous_state + model(x) | No |  No |

An exemple to use the **diff_ar** strategy:

```bash
runai exec_gpu python bin/train.py --dataset smeagol --model halfunet --strategy diff_ar
```

## NamedTensors

PyTorch provides an experimental feature called [**named tensors**](https://pytorch.org/docs/stable/named_tensor.html), at this time it is subject to change so we don't use it. That's why we provide our own implementation.

NamedTensors are a way to give names to dimensions of tensors and to keep track of the names of the physical/weather parameters along the features dimension.

The **NamedTensor** class is a wrapper around a PyTorch tensor, it allows us to pass consistent object linking data and metadata with extra utility methods (concat along features dimension, flatten in place, ...). See the implementation [here](py4cast/datasets/base.py) and usage for plots [here](py4cast/observer.py)

Some examples of NamedTensors usage, here for gridded data on a 256x256 grid:

```python

tensor = torch.rand(4, 256, 256, 3)

nt = NamedTensor(
    tensor,
    names=["batch", "lat", "lon", "features"],
    feature_names=["u", "v", "t2m"],
)

print(nt.dim_size("lat"))
# 256

nt2 = NamedTensor(
    torch.rand(4, 256, 256, 1),
    names=["batch", "lat", "lon", "features"],
    feature_names=["q"],
)

# concat along the features dimension
nt3 = nt | nt2

# index by feature name
nt3["u"]

# Create a new NamedTensor with the same names but different data (useful for autoregressive models)
nt4 = NamedTensor.new_like(torch.rand(4, 256, 256, 4), nt3)

# Flatten in place the lat and lon dimensions and rename the new dim to 'ngrid'
# this is typically to feed our gridded data to GNNs
nt3.flatten_("ngrid", 1, 2)

# str representation of the NamedTensor yields useful statistics
>>> print(nt)
--- NamedTensor ---
Names: ['batch', 'lat', 'lon', 'features']
Tensor Shape: torch.Size([4, 256, 256, 3]))
Features:
┌────────────────┬─────────────┬──────────┐
│ Feature name   │         Min │      Max │
├────────────────┼─────────────┼──────────┤
│ u              │ 1.3113e-06  │ 0.999996 │
│ v              │ 8.9407e-07  │ 0.999997 │
│ t2m            │ 5.06639e-06 │ 0.999995 │

```

## Plots

Plots are done using the **matplotlib** library. We wrap each plot in a **ErrorObserver** class. Below is an example of a plot that shows the spatial distribution of the error for all the variables together. See our [observer.py](py4cast/observer.py) for more examples.

```python
class SpatialErrorPlot(ErrorObserver):
    """
    Produce a map which shows where the error are accumulating (all variables together).
    """

    def __init__(self, prefix: str = "Test"):
        self.spatial_loss_maps = []
        self.prefix = prefix

    def update(
        self,
        obj: "AutoRegressiveLightning",
        prediction: NamedTensor,
        target: NamedTensor,
    ) -> None:
        spatial_loss = obj.loss(prediction, target, reduce_spatial_dim=False)
        # Getting only spatial loss for the required val_step_errors
        if obj.model.info.output_dim == 1:
            spatial_loss = einops.rearrange(
                spatial_loss, "b t (x y) -> b t x y ", x=obj.grid_shape[0]
            )
        self.spatial_loss_maps.append(spatial_loss)  # (B, N_log, N_lat, N_lon)

    def on_step_end(self, obj: "AutoRegressiveLightning") -> None:
        """
        Make the summary figure
        """
```

In order to add your own plot, you can create a new class that inherits from **ErrorObserver** and implement the **update** and **on_step_end** methods. You can then add your plot to the **AutoRegressiveLightning** class in the **valid_plotters** or **test_plotters** [list](py4cast/lightning.py).

```python
self.test_plotters = [
    StateErrorPlot(metrics),
    SpatialErrorPlot(),
    PredictionPlot(self.hparams["hparams"].num_samples_to_plot),
]
```

## Experiment tracking

We use [Tensorboad](https://www.tensorflow.org/tensorboard) to track the experiments. You can launch a tensorboard server using the following command:

1. At Météo-France


**runai** will handle port forwarding for you.

```bash
runai tensorboard --logdir PATH_TO_YOUR_ROOT_PATH
```

2. Elsewhere

```bash
tensorboard --logdir PATH_TO_YOUR_ROOT_PATH
```


Then you can access the tensorboard server at the following address: http://YOUR_SERVER_IP:YOUR_PORT/

## Unit tests

We provide a first set of unit tests to ensure the correctness of the codebase. You can run them using the following command:

```bash
python -m pytest
```

Our tests cover:
- The NamedTensor class
- The models, we make sure they can be instanciated and trained in a pure PyTorch training loop.


## Continuous Integration

We have a gitlab CI pipeline that runs linting (flake8, isort, black, bandit) and tests on every push to the repository. See the [gitlab-ci.yml](.gitlab-ci.yml) file for more details.

Our CI also launches two runs of the full system (*bin/train.py*) with our **Dummy** dataset using **HiLam** and **HalfUnet32**.


## TLDR design choices

- We define **interface contracts** between the components of the system using [Python ABCs](https://docs.python.org/3/library/abc.html). As long as the Python classes respect the interface contract, they can be used interchangeably in the system and the underlying implementation can be very different. For instance datasets with any underlying storage (grib2, netcdf, mmap+numpy, ...) and real-time or ahead of time concat and pre-processing could be used with the same neural network architectures and training strategies.

- Neural network architectures are Python classes that inherit from both **ModelABC** and PyTorch's **nn.Module**. The later means it is quick to insert a third-party pure PyTorch model in the system (see for instance the code for Lucidrains' Segformer or a U-Net).

- We use **dataclasses** and **dataclass_json** to define the settings whenever possible. This allows us to easily serialize and deserialize the settings to/from json files with Schema validation.

- The NamedTensor allows us to keep track of the physical/weather parameters along the features dimension and to pass a single consistent object in the system. It is also a way to factorize common operations on tensors (concat along features dimension, flatten in place, ...) while keeping the dimension and feature names metadata in sync.

- We use **PyTorch-lightning** to train the models. This allows us to easily scale the training to multiple GPUs and to use the same training loop for all the models. We also use the **PyTorch-lightning** logging system to log the training metrics and the hyperparameters.

### Ideas for future improvements

- Ideally, we could end up with a simple based class system for the training strategies to allow for easy addition of new strategies.

- The **ItemBatch** class attributes could be generalized to have multiple inputs, outputs and forcing tensors referenced by name, this would allow for more flexibility in the models and plug metnet-3 and Pangu.

- The distinction between **prognostic** and **diagnostic** variables should be made explicit in the system.

- We should probably reshape back the GNN outputs to (lat, lon) gridded shape as early as possible to have this as a common/standard output format for all the models. This would simplify the post-processing, plotting, ... We still have if statements in the code to handle the different output shapes of the models.
