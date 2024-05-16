# PNIA

This project built using **PyTorch** and **PyTorch-lightning** is designed to train a variety of Neural Network architectures (GNNs, CNNs, Vision Transformers, ...) on weather forecasting datasets.

Developped at Météo-France by **DSM/Lab IA** and **CNRM/GMAP/PREV**.

Contributions are welcome.

This project is licensed under the [APACHE 2.0 license.](LICENSE-2.0.txt)

# Table of contents

1. [Running at MF](#running-using-runai-météo-france-internal)
2. [Available PyTorch's architecture](#available-pytorchs-architecture)
    1. [Adding a new neural network architecture to the project.](#adding-a-new-neural-network-architecture-to-the-project)
3. [Available Training strategies](#available-training-strategies)
4. [NamedTensors](#namedtensors)

## Running using **runai** (Météo-France internal)

**runai** is our docker wrapper for training neural networks. See our [repository](https://git.meteo.fr/dsm-labia/monorepo4ai) for installation instructions.

`runai` commands must be issued at the root directory of the `pnia` project:

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
runai exec_gpu python bin/train.py --dataset smeagol --model halfunet --model_conf config/halfunet128_ghost.json
```

You can also override the dataset default configuration file:

```bash
runai exec_gpu python bin/train.py --dataset smeagol --model halfunet --dataset_conf config/smeagol.json
```

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

Below is a default working implementation for CNNs and VTs:

```python
def transform_statics(self, statics: Statics) -> Statics:
    return statics

def transform_batch(
    self, batch: Item
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    return batch.cat_2D()
```

For GNNs we need to flatten the width and height to graph_node_ids, since our datasets produce gridded shape data.

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
    new_batch = batch.cat_2D()

    new_batch.inputs[0].flatten_("ngrid", 2, 3)
    new_batch.outputs[0].flatten_("ngrid", 2, 3)

    if new_batch.forcing is not None:
        new_batch.forcing[0].flatten_("ngrid", 2, 3)

    return new_batch
```

Now your model can be either registered explicitely in the system (in case the code is in this repository) or injected in the system as a plugin (in case the code is hosted on a third party repository).

1. Model in the same git repository

Add your **NewModel** class to the registry explicitly in the models package [__init__.py](pnia/models/__init__.py)

```python
registry = {}
for kls in (HalfUnet, Unet, GraphLAM, HiLAM, HiLAMParallel, Segformer, NewModel):
    registry[kls.__name__.lower()] = kls
```

2. Model as a third party plugin

In order to be discovered, your model Python class MUST:

* be contained in a python module prefixed with **pnia_plugin_**
* inherit from **ModelABC** and **nn.Module**
* have a different name than the models already present in the system
* be discoverable by the system (e.g. in the PYTHONPATH or pip installed)

We provide an example module [here](pnia_plugin_example.py) to help you create your own plugin. This approach is based on the [official python packaging guidelines](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/).

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

The **NamedTensor** class is a wrapper around a PyTorch tensor.

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

See the implementation [here](pnia/datasets/base.py)