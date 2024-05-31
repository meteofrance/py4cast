# PY4CAST : Adding Features and Contributing

## Table of contents

1. [Adding new features](#adding-new-features)
    1. [Adding a new neural network architecture](#adding-a-new-neural-network-architecture-to-the-project)
    2. [Adding a new dataset](#adding-a-new-dataset)
    3. [Adding Training Plots](#adding-training-plots)
2. [Contribution Guidelines](#contribution-guidelines)
    1. [Unit tests](#unit-tests)
    2. [Continuous Integration](#continuous-integration)

## Adding New Features

### Adding a new neural network architecture to the project

1. Neural network architectures MUST be Python classes and inherit from both **ModelABC** and  **nn.Module**, in that order.

```python
class NewModel(ModelABC, nn.Module):
    settings_kls = NewModelSettings
    onnx_supported = True
    input_dims: Tuple[str, ...] = ("batch", "height", "width", "features")
    output_dims: Tuple[str, ...] = ("batch", "height", "width", "features")
```

2. The **onnx_supported** attribute MUST be present and set to **True** if the architecture is onnx exportable, see [our unit tests](tests/test_models.py).

3. the **input_dims** and **output_dims** attributes MUST be present to explicit the model's input and output tensor shapes.

4. The **settings_kls** attribute MUST be present and set to the **dataclass_json** setting class of the architecture.


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


### Adding a new dataset

1. Your dataset **MUST** inherit from **DatasetABC** and **data.Dataset**, in thats order.

```python
class TitanDataset(DatasetABC, Dataset):
    ...
```

2. Your dataset **MUST** implement
    * ALL the abstract properties from **DatasetABC**: `dataset_info`, `meshgrid`, `geopotential_info` and `cache_dir`.
    * ALL the abstract methods from **DatasetABC**: `torch_dataloader` and `from_json`.
    * 2 methods from **data.Dataset**: `__len__` and `__getitem__`.

3. Your `__getitem__` method MUST return an Item object.

```python
    def __getitem__(self, index: int) -> Item:
        ...
```

4. It is MANDATORY that your `__getitem__` method returns [Item](../py4cast/datasets/base.py#L288) instances containing NamedTensors with precise feature and dimension names. By convention we use these names for tensor dimensions: **("batch", "timestep", "lat", "lon", "features")**.

5. It is HIGHLY RECOMMENDED that your dataset implements a `prepare` method to easily compute and save all the statics needed by your dataset.


### Adding Training Plots

Plots are done using the **matplotlib** library. We wrap each plot in a **ErrorObserver** class. Below is an example of a plot that shows the spatial distribution of the error for all the variables together. See our [observer.py](../py4cast/observer.py#L40) for more examples.

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

## Contribution guidelines

Anyone can contribute through a merge request. The code must pass the unit tests and continuous integration before it is merged.

### Unit tests

We provide a first set of unit tests to ensure the correctness of the codebase. You can run them using the following command:

```bash
python -m pytest
```

Our tests cover:
- The NamedTensor class
- The models, we make sure they can be instanciated and trained in a pure PyTorch training loop.


### Continuous Integration

We have a gitlab CI pipeline that runs linting (flake8, isort, black, bandit) and tests on every push to the repository. See the [gitlab-ci.yml](../.gitlab-ci.yml) file for more details.

Our CI also launches two runs of the full system (*bin/train.py*) with our **Dummy** dataset using **HiLam** and **HalfUnet32**.

```bash
python bin/train.py --model hilam --dataset dummy --epochs 1 --batch_size 1 --num_pred_steps_train 1 --limit_train_batches 1
```
