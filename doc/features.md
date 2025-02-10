# PY4CAST : Features

## Table of contents

1. [PyTorch's architectures](#available-pytorchs-architectures)
2. [Datasets](#available-datasets)
3. [Training strategies](#available-training-strategies)
4. [Losses](#available-losses)
5. [Plots](#available-plots)
5. [NamedTensors](#namedtensors)

## Available PyTorch's architectures

Currently we support the neural network architectures available in our [mfai Library](https://github.com/meteofrance/mfai)

## Available datasets

Currently we support the following datasets:

| Dataset  | Domain  | Description    | Documentation  | Maintainer(s) |
| :---:   | :---: | :---: | :---: | :---: |
| titan| France | AROME Analyses + ARPEGE Analyses and forecasts + 1h Rainfall; Timestep 1h; 2022-2023; [download link](https://huggingface.co/datasets/meteofrance/titan)  | [link](titan.md) | Léa Berthomier |
| smeagol | France | A private dataset for assimilation research  | ... |  Vincent Chabot |
| dummy | Persian Gulf  | A 64x64 dataset for doc and testing purposes using random data, also used in our CI to test the whole system | See [the code](../py4cast/datasets/dummy.py) |  Vincent Chabot / Frank Guibert |

## Available Training strategies

The training strategy determines how the next timestep is computed
in the forward pass. **x** are the neural network inputs and **model(x)** is the returned value by the neural network when fed **x** as input.

| Strategy Name | Reference | Update Rule | Boundary forcing |  Intermediary Steps |
| :---:   | :---: | :---: | :---: | :---: |
| scaled_ar |  | next_state = previous_state + model(x)*diff_std + diff_mean | y_true  | Yes |
|  diff_ar | | next_state = previous_state + model(x) | No |  No |
| downscaling_only | next_state = model(x) | No |  No |


## Available Losses

Implemented losses are based on RMSE and MAE measurements between each element in the input **x** and the output **y**. Their weighted versions are also proposed. Py4CastLoss class is designed to inherit from 'ABC'. This class serves as a template for defining custom loss functions in a PyTorch Lightning system, ensuring that subclasses implement the necessary preparation and computation methods. It also includes functionality to manage and register additional state information required for these loss functions. For that:

'prepare' is an abstract method which prepares the loss function using the dataset information and an interior mask.

```python
@abstractmethod
def prepare(
    self,
    lm: pl.LightningModule,
    interior_mask: torch.Tensor,
    dataset_info: DatasetInfo,
) -> None:
    """
    Prepare the loss function using the dataset informations and the interior mask
    """
```

'forward' computes the loss given the predictions and targets.
```python
@abstractmethod
def forward(self, prediction: NamedTensor, target: NamedTensor) -> torch.Tensor:
    """
    Compute the loss function
    """
```

'register_loss_state_buffers' registers various state buffers to a PyTorch Lightning module.
```python
def register_loss_state_buffers(
    self,
    lm: pl.LightningModule,
    interior_mask: torch.Tensor,
    loss_state_weight: dict,
    squeeze_mask: bool = False,
) -> None:
    """
    We register the state_weight buffer to the lightning module
    and keep references to other buffers of interest
    """
```

## Available Plots

Plots are done using the **matplotlib** library. We wrap each plot in a **Plotter** class. Below is an example of a plot that shows the spatial distribution of the error for all the variables together. See our [plots.py](../py4cast/plots.py#L212) for more examples.

```python
class SpatialErrorPlot(Plotter):
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

In order to add your own plot, you can create a new class that inherits from **Plotter** and implement the **update** and **on_step_end** methods. You can then add your plot to the **AutoRegressiveLightning** class in the **valid_plotters** or [**test_plotters** list](../py4cast/lightning.py#L398).

```python
self.test_plotters = [
    StateErrorPlot(metrics),
    SpatialErrorPlot(),
    PredictionPlot(self.hparams["hparams"].num_samples_to_plot),
]
```
## NamedTensors

PyTorch provides an experimental feature called [**named tensors**](https://pytorch.org/docs/stable/named_tensor.html), at this time it is subject to change so we don't use it. That's why we provide our own implementation.

NamedTensors are a way to give names to dimensions of tensors and to keep track of the names of the physical/weather parameters along the features dimension.

The [**NamedTensor**](../py4cast/datasets/base.py#L38) class is a wrapper around a PyTorch tensor, it allows us to pass consistent object linking data and metadata with extra utility methods (concat along features dimension, flatten in place, ...). See the implementation [here](../py4cast/datasets/base.py#L38) and usage for plots [here](../py4cast/observer.py)

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