import ssl

import cartopy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pnia.models.nlam import utils

ssl._create_default_https_context = ssl._create_unverified_context

# Projection and grid
# TODO Do not hard code this, make part of static dataset files
# grid_shape = (268, 238)  # (y, x)

lambert_proj_params = {
    "a": 6367470,
    "b": 6367470,
    "lat_0": 63.3,
    "lat_1": 63.3,
    "lat_2": 63.3,
    "lon_0": 15.0,
    "proj": "lcc",
}

grid_limits = [  # In projection
    -1059506.5523409774,  # min x
    1310493.4476590226,  # max x
    -1331732.4471934352,  # min y
    1338267.5528065648,  # max y
]

# Create projection
lambert_proj = cartopy.crs.LambertConformal(
    central_longitude=lambert_proj_params["lon_0"],
    central_latitude=lambert_proj_params["lat_0"],
    standard_parallels=(lambert_proj_params["lat_1"], lambert_proj_params["lat_2"]),
)


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(errors, dataset, title=None, step_length=3):
    """
    Plot a heatmap of errors of different variables at different predictions horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape

    # Normalize all errors to [0,1] for color map
    max_errors = errors_np.max(axis=1)  # d_f
    errors_norm = errors_np / np.expand_dims(max_errors, axis=1)

    fig, ax = plt.subplots(figsize=(15, 10))

    ax.imshow(
        errors_norm,
        cmap="OrRd",
        vmin=0,
        vmax=1.0,
        interpolation="none",
        aspect="auto",
        alpha=0.8,
    )

    # ax and labels
    for (j, i), error in np.ndenumerate(errors_np):
        # Numbers > 9999 will be too large to fit
        formatted_error = f"{error:.3f}" if error < 9999 else f"{error:.2E}"
        ax.text(i, j, formatted_error, ha="center", va="center", usetex=False)

    # Ticks and labels
    label_size = 15
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
    pred_hor_h = step_length * pred_hor_i  # Prediction horiz. in hours
    ax.set_xticklabels(pred_hor_h, size=label_size)
    ax.set_xlabel("Lead time (h)", size=label_size)

    ax.set_yticks(np.arange(d_f))
    y_ticklabels = [
        f"{name} ({unit})"
        for name, unit in zip(dataset.shortnames("output"), dataset.units("output"))
    ]
    ax.set_yticklabels(y_ticklabels, rotation=30, size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(
    pred,
    target,
    obs_mask,
    grid_shape,
    title=None,
    vrange=None,
    projection=lambert_proj,
    grid_limits=grid_limits,
):
    """
    Plot example prediction and grond truth.
    Each has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange

    # Set up masking of border region
    mask_reshaped = obs_mask.reshape(*grid_shape)
    pixel_alpha = mask_reshaped.clamp(0.7, 1).cpu().numpy()  # Faded border region

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 7), subplot_kw={"projection": projection}
    )

    # Plot pred and target
    for ax, data in zip(axes, (target, pred)):
        ax.coastlines()  # Add coastline outlines
        data_grid = data.reshape(*grid_shape).cpu().numpy()
        im = ax.imshow(
            data_grid,
            origin="lower",
            extent=grid_limits,
            alpha=pixel_alpha,
            vmin=vmin,
            vmax=vmax,
            cmap="plasma",
        )

    # Ticks and labels
    axes[0].set_title("Ground Truth", size=15)
    axes[1].set_title("Prediction", size=15)
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error(
    error,
    obs_mask,
    grid_shape,
    title=None,
    vrange=None,
    projection=lambert_proj,
    grid_limits=grid_limits,
):
    """
    Plot errors over spatial map
    Error and obs_mask has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange

    # Set up masking of border region
    mask_reshaped = obs_mask.reshape(*grid_shape)
    pixel_alpha = mask_reshaped.clamp(0.7, 1).cpu().numpy()  # Faded border region

    fig, ax = plt.subplots(figsize=(5, 4.8), subplot_kw={"projection": projection})

    ax.coastlines()  # Add coastline outlines
    print(error.shape)
    error_grid = error.reshape(*grid_shape).cpu().numpy()

    im = ax.imshow(
        error_grid,
        origin="lower",
        extent=grid_limits,
        alpha=pixel_alpha,
        vmin=vmin,
        vmax=vmax,
        cmap="OrRd",
    )

    # Ticks and labels
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig
