import numpy as np
import torch
from scipy.fftpack import dct
from torchmetrics import Metric

from py4cast.datasets.base import NamedTensor
from py4cast.plots import plot_log_psd


class MetricPSDK(Metric):
    """
    Compute the PSD as a function of the wave number for each channels/features
    """

    def __init__(self):
        super().__init__()

        # Declaration of state, states are reset when self.reset() is called
        # Sum PSD prediction at each epoch
        self.add_state(
            "sum_psd_pred",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        # Sum PSD target at each epoch
        self.add_state(
            "sum_psd_target",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        # Step counter, needed to compute psd mean at each epoch.
        self.add_state("step_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: NamedTensor, target: NamedTensor):
        """
        Add PSD to their respective sum. Should be called at each validation step's end
        """
        if preds.tensor.shape != target.tensor.shape:
            raise ValueError("preds and target must have the same shape")

        if self.step_count == 0:
            self.feature_names = preds.feature_names

        # Compute PSD on first pred_step only
        pred_step = 0

        # Add PSD to the sums
        self.sum_psd_pred = self.add_psd(preds.tensor, self.sum_psd_pred, pred_step)
        self.sum_psd_target = self.add_psd(
            target.tensor, self.sum_psd_target, pred_step
        )

        # Increment count_step
        self.step_count += 1

    def compute(self) -> dict:
        """
        Compute PSD mean for each channels/features, plot the figure, return a dict.
        Should be called at each epoch's end
        """

        # Compute PSD mean over an epoch
        mean_psd_pred = self.sum_psd_pred / self.step_count
        mean_psd_target = self.sum_psd_target / self.step_count
        features_name = self.feature_names
        nb_channel = mean_psd_pred.shape[0]

        # Compute max wave number
        Rmax = mean_psd_pred.shape[1]
        # Compute all wave number
        k = np.linspace(2 * np.pi / 2.6, Rmax * 2 * np.pi / 2.6, Rmax)

        # dict {"feature_name" : fig}
        dict_psd_metric = dict()
        for c in range(nb_channel):
            fig = plot_log_psd(
                k, mean_psd_pred[c].cpu().numpy(), mean_psd_target[c].cpu().numpy()
            )
            dict_psd_metric[f"mean_psd_k/{features_name[c]}"] = fig

        # Reset metric's state
        self.reset()

        return dict_psd_metric

    def add_psd(self, x: torch.Tensor, sum_psd: torch.Tensor, pred_step: int):
        """
        Add the PSD of x to the variable sum_psd
        """
        # x should be (Batch, channels, Lon, Lat) to be an argument of power_spectral_density

        x = x.permute(0, -1, *range(2, x.dim() - 1), 1)
        channels = x.shape[1]

        # Compute the PSD for each channel of x.
        for c in range(channels):

            psd = power_spectral_density(
                x[:, c : c + 1, :, :, pred_step].cpu().numpy()
            ).squeeze()

            psd_torch = torch.from_numpy(psd).to(device=self.device)

            # Initialize sum_psd as a tensor of dim (nb_features, wave_number)
            if not sum_psd.ndim:
                sum_psd = torch.zeros([x.shape[1], psd.shape[0]], device=self.device)

            # Add PSD for this channel
            sum_psd[c, :] += psd_torch

        return sum_psd


class MetricPSDVar(Metric):
    """
    Compute the RMSE between the target and the prediciton PSD for each channels/features.
    """

    def __init__(self):
        super().__init__()

        # Declaration of state, states are reset when self.reset() is called
        # Sum RMSE PSD prediction at each epoch
        self.add_state(
            "sum_rmse",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        # Step counter, needed to compute psd mean at each epoch.
        self.add_state("step_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: NamedTensor, target: NamedTensor):
        """
        compute the RSME between target and pred PSD.
        called at each end of step
        """
        if self.step_count == 0:
            self.feature_names = preds.feature_names

        # tensor should be (Batch, channels, Lon, Lat) or (Batch, channels, ngrids)
        # to be an argument of power_spectral_density
        preds = preds.tensor.permute(0, -1, *range(2, preds.tensor.dim() - 1), 1)
        target = target.tensor.permute(0, -1, *range(2, target.tensor.dim() - 1), 1)

        # Compute PSD on first pred_step only
        pred_step = 0
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        channels = target.shape[1]
        res = np.zeros((channels,))

        # Compute the RMSE for each channel of x.
        for c in range(channels):

            # Compute PSD
            psd_target = power_spectral_density(
                target[:, c : c + 1, :, :, pred_step].cpu().numpy()
            )
            psd_pred = power_spectral_density(
                preds[:, c : c + 1, :, :, pred_step].cpu().numpy()
            )
            # Compute RMSE
            res[c] = np.sqrt(np.mean((np.log10(psd_target) - np.log10(psd_pred)) ** 2))

        # Initialize sum_rmse as a tensor of dim (nb_features)
        if not self.sum_rmse.ndim:  # self.sum_rmse not yet initalized as a tensor
            self.sum_rmse = torch.zeros(channels, device=self.device)

        # Add RMSE for this channel
        self.sum_rmse += torch.from_numpy(res).to(device=self.device)

        # Increment step_count
        self.step_count += 1

    def compute(self) -> dict:
        """
        Compute PSD mean for each channels/features, return a dict.
        Should be called at each epoch's end
        """
        # Compute mean RMSE over an epoch
        mean_psd_rmse = self.sum_rmse / self.step_count
        feature_names = self.feature_names

        # dict {"feature_name" : RMSE mean}
        metric_log_dict = {
            f"val_rmse_psd/{name}": mean_psd_rmse[i]
            for i, name in enumerate(feature_names)
        }

        # Reset metric's state
        self.reset()

        return metric_log_dict


# Useful function for spectral compute
def dct_2d(x):
    """
    2D dicrete_cosine_transf transform for 2D (square) numpy array
    or for each sample b of BxNxN numpy array. Applies the scipy
    dct function (Type II) along rows and columns.

    """
    assert x.ndim in [2, 3]
    if x.ndim == 3:
        res = dct(
            dct(x.transpose((0, 2, 1)), norm="ortho").transpose((0, 2, 1)),
            norm="ortho",
        )
    else:
        res = dct(dct(x.T, norm="ortho").T, norm="ortho")
    return res


def dct_var(x: np.ndarray) -> np.ndarray:
    """
    compute the bidirectional variance spectrum of the (square) numpy array x

    Inputs :
        x : numpy array, shape is B x 1 x N x N

    Return :
        sigma : 2D numpy array representing the variance spectrum
    """
    n = x.shape[-1]
    # compute the discret cosine transform in 2D of x
    fx = dct_2d(x)
    # compute the variance
    sigma = (1 / n**2) * fx**2
    return sigma


def radial_bin_dct(dct_sig: np.ndarray, center: tuple) -> np.ndarray:
    """
    radial binning on the variance spectrum obtained from the DCT

    Inputs :
        dct_sig : variance spectrum
        center : center of the radial binning (typically the center of the array)

    Return :
        radial_profile : 1D numpy array representing the radially-averaged power spectrum
    """
    y, x = np.indices(dct_sig.shape)

    # computation of radial distance from the center for each point of the spectrum
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)

    # maximum radial distance
    Rmax = min(x.max(), y.max(), r.max()) // 2

    # double binning for dct
    dct = (
        dct_sig.ravel()[2 * r.ravel()]
        + 0.5 * dct_sig.ravel()[2 * r.ravel() - 1]
        + 0.5 * dct_sig.ravel()[2 * r.ravel() + 1]
    )

    # aggregate binned values and compute the radial profile by averaging
    tbin = np.bincount(r.ravel()[r.ravel() < Rmax], dct[r.ravel() < Rmax])
    nr = np.bincount(r.ravel()[r.ravel() < Rmax])
    radial_profile = tbin / nr

    return radial_profile


def power_spectral_density(x: np.ndarray) -> np.ndarray:
    """
    compute the radially-averaged, sample-averaged power spectral density
    of the data x

    Inputs :
        x : numpy array, shape is B x C x N x N

    Return :
        out : numpy array, shape is (C, Rmax), with R_max defined in radial_bin_dct function
    """
    out_list = []
    channels = x.shape[1]

    for c in range(channels):
        x_c = x[:, c, :, :]

        # compute the variance spectrum and average over the batch dimension
        sig = dct_var(x_c).mean(axis=0)

        # determine the center for radial binning
        center = (sig.shape[0] // 2, sig.shape[1] // 2)

        # compute the radial profile
        out_list.append(radial_bin_dct(sig, center))

    # concatenate for all channels
    out = np.concatenate([np.expand_dims(o, axis=0) for o in out_list], axis=0)
    return out
