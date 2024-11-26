import torch
import numpy as np
from PIL import Image
import io


###--------------------- PLOT ---------------------###


class PlotLightningModule:
    def __init__(self):
        pass

    def log_loss(self, tb, metrics_loss_dict, label):
        """Logs loss to TensorBoard.
        Args:
            label: The current label (train, val, test).
            loss: The loss value.
        """
        step = self.current_epoch if label != "train" else self.global_step
        avg_loss = torch.stack([x["loss"] for x in metrics_loss_dict]).mean()
        tb.add_scalar(f"loss/{label}", avg_loss, step)

    def log_metrics(self, tb, metrics_loss_dict, label):
        """Logs metrics to TensorBoard.
        Args:
            label: The current label (train, val, test).
            metrics: A dictionary of metrics.
        """
        step = self.current_epoch if label != "train" else self.global_step
        for metric in self.metrics:
            avg_m = torch.stack([x[metric] for x in metrics_loss_dict]).mean()
            tb.add_scalar(f"metrics/{label}_{metric}", avg_m, step)

    def log_images(self, tb, label, y, y_hat, num_channels=3, channel_indices=None):
        """Logs images to TensorBoard. Batching is supposedly done on chronologically adjacent samples. (B=0 <=> t=i), (B=1 <=> t=i+1), etc.
        Args:
            label: The current label (train, val, test).
            y: The ground truth image tensor.
            y_hat: The predicted image tensor.
            num_channels: The number of channels to plot (default: 3).
            channel_indices: A list of channel indices to plot (optional).
                            If None, the first num_channels will be plotted.
        """
        step = self.current_epoch if label != "train" else self.global_step

        # Process and log the ground truth image (y)
        if channel_indices is None:
            channels_to_plot = y[:num_channels]
        else:
            channels_to_plot = y[channel_indices]
        image_to_plot = channels_to_plot.permute(1, 2, 0)
        tb.add_image(f"{label}/ground_truth", image_to_plot, step, dataformats="HW")

        # Process and log the predicted image (y_hat)
        if channel_indices is None:
            channels_to_plot = y_hat[:num_channels]
        else:
            channels_to_plot = y_hat[channel_indices]
        image_to_plot = channels_to_plot.permute(1, 2, 0)
        tb.add_image(f"{label}/prediction", image_to_plot, step, dataformats="HW")

    def log_gif(self, tb, label, y, y_hat, number_of_steps, channel_indices=0):
        """Logs a GIF to TensorBoard showing the temporal evolution of a specific channel.

        Args:
            label: The current label (train, val, test).
            y: The ground truth image tensor (size=B,C,H,W).
            y_hat: The predicted image tensor (size=B,C,H,W).
            number_of_steps: The number of steps (batches) to include in the GIF.
            channel_indices: The index of the channel to display (default: 1).
        """

        # Select the specified channel for both ground truth and prediction
        y_channel = y[:number_of_steps, channel_indices, :, :]
        y_hat_channel = y_hat[:number_of_steps, channel_indices, :, :]

        # Create lists to store frames for the GIF
        y_frames = []
        y_hat_frames = []

        for i in range(number_of_steps):
            # Convert tensors to PIL images
            y_image = Image.fromarray(
                y_channel[i].cpu().numpy(), mode="F"
            )  # Assuming 'F' mode for single-channel data
            y_hat_image = Image.fromarray(y_hat_channel[i].cpu().numpy(), mode="F")

            # Add frames to the lists
            y_frames.append(y_image)
            y_hat_frames.append(y_hat_image)

        # Save GIFs to bytes buffers
        y_buffer = io.BytesIO()
        y_frames[0].save(
            y_buffer, format="GIF", append_images=y_frames[1:], save_all=True, loop=0
        )

        y_hat_buffer = io.BytesIO()
        y_hat_frames[0].save(
            y_hat_buffer,
            format="GIF",
            append_images=y_hat_frames[1:],
            save_all=True,
            loop=0,
        )

        # Log GIFs to TensorBoard
        tb.add_image(
            f"{label}/ground_truth_gif",
            np.array(Image.open(y_buffer)),
            0,
            dataformats="HW",
        )
        tb.add_image(
            f"{label}/prediction_gif",
            np.array(Image.open(y_hat_buffer)),
            0,
            dataformats="HW",
        )


###--------------------- MASK ---------------------###


class MaskLightningModule:
    def __init__(self):
        pass

    def create_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Creates a random binary mask with masking ratio applied per channel.
        Args:
            x: The input tensor (shape: B, C, H, W).
            mask_ratio: The fraction of elements to mask (between 0 and 1).
        Returns:
            A binary mask tensor of the same size as x, with 0s and 1s.
        """
        B, C, H, W = x.shape
        num_mask_per_channel = int(
            mask_ratio * H * W
        )  # number of elements to mask per channel
        mask = torch.ones_like(
            x, dtype=torch.bool
        )  # empty mask with the same shape as x
        for b in range(B):
            for c in range(C):
                mask_indices = torch.randperm(H * W)[
                    :num_mask_per_channel
                ]  # random indices for the current channel
                mask[b, c].view(-1)[mask_indices] = False  # Set to False for masking
        return mask
