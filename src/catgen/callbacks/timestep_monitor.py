"""Callback for monitoring timestep distribution during training."""

import torch
from lightning import Callback, LightningModule, Trainer


class TimestepDistributionMonitor(Callback):
    """Log timestep distribution statistics during training.

    Tracks mean, std, min, max of sampled timesteps and logs histogram
    to W&B for visual verification that training covers all timesteps.
    """

    def __init__(self, log_every_n_steps: int = 10, log_histogram: bool = True):
        """Initialize the callback.

        Args:
            log_every_n_steps: How often to log timestep statistics.
            log_histogram: Whether to log histogram to W&B.
        """
        self.log_every_n_steps = log_every_n_steps
        self.log_histogram = log_histogram
        self._timestep_buffer = []
        self._buffer_size = 500  # Accumulate timesteps for histogram

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Log timestep statistics after each training batch."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        # Access the last sampled timesteps from the module
        if not hasattr(pl_module, "_last_timesteps") or pl_module._last_timesteps is None:
            return

        times = pl_module._last_timesteps

        # Log statistics
        pl_module.log("train/timestep/mean", times.mean())
        pl_module.log("train/timestep/std", times.std())
        pl_module.log("train/timestep/min", times.min())
        pl_module.log("train/timestep/max", times.max())

        # Log bin counts for verification
        bins = [
            (0.0, 0.25, "0.00-0.25"),
            (0.25, 0.5, "0.25-0.50"),
            (0.5, 0.75, "0.50-0.75"),
            (0.75, 1.0, "0.75-1.00"),
        ]
        for low, high, name in bins:
            if high < 1.0:
                count = ((times >= low) & (times < high)).sum().float()
            else:
                count = ((times >= low) & (times <= high)).sum().float()
            pl_module.log(f"train/timestep/bin/{name}", count)

        # Accumulate for histogram
        if self.log_histogram:
            self._timestep_buffer.extend(times.cpu().tolist())
            if len(self._timestep_buffer) >= self._buffer_size:
                self._log_histogram(trainer, pl_module)
                self._timestep_buffer = []

    def _log_histogram(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log histogram to W&B."""
        if not pl_module.logger or not hasattr(pl_module.logger, "experiment"):
            return

        try:
            import wandb

            pl_module.logger.experiment.log(
                {
                    "train/timestep/histogram": wandb.Histogram(self._timestep_buffer),
                    "trainer/global_step": trainer.global_step,
                }
            )
        except Exception:
            # Silently ignore histogram logging errors
            pass
