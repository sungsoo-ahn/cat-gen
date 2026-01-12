"""Callback for monitoring gradient norms during training."""

import torch
from lightning import Callback, LightningModule, Trainer


class GradientNormMonitor(Callback):
    """Log gradient norms for debugging convergence issues.

    Logs total gradient norm and per-module gradient norms to help identify
    vanishing/exploding gradients or imbalanced loss components.
    """

    def __init__(self, log_every_n_steps: int = 10):
        """Initialize the callback.

        Args:
            log_every_n_steps: How often to log gradient norms.
        """
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer
    ) -> None:
        """Log gradient norms before optimizer step."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        # Compute total gradient norm
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        pl_module.log("train/grad_norm_total", total_norm)

        # Log per-module gradient norms for the flow model
        if hasattr(pl_module, "structure_module"):
            flow_model = pl_module.structure_module.flow_model
            for name, module in flow_model.named_children():
                module_norm = 0.0
                for p in module.parameters():
                    if p.grad is not None:
                        module_norm += p.grad.data.norm(2).item() ** 2
                module_norm = module_norm ** 0.5
                pl_module.log(f"train/grad_norm_{name}", module_norm)
