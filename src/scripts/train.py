#!/usr/bin/env python3
"""Train CatGen model.

Usage:
    uv run python src/scripts/train.py configs/default/default.yaml
    uv run python src/scripts/train.py configs/default/default.yaml --overwrite
"""

import argparse
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger

from src.utils import init_directory
from src.catgen.data.datamodule import LMDBDataModule
from src.catgen.module.catgen_module import CatGen
from src.catgen.callbacks import GradientNormMonitor


def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load and validate config from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required fields - fail fast!
    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' required in config")

    required_sections = ["experiment", "data", "model", "training"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"FATAL: '{section}' section required in config")

    return config


def build_callbacks(config: dict, output_dir: Path) -> list:
    """Build Lightning callbacks from config."""
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
    ]

    # Model checkpointing
    ckpt_config = config["training"]["checkpoint"]
    callbacks.append(
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="epoch={epoch}-val_loss={val/total_loss:.4f}",
            monitor=ckpt_config["monitor"],
            mode=ckpt_config["mode"],
            save_top_k=ckpt_config["save_top_k"],
            save_last=ckpt_config["save_last"],
            auto_insert_metric_name=False,
        )
    )

    # Gradient norm monitoring for debugging
    wandb_config = config.get("wandb", {})
    if wandb_config.get("watch_gradients", False):
        callbacks.append(GradientNormMonitor(log_every_n_steps=10))

    return callbacks


def create_model(config: dict) -> CatGen:
    """Create CatGen from config."""
    model_config = config["model"]
    training_config = config["training"]
    validation_config = config["validation"]
    prediction_config = config.get("prediction", {})

    model = CatGen(
        hidden_dim=model_config["hidden_dim"],
        flow_model_args=model_config["flow_model_args"],
        training_args={
            "train_multiplicity": training_config["train_multiplicity"],
            "loss_type": training_config["loss_type"],
            "prim_slab_coord_loss_weight": training_config["prim_slab_coord_loss_weight"],
            "ads_coord_loss_weight": training_config["ads_coord_loss_weight"],
            "prim_virtual_loss_weight": training_config["prim_virtual_loss_weight"],
            "supercell_virtual_loss_weight": training_config["supercell_virtual_loss_weight"],
            "scaling_factor_loss_weight": training_config["scaling_factor_loss_weight"],
            "loss_space": training_config.get("loss_space", "raw"),
            # Optimizer configuration
            "optimizer": training_config.get("optimizer", "adamw"),
            "lr": training_config["lr"],
            "weight_decay": training_config["weight_decay"],
            "adam_beta1": training_config.get("adam_beta1", 0.9),
            "adam_beta2": training_config.get("adam_beta2", 0.999),
            # Muon-specific parameters
            "muon_momentum": training_config.get("muon_momentum", 0.95),
            "muon_nesterov": training_config.get("muon_nesterov", True),
            "muon_ns_steps": training_config.get("muon_ns_steps", 5),
            "adamw_lr": training_config.get("adamw_lr"),
            "scheduler": training_config.get("scheduler", {}),
        },
        validation_args=validation_config,
        flow_process_args=model_config["flow_process_args"],
        prior_sampler_args=model_config["prior_sampler_args"],
        predict_args=prediction_config,
        use_kernels=model_config.get("use_kernels", False),
        use_ema=training_config.get("use_ema", False),
        ema_decay=training_config.get("ema_decay", 0.9999),
    )

    return model


def create_datamodule(config: dict) -> LMDBDataModule:
    """Create LMDBDataModule from config."""
    data_config = config["data"]

    return LMDBDataModule(
        train_lmdb_path=data_config["train_lmdb_path"],
        val_lmdb_path=data_config.get("val_lmdb_path"),
        test_lmdb_path=data_config.get("test_lmdb_path"),
        batch_size=SimpleNamespace(**data_config["batch_size"]),
        num_workers=SimpleNamespace(**data_config["num_workers"]),
        preload_to_ram=data_config.get("preload_to_ram", True),
        use_pyg=data_config.get("use_pyg", False),
        train_shuffle=data_config.get("train_shuffle", True),
    )


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    """Main training function."""
    # Load config
    config = load_config(config_path)

    # Initialize output directory
    output_dir = init_directory(config["output_dir"], overwrite=overwrite)

    # Create subdirectories
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    # Copy config for reproducibility
    shutil.copy(config_path, output_dir / "config.yaml")

    # Set seed
    set_seed(
        config["experiment"]["seed"], config["experiment"]["deterministic"]
    )

    # Initialize WandB
    wandb_logger = None
    wandb_config = config.get("wandb", {})
    if wandb_config.get("enabled", False):
        wandb_logger = WandbLogger(
            project=wandb_config.get("project", "CatGen"),
            name=f"{config['experiment']['name']}_{config['code_version']}",
            save_dir=str(output_dir / "logs"),
            config=config,
            tags=[
                config["code_version"],
                f"seed_{config['experiment']['seed']}",
                *wandb_config.get("tags", []),
            ],
            group=config["experiment"]["name"],
            job_type=config["code_version"],
            mode=wandb_config.get("mode", "online"),
        )

    # Create data module
    print("Creating data module...")
    data_module = create_datamodule(config)

    # Create model
    print("Creating model...")
    model = create_model(config)

    # Enable WandB gradient watching if configured
    if wandb_logger is not None and wandb_config.get("watch_gradients", False):
        watch_freq = wandb_config.get("watch_freq", 100)
        wandb_logger.watch(model, log="gradients", log_freq=watch_freq)
        print(f"WandB gradient watching enabled (freq={watch_freq})")

    # Create trainer
    training_config = config["training"]
    trainer = L.Trainer(
        max_epochs=training_config["max_epochs"],
        accelerator=training_config["accelerator"],
        devices=training_config["devices"],
        strategy=training_config["strategy"],
        precision=training_config["precision"],
        gradient_clip_val=training_config["gradient_clip_val"],
        gradient_clip_algorithm=training_config["gradient_clip_algorithm"],
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        val_check_interval=training_config.get("val_check_interval"),
        check_val_every_n_epoch=training_config.get("check_val_every_n_epoch"),
        num_sanity_val_steps=training_config["num_sanity_val_steps"],
        limit_train_batches=training_config.get("limit_train_batches"),
        limit_val_batches=training_config.get("limit_val_batches"),
        callbacks=build_callbacks(config, output_dir),
        logger=wandb_logger,
        default_root_dir=str(output_dir),
        deterministic=config["experiment"]["deterministic"],
        fast_dev_run=debug,
    )

    # Train
    print("Starting training...")
    trainer.fit(model, data_module)

    print(f"Training complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CatGen")
    parser.add_argument("config_path", type=str, help="Path to config YAML")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode (fast_dev_run)")
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
