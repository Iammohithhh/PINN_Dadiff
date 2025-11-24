"""
Training Script for CT-PINN-DADif.

Implements the complete training pipeline with:
- SAM optimizer for robust generalization
- Physics-informed loss functions
- Validation and checkpointing
- Logging and visualization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
import time
import json
from tqdm import tqdm

from .model import CT_PINN_DADif, CTReconstructionLoss, SAM, create_model, create_loss
from .data_loader import create_dataloaders
from .ct_physics import RadonTransform


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute PSNR, SSIM, RMSE, and MAE metrics.

    FIXED: Added RMSE and MAE - standard CT reconstruction metrics.

    Args:
        pred: Predicted image (B, 1, H, W)
        target: Ground truth image (B, 1, H, W)

    Returns:
        Dictionary with PSNR, SSIM, RMSE, and MAE values
    """
    # Ensure same range
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    # MSE
    mse = ((pred - target) ** 2).mean().item()

    # RMSE (Root Mean Square Error) - ADDED
    rmse = np.sqrt(mse)

    # MAE (Mean Absolute Error) - ADDED
    mae = (torch.abs(pred - target)).mean().item()

    # PSNR
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = 10 * np.log10(1.0 / mse)

    # SSIM (simplified)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_pred = pred.mean(dim=(2, 3), keepdim=True)
    mu_target = target.mean(dim=(2, 3), keepdim=True)

    sigma_pred = ((pred - mu_pred) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_target = ((target - mu_target) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_both = ((pred - mu_pred) * (target - mu_target)).mean(dim=(2, 3), keepdim=True)

    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_both + C2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))

    return {
        'psnr': psnr,
        'ssim': ssim.mean().item() * 100,  # Convert to percentage
        'rmse': rmse,
        'mae': mae
    }


class Trainer:
    """
    Trainer class for CT-PINN-DADif.
    """

    def __init__(
        self,
        model: CT_PINN_DADif,
        loss_fn: CTReconstructionLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        lr = config.get('learning_rate', 6e-3)
        use_sam = config.get('use_sam', True)

        if use_sam:
            base_optimizer = torch.optim.Adam
            self.optimizer = SAM(
                model.parameters(),
                base_optimizer,
                lr=lr,
                betas=(0.9, 0.999),
                rho=config.get('sam_rho', 0.05)
            )
            self.use_sam = True
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.999)
            )
            self.use_sam = False

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer.base_optimizer if use_sam else self.optimizer,
            T_max=config.get('num_epochs', 600),
            eta_min=1e-6
        )

        # Mixed precision
        self.use_amp = config.get('use_amp', True) and self.device.startswith('cuda')
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'experiments/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.log_dir = Path(config.get('log_dir', 'experiments/logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Best metrics
        self.best_psnr = 0.0
        self.best_ssim = 0.0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch in pbar:
            # Move to device
            sinogram = batch['sinogram_noisy'].to(self.device)
            target = batch['image'].to(self.device)
            weights = batch['weights'].to(self.device)
            mask = batch['mask'].to(self.device)

            if self.use_sam:
                # SAM: First forward-backward
                self.optimizer.zero_grad()

                with autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(sinogram, weights, mask)
                    losses = self.loss_fn(
                        outputs['reconstruction'], target,
                        sinogram, weights
                    )
                    loss = losses['total']

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer.base_optimizer)
                else:
                    loss.backward()

                self.optimizer.first_step(zero_grad=True)

                # SAM: Second forward-backward
                with autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(sinogram, weights, mask)
                    losses = self.loss_fn(
                        outputs['reconstruction'], target,
                        sinogram, weights
                    )
                    loss = losses['total']

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer.base_optimizer)
                else:
                    loss.backward()

                self.optimizer.second_step(zero_grad=True)

                if self.use_amp:
                    self.scaler.update()
            else:
                # Standard optimizer
                self.optimizer.zero_grad()

                with autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(sinogram, weights, mask)
                    losses = self.loss_fn(
                        outputs['reconstruction'], target,
                        sinogram, weights
                    )
                    loss = losses['total']

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            # Compute metrics
            with torch.no_grad():
                metrics = compute_metrics(outputs['reconstruction'], target)

            total_loss += loss.item()
            total_psnr += metrics['psnr']
            total_ssim += metrics['ssim']
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'psnr': f"{metrics['psnr']:.2f}",
                'ssim': f"{metrics['ssim']:.2f}"
            })

        self.scheduler.step()

        return {
            'loss': total_loss / num_batches,
            'psnr': total_psnr / num_batches,
            'ssim': total_ssim / num_batches
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc='Validation'):
            sinogram = batch['sinogram_noisy'].to(self.device)
            target = batch['image'].to(self.device)
            weights = batch['weights'].to(self.device)
            mask = batch['mask'].to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(sinogram, weights, mask)
                losses = self.loss_fn(
                    outputs['reconstruction'], target,
                    sinogram, weights
                )

            metrics = compute_metrics(outputs['reconstruction'], target)

            total_loss += losses['total'].item()
            total_psnr += metrics['psnr']
            total_ssim += metrics['ssim']
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'psnr': total_psnr / num_batches,
            'ssim': total_ssim / num_batches
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if not self.use_sam else self.optimizer.base_optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # Save regular checkpoint
        torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pt')

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if not self.use_sam:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.optimizer.base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch'], checkpoint['metrics']

    def train(self, num_epochs: int, start_epoch: int = 0):
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
            start_epoch: Starting epoch (for resuming)
        """
        history = {
            'train_loss': [], 'train_psnr': [], 'train_ssim': [],
            'val_loss': [], 'val_psnr': [], 'val_ssim': []
        }

        for epoch in range(start_epoch, num_epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            history['train_loss'].append(train_metrics['loss'])
            history['train_psnr'].append(train_metrics['psnr'])
            history['train_ssim'].append(train_metrics['ssim'])

            # Validation
            val_metrics = self.validate()
            history['val_loss'].append(val_metrics['loss'])
            history['val_psnr'].append(val_metrics['psnr'])
            history['val_ssim'].append(val_metrics['ssim'])

            # Check if best
            is_best = val_metrics['psnr'] > self.best_psnr
            if is_best:
                self.best_psnr = val_metrics['psnr']
                self.best_ssim = val_metrics['ssim']

            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 50) == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)

            # Log
            print(f"\nEpoch {epoch}: "
                  f"Train Loss={train_metrics['loss']:.4f}, PSNR={train_metrics['psnr']:.2f}, SSIM={train_metrics['ssim']:.2f} | "
                  f"Val Loss={val_metrics['loss']:.4f}, PSNR={val_metrics['psnr']:.2f}, SSIM={val_metrics['ssim']:.2f}")

            # Save history
            with open(self.log_dir / 'history.json', 'w') as f:
                json.dump(history, f)

        return history


def train_ct_pinn_dadif(config: Dict[str, Any]):
    """
    Main training function.

    Args:
        config: Training configuration dictionary

    Returns:
        Training history
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        batch_size=config.get('batch_size', 4),
        num_workers=config.get('num_workers', 4)
    )

    # Create model
    print("Creating model...")
    model = create_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss function
    loss_fn = create_loss(config)

    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Train
    print("Starting training...")
    history = trainer.train(
        num_epochs=config.get('num_epochs', 600),
        start_epoch=0
    )

    print(f"\nTraining complete!")
    print(f"Best PSNR: {trainer.best_psnr:.2f} dB")
    print(f"Best SSIM: {trainer.best_ssim:.2f} %")

    return history


# Default configuration
DEFAULT_CONFIG = {
    # Model
    'img_size': 256,
    'num_angles': 180,
    'num_detectors': None,
    'base_channels': 64,
    'latent_dim': 128,
    'context_dim': 256,
    'num_diffusion_steps': 12,
    'lambda_phys_lpce': 0.3,
    'lambda_phys_pace': 0.1,
    'use_final_dc': True,

    # Data
    'dataset_type': 'simulated',
    'num_train_samples': 1000,
    'num_val_samples': 200,
    'num_test_samples': 200,
    'phantom_type': 'mixed',
    'noise_level': 'low',
    'acquisition_type': 'full',  # 'full', 'sparse', 'limited'
    'num_views': 60,  # for sparse-view

    # Training
    'batch_size': 4,
    'num_epochs': 600,
    'learning_rate': 6e-3,
    'use_sam': True,
    'sam_rho': 0.05,
    'use_amp': True,
    'num_workers': 4,

    # Loss weights - FIXED: CT-optimized weights (physics is primary)
    'alpha': 0.4,  # pixel loss (reduced for CT)
    'beta': 0.1,   # perceptual loss (reduced - VGG not ideal for CT)
    'gamma': 0.5,  # physics loss (increased - physics is primary for CT)
    'tv_weight': 1e-4,
    'nonneg_weight': 1e-3,
    'use_poisson': False,
    'use_perceptual': False,  # Disabled by default - VGG trained on natural images

    # Checkpointing
    'checkpoint_dir': 'experiments/checkpoints',
    'log_dir': 'experiments/logs',
    'save_every': 50
}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train CT-PINN-DADif')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--epochs', type=int, default=600, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=6e-3, help='Learning rate')
    parser.add_argument('--acquisition', type=str, default='full',
                        choices=['full', 'sparse', 'limited'],
                        help='CT acquisition type')
    args = parser.parse_args()

    # Load or create config
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG.copy()

    # Override with command line arguments
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['acquisition_type'] = args.acquisition

    # Train
    train_ct_pinn_dadif(config)
