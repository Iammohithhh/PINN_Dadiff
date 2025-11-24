#!/usr/bin/env python3
"""
Simple Training Script for CT-PINN-DADif
=========================================

This script provides an easy way to train and validate the CT reconstruction model.

Usage:
    python run_training.py                    # Full training (default settings)
    python run_training.py --mode demo        # Quick demo (5 epochs)
    python run_training.py --acquisition sparse --views 60  # Sparse-view CT
    python run_training.py --noise high       # High noise scenario
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Train CT-PINN-DADif')

    # Mode
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'demo', 'test', 'visualize'],
                        help='Mode: train, demo (quick test), test (evaluate), visualize')

    # Model settings
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--num_angles', type=int, default=180, help='Number of projection angles')

    # Data settings
    parser.add_argument('--acquisition', type=str, default='full',
                        choices=['full', 'sparse', 'limited'],
                        help='CT acquisition type')
    parser.add_argument('--views', type=int, default=60, help='Number of views (for sparse)')
    parser.add_argument('--noise', type=str, default='low',
                        choices=['none', 'low', 'medium', 'high'],
                        help='Noise level')
    parser.add_argument('--phantom', type=str, default='mixed',
                        choices=['shepp_logan', 'random', 'mixed'],
                        help='Phantom type for simulated data')

    # Training settings
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=6e-3, help='Learning rate')
    parser.add_argument('--no_sam', action='store_true', help='Disable SAM optimizer')

    # Checkpointing
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Output directory')

    args = parser.parse_args()

    # Set default epochs based on mode
    if args.epochs is None:
        args.epochs = 5 if args.mode == 'demo' else 100

    # Run appropriate mode
    if args.mode in ['train', 'demo']:
        train_model(args)
    elif args.mode == 'test':
        test_model(args)
    elif args.mode == 'visualize':
        visualize_results(args)


def create_config(args):
    """Create configuration from command line arguments."""
    return {
        # Model
        'img_size': args.img_size,
        'num_angles': args.num_angles,
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
        'num_train_samples': 100 if args.mode == 'demo' else 500,
        'num_val_samples': 20 if args.mode == 'demo' else 100,
        'num_test_samples': 20 if args.mode == 'demo' else 100,
        'phantom_type': args.phantom,
        'noise_level': args.noise,
        'acquisition_type': args.acquisition,
        'num_views': args.views,

        # Training
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'use_sam': not args.no_sam,
        'sam_rho': 0.05,
        'use_amp': True,
        'num_workers': 2,

        # Loss
        'alpha': 0.5,
        'beta': 0.2,
        'gamma': 0.3,
        'tv_weight': 1e-4,
        'nonneg_weight': 1e-3,
        'use_poisson': False,
        'use_perceptual': False,

        # Output
        'checkpoint_dir': f'{args.output_dir}/checkpoints',
        'log_dir': f'{args.output_dir}/logs',
        'save_every': 25
    }


def train_model(args):
    """Train the CT-PINN-DADif model."""
    from model import CT_PINN_DADif, CTReconstructionLoss, SAM, create_model, create_loss
    from data_loader import create_dataloaders
    from train import Trainer, compute_metrics

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"CT-PINN-DADif Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create config
    config = create_config(args)

    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Acquisition: {config['acquisition_type']}")
    print(f"  Noise level: {config['noise_level']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  SAM optimizer: {config['use_sam']}")

    # Create output directories
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['log_dir']).mkdir(parents=True, exist_ok=True)

    # Save config
    with open(f"{config['log_dir']}/config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Create dataloaders
    print(f"\nCreating datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")

    # Create model
    print(f"\nCreating model...")
    model = create_model(config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1

    # Create loss and trainer
    loss_fn = create_loss(config).to(device)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Train
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}\n")

    history = trainer.train(
        num_epochs=config['num_epochs'],
        start_epoch=start_epoch
    )

    # Save final results
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best PSNR: {trainer.best_psnr:.2f} dB")
    print(f"Best SSIM: {trainer.best_ssim:.2f} %")
    print(f"\nCheckpoints saved to: {config['checkpoint_dir']}")
    print(f"Logs saved to: {config['log_dir']}")

    # Plot training curves
    plot_training_history(history, config['log_dir'])

    return history


def test_model(args):
    """Test a trained model."""
    from model import create_model
    from data_loader import create_dataloaders
    from train import compute_metrics

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = create_config(args)

    print(f"\n{'='*60}")
    print(f"Testing CT-PINN-DADif")
    print(f"{'='*60}")

    # Load model
    checkpoint_path = args.checkpoint or f"{args.output_dir}/checkpoints/best_model.pt"
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Please train a model first or provide --checkpoint path")
        return

    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_model(checkpoint.get('config', config))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create test loader
    _, _, test_loader = create_dataloaders(config, batch_size=1, num_workers=2)

    # Evaluate
    psnr_list = []
    ssim_list = []

    print(f"\nEvaluating on {len(test_loader.dataset)} test samples...")

    with torch.no_grad():
        for batch in tqdm(test_loader):
            sinogram = batch['sinogram_noisy'].to(device)
            target = batch['image'].to(device)
            weights = batch['weights'].to(device)
            mask = batch['mask'].to(device)

            outputs = model(sinogram, weights, mask)
            metrics = compute_metrics(outputs['reconstruction'], target)

            psnr_list.append(metrics['psnr'])
            ssim_list.append(metrics['ssim'])

    print(f"\n{'='*60}")
    print(f"Test Results")
    print(f"{'='*60}")
    print(f"PSNR: {np.mean(psnr_list):.2f} +/- {np.std(psnr_list):.2f} dB")
    print(f"SSIM: {np.mean(ssim_list):.2f} +/- {np.std(ssim_list):.2f} %")


def visualize_results(args):
    """Visualize reconstruction results."""
    from model import create_model
    from data_loader import create_dataloaders
    from train import compute_metrics

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = create_config(args)

    # Load model
    checkpoint_path = args.checkpoint or f"{args.output_dir}/checkpoints/best_model.pt"
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(checkpoint.get('config', config))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Get test samples
    _, _, test_loader = create_dataloaders(config, batch_size=4, num_workers=2)
    batch = next(iter(test_loader))

    with torch.no_grad():
        sinogram = batch['sinogram_noisy'].to(device)
        target = batch['image'].to(device)
        weights = batch['weights'].to(device)
        mask = batch['mask'].to(device)
        fbp_recon = batch['fbp'].to(device)

        outputs = model(sinogram, weights, mask)

    # Plot
    n_samples = min(4, sinogram.shape[0])
    fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4*n_samples))

    for i in range(n_samples):
        # Ground truth
        axes[i, 0].imshow(target[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 0].set_title('Ground Truth')
        axes[i, 0].axis('off')

        # Sinogram
        axes[i, 1].imshow(sinogram[i, 0].cpu().numpy(), cmap='gray', aspect='auto')
        axes[i, 1].set_title('Sinogram')

        # FBP
        fbp_metrics = compute_metrics(fbp_recon[i:i+1], target[i:i+1])
        axes[i, 2].imshow(fbp_recon[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 2].set_title(f'FBP (PSNR: {fbp_metrics["psnr"]:.1f})')
        axes[i, 2].axis('off')

        # Reconstruction
        rec = outputs['reconstruction']
        rec_metrics = compute_metrics(rec[i:i+1], target[i:i+1])
        axes[i, 3].imshow(rec[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 3].set_title(f'CT-PINN-DADif (PSNR: {rec_metrics["psnr"]:.1f})')
        axes[i, 3].axis('off')

        # Error
        error = torch.abs(rec[i, 0] - target[i, 0]).cpu().numpy()
        im = axes[i, 4].imshow(error, cmap='hot', vmin=0, vmax=0.1)
        axes[i, 4].set_title('Error Map')
        axes[i, 4].axis('off')
        plt.colorbar(im, ax=axes[i, 4], fraction=0.046)

    plt.tight_layout()
    save_path = f"{args.output_dir}/reconstructions.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")
    plt.show()


def plot_training_history(history, log_dir):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PSNR
    axes[1].plot(history['train_psnr'], label='Train', linewidth=2)
    axes[1].plot(history['val_psnr'], label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('PSNR')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # SSIM
    axes[2].plot(history['train_ssim'], label='Train', linewidth=2)
    axes[2].plot(history['val_ssim'], label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('SSIM (%)')
    axes[2].set_title('SSIM')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"{log_dir}/training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training history to: {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
