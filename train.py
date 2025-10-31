"""
PINN-DADif Training Pipeline
Complete training, validation, and testing implementation

Paper: Ahmed et al. (2025) - Digital Signal Processing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from pinn_dadif import PINNDADif, PINNDADifLoss


# ==================== Dataset ====================

class MRIDataset(Dataset):
    """
    MRI Dataset for PINN-DADif training
    
    Supports both private dataset and fastMRI format
    """
    
    def __init__(self,
                 data_path: str,
                 mode: str = 'train',
                 acceleration_factor: int = 4,
                 sequence_types: list = ['t1', 't2', 'pd'],
                 image_size: Tuple[int, int] = (256, 256),
                 normalize: bool = True):
        """
        Args:
            data_path: Path to dataset
            mode: 'train', 'val', or 'test'
            acceleration_factor: Undersampling factor (4, 8, 12)
            sequence_types: List of MRI sequences to include
            image_size: Target image size (H, W)
            normalize: Whether to normalize images
        """
        self.data_path = Path(data_path)
        self.mode = mode
        self.R = acceleration_factor
        self.sequence_types = sequence_types
        self.image_size = image_size
        self.normalize = normalize
        
        # Load file list
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} {mode} samples (R={self.R}x)")
    
    def _load_samples(self):
        """Load list of data samples"""
        samples = []
        
        for seq_type in self.sequence_types:
            seq_path = self.data_path / self.mode / seq_type
            if seq_path.exists():
                files = sorted(list(seq_path.glob('*.npy')))
                samples.extend([(f, seq_type) for f in files])
        
        return samples
    
    def _create_undersampling_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Create variable-density random undersampling mask
        Higher density at k-space center
        """
        H, W = shape
        mask = np.zeros((H, W), dtype=np.float32)
        
        # Center fraction (fully sampled)
        center_fraction = 0.08
        center_size = int(W * center_fraction)
        center_start = W // 2 - center_size // 2
        mask[:, center_start:center_start + center_size] = 1.0
        
        # Random sampling for remaining
        num_sampled = int((W * H) / self.R)
        num_remaining = num_sampled - (H * center_size)
        
        # Variable density (higher probability near center)
        prob = np.zeros((H, W))
        for i in range(W):
            dist = abs(i - W // 2) / (W // 2)
            prob[:, i] = np.exp(-3 * dist)  # Exponential decay
        
        prob[:, center_start:center_start + center_size] = 0  # Already sampled
        prob = prob.flatten()
        prob = prob / prob.sum()
        
        indices = np.random.choice(H * W, num_remaining, replace=False, p=prob)
        mask.flat[indices] = 1.0
        
        return mask
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            Dictionary containing:
            - 'undersampled': Undersampled magnitude image (1, H, W)
            - 'target': Fully-sampled ground truth (1, H, W)
            - 'k_space': Measured k-space data (H, W) complex
            - 'mask': Undersampling mask (1, H, W)
            - 'sequence': Sequence type string
        """
        file_path, seq_type = self.samples[idx]
        
        # Load fully-sampled image
        image = np.load(file_path).astype(np.float32)
        
        # Resize if needed
        if image.shape != self.image_size:
            from scipy.ndimage import zoom
            scale = (self.image_size[0] / image.shape[0],
                    self.image_size[1] / image.shape[1])
            image = zoom(image, scale, order=1)
        
        # Normalize
        if self.normalize:
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Create k-space
        k_space = np.fft.fft2(image)
        k_space = np.fft.fftshift(k_space)
        
        # Create undersampling mask
        mask = self._create_undersampling_mask(image.shape)
        
        # Apply mask
        k_space_under = k_space * mask
        
        # Inverse FFT to get undersampled image
        k_space_under = np.fft.ifftshift(k_space_under)
        image_under = np.fft.ifft2(k_space_under)
        image_under = np.abs(image_under).astype(np.float32)
        
        # Convert to tensors
        return {
            'undersampled': torch.from_numpy(image_under).unsqueeze(0),  # (1, H, W)
            'target': torch.from_numpy(image).unsqueeze(0),  # (1, H, W)
            'k_space': torch.from_numpy(k_space),  # (H, W) complex
            'mask': torch.from_numpy(mask).unsqueeze(0),  # (1, H, W)
            'sequence': seq_type,
            'filename': file_path.name
        }


# ==================== Metrics ====================

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """
    Calculate Structural Similarity Index Measure
    Simplified implementation
    """
    from torch.nn.functional import conv2d
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                          for x in range(window_size)])
    window = gauss / gauss.sum()
    window = window.unsqueeze(1)
    window = window.mm(window.t()).unsqueeze(0).unsqueeze(0)
    window = window.to(pred.device)
    
    # Compute statistics
    mu1 = conv2d(pred, window, padding=window_size//2, groups=1)
    mu2 = conv2d(target, window, padding=window_size//2, groups=1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = conv2d(pred * pred, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = conv2d(target * target, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = conv2d(pred * target, window, padding=window_size//2, groups=1) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


class MetricsTracker:
    """Track and compute metrics during training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.psnr_list = []
        self.ssim_list = []
        self.loss_list = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, loss: float):
        psnr = calculate_psnr(pred, target)
        ssim = calculate_ssim(pred, target)
        
        self.psnr_list.append(psnr)
        self.ssim_list.append(ssim)
        self.loss_list.append(loss)
    
    def get_average(self) -> Dict[str, float]:
        return {
            'psnr': np.mean(self.psnr_list),
            'ssim': np.mean(self.ssim_list),
            'loss': np.mean(self.loss_list)
        }


# ==================== Trainer ====================

class PINNDADifTrainer:
    """
    Complete training pipeline for PINN-DADif
    """
    
    def __init__(self,
                 model: PINNDADif,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: PINNDADifLoss,
                 optimizer: optim.Optimizer,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler],
                 config: Dict,
                 device: str = 'cuda'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # Tracking
        self.epoch = 0
        self.global_step = 0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        
        print(f"\n✓ Trainer initialized")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        metrics = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.config['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            x_under = batch['undersampled'].to(self.device)
            x_target = batch['target'].to(self.device)
            k_space = batch['k_space'].to(self.device)
            mask = batch['mask'].to(self.device)
            seq_type = batch['sequence'][0] if len(batch['sequence']) == 1 else 'mixed'
            
            # Forward pass
            x_recon, model_losses = self.model(
                x_under,
                k_space=k_space,
                mask=mask,
                sequence_type=seq_type,
                return_losses=True
            )
            
            # Compute loss
            total_loss, loss_dict = self.criterion(
                x_recon,
                x_target,
                k_space,
                mask,
                model_losses=model_losses
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                metrics.update(x_recon, x_target, total_loss.item())
            
            # Log to TensorBoard
            if self.global_step % self.config['log_interval'] == 0:
                self.writer.add_scalar('train/total_loss', total_loss.item(), self.global_step)
                self.writer.add_scalar('train/pixel_loss', loss_dict['pixel'].item(), self.global_step)
                self.writer.add_scalar('train/perceptual_loss', loss_dict['perceptual'].item(), self.global_step)
                self.writer.add_scalar('train/physics_loss', loss_dict['physics'].item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'psnr': f"{metrics.psnr_list[-1]:.2f}",
                'ssim': f"{metrics.ssim_list[-1]:.4f}"
            })
        
        return metrics.get_average()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        metrics = MetricsTracker()
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            # Move to device
            x_under = batch['undersampled'].to(self.device)
            x_target = batch['target'].to(self.device)
            k_space = batch['k_space'].to(self.device)
            mask = batch['mask'].to(self.device)
            seq_type = batch['sequence'][0] if len(batch['sequence']) == 1 else 'mixed'
            
            # Forward pass
            x_recon = self.model.reconstruct(
                x_under,
                k_space=k_space,
                mask=mask,
                sequence_type=seq_type
            )
            
            # Compute loss
            total_loss, _ = self.criterion(x_recon, x_target, k_space, mask)
            
            # Update metrics
            metrics.update(x_recon, x_target, total_loss.item())
            
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'psnr': f"{metrics.psnr_list[-1]:.2f}",
                'ssim': f"{metrics.ssim_list[-1]:.4f}"
            })
        
        return metrics.get_average()
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'config': self.config
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (PSNR: {self.best_psnr:.2f} dB)")
        
        # Save epoch checkpoint
        if (self.epoch + 1) % self.config['save_interval'] == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch+1:03d}.pth'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_psnr = checkpoint['best_psnr']
        self.best_ssim = checkpoint['best_ssim']
        
        print(f"✓ Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Complete training loop"""
        print("\n" + "=" * 70)
        print("Starting Training")
        print("=" * 70)
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, self.config['num_epochs']):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['psnr'])
                else:
                    self.scheduler.step()
            
            # Log to TensorBoard
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/train_psnr', train_metrics['psnr'], epoch)
            self.writer.add_scalar('epoch/train_ssim', train_metrics['ssim'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/val_psnr', val_metrics['psnr'], epoch)
            self.writer.add_scalar('epoch/val_ssim', val_metrics['ssim'], epoch)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} - {epoch_time:.1f}s")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"PSNR: {train_metrics['psnr']:.2f} dB, "
                  f"SSIM: {train_metrics['ssim']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"PSNR: {val_metrics['psnr']:.2f} dB, "
                  f"SSIM: {val_metrics['ssim']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['psnr'] > self.best_psnr
            if is_best:
                self.best_psnr = val_metrics['psnr']
                self.best_ssim = val_metrics['ssim']
            
            self.save_checkpoint(is_best)
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Best SSIM: {self.best_ssim:.4f}")
        print("=" * 70)
        
        self.writer.close()


# ==================== Main Training Script ====================

def main(args):
    """Main training function"""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"PINN-DADif Training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = MRIDataset(
        data_path=args.data_path,
        mode='train',
        acceleration_factor=args.acceleration,
        sequence_types=args.sequences,
        image_size=(args.image_size, args.image_size),
        normalize=True
    )
    
    val_dataset = MRIDataset(
        data_path=args.data_path,
        mode='val',
        acceleration_factor=args.acceleration,
        sequence_types=args.sequences,
        image_size=(args.image_size, args.image_size),
        normalize=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print("\nInitializing model...")
    model = PINNDADif(
        in_channels=1,
        out_channels=1,
        lpce_latent_dim=128,
        pace_out_channels=256,
        adrn_num_diffusion_steps=args.diffusion_steps,
        adrn_num_reverse_iterations=args.reverse_iterations,
        lambda_phys=args.lambda_phys,
        device=device
    ).to(device)
    
    model.print_model_summary()
    
    # Loss function
    criterion = PINNDADifLoss(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    
    # Optimizer (SAM from paper)
    base_optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        base_optimizer,
        step_size=args.step_size,
        gamma=0.1
    )
    
    # Configuration
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'acceleration': args.acceleration,
        'output_dir': args.output_dir,
        'log_interval': args.log_interval,
        'save_interval': args.save_interval,
        'grad_clip': args.grad_clip
    }
    
    # Create trainer
    trainer = PINNDADifTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=base_optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )
    
    # Load checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINN-DADif Training')
    
    # Data
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--sequences', nargs='+', default=['t1', 't2', 'pd'],
                       help='MRI sequences to use')
    parser.add_argument('--acceleration', type=int, default=4,
                       help='Acceleration factor (4, 8, or 12)')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size')
    
    # Model
    parser.add_argument('--diffusion_steps', type=int, default=10,
                       help='Number of diffusion steps (T/k)')
    parser.add_argument('--reverse_iterations', type=int, default=12,
                       help='Number of reverse diffusion iterations')
    parser.add_argument('--lambda_phys', type=float, default=0.2,
                       help='Physics regularization weight')
    
    # Loss
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Pixel loss weight')
    parser.add_argument('--beta', type=float, default=0.3,
                       help='Perceptual loss weight')
    parser.add_argument('--gamma', type=float, default=0.2,
                       help='Physics loss weight')
    
    # Training
    parser.add_argument('--epochs', type=int, default=600,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=6e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--step_size', type=int, default=125,
                       help='LR scheduler step size')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping')
    
    # Misc
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=50,
                       help='Checkpoint save interval')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    main(args)