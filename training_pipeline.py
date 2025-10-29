import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm
import torchvision.models as models
from lpce import LatentPhysicsConstrainedEncoder
from pace import PhysicsAwareContextEncoder
from adrn import AdaptiveDiffusionRefinementNetwork
from art import AdaptiveReconstructionTransformer


# Assuming the previous modules are imported
# from lpce import LatentPhysicsConstrainedEncoder
# from pace import PhysicsAwareContextEncoder
# from adrn import AdaptiveDiffusionRefinementNetwork
# from art import AdaptiveReconstructionTransformer


class ComplexToReal(nn.Module):
    """Convert complex k-space data to 2-channel real representation"""
    def forward(self, x):
        if torch.is_complex(x):
            return torch.stack([x.real, x.imag], dim=1)
        return x


class PINNDADif(nn.Module):
    """
    Complete PINN-DADif model integrating all components.
    """
    def __init__(self,
                 in_channels=2,
                 lpce_hidden=64,
                 lpce_latent=128,
                 pace_hidden=256,
                 pace_out=256,
                 adrn_channels=256,
                 art_hidden=512,
                 lambda_phys=0.2,
                 num_diffusion_steps=10):
        super().__init__()
        
        # Convert complex k-space to real 2-channel
        self.complex_to_real = ComplexToReal()
        
        # LPCE: Latent Physics-Constrained Encoder
        self.lpce = LatentPhysicsConstrainedEncoder(
            in_channels=in_channels,
            hidden_channels=lpce_hidden,
            latent_dim=lpce_latent,
            lambda_phys=lambda_phys,
            use_sequence_specific=True
        )
        
        # PACE: Physics-Aware Context Encoder
        self.pace = PhysicsAwareContextEncoder(
            in_channels=lpce_latent,
            hidden_channels=pace_hidden,
            out_channels=pace_out,
            lambda_phys=lambda_phys
        )
        
        # ADRN: Adaptive Diffusion Refinement Network
        self.adrn = AdaptiveDiffusionRefinementNetwork(
            in_channels=pace_out,
            model_channels=64,
            num_timesteps=1000,
            num_inference_steps=num_diffusion_steps,
            lambda_phys=lambda_phys
        )
        
        # ART: Adaptive Reconstruction Transformer
        self.art = AdaptiveReconstructionTransformer(
            pace_channels=pace_out,
            adrn_channels=adrn_channels,
            hidden_channels=art_hidden,
            out_channels=1,
            lambda_phys=lambda_phys
        )
    
    def forward(self, x_under, sequence_type='mixed', k_measured=None, 
                mask=None, mode='train'):
        """
        Complete forward pass through PINN-DADif.
        
        Args:
            x_under: Undersampled k-space (B, 2, H, W) or complex (B, H, W)
            sequence_type: 't1', 't2', 'pd', or 'mixed'
            k_measured: Measured k-space for data consistency
            mask: Sampling mask
            mode: 'train' or 'inference'
        """
        # Ensure 2-channel real representation
        if x_under.shape[1] != 2:
            x_under = self.complex_to_real(x_under)
        
        # LPCE: Extract latent features
        z_latent = self.lpce(x_under, sequence_type)
        
        # PACE: Extract context-aware features
        z_pace = self.pace(z_latent)
        
        # ADRN: Refine with adaptive diffusion
        z_adrn = self.adrn(z_pace, k_measured, mask, mode=mode)
        
        # ART: Final reconstruction
        output = self.art(z_pace, z_adrn)
        
        return output


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    Implements Equation 28 from the paper.
    """
    def __init__(self, layers=[3, 8, 15, 22]):
        super().__init__()
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).features
        self.vgg = nn.ModuleList([vgg[:l+1] for l in layers])
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.layers = layers
    
    def forward(self, pred, target):
        """
        Compute perceptual loss.
        L_perc = Σ ||φ_i(I_REC) - φ_i(I_GT)||²
        """
        # Convert single channel to 3 channels for VGG
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        loss = 0.0
        for vgg_layer in self.vgg:
            pred = vgg_layer(pred)
            target = vgg_layer(target)
            loss += F.mse_loss(pred, target)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Complete loss function combining all components.
    Implements Equation 30 from the paper.
    """
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        
        self.alpha = alpha  # Pixel-wise loss weight
        self.beta = beta    # Perceptual loss weight
        self.gamma = gamma  # Physics-informed loss weight
        
        self.perceptual_loss = PerceptualLoss()
    
    def pixel_loss(self, pred, target):
        """
        L2 pixel-wise loss (Equation 27).
        L_pixel = ||I_REC - I_GT||²
        """
        return F.mse_loss(pred, target)
    
    def physics_loss(self, pred, k_measured, mask):
        """
        Physics-informed loss (Equation 29).
        L_phys = λ_phys * R_phys(I_REC)
        """
        # K-space consistency
        pred_kspace = torch.fft.fft2(pred.squeeze(1), norm='ortho')
        measured_kspace = torch.complex(k_measured[..., 0], k_measured[..., 1])
        
        # Data consistency loss
        kspace_loss = F.mse_loss(
            pred_kspace * mask.squeeze(1),
            measured_kspace * mask.squeeze(1)
        )
        
        # Gradient smoothness
        dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        smoothness = torch.mean(dx**2) + torch.mean(dy**2)
        
        return kspace_loss + 0.1 * smoothness
    
    def forward(self, pred, target, k_measured=None, mask=None):
        """
        Total loss (Equation 30).
        L_total = α*L_pixel + β*L_perc + γ*L_phys
        """
        # Pixel-wise loss
        l_pixel = self.pixel_loss(pred, target)
        
        # Perceptual loss
        l_perc = self.perceptual_loss(pred, target)
        
        # Physics-informed loss
        if k_measured is not None and mask is not None:
            l_phys = self.physics_loss(pred, k_measured, mask)
        else:
            l_phys = torch.tensor(0.0, device=pred.device)
        
        # Combined loss
        total_loss = (self.alpha * l_pixel + 
                     self.beta * l_perc + 
                     self.gamma * l_phys)
        
        return {
            'total': total_loss,
            'pixel': l_pixel,
            'perceptual': l_perc,
            'physics': l_phys
        }


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.
    Implements Equations 31-32 from the paper.
    """
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step: perturb parameters (Equation 31).
        θ+ = θ + ρ * ∇_θ L(θ) / ||∇_θ L(θ)||
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                # Save old parameters
                self.state[p]['old_p'] = p.data.clone()
                # Perturb: θ+ = θ + ρ * gradient / ||gradient||
                e_w = p.grad * scale
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step: update at perturbed location (Equation 32).
        θ = θ - η*∇_θ L(θ+)
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Restore original parameters
                p.data = self.state[p]['old_p']
        
        # Update with base optimizer
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    def _grad_norm(self):
        """Compute gradient norm for scaling"""
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def step(self, closure=None):
        """Not used - use first_step() and second_step() instead"""
        raise NotImplementedError("Use first_step() and second_step()")


class MRIDataset(Dataset):
    """
    Dataset for MRI reconstruction from k-space data.
    """
    def __init__(self, data_path, acceleration=4, sequence_type='mixed'):
        """
        Args:
            data_path: Path to h5 files
            acceleration: Undersampling rate (4x, 8x, etc.)
            sequence_type: 't1', 't2', 'pd', or 'mixed'
        """
        self.data_path = Path(data_path)
        self.acceleration = acceleration
        self.sequence_type = sequence_type
        
        # Load file list
        self.files = sorted(list(self.data_path.glob('*.h5')))
        print(f"Found {len(self.files)} MRI files")
    
    def __len__(self):
        return len(self.files)
    
    def undersample_kspace(self, kspace, acceleration):
        """Create undersampled k-space with random mask"""
        _, _, h, w = kspace.shape
        
        # Random variable-density mask
        mask = self._create_mask(h, w, acceleration)
        
        # Apply mask
        kspace_under = kspace * mask
        
        return kspace_under, mask
    
    def _create_mask(self, h, w, acceleration):
        """Create variable-density random undersampling mask"""
        # Center fraction (fully sampled)
        center_fraction = 0.08
        
        # Create mask
        mask = torch.zeros(1, 1, h, w)
        
        # Fully sample center
        center_h = int(h * center_fraction)
        mask[:, :, h//2 - center_h//2:h//2 + center_h//2, :] = 1
        
        # Random sampling for rest
        prob = (1 - center_fraction) / acceleration
        random_mask = torch.rand(1, 1, h, w) < prob
        mask = torch.logical_or(mask, random_mask).float()
        
        return mask
    
    def __getitem__(self, idx):
        """Load MRI data"""
        file_path = self.files[idx]
        
        with h5py.File(file_path, 'r') as f:
            # Load k-space data (assuming stored as 'kspace')
            kspace = torch.tensor(f['kspace'][:], dtype=torch.complex64)
            
            # Get sequence type if stored
            seq_type = f.attrs.get('sequence', self.sequence_type)
        
        # Add batch and channel dimensions
        kspace = kspace.unsqueeze(0)  # (1, H, W)
        
        # Create undersampled version
        kspace_under, mask = self.undersample_kspace(kspace, self.acceleration)
        
        # Ground truth image (inverse FFT of full k-space)
        image_gt = torch.fft.ifft2(kspace, norm='ortho').abs()
        
        # Normalize to [0, 1]
        image_gt = (image_gt - image_gt.min()) / (image_gt.max() - image_gt.min() + 1e-8)
        
        # Convert k-space to 2-channel real representation
        kspace_under_real = torch.stack([kspace_under.real, kspace_under.imag], dim=1)
        kspace_full_real = torch.stack([kspace.real, kspace.imag], dim=1)
        
        return {
            'kspace_under': kspace_under_real.squeeze(0),  # (2, H, W)
            'kspace_full': kspace_full_real.squeeze(0),    # (2, H, W)
            'image_gt': image_gt,                          # (1, H, W)
            'mask': mask.squeeze(0),                       # (1, H, W)
            'sequence': seq_type
        }


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Training loop for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        kspace_under = batch['kspace_under'].to(device)
        kspace_full = batch['kspace_full'].to(device)
        image_gt = batch['image_gt'].to(device)
        mask = batch['mask'].to(device)
        sequence = batch['sequence'][0] if isinstance(batch['sequence'], list) else 'mixed'
        
        # First forward-backward pass (SAM)
        def closure():
            optimizer.zero_grad()
            
            # Forward pass
            output = model(kspace_under, sequence, kspace_full, mask, mode='train')
            
            # Compute loss
            losses = criterion(output, image_gt, kspace_full, mask)
            
            # Backward
            losses['total'].backward()
            return losses
        
        # First step: compute gradient at current parameters
        losses = closure()
        optimizer.first_step(zero_grad=True)
        
        # Second step: compute gradient at perturbed parameters and update
        closure()
        optimizer.second_step(zero_grad=True)
        
        # Update progress bar
        total_loss += losses['total'].item()
        pbar.set_postfix({
            'loss': losses['total'].item(),
            'pixel': losses['pixel'].item(),
            'perc': losses['perceptual'].item(),
            'phys': losses['physics'].item()
        })
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validation loop"""
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            kspace_under = batch['kspace_under'].to(device)
            kspace_full = batch['kspace_full'].to(device)
            image_gt = batch['image_gt'].to(device)
            mask = batch['mask'].to(device)
            sequence = batch['sequence'][0] if isinstance(batch['sequence'], list) else 'mixed'
            
            # Forward pass
            output = model(kspace_under, sequence, kspace_full, mask, mode='inference')
            
            # Compute loss
            losses = criterion(output, image_gt, kspace_full, mask)
            total_loss += losses['total'].item()
            
            # Compute metrics
            psnr = compute_psnr(output, image_gt)
            ssim = compute_ssim(output, image_gt)
            
            total_psnr += psnr
            total_ssim += ssim
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'psnr': total_psnr / n,
        'ssim': total_ssim / n
    }


def compute_psnr(pred, target):
    """Compute PSNR"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def compute_ssim(pred, target, window_size=11):
    """Compute SSIM (simplified version)"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean() * 100  # Return as percentage


def main():
    """Main training script"""
    # Hyperparameters (from paper)
    config = {
        'batch_size': 4,
        'num_epochs': 600,
        'learning_rate': 6e-3,
        'weight_decay': 1e-4,
        'acceleration': 4,
        'num_diffusion_steps': 10,
        'lambda_phys': 0.2,
        'rho': 0.05,  # SAM perturbation radius
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = MRIDataset('path/to/train/data', 
                              acceleration=config['acceleration'])
    val_dataset = MRIDataset('path/to/val/data', 
                            acceleration=config['acceleration'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                          shuffle=False, num_workers=4)
    
    # Create model
    model = PINNDADif(
        lambda_phys=config['lambda_phys'],
        num_diffusion_steps=config['num_diffusion_steps']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    criterion = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)
    
    # SAM optimizer with Adam base
    base_optimizer = torch.optim.Adam
    optimizer = SAM(
        model.parameters(),
        base_optimizer,
        lr=config['learning_rate'],
        rho=config['rho'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer.base_optimizer, 
                                               step_size=125, gamma=0.5)
    
    # Training loop
    best_psnr = 0
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, 
                                criterion, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val PSNR: {val_metrics['psnr']:.2f} dB")
        print(f"Val SSIM: {val_metrics['ssim']:.2f} %")
        
        # Save best model
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
                'config': config
            }, 'best_model.pth')
            print(f"✓ Saved best model (PSNR: {best_psnr:.2f} dB)")
        
        # Save checkpoint every 50 epochs
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, f'checkpoint_epoch_{epoch}.pth')
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print("="*50)


if __name__ == "__main__":
    main()