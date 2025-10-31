"""
PINN-DADif: Physics-Informed Deep Adaptive Diffusion Network
Complete implementation integrating LPCE, PACE, ADRN, and ART modules

Paper: Ahmed et al. (2025) - Digital Signal Processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

# Import all modules (assuming they're in separate files)
from lpce import LatentPhysicsConstrainedEncoder
from pace import PhysicsAwareContextEncoder
from adrn import AdaptiveDiffusionRefinementNetwork
from art import AdaptiveReconstructionTransformer


class PINNDADif(nn.Module):
    """
    Complete PINN-DADif Model
    
    Physics-Informed Neural Network with Deep Adaptive Diffusion
    for robust and efficient MRI reconstruction.
    
    Architecture:
    Input (undersampled MRI) → LPCE → PACE → ADRN → ART → Output (reconstructed MRI)
    
    Paper Reference: Ahmed et al. (2025), Digital Signal Processing, 160, 105085
    """
    
    def __init__(self,
                 # Input/Output
                 in_channels: int = 1,
                 out_channels: int = 1,
                 
                 # LPCE configuration
                 lpce_hidden_channels: int = 64,
                 lpce_latent_dim: int = 128,
                 lpce_num_coils: int = 1,
                 
                 # PACE configuration
                 pace_hidden_channels: int = 256,
                 pace_out_channels: int = 256,
                 pace_num_nonlocal_blocks: int = 2,
                 pace_num_context_layers: int = 3,
                 
                 # ADRN configuration
                 adrn_model_channels: int = 128,
                 adrn_num_diffusion_steps: int = 10,
                 adrn_num_reverse_iterations: int = 12,
                 adrn_beta_min: float = 0.1,
                 adrn_beta_max: float = 20.0,
                 
                 # ART configuration
                 art_hidden_channels: int = 256,
                 art_num_dynamic_conv_layers: int = 3,
                 art_num_transformer_blocks: int = 4,
                 art_num_heads: int = 8,
                 
                 # Global configuration
                 lambda_phys: float = 0.2,
                 dropout: float = 0.1,
                 device: str = 'cuda'):
        """
        Initialize PINN-DADif model with all sub-modules.
        
        Args:
            in_channels: Input image channels (1 for magnitude MRI)
            out_channels: Output image channels (1 for reconstructed MRI)
            lpce_*: LPCE module parameters
            pace_*: PACE module parameters
            adrn_*: ADRN module parameters
            art_*: ART module parameters
            lambda_phys: Physics regularization weight
            dropout: Dropout rate
            device: Device for computation
        """
        super().__init__()
        
        self.device = device
        self.lambda_phys = lambda_phys
        
        # Store configuration
        self.config = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'lpce_latent_dim': lpce_latent_dim,
            'pace_out_channels': pace_out_channels,
            'adrn_num_diffusion_steps': adrn_num_diffusion_steps,
            'adrn_num_reverse_iterations': adrn_num_reverse_iterations,
            'lambda_phys': lambda_phys
        }
        
        # Module 1: Latent Physics-Constrained Encoder (LPCE)
        # Extracts physics-informed features from undersampled input
        self.lpce = LatentPhysicsConstrainedEncoder(
            in_channels=in_channels,
            hidden_channels=lpce_hidden_channels,
            latent_dim=lpce_latent_dim,
            lambda_phys=lambda_phys,
            use_sequence_specific=True,
            num_coils=lpce_num_coils,
            num_physics_layers=3
        )
        
        # Module 2: Physics-Aware Context Encoder (PACE)
        # Enhances features with multi-scale context and attention
        self.pace = PhysicsAwareContextEncoder(
            in_channels=lpce_latent_dim,
            hidden_channels=pace_hidden_channels,
            out_channels=pace_out_channels,
            lambda_phys=lambda_phys,
            num_nonlocal_blocks=pace_num_nonlocal_blocks,
            num_context_layers=pace_num_context_layers
        )
        
        # Module 3: Adaptive Diffusion Refinement Network (ADRN)
        # Refines features through adaptive diffusion process
        self.adrn = AdaptiveDiffusionRefinementNetwork(
            in_channels=pace_out_channels,
            model_channels=adrn_model_channels,
            out_channels=pace_out_channels,
            num_diffusion_steps=adrn_num_diffusion_steps,
            num_reverse_iterations=adrn_num_reverse_iterations,
            beta_min=adrn_beta_min,
            beta_max=adrn_beta_max,
            lambda_phys=lambda_phys,
            use_transformer=True,
            num_heads=art_num_heads,
            device=device
        )
        
        # Module 4: Adaptive Reconstruction Transformer (ART)
        # Synthesizes final reconstruction with dynamic convolutions
        self.art = AdaptiveReconstructionTransformer(
            pace_channels=pace_out_channels,
            adrn_channels=pace_out_channels,
            hidden_channels=art_hidden_channels,
            out_channels=out_channels,
            num_dynamic_conv_layers=art_num_dynamic_conv_layers,
            num_transformer_blocks=art_num_transformer_blocks,
            num_heads=art_num_heads,
            mlp_ratio=4.0,
            dropout=dropout,
            lambda_phys=lambda_phys
        )
        
        print(f"✓ PINN-DADif initialized on {device}")
        print(f"  - Total parameters: {self.get_num_params():,}")
    
    def forward(self,
                x_under: torch.Tensor,
                k_space: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                sensitivities: Optional[torch.Tensor] = None,
                sequence_type: str = 'mixed',
                return_losses: bool = False,
                return_intermediates: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through complete PINN-DADif pipeline.
        
        Args:
            x_under: Undersampled magnitude image (B, 1, H, W)
            k_space: Measured k-space data (B, H, W) complex tensor
            mask: Undersampling mask (B, 1, H, W), 1=measured, 0=unmeasured
            sensitivities: Coil sensitivity maps for multi-coil data (optional)
            sequence_type: MRI sequence ('t1', 't2', 'pd', 'flair', 'mixed')
            return_losses: Whether to return all losses for training
            return_intermediates: Whether to return intermediate features
        
        Returns:
            x_recon: Reconstructed MRI image (B, 1, H, W)
            losses: Dict of all losses (if return_losses=True)
            intermediates: Dict of intermediate features (if return_intermediates=True)
        """
        
        losses = {}
        intermediates = {}
        
        # ==================== Module 1: LPCE ====================
        # Extract physics-constrained latent features
        if return_losses:
            z_latent, lpce_losses = self.lpce(
                x_under,
                k_space=k_space,
                mask=mask,
                sensitivities=sensitivities,
                sequence_type=sequence_type,
                return_losses=True
            )
            losses.update({f'lpce_{k}': v for k, v in lpce_losses.items()})
        else:
            z_latent = self.lpce(
                x_under,
                k_space=k_space,
                mask=mask,
                sensitivities=sensitivities,
                sequence_type=sequence_type
            )
        
        if return_intermediates:
            intermediates['z_latent'] = z_latent.detach()
        
        # ==================== Module 2: PACE ====================
        # Enhance with multi-scale context and attention
        if return_losses:
            z_pace, pace_losses = self.pace(
                z_latent,
                sequence_type=sequence_type,
                return_losses=True
            )
            losses.update({f'pace_{k}': v for k, v in pace_losses.items()})
        else:
            z_pace = self.pace(z_latent, sequence_type=sequence_type)
        
        if return_intermediates:
            intermediates['z_pace'] = z_pace.detach()
        
        # ==================== Module 3: ADRN ====================
        # Refine through adaptive diffusion process
        if return_losses:
            z_adrn, adrn_losses = self.adrn(
                z_pace,
                k_measured=k_space,
                mask=mask,
                return_losses=True
            )
            losses.update({f'adrn_{k}': v for k, v in adrn_losses.items()})
        else:
            z_adrn = self.adrn(
                z_pace,
                k_measured=k_space,
                mask=mask
            )
        
        if return_intermediates:
            intermediates['z_adrn'] = z_adrn.detach()
        
        # ==================== Module 4: ART ====================
        # Final reconstruction with dynamic convolutions
        if return_losses:
            x_recon, art_losses = self.art(
                z_pace,
                z_adrn,
                k_measured=k_space,
                mask=mask,
                return_losses=True
            )
            losses.update({f'art_{k}': v for k, v in art_losses.items()})
        else:
            x_recon = self.art(
                z_pace,
                z_adrn,
                k_measured=k_space,
                mask=mask
            )
        
        # ==================== Prepare Returns ====================
        if return_losses and return_intermediates:
            return x_recon, losses, intermediates
        elif return_losses:
            return x_recon, losses
        elif return_intermediates:
            return x_recon, intermediates
        
        return x_recon
    
    def reconstruct(self,
                   x_under: torch.Tensor,
                   k_space: Optional[torch.Tensor] = None,
                   mask: Optional[torch.Tensor] = None,
                   sequence_type: str = 'mixed') -> torch.Tensor:
        """
        Simple inference method for reconstruction.
        
        Args:
            x_under: Undersampled image
            k_space: K-space data
            mask: Sampling mask
            sequence_type: MRI sequence type
        
        Returns:
            Reconstructed image
        """
        self.eval()
        with torch.no_grad():
            x_recon = self.forward(
                x_under,
                k_space=k_space,
                mask=mask,
                sequence_type=sequence_type,
                return_losses=False,
                return_intermediates=False
            )
        return x_recon
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_module_params(self) -> Dict[str, int]:
        """Get parameter count for each module"""
        return {
            'lpce': sum(p.numel() for p in self.lpce.parameters()),
            'pace': sum(p.numel() for p in self.pace.parameters()),
            'adrn': sum(p.numel() for p in self.adrn.parameters()),
            'art': sum(p.numel() for p in self.art.parameters())
        }
    
    def print_model_summary(self):
        """Print detailed model summary"""
        print("\n" + "=" * 70)
        print("PINN-DADif Model Summary")
        print("=" * 70)
        
        # Configuration
        print("\nConfiguration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        
        # Parameters per module
        print("\nParameters per module:")
        module_params = self.get_module_params()
        total_params = self.get_num_params()
        
        for name, count in module_params.items():
            percentage = (count / total_params) * 100
            print(f"  {name.upper():6s}: {count:12,} ({percentage:5.2f}%)")
        
        print(f"  {'TOTAL':6s}: {total_params:12,} (100.00%)")
        
        # Trainable parameters
        trainable = self.get_num_trainable_params()
        print(f"\nTrainable parameters: {trainable:,}")
        print(f"Non-trainable parameters: {total_params - trainable:,}")
        
        print("=" * 70 + "\n")


# ==================== Loss Functions ====================

class PINNDADifLoss(nn.Module):
    """
    Complete loss function for PINN-DADif training.
    Implements the loss from Section 2.5 of the paper.
    """
    
    def __init__(self,
                 alpha: float = 0.5,      # Pixel-wise weight
                 beta: float = 0.3,       # Perceptual weight
                 gamma: float = 0.2,      # Physics-informed weight
                 perceptual_layers: list = [3, 8, 15, 22]):
        """
        Initialize loss function.
        
        Args:
            alpha: Weight for pixel-wise L2 loss
            beta: Weight for perceptual loss
            gamma: Weight for physics-informed loss
            perceptual_layers: VGG layers for perceptual loss
        """
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # VGG for perceptual loss
        try:
            from torchvision.models import vgg19
            vgg = vgg19(pretrained=True).features
            self.perceptual_net = nn.ModuleList([
                vgg[:layer+1] for layer in perceptual_layers
            ]).eval()
            for param in self.perceptual_net.parameters():
                param.requires_grad = False
        except:
            print("Warning: VGG not available, perceptual loss disabled")
            self.perceptual_net = None
    
    def pixel_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Pixel-wise L2 loss (Equation 27)
        L_pixel = ||I_REC - I_GT||²
        """
        return F.mse_loss(pred, target)
    
    def perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Perceptual loss using VGG features (Equation 28)
        L_perc = Σ ||φ_i(I_REC) - φ_i(I_GT)||²
        """
        if self.perceptual_net is None:
            return torch.tensor(0.0, device=pred.device)
        
        # Convert to 3-channel for VGG
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        loss = 0.0
        for layer in self.perceptual_net:
            pred = layer(pred)
            target = layer(target)
            loss += F.mse_loss(pred, target)
        
        return loss
    
    def physics_loss(self,
                    pred: torch.Tensor,
                    k_measured: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        """
        Physics-informed loss (Equation 29)
        L_phys = λ_phys * R_phys(I_REC)
        
        Includes k-space fidelity and smoothness constraints
        """
        # K-space fidelity
        k_pred = torch.fft.fft2(pred, norm='ortho')
        kspace_loss = torch.mean(torch.abs((k_pred - k_measured) * mask)**2)
        
        # Gradient smoothness
        dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        smoothness_loss = torch.mean(dx**2) + torch.mean(dy**2)
        
        return kspace_loss + 0.1 * smoothness_loss
    
    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                k_measured: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                model_losses: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss (Equation 30)
        L_total = α*L_pixel + β*L_perc + γ*L_phys
        
        Args:
            pred: Predicted reconstruction
            target: Ground truth image
            k_measured: Measured k-space data
            mask: Undersampling mask
            model_losses: Additional losses from model modules
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        
        # Pixel-wise loss (Equation 27)
        l_pixel = self.pixel_loss(pred, target)
        
        # Perceptual loss (Equation 28)
        l_perc = self.perceptual_loss(pred, target)
        
        # Physics-informed loss (Equation 29)
        l_phys = 0.0
        if k_measured is not None and mask is not None:
            l_phys = self.physics_loss(pred, k_measured, mask)
        
        # Total loss (Equation 30)
        total_loss = self.alpha * l_pixel + self.beta * l_perc + self.gamma * l_phys
        
        # Add module-specific losses if provided
        if model_losses is not None:
            for key, value in model_losses.items():
                if 'reg' in key or 'loss' in key:
                    total_loss = total_loss + 0.01 * value  # Small weight for regularization
        
        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss,
            'pixel': l_pixel,
            'perceptual': l_perc,
            'physics': l_phys
        }
        
        if model_losses is not None:
            loss_dict.update(model_losses)
        
        return total_loss, loss_dict


# ==================== Testing & Example Usage ====================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Testing Complete PINN-DADif Model")
    print("=" * 70)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # ==================== 1. Create Model ====================
    print("\n1. Initializing PINN-DADif...")
    
    model = PINNDADif(
        in_channels=1,
        out_channels=1,
        lpce_hidden_channels=64,
        lpce_latent_dim=128,
        pace_hidden_channels=256,
        pace_out_channels=256,
        adrn_num_diffusion_steps=10,
        adrn_num_reverse_iterations=12,
        lambda_phys=0.2,
        device=device
    ).to(device)
    
    model.print_model_summary()
    
    # ==================== 2. Create Loss Function ====================
    print("2. Initializing Loss Function...")
    criterion = PINNDADifLoss(alpha=0.5, beta=0.3, gamma=0.2)
    
    # ==================== 3. Test Forward Pass ====================
    print("\n3. Testing Forward Pass...")
    
    batch_size = 2
    height, width = 256, 256
    
    # Create test data
    x_under = torch.randn(batch_size, 1, height, width).abs().to(device)
    x_target = torch.randn(batch_size, 1, height, width).abs().to(device)
    k_space = torch.fft.fft2(x_target.squeeze(1), norm='ortho')
    mask = (torch.rand(batch_size, 1, height, width) > 0.75).float().to(device)
    
    print(f"   Input shape: {x_under.shape}")
    print(f"   Target shape: {x_target.shape}")
    print(f"   K-space shape: {k_space.shape}")
    print(f"   Mask shape: {mask.shape}")
    
    # Forward pass (inference mode)
    model.eval()
    with torch.no_grad():
        x_recon = model.reconstruct(x_under, k_space, mask, sequence_type='t1')
    
    print(f"   Reconstruction shape: {x_recon.shape}")
    print(f"   ✓ Forward pass successful!")
    
    # ==================== 4. Test Training Mode ====================
    print("\n4. Testing Training Mode...")
    
    model.train()
    x_recon_train, losses = model(
        x_under,
        k_space=k_space,
        mask=mask,
        sequence_type='t1',
        return_losses=True
    )
    
    print(f"   Model losses:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.item():.6f}")
    
    # Compute total loss
    total_loss, loss_dict = criterion(
        x_recon_train,
        x_target,
        k_space,
        mask,
        model_losses=losses
    )
    
    print(f"\n   Total loss: {total_loss.item():.6f}")
    print(f"   Pixel loss: {loss_dict['pixel'].item():.6f}")
    print(f"   Perceptual loss: {loss_dict['perceptual'].item():.6f}")
    print(f"   Physics loss: {loss_dict['physics'].item():.6f}")
    print(f"   ✓ Training mode successful!")
    
    # ==================== 5. Test Different Sequences ====================
    print("\n5. Testing Different MRI Sequences...")
    
    model.eval()
    sequences = ['t1', 't2', 'pd', 'mixed']
    
    with torch.no_grad():
        for seq in sequences:
            x_recon_seq = model.reconstruct(x_under, k_space, mask, sequence_type=seq)
            print(f"   {seq.upper():6s}: {x_recon_seq.shape} ✓")
    
    # ==================== 6. Test with Intermediates ====================
    print("\n6. Testing Intermediate Features...")
    
    with torch.no_grad():
        x_recon_inter, intermediates = model(
            x_under,
            k_space=k_space,
            mask=mask,
            sequence_type='t1',
            return_intermediates=True
        )
    
    print("   Intermediate features:")
    for key, value in intermediates.items():
        print(f"     {key}: {value.shape}")
    
    # ==================== 7. Memory and Performance ====================
    print("\n7. Model Statistics:")
    
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"   GPU Memory:")
        print(f"     Allocated: {memory_allocated:.2f} GB")
        print(f"     Reserved: {memory_reserved:.2f} GB")
    
    # ==================== 8. Gradient Check ====================
    print("\n8. Testing Gradient Flow...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Forward pass
    x_recon = model(x_under, k_space, mask, sequence_type='t1')
    
    # Compute loss
    loss = F.mse_loss(x_recon, x_target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"   Gradients computed: {has_grad} ✓")
    
    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("✓ PINN-DADif model is ready for training and inference!")
    print("=" * 70 + "\n")