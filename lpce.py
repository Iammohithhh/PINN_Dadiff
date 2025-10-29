import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhysicsInformedRegularization(nn.Module):
    """
    Physics-based regularization module that enforces MRI signal properties.
    Implements Equation 6 from the paper.
    """
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, activations, physics_model=None):
        """
        Args:
            activations: Current layer activations (B, C, H, W)
            physics_model: Optional physics model for signal behavior
        Returns:
            Regularization loss
        """
        if physics_model is None:
            # Default: enforce smoothness constraint
            # Compute gradient-based smoothness
            dx = activations[:, :, :, 1:] - activations[:, :, :, :-1]
            dy = activations[:, :, 1:, :] - activations[:, :, :-1, :]
            
            smoothness_loss = torch.mean(dx**2) + torch.mean(dy**2)
            return self.gamma * smoothness_loss
        else:
            # Use provided physics model
            expected = physics_model(activations)
            return self.gamma * torch.mean((activations - expected)**2)


class FourierTransformLayer(nn.Module):
    """
    Implements Fourier Transform operations (Equations 3 and 4).
    Converts between spatial and frequency domains.
    """
    def __init__(self):
        super().__init__()
    
    def forward_fft(self, x):
        """
        Forward Fourier Transform: spatial -> frequency domain
        Implements Equation 3
        """
        # Input: (B, C, H, W) - real-valued spatial domain
        # Output: (B, C, H, W, 2) - complex frequency domain
        
        # Apply 2D FFT
        x_freq = torch.fft.fft2(x, norm='ortho')
        
        # Convert complex to real representation (real, imag)
        x_freq_real = torch.stack([x_freq.real, x_freq.imag], dim=-1)
        
        return x_freq_real
    
    def inverse_fft(self, x_freq):
        """
        Inverse Fourier Transform: frequency -> spatial domain
        Implements Equation 4
        """
        # Input: (B, C, H, W, 2) - complex frequency domain
        # Output: (B, C, H, W) - real-valued spatial domain
        
        # Reconstruct complex tensor
        x_complex = torch.complex(x_freq[..., 0], x_freq[..., 1])
        
        # Apply 2D IFFT
        x_spatial = torch.fft.ifft2(x_complex, norm='ortho')
        
        # Return real part (imaginary should be negligible)
        return x_spatial.real


class DataDrivenBranch(nn.Module):
    """
    Data-driven feature extraction branch using CNNs.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, 
                               kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Standard convolutional feature extraction
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        return x


class PhysicsInformedBranch(nn.Module):
    """
    Physics-informed feature extraction branch.
    Models MRI physics including T1/T2 relaxation and k-space properties.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        self.fourier = FourierTransformLayer()
        
        # Process in k-space (frequency domain)
        # Input has 2 additional channels for real/imag components
        self.kspace_conv1 = nn.Conv2d(in_channels * 2, hidden_channels, 
                                      kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        self.kspace_conv2 = nn.Conv2d(hidden_channels, hidden_channels, 
                                      kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        # Project back to desired channels
        self.kspace_conv3 = nn.Conv2d(hidden_channels, in_channels * 2, 
                                      kernel_size=3, padding=1)
        
        # Spatial domain refinement after IFFT
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, 
                                      kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Physics regularization
        self.physics_reg = PhysicsInformedRegularization(gamma=0.1)
    
    def forward(self, x):
        """
        Physics-guided feature extraction through k-space
        """
        # Transform to k-space (Equation 3)
        x_kspace = self.fourier.forward_fft(x)  # (B, C, H, W, 2)
        B, C, H, W, _ = x_kspace.shape
        
        # Reshape for convolution: merge real/imag into channels
        x_kspace = x_kspace.permute(0, 1, 4, 2, 3).reshape(B, C*2, H, W)
        
        # Process in k-space domain
        x_kspace = self.relu(self.bn1(self.kspace_conv1(x_kspace)))
        x_kspace = self.relu(self.bn2(self.kspace_conv2(x_kspace)))
        x_kspace = self.kspace_conv3(x_kspace)
        
        # Reshape back to complex format
        x_kspace = x_kspace.reshape(B, C, 2, H, W).permute(0, 1, 3, 4, 2)
        
        # Transform back to spatial domain (Equation 4)
        x_spatial = self.fourier.inverse_fft(x_kspace)
        
        # Spatial refinement
        x_out = self.relu(self.bn3(self.spatial_conv(x_spatial)))
        
        return x_out


class SequenceSpecificProcessing(nn.Module):
    """
    Handles different MRI sequences (T1, T2, PD) with specialized processing.
    Implements Equation 7 from the paper.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Separate convolutional layers for each sequence type
        self.conv_t1 = nn.Conv2d(in_channels, out_channels, 
                                 kernel_size=3, padding=1)
        self.conv_t2 = nn.Conv2d(in_channels, out_channels, 
                                 kernel_size=3, padding=1)
        self.conv_pd = nn.Conv2d(in_channels, out_channels, 
                                 kernel_size=3, padding=1)
        
        # Batch normalization for each sequence
        self.bn_t1 = nn.BatchNorm2d(out_channels)
        self.bn_t2 = nn.BatchNorm2d(out_channels)
        self.bn_pd = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, sequence_type='mixed'):
        """
        Args:
            x: Input features
            sequence_type: 't1', 't2', 'pd', or 'mixed'
        """
        if sequence_type == 't1':
            return self.relu(self.bn_t1(self.conv_t1(x)))
        elif sequence_type == 't2':
            return self.relu(self.bn_t2(self.conv_t2(x)))
        elif sequence_type == 'pd':
            return self.relu(self.bn_pd(self.conv_pd(x)))
        else:
            # Mixed: combine all three
            out_t1 = self.relu(self.bn_t1(self.conv_t1(x)))
            out_t2 = self.relu(self.bn_t2(self.conv_t2(x)))
            out_pd = self.relu(self.bn_pd(self.conv_pd(x)))
            return (out_t1 + out_t2 + out_pd) / 3.0


class PhysicsConstrainedLayer(nn.Module):
    """
    Individual layer with physics-based regularization.
    Implements Equation 5 from the paper.
    """
    def __init__(self, in_channels, out_channels, lambda_phys=0.1):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.lambda_phys = lambda_phys
        self.physics_reg = PhysicsInformedRegularization()
    
    def forward(self, x, return_reg_loss=False):
        """
        Forward pass with optional physics regularization loss
        """
        # Standard convolution: σ(Wl·Al-1 + bl)
        out = self.relu(self.bn(self.conv(x)))
        
        if return_reg_loss:
            # Compute physics regularization: λphys * Rphys(Al)
            reg_loss = self.lambda_phys * self.physics_reg(out)
            return out, reg_loss
        
        return out


class LatentPhysicsConstrainedEncoder(nn.Module):
    """
    Complete LPCE implementation combining all components.
    Implements the dual-branch architecture with physics-informed priors.
    """
    def __init__(self, 
                 in_channels=2,           # Complex k-space (real, imag)
                 hidden_channels=64,
                 latent_dim=128,
                 lambda_phys=0.2,
                 use_sequence_specific=True):
        super().__init__()
        
        self.lambda_phys = lambda_phys
        self.use_sequence_specific = use_sequence_specific
        
        # Initial processing
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, 
                                    kernel_size=3, padding=1)
        
        # Dual-branch architecture
        self.data_branch = DataDrivenBranch(
            hidden_channels, 
            hidden_channels * 2, 
            hidden_channels
        )
        
        self.physics_branch = PhysicsInformedBranch(
            hidden_channels,
            hidden_channels * 2,
            hidden_channels
        )
        
        # Sequence-specific processing (optional)
        if use_sequence_specific:
            self.sequence_processor = SequenceSpecificProcessing(
                hidden_channels * 2,  # Combined branches
                hidden_channels * 2
            )
        
        # Physics-constrained layers
        self.physics_layers = nn.ModuleList([
            PhysicsConstrainedLayer(hidden_channels * 2, 
                                   hidden_channels * 2, 
                                   lambda_phys)
            for _ in range(3)
        ])
        
        # Final projection to latent space
        self.latent_proj = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, latent_dim, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )
        
        self.physics_reg = PhysicsInformedRegularization()
    
    def forward(self, x_under, sequence_type='mixed', return_losses=False):
        """
        Forward pass through LPCE.
        
        Args:
            x_under: Undersampled k-space data (B, 2, H, W) - real/imag channels
            sequence_type: MRI sequence type ('t1', 't2', 'pd', or 'mixed')
            return_losses: Whether to return regularization losses
            
        Returns:
            z_latent: Latent representation
            losses: Dictionary of regularization losses (if return_losses=True)
        """
        losses = {}
        
        # Initial projection
        x = self.input_proj(x_under)
        
        # Dual-branch feature extraction (Equation 2)
        # f_data(X_under)
        z_data = self.data_branch(x)
        
        # f_phys(X_under)
        z_phys = self.physics_branch(x)
        
        # Combine branches: Z_latent = f_data + λ_phys * f_phys
        z_combined = torch.cat([z_data, z_phys], dim=1)
        
        # Sequence-specific processing (Equation 7)
        if self.use_sequence_specific:
            z_combined = self.sequence_processor(z_combined, sequence_type)
        
        # Apply physics-constrained layers (Equation 5)
        total_reg_loss = 0
        for i, layer in enumerate(self.physics_layers):
            if return_losses:
                z_combined, reg_loss = layer(z_combined, return_reg_loss=True)
                losses[f'layer_{i}_reg'] = reg_loss
                total_reg_loss += reg_loss
            else:
                z_combined = layer(z_combined)
        
        # Project to latent space
        z_latent = self.latent_proj(z_combined)
        
        # Additional physics regularization on final output
        if return_losses:
            final_reg = self.lambda_phys * self.physics_reg(z_latent)
            losses['final_reg'] = final_reg
            losses['total_reg'] = total_reg_loss + final_reg
            return z_latent, losses
        
        return z_latent


# Example usage
if __name__ == "__main__":
    # Create LPCE module
    lpce = LatentPhysicsConstrainedEncoder(
        in_channels=2,        # Complex k-space (real, imag)
        hidden_channels=64,
        latent_dim=128,
        lambda_phys=0.2,
        use_sequence_specific=True
    )
    
    # Example input: undersampled k-space data
    batch_size = 4
    height, width = 256, 256
    x_under = torch.randn(batch_size, 2, height, width)
    
    # Forward pass
    z_latent, losses = lpce(x_under, sequence_type='t1', return_losses=True)
    
    print(f"Input shape: {x_under.shape}")
    print(f"Output latent shape: {z_latent.shape}")
    print(f"Regularization losses: {losses}")
    
    # Total parameters
    total_params = sum(p.numel() for p in lpce.parameters())
    print(f"\nTotal parameters: {total_params:,}")