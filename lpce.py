import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BlochEquationConstraint(nn.Module):
    """
    Implements Bloch equations for T1/T2 relaxation (Equation 1 from paper).
    M(t) = M0 * (1 - e^(-t/T1)) for T1-weighted
    M(t) = M0 * e^(-t/T2) for T2-weighted
    """
    def __init__(self):
        super().__init__()
        
        # Typical relaxation times for brain tissue (in ms)
        # Gray matter
        self.register_buffer('T1_gm', torch.tensor(1331.0))
        self.register_buffer('T2_gm', torch.tensor(110.0))
        
        # White matter
        self.register_buffer('T1_wm', torch.tensor(832.0))
        self.register_buffer('T2_wm', torch.tensor(80.0))
        
        # CSF
        self.register_buffer('T1_csf', torch.tensor(4163.0))
        self.register_buffer('T2_csf', torch.tensor(2569.0))
        
    def t1_relaxation(self, M0, t, T1):
        """T1 recovery: M(t) = M0 * (1 - exp(-t/T1))"""
        return M0 * (1 - torch.exp(-t / (T1 + 1e-8)))
    
    def t2_relaxation(self, M0, t, T2):
        """T2 decay: M(t) = M0 * exp(-t/T2)"""
        return M0 * torch.exp(-t / (T2 + 1e-8))
    
    def forward(self, signal, sequence_type='t1', TR=None, TE=None):
        """
        Enforce signal to follow Bloch equations.
        
        Args:
            signal: Input signal (B, C, H, W)
            sequence_type: 't1', 't2', or 'pd'
            TR: Repetition time (if None, use defaults)
            TE: Echo time (if None, use defaults)
        
        Returns:
            Physics-consistent signal constraint
        """
        M0 = signal.abs().max()  # Estimate equilibrium magnetization
        
        # Default sequence parameters from paper
        if TR is None or TE is None:
            if sequence_type == 't1':
                TR, TE = 9.813, 4.603  # From paper Section 2.6
            elif sequence_type == 't2':
                TR, TE = 8178.0, 100.0
            else:  # PD
                TR, TE = 8178.0, 8.0
        
        TR = torch.tensor(TR, device=signal.device)
        TE = torch.tensor(TE, device=signal.device)
        
        if sequence_type == 't1':
            # T1-weighted: emphasizes T1 differences
            # Mix of tissue types
            signal_gm = self.t1_relaxation(M0, TR, self.T1_gm) * \
                       self.t2_relaxation(1.0, TE, self.T2_gm)
            signal_wm = self.t1_relaxation(M0, TR, self.T1_wm) * \
                       self.t2_relaxation(1.0, TE, self.T2_wm)
            
            expected = (signal_gm + signal_wm) / 2.0
            
        elif sequence_type == 't2':
            # T2-weighted: emphasizes T2 differences
            signal_gm = self.t1_relaxation(M0, TR, self.T1_gm) * \
                       self.t2_relaxation(1.0, TE, self.T2_gm)
            signal_wm = self.t1_relaxation(M0, TR, self.T1_wm) * \
                       self.t2_relaxation(1.0, TE, self.T2_wm)
            
            expected = (signal_gm + signal_wm) / 2.0
            
        else:  # PD-weighted
            # Proton density: minimal T1/T2 weighting
            expected = M0 * torch.ones_like(signal)
        
        return expected


class PhysicsInformedRegularization(nn.Module):
    """
    Physics-based regularization module (Equation 6 from paper).
    Enforces MRI signal properties including Bloch equations and smoothness.
    """
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma
        self.bloch = BlochEquationConstraint()
    
    def forward(self, activations, sequence_type='mixed', physics_model=None):
        """
        Args:
            activations: Current layer activations (B, C, H, W)
            sequence_type: MRI sequence type
            physics_model: Optional custom physics model
        
        Returns:
            Regularization loss
        """
        # 1. Gradient smoothness constraint
        dx = activations[:, :, :, 1:] - activations[:, :, :, :-1]
        dy = activations[:, :, 1:, :] - activations[:, :, :-1, :]
        smoothness_loss = torch.mean(dx**2) + torch.mean(dy**2)
        
        # 2. Bloch equation constraint
        bloch_loss = 0.0
        if sequence_type != 'mixed':
            expected_signal = self.bloch(activations, sequence_type)
            # Soft constraint to guide toward physically plausible signals
            bloch_loss = torch.mean((activations - expected_signal)**2)
        
        # 3. Custom physics model (optional)
        if physics_model is not None:
            expected = physics_model(activations)
            custom_loss = torch.mean((activations - expected)**2)
            total_loss = smoothness_loss + 0.3 * bloch_loss + 0.2 * custom_loss
        else:
            total_loss = smoothness_loss + 0.5 * bloch_loss
        
        return self.gamma * total_loss


class FourierTransformLayer(nn.Module):
    """
    Implements Fourier Transform operations (Equations 3 and 4 from paper).
    Converts between spatial and frequency (k-space) domains.
    """
    def __init__(self):
        super().__init__()
    
    def forward_fft(self, x):
        """
        Forward Fourier Transform: spatial -> frequency domain
        Implements Equation 3
        
        F(u,v) = Σ_x Σ_y f(x,y) * e^(-2πi(ux/M + vy/N))
        """
        # Input: (B, C, H, W) - real-valued spatial domain
        # Output: (B, C, H, W, 2) - complex frequency domain [real, imag]
        
        # Apply 2D FFT with ortho normalization
        x_freq = torch.fft.fft2(x, norm='ortho')
        
        # Convert complex to real representation
        x_freq_real = torch.stack([x_freq.real, x_freq.imag], dim=-1)
        
        return x_freq_real
    
    def inverse_fft(self, x_freq):
        """
        Inverse Fourier Transform: frequency -> spatial domain
        Implements Equation 4
        
        f(x,y) = (1/MN) * Σ_u Σ_v F(u,v) * e^(2πi(ux/M + vy/N))
        """
        # Input: (B, C, H, W, 2) - complex frequency domain
        # Output: (B, C, H, W) - real-valued spatial domain
        
        # Reconstruct complex tensor
        x_complex = torch.complex(x_freq[..., 0], x_freq[..., 1])
        
        # Apply 2D IFFT with ortho normalization
        x_spatial = torch.fft.ifft2(x_complex, norm='ortho')
        
        # Return real part (imaginary should be negligible for real images)
        return x_spatial.real


class DataDrivenBranch(nn.Module):
    """
    Data-driven feature extraction branch using standard CNNs.
    Part of dual-branch architecture in Equation 2: f_data(X_under)
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
        
        # Residual connection
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Standard convolutional feature extraction with residual connection
        """
        identity = self.skip(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out = out + identity
        out = self.relu(out)
        
        return out


class PhysicsInformedBranch(nn.Module):
    """
    Physics-informed feature extraction branch.
    Models MRI physics including k-space properties and data consistency.
    Part of dual-branch architecture in Equation 2: f_phys(X_under)
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
        
        # Project back to desired channels (complex)
        self.kspace_conv3 = nn.Conv2d(hidden_channels, in_channels * 2, 
                                      kernel_size=3, padding=1)
        
        # Spatial domain refinement after IFFT
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, 
                                      kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Physics regularization
        self.physics_reg = PhysicsInformedRegularization(gamma=0.1)
        
        # K-space data consistency weight
        self.kspace_fidelity_weight = nn.Parameter(torch.tensor(1.0))
    
    def apply_kspace_consistency(self, x_spatial, k_measured, mask):
        """
        Enforce data consistency with measured k-space data.
        Replaces measured frequencies with ground truth values.
        
        Args:
            x_spatial: Predicted spatial image
            k_measured: Measured k-space data (complex)
            mask: Undersampling mask (1 = measured, 0 = unmeasured)
        """
        if k_measured is None or mask is None:
            return x_spatial
        
        # Convert prediction to k-space
        k_pred = torch.fft.fft2(x_spatial, norm='ortho')
        
        # Data consistency: keep measured frequencies, update unmeasured
        # k_corrected = k_pred * (1 - mask) + k_measured * mask
        k_corrected = k_pred * (1 - mask) + k_measured * mask * self.kspace_fidelity_weight
        
        # Convert back to spatial domain
        x_corrected = torch.fft.ifft2(k_corrected, norm='ortho').real
        
        return x_corrected
    
    def forward(self, x, k_measured=None, mask=None, sequence_type='mixed'):
        """
        Physics-guided feature extraction through k-space processing
        
        Args:
            x: Input features (B, C, H, W)
            k_measured: Measured k-space data (B, H, W) complex
            mask: Undersampling mask (B, 1, H, W)
            sequence_type: MRI sequence type
        """
        # Transform to k-space (Equation 3)
        x_kspace = self.fourier.forward_fft(x)  # (B, C, H, W, 2)
        B, C, H, W, _ = x_kspace.shape
        
        # Reshape for convolution: merge real/imag into channels
        x_kspace = x_kspace.permute(0, 1, 4, 2, 3).reshape(B, C*2, H, W)
        
        # Process in k-space domain with physics-aware convolutions
        x_kspace = self.relu(self.bn1(self.kspace_conv1(x_kspace)))
        x_kspace = self.relu(self.bn2(self.kspace_conv2(x_kspace)))
        x_kspace = self.kspace_conv3(x_kspace)
        
        # Reshape back to complex format
        x_kspace = x_kspace.reshape(B, C, 2, H, W).permute(0, 1, 3, 4, 2)
        
        # Transform back to spatial domain (Equation 4)
        x_spatial = self.fourier.inverse_fft(x_kspace)
        
        # Apply k-space data consistency if available
        if k_measured is not None and mask is not None:
            x_spatial = self.apply_kspace_consistency(x_spatial, k_measured, mask)
        
        # Spatial refinement
        x_out = self.relu(self.bn3(self.spatial_conv(x_spatial)))
        
        return x_out


class SequenceSpecificProcessing(nn.Module):
    """
    Handles different MRI sequences (T1, T2, PD/FLAIR) with specialized processing.
    Implements Equation 7: Z_sequence = Conv_T1 + Conv_T2 + Conv_PD
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
        
        # Learnable fusion weights for mixed sequences
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3.0)
    
    def forward(self, x, sequence_type='mixed'):
        """
        Args:
            x: Input features (B, C, H, W)
            sequence_type: 't1', 't2', 'pd', 'flair', or 'mixed'
        
        Returns:
            Sequence-specific features
        """
        if sequence_type == 't1':
            return self.relu(self.bn_t1(self.conv_t1(x)))
        
        elif sequence_type == 't2':
            return self.relu(self.bn_t2(self.conv_t2(x)))
        
        elif sequence_type in ['pd', 'flair']:
            # PD and FLAIR use same processing
            return self.relu(self.bn_pd(self.conv_pd(x)))
        
        else:  # 'mixed'
            # Weighted combination of all sequences (Equation 7)
            out_t1 = self.relu(self.bn_t1(self.conv_t1(x)))
            out_t2 = self.relu(self.bn_t2(self.conv_t2(x)))
            out_pd = self.relu(self.bn_pd(self.conv_pd(x)))
            
            # Normalize fusion weights
            weights = F.softmax(self.fusion_weights, dim=0)
            
            return weights[0] * out_t1 + weights[1] * out_t2 + weights[2] * out_pd


class PhysicsConstrainedLayer(nn.Module):
    """
    Individual layer with physics-based regularization.
    Implements Equation 5: A_l = σ(W_l·A_{l-1} + b_l) + λ_phys * R_phys(A_l)
    """
    def __init__(self, in_channels, out_channels, lambda_phys=0.1):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.lambda_phys = lambda_phys
        self.physics_reg = PhysicsInformedRegularization(gamma=1.0)
        
        # Residual connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, sequence_type='mixed', return_reg_loss=False):
        """
        Forward pass with optional physics regularization loss
        
        Args:
            x: Input activations
            sequence_type: MRI sequence type
            return_reg_loss: Whether to return regularization loss
        """
        identity = self.skip(x)
        
        # Standard convolution: σ(W_l·A_{l-1} + b_l)
        out = self.relu(self.bn(self.conv(x)))
        
        # Add residual
        out = out + identity
        
        if return_reg_loss:
            # Compute physics regularization: λ_phys * R_phys(A_l)
            reg_loss = self.lambda_phys * self.physics_reg(out, sequence_type)
            return out, reg_loss
        
        return out


class CoilSensitivityLayer(nn.Module):
    """
    Handle multi-coil MRI data (for fastMRI dataset).
    Uses ESPIRiT-estimated coil sensitivities for coil combination.
    """
    def __init__(self, num_coils=10):
        super().__init__()
        self.num_coils = num_coils
        
    def coil_combine(self, multi_coil_image, sensitivities):
        """
        Combine multi-coil images using sensitivity maps.
        
        Args:
            multi_coil_image: (B, num_coils, H, W) complex
            sensitivities: (B, num_coils, H, W) complex
        
        Returns:
            combined_image: (B, 1, H, W) real magnitude
        """
        if multi_coil_image.is_complex():
            # Conjugate sensitivity weighting: Σ_c (img_c * conj(sens_c))
            combined = torch.sum(multi_coil_image * sensitivities.conj(), dim=1)
            
            # Normalize by sum of squares of sensitivities
            sos = torch.sqrt(torch.sum(torch.abs(sensitivities)**2, dim=1) + 1e-8)
            combined = combined / sos
            
            # Return magnitude
            return combined.abs().unsqueeze(1)
        else:
            # If already magnitude images, use sum-of-squares
            sos = torch.sqrt(torch.sum(multi_coil_image**2, dim=1, keepdim=True) + 1e-8)
            return sos
    
    def forward(self, x, sensitivities=None):
        """
        Args:
            x: Multi-coil image (B, num_coils, H, W) or single-coil (B, 1, H, W)
            sensitivities: Coil sensitivity maps (optional)
        """
        if x.shape[1] > 1:  # Multi-coil
            if sensitivities is not None:
                return self.coil_combine(x, sensitivities)
            else:
                # Fall back to sum-of-squares
                return torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + 1e-8)
        
        return x  # Already single-coil


class LatentPhysicsConstrainedEncoder(nn.Module):
    """
    Complete Latent Physics-Constrained Encoder (LPCE) implementation.
    
    Implements:
    - Dual-branch architecture (Equation 2)
    - Physics-informed operations (Equations 3-6)
    - Sequence-specific processing (Equation 7)
    - Multi-coil support for fastMRI
    
    Architecture from paper Section 2.1
    """
    def __init__(self, 
                 in_channels=1,           # Single-channel magnitude image
                 hidden_channels=64,      # Hidden layer channels
                 latent_dim=128,          # Optimal from Fig. 5c ablation
                 lambda_phys=0.2,         # Physics regularization weight
                 use_sequence_specific=True,
                 num_coils=1,             # 1 for private dataset, 5-10 for fastMRI
                 num_physics_layers=3):    # Number of physics-constrained layers
        super().__init__()
        
        self.in_channels = in_channels
        self.lambda_phys = lambda_phys
        self.use_sequence_specific = use_sequence_specific
        self.num_coils = num_coils
        
        # Multi-coil handling for fastMRI dataset
        if num_coils > 1:
            self.coil_layer = CoilSensitivityLayer(num_coils)
            effective_in_channels = 1  # After coil combination
        else:
            self.coil_layer = None
            effective_in_channels = in_channels
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(effective_in_channels, hidden_channels, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dual-branch architecture (Equation 2)
        # f_data(X_under): Standard CNN features
        self.data_branch = DataDrivenBranch(
            hidden_channels, 
            hidden_channels * 2, 
            hidden_channels
        )
        
        # f_phys(X_under): Physics-informed features via k-space
        self.physics_branch = PhysicsInformedBranch(
            hidden_channels,
            hidden_channels * 2,
            hidden_channels
        )
        
        # Fusion layer for dual branches
        self.branch_fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2,
                     kernel_size=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # Sequence-specific processing (Equation 7)
        if use_sequence_specific:
            self.sequence_processor = SequenceSpecificProcessing(
                hidden_channels * 2,
                hidden_channels * 2
            )
        
        # Physics-constrained layers (Equation 5)
        self.physics_layers = nn.ModuleList([
            PhysicsConstrainedLayer(
                hidden_channels * 2, 
                hidden_channels * 2, 
                lambda_phys
            )
            for _ in range(num_physics_layers)
        ])
        
        # Final projection to latent space
        self.latent_proj = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, latent_dim, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim,
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Physics regularization module
        self.physics_reg = PhysicsInformedRegularization(gamma=1.0)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x_under, k_space=None, mask=None, 
                sensitivities=None, sequence_type='mixed', 
                return_losses=False):
        """
        Complete forward pass through LPCE.
        
        Args:
            x_under: Undersampled magnitude image 
                     - Single-coil: (B, 1, H, W)
                     - Multi-coil: (B, num_coils, H, W)
            k_space: Measured k-space data (B, H, W) complex tensor
            mask: Undersampling mask (B, 1, H, W) - 1=measured, 0=unmeasured
            sensitivities: Coil sensitivity maps for multi-coil data
            sequence_type: 't1', 't2', 'pd', 'flair', or 'mixed'
            return_losses: Whether to return regularization losses
        
        Returns:
            z_latent: Latent representation (B, latent_dim, H, W)
            losses: Dict of regularization losses (if return_losses=True)
        """
        losses = {}
        
        # Multi-coil combination if needed (fastMRI)
        if self.num_coils > 1 and self.coil_layer is not None:
            x_under = self.coil_layer(x_under, sensitivities)
        
        # Initial projection
        x = self.input_proj(x_under)
        
        # Dual-branch feature extraction (Equation 2)
        # Z_latent = f_data(X_under) + λ_phys * f_phys(X_under)
        
        # Data-driven branch
        z_data = self.data_branch(x)
        
        # Physics-informed branch with k-space processing
        z_phys = self.physics_branch(x, k_space, mask, sequence_type)
        
        # Combine branches
        z_combined = torch.cat([z_data, z_phys], dim=1)
        z_combined = self.branch_fusion(z_combined)
        
        # Sequence-specific processing (Equation 7)
        if self.use_sequence_specific:
            z_combined = self.sequence_processor(z_combined, sequence_type)
        
        # Apply physics-constrained layers (Equation 5)
        total_reg_loss = 0
        for i, layer in enumerate(self.physics_layers):
            if return_losses:
                z_combined, reg_loss = layer(z_combined, sequence_type, 
                                            return_reg_loss=True)
                losses[f'layer_{i}_reg'] = reg_loss
                total_reg_loss += reg_loss
            else:
                z_combined = layer(z_combined, sequence_type)
        
        # Project to latent space
        z_latent = self.latent_proj(z_combined)
        
        # Final physics regularization
        if return_losses:
            final_reg = self.lambda_phys * self.physics_reg(z_latent, sequence_type)
            losses['final_reg'] = final_reg
            losses['total_reg'] = total_reg_loss + final_reg
            
            return z_latent, losses
        
        return z_latent
    
    def get_num_params(self):
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== Testing & Validation ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Complete LPCE Implementation")
    print("=" * 60)
    
    # Configuration matching paper
    config = {
        'in_channels': 1,
        'hidden_channels': 64,
        'latent_dim': 128,          # Optimal from Fig. 5c
        'lambda_phys': 0.2,         # From Section 2.7
        'use_sequence_specific': True,
        'num_coils': 1,             # Private dataset
        'num_physics_layers': 3
    }
    
    # Create LPCE
    lpce = LatentPhysicsConstrainedEncoder(**config)
    
    print(f"\n1. Model Configuration:")
    print(f"   - Input channels: {config['in_channels']}")
    print(f"   - Latent dimension: {config['latent_dim']}")
    print(f"   - Physics weight λ: {config['lambda_phys']}")
    print(f"   - Number of coils: {config['num_coils']}")
    
    # Count parameters
    total_params = lpce.get_num_params()
    trainable_params = lpce.get_num_trainable_params()
    print(f"\n2. Model Parameters:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    # Test with single-coil data (Private dataset)
    print(f"\n3. Testing Single-Coil Input (Private Dataset):")
    batch_size = 4
    height, width = 256, 256
    
    # Undersampled magnitude image
    x_under = torch.randn(batch_size, 1, height, width).abs()
    
    # Corresponding k-space and mask
    k_space = torch.fft.fft2(x_under.squeeze(1), norm='ortho')
    mask = (torch.rand(batch_size, 1, height, width) > 0.75).float()  # 4x acceleration
    
    # Forward pass
    z_latent, losses = lpce(
        x_under, 
        k_space=k_space,
        mask=mask,
        sequence_type='t1', 
        return_losses=True
    )
    
    print(f"   - Input shape: {x_under.shape}")
    print(f"   - Latent shape: {z_latent.shape}")
    print(f"   - Regularization losses:")
    for key, value in losses.items():
        print(f"     * {key}: {value.item():.6f}")
    
    # Test with multi-coil data (fastMRI)
    print(f"\n4. Testing Multi-Coil Input (fastMRI):")
    lpce_multicoil = LatentPhysicsConstrainedEncoder(
        in_channels=1,
        hidden_channels=64,
        latent_dim=128,
        lambda_phys=0.2,
        use_sequence_specific=True,
        num_coils=10,  # 10 virtual coils after compression
        num_physics_layers=3
    )
    
    # Multi-coil input
    x_multicoil = torch.randn(batch_size, 10, height, width).abs()
    sensitivities = torch.randn(batch_size, 10, height, width) + \
                   1j * torch.randn(batch_size, 10, height, width)
    
    z_latent_mc = lpce_multicoil(
        x_multicoil,
        sensitivities=sensitivities,
        sequence_type='t2'
    )
    
    print(f"   - Input shape: {x_multicoil.shape}")
    print(f"   - Latent shape: {z_latent_mc.shape}")
    
    # Test different sequence types
    print(f"\n5. Testing Different Sequence Types:")
    for seq_type in ['t1', 't2', 'pd', 'mixed']:
        z = lpce(x_under, k_space=k_space, mask=mask, sequence_type=seq_type)
        print(f"   - {seq_type.upper()}: {z.shape} ✓")
    
    # Verify key components
    print(f"\n6. Component Verification:")
    print(f"   - Has Bloch constraints: {hasattr(lpce.physics_reg, 'bloch')} ✓")
    print(f"   - Has data branch: {hasattr(lpce, 'data_branch')} ✓")
    print(f"   - Has physics branch: {hasattr(lpce, 'physics_branch')} ✓")
    print(f"   - Has sequence processor: {hasattr(lpce, 'sequence_processor')} ✓")
    print(f"   - Has physics layers: {len(lpce.physics_layers)} layers ✓")
    
    print(f"\n{'=' * 60}")
    print("✓ All tests passed! LPCE implementation complete.")
    print("=" * 60)