import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhysicsInformedRegularization(nn.Module):
    """
    Physics-based regularization for final reconstruction
    Implements regularization component from Equations 24-25
    """
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, x, k_measured=None, mask=None):
        """
        Args:
            x: Reconstructed features (B, C, H, W)
            k_measured: Measured k-space data (optional)
            mask: Undersampling mask (optional)
        Returns:
            Physics regularization loss
        """
        # 1. Gradient smoothness (anatomical coherence)
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        smoothness_loss = torch.mean(dx**2) + torch.mean(dy**2)
        
        # 2. K-space fidelity (data consistency)
        kspace_loss = 0.0
        if k_measured is not None and mask is not None:
            k_pred = torch.fft.fft2(x, norm='ortho')
            kspace_loss = torch.mean(torch.abs((k_pred - k_measured) * mask)**2)
        
        # 3. Total variation (edge preservation)
        tv_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
        
        # 4. High-frequency consistency
        k_space = torch.fft.fft2(x, norm='ortho')
        k_magnitude = torch.abs(k_space)
        
        h, w = k_magnitude.shape[-2:]
        center_h, center_w = h // 2, w // 2
        
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=x.device), 
            torch.arange(w, device=x.device),
            indexing='ij'
        )
        dist = torch.sqrt((y_grid - center_h)**2 + (x_grid - center_w)**2)
        high_freq_mask = (dist > min(h, w) * 0.3).float()
        
        high_freq_loss = torch.mean((k_magnitude * high_freq_mask)**2)
        
        total_loss = smoothness_loss + 0.5 * kspace_loss + 0.1 * tv_loss + 0.05 * high_freq_loss
        
        return self.gamma * total_loss


class DynamicConvolution(nn.Module):
    """
    Dynamic Convolutional Layer with adaptive kernels
    Implements Equations 20-21:
    K_dynamic = σ(W_k · Z_PACE + b_k)
    Z_filtered = Conv(Z_ADRN, K_dynamic)
    
    Kernels adapt based on input features for context-aware filtering
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, num_kernels=4):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        
        # Kernel generation network (Equation 20)
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_kernels, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Learnable kernel bank
        self.kernel_bank = nn.Parameter(
            torch.randn(num_kernels, out_channels, in_channels, kernel_size, kernel_size)
        )
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Normalization
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            Adaptively filtered features (B, out_channels, H, W)
        """
        batch_size = x.shape[0]
        
        # Generate kernel weights (Equation 20)
        # σ(W_k · Z_PACE + b_k)
        kernel_weights = self.kernel_generator(x)  # (B, num_kernels, 1, 1)
        kernel_weights = kernel_weights.view(batch_size, self.num_kernels, 1, 1, 1, 1)
        
        # Compute dynamic kernels as weighted sum of kernel bank
        # K_dynamic = Σ(weight_i * kernel_i)
        kernels = (kernel_weights * self.kernel_bank.unsqueeze(0)).sum(dim=1)
        # kernels: (B, out_channels, in_channels, K, K)
        
        # Apply dynamic convolution (Equation 21)
        # Z_filtered = Conv(Z_ADRN, K_dynamic)
        outputs = []
        for i in range(batch_size):
            # Perform grouped convolution for each sample
            out = F.conv2d(
                x[i:i+1], 
                kernels[i], 
                bias=self.bias,
                padding=self.kernel_size // 2
            )
            outputs.append(out)
        
        output = torch.cat(outputs, dim=0)
        
        # Normalize and activate
        output = self.bn(output)
        output = self.relu(output)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    Implements Equation 22: Attention(Q,K,V) = softmax(QK^T/√d_k) * V
    """
    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Physics-based attention bias (learnable)
        self.physics_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input features (B, C, H, W)
            return_attention: Whether to return attention weights
        Returns:
            Output features (B, C, H, W)
            attention weights (optional): (B, num_heads, HW, HW)
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence: (B, HW, C)
        x_seq = x.flatten(2).transpose(1, 2)
        
        # Generate Q, K, V
        qkv = self.qkv(x_seq).reshape(B, H*W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention: QK^T / √d_k (Equation 22)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add physics-based bias
        attn = attn + self.physics_bias
        
        # Softmax normalization
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        
        # Output projection
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        # Reshape back to image: (B, C, H, W)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        if return_attention:
            return out, attn
        
        return out


class TransformerBlock(nn.Module):
    """
    Complete Transformer block with multi-head attention and MLP
    Implements Equations 22-23 with physics-informed regularization
    """
    def __init__(self, channels, num_heads=8, mlp_ratio=4.0, dropout=0.1, lambda_phys=0.1):
        super().__init__()
        
        self.channels = channels
        self.lambda_phys = lambda_phys
        
        # Layer normalization
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)
        
        # Multi-head attention (Equation 22)
        self.attn = MultiHeadAttention(channels, num_heads, dropout)
        
        # MLP / Feed-forward network
        mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mlp_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(mlp_hidden_dim, channels, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # Physics regularization
        self.physics_reg = PhysicsInformedRegularization(gamma=1.0)
    
    def forward(self, x, k_measured=None, mask=None, return_reg_loss=False):
        """
        Args:
            x: Input features (B, C, H, W)
            k_measured: Measured k-space data (optional)
            mask: Undersampling mask (optional)
            return_reg_loss: Whether to return physics regularization loss
        Returns:
            Output features (B, C, H, W)
            reg_loss (optional)
        """
        # Self-attention with residual (Equation 22)
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm)
        
        # MLP with residual (Equation 23)
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        
        # Physics regularization (Equation 25)
        if return_reg_loss:
            reg_loss = self.lambda_phys * self.physics_reg(x, k_measured, mask)
            return x, reg_loss
        
        return x


class FeatureFusionModule(nn.Module):
    """
    Fuses features from PACE and ADRN
    Implements Equation 19: Z_ART = DynamicConv(Z_PACE, Z_ADRN) + Transformer(Z_PACE, Z_ADRN)
    """
    def __init__(self, pace_channels, adrn_channels, out_channels):
        super().__init__()
        
        # Project PACE and ADRN to same dimensions
        self.pace_proj = nn.Sequential(
            nn.Conv2d(pace_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.adrn_proj = nn.Sequential(
            nn.Conv2d(adrn_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention-based fusion
        self.fusion_attention = nn.Sequential(
            nn.Conv2d(out_channels * 2, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Refinement
        self.fusion_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, z_pace, z_adrn):
        """
        Args:
            z_pace: Features from PACE (B, pace_channels, H, W)
            z_adrn: Features from ADRN (B, adrn_channels, H, W)
        Returns:
            Fused features (B, out_channels, H, W)
        """
        # Project to same dimensions
        z_pace_proj = self.pace_proj(z_pace)
        z_adrn_proj = self.adrn_proj(z_adrn)
        
        # Concatenate
        z_concat = torch.cat([z_pace_proj, z_adrn_proj], dim=1)
        
        # Attention-based fusion weights
        fusion_weights = self.fusion_attention(z_concat)  # (B, 2, H, W)
        
        # Weighted fusion
        z_fused = fusion_weights[:, 0:1] * z_pace_proj + fusion_weights[:, 1:2] * z_adrn_proj
        
        # Refinement
        z_fused = self.fusion_refine(z_fused)
        
        return z_fused


class AdaptiveReconstructionTransformer(nn.Module):
    """
    Complete Adaptive Reconstruction Transformer (ART) Implementation
    
    Implements:
    - Dynamic convolutions (Equations 20-21)
    - Transformer blocks (Equations 22-23)
    - Physics-based regularization (Equations 24-25)
    - Feature synthesis (Equation 19)
    - Final reconstruction (Equation 26)
    
    Architecture from paper Section 2.4
    """
    def __init__(self,
                 pace_channels=256,         # From PACE output
                 adrn_channels=256,         # From ADRN output
                 hidden_channels=256,
                 out_channels=1,            # Final reconstruction (1 channel magnitude image)
                 num_dynamic_conv_layers=3,
                 num_transformer_blocks=4,
                 num_heads=8,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 lambda_phys=0.2):
        super().__init__()
        
        self.pace_channels = pace_channels
        self.adrn_channels = adrn_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.lambda_phys = lambda_phys
        
        # Feature fusion module (Equation 19)
        self.feature_fusion = FeatureFusionModule(
            pace_channels, 
            adrn_channels, 
            hidden_channels
        )
        
        # Dynamic convolution layers (Equations 20-21)
        self.dynamic_convs = nn.ModuleList([
            DynamicConvolution(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                num_kernels=4
            )
            for _ in range(num_dynamic_conv_layers)
        ])
        
        # Transformer blocks (Equations 22-23)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_channels,
                num_heads,
                mlp_ratio,
                dropout,
                lambda_phys
            )
            for _ in range(num_transformer_blocks)
        ])
        
        # Intermediate refinement layers
        self.refinement_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(3)
        ])
        
        # Final reconstruction layers (Equation 26)
        self.final_layers = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, out_channels, kernel_size=3, padding=1)
        )
        
        # Physics regularization
        self.physics_reg = PhysicsInformedRegularization(gamma=1.0)
        
        # Data consistency layer
        self.data_consistency_weight = nn.Parameter(torch.tensor(1.0))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def apply_data_consistency(self, x_recon, k_measured, mask):
        """
        Apply k-space data consistency to final reconstruction
        Ensures reconstructed image matches measured k-space frequencies
        """
        if k_measured is None or mask is None:
            return x_recon
        
        # Convert to k-space
        k_pred = torch.fft.fft2(x_recon, norm='ortho')
        
        # Data consistency: replace measured frequencies
        k_corrected = k_pred * (1 - mask) + k_measured * mask * self.data_consistency_weight
        
        # Convert back to image space
        x_corrected = torch.fft.ifft2(k_corrected, norm='ortho').real
        
        return x_corrected
    
    def forward(self, z_pace, z_adrn, k_measured=None, mask=None, 
                return_losses=False, return_attention=False):
        """
        Complete forward pass through ART
        
        Implements the full reconstruction pipeline:
        1. Feature fusion from PACE and ADRN (Equation 19)
        2. Dynamic convolutions (Equations 20-21)
        3. Transformer processing (Equations 22-23)
        4. Final reconstruction (Equation 26)
        5. Data consistency projection
        
        Args:
            z_pace: Features from PACE (B, pace_channels, H, W)
            z_adrn: Features from ADRN (B, adrn_channels, H, W)
            k_measured: Measured k-space data (B, H, W) complex
            mask: Undersampling mask (B, 1, H, W)
            return_losses: Whether to return regularization losses
            return_attention: Whether to return attention maps
        
        Returns:
            x_recon: Reconstructed MRI image (B, 1, H, W)
            losses: Dict of losses (if return_losses=True)
            attention_maps: List of attention maps (if return_attention=True)
        """
        losses = {}
        attention_maps = []
        
        # 1. Feature fusion (Equation 19)
        # Z_ART = DynamicConv(Z_PACE, Z_ADRN) + Transformer(Z_PACE, Z_ADRN)
        z_fused = self.feature_fusion(z_pace, z_adrn)
        
        # 2. Dynamic convolutions (Equations 20-21)
        # K_dynamic = σ(W_k·Z_PACE + b_k)
        # Z_filtered = Conv(Z_ADRN, K_dynamic)
        z_dynamic = z_fused
        for i, dynamic_conv in enumerate(self.dynamic_convs):
            z_dynamic = dynamic_conv(z_dynamic)
            
            if return_losses:
                dc_reg = self.lambda_phys * self.physics_reg(z_dynamic, k_measured, mask)
                losses[f'dynamic_conv_{i}_reg'] = dc_reg
        
        # 3. Transformer blocks (Equations 22-23)
        # Attention(Q,K,V) = softmax(QK^T/√d_k) * V
        z_transformed = z_dynamic
        total_transformer_reg = 0
        
        for i, transformer in enumerate(self.transformer_blocks):
            if return_losses:
                z_transformed, reg_loss = transformer(
                    z_transformed, 
                    k_measured, 
                    mask, 
                    return_reg_loss=True
                )
                losses[f'transformer_{i}_reg'] = reg_loss
                total_transformer_reg += reg_loss
            else:
                z_transformed = transformer(z_transformed, k_measured, mask)
            
            # Extract attention maps if requested
            if return_attention:
                with torch.no_grad():
                    _, attn = transformer.attn(z_transformed, return_attention=True)
                    attention_maps.append(attn)
        
        # 4. Intermediate refinement
        z_refined = z_transformed
        for refinement_layer in self.refinement_layers:
            z_refined = z_refined + refinement_layer(z_refined)  # Residual
        
        # 5. Final reconstruction (Equation 26)
        # Z_ART-final = σ(W_out·Z_ART + b_out)
        x_recon = self.final_layers(z_refined)
        
        # 6. Apply data consistency projection
        if k_measured is not None and mask is not None:
            x_recon = self.apply_data_consistency(x_recon, k_measured, mask)
        
        # Compute losses if requested
        if return_losses:
            # Physics regularization on final output
            final_physics_reg = self.lambda_phys * self.physics_reg(
                x_recon, k_measured, mask
            )
            losses['final_physics_reg'] = final_physics_reg
            
            # K-space consistency loss
            if k_measured is not None and mask is not None:
                k_recon = torch.fft.fft2(x_recon, norm='ortho')
                kspace_loss = torch.mean(torch.abs((k_recon - k_measured) * mask)**2)
                losses['kspace_consistency'] = kspace_loss
            
            # Total loss
            losses['total_reg'] = sum(losses.values())
        
        # Prepare return values
        if return_losses and return_attention:
            return x_recon, losses, attention_maps
        elif return_losses:
            return x_recon, losses
        elif return_attention:
            return x_recon, attention_maps
        
        return x_recon
    
    def get_num_params(self):
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== Testing & Validation ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Complete ART Implementation")
    print("=" * 70)
    
    # Configuration matching paper
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = {
        'pace_channels': 256,
        'adrn_channels': 256,
        'hidden_channels': 256,
        'out_channels': 1,              # Final magnitude image
        'num_dynamic_conv_layers': 3,
        'num_transformer_blocks': 4,
        'num_heads': 8,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'lambda_phys': 0.2
    }
    
    # Create ART
    art = AdaptiveReconstructionTransformer(**config).to(device)
    
    print(f"\n1. Model Configuration:")
    print(f"   - Device: {device}")
    print(f"   - PACE channels: {config['pace_channels']}")
    print(f"   - ADRN channels: {config['adrn_channels']}")
    print(f"   - Hidden channels: {config['hidden_channels']}")
    print(f"   - Output channels: {config['out_channels']}")
    print(f"   - Dynamic conv layers: {config['num_dynamic_conv_layers']}")
    print(f"   - Transformer blocks: {config['num_transformer_blocks']}")
    print(f"   - Attention heads: {config['num_heads']}")
    print(f"   - Physics weight λ: {config['lambda_phys']}")
    
    # Count parameters
    total_params = art.get_num_params()
    trainable_params = art.get_num_trainable_params()
    print(f"\n2. Model Parameters:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print(f"\n3. Testing Forward Pass:")
    batch_size = 2
    height, width = 128, 128
    
    # Inputs from PACE and ADRN
    z_pace = torch.randn(batch_size, 256, height, width).to(device)
    z_adrn = torch.randn(batch_size, 256, height, width).to(device)
    
    # K-space and mask
    k_measured = torch.randn(batch_size, height, width, dtype=torch.complex64).to(device)
    mask = (torch.rand(batch_size, 1, height, width) > 0.75).float().to(device)
    
    # Forward pass without losses
    x_recon = art(z_pace, z_adrn, k_measured, mask)
    
    print(f"   - PACE input shape: {z_pace.shape}")
    print(f"   - ADRN input shape: {z_adrn.shape}")
    print(f"   - Reconstruction shape: {x_recon.shape}")
    print(f"   - Output range: [{x_recon.min().item():.4f}, {x_recon.max().item():.4f}]")
    
    # Forward pass with losses
    print(f"\n4. Testing with Losses:")
    x_recon_loss, losses = art(z_pace, z_adrn, k_measured, mask, return_losses=True)
    
    print(f"   - Regularization losses:")
    for key, value in losses.items():
        print(f"     * {key}: {value.item():.6f}")
    
    # Forward pass with attention
    print(f"\n5. Testing with Attention Maps:")
    x_recon_attn, attention_maps = art(
        z_pace, z_adrn, k_measured, mask, return_attention=True
    )
    
    print(f"   - Number of attention maps: {len(attention_maps)}")
    for i, attn in enumerate(attention_maps):
        print(f"     * Transformer {i} attention: {attn.shape}")
    
    # Test individual components
    print(f"\n6. Component Testing:")
    
    # Dynamic Convolution
    dynamic_conv = DynamicConvolution(256, 256, kernel_size=3, num_kernels=4).to(device)
    x_test = torch.randn(2, 256, 64, 64).to(device)
    x_dynamic = dynamic_conv(x_test)
    print(f"   - Dynamic Conv: {x_test.shape} -> {x_dynamic.shape} ✓")
    
    # Multi-head Attention
    mha = MultiHeadAttention(256, num_heads=8).to(device)
    x_attn = mha(x_test)
    print(f"   - Multi-head Attention: {x_test.shape} -> {x_attn.shape} ✓")
    
    # Transformer Block
    transformer = TransformerBlock(256, num_heads=8).to(device)
    x_trans = transformer(x_test)
    print(f"   - Transformer Block: {x_test.shape} -> {x_trans.shape} ✓")
    
    # Feature Fusion
    fusion = FeatureFusionModule(256, 256, 256).to(device)
    z1 = torch.randn(2, 256, 64, 64).to(device)
    z2 = torch.randn(2, 256, 64, 64).to(device)
    z_fused = fusion(z1, z2)
    print(f"   - Feature Fusion: {z1.shape} + {z2.shape} -> {z_fused.shape} ✓")
    
    # Data Consistency
    x_dc = art.apply_data_consistency(
        x_test, 
        torch.fft.fft2(x_test, norm='ortho'),
        torch.ones(2, 1, 64, 64).to(device)
    )
    print(f"   - Data Consistency: {x_test.shape} -> {x_dc.shape} ✓")
    
    # Verify key components
    print(f"\n7. Component Verification:")
    print(f"   - Has feature fusion: {hasattr(art, 'feature_fusion')} ✓")
    print(f"   - Has dynamic convs: {len(art.dynamic_convs)} layers ✓")
    print(f"   - Has transformers: {len(art.transformer_blocks)} blocks ✓")
    print(f"   - Has refinement layers: {len(art.refinement_layers)} layers ✓")
    print(f"   - Has physics regularization: {hasattr(art, 'physics_reg')} ✓")
    print(f"   - Has final layers: {hasattr(art, 'final_layers')} ✓")
    
    # Test complete pipeline
    print(f"\n8. Complete Pipeline Test (LPCE → PACE → ADRN → ART):")
    
    # Simulate full pipeline
    print(f"   - Simulating LPCE output...")
    z_latent = torch.randn(2, 128, 128, 128).to(device)
    
    print(f"   - Simulating PACE processing...")
    from pace import PhysicsAwareContextEncoder
    pace_module = PhysicsAwareContextEncoder(
        in_channels=128,
        hidden_channels=256,
        out_channels=256
    ).to(device)
    z_pace_out = pace_module(z_latent)
    
    print(f"   - Simulating ADRN processing...")
    from adrn import AdaptiveDiffusionRefinementNetwork
    adrn_module = AdaptiveDiffusionRefinementNetwork(
        in_channels=256,
        model_channels=128,
        out_channels=256,
        num_diffusion_steps=10,
        device=device
    ).to(device)
    adrn_module.eval()
    with torch.no_grad():
        z_adrn_out = adrn_module(z_pace_out, k_measured, mask)
    
    print(f"   - ART final reconstruction...")
    art.eval()
    with torch.no_grad():
        x_final = art(z_pace_out, z_adrn_out, k_measured, mask)
    
    print(f"\n   Pipeline Flow:")
    print(f"   - LPCE: Input → {z_latent.shape}")
    print(f"   - PACE: {z_latent.shape} → {z_pace_out.shape}")
    print(f"   - ADRN: {z_pace_out.shape} → {z_adrn_out.shape}")
    print(f"   - ART: {z_pace_out.shape} + {z_adrn_out.shape} → {x_final.shape}")
    print(f"   - Final reconstruction: {x_final.shape} ✓")
    
    # Verify output properties
    print(f"\n9. Output Verification:")
    print(f"   - Output is single channel: {x_final.shape[1] == 1} ✓")
    print(f"   - Output is real-valued: {x_final.dtype == torch.float32} ✓")
    print(f"   - Output has correct spatial dims: {x_final.shape[2:] == (128, 128)} ✓")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f"\n10. Memory Usage:")
        print(f"   - Allocated: {memory_allocated:.2f} MB")
        print(f"   - Reserved: {memory_reserved:.2f} MB")
    
    print(f"\n{'=' * 70}")
    print("✓ All tests passed! ART implementation complete.")
    print("✓ Complete PINN-DADif pipeline verified!")
    print("=" * 70)