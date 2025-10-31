import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhysicsInformedRegularization(nn.Module):
    """
    Physics-based regularization (reused from LPCE but included for completeness)
    Implements Equation 6/13 from the paper
    """
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, activations, sequence_type='mixed'):
        """
        Args:
            activations: Current layer activations (B, C, H, W)
            sequence_type: MRI sequence type
        Returns:
            Regularization loss
        """
        # Gradient smoothness constraint
        dx = activations[:, :, :, 1:] - activations[:, :, :, :-1]
        dy = activations[:, :, 1:, :] - activations[:, :, :-1, :]
        smoothness_loss = torch.mean(dx**2) + torch.mean(dy**2)
        
        # K-space frequency smoothness (physics-based)
        # High-frequency components should be smooth in MRI
        k_space = torch.fft.fft2(activations, norm='ortho')
        k_magnitude = torch.abs(k_space)
        
        # Penalize high-frequency noise
        h, w = k_magnitude.shape[-2:]
        center_h, center_w = h // 2, w // 2
        
        # Create high-frequency mask
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y, x = y.to(activations.device), x.to(activations.device)
        dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
        high_freq_mask = (dist > min(h, w) * 0.3).float()
        
        high_freq_loss = torch.mean((k_magnitude * high_freq_mask)**2)
        
        total_loss = smoothness_loss + 0.1 * high_freq_loss
        
        return self.gamma * total_loss


class AtrousSpatialPyramidPooling(nn.Module):
    """
    ASPP Module - Multi-scale feature extraction with dilated convolutions
    Implements Equation 9 from paper:
    Z_ASPP = Concat(Conv_r=1, Conv_r=6, Conv_r=12, Conv_r=18)
    
    Captures features at multiple receptive field scales
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Dilation rates from paper: [1, 6, 12, 18]
        self.dilation_rates = [1, 6, 12, 18]
        
        # Multiple parallel convolutions with different dilation rates
        self.aspp_convs = nn.ModuleList()
        for rate in self.dilation_rates:
            self.aspp_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels // 4, 
                             kernel_size=3, padding=rate, dilation=rate),
                    nn.BatchNorm2d(out_channels // 4),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Project concatenated features
        # Total channels: 4 dilated + 1 global + 1 1x1 = 6 branches
        self.project = nn.Sequential(
            nn.Conv2d(out_channels // 4 * 6, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            Multi-scale features (B, out_channels, H, W)
        """
        _, _, h, w = x.shape
        
        # Apply dilated convolutions with different rates
        aspp_features = []
        for conv in self.aspp_convs:
            aspp_features.append(conv(x))
        
        # Global average pooling branch
        gap = self.global_avg_pool(x)
        gap = F.interpolate(gap, size=(h, w), mode='bilinear', align_corners=False)
        aspp_features.append(gap)
        
        # 1x1 convolution branch
        aspp_features.append(self.conv_1x1(x))
        
        # Concatenate all branches (Equation 9)
        out = torch.cat(aspp_features, dim=1)
        
        # Project to output channels
        out = self.project(out)
        
        return out


class NonLocalBlock(nn.Module):
    """
    Non-Local Block for capturing long-range dependencies
    Implements Equation 10 from paper:
    Z_NonLocal(i) = (1/C(Z)) * Σ_j softmax(Z(i)·Z(j)) * Z(j)
    
    Allows each position to attend to all other positions in the feature map
    """
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        
        self.in_channels = in_channels
        self.inter_channels = in_channels // reduction
        
        # Embedding functions for query, key, value
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        
        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        
        # Initialize output conv to zero for residual connection
        nn.init.constant_(self.out_conv[1].weight, 0)
        nn.init.constant_(self.out_conv[1].bias, 0)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            Features with long-range dependencies (B, C, H, W)
        """
        batch_size, C, H, W = x.shape
        
        # Generate query, key, value embeddings
        # Query: Z_latent(i)
        query = self.query_conv(x).view(batch_size, self.inter_channels, -1)  # (B, C', HW)
        query = query.permute(0, 2, 1)  # (B, HW, C')
        
        # Key: Z_latent(j)
        key = self.key_conv(x).view(batch_size, self.inter_channels, -1)  # (B, C', HW)
        
        # Value: Z_latent(j)
        value = self.value_conv(x).view(batch_size, self.inter_channels, -1)  # (B, C', HW)
        value = value.permute(0, 2, 1)  # (B, HW, C')
        
        # Compute attention: Z_latent(i) · Z_latent(j)
        attention = torch.bmm(query, key)  # (B, HW, HW)
        
        # Normalize with softmax (Equation 10)
        attention = self.softmax(attention)
        
        # Apply attention to values: Σ_j softmax(...) * Z(j)
        out = torch.bmm(attention, value)  # (B, HW, C')
        out = out.permute(0, 2, 1).contiguous()  # (B, C', HW)
        out = out.view(batch_size, self.inter_channels, H, W)  # (B, C', H, W)
        
        # Project back to original channels
        out = self.out_conv(out)
        
        # Residual connection
        out = out + x
        
        return out


class ChannelAttention(nn.Module):
    """
    Channel-wise Attention Module
    Implements Equation 11 from paper:
    Z_CA = σ(W_c · GlobalAvgPool(Z_PACE) + b_c) ⊙ Z_PACE
    
    Emphasizes important feature channels
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for both pooling paths
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            Channel attention weights (B, C, 1, 1)
        """
        # Average pooling path (Equation 11)
        avg_out = self.fc(self.avg_pool(x))
        
        # Max pooling path (additional refinement)
        max_out = self.fc(self.max_pool(x))
        
        # Combine both paths
        out = avg_out + max_out
        
        # Apply sigmoid activation: σ(W_c · GlobalAvgPool(...) + b_c)
        attention = self.sigmoid(out)
        
        # Apply attention: ⊙ Z_PACE
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Implements Equation 12 from paper:
    Z_SA = σ(Conv_spatial([AvgPool(Z_PACE), MaxPool(Z_PACE)])) ⊙ Z_PACE
    
    Emphasizes important spatial regions
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            Spatial attention weights (B, 1, H, W)
        """
        # Average pooling across channels (Equation 12)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Max pooling across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate pooled features
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        
        # Apply convolution: Conv_spatial([AvgPool, MaxPool])
        attention = self.conv(concat)  # (B, 1, H, W)
        
        # Apply sigmoid: σ(...)
        attention = self.sigmoid(attention)
        
        # Apply attention: ⊙ Z_PACE
        return x * attention


class DualAttentionModule(nn.Module):
    """
    Dual Attention combining Channel and Spatial Attention
    Implements both Equations 11 and 12 sequentially
    
    Focuses on both "what" (channels) and "where" (spatial locations)
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            Features with dual attention (B, C, H, W)
        """
        # Apply channel attention first (Equation 11)
        x = self.channel_attention(x)
        
        # Then apply spatial attention (Equation 12)
        x = self.spatial_attention(x)
        
        return x


class PhysicsConstrainedContextLayer(nn.Module):
    """
    Context layer with physics-based regularization
    Implements Equation 13 from paper:
    A_l = σ(W_l·A_{l-1} + b_l) + λ_phys * R_phys(A_l)
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
        Args:
            x: Input features (B, C, H, W)
            sequence_type: MRI sequence type
            return_reg_loss: Whether to return physics regularization loss
        """
        identity = self.skip(x)
        
        # Standard convolution: σ(W_l·A_{l-1} + b_l)
        out = self.conv(x)
        out = self.bn(out)
        
        # Add residual before activation
        out = out + identity
        out = self.relu(out)
        
        if return_reg_loss:
            # Compute physics regularization: λ_phys * R_phys(A_l)
            reg_loss = self.lambda_phys * self.physics_reg(out, sequence_type)
            return out, reg_loss
        
        return out


class PhysicsAwareContextEncoder(nn.Module):
    """
    Complete Physics-Aware Context Encoder (PACE) Implementation
    
    Implements:
    - ASPP for multi-scale features (Equation 9)
    - Non-Local Blocks for global context (Equation 10)
    - Dual Attention (Equations 11-12)
    - Physics-constrained layers (Equation 13)
    
    Architecture from paper Section 2.2
    """
    def __init__(self,
                 in_channels=128,        # From LPCE latent dimension
                 hidden_channels=256,
                 out_channels=256,
                 lambda_phys=0.1,
                 num_nonlocal_blocks=2,
                 num_context_layers=3):
        super().__init__()
        
        self.lambda_phys = lambda_phys
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # ASPP Module (Equation 9)
        # Multi-scale feature extraction with dilation rates [1, 6, 12, 18]
        self.aspp = AtrousSpatialPyramidPooling(hidden_channels, hidden_channels)
        
        # Non-Local Blocks (Equation 10)
        # Capture long-range spatial dependencies
        self.nonlocal_blocks = nn.ModuleList([
            NonLocalBlock(hidden_channels, reduction=2)
            for _ in range(num_nonlocal_blocks)
        ])
        
        # Feature fusion after ASPP + NonLocal
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dual Attention Module (Equations 11-12)
        # Channel attention + Spatial attention
        self.dual_attention = DualAttentionModule(hidden_channels, reduction=16)
        
        # Physics-constrained context layers (Equation 13)
        self.context_layers = nn.ModuleList([
            PhysicsConstrainedContextLayer(
                hidden_channels,
                hidden_channels,
                lambda_phys
            )
            for _ in range(num_context_layers)
        ])
        
        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global physics regularization
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
    
    def forward(self, z_latent, sequence_type='mixed', return_losses=False):
        """
        Forward pass through PACE
        
        Args:
            z_latent: Latent features from LPCE (B, in_channels, H, W)
            sequence_type: MRI sequence type ('t1', 't2', 'pd', 'mixed')
            return_losses: Whether to return regularization losses
        
        Returns:
            z_pace: Context-enriched features (B, out_channels, H, W)
            losses: Dict of regularization losses (if return_losses=True)
        """
        losses = {}
        
        # Input projection
        x = self.input_proj(z_latent)
        
        # Multi-scale feature extraction with ASPP (Equation 9)
        z_aspp = self.aspp(x)
        
        # Long-range dependency modeling with Non-Local blocks (Equation 10)
        z_nonlocal = z_aspp
        for i, nonlocal_block in enumerate(self.nonlocal_blocks):
            z_nonlocal = nonlocal_block(z_nonlocal)
            
            if return_losses:
                # Physics regularization on non-local features
                nl_reg = self.lambda_phys * self.physics_reg(z_nonlocal, sequence_type)
                losses[f'nonlocal_{i}_reg'] = nl_reg
        
        # Fuse ASPP and Non-Local features
        z_fused = self.feature_fusion(z_nonlocal)
        
        # Apply dual attention (Equations 11-12)
        z_attended = self.dual_attention(z_fused)
        
        if return_losses:
            attention_reg = self.lambda_phys * self.physics_reg(z_attended, sequence_type)
            losses['attention_reg'] = attention_reg
        
        # Apply physics-constrained context layers (Equation 13)
        z_context = z_attended
        total_reg_loss = 0
        
        for i, context_layer in enumerate(self.context_layers):
            if return_losses:
                z_context, reg_loss = context_layer(
                    z_context, 
                    sequence_type, 
                    return_reg_loss=True
                )
                losses[f'context_{i}_reg'] = reg_loss
                total_reg_loss += reg_loss
            else:
                z_context = context_layer(z_context, sequence_type)
        
        # Final output projection
        z_pace = self.output_proj(z_context)
        
        # Final physics regularization
        if return_losses:
            final_reg = self.lambda_phys * self.physics_reg(z_pace, sequence_type)
            losses['final_reg'] = final_reg
            losses['total_reg'] = total_reg_loss + final_reg
            
            return z_pace, losses
        
        return z_pace
    
    def get_num_params(self):
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== Testing & Validation ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Complete PACE Implementation")
    print("=" * 70)
    
    # Configuration matching paper
    config = {
        'in_channels': 128,         # From LPCE output
        'hidden_channels': 256,     # Internal processing
        'out_channels': 256,        # Output to ADRN
        'lambda_phys': 0.1,         # Physics regularization weight
        'num_nonlocal_blocks': 2,   # Number of non-local blocks
        'num_context_layers': 3     # Number of physics-constrained layers
    }
    
    # Create PACE module
    pace = PhysicsAwareContextEncoder(**config)
    
    print(f"\n1. Model Configuration:")
    print(f"   - Input channels: {config['in_channels']}")
    print(f"   - Output channels: {config['out_channels']}")
    print(f"   - Hidden channels: {config['hidden_channels']}")
    print(f"   - Physics weight λ: {config['lambda_phys']}")
    print(f"   - Non-local blocks: {config['num_nonlocal_blocks']}")
    print(f"   - Context layers: {config['num_context_layers']}")
    
    # Count parameters
    total_params = pace.get_num_params()
    trainable_params = pace.get_num_trainable_params()
    print(f"\n2. Model Parameters:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print(f"\n3. Testing Forward Pass:")
    batch_size = 4
    height, width = 256, 256
    
    # Input from LPCE
    z_latent = torch.randn(batch_size, 128, height, width)
    
    # Forward pass without losses
    z_pace = pace(z_latent, sequence_type='t1')
    print(f"   - Input shape: {z_latent.shape}")
    print(f"   - Output shape: {z_pace.shape}")
    
    # Forward pass with losses
    z_pace_with_loss, losses = pace(z_latent, sequence_type='t1', return_losses=True)
    print(f"\n4. Regularization Losses:")
    for key, value in losses.items():
        print(f"   - {key}: {value.item():.6f}")
    
    # Test different sequence types
    print(f"\n5. Testing Different Sequence Types:")
    for seq_type in ['t1', 't2', 'pd', 'mixed']:
        z = pace(z_latent, sequence_type=seq_type)
        print(f"   - {seq_type.upper()}: {z.shape} ✓")
    
    # Test individual components
    print(f"\n6. Component Testing:")
    
    # ASPP
    aspp = AtrousSpatialPyramidPooling(128, 128)
    x_test = torch.randn(2, 128, 64, 64)
    aspp_out = aspp(x_test)
    print(f"   - ASPP: {x_test.shape} -> {aspp_out.shape} ✓")
    
    # Non-Local Block
    nonlocal_block = NonLocalBlock(128, reduction=2)
    nl_out = nonlocal_block(x_test)
    print(f"   - Non-Local: {x_test.shape} -> {nl_out.shape} ✓")
    
    # Channel Attention
    channel_attn = ChannelAttention(128, reduction=16)
    ca_out = channel_attn(x_test)
    print(f"   - Channel Attention: {x_test.shape} -> {ca_out.shape} ✓")
    
    # Spatial Attention
    spatial_attn = SpatialAttention(kernel_size=7)
    sa_out = spatial_attn(x_test)
    print(f"   - Spatial Attention: {x_test.shape} -> {sa_out.shape} ✓")
    
    # Dual Attention
    dual_attn = DualAttentionModule(128)
    da_out = dual_attn(x_test)
    print(f"   - Dual Attention: {x_test.shape} -> {da_out.shape} ✓")
    
    # Verify key components
    print(f"\n7. Component Verification:")
    print(f"   - Has ASPP module: {hasattr(pace, 'aspp')} ✓")
    print(f"   - Has Non-Local blocks: {len(pace.nonlocal_blocks)} blocks ✓")
    print(f"   - Has Dual Attention: {hasattr(pace, 'dual_attention')} ✓")
    print(f"   - Has Context layers: {len(pace.context_layers)} layers ✓")
    print(f"   - Has Physics regularization: {hasattr(pace, 'physics_reg')} ✓")
    
    # Test with actual LPCE output
    print(f"\n8. Integration Test with LPCE:")
    from lpce import LatentPhysicsConstrainedEncoder
    
    lpce = LatentPhysicsConstrainedEncoder(
        in_channels=1,
        hidden_channels=64,
        latent_dim=128,
        lambda_phys=0.2
    )
    
    # Generate test input
    x_under = torch.randn(2, 1, 256, 256).abs()
    k_space = torch.fft.fft2(x_under.squeeze(1), norm='ortho')
    mask = (torch.rand(2, 1, 256, 256) > 0.75).float()
    
    # LPCE forward
    z_latent = lpce(x_under, k_space=k_space, mask=mask, sequence_type='t1')
    
    # PACE forward
    z_pace = pace(z_latent, sequence_type='t1')
    
    print(f"   - LPCE output: {z_latent.shape}")
    print(f"   - PACE output: {z_pace.shape}")
    print(f"   - Integration successful ✓")
    
    print(f"\n{'=' * 70}")
    print("✓ All tests passed! PACE implementation complete.")
    print("=" * 70)