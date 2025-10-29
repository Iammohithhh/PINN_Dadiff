import torch
import torch.nn as nn
import torch.nn.functional as F


class AtrousSpatialPyramidPooling(nn.Module):
    """
    ASPP module for multi-scale feature extraction.
    Implements Equation 9 from the paper.
    Uses parallel convolutions with different dilation rates.
    """
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 6, 12, 18]):
        super().__init__()
        
        self.dilation_rates = dilation_rates
        
        # Parallel atrous convolutions with different dilation rates
        self.aspp_branches = nn.ModuleList()
        for rate in dilation_rates:
            if rate == 1:
                # Standard convolution for rate=1
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 
                             kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                # Atrous convolution with dilation
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 
                             kernel_size=3, padding=rate, 
                             dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            self.aspp_branches.append(branch)
        
        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 
                     kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Projection after concatenation
        total_channels = out_channels * (len(dilation_rates) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 
                     kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        """
        Multi-scale feature extraction using atrous convolutions.
        Z_ASPP = Concat(Conv_r=1, Conv_r=6, Conv_r=12, Conv_r=18)
        """
        H, W = x.shape[2:]
        
        # Apply all ASPP branches
        aspp_features = []
        for branch in self.aspp_branches:
            aspp_features.append(branch(x))
        
        # Global pooling branch
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=(H, W), 
                                    mode='bilinear', align_corners=False)
        aspp_features.append(global_feat)
        
        # Concatenate all features
        out = torch.cat(aspp_features, dim=1)
        
        # Project to output channels
        out = self.project(out)
        
        return out


class NonLocalBlock(nn.Module):
    """
    Non-local block for capturing long-range dependencies.
    Implements Equation 10 from the paper.
    """
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        
        self.in_channels = in_channels
        self.inter_channels = in_channels // reduction
        
        # Query, Key, Value transformations
        self.query = nn.Conv2d(in_channels, self.inter_channels, 
                              kernel_size=1)
        self.key = nn.Conv2d(in_channels, self.inter_channels, 
                            kernel_size=1)
        self.value = nn.Conv2d(in_channels, self.inter_channels, 
                              kernel_size=1)
        
        # Output projection
        self.out = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, 
                     kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        
        # Initialize weights
        nn.init.constant_(self.out[1].weight, 0)
        nn.init.constant_(self.out[1].bias, 0)
    
    def forward(self, x):
        """
        Non-local operation with self-attention.
        Z_NonLocal(i) = (1/C(Z)) * Σ_j softmax(Z(i)·Z(j)) * Z(j)
        """
        B, C, H, W = x.shape
        
        # Generate query, key, value
        q = self.query(x).view(B, self.inter_channels, -1)  # (B, C', HW)
        k = self.key(x).view(B, self.inter_channels, -1)    # (B, C', HW)
        v = self.value(x).view(B, self.inter_channels, -1)  # (B, C', HW)
        
        # Transpose for matrix multiplication
        q = q.permute(0, 2, 1)  # (B, HW, C')
        
        # Compute attention: softmax(Q·K^T)
        attention = torch.bmm(q, k)  # (B, HW, HW)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values: Attention·V
        v = v.permute(0, 2, 1)  # (B, HW, C')
        out = torch.bmm(attention, v)  # (B, HW, C')
        
        # Reshape back
        out = out.permute(0, 2, 1).contiguous()  # (B, C', HW)
        out = out.view(B, self.inter_channels, H, W)
        
        # Project back to original channels
        out = self.out(out)
        
        # Residual connection
        return x + out


class ChannelAttention(nn.Module):
    """
    Channel-wise attention mechanism.
    Implements Equation 11 from the paper.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 
                     kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 
                     kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Channel attention: Z_CA = σ(W_c·GlobalAvgPool(Z_PACE) + b_c)·Z_PACE
        """
        # Average pooling path
        avg_out = self.fc(self.avg_pool(x))
        
        # Max pooling path
        max_out = self.fc(self.max_pool(x))
        
        # Combine and generate attention weights
        attention = self.sigmoid(avg_out + max_out)
        
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism.
    Implements Equation 12 from the paper.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Spatial attention: Z_SA = σ(Conv([AvgPool(Z), MaxPool(Z)]))·Z_PACE
        """
        # Average pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # Max pooling across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate
        concat = torch.cat([avg_out, max_out], dim=1)
        
        # Generate spatial attention map
        attention = self.sigmoid(self.conv(concat))
        
        return x * attention


class DualAttentionModule(nn.Module):
    """
    Combined channel and spatial attention.
    Implements both Equations 11 and 12.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)
    
    def forward(self, x):
        """
        Apply channel attention followed by spatial attention
        """
        # Channel attention
        x = self.channel_attention(x)
        
        # Spatial attention
        x = self.spatial_attention(x)
        
        return x


class PhysicsAwareContextEncoder(nn.Module):
    """
    Complete PACE implementation.
    Combines ASPP, Non-Local Blocks, and Dual Attention.
    Implements Equation 8 from the paper.
    """
    def __init__(self, 
                 in_channels=128,
                 hidden_channels=256,
                 out_channels=256,
                 dilation_rates=[1, 6, 12, 18],
                 num_nonlocal_blocks=2,
                 lambda_phys=0.2):
        super().__init__()
        
        self.lambda_phys = lambda_phys
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # ASPP module for multi-scale features (Equation 9)
        self.aspp = AtrousSpatialPyramidPooling(
            hidden_channels, 
            hidden_channels, 
            dilation_rates
        )
        
        # Multiple Non-Local blocks for long-range dependencies (Equation 10)
        self.nonlocal_blocks = nn.ModuleList([
            NonLocalBlock(hidden_channels, reduction=2)
            for _ in range(num_nonlocal_blocks)
        ])
        
        # Fusion of ASPP and Non-Local features
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 
                     kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dual attention mechanism (Equations 11 & 12)
        self.dual_attention = DualAttentionModule(
            hidden_channels, 
            reduction=16
        )
        
        # Physics-based regularization layers (Equation 13)
        self.physics_layers = nn.ModuleList([
            self._make_physics_layer(hidden_channels, hidden_channels)
            for _ in range(2)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Physics regularization module
        self.physics_reg = PhysicsInformedRegularization(gamma=0.1)
    
    def _make_physics_layer(self, in_channels, out_channels):
        """
        Create a physics-constrained layer (Equation 13)
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, z_latent, return_losses=False):
        """
        Forward pass through PACE.
        Implements: Z_PACE = ASPP(Z_latent) + NonLocal(Z_latent)
        
        Args:
            z_latent: Latent features from LPCE (B, C, H, W)
            return_losses: Whether to return regularization losses
            
        Returns:
            z_pace: Context-aware features
            losses: Dictionary of regularization losses (optional)
        """
        losses = {}
        
        # Input projection
        x = self.input_proj(z_latent)
        
        # ASPP for multi-scale features
        aspp_features = self.aspp(x)
        
        # Non-Local blocks for long-range dependencies
        nonlocal_features = x
        for nl_block in self.nonlocal_blocks:
            nonlocal_features = nl_block(nonlocal_features)
        
        # Combine ASPP and Non-Local features (Equation 8)
        # Z_PACE = ASPP(Z_latent) + NonLocal(Z_latent)
        combined = torch.cat([aspp_features, nonlocal_features], dim=1)
        fused = self.fusion(combined)
        
        # Apply dual attention (Equations 11 & 12)
        attended = self.dual_attention(fused)
        
        # Physics-constrained refinement (Equation 13)
        # A_l = σ(W_l·A_{l-1} + b_l) + λ_phys·R_phys(A_l)
        total_reg_loss = 0
        x_refined = attended
        for i, layer in enumerate(self.physics_layers):
            x_refined = layer(x_refined)
            
            if return_losses:
                # Compute physics regularization
                reg_loss = self.lambda_phys * self.physics_reg(x_refined)
                losses[f'physics_layer_{i}_reg'] = reg_loss
                total_reg_loss += reg_loss
        
        # Output projection
        z_pace = self.output_proj(x_refined)
        
        # Final physics regularization
        if return_losses:
            final_reg = self.lambda_phys * self.physics_reg(z_pace)
            losses['final_reg'] = final_reg
            losses['total_reg'] = total_reg_loss + final_reg
            return z_pace, losses
        
        return z_pace


class PhysicsInformedRegularization(nn.Module):
    """
    Physics-based regularization for PACE.
    Enforces smoothness and physical consistency.
    """
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, activations):
        """
        Compute physics-based regularization loss.
        Encourages smoothness in the feature maps.
        """
        # Gradient-based smoothness
        dx = activations[:, :, :, 1:] - activations[:, :, :, :-1]
        dy = activations[:, :, 1:, :] - activations[:, :, :-1, :]
        
        # Total variation loss for smoothness
        smoothness_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
        
        return self.gamma * smoothness_loss


# Example usage and testing
if __name__ == "__main__":
    # Create PACE module
    pace = PhysicsAwareContextEncoder(
        in_channels=128,      # From LPCE output
        hidden_channels=256,
        out_channels=256,
        dilation_rates=[1, 6, 12, 18],
        num_nonlocal_blocks=2,
        lambda_phys=0.2
    )
    
    # Example input from LPCE
    batch_size = 4
    height, width = 256, 256
    z_latent = torch.randn(batch_size, 128, height, width)
    
    # Forward pass
    z_pace, losses = pace(z_latent, return_losses=True)
    
    print(f"Input shape: {z_latent.shape}")
    print(f"Output shape: {z_pace.shape}")
    print(f"Regularization losses: {losses}")
    
    # Test individual components
    print("\n--- Testing ASPP ---")
    aspp = AtrousSpatialPyramidPooling(128, 128)
    aspp_out = aspp(z_latent)
    print(f"ASPP output shape: {aspp_out.shape}")
    
    print("\n--- Testing Non-Local Block ---")
    nonlocal = NonLocalBlock(128)
    nl_out = nonlocal(z_latent)
    print(f"Non-Local output shape: {nl_out.shape}")
    
    print("\n--- Testing Dual Attention ---")
    dual_attn = DualAttentionModule(128)
    attn_out = dual_attn(z_latent)
    print(f"Dual Attention output shape: {attn_out.shape}")
    
    # Total parameters
    total_params = sum(p.numel() for p in pace.parameters())
    print(f"\nTotal PACE parameters: {total_params:,}")
    
    # Memory usage estimation
    print(f"\nEstimated memory for single forward pass:")
    print(f"  Input: {z_latent.element_size() * z_latent.nelement() / 1024**2:.2f} MB")
    print(f"  Output: {z_pace.element_size() * z_pace.nelement() / 1024**2:.2f} MB")