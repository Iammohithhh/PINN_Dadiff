import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicConvolution(nn.Module):
    """
    Dynamic convolution layer that adapts kernels based on input features.
    Implements Equations 20-21 from the paper.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, num_kernels=4):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.padding = kernel_size // 2
        
        # Generate multiple kernels
        self.kernels = nn.Parameter(
            torch.randn(num_kernels, out_channels, in_channels, 
                       kernel_size, kernel_size)
        )
        
        # Attention network to generate kernel weights
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_kernels, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Learnable parameters for dynamic kernel generation
        self.weight_generator = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, out_channels * in_channels * kernel_size * kernel_size)
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, context_features):
        """
        Dynamic convolution with adaptive kernels.
        Implements: K_dynamic = σ(W_k·Z_PACE + b_k)
                    Z_filtered = Conv(Z_ADRN, K_dynamic)
        
        Args:
            x: Input features to be filtered (Z_ADRN)
            context_features: Context for kernel adaptation (Z_PACE)
        """
        B, C, H, W = x.shape
        
        # Generate attention weights for kernel selection (Equation 20)
        attn_weights = self.attention(context_features)  # (B, num_kernels, 1, 1)
        
        # Weighted combination of kernels
        # K_dynamic = Σ(attention_i * kernel_i)
        dynamic_kernel = torch.zeros(B, self.out_channels, self.in_channels, 
                                     self.kernel_size, self.kernel_size, 
                                     device=x.device)
        
        for i in range(self.num_kernels):
            dynamic_kernel += attn_weights[:, i:i+1, :, :] * self.kernels[i:i+1]
        
        # Apply dynamic convolution (Equation 21)
        output = []
        for b in range(B):
            # Apply convolution with batch-specific kernel
            out_b = F.conv2d(x[b:b+1], dynamic_kernel[b], 
                            padding=self.padding)
            output.append(out_b)
        
        output = torch.cat(output, dim=0)
        output = self.bn(output)
        
        return output


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    Implements Equations 22-23 from the paper.
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        
        # Layer norm
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x):
        """
        Multi-head self-attention.
        Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
        """
        B, C, H, W = x.shape
        
        # Reshape: (B, C, H, W) -> (B, HW, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Apply layer norm
        x_norm = self.norm(x_flat)
        
        # Generate Q, K, V (Equation 22)
        qkv = self.qkv(x_norm).reshape(B, H*W, 3, self.num_heads, 
                                        self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, HW, head_dim)
        
        # Scaled dot-product attention
        # Attention = softmax(Q·K^T / sqrt(d_k))
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, HW, HW)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values: Attention·V
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        
        # Output projection
        x_attn = self.proj(x_attn)
        
        # Residual connection
        x_out = x_flat + x_attn
        
        # Reshape back: (B, HW, C) -> (B, C, H, W)
        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        
        return x_out


class TransformerBlock(nn.Module):
    """
    Complete transformer block with self-attention and feed-forward network.
    Implements Equation 23 with multi-head attention.
    """
    def __init__(self, channels, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(channels, num_heads)
        
        # Feed-forward network
        mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mlp_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(mlp_hidden_dim, channels, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
    
    def forward(self, x):
        """
        Transformer block with residual connections.
        Implements: MultiHead(Q, K, V) = Concat(head_1, ..., head_h)·W_O
        """
        # Self-attention with residual
        x = x + self.attn(x)
        
        # Feed-forward with residual
        identity = x
        B, C, H, W = x.shape
        
        # Apply layer norm in channel dimension
        x_flat = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x_norm = self.norm2(x_flat).transpose(1, 2).reshape(B, C, H, W)
        
        x = identity + self.mlp(x_norm)
        
        return x


class PhysicsInformedRegularization(nn.Module):
    """
    Physics-based regularization for ART.
    Enforces MRI signal properties and smoothness.
    Implements Equation 24-25.
    """
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, activations):
        """
        Compute physics-based regularization.
        R_phys ensures smoothness and physical consistency.
        """
        # Gradient-based smoothness (spatial regularization)
        dx = activations[:, :, :, 1:] - activations[:, :, :, :-1]
        dy = activations[:, :, 1:, :] - activations[:, :, :-1, :]
        
        # Total variation for smoothness
        smoothness_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
        
        # Signal intensity consistency (encourage realistic MRI intensities)
        intensity_reg = torch.mean((activations - torch.mean(activations))**2)
        
        return self.gamma * (smoothness_loss + 0.1 * intensity_reg)


class PhysicsConstrainedDynamicConv(nn.Module):
    """
    Dynamic convolution with physics-based regularization.
    Implements Equation 24 from the paper.
    """
    def __init__(self, in_channels, out_channels, lambda_phys=0.2):
        super().__init__()
        
        self.dynamic_conv = DynamicConvolution(in_channels, out_channels)
        self.lambda_phys = lambda_phys
        self.physics_reg = PhysicsInformedRegularization(gamma=0.1)
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x, context, return_reg_loss=False):
        """
        Dynamic convolution with physics regularization.
        K_dynamic = σ(W_k·Z_PACE + b_k) + λ_phys·R_phys(Z_PACE)
        """
        # Apply dynamic convolution (Equation 24)
        out = self.dynamic_conv(x, context)
        out = self.activation(out)
        
        if return_reg_loss:
            # Compute physics regularization
            reg_loss = self.lambda_phys * self.physics_reg(out)
            return out, reg_loss
        
        return out


class PhysicsConstrainedAttention(nn.Module):
    """
    Self-attention with physics-based regularization.
    Implements Equation 25 from the paper.
    """
    def __init__(self, channels, num_heads=8, lambda_phys=0.2):
        super().__init__()
        
        self.attn = MultiHeadSelfAttention(channels, num_heads)
        self.lambda_phys = lambda_phys
        self.physics_reg = PhysicsInformedRegularization(gamma=0.1)
    
    def forward(self, x, return_reg_loss=False):
        """
        Self-attention with physics regularization.
        Attention(Q,K,V) = softmax(Q·K^T/sqrt(d_k)) + λ_phys·R_phys(Q,K,V)
        """
        # Apply self-attention (Equation 25)
        out = self.attn(x)
        
        if return_reg_loss:
            # Compute physics regularization on attention output
            reg_loss = self.lambda_phys * self.physics_reg(out)
            return out, reg_loss
        
        return out


class AdaptiveReconstructionTransformer(nn.Module):
    """
    Complete ART implementation.
    Synthesizes features from PACE and ADRN with dynamic convolutions,
    transformer blocks, and physics-informed constraints.
    Implements Equation 19 and 26 from the paper.
    """
    def __init__(self,
                 pace_channels=256,
                 adrn_channels=256,
                 hidden_channels=512,
                 out_channels=1,
                 num_transformer_blocks=4,
                 num_heads=8,
                 lambda_phys=0.2,
                 dropout=0.1):
        super().__init__()
        
        self.pace_channels = pace_channels
        self.adrn_channels = adrn_channels
        self.hidden_channels = hidden_channels
        self.lambda_phys = lambda_phys
        
        # Input projection for PACE and ADRN features
        self.pace_proj = nn.Sequential(
            nn.Conv2d(pace_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        self.adrn_proj = nn.Sequential(
            nn.Conv2d(adrn_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Physics-constrained dynamic convolutions (Equation 24)
        self.dynamic_conv_layers = nn.ModuleList([
            PhysicsConstrainedDynamicConv(
                hidden_channels if i == 0 else hidden_channels,
                hidden_channels,
                lambda_phys
            )
            for i in range(2)
        ])
        
        # Transformer blocks for global feature refinement (Equation 23)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_channels, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        # Physics-constrained attention layers (Equation 25)
        self.physics_attn_layers = nn.ModuleList([
            PhysicsConstrainedAttention(hidden_channels, num_heads, lambda_phys)
            for _ in range(2)
        ])
        
        # Feature fusion module
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Refinement layers
        self.refinement = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 
                         kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(2)
        ])
        
        # Output projection (Equation 26)
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, out_channels, 
                     kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1] range for MRI
        )
        
        # Physics regularization
        self.physics_reg = PhysicsInformedRegularization(gamma=0.1)
    
    def forward(self, z_pace, z_adrn, return_losses=False):
        """
        Complete ART forward pass.
        Implements: Z_ART = DynamicConv(Z_PACE, Z_ADRN) + Transformer(Z_PACE, Z_ADRN)
        
        Args:
            z_pace: Context features from PACE (B, C_pace, H, W)
            z_adrn: Refined features from ADRN (B, C_adrn, H, W)
            return_losses: Whether to return regularization losses
            
        Returns:
            z_art_final: Final reconstructed MRI image (B, 1, H, W)
            losses: Dictionary of regularization losses (optional)
        """
        losses = {}
        total_reg_loss = 0
        
        # Project inputs to hidden dimension
        z_pace_proj = self.pace_proj(z_pace)
        z_adrn_proj = self.adrn_proj(z_adrn)
        
        # Dynamic convolution path (Equation 19, 24)
        dynamic_features = z_adrn_proj
        for i, dyn_conv in enumerate(self.dynamic_conv_layers):
            if return_losses:
                dynamic_features, reg_loss = dyn_conv(
                    dynamic_features, z_pace_proj, return_reg_loss=True
                )
                losses[f'dynamic_conv_{i}_reg'] = reg_loss
                total_reg_loss += reg_loss
            else:
                dynamic_features = dyn_conv(dynamic_features, z_pace_proj)
        
        # Transformer path for global dependencies (Equation 23)
        transformer_features = z_pace_proj
        for i, transformer in enumerate(self.transformer_blocks):
            transformer_features = transformer(transformer_features)
        
        # Apply physics-constrained attention (Equation 25)
        for i, phys_attn in enumerate(self.physics_attn_layers):
            if return_losses:
                transformer_features, reg_loss = phys_attn(
                    transformer_features, return_reg_loss=True
                )
                losses[f'physics_attn_{i}_reg'] = reg_loss
                total_reg_loss += reg_loss
            else:
                transformer_features = phys_attn(transformer_features)
        
        # Combine dynamic conv and transformer paths (Equation 19)
        # Z_ART = DynamicConv(Z_PACE, Z_ADRN) + Transformer(Z_PACE, Z_ADRN)
        combined = torch.cat([dynamic_features, transformer_features], dim=1)
        z_art = self.fusion(combined)
        
        # Refinement layers
        for refine_layer in self.refinement:
            z_art = z_art + refine_layer(z_art)  # Residual connection
        
        # Final output projection (Equation 26)
        # Z_ART_final = σ(W_out·Z_ART + b_out)
        z_art_final = self.output_proj(z_art)
        
        # Final physics regularization
        if return_losses:
            final_reg = self.lambda_phys * self.physics_reg(z_art_final)
            losses['final_reg'] = final_reg
            losses['total_reg'] = total_reg_loss + final_reg
            return z_art_final, losses
        
        return z_art_final


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Adaptive Reconstruction Transformer (ART) ===\n")
    
    # Create ART module
    art = AdaptiveReconstructionTransformer(
        pace_channels=256,
        adrn_channels=256,
        hidden_channels=512,
        out_channels=1,
        num_transformer_blocks=4,
        num_heads=8,
        lambda_phys=0.2,
        dropout=0.1
    )
    
    # Example inputs from PACE and ADRN
    batch_size = 2
    height, width = 128, 128
    z_pace = torch.randn(batch_size, 256, height, width)
    z_adrn = torch.randn(batch_size, 256, height, width)
    
    # Forward pass with losses
    print("--- Forward Pass with Regularization Losses ---")
    z_art_final, losses = art(z_pace, z_adrn, return_losses=True)
    
    print(f"Z_PACE shape: {z_pace.shape}")
    print(f"Z_ADRN shape: {z_adrn.shape}")
    print(f"Z_ART_final shape: {z_art_final.shape}")
    print(f"\nRegularization losses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")
    
    # Forward pass without losses (inference mode)
    print("\n--- Inference Mode ---")
    art.eval()
    with torch.no_grad():
        z_art_inference = art(z_pace, z_adrn, return_losses=False)
    print(f"Inference output shape: {z_art_inference.shape}")
    print(f"Output range: [{z_art_inference.min().item():.4f}, "
          f"{z_art_inference.max().item():.4f}]")
    
    # Test individual components
    print("\n--- Testing Individual Components ---")
    
    # Dynamic Convolution
    print("\n1. Dynamic Convolution:")
    dyn_conv = DynamicConvolution(256, 512, kernel_size=3, num_kernels=4)
    dyn_out = dyn_conv(z_adrn, z_pace)
    print(f"   Input shape: {z_adrn.shape}")
    print(f"   Output shape: {dyn_out.shape}")
    
    # Multi-head Self-Attention
    print("\n2. Multi-head Self-Attention:")
    mhsa = MultiHeadSelfAttention(256, num_heads=8)
    attn_out = mhsa(z_pace)
    print(f"   Input shape: {z_pace.shape}")
    print(f"   Output shape: {attn_out.shape}")
    
    # Transformer Block
    print("\n3. Transformer Block:")
    transformer = TransformerBlock(256, num_heads=8)
    trans_out = transformer(z_pace)
    print(f"   Input shape: {z_pace.shape}")
    print(f"   Output shape: {trans_out.shape}")
    
    # Physics-Constrained Dynamic Conv
    print("\n4. Physics-Constrained Dynamic Convolution:")
    phys_dyn_conv = PhysicsConstrainedDynamicConv(256, 256, lambda_phys=0.2)
    phys_out, reg_loss = phys_dyn_conv(z_adrn, z_pace, return_reg_loss=True)
    print(f"   Input shape: {z_adrn.shape}")
    print(f"   Output shape: {phys_out.shape}")
    print(f"   Regularization loss: {reg_loss.item():.6f}")
    
    # Physics-Constrained Attention
    print("\n5. Physics-Constrained Attention:")
    phys_attn = PhysicsConstrainedAttention(256, num_heads=8, lambda_phys=0.2)
    phys_attn_out, attn_reg = phys_attn(z_pace, return_reg_loss=True)
    print(f"   Input shape: {z_pace.shape}")
    print(f"   Output shape: {phys_attn_out.shape}")
    print(f"   Regularization loss: {attn_reg.item():.6f}")
    
    # Model statistics
    print("\n--- Model Statistics ---")
    total_params = sum(p.numel() for p in art.parameters())
    trainable_params = sum(p.numel() for p in art.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Memory estimation
    input_memory = (z_pace.element_size() * z_pace.nelement() + 
                   z_adrn.element_size() * z_adrn.nelement()) / (1024**2)
    output_memory = z_art_final.element_size() * z_art_final.nelement() / (1024**2)
    print(f"\nMemory usage:")
    print(f"  Input (PACE + ADRN): {input_memory:.2f} MB")
    print(f"  Output: {output_memory:.2f} MB")
    
    print("\n=== ART Module Testing Complete ===")