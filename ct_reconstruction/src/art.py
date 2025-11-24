"""
Adaptive Reconstruction Transformer (ART) for CT Reconstruction.

Final synthesis module that combines PACE and ADRN outputs through
dynamic convolutions and transformer-based attention with CT physics constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .ct_physics import (
    RadonTransform,
    FilteredBackProjection,
    CTDataConsistency,
    WeightedLeastSquaresLoss,
    TotalVariationLoss
)


class DynamicConv(nn.Module):
    """
    Dynamic Convolution layer.

    Generates input-dependent convolution kernels for adaptive filtering.
    K_dynamic = sigma(W_k @ Z_PACE + b_k)
    Z_filtered = Conv(Z_ADRN, K_dynamic)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_experts: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_experts = num_experts

        # Expert convolution kernels
        self.weight = nn.Parameter(
            torch.randn(num_experts, out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(num_experts, out_channels))

        # Attention over experts
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic convolution.

        Args:
            x: Input features (B, C, H, W)
            context: Context for kernel generation (B, C, H, W)

        Returns:
            Filtered features (B, out_channels, H, W)
        """
        B = x.shape[0]

        # Compute attention weights over experts
        attn = self.attention(context)  # (B, num_experts)

        # Combine expert weights
        # weight: (num_experts, out_ch, in_ch, k, k)
        # attn: (B, num_experts)
        weight = torch.einsum('be,eoikk->boikk', attn, self.weight.flatten(-2, -1).unsqueeze(0).expand(B, -1, -1, -1, -1).squeeze(0))
        weight = weight.view(B * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        bias = torch.einsum('be,eo->bo', attn, self.bias).view(-1)

        # Apply grouped convolution
        x = x.view(1, B * self.in_channels, x.shape[2], x.shape[3])
        padding = self.kernel_size // 2
        out = F.conv2d(x, weight, bias, padding=padding, groups=B)
        out = out.view(B, self.out_channels, out.shape[2], out.shape[3])

        return out


class SimpleDynamicConv(nn.Module):
    """Simplified dynamic convolution using feature modulation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)

        # Dynamic kernel generator
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply modulated convolution."""
        # Generate modulation from context
        mod = self.kernel_gen(context)  # (B, out_channels)
        mod = mod.unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1)

        # Convolve and modulate
        out = self.conv(x)
        out = self.bn(out)
        out = out * mod

        return F.relu(out)


class TransformerFusion(nn.Module):
    """
    Transformer-based feature fusion.

    Uses self-attention to integrate PACE and ADRN features.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList([
                    nn.LayerNorm(channels),
                    nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True),
                    nn.LayerNorm(channels),
                    nn.Sequential(
                        nn.Linear(channels, channels * 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(channels * 4, channels),
                        nn.Dropout(dropout)
                    )
                ])
            )

        self.final_norm = nn.LayerNorm(channels)

    def forward(self, pace_features: torch.Tensor, adrn_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse features using cross-attention.

        Args:
            pace_features: Features from PACE (B, C, H, W)
            adrn_features: Features from ADRN (B, C, H, W)

        Returns:
            Fused features (B, C, H, W)
        """
        B, C, H, W = pace_features.shape

        # Flatten to sequence
        pace_seq = pace_features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        adrn_seq = adrn_features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        # Concatenate for joint processing
        x = torch.cat([pace_seq, adrn_seq], dim=1)  # (B, 2*H*W, C)

        # Apply transformer layers
        for norm1, attn, norm2, mlp in self.layers:
            # Self-attention with residual
            x_norm = norm1(x)
            x = x + attn(x_norm, x_norm, x_norm)[0]

            # MLP with residual
            x = x + mlp(norm2(x))

        x = self.final_norm(x)

        # Take first half (PACE-aligned) and reshape
        x = x[:, :H*W, :].permute(0, 2, 1).view(B, C, H, W)

        return x


class PhysicsRegularizationART(nn.Module):
    """
    Physics-based regularization for ART.

    Ensures final reconstruction adheres to CT physics constraints.
    """

    def __init__(
        self,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        sino_weight: float = 1.0,
        tv_weight: float = 0.01
    ):
        super().__init__()
        self.sino_weight = sino_weight
        self.tv_weight = tv_weight

        self.radon = RadonTransform(img_size, num_angles, num_detectors)
        self.wls_loss = WeightedLeastSquaresLoss()
        self.tv_loss = TotalVariationLoss()

    def forward(
        self,
        x_rec: torch.Tensor,
        sinogram_target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute physics regularization loss."""
        # Forward projection
        sinogram_pred = self.radon.forward_fast(x_rec)

        # Resize if needed
        if sinogram_pred.shape != sinogram_target.shape:
            sinogram_pred = F.interpolate(
                sinogram_pred, size=sinogram_target.shape[2:], mode='bilinear'
            )

        # Sinogram consistency
        sino_loss = self.wls_loss(sinogram_pred, sinogram_target, weights)

        # TV regularization
        tv_loss = self.tv_loss(x_rec)

        return self.sino_weight * sino_loss + self.tv_weight * tv_loss


class FinalDataConsistency(nn.Module):
    """
    Final data consistency projection.

    Solves: x_out = argmin_x 0.5 * ||x - x_ART||^2 + mu * L_phys(x)
    using conjugate gradient iterations.
    """

    def __init__(
        self,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        num_iterations: int = 5,
        mu: float = 0.1
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.mu = mu

        self.data_consistency = CTDataConsistency(
            img_size=img_size,
            num_angles=num_angles,
            num_detectors=num_detectors,
            num_iterations=num_iterations,
            step_size=mu
        )

    def forward(
        self,
        x_art: torch.Tensor,
        sinogram: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply final data consistency."""
        return self.data_consistency(x_art, sinogram, weights, mask)


class CT_ART(nn.Module):
    """
    Adaptive Reconstruction Transformer for CT Reconstruction.

    Final synthesis module that produces high-quality CT reconstructions
    by combining PACE and ADRN features through dynamic convolutions
    and transformer-based fusion with physics constraints.

    Z_ART = DynamicConv(Z_PACE, Z_ADRN) + Transformer(Z_PACE, Z_ADRN)
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 1,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        num_heads: int = 8,
        num_transformer_layers: int = 2,
        lambda_phys: float = 0.1,
        use_final_dc: bool = True,
        num_dc_iterations: int = 5
    ):
        super().__init__()
        self.lambda_phys = lambda_phys
        self.use_final_dc = use_final_dc
        self.img_size = img_size

        # Dynamic convolutions
        self.dynamic_conv1 = SimpleDynamicConv(in_channels, in_channels)
        self.dynamic_conv2 = SimpleDynamicConv(in_channels, in_channels)

        # Transformer fusion
        self.transformer_fusion = TransformerFusion(
            channels=in_channels,
            num_heads=num_heads,
            num_layers=num_transformer_layers
        )

        # Feature combination
        self.combine = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Decoder to image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, out_channels, 3, padding=1)
        )

        # Physics regularization
        self.physics_reg = PhysicsRegularizationART(
            img_size=img_size,
            num_angles=num_angles,
            num_detectors=num_detectors
        )

        # Final data consistency
        if use_final_dc:
            self.final_dc = FinalDataConsistency(
                img_size=img_size,
                num_angles=num_angles,
                num_detectors=num_detectors,
                num_iterations=num_dc_iterations
            )

    def forward(
        self,
        z_pace: torch.Tensor,
        z_adrn: torch.Tensor,
        sinogram: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_reg_loss: bool = False
    ) -> torch.Tensor:
        """
        Synthesize final CT reconstruction.

        Args:
            z_pace: Features from PACE (B, C, H, W)
            z_adrn: Features from ADRN (B, C, H, W)
            sinogram: Measured sinogram for physics constraint
            weights: Optional WLS weights
            mask: Optional sparse-view mask
            return_reg_loss: Whether to return regularization loss

        Returns:
            Reconstructed CT image (B, 1, img_size, img_size)
        """
        # Dynamic convolutions (ADRN features conditioned on PACE)
        z_dyn1 = self.dynamic_conv1(z_adrn, z_pace)
        z_dyn2 = self.dynamic_conv2(z_pace, z_adrn)

        # Transformer fusion
        z_trans = self.transformer_fusion(z_pace, z_adrn)

        # Combine all features
        z_combined = torch.cat([z_dyn1 + z_dyn2, z_trans], dim=1)
        z_art = self.combine(z_combined)

        # Decode to image
        x_rec = self.decoder(z_art)

        # Resize if needed
        if x_rec.shape[2:] != (self.img_size, self.img_size):
            x_rec = F.interpolate(x_rec, size=(self.img_size, self.img_size), mode='bilinear')

        # Final data consistency
        if self.use_final_dc and sinogram is not None:
            x_rec = self.final_dc(x_rec, sinogram, weights, mask)

        # Non-negativity for CT attenuation
        x_rec = F.relu(x_rec)

        if return_reg_loss and sinogram is not None:
            reg_loss = self.physics_reg(x_rec, sinogram, weights)
            return x_rec, reg_loss

        return x_rec


class MultiScaleART(nn.Module):
    """
    Multi-scale variant of ART for improved detail preservation.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 1,
        img_size: int = 256,
        num_angles: int = 180,
        scales: Tuple[int, ...] = (1, 2, 4)
    ):
        super().__init__()
        self.scales = scales

        # Scale-specific decoders
        self.decoders = nn.ModuleList()
        for scale in scales:
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
                    nn.BatchNorm2d(in_channels // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels // 2, out_channels, 1)
                )
            )

        # Final fusion
        self.fusion = nn.Conv2d(out_channels * len(scales), out_channels, 1)

    def forward(
        self,
        z_pace: torch.Tensor,
        z_adrn: torch.Tensor,
        target_size: int = 256
    ) -> torch.Tensor:
        """Generate multi-scale reconstruction and fuse."""
        outputs = []

        # Combined features
        z = z_pace + z_adrn

        for scale, decoder in zip(self.scales, self.decoders):
            # Resize features
            if scale > 1:
                z_scaled = F.interpolate(z, scale_factor=1/scale, mode='bilinear')
            else:
                z_scaled = z

            # Decode
            out = decoder(z_scaled)

            # Upsample to target size
            out = F.interpolate(out, size=(target_size, target_size), mode='bilinear')
            outputs.append(out)

        # Fuse scales
        fused = torch.cat(outputs, dim=1)
        return self.fusion(fused)
