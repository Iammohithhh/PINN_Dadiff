"""
Physics-Aware Context Encoder (PACE) for CT Reconstruction.

Enhances latent features from LPCE with multi-scale and long-range contextual
information using ASPP, Non-Local Blocks, and Dual Attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .ct_physics import (
    RadonTransform,
    WeightedLeastSquaresLoss,
    TotalVariationLoss
)


class ASPPConv(nn.Module):
    """Atrous Separable Convolution for ASPP."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3,
                padding=dilation, dilation=dilation, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ASPPPooling(nn.Module):
    """
    Global Average Pooling branch for ASPP.

    FIXED: Use GroupNorm instead of BatchNorm to support batch_size=1
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),  # FIXED: Was BatchNorm2d (fails with batch_size=1)
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        x = self.pool(x)
        x = self.conv(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for CT.

    Captures features at multiple scales using dilated convolutions.
    Z_ASPP = Concat(Conv_r=1, Conv_r=6, Conv_r=12, Conv_r=18, GlobalPool)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates: Tuple[int, ...] = (6, 12, 18)
    ):
        super().__init__()
        modules = []

        # 1x1 convolution
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )

        # Atrous convolutions at different rates
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # Global average pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # Project concatenated features
        # FIXED: Use GroupNorm instead of BatchNorm to support batch_size=1
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),  # FIXED: Was BatchNorm2d
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ASPP to input features.

        Args:
            x: Input features (B, C, H, W)

        Returns:
            Multi-scale features (B, out_channels, H, W)
        """
        results = []
        for conv in self.convs:
            results.append(conv(x))

        concat = torch.cat(results, dim=1)
        return self.project(concat)


class NonLocalBlock(nn.Module):
    """
    Non-Local Block for capturing long-range dependencies.

    Z_NonLocal(i) = (1/C(Z)) * sum_j softmax(Z(i) . Z(j)) * Z(j)

    Enables the network to integrate information from distant parts of the image.
    """

    def __init__(
        self,
        in_channels: int,
        inter_channels: Optional[int] = None,
        sub_sample: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2

        # Theta, Phi, G transformations
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)

        # Output transformation
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )

        # Optional subsampling for efficiency
        self.sub_sample = sub_sample
        if sub_sample:
            self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply non-local attention.

        Args:
            x: Input features (B, C, H, W)

        Returns:
            Features with non-local attention (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Compute theta, phi, g
        theta = self.theta(x).view(B, self.inter_channels, -1)  # (B, C', H*W)
        theta = theta.permute(0, 2, 1)  # (B, H*W, C')

        phi = self.phi(x)
        g = self.g(x)

        if self.sub_sample:
            phi = self.pool(phi)
            g = self.pool(g)

        phi = phi.view(B, self.inter_channels, -1)  # (B, C', H'*W')
        g = g.view(B, self.inter_channels, -1).permute(0, 2, 1)  # (B, H'*W', C')

        # Attention: softmax(theta @ phi)
        attn = torch.bmm(theta, phi)  # (B, H*W, H'*W')
        attn = F.softmax(attn, dim=-1)

        # Apply attention to g
        y = torch.bmm(attn, g)  # (B, H*W, C')
        y = y.permute(0, 2, 1).view(B, self.inter_channels, H, W)

        # Output projection with residual
        y = self.W(y)
        return x + y


class ChannelAttention(nn.Module):
    """
    Channel-wise Attention for feature recalibration.

    Z_CA = sigmoid(W_c @ GlobalAvgPool(Z) + b_c) * Z
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention."""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention(nn.Module):
    """
    Spatial Attention for focusing on important regions.

    Z_SA = sigmoid(Conv([AvgPool(Z), MaxPool(Z)])) * Z
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(concat))
        return x * attn


class DualAttention(nn.Module):
    """
    Dual Attention Mechanism combining Channel and Spatial attention.

    Enhances features by focusing on both important channels and spatial regions.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dual attention."""
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class PhysicsRegularizationPACE(nn.Module):
    """
    Physics-based regularization for PACE.

    Ensures contextual features adhere to CT physics constraints.
    R_phys(A_l) = ||Ax - y||_W^2 + TV(x)

    FIXED: Use configurable in_channels and proper sinogram geometry.
    """

    def __init__(
        self,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        in_channels: int = 256,
        sino_weight: float = 1.0,
        tv_weight: float = 0.01
    ):
        super().__init__()
        self.sino_weight = sino_weight
        self.tv_weight = tv_weight
        self.img_size = img_size

        self.radon = RadonTransform(img_size, num_angles, num_detectors)
        self.wls_loss = WeightedLeastSquaresLoss()
        self.tv_loss = TotalVariationLoss()

        # Learnable projection to single channel
        self.proj = nn.Conv2d(in_channels, 1, 1, bias=False)

    def forward(
        self,
        features: torch.Tensor,
        sinogram_target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute physics regularization loss."""
        # Project to single channel for sinogram consistency
        features_1ch = self.proj(features)

        # Forward projection
        sinogram_pred = self.radon.forward(features_1ch)

        # Resize sinogram if needed (geometry-aware, same num_angles/num_detectors)
        if sinogram_pred.shape != sinogram_target.shape:
            sinogram_pred = F.interpolate(
                sinogram_pred, size=sinogram_target.shape[2:],
                mode='bilinear', align_corners=True
            )

        # Sinogram consistency loss
        sino_loss = self.wls_loss(sinogram_pred, sinogram_target, weights)

        # TV regularization
        tv_loss = self.tv_loss(features_1ch)

        return self.sino_weight * sino_loss + self.tv_weight * tv_loss


class CT_PACE(nn.Module):
    """
    Physics-Aware Context Encoder for CT Reconstruction.

    Enhances LPCE latent features with multi-scale and long-range context
    while incorporating CT physics constraints.

    Z_PACE = ASPP(Z_latent) + NonLocal(Z_latent) + DualAttn(Z_latent)
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 256,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        atrous_rates: Tuple[int, ...] = (6, 12, 18),
        lambda_phys: float = 0.1
    ):
        super().__init__()
        self.lambda_phys = lambda_phys
        self.img_size = img_size  # Store for upsampling in forward

        # ASPP for multi-scale features
        self.aspp = ASPP(in_channels, out_channels, atrous_rates)

        # Non-Local block for long-range dependencies
        self.non_local = NonLocalBlock(out_channels)

        # Dual attention
        self.dual_attn = DualAttention(out_channels)

        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Physics regularization - pass out_channels for channel projection
        self.physics_reg = PhysicsRegularizationPACE(
            img_size=img_size,
            num_angles=num_angles,
            num_detectors=num_detectors,
            in_channels=out_channels
        )

        # Residual connection projection
        self.residual_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(
        self,
        z_latent: torch.Tensor,
        sinogram: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        return_reg_loss: bool = False
    ) -> torch.Tensor:
        """
        Encode latent features with contextual information.

        Args:
            z_latent: Latent features from LPCE (B, C, H, W)
            sinogram: Target sinogram for physics regularization
            weights: Optional WLS weights
            return_reg_loss: Whether to return regularization loss

        Returns:
            Z_PACE: Context-enhanced features (B, out_channels, H, W)
        """
        # Multi-scale features via ASPP
        z_aspp = self.aspp(z_latent)

        # Long-range dependencies via Non-Local
        z_nonlocal = self.non_local(z_aspp)

        # Channel and spatial attention
        z_attn = self.dual_attn(z_nonlocal)

        # Refine with residual connection
        residual = self.residual_proj(z_latent)
        z_pace = self.refine(z_attn) + residual

        if return_reg_loss and sinogram is not None:
            # FIXED: Upsample to IMAGE dimensions, not sinogram dimensions
            # Physics regularization expects image-space features to project to sinogram
            z_upsampled = F.interpolate(z_pace, size=(self.img_size, self.img_size),
                                        mode='bilinear', align_corners=True)
            reg_loss = self.physics_reg(z_upsampled, sinogram, weights)
            return z_pace, reg_loss

        return z_pace


class ContextualFeatureExtractor(nn.Module):
    """
    Combined Contextual Feature Extractor (CFE).

    Wrapper that combines ASPP, Non-Local, and Dual Attention
    with configurable components.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_aspp: bool = True,
        use_nonlocal: bool = True,
        use_dual_attn: bool = True,
        atrous_rates: Tuple[int, ...] = (6, 12, 18)
    ):
        super().__init__()
        self.use_aspp = use_aspp
        self.use_nonlocal = use_nonlocal
        self.use_dual_attn = use_dual_attn

        current_channels = in_channels

        if use_aspp:
            self.aspp = ASPP(current_channels, out_channels, atrous_rates)
            current_channels = out_channels
        else:
            self.aspp = nn.Conv2d(current_channels, out_channels, 1)
            current_channels = out_channels

        if use_nonlocal:
            self.nonlocal_block = NonLocalBlock(current_channels)

        if use_dual_attn:
            self.dual_attn = DualAttention(current_channels)

        # Final projection
        self.final = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract contextual features."""
        out = self.aspp(x)

        if self.use_nonlocal:
            out = self.nonlocal_block(out)

        if self.use_dual_attn:
            out = self.dual_attn(out)

        return self.final(out)
