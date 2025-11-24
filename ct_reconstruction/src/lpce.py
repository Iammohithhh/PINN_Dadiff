"""
Latent Physics-Constrained Encoder (LPCE) for CT Reconstruction.

Adapted from MRI PINN-DADif to use Radon/FBP physics instead of Fourier/k-space.
Extracts latent features from undersampled sinogram data while respecting CT physics.
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


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for feature extraction."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class PhysicsInformedConv(nn.Module):
    """
    Physics-informed convolutional layer for CT.

    Incorporates CT physics constraints (sinogram consistency, smoothness)
    into the convolutional operations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: int = 256,
        num_angles: int = 180,
        physics_weight: float = 0.1
    ):
        super().__init__()
        self.physics_weight = physics_weight

        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Physics constraint layers
        self.physics_proj = nn.Conv2d(out_channels, out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        physics_term: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional physics constraint.

        Args:
            x: Input features
            physics_term: Physics-based regularization term

        Returns:
            Output features with physics constraints
        """
        out = self.relu(self.bn(self.conv(x)))

        if physics_term is not None:
            # Add physics-informed regularization
            physics_correction = self.physics_proj(physics_term)
            out = out + self.physics_weight * physics_correction

        return out


class DataDrivenBranch(nn.Module):
    """
    Data-driven feature extraction branch.

    Processes FBP reconstruction using CNN to extract data-driven features.
    f_data(X_FBP)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        latent_dim: int = 128
    ):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, latent_dim)

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Residual refinement
        self.res_blocks = nn.Sequential(
            ResidualBlock(latent_dim),
            ResidualBlock(latent_dim),
            ResidualBlock(latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract data-driven features from FBP reconstruction.

        Args:
            x: FBP reconstructed image (B, 1, H, W)

        Returns:
            Latent features (B, latent_dim, H/8, W/8)
        """
        # Encoder path
        e1 = self.enc1(x)          # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))   # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))   # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))   # (B, latent_dim, H/8, W/8)

        # Residual refinement
        out = self.res_blocks(e4)

        return out


class PhysicsBranch(nn.Module):
    """
    Physics-informed feature extraction branch for CT.

    Processes sinogram residuals and applies CT physics constraints.
    f_phys(sinogram_residual)

    Replaces MRI's Fourier/k-space operations with Radon/sinogram operations.
    """

    def __init__(
        self,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        base_channels: int = 64,
        latent_dim: int = 128
    ):
        super().__init__()
        self.img_size = img_size
        self.num_angles = num_angles
        self.num_detectors = num_detectors or int(torch.ceil(torch.tensor(2 ** 0.5 * img_size)).item())

        # Radon transform (forward projection)
        self.radon = RadonTransform(img_size, num_angles, self.num_detectors)

        # FBP (backprojection)
        self.fbp = FilteredBackProjection(img_size, num_angles, self.num_detectors)

        # Sinogram domain CNN (processes residual sinogram)
        self.sino_encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Project sinogram features back to image domain
        self.sino_to_image = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Image domain refinement
        self.image_encoder = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            ConvBlock(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2),
            ConvBlock(base_channels * 4, latent_dim),
            nn.MaxPool2d(2),
        )

        # Physics-informed residual blocks
        self.physics_blocks = nn.Sequential(
            ResidualBlock(latent_dim),
            ResidualBlock(latent_dim),
        )

    def forward(
        self,
        x_fbp: torch.Tensor,
        sinogram_measured: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract physics-informed features.

        Args:
            x_fbp: FBP reconstruction (B, 1, H, W)
            sinogram_measured: Measured sinogram (B, 1, num_angles, num_detectors)

        Returns:
            Physics-informed latent features (B, latent_dim, H/8, W/8)
        """
        # Compute residual sinogram
        sinogram_fbp = self.radon.forward_fast(x_fbp)
        residual_sino = sinogram_measured - sinogram_fbp

        # Process residual in sinogram domain
        sino_features = self.sino_encoder(residual_sino)

        # Backproject features to image domain
        # Average pooling to reduce channels before backprojection
        sino_pooled = sino_features.mean(dim=1, keepdim=True)
        backprojected = self.fbp.backproject(sino_pooled)

        # Expand channels
        backprojected = backprojected.expand(-1, sino_features.shape[1], -1, -1)

        # Refine in image domain
        image_features = self.sino_to_image(backprojected)
        latent = self.image_encoder(image_features)

        # Physics-informed refinement
        out = self.physics_blocks(latent)

        return out


class PhysicsRegularization(nn.Module):
    """
    Physics-based regularization for CT LPCE.

    Enforces sinogram consistency and smoothness constraints.
    R_phys(A_l) = ||A(A_l) - y||_W^2 + eta * TV(A_l)
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
        features: torch.Tensor,
        sinogram_target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute physics regularization loss.

        Args:
            features: Intermediate feature activation (treated as image)
            sinogram_target: Target sinogram
            weights: Optional WLS weights

        Returns:
            Regularization loss
        """
        # Project features to sinogram domain
        # Reduce channels first
        if features.shape[1] > 1:
            features_1ch = features.mean(dim=1, keepdim=True)
        else:
            features_1ch = features

        sinogram_features = self.radon.forward_fast(features_1ch)

        # Sinogram consistency
        sino_loss = self.wls_loss(sinogram_features, sinogram_target, weights)

        # TV regularization
        tv_loss = self.tv_loss(features_1ch)

        return self.sino_weight * sino_loss + self.tv_weight * tv_loss


class CT_LPCE(nn.Module):
    """
    Latent Physics-Constrained Encoder for CT Reconstruction.

    Multi-branch encoder that extracts both data-driven and physics-informed
    features from undersampled CT data.

    Z_latent = f_data(X_FBP) + lambda_phys * f_phys(X_under, sinogram)

    Replaces MRI LPCE by using Radon/FBP instead of FFT/IFFT.
    """

    def __init__(
        self,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        in_channels: int = 1,
        base_channels: int = 64,
        latent_dim: int = 128,
        lambda_phys: float = 0.3
    ):
        super().__init__()
        self.img_size = img_size
        self.lambda_phys = lambda_phys
        self.latent_dim = latent_dim

        # FBP for initial reconstruction
        self.fbp = FilteredBackProjection(img_size, num_angles, num_detectors)

        # Data-driven branch
        self.data_branch = DataDrivenBranch(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_dim=latent_dim
        )

        # Physics-informed branch
        self.physics_branch = PhysicsBranch(
            img_size=img_size,
            num_angles=num_angles,
            num_detectors=num_detectors,
            base_channels=base_channels,
            latent_dim=latent_dim
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(latent_dim * 2, latent_dim, 1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            ResidualBlock(latent_dim)
        )

        # Physics regularization
        self.physics_reg = PhysicsRegularization(
            img_size=img_size,
            num_angles=num_angles,
            num_detectors=num_detectors
        )

    def forward(
        self,
        sinogram: torch.Tensor,
        x_fbp: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        return_reg_loss: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sinogram to latent features.

        Args:
            sinogram: Measured sinogram (B, 1, num_angles, num_detectors)
            x_fbp: Optional pre-computed FBP reconstruction
            weights: Optional WLS weights
            return_reg_loss: Whether to return regularization loss

        Returns:
            z_latent: Latent features (B, latent_dim, H/8, W/8)
            x_fbp: FBP reconstruction (B, 1, H, W)
            reg_loss: (optional) Physics regularization loss
        """
        # Initial FBP reconstruction
        if x_fbp is None:
            x_fbp = self.fbp(sinogram)

        # Data-driven features
        z_data = self.data_branch(x_fbp)

        # Physics-informed features
        z_phys = self.physics_branch(x_fbp, sinogram)

        # Combine features
        z_combined = torch.cat([z_data, self.lambda_phys * z_phys], dim=1)
        z_latent = self.fusion(z_combined)

        if return_reg_loss:
            # Compute physics regularization
            # Use upsampled latent features
            z_upsampled = F.interpolate(z_latent, size=(self.img_size, self.img_size), mode='bilinear')
            reg_loss = self.physics_reg(z_upsampled, sinogram, weights)
            return z_latent, x_fbp, reg_loss

        return z_latent, x_fbp


class ScanTypeConditioning(nn.Module):
    """
    Conditioning module for different CT scan types.

    Similar to sequence-specific processing in MRI LPCE,
    this handles different CT protocols (low-dose, sparse-view, limited-angle).
    """

    def __init__(self, latent_dim: int = 128, num_scan_types: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(num_scan_types, latent_dim)
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(
        self,
        z_latent: torch.Tensor,
        scan_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply scan-type conditioning.

        Args:
            z_latent: Latent features (B, C, H, W)
            scan_type: Scan type indices (B,)

        Returns:
            Conditioned features
        """
        # Get embedding
        emb = self.embedding(scan_type)  # (B, latent_dim)
        emb = self.proj(emb)  # (B, latent_dim)

        # Add to latent features (broadcast over spatial dimensions)
        emb = emb.unsqueeze(-1).unsqueeze(-1)  # (B, latent_dim, 1, 1)
        return z_latent + emb
