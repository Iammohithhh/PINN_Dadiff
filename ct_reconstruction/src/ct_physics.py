"""
CT Physics Module for PINN-DADif CT Reconstruction
Implements differentiable Radon Transform, Filtered Back Projection (FBP),
and CT-specific physics operations.

This module replaces the MRI k-space/Fourier operations with CT sinogram/Radon operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


class RadonTransform(nn.Module):
    """
    Differentiable Radon Transform (Forward Projection).

    Computes line integrals of the image along specified angles,
    producing a sinogram.

    p(theta, s) = integral of mu(x, y) along line at angle theta, offset s

    In matrix form: y = A @ x
    where A is the system/projection matrix
    """

    def __init__(
        self,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        angles: Optional[torch.Tensor] = None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.img_size = img_size
        self.num_angles = num_angles
        self.num_detectors = num_detectors or int(np.ceil(np.sqrt(2) * img_size))
        self.device = device

        # Generate angles if not provided
        if angles is None:
            angles = torch.linspace(0, np.pi, num_angles, device=device)
        self.register_buffer('angles', angles)

        # Precompute projection geometry
        self._setup_geometry()

    def _setup_geometry(self):
        """Setup projection geometry for Radon transform."""
        # Detector positions (centered)
        det_pos = torch.linspace(
            -self.num_detectors // 2,
            self.num_detectors // 2,
            self.num_detectors,
            device=self.device
        )
        self.register_buffer('det_pos', det_pos)

        # Image coordinates
        coords = torch.linspace(
            -self.img_size // 2,
            self.img_size // 2,
            self.img_size,
            device=self.device
        )
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('x_grid', x_grid)
        self.register_buffer('y_grid', y_grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward projection (Radon transform).

        Args:
            x: Input image tensor of shape (B, 1, H, W) or (B, H, W)

        Returns:
            Sinogram of shape (B, 1, num_angles, num_detectors)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        B, C, H, W = x.shape
        sinogram = torch.zeros(B, C, self.num_angles, self.num_detectors, device=x.device)

        for i, theta in enumerate(self.angles):
            # Rotate coordinates
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)

            # Project coordinates onto detector
            proj_coords = self.x_grid * cos_t + self.y_grid * sin_t

            # Normalize to [-1, 1] for grid_sample
            proj_coords_norm = proj_coords / (self.num_detectors // 2)

            # Create sampling grid
            # For each detector position, sample along the perpendicular line
            for j, det in enumerate(self.det_pos):
                # Find pixels that project to this detector
                mask = (proj_coords >= det - 0.5) & (proj_coords < det + 0.5)
                sinogram[:, :, i, j] = (x * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))

        return sinogram

    def forward_fast(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast forward projection using interpolation.

        Args:
            x: Input image tensor of shape (B, 1, H, W)

        Returns:
            Sinogram of shape (B, 1, num_angles, num_detectors)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        B, C, H, W = x.shape
        sinograms = []

        for theta in self.angles:
            # Rotate image by -theta and sum along columns
            cos_t, sin_t = torch.cos(-theta), torch.sin(-theta)

            # Create rotation grid
            y_coords = torch.linspace(-1, 1, H, device=x.device)
            x_coords = torch.linspace(-1, 1, W, device=x.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

            # Rotated coordinates
            xx_rot = xx * cos_t - yy * sin_t
            yy_rot = xx * sin_t + yy * cos_t

            grid = torch.stack([xx_rot, yy_rot], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

            # Sample rotated image
            rotated = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

            # Sum along rows (vertical direction after rotation)
            projection = rotated.sum(dim=2, keepdim=True)
            sinograms.append(projection)

        sinogram = torch.cat(sinograms, dim=2)

        # Resize to num_detectors if needed
        if sinogram.shape[-1] != self.num_detectors:
            sinogram = F.interpolate(sinogram, size=(self.num_angles, self.num_detectors), mode='bilinear')

        return sinogram


class FilteredBackProjection(nn.Module):
    """
    Filtered Back Projection (FBP) for CT reconstruction.

    Implements the adjoint of the Radon transform with ramp filtering.
    """

    def __init__(
        self,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        filter_type: str = 'ramp',
        device: str = 'cuda'
    ):
        super().__init__()
        self.img_size = img_size
        self.num_angles = num_angles
        self.num_detectors = num_detectors or int(np.ceil(np.sqrt(2) * img_size))
        self.filter_type = filter_type
        self.device = device

        # Generate angles
        angles = torch.linspace(0, np.pi, num_angles, device=device)
        self.register_buffer('angles', angles)

        # Create filter
        self._create_filter()

    def _create_filter(self):
        """Create frequency domain filter for FBP."""
        n = self.num_detectors
        freq = torch.fft.fftfreq(n, device=self.device)

        if self.filter_type == 'ramp':
            # Ram-Lak (ramp) filter
            filt = torch.abs(freq) * 2
        elif self.filter_type == 'shepp-logan':
            # Shepp-Logan filter
            filt = torch.abs(freq) * torch.sinc(freq)
        elif self.filter_type == 'cosine':
            # Cosine filter
            filt = torch.abs(freq) * torch.cos(np.pi * freq / 2)
        elif self.filter_type == 'hamming':
            # Hamming filter
            filt = torch.abs(freq) * (0.54 + 0.46 * torch.cos(np.pi * freq))
        elif self.filter_type == 'hann':
            # Hann filter
            filt = torch.abs(freq) * (0.5 + 0.5 * torch.cos(np.pi * freq))
        else:
            filt = torch.abs(freq) * 2  # Default to ramp

        self.register_buffer('filter', filt)

    def filter_sinogram(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Apply filter to sinogram in frequency domain."""
        # FFT along detector dimension
        sino_fft = torch.fft.fft(sinogram, dim=-1)

        # Apply filter
        filtered_fft = sino_fft * self.filter.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Inverse FFT
        filtered_sino = torch.fft.ifft(filtered_fft, dim=-1).real

        return filtered_sino

    def backproject(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        Backproject filtered sinogram.

        Args:
            sinogram: Filtered sinogram of shape (B, 1, num_angles, num_detectors)

        Returns:
            Reconstructed image of shape (B, 1, H, W)
        """
        if sinogram.dim() == 3:
            sinogram = sinogram.unsqueeze(1)

        B, C, A, D = sinogram.shape

        # Create coordinate grid
        coords = torch.linspace(-1, 1, self.img_size, device=sinogram.device)
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')

        reconstruction = torch.zeros(B, C, self.img_size, self.img_size, device=sinogram.device)

        for i, theta in enumerate(self.angles):
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)

            # Calculate projection position for each pixel
            t = x_grid * cos_t + y_grid * sin_t

            # Normalize to [0, 1] for grid_sample
            t_norm = t  # Already in [-1, 1]

            # Create sampling grid (sample along detector dimension)
            zeros = torch.zeros_like(t_norm)
            grid = torch.stack([t_norm, zeros], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

            # Get projection values for this angle
            proj = sinogram[:, :, i:i+1, :]  # (B, C, 1, D)

            # Sample from projection
            sampled = F.grid_sample(proj, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

            reconstruction += sampled.squeeze(2)

        # Normalize by number of angles
        reconstruction = reconstruction * np.pi / self.num_angles

        return reconstruction

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        Full FBP reconstruction.

        Args:
            sinogram: Sinogram of shape (B, 1, num_angles, num_detectors)

        Returns:
            Reconstructed image of shape (B, 1, H, W)
        """
        filtered = self.filter_sinogram(sinogram)
        reconstruction = self.backproject(filtered)
        return reconstruction


class CTForwardModel(nn.Module):
    """
    Complete CT Forward Model with noise modeling.

    Implements:
    - Linear forward model: y = A @ x
    - Poisson noise model: Y ~ Poisson(I0 * exp(-Ax) + scatter)
    - Log-domain transformation
    """

    def __init__(
        self,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        I0: float = 1e4,  # Incident photon intensity
        scatter: float = 0.0,  # Scatter/background
        device: str = 'cuda'
    ):
        super().__init__()
        self.I0 = I0
        self.scatter = scatter

        self.radon = RadonTransform(img_size, num_angles, num_detectors, device=device)
        self.fbp = FilteredBackProjection(img_size, num_angles, num_detectors, device=device)

    def forward_project(self, x: torch.Tensor) -> torch.Tensor:
        """Compute line integrals (noiseless sinogram)."""
        return self.radon.forward_fast(x)

    def add_poisson_noise(self, sinogram: torch.Tensor, I0: Optional[float] = None) -> torch.Tensor:
        """
        Add Poisson noise to sinogram.

        Args:
            sinogram: Noiseless sinogram (line integrals)
            I0: Incident intensity (overrides default if provided)

        Returns:
            Noisy photon counts
        """
        I0 = I0 or self.I0

        # Expected photon counts: I0 * exp(-sinogram) + scatter
        expected_counts = I0 * torch.exp(-sinogram) + self.scatter

        # Add Poisson noise
        noisy_counts = torch.poisson(expected_counts.clamp(min=1e-10))

        return noisy_counts

    def counts_to_sinogram(self, counts: torch.Tensor, I0: Optional[float] = None) -> torch.Tensor:
        """
        Convert photon counts back to line integrals (log transform).

        p = -log(Y / I0)
        """
        I0 = I0 or self.I0

        # Avoid log(0)
        counts_safe = counts.clamp(min=1.0)
        sinogram = -torch.log(counts_safe / I0)

        return sinogram

    def get_weights(self, counts: torch.Tensor) -> torch.Tensor:
        """
        Get weights for weighted least squares.

        For log-domain, Var(p) ≈ 1/Y, so weight = Y
        """
        return counts.clamp(min=1.0)

    def forward(
        self,
        x: torch.Tensor,
        add_noise: bool = True,
        return_counts: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Complete forward model.

        Args:
            x: Attenuation image
            add_noise: Whether to add Poisson noise
            return_counts: Whether to return raw counts

        Returns:
            Sinogram (and optionally raw counts)
        """
        # Forward projection
        sinogram = self.forward_project(x)

        if add_noise:
            counts = self.add_poisson_noise(sinogram)
            sinogram_noisy = self.counts_to_sinogram(counts)

            if return_counts:
                return sinogram_noisy, counts
            return sinogram_noisy

        if return_counts:
            counts = self.I0 * torch.exp(-sinogram)
            return sinogram, counts
        return sinogram


class CTDataConsistency(nn.Module):
    """
    CT Data Consistency Layer.

    Enforces consistency between reconstructed image and measured sinogram.
    Replaces MRI k-space data consistency.
    """

    def __init__(
        self,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        num_iterations: int = 5,
        step_size: float = 0.1,
        use_weights: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.use_weights = use_weights

        self.radon = RadonTransform(img_size, num_angles, num_detectors, device=device)
        self.fbp = FilteredBackProjection(img_size, num_angles, num_detectors, device=device)

    def forward(
        self,
        x_pred: torch.Tensor,
        sinogram_measured: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply data consistency.

        Solves: x* = argmin_x ||x - x_pred||^2 + mu * ||W^(1/2)(Ax - p)||^2

        Using gradient descent steps:
        x = x - eta * A^T W (Ax - p)

        Args:
            x_pred: Predicted image from network
            sinogram_measured: Measured sinogram
            weights: Optional weights for WLS
            mask: Optional mask for sparse-view CT

        Returns:
            Data-consistent reconstruction
        """
        x = x_pred.clone()

        for _ in range(self.num_iterations):
            # Forward projection
            sinogram_pred = self.radon.forward_fast(x)

            # Compute residual
            residual = sinogram_pred - sinogram_measured

            # Apply mask for sparse-view
            if mask is not None:
                residual = residual * mask

            # Apply weights for WLS
            if self.use_weights and weights is not None:
                residual = residual * weights

            # Backproject residual (A^T)
            gradient = self.fbp.backproject(residual)

            # Update
            x = x - self.step_size * gradient

            # Enforce non-negativity
            x = F.relu(x)

        return x


class PoissonNLLLoss(nn.Module):
    """
    Poisson Negative Log-Likelihood Loss for CT.

    L = sum_i [ lambda_i - Y_i * log(lambda_i) ]
    where lambda_i = I0 * exp(-[Ax]_i) + scatter
    """

    def __init__(self, I0: float = 1e4, scatter: float = 0.0, eps: float = 1e-10):
        super().__init__()
        self.I0 = I0
        self.scatter = scatter
        self.eps = eps

    def forward(
        self,
        sinogram_pred: torch.Tensor,
        counts_measured: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Poisson NLL.

        Args:
            sinogram_pred: Predicted line integrals (Ax)
            counts_measured: Measured photon counts (Y)

        Returns:
            Poisson NLL loss
        """
        # Expected counts
        lambda_pred = self.I0 * torch.exp(-sinogram_pred) + self.scatter
        lambda_pred = lambda_pred.clamp(min=self.eps)

        # Poisson NLL
        loss = lambda_pred - counts_measured * torch.log(lambda_pred)

        return loss.mean()


class WeightedLeastSquaresLoss(nn.Module):
    """
    Weighted Least Squares Loss for CT (log-domain).

    L = 0.5 * sum_i w_i * (Ax_i - p_i)^2
    where w_i ≈ Y_i (photon counts)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        sinogram_pred: torch.Tensor,
        sinogram_measured: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute WLS loss.

        Args:
            sinogram_pred: Predicted sinogram
            sinogram_measured: Measured sinogram
            weights: Optional weights (default: uniform)

        Returns:
            WLS loss
        """
        residual = sinogram_pred - sinogram_measured

        if weights is not None:
            loss = 0.5 * (weights * residual ** 2).mean()
        else:
            loss = 0.5 * (residual ** 2).mean()

        return loss


class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss for CT regularization.

    TV(x) = sum_ij sqrt((x_i+1,j - x_ij)^2 + (x_i,j+1 - x_ij)^2 + eps)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute isotropic TV loss."""
        # Horizontal differences
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        # Vertical differences
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]

        # Pad to same size
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))

        # Isotropic TV
        tv = torch.sqrt(dx ** 2 + dy ** 2 + self.eps)

        return tv.mean()


class NonNegativityLoss(nn.Module):
    """
    Non-negativity constraint for CT (attenuation must be >= 0).

    L = ||min(x, 0)||^2
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Penalize negative values."""
        return (F.relu(-x) ** 2).mean()


def create_sparse_view_mask(
    num_angles: int,
    num_views: int,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create mask for sparse-view CT.

    Args:
        num_angles: Total number of angles
        num_views: Number of views to keep
        device: Device for tensor

    Returns:
        Binary mask of shape (1, 1, num_angles, 1)
    """
    mask = torch.zeros(num_angles, device=device)
    indices = torch.linspace(0, num_angles - 1, num_views, device=device).long()
    mask[indices] = 1.0
    return mask.view(1, 1, -1, 1)


def create_limited_angle_mask(
    num_angles: int,
    start_angle: float = 0.0,
    end_angle: float = 180.0,
    missing_start: float = 60.0,
    missing_end: float = 120.0,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create mask for limited-angle CT.

    Args:
        num_angles: Number of angles
        start_angle, end_angle: Angular range
        missing_start, missing_end: Missing angular range
        device: Device for tensor

    Returns:
        Binary mask
    """
    angles = torch.linspace(start_angle, end_angle, num_angles, device=device)
    mask = ~((angles >= missing_start) & (angles <= missing_end))
    return mask.float().view(1, 1, -1, 1)
