"""
CT Physics Module for PINN-DADif CT Reconstruction
Implements differentiable Radon Transform, Filtered Back Projection (FBP),
and CT-specific physics operations.

This module replaces the MRI k-space/Fourier operations with CT sinogram/Radon operations.

FIXED ISSUES:
- Correct coordinate normalization using floating-point division
- Proper line-integral sampling using grid_sample
- Consistent forward projection implementation
- Correct FBP backprojection geometry
- Fixed noise model clamping (use epsilon, not 1.0)
- Centralized num_detectors computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


def compute_num_detectors(img_size: int) -> int:
    """
    Centralized computation of number of detectors.
    Use this everywhere to ensure consistency.

    Args:
        img_size: Image size

    Returns:
        Number of detectors (diagonal of image)
    """
    return int(np.ceil(np.sqrt(2) * img_size))


class RadonTransform(nn.Module):
    """
    Differentiable Radon Transform (Forward Projection).

    Computes line integrals of the image along specified angles,
    producing a sinogram.

    p(theta, s) = integral of mu(x, y) along line at angle theta, offset s

    In matrix form: y = A @ x
    where A is the system/projection matrix

    FIXED: Proper coordinate normalization and consistent implementation.
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
        # Use centralized computation
        self.num_detectors = num_detectors or compute_num_detectors(img_size)
        self.device = device

        # Generate angles: [0, pi) - exclude pi to avoid duplication
        if angles is None:
            angles = torch.linspace(0, np.pi * (1 - 1/num_angles), num_angles, device=device)
        self.register_buffer('angles', angles)

        # Precompute projection geometry
        self._setup_geometry()

    def _setup_geometry(self):
        """Setup projection geometry for Radon transform."""
        # Detector positions (centered) - use floating point division
        # Range: [-detector_span/2, detector_span/2]
        detector_span = self.num_detectors - 1
        det_pos = torch.linspace(
            -detector_span / 2.0,
            detector_span / 2.0,
            self.num_detectors,
            device=self.device
        )
        self.register_buffer('det_pos', det_pos)

        # Image coordinates - use floating point division
        img_span = self.img_size - 1
        coords = torch.linspace(
            -img_span / 2.0,
            img_span / 2.0,
            self.img_size,
            device=self.device
        )
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('x_grid', x_grid)
        self.register_buffer('y_grid', y_grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward projection (Radon transform) using rotation + summation.

        This is the canonical implementation - use this one.

        Args:
            x: Input image tensor of shape (B, 1, H, W) or (B, H, W)

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

            # Create rotation grid for grid_sample
            # grid_sample expects coordinates in [-1, 1]
            y_coords = torch.linspace(-1, 1, H, device=x.device)
            x_coords = torch.linspace(-1, 1, W, device=x.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

            # Rotated coordinates (inverse rotation for grid_sample)
            xx_rot = xx * cos_t - yy * sin_t
            yy_rot = xx * sin_t + yy * cos_t

            grid = torch.stack([xx_rot, yy_rot], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

            # Sample rotated image
            rotated = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

            # Sum along rows (vertical direction after rotation) to get projection
            projection = rotated.sum(dim=2, keepdim=True)  # (B, C, 1, W)
            sinograms.append(projection)

        sinogram = torch.cat(sinograms, dim=2)  # (B, C, num_angles, W)

        # Resize to num_detectors if needed
        if sinogram.shape[-1] != self.num_detectors:
            sinogram = F.interpolate(sinogram, size=(self.num_angles, self.num_detectors),
                                     mode='bilinear', align_corners=True)

        return sinogram

    def forward_fast(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward() - use single canonical implementation."""
        return self.forward(x)


class FilteredBackProjection(nn.Module):
    """
    Filtered Back Projection (FBP) for CT reconstruction.

    Implements the adjoint of the Radon transform with ramp filtering.

    FIXED: Correct backprojection geometry with proper sampling.
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
        self.num_detectors = num_detectors or compute_num_detectors(img_size)
        self.filter_type = filter_type
        self.device = device

        # Generate angles: [0, pi) - same convention as RadonTransform
        angles = torch.linspace(0, np.pi * (1 - 1/num_angles), num_angles, device=device)
        self.register_buffer('angles', angles)

        # Create filter
        self._create_filter()

    def _create_filter(self):
        """Create frequency domain filter for FBP."""
        n = self.num_detectors
        # Use proper FFT frequency computation
        freq = torch.fft.fftfreq(n, d=1.0, device=self.device)

        if self.filter_type == 'ramp':
            # Ram-Lak (ramp) filter: |omega|
            filt = torch.abs(freq) * 2
        elif self.filter_type == 'shepp-logan':
            # Shepp-Logan filter: |omega| * sinc(omega/(2*f_max))
            # where f_max = 0.5 (Nyquist frequency)
            filt = torch.abs(freq) * torch.sinc(freq)  # sinc includes pi
        elif self.filter_type == 'cosine':
            # Cosine filter: |omega| * cos(pi*omega/(2*f_max))
            filt = torch.abs(freq) * torch.cos(np.pi * freq)
        elif self.filter_type == 'hamming':
            # Hamming filter: |omega| * (0.54 + 0.46*cos(pi*omega/f_max))
            # Note: inside passband, use + not -
            filt = torch.abs(freq) * (0.54 + 0.46 * torch.cos(2 * np.pi * freq))
        elif self.filter_type == 'hann':
            # Hann filter: |omega| * 0.5*(1 + cos(pi*omega/f_max))
            filt = torch.abs(freq) * 0.5 * (1 + torch.cos(2 * np.pi * freq))
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

        FIXED: Correct geometry - for each pixel, compute where it projects
        onto each detector array, then sample the sinogram at that position.

        Args:
            sinogram: Filtered sinogram of shape (B, 1, num_angles, num_detectors)

        Returns:
            Reconstructed image of shape (B, 1, H, W)
        """
        if sinogram.dim() == 3:
            sinogram = sinogram.unsqueeze(1)

        B, C, A, D = sinogram.shape

        # Create coordinate grid for output image in [-1, 1]
        coords = torch.linspace(-1, 1, self.img_size, device=sinogram.device)
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')

        reconstruction = torch.zeros(B, C, self.img_size, self.img_size, device=sinogram.device)

        # Scale factor: map pixel coordinates to detector indices
        # Detector positions span [-D/2, D/2], image coords span [-1, 1]
        # Scale = (img_size-1)/2 * sqrt(2) to account for diagonal
        scale = (self.img_size - 1) / 2.0 * np.sqrt(2) / ((D - 1) / 2.0)

        for i, theta in enumerate(self.angles):
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)

            # For each pixel (x, y), compute projection position t = x*cos(theta) + y*sin(theta)
            # This gives where the pixel projects onto the detector
            t = x_grid * cos_t + y_grid * sin_t

            # Normalize t to [-1, 1] range for grid_sample
            # t is in image coordinate units, scale to detector units
            t_norm = t / scale if scale > 0 else t

            # For grid_sample on a 1D slice, we need (B, H, W, 2) grid
            # where we sample along the detector dimension
            # Create grid: x-coordinate is t_norm (detector position), y is 0 (single row)
            zeros = torch.zeros_like(t_norm)
            grid = torch.stack([t_norm, zeros], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

            # Get projection values for this angle (B, C, 1, D)
            proj = sinogram[:, :, i:i+1, :]

            # Sample from projection using grid_sample
            # grid_sample expects input (B, C, H_in, W_in) and grid (B, H_out, W_out, 2)
            sampled = F.grid_sample(proj, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

            # sampled is (B, C, img_size, img_size)
            reconstruction += sampled

        # Normalize by angular step
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

    FIXED: Proper epsilon for log-safety, no hard floor to 1.0
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
        self.eps = 1e-6  # Epsilon for log-safety

        self.radon = RadonTransform(img_size, num_angles, num_detectors, device=device)
        self.fbp = FilteredBackProjection(img_size, num_angles, num_detectors, device=device)

    def forward_project(self, x: torch.Tensor) -> torch.Tensor:
        """Compute line integrals (noiseless sinogram)."""
        return self.radon.forward(x)

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

        # Add Poisson noise - clamp to small positive for numerical stability
        noisy_counts = torch.poisson(expected_counts.clamp(min=self.eps))

        return noisy_counts

    def counts_to_sinogram(self, counts: torch.Tensor, I0: Optional[float] = None) -> torch.Tensor:
        """
        Convert photon counts back to line integrals (log transform).

        FIXED: Use epsilon, not floor to 1.0

        p = -log(Y / I0)
        """
        I0 = I0 or self.I0

        # Avoid log(0) with small epsilon, not hard floor to 1
        counts_safe = counts.clamp(min=self.eps)
        sinogram = -torch.log(counts_safe / I0)

        return sinogram

    def get_weights(self, counts: torch.Tensor) -> torch.Tensor:
        """
        Get weights for weighted least squares.

        FIXED: Use epsilon, not floor to 1.0

        For log-domain, Var(p) ≈ 1/Y, so weight = Y
        """
        return counts.clamp(min=self.eps)

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
            sinogram_pred = self.radon.forward(x)

            # Handle size mismatch
            if sinogram_pred.shape != sinogram_measured.shape:
                sinogram_pred = F.interpolate(sinogram_pred, size=sinogram_measured.shape[2:],
                                              mode='bilinear', align_corners=True)

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

    def __init__(self, I0: float = 1e4, scatter: float = 0.0, eps: float = 1e-6):
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

    Note: Uses sum reduction normalized by total weight for consistency.
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
            # Normalized weighted sum
            weighted_sq = weights * residual ** 2
            loss = 0.5 * weighted_sq.sum() / (weights.sum() + 1e-8)
        else:
            loss = 0.5 * (residual ** 2).mean()

        return loss


class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss for CT regularization.

    TV(x) = sum_ij sqrt((x_i+1,j - x_ij)^2 + (x_i,j+1 - x_ij)^2 + eps)

    FIXED: Correct padding order for (B, C, H, W) tensors.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute isotropic TV loss.

        Expects x of shape (B, C, H, W).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Horizontal differences (along W dimension)
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        # Vertical differences (along H dimension)
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]

        # Pad to same size
        # F.pad order is (left, right, top, bottom) for 2D
        dx = F.pad(dx, (0, 1, 0, 0))  # pad right
        dy = F.pad(dy, (0, 0, 0, 1))  # pad bottom

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


# ============== Validation Functions ==============

def test_adjoint(radon: RadonTransform, fbp: FilteredBackProjection,
                 device: str = 'cuda', tol: float = 0.1) -> Tuple[bool, float]:
    """
    Test adjoint relationship: <Ax, y> ≈ <x, A^T y>

    Args:
        radon: Radon transform
        fbp: Filtered back projection
        device: Device
        tol: Relative tolerance

    Returns:
        (passed, relative_error)
    """
    with torch.no_grad():
        # Random image and sinogram
        x = torch.rand(1, 1, radon.img_size, radon.img_size, device=device)
        y = torch.rand(1, 1, radon.num_angles, radon.num_detectors, device=device)

        # Forward: Ax
        Ax = radon.forward(x)

        # Adjoint (backproject without filter): A^T y
        ATy = fbp.backproject(y)

        # Inner products
        inner1 = (Ax * y).sum()
        inner2 = (x * ATy).sum()

        rel_error = abs(inner1 - inner2) / (abs(inner1) + abs(inner2) + 1e-10)
        passed = rel_error < tol

        return passed, rel_error.item()


def test_fbp_roundtrip(img_size: int = 256, num_angles: int = 180,
                       device: str = 'cuda') -> Tuple[bool, float]:
    """
    Test FBP round-trip: FBP(Radon(x)) ≈ x

    Args:
        img_size: Image size
        num_angles: Number of angles
        device: Device

    Returns:
        (passed, psnr)
    """
    from . import create_shepp_logan_phantom  # Import from data_loader

    with torch.no_grad():
        # Create phantom
        phantom = create_shepp_logan_phantom(img_size)
        x = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0).float().to(device)

        # Forward and back
        radon = RadonTransform(img_size, num_angles, device=device).to(device)
        fbp = FilteredBackProjection(img_size, num_angles, device=device).to(device)

        sino = radon.forward(x)
        recon = fbp.forward(sino)

        # Compute PSNR
        mse = ((recon - x) ** 2).mean()
        if mse < 1e-10:
            psnr = 100.0
        else:
            psnr = 10 * np.log10(1.0 / mse.item())

        # Good FBP should achieve > 20 dB PSNR
        passed = psnr > 20.0

        return passed, psnr
