"""
CT-PINN-DADif: Complete Model for CT Reconstruction.

Physics-Informed Deep Adaptive Diffusion Network adapted for
Computed Tomography reconstruction from undersampled sinogram data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any

from .ct_physics import (
    RadonTransform,
    FilteredBackProjection,
    CTForwardModel,
    PoissonNLLLoss,
    WeightedLeastSquaresLoss,
    TotalVariationLoss,
    NonNegativityLoss
)
from .lpce import CT_LPCE
from .pace import CT_PACE
from .adrn import CT_ADRN
from .art import CT_ART


class CTReconstructionLoss(nn.Module):
    """
    Combined loss function for CT reconstruction.

    L_total = alpha * L_pixel + beta * L_perc + gamma * L_phys

    Where:
    - L_pixel: Pixel-wise L2 loss
    - L_perc: Perceptual loss (VGG features)
    - L_phys: Physics-informed loss (Poisson NLL or WLS + TV + nonnegativity)
    """

    def __init__(
        self,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        alpha: float = 0.5,   # Pixel loss weight
        beta: float = 0.2,    # Perceptual loss weight
        gamma: float = 0.3,   # Physics loss weight
        use_poisson: bool = False,  # Use Poisson NLL vs WLS
        I0: float = 1e4,
        tv_weight: float = 1e-4,
        nonneg_weight: float = 1e-3,
        use_perceptual: bool = False
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_poisson = use_poisson
        self.use_perceptual = use_perceptual

        # Pixel loss
        self.pixel_loss = nn.MSELoss()

        # Physics losses
        self.radon = RadonTransform(img_size, num_angles, num_detectors)

        if use_poisson:
            self.physics_loss = PoissonNLLLoss(I0=I0)
        else:
            self.physics_loss = WeightedLeastSquaresLoss()

        self.tv_loss = TotalVariationLoss()
        self.nonneg_loss = NonNegativityLoss()

        self.tv_weight = tv_weight
        self.nonneg_weight = nonneg_weight

        # Perceptual loss (optional - uses VGG features)
        if use_perceptual:
            try:
                from torchvision.models import vgg16, VGG16_Weights
                vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
                for param in vgg.parameters():
                    param.requires_grad = False
                self.vgg = vgg.eval()
            except ImportError:
                self.use_perceptual = False
                self.vgg = None

    def forward(
        self,
        x_rec: torch.Tensor,
        x_gt: torch.Tensor,
        sinogram_measured: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        counts: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            x_rec: Reconstructed image (B, 1, H, W)
            x_gt: Ground truth image (B, 1, H, W)
            sinogram_measured: Measured sinogram
            weights: Optional WLS weights
            counts: Optional raw photon counts (for Poisson loss)

        Returns:
            Dictionary with total loss and individual components
        """
        losses = {}

        # Pixel loss
        losses['pixel'] = self.pixel_loss(x_rec, x_gt)

        # Perceptual loss
        if self.use_perceptual and self.vgg is not None:
            # Convert to 3-channel for VGG
            x_rec_3ch = x_rec.repeat(1, 3, 1, 1)
            x_gt_3ch = x_gt.repeat(1, 3, 1, 1)

            with torch.no_grad():
                gt_features = self.vgg(x_gt_3ch)
            rec_features = self.vgg(x_rec_3ch)
            losses['perceptual'] = F.mse_loss(rec_features, gt_features)
        else:
            losses['perceptual'] = torch.tensor(0.0, device=x_rec.device)

        # Physics loss
        sinogram_pred = self.radon.forward_fast(x_rec)

        # Resize if needed
        if sinogram_pred.shape != sinogram_measured.shape:
            sinogram_pred = F.interpolate(
                sinogram_pred, size=sinogram_measured.shape[2:], mode='bilinear'
            )

        if self.use_poisson and counts is not None:
            losses['sino_consistency'] = self.physics_loss(sinogram_pred, counts)
        else:
            losses['sino_consistency'] = self.physics_loss(sinogram_pred, sinogram_measured, weights)

        losses['tv'] = self.tv_loss(x_rec) * self.tv_weight
        losses['nonneg'] = self.nonneg_loss(x_rec) * self.nonneg_weight

        losses['physics'] = losses['sino_consistency'] + losses['tv'] + losses['nonneg']

        # Total loss
        losses['total'] = (
            self.alpha * losses['pixel'] +
            self.beta * losses['perceptual'] +
            self.gamma * losses['physics']
        )

        return losses


class CT_PINN_DADif(nn.Module):
    """
    CT-PINN-DADif: Physics-Informed Deep Adaptive Diffusion Network for CT.

    Complete pipeline for CT reconstruction:
    1. LPCE: Encode sinogram to latent features with physics constraints
    2. PACE: Enhance with multi-scale context and attention
    3. ADRN: Refine through adaptive diffusion with sinogram consistency
    4. ART: Final synthesis with dynamic convolutions and transformers

    Algorithm:
    1. x0 = FBP(sinogram)  # Initial reconstruction
    2. z_latent = CT_LPCE(x0, sinogram)  # Encode with physics
    3. z_pace = PACE(z_latent)  # Context enhancement
    4. z_adrn = ADRN(z_pace, sinogram)  # Diffusion refinement
    5. x_rec = ART(z_pace, z_adrn, sinogram)  # Final synthesis
    """

    def __init__(
        self,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        in_channels: int = 1,
        base_channels: int = 64,
        latent_dim: int = 128,
        context_dim: int = 256,
        lambda_phys_lpce: float = 0.3,
        lambda_phys_pace: float = 0.1,
        num_diffusion_steps: int = 12,
        use_final_dc: bool = True
    ):
        super().__init__()
        self.img_size = img_size
        self.num_angles = num_angles
        self.num_detectors = num_detectors or int(torch.ceil(torch.tensor(2 ** 0.5 * img_size)).item())

        # FBP for initial reconstruction
        self.fbp = FilteredBackProjection(img_size, num_angles, self.num_detectors)

        # LPCE: Latent Physics-Constrained Encoder
        self.lpce = CT_LPCE(
            img_size=img_size,
            num_angles=num_angles,
            num_detectors=self.num_detectors,
            in_channels=in_channels,
            base_channels=base_channels,
            latent_dim=latent_dim,
            lambda_phys=lambda_phys_lpce
        )

        # PACE: Physics-Aware Context Encoder
        self.pace = CT_PACE(
            in_channels=latent_dim,
            out_channels=context_dim,
            img_size=img_size,
            num_angles=num_angles,
            num_detectors=self.num_detectors,
            lambda_phys=lambda_phys_pace
        )

        # ADRN: Adaptive Diffusion Refinement Network
        self.adrn = CT_ADRN(
            in_channels=context_dim,
            out_channels=context_dim,
            img_size=img_size,
            num_angles=num_angles,
            num_detectors=self.num_detectors,
            num_inference_steps=num_diffusion_steps
        )

        # ART: Adaptive Reconstruction Transformer
        self.art = CT_ART(
            in_channels=context_dim,
            out_channels=in_channels,
            img_size=img_size,
            num_angles=num_angles,
            num_detectors=self.num_detectors,
            use_final_dc=use_final_dc
        )

    def forward(
        self,
        sinogram: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
        return_losses: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct CT image from sinogram.

        Args:
            sinogram: Measured sinogram (B, 1, num_angles, num_detectors)
            weights: Optional WLS weights (e.g., photon counts)
            mask: Optional mask for sparse-view CT
            return_intermediate: Return intermediate outputs
            return_losses: Return regularization losses

        Returns:
            Dictionary containing:
            - 'reconstruction': Final CT reconstruction (B, 1, H, W)
            - 'fbp': Initial FBP reconstruction
            - (optional) intermediate outputs and losses
        """
        outputs = {}

        # 1. Initial FBP reconstruction
        x_fbp = self.fbp(sinogram)
        outputs['fbp'] = x_fbp

        # 2. LPCE encoding
        if return_losses:
            z_latent, x_fbp_out, lpce_loss = self.lpce(
                sinogram, x_fbp, weights, return_reg_loss=True
            )
            outputs['lpce_loss'] = lpce_loss
        else:
            z_latent, x_fbp_out = self.lpce(sinogram, x_fbp, weights)

        if return_intermediate:
            outputs['z_latent'] = z_latent

        # 3. PACE context encoding
        if return_losses:
            z_pace, pace_loss = self.pace(
                z_latent, sinogram, weights, return_reg_loss=True
            )
            outputs['pace_loss'] = pace_loss
        else:
            z_pace = self.pace(z_latent, sinogram, weights)

        if return_intermediate:
            outputs['z_pace'] = z_pace

        # 4. ADRN diffusion refinement
        z_adrn = self.adrn(z_pace, sinogram, weights, mask)

        if return_intermediate:
            outputs['z_adrn'] = z_adrn

        # 5. ART final synthesis
        if return_losses:
            x_rec, art_loss = self.art(
                z_pace, z_adrn, sinogram, weights, mask, return_reg_loss=True
            )
            outputs['art_loss'] = art_loss
        else:
            x_rec = self.art(z_pace, z_adrn, sinogram, weights, mask)

        outputs['reconstruction'] = x_rec

        return outputs

    def get_parameters_for_optimizer(self, lr: float = 6e-3) -> list:
        """Get parameter groups for optimizer."""
        return [
            {'params': self.lpce.parameters(), 'lr': lr},
            {'params': self.pace.parameters(), 'lr': lr},
            {'params': self.adrn.parameters(), 'lr': lr * 0.5},  # Lower LR for diffusion
            {'params': self.art.parameters(), 'lr': lr}
        ]


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) Optimizer.

    Improves generalization by minimizing both loss and loss sharpness.
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Perturb parameters in gradient direction."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)

            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # Perturb
                self.state[p]['e_w'] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Unperturb and update."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]['e_w'])  # Unperturb

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        """Compute gradient norm."""
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def step(self, closure=None):
        """Not used directly - use first_step and second_step."""
        raise NotImplementedError("Use first_step and second_step")


def create_model(config: Dict[str, Any]) -> CT_PINN_DADif:
    """
    Create CT-PINN-DADif model from configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        CT_PINN_DADif model
    """
    return CT_PINN_DADif(
        img_size=config.get('img_size', 256),
        num_angles=config.get('num_angles', 180),
        num_detectors=config.get('num_detectors', None),
        in_channels=config.get('in_channels', 1),
        base_channels=config.get('base_channels', 64),
        latent_dim=config.get('latent_dim', 128),
        context_dim=config.get('context_dim', 256),
        lambda_phys_lpce=config.get('lambda_phys_lpce', 0.3),
        lambda_phys_pace=config.get('lambda_phys_pace', 0.1),
        num_diffusion_steps=config.get('num_diffusion_steps', 12),
        use_final_dc=config.get('use_final_dc', True)
    )


def create_loss(config: Dict[str, Any]) -> CTReconstructionLoss:
    """
    Create loss function from configuration.

    Args:
        config: Loss configuration dictionary

    Returns:
        CTReconstructionLoss
    """
    return CTReconstructionLoss(
        img_size=config.get('img_size', 256),
        num_angles=config.get('num_angles', 180),
        num_detectors=config.get('num_detectors', None),
        alpha=config.get('alpha', 0.5),
        beta=config.get('beta', 0.2),
        gamma=config.get('gamma', 0.3),
        use_poisson=config.get('use_poisson', False),
        I0=config.get('I0', 1e4),
        tv_weight=config.get('tv_weight', 1e-4),
        nonneg_weight=config.get('nonneg_weight', 1e-3),
        use_perceptual=config.get('use_perceptual', False)
    )
