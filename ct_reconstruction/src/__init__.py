"""
CT-PINN-DADif: Physics-Informed Deep Adaptive Diffusion Network for CT Reconstruction.

This package adapts the PINN-DADif architecture from MRI to CT reconstruction,
replacing k-space/Fourier physics with Radon transform/sinogram physics.

Modules:
    ct_physics: CT physics operations (Radon, FBP, noise models, losses)
    lpce: Latent Physics-Constrained Encoder
    pace: Physics-Aware Context Encoder
    adrn: Adaptive Diffusion Refinement Network
    art: Adaptive Reconstruction Transformer
    model: Complete CT-PINN-DADif model
    data_loader: Dataset and dataloader utilities
"""

from .ct_physics import (
    RadonTransform,
    FilteredBackProjection,
    CTForwardModel,
    CTDataConsistency,
    PoissonNLLLoss,
    WeightedLeastSquaresLoss,
    TotalVariationLoss,
    NonNegativityLoss,
    create_sparse_view_mask,
    create_limited_angle_mask
)

from .lpce import CT_LPCE
from .pace import CT_PACE
from .adrn import CT_ADRN
from .art import CT_ART
from .model import CT_PINN_DADif, CTReconstructionLoss, SAM, create_model, create_loss
from .data_loader import (
    SimulatedCTDataset,
    RealCTDataset,
    create_dataloaders,
    create_shepp_logan_phantom
)

__version__ = '1.0.0'
__author__ = 'Adapted from PINN-DADif (Ahmed et al., 2025)'

__all__ = [
    # Physics
    'RadonTransform',
    'FilteredBackProjection',
    'CTForwardModel',
    'CTDataConsistency',
    'PoissonNLLLoss',
    'WeightedLeastSquaresLoss',
    'TotalVariationLoss',
    'NonNegativityLoss',
    'create_sparse_view_mask',
    'create_limited_angle_mask',
    # Model components
    'CT_LPCE',
    'CT_PACE',
    'CT_ADRN',
    'CT_ART',
    # Main model
    'CT_PINN_DADif',
    'CTReconstructionLoss',
    'SAM',
    'create_model',
    'create_loss',
    # Data
    'SimulatedCTDataset',
    'RealCTDataset',
    'create_dataloaders',
    'create_shepp_logan_phantom'
]
