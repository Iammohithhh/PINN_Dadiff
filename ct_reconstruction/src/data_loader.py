"""
Data Loader for CT Reconstruction.

Supports various CT datasets including:
- LIDC-IDRI (Lung Image Database Consortium)
- Mayo Low Dose CT Challenge
- AAPM Low Dose CT Grand Challenge
- Simulated phantoms (Shepp-Logan)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union, Callable
import random

from .ct_physics import (
    RadonTransform,
    CTForwardModel,
    create_sparse_view_mask,
    create_limited_angle_mask
)


def create_shepp_logan_phantom(size: int = 256) -> np.ndarray:
    """
    Create Shepp-Logan phantom.

    Args:
        size: Image size

    Returns:
        Phantom image (size, size)
    """
    phantom = np.zeros((size, size), dtype=np.float32)

    # Ellipse parameters: (value, a, b, x0, y0, phi)
    ellipses = [
        (1.0, 0.69, 0.92, 0.0, 0.0, 0),
        (-0.8, 0.6624, 0.8740, 0.0, -0.0184, 0),
        (-0.2, 0.1100, 0.3100, 0.22, 0.0, -18),
        (-0.2, 0.1600, 0.4100, -0.22, 0.0, 18),
        (0.1, 0.2100, 0.2500, 0.0, 0.35, 0),
        (0.1, 0.0460, 0.0460, 0.0, 0.1, 0),
        (0.1, 0.0460, 0.0460, 0.0, -0.1, 0),
        (0.1, 0.0460, 0.0230, -0.08, -0.605, 0),
        (0.1, 0.0230, 0.0230, 0.0, -0.606, 0),
        (0.1, 0.0230, 0.0460, 0.06, -0.605, 0),
    ]

    # Create coordinate grid
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    for value, a, b, x0, y0, phi in ellipses:
        phi_rad = np.deg2rad(phi)
        cos_p, sin_p = np.cos(phi_rad), np.sin(phi_rad)

        # Rotated coordinates
        X_rot = (X - x0) * cos_p + (Y - y0) * sin_p
        Y_rot = -(X - x0) * sin_p + (Y - y0) * cos_p

        # Ellipse mask
        mask = (X_rot / a) ** 2 + (Y_rot / b) ** 2 <= 1
        phantom[mask] += value

    # Normalize to [0, 1]
    phantom = np.clip(phantom, 0, 1)

    return phantom


def add_gaussian_noise(image: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to image."""
    noise = np.random.randn(*image.shape).astype(np.float32) * std
    return np.clip(image + noise, 0, 1)


def create_random_phantom(size: int = 256, num_ellipses: int = 10) -> np.ndarray:
    """Create random phantom with ellipses."""
    phantom = np.zeros((size, size), dtype=np.float32)

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    for _ in range(num_ellipses):
        value = np.random.uniform(0.1, 0.5)
        a = np.random.uniform(0.05, 0.3)
        b = np.random.uniform(0.05, 0.3)
        x0 = np.random.uniform(-0.5, 0.5)
        y0 = np.random.uniform(-0.5, 0.5)
        phi = np.random.uniform(0, 180)

        phi_rad = np.deg2rad(phi)
        cos_p, sin_p = np.cos(phi_rad), np.sin(phi_rad)

        X_rot = (X - x0) * cos_p + (Y - y0) * sin_p
        Y_rot = -(X - x0) * sin_p + (Y - y0) * cos_p

        mask = (X_rot / a) ** 2 + (Y_rot / b) ** 2 <= 1
        phantom[mask] += value

    return np.clip(phantom, 0, 1)


class SimulatedCTDataset(Dataset):
    """
    Simulated CT dataset using phantoms.

    Generates training data with:
    - Random or Shepp-Logan phantoms
    - Forward projection (Radon transform)
    - Optional Poisson noise for low-dose simulation
    - Optional sparse-view/limited-angle masks
    """

    def __init__(
        self,
        num_samples: int = 1000,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        phantom_type: str = 'random',  # 'random', 'shepp_logan', 'mixed'
        noise_level: str = 'low',  # 'none', 'low', 'medium', 'high'
        I0: float = 1e4,  # Incident intensity for Poisson noise
        acquisition_type: str = 'full',  # 'full', 'sparse', 'limited'
        num_views: int = 60,  # For sparse-view
        missing_angle_range: Tuple[float, float] = (60, 120),  # For limited-angle
        transform: Optional[Callable] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_angles = num_angles
        self.num_detectors = num_detectors or int(np.ceil(np.sqrt(2) * img_size))
        self.phantom_type = phantom_type
        self.noise_level = noise_level
        self.I0 = I0
        self.acquisition_type = acquisition_type
        self.num_views = num_views
        self.missing_angle_range = missing_angle_range
        self.transform = transform

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Noise levels (standard deviation)
        self.noise_std = {
            'none': 0.0,
            'low': 0.01,
            'medium': 0.03,
            'high': 0.05
        }

        # Pre-generate phantoms
        self._generate_phantoms()

    def _generate_phantoms(self):
        """Pre-generate phantom images."""
        self.phantoms = []

        for i in range(self.num_samples):
            if self.phantom_type == 'shepp_logan':
                phantom = create_shepp_logan_phantom(self.img_size)
            elif self.phantom_type == 'random':
                phantom = create_random_phantom(self.img_size)
            else:  # mixed
                if random.random() < 0.3:
                    phantom = create_shepp_logan_phantom(self.img_size)
                else:
                    phantom = create_random_phantom(self.img_size)

            # Add slight variations
            phantom = add_gaussian_noise(phantom, std=0.005)
            self.phantoms.append(phantom)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.

        Returns:
            Dictionary with:
            - 'image': Ground truth image (1, H, W)
            - 'sinogram': Sinogram (1, num_angles, num_detectors)
            - 'sinogram_noisy': Noisy sinogram
            - 'weights': WLS weights
            - 'mask': Acquisition mask (for sparse/limited angle)
            - 'fbp': FBP reconstruction from noisy sinogram
        """
        # Get phantom
        image = self.phantoms[idx].copy()

        # Convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W)

        # Create forward model
        device = 'cpu'  # Use CPU for data loading
        radon = RadonTransform(
            self.img_size, self.num_angles, self.num_detectors, device=device
        )

        # Forward projection
        sinogram = radon.forward_fast(image_tensor.unsqueeze(0)).squeeze(0)  # (1, A, D)

        # Add noise
        if self.noise_level != 'none':
            # Poisson noise simulation
            counts = self.I0 * torch.exp(-sinogram)
            counts_noisy = torch.poisson(counts.clamp(min=1))
            sinogram_noisy = -torch.log(counts_noisy.clamp(min=1) / self.I0)
            weights = counts_noisy.clamp(min=1)

            # Additional Gaussian noise
            noise_std = self.noise_std[self.noise_level]
            sinogram_noisy = sinogram_noisy + noise_std * torch.randn_like(sinogram_noisy)
        else:
            sinogram_noisy = sinogram.clone()
            weights = torch.ones_like(sinogram)

        # Create acquisition mask
        if self.acquisition_type == 'sparse':
            mask = create_sparse_view_mask(self.num_angles, self.num_views, device)
            sinogram_noisy = sinogram_noisy * mask.squeeze(0)
        elif self.acquisition_type == 'limited':
            mask = create_limited_angle_mask(
                self.num_angles,
                missing_start=self.missing_angle_range[0],
                missing_end=self.missing_angle_range[1],
                device=device
            )
            sinogram_noisy = sinogram_noisy * mask.squeeze(0)
        else:
            mask = torch.ones(1, 1, self.num_angles, 1)

        # FBP reconstruction
        fbp = FilteredBackProjection(
            self.img_size, self.num_angles, self.num_detectors, device=device
        )
        fbp_recon = fbp(sinogram_noisy.unsqueeze(0)).squeeze(0)

        sample = {
            'image': image_tensor,
            'sinogram': sinogram,
            'sinogram_noisy': sinogram_noisy,
            'weights': weights,
            'mask': mask.squeeze(0),
            'fbp': fbp_recon,
            'acquisition_type': self.acquisition_type
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class RealCTDataset(Dataset):
    """
    Dataset for real CT data (DICOM files or preprocessed arrays).

    Supports:
    - LIDC-IDRI dataset
    - Mayo Low Dose CT
    - Any dataset in numpy format
    """

    def __init__(
        self,
        data_dir: str,
        img_size: int = 256,
        num_angles: int = 180,
        split: str = 'train',
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.num_angles = num_angles
        self.split = split
        self.transform = transform

        # Find data files
        self.samples = self._find_samples()

    def _find_samples(self) -> List[Path]:
        """Find all sample files in data directory."""
        samples = []

        # Look for numpy files
        for ext in ['*.npy', '*.npz']:
            samples.extend(self.data_dir.glob(f'**/{ext}'))

        # Sort for reproducibility
        samples = sorted(samples)

        # Split into train/val/test (70/15/15)
        n = len(samples)
        if self.split == 'train':
            samples = samples[:int(0.7 * n)]
        elif self.split == 'val':
            samples = samples[int(0.7 * n):int(0.85 * n)]
        else:  # test
            samples = samples[int(0.85 * n):]

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and process a sample."""
        path = self.samples[idx]

        # Load data
        if path.suffix == '.npy':
            data = np.load(path)
            image = torch.from_numpy(data).float()
        else:  # .npz
            data = np.load(path)
            image = torch.from_numpy(data['image']).float()
            sinogram = torch.from_numpy(data.get('sinogram', None))

        # Ensure correct shape
        if image.dim() == 2:
            image = image.unsqueeze(0)

        # Resize if needed
        if image.shape[1:] != (self.img_size, self.img_size):
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode='bilinear'
            ).squeeze(0)

        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


def create_dataloaders(
    config: Dict,
    batch_size: int = 4,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Dataset configuration
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_type = config.get('dataset_type', 'simulated')

    if dataset_type == 'simulated':
        # Create simulated datasets
        train_dataset = SimulatedCTDataset(
            num_samples=config.get('num_train_samples', 1000),
            img_size=config.get('img_size', 256),
            num_angles=config.get('num_angles', 180),
            phantom_type=config.get('phantom_type', 'mixed'),
            noise_level=config.get('noise_level', 'low'),
            acquisition_type=config.get('acquisition_type', 'full'),
            num_views=config.get('num_views', 60),
            seed=42
        )

        val_dataset = SimulatedCTDataset(
            num_samples=config.get('num_val_samples', 200),
            img_size=config.get('img_size', 256),
            num_angles=config.get('num_angles', 180),
            phantom_type=config.get('phantom_type', 'mixed'),
            noise_level=config.get('noise_level', 'low'),
            acquisition_type=config.get('acquisition_type', 'full'),
            num_views=config.get('num_views', 60),
            seed=123
        )

        test_dataset = SimulatedCTDataset(
            num_samples=config.get('num_test_samples', 200),
            img_size=config.get('img_size', 256),
            num_angles=config.get('num_angles', 180),
            phantom_type=config.get('phantom_type', 'mixed'),
            noise_level=config.get('noise_level', 'low'),
            acquisition_type=config.get('acquisition_type', 'full'),
            num_views=config.get('num_views', 60),
            seed=456
        )
    else:
        # Real dataset
        train_dataset = RealCTDataset(
            config['data_dir'], config.get('img_size', 256),
            config.get('num_angles', 180), 'train'
        )
        val_dataset = RealCTDataset(
            config['data_dir'], config.get('img_size', 256),
            config.get('num_angles', 180), 'val'
        )
        test_dataset = RealCTDataset(
            config['data_dir'], config.get('img_size', 256),
            config.get('num_angles', 180), 'test'
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# Need to import FilteredBackProjection for the dataset
from .ct_physics import FilteredBackProjection
