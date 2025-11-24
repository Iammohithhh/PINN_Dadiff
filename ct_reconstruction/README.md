# CT-PINN-DADif: Physics-Informed Deep Adaptive Diffusion Network for CT Reconstruction

This module adapts the PINN-DADif architecture (Ahmed et al., 2025) from MRI to **Computed Tomography (CT)** reconstruction. The key innovation is replacing MRI k-space/Fourier physics with CT Radon transform/sinogram physics while preserving the powerful deep learning components.

## Key Adaptations from MRI to CT

| Component | MRI (Original) | CT (This Implementation) |
|-----------|---------------|-------------------------|
| Forward Model | Fourier Transform (k-space) | Radon Transform (sinogram) |
| Inverse | Inverse FFT | Filtered Back Projection (FBP) |
| Noise Model | Gaussian | Poisson (photon counting) |
| Data Consistency | k-space consistency | Sinogram consistency |
| Physics Regularization | Bloch equations, T1/T2 | Total Variation, non-negativity |
| Undersampling | k-space masks | Sparse-view, limited-angle |

## Architecture Overview

```
Input: Measured Sinogram y (sparse/noisy)
         │
         ▼
    ┌─────────────┐
    │  FBP (x₀)   │  Initial reconstruction
    └─────────────┘
         │
         ▼
    ┌─────────────────────────────────────────┐
    │           CT-LPCE                        │
    │  Latent Physics-Constrained Encoder     │
    │  • Data-driven branch (CNN on FBP)      │
    │  • Physics branch (sinogram residuals)  │
    │  Z_latent = f_data(x_FBP) + λ·f_phys(r) │
    └─────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────┐
    │           CT-PACE                        │
    │  Physics-Aware Context Encoder          │
    │  • ASPP (multi-scale features)          │
    │  • Non-Local blocks (long-range deps)   │
    │  • Dual Attention (channel + spatial)   │
    └─────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────┐
    │           CT-ADRN                        │
    │  Adaptive Diffusion Refinement Network  │
    │  • Forward/Reverse diffusion            │
    │  • Physics projection each step:        │
    │    x ← x - η·Aᵀ W(Ax - y)               │
    │  • Transformer for global context       │
    │  • Adaptive prior adjustment            │
    └─────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────┐
    │           CT-ART                         │
    │  Adaptive Reconstruction Transformer    │
    │  • Dynamic convolutions                 │
    │  • Transformer fusion                   │
    │  • Final data consistency (CG steps)    │
    └─────────────────────────────────────────┘
         │
         ▼
    Output: Reconstructed CT Image x_rec
```

## CT Physics Equations

### 1. Forward Model (Radon Transform)
```
p(θ, s) = ∫ μ(x, y) dl   (line integral of attenuation)

Matrix form: y = A·x
```

### 2. Poisson Noise Model
```
Y_i ~ Poisson(I₀ · exp(-[Ax]_i) + scatter)
```

### 3. Log-domain Approximation
```
p = -log(Y/I₀) ≈ Ax + noise
Var(p_i) ≈ 1/Y_i  →  weights w_i = Y_i
```

### 4. Data Consistency Loss
**Weighted Least Squares:**
```
L_WLS = (1/2) · ||W^(1/2)(Ax - p)||²
```

**Poisson NLL:**
```
L_Poisson = Σᵢ [λᵢ - Yᵢ·log(λᵢ)]
where λᵢ = I₀·exp(-[Ax]ᵢ) + scatter
```

### 5. Total Loss Function
```
L_total = α·L_pixel + β·L_perc + γ·L_phys

where L_phys = L_WLS + η_TV·TV(x) + η_nonneg·||min(x,0)||²
```

## Project Structure

```
ct_reconstruction/
├── src/
│   ├── __init__.py          # Package exports
│   ├── ct_physics.py        # Radon, FBP, noise models, losses
│   ├── lpce.py              # Latent Physics-Constrained Encoder
│   ├── pace.py              # Physics-Aware Context Encoder
│   ├── adrn.py              # Adaptive Diffusion Refinement Network
│   ├── art.py               # Adaptive Reconstruction Transformer
│   ├── model.py             # Complete CT-PINN-DADif model
│   ├── data_loader.py       # Datasets and dataloaders
│   └── train.py             # Training script
├── notebooks/
│   └── CT_PINN_DADif_Training.ipynb  # Colab training notebook
├── experiments/             # Checkpoints and logs
├── data/                    # Dataset storage
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone repository
git clone https://github.com/Iammohithhh/PINN_Dadiff.git
cd PINN_Dadiff/ct_reconstruction

# Install dependencies
pip install -r requirements.txt

# Optional: Install ASTRA toolbox for fast GPU projections
# conda install -c astra-toolbox astra-toolbox
```

## Quick Start

### 1. Training with Simulated Data

```python
from src import CT_PINN_DADif, create_dataloaders, create_model, create_loss
from src.train import train_ct_pinn_dadif

config = {
    'img_size': 256,
    'num_angles': 180,
    'num_train_samples': 1000,
    'noise_level': 'low',
    'acquisition_type': 'sparse',  # 'full', 'sparse', 'limited'
    'num_views': 60,
    'batch_size': 4,
    'num_epochs': 600,
    'learning_rate': 6e-3,
    'use_sam': True
}

history = train_ct_pinn_dadif(config)
```

### 2. Inference

```python
import torch
from src import CT_PINN_DADif, FilteredBackProjection, RadonTransform

# Load model
model = CT_PINN_DADif(img_size=256, num_angles=180)
model.load_state_dict(torch.load('best_model.pt')['model_state_dict'])
model.eval()

# Reconstruct
with torch.no_grad():
    outputs = model(sinogram, weights=counts, mask=sparse_mask)
    reconstruction = outputs['reconstruction']
```

### 3. Google Colab

Open `notebooks/CT_PINN_DADif_Training.ipynb` in Google Colab for a complete training tutorial with GPU support.

## Supported CT Scenarios

### 1. Low-Dose CT
- Reduced photon counts (Poisson noise)
- Adjust `I0` parameter (e.g., 1e3 for low-dose, 1e4 for normal)

### 2. Sparse-View CT
- Reduced number of projection angles
- Use `acquisition_type='sparse'` and `num_views=30/60/90`

### 3. Limited-Angle CT
- Missing angular range
- Use `acquisition_type='limited'` and specify `missing_angle_range`

### 4. Combined Scenarios
- Low-dose + sparse-view for maximum acceleration

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_diffusion_steps` | 12 | Reverse diffusion iterations |
| `lambda_phys_lpce` | 0.3 | Physics weight in LPCE |
| `lambda_phys_pace` | 0.1 | Physics weight in PACE |
| `alpha` | 0.5 | Pixel loss weight |
| `beta` | 0.2 | Perceptual loss weight |
| `gamma` | 0.3 | Physics loss weight |
| `tv_weight` | 1e-4 | Total Variation regularization |
| `learning_rate` | 6e-3 | Initial learning rate |
| `sam_rho` | 0.05 | SAM perturbation radius |

## Datasets

### Supported Datasets
- **LIDC-IDRI**: Lung CT scans
- **Mayo Low Dose CT**: Full and quarter-dose pairs
- **AAPM Low Dose CT Grand Challenge**
- **Simulated phantoms**: Shepp-Logan, random ellipses

### Data Format
```python
# Expected format for real data
sample = {
    'image': torch.Tensor,      # (1, H, W) ground truth
    'sinogram': torch.Tensor,   # (1, num_angles, num_detectors)
    'weights': torch.Tensor,    # (1, num_angles, num_detectors) WLS weights
}
```

## Expected Results

Based on the MRI PINN-DADif paper, expected CT reconstruction performance:

| Scenario | PSNR (dB) | SSIM (%) |
|----------|-----------|----------|
| Full-view, low noise | 40-42 | 98-99 |
| Sparse-view (60 views) | 36-38 | 95-97 |
| Sparse-view (30 views) | 33-35 | 92-94 |
| Limited-angle (±30° missing) | 34-36 | 93-95 |
| Low-dose (I0=1e3) | 35-37 | 94-96 |

## Citation

If you use this code, please cite:

```bibtex
@article{ahmed2025pinn,
  title={PINN-DADif: Physics-Informed Deep Adaptive Diffusion Network for Robust and Efficient MRI Reconstruction},
  author={Ahmed, Shahzad and Feng, Jinchao and Mehmood, Atif and Ali, Muhammad Usman and Yaqub, Muhammad and Manan, Malik Abdul and Raheem, Abdul},
  journal={Digital Signal Processing},
  volume={160},
  pages={105085},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Original PINN-DADif paper authors (Ahmed et al., 2025)
- ASTRA Toolbox for CT projection operators
- PyTorch team for deep learning framework
