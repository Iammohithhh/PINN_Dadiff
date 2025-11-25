# CT-PINN-DADif: Physics-Informed CT Reconstruction

Adaptation of PINN-DADif from MRI to CT reconstruction using Radon transform physics.

## Quick Start
```bash
# Install
pip install torch numpy scipy matplotlib tqdm

# Clone
git clone https://github.com/Iammohithhh/PINN_Dadiff.git
cd PINN_Dadiff/ct_reconstruction

# Train (5 epochs proof-of-concept)
python run_training.py --mode demo

# Or use notebook
# Open: notebooks/CT_PINN_DADif_Training.ipynb in Colab
```

## Results (5 Epochs)

| Method | PSNR | SSIM |
|--------|------|------|
| FBP | 26.14 dB | 71.82% |
| CT-PINN-DADif | 26.73 dB | 73.91% |
| Improvement | +0.59 dB | +2.09% |

⚠️ **Note**: Proof-of-concept only. Full training (600-1000 epochs) needed for clinical use.

## Key Features

- ✅ Differentiable Radon transform (validated: 4.3% adjoint error)
- ✅ Poisson noise modeling
- ✅ Four-stage pipeline: LPCE → PACE → ADRN → ART
- ✅ Stable training (no NaN losses)

## Citation
```bibtex
@article{ahmed2025pinn,
  title={PINN-DADif for MRI Reconstruction},
  author={Ahmed et al.},
  journal={Digital Signal Processing},
  year={2025}
}
```

## License

MIT
