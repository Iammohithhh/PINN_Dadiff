# PINN-DADif: Physics-Informed Deep Adaptive Diffusion Network  
Implementation of the PINN-DADif research paper (Ahmed et al., 2025)  
---

## Introduction  
This repository hosts the implementation of the PINN-DADif model: a **Physics-Informed Deep Adaptive Diffusion Network** designed for **robust and efficient MRI reconstruction**. The original paper:  
> Ahmed S., Feng J., Mehmood A., Ali M. U., Yaqub M., Azfar Yaqub M. A., Manan M. A., Raheem A. (2025). *PINN-DADif: Physics-Informed Deep Adaptive Diffusion Network for Robust and Efficient MRI Reconstruction*. _Digital Signal Processing: A Review Journal_, 160, 105085. :contentReference[oaicite:1]{index=1}  

The core innovation integrates physics-informed constraints (via a PINN framework) with a deep adaptive diffusion network to improve reconstruction quality and robustness in MRI imaging tasks.

---

## Project Structure  
```text
/  
├─ data/                    # raw and processed data  
│   ├─ raw/                 # original MRI scans (e.g., k-space, image space)  
│   └─ processed/           # pre-processed / paired data for training  
├─ experiments/             # experiment logs, checkpoints  
├─ models/                  # model definitions & saved weights  
├─ src/                     # source code  
│   ├─ data_loader.py       # dataset loading & preprocessing  
│   ├─ model.py             # model architecture: PINN + diffusion network  
│   ├─ loss_functions.py    # customized losses (physics residuals + data fidelity)  
│   ├─ train.py             # training script  
│   ├─ evaluate.py          # evaluation / inference script  
│   └─ utils.py             # helper functions, logging, visualization  
├─ notebooks/               # optional Jupyter notebooks for exploration  
├─ README.md                # this file  
└─ requirements.txt         # Python dependencies  
