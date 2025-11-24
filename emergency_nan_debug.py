"""
EMERGENCY NaN DEBUG SCRIPT

This script traces EXACTLY where NaN appears by checking after every operation.
Run this FIRST before training to identify the problem.
"""

import sys
sys.path.insert(0, '/home/user/PINN_Dadiff')

import torch
import numpy as np
from ct_reconstruction.src import (
    create_model,
    create_shepp_logan_phantom,
    CTForwardModel
)

def check_tensor(name, tensor, step=""):
    """Check if tensor has NaN or Inf and print detailed stats."""
    if tensor is None:
        print(f"  ⚠️  {name} is None!")
        return True

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    prefix = "  ❌" if (has_nan or has_inf) else "  ✓"

    print(f"{prefix} {step}{name}:")
    print(f"      Shape: {tensor.shape}")
    print(f"      Range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
    print(f"      Mean: {tensor.mean().item():.6f}, Std: {tensor.std().item():.6f}")
    print(f"      Has NaN: {has_nan}, Has Inf: {has_inf}")

    if has_nan or has_inf:
        # Show where NaN/Inf appears
        nan_mask = torch.isnan(tensor) | torch.isinf(tensor)
        num_bad = nan_mask.sum().item()
        total = tensor.numel()
        print(f"      ⚠️  {num_bad}/{total} ({100*num_bad/total:.2f}%) values are NaN/Inf")
        return True

    return False

def trace_model_forward():
    """Trace model forward pass step by step."""

    print("="*70)
    print("EMERGENCY NaN DEBUGGING - STEP-BY-STEP TRACE")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}\n")

    # Ultra-conservative config
    config = {
        'img_size': 256,
        'num_angles': 180,
        'num_detectors': None,
        'base_channels': 64,
        'latent_dim': 128,
        'context_dim': 256,
        'num_diffusion_steps': 1,  # MINIMAL for debugging
        'lambda_phys_lpce': 0.0,   # DISABLED for debugging
        'lambda_phys_pace': 0.0,   # DISABLED for debugging
        'use_final_dc': False,
    }

    print("-"*70)
    print("Step 1: Creating model with DEBUG config")
    print("  num_diffusion_steps: 1 (minimal)")
    print("  lambda_phys: 0.0 (physics disabled)")
    print("-"*70)

    model = create_model(config).to(device)

    # Check if ADRN has correct beta values
    if hasattr(model, 'adrn'):
        print("\n🔍 Checking ADRN beta values...")
        print(f"  beta_min: {model.adrn.betas[0].item():.6f}")
        print(f"  beta_max: {model.adrn.betas[-1].item():.6f}")

        if model.adrn.betas[-1].item() > 0.1:
            print("  ❌ ERROR: beta_max is still too high! Old code is being used!")
            print("  This means the notebook is loading OLD code from GitHub, not your local changes!")
            return False
        else:
            print("  ✅ Beta values are correct")

    model.eval()

    print("\n" + "-"*70)
    print("Step 2: Creating test data")
    print("-"*70)

    phantom = create_shepp_logan_phantom(256)
    x_gt = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0).float().to(device)

    ct_forward = CTForwardModel(img_size=256, num_angles=180, I0=1e4, device=device)
    sinogram, counts = ct_forward(x_gt, add_noise=True, return_counts=True)
    weights = ct_forward.get_weights(counts)

    check_tensor("Input phantom", x_gt, "INPUT: ")
    check_tensor("Sinogram", sinogram, "INPUT: ")
    check_tensor("Weights", weights, "INPUT: ")

    print("\n" + "="*70)
    print("Step 3: TRACING MODEL FORWARD PASS")
    print("="*70)

    with torch.no_grad():
        # Manually trace through model
        print("\n>>> LPCE Forward")
        try:
            z_lpce, x_fbp = model.lpce(sinogram, weights=weights, return_reg_loss=False)
            if check_tensor("LPCE output (z_lpce)", z_lpce):
                print("\n❌ NaN FOUND IN LPCE!")
                return False
            if check_tensor("FBP reconstruction", x_fbp):
                print("\n❌ NaN FOUND IN FBP!")
                return False
        except Exception as e:
            print(f"❌ LPCE FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

        print("\n>>> PACE Forward")
        try:
            z_pace = model.pace(z_lpce, sinogram=sinogram, weights=weights, return_reg_loss=False)
            if check_tensor("PACE output (z_pace)", z_pace):
                print("\n❌ NaN FOUND IN PACE!")
                return False
        except Exception as e:
            print(f"❌ PACE FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

        print("\n>>> ADRN Forward")
        try:
            # Check ADRN inputs
            print("  ADRN inputs:")
            check_tensor("  z_pace (input to ADRN)", z_pace)

            z_adrn = model.adrn(z_pace, sinogram=sinogram, weights=weights)
            if check_tensor("ADRN output (z_adrn)", z_adrn):
                print("\n❌ NaN FOUND IN ADRN!")
                print("\nDebugging ADRN internals...")

                # Check diffusion schedule
                print(f"  Num timesteps: {model.adrn.num_timesteps}")
                print(f"  Num inference steps: {model.adrn.num_inference_steps}")
                print(f"  Beta range: [{model.adrn.betas.min().item():.6f}, {model.adrn.betas.max().item():.6f}]")
                print(f"  Alpha range: [{model.adrn.alphas.min().item():.6f}, {model.adrn.alphas.max().item():.6f}]")
                print(f"  Alpha cumprod range: [{model.adrn.alphas_cumprod.min().item():.6f}, {model.adrn.alphas_cumprod.max().item():.6f}]")

                return False
        except Exception as e:
            print(f"❌ ADRN FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

        print("\n>>> ART Forward")
        try:
            x_rec = model.art(z_pace, z_adrn, sinogram=sinogram, weights=weights, return_reg_loss=False)
            if check_tensor("ART output (reconstruction)", x_rec):
                print("\n❌ NaN FOUND IN ART!")
                return False
        except Exception as e:
            print(f"❌ ART FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "="*70)
    print("✅ SUCCESS: No NaN found in any module!")
    print("="*70)
    print("\nIf you're still getting NaN during training, the issue is in:")
    print("  1. Loss computation")
    print("  2. Backward pass / gradients")
    print("  3. Optimizer step")
    print("\nNext: Run emergency_loss_debug.py to check loss computation")

    return True

if __name__ == '__main__':
    success = trace_model_forward()

    if not success:
        print("\n" + "="*70)
        print("❌ NaN DETECTED - See above for exact location")
        print("="*70)
        print("\nPossible causes:")
        print("  1. OLD CODE: Notebook is loading code from GitHub, not local changes")
        print("     Solution: In Colab, restart runtime and re-run setup cells")
        print("  2. Wrong beta values still in ADRN")
        print("  3. Data issue: Sinogram or weights have NaN/Inf")
        print("  4. Numerical instability in a specific module")
        sys.exit(1)
    else:
        print("\n🎉 All modules are stable!")
        sys.exit(0)
