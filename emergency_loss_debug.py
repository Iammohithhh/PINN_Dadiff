"""
EMERGENCY LOSS & GRADIENT DEBUG

Checks loss computation and gradient flow for NaN.
Run this if emergency_nan_debug.py passes but training still fails.
"""

import sys
sys.path.insert(0, '/home/user/PINN_Dadiff')

import torch
import numpy as np
from ct_reconstruction.src import (
    create_model,
    create_loss,
    create_shepp_logan_phantom,
    CTForwardModel
)

def check_loss_and_gradients():
    """Check loss computation and gradient flow."""

    print("="*70)
    print("EMERGENCY LOSS & GRADIENT DEBUGGING")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}\n")

    # Minimal config
    config = {
        'img_size': 256,
        'num_angles': 180,
        'num_detectors': None,
        'base_channels': 64,
        'latent_dim': 128,
        'context_dim': 256,
        'num_diffusion_steps': 1,
        'lambda_phys_lpce': 0.0,   # Start with 0
        'lambda_phys_pace': 0.0,
        'use_final_dc': False,
        'alpha': 1.0,              # Only pixel loss
        'beta': 0.0,
        'gamma': 0.0,              # No physics loss
        'tv_weight': 0.0,
        'nonneg_weight': 0.0,
        'use_perceptual': False,
        'use_poisson': False,
    }

    print("-"*70)
    print("Creating model and loss (MINIMAL config)")
    print("  Only pixel loss enabled (alpha=1.0)")
    print("  All physics losses disabled")
    print("-"*70)

    model = create_model(config).to(device)
    loss_fn = create_loss(config).to(device)

    model.train()  # Training mode

    print("\n" + "-"*70)
    print("Creating test batch")
    print("-"*70)

    phantom = create_shepp_logan_phantom(256)
    x_gt = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0).float().to(device)

    ct_forward = CTForwardModel(img_size=256, num_angles=180, I0=1e4, device=device)
    sinogram, counts = ct_forward(x_gt, add_noise=True, return_counts=True)
    weights = ct_forward.get_weights(counts)

    print(f"✓ Data created")

    print("\n" + "="*70)
    print("TEST 1: Forward Pass (Training Mode)")
    print("="*70)

    try:
        outputs = model(sinogram, weights=weights)
        x_rec = outputs['reconstruction']

        print(f"✓ Forward pass successful")
        print(f"  Reconstruction range: [{x_rec.min():.3f}, {x_rec.max():.3f}]")

        has_nan = torch.isnan(x_rec).any().item()
        has_inf = torch.isinf(x_rec).any().item()

        if has_nan or has_inf:
            print(f"❌ Output has NaN/Inf!")
            return False

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*70)
    print("TEST 2: Loss Computation (Pixel Loss Only)")
    print("="*70)

    try:
        loss_dict = loss_fn(outputs, x_gt, sinogram, weights)
        total_loss = loss_dict['total']

        print(f"✓ Loss computation successful")
        print(f"  Total loss: {total_loss.item():.6f}")
        print(f"  Pixel loss: {loss_dict['pixel'].item():.6f}")

        if torch.isnan(total_loss):
            print(f"❌ Loss is NaN!")
            print(f"Loss breakdown:")
            for k, v in loss_dict.items():
                print(f"  {k}: {v.item() if not torch.isnan(v) else 'NaN'}")
            return False

    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*70)
    print("TEST 3: Backward Pass (Gradient Computation)")
    print("="*70)

    try:
        model.zero_grad()
        total_loss.backward()

        print(f"✓ Backward pass successful")

        # Check gradients
        nan_grads = []
        max_grad = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
                grad_norm = param.grad.norm().item()
                max_grad = max(max_grad, grad_norm)

        if nan_grads:
            print(f"❌ NaN gradients found in {len(nan_grads)} parameters:")
            for name in nan_grads[:5]:  # Show first 5
                print(f"  - {name}")
            return False

        print(f"  Max gradient norm: {max_grad:.6f}")

        if max_grad > 1e6:
            print(f"  ⚠️  WARNING: Very large gradients detected!")

    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*70)
    print("TEST 4: Optimizer Step")
    print("="*70)

    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Very small LR
        optimizer.step()

        print(f"✓ Optimizer step successful")

        # Check parameters after update
        nan_params = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)

        if nan_params:
            print(f"❌ NaN in parameters after optimizer step:")
            for name in nan_params[:5]:
                print(f"  - {name}")
            return False

    except Exception as e:
        print(f"❌ Optimizer step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*70)
    print("TEST 5: Second Forward Pass (After Update)")
    print("="*70)

    try:
        outputs2 = model(sinogram, weights=weights)
        x_rec2 = outputs2['reconstruction']

        has_nan = torch.isnan(x_rec2).any().item()
        has_inf = torch.isinf(x_rec2).any().item()

        if has_nan or has_inf:
            print(f"❌ NaN/Inf appeared after one training step!")
            return False

        print(f"✓ Second forward pass successful")
        print(f"  Reconstruction range: [{x_rec2.min():.3f}, {x_rec2.max():.3f}]")

    except Exception as e:
        print(f"❌ Second forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nTraining should work with these settings.")
    print("\nNow gradually enable physics losses:")
    print("  1. Try gamma=0.01 (small physics weight)")
    print("  2. Try gamma=0.05 (recommended)")
    print("  3. Enable lambda_phys_lpce=0.05")
    print("  4. Enable lambda_phys_pace=0.01")

    return True

if __name__ == '__main__':
    success = check_loss_and_gradients()

    if not success:
        print("\n" + "="*70)
        print("❌ PROBLEM FOUND - See above")
        print("="*70)
        print("\nThe issue is in:")
        print("  - Loss computation, OR")
        print("  - Gradient calculation, OR")
        print("  - Parameter update")
        sys.exit(1)
    else:
        print("\n🎉 Loss and gradients are stable!")
        sys.exit(0)
