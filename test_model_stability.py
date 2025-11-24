"""
Quick test to verify model stability - no NaN in forward pass.
"""

import sys
import torch
import numpy as np

# Add to path
sys.path.insert(0, '/home/user/PINN_Dadiff')

from ct_reconstruction.src import (
    CT_PINN_DADif,
    create_model,
    create_shepp_logan_phantom,
    CTForwardModel
)

def test_forward_pass():
    """Test that model forward pass doesn't produce NaN."""

    print("="*60)
    print("TESTING MODEL STABILITY - Forward Pass")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create model with stable config
    config = {
        'img_size': 256,
        'num_angles': 180,
        'num_detectors': None,
        'base_channels': 64,
        'latent_dim': 128,
        'context_dim': 256,
        'num_diffusion_steps': 12,
        'lambda_phys_lpce': 0.05,
        'lambda_phys_pace': 0.01,
        'use_final_dc': False,
    }

    print("\n" + "-"*60)
    print("Creating model...")
    print("-"*60)

    model = create_model(config).to(device)
    model.eval()

    print("✓ Model created successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create test data
    print("\n" + "-"*60)
    print("Creating test phantom and sinogram...")
    print("-"*60)

    phantom = create_shepp_logan_phantom(256)
    x_gt = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0).float().to(device)

    # Forward model
    ct_forward = CTForwardModel(img_size=256, num_angles=180, I0=1e4, device=device)
    sinogram, counts = ct_forward(x_gt, add_noise=True, return_counts=True)
    weights = ct_forward.get_weights(counts)

    print(f"✓ Test data created")
    print(f"  Phantom: {x_gt.shape}, range=[{x_gt.min():.3f}, {x_gt.max():.3f}]")
    print(f"  Sinogram: {sinogram.shape}, range=[{sinogram.min():.3f}, {sinogram.max():.3f}]")
    print(f"  Weights: {weights.shape}, range=[{weights.min():.1f}, {weights.max():.1f}]")

    # Forward pass
    print("\n" + "-"*60)
    print("Running forward pass...")
    print("-"*60)

    with torch.no_grad():
        try:
            x_rec = model(sinogram, weights=weights)

            # Check for NaN
            has_nan = torch.isnan(x_rec).any().item()
            has_inf = torch.isinf(x_rec).any().item()

            print(f"\n✓ Forward pass completed")
            print(f"  Output: {x_rec.shape}")
            print(f"  Range: [{x_rec.min():.3f}, {x_rec.max():.3f}]")
            print(f"  Mean: {x_rec.mean():.3f}, Std: {x_rec.std():.3f}")
            print(f"  Has NaN: {has_nan}")
            print(f"  Has Inf: {has_inf}")

            if has_nan or has_inf:
                print("\n❌ FAILED: Model output contains NaN or Inf!")
                return False
            else:
                print("\n✅ SUCCESS: Model output is stable (no NaN or Inf)")
                return True

        except Exception as e:
            print(f"\n❌ FAILED: Exception during forward pass")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def test_multiple_iterations(n=5):
    """Run multiple forward passes to check consistency."""

    print("\n\n" + "="*60)
    print(f"TESTING CONSISTENCY - {n} Forward Passes")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model
    config = {
        'img_size': 256,
        'num_angles': 180,
        'num_detectors': None,
        'base_channels': 64,
        'latent_dim': 128,
        'context_dim': 256,
        'num_diffusion_steps': 12,
        'lambda_phys_lpce': 0.05,
        'lambda_phys_pace': 0.01,
        'use_final_dc': False,
    }

    model = create_model(config).to(device)
    model.eval()

    # Create test data
    phantom = create_shepp_logan_phantom(256)
    x_gt = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0).float().to(device)

    ct_forward = CTForwardModel(img_size=256, num_angles=180, I0=1e4, device=device)
    sinogram, counts = ct_forward(x_gt, add_noise=True, return_counts=True)
    weights = ct_forward.get_weights(counts)

    results = []
    failures = 0

    for i in range(n):
        with torch.no_grad():
            try:
                x_rec = model(sinogram, weights=weights)

                has_nan = torch.isnan(x_rec).any().item()
                has_inf = torch.isinf(x_rec).any().item()

                status = "✓" if not (has_nan or has_inf) else "✗"
                results.append(not (has_nan or has_inf))

                if has_nan or has_inf:
                    failures += 1

                print(f"  Iteration {i+1}/{n}: {status} "
                      f"range=[{x_rec.min():.3f}, {x_rec.max():.3f}], "
                      f"mean={x_rec.mean():.3f}")

            except Exception as e:
                print(f"  Iteration {i+1}/{n}: ✗ Exception: {str(e)}")
                results.append(False)
                failures += 1

    success_rate = (n - failures) / n * 100

    print(f"\n{'='*60}")
    print(f"Results: {n-failures}/{n} successful ({success_rate:.1f}%)")
    print(f"{'='*60}")

    if failures == 0:
        print("✅ ALL TESTS PASSED - Model is stable!")
        return True
    else:
        print(f"❌ {failures} TESTS FAILED - Model has stability issues")
        return False

if __name__ == '__main__':
    # Run tests
    test1_passed = test_forward_pass()
    test2_passed = test_multiple_iterations(n=5)

    print("\n\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Single forward pass: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Multiple iterations: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print("="*60)

    if test1_passed and test2_passed:
        print("\n🎉 All stability tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed - check model stability")
        sys.exit(1)
