"""
Diagnostic script to analyze CT reconstruction test results.

This script will:
1. Load your trained model
2. Run evaluation with CORRECTED metrics
3. Analyze model output ranges
4. Create proper visualizations
5. Identify issues with training
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add to path
sys.path.insert(0, '/home/user/PINN_Dadiff')

from ct_reconstruction.src import (
    create_model,
    create_dataloaders,
    compute_metrics,
    DEFAULT_CONFIG
)

def diagnose_model(checkpoint_path: str = 'experiments/checkpoints/best_model.pt'):
    """
    Comprehensive diagnosis of model performance.
    """
    print("="*70)
    print("CT-PINN-DADif DIAGNOSTIC REPORT")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    print(f"  Epochs trained: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Config: {config['num_epochs']} total epochs planned")

    # Create model
    print("\nCreating model...")
    model = create_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("  ✓ Model loaded successfully")

    # Create test data
    print("\nCreating test dataloader...")
    _, _, test_loader = create_dataloaders(
        config, batch_size=config['batch_size'], num_workers=2
    )
    print(f"  Test samples: {len(test_loader.dataset)}")

    # Evaluate with corrected metrics
    print("\n" + "="*70)
    print("EVALUATING WITH CORRECTED METRICS")
    print("="*70)

    test_metrics = {'psnr': [], 'ssim': [], 'rmse': [], 'mae': []}
    fbp_metrics = {'psnr': [], 'ssim': [], 'rmse': [], 'mae': []}

    # Track output ranges
    model_outputs_min = []
    model_outputs_max = []
    model_outputs_mean = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            sinogram = batch['sinogram_noisy'].to(device)
            target = batch['image'].to(device)
            weights = batch['weights'].to(device)
            mask = batch['mask'].to(device)
            fbp_recon = batch['fbp'].to(device)

            # Model reconstruction
            outputs = model(sinogram, weights, mask)
            pred = outputs['reconstruction']

            # Track output statistics
            model_outputs_min.append(pred.min().item())
            model_outputs_max.append(pred.max().item())
            model_outputs_mean.append(pred.mean().item())

            # Compute metrics
            m = compute_metrics(pred, target)
            for k in test_metrics:
                test_metrics[k].append(m[k])

            m_fbp = compute_metrics(fbp_recon, target)
            for k in fbp_metrics:
                fbp_metrics[k].append(m_fbp[k])

    # Print output range analysis
    print("\n" + "="*70)
    print("MODEL OUTPUT ANALYSIS")
    print("="*70)
    print(f"Output range:")
    print(f"  Min: {min(model_outputs_min):.4f}")
    print(f"  Max: {max(model_outputs_max):.4f}")
    print(f"  Mean: {np.mean(model_outputs_mean):.4f}")
    print(f"\nExpected range: [0.0, 1.0]")

    if max(model_outputs_max) < 0.1:
        print("\n⚠️  WARNING: Model outputs are very small (near zero)")
        print("   This suggests insufficient training or learning rate issues")
    elif max(model_outputs_max) > 1.5:
        print("\n⚠️  WARNING: Model outputs exceed expected range")
        print("   This may indicate normalization issues")
    else:
        print("\n✓ Output range looks reasonable")

    # Print corrected results
    print("\n" + "="*70)
    print("📊 CORRECTED TEST SET RESULTS (REAL CT DATA)")
    print("="*70)
    print(f"{'Metric':<20} {'FBP':>12} {'PINN-DADif':>15} {'Improvement':>15}")
    print("-"*70)

    for k in ['psnr', 'ssim', 'rmse', 'mae']:
        fbp_val = np.mean(fbp_metrics[k])
        model_val = np.mean(test_metrics[k])

        if k in ['psnr', 'ssim']:
            diff = model_val - fbp_val
            print(f'{k.upper():<20} {fbp_val:>12.2f} {model_val:>15.2f} {diff:>+15.2f}')
        else:
            diff_pct = (1 - model_val/fbp_val) * 100 if fbp_val > 0 else 0
            print(f'{k.upper():<20} {fbp_val:>12.4f} {model_val:>15.4f} {diff_pct:>+14.1f}%')

    print("="*70)

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    avg_psnr = np.mean(test_metrics['psnr'])
    avg_ssim = np.mean(test_metrics['ssim'])

    if avg_psnr < 10:
        print("\n❌ VERY POOR RESULTS (PSNR < 10 dB)")
        print("   Likely causes:")
        print("   1. Insufficient training (only 5 epochs - need 50-200)")
        print("   2. Learning rate too low or optimizer not converging")
        print("   3. Model architecture issues")
        print("\n   RECOMMENDATION: Train for at least 50 epochs")

    elif avg_psnr < 20:
        print("\n⚠️  POOR RESULTS (PSNR 10-20 dB)")
        print("   Results are below FBP baseline quality")
        print("   RECOMMENDATION: Continue training for more epochs")

    elif avg_psnr < 25:
        print("\n⚠️  BELOW FBP BASELINE (PSNR 20-25 dB)")
        print("   Model is learning but not yet competitive with FBP")
        print("   RECOMMENDATION: Continue training")

    elif avg_psnr < 28:
        print("\n✓ MODERATE RESULTS (PSNR 25-28 dB)")
        print("   Model performs similar to FBP baseline")

    else:
        print("\n✅ GOOD RESULTS (PSNR > 28 dB)")
        print("   Model outperforms FBP baseline significantly")

    if avg_ssim > 100:
        print("\n❌ ERROR: SSIM > 100% (This should be fixed now)")
    elif avg_ssim < 10:
        print(f"\n⚠️  WARNING: Very low SSIM ({avg_ssim:.1f}%)")
        print("   Structural similarity is poor")

    # Create visualization
    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)

    # Get one batch for visualization
    batch = next(iter(test_loader))
    sinogram = batch['sinogram_noisy'].to(device)
    target = batch['image'].to(device)
    weights = batch['weights'].to(device)
    mask = batch['mask'].to(device)
    fbp_recon = batch['fbp'].to(device)

    with torch.no_grad():
        outputs = model(sinogram, weights, mask)
        pred = outputs['reconstruction']

    n = min(4, len(target))
    fig, axes = plt.subplots(4, n, figsize=(4*n, 16))

    for i in range(n):
        # Ground truth
        gt_img = target[i, 0].cpu().numpy()
        axes[0, i].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Ground Truth', fontsize=14, rotation=90, labelpad=20)

        # FBP
        fbp_img = fbp_recon[i, 0].cpu().numpy()
        m_fbp = compute_metrics(fbp_recon[i:i+1], target[i:i+1])
        axes[1, i].imshow(fbp_img, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'FBP: {m_fbp["psnr"]:.1f} dB', fontsize=10)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('FBP', fontsize=14, rotation=90, labelpad=20)

        # PINN-DADif
        pred_img = pred[i, 0].cpu().numpy()
        m_pred = compute_metrics(pred[i:i+1], target[i:i+1])
        axes[2, i].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f'PINN-DADif: {m_pred["psnr"]:.1f} dB', fontsize=10)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('PINN-DADif', fontsize=14, rotation=90, labelpad=20)

        # Error map
        error = np.abs(pred_img - gt_img)
        im = axes[3, i].imshow(error, cmap='hot', vmin=0, vmax=0.3)
        axes[3, i].set_title(f'MAE: {m_pred["mae"]:.4f}', fontsize=10)
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_ylabel('Error Map', fontsize=14, rotation=90, labelpad=20)

    plt.suptitle('CT Reconstruction Comparison (Corrected Visualization)', fontsize=16)
    plt.tight_layout()

    output_path = 'diagnostic_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()

    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)

    return {
        'test_metrics': test_metrics,
        'fbp_metrics': fbp_metrics,
        'output_range': {
            'min': min(model_outputs_min),
            'max': max(model_outputs_max),
            'mean': np.mean(model_outputs_mean)
        }
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose CT reconstruction test results')
    parser.add_argument('--checkpoint', type=str, default='experiments/checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    args = parser.parse_args()

    try:
        results = diagnose_model(args.checkpoint)
        print("\n✅ Diagnosis successful!")
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find checkpoint file: {args.checkpoint}")
        print("   Please specify the correct path using --checkpoint")
    except Exception as e:
        print(f"\n❌ Error during diagnosis: {str(e)}")
        import traceback
        traceback.print_exc()
