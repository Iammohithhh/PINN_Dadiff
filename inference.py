import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from tqdm import tqdm
import argparse

# Assuming model and dataset classes are imported
# from training import PINNDADif, MRIDataset, compute_psnr, compute_ssim


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with saved config
    config = checkpoint['config']
    model = PINNDADif(
        lambda_phys=config.get('lambda_phys', 0.2),
        num_diffusion_steps=config.get('num_diffusion_steps', 10)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Best PSNR: {checkpoint.get('psnr', 'N/A'):.2f} dB")
    
    return model, config


def reconstruct_image(model, kspace_under, mask, sequence_type='mixed', device='cuda'):
    """
    Reconstruct MRI image from undersampled k-space.
    
    Args:
        model: Trained PINN-DADif model
        kspace_under: Undersampled k-space (2, H, W)
        mask: Sampling mask (1, H, W)
        sequence_type: MRI sequence type
        device: Computing device
    
    Returns:
        reconstructed: Reconstructed image (H, W)
    """
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension
        kspace_under = kspace_under.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        
        # Reconstruct
        output = model(
            kspace_under, 
            sequence_type=sequence_type,
            k_measured=kspace_under,
            mask=mask,
            mode='inference'
        )
        
        # Remove batch and channel dimensions
        reconstructed = output.squeeze().cpu()
    
    return reconstructed


def evaluate_reconstruction(pred, target):
    """Compute reconstruction metrics"""
    # Ensure same shape
    if pred.shape != target.shape:
        pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), 
                            size=target.shape, mode='bilinear').squeeze()
    
    # PSNR
    psnr = compute_psnr(pred.unsqueeze(0).unsqueeze(0), 
                       target.unsqueeze(0).unsqueeze(0))
    
    # SSIM
    ssim = compute_ssim(pred.unsqueeze(0).unsqueeze(0), 
                       target.unsqueeze(0).unsqueeze(0))
    
    # NMSE (Normalized Mean Squared Error)
    nmse = torch.mean((pred - target) ** 2) / torch.mean(target ** 2)
    
    return {
        'psnr': psnr.item(),
        'ssim': ssim.item(),
        'nmse': nmse.item()
    }


def visualize_reconstruction(kspace_under, mask, reconstructed, ground_truth, 
                            metrics, save_path=None):
    """
    Visualize reconstruction results.
    
    Creates a figure with:
    - Undersampled reconstruction (zero-filled)
    - PINN-DADif reconstruction
    - Ground truth
    - Error maps
    - K-space mask
    """
    # Zero-filled reconstruction (baseline)
    kspace_complex = torch.complex(kspace_under[0], kspace_under[1])
    zero_filled = torch.fft.ifft2(kspace_complex, norm='ortho').abs()
    zero_filled = (zero_filled - zero_filled.min()) / (zero_filled.max() - zero_filled.min())
    
    # Error maps
    error_zero_filled = torch.abs(zero_filled - ground_truth)
    error_pinn = torch.abs(reconstructed - ground_truth)
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Images
    im1 = axes[0, 0].imshow(zero_filled.cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Zero-Filled Recon')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(reconstructed.cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'PINN-DADif Recon\nPSNR: {metrics["psnr"]:.2f} dB')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    im3 = axes[0, 2].imshow(ground_truth.cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Ground Truth')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    im4 = axes[0, 3].imshow(mask.squeeze().cpu(), cmap='gray')
    axes[0, 3].set_title('K-space Mask')
    axes[0, 3].axis('off')
    plt.colorbar(im4, ax=axes[0, 3], fraction=0.046)
    
    # Row 2: Error maps and ROI
    error_max = max(error_zero_filled.max(), error_pinn.max())
    
    im5 = axes[1, 0].imshow(error_zero_filled.cpu(), cmap='hot', vmin=0, vmax=error_max)
    axes[1, 0].set_title('Zero-Filled Error')
    axes[1, 0].axis('off')
    plt.colorbar(im5, ax=axes[1, 0], fraction=0.046)
    
    im6 = axes[1, 1].imshow(error_pinn.cpu(), cmap='hot', vmin=0, vmax=error_max)
    axes[1, 1].set_title(f'PINN-DADif Error\nSSIM: {metrics["ssim"]:.2f}%')
    axes[1, 1].axis('off')
    plt.colorbar(im6, ax=axes[1, 1], fraction=0.046)
    
    # ROI comparison (center crop)
    h, w = ground_truth.shape
    roi_size = min(h, w) // 4
    roi_h = slice(h//2 - roi_size//2, h//2 + roi_size//2)
    roi_w = slice(w//2 - roi_size//2, w//2 + roi_size//2)
    
    im7 = axes[1, 2].imshow(reconstructed[roi_h, roi_w].cpu(), cmap='gray')
    axes[1, 2].set_title('PINN-DADif ROI')
    axes[1, 2].axis('off')
    plt.colorbar(im7, ax=axes[1, 2], fraction=0.046)
    
    im8 = axes[1, 3].imshow(ground_truth[roi_h, roi_w].cpu(), cmap='gray')
    axes[1, 3].set_title('Ground Truth ROI')
    axes[1, 3].axis('off')
    plt.colorbar(im8, ax=axes[1, 3], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    plt.show()
    
    return fig


def batch_inference(model, test_dataset, device='cuda', save_dir='results'):
    """
    Run inference on entire test dataset and compute statistics.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    results = {
        'psnr': [],
        'ssim': [],
        'nmse': []
    }
    
    model.eval()
    
    print("\nRunning inference on test dataset...")
    for idx in tqdm(range(len(test_dataset))):
        # Load sample
        sample = test_dataset[idx]
        kspace_under = sample['kspace_under']
        kspace_full = sample['kspace_full']
        image_gt = sample['image_gt'].squeeze()
        mask = sample['mask']
        sequence = sample['sequence']
        
        # Reconstruct
        reconstructed = reconstruct_image(
            model, kspace_under, mask, sequence, device
        )
        
        # Evaluate
        metrics = evaluate_reconstruction(reconstructed, image_gt)
        
        # Store results
        for key in results:
            results[key].append(metrics[key])
        
        # Save visualization for first 10 samples
        if idx < 10:
            fig = visualize_reconstruction(
                kspace_under, mask, reconstructed, image_gt, metrics,
                save_path=save_dir / f'reconstruction_{idx:03d}.png'
            )
            plt.close(fig)
    
    # Compute statistics
    print("\n" + "="*50)
    print("RECONSTRUCTION RESULTS")
    print("="*50)
    for metric_name, values in results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric_name.upper():8s}: {mean_val:7.3f} ± {std_val:6.3f}")
    print("="*50)
    
    # Save results to file
    np.save(save_dir / 'metrics.npy', results)
    
    return results


def compare_acceleration_rates(model, sample, device='cuda', 
                               rates=[4, 8, 12], save_path='comparison.png'):
    """
    Compare reconstruction quality at different acceleration rates.
    """
    kspace_full = sample['kspace_full']
    image_gt = sample['image_gt'].squeeze()
    sequence = sample['sequence']
    
    fig, axes = plt.subplots(2, len(rates) + 1, figsize=(5*(len(rates)+1), 10))
    
    # Ground truth
    axes[0, 0].imshow(image_gt.cpu(), cmap='gray')
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Reconstruct at different rates
    for i, rate in enumerate(rates):
        # Create mask for this acceleration rate
        h, w = kspace_full.shape[1:]
        mask = create_variable_density_mask(h, w, rate)
        
        # Undersample
        kspace_under = kspace_full * mask
        
        # Reconstruct
        reconstructed = reconstruct_image(
            model, kspace_under, mask, sequence, device
        )
        
        # Evaluate
        metrics = evaluate_reconstruction(reconstructed, image_gt)
        
        # Plot reconstruction
        axes[0, i+1].imshow(reconstructed.cpu(), cmap='gray')
        axes[0, i+1].set_title(f'R={rate}x\nPSNR: {metrics["psnr"]:.2f} dB')
        axes[0, i+1].axis('off')
        
        # Plot error
        error = torch.abs(reconstructed - image_gt)
        axes[1, i+1].imshow(error.cpu(), cmap='hot')
        axes[1, i+1].set_title(f'SSIM: {metrics["ssim"]:.2f}%')
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved acceleration comparison to {save_path}")
    plt.show()


def create_variable_density_mask(h, w, acceleration):
    """Create variable-density undersampling mask"""
    center_fraction = 0.08
    mask = torch.zeros(1, h, w)
    
    # Fully sample center
    center_h = int(h * center_fraction)
    mask[:, h//2 - center_h//2:h//2 + center_h//2, :] = 1
    
    # Random sampling for rest
    prob = (1 - center_fraction) / acceleration
    random_mask = torch.rand(1, h, w) < prob
    mask = torch.logical_or(mask, random_mask).float()
    
    return mask


def main():
    """Main inference script"""
    parser = argparse.ArgumentParser(description='PINN-DADif Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--acceleration', type=int, default=4,
                       help='Acceleration rate')
    parser.add_argument('--batch_inference', action='store_true',
                       help='Run batch inference on entire dataset')
    parser.add_argument('--compare_rates', action='store_true',
                       help='Compare different acceleration rates')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Computing device')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    
    # Create test dataset
    print(f"\nLoading test data from {args.test_data}...")
    test_dataset = MRIDataset(
        args.test_data,
        acceleration=args.acceleration,
        sequence_type='mixed'
    )
    print(f"✓ Loaded {len(test_dataset)} test samples")
    
    if args.batch_inference:
        # Run batch inference
        results = batch_inference(model, test_dataset, device, args.output_dir)
    
    elif args.compare_rates:
        # Compare acceleration rates
        sample = test_dataset[0]
        compare_acceleration_rates(
            model, sample, device,
            rates=[4, 8, 12],
            save_path=Path(args.output_dir) / 'acceleration_comparison.png'
        )
    
    else:
        # Single sample inference
        sample = test_dataset[0]
        
        print("\nReconstructing single sample...")
        reconstructed = reconstruct_image(
            model,
            sample['kspace_under'],
            sample['mask'],
            sample['sequence'],
            device
        )
        
        # Evaluate
        metrics = evaluate_reconstruction(reconstructed, sample['image_gt'].squeeze())
        
        # Visualize
        visualize_reconstruction(
            sample['kspace_under'],
            sample['mask'],
            reconstructed,
            sample['image_gt'].squeeze(),
            metrics,
            save_path=Path(args.output_dir) / 'reconstruction.png'
        )
        
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key.upper()}: {value:.3f}")


if __name__ == "__main__":
    main()