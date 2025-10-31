"""
PINN-DADif Inference Pipeline
Complete inference, evaluation, and visualization implementation

Paper: Ahmed et al. (2025) - Digital Signal Processing
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import json
from datetime import datetime

from pinn_dadif import PINNDADif
from train import MRIDataset, calculate_psnr, calculate_ssim


# ==================== Inference Engine ====================

class PINNDADifInference:
    """
    Complete inference pipeline for PINN-DADif
    Handles testing, evaluation, and visualization
    """
    
    def __init__(self,
                 model: PINNDADif,
                 checkpoint_path: str,
                 device: str = 'cuda',
                 save_reconstructions: bool = True,
                 save_intermediates: bool = False):
        """
        Args:
            model: PINN-DADif model
            checkpoint_path: Path to trained checkpoint
            device: Device for inference
            save_reconstructions: Whether to save reconstructed images
            save_intermediates: Whether to save intermediate features
        """
        self.model = model.to(device)
        self.device = device
        self.save_reconstructions = save_reconstructions
        self.save_intermediates = save_intermediates
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_path)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Statistics
        self.inference_times = []
        
        print(f"✓ Inference engine initialized")
        print(f"  Device: {device}")
        print(f"  Checkpoint: {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load trained model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.best_psnr = checkpoint.get('best_psnr', 0)
        self.best_ssim = checkpoint.get('best_ssim', 0)
        
        print(f"  Loaded from epoch {self.epoch}")
        print(f"  Best training PSNR: {self.best_psnr:.2f} dB")
        print(f"  Best training SSIM: {self.best_ssim:.4f}")
    
    @torch.no_grad()
    def reconstruct_single(self,
                          x_under: torch.Tensor,
                          k_space: Optional[torch.Tensor] = None,
                          mask: Optional[torch.Tensor] = None,
                          sequence_type: str = 'mixed',
                          return_time: bool = False) -> Dict:
        """
        Reconstruct a single image
        
        Args:
            x_under: Undersampled image (1, 1, H, W)
            k_space: K-space data (1, H, W) complex
            mask: Undersampling mask (1, 1, H, W)
            sequence_type: MRI sequence type
            return_time: Whether to return inference time
        
        Returns:
            Dictionary with reconstruction and optional intermediates
        """
        x_under = x_under.to(self.device)
        if k_space is not None:
            k_space = k_space.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        
        # Measure inference time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Inference
        if self.save_intermediates:
            x_recon, intermediates = self.model(
                x_under,
                k_space=k_space,
                mask=mask,
                sequence_type=sequence_type,
                return_intermediates=True
            )
        else:
            x_recon = self.model.reconstruct(
                x_under,
                k_space=k_space,
                mask=mask,
                sequence_type=sequence_type
            )
            intermediates = None
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            self.inference_times.append(inference_time)
        else:
            inference_time = None
        
        result = {
            'reconstruction': x_recon.cpu(),
            'intermediates': {k: v.cpu() for k, v in intermediates.items()} if intermediates else None,
            'inference_time': inference_time
        }
        
        return result
    
    @torch.no_grad()
    def evaluate_dataset(self,
                        test_loader,
                        output_dir: str,
                        save_images: bool = True,
                        save_metrics: bool = True) -> Dict:
        """
        Evaluate on complete test dataset
        
        Args:
            test_loader: DataLoader for test data
            output_dir: Directory to save results
            save_images: Whether to save reconstructed images
            save_metrics: Whether to save metrics per image
        
        Returns:
            Dictionary with aggregate metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        if save_images:
            recon_dir = output_dir / 'reconstructions'
            recon_dir.mkdir(exist_ok=True)
        
        # Metrics storage
        all_metrics = {
            'psnr': [],
            'ssim': [],
            'inference_time': [],
            'per_sequence': {}
        }
        
        per_image_metrics = []
        
        print(f"\n{'='*70}")
        print(f"Evaluating on {len(test_loader.dataset)} test samples")
        print(f"{'='*70}\n")
        
        pbar = tqdm(test_loader, desc="Inference")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            x_under = batch['undersampled']
            x_target = batch['target']
            k_space = batch['k_space']
            mask = batch['mask']
            seq_type = batch['sequence'][0] if len(batch['sequence']) == 1 else 'mixed'
            filename = batch['filename'][0]
            
            # Reconstruct
            result = self.reconstruct_single(
                x_under,
                k_space=k_space,
                mask=mask,
                sequence_type=seq_type,
                return_time=True
            )
            
            x_recon = result['reconstruction']
            inference_time = result['inference_time']
            
            # Compute metrics
            psnr = calculate_psnr(x_recon, x_target)
            ssim = calculate_ssim(x_recon, x_target)
            
            # Store metrics
            all_metrics['psnr'].append(psnr)
            all_metrics['ssim'].append(ssim)
            if inference_time:
                all_metrics['inference_time'].append(inference_time)
            
            # Per-sequence metrics
            if seq_type not in all_metrics['per_sequence']:
                all_metrics['per_sequence'][seq_type] = {'psnr': [], 'ssim': []}
            all_metrics['per_sequence'][seq_type]['psnr'].append(psnr)
            all_metrics['per_sequence'][seq_type]['ssim'].append(ssim)
            
            # Per-image metrics
            per_image_metrics.append({
                'filename': filename,
                'sequence': seq_type,
                'psnr': psnr,
                'ssim': ssim,
                'inference_time': inference_time
            })
            
            # Save reconstruction
            if save_images:
                self._save_reconstruction(
                    x_under.squeeze().numpy(),
                    x_recon.squeeze().numpy(),
                    x_target.squeeze().numpy(),
                    recon_dir / f"{Path(filename).stem}_{seq_type}.png",
                    psnr,
                    ssim
                )
            
            # Update progress bar
            pbar.set_postfix({
                'PSNR': f'{psnr:.2f}',
                'SSIM': f'{ssim:.4f}',
                'Time': f'{inference_time:.3f}s' if inference_time else 'N/A'
            })
        
        # Compute aggregate statistics
        aggregate_metrics = {
            'mean_psnr': np.mean(all_metrics['psnr']),
            'std_psnr': np.std(all_metrics['psnr']),
            'mean_ssim': np.mean(all_metrics['ssim']),
            'std_ssim': np.std(all_metrics['ssim']),
            'mean_inference_time': np.mean(all_metrics['inference_time']) if all_metrics['inference_time'] else None,
            'per_sequence': {}
        }
        
        # Per-sequence statistics
        for seq_type, metrics in all_metrics['per_sequence'].items():
            aggregate_metrics['per_sequence'][seq_type] = {
                'mean_psnr': np.mean(metrics['psnr']),
                'std_psnr': np.std(metrics['psnr']),
                'mean_ssim': np.mean(metrics['ssim']),
                'std_ssim': np.std(metrics['ssim']),
                'num_samples': len(metrics['psnr'])
            }
        
        # Save metrics
        if save_metrics:
            # Aggregate metrics
            with open(output_dir / 'aggregate_metrics.json', 'w') as f:
                json.dump(aggregate_metrics, f, indent=2)
            
            # Per-image metrics
            with open(output_dir / 'per_image_metrics.json', 'w') as f:
                json.dump(per_image_metrics, f, indent=2)
        
        # Print summary
        self._print_summary(aggregate_metrics)
        
        return aggregate_metrics
    
    def _save_reconstruction(self,
                            x_under: np.ndarray,
                            x_recon: np.ndarray,
                            x_target: np.ndarray,
                            save_path: Path,
                            psnr: float,
                            ssim: float):
        """Save reconstruction visualization"""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Undersampled
        axes[0].imshow(x_under, cmap='gray')
        axes[0].set_title('Undersampled')
        axes[0].axis('off')
        
        # Reconstruction
        axes[1].imshow(x_recon, cmap='gray')
        axes[1].set_title(f'PINN-DADif\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}')
        axes[1].axis('off')
        
        # Ground Truth
        axes[2].imshow(x_target, cmap='gray')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        
        # Error Map
        error = np.abs(x_recon - x_target)
        im = axes[3].imshow(error, cmap='hot')
        axes[3].set_title('Error Map')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _print_summary(self, metrics: Dict):
        """Print evaluation summary"""
        print(f"\n{'='*70}")
        print("Evaluation Summary")
        print(f"{'='*70}")
        print(f"\nOverall Performance:")
        print(f"  PSNR: {metrics['mean_psnr']:.2f} ± {metrics['std_psnr']:.2f} dB")
        print(f"  SSIM: {metrics['mean_ssim']:.4f} ± {metrics['std_ssim']:.4f}")
        if metrics['mean_inference_time']:
            print(f"  Inference Time: {metrics['mean_inference_time']:.3f} s/image")
        
        print(f"\nPer-Sequence Performance:")
        for seq_type, seq_metrics in metrics['per_sequence'].items():
            print(f"  {seq_type.upper()} ({seq_metrics['num_samples']} samples):")
            print(f"    PSNR: {seq_metrics['mean_psnr']:.2f} ± {seq_metrics['std_psnr']:.2f} dB")
            print(f"    SSIM: {seq_metrics['mean_ssim']:.4f} ± {seq_metrics['std_ssim']:.4f}")
        
        print(f"{'='*70}\n")
    
    def compare_with_baselines(self,
                              test_loader,
                              baseline_methods: Dict[str, callable],
                              output_dir: str):
        """
        Compare PINN-DADif with baseline methods
        
        Args:
            test_loader: Test data loader
            baseline_methods: Dict of {method_name: reconstruction_function}
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for all methods
        all_results = {
            'PINN-DADif': {'psnr': [], 'ssim': [], 'time': []},
        }
        
        for method_name in baseline_methods.keys():
            all_results[method_name] = {'psnr': [], 'ssim': [], 'time': []}
        
        print(f"\n{'='*70}")
        print(f"Comparing with {len(baseline_methods)} baseline methods")
        print(f"{'='*70}\n")
        
        pbar = tqdm(test_loader, desc="Comparison")
        
        for batch_idx, batch in enumerate(pbar):
            x_under = batch['undersampled']
            x_target = batch['target']
            k_space = batch['k_space']
            mask = batch['mask']
            seq_type = batch['sequence'][0]
            
            # PINN-DADif
            result = self.reconstruct_single(
                x_under, k_space, mask, seq_type, return_time=True
            )
            x_recon_pinn = result['reconstruction']
            
            psnr = calculate_psnr(x_recon_pinn, x_target)
            ssim = calculate_ssim(x_recon_pinn, x_target)
            
            all_results['PINN-DADif']['psnr'].append(psnr)
            all_results['PINN-DADif']['ssim'].append(ssim)
            all_results['PINN-DADif']['time'].append(result['inference_time'])
            
            # Baseline methods
            for method_name, method_func in baseline_methods.items():
                x_recon_base = method_func(x_under, k_space, mask)
                
                psnr = calculate_psnr(x_recon_base, x_target)
                ssim = calculate_ssim(x_recon_base, x_target)
                
                all_results[method_name]['psnr'].append(psnr)
                all_results[method_name]['ssim'].append(ssim)
        
        # Compute statistics
        comparison_stats = {}
        for method_name, results in all_results.items():
            comparison_stats[method_name] = {
                'mean_psnr': np.mean(results['psnr']),
                'std_psnr': np.std(results['psnr']),
                'mean_ssim': np.mean(results['ssim']),
                'std_ssim': np.std(results['ssim']),
                'mean_time': np.mean(results['time']) if results['time'] and results['time'][0] else None
            }
        
        # Save comparison
        with open(output_dir / 'comparison_results.json', 'w') as f:
            json.dump(comparison_stats, f, indent=2)
        
        # Create comparison plots
        self._plot_comparison(comparison_stats, output_dir)
        
        # Print comparison
        print(f"\n{'='*70}")
        print("Comparison Results")
        print(f"{'='*70}")
        for method_name, stats in comparison_stats.items():
            print(f"\n{method_name}:")
            print(f"  PSNR: {stats['mean_psnr']:.2f} ± {stats['std_psnr']:.2f} dB")
            print(f"  SSIM: {stats['mean_ssim']:.4f} ± {stats['std_ssim']:.4f}")
            if stats['mean_time']:
                print(f"  Time: {stats['mean_time']:.3f} s")
        print(f"{'='*70}\n")
        
        return comparison_stats
    
    def _plot_comparison(self, stats: Dict, output_dir: Path):
        """Create comparison plots"""
        methods = list(stats.keys())
        psnr_means = [stats[m]['mean_psnr'] for m in methods]
        psnr_stds = [stats[m]['std_psnr'] for m in methods]
        ssim_means = [stats[m]['mean_ssim'] for m in methods]
        ssim_stds = [stats[m]['std_ssim'] for m in methods]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # PSNR comparison
        x_pos = np.arange(len(methods))
        axes[0].bar(x_pos, psnr_means, yerr=psnr_stds, capsize=5, alpha=0.7)
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('PSNR Comparison')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0].grid(axis='y', alpha=0.3)
        
        # SSIM comparison
        axes[1].bar(x_pos, ssim_means, yerr=ssim_stds, capsize=5, alpha=0.7, color='orange')
        axes[1].set_ylabel('SSIM')
        axes[1].set_title('SSIM Comparison')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(methods, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'comparison_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_intermediate_features(self,
                                       x_under: torch.Tensor,
                                       k_space: torch.Tensor,
                                       mask: torch.Tensor,
                                       sequence_type: str,
                                       output_dir: str):
        """
        Visualize intermediate features from all modules
        
        Args:
            x_under: Undersampled image
            k_space: K-space data
            mask: Undersampling mask
            sequence_type: Sequence type
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get reconstruction with intermediates
        result = self.reconstruct_single(
            x_under.unsqueeze(0),
            k_space.unsqueeze(0),
            mask.unsqueeze(0),
            sequence_type,
            return_time=False
        )
        
        if result['intermediates'] is None:
            print("Warning: Intermediates not available. Set save_intermediates=True")
            return
        
        intermediates = result['intermediates']
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Input
        axes[0, 0].imshow(x_under.squeeze().numpy(), cmap='gray')
        axes[0, 0].set_title('Input (Undersampled)')
        axes[0, 0].axis('off')
        
        # LPCE output
        z_latent = intermediates['z_latent'].squeeze()
        # Show first channel
        axes[0, 1].imshow(z_latent[0].numpy(), cmap='viridis')
        axes[0, 1].set_title(f'LPCE Output\n{z_latent.shape[0]} channels')
        axes[0, 1].axis('off')
        
        # PACE output
        z_pace = intermediates['z_pace'].squeeze()
        axes[0, 2].imshow(z_pace[0].numpy(), cmap='viridis')
        axes[0, 2].set_title(f'PACE Output\n{z_pace.shape[0]} channels')
        axes[0, 2].axis('off')
        
        # ADRN output
        z_adrn = intermediates['z_adrn'].squeeze()
        axes[1, 0].imshow(z_adrn[0].numpy(), cmap='viridis')
        axes[1, 0].set_title(f'ADRN Output\n{z_adrn.shape[0]} channels')
        axes[1, 0].axis('off')
        
        # Final reconstruction
        x_recon = result['reconstruction'].squeeze().numpy()
        axes[1, 1].imshow(x_recon, cmap='gray')
        axes[1, 1].set_title('Final Reconstruction')
        axes[1, 1].axis('off')
        
        # Feature statistics
        axes[1, 2].axis('off')
        stats_text = f"Feature Statistics:\n\n"
        stats_text += f"LPCE: {z_latent.shape}\n"
        stats_text += f"  Mean: {z_latent.mean():.4f}\n"
        stats_text += f"  Std: {z_latent.std():.4f}\n\n"
        stats_text += f"PACE: {z_pace.shape}\n"
        stats_text += f"  Mean: {z_pace.mean():.4f}\n"
        stats_text += f"  Std: {z_pace.std():.4f}\n\n"
        stats_text += f"ADRN: {z_adrn.shape}\n"
        stats_text += f"  Mean: {z_adrn.mean():.4f}\n"
        stats_text += f"  Std: {z_adrn.std():.4f}"
        
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, 
                       verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'intermediates_{sequence_type}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved intermediate visualizations to {output_dir}")


# ==================== Main Inference Script ====================

def main(args):
    """Main inference function"""
    
    print(f"\n{'='*70}")
    print("PINN-DADif Inference")
    print(f"{'='*70}\n")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model
    print("\nInitializing model...")
    model = PINNDADif(
        in_channels=1,
        out_channels=1,
        lpce_latent_dim=128,
        pace_out_channels=256,
        adrn_num_diffusion_steps=10,
        adrn_num_reverse_iterations=12,
        lambda_phys=0.2,
        device=device
    )
    
    # Create inference engine
    inference = PINNDADifInference(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
        save_reconstructions=args.save_images,
        save_intermediates=args.save_intermediates
    )
    
    # Create test dataset
    print("\nLoading test dataset...")
    test_dataset = MRIDataset(
        data_path=args.data_path,
        mode='test',
        acceleration_factor=args.acceleration,
        sequence_types=args.sequences,
        image_size=(args.image_size, args.image_size),
        normalize=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # Inference one at a time
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"inference_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Run evaluation
    print("\nRunning evaluation...")
    metrics = inference.evaluate_dataset(
        test_loader=test_loader,
        output_dir=output_dir,
        save_images=args.save_images,
        save_metrics=True
    )
    
    # Visualize intermediates (if requested)
    if args.save_intermediates and len(test_dataset) > 0:
        print("\nGenerating intermediate visualizations...")
        sample = test_dataset[0]
        inference.visualize_intermediate_features(
            sample['undersampled'],
            sample['k_space'],
            sample['mask'],
            sample['sequence'],
            output_dir / 'intermediates'
        )
    
    # Save configuration
    config = {
        'checkpoint': args.checkpoint,
        'acceleration': args.acceleration,
        'sequences': args.sequences,
        'num_samples': len(test_dataset),
        'metrics': metrics,
        'timestamp': timestamp
    }
    
    with open(output_dir / 'inference_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Inference complete!")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINN-DADif Inference')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    
    # Data
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Output directory')
    parser.add_argument('--sequences', nargs='+', default=['t1', 't2', 'pd'],
                       help='MRI sequences to test')
    parser.add_argument('--acceleration', type=int, default=4,
                       help='Acceleration factor (4, 8, or 12)')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size')
    
    # Options
    parser.add_argument('--save_images', action='store_true',
                       help='Save reconstructed images')
    parser.add_argument('--save_intermediates', action='store_true',
                       help='Save intermediate features')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    main(args)