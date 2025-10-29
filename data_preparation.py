import torch
import numpy as np
import h5py
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
import argparse


def normalize_image(image):
    """Normalize image to [0, 1] range"""
    image = image - image.min()
    image = image / (image.max() + 1e-8)
    return image


def convert_nifti_to_h5(nifti_path, output_path, sequence_type='t1'):
    """
    Convert NIfTI MRI file to HDF5 format with k-space data.
    
    Args:
        nifti_path: Path to .nii or .nii.gz file
        output_path: Path to save .h5 file
        sequence_type: 't1', 't2', or 'pd'
    """
    # Load NIfTI file
    nii = nib.load(nifti_path)
    image = nii.get_fdata()
    
    # Handle 3D volumes - process each slice
    if image.ndim == 3:
        slices = []
        for i in range(image.shape[2]):
            slice_2d = image[:, :, i]
            
            # Normalize
            slice_2d = normalize_image(slice_2d)
            
            # Convert to k-space
            kspace = np.fft.fft2(slice_2d)
            kspace = np.fft.fftshift(kspace)
            
            slices.append(kspace)
        
        # Save all slices to single file
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('kspace', data=np.array(slices), 
                           compression='gzip')
            f.attrs['sequence'] = sequence_type
            f.attrs['num_slices'] = len(slices)
    
    elif image.ndim == 2:
        # Single 2D image
        image = normalize_image(image)
        kspace = np.fft.fft2(image)
        kspace = np.fft.fftshift(kspace)
        
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('kspace', data=kspace, compression='gzip')
            f.attrs['sequence'] = sequence_type
    
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")
    
    print(f"✓ Converted {nifti_path.name} to {output_path.name}")


def convert_dicom_to_h5(dicom_dir, output_path, sequence_type='t1'):
    """
    Convert DICOM series to HDF5 format.
    Requires pydicom: pip install pydicom
    """
    try:
        import pydicom
    except ImportError:
        print("Error: pydicom not installed. Install with: pip install pydicom")
        return
    
    # Read DICOM files
    dicom_files = sorted(Path(dicom_dir).glob('*.dcm'))
    
    if not dicom_files:
        print(f"No DICOM files found in {dicom_dir}")
        return
    
    slices = []
    for dcm_file in dicom_files:
        ds = pydicom.dcmread(dcm_file)
        image = ds.pixel_array.astype(float)
        
        # Normalize
        image = normalize_image(image)
        
        # Convert to k-space
        kspace = np.fft.fft2(image)
        kspace = np.fft.fftshift(kspace)
        
        slices.append(kspace)
    
    # Save to HDF5
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('kspace', data=np.array(slices), compression='gzip')
        f.attrs['sequence'] = sequence_type
        f.attrs['num_slices'] = len(slices)
    
    print(f"✓ Converted {len(slices)} DICOM slices to {output_path.name}")


def prepare_fastmri_data(fastmri_path, output_dir, split='train', 
                        num_samples=None):
    """
    Prepare fastMRI dataset for training.
    
    Args:
        fastmri_path: Path to fastMRI dataset
        output_dir: Output directory
        split: 'train', 'val', or 'test'
        num_samples: Number of samples to process (None for all)
    """
    fastmri_path = Path(fastmri_path)
    output_dir = Path(output_dir) / split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all h5 files
    h5_files = sorted(list(fastmri_path.glob(f'{split}/**/*.h5')))
    
    if num_samples:
        h5_files = h5_files[:num_samples]
    
    print(f"\nProcessing {len(h5_files)} files from {split} set...")
    
    for idx, h5_file in enumerate(tqdm(h5_files)):
        try:
            with h5py.File(h5_file, 'r') as f:
                # Load k-space data
                kspace = f['kspace'][()]
                
                # Get attributes
                attrs = dict(f.attrs)
                
                # Process each slice
                for slice_idx in range(kspace.shape[0]):
                    kspace_slice = kspace[slice_idx]
                    
                    # Save to new file
                    output_file = output_dir / f'{idx:05d}_slice_{slice_idx:03d}.h5'
                    
                    with h5py.File(output_file, 'w') as out_f:
                        out_f.create_dataset('kspace', data=kspace_slice,
                                           compression='gzip')
                        
                        # Copy attributes
                        for key, value in attrs.items():
                            out_f.attrs[key] = value
        
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")
            continue
    
    print(f"✓ Processed {len(h5_files)} files")
    print(f"✓ Output saved to {output_dir}")


def create_synthetic_mri_data(output_dir, num_samples=100, image_size=256):
    """
    Create synthetic MRI data for testing.
    Generates simple phantoms with different tissue contrasts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {num_samples} synthetic MRI samples...")
    
    for idx in tqdm(range(num_samples)):
        # Create phantom
        image = create_shepp_logan_phantom(image_size)
        
        # Add noise
        noise = np.random.randn(*image.shape) * 0.02
        image = image + noise
        image = np.clip(image, 0, 1)
        
        # Convert to k-space
        kspace = np.fft.fft2(image)
        kspace = np.fft.fftshift(kspace)
        
        # Determine sequence type (vary for diversity)
        sequence_types = ['t1', 't2', 'pd']
        sequence = sequence_types[idx % len(sequence_types)]
        
        # Save
        output_file = output_dir / f'synthetic_{idx:05d}.h5'
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('kspace', data=kspace, compression='gzip')
            f.attrs['sequence'] = sequence
            f.attrs['synthetic'] = True
    
    print(f"✓ Generated {num_samples} synthetic samples")
    print(f"✓ Saved to {output_dir}")


def create_shepp_logan_phantom(size=256):
    """
    Create Shepp-Logan phantom - standard test image for MRI.
    """
    # Define ellipses: [A, a, b, x0, y0, phi]
    ellipses = [
        [1.0, 0.69, 0.92, 0, 0, 0],
        [-0.8, 0.6624, 0.874, 0, -0.0184, 0],
        [-0.2, 0.11, 0.31, 0.22, 0, -18],
        [-0.2, 0.16, 0.41, -0.22, 0, 18],
        [0.1, 0.21, 0.25, 0, 0.35, 0],
        [0.1, 0.046, 0.046, 0, 0.1, 0],
        [0.1, 0.046, 0.046, 0, -0.1, 0],
        [0.1, 0.046, 0.023, -0.08, -0.605, 0],
        [0.1, 0.023, 0.023, 0, -0.606, 0],
        [0.1, 0.023, 0.046, 0.06, -0.605, 0],
    ]
    
    # Create coordinate grid
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Initialize phantom
    phantom = np.zeros((size, size))
    
    # Add each ellipse
    for ellipse in ellipses:
        A, a, b, x0, y0, phi = ellipse
        phi = phi * np.pi / 180  # Convert to radians
        
        # Rotation
        X_rot = (X - x0) * np.cos(phi) + (Y - y0) * np.sin(phi)
        Y_rot = -(X - x0) * np.sin(phi) + (Y - y0) * np.cos(phi)
        
        # Ellipse equation
        inside = (X_rot / a) ** 2 + (Y_rot / b) ** 2 <= 1
        phantom[inside] += A
    
    # Normalize
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min())
    
    return phantom


def visualize_kspace(h5_path, save_path=None):
    """Visualize k-space data from h5 file"""
    import matplotlib.pyplot as plt
    
    with h5py.File(h5_path, 'r') as f:
        kspace = f['kspace'][()]
        sequence = f.attrs.get('sequence', 'unknown')
    
    # Handle multiple slices
    if kspace.ndim == 3:
        kspace = kspace[kspace.shape[0] // 2]  # Middle slice
    
    # Reconstruct image
    image = np.fft.ifftshift(kspace)
    image = np.fft.ifft2(image)
    image = np.abs(image)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # K-space magnitude
    axes[0].imshow(np.log(np.abs(kspace) + 1), cmap='gray')
    axes[0].set_title('K-space (log magnitude)')
    axes[0].axis('off')
    
    # K-space phase
    axes[1].imshow(np.angle(kspace), cmap='hsv')
    axes[1].set_title('K-space (phase)')
    axes[1].axis('off')
    
    # Reconstructed image
    axes[2].imshow(image, cmap='gray')
    axes[2].set_title(f'Reconstructed Image\nSequence: {sequence}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    plt.show()


def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train/val/test sets.
    """
    data_dir = Path(data_dir)
    
    # Get all h5 files
    all_files = sorted(list(data_dir.glob('*.h5')))
    num_files = len(all_files)
    
    # Calculate splits
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)
    
    # Shuffle
    np.random.shuffle(all_files)
    
    # Split
    train_files = all_files[:num_train]
    val_files = all_files[num_train:num_train + num_val]
    test_files = all_files[num_train + num_val:]
    
    # Create directories and move files
    for split_name, files in [('train', train_files), 
                              ('val', val_files), 
                              ('test', test_files)]:
        split_dir = data_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        for file in tqdm(files, desc=f'Moving {split_name} files'):
            file.rename(split_dir / file.name)
    
    print(f"\n✓ Dataset split complete:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val:   {len(val_files)} files")
    print(f"  Test:  {len(test_files)} files")


def main():
    """Main data preparation script"""
    parser = argparse.ArgumentParser(description='MRI Data Preparation')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['nifti', 'dicom', 'fastmri', 'synthetic', 
                               'split', 'visualize'],
                       help='Preparation mode')
    parser.add_argument('--input', type=str, help='Input path')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--sequence', type=str, default='t1',
                       choices=['t1', 't2', 'pd'],
                       help='MRI sequence type')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples for synthetic data')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size for synthetic data')
    
    args = parser.parse_args()
    
    if args.mode == 'nifti':
        # Convert NIfTI to H5
        convert_nifti_to_h5(args.input, args.output, args.sequence)
    
    elif args.mode == 'dicom':
        # Convert DICOM to H5
        convert_dicom_to_h5(args.input, args.output, args.sequence)
    
    elif args.mode == 'fastmri':
        # Prepare fastMRI data
        prepare_fastmri_data(args.input, args.output, split='train')
    
    elif args.mode == 'synthetic':
        # Generate synthetic data
        create_synthetic_mri_data(args.output, args.num_samples, args.image_size)
    
    elif args.mode == 'split':
        # Split dataset
        split_dataset(args.input)
    
    elif args.mode == 'visualize':
        # Visualize k-space
        visualize_kspace(args.input, args.output)


if __name__ == "__main__":
    main()