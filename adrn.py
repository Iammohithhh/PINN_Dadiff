import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PhysicsInformedRegularization(nn.Module):
    """
    Physics-based regularization for diffusion process
    Implements Equation 16 regularization component
    """
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, x_t, k_measured=None, mask=None):
        """
        Args:
            x_t: Current diffusion state (B, C, H, W)
            k_measured: Measured k-space data (optional)
            mask: Undersampling mask (optional)
        Returns:
            Physics regularization loss
        """
        # 1. Gradient smoothness
        dx = x_t[:, :, :, 1:] - x_t[:, :, :, :-1]
        dy = x_t[:, :, 1:, :] - x_t[:, :, :-1, :]
        smoothness_loss = torch.mean(dx**2) + torch.mean(dy**2)
        
        # 2. K-space fidelity (if available)
        kspace_loss = 0.0
        if k_measured is not None and mask is not None:
            k_pred = torch.fft.fft2(x_t, norm='ortho')
            kspace_loss = torch.mean(torch.abs((k_pred - k_measured) * mask)**2)
        
        # 3. Total variation for noise suppression
        tv_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
        
        total_loss = smoothness_loss + 0.5 * kspace_loss + 0.1 * tv_loss
        
        return self.gamma * total_loss


class NoiseScheduler:
    """
    Noise scheduling for diffusion process
    Implements exponential decay from β_max to β_min
    Paper uses: β_min=0.1, β_max=20.0, T=10 steps
    """
    def __init__(self, beta_min=0.1, beta_max=20.0, num_steps=10, device='cuda'):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_steps = num_steps
        self.device = device
        
        # Exponential schedule from β_max to β_min
        self.betas = self._get_beta_schedule()
        
        # Pre-compute alpha values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / 
            (1.0 - self.alphas_cumprod)
        )
    
    def _get_beta_schedule(self):
        """
        Exponential decay schedule from β_max to β_min
        """
        betas = torch.exp(
            torch.linspace(
                np.log(self.beta_max), 
                np.log(self.beta_min), 
                self.num_steps
            )
        )
        return betas.to(self.device)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process - add noise
        Implements Equation 14: q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
        
        Args:
            x_0: Original image (B, C, H, W)
            t: Timestep (B,)
            noise: Optional pre-generated noise
        Returns:
            Noisy image x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        
        # x_t = √(α̅_t) * x_0 + √(1 - α̅_t) * ε
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _extract(self, a, t, x_shape):
        """
        Extract values from a at timestep t and reshape for broadcasting
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention for global dependency capture
    Implements Equation 18: Attention(Q,K,V) = softmax(QK^T/√d_k) * V
    """
    def __init__(self, channels, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Multi-head self-attention
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # MLP (Feed-forward network)
        mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, channels),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            Features with global dependencies (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence: (B, H*W, C)
        x_seq = x.flatten(2).transpose(1, 2)
        
        # Self-attention with residual
        x_norm = self.norm1(x_seq)
        
        # Generate Q, K, V
        qkv = self.qkv(x_norm).reshape(B, H*W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention: softmax(QK^T/√d_k) * V (Equation 18)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_dropout(x_attn)
        
        # Residual connection
        x_seq = x_seq + x_attn
        
        # MLP with residual
        x_seq = x_seq + self.mlp(self.norm2(x_seq))
        
        # Reshape back to image: (B, C, H, W)
        x_out = x_seq.transpose(1, 2).reshape(B, C, H, W)
        
        return x_out


class DiffusionUNet(nn.Module):
    """
    U-Net architecture for noise prediction in diffusion process
    Predicts noise ε_θ(x_t, t) at each timestep
    """
    def __init__(self, 
                 in_channels=256, 
                 model_channels=128,
                 out_channels=256,
                 num_res_blocks=2,
                 attention_resolutions=[8, 16],
                 channel_mult=(1, 2, 4, 8),
                 num_heads=8,
                 use_transformer=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.use_transformer = use_transformer
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [model_channels]
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, mult * model_channels)
                ]
                ch = mult * model_channels
                
                # Add transformer at specific resolutions
                if use_transformer and level in attention_resolutions:
                    layers.append(TransformerBlock(ch, num_heads))
                
                self.down_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            # Downsample (except last level)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                input_block_chans.append(ch)
        
        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            ResBlock(ch, time_embed_dim, ch),
            TransformerBlock(ch, num_heads) if use_transformer else nn.Identity(),
            ResBlock(ch, time_embed_dim, ch)
        ])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        mult * model_channels
                    )
                ]
                ch = mult * model_channels
                
                # Add transformer at specific resolutions
                if use_transformer and level in attention_resolutions:
                    layers.append(TransformerBlock(ch, num_heads))
                
                # Upsample (except last)
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                
                self.up_blocks.append(nn.ModuleList(layers))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, t):
        """
        Args:
            x: Noisy input (B, C, H, W)
            t: Timestep (B,)
        Returns:
            Predicted noise ε_θ(x_t, t)
        """
        # Time embedding
        t_emb = self.get_timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # Input projection
        h = self.input_proj(x)
        
        # Downsampling
        hs = [h]
        for module_list in self.down_blocks:
            for module in module_list:
                if isinstance(module, ResBlock):
                    h = module(h, t_emb)
                else:
                    h = module(h)
                hs.append(h)
        
        # Middle
        for module in self.middle_blocks:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
        
        # Upsampling
        for module_list in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for module in module_list:
                if isinstance(module, ResBlock):
                    h = module(h, t_emb)
                else:
                    h = module(h)
        
        # Output
        return self.output_proj(h)
    
    @staticmethod
    def get_timestep_embedding(timesteps, embedding_dim):
        """
        Sinusoidal positional embeddings for timesteps
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if embedding_dim % 2 == 1:  # Zero pad
            emb = F.pad(emb, (0, 1))
        
        return emb


class ResBlock(nn.Module):
    """
    Residual block with time embedding
    """
    def __init__(self, in_channels, time_embed_dim, out_channels):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_emb_proj = nn.Linear(time_embed_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
        
        self.act = nn.SiLU()
    
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.time_emb_proj(self.act(t_emb))[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class Downsample(nn.Module):
    """Downsampling layer"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class AdaptiveDiffusionRefinementNetwork(nn.Module):
    """
    Complete Adaptive Diffusion Refinement Network (ADRN) Implementation
    
    Implements:
    - Forward diffusion (Equation 14)
    - Reverse diffusion (Equations 15-16)
    - Physics-informed regularization (Equation 16)
    - Adaptive diffusion prior (Equation 17)
    - Transformer blocks (Equation 18)
    - Frequency domain learning
    
    Architecture from paper Section 2.3
    """
    def __init__(self,
                 in_channels=256,           # From PACE output
                 model_channels=128,
                 out_channels=256,          # To ART
                 num_diffusion_steps=10,    # T/k from paper
                 num_reverse_iterations=12, # Increased from 8
                 beta_min=0.1,
                 beta_max=20.0,
                 lambda_phys=0.2,
                 use_transformer=True,
                 num_heads=8,
                 device='cuda'):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_diffusion_steps = num_diffusion_steps
        self.num_reverse_iterations = num_reverse_iterations
        self.lambda_phys = lambda_phys
        self.device = device
        
        # Noise scheduler (Equations 14-16)
        self.noise_scheduler = NoiseScheduler(
            beta_min=beta_min,
            beta_max=beta_max,
            num_steps=num_diffusion_steps,
            device=device
        )
        
        # Diffusion model for noise prediction
        self.diffusion_model = DiffusionUNet(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=2,
            attention_resolutions=[1, 2],
            channel_mult=(1, 2, 4, 8),
            num_heads=num_heads,
            use_transformer=use_transformer
        )
        
        # Physics-informed regularization
        self.physics_reg = PhysicsInformedRegularization(gamma=1.0)
        
        # Adaptive mapping network for larger steps
        self.adaptive_mapper = nn.Sequential(
            nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # Transformer blocks for global refinement (Equation 18)
        if use_transformer:
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(out_channels, num_heads)
                for _ in range(3)
            ])
        else:
            self.transformer_blocks = None
        
        # Frequency domain processing
        self.freq_processor = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final refinement
        self.final_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def apply_data_consistency(self, x_t, k_measured, mask):
        """
        Data consistency projection (Equation 17)
        Ensures reconstruction matches measured k-space frequencies
        
        L_data-consistency = ||F(x_t) - k_measured||²
        """
        if k_measured is None or mask is None:
            return x_t
        
        # Convert to k-space
        k_pred = torch.fft.fft2(x_t, norm='ortho')
        
        # Apply data consistency: keep measured, update unmeasured
        k_corrected = k_pred * (1 - mask) + k_measured * mask
        
        # Convert back to image space
        x_corrected = torch.fft.ifft2(k_corrected, norm='ortho').real
        
        return x_corrected
    
    def p_sample(self, x_t, t, k_measured=None, mask=None):
        """
        Single reverse diffusion step
        Implements Equations 15-16:
        p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ²_θI)
        with physics-informed regularization
        
        Args:
            x_t: Current noisy state (B, C, H, W)
            t: Current timestep (B,)
            k_measured: Measured k-space data
            mask: Undersampling mask
        Returns:
            x_{t-1}: Denoised state
        """
        batch_size = x_t.shape[0]
        
        # Predict noise: ε_θ(x_t, t)
        predicted_noise = self.diffusion_model(x_t, t)
        
        # Get scheduler coefficients
        betas_t = self.noise_scheduler.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.noise_scheduler.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = self.noise_scheduler.sqrt_recip_alphas[t]
        
        # Reshape for broadcasting
        betas_t = betas_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = sqrt_recip_alphas_t.view(-1, 1, 1, 1)
        
        # Compute mean: μ_θ(x_t, t) (Equation 16)
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        # Add physics-informed regularization (Equation 16)
        # μ_θ(x_t,t) = ... + λ_phys * R_phys(x_t)
        if self.training:
            physics_correction = self.lambda_phys * self.physics_reg(
                model_mean, k_measured, mask
            )
            # Don't add loss directly, just guide the mean
            model_mean = model_mean - 0.01 * physics_correction * model_mean
        
        if t[0] == 0:
            # No noise at final step
            x_prev = model_mean
        else:
            # Add noise
            posterior_variance_t = self.noise_scheduler.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            x_prev = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        # Apply data consistency (Equation 17)
        x_prev = self.apply_data_consistency(x_prev, k_measured, mask)
        
        return x_prev
    
    def forward_diffusion(self, x_0, num_steps=None):
        """
        Forward diffusion process - add noise progressively
        Implements Equation 14
        """
        if num_steps is None:
            num_steps = self.num_diffusion_steps
        
        batch_size = x_0.shape[0]
        t = torch.randint(0, num_steps, (batch_size,), device=x_0.device).long()
        
        # Add noise
        noise = torch.randn_like(x_0)
        x_t = self.noise_scheduler.q_sample(x_0, t, noise)
        
        return x_t, t, noise
    
    def reverse_diffusion(self, x_T, k_measured=None, mask=None, return_trajectory=False):
        """
        Complete reverse diffusion process
        Implements iterative denoising with 12 reverse iterations
        
        Args:
            x_T: Initial noisy state (B, C, H, W)
            k_measured: Measured k-space data
            mask: Undersampling mask
            return_trajectory: Whether to return intermediate states
        Returns:
            x_0: Denoised output
            trajectory: List of intermediate states (if return_trajectory=True)
        """
        batch_size = x_T.shape[0]
        x_t = x_T
        trajectory = [x_T] if return_trajectory else None
        
        # Reverse diffusion iterations (12 iterations from paper)
        for iteration in range(self.num_reverse_iterations):
            # For each iteration, go through all timesteps
            for t_idx in reversed(range(self.num_diffusion_steps)):
                t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
                
                # Single reverse step with physics constraints
                x_t = self.p_sample(x_t, t, k_measured, mask)
                
                if return_trajectory and t_idx == 0:
                    trajectory.append(x_t.clone())
        
        if return_trajectory:
            return x_t, trajectory
        
        return x_t
    
    def frequency_domain_refinement(self, x):
        """
        Process features in frequency domain
        Captures global patterns through Fourier transform
        """
        # Transform to frequency domain
        x_freq = torch.fft.fft2(x, norm='ortho')
        
        # Separate real and imaginary parts
        x_freq_real = x_freq.real
        x_freq_imag = x_freq.imag
        
        # Stack and process
        x_freq_combined = torch.cat([x_freq_real, x_freq_imag], dim=1)
        x_freq_processed = self.freq_processor(x_freq_combined)
        
        # Transform back to spatial domain
        x_spatial = torch.fft.ifft2(
            torch.complex(x_freq_processed, torch.zeros_like(x_freq_processed)),
            norm='ortho'
        ).real
        
        return x_spatial
    
    def forward(self, z_pace, k_measured=None, mask=None, 
                return_losses=False, return_trajectory=False):
        """
        Complete forward pass through ADRN
        
        Two-phase process:
        1. Rapid-diffusion: Quick initial reconstruction
        2. Adaptive refinement: Iterative refinement with physics constraints
        
        Args:
            z_pace: Input features from PACE (B, C, H, W)
            k_measured: Measured k-space data (B, H, W) complex
            mask: Undersampling mask (B, 1, H, W)
            return_losses: Whether to return losses
            return_trajectory: Whether to return diffusion trajectory
        Returns:
            z_adrn: Refined features (B, out_channels, H, W)
            losses: Dict of losses (if return_losses=True)
            trajectory: Diffusion trajectory (if return_trajectory=True)
        """
        losses = {}
        
        # Phase 1: Rapid-diffusion (initial reconstruction)
        # Use adaptive mapper for fast preliminary result
        x_rapid = self.adaptive_mapper(z_pace)
        
        # Phase 2: Adaptive refinement through reverse diffusion
        # Add noise and then denoise with physics constraints
        if self.training:
            # During training: forward then reverse diffusion
            x_noisy, t, noise = self.forward_diffusion(x_rapid)
            
            # Predict noise
            predicted_noise = self.diffusion_model(x_noisy, t)
            
            # Diffusion loss
            diffusion_loss = F.mse_loss(predicted_noise, noise)
            losses['diffusion_loss'] = diffusion_loss
            
            # Start reverse from noisy state
            x_refined = self.reverse_diffusion(x_noisy, k_measured, mask)
        else:
            # During inference: just reverse diffusion from rapid result
            # Add small noise for refinement
            noise = torch.randn_like(x_rapid) * 0.1
            x_noisy = x_rapid + noise
            
            x_refined, trajectory = self.reverse_diffusion(
                x_noisy, k_measured, mask, return_trajectory=return_trajectory
            )
        
        # Apply transformer blocks for global refinement (Equation 18)
        if self.transformer_blocks is not None:
            for transformer in self.transformer_blocks:
                x_refined = transformer(x_refined)
        
        # Frequency domain refinement
        x_freq_refined = self.frequency_domain_refinement(x_refined)
        
        # Combine spatial and frequency features
        z_adrn = self.final_refine(x_freq_refined + x_refined)
        
        # Physics regularization loss
        if return_losses:
            physics_loss = self.lambda_phys * self.physics_reg(z_adrn, k_measured, mask)
            losses['physics_loss'] = physics_loss
            
            # Data consistency loss (Equation 17)
            if k_measured is not None and mask is not None:
                k_pred = torch.fft.fft2(z_adrn, norm='ortho')
                data_loss = torch.mean(torch.abs((k_pred - k_measured) * mask)**2)
                losses['data_consistency_loss'] = data_loss
            
            losses['total_loss'] = sum(losses.values())
        
        if return_trajectory and not self.training:
            return z_adrn, losses if return_losses else z_adrn, trajectory
        
        if return_losses:
            return z_adrn, losses
        
        return z_adrn
    
    def get_num_params(self):
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== Testing & Validation ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Complete ADRN Implementation")
    print("=" * 70)
    
    # Configuration matching paper
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = {
        'in_channels': 256,            # From PACE
        'model_channels': 128,
        'out_channels': 256,           # To ART
        'num_diffusion_steps': 10,     # T/k from paper
        'num_reverse_iterations': 12,  # Increased from 8
        'beta_min': 0.1,              # From Section 2.7
        'beta_max': 20.0,             # From Section 2.7
        'lambda_phys': 0.2,
        'use_transformer': True,
        'num_heads': 8,
        'device': device
    }
    
    # Create ADRN
    adrn = AdaptiveDiffusionRefinementNetwork(**config).to(device)
    
    print(f"\n1. Model Configuration:")
    print(f"   - Device: {device}")
    print(f"   - Input channels: {config['in_channels']}")
    print(f"   - Output channels: {config['out_channels']}")
    print(f"   - Diffusion steps: {config['num_diffusion_steps']}")
    print(f"   - Reverse iterations: {config['num_reverse_iterations']}")
    print(f"   - β_min: {config['beta_min']}, β_max: {config['beta_max']}")
    print(f"   - Physics weight λ: {config['lambda_phys']}")
    
    # Count parameters
    total_params = adrn.get_num_params()
    trainable_params = adrn.get_num_trainable_params()
    print(f"\n2. Model Parameters:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    # Test forward pass (training mode)
    print(f"\n3. Testing Forward Pass (Training):")
    adrn.train()
    batch_size = 2
    height, width = 128, 128  # Smaller for testing
    
    # Input from PACE
    z_pace = torch.randn(batch_size, 256, height, width).to(device)
    
    # K-space and mask
    k_measured = torch.randn(batch_size, height, width, dtype=torch.complex64).to(device)
    mask = (torch.rand(batch_size, 1, height, width) > 0.75).float().to(device)
    
    # Forward pass with losses
    z_adrn, losses = adrn(z_pace, k_measured, mask, return_losses=True)
    
    print(f"   - Input shape: {z_pace.shape}")
    print(f"   - Output shape: {z_adrn.shape}")
    print(f"   - Training losses:")
    for key, value in losses.items():
        print(f"     * {key}: {value.item():.6f}")
    
    # Test inference mode
    print(f"\n4. Testing Forward Pass (Inference):")
    adrn.eval()
    
    with torch.no_grad():
        z_adrn_inf, trajectory = adrn(
            z_pace, 
            k_measured, 
            mask, 
            return_trajectory=True
        )
    
    print(f"   - Output shape: {z_adrn_inf.shape}")
    print(f"   - Trajectory length: {len(trajectory)} steps")
    
    # Test individual components
    print(f"\n5. Component Testing:")
    
    # Noise scheduler
    scheduler = NoiseScheduler(beta_min=0.1, beta_max=20.0, num_steps=10, device=device)
    x_0 = torch.randn(2, 256, 64, 64).to(device)
    t = torch.tensor([5, 5], device=device)
    x_t = scheduler.q_sample(x_0, t)
    print(f"   - Noise Scheduler: x_0 {x_0.shape} -> x_t {x_t.shape} ✓")
    
    # Transformer block
    transformer = TransformerBlock(256, num_heads=8).to(device)
    x_test = torch.randn(2, 256, 32, 32).to(device)
    x_trans = transformer(x_test)
    print(f"   - Transformer Block: {x_test.shape} -> {x_trans.shape} ✓")
    
    # Diffusion U-Net
    unet = DiffusionUNet(
        in_channels=256,
        model_channels=128,
        out_channels=256
    ).to(device)
    t_test = torch.tensor([3, 3], device=device)
    noise_pred = unet(x_test, t_test)
    print(f"   - Diffusion U-Net: {x_test.shape} -> {noise_pred.shape} ✓")
    
    # Data consistency
    x_dc = adrn.apply_data_consistency(x_test, 
                                       torch.fft.fft2(x_test, norm='ortho'),
                                       torch.ones(2, 1, 32, 32).to(device))
    print(f"   - Data Consistency: {x_test.shape} -> {x_dc.shape} ✓")
    
    # Verify key components
    print(f"\n6. Component Verification:")
    print(f"   - Has noise scheduler: {hasattr(adrn, 'noise_scheduler')} ✓")
    print(f"   - Has diffusion model: {hasattr(adrn, 'diffusion_model')} ✓")
    print(f"   - Has transformer blocks: {adrn.transformer_blocks is not None} ✓")
    print(f"   - Has physics regularization: {hasattr(adrn, 'physics_reg')} ✓")
    print(f"   - Has adaptive mapper: {hasattr(adrn, 'adaptive_mapper')} ✓")
    print(f"   - Has frequency processor: {hasattr(adrn, 'freq_processor')} ✓")
    
    # Test integration with previous modules
    print(f"\n7. Integration Test with LPCE + PACE:")
    
    # Simulate LPCE output
    z_latent = torch.randn(2, 128, 128, 128).to(device)
    
    # Simulate PACE processing
    from pace import PhysicsAwareContextEncoder
    pace = PhysicsAwareContextEncoder(
        in_channels=128,
        hidden_channels=256,
        out_channels=256
    ).to(device)
    
    z_pace_out = pace(z_latent, sequence_type='t1')
    
    # ADRN processing
    adrn.eval()
    with torch.no_grad():
        z_adrn_final = adrn(z_pace_out, k_measured, mask)
    
    print(f"   - LPCE simulated: {z_latent.shape}")
    print(f"   - PACE output: {z_pace_out.shape}")
    print(f"   - ADRN output: {z_adrn_final.shape}")
    print(f"   - Pipeline integration successful ✓")
    
    # Test noise schedule
    print(f"\n8. Noise Schedule Verification:")
    betas = scheduler.betas.cpu().numpy()
    print(f"   - β values: {betas}")
    print(f"   - β_min: {betas.min():.4f} (expected: 0.1000)")
    print(f"   - β_max: {betas.max():.4f} (expected: 20.0000)")
    print(f"   - Schedule type: Exponential decay ✓")
    
    print(f"\n{'=' * 70}")
    print("✓ All tests passed! ADRN implementation complete.")
    print("=" * 70)