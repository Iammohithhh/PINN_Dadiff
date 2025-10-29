import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embedding for timestep encoding in diffusion.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps):
        """
        Args:
            timesteps: (B,) tensor of timestep values
        Returns:
            embeddings: (B, dim) tensor of position embeddings
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class TransformerBlock(nn.Module):
    """
    Transformer block for capturing global dependencies.
    Implements Equation 18 from the paper.
    """
    def __init__(self, channels, num_heads=8, mlp_ratio=4.0):
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
        
        # MLP
        mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, channels)
        )
    
    def forward(self, x):
        """
        Multi-head self-attention with residual connections.
        Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
        """
        B, C, H, W = x.shape
        
        # Reshape for attention: (B, C, H, W) -> (B, HW, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Self-attention with residual
        shortcut = x_flat
        x_flat = self.norm1(x_flat)
        
        # Generate Q, K, V
        qkv = self.qkv(x_flat).reshape(B, H*W, 3, self.num_heads, 
                                        C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x_attn = self.proj(x_attn)
        
        # First residual connection
        x_flat = shortcut + x_attn
        
        # MLP with residual
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        # Reshape back: (B, HW, C) -> (B, C, H, W)
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return x


class NoiseScheduler(nn.Module):
    """
    Physics-informed noise scheduler for diffusion process.
    Implements the variance schedule β_t.
    """
    def __init__(self, beta_min=0.1, beta_max=20, num_timesteps=1000):
        super().__init__()
        
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_timesteps = num_timesteps
        
        # Exponential decay schedule
        self.register_buffer('betas', self._get_beta_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', 
                            torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                            torch.sqrt(1.0 - self.alphas_cumprod))
    
    def _get_beta_schedule(self):
        """
        Exponential decay schedule for β_t
        """
        scale = 1000 / self.num_timesteps
        beta_start = scale * self.beta_min / 1000
        beta_end = scale * self.beta_max / 1000
        
        return torch.linspace(beta_start, beta_end, self.num_timesteps)
    
    def add_noise(self, x_0, t, noise=None):
        """
        Forward diffusion process (Equation 14).
        q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t)·x_{t-1}, β_t·I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # x_t = sqrt(α̅_t)·x_0 + sqrt(1-α̅_t)·ε
        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        
        return x_t, noise


class DiffusionUNet(nn.Module):
    """
    U-Net architecture for noise prediction in diffusion.
    Predicts ε_θ(x_t, t) for reverse diffusion.
    """
    def __init__(self, in_channels, model_channels=64, num_res_blocks=2):
        super().__init__()
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Encoder
        self.conv_in = nn.Conv2d(in_channels, model_channels, 
                                kernel_size=3, padding=1)
        
        ch_mult = [1, 2, 4, 8]
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        in_ch = model_channels
        for i, mult in enumerate(ch_mult):
            out_ch = model_channels * mult
            
            # Residual blocks
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_ch, out_ch, time_embed_dim))
                in_ch = out_ch
            self.down_blocks.append(nn.ModuleList(blocks))
            
            # Downsample
            if i < len(ch_mult) - 1:
                self.down_samples.append(
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, 
                             stride=2, padding=1)
                )
            else:
                self.down_samples.append(nn.Identity())
        
        # Middle
        self.middle_block1 = ResidualBlock(in_ch, in_ch, time_embed_dim)
        self.middle_attn = TransformerBlock(in_ch, num_heads=8)
        self.middle_block2 = ResidualBlock(in_ch, in_ch, time_embed_dim)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = model_channels * mult
            
            # Residual blocks
            blocks = []
            for j in range(num_res_blocks + 1):
                # Add skip connection channels for first block
                skip_ch = in_ch if j == 0 else 0
                blocks.append(ResidualBlock(out_ch + skip_ch, out_ch, 
                                           time_embed_dim))
            self.up_blocks.append(nn.ModuleList(blocks))
            
            # Upsample
            if i > 0:
                self.up_samples.append(
                    nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, 
                                      stride=2, padding=1)
                )
            else:
                self.up_samples.append(nn.Identity())
            
            in_ch = out_ch
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, t):
        """
        Predict noise ε_θ(x_t, t)
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Initial conv
        h = self.conv_in(x)
        
        # Encoder with skip connections
        skip_connections = []
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                h = block(h, t_emb)
            skip_connections.append(h)
            h = downsample(h)
        
        # Middle
        h = self.middle_block1(h, t_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, t_emb)
        
        # Decoder with skip connections
        for blocks, upsample in zip(self.up_blocks, self.up_samples):
            skip = skip_connections.pop()
            for i, block in enumerate(blocks):
                if i == 0:
                    h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
            h = upsample(h)
        
        # Output
        return self.conv_out(h)


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding.
    """
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
        self.skip = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        self.act = nn.SiLU()
    
    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        
        return h + self.skip(x)


class DataConsistencyLayer(nn.Module):
    """
    Data consistency layer for k-space enforcement.
    Implements Equation 17 from the paper.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x_recon, k_measured, mask):
        """
        Enforce data consistency in k-space.
        L_data_consistency = ||F(x_t) - k_measured||²
        
        Args:
            x_recon: Reconstructed image (B, C, H, W)
            k_measured: Measured k-space data (B, C, H, W, 2)
            mask: Sampling mask (B, 1, H, W)
        """
        # Transform to k-space
        x_kspace = torch.fft.fft2(x_recon, norm='ortho')
        x_kspace = torch.stack([x_kspace.real, x_kspace.imag], dim=-1)
        
        # Apply data consistency
        k_measured_complex = torch.complex(k_measured[..., 0], k_measured[..., 1])
        x_kspace_complex = torch.complex(x_kspace[..., 0], x_kspace[..., 1])
        
        # Replace measured k-space values
        x_kspace_dc = torch.where(
            mask.unsqueeze(-1).expand_as(x_kspace).bool(),
            k_measured,
            x_kspace
        )
        
        # Transform back to image space
        x_kspace_dc_complex = torch.complex(x_kspace_dc[..., 0], x_kspace_dc[..., 1])
        x_dc = torch.fft.ifft2(x_kspace_dc_complex, norm='ortho').real
        
        return x_dc


class AdaptiveDiffusionRefinementNetwork(nn.Module):
    """
    Complete ADRN implementation with forward and reverse diffusion.
    Implements adaptive diffusion with physics-informed priors.
    """
    def __init__(self,
                 in_channels=256,
                 model_channels=64,
                 num_timesteps=1000,
                 num_inference_steps=12,
                 beta_min=0.1,
                 beta_max=20.0,
                 lambda_phys=0.2):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_timesteps = num_timesteps
        self.num_inference_steps = num_inference_steps
        self.lambda_phys = lambda_phys
        
        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(beta_min, beta_max, num_timesteps)
        
        # Diffusion U-Net for noise prediction
        self.unet = DiffusionUNet(in_channels, model_channels, num_res_blocks=2)
        
        # Transformer blocks for global dependencies (Equation 18)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(in_channels, num_heads=8)
            for _ in range(2)
        ])
        
        # Data consistency layer
        self.data_consistency = DataConsistencyLayer()
        
        # Physics regularization
        self.physics_reg = PhysicsRegularization(gamma=0.1)
        
        # Adaptive prior network
        self.adaptive_prior = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
    
    def forward_diffusion(self, x_0, num_steps=None):
        """
        Forward diffusion process: add noise gradually.
        Implements Equation 14.
        """
        if num_steps is None:
            num_steps = self.num_timesteps
        
        B = x_0.shape[0]
        t = torch.randint(0, num_steps, (B,), device=x_0.device).long()
        
        # Add noise
        x_t, noise = self.noise_scheduler.add_noise(x_0, t)
        
        return x_t, noise, t
    
    def reverse_diffusion_step(self, x_t, t, k_measured=None, mask=None):
        """
        Single reverse diffusion step with physics regularization.
        Implements Equation 15 and 16.
        
        p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ²_θ·I)
        """
        # Predict noise
        noise_pred = self.unet(x_t, t)
        
        # Get schedule parameters
        alpha_t = self.noise_scheduler.alphas[t].view(-1, 1, 1, 1)
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.noise_scheduler.betas[t].view(-1, 1, 1, 1)
        
        # Compute mean (Equation 16)
        # μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t))·ε_θ(x_t, t))
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_prod_t = torch.sqrt(1.0 - alpha_prod_t)
        
        mean = (x_t - (beta_t / sqrt_one_minus_alpha_prod_t) * noise_pred) / sqrt_alpha_t
        
        # Add physics regularization (Equation 16)
        if self.training:
            physics_term = self.lambda_phys * self.physics_reg(x_t)
            mean = mean + physics_term
        
        # Add noise for t > 0
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = beta_t
            x_t_minus_1 = mean + torch.sqrt(variance) * noise
        else:
            x_t_minus_1 = mean
        
        # Apply data consistency if k-space data provided
        if k_measured is not None and mask is not None:
            x_t_minus_1 = self.data_consistency(x_t_minus_1, k_measured, mask)
        
        return x_t_minus_1
    
    def forward(self, z_pace, k_measured=None, mask=None, 
                return_losses=False, mode='train'):
        """
        Complete ADRN forward pass.
        
        Args:
            z_pace: Features from PACE (B, C, H, W)
            k_measured: Measured k-space data (optional)
            mask: Sampling mask (optional)
            return_losses: Whether to return losses
            mode: 'train' or 'inference'
        """
        losses = {}
        
        # Apply transformer blocks for global dependencies
        x = z_pace
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        if mode == 'train':
            # Training: forward then reverse diffusion
            # Forward diffusion
            x_noisy, noise_gt, t = self.forward_diffusion(x)
            
            # Predict noise
            noise_pred = self.unet(x_noisy, t)
            
            # Diffusion loss
            diff_loss = F.mse_loss(noise_pred, noise_gt)
            losses['diffusion_loss'] = diff_loss
            
            # Physics regularization
            physics_loss = self.lambda_phys * self.physics_reg(x_noisy)
            losses['physics_loss'] = physics_loss
            
            # Reconstruct x_0 from predicted noise
            alpha_prod_t = self.noise_scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alpha_prod = torch.sqrt(alpha_prod_t)
            sqrt_one_minus_alpha_prod = torch.sqrt(1.0 - alpha_prod_t)
            
            x_recon = (x_noisy - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
            
            if return_losses:
                losses['total_loss'] = diff_loss + physics_loss
                return x_recon, losses
            
            return x_recon
        
        else:
            # Inference: iterative reverse diffusion
            B, C, H, W = x.shape
            
            # Apply adaptive prior
            x = self.adaptive_prior(x)
            
            # Start from noisy version
            x_t = x + torch.randn_like(x) * 0.1
            
            # Reverse diffusion iterations
            timesteps = torch.linspace(self.num_timesteps - 1, 0, 
                                      self.num_inference_steps, 
                                      device=x.device).long()
            
            for t in timesteps:
                t_batch = t.repeat(B)
                x_t = self.reverse_diffusion_step(x_t, t_batch, 
                                                  k_measured, mask)
            
            return x_t


class PhysicsRegularization(nn.Module):
    """
    Physics-based regularization for diffusion process.
    """
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, x):
        """
        Enforce smoothness and physical constraints
        """
        # Gradient smoothness
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        
        smoothness = torch.mean(dx**2) + torch.mean(dy**2)
        
        return self.gamma * smoothness


# Example usage
if __name__ == "__main__":
    # Create ADRN
    adrn = AdaptiveDiffusionRefinementNetwork(
        in_channels=256,
        model_channels=64,
        num_timesteps=1000,
        num_inference_steps=12,
        beta_min=0.1,
        beta_max=20.0,
        lambda_phys=0.2
    )
    
    # Example input from PACE
    batch_size = 2
    height, width = 128, 128
    z_pace = torch.randn(batch_size, 256, height, width)
    
    # Training mode
    print("=== Training Mode ===")
    adrn.train()
    x_recon, losses = adrn(z_pace, return_losses=True, mode='train')
    print(f"Input shape: {z_pace.shape}")
    print(f"Reconstructed shape: {x_recon.shape}")
    print(f"Losses: {losses}")
    
    # Inference mode
    print("\n=== Inference Mode ===")
    adrn.eval()
    with torch.no_grad():
        x_refined = adrn(z_pace, mode='inference')
    print(f"Refined shape: {x_refined.shape}")
    
    # Total parameters
    total_params = sum(p.numel() for p in adrn.parameters())
    print(f"\nTotal ADRN parameters: {total_params:,}")