"""
Adaptive Diffusion Refinement Network (ADRN) for CT Reconstruction.

Implements iterative refinement through adaptive diffusion process with
CT physics constraints (sinogram consistency) replacing MRI k-space constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

from .ct_physics import (
    RadonTransform,
    FilteredBackProjection,
    CTDataConsistency,
    WeightedLeastSquaresLoss
)


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embedding.

    Args:
        timesteps: Tensor of timesteps (B,)
        embedding_dim: Dimension of embedding

    Returns:
        Timestep embeddings (B, embedding_dim)
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


class ResidualBlock(nn.Module):
    """Residual block with timestep conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        # Timestep embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward with timestep conditioning."""
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Add timestep embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]

        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Self-attention layer for capturing global dependencies."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Attention
        attn = torch.einsum('bncd,bncf->bndf', q, k) * (C // self.num_heads) ** -0.5
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out = torch.einsum('bndf,bncf->bncd', attn, v)
        out = out.reshape(B, C, H, W)

        return x + self.proj(out)


class TransformerBlock(nn.Module):
    """
    Transformer block for global dependency capture.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(channels * mlp_ratio), channels),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block.

        Args:
            x: Input (B, N, C) where N = H*W

        Returns:
            Output (B, N, C)
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class UNetEncoder(nn.Module):
    """U-Net encoder for diffusion model."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        time_emb_dim: int = 256
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch = in_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.downs.append(
                nn.ModuleList([
                    ResidualBlock(ch, out_ch, time_emb_dim),
                    ResidualBlock(out_ch, out_ch, time_emb_dim),
                    SelfAttention(out_ch) if mult >= 4 else nn.Identity()
                ])
            )
            self.pools.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            ch = out_ch

        self.out_channels = ch

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode with skip connections."""
        skips = []
        for (res1, res2, attn), pool in zip(self.downs, self.pools):
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            x = attn(x)
            skips.append(x)
            x = pool(x)
        return x, skips


class UNetDecoder(nn.Module):
    """U-Net decoder for diffusion model."""

    def __init__(
        self,
        out_channels: int,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        time_emb_dim: int = 256
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # Decoder needs to have same number of levels as encoder
        # Encoder: [1, 2, 4, 8] -> 4 downsamples, 4 skips at sizes [256, 128, 64, 32]
        # Decoder: needs 4 upsamples to go from 16 -> 32 -> 64 -> 128 -> 256

        reversed_mults = list(reversed(channel_mults))  # [8, 4, 2, 1]

        # Start with bottleneck channels
        ch = base_channels * reversed_mults[0]  # 512

        # Create upsampling layers for ALL levels
        for i in range(len(channel_mults)):  # 4 levels
            # Output channels for this level
            if i < len(channel_mults) - 1:
                out_ch = base_channels * reversed_mults[i + 1]
            else:
                out_ch = base_channels * reversed_mults[-1]  # Last level stays at base

            self.upsamples.append(
                nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
            )
            self.ups.append(
                nn.ModuleList([
                    ResidualBlock(ch * 2, out_ch, time_emb_dim),  # *2 for skip concat
                    ResidualBlock(out_ch, out_ch, time_emb_dim),
                    SelfAttention(out_ch) if out_ch >= base_channels * 4 else nn.Identity()
                ])
            )
            ch = out_ch

        self.final = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        skips: List[torch.Tensor]
    ) -> torch.Tensor:
        """Decode with skip connections."""
        skips = list(reversed(skips))  # Now [32x32, 64x64, 128x128, 256x256]

        for i, ((res1, res2, attn), upsample) in enumerate(zip(self.ups, self.upsamples)):
            x = upsample(x)
            # Handle size mismatch
            if x.shape[2:] != skips[i].shape[2:]:
                x = F.interpolate(x, size=skips[i].shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skips[i]], dim=1)
            x = res1(x, t_emb)
            x = res2(x, t_emb)
            x = attn(x)

        return self.final(x)


class DiffusionUNet(nn.Module):
    """
    U-Net for diffusion-based denoising.

    Predicts noise epsilon_theta(x_t, t) given noisy image and timestep.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        time_emb_dim: int = 256
    ):
        super().__init__()

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.encoder = UNetEncoder(base_channels, base_channels, channel_mults, time_emb_dim)

        # Middle
        mid_ch = base_channels * channel_mults[-1]
        self.mid = nn.ModuleList([
            ResidualBlock(mid_ch, mid_ch, time_emb_dim),
            SelfAttention(mid_ch),
            ResidualBlock(mid_ch, mid_ch, time_emb_dim)
        ])

        # Decoder
        self.decoder = UNetDecoder(base_channels, base_channels, channel_mults, time_emb_dim)

        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise.

        Args:
            x: Noisy image (B, C, H, W)
            t: Timesteps (B,)

        Returns:
            Predicted noise (B, C, H, W)
        """
        # Timestep embedding
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        # Initial conv
        h = self.init_conv(x)

        # Encoder
        h, skips = self.encoder(h, t_emb)

        # Middle
        for layer in self.mid:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)

        # Decoder
        h = self.decoder(h, t_emb, skips)

        return self.out_conv(h)


class CTPhysicsProjection(nn.Module):
    """
    Physics projection step for CT reconstruction.

    Replaces MRI k-space projection with sinogram consistency.
    x' = x - eta * A^T W (Ax - p)
    """

    def __init__(
        self,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        step_size: float = 0.1
    ):
        super().__init__()
        self.step_size = step_size

        self.radon = RadonTransform(img_size, num_angles, num_detectors)
        self.fbp = FilteredBackProjection(img_size, num_angles, num_detectors)

    def forward(
        self,
        x: torch.Tensor,
        sinogram_measured: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply physics projection step.

        Args:
            x: Current reconstruction (B, 1, H, W)
            sinogram_measured: Measured sinogram
            weights: Optional WLS weights
            mask: Optional mask for sparse-view

        Returns:
            Updated reconstruction
        """
        # Forward projection
        sinogram_pred = self.radon.forward_fast(x)

        # Compute residual
        residual = sinogram_pred - sinogram_measured

        # Apply mask
        if mask is not None:
            residual = residual * mask

        # Apply weights
        if weights is not None:
            residual = residual * weights

        # Backproject gradient
        gradient = self.fbp.backproject(residual)

        # Update step
        x_new = x - self.step_size * gradient

        # Non-negativity
        x_new = F.relu(x_new)

        return x_new


class AdaptiveDiffusionPrior(nn.Module):
    """
    Adaptive Diffusion Prior for CT.

    Dynamically adjusts based on data-consistency loss during inference.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512
    ):
        super().__init__()

        # Adaptation network
        self.adapter = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )

        # Learnable base prior
        self.base_prior = nn.Parameter(torch.zeros(1, latent_dim, 1, 1))

    def forward(
        self,
        data_consistency_loss: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Compute adaptive prior based on data consistency.

        Args:
            data_consistency_loss: Current DC loss (scalar or batch)
            batch_size: Batch size

        Returns:
            Adaptive prior (B, latent_dim, 1, 1)
        """
        # Normalize loss
        loss_normalized = data_consistency_loss.view(-1, 1)

        # Compute adaptation
        adaptation = self.adapter(loss_normalized)
        adaptation = adaptation.view(-1, adaptation.shape[-1], 1, 1)

        # Combine with base prior
        prior = self.base_prior.expand(batch_size, -1, -1, -1) + adaptation

        return prior


class CT_ADRN(nn.Module):
    """
    Adaptive Diffusion Refinement Network for CT Reconstruction.

    Refines features through adaptive diffusion with CT physics constraints.
    Two-phase process:
    1. Rapid diffusion for initial reconstruction
    2. Adaptation phase for physics-consistent refinement
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        img_size: int = 256,
        num_angles: int = 180,
        num_detectors: Optional[int] = None,
        num_timesteps: int = 1000,
        num_inference_steps: int = 12,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        physics_step_size: float = 0.1
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_inference_steps = num_inference_steps
        self.img_size = img_size

        # Diffusion noise schedule (exponential)
        betas = torch.linspace(beta_min, beta_max, num_timesteps) / num_timesteps
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # Denoising network
        self.denoise_net = DiffusionUNet(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=64
        )

        # Physics projection
        self.physics_proj = CTPhysicsProjection(
            img_size=img_size,
            num_angles=num_angles,
            num_detectors=num_detectors,
            step_size=physics_step_size
        )

        # Adaptive prior
        self.adaptive_prior = AdaptiveDiffusionPrior(latent_dim=in_channels)

        # Feature to image projection
        self.to_image = nn.Conv2d(in_channels, 1, 1)
        self.from_image = nn.Conv2d(1, in_channels, 1)

        # Transformer for global dependencies
        self.transformer = TransformerBlock(in_channels, num_heads=8)

        # Physics loss
        self.wls_loss = WeightedLeastSquaresLoss()
        self.radon = RadonTransform(img_size, num_angles, num_detectors)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion (add noise)."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        sinogram: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Reverse diffusion step with physics projection."""
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)

        # Predict noise
        noise_pred = self.denoise_net(x_t, t_tensor)

        # Compute mean
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]

        mean = (1.0 / torch.sqrt(alpha)) * (
            x_t - (beta / self.sqrt_one_minus_alphas_cumprod[t]) * noise_pred
        )

        # Add physics projection step
        # FIXED: Adaptive blending - physics strength increases with lower t (later steps)
        if sinogram is not None:
            # Convert features to image domain (keeps spatial size)
            mean_image = self.to_image(mean)
            feature_size = mean_image.shape[2:]  # Store original feature spatial size

            # FIXED: Upsample to full image size for physics projection
            mean_image_full = F.interpolate(
                mean_image, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False
            )

            # Physics step at full resolution
            mean_image_corrected = self.physics_proj(mean_image_full, sinogram, weights, mask)

            # Downsample back to feature size
            mean_image_corrected = F.interpolate(
                mean_image_corrected, size=feature_size,
                mode='bilinear', align_corners=False
            )

            # Convert back to feature domain
            mean_corrected = self.from_image(mean_image_corrected)

            # Adaptive blending: stronger physics at later steps (lower t)
            # At t=num_timesteps: blend=0.2, at t=0: blend=1.0 (full physics)
            physics_weight = 0.2 + 0.8 * (1.0 - t / self.num_timesteps)
            mean = physics_weight * mean_corrected + (1.0 - physics_weight) * mean

        # Add noise for non-final steps
        if t > 0:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(beta)
            mean = mean + variance * noise

        return mean

    def forward(
        self,
        z_pace: torch.Tensor,
        sinogram: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Refine features through diffusion.

        Args:
            z_pace: Features from PACE (B, C, H, W)
            sinogram: Measured sinogram for physics constraint
            weights: Optional WLS weights
            mask: Optional sparse-view mask
            return_intermediate: Return intermediate reconstructions

        Returns:
            Refined features (B, C, H, W)
        """
        B, C, H, W = z_pace.shape
        intermediates = []

        # Initialize from PACE features (with some noise for diffusion)
        x_t = z_pace + 0.1 * torch.randn_like(z_pace)

        # Inference timesteps (evenly spaced)
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, self.num_inference_steps,
            device=z_pace.device
        ).long()

        for t in timesteps:
            # Reverse diffusion step
            x_t = self.p_sample(x_t, t.item(), sinogram, weights, mask)

            # Apply transformer for global context
            x_flat = x_t.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            x_flat = self.transformer(x_flat)
            x_t = x_flat.permute(0, 2, 1).view(B, C, H, W)

            if return_intermediate:
                intermediates.append(x_t.clone())

        # Adaptive prior adjustment
        if sinogram is not None:
            # Compute data consistency loss
            x_image = self.to_image(x_t)
            sino_pred = self.radon.forward_fast(x_image)
            dc_loss = self.wls_loss(sino_pred, sinogram, weights)

            # Get adaptive prior
            prior = self.adaptive_prior(dc_loss.detach(), B)
            x_t = x_t + prior

        if return_intermediate:
            return x_t, intermediates

        return x_t
