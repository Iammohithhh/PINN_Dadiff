"""
LPCE implementation (PyTorch) for PINN-DADif paper.

- Multi-branch:
    * data_branch: standard conv encoder (sequence-specific options)
    * physics_branch: uses FFT/IFFT, predicts simple Bloch-like maps (M0, T1, T2)
      and produces a physics-consistent feature map f_phys.
- Outputs latent features Zlatent = f_data + lambda_phys * f_phys
- Also returns a physics regularization scalar (phys_reg) that can be added to total loss.

Assumptions / API:
- Input `kspace` is a complex tensor represented as real/imag channels:
    shape: (B, 2, H, W)  OR (B, coils, 2, H, W)
  For multi-coil, pass (B, C, 2, H, W) and set multi_coil=True.
- sequence_type: "T1", "T2", or "PD" (affects simple Bloch model used).
- latent_dim controls final channel count.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---- small util building blocks ----
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(x + self.net(x))

# ---- LPCE module ----
class LPCE(nn.Module):
    def __init__(
        self,
        in_channels=2,            # real/imag input channels per coil or per image
        initial_channels=32,
        latent_dim=128,
        num_res_blocks=2,
        lambda_phys=0.2,          # weight mixing f_phys into Zlatent
        phys_reg_gamma=1.0,       # scales phys_reg term
        multi_coil=False,
        sequence_types=("T1","T2","PD")
    ):
        super().__init__()
        self.multi_coil = multi_coil
        self.lambda_phys = lambda_phys
        self.phys_reg_gamma = phys_reg_gamma
        self.sequence_types = sequence_types

        # sequence-specific small heads (paper: LPCE employs sequence-specific processing)
        # we'll create a small conv per sequence that maps input->feat
        self.seq_heads = nn.ModuleDict()
        for seq in sequence_types:
            self.seq_heads[seq] = nn.Sequential(
                ConvBlock(in_channels, initial_channels),
                ConvBlock(initial_channels, initial_channels)
            )

        # data_branch encoder (shared after seq head)
        self.encoder = nn.Sequential(
            ConvBlock(initial_channels, initial_channels * 2, k=3, stride=2, padding=1),
            *[ResidualBlock(initial_channels * 2) for _ in range(num_res_blocks)],
            ConvBlock(initial_channels * 2, latent_dim, k=1, stride=1, padding=0)
        )

        # physics branch: small CNNs that work in image space; will predict M0 and T1/T2 maps
        # Assume physics prediction uses image-domain representation (IFFT of k-space)
        # We'll predict maps with same spatial dims as input feature map; channels: M0, Tmap
        self.phys_predictor = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1, 1, 0)  # output 2 channels: (M0_est, T_est) -- T interpreted per-sequence
        )

        # map physics prediction into latent space (feature transform)
        self.phys_to_latent = nn.Sequential(
            nn.Conv2d(2, latent_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )

        # small learnable coil-sensitivity regularizer (if multi-coil)
        if self.multi_coil:
            self.coil_compress = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False)

    def forward(self, kspace, sequence_type="T1"):
        """
        kspace: tensor shape
           if multi_coil: (B, coils, 2, H, W)  (real/imag last two channels)
           else: (B, 2, H, W)
        sequence_type: "T1" | "T2" | "PD" (affects physics model)
        Returns:
          zlatent: (B, latent_dim, H_lat, W_lat)
          phys_reg: scalar tensor (batch mean) representing physics regularization
          aux: dict with 'M0', 'T' predicted maps
        """
        # ===== normalize & prepare complex -> image domain =====
        # convert to complex image via ifft2
        # unify dims: make shape (B,  C_coils_or_1, 2, H, W)
        if kspace.dim() == 4:
            # (B, 2, H, W)
            multi = False
            kc = kspace
        elif kspace.dim() == 5:
            multi = True
            kc = kspace
        else:
            raise ValueError("kspace must be shape (B,2,H,W) or (B,coils,2,H,W)")

        # convert to complex-valued tensor using torch.view as complex or two channels
        # We'll implement IFFT using torch.fft with complex dtype
        def ifft2_from_realimag(x):  # x: (..., 2, H, W) or (..., H, W, 2)
            # bring channels last for complex creation
            # input x shape: (B, 2, H, W)
            real = x[...,0,:,:] if x.ndim==4 else x[...,0,:,:]
            imag = x[...,1,:,:] if x.ndim==4 else x[...,1,:,:]
            complex_img = torch.complex(real, imag)  # shape (..., H, W)
            img = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(complex_img, dim=(-2,-1)), norm='ortho'), dim=(-2,-1))
            # return real/imag stacked as float channels
            return torch.stack([img.real, img.imag], dim=1)  # (B, 2, H, W) or (B, coils, 2, H, W)

        if multi:
            B, coils, _, H, W = kc.shape
            # reshape to (B*coils, 2, H, W) for IFFT
            kc_rs = kc.reshape(B*coils, kc.shape[2], H, W)
            img_rs = ifft2_from_realimag(kc_rs)  # (B*coils, 2, H, W)
            img = img_rs.reshape(B, coils, 2, H, W)
            # coil combine (magnitude-sum or learned)
            # We'll simple take root-sum-of-squares magnitude as quick combine to feed physics predictor,
            # but keep complex channels by summing complex signals across coils
            # sum complex signals across coils:
            complex_imgs = torch.complex(img[:,0,:,:,:], img[:,1,:,:,:])  # (B, coils, H, W)
            combined = complex_imgs.sum(dim=1)  # (B, H, W)
            img_for_phys = torch.stack([combined.real, combined.imag], dim=1)  # (B, 2, H, W)
        else:
            img_for_phys = ifft2_from_realimag(kc)  # (B, 2, H, W)

        # ===== sequence-specific head + data branch =====
        # pass the raw input (kspace image-domain combined) through sequence-specific head
        # use imag+real channels as input
        seq_head = self.seq_heads.get(sequence_type, None)
        if seq_head is None:
            # fallback to first head if unknown
            seq_head = list(self.seq_heads.values())[0]
        x = seq_head(img_for_phys)  # (B, C, H, W)

        # encode to latent (data-driven)
        f_data = self.encoder(x)  # (B, latent_dim, H_lat, W_lat)

        # ===== physics branch: predict M0 and T maps from image domain =====
        phys_in = img_for_phys  # (B, 2, H, W)
        phys_pred = self.phys_predictor(phys_in)  # (B, 2, H, W) channels: M0_est, T_est
        M0 = phys_pred[:,0:1,:,:]
        Tmap = phys_pred[:,1:2,:,:]  # positive/negative allowed; later apply softplus if needed

        # compute a simple Bloch-inspired physics map f_phys:
        # using paper formulas:
        #   M_T1(t) = M0 * (1 - exp(-t/T1))   (T1-like)
        #   M_T2(t) = M0 * exp(-t/T2)         (T2-like)
        # We don't have real 't' (echo/tr) per slice — use a learnable nominal time scalar per sequence
        # Make small learnable scalar (register buffer)
        device = phys_pred.device
        if not hasattr(self, "_t_params"):
            # create tiny parameters
            self.register_parameter("_t_T1", nn.Parameter(torch.tensor(10.0)))
            self.register_parameter("_t_T2", nn.Parameter(torch.tensor(10.0)))
        t_T1 = torch.clamp(self._t_T1, 1e-3, 1e6)
        t_T2 = torch.clamp(self._t_T2, 1e-3, 1e6)

        if sequence_type.upper().startswith("T1"):
            # M = M0 * (1 - exp(-t/T))
            f_phys_map = M0 * (1.0 - torch.exp(- t_T1 / (torch.abs(Tmap) + 1e-6)))
        elif sequence_type.upper().startswith("T2"):
            f_phys_map = M0 * torch.exp(- t_T2 / (torch.abs(Tmap) + 1e-6))
        else:
            # PD or unknown -> simple proportional map
            f_phys_map = M0

        # now map physics map into same latent channel dimension
        f_phys_latent = self.phys_to_latent(f_phys_map)  # (B, latent_dim, H, W)
        # if encoder spatial dims differ, resize to match f_data dims
        if f_phys_latent.shape[2:] != f_data.shape[2:]:
            f_phys_latent = F.interpolate(f_phys_latent, size=f_data.shape[2:], mode='bilinear', align_corners=False)

        # combine
        zlatent = f_data + self.lambda_phys * f_phys_latent

        # physics regularization scalar (R_phys from paper eq (6))
        # compute per-pixel squared difference between activation (we'll use f_data activations) and f_phys (mapped to same space)
        # reduce to scalar per-batch element then mean
        diff = f_data - (f_phys_latent / (self.lambda_phys + 1e-12))
        phys_reg_map = (diff ** 2).sum(dim=1, keepdim=True)  # (B,1,H_lat,W_lat)
        phys_reg_scalar = phys_reg_map.mean() * self.phys_reg_gamma

        aux = {
            "M0": M0,
            "Tmap": Tmap,
            "f_phys_map": f_phys_map
        }

        return zlatent, phys_reg_scalar, aux

# ---- Example usage ----
# if __name__ == "__main__":
#     # quick smoke test
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = LPCE(in_channels=2, initial_channels=16, latent_dim=64, num_res_blocks=1).to(device)
#     # fake k-space: (B,2,H,W)
#     B, H, W = 2, 128, 128
#     fake_k = torch.randn(B, 2, H, W, device=device)
#     z, phys_reg, aux = model(fake_k, sequence_type="T1")
#     print("z:", z.shape, "phys_reg:", phys_reg.item())
