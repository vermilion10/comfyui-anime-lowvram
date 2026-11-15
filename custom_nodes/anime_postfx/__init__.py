"""Lightweight photographic finishing effects for generated images.

Provides a single node, "Anime Post FX", that applies bloom, chromatic
aberration, vignette and film grain. Everything runs with plain PyTorch on
the CPU, so it uses no VRAM.
"""

import torch
import torch.nn.functional as F


def _gaussian_kernel(radius: float) -> torch.Tensor:
    sigma = max(radius / 2.0, 0.5)
    size = int(radius) * 2 + 1
    x = torch.arange(size, dtype=torch.float32) - size // 2
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _blur(x: torch.Tensor, radius: float) -> torch.Tensor:
    """Separable gaussian blur on a [B, C, H, W] tensor."""
    if radius < 0.5:
        return x
    k = _gaussian_kernel(radius).to(x)
    c = x.shape[1]
    pad = k.numel() // 2
    kh = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kv = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="replicate"), kh, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="replicate"), kv, groups=c)
    return x


def _bloom(img: torch.Tensor, strength: float, radius: float, threshold: float) -> torch.Tensor:
    if strength <= 0:
        return img
    luma = (img * torch.tensor([0.299, 0.587, 0.114], device=img.device).view(1, 3, 1, 1)).sum(1, keepdim=True)
    mask = ((luma - threshold) / max(1.0 - threshold, 1e-4)).clamp(0, 1)
    bright = img * mask

    # Blur at quarter resolution: same look, far cheaper for large radii.
    h, w = img.shape[-2:]
    small = F.interpolate(bright, scale_factor=0.25, mode="area")
    small = _blur(small, radius / 4.0)
    glow = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)

    glow = (glow * strength).clamp(0, 1)
    return 1.0 - (1.0 - img) * (1.0 - glow)  # screen blend


def _chromatic_aberration(img: torch.Tensor, amount_px: float) -> torch.Tensor:
    """Radial lens-style fringing: red scaled outward, blue inward."""
    if amount_px <= 0:
        return img
    b, _, h, w = img.shape
    s = amount_px / (max(h, w) / 2.0)
    ys = torch.linspace(-1, 1, h, device=img.device)
    xs = torch.linspace(-1, 1, w, device=img.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)

    def sample(channel: torch.Tensor, scale: float) -> torch.Tensor:
        return F.grid_sample(channel, base * scale, mode="bilinear", padding_mode="border", align_corners=True)

    r = sample(img[:, 0:1], 1.0 - s)
    g = img[:, 1:2]
    bl = sample(img[:, 2:3], 1.0 + s)
    return torch.cat((r, g, bl), dim=1)


def _vignette(img: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0:
        return img
    h, w = img.shape[-2:]
    ys = torch.linspace(-1, 1, h, device=img.device).view(-1, 1)
    xs = torch.linspace(-1, 1, w, device=img.device).view(1, -1)
    r = torch.sqrt(xs ** 2 + ys ** 2) / (2 ** 0.5)
    t = ((r - 0.35) / 0.65).clamp(0, 1)
    falloff = t * t * (3 - 2 * t)
    return img * (1.0 - strength * falloff)


def _grain(img: torch.Tensor, strength: float, seed: int) -> torch.Tensor:
    if strength <= 0:
        return img
    gen = torch.Generator(device="cpu").manual_seed(seed)
    b, _, h, w = img.shape
    noise = torch.randn((b, 1, h, w), generator=gen).to(img)
    return img + noise * strength


class AnimePostFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bloom_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05,
                                             "tooltip": "Glow added around bright areas."}),
                "bloom_radius": ("FLOAT", {"default": 40.0, "min": 1.0, "max": 200.0, "step": 1.0,
                                           "tooltip": "Spread of the glow in pixels."}),
                "bloom_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 0.99, "step": 0.01,
                                              "tooltip": "Brightness above which areas start to glow."}),
                "chromatic_aberration": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.5,
                                                   "tooltip": "Red/blue fringe offset at the image edge, in pixels."}),
                "vignette": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "Darkening toward the corners."}),
                "grain": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005,
                                    "tooltip": "Film grain intensity."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                                 "tooltip": "Seed for the grain pattern."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/postprocessing"

    def apply(self, image, bloom_strength, bloom_radius, bloom_threshold,
              chromatic_aberration, vignette, grain, seed):
        x = image.detach().to("cpu", torch.float32).movedim(-1, 1)  # [B, C, H, W]
        rgb, alpha = x[:, :3], x[:, 3:]

        rgb = _bloom(rgb, bloom_strength, bloom_radius, bloom_threshold)
        rgb = _chromatic_aberration(rgb, chromatic_aberration)
        rgb = _vignette(rgb, vignette)
        rgb = _grain(rgb, grain, seed)

        out = torch.cat((rgb.clamp(0, 1), alpha), dim=1).movedim(1, -1)
        return (out.to(image.dtype),)


NODE_CLASS_MAPPINGS = {"AnimePostFX": AnimePostFX}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimePostFX": "Anime Post FX"}
