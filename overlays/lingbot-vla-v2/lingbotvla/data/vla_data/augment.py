"""C2R appearance-randomization augmentation (recipe ``c2r_v1``).

Online photometric + channel-statistics + frequency-domain augmentation for
clean-only training, targeting the clean -> randomized visual domain gap of
RoboTwin 2.0 (random background textures, table clutter, lighting, exposure).

Design contracts (operator set and parameter ranges follow the PatchWAM
clean-to-randomized recipe, arXiv 2609.25961, adapted to this repo's pipeline):

  * params are sampled per camera view, independently across views;
  * one sample's params are replayed for that camera's current AND future
    frames (``prepare_images`` forwards the same params dict to the
    future_image call), so future-frame supervision never sees per-frame
    appearance flicker;
  * purely photometric / statistical: no crop, rotation, or flip of any kind
    -- spatial targets (depth alignment, future-frame spatial prediction)
    stay valid and only RGB is ever touched;
  * 20% of samples pass through unmodified (``sample_c2r_params`` -> None);
  * applied only at train time; policy eval never augments (enforced by the
    existing ``train=self.image_augment and not policy_eval`` call sites).

All ranges live in this file so the recipe is a single auditable place.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F

RECIPE_NAME = "c2r_v1"

PASSTHROUGH_P = 0.2
BRIGHTNESS_RANGE = (0.6, 1.4)
CONTRAST_RANGE = (0.6, 1.4)
SATURATION_RANGE = (0.6, 1.4)
HUE_RANGE = (-0.03, 0.03)       # +-11 deg. Deliberately tighter than the
                                # PatchWAM +-0.1: the randomized eval setting
                                # varies backgrounds/lighting/clutter but never
                                # recolors task objects, so a wide hue shift
                                # trains on a domain the eval never presents
                                # while risking color-word grounding (acceptance
                                # grid task#02: +0.08 turned the red block
                                # orange under a "pick the red block"
                                # instruction). +-0.03 keeps mild lighting-color
                                # drift without crossing color classes.
GAMMA_RANGE = (0.7, 1.4)
EXPOSURE_STOPS_RANGE = (-0.5, 0.5)
COLOR_TEMP_RANGE = (-0.15, 0.15)  # R gain (1+t) / B gain (1-t)
BLUR_P = 0.3
BLUR_SIGMA_RANGE = (0.5, 1.5)
NOISE_STD = 0.03
ADAIN_P = 0.5
ADAIN_BLEND_RANGE = (0.5, 1.0)
ADAIN_MEAN_RANGE = (0.15, 0.85)
ADAIN_STD_RANGE = (0.10, 0.35)
FFT_P = 0.5
FFT_AMP_RANGE = (0.5, 1.5)
FFT_BANDS = 5


def _u(range_):
    lo, hi = range_
    return lo + (hi - lo) * torch.rand(()).item()


def sample_c2r_params() -> dict | None:
    """Sample one camera's augmentation config for one training sample.

    Returns None for the pass-through share of samples (identity, zero cost).
    """
    if torch.rand(()).item() < PASSTHROUGH_P:
        return None
    params = {
        "brightness": _u(BRIGHTNESS_RANGE),
        "contrast": _u(CONTRAST_RANGE),
        "saturation": _u(SATURATION_RANGE),
        "hue": _u(HUE_RANGE),
        "gamma": _u(GAMMA_RANGE),
        "exposure_stops": _u(EXPOSURE_STOPS_RANGE),
        "color_temp": _u(COLOR_TEMP_RANGE),
        "noise_std": NOISE_STD,
        "blur_sigma": _u(BLUR_SIGMA_RANGE) if torch.rand(()).item() < BLUR_P else None,
    }
    if torch.rand(()).item() < ADAIN_P:
        params["adain"] = {
            "mean": [_u(ADAIN_MEAN_RANGE) for _ in range(3)],
            "std": [_u(ADAIN_STD_RANGE) for _ in range(3)],
            "blend": _u(ADAIN_BLEND_RANGE),
        }
    if torch.rand(()).item() < FFT_P:
        params["fft_amp"] = [_u(FFT_AMP_RANGE) for _ in range(FFT_BANDS)]
    return params


def apply_c2r_augmentation(visual: Tensor, params: dict) -> Tensor:
    """Apply one sampled config to an image (C,H,W) or clip (T,C,H,W).

    Keeps the input's shape and dtype (uint8 round-trips exactly like the
    legacy color jitter in this package).
    """
    squeeze = visual.ndim == 3
    x = visual.unsqueeze(0) if squeeze else visual  # (T,C,H,W)
    orig_dtype = visual.dtype
    x = x.float()
    if float(x.max()) > 1.0:
        x = x / 255.0

    if params.get("blur_sigma") is not None:
        x = _gaussian_blur(x, params["blur_sigma"])
    x = x * params["brightness"]
    x = x * (2.0 ** params["exposure_stops"])
    t = params["color_temp"]
    if t != 0.0:
        gains = torch.ones(3, device=x.device, dtype=x.dtype)
        gains[0] = 1.0 + t  # R
        gains[2] = 1.0 - t  # B
        x = x * gains.view(1, 3, 1, 1)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * params["contrast"] + mean
    gray = x.mean(dim=1, keepdim=True)
    x = gray + (x - gray) * params["saturation"]
    if params["hue"] != 0.0:
        x = _shift_hue(x, params["hue"])
    x = x.clamp(min=0.0) ** params["gamma"]
    if "adain" in params:
        x = _adain(x, params["adain"])
    if "fft_amp" in params:
        x = _fft_amp_scale(x, params["fft_amp"])
    x = x + torch.randn_like(x) * params["noise_std"]
    x = x.clamp(0.0, 1.0)

    if orig_dtype == torch.uint8:
        x = (x * 255.0).round().to(torch.uint8)
    else:
        x = x.to(orig_dtype)
    return x.squeeze(0) if squeeze else x


# ---------------------------------------------------------------------------
# operator helpers (all operate on (T,C,H,W) float tensors in [0,1])
# ---------------------------------------------------------------------------

def _gaussian_kernel1d(sigma: float, device, dtype) -> Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    return k / k.sum()


def _gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    T, C, H, W = x.shape
    f = x.reshape(1, T * C, H, W)
    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    kh = k.view(1, 1, 1, -1).expand(T * C, 1, 1, -1)
    f = F.conv2d(f, kh, padding=(0, r), groups=T * C)
    kw = k.view(1, 1, -1, 1).expand(T * C, 1, -1, 1)
    f = F.conv2d(f, kw, padding=(r, 0), groups=T * C)
    return f.reshape(T, C, H, W)


def _shift_hue(x: Tensor, shift: float) -> Tensor:
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(dim=1)
    minc, _ = x.min(dim=1)
    v = maxc
    delta = maxc - minc
    s = torch.where(maxc > 0, delta / maxc.clamp(min=1e-8), torch.zeros_like(maxc))
    dz = delta.clamp(min=1e-8)
    rc = (maxc - r) / dz
    gc = (maxc - g) / dz
    bc = (maxc - b) / dz
    h = torch.where(
        r == maxc,
        bc - gc,
        torch.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc),
    )
    h = (h / 6.0 + shift) % 1.0
    # HSV -> RGB
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i.long() % 6
    choose = lambda a, b, c, d, e, f2: torch.where(
        i == 0, a, torch.where(i == 1, b, torch.where(i == 2, c, torch.where(i == 3, d, torch.where(i == 4, e, f2))))
    )
    out = torch.stack(
        [choose(v, q, p, p, t, v), choose(t, v, v, q, p, p), choose(p, p, t, v, v, q)],
        dim=1,
    )
    return out


def _adain(x: Tensor, cfg: dict) -> Tensor:
    C = x.shape[1]
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    tgt_mean = torch.tensor(cfg["mean"], device=x.device, dtype=x.dtype).view(1, C, 1, 1)
    tgt_std = torch.tensor(cfg["std"], device=x.device, dtype=x.dtype).view(1, C, 1, 1)
    stylized = (x - mean) / std * tgt_std + tgt_mean
    blend = cfg["blend"]
    return (1.0 - blend) * x + blend * stylized


_BAND_CACHE: dict[tuple[int, int], Tensor] = {}


def _band_map(H: int, W: int) -> Tensor:
    key = (H, W)
    bm = _BAND_CACHE.get(key)
    if bm is None:
        fy = torch.fft.fftfreq(H).abs() * 2.0
        fx = torch.fft.rfftfreq(W).abs() * 2.0
        r = torch.sqrt(fy.view(-1, 1) ** 2 + fx.view(1, -1) ** 2).clamp(max=1.0)
        bm = (r * FFT_BANDS).long().clamp(max=FFT_BANDS - 1)
        _BAND_CACHE[key] = bm
    return bm


def _fft_amp_scale(x: Tensor, band_amps) -> Tensor:
    T, C, H, W = x.shape
    bm = _band_map(H, W).to(x.device)
    factors = torch.tensor(band_amps, device=x.device, dtype=x.dtype)
    scale = factors[bm].unsqueeze(0).unsqueeze(0)  # (1,1,H,Wf)
    spec = torch.fft.rfft2(x, dim=(2, 3))
    amp = spec.abs()
    phase = spec / amp.clamp(min=1e-12)
    out = torch.fft.irfft2(amp * scale * phase, s=(H, W), dim=(2, 3))
    return out
