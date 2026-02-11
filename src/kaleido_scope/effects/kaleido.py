import math
from typing import Any

import torch
import torch.nn.functional as F

_GRID_CACHE: dict[
    tuple[tuple[str, int | None], torch.dtype, int, int],
    tuple[torch.Tensor, torch.Tensor],
] = {}


def _device_key(device: torch.device) -> tuple[str, int | None]:
    if device.type != "cuda":
        return (device.type, None)
    return (device.type, device.index)


def _get_base_grid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (_device_key(device), dtype, height, width)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    grid_y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    grid_x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    _GRID_CACHE[key] = (gx, gy)
    return gx, gy


def _as_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    if hasattr(x, "value"):
        return str(x.value)
    return str(x)


def kaleido_effect(
    frames: torch.Tensor,
    *,
    enabled: bool = True,
    mix: float = 1.0,
    mirror_mode: Any = "none",
    rotational_enabled: bool = True,
    rotational_slices: int = 6,
    rotation_deg: float = 0.0,
    zoom: float = 1.0,
    warp: float = 0.0,
) -> torch.Tensor:
    if not enabled or mix <= 0.0:
        return frames
    if frames.ndim != 4:
        raise ValueError("kaleido_effect expects frames in THWC format: (T, H, W, C)")

    T, H, W, _C = frames.shape
    device = frames.device

    gx, gy = _get_base_grid(H, W, device=device, dtype=torch.float32)
    u = gx
    v = gy

    if zoom != 1.0 and zoom > 0:
        u = u / float(zoom)
        v = v / float(zoom)

    if warp != 0.0:
        r2 = u * u + v * v
        factor = 1.0 + float(warp) * r2
        u = u * factor
        v = v * factor

    mm = _as_str(mirror_mode).lower()

    # --- NEW: "kaleido6" preset (your requested 6-kaleidoscope mode) ---
    if mm == "kaleido6":
        rotational_enabled = True
        rotational_slices = 6

    # Rotational folding
    if rotational_enabled and rotational_slices >= 3:
        r = torch.sqrt(u * u + v * v + 1e-8)
        theta = torch.atan2(v, u)
        theta = theta + math.radians(float(rotation_deg))

        wedge = 2.0 * math.pi / float(rotational_slices)
        phi = torch.remainder(theta, wedge)
        phi = torch.minimum(phi, wedge - phi)

        u = r * torch.cos(phi)
        v = r * torch.sin(phi)

    # Mirror modes
    if mm == "2x":
        u = torch.abs(u)
    elif mm == "4x":
        u = torch.abs(u)
        v = torch.abs(v)

    u = u.clamp(-1.0, 1.0)
    v = v.clamp(-1.0, 1.0)

    grid = torch.stack((u, v), dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0).expand(T, -1, -1, -1).to(device=device, dtype=frames.dtype)

    nchw = frames.permute(0, 3, 1, 2)
    sampled = F.grid_sample(
        nchw,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    out = sampled.permute(0, 2, 3, 1)

    if mix < 1.0:
        out = frames * (1.0 - float(mix)) + out * float(mix)

    return out.clamp(0.0, 1.0)
