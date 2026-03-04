#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26"]
# ///
"""Generate a fully synthetic, deterministic HDR environment map for regression
testing.

The map is an equirectangular image encoding three components in HDR (linear,
physical-scale radiance):

  * Sky gradient  — deep blue at the horizon brightening to pale blue-white at
                    the zenith via a smooth power curve.
  * Sun disc      — a Gaussian spot at a configurable elevation/azimuth with
                    peak radiance comparable to the real sun (≈80 000 nit).
  * Ground fill   — flat mid-grey below the horizon to avoid pure-black
                    contribution from downward rays.

Output: a Radiance RGBE (.hdr) file written directly in numpy — no extra
Python dependencies beyond numpy itself.

Run with uv (no manual venv required):

    uv run tests/assets/gen_test_skymap.py
    uv run tests/assets/gen_test_skymap.py --width 1024 --height 512 \\
        --output tmp/regression/skyboxes/test_sky.hdr
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Radiance RGBE writer (no external dependency beyond numpy)
# ---------------------------------------------------------------------------

def _float_to_rgbe(img: np.ndarray) -> np.ndarray:
    """Encode a float32 ``(H, W, 3)`` linear-radiance image to ``(H, W, 4)``
    uint8 using the Radiance RGBE packing scheme.

    RGBE encoding: choose the shared exponent from the maximum channel, then
    scale each channel into [0, 255] sharing that exponent in the 4th byte.
    The exponent byte stores ``exp + 128`` (biased) so a scene-black pixel has
    ``E = 0``.
    """
    h, w, _ = img.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)

    max_c = np.max(img, axis=-1)  # (H, W)
    nonzero = max_c > 1e-32

    if np.any(nonzero):
        mantissa, exponent = np.frexp(max_c[nonzero])  # mantissa ∈ [0.5, 1)
        scale = mantissa * 256.0 / max_c[nonzero]       # maps max channel → 255
        out[nonzero, 0] = np.clip(img[nonzero, 0] * scale, 0, 255).astype(np.uint8)
        out[nonzero, 1] = np.clip(img[nonzero, 1] * scale, 0, 255).astype(np.uint8)
        out[nonzero, 2] = np.clip(img[nonzero, 2] * scale, 0, 255).astype(np.uint8)
        out[nonzero, 3] = np.clip(exponent + 128, 0, 255).astype(np.uint8)

    return out


def save_hdr(img: np.ndarray, path: str) -> None:
    """Write a float32 ``(H, W, 3)`` image to a Radiance HDR (.hdr) file.

    The output uses the `#?RADIANCE` magic and the standard
    `FORMAT=32-bit_rle_rgbe` header tag, followed by **flat (non-RLE)** RGBE
    scanlines.  The ``image`` crate's HDR decoder (used by `try_load_hdr`)
    detects the absence of per-scanline RLE magic bytes and falls back to raw
    mode automatically, so the file is read correctly.  Other viewers that
    strictly require RLE-compressed scanlines may reject this file; implement
    a proper scanline-RLE encoder (Walker's scheme) if broader compatibility
    is ever needed.
    """
    h, w, _ = img.shape
    rgbe = _float_to_rgbe(img)  # (H, W, 4) uint8

    header = (
        "#?RADIANCE\n"
        "FORMAT=32-bit_rle_rgbe\n"
        "\n"
        f"-Y {h} +X {w}\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rgbe.tobytes())


# ---------------------------------------------------------------------------
# Colour / radiance helpers
# ---------------------------------------------------------------------------

def _hex_to_linear(hex_color: str) -> np.ndarray:
    """Convert a CSS hex colour string (``#RRGGBB``) to a linear float32 RGB
    array.  The input is assumed to be in sRGB; the conversion uses the
    standard sRGB gamma ≈ 2.2 approximation (``v ** 2.2``).
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return np.array([r ** 2.2, g ** 2.2, b ** 2.2], dtype=np.float32)


def _lerp(
    a: "np.ndarray | float",
    b: "np.ndarray | float",
    t: "float | np.ndarray",
) -> np.ndarray:
    """Linear interpolation between ``a`` and ``b`` by scalar or array
    ``t`` ∈ [0, 1].  Both ``a`` / ``b`` may be arrays or plain floats.
    """
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# Sky model
# ---------------------------------------------------------------------------

def generate_skymap(
    width: int,
    height: int,
    *,
    horizon_hex: str = "#0033AA",
    zenith_hex: str = "#88BBFF",
    ground_hex: str = "#333333",
    sun_color_hex: str = "#FFEEDD",
    sun_elevation_deg: float = 45.0,
    sun_azimuth_deg: float = 30.0,
    sun_peak_radiance: float = 80_000.0,
    sun_sigma_deg: float = 2.0,
    sky_horizon_nit: float = 1.0,
    sky_zenith_nit: float = 3.0,
    ground_nit: float = 0.1,
) -> np.ndarray:
    """Synthesise an equirectangular HDR sky image.

    Parameters
    ----------
    width, height:
        Image dimensions in pixels.  Standard choices: 512×256, 1024×512.
    horizon_hex, zenith_hex, ground_hex, sun_color_hex:
        sRGB hex colours for the sky components.
    sun_elevation_deg:
        Sun centre elevation above the horizon (degrees).
    sun_azimuth_deg:
        Sun centre azimuth measured from the +X axis toward +Z (degrees).
    sun_peak_radiance:
        Peak radiance of the sun disc centre (nit / linear float scale).
    sun_sigma_deg:
        Gaussian standard deviation of the sun disc (degrees).  Use ≈2° for
        a sharp but anti-aliased edge.
    sky_horizon_nit, sky_zenith_nit:
        Luminance scale factors applied to the sky gradient at the horizon
        and zenith respectively.
    ground_nit:
        Constant luminance of the below-horizon ground fill.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(height, width, 3)`` containing physical-
        scale linear HDR radiance values ready for ``imageio`` to write as
        a Radiance `.hdr` file.
    """
    horizon_color = _hex_to_linear(horizon_hex)
    zenith_color  = _hex_to_linear(zenith_hex)
    ground_color  = _hex_to_linear(ground_hex)
    sun_color     = _hex_to_linear(sun_color_hex)

    # Pre-compute sun direction in Cartesian world space.
    # Convention: +Y is up, azimuth is measured in the XZ plane.
    sun_el  = math.radians(sun_elevation_deg)
    sun_az  = math.radians(sun_azimuth_deg)
    sun_dir = np.array([
        math.cos(sun_el) * math.cos(sun_az),
        math.sin(sun_el),
        math.cos(sun_el) * math.sin(sun_az),
    ], dtype=np.float64)

    sun_sigma_rad = math.radians(sun_sigma_deg)
    sun_sigma2    = 2.0 * sun_sigma_rad ** 2  # denominator in Gaussian exponent

    # Pixel coordinate grids — u in [0, width), v in [0, height).
    u = np.arange(width,  dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)  # both shape (height, width)

    # Spherical coordinates matching the equirectangular convention.
    phi   = (uu / width  - 0.5) * (2.0 * math.pi)   # azimuth  -π … +π
    theta = (0.5 - vv / height) * math.pi             # elevation -π/2 … +π/2

    # ---- Sky gradient -------------------------------------------------------
    # t = 0 at the horizon, 1 at the zenith; smooth power curve.
    sin_theta = np.sin(theta)
    t = np.maximum(sin_theta, 0.0) ** 0.6            # shape (H, W)

    # Interpolate colour and luminance simultaneously using vectorised ops.
    sky = (                                           # shape (H, W, 3)
        _lerp(horizon_color, zenith_color, t[:, :, np.newaxis])
        * _lerp(sky_horizon_nit, sky_zenith_nit, t)[:, :, np.newaxis]
    )

    # ---- Sun disc -----------------------------------------------------------
    # Cartesian direction for each pixel.
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_el  = np.cos(theta)
    sin_el  = np.sin(theta)

    # Direction vectors for every pixel:  (H, W, 3)
    px = cos_el * cos_phi
    py = sin_el
    pz = cos_el * sin_phi
    dirs = np.stack([px, py, pz], axis=-1)  # (H, W, 3)

    # Dot product with sun direction → cos(angle to sun).
    cos_angle = np.clip(
        dirs @ sun_dir,           # (H, W) · (3,) → (H, W)
        -1.0, 1.0,
    )
    angle     = np.arccos(cos_angle)                 # angular separation (rad)
    sun_val   = sun_peak_radiance * np.exp(-(angle ** 2) / sun_sigma2)

    # Sun is only visible above the horizon (theta ≥ 0).
    sun_val   = np.where(theta >= 0, sun_val, 0.0)

    # ---- Below-horizon ground fill -----------------------------------------
    above = (theta >= 0)[:, :, np.newaxis]           # (H, W, 1) bool mask
    sky   = np.where(above, sky, ground_color * ground_nit)

    # ---- Composite ----------------------------------------------------------
    pixel = sky + sun_val[:, :, np.newaxis] * sun_color   # (H, W, 3)

    return pixel.astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a synthetic HDR environment map for regression testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--width",  type=int, default=512,
                   help="Image width  in pixels.")
    p.add_argument("--height", type=int, default=256,
                   help="Image height in pixels.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/regression/skyboxes/test_sky.hdr"),
        help="Output path for the generated .hdr file.",
    )
    # Sky colours
    p.add_argument("--horizon-color", default="#0033AA",
                   metavar="HEX", help="sRGB hex colour at the horizon.")
    p.add_argument("--zenith-color",  default="#88BBFF",
                   metavar="HEX", help="sRGB hex colour at the zenith.")
    p.add_argument("--ground-color",  default="#333333",
                   metavar="HEX", help="sRGB hex colour below the horizon.")
    p.add_argument("--sun-color",     default="#FFEEDD",
                   metavar="HEX", help="sRGB hex colour of the sun disc.")
    # Sun position
    p.add_argument("--sun-elevation", type=float, default=45.0,
                   metavar="DEG", help="Sun elevation above the horizon (degrees).")
    p.add_argument("--sun-azimuth",   type=float, default=30.0,
                   metavar="DEG", help="Sun azimuth from +X toward +Z (degrees).")
    # Sun appearance
    p.add_argument("--sun-radiance",  type=float, default=80_000.0,
                   metavar="NIT", help="Peak radiance of the sun disc centre.")
    p.add_argument("--sun-sigma",     type=float, default=2.0,
                   metavar="DEG", help="Gaussian sigma of the sun disc (degrees).")
    # Sky luminance scale
    p.add_argument("--sky-horizon-nit", type=float, default=1.0,
                   help="Luminance scale of the sky at the horizon (nit).")
    p.add_argument("--sky-zenith-nit",  type=float, default=3.0,
                   help="Luminance scale of the sky at the zenith (nit).")
    p.add_argument("--ground-nit",      type=float, default=0.1,
                   help="Luminance scale of the below-horizon ground fill (nit).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    print(f"Generating {args.width}×{args.height} synthetic HDR skymap …")

    img = generate_skymap(
        args.width,
        args.height,
        horizon_hex=args.horizon_color,
        zenith_hex=args.zenith_color,
        ground_hex=args.ground_color,
        sun_color_hex=args.sun_color,
        sun_elevation_deg=args.sun_elevation,
        sun_azimuth_deg=args.sun_azimuth,
        sun_peak_radiance=args.sun_radiance,
        sun_sigma_deg=args.sun_sigma,
        sky_horizon_nit=args.sky_horizon_nit,
        sky_zenith_nit=args.sky_zenith_nit,
        ground_nit=args.ground_nit,
    )

    # Ensure the output directory exists.
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write as Radiance RGBE (.hdr) — compatible with the image crate and
    # imageio.  The format is the flat (non-RLE) variant of 32-bit_rle_rgbe.
    save_hdr(img, str(args.output))

    size_kb = args.output.stat().st_size / 1024
    print(f"Written {args.output}  ({size_kb:.1f} KB, "
          f"max_val={img.max():.1f}, min_val={img.min():.4f})")


if __name__ == "__main__":
    main()
