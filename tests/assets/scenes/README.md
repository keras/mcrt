# Test Scene Assets

This directory contains the YAML scene files and their TOML sidecar metadata
used by the render regression testing framework (Phase RT).

## Scene inventory

| Scene file | What it tests |
|---|---|
| `diffuse_sphere.yaml` | Lambertian shading, indirect diffuse bounce |
| `mirror_sphere.yaml` | Perfect specular reflection (metal fuzz = 0) |
| `glass_sphere.yaml` | Dielectric refraction and Fresnel at IOR 1.5 |
| `emissive_box.yaml` | Area-light emission, soft shadows, colour bleed |
| `cornell_box_mini.yaml` | Full Cornell Box feature set at low SPP |

All scenes are deliberately simple: low primitive counts, no external mesh or
texture assets, and the default procedural sky gradient as background (no
`env_map`).  This keeps render times short and makes the framework fully
self-contained.

---

## Sidecar format (`<scene>.test.toml`)

Each scene YAML has a companion `.test.toml` file in the same directory.  The
sidecar controls how the regression test renders the scene and how strictly the
resulting image is compared to the baseline.

```toml
[render]
width  = 128    # output image width  in pixels
height = 128    # output image height in pixels
spp    = 64     # samples per pixel before the frame is saved

[regression]
threshold_mse  = 0.002  # maximum allowed mean squared error  (linear 0-1)
threshold_psnr = 38.0   # minimum required PSNR in decibels
```

### Field reference

#### `[render]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `width` | int | 128 | Rendered image width in pixels. |
| `height` | int | 128 | Rendered image height in pixels. |
| `spp` | int | 64 | Samples per pixel accumulated before saving. |

#### `[regression]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `threshold_mse` | float | 0.002 | Maximum mean squared error between candidate and baseline images.  Computed in linear (not gamma) space over all RGB channels.  Values above this threshold fail the test. |
| `threshold_psnr` | float | 38.0 | Minimum Peak Signal-to-Noise Ratio (dB).  A lower PSNR than this threshold fails the test. |

### Per-scene notes

**`emissive_box`** uses a higher SPP (`256`) and a looser MSE threshold
(`0.005`) because emission-only scenes converge more slowly than scenes with
direct-light contribution from the sky, and residual high-frequency noise from
Monte Carlo sampling would otherwise trigger false negatives.

---

## Design principles

- **No GPU window** — renderers run headless; scenes must load without any
  interactive window being present.
- **No external assets** — scenes intentionally omit `env_map` and OBJ meshes
  so the test suite works in fresh checkouts without downloading large files.
- **Stable baselines** — baselines are generated from a known commit using
  `tmp/regression/baselines/` (excluded from git via `tmp/` in `.gitignore`).
  Only source files in this directory are version-controlled.
- **Perceptual thresholds** — MSE and PSNR are used instead of byte-exact
  comparison to tolerate minor floating-point variance across platforms.
