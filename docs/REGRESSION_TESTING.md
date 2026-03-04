# Regression Testing

End-to-end image-based regression testing for `mcrt`.  The suite renders every
test scene from the current working copy, compares each output against a known-
good baseline, and fails automatically when a scene exceeds its configured
MSE/PSNR thresholds.

---

## Overview

The workflow is deliberately split into two independent steps to avoid
redundant re-renders and to make CI configuration straightforward:

```
Step 1 — Baseline capture (once, from a known-good commit)
          scripts/regress_baseline.sh --from <commitish>

Step 2 — Regression check (every time you want to validate)
          MCRT_REGRESSION_BASELINE=tmp/regression/baselines/<sha> \
          MCRT_REGRESSION_CURRENT=tmp/regression/current \
            cargo test regression -- --nocapture
```

The baseline images are **not committed** — they live in `tmp/regression/`
which is git-ignored.  Only the generator scripts and scene definition files
are committed, so the repository never accumulates large binary blobs.

---

## Quick Start

### 1. Build the release binary

```sh
cargo build --release
```

### 2. Capture a baseline from the last commit

```sh
scripts/regress_baseline.sh --from HEAD
```

This creates `tmp/regression/baselines/<short-sha>/` containing one PNG per
scene and a `manifest.json` with metadata.

### 3. Run the regression suite against your working copy

```sh
SHA=$(git rev-parse --short HEAD)
MCRT_REGRESSION_BASELINE=tmp/regression/baselines/$SHA \
MCRT_REGRESSION_CURRENT=tmp/regression/current \
  cargo test regression -- --nocapture
```

On the first run, each scene is rendered into `tmp/regression/current/`.  On
subsequent runs without code changes, the PNGs are reused (mtime-checked).

### 4. Force a clean re-render

```sh
MCRT_REGRESSION_FORCE=1 \
MCRT_REGRESSION_BASELINE=tmp/regression/baselines/$SHA \
MCRT_REGRESSION_CURRENT=tmp/regression/current \
  cargo test regression -- --nocapture
```

---

## Environment Variables

| Variable                   | Required | Default               | Description |
|----------------------------|----------|-----------------------|-------------|
| `MCRT_REGRESSION_BASELINE` | **Yes**  | —                     | Path to baseline directory (output of `regress_baseline.sh`) |
| `MCRT_REGRESSION_CURRENT`  | **Yes**  | —                     | Directory for current-render PNGs (created if absent) |
| `MCRT_REGRESSION_FORCE`    | No       | `0`                   | Set to `1` to re-render all scenes regardless of mtime |
| `MCRT_REGRESSION_DIFF_DIR` | No       | `tmp/regression/diff` | Directory for diff PNG output when a scene fails |

When `MCRT_REGRESSION_BASELINE` or `MCRT_REGRESSION_CURRENT` are **not set**,
the test is **skipped** — not failed.  This allows `cargo test` to run cleanly
on machines without a baseline.

---

## Pass / Fail Criteria

Each test scene has a sidecar `<scene>.test.toml` that defines its thresholds:

```toml
[regression]
threshold_mse  = 0.002   # fail if MSE  > this value
threshold_psnr = 38.0    # fail if PSNR < this value (dB)
```

Metrics are computed in the encoded sRGB (gamma) space of the 8-bit PNG, over
the R, G, and B channels only (alpha is ignored).  See `src/regression.rs` for
exact definitions.

| Metric        | Formula                               | Pass condition              |
|---------------|---------------------------------------|-----------------------------|
| MSE           | Mean of (a−b)² over all RGB samples   | `mse ≤ threshold_mse`       |
| PSNR          | 10·log₁₀(1/MSE) dB                    | `psnr ≥ threshold_psnr`     |
| Max delta     | Max |a−b| over all RGB samples        | Informational only           |

---

## Output

### Console output

Running with `-- --nocapture` prints progress for both steps and a Markdown
summary table:

```
=== Render step (baseline: tmp/regression/baselines/0c186bc) ===

[CACHED ] cornell_box_mini
[RENDER ] diffuse_sphere … ok
[RENDER ] emissive_box … ok
...

=== Compare step ===

[PASS   ] cornell_box_mini
[PASS   ] diffuse_sphere
...

## Regression summary — baseline `tmp/regression/baselines/0c186bc`

| Scene                  |       MSE | PSNR (dB) |   Max Δ | Status |
|------------------------|-----------|-----------|---------|--------|
| cornell_box_mini       |  0.000021 |     46.77 |  0.0235 | ✓ PASS |
| diffuse_sphere         |  0.000018 |     47.44 |  0.0198 | ✓ PASS |
| emissive_box           |  0.000089 |     40.51 |  0.0431 | ✓ PASS |
...
```

### Diff images

When a scene fails, a false-colour diff PNG is written to
`tmp/regression/diff/<scene>_diff.png`.  The heat-map palette encodes error
magnitude: black (identical) → blue → green → yellow → red (maximum difference).
An `amplify` factor of 4.0 is applied so that a quarter-range error appears as
fully red.

---

## Test Scenes

All scene definitions and their sidecars live in `tests/assets/scenes/`.

| Scene              | What it tests                              | SPP | MSE threshold | PSNR threshold |
|--------------------|--------------------------------------------|-----|---------------|----------------|
| `diffuse_sphere`   | Lambertian shading, indirect bounce        | 64  | 0.002         | 38.0 dB        |
| `mirror_sphere`    | Perfect specular reflection                | 64  | 0.002         | 38.0 dB        |
| `glass_sphere`     | Dielectric refraction, Fresnel             | 64  | 0.002         | 38.0 dB        |
| `emissive_box`     | Area-light emission, soft shadows          | 256 | 0.005         | 35.0 dB        |
| `cornell_box_mini` | Full Cornell Box at low SPP                | 64  | 0.002         | 38.0 dB        |
| `skymap_sphere`    | Environment-map importance sampling        | 128 | 0.003         | 36.0 dB        |

---

## Updating Baselines

After an **intentional** rendering change (e.g. a bug fix that improves quality,
a new material feature) the baseline must be regenerated:

```sh
# Re-capture baseline from the new HEAD (after committing the fix):
scripts/regress_baseline.sh --from HEAD

# Run the suite against the new baseline:
SHA=$(git rev-parse --short HEAD)
MCRT_REGRESSION_BASELINE=tmp/regression/baselines/$SHA \
MCRT_REGRESSION_CURRENT=tmp/regression/current \
  cargo test regression -- --nocapture
```

The old baseline directory (`tmp/regression/baselines/<old-sha>/`) can be left
in place or deleted — it is git-ignored and does not affect anything once the
env var points to the new one.

---

## Adjusting Thresholds

If a scene consistently exceeds its threshold due to inherent Monte Carlo
variance at low SPP (rather than a bug), you can:

1. Increase `spp` in `<scene>.test.toml` to reduce variance.
2. Relax `threshold_mse` / `threshold_psnr` slightly and re-capture the
   baseline.

Use `--nocapture` output to see the actual MSE/PSNR values and decide on
appropriate new thresholds.

---

## Architecture

```
src/regression.rs       — Image comparison (MSE, PSNR, heat-map diff)
                          + SceneThresholds / load_thresholds API
src/lib.rs              — Re-exports regression module for integration tests
src/headless.rs         — Headless renderer (--headless CLI)
tests/regression_tests.rs — cargo-test-compatible regression runner
tests/assets/scenes/    — Scene YAML + .test.toml sidecars
scripts/regress_baseline.sh — Baseline capture from a git commitish
tmp/regression/         — All generated artefacts (git-ignored)
  baselines/<sha>/      — Baseline PNGs + manifest.json
  current/              — Current-render PNGs
  diff/                 — Diff images written on failure
```

The `regression` module is compiled into both the binary crate and into the
library crate (`src/lib.rs`).  Integration tests import it as `mcrt::regression`
and directly call `compare_images`, `write_diff_image`, and `load_thresholds`.
The binary uses the same module (via `mod regression;` in `main.rs`) for future
command-line diff tooling.
