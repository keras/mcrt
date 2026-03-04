# Render Regression Testing Framework — Roadmap

Establish a deterministic, image-based regression testing framework that
catches unintended visual changes introduced by code modifications.  The
workflow compares rendered images produced by the current working copy against
a baseline rendered from an older commit, computing pixel-level diff metrics
and failing automatically when the difference exceeds a configured threshold.

> **Design goals:**
> - Zero GPU-window dependency — all rendering is headless and offline.
> - Scenes are small and cheap (low resolution) so the suite runs in
>   seconds in CI.
> - Baselines are generated on-demand from any git commitish, not stored as
>   binary blobs in the repository.
> - The comparison is perceptually meaningful (not raw bit-equality) to tolerate
>   minor floating-point variance across platforms.
> - All generated artefacts (rendered PNGs, diff images, baselines, synthesised
>   HDR skymaps) are written to `tmp/regression/` which is excluded from git.
>   Only source files (scene YAMLs, sidecar TOMLs, generator scripts) are
>   committed.

---

## Phase RT-1: Test Scene Assets

**Goal:** Produce a small set of carefully chosen YAML scene files whose visual
output exercises distinct rendering paths.  Each scene must be as simple as
possible while still providing meaningful coverage.

### Proposed scenes

| File | What it tests |
|------|---------------|
| `tests/assets/scenes/diffuse_sphere.yaml` | Lambertian shading, indirect bounce |
| `tests/assets/scenes/mirror_sphere.yaml` | Perfect specular reflection |
| `tests/assets/scenes/glass_sphere.yaml` | Dielectric refraction, Fresnel |
| `tests/assets/scenes/emissive_box.yaml` | Area-light emission, soft shadows — needs higher SPP (see note below) |
| `tests/assets/scenes/cornell_box_mini.yaml` | Full Cornell Box at low SPP |

### Per-scene metadata sidecar

Each YAML scene file is accompanied by a small `<scene>.test.toml` sidecar in
the same directory that controls regression-test parameters:

```toml
# tests/assets/scenes/diffuse_sphere.test.toml
[render]
width  = 128
height = 128
spp    = 64          # samples per pixel before the frame is saved

[regression]
threshold_mse    = 0.002   # mean squared error (linear, 0–1 range)
threshold_psnr   = 38.0    # minimum Peak Signal-to-Noise Ratio (dB)
```

### Tasks

- [x] Add `tmp/` to `.gitignore` (a single line `tmp/` covers all generated
  artefacts for the entire regression framework).
- [x] Create `tests/assets/scenes/` directory.
- [x] Author `diffuse_sphere.yaml` — single Lambertian sphere on a white floor,
  one point light above, sky colour background.
- [x] Author `mirror_sphere.yaml` — single perfectly specular sphere, same
  lighting as above.
- [x] Author `glass_sphere.yaml` — dielectric sphere (IOR 1.5) suspended in
  air, same lighting.
- [x] Author `emissive_box.yaml` — room with a ceiling emissive panel, no
  other light source.  Because emission-only scenes converge slowly, set
  `spp = 256` in its sidecar (4× the default) and use a looser threshold
  (`threshold_mse = 0.005`) to accommodate the remaining variance.
- [x] Author `cornell_box_mini.yaml` — trimmed-down copy of
  `assets/cornell-box.yaml` (remove complex geometry that adds render time).
- [x] Write a `<scene>.test.toml` sidecar for every scene above.
- [x] Add a brief `tests/assets/scenes/README.md` documenting the purpose of
  each scene and the sidecar format.

**Output:** A `tests/assets/scenes/` directory containing five scene+sidecar
pairs and a README.  No Rust code is changed.

---

## Phase RT-1b: Procedural Test HDR Skymap Generator

**Goal:** Produce a fully synthetic, deterministic `.hdr` environment map that
provides meaningful lighting for regression tests — a blue sky gradient with a
single bright sun disc — without depending on any external asset such as
`data/golden_gate_hills_2k.hdr`.

Using a generated skymap instead of a real photograph makes the test suite
fully self-contained and guarantees that the environmental lighting signal never
changes unless the generator script itself is modified.

### Map description

The synthesised equirectangular image encodes three components in HDR (linear,
physical-scale radiance):

| Component | Description |
|-----------|-------------|
| **Sky gradient** | Deep blue (`#0033AA`, ~1 nit) at the horizon, brightening to pale blue-white (`#88BBFF`, ~3 nit) at the zenith.  Interpolated via `elevation` angle using a smooth power curve. |
| **Sun disc** | A Gaussian spot centred at elevation 45 °, azimuth 30 °.  Peak radiance ~80 000 nit (comparable to real sun).  Gaussian σ ≈ 2 °, giving a sharp but anti-aliased edge. |
| **Below-horizon ground** | Flat mid-grey (~0.1 nit) for the lower hemisphere, preventing pure-black contribution from below-horizon rays. |

All values are linear float32 stored in standard Radiance RGBE (`.hdr`) format,
matching the format already read by `texture::try_load_hdr`.

### Generator script

A self-contained Python script at `tests/assets/gen_test_skymap.py`, managed
with **`uv`**.  Dependencies are declared inline using PEP 723 script metadata
so no separate virtual environment or `requirements.txt` is needed:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26"]
# ///
```

The HDR file is written using a self-contained RGBE encoder built on numpy;
no additional Python packages are needed.

Running the script is then a single, reproducible command — `uv` creates an
isolated environment automatically:

```
uv run tests/assets/gen_test_skymap.py [--width W] [--height H] [--output PATH]
# defaults: 512 × 256, tmp/regression/skyboxes/test_sky.hdr
```

#### Algorithm sketch

```python
for each pixel (u, v):
    phi   = (u / width  - 0.5) * 2π      # azimuth  -π … +π
    theta = (0.5 - v / height) * π        # elevation -π/2 … +π/2

    # Sky gradient
    t     = max(sin(theta), 0.0) ** 0.6   # 0 at horizon, 1 at zenith
    sky   = lerp(horizon_color, zenith_color, t)
    sky  *= lerp(1.0, 3.0, t)             # HDR brightening toward zenith

    # Sun disc
    cos_angle = dot(direction(phi, theta), sun_direction)
    sun_sigma = radians(2.0)
    sun_val   = 80_000 * exp(-((acos(cos_angle))^2) / (2 * sun_sigma^2))

    # Below-horizon ground fill
    if theta < 0:
        sky = ground_color  # flat grey

    pixel = sky + sun_val * sun_color
```

### Test scene

A new scene YAML `tests/assets/scenes/skymap_sphere.yaml` renders a single
white Lambertian sphere on a white floor lit exclusively by the synthetic
skymap.  There is no other light source, so all illumination comes from the
environment map — making it a direct regression test for the env-map sampling
and importance-weighting code paths.

```yaml
env_map: tmp/regression/skyboxes/test_sky.hdr

camera:
  look_from: [0.0, 1.5, 5.0]
  look_at:   [0.0, 1.0, 0.0]
  vfov:      45.0
  aperture:  0.0

materials:
  white:
    type: lambertian
    albedo: [0.9, 0.9, 0.9]

objects:
  - type: sphere
    center: [0.0, 1.0, 0.0]
    radius: 1.0
    material: white
  - type: plane
    normal:   [0.0, 1.0, 0.0]
    half_extents: [5, 5]
    material: white
    backface_culling: true
    transform:
      translate: [0.0, 0.0, 0.0]
```

Its sidecar `skymap_sphere.test.toml` uses a slightly higher SPP than the
simple scenes because env-map lighting converges more slowly:

```toml
[render]
width  = 128
height = 128
spp    = 128

[regression]
threshold_mse  = 0.003
threshold_psnr = 36.0
```

### Tasks

- [x] Write `tests/assets/gen_test_skymap.py` following the algorithm above.
  Add PEP 723 inline script metadata declaring `numpy` as the only
  dependency.  The HDR file is written by a self-contained RGBE encoder
  (no imageio required).  Parameterise sun direction, sun peak radiance,
  σ, sky colours, and output path via `argparse`.
- [x] Run the script with `uv run tests/assets/gen_test_skymap.py`; the HDR is
  written to `tmp/regression/skyboxes/test_sky.hdr` which is git-ignored.
  The generator script is the source of truth — the `.hdr` is a build artefact
  that is never committed.
- [x] Add a `Makefile` (or `scripts/gen_test_assets.sh`) target
  `gen-test-assets` that runs `uv run tests/assets/gen_test_skymap.py` and
  creates `tmp/regression/skyboxes/` if it does not exist.
- [x] Author `tests/assets/scenes/skymap_sphere.yaml` and its sidecar
  `skymap_sphere.test.toml`.
- [x] Add the new scene to `tests/assets/scenes/README.md`.
- [x] Verify visually: open the scene in the interactive renderer and confirm
  the sky gradient and sun highlight are visible on the sphere.

**Output:** `tmp/regression/skyboxes/test_sky.hdr` (generated, git-ignored),
and a new `skymap_sphere` scene+sidecar pair in `tests/assets/scenes/`
(committed).

---

## Phase RT-2: Headless Offline Renderer

**Goal:** Add a rendering mode that can run without a windowing system, reads a
scene file, accumulates exactly `spp` samples, then writes the result to a PNG
on disk.

### CLI interface

```
mcrt --headless <scene.yaml> --output <out.png> [--width W] [--height H] [--spp N]
```

CLI flags override any values found in a `.test.toml` sidecar.  Width, height,
and SPP fall back to sidecar values, then to built-in defaults (512×512, 64 spp)
if neither is provided.

### Architecture notes

- The existing wgpu render loop is event-driven and tied to `winit`.  The
  headless path should bypass `winit` entirely, creating a `wgpu::Device` and
  `wgpu::Queue` via a headless surface (using `wgpu::Instance::create_surface`
  on an offscreen target, or a plain storage-texture pipeline without a
  swapchain).
- Accumulation must be fully deterministic: seed the per-pixel RNG with
  `pixel_index * spp_count` (or equivalent) so that the same `(scene, spp)`
  pair always produces the same image on the same GPU.
- After the final sample, read back the accumulated texture into a CPU buffer,
  apply a simple gamma-2.2 tonemap (linear → sRGB), and encode it as a
  standard 8-bit sRGB PNG.  At 64–128 SPP the Monte Carlo noise floor
  completely dominates quantisation error, so 16-bit depth adds pipeline
  complexity for no measurable benefit.

### Tasks

- [x] Audit the current render pipeline to identify which parts are
  `winit`/surface-dependent and which are purely compute.
- [x] Design a `HeadlessRenderer` struct that holds the wgpu device/queue and
  the compute + display pipelines but owns a plain `wgpu::Texture` instead of
  a swapchain surface.
- [x] Implement the `--headless` CLI flag in `main.rs`; parse width, height,
  and SPP overrides.
- [x] Implement sidecar `.test.toml` discovery and parsing (look for
  `<scene_stem>.test.toml` next to the scene YAML).
- [x] Implement the accumulation loop: dispatch the path-trace compute shader
  `spp` times, accumulating into the existing float32 texture.
- [x] Implement texture readback: map the texture to CPU memory, apply
  per-channel gamma-2.2 tonemap (`v.powf(1.0/2.2).clamp(0.0,1.0)`),
  convert to `u8`, and encode as 8-bit sRGB PNG using the `image` crate.
- [x] Verify determinism: render the same scene twice consecutively and assert
  the output PNGs are byte-identical.

**Output:** `cargo run --release -- --headless tests/assets/scenes/diffuse_sphere.yaml --output tmp/regression/current/diffuse_sphere.png --spp 64` produces a valid PNG.

---

## Phase RT-3: Baseline Capture from a Git Commitish

**Goal:** Provide a helper command (or script) that checks out a given git
commitish to a temporary directory, builds it, and renders all regression
scenes, storing the results as the reference baseline.

### Workflow

```
mcrt-regress baseline --from HEAD^  [--scenes tests/assets/scenes/]
```

1. Resolve the commitish to a short SHA: `git rev-parse --short HEAD^`
   (produces e.g. `a1b2c3d`).
2. `git worktree add /tmp/mcrt-baseline-<short-sha> <short-sha>` — creates an
   isolated worktree without disturbing the current checkout.
3. Build the baseline binary inside that worktree:
   `cargo build --release --manifest-path /tmp/mcrt-baseline-<short-sha>/Cargo.toml`.
4. For each scene in the scenes directory, render using the baseline binary and
   write the output to `tmp/regression/baselines/<short-sha>/<scene_stem>.png`.
5. Remove the temporary worktree (`git worktree remove`).
6. Write a `tmp/regression/baselines/<short-sha>/manifest.json` recording the
   full commit SHA, short SHA, date, and per-scene render parameters used.

### Tasks

- [ ] Decide on the delivery mechanism: a subcommand of the main binary (`mcrt
  regress baseline`) or a standalone shell/Python script
  (`scripts/regress_baseline.sh`).  A shell script is simpler and avoids
  adding git-interaction logic to the Rust binary.
- [ ] Implement worktree creation, with error handling for already-existing
  worktrees (resume or remove-and-recreate).
- [ ] Implement sequential scene enumeration: find all `*.test.toml` sidecars
  and render the corresponding YAML with the commitish binary.
- [ ] Handle scene YAML schema changes gracefully: if the baseline binary
  rejects a scene file (non-zero exit, parse error logged to stderr), log a
  warning `[SKIP] <scene>: schema incompatible with baseline <short-sha>` and
  continue to the next scene.  Do **not** fail the baseline-capture run.
  The missing PNG will cause the later comparison step to log a skip for that
  scene rather than a failure.
- [ ] Implement `manifest.json` generation.
- [ ] Implement worktree cleanup (always, even on failure — use `trap` in
  shell).
- [ ] Test against a real prior commit to confirm images are produced and the
  worktree is cleaned up.

**Output:** After running the baseline command,
`tmp/regression/baselines/<short-sha>/` contains one PNG per scene plus a
`manifest.json`.  The working copy is untouched.  The `tmp/` tree is
git-ignored so baseline images are never accidentally staged.

---

## Phase RT-4: Image Comparison & Diff Metrics

**Goal:** Implement image comparison logic that reads two PNG files (current vs.
baseline) and computes perceptual diff metrics.

### Metrics

| Metric | Symbol | Description | Pass condition |
|--------|--------|-------------|----------------|
| Mean Squared Error | MSE | Average squared per-channel pixel difference (linear 0–1) | `mse ≤ threshold_mse` |
| Peak Signal-to-Noise Ratio | PSNR | $10 \log_{10}(1 / \text{MSE})$ in dB | `psnr ≥ threshold_psnr` |
| SSIM | — | Structural Similarity Index (optional stretch goal) | `ssim ≥ threshold_ssim` |
| Max pixel delta | — | Maximum absolute difference across all channels | informational |

Both images must have the same dimensions; a dimension mismatch is an
immediate failure.

### Diff image output

When a regression is detected, write a false-colour difference image to
`tmp/regression/diff/<scene_stem>_diff.png` where each pixel encodes the
magnitude of the per-channel error (scaled for visibility).  This helps with
visual debugging.  The `tmp/` tree is git-ignored so diff images never
clutter `git status`.

### Tasks

- [ ] Add `image` crate usage (already planned for Phase RT-2) for PNG I/O.
- [ ] Implement `fn compare_images(a: &DynamicImage, b: &DynamicImage) -> DiffResult` returning MSE, PSNR, and max delta.
- [ ] Implement false-colour diff image generation.
- [ ] Write unit tests for the comparison function with synthetic pixel data
  (identical images → MSE = 0; fully inverted image → expected MSE value).
- [ ] Verify the metrics against known reference values using at least one real
  scene pair.

**Output:** A `src/regression.rs` module (or `tests/regression/compare.rs`)
with a tested comparison function and diff-image writer.

---

## Phase RT-5: Regression Test Runner & CI Integration

**Goal:** Wire everything together into an automated test that can be run with
`cargo test` (or a dedicated script) and fails with a human-readable report
when images diverge too much from the baseline.

### Test runner design

The runner is split into two independent steps to avoid redundant re-renders:

```
# Step 1 — render current working copy (skips scenes whose PNG already exists)
mcrt-regress render  --scenes tests/assets/scenes/
                     --output tmp/regression/current/

# Step 2 — compare current renders against a baseline
mcrt-regress compare --baseline tmp/regression/baselines/<short-sha>/
                     --current  tmp/regression/current/
                     --output   tmp/regression/diff/
```

The `render` step checks whether `tmp/regression/current/<scene_stem>.png`
already exists **and** its modification time is newer than the scene YAML and
binary.  If so, it is reused and the render is skipped.  Pass `--force` to
always re-render.

The `compare` step does no rendering at all; it only reads PNGs from
`--current` and `--baseline` and computes metrics.

For each scene in the comparison:

1. If the baseline PNG is absent (schema change skipped it — see RT-3), log
   `[SKIP] <scene>: no baseline image` and continue.  Do **not** count as
   a failure.
2. Load both PNGs and verify they have the same dimensions; a mismatch is
   an immediate per-scene failure.
3. Compute diff metrics.
4. Compare against per-scene thresholds from the `.test.toml` sidecar.
5. Append a row to a summary table (scene name, MSE, PSNR, status).
6. If any scene fails, exit with code 1 and print the full summary.

If `--output` is given and a scene fails, write the diff PNG there.

### Integration with `cargo test`

Expose the runner as a Rust integration test in `tests/regression_tests.rs` so
that `cargo test regression` works out of the box.  The test requires two
environment variables; if either is absent the test is skipped (not failed):

```
MCRT_REGRESSION_BASELINE=tmp/regression/baselines/<short-sha> \
MCRT_REGRESSION_CURRENT=tmp/regression/current \
  cargo test regression
```

The integration test calls `mcrt-regress render` first (skipping already-fresh
images), then `mcrt-regress compare`, and reports any failures as test
assertions.

### CI workflow (GitHub Actions sketch)

```yaml
jobs:
  regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }   # full history needed for git worktree
      - run: cargo build --release
      - run: scripts/regress_baseline.sh HEAD^
      - run: |
          SHA=$(git rev-parse --short HEAD^)
          mcrt-regress render --scenes tests/assets/scenes/ \
                              --output tmp/regression/current/
          MCRT_REGRESSION_BASELINE=tmp/regression/baselines/$SHA \
          MCRT_REGRESSION_CURRENT=tmp/regression/current \
            cargo test regression
```

### Tasks

- [ ] Implement the `render` subcommand / script: enumerate scenes, skip
  already-fresh PNGs (mtime check), render the rest, write to
  `tmp/regression/current/`.  Support `--force` to bypass the cache.
- [ ] Implement the `compare` subcommand / script: load PNGs from `--current`
  and `--baseline`, compute metrics, skip scenes where the baseline PNG is
  absent (log `[SKIP]`), report failures.
- [ ] Generate a Markdown summary table and print it to stdout upon completion.
- [ ] Write `tests/regression_tests.rs`: the `cargo test`-compatible wrapper
  that skips gracefully when `MCRT_REGRESSION_BASELINE` or
  `MCRT_REGRESSION_CURRENT` are not set.
- [ ] Document the full workflow in `docs/REGRESSION_TESTING.md`:
  initial baseline capture, running comparisons, interpreting results,
  updating baselines after intentional changes.
- [ ] (Stretch) Add a GitHub Actions workflow file at
  `.github/workflows/regression.yml`.

**Output:** With a baseline captured and current renders already present,
`MCRT_REGRESSION_BASELINE=tmp/regression/baselines/<short-sha> MCRT_REGRESSION_CURRENT=tmp/regression/current cargo test regression`
passes on an unmodified codebase and fails (with a diff PNG written to
`tmp/regression/diff/` and a summary table) when a rendering change is
introduced.  Re-running the test suite without source changes reuses the
existing current renders without re-rendering.

---

## Dependency Summary

### Rust crates (`Cargo.toml`)

| Crate | Purpose | Already present? |
|-------|---------|-----------------|
| `image` | PNG read/write, pixel arithmetic | No — add to `Cargo.toml` |
| `serde` + `toml` | `.test.toml` sidecar parsing | Check — likely via `serde_yaml` already |
| `clap` | CLI subcommands | Check current `main.rs` |
| `serde_json` | `manifest.json` generation | No — add or use `toml` |

### Python scripts (PEP 723 inline metadata, run via `uv run`)

Python script dependencies are declared **inside each script** using PEP 723
inline metadata.  There is no shared `requirements.txt`, no `.venv` to
maintain, and no manual activation step — `uv` resolves and caches the
environment automatically on first run.

| Script | Python packages declared in script |
|--------|-------------------------------------|
| `tests/assets/gen_test_skymap.py` | `numpy>=1.26` (RGBE encoding done with numpy; no imageio needed) |

If a helper script is added later (e.g. a comparison visualiser), it follows
the same pattern: declare its own `# dependencies = [...]` block and run with
`uv run`.

---

## Open Questions

1. **RNG seeding strategy** — the current path tracer uses a per-frame seed.
   We need a strictly deterministic seed that does not depend on wall-clock
   time or frame counter so that the same scene + SPP always produces the same
   image.  Confirm the existing shader's RNG inputs before implementing Phase RT-2.

2. **GPU variance across machines** — floating-point rounding may differ
   between GPU vendors/drivers.  Consider whether the regression baselines
   should be machine-specific (captured and compared on the same machine / CI
   runner) or whether thresholds should be loose enough to tolerate
   cross-vendor variance.

3. **Baseline storage strategy** — all generated baselines live in `tmp/regression/`
   which is git-ignored, so they are never committed.  For CI, the baseline
   capture step (`scripts/regress_baseline.sh`) runs as part of the same job
   that does the comparison, so no persistent storage is needed.  If baselines
   ever need to be shared across CI jobs or preserved as artifacts, upload
   `tmp/regression/baselines/<short-sha>/` as a CI job artifact (e.g. GitHub
   Actions `upload-artifact`) rather than committing or using Git LFS.

4. **Parallel rendering** — for a larger scene suite, consider rendering scenes
   concurrently (one wgpu device per scene, or sequentially on one device).
   Phase RT-2 renders one scene at a time; parallelisation is a follow-on.
