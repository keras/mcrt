# World-Space Irradiance Cache — Roadmap

A GPU-accelerated irradiance caching system layered on top of the existing path
tracer, drawing on ideas from DDGI (Majercik et al., 2019) and Unreal Engine's
Lumen.  The goal is to replace the costly multi-bounce diffuse path with a
structured, updateable world-space cache of indirect lighting — enabling
real-time or near-real-time GI.

> **Design note:** This implementation uses **L2 Spherical Harmonics** for
> per-probe irradiance storage rather than the octahedral irradiance maps used
> in the original DDGI papers.  SH was chosen for simplicity (flat buffer, no
> texture atlas management, no border-texel handling) and because L2 SH captures
> >99 % of diffuse irradiance energy (Ramamoorthi & Hanrahan, 2001).  The
> trade-off is lower angular resolution per probe compared to octahedral maps,
> which is acceptable for smooth indirect lighting.

---

## Background: Lumen Architecture (Simplified)

Unreal Engine's Lumen answers *"how much indirect light arrives at this surface
point?"* through a multi-layered system.  The diagram below is a simplified
overview — Lumen's full pipeline is tightly coupled to UE5's Nanite, Mesh
Cards, and virtual texturing, which are out of scope here.

```
Final Gather (screen probes placed at ~16×16 pixel intervals)
  Each screen probe shoots a small fan of rays through:
    Screen traces (Hi-Z ray march, cheapest)
          ↓ miss
    Mesh SDF traces (per-object signed-distance fields)
          ↓ miss
    Global SDF (voxelised whole-scene SDF, far field)
  For far-field contributions:
    → Sample from World-Space Radiance Cache
        (clipmap cascade of world-space probes, updated per frame)
```

The Final Gather is the **orchestrator** — it drives both short-range traces
and far-field cache lookups.  The Radiance Cache is a read-only input to the
Final Gather, not a downstream stage.

### Ideas borrowed (loosely) for this implementation

| Lumen concept                | Adaptation in this project                                  |
|------------------------------|-------------------------------------------------------------|
| World-Space Radiance Cache   | SH irradiance probe grid (see Design note above)           |
| Clipmap cascade probes       | Multi-resolution probe cascades around camera (IC-5)       |
| Probe radiance capture       | Compute pass: shoot ray fans per probe, accumulate SH      |
| Visibility / depth encoding  | Per-probe depth octahedron — Chebyshev weighting (IC-4)    |
| Surface Cache (Mesh Cards)   | Simplified surface lightmap on mesh UVs (IC-6) — *not* equivalent to Lumen's Mesh Card shading cache |
| Screen traces                | Short-range screen-space irradiance gather (SSGI, IC-7)    |

> **Lumen's Surface Cache** in UE5 uses axis-aligned Mesh Card projections (up
> to 6 per mesh) to create a compact shading atlas for accelerating ray-hit
> evaluation.  Our IC-6 takes the simpler lightmap-on-UV approach for static
> geometry, which serves a different purpose (caching *irradiance* rather than
> accelerating *material lookups*).

---

## Glossary

| Term | Definition |
|------|-----------|
| **Irradiance probe** | World-space point that stores pre-integrated incoming radiance in all directions as a compact representation (L2 SH in this implementation). |
| **Spherical Harmonics (SH)** | A basis for low-frequency signals on the sphere; L2 SH (9 coefficients per colour channel) captures smooth indirect lighting with >99 % energy accuracy for diffuse irradiance (Ramamoorthi & Hanrahan, 2001). |
| **DDGI** | Dynamic Diffuse Global Illumination (Majercik et al., 2019).  Uses octahedral irradiance/depth maps in a texture atlas.  This roadmap adapts DDGI's probe placement, update scheduling, and visibility weighting concepts but substitutes SH for irradiance storage. |
| **Octahedral map** | Equal-area 2-D projection of a spherical signal; used in this implementation for per-probe depth encoding (IC-4). |
| **Cascade** | A concentric shell of probe grids at increasing spacing; inner cascades are fine-grained, outer are coarse. |
| **Time-slice update** | Updating a fixed budget of probes per frame rather than all at once. |
| **Backface weighting** | During probe interpolation, down-weight probes where `dot(probe_to_point, surface_normal) < 0`, indicating the probe is behind the surface — a critical light-leak reduction technique alongside Chebyshev visibility. |
| **Probe relocation** | Detecting probes embedded inside geometry and shifting them to valid positions; prevents probes from contributing incorrect lighting from inside walls. |
| **Self-shadow bias** | A small offset along the surface normal applied to the query point before probe lookup, preventing self-shadowing artifacts at surface/probe boundaries. |
| **Surface cache** | A per-surface lightmap whose texels are updated lazily and used during shading. |
| **SSGI** | Screen-Space Global Illumination — a screen-space ray-march that cheaply resolves nearby indirect details. |

---

## Phase IC-1: Probe Grid Foundation

**Goal:** Lay the CPU/GPU data structure foundation for a uniform 3-D grid of
irradiance probes and visualise their placement on screen.

### Tasks

- [ ] Define `IrradianceProbe` GPU struct:
  - `position: [f32; 4]` (world-space centre; w = cascade index)
  - `radiance_offset: u32` (index into the flat irradiance buffer)
  - `depth_offset: u32`
  - `flags: u32` (dirty / valid / invalid / needs-relocation / cascade)
  - 4-byte pad
- [ ] Define `ProbeGrid` CPU struct:
  - `origin: Vec3`, `spacing: f32`, `dimensions: [u32; 3]`
  - Probe array: `Vec<IrradianceProbe>`
  - Helper: `probe_index(x, y, z) -> u32`
- [ ] Allocate two flat GPU storage buffers:
  - `irradiance_buffer`: `N × 9 × 3 × f32` (L2 SH, 9 coeffs × RGB per probe)
  - `depth_buffer`: `N × 16 × 16 × 2 × f32` (octahedral depth: R = mean distance, G = mean distance² — variance derived at lookup time as `mean² − mean²`)
- [ ] Write WGSL struct `Probe` and accessor helpers used in later passes
- [ ] Upload an initial grid centred on the scene AABB (computed from
  `GpuSphere` + `GpuVertex` bounds)
- [ ] Mark all probes dirty on scene load / camera warp

> **Bind group constraint:** The compute pass already uses 14 of 16 bindings.
> Adding `irradiance_buffer` and `probe_grid_uniform` fills the group exactly.
> Phase IC-4+ will require a **second bind group** for depth atlas, surface
> cache, and other probe-related resources.

#### Debug Visualisation

- [ ] Render probe positions as small spheres in the path-trace or a separate
  debug compute pass — colour-encode by cascade / validity / relocation state
- [ ] Add egui checkbox **"Show probe grid"** to toggle the overlay
- [ ] Add egui slider **"Probe spacing"** to adjust grid density at runtime
  (triggers full cache invalidation)

**Output:** Probe grid visible as coloured dots overlaid on the rendered scene.

---

## Phase IC-2: Probe Radiance Capture

**Goal:** Implement the GPU compute pass that shoots rays from each probe and
accumulates the result into its SH irradiance representation.

### Algorithm

For each probe scheduled for update this frame:
1. Distribute `NUM_RAYS_PER_PROBE` (e.g., 128 or 256) directions uniformly over
   the sphere using a Fibonacci lattice, then apply a **per-frame random
   rotation** to the entire direction set (essential for temporal convergence —
   a static lattice always misses features between sample directions).
2. For each ray, trace the existing BVH scene:
   - On hit → evaluate direct lighting (same NEE as `path_trace.wgsl`) + one
     bounce of diffuse using the **surrounding probes' current irradiance
     estimates** at the hit point (bootstrapping — this is how multi-bounce
     propagates across frames without tracing additional ray bounces).
   - On miss → sample the HDR environment map.
3. Project the returned radiance sample onto L2 spherical harmonics and
   accumulate via a hysteresis blend (exponential moving average):
   `SH_new = lerp(SH_old, SH_sample, α)` where `α ≈ 0.02–0.04` (i.e.,
   hysteresis ~0.97 for the old value).
   - **Asymmetric hysteresis:** when the new sample's luminance is significantly
     *lower* than the current stored value, increase α (blend faster toward
     darkness) so that lights turning off or objects blocking light are reflected
     quickly.  Suggested: `α_dark = min(α * 4, 0.25)`.
4. Simultaneously record the ray hit distance into the octahedral depth map
   (for Phase IC-4 light-leak reduction): store `distance` in R and `distance²`
   in G at the octahedral texel for the ray direction.

> **Monte Carlo weighting:** Each frame's `SH_sample` must be the properly
> weighted average of the ray batch.  For `N` uniformly distributed rays over
> the sphere, each sample contributes with weight `4π / N`.  After summing
> weighted SH projections, divide by `N` (or equivalently, multiply by
> `4π / N²` total) to produce the per-frame irradiance estimate before blending
> with the EMA.

### Tasks

- [ ] Write `probe_update.wgsl` compute shader:
  - Workgroup: one probe per workgroup, `NUM_RAYS_PER_PROBE` invocations
  - Shared memory: per-thread SH accumulator; reduce at end of workgroup
  - Bindings: scene BVH, vertices, indices, materials, environment map,
    probe grid uniform, irradiance buffer (read/write), depth buffer (read/write)
- [ ] Implement per-frame random rotation matrix (uniform random axis + angle,
  seeded from `frame_count`; apply to all ray directions in the workgroup)
- [ ] Implement `sh_project(dir: vec3f, radiance: vec3f) -> array<vec3f, 9>` in WGSL
- [ ] Implement `sh_irradiance(coeffs: array<vec3f, 9>, normal: vec3f) -> vec3f`
  — used at shading time (ZH-convolved evaluation: band 0 × π, band 1 × 2π/3,
  band 2 × π/4)
- [ ] Wire up a new `ProbeUpdatePass` in `gpu.rs`:
  - Dispatch with `ceil(N_dirty_probes / WORKGROUP_SIZE)` workgroups
  - CPU-side dirty list: only update probes flagged as dirty or due for refresh
- [ ] Time-slice: cap the per-frame update budget to `MAX_PROBES_PER_FRAME` (e.g., 64)
- [ ] Expose `MAX_PROBES_PER_FRAME` in the egui panel

**Output:** Probe SH coefficients converge to correct indirect lighting values
over several frames.  Indirect lighting visible on diffuse surfaces when probes
are sampled.

---

## Phase IC-3: Probe Interpolation — Indirect Diffuse Lighting

**Goal:** Use the probe grid at the *first diffuse hit* in `path_trace.wgsl` to
answer indirect lighting queries instead of tracing additional bounces.

### Algorithm

At a surface hit point `P` with normal `N`:
1. Apply **self-shadow bias**: offset the query point along the normal —
   `P_biased = P + N * SELF_SHADOW_BIAS` (suggested: `SELF_SHADOW_BIAS ≈ 0.1 ×
   probe_spacing`) to prevent probes behind the surface from dominating.
2. Identify the 8 surrounding probe grid cells (trilinear neighbourhood).
3. For each of the 8 probes, evaluate `sh_irradiance(probe.sh, N)`.
4. Blend the 8 contributions using trilinear weights, multiplied by:
   - **Probe validity weight**: skip probes with `flags & FLAG_INVALID`
   - **Normal-based backface weight**: `max(0.0001, dot(normalize(P - probe.pos), N))`
     — down-weights probes that are on the wrong side of the surface (a critical
     light-leak reduction technique, complementary to IC-4's Chebyshev test)
5. Multiply by the surface `albedo / π` to convert irradiance to exitant radiance.
6. Add direct lighting on top (still use NEE from Phase 13).

### Tasks

- [ ] Add `irradiance_buffer` and `probe_grid_uniform` bindings to `path_trace.wgsl`
  (uses the last 2 available slots in the existing bind group — see IC-1 note)
- [ ] Implement `sample_irradiance_cache(p: vec3f, n: vec3f) -> vec3f` in WGSL
  - Trilinear interpolation with 8-probe neighbourhood
  - Self-shadow bias on input position
  - Normal-based backface weighting per probe
  - Skip probes with `flags & FLAG_INVALID`
- [ ] Add a render mode toggle: **"Use irradiance cache"** vs **"Full path trace"**
  (egui radio or key shortcut `I`)
- [ ] In cache mode, limit path bounces to 1 (direct) + cache lookup for
  higher bounces — this is the primary performance gain
- [ ] Verify energy conservation: cache-lit surfaces should roughly match the
  fully path-traced ground truth after cache convergence
- [ ] Add a debug visualisation: **"Irradiance only"** (no direct, probe lookup only)

**Output:** Visually indistinguishable indirect lighting from the cache compared
to ground-truth path tracing, achieved in a fraction of the per-pixel ray budget.

---

## Phase IC-4: Visibility Encoding & Light-Leak Reduction

**Goal:** Add per-probe depth/visibility information to suppress light leaking
through thin walls and across shadow boundaries.

### Algorithm (Chebyshev visibility weighting)

Each probe stores an octahedral map of hit distances (written during IC-2):
- `depth_mean[dir]` — mean ray hit distance in that direction (R channel)
- `depth_mean_sq[dir]` — mean of distance² (G channel)

Variance is derived at lookup time: `variance = mean_sq - mean * mean`

During interpolation (Phase IC-3), each probe's contribution is additionally
weighted by a *visibility weight*:
```
dist = length(P - probe.pos)
dir  = normalize(P - probe.pos)
mean, mean_sq = sample_depth_octahedron(probe, dir)
variance = max(0, mean_sq - mean * mean)
w_vis = variance / (variance + max(0, dist - mean)²)  // Chebyshev upper bound
```
Probes whose stored depth suggests geometry between the probe centre and the
query point are down-weighted, preventing light from penetrating walls.

### Probe Relocation

Probes that are consistently embedded inside geometry (all/most rays hit at
very short distances) should be detected and relocated:
- [ ] Each update cycle, track the fraction of rays hitting within a short
  threshold (e.g., < 0.1 × probe_spacing)
- [ ] If > 75 % of rays are short hits, flag the probe as `NEEDS_RELOCATION`
- [ ] CPU pass: shift relocated probes along the direction of least obstruction
  (average of the longest-distance ray directions)
- [ ] If no valid position is found, mark as `INACTIVE` and exclude from
  interpolation entirely

### Tasks

- [ ] Extend `probe_update.wgsl` to write hit distances into the octahedral depth map
  - Allocate a `depth_atlas` 2-D texture array: `N × DEPTH_RES × DEPTH_RES × rg32float`
  - `DEPTH_RES = 16` (mean distance in R, mean distance² in G)
  - Store with the same hysteresis blend as irradiance
- [ ] Implement `chebyshev_weight(mean: f32, mean_sq: f32, dist: f32) -> f32` in WGSL
  - Derive variance internally: `var = max(0, mean_sq - mean * mean)`
  - Return `var / (var + max(0, dist - mean)²)`
  - Clamp output to `[0, 1]`, apply `CHEBYSHEV_BIAS` to avoid hard cut-offs
- [ ] Modify `sample_irradiance_cache` to multiply each probe weight by:
  `chebyshev_weight(...)` × backface weight (from IC-3)
- [ ] Add octahedral encode/decode functions (`oct_encode`, `oct_decode`) as shared WGSL utility
- [ ] Implement probe relocation detection + CPU-side shift (see above)
- [ ] Test on the Cornell Box scene (lights-through-walls scenario)
- [ ] Expose `DEPTH_RES`, `CHEBYSHEV_BIAS`, and relocation threshold as tunable constants
- [ ] Requires **second bind group** for depth atlas texture (see IC-1 bind group note)

**Output:** No more light bleeding through thin walls or across shadow
boundaries; probes inside geometry are relocated or deactivated.

---

## Phase IC-5: Multi-Resolution Cascade System

**Goal:** Support multiple nested probe grids (cascades) at increasing spacing
to cover both nearby fine-detail GI and distant ambient lighting.

### Cascade Design

Based on the DDGI cascade extension (Majercik et al., JCGT 2021), using a 2×
spacing ratio per level with equal grid resolution across cascades:

| Cascade | Spacing | Grid Dims | Update rate | Coverage |
|---------|---------|-----------|-------------|----------|
| 0 (inner) | `s`    | 16³       | Every frame  | ~16s volume around camera |
| 1         | `2s`   | 16³       | Every 2 frames | ~32s volume |
| 2 (outer) | `4s`   | 16³       | Every 4 frames | ~64s volume (far field / sky) |

- Cascades follow the camera; grids scroll in world-space as the camera moves
  (toroidal/wrapping address scheme to avoid data copies — shift the logical
  origin and wrap indices).
- When the camera moves enough to advance the inner grid by one cell, shift
  the grid origin and mark the newly uncovered ring of probes dirty.
- The 2× spacing ratio (rather than 4×) reduces visible quality transitions
  at cascade boundaries.

### Tasks

- [ ] Generalise `ProbeGrid` to `ProbeGridCascade(level: u32, spacing: f32, ...)`
- [ ] Allocate separate irradiance buffers and depth atlases per cascade
- [ ] Implement grid scrolling: detect camera movement > 1 cell, translate
  grid origin (toroidal wrap), invalidate the newly uncovered ring of probes
- [ ] During shading: select cascade based on distance of query point to camera
  origin; blend between cascades at boundaries for smooth transitions
- [ ] In `probe_update.wgsl`, support reading a coarser cascade's SH when
  computing indirect for a finer cascade (ensures multi-bounce information
  propagates inward)
- [ ] Expose cascade count and spacing ratio as runtime parameters in egui
- [ ] Visualise cascade boundaries using coloured debug overlays

**Output:** High-resolution indirect lighting near the camera with automatic
fall-off to low-resolution at distance; no visible seams at cascade boundaries.

---

## Phase IC-6: Surface Irradiance Cache (Virtual Lightmap)

**Goal:** Cache irradiance at surface texels (traditional lightmapping style),
providing stable, low-noise indirect lighting on static geometry without
per-pixel probe lookups.

> **Relation to Lumen's Surface Cache:** Lumen uses axis-aligned Mesh Card
> projections to create a shading cache for cheap ray-hit material evaluation.
> Our approach is simpler: a UV-space lightmap that caches *irradiance* on
> static geometry, falling back to the probe grid for dynamic objects.  This
> avoids the Mesh Card generation pipeline but does not accelerate ray-hit
> shading the way Lumen's Surface Cache does.

### Concept

- Each mesh triangle is assigned a unique 2-D texel region within a large
  `surface_irradiance_atlas` texture (a virtual lightmap).
- A compute pass selects a budget of atlas texels to update per frame
  (time-sliced lightmap baking).
- During shading, the first-hit surface samples the atlas texel for its
  indirect irradiance before falling back to the probe grid.
- Static geometry converges to a fully baked lightmap appearance over time;
  dynamic objects always use the probe grid.

### Tasks

- [ ] Implement UV atlas packing on the CPU (`AtlasPacker`):
  - Assign each mesh a unique rect in a power-of-two atlas (e.g., 2048²)
  - Store per-triangle `(atlas_uv_origin, atlas_uv_extent)` in an extra
    vertex attribute buffer
- [ ] Allocate `surface_irradiance_atlas: Rgba32Float` GPU texture
- [ ] Write `surface_cache_update.wgsl` compute pass:
  - Each invocation corresponds to one atlas texel
  - Reconstruct world-space position + normal from stored data
  - Shoot `NUM_SURFACE_RAYS` hemisphere rays and accumulate irradiance
  - Blend result into the atlas texel (exponential moving average)
- [ ] Modify `path_trace.wgsl` to look up the atlas at first diffuse hit
  (fall back to probe grid if atlas UV is unavailable or texel is stale)
- [ ] Tag mesh objects as `static` or `dynamic` in the YAML scene format;
  only static meshes participate in the surface cache
- [ ] Expose atlas resolution and per-frame texel budget in egui
- [ ] Debug view: render the raw atlas texture in a separate egui window

**Output:** Near-noiseless indirect lighting on static geometry, convergent
over seconds, with dynamic objects smoothly blending in probe-grid GI.

---

## Phase IC-7: Screen-Space Irradiance Gather (SSGI)

**Goal:** Implement a short-range screen-space AO/GI pass that fills in
high-frequency contact shadows and local colour bleeding that probe grids cannot
resolve, complementing the world-space cache.

### Algorithm

For each screen pixel:
1. Reconstruct world-space position and normal from the G-buffer (already
   available from Phase 14: `gbuffer_normal_depth`).
2. Cast `NUM_SS_RAYS` short rays using depth-buffer reprojection (ray march
   along screen-space position).
3. Accumulate irradiance from hit texels.
4. Blend SSGI result with the world-space probe lookup:
   `indirect_final = lerp(probe_irradiance, ssgi_irradiance, ss_confidence)`
   where `ss_confidence` depends on the fraction of rays that stayed on screen.

### Tasks

- [ ] Write `ssgi.wgsl` compute pass executed after `path_trace.wgsl` but
  reading the G-buffer textures produced in that pass
- [ ] Implement screen-space ray-march (linearised depth buffer)
- [ ] Compute a `confidence` map based on how many SSGI rays resolved on screen
- [ ] Blend with the probe grid result in a `combine.wgsl` pass (or extend `display.wgsl`)
- [ ] Add temporal reprojection for SSGI to reduce noise across frames
  (reproject using camera motion vectors)
- [ ] Expose SSGI sample count, max distance, and blend weight in egui
- [ ] Add toggle **"Enable SSGI"** (key shortcut `G`) for A/B comparison

**Output:** High-quality contact shadows, colour bleeding in corners, and
improved local indirect detail with little additional ray budget.

---

## Phase IC-8: Performance, Stability & Polish

**Goal:** Optimise the cache system for real-time performance, reduce temporal
flickering, and expose production-quality controls.

### Performance Optimizations

- [ ] **Probe culling**: cull probes whose probe-to-camera ray is fully occluded
  (no visible surfaces in range); mark them low-priority for updates
- [ ] **Asynchronous readback**: overlap probe update dispatch with display
  render using wgpu's async queue submission model
- [ ] **Hierarchical update scheduling**: prioritise probes that are
  visible from the camera position (frustum-cull the dirty list)
- [ ] **Compressed SH storage**: pack 9 × RGB f32 coefficients into f16 to
  halve the irradiance buffer bandwidth (accumulate in f32, store in f16;
  f16 precision is sufficient — industry standard in Filament, UE, NVIDIA DDGI SDK)
- [ ] **Temporal reprojection for probes**: when the camera moves, reproject
  previous frame's probe irradiance as a warm-start to reduce re-convergence time
- [ ] Profile with `wgpu` timestamp queries; expose per-pass GPU times in egui

### Temporal Stability

- [ ] Clamp maximum SH coefficient change per frame to avoid sudden irradiance
  pops when a large emitter enters the scene
- [ ] Implement probe relighting hysteresis — weight sudden light changes
  smoothly over multiple update cycles (asymmetric: faster toward darkness,
  see IC-2 algorithm notes)
- [ ] Add optional **temporal anti-flicker** pass: reproject last frame's
  indirect buffer and blend (Catmull–Rom resampling)
- [ ] Detect disocclusion (large depth differences between reprojection and
  current depth) and fall back to fresh probe lookups in those regions

### Quality Controls (egui)

- [ ] Probe cache panel:
  - Cascade count, inner spacing, update rates
  - Rays per probe, hysteresis blend factor
  - Cache mode (full path trace / 1-bounce + cache / cache only)
- [ ] SSGI panel: sample count, max trace distance, blend strength
- [ ] Surface cache panel: atlas resolution, per-frame texel budget, static flag per object
- [ ] GI debug views: probes, cascade boundaries, irradiance only, SSGI only,
  surface cache atlas

**Output:** Stable, interactive GI with less than 10% frame-time overhead over
the direct-only renderer; full set of quality/debug controls exposed in the UI.

---

## Phase IC-9: Validation & Integration Tests

**Goal:** Ensure correctness, energy conservation, and regression safety.

### Tasks

- [ ] **Energy conservation test**: a fully Lambertian sphere inside a white
  box must converge to a uniform 1.0 irradiance field; assert maximum deviation < 5 %
- [ ] **Light-leak test**: render the Cornell Box; measure irradiance inside
  the box from the outside — must be < threshold
- [ ] **Convergence speed test**: record samples-to-convergence for the Cornell
  Box with and without the cache; document speed-up factor
- [ ] **Visual regression**: save reference images per phase; compare with
  golden images in CI using structural similarity (SSIM)
- [ ] **Multi-GPU compatibility**: test on NVIDIA (Vulkan/Metal), Apple Silicon
  (Metal), and software rasteriser (DX12 / Vulkan CPU fallback)
- [ ] Add `cargo test irradiance` suite for CPU-side logic (SH projection,
  SH evaluation, cascade grid arithmetic, probe relocation, atlas UV packing)

**Output:** Verified, regression-tested irradiance cache subsystem with
documented performance characteristics.

---

## Architecture Integration

```
┌──────────────────────────────────────────────────────────────┐
│                        CPU (Rust)                            │
│                                                              │
│  ProbeGridCascade × 3  ──► dirty list ──► ProbeUpdatePass   │
│  ProbeRelocator         ──► shift list ──► (CPU adjust)     │
│  AtlasPacker            ──► UV map    ──► SurfaceCachePass   │
│  CameraState movement   ──► grid scroll / probe invalidation │
└──────────────────────┬───────────────────────────────────────┘
                       │ wgpu
┌──────────────────────▼───────────────────────────────────────┐
│                        GPU (WGSL)                            │
│                                                              │
│  probe_update.wgsl   — shoot rays, accumulate SH + depth    │
│  surface_cache.wgsl  — texel-by-texel lightmap baking       │
│  path_trace.wgsl     — 1st hit: lookup cache, NEE direct    │
│    ├─ sample_irradiance_cache() → trilinear probe interp     │
│    │   with self-shadow bias + backface weight + Chebyshev   │
│    ├─ surface atlas lookup (static meshes)                   │
│    └─ fallback: path-trace bounce if cache invalid           │
│  ssgi.wgsl           — screen-space short-range gather       │
│  combine.wgsl        — blend SSGI + world-space cache        │
│  display.wgsl        — tone map + gamma (unchanged)         │
└──────────────────────────────────────────────────────────────┘

Bind groups:
  Group 0 (existing, 14→16 bindings):
    bindings 0–13: existing path-trace resources
    binding  14:   irradiance_buffer (L2 SH per probe)
    binding  15:   probe_grid_uniform
  Group 1 (new, IC-4+):
    depth_atlas, probe_meta_buffer, surface_irradiance_atlas,
    atlas_uv_buffer, cascade uniforms

Buffer inventory (new):
  irradiance_buffer[]         — L2 SH per probe (f16 after IC-8)
  depth_atlas[]               — oct depth mean + mean² per probe
  surface_irradiance_atlas    — 2048² Rgba32Float
  probe_meta_buffer           — position, flags, offsets
  atlas_uv_buffer             — per-triangle atlas UV rects
```

---

## Reference Material

| Resource | Notes |
|----------|-------|
| Majercik et al. **"Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields"** (2019) | Core DDGI algorithm; probe layout, octahedral irradiance/depth maps, Chebyshev visibility.  *This implementation substitutes SH for octahedral irradiance maps.* |
| Majercik et al. **"Scaling Probe-Based Real-Time Dynamic Global Illumination for Production"** (JCGT 2021) | Multi-cascade DDGI, probe relighting, probe relocation |
| Wright **"Lumen: Real-time Global Illumination in Unreal Engine 5"** (SIGGRAPH 2022) | Lumen's full pipeline; surface cache (Mesh Cards), final gather (screen probes), radiance cache (clipmap) |
| Ramamoorthi & Hanrahan **"An Efficient Representation for Irradiance Environment Maps"** (2001) | SH theory: L2 captures >99 % of diffuse irradiance energy; ZH convolution |
| Sloan **"Stupid Spherical Harmonics (SH) Tricks"** (GDC 2008) | Practical SH implementation: projection, evaluation, rotation, windowing |
| McGuire **"Real-Time Rendering Resources"** — DDGI implementation notes | wgpu/WGSL implementation patterns |

---

## Milestone Summary

| Phase  | Milestone                        | Key Deliverable                               |
|--------|----------------------------------|-----------------------------------------------|
| IC-1   | Probe Grid Foundation            | Visualised probe grid in scene                |
| IC-2   | Probe Radiance Capture           | SH probe coefficients converging              |
| IC-3   | Probe Interpolation              | Cache-driven indirect diffuse lighting        |
| IC-4   | Visibility & Relocation          | Light-leak-free interpolation, probe relocation |
| IC-5   | Multi-Resolution Cascades        | Far-field + near-field GI, no seams           |
| IC-6   | Surface Irradiance Cache         | Stable per-surface indirect on static meshes  |
| IC-7   | SSGI                             | High-frequency contact shadow & colour bleed  |
| IC-8   | Performance & Stability          | Real-time GI with full debug UI               |
| IC-9   | Validation & Integration Tests   | CI-tested, regression-verified subsystem      |

---

*Each phase is independently verifiable and buildable on the previous one.
Phases IC-1 through IC-4 form the minimum viable cache; IC-5 through IC-9 are
production-quality extensions.*
