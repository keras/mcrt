# Procedural Voxel World Generator — Roadmap

Extend the path-tracer renderer with a fully procedural, infinite voxel world
generator inspired by Minecraft but operating at **1/8 the block scale** — a
voxel edge length of **0.125 m (12.5 cm)** compared to Minecraft's 1 m.  At
this scale a human-sized character is ~14 voxels tall and fine surface details
(brickwork, gravel, bark) can be expressed geometrically rather than through
texture tiling alone.

> **Design goals:**
> - The world is infinite and generated fully on the fly from a 64-bit integer
>   seed; no pre-authored assets are required.
> - Geometry is expressed as triangle meshes fed into the existing BVH pipeline —
>   no new GPU primitives or shader rewrites are needed for the core path tracer.
> - A greedy meshing pass keeps triangle counts manageable; flat same-material
>   faces are merged into quads before triangulation.
> - The scene YAML gains a `world:` key so a voxel world can be composed with
>   ordinary objects (spheres, meshes, lights).
> - All world-generation code lives in pure Rust with no wgpu dependencies,
>   making it independently testable.
> - Hot-reload: modifying generator parameters in the YAML rebuilds only the
>   affected chunks and re-uploads geometry to the GPU.

---

## Scale Reference

| Entity | Minecraft (m) | This renderer (m) | Voxels tall |
|--------|--------------|-------------------|-------------|
| Block edge | 1.00 | 0.125 | 1 |
| Player height | 1.80 | 0.225 | ~1.8 |
| Tree trunk height | 4–8 | 0.50–1.00 | 4–8 |
| Chunk width (XZ) | 16 | 2.00 | 16 |
| Chunk height (Y) | 256 | 32.00 | 256 |
| View distance (near) | 128 | 16.00 | 128 |

Voxel coordinates are integers; world coordinates are `ivec3 * 0.125`.

---

## Phase VW-1: Voxel Data Model & Chunk System

**Goal:** Establish the in-memory representation of the voxel world: block type
identifiers, a fixed-size chunk, and a sparse chunk map.  No rendering or
generation yet — just the data structures that everything else builds on.

### Block type registry

Blocks are identified by a `u16` block ID.  A global palette array maps IDs to
metadata:

```rust
// src/world/block.rs
pub const AIR:   u16 = 0;
pub const STONE: u16 = 1;
pub const DIRT:  u16 = 2;
pub const GRASS: u16 = 3;
pub const SAND:  u16 = 4;
pub const WATER: u16 = 5;   // transparent, future phase
pub const WOOD:  u16 = 6;
pub const LEAVES: u16 = 7;
pub const SNOW:  u16 = 8;
pub const GRAVEL: u16 = 9;
// ... up to 256 initial block types

pub struct BlockMeta {
    pub name:          &'static str,
    pub opaque:        bool,    // false → skip greedy-mesh face culling against it
    pub material_name: &'static str, // name of the GPU material slot
    pub tint:          [f32; 3], // base albedo tint (multiplied with material albedo)
}
```

A compile-time array `BLOCK_META: [BlockMeta; 256]` is the single source of
truth.  New block types are added by appending to this array; existing indices
must never be renumbered to keep saved worlds valid.

### Chunk layout

A chunk is a fixed 16 × 256 × 16 voxel column (XZY order, matching Minecraft
convention).  Y = 0 is the bedrock floor; Y = 255 is the build limit.

```rust
// src/world/chunk.rs
pub const CHUNK_XZ: usize = 16;
pub const CHUNK_Y:  usize = 256;
pub const CHUNK_SIZE: usize = CHUNK_XZ * CHUNK_Y * CHUNK_XZ; // 65 536

pub struct Chunk {
    /// Block IDs in XZY-major order: index = x * CHUNK_Y * CHUNK_XZ + y * CHUNK_XZ + z
    pub blocks: Box<[u16; CHUNK_SIZE]>,
}
```

Storing 65 536 `u16` values per chunk consumes 128 KiB per chunk.  With a 9×9
view region (81 chunks) that is ~10 MiB — negligible.

### Chunk coordinate system

Chunk coordinates `(cx, cz): (i32, i32)` identify a column; world block
coordinates are `(cx * 16 + lx, y, cz * 16 + lz)`.  A `ChunkMap` holds live
chunks in a `HashMap<(i32,i32), Chunk>`.

### Tasks

- [ ] Create `src/world/` module directory and `src/world/mod.rs`.
- [ ] Implement `src/world/block.rs`: `u16` block IDs, `BlockMeta` struct, and
  a `BLOCK_META` static array with at least 10 initial block types.
- [ ] Implement `src/world/chunk.rs`: `Chunk` struct with flat `u16` block
  storage, `inline fn get(x,y,z)` and `set(x,y,z,id)` accessors with bounds
  debug-assertions.
- [ ] Implement `ChunkMap` in `src/world/chunk.rs`: `HashMap<(i32,i32), Box<Chunk>>`,
  `insert`, `get`, and `remove` methods.
- [ ] Write unit tests: round-trip block set/get in all eight chunk corners,
  index-formula verification.

**Output:** A compilable `src/world/` module with zero GPU or wgpu imports.
`cargo test world` passes.

---

## Phase VW-2: Procedural Terrain Generator

**Goal:** Implement a deterministic noise-based heightmap + biome terrain
generator that fills `Chunk` objects from a world seed.

### Noise library

Use the `noise` crate (already common in Rust game projects) for
`Simplex`/`Perlin` noise.  Add it to `Cargo.toml`:

```toml
noise = "0.9"
```

### Terrain layers

| Pass | Algorithm | Controls |
|------|-----------|---------|
| **Continent** | Low-frequency Simplex (octave 1) | Large-scale continent/ocean shape |
| **Heightmap** | 5-octave fractional Brownian motion (fBm) | Hill & valley profile |
| **Biome** | 2D Voronoi + temperature/humidity noise | Block palette selection |
| **Cave** | 3D Perlin worm + threshold | Underground cavities |
| **Surface decoration** | Point Poisson sampling + templates | Trees, boulders |

#### Heightmap formula

```
base_height   = 64 + continent_noise(cx,cz) * 48
detail_height = fBm(x, z, octaves=5, lacunarity=2.0, gain=0.5) * 24
surface_y     = clamp(base_height + detail_height, 4, 248)
```

All noise evaluations use `(x * VOXEL_SCALE, z * VOXEL_SCALE)` world
coordinates so terrain features scale correctly.

#### Column filling

```
for y in 0..surface_y:
    block = stone
for y in surface_y-3..surface_y:
    block = dirt (or biome-appropriate fill)
blocks[surface_y] = grass / sand / snow (biome top)
for y in surface_y+1..sea_level:
    block = water
```

### Biome system

Four initial biomes, selected by a `(temperature, humidity)` pair sampled from
two low-frequency noise maps:

| Biome | Top block | Fill | Tree type |
|-------|-----------|------|-----------|
| Plains | Grass | Dirt | Oak (trunk + leaf sphere) |
| Desert | Sand | Sand | Cactus pillar |
| Taiga | Snow | Dirt | Spruce (tall trunk + cone cap) |
| Badlands | Red sand | Red stone | None |

### Cave generation

3D Perlin noise evaluated at each voxel: if `cave_noise(x, y, z) > 0.72` and
`y < surface_y - 8` and `y > 2`, the voxel is carved to AIR.  A secondary
worm-tube pass uses the gradient of the noise field to produce more tunnel-like
passages.

### World generator struct

```rust
// src/world/generator.rs
pub struct WorldGenerator {
    seed: u64,
    // per-noise-layer Simplex generators derived from seeded RNG
}

impl WorldGenerator {
    pub fn new(seed: u64) -> Self;
    pub fn generate_chunk(&self, cx: i32, cz: i32) -> Box<Chunk>;
}
```

`generate_chunk` is pure (no mutable state) so multiple chunks can be generated
in parallel with `rayon`.

### Tasks

- [ ] Add `noise = "0.9"` to `Cargo.toml`.
- [ ] Implement `src/world/generator.rs`: `WorldGenerator::new(seed)` and
  `generate_chunk(cx, cz)`.
- [ ] Implement the 5-layer generation pipeline: continent, heightmap, biome
  selection, cave carve, surface decoration stubs.
- [ ] Implement the four initial biomes with correct top/fill block palettes.
- [ ] Implement tree template spawning (oak: 4–6-voxel trunk + leaf sphere of
  radius 3–4; spruce: 6–10-voxel trunk + stacked horizontal leaf layers).
- [ ] Write determinism tests: `generate_chunk(0,0)` called twice returns
  byte-identical `Chunk` contents.
- [ ] Write coverage tests: at least 50 % of generated surface blocks are
  non-air across a 5×5 chunk region.

**Output:** `WorldGenerator::new(42).generate_chunk(0, 0)` returns a fully
filled chunk.  `cargo test world` passes.

---

## Phase VW-3: Greedy Mesh Extraction

**Goal:** Convert a `Chunk` (and its six neighbours for cross-boundary face
culling) into a minimal triangle mesh via the greedy meshing algorithm.  This
is the performance-critical step that keeps triangle counts reasonable despite
the fine voxel scale.

### Face culling

A face between two adjacent voxels is skipped if the neighbour is opaque
(i.e., `BLOCK_META[neighbour_id].opaque == true`).  Only the six faces of
each non-air, opaque block that border a non-opaque voxel are emitted.

### Greedy meshing algorithm

For each of the six face orientations (±X, ±Y, ±Z):

1. Sweep layer by layer along the face-normal axis.
2. Build a 2D boolean mask of visible, same-material faces in this layer.
3. Greedily expand rectangles: for each unvisited `true` cell, extend as far
   right then as far down as possible while all cells share the same block ID.
4. Emit a quad (two triangles) for each merged rectangle.
5. Mark the consumed cells as visited.

This reduces the worst case of `N³` individual quads to O(N²) merged quads for
flat terrain — typically a 10–50× reduction over naïve per-face triangulation.

### Chunk mesh output

```rust
// src/world/mesher.rs
pub struct ChunkMesh {
    pub vertices:  Vec<crate::mesh::GpuVertex>,
    pub triangles: Vec<crate::mesh::GpuTriangle>,
}

pub fn mesh_chunk(
    chunk: &Chunk,
    neighbours: [Option<&Chunk>; 6], // +X, -X, +Y, -Y, +Z, -Z
    world_origin: [f32; 3],          // corner of this chunk in world space
    material_map: &BlockMaterialMap, // block_id → GPU material index
) -> ChunkMesh;
```

`GpuVertex` and `GpuTriangle` are the same types already used by `mesh.rs`,
so the output plugs directly into the existing mesh BVH pipeline.

`world_origin` is `[cx as f32 * 16.0 * 0.125, 0.0, cz as f32 * 16.0 * 0.125]`
— vertex positions are in world metres.

### Block-to-material mapping

```rust
pub struct BlockMaterialMap {
    /// block_id → GPU material slot index (into the global GpuMaterialData table)
    pub slots: [u32; 256],
}
```

Built once at startup from `BLOCK_META` by walking the named material palette.

### Tasks

- [ ] Implement `src/world/mesher.rs` with the greedy meshing algorithm for
  all six face orientations.
- [ ] Implement `BlockMaterialMap` construction from `BLOCK_META` and a
  material name palette.
- [ ] Implement correct per-vertex normals (constant per face orientation) and
  placeholder UV coordinates (left for texture phase).
- [ ] Handle chunk boundary faces using the `neighbours` argument; treat
  missing neighbours as all-AIR (conservative — face is emitted).
- [ ] Write unit tests:
  - A 1×1×1 solid voxel produces exactly 12 triangles (6 quads × 2).
  - A 2×1×2 flat slab produces fewer than 24 triangles (greedy merges XZ faces).
  - A fully air chunk produces 0 triangles.
- [ ] Benchmark: `mesh_chunk` for a full terrain chunk (16×256×16) takes ≤ 5 ms
  on a development machine.

**Output:** `mesh_chunk` converts any `Chunk` to `GpuVertex`/`GpuTriangle`
arrays compatible with `mesh.rs`.  All unit tests pass.

---

## Phase VW-4: Block Material Palette

**Goal:** Register block visual properties in the renderer's material system so
each block type renders with appropriate albedo, roughness, and other PBR
properties under path tracing.

### Current material system

The renderer supports four material types:

| Type | Shader ID | Key parameters |
|------|-----------|----------------|
| Lambertian | 0 | `albedo` |
| Metal | 1 | `albedo`, `fuzz` |
| Dielectric | 2 | `ior` |
| Emissive | 3 | `albedo`, `emission_strength` |

All block types initially use **Lambertian** with per-block albedo.  Metal and
dielectric variants are used for ore veins and ice/water blocks respectively.

### Initial block palette (≤ 64 GPU material slots)

| Block | Material type | Albedo (linear) | Notes |
|-------|--------------|-----------------|-------|
| Stone | Lambertian | [0.45, 0.45, 0.45] | |
| Dirt | Lambertian | [0.42, 0.27, 0.14] | |
| Grass | Lambertian | [0.22, 0.48, 0.10] | |
| Sand | Lambertian | [0.76, 0.69, 0.50] | |
| Gravel | Lambertian | [0.40, 0.38, 0.36] | |
| Wood | Lambertian | [0.35, 0.21, 0.10] | |
| Leaves | Lambertian | [0.10, 0.35, 0.05] | |
| Snow | Lambertian | [0.90, 0.92, 0.95] | |
| Coal ore | Lambertian | [0.18, 0.18, 0.18] | |
| Iron ore | Metal | [0.62, 0.53, 0.45] fuzz=0.6 | |
| Gold ore | Metal | [0.83, 0.68, 0.22] fuzz=0.3 | |
| Diamond ore | Metal | [0.20, 0.80, 0.90] fuzz=0.1 | |
| Ice | Dielectric | ior=1.31 | water ice |
| Lava | Emissive | [1.0, 0.35, 0.02] strength=6.0 | area light source |
| Glowstone | Emissive | [1.0, 0.90, 0.60] strength=4.0 | |
| Red sand | Lambertian | [0.70, 0.35, 0.12] | |

The current `MAX_MATERIALS = 64` limit in `material.rs` / `path_trace.wgsl`
accommodates a palette of ~64 block types.  If more are needed, bump
`MAX_MATERIALS` to 256 and update the corresponding WGSL constant.

### YAML material declaration

World block materials are declared in a special `world_materials:` top-level
key in the scene YAML (distinct from the regular per-object `materials:` map)
so they are registered before any scene objects are processed:

```yaml
world_materials:
  stone:   { type: lambertian, albedo: [0.45, 0.45, 0.45] }
  dirt:    { type: lambertian, albedo: [0.42, 0.27, 0.14] }
  grass:   { type: lambertian, albedo: [0.22, 0.48, 0.10] }
  # ... one entry per block type used in this scene's world
```

### Tasks

- [ ] Define the initial 16-block palette above in `src/world/block.rs`,
  matching `material_name` strings to `world_materials:` YAML keys.
- [ ] Extend `scene.rs` to parse an optional `world_materials:` top-level map
  and pre-populate the GPU material table before processing `objects:`.
- [ ] Implement `BlockMaterialMap::from_palette(palette, block_meta)` that
  resolves each block's `material_name` to a GPU slot index, falling back
  to a bright-pink error material for unmapped blocks.
- [ ] Write unit test: all 16 initial block types resolve to distinct, in-range
  GPU slots.
- [ ] Consider `MAX_MATERIALS`: if the palette exceeds 64 later, document the
  constant-bump procedure.

**Output:** A `world_materials:` YAML section correctly populates GPU material
slots for block rendering.  `cargo test` passes.

---

## Phase VW-5: GPU Integration & Chunk BVH

**Goal:** Feed chunk meshes into the existing `mesh.rs` / `gpu.rs` pipeline and
build a GPU-side BVH that covers an entire rendered world region.  The path
tracer should shoot rays into the voxel world with no shader modifications.

### Integration strategy

The renderer already holds a single flat `Vec<GpuVertex>` and
`Vec<GpuTriangle>` merged from all scene mesh objects, covered by one
`MeshBvhResult`.  The voxel world is a large collection of chunk meshes that
slots into exactly this pipeline:

1. For each visible chunk, call `mesh_chunk(…)` → `ChunkMesh`.
2. Merge all chunk `GpuVertex` / `GpuTriangle` arrays, offsetting triangle
   vertex indices by the running vertex base.
3. Append each block's Lambertian/metal/emissive material to the palette
   (already done in Phase VW-4).
4. Build a single mesh BVH over the merged triangle list via the existing
   `mesh::build_mesh_bvh`.
5. Upload the combined vertex + triangle + BVH buffers to the GPU and dispatch
   the path tracer as normal.

No GPU code changes are required — the world is simply "a lot of triangles".

### Chunk visibility set

A square `VIEW_DISTANCE × VIEW_DISTANCE` chunk region centred on the camera is
always loaded.  For the initial implementation, `VIEW_DISTANCE = 9` (9×9 = 81
chunks) gives a rendered area of 18 m × 18 m in world space at default scale —
suitable for screenshots and regression tests.

```rust
pub fn visible_chunk_coords(cam_pos: [f32; 3], view_dist: i32) -> Vec<(i32, i32)>
```

### Scene integration point

Add a `WorldState` field to `GpuState`:

```rust
struct WorldState {
    generator:    WorldGenerator,
    loaded_chunks: ChunkMap,
    material_map:  BlockMaterialMap,
    view_distance: i32,
}
```

On scene load, if a `world:` YAML block is present, instantiate `WorldState`,
generate all visible chunks, mesh them, and merge their geometry with any
additional static mesh objects from the scene.

### Tasks

- [ ] Implement `merge_chunk_meshes(chunks: &[(ChunkMesh, [f32;3])]) -> (Vec<GpuVertex>, Vec<GpuTriangle>)` in `src/world/mesher.rs`.
- [ ] Add `WorldState` to `gpu.rs` and wire it into the `GpuState::new /
  reload_scene` path.
- [ ] Implement `visible_chunk_coords`; use it at scene load to generate the
  initial chunk set.
- [ ] Ensure the merged voxel geometry feeds through `mesh::build_mesh_bvh`
  unchanged.
- [ ] Test: load a world-only scene YAML (no extra objects) and confirm the
  render completes without GPU validation errors.
- [ ] Test: the merged vertex / triangle buffers stay within the wgpu
  `max_buffer_size` limit for 81 chunks at default VIEW_DISTANCE.  Log a
  warning if the merged mesh exceeds 512 MB.

**Output:** A world YAML scene renders through the existing path tracer with no
shader changes.

---

## Phase VW-6: Scene YAML `world:` Key

**Goal:** Expose the world generator through the scene YAML so users can
control seed, view distance, and rendering parameters without touching Rust
code.

### YAML schema

```yaml
# Procedural voxel world
world:
  seed: 123456789       # u64 — deterministic generation
  view_distance: 9      # half-width in chunks (9 → 9×9 = 81 chunks)
  sea_level: 64         # y-coordinate of water surface
  terrain:
    base_height: 64     # average surface y
    amplitude:   24.0   # ±height of fBm detail (metres)
    octaves:     5
    lacunarity:  2.0
    gain:        0.5
  biomes:
    enabled: true       # false → single biome (plains)
  caves:
    enabled: true
    threshold: 0.72     # Perlin threshold above which voxels are carved

# World materials block (optional — built-in defaults are used if absent)
world_materials:
  stone: { type: lambertian, albedo: [0.45, 0.45, 0.45] }
  ...

# Regular scene additions (lights, props, camera) compose with the world
camera:
  look_from: [0.0, 12.0, 10.0]
  look_at:   [0.0,  9.0,  0.0]
  vfov: 70.0
  aperture: 0.0

objects:
  - type: sphere
    center: [0.0, 15.0, 0.0]
    radius: 0.5
    material: { type: emissive, albedo: [1.0, 0.9, 0.8], emission_strength: 20.0 }
```

### Serde schema types

```rust
// src/world/config.rs
#[derive(serde::Deserialize, Default)]
pub struct WorldDesc {
    pub seed:          u64,
    #[serde(default = "default_view_distance")]
    pub view_distance: i32,
    #[serde(default = "default_sea_level")]
    pub sea_level:     i32,
    #[serde(default)]
    pub terrain:       TerrainDesc,
    #[serde(default)]
    pub biomes:        BiomeDesc,
    #[serde(default)]
    pub caves:         CaveDesc,
}
```

When the `world:` key is absent the existing behaviour (static mesh + sphere
scene) is unchanged.

### Hot-reload support

Changing world generator parameters in the YAML triggers `reload_scene()` in
`gpu.rs`.  The reload regenerates all visible chunks from scratch (incrementally
dirty only when chunk count is large — a future optimisation).

### Tasks

- [ ] Implement `src/world/config.rs` with `WorldDesc`, `TerrainDesc`,
  `BiomeDesc`, `CaveDesc` serde structs and `Default` impls matching the
  defaults above.
- [ ] Extend `SceneFile` in `scene.rs` to include `world: Option<WorldDesc>`.
- [ ] Wire `WorldDesc` → `WorldGenerator::new_from_desc` factory.
- [ ] Write `assets/world.yaml` — a minimal example world scene that can be
  opened with the interactive renderer.
- [ ] Confirm hot-reload: edit `seed:` in the running interactive renderer and
  verify the world re-generates.

**Output:** `cargo run -- assets/world.yaml` opens the interactive renderer
showing a procedurally generated voxel landscape.

---

## Phase VW-7: Dynamic Chunk Streaming

**Goal:** As the camera moves in the interactive renderer, load newly visible
chunks and unload out-of-range chunks without stalling the render loop.

### Two-thread model

World generation and meshing are CPU-bound and slow relative to a single frame.
Move them to a worker thread and send completed meshes to the render thread via
a channel:

```
┌──────────────┐  (cx,cz) requests   ┌─────────────────────┐
│  Render loop │ ──────────────────► │  WorldWorker thread  │
│  (GPU thread)│                     │  generate + mesh      │
│              │ ◄────────────────── │  → send ChunkMesh    │
└──────────────┘  ChunkMesh results  └─────────────────────┘
```

1. Each frame, compute the desired chunk set from the camera position.
2. Diff against the currently loaded set: emit `Load(cx,cz)` and
   `Unload(cx,cz)` requests.
3. Worker generates + meshes each requested chunk and sends it back.
4. The render thread injects new chunks and rebuilds the BVH at a controlled
   rate (e.g., at most 4 new chunks per frame) to avoid frame spikes.
5. Memoize generated chunks in `ChunkMap`; regenerate only on seed change.

### BVH rebuild strategy

Rebuilding the full mesh BVH every frame is expensive for large worlds.  Two
mitigations:

- **Rebuild on chunk change only:** track a `geometry_dirty` flag; skip BVH
  rebuild if no chunks changed this frame.
- **Hierarchical BVH:** build a per-chunk BVH during meshing; combine chunk
  BVHs under a two-level top BVH.  The top level is rebuilt cheaply when
  chunks are swapped; per-chunk BVHs are rebuilt only for changed chunks.

The hierarchical BVH is a stretch goal; flat rebuild suffices for initial
interactive use.

### Tasks

- [ ] Implement `WorldWorker` in `src/world/worker.rs`: a thread that owns a
  `WorldGenerator` and processes `(i32,i32)` requests from a
  `crossbeam_channel` (or `std::sync::mpsc`), returning `(ChunkPos, ChunkMesh)`.
- [ ] Implement camera-position-based desired-chunk-set diffing in `gpu.rs`.
- [ ] Implement chunk injection: merge new chunk mesh into the live geometry
  buffers and mark `geometry_dirty`.
- [ ] Implement chunk eviction: remove old chunk geometry (track byte ranges or
  rebuild from scratch — the latter is simpler and acceptable for ≤ 81 chunks).
- [ ] Rate-limit BVH rebuilds to avoid per-frame stalls.
- [ ] Add `crossbeam-channel = "0.5"` (or use `std::sync::mpsc`) to
  `Cargo.toml`.
- [ ] Test: move the camera 32 m (256 voxels) and confirm new chunks appear
  without visual tearing or GPU validation errors.

**Output:** Smooth camera movement through an infinite procedural world with
chunk pop-in limited to the background loading rate.

---

## Phase VW-8: World Regression Test Scenes

**Goal:** Add regression test scenes that cover the voxel world rendering path
so the Phase RT framework can detect unintended visual regressions in terrain
generation or meshing.

### Proposed world scenes

| File | What it tests |
|------|---------------|
| `tests/assets/scenes/world_plains.yaml` | Basic Lambertian terrain, no caves, no decoration |
| `tests/assets/scenes/world_caves.yaml` | Underground camera angle testing cave interiors and emissive lava |
| `tests/assets/scenes/world_biomes.yaml` | Camera positioned to capture a biome boundary edge |

All three scenes use `view_distance: 3` (7×7 = 49 chunks) and low SPP for
fast headless renders.  The seed and camera are fixed so output is fully
deterministic.

### Sidecar example

```toml
# tests/assets/scenes/world_plains.test.toml
[render]
width  = 256
height = 144
spp    = 32

[regression]
threshold_mse  = 0.004   # slightly looser than mesh scenes due to large flat surfaces
threshold_psnr = 34.0
```

### Tasks

- [ ] Author `tests/assets/scenes/world_plains.yaml` with fixed seed, small
  `view_distance`, and deterministic top-down camera.
- [ ] Author `tests/assets/scenes/world_caves.yaml` with a camera placed below
  the surface highlighting lava emissive lighting.
- [ ] Author `tests/assets/scenes/world_biomes.yaml` with camera at a
  plains-desert boundary.
- [ ] Write sidecar `.test.toml` for each scene.
- [ ] Add all three scenes to `tests/assets/scenes/README.md`.
- [ ] Run `scripts/regress_baseline.sh --from HEAD` to capture initial baselines.

**Output:** Three new scenes in the regression suite; `cargo test regression`
includes world rendering coverage.

---

## Dependency Summary

### New Rust crates (`Cargo.toml`)

| Crate | Purpose | Already present? |
|-------|---------|-----------------|
| `noise` | Simplex/Perlin noise for terrain generation | No — add `noise = "0.9"` |
| `crossbeam-channel` | Lock-free MPSC channel for world worker thread | No — add or use `std::sync::mpsc` |
| `rayon` | Parallel chunk generation | No — add `rayon = "1"` (optional) |

Everything else (`serde`, `image`, `bytemuck`, `wgpu`, `clap`) is already used.

---

## Open Questions

1. **`MAX_MATERIALS` limit** — the current cap of 64 GPU material slots may
   be tight for a rich block palette.  Raising it to 256 requires updating the
   constant in both `material.rs` and `path_trace.wgsl`; document this as a
   known upgrade path.

2. **Large-world memory** — at `VIEW_DISTANCE = 16` (33×33 = 1 089 chunks) the
   merged vertex buffer could exceed 1 GB.  Profile triangle counts from greedy
   meshing under typical terrain before committing to a view-distance default.

3. **Determinism across platforms** — the noise library must produce identical
   float outputs across CPU architectures for regression baselines to be
   portable.  Verify by running terrain generation on both ARM and x86 machines
   and comparing `Chunk` byte content.

4. **LOD strategy** — at long range, individual 12.5 cm voxels are sub-pixel.
   A multi-resolution LOD (full resolution near, 2× merged blocks at mid-range,
   impostor geometry at far range) would reduce triangle count and BVH depth.
   This is not planned in the above phases but should be kept in mind when
   designing the mesher API.

5. **Water rendering** — transparent/refractive water blocks require a separate
   draw pass or mixed-BVH handling not currently supported.  Water is defined
   in the block registry but rendered as opaque placeholder blue until a
   transparency phase is implemented.

6. **Texture atlas** — the current material system encodes per-material albedo
   as a uniform colour.  A texture atlas would allow per-face variation (grass
   side vs. top) and dramatically reduce material slot consumption.  This is
   best deferred until the implicit-surfaces or texture-streaming roadmap is
   tackled.
