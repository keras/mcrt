// world/generator.rs — Mediterranean terrain generator (Phase VW-2)
//
// Produces deterministic voxel chunks inspired by Mediterranean coastal
// landscapes: eroded limestone cliffs, terra-rossa valleys, karst caves,
// and light-driven vegetation (olive trees, dry shrubs, grass patches).
//
// Algorithm overview for each chunk:
//  1. Erosion-inspired heightmap via multi-octave FBM
//  2. Column fill: bedrock / limestone / red-soil transition
//  3. Cave carving: 3D Perlin "swiss cheese" noise
//  4. Surface dressing: grass vs. exposed rock based on slope
//  5. Vegetation: olive trees and shrubs placed by deterministic hash sampling
//
// No wgpu imports — independently testable.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin, SuperSimplex};

use super::block::*;
use super::chunk::{CHUNK_XZ, CHUNK_Y, Chunk};

// ---------------------------------------------------------------------------
// Terrain constants
// ---------------------------------------------------------------------------

/// Cave noise threshold above which a voxel is carved to AIR.
const CAVE_THRESHOLD: f64 = 0.65;
/// Minimum absolute Y for caves (keep bedrock intact).
const CAVE_MIN_Y: i32 = 6;
/// Hash-based vegetation density (lower = more vegetation).
const TREE_DENSITY: u64 = 5;
const SHRUB_DENSITY: u64 = 3;

// TERRAIN_BASE, TERRAIN_AMP and grass_slope_max are intentionally NOT
// constants — they are computed at runtime from `voxels_per_block` so the
// physical scale of the world can be tuned without recompiling.
//
// With vpb = 8 (the "1/8 Minecraft block scale" default):
//   terrain_base     = 8  × 8  =  64 voxels =  8 m
//   terrain_amp      = 10 × 8  =  80 voxels = 10 m  (→ max surface ≈ 18 m)
//   grass_slope_max  = 1.8 × 8 = 14.4 voxels/cell

// ---------------------------------------------------------------------------
// WorldGenerator
// ---------------------------------------------------------------------------

/// Deterministic, seed-based procedural world generator.
pub struct WorldGenerator {
    seed: u64,
    /// Voxels per Minecraft block; scales terrain heights and vegetation sizes.
    /// Set via the `voxels_per_block` field in the `world:` YAML block.
    voxels_per_block: u32,
    /// Multi-octave SuperSimplex noise for the base heightmap.
    height_fbm: Fbm<SuperSimplex>,
    /// Secondary FBM layer for slope-driven erosion detail.
    detail_fbm: Fbm<SuperSimplex>,
    /// 3D Perlin FBM used for karst cave carving.
    cave_fbm: Fbm<Perlin>,
    /// 3D Perlin FBM that warps the leaf-sphere radius, giving each tree a
    /// unique bumpy silhouette instead of a perfect sphere.
    leaf_fbm: Fbm<Perlin>,
}

impl WorldGenerator {
    /// Create a new generator from a 64-bit seed.
    ///
    /// The `noise` crate accepts a `u32` seed, so we fold the 64-bit seed
    /// into 32 bits via XOR-fold and mix offsets for each noise layer.
    pub fn new(seed: u64, voxels_per_block: u32) -> Self {
        let s32 = ((seed ^ (seed >> 32)) & 0xffff_ffff) as u32;

        let height_fbm = Fbm::<SuperSimplex>::new(s32)
            .set_octaves(6)
            .set_frequency(1.0)
            .set_lacunarity(2.0)
            .set_persistence(0.5);

        let detail_fbm = Fbm::<SuperSimplex>::new(s32.wrapping_add(0x9e37_79b9))
            .set_octaves(4)
            .set_frequency(3.0)
            .set_lacunarity(2.0)
            .set_persistence(0.45);

        let cave_fbm = Fbm::<Perlin>::new(s32.wrapping_add(0x6c62_272e))
            .set_octaves(4)
            .set_frequency(1.0)
            .set_lacunarity(2.0)
            .set_persistence(0.5);

        // Base frequency 1.0 in MC-block space → ~1-block-scale coarse bumps,
        // with finer octaves adding surface detail.  At vpb=8 this gives
        // variation every ~8 voxels, well below the tree radius.
        let leaf_fbm = Fbm::<Perlin>::new(s32.wrapping_add(0x2d35_9631))
            .set_octaves(4)
            .set_frequency(0.5)
            .set_lacunarity(2.0)
            .set_persistence(0.5);

        Self {
            seed,
            voxels_per_block,
            height_fbm,
            detail_fbm,
            cave_fbm,
            leaf_fbm,
        }
    }

    /// Generate the chunk at column coordinates `(cx, cz)`.
    pub fn generate_chunk(&self, cx: i32, cz: i32) -> Box<Chunk> {
        let mut chunk = Chunk::new();

        // Scale-dependent parameters derived from `voxels_per_block` (vpb).
        let vpb = self.voxels_per_block as f32;
        let vpb_d = self.voxels_per_block as f64;
        let terrain_base: i32 = (8.0 * vpb) as i32; // 8 MC blocks above bedrock
        let terrain_amp: f32 = 3.0 * vpb; // ±3 MC blocks amplitude → gentle rolling hills

        // Slope threshold in voxels-of-height per voxel-of-horizontal-distance.
        // With amp=3·vpb and period≈25·vpb voxels, typical slopes are 0.1–0.35.
        // Set threshold at 0.25 so steeper faces show rock, gentler ones grass.
        let grass_slope_max: f32 = 0.25;

        // ---- Step 1: build per-column heightmap and slope ----------------
        // We sample a (CHUNK_XZ+1)×(CHUNK_XZ+1) grid so we can compute
        // central-difference gradients at every interior voxel column.

        const G: usize = CHUNK_XZ + 1;
        let mut height_grid = [[0i32; G]; G];

        for (gx, row) in height_grid.iter_mut().enumerate() {
            for (gz, cell) in row.iter_mut().enumerate() {
                // Sample noise in Minecraft-block coordinates (voxel index / vpb).
                // This makes terrain feature width scale with vpb so hills cover
                // the same number of MC blocks at any voxel resolution.
                let mx = (cx * CHUNK_XZ as i32 + gx as i32) as f64 / vpb_d;
                let mz = (cz * CHUNK_XZ as i32 + gz as i32) as f64 / vpb_d;

                // Base FBM in [−1, +1].  Low frequency (0.04 in MC-block coords)
                // → feature period ≈ 25 MC blocks, giving broad rolling hills.
                let h = self.height_fbm.get([mx * 0.04, mz * 0.04]);
                // Detail layer adds gentle surface variation (reduced weight).
                let d = self.detail_fbm.get([mx * 0.10, mz * 0.10]) * 0.08;
                let combined = (h + d).clamp(-1.0, 1.0);
                // Flatten peaks (powf > 1 pulls values toward 0 → plateaus and
                // gentle valleys rather than sharp spikes).
                let eroded = combined.signum() * combined.abs().powf(1.4);

                let surf_y = terrain_base + (eroded as f32 * terrain_amp) as i32;
                *cell = surf_y.clamp(CAVE_MIN_Y + 8, CHUNK_Y as i32 - 2);
            }
        }

        // ---- Step 2: fill each column ------------------------------------
        for x in 0..CHUNK_XZ {
            for z in 0..CHUNK_XZ {
                let surface_y = height_grid[x][z];

                // Approximate gradient magnitude via finite differences.
                let dh_dx = (height_grid[x + 1][z] - height_grid[x][z]) as f32;
                let dh_dz = (height_grid[x][z + 1] - height_grid[x][z]) as f32;
                let slope = (dh_dx * dh_dx + dh_dz * dh_dz).sqrt();

                // Determine surface material.
                let surface_block = if slope < grass_slope_max {
                    GRASS_TYP
                } else {
                    LIMESTONE
                };

                // Fill columns from bottom to surface.
                for y in 0..(surface_y as usize).min(CHUNK_Y) {
                    // Bedrock zone: solid, no caves.
                    if y < CAVE_MIN_Y as usize {
                        chunk.set(x, y, z, LIMESTONE);
                        continue;
                    }

                    // Cave carving — sample in MC-block coords so cave feature
                    // size scales with vpb just like surface terrain.
                    let mc_x = (cx * CHUNK_XZ as i32 + x as i32) as f64 / vpb_d;
                    let mc_z = (cz * CHUNK_XZ as i32 + z as i32) as f64 / vpb_d;
                    // Compress Y so caves are elongated horizontally (tunnel-like).
                    let mc_y = y as f64 / vpb_d * 0.5;
                    let cave_n = self.cave_fbm.get([mc_x * 0.5, mc_y * 0.7, mc_z * 0.5]);
                    if cave_n > CAVE_THRESHOLD && y > CAVE_MIN_Y as usize {
                        // Check for cave glow: small clusters at mid-cave height.
                        let glow_hash =
                            lcg_hash(self.seed ^ (cx as u64) ^ ((cz as u64) << 32) ^ y as u64);
                        if glow_hash.is_multiple_of(120)
                            && y < (surface_y as usize).saturating_sub(6)
                            && y > 10
                        {
                            chunk.set(x, y, z, CAVE_GLOW);
                        } else {
                            chunk.set(x, y, z, AIR);
                        }
                        continue;
                    }

                    // Sub-surface stratigraphy:
                    //   top 5 layers → terra-rossa red soil on gentle terrain
                    //   below → limestone bedrock
                    let depth_from_surface = surface_y as usize - y;
                    if depth_from_surface < 5 && slope < grass_slope_max * 2.0 {
                        chunk.set(x, y, z, RED_SOIL);
                    } else {
                        chunk.set(x, y, z, LIMESTONE);
                    }
                }

                // Surface block.
                if (surface_y as usize) < CHUNK_Y {
                    chunk.set(x, surface_y as usize, z, surface_block);
                }
            }
        }

        // ---- Step 3: trunks (this chunk only) ------------------------------
        // Iterate at MC-block resolution; place only the wood trunk column here.
        // Canopy is handled in step 4 with cross-chunk awareness.
        let vpb_usize = self.voxels_per_block as usize;
        for x in (0..CHUNK_XZ).step_by(vpb_usize.max(1)) {
            for z in (0..CHUNK_XZ).step_by(vpb_usize.max(1)) {
                let surface_y = height_grid[x][z];
                if surface_y as usize >= CHUNK_Y - 10 || surface_y < CAVE_MIN_Y + 4 {
                    continue;
                }

                let dh_dx = (height_grid[x + 1][z] - height_grid[x][z]) as f32;
                let dh_dz = (height_grid[x][z + 1] - height_grid[x][z]) as f32;
                let slope = (dh_dx * dh_dx + dh_dz * dh_dz).sqrt();

                let mc_bx = (cx * CHUNK_XZ as i32 + x as i32) / self.voxels_per_block as i32;
                let mc_bz = (cz * CHUNK_XZ as i32 + z as i32) / self.voxels_per_block as i32;
                let col_hash = tree_hash(self.seed, mc_bx, mc_bz);

                if slope < grass_slope_max * 0.6 && col_hash.is_multiple_of(TREE_DENSITY) {
                    // Trunk only; canopy is placed in the cross-chunk step below.
                    let trunk_h = (2 + (col_hash % 4) as usize) * self.voxels_per_block as usize;
                    for ty in 0..trunk_h {
                        let y = surface_y as usize + 1 + ty;
                        if y >= CHUNK_Y {
                            break;
                        }
                        chunk.set(x, y, z, WOOD);
                    }
                } else if slope < grass_slope_max * 1.5
                    && col_hash % SHRUB_DENSITY == 1
                    && surface_y as usize + 1 < CHUNK_Y
                {
                    chunk.set(x, surface_y as usize + 1, z, SHRUB);
                }
            }
        }

        // ---- Step 4: cross-chunk-aware canopy pass -------------------------
        // Trees whose trunks sit near a chunk boundary would have their canopy
        // clipped if we only wrote leaves within this chunk's local bounds.
        // Instead, we iterate over every MC-block position whose canopy *could*
        // overlap this chunk (including positions in adjacent chunks), recompute
        // the surface height deterministically, and write only the leaf voxels
        // that fall inside our chunk — giving fully-round canopies everywhere.
        let vpb_i = self.voxels_per_block as i32;
        let max_r_vox = 2 * vpb_i; // maximum canopy radius in voxels
        let chunk_ox = cx * CHUNK_XZ as i32;
        let chunk_oz = cz * CHUNK_XZ as i32;

        let mc_bx_min = (chunk_ox - max_r_vox).div_euclid(vpb_i);
        let mc_bx_max = (chunk_ox + CHUNK_XZ as i32 - 1 + max_r_vox).div_euclid(vpb_i);
        let mc_bz_min = (chunk_oz - max_r_vox).div_euclid(vpb_i);
        let mc_bz_max = (chunk_oz + CHUNK_XZ as i32 - 1 + max_r_vox).div_euclid(vpb_i);

        for mc_bx in mc_bx_min..=mc_bx_max {
            for mc_bz in mc_bz_min..=mc_bz_max {
                let col_hash = tree_hash(self.seed, mc_bx, mc_bz);
                if !col_hash.is_multiple_of(TREE_DENSITY) {
                    continue;
                }

                // World voxel position of this tree's trunk.
                let wvx = mc_bx * vpb_i;
                let wvz = mc_bz * vpb_i;

                // Recompute surface height and slope at this position.
                let surf = self.surface_height_at(wvx, wvz);
                if surf as usize >= CHUNK_Y - 10 || surf < CAVE_MIN_Y + 4 {
                    continue;
                }
                let surf_px = self.surface_height_at(wvx + vpb_i, wvz);
                let surf_pz = self.surface_height_at(wvx, wvz + vpb_i);
                let slope = {
                    let dx = (surf_px - surf) as f32 / vpb_i as f32;
                    let dz = (surf_pz - surf) as f32 / vpb_i as f32;
                    (dx * dx + dz * dz).sqrt()
                };
                if slope >= grass_slope_max * 0.6 {
                    continue;
                }

                let trunk_h = (2 + (col_hash % 4) as i32) * vpb_i;
                let canopy_wy = surf + 1 + trunk_h;
                let radius = (1 + (col_hash >> 3 & 1) as i32) * vpb_i;

                self.place_canopy_clipped(
                    &mut chunk, chunk_ox, chunk_oz, wvx, canopy_wy, wvz, radius,
                );
            }
        }

        // Step 5 — Connectivity prune.
        //
        // Remove LEAVES that float in the air disconnected from any trunk.
        // We BFS from two seed sets:
        //  a) every WOOD voxel in the chunk  (in-chunk tree trunks)
        //  b) every LEAVES voxel on the chunk XZ boundary (x==0, x==last,
        //     z==0, z==last) — these may belong to a tree whose trunk sits in
        //     a neighbouring chunk; treating them as trusted prevents the pass
        //     from stripping the cross-chunk canopy contribution.
        //
        // Only LEAVES are expanded during BFS; WOOD acts as a seed but not as
        // a propagation medium.
        {
            let vol = CHUNK_XZ * CHUNK_XZ * CHUNK_Y;
            let idx = |x: usize, y: usize, z: usize| x * CHUNK_XZ * CHUNK_Y + z * CHUNK_Y + y;
            let mut visited = vec![false; vol];
            let mut queue: std::collections::VecDeque<(usize, usize, usize)> =
                std::collections::VecDeque::new();

            for y in 0..CHUNK_Y {
                for x in 0..CHUNK_XZ {
                    for z in 0..CHUNK_XZ {
                        let b = chunk.get(x, y, z);
                        let on_border = x == 0 || x == CHUNK_XZ - 1 || z == 0 || z == CHUNK_XZ - 1;
                        let is_seed = b == WOOD || (on_border && b == LEAVES);
                        if is_seed && !visited[idx(x, y, z)] {
                            visited[idx(x, y, z)] = true;
                            queue.push_back((x, y, z));
                        }
                    }
                }
            }

            const DIRS: [(i32, i32, i32); 6] = [
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ];
            while let Some((x, y, z)) = queue.pop_front() {
                for (dx, dy, dz) in DIRS {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    if nx < 0
                        || nx >= CHUNK_XZ as i32
                        || ny < 0
                        || ny >= CHUNK_Y as i32
                        || nz < 0
                        || nz >= CHUNK_XZ as i32
                    {
                        continue;
                    }
                    let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
                    if !visited[idx(nx, ny, nz)] && chunk.get(nx, ny, nz) == LEAVES {
                        visited[idx(nx, ny, nz)] = true;
                        queue.push_back((nx, ny, nz));
                    }
                }
            }

            for y in 0..CHUNK_Y {
                for x in 0..CHUNK_XZ {
                    for z in 0..CHUNK_XZ {
                        if chunk.get(x, y, z) == LEAVES && !visited[idx(x, y, z)] {
                            chunk.set(x, y, z, AIR);
                        }
                    }
                }
            }
        }

        chunk
    }

    /// Compute terrain surface height at an arbitrary world voxel position.
    /// Must match the heightmap formula in `generate_chunk` exactly.
    fn surface_height_at(&self, wvx: i32, wvz: i32) -> i32 {
        let vpb = self.voxels_per_block as f64;
        let vpb_f = self.voxels_per_block as f32;
        let mx = wvx as f64 / vpb;
        let mz = wvz as f64 / vpb;
        let h = self.height_fbm.get([mx * 0.04, mz * 0.04]);
        let d = self.detail_fbm.get([mx * 0.10, mz * 0.10]) * 0.08;
        let combined = (h + d).clamp(-1.0, 1.0) as f32;
        let eroded = combined.signum() * combined.abs().powf(1.4);
        let terrain_base = (8.0 * vpb_f) as i32;
        let terrain_amp = 3.0 * vpb_f;
        (terrain_base + (eroded * terrain_amp) as i32).clamp(CAVE_MIN_Y + 8, CHUNK_Y as i32 - 2)
    }

    /// Write the leaves of a canopy centred at world voxel `(wvx, canopy_wy, wvz)`
    /// into `chunk`, clipping to the chunk's local bounds.  Only AIR voxels are
    /// overwritten (trunks and terrain are preserved).
    ///
    /// The sphere radius is warped per-voxel by multi-frequency 3-D Perlin noise
    /// (±40 %), so each tree gets an organic, bumpy silhouette.  The inner 60 %
    /// of the sphere is always solid; only the outer shell is noise-dependent,
    /// so the canopy stays well-connected.  Isolated floating voxels are removed
    /// by the BFS connectivity pass that runs after all vegetation is placed.
    fn place_canopy_clipped(
        &self,
        chunk: &mut Chunk,
        chunk_ox: i32,
        chunk_oz: i32,
        wvx: i32,
        canopy_wy: i32,
        wvz: i32,
        radius: i32,
    ) {
        let vpb = self.voxels_per_block as f64;
        let r = radius as f32;
        // The noise can add at most 40 % to the radius, so we extend the
        // bounding box by that amount (plus one voxel for safety).
        let r_max = (r * 1.40) as i32 + 1;
        // Precompute thresholds to skip the FBM call for the solid inner core
        // and the empty outer region.
        let r_inner_sq = (r * 0.10) * (r * 0.10);
        let r_outer_sq = (r * 1.40 + 1.0) * (r * 1.40 + 1.0);

        for dy in -r_max..=r_max {
            for dx in -r_max..=r_max {
                for dz in -r_max..=r_max {
                    let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                    // Definitely outside — skip without noise lookup.
                    if dist_sq > r_outer_sq {
                        continue;
                    }

                    let place = if dist_sq <= r_inner_sq {
                        // Definitely inside — no noise needed.
                        true
                    } else {
                        // Transition shell: warp radius with 3-D FBM noise.
                        let nx = (wvx + dx) as f64 / vpb;
                        let ny = (canopy_wy + dy) as f64 / vpb;
                        let nz = (wvz + dz) as f64 / vpb;
                        let n = self.leaf_fbm.get([nx, ny, nz]) as f32;
                        let r_eff = r * (1.0 + n * 1.40);
                        dist_sq <= r_eff * r_eff
                    };
                    if !place {
                        continue;
                    }

                    let lx = wvx + dx - chunk_ox;
                    let ly = canopy_wy + dy;
                    let lz = wvz + dz - chunk_oz;
                    if lx < 0 || lx >= CHUNK_XZ as i32 {
                        continue;
                    }
                    if lz < 0 || lz >= CHUNK_XZ as i32 {
                        continue;
                    }
                    if ly < 0 || ly >= CHUNK_Y as i32 {
                        continue;
                    }
                    if chunk.get(lx as usize, ly as usize, lz as usize) == AIR {
                        chunk.set(lx as usize, ly as usize, lz as usize, LEAVES);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Fast, deterministic LCG hash.  Returns a pseudo-random u64 from a seed.
#[inline]
fn lcg_hash(seed: u64) -> u64 {
    seed.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Deterministic hash for the tree/vegetation at MC-block position (mc_bx, mc_bz).
/// Used by both the trunk pass (step 3) and the cross-chunk canopy pass (step 4)
/// so that both passes agree on which MC blocks have trees.
#[inline]
fn tree_hash(seed: u64, mc_bx: i32, mc_bz: i32) -> u64 {
    lcg_hash(seed ^ ((mc_bx as u64).wrapping_mul(0x517c_c1b7_2722_0a95)) ^ ((mc_bz as u64) << 32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::block::{AIR, LIMESTONE};
    use crate::world::chunk::{CHUNK_XZ, CHUNK_Y};

    fn make_gen(seed: u64) -> WorldGenerator {
        WorldGenerator::new(seed, 8)
    }

    #[test]
    fn chunk_has_solid_ground() {
        let wgen = make_gen(42);
        let chunk = wgen.generate_chunk(0, 0);
        // At least some column must have a solid block at bedrock level (y=1).
        let mut found_solid = false;
        'outer: for x in 0..CHUNK_XZ {
            for z in 0..CHUNK_XZ {
                if chunk.get(x, 1, z) != AIR {
                    found_solid = true;
                    break 'outer;
                }
            }
        }
        assert!(found_solid, "no solid blocks near bedrock");
    }

    #[test]
    fn bedrock_zone_is_solid() {
        let wgen = make_gen(1234);
        let chunk = wgen.generate_chunk(1, -2);
        // Y=0..5 must be solid limestone (cave carving is disabled there).
        for x in 0..CHUNK_XZ {
            for z in 0..CHUNK_XZ {
                for y in 0..CAVE_MIN_Y as usize {
                    assert_eq!(
                        chunk.get(x, y, z),
                        LIMESTONE,
                        "non-limestone at bedrock ({x},{y},{z})"
                    );
                }
            }
        }
    }

    #[test]
    fn top_of_chunk_is_air() {
        let wgen = make_gen(99);
        let chunk = wgen.generate_chunk(0, 0);
        // The very top Y layer should always be AIR (terrain never reaches 255).
        for x in 0..CHUNK_XZ {
            for z in 0..CHUNK_XZ {
                assert_eq!(
                    chunk.get(x, CHUNK_Y - 1, z),
                    AIR,
                    "top row not AIR at ({x},{z})"
                );
            }
        }
    }

    #[test]
    fn different_seeds_differ() {
        let c1 = WorldGenerator::new(1, 8).generate_chunk(0, 0);
        let c2 = WorldGenerator::new(2, 8).generate_chunk(0, 0);
        assert_ne!(
            c1.blocks, c2.blocks,
            "different seeds produced identical chunks"
        );
    }

    #[test]
    fn deterministic_across_calls() {
        let wgen = make_gen(777);
        let c1 = wgen.generate_chunk(3, -5);
        let c2 = wgen.generate_chunk(3, -5);
        assert_eq!(
            c1.blocks, c2.blocks,
            "same chunk generated differently on second call"
        );
    }
}
