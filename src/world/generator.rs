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
use super::chunk::{CHUNK_XZ, CHUNK_Y, Chunk, VOXEL_SIZE};

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

        Self {
            seed,
            voxels_per_block,
            height_fbm,
            detail_fbm,
            cave_fbm,
        }
    }

    /// Generate the chunk at column coordinates `(cx, cz)`.
    pub fn generate_chunk(&self, cx: i32, cz: i32) -> Box<Chunk> {
        let mut chunk = Chunk::new();

        // Scale-dependent parameters derived from `voxels_per_block` (vpb).
        // Expressing heights in Minecraft-block units and multiplying by vpb
        // keeps the world visually consistent regardless of voxel resolution.
        let vpb = self.voxels_per_block as f32;
        let terrain_base: i32 = (8.0 * vpb) as i32; // 8 MC blocks above bedrock
        let terrain_amp: f32 = 10.0 * vpb; // ±10 MC blocks amplitude
        let grass_slope_max: f32 = 1.8 * vpb; // voxels/cell threshold for grass

        // ---- Step 1: build per-column heightmap and slope ----------------
        // We sample a (CHUNK_XZ+1)×(CHUNK_XZ+1) grid so we can compute
        // central-difference gradients at every interior voxel column.

        const G: usize = CHUNK_XZ + 1;
        let mut height_grid = [[0i32; G]; G];

        for (gx, row) in height_grid.iter_mut().enumerate() {
            for (gz, cell) in row.iter_mut().enumerate() {
                let wx = (cx * CHUNK_XZ as i32 + gx as i32) as f64 * VOXEL_SIZE as f64;
                let wz = (cz * CHUNK_XZ as i32 + gz as i32) as f64 * VOXEL_SIZE as f64;

                // Base FBM in [−1, +1].  Low frequency (0.12) keeps features
                // broad relative to the 18 m viewport — no micro-spikes.
                let h = self.height_fbm.get([wx * 0.12, wz * 0.12]);
                // Detail layer adds gentle surface variation (reduced weight).
                let d = self.detail_fbm.get([wx * 0.28, wz * 0.28]) * 0.08;
                let combined = (h + d).clamp(-1.0, 1.0);
                // Mild non-linear bias: slightly flatten peaks, deepen valleys
                // to give a Mediterranean eroded-limestone feel without spikes.
                let eroded = combined.signum() * combined.abs().powf(0.85);

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

                    // Cave carving.
                    let world_x = (cx * CHUNK_XZ as i32 + x as i32) as f64 * VOXEL_SIZE as f64;
                    let world_z = (cz * CHUNK_XZ as i32 + z as i32) as f64 * VOXEL_SIZE as f64;
                    // Compress Y so caves are elongated horizontally (tunnel-like).
                    let world_y = y as f64 * VOXEL_SIZE as f64 * 0.5;
                    let cave_n = self.cave_fbm.get([world_x * 0.5, world_y * 0.7, world_z * 0.5]);
                    if cave_n > CAVE_THRESHOLD && y > CAVE_MIN_Y as usize {
                        // Check for cave glow: small clusters at mid-cave height.
                        let glow_hash = lcg_hash(
                            self.seed ^ (cx as u64) ^ ((cz as u64) << 32) ^ y as u64,
                        );
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

        // ---- Step 3: vegetation -----------------------------------------
        // Vegetation is placed based on slope only — the surface block is by
        // definition the topmost solid block, so sky visibility is always 1.0
        // and the old sky_vis proxy (which was based on absolute Y height and
        // always returned near-zero for low-amplitude terrain) is removed.
        for x in 0..CHUNK_XZ {
            for z in 0..CHUNK_XZ {
                let surface_y = height_grid[x][z];
                if surface_y as usize >= CHUNK_Y - 10 || surface_y < CAVE_MIN_Y + 4 {
                    continue;
                }

                // Slope at this column (voxel-height difference per adjacent cell).
                let dh_dx = (height_grid[x + 1][z] - height_grid[x][z]) as f32;
                let dh_dz = (height_grid[x][z + 1] - height_grid[x][z]) as f32;
                let slope = (dh_dx * dh_dx + dh_dz * dh_dz).sqrt();

                // Unique per-column deterministic hash.
                let col_hash = lcg_hash(
                    self.seed
                        ^ ((cx as u64).wrapping_mul(0x517c_c1b7_2722_0a95))
                        ^ ((cz as u64) << 32)
                        ^ (x as u64).wrapping_mul(31)
                        ^ (z as u64).wrapping_mul(97),
                );

                // Olive tree: gentle slope only, hash-gated density.
                if slope < 1.0 * vpb && col_hash.is_multiple_of(TREE_DENSITY) {
                    self.place_tree(&mut chunk, x, surface_y as usize + 1, z, col_hash);
                    continue; // don't also add a shrub beneath the trunk
                }

                // Shrub: tolerates moderate slopes.
                if slope < 2.5 * vpb
                    && col_hash % SHRUB_DENSITY == 1
                    && surface_y as usize + 1 < CHUNK_Y
                {
                    chunk.set(x, surface_y as usize + 1, z, SHRUB);
                }
            }
        }

        chunk
    }

    /// Place a small olive-style tree at `(bx, base_y, bz)`.
    fn place_tree(&self, chunk: &mut Chunk, bx: usize, base_y: usize, bz: usize, hash: u64) {
        // Trunk height: 2–5 Minecraft blocks, scaled to voxels.
        let trunk_h = (2 + (hash % 4) as usize) * self.voxels_per_block as usize;
        for ty in 0..trunk_h {
            let y = base_y + ty;
            if y >= CHUNK_Y {
                return;
            }
            if bx < CHUNK_XZ && bz < CHUNK_XZ {
                chunk.set(bx, y, bz, WOOD);
            }
        }

        // Canopy: small irregular sphere of leaves, radius 1–2 voxels.
        let canopy_y = base_y + trunk_h;
        // Canopy radius: 1–2 Minecraft blocks, scaled to voxels.
        let radius: i32 = (1 + (hash >> 3 & 1) as i32) * self.voxels_per_block as i32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    // Rough sphere test.
                    if dx * dx + dy * dy + dz * dz > radius * radius + 1 {
                        continue;
                    }
                    let lx = bx as i32 + dx;
                    let ly = canopy_y as i32 + dy;
                    let lz = bz as i32 + dz;
                    if lx < 0
                        || lx >= CHUNK_XZ as i32
                        || ly < 0
                        || ly >= CHUNK_Y as i32
                        || lz < 0
                        || lz >= CHUNK_XZ as i32
                    {
                        continue;
                    }
                    // Don't overwrite trunk or solid terrain.
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
    seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
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
        assert_ne!(c1.blocks, c2.blocks, "different seeds produced identical chunks");
    }

    #[test]
    fn deterministic_across_calls() {
        let wgen = make_gen(777);
        let c1 = wgen.generate_chunk(3, -5);
        let c2 = wgen.generate_chunk(3, -5);
        assert_eq!(c1.blocks, c2.blocks, "same chunk generated differently on second call");
    }
}
