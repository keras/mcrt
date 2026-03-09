// world/mod.rs — Procedural voxel world generator (Phases VW-1 through VW-5)
//
// This module contains all world-generation code with zero wgpu dependencies,
// keeping it independently testable and usable from both the interactive
// renderer and headless batch renderer.
//
// Module layout:
//   block     — Block ID constants, BlockMeta, BLOCK_META[256]
//   chunk     — Chunk (65 536 u16 flat array), ChunkMap
//   generator — WorldGenerator: FBM terrain + cave + vegetation
//   mesher    — Greedy mesh extraction, BlockMaterialMap

pub mod block;
pub mod chunk;
pub mod generator;
pub mod mesher;

// Re-export the most commonly used types at the crate level.
pub use chunk::{CHUNK_XZ, VOXEL_SIZE, ChunkMap, chunk_world_origin};
pub use generator::WorldGenerator;
pub use mesher::{BlockMaterialMap, merge_chunk_meshes, mesh_chunk};

// ---------------------------------------------------------------------------
// WorldConfig — parsed from the `world:` YAML block in scene files
// ---------------------------------------------------------------------------

/// World generation parameters returned by the scene loader.
///
/// Kept in this module so it has no serde dependency (the YAML parsing lives
/// in `scene.rs` which owns the serde derivations).
#[derive(Clone, Debug)]
pub struct WorldConfig {
    /// 64-bit seed for deterministic terrain and vegetation generation.
    pub seed: u64,
    /// Half-size of the square chunk region rendered around the world origin.
    /// `view_distance = 4` loads a 9×9 = 81 chunk area (≈ 18 m × 18 m).
    pub view_distance: i32,
    /// Number of MCRT voxels that equal one Minecraft block.
    ///
    /// At `VOXEL_SIZE = 0.125 m`, one Minecraft block = 8 voxels (the
    /// "1/8 block scale" design target).  Set to 1 for a 1:1 voxel-to-block
    /// mapping (1 voxel = 1 m), or any positive integer in between.
    /// All terrain heights and vegetation sizes scale proportionally.
    pub voxels_per_block: u32,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            view_distance: 4,
            voxels_per_block: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// visible_chunk_coords — camera-centred chunk set
// ---------------------------------------------------------------------------

/// Return all chunk column coordinates `(cx, cz)` within `view_dist` steps
/// of the chunk containing `cam_pos`.
///
/// Produces a `(2*view_dist + 1)²` grid centred on the camera's chunk.
pub fn visible_chunk_coords(cam_pos: [f32; 3], view_dist: i32) -> Vec<(i32, i32)> {
    let cam_cx = (cam_pos[0] / (CHUNK_XZ as f32 * VOXEL_SIZE)).floor() as i32;
    let cam_cz = (cam_pos[2] / (CHUNK_XZ as f32 * VOXEL_SIZE)).floor() as i32;

    let mut coords = Vec::new();
    for dcx in -view_dist..=view_dist {
        for dcz in -view_dist..=view_dist {
            coords.push((cam_cx + dcx, cam_cz + dcz));
        }
    }
    coords
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_coords_count() {
        let coords = visible_chunk_coords([0.0, 0.0, 0.0], 4);
        assert_eq!(coords.len(), 9 * 9); // (2*4+1)^2
    }

    #[test]
    fn visible_coords_centred_on_camera_chunk() {
        // Camera inside chunk (2, 3) (world x = 2*16*0.125 = 4.0, z = 3*16*0.125 = 6.0).
        let coords =
            visible_chunk_coords([4.0 + 0.5, 0.0, 6.0 + 0.5], 2);
        // Centre chunk must be present.
        assert!(coords.contains(&(2, 3)));
    }
}
