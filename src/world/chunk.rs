// world/chunk.rs — Chunk data structure and sparse chunk map (Phase VW-1)
//
// A chunk is a fixed-size 16 × 256 × 16 column of voxels stored as a flat
// u16 block-ID array in XZY-major order (matching Minecraft's convention).
// Y = 0 is the bedrock floor; Y = 255 is the build limit.
//
// No wgpu imports — this module is independently testable.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Chunk dimensions
// ---------------------------------------------------------------------------

pub const CHUNK_XZ: usize = 16;
pub const CHUNK_Y: usize = 256;
pub const CHUNK_SIZE: usize = CHUNK_XZ * CHUNK_Y * CHUNK_XZ; // 65 536

/// Voxel edge length in world metres (1/8 Minecraft scale).
pub const VOXEL_SIZE: f32 = 0.125;

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------

/// A 16 × 256 × 16 column of voxels.
///
/// Block IDs are stored in **XZY-major order**:
/// ```text
/// index(x, y, z) = x * CHUNK_Y * CHUNK_XZ + y * CHUNK_XZ + z
/// ```
/// where `x`, `z` ∈ `[0, CHUNK_XZ)` and `y` ∈ `[0, CHUNK_Y)`.
///
/// Block 0 (`AIR`) is the default fill value when a new chunk is allocated.
pub struct Chunk {
    pub blocks: Box<[u16; CHUNK_SIZE]>,
}

impl Chunk {
    /// Allocate a new all-AIR chunk on the heap.
    ///
    /// Using `Box` prevents a 128 KiB stack allocation.
    pub fn new() -> Box<Self> {
        Box::new(Self {
            blocks: Box::new([0u16; CHUNK_SIZE]),
        })
    }

    /// Compute the flat index for `(x, y, z)`.  Panics in debug mode on
    /// out-of-bounds coordinates.
    #[inline]
    pub fn idx(x: usize, y: usize, z: usize) -> usize {
        debug_assert!(x < CHUNK_XZ, "x={x} out of [0, {CHUNK_XZ})");
        debug_assert!(y < CHUNK_Y, "y={y} out of [0, {CHUNK_Y})");
        debug_assert!(z < CHUNK_XZ, "z={z} out of [0, {CHUNK_XZ})");
        x * CHUNK_Y * CHUNK_XZ + y * CHUNK_XZ + z
    }

    /// Return the block ID at `(x, y, z)`.
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> u16 {
        self.blocks[Self::idx(x, y, z)]
    }

    /// Set the block ID at `(x, y, z)`.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, id: u16) {
        self.blocks[Self::idx(x, y, z)] = id;
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self {
            blocks: Box::new([0u16; CHUNK_SIZE]),
        }
    }
}

// ---------------------------------------------------------------------------
// ChunkMap
// ---------------------------------------------------------------------------

/// Sparse map from chunk column coordinates `(cx, cz)` to loaded chunks.
///
/// Chunk coordinates relate to world block coordinates as:
/// ```text
/// world_x = cx * CHUNK_XZ + local_x
/// world_z = cz * CHUNK_XZ + local_z
/// ```
#[derive(Default)]
pub struct ChunkMap(pub HashMap<(i32, i32), Box<Chunk>>);

impl ChunkMap {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn insert(&mut self, cx: i32, cz: i32, chunk: Box<Chunk>) {
        self.0.insert((cx, cz), chunk);
    }

    pub fn get(&self, cx: i32, cz: i32) -> Option<&Chunk> {
        self.0.get(&(cx, cz)).map(|b| b.as_ref())
    }

    #[allow(dead_code)]
    pub fn remove(&mut self, cx: i32, cz: i32) -> Option<Box<Chunk>> {
        self.0.remove(&(cx, cz))
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Helper: world origin of a chunk column (bottom-left-front corner)
// ---------------------------------------------------------------------------

/// Return the world-space XZ origin of chunk `(cx, cz)` at Y = 0.
///
/// Vertex positions in the mesher are computed relative to this origin.
#[inline]
pub fn chunk_world_origin(cx: i32, cz: i32) -> [f32; 3] {
    [
        cx as f32 * CHUNK_XZ as f32 * VOXEL_SIZE,
        0.0,
        cz as f32 * CHUNK_XZ as f32 * VOXEL_SIZE,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_formula() {
        let mut chunk = Chunk::new();
        // Set each corner to a distinct value and read back.
        let corners = [
            (0usize, 0usize, 0usize, 1u16),
            (15, 0, 0, 2),
            (0, 255, 0, 3),
            (0, 0, 15, 4),
            (15, 255, 0, 5),
            (15, 0, 15, 6),
            (0, 255, 15, 7),
            (15, 255, 15, 8),
        ];
        for (x, y, z, id) in corners {
            chunk.set(x, y, z, id);
        }
        for (x, y, z, id) in corners {
            assert_eq!(chunk.get(x, y, z), id, "mismatch at ({x},{y},{z})");
        }
    }

    #[test]
    fn new_chunk_is_all_air() {
        let chunk = Chunk::new();
        for x in 0..CHUNK_XZ {
            for y in 0..CHUNK_Y {
                for z in 0..CHUNK_XZ {
                    assert_eq!(chunk.get(x, y, z), 0);
                }
            }
        }
    }

    #[test]
    fn chunk_map_insert_get_remove() {
        let mut map = ChunkMap::new();
        let mut c = Chunk::new();
        c.set(1, 2, 3, 42);
        map.insert(5, -3, c);
        assert_eq!(map.get(5, -3).unwrap().get(1, 2, 3), 42);
        assert!(map.get(0, 0).is_none());
        map.remove(5, -3);
        assert!(map.get(5, -3).is_none());
    }

    #[test]
    fn chunk_world_origin_values() {
        let [ox, oy, oz] = chunk_world_origin(2, -3);
        assert!((ox - 2.0 * 16.0 * 0.125).abs() < 1e-6);
        assert_eq!(oy, 0.0);
        assert!((oz - (-3.0) * 16.0 * 0.125).abs() < 1e-6);
    }
}
