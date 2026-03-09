// world/mesher.rs — Greedy voxel mesh extraction (Phase VW-3)
//
// Converts a `Chunk` (plus optional neighbours for cross-boundary face
// culling) into a minimal triangle mesh by the greedy meshing algorithm.
//
// For each of the six face orientations (±X, ±Y, ±Z):
//  1. Sweep layer-by-layer along the face-normal axis.
//  2. Build a 2D visibility mask: each cell holds the block ID of a visible
//     face (non-AIR voxel whose neighbour in the face direction is non-opaque),
//     or 0 if the face is hidden.
//  3. Greedily grow rectangles: extend width then height while the block IDs
//     match and cells are unvisited.
//  4. Emit one quad (two CCW triangles) per merged rectangle with flat normal
//     and normalised UV coordinates spanning the quad.
//  5. Mark consumed cells as visited.
//
// The output `GpuVertex` / `GpuTriangle` types are the same types used by the
// existing mesh/BVH pipeline, so the world plugs in without any GPU changes.
//
// No wgpu imports — independently testable.

use super::block::{AIR, BLOCK_META};
use super::chunk::{CHUNK_XZ, CHUNK_Y, Chunk, VOXEL_SIZE};
use crate::mesh::{GpuTriangle, GpuVertex};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Triangle mesh produced by meshing one chunk.
pub struct ChunkMesh {
    pub vertices: Vec<GpuVertex>,
    pub triangles: Vec<GpuTriangle>,
}

/// Maps block IDs (u16) to GPU material slot indices (u32).
///
/// Built once at startup by [`BlockMaterialMap::from_palette`].
#[derive(Clone)]
pub struct BlockMaterialMap {
    pub slots: [u32; 256],
}

impl BlockMaterialMap {
    /// Build the map from a `name → GPU_slot` lookup table.
    ///
    /// Each `BLOCK_META[id].material_name` is looked up in `name_to_slot`; if
    /// not found the block falls back to `error_slot` (bright pink / index 0).
    pub fn from_palette(name_to_slot: &std::collections::HashMap<String, u32>) -> Self {
        let mut slots = [0u32; 256];
        for (id, meta) in BLOCK_META.iter().enumerate() {
            if let Some(&slot) = name_to_slot.get(meta.material_name) {
                slots[id] = slot;
            }
            // AIR or unmapped blocks stay at slot 0 (never actually rendered).
        }
        Self { slots }
    }
}

// ---------------------------------------------------------------------------
// Face orientation descriptors
// ---------------------------------------------------------------------------

/// One of the six cardinal face directions.
#[derive(Clone, Copy)]
struct FaceDir {
    /// Which axis the face normal points along (0=X, 1=Y, 2=Z).
    axis: usize,
    /// +1 or -1: which direction the normal points along `axis`.
    sign: i32,
}

const FACE_DIRS: [FaceDir; 6] = [
    FaceDir { axis: 0, sign: 1 },  // +X (right)
    FaceDir { axis: 0, sign: -1 }, // −X (left)
    FaceDir { axis: 1, sign: 1 },  // +Y (up)
    FaceDir { axis: 1, sign: -1 }, // −Y (down)
    FaceDir { axis: 2, sign: 1 },  // +Z (forward)
    FaceDir { axis: 2, sign: -1 }, // −Z (backward)
];

// ---------------------------------------------------------------------------
// Main meshing entry point
// ---------------------------------------------------------------------------

/// Convert a `Chunk` into an optimised triangle mesh.
///
/// `neighbours` provides adjacent chunks for cross-boundary face culling in
/// the order `[+X, −X, +Y, −Y, +Z, −Z]`.  `None` means treat that boundary
/// as all-AIR (conservative — faces on that edge are emitted).
///
/// `world_origin` is the world-space position of the chunk's `(0,0,0)` corner
/// (typically `chunk_world_origin(cx, cz)` from `chunk.rs`).
///
/// `material_map` resolves block IDs to GPU material indices.
pub fn mesh_chunk(
    chunk: &Chunk,
    neighbours: [Option<&Chunk>; 6],
    world_origin: [f32; 3],
    material_map: &BlockMaterialMap,
) -> ChunkMesh {
    let mut vertices: Vec<GpuVertex> = Vec::new();
    let mut triangles: Vec<GpuTriangle> = Vec::new();

    for face in FACE_DIRS {
        mesh_face_direction(
            chunk,
            &neighbours,
            world_origin,
            material_map,
            face,
            &mut vertices,
            &mut triangles,
        );
    }

    ChunkMesh { vertices, triangles }
}

// ---------------------------------------------------------------------------
// Per-direction greedy pass
// ---------------------------------------------------------------------------

fn mesh_face_direction(
    chunk: &Chunk,
    neighbours: &[Option<&Chunk>; 6],
    world_origin: [f32; 3],
    material_map: &BlockMaterialMap,
    face: FaceDir,
    vertices: &mut Vec<GpuVertex>,
    triangles: &mut Vec<GpuTriangle>,
) {
    // Map axis directions to dimension indices.
    // We sweep "layer" along `face.axis`, and the 2D mask spans the other two.
    let axis = face.axis;
    let (u_axis, v_axis) = match axis {
        0 => (1, 2), // sweep X → mask is (Y, Z)
        1 => (0, 2), // sweep Y → mask is (X, Z)
        _ => (0, 1), // sweep Z → mask is (X, Y)
    };

    // Dimension sizes for the three axes.
    let dim: [usize; 3] = [CHUNK_XZ, CHUNK_Y, CHUNK_XZ];

    let layer_dim = dim[axis];
    let u_dim = dim[u_axis];
    let v_dim = dim[v_axis];

    // Normal vector (world space).
    let mut normal = [0.0f32; 3];
    normal[axis] = face.sign as f32;

    // Neighbour chunk index for the positive or negative face boundary.
    // Ordering of `neighbours`: [+X=0, −X=1, +Y=2, −Y=3, +Z=4, −Z=5].
    let pos_neighbour_idx = axis * 2;     // +axis
    let neg_neighbour_idx = axis * 2 + 1; // −axis

    // Reusable mask buffer (u_dim × v_dim), 0 means "no visible face here".
    let mut mask = vec![0u16; u_dim * v_dim];
    // Visited flags for greedy expansion.
    let mut visited = vec![false; u_dim * v_dim];

    for layer in 0..layer_dim {
        // Build the 2D mask for this slice.
        mask.iter_mut().for_each(|m| *m = 0);
        visited.iter_mut().for_each(|v| *v = false);

        for u in 0..u_dim {
            for v in 0..v_dim {
                let mut pos = [0usize; 3];
                pos[axis] = layer;
                pos[u_axis] = u;
                pos[v_axis] = v;

                let block_id = chunk.get(pos[0], pos[1], pos[2]);
                if block_id == AIR || !BLOCK_META[block_id as usize].opaque {
                    continue;
                }

                // Compute neighbour voxel position (in direction of face normal).
                let neighbour_block =
                    sample_neighbour(chunk, neighbours, pos, axis, face.sign, pos_neighbour_idx, neg_neighbour_idx);

                // Emit face only if the neighbour is non-opaque.
                if !BLOCK_META[neighbour_block as usize].opaque {
                    mask[u * v_dim + v] = block_id;
                }
            }
        }

        // Greedy expansion.
        for u0 in 0..u_dim {
            let mut v0 = 0;
            while v0 < v_dim {
                let start_id = mask[u0 * v_dim + v0];
                if start_id == 0 || visited[u0 * v_dim + v0] {
                    v0 += 1;
                    continue;
                }

                // Extend width (along V axis) while same block ID and unvisited.
                let mut w = 1;
                while v0 + w < v_dim
                    && mask[u0 * v_dim + v0 + w] == start_id
                    && !visited[u0 * v_dim + v0 + w]
                {
                    w += 1;
                }

                // Extend height (along U axis) while the entire row matches.
                let mut h = 1;
                'outer: while u0 + h < u_dim {
                    for dv in 0..w {
                        if mask[(u0 + h) * v_dim + v0 + dv] != start_id
                            || visited[(u0 + h) * v_dim + v0 + dv]
                        {
                            break 'outer;
                        }
                    }
                    h += 1;
                }

                // Mark all consumed cells as visited.
                for du in 0..h {
                    for dv in 0..w {
                        visited[(u0 + du) * v_dim + v0 + dv] = true;
                    }
                }

                // Emit quad for this merged rectangle.
                emit_quad(
                    world_origin,
                    axis,
                    u_axis,
                    v_axis,
                    layer,
                    u0,
                    v0,
                    h,  // extent along U
                    w,  // extent along V
                    face.sign,
                    normal,
                    material_map.slots[start_id as usize],
                    vertices,
                    triangles,
                );

                v0 += w;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Neighbour voxel sampling
// ---------------------------------------------------------------------------

/// Return the block ID of the voxel adjacent to `pos` along `axis` in
/// direction `sign`.  Crosses into a neighbouring chunk if needed.
fn sample_neighbour(
    chunk: &Chunk,
    neighbours: &[Option<&Chunk>; 6],
    pos: [usize; 3],
    axis: usize,
    sign: i32,
    pos_nb_idx: usize,
    neg_nb_idx: usize,
) -> u16 {
    let dim = [CHUNK_XZ, CHUNK_Y, CHUNK_XZ];

    let coord = pos[axis] as i32 + sign;
    if coord < 0 {
        // Below lower boundary on this axis.
        if let Some(nb) = neighbours[neg_nb_idx] {
            let mut np = pos;
            np[axis] = dim[axis] - 1; // last slice of neighbour
            nb.get(np[0], np[1], np[2])
        } else {
            AIR // treat missing neighbour as air → face is emitted
        }
    } else if coord >= dim[axis] as i32 {
        // Beyond upper boundary on this axis.
        if let Some(nb) = neighbours[pos_nb_idx] {
            let mut np = pos;
            np[axis] = 0; // first slice of neighbour
            nb.get(np[0], np[1], np[2])
        } else {
            AIR
        }
    } else {
        let mut np = pos;
        np[axis] = coord as usize;
        chunk.get(np[0], np[1], np[2])
    }
}

// ---------------------------------------------------------------------------
// Quad emission
// ---------------------------------------------------------------------------

/// Emit a quad (2 CCW triangles) into `vertices` / `triangles`.
///
/// The quad spans `h` voxels along `u_axis` and `w` voxels along `v_axis`
/// starting at layer `layer` on `axis`.  `sign` selects which side of the
/// layer the face sits on (+1 = far side, −1 = near side).
#[allow(clippy::too_many_arguments)]
fn emit_quad(
    world_origin: [f32; 3],
    axis: usize,
    u_axis: usize,
    v_axis: usize,
    layer: usize,
    u0: usize,
    v0: usize,
    h: usize,
    w: usize,
    sign: i32,
    normal: [f32; 3],
    mat_idx: u32,
    vertices: &mut Vec<GpuVertex>,
    triangles: &mut Vec<GpuTriangle>,
) {
    // Compute the four corners of the quad in voxel-local coordinates.
    // The face sits at `layer + (sign == +1 ? 1 : 0)` along `axis`.
    let face_offset = if sign > 0 { 1 } else { 0 };

    // Helpers: build a [f32;3] with values on axis, u_axis, v_axis.
    let make_pos = |a: usize, u: usize, v: usize| -> [f32; 3] {
        let mut p = [0.0f32; 3];
        p[axis] = (layer + face_offset) as f32 * VOXEL_SIZE;
        p[u_axis] = u as f32 * VOXEL_SIZE;
        p[v_axis] = v as f32 * VOXEL_SIZE;
        let _ = a; // suppress unused warning
        [
            world_origin[0] + p[0],
            world_origin[1] + p[1],
            world_origin[2] + p[2],
        ]
    };

    // Four corners: (u0,v0), (u0+h,v0), (u0+h,v0+w), (u0,v0+w).
    let p0 = make_pos(layer, u0, v0);
    let p1 = make_pos(layer, u0 + h, v0);
    let p2 = make_pos(layer, u0 + h, v0 + w);
    let p3 = make_pos(layer, u0, v0 + w);

    let n4 = [normal[0], normal[1], normal[2], 0.0];
    let base = vertices.len() as u32;

    // UV: normalised across the quad extent.
    vertices.push(GpuVertex { position: [p0[0], p0[1], p0[2], 1.0], normal: n4, uv: [0.0, 0.0, 0.0, 0.0] });
    vertices.push(GpuVertex { position: [p1[0], p1[1], p1[2], 1.0], normal: n4, uv: [1.0, 0.0, 0.0, 0.0] });
    vertices.push(GpuVertex { position: [p2[0], p2[1], p2[2], 1.0], normal: n4, uv: [1.0, 1.0, 0.0, 0.0] });
    vertices.push(GpuVertex { position: [p3[0], p3[1], p3[2], 1.0], normal: n4, uv: [0.0, 1.0, 0.0, 0.0] });

    // Wind CCW when viewed from outside (i.e. in the direction of the normal).
    if sign > 0 {
        triangles.push(GpuTriangle { v: [base, base + 1, base + 2], mat_idx });
        triangles.push(GpuTriangle { v: [base, base + 2, base + 3], mat_idx });
    } else {
        // Flip winding for back faces.
        triangles.push(GpuTriangle { v: [base, base + 2, base + 1], mat_idx });
        triangles.push(GpuTriangle { v: [base, base + 3, base + 2], mat_idx });
    }
}

// ---------------------------------------------------------------------------
// Mesh merging helper
// ---------------------------------------------------------------------------

/// Merge multiple per-chunk meshes into a single flat vertex+triangle list
/// compatible with `mesh::build_mesh_bvh`.
///
/// World origins are already baked into the vertex positions by `mesh_chunk`,
/// so this function simply concatenates and offsets triangle indices.
pub fn merge_chunk_meshes(meshes: &[ChunkMesh]) -> (Vec<GpuVertex>, Vec<GpuTriangle>) {
    let total_verts: usize = meshes.iter().map(|m| m.vertices.len()).sum();
    let total_tris: usize = meshes.iter().map(|m| m.triangles.len()).sum();

    let mut vertices = Vec::with_capacity(total_verts);
    let mut triangles = Vec::with_capacity(total_tris);

    for mesh in meshes {
        let vert_offset = vertices.len() as u32;
        vertices.extend_from_slice(&mesh.vertices);
        for tri in &mesh.triangles {
            triangles.push(GpuTriangle {
                v: [
                    tri.v[0] + vert_offset,
                    tri.v[1] + vert_offset,
                    tri.v[2] + vert_offset,
                ],
                mat_idx: tri.mat_idx,
            });
        }
    }

    (vertices, triangles)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::block::LIMESTONE;
    use crate::world::chunk::Chunk;
    use std::collections::HashMap;

    fn dummy_material_map() -> BlockMaterialMap {
        let mut m = HashMap::new();
        // Map every named block to material slot 0.
        m.insert("limestone".to_string(), 0u32);
        m.insert("red_soil".to_string(), 0u32);
        m.insert("grass".to_string(), 0u32);
        m.insert("sand".to_string(), 0u32);
        m.insert("water".to_string(), 0u32);
        m.insert("wood".to_string(), 0u32);
        m.insert("leaves".to_string(), 0u32);
        m.insert("shrub".to_string(), 0u32);
        m.insert("gravel".to_string(), 0u32);
        m.insert("cave_glow".to_string(), 0u32);
        BlockMaterialMap::from_palette(&m)
    }

    #[test]
    fn air_chunk_produces_no_triangles() {
        let chunk = Chunk::new(); // all AIR
        let mat = dummy_material_map();
        let mesh = mesh_chunk(&chunk, [None; 6], [0.0, 0.0, 0.0], &mat);
        assert_eq!(mesh.triangles.len(), 0, "all-air chunk should have no triangles");
    }

    #[test]
    fn single_voxel_produces_12_triangles() {
        let mut chunk = Chunk::new();
        chunk.set(8, 8, 8, LIMESTONE);
        let mat = dummy_material_map();
        let mesh = mesh_chunk(&chunk, [None; 6], [0.0, 0.0, 0.0], &mat);
        // 6 faces × 2 triangles each = 12
        assert_eq!(
            mesh.triangles.len(),
            12,
            "single voxel should produce exactly 12 triangles"
        );
    }

    #[test]
    fn flat_2x2_slab_top_face_is_merged() {
        // A 2×1×2 slab (x=0..1, y=0, z=0..1): the top face (+Y) should be
        // merged into one quad → 2 triangles instead of 4×2=8 for naïve meshing.
        let mut chunk = Chunk::new();
        for x in 0..2 {
            for z in 0..2 {
                chunk.set(x, 0, z, LIMESTONE);
            }
        }
        let mat = dummy_material_map();
        let mesh = mesh_chunk(&chunk, [None; 6], [0.0, 0.0, 0.0], &mat);

        // Count triangles contributed by the +Y face (normal ≈ [0,1,0]).
        // The slab has 1 top face (merged → 2 tris), 4 side faces (4 quads →
        // 8 tris), 1 bottom face (merged → 2 tris): total ≤ 12.
        // Without greedy, naïve would produce 4*2 + 4*4 + 4*2 = 24 tris.
        assert!(
            mesh.triangles.len() < 24,
            "expected greedy merge to reduce triangle count below 24, got {}",
            mesh.triangles.len()
        );
    }

    #[test]
    fn adjacent_blocks_cull_shared_face() {
        // Two adjacent limestone blocks: no face between them should be emitted.
        let mut chunk = Chunk::new();
        chunk.set(0, 0, 0, LIMESTONE);
        chunk.set(1, 0, 0, LIMESTONE); // shares +X face with (0,0,0)
        let mat = dummy_material_map();
        let mesh = mesh_chunk(&chunk, [None; 6], [0.0, 0.0, 0.0], &mat);
        // Each lone 1×1×1 cube has 12 tris; two cubes sharing one face = 10 tris.
        // Greedy may merge some outer faces further, so assert ≤ 20.
        assert!(
            mesh.triangles.len() < 24,
            "shared face between adjacent blocks not culled; got {} triangles",
            mesh.triangles.len()
        );
    }

    #[test]
    fn merge_chunk_meshes_offsets_correctly() {
        let mut chunk = Chunk::new();
        chunk.set(0, 0, 0, LIMESTONE);
        let mat = dummy_material_map();
        let m1 = mesh_chunk(&chunk, [None; 6], [0.0, 0.0, 0.0], &mat);
        let m2 = mesh_chunk(&chunk, [None; 6], [2.0, 0.0, 0.0], &mat);
        let v1_count = m1.vertices.len() as u32;
        let (verts, tris) = merge_chunk_meshes(&[m1, m2]);
        // All triangle indices in the second mesh should be offset by v1_count.
        let max_idx = tris.iter().flat_map(|t| t.v).max().unwrap_or(0);
        assert_eq!(verts.len() as u32, max_idx + 1);
        // Second mesh triangle indices start at v1_count.
        let second_mesh_min_idx = tris[12..].iter().flat_map(|t| t.v).min().unwrap_or(0);
        assert_eq!(second_mesh_min_idx, v1_count);
    }
}
