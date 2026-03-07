// world/block.rs — Block type registry for the procedural voxel world (Phase VW-1)
//
// Block IDs are stable u16 integers.  The BLOCK_META static array maps each ID
// to its rendering and simulation properties.  New block types are always
// appended; existing indices must never be renumbered so that saved worlds
// remain valid.
//
// No wgpu imports — this module is independently testable.

// ---------------------------------------------------------------------------
// Block ID constants
// ---------------------------------------------------------------------------

pub const AIR: u16 = 0;
pub const LIMESTONE: u16 = 1; // primary rocky material
pub const RED_SOIL: u16 = 2; // Terra Rossa Mediterranean soil
pub const GRASS_TYP: u16 = 3; // dry grass / scrubland surface
pub const SAND: u16 = 4;
pub const WATER: u16 = 5;
pub const WOOD: u16 = 6; // olive / pine wood
pub const LEAVES: u16 = 7; // olive / pine foliage
pub const SHRUB: u16 = 8; // small bushes
pub const GRAVEL: u16 = 9; // weathered limestone debris
pub const CAVE_GLOW: u16 = 10; // phosphorescent cave flora (emissive)

/// Total number of named block types.
#[allow(dead_code)]
pub const NUM_BLOCK_TYPES: usize = 11;

// ---------------------------------------------------------------------------
// Block metadata
// ---------------------------------------------------------------------------

/// Per-block-type properties used by the mesher and generator.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct BlockMeta {
    /// Human-readable name.
    pub name: &'static str,
    /// True means this block occludes its neighbours: faces between two opaque
    /// blocks are culled by the mesher.  AIR is the only non-opaque block
    /// initially; water and leaves can be made semi-transparent later.
    pub opaque: bool,
    /// Key into the `world_materials:` YAML palette.  The mesher uses this to
    /// look up the GPU material slot index from `BlockMaterialMap`.
    pub material_name: &'static str,
    /// Linear-space base tint — informational only (albedo is controlled by
    /// the GPU material entry referenced by `material_name`).
    pub tint: [f32; 3],
}

/// Global block metadata table.  Entry 0 is AIR; entries 1–10 are the named
/// blocks above; entries 11–255 all fall back to AIR-like properties.
pub static BLOCK_META: [BlockMeta; 256] = {
    const AIR_META: BlockMeta = BlockMeta {
        name: "air",
        opaque: false,
        material_name: "air",
        tint: [0.0, 0.0, 0.0],
    };
    let mut table = [AIR_META; 256];

    table[AIR as usize] = AIR_META;
    table[LIMESTONE as usize] = BlockMeta {
        name: "limestone",
        opaque: true,
        material_name: "limestone",
        tint: [0.82, 0.78, 0.70],
    };
    table[RED_SOIL as usize] = BlockMeta {
        name: "red_soil",
        opaque: true,
        material_name: "red_soil",
        tint: [0.55, 0.25, 0.15],
    };
    table[GRASS_TYP as usize] = BlockMeta {
        name: "grass",
        opaque: true,
        material_name: "grass",
        tint: [0.35, 0.42, 0.18],
    };
    table[SAND as usize] = BlockMeta {
        name: "sand",
        opaque: true,
        material_name: "sand",
        tint: [0.76, 0.69, 0.50],
    };
    table[WATER as usize] = BlockMeta {
        name: "water",
        // Rendered as opaque placeholder until transparency phase.
        opaque: true,
        material_name: "water",
        tint: [0.10, 0.30, 0.70],
    };
    table[WOOD as usize] = BlockMeta {
        name: "wood",
        opaque: true,
        material_name: "wood",
        tint: [0.32, 0.28, 0.20],
    };
    table[LEAVES as usize] = BlockMeta {
        name: "leaves",
        opaque: true,
        material_name: "leaves",
        tint: [0.20, 0.35, 0.15],
    };
    table[SHRUB as usize] = BlockMeta {
        name: "shrub",
        opaque: true,
        material_name: "shrub",
        tint: [0.25, 0.35, 0.12],
    };
    table[GRAVEL as usize] = BlockMeta {
        name: "gravel",
        opaque: true,
        material_name: "gravel",
        tint: [0.45, 0.42, 0.40],
    };
    table[CAVE_GLOW as usize] = BlockMeta {
        name: "cave_glow",
        opaque: true,
        material_name: "cave_glow",
        tint: [0.60, 0.85, 1.0],
    };

    table
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn air_is_not_opaque() {
        assert!(!BLOCK_META[AIR as usize].opaque);
    }

    #[test]
    fn all_named_blocks_have_material_names() {
        let named = [
            LIMESTONE, RED_SOIL, GRASS_TYP, SAND, WATER, WOOD, LEAVES, SHRUB, GRAVEL, CAVE_GLOW,
        ];
        for id in named {
            let m = &BLOCK_META[id as usize];
            assert!(!m.material_name.is_empty(), "block {id} has empty material_name");
            assert!(m.opaque, "named block {id} ({}) should be opaque by default", m.name);
        }
    }

    #[test]
    fn table_is_full_256() {
        assert_eq!(BLOCK_META.len(), 256);
    }
}
