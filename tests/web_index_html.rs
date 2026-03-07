//! Sanity checks for web/index.html.
//!
//! The browser enforces strict MIME-type checking for ES module scripts, so
//! `web/index.html` must let Trunk inject the hashed `<script type="module">`
//! rather than hard-coding the output filename itself.  These tests catch
//! regressions that would cause a MIME-type error at runtime without any
//! compile-time signal.

use std::fs;

fn html() -> String {
    // Resolve relative to the crate root regardless of where `cargo test` is
    // invoked from.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/web/index.html");
    fs::read_to_string(path).expect("web/index.html must exist")
}

/// Trunk must own the module script.  A hard-coded `import init from
/// './mcrt.js'` (or any bare `.js` import) breaks as soon as Trunk adds a
/// content hash to the filename.
#[test]
fn no_hardcoded_wasm_import() {
    let html = html();
    assert!(
        !html.contains("import init from"),
        "web/index.html must not contain a hard-coded `import init from …` — \
         use `<link data-trunk rel=\"rust\" />` instead and let Trunk inject \
         the hashed module script automatically"
    );
}

/// The `<link data-trunk rel="rust" />` directive is what tells Trunk to
/// compile the crate to WASM and inject the correct `<script type="module">`.
/// Without it the page loads no WASM at all.
#[test]
fn has_trunk_rust_link() {
    let html = html();
    assert!(
        html.contains("data-trunk") && html.contains(r#"rel="rust""#),
        "web/index.html must contain `<link data-trunk rel=\"rust\" />` so \
         Trunk injects the hashed WASM module script"
    );
}
