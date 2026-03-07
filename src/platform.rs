// platform.rs — platform-abstracted asset loading
//
// Provides a uniform `load_bytes(path)` API used by scene.rs, mesh.rs, and
// texture.rs.  On native the bytes come straight from the file system; on
// WASM they are served from an in-memory cache that the browser entry point
// (web.rs) populates with HTTP fetch responses before the render loop starts.

// ---------------------------------------------------------------------------
// Native implementation
// ---------------------------------------------------------------------------

/// Read the file at `path` and return its raw bytes.
///
/// Returns `Err(message)` on any I/O failure.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_bytes(path: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// WASM implementation
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
use std::{cell::RefCell, collections::HashMap};

// Thread-local pre-fetched asset cache used on WASM.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static ASSET_CACHE: RefCell<HashMap<String, Vec<u8>>> = RefCell::new(HashMap::new());
}

/// Store `bytes` in the asset cache under `path`.
///
/// Must be called by the browser entry point (web.rs) for every asset that
/// the renderer will request via [`load_bytes`] before the event loop runs.
#[cfg(target_arch = "wasm32")]
pub fn cache_asset(path: &str, bytes: Vec<u8>) {
    ASSET_CACHE.with(|c| {
        c.borrow_mut().insert(path.to_string(), bytes);
    });
}

/// Return the pre-fetched bytes for `path` from the in-memory cache.
///
/// Returns `Err` if the asset was not previously stored with [`cache_asset`].
#[cfg(target_arch = "wasm32")]
pub fn load_bytes(path: &str) -> Result<Vec<u8>, String> {
    ASSET_CACHE.with(|c| {
        c.borrow()
            .get(path)
            .cloned()
            .ok_or_else(|| format!("asset not pre-fetched: '{path}'"))
    })
}

/// Asynchronously fetch `url` (relative to the page origin) via the browser
/// Fetch API and return the raw response bytes.
///
/// The result should be forwarded to [`cache_asset`] so that the synchronous
/// [`load_bytes`] can serve it later.
#[cfg(target_arch = "wasm32")]
pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    use wasm_bindgen::JsCast as _;
    use wasm_bindgen_futures::JsFuture;

    let window = web_sys::window().ok_or("no global window object")?;
    let resp_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|e| format!("fetch failed: {e:?}"))?;
    let resp: web_sys::Response =
        resp_value.dyn_into().map_err(|e| format!("response cast failed: {e:?}"))?;
    if !resp.ok() {
        return Err(format!("HTTP {}: {url}", resp.status()));
    }
    let ab = JsFuture::from(resp.array_buffer().map_err(|e| format!("{e:?}"))?)
        .await
        .map_err(|e| format!("array_buffer failed: {e:?}"))?;
    Ok(js_sys::Uint8Array::new(&ab).to_vec())
}
