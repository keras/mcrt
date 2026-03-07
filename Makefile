# Makefile — helper targets for the mcrt render regression testing framework.
#
# All generated artefacts land in tmp/regression/ which is excluded from git.
# Only source files (scene YAMLs, sidecar TOMLs, generator scripts) are
# committed.  Run `make help` for a brief description of each target.

.PHONY: help web web-release web-serve gen-test-assets regress-baseline regress clean-regression

TRUNK := ./bin/trunk
TRUNK_VERSION := 0.21.4

# Default FROM commitish when not supplied by the caller.
FROM ?= HEAD^

# ---------------------------------------------------------------------------
# web / web-release / web-serve — WASM build targets using Trunk.
#
# trunk is installed project-locally into ./bin/ on first use; no global
# install is needed.  Requires the wasm32-unknown-unknown Rust target:
#   rustup target add wasm32-unknown-unknown
#
# trunk reads web/index.html and Trunk.toml automatically.
# Output lands in dist/ (see Trunk.toml: dist = "dist").
# ---------------------------------------------------------------------------

# Bootstrap: install trunk into ./bin/ if not already present.
$(TRUNK):
	@echo "→ Installing trunk $(TRUNK_VERSION) into ./bin/ …"
	cargo install --locked trunk --version $(TRUNK_VERSION) --root .
	@echo "✓ trunk installed at $(TRUNK)"

# Bootstrap: add wasm32 target if not already installed.
.PHONY: wasm-target
wasm-target:
	@rustup target list --installed | grep -q wasm32-unknown-unknown \
		|| (echo "→ Installing wasm32-unknown-unknown target …" \
		    && rustup target add wasm32-unknown-unknown)

web: $(TRUNK) wasm-target
	$(TRUNK) build

web-release: $(TRUNK) wasm-target
	$(TRUNK) build --release

web-serve: $(TRUNK) wasm-target
	$(TRUNK) serve

# Default target: print help.
help:
	@echo "Targets:"
	@echo "  web               Build the WASM web renderer (debug) into dist/"
	@echo "  web-release       Build the WASM web renderer (release) into dist/"
	@echo "  web-serve         Build and serve the web renderer at http://localhost:8080"
	@echo "  gen-test-assets   Generate all synthetic test assets (HDR skymaps, etc.)"
	@echo "  regress-baseline  Capture regression baseline from a git commitish."
	@echo "                    Usage: make regress-baseline FROM=HEAD^"
	@echo "  regress           Full pipeline: build, capture baseline, render, compare."
	@echo "                    Usage: make regress [FROM=HEAD^]"
	@echo "  clean-regression  Remove all generated artefacts in tmp/regression/"

# ---------------------------------------------------------------------------
# gen-test-assets — create the synthetic HDR environment map used by regression
# tests that exercise the env-map sampling code path.
#
# Requires uv (https://github.com/astral-sh/uv) to be on PATH.
# uv automatically resolves and caches the Python environment declared in the
# PEP 723 inline script metadata; no manual venv activation is needed.
# ---------------------------------------------------------------------------
gen-test-assets:
	@echo "→ Generating synthetic test HDR skymap …"
	@mkdir -p tmp/regression/skyboxes
	uv run tests/assets/gen_test_skymap.py \
		--width 512 --height 256 \
		--sky-horizon-nit 0.4 --sky-zenith-nit 0.8 \
		--output tmp/regression/skyboxes/test_sky.hdr
	@echo "✓ Test assets generated in tmp/regression/"

# ---------------------------------------------------------------------------
# regress — full regression pipeline in one command.
#
# 1. Builds the release binary from the current working copy.
# 2. Captures a baseline from <FROM> (default: HEAD^) using
#    scripts/regress_baseline.sh.
# 3. Resolves the short SHA so the env var points to the right directory.
# 4. Runs `cargo test regression_suite` with the env vars set.
#
# Usage:
#   make regress              # compare HEAD vs HEAD^
#   make regress FROM=a1b2c3  # compare HEAD vs a specific commit
#   make regress FROM=HEAD~5  # compare HEAD vs 5 commits back
# ---------------------------------------------------------------------------
regress:
	@echo "→ Building release binary …"
	cargo build --release
	@echo "→ Capturing baseline from $(FROM) …"
	bash scripts/regress_baseline.sh --from $(FROM)
	$(eval SHA := $(shell git rev-parse --short $(FROM)))
	@echo "→ Running regression suite (baseline: $(SHA)) …"
	MCRT_REGRESSION_BASELINE=tmp/regression/baselines/$(SHA) \
	MCRT_REGRESSION_CURRENT=tmp/regression/current \
	cargo test --release regression_suite -- --nocapture

# ---------------------------------------------------------------------------
# regress-baseline — check out <FROM>, build it, render all regression scenes,
# and store results in tmp/regression/baselines/<short-sha>/.
#
# Usage:  make regress-baseline FROM=HEAD^
#         make regress-baseline FROM=a1b2c3d
# ---------------------------------------------------------------------------
regress-baseline:
	@test -n "$(FROM)" || (echo "error: FROM is required — e.g. make regress-baseline FROM=HEAD^"; exit 1)
	bash scripts/regress_baseline.sh --from $(FROM)

# ---------------------------------------------------------------------------
# clean-regression — wipe all generated regression artefacts.
# The tmp/ directory is git-ignored, so this is safe to run at any time.
# ---------------------------------------------------------------------------
clean-regression:
	@echo "→ Removing tmp/regression/ …"
	rm -rf tmp/regression/
	@echo "✓ Done."
