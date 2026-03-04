# Makefile — helper targets for the mcrt render regression testing framework.
#
# All generated artefacts land in tmp/regression/ which is excluded from git.
# Only source files (scene YAMLs, sidecar TOMLs, generator scripts) are
# committed.  Run `make help` for a brief description of each target.

.PHONY: help gen-test-assets clean-regression

# Default target: print help.
help:
	@echo "Targets:"
	@echo "  gen-test-assets   Generate all synthetic test assets (HDR skymaps, etc.)"
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
		--output tmp/regression/skyboxes/test_sky.hdr
	@echo "✓ Test assets generated in tmp/regression/"

# ---------------------------------------------------------------------------
# clean-regression — wipe all generated regression artefacts.
# The tmp/ directory is git-ignored, so this is safe to run at any time.
# ---------------------------------------------------------------------------
clean-regression:
	@echo "→ Removing tmp/regression/ …"
	rm -rf tmp/regression/
	@echo "✓ Done."
