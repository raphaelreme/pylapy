# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-10-28

Large refactor and upgrade of the code base, with a unified, clearer solving
API, a new (faster) backend, a reworked sparse path, a documented benchmark
suite, and a modern tooling/packaging setup. The solving API has been reshaped
around a new `cost_limit` / `alternative_cost` model; the previous `eta`
thresholding API and the `[sparse]` extra are kept as deprecated compatibility
shims that still work but emit a `DeprecationWarning`. This prepares the major
version bump (2.0.0) that will drop the support of this previous `eta` API.

### Breaking Changes

- **Extension functions renamed and reorganised.** The pluggable shape/cost
  extensions were renamed to a consistent scheme and the sparse extensions were
  cleanly rewritten in their own `sparse` module. Notably
  `smallest_fill_0` → `smallest_fill_zero`; the cost extensions moved to a
  `fill_*_pad_{zero,inf}` / `diag_*_pad_{zero,inf}` naming (e.g. `row_cost` →
  `fill_row_pad_zero`, `diag_split_cost` → `diag_split`); and the old sparse
  extensions (`symmetric_cost`, `diag_row_cost`) were replaced by
  `diag_split_sparse`, `diag_row_no_pad` and `diag_col_no_pad`.
- **Extension attributes are now resolvable by name.** `shape_extension`,
  `cost_extension` and `sparse_extension` are properties that accept either a
  callable (as before) or a registry name (a string), resolved against the
  `SHAPE_EXTENSIONS` / `COST_EXTENSIONS` / `SPARSE_EXTENSIONS` registries.
- **`lapjv` backend now ships through `lapx`.** The `lapjv` extra installs
  `lapx` (instead of the less maintained `lapjv` package, which do not work on
  macOS). All of `lap`, `lapmod`, `lapsolver`, `lapjv` and the new `laprust`
  backend are now provided by the single `lapx` dependency.
- **Dropped Python 3.9 support.** The project now requires Python >= 3.10.

### Added

- New `laprust` backend (the Rust `JV` implementation from `lapx`), which is
  noticeably faster than all previously wrapped solvers and is now the first
  choice when auto-selecting a backend.
- Unified `cost_limit` / `alternative_cost` thresholding model shared by both
  `solve` and `sparse_solve` (both keyword-only):
  - `cost_limit`: forbids any link whose cost is above it (a node with no
    admissible link is left unmatched).
  - `alternative_cost`: the cost of leaving a row/column unmatched (the "sink"
    cost), i.e. the former soft-thresholding behaviour; it supersedes
    `cost_limit` (forbidding any link more expensive than the alternative).

  `sparse_solve` additionally renames the old `feasible` flag to
  `assume_feasible`. See *Deprecated* for the v1 `eta` → new-API mapping.
- `enforce_less_row_than_col` attribute (default `True`): transposes the
  problem internally to keep the number of rows below the number of columns,
  which is faster for most backends.
- Support for negative costs: the internal "infinity" sentinel is now the sum
  of absolute values (+1) rather than the sum of finite entries, and the
  `csgraph` / `lapmod` sparse backends shift costs into the positives
  (objective-invariant for the square matching they solve).
- `assume_feasible` fast path in `sparse_solve`: when the caller knows a full
  matching exists within `cost_limit` (and no `alternative_cost` is used), the
  feasibility sink extension is skipped and the gated matrix is solved
  directly.
- Extension registries (`SHAPE_EXTENSIONS`, `COST_EXTENSIONS`,
  `SPARSE_EXTENSIONS`) plus `no_shape_extension` / `no_cost_extension`
  pass-through extensions for backends that handle the general problem
  natively.
- Comprehensive test suite (`tests/`) covering the solver, the shape/cost/
  sparse extensions, and integration/edge cases, run with coverage.
- A reworked, documented benchmark suite: cleaner presentation of results, a
  method-comparison script, and a dedicated sparse-extension benchmark.


### Changed

- The sparse solving path was entirely rewritten for reliability and speed: the
  dense→COO conversion is now numba-JIT'd and the extended sparse matrix is
  built directly, fixing several bugs in the previously under-tested sparse
  code.
- Migrated to modern Python tooling: packaging and environment management now
  use **uv** (with the `uv_build` backend and a `src/` layout) instead of
  setuptools + pip/conda; linting and formatting use **ruff** + **mypy**
  instead of pylint + black; a pre-commit hook is provided for developers.
- CI/CD updated to the new toolchain and now runs the test suite with coverage.

### Deprecated

- **The v1 `eta` thresholding arguments.** `solve(dist, eta)` and
  `sparse_solve(dist, eta, hard, feasible)` still work (including positionally)
  but now emit a `DeprecationWarning`; they are mapped onto the new API:
  - `solve(dist, eta)` → `solve(dist, alternative_cost=eta)`.
  - `sparse_solve(dist, eta, hard, feasible)` →
    `sparse_solve(dist, cost_limit=eta, alternative_cost=(inf if hard else eta),
    assume_feasible=feasible)`.

  They will be removed in the future major release (v2).
- **The `[sparse]` extra.** Kept as a backward-compatible alias for `[csgraph]`
  (`scipy` + `numba`). Depend on `pylapy[csgraph]` (or `pylapy[lapmod]`)
  instead; it will be removed in the future major release (v2).

### Fixed

- Several edge cases in dense and sparse solving that were not tested.
