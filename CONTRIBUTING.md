---
>-
  Participation is governed by our **[Code of Conduct](CODE_OF_CONDUCT.md)**. Be
  respectful and constructive.
---

# Contributing to PySpatialStats

Thanks for helping improve **PySpatialStats**. This guide explains how to set up your environment, coding standards, how to run tests and docs, and how to propose changes.

> TL;DR checklist is at the end. Please read the standards once before opening your first PR.

## Development Environment

**Requirements**

- Python **3.11 or newer** (CI covers 3.11 through 3.14)
- Rust **stable** (with `cargo`, `rustfmt`, `clippy`)
- Git

No BLAS or LAPACK is required: `ndarray` and `nalgebra` are used in their
pure-Rust configurations.

### Quickstart

```bash
git clone https://github.com/DiogoRibeiro7/pygeostats.git
cd pygeostats

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# maturin is the build backend, so pip compiles the Rust extension for you
pip install --upgrade pip
pip install -e ".[dev,test]"
```

With the virtualenv active, `maturin develop --extras dev` rebuilds the
extension in place. It requires an active virtualenv and fails without one.

--------------------------------------------------------------------------------

## Project Layout (key paths)

```
pygeostats/
├─ src/python/pygeostats/   # Python API (variogram, kriging, etc.)
├─ src/rust/                    # Rust core crates (if split)
├─ tests/                       # pytest suite
├─ examples/                    # scripts & notebooks
└─ docs/                        # MkDocs site
```

--------------------------------------------------------------------------------

## Coding Standards

### Python

- **Typing** mandatory on all public functions, classes, and module-level variables.
- Use **NumPy-style docstrings**; the API reference is generated from them.
- Keep **inline comments** concise and informative. Prefer explaining _why_ over _what_ when the code is self-evident.
- Prefer **NumPy**/**SciPy** primitives; avoid heavy deps unless justified.
- Raise **specific exceptions** with clear, actionable messages.
- Public API should be stable and minimal. Mark experimental APIs with a `Warnings` section in the docstring and `@deprecated` notes when applicable.

**Linters & Formatters** (run locally or use `pre-commit`):

```bash
# Format & lint
black src/python/ tests/
ruff check src/python/ tests/
ruff format src/python/ tests/     # if using ruff format
mypy src/python/pygeostats/
```

**Mypy**: aim for `--strict` cleanliness in new/modified modules. If a narrow `# type: ignore` is needed, include a short justification.

### Rust

- Keep code `cargo fmt`-clean and **clippy**-clean (`-D warnings`).
- Prefer explicit types and **document safety** on `unsafe` blocks.
- Benchmarks go under `benches/` with `criterion` where possible.

--------------------------------------------------------------------------------

## Tests

We require tests for new features and bug fixes.

- **Unit tests**: `pytest -v` under `tests/`
- **Coverage**: CI enforces coverage via Codecov; add tests to keep or raise coverage.
- **Property-based tests**: prefer `hypothesis` for numerical properties (e.g., semivariance non-negativity, covariance monotonicity).
- **Numerical tolerances**: use `np.isclose(..., rtol=..., atol=...)` with documented rationale.

**Run locally**

```bash
pytest tests/ -v --cov=pygeostats --cov-report=term-missing
```

--------------------------------------------------------------------------------

## Documentation

- The site is built with MkDocs and the Material theme. The API reference is
  generated from the docstrings by mkdocstrings.
- Docstrings use the NumPy style. All public APIs need one, with **usage notes**
  and **parameter constraints**.
- Every `python` code block under `docs/` runs in `tests/test_docs_examples.py`, one
  namespace per page. Keep examples short and fast. Fence Python that should not
  run, such as an example needing an optional dependency, as `py`: it is
  highlighted the same way but skipped.

**Build docs**

```bash
pip install -r docs/requirements.txt
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build --strict   # what CI runs
```

--------------------------------------------------------------------------------

## Benchmarks

If you change core numerics (variogram, kriging kernels, solvers), run and paste results from `benchmarks/` scripts in the PR description.

```bash
python benchmarks/benchmark_variogram.py
python benchmarks/compare_with_gstat.py
```

State CPU, Python, Rust, BLAS, and OS.

--------------------------------------------------------------------------------

## Git & PR Process

- Work from a feature branch off `main`.
- **Conventional Commits** for messages (e.g., `feat:`, `fix:`, `perf:`, `docs:`).
- Small, focused PRs. Update tests and docs alongside code.
- PR description should include: motivation, approach, validation (tests/bench/plots), and risk/limitations.
- All status checks must pass: format, lint, type-check, tests, docs build.

**Branch protection**

- `main`: the default branch and the target for pull requests; merges via
  squash or rebase. Releases are tagged from it.

The repository previously used `develop` as its default branch. It has been
renamed to `main`, and there is no longer a separate integration branch.

--------------------------------------------------------------------------------

## Versioning & Releases

- We use **Semantic Versioning**.
- Deprecate before breaking. Add warnings and document replacements.
- Releases are cut from `main` with tags `vX.Y.Z`. Publishing wheels is automated via CI (`release.yml`).

--------------------------------------------------------------------------------

## Issue Reporting

Please include:

- Environment (OS, Python, Rust, BLAS)
- Exact versions (`pip freeze`, `rustc --version`)
- Minimal reproducible example (code + data shape/sizes)
- Expected vs actual behavior and error trace

--------------------------------------------------------------------------------

## Large Files & Data

- Do **not** commit large datasets. Use small synthetic fixtures.
- Use Git LFS only when strictly necessary.

--------------------------------------------------------------------------------

## PR Checklist (copy into your PR)

- [ ] Code follows style guides; public APIs fully typed
- [ ] Inline comments explain non-obvious logic; docstrings updated
- [ ] Lint/format pass: `black`, `ruff`, `mypy`
- [ ] Rust checks pass: `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`, and
      `cargo test --no-default-features --features parallel` (the flags matter: `extension-module`
      stops libpython being linked, so a plain `cargo test` fails to link on Linux and macOS)
- [ ] Tests added/updated; coverage not reduced
- [ ] Docs updated and `mkdocs build --strict` succeeds
- [ ] Benchmarks run for core numeric changes; results included
- [ ] No large files; CI green

Thank you for contributing to PySpatialStats. Your efforts help make spatial analysis faster and more reliable for everyone.
