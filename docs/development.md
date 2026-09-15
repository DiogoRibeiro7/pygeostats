# Development

## Set up

A Rust toolchain is required, since the core extension is compiled. The compiler
version is pinned in `rust-toolchain.toml`, and rustup fetches it automatically.

```bash
git clone https://github.com/DiogoRibeiro7/pygeostats.git
cd pygeostats
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,test]"
```

maturin is the build backend, so `pip` compiles the extension. With the virtual
environment active, `maturin develop` rebuilds it in place after changes to the Rust
sources.

## Tests and checks

```bash
pytest tests/
black --check src/python/ tests/
ruff check src/python/ tests/
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings

# extension-module tells the linker not to link libpython, which is right for the
# extension but leaves a plain `cargo test` with undefined Python symbols on Linux
# and macOS, so the Rust tests run without it
cargo test --no-default-features --features parallel
```

CI runs all of these on Linux, macOS and Windows.

## Documentation

This site is built with [MkDocs](https://www.mkdocs.org) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme. The API reference is
generated from the docstrings by [mkdocstrings](https://mkdocstrings.github.io),
which reads the Python sources directly, so the extension does not need to be built
to work on the site.

```bash
pip install -r docs/requirements.txt
mkdocs serve   # live preview at http://127.0.0.1:8000
mkdocs build --strict
```

Docstrings use the NumPy style. Every `python` code block in the pages, and in the
README, runs as part of the test suite, in `tests/test_docs_examples.py`, one namespace per page, so keep
examples fast. Fence Python that should not run, such as an example that needs an
optional dependency, as `py`: it is highlighted the same way but skipped.

CI builds the site with `--strict` on every pull request, and the site is published
to GitHub Pages from `main`.

## Contributing

See
[CONTRIBUTING.md](https://github.com/DiogoRibeiro7/pygeostats/blob/main/CONTRIBUTING.md)
for coding standards and the pull request process.
