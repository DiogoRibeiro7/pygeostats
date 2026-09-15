# Installation

## From PyPI

```bash
pip install pygeostats
```

pygeostats needs Python 3.11 or newer. Wheels are published for:

| Platform | Architectures |
|----------|---------------|
| Linux (manylinux 2014) | x86_64, aarch64 |
| macOS | Intel (10.12 and later), Apple silicon (11.0 and later) |
| Windows | x86_64 |

The wheels are built against Python's stable ABI, so one wheel per platform covers
every supported Python version. On a platform without a wheel, pip builds from the
source distribution, which needs a [Rust toolchain](https://rustup.rs).

!!! note "Pre-releases"

    The current release, 0.1.0a1, is a pre-release. pip installs it while no stable
    release exists. Once one does, use `pip install --pre pygeostats` to get
    pre-releases.

## Dependencies

pip installs these automatically:

| Package | Minimum version |
|---------|-----------------|
| NumPy | 1.23.2 |
| SciPy | 1.9.2 |
| pandas | 1.5.0 |
| GeoPandas | 0.11.0 |
| scikit-learn | 1.1.3 |
| Matplotlib | 3.6.0 |
| psutil | 5.9.4 |

No system libraries are needed. The Rust core uses pure-Rust linear algebra, so
there is no BLAS or LAPACK to install.

A few features use optional packages, available as extras:

| Extra | Installs | Used for |
|-------|----------|----------|
| `plotting` | plotly | Interactive plots with `backend="plotly"` |
| `progress` | tqdm | Progress bars in the parallel executors |
| `approx` | annoy | The Annoy backend for approximate neighbour search |

```bash
pip install "pygeostats[plotting,progress]"
```

## From source

Building from source needs a Rust toolchain. [rustup](https://rustup.rs) installs
one; the repository pins the compiler version in `rust-toolchain.toml`, and rustup
fetches it automatically.

```bash
git clone https://github.com/DiogoRibeiro7/pygeostats.git
cd pygeostats
pip install .
```

For development, install in editable mode with the test dependencies:

```bash
pip install -e ".[dev,test]"
```

See [Development](../development.md) for running the tests and building this site.

## Check the installation

```python
import pygeostats
from pygeostats.variogram import EmpiricalVariogram
import numpy as np

print(pygeostats.__version__)

coords = np.random.default_rng(0).uniform(0, 1, size=(10, 2))
values = np.random.default_rng(1).normal(size=10)
EmpiricalVariogram(coords, values, n_bins=4).compute()
```

If importing a kriging or variogram class fails with `No module named
'pygeostats._core'`, the compiled extension is missing. Reinstall without the cache:

```bash
pip uninstall pygeostats
pip install --no-cache-dir pygeostats
```
