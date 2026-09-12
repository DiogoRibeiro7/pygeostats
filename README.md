# PySpatialStats

[![CI](https://github.com/diogoribeiro7/pyspatialstats/workflows/CI/badge.svg)](https://github.com/diogoribeiro7/pyspatialstats/actions)
[![PyPI version](https://badge.fury.io/py/pyspatialstats.svg)](https://badge.fury.io/py/pyspatialstats)
[![Documentation Status](https://readthedocs.org/projects/pyspatialstats/badge/?version=latest)](https://pyspatialstats.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://codecov.io/gh/diogoribeiro7/pyspatialstats/branch/main/graph/badge.svg)](https://codecov.io/gh/diogoribeiro7/pyspatialstats)

High-performance spatial statistics library for Python with Rust-accelerated core algorithms.

## Features

* **Variogram Analysis**: Empirical variogram computation and theoretical model fitting
* **Kriging Interpolation**: Ordinary, simple, and universal kriging with uncertainty quantification  
* **High Performance**: Rust-accelerated core algorithms for maximum speed
* **Large-Data Variograms**: Streaming accumulators, memory-mapped I/O, and sparse summaries for million-point datasets
* **Scikit-learn Compatible**: Familiar API with `.fit()`, `.predict()`, and `.score()` methods
* **GeoPandas Integration**: Native support for spatial data structures
* **Scalable Kriging**: Approximate neighbours, tiling, progress tracking, and checkpointing for large grids

## Quick Start

```python
import numpy as np
from pygeostats.variogram import EmpiricalVariogram, Variogram
from pygeostats.kriging import OrdinaryKriging

# Generate sample data
coords = np.random.uniform(0, 10, size=(100, 2))
values = np.random.randn(100)

# Compute empirical variogram
emp_vario = EmpiricalVariogram(coords, values)
emp_vario.compute()

# Fit theoretical model
model = Variogram(model='exponential')
model.fit(emp_vario.distances_, emp_vario.gamma_)

# Perform kriging
kriging = OrdinaryKriging(model)
kriging.fit(coords, values)
predictions = kriging.predict(coords)
```

## Installation

You can install PySpatialStats via pip:

```bash
pip install pygeostats
``` 

## Documentation

Comprehensive documentation is available at [https://pyspatialstats.readthedocs.io](https://pyspatialstats.readthedocs.io).

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

