PySpatialStats Documentation
============================

Welcome to PySpatialStats, a high-performance spatial statistics library for Python with Rust-accelerated core algorithms.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   tutorials/index
   api/index
   examples/index
   benchmarks/variogram_streaming
   benchmarks/kriging_parallel
   development
   changelog

Overview
--------

PySpatialStats provides comprehensive spatial statistics functionality with a focus on performance and ease of use. The library includes:

* **Variogram Analysis**: Empirical variogram computation and theoretical model fitting
* **Kriging Interpolation**: Ordinary, simple, and universal kriging with uncertainty quantification
* **Point Pattern Analysis**: Spatial clustering and pattern detection (coming soon)
* **Spatial Regression**: Spatial autocorrelation and regression modeling (coming soon)

Key Features
------------

* **High Performance**: Rust-accelerated core algorithms for maximum speed
* **Scikit-learn Compatible**: Familiar API with `.fit()`, `.predict()`, and `.score()` methods
* **GeoPandas Integration**: Native support for spatial data structures
* **Comprehensive**: Full spatial statistics workflow from data exploration to modeling
* **Well Tested**: Extensive test suite with benchmarks against established packages

Quick Example
-------------

.. code-block:: python

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

   # Make predictions
   pred_coords = np.random.uniform(0, 10, size=(50, 2))
   predictions = kriging.predict(pred_coords)

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install pygeostats

Or install from source:

.. code-block:: bash

   git clone https://github.com/username/pyspatialstats.git
   cd pygeostats
   maturin develop --extras dev

Contributing
------------

We welcome contributions! Please see our `development guide <development.html>`_ for details on how to contribute to PySpatialStats.

License
-------

PySpatialStats is released under the MIT License. See the LICENSE file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


