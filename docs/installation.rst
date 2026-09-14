Installation
============

Requirements
------------

pygeostats requires:

* Python 3.11 or later
* NumPy >= 1.23.2
* SciPy >= 1.9.2
* Pandas >= 1.5.0
* GeoPandas >= 0.11.0
* Scikit-learn >= 1.1.3
* Matplotlib >= 3.6.0
* psutil >= 5.9.4

No system libraries are needed. The Rust core uses pure-Rust linear algebra, so
there is no BLAS or LAPACK to install.

Installation Methods
--------------------

From PyPI
~~~~~~~~~

.. code-block:: bash

   pip install pygeostats

Wheels are published for Linux (x86_64 and aarch64), macOS (Intel and Apple
silicon) and Windows (x86_64). They are built against the stable ABI, so one
wheel per platform covers Python 3.11 and newer. On platforms without a wheel,
pip builds from the source distribution, which needs a Rust toolchain.

The current release, 0.1.0a1, is a pre-release. pip installs it while no stable
release exists; once one does, use ``pip install --pre pygeostats`` to get
pre-releases.

From Source
~~~~~~~~~~~

For development or the latest changes:

.. code-block:: bash

   # Install Rust (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env

   # Clone and install, with the development and test dependencies
   git clone https://github.com/DiogoRibeiro7/pygeostats.git
   cd pygeostats
   pip install -e ".[dev,test]"

Verification
------------

Test your installation:

.. code-block:: python

   import pygeostats
   print(pygeostats.__version__)

   # Run a simple test
   import numpy as np
   from pygeostats.variogram import EmpiricalVariogram

   coords = np.random.uniform(0, 1, size=(10, 2))
   values = np.random.randn(10)
   emp_vario = EmpiricalVariogram(coords, values)
   emp_vario.compute()

   print("Installation successful!")

Troubleshooting
---------------

**Import Error: "No module named '_core'"**

This usually indicates the Rust extension wasn't compiled properly. Try:

.. code-block:: bash

   pip uninstall pygeostats
   pip install --no-cache-dir pygeostats

For more help, please open an issue on our `GitHub repository <https://github.com/DiogoRibeiro7/pygeostats>`_.
