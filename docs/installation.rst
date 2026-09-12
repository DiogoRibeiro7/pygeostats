Installation
============

Requirements
------------

PySpatialStats requires:

* Python 3.9 or later
* NumPy >= 1.20.0
* SciPy >= 1.7.0
* Pandas >= 1.3.0
* GeoPandas >= 0.10.0
* Scikit-learn >= 1.0.0
* Matplotlib >= 3.5.0

System Dependencies
-------------------

For optimal performance, install BLAS/LAPACK libraries:

**Ubuntu/Debian:**

.. code-block:: bash

   sudo apt-get install libopenblas-dev liblapack-dev

**macOS:**

.. code-block:: bash

   brew install openblas lapack

**Windows:**

BLAS/LAPACK libraries are typically included with scientific Python distributions like Anaconda.

Installation Methods
--------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install pygeostats

From Conda-Forge
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda install -c conda-forge pygeostats

From Source
~~~~~~~~~~~

For development or the latest features:

.. code-block:: bash

   # Install Rust (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env

   # Clone and install
   git clone https://github.com/DiogoRibeiro7/pygeostats.git
   cd pygeostats
   
   # Install maturin
   pip install maturin
   
   # Build and install in development mode
   maturin develop --extras dev

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

**BLAS/LAPACK Errors**

Install the appropriate system libraries as described above, then reinstall:

.. code-block:: bash

   pip install --force-reinstall --no-cache-dir pygeostats

**Compilation Issues on Apple Silicon**

If you encounter issues on M1/M2 Macs:

.. code-block:: bash

   export MACOSX_DEPLOYMENT_TARGET=11.0
   pip install pygeostats

For more help, please open an issue on our `GitHub repository <https://github.com/DiogoRibeiro7/pygeostats>`_.
