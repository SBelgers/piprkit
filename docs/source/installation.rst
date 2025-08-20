Installation
============

Requirements
------------

Pypipr requires Python 3.12 or higher and the following dependencies:

* numpy
* matplotlib
* scipy

Installation from PyPI
----------------------

.. code-block:: bash

   pip install pypipr

Development Installation
------------------------

For development purposes, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/SBelgers/pypipr_package.git
   cd pypipr_package
   pip install -e .

Optional Dependencies
---------------------

For documentation building:

.. code-block:: bash

   pip install -r docs/requirements.txt

For running tests:

.. code-block:: bash

   pip install pytest

Verification
------------

To verify your installation, run:

.. code-block:: python

   import pypipr
   print(pypipr.__version__)

   # Load example data
   measurement = pypipr.load_simulated_pupil()
   measurement.plot()

Troubleshooting
---------------

If you encounter installation issues:

1. Ensure you have Python 3.12 or higher
2. Update pip: ``pip install --upgrade pip``
3. Check that all dependencies are compatible
4. For development installations, ensure git is available
