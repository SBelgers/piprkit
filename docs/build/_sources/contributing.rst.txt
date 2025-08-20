Contributing
============

We welcome contributions to Pypipr! This guide will help you get started.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/yourusername/pypipr_package.git
      cd pypipr_package

3. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install in development mode:

   .. code-block:: bash

      pip install -e .
      pip install -r docs/requirements.txt

Running Tests
-------------

Run the test suite to ensure everything works:

.. code-block:: bash

   pytest

Code Style
-----------

We follow PEP 8 style guidelines. Please ensure your code:

* Uses descriptive variable names
* Includes appropriate docstrings
* Follows the existing code structure
* Includes type hints where appropriate

Documentation
-------------

When adding new features:

* Update docstrings using NumPy style
* Add examples to docstrings when helpful
* Update relevant example notebooks
* Add new API documentation if needed

Submitting Changes
------------------

1. Create a new branch for your feature:

   .. code-block:: bash

      git checkout -b feature-name

2. Make your changes and commit them:

   .. code-block:: bash

      git add .
      git commit -m "Description of changes"

3. Push to your fork and submit a pull request

Pull Request Guidelines
-----------------------

* Include a clear description of the changes
* Reference any related issues
* Ensure tests pass
* Update documentation as needed
* Keep changes focused and atomic

Bug Reports
-----------

When reporting bugs, please include:

* Python version and operating system
* Pypipr version
* Minimal code example reproducing the issue
* Full error traceback
* Expected vs. actual behavior

Feature Requests
----------------

Feature requests should include:

* Clear description of the proposed feature
* Use case and motivation
* Possible implementation approach
* Willingness to contribute the implementation

Code of Conduct
---------------

Please be respectful and constructive in all interactions. We strive to maintain
a welcoming environment for contributors of all backgrounds and experience levels.
