=======
piprkit
=======

**A Python package for processing pupil data, with a focus on the post-illumination pupil response (PIPR).**

.. image:: https://img.shields.io/pypi/v/piprkit.svg
   :target: https://pypi.org/project/piprkit/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/piprkit.svg
   :target: https://pypi.org/project/piprkit/
   :alt: Python versions

.. image:: https://img.shields.io/pypi/l/piprkit.svg
   :target: https://pypi.org/project/piprkit/
   :alt: License

.. image:: https://github.com/SBelgers/piprkit/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/SBelgers/piprkit/actions/workflows/tests.yml
   :alt: Tests


Overview
========


piprkit is made to process pupil data after collecting it with a pupillometer. It allows
processing series of measurements and individual measurements. Using very simple commands, it
offers features such as filtering, blink deletion, data smoothing, baseline correction, and
more. Common metrics such as the 6s-PIPR or AUC are easily extracted. Models of the pupil
response can be fitted to the data to improve data quality and for advanced analysis.

Everything is strongly grounded in literature; references are provided where relevant.


Installation
============

.. code-block:: bash

   pip install piprkit


.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/pupil_measurement
   examples/pupil_series
   examples/pupil_fit
   examples/example_ipynb/full_example

Clear and extensive examples can be found for :doc:`single measurements <examples/pupil_measurement>`,
:doc:`series <examples/pupil_series>` and :doc:`fitting <examples/pupil_fit>`; the full API
reference is found in :doc:`API reference <api>`.

.. toctree::
   :maxdepth: 1
   :caption: Reference

   planned_features
   api
   references
   changelog
   license


AI statement
============

The code is initially fully handwritten and checked by hand. All documentation apart from
docstrings is handwritten. Claude Opus 5.5 is used to generate unittests, aid in writing
docstrings, and to aid with optimization of the code base.
