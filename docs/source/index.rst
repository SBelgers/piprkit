.. Pypipr documentation master file, created by
   sphinx-quickstart on Tue Aug 19 11:25:46 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==============
PyPipr Package
==============


**A Python package for processing pupil data, with a focus on the post illumination pupil response (PIPR).**

.. image:: https://img.shields.io/badge/python-3.12+-blue.svg
   :target: https://python.org
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :alt: License


Overview
========

PyPipr is a comprehensive Python package designed for analyzing pupil diameter data, with specialized functionality for post-illumination pupil responses (PIPR). The package provides tools for data loading, preprocessing, analysis, and visualization of pupillometry data.

Key Features
============

* **Data Loading**: Support for loading continuous pupil data and simulated data generation
* **Preprocessing**: Filtering and baseline correction
* **Analysis**: Comprehensive pupil metrics and fitting algorithms
* **Pupil measurement series**: Batch processing and analysis of multiple measurements
* **Visualization**: Pplotting capabilities for pupil data exploration


Installation
------------

.. code-block:: bash

   git clone https://github.com/SBelgers/pypipr_package.git
   cd pypipr_package
   pip install -e .

Examples and Tutorials
======================

The package includes comprehensive Jupyter notebook examples


.. toctree::
   :maxdepth: 2
   :caption: Examples:

   
   examples/pupil_measurement
   examples/pupil_fit
   examples/pupil_series

Additional Information
======================

The package includes comprehensive Jupyter notebook examples

.. toctree::
   :maxdepth: 1
   :caption: Additional Information:
   
   index
   contributing
   changelog
   license


API Reference
=============

Main Classes and Functions
--------------------------

.. autosummary::
   :toctree: _autosummary
   :caption: Main Classes and Functions
   :recursive:
   
   pypipr.PupilMeasurement
   pypipr.PupilSeries
   pypipr.PupilFit
   pypipr.LightStimulus
   pypipr.LightStimuliSeries
   pypipr.load_simulated_pupil
   pypipr.load_real_series
   pypipr.simulate_pupil_measurement
   pypipr.check_time_series

.. note::
   This documentation is automatically generated from the source code.
