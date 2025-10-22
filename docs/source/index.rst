.. Pypipr documentation master file, created by
   sphinx-quickstart on Tue Aug 19 11:25:46 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==============
Pypipr Package
==============

**A Python package for processing pupil data, with a focus on the post illumination pupil response (PIPR).**

.. image:: https://img.shields.io/badge/python-3.12+-blue.svg
   :target: https://python.org
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :alt: License

Overview
========

Pypipr is a comprehensive Python package designed for analyzing pupil diameter data, with specialized functionality for post-illumination pupil responses (PIPR). The package provides tools for data loading, preprocessing, analysis, and visualization of pupillometry data.

Key Features
============

* **Data Loading**: Support for various pupil data formats and simulated data generation
* **Preprocessing**: Advanced filtering, blink detection, and baseline correction
* **Analysis**: Comprehensive pupil metrics and fitting algorithms
* **Time Series**: Batch processing and analysis of multiple measurements
* **Visualization**: Rich plotting capabilities for pupil data exploration
* **Light Stimuli**: Tools for modeling and analyzing light stimulus effects

Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install pypipr

Basic Usage
-----------

.. code-block:: python

   import pypipr
   
   # Load pupil data
   measurement = pypipr.load_simulated_pupil()
   
   # Basic preprocessing
   measurement.filter_data()
   measurement.detect_blinks()
   
   # Calculate metrics
   metrics = pypipr.pupil_metrics.calculate_basic_metrics(measurement)
   
   # Visualize
   measurement.plot()

API Reference
=============

Main Classes and Functions
--------------------------

.. autosummary::
   :toctree: _autosummary
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

Examples and Tutorials
======================

The package includes comprehensive Jupyter notebook examples:

**Pupil Measurement Examples:**

* Basic pupil simulation and data loading
* Data visualization and exploration
* Preprocessing and filtering
* Baseline correction techniques
* Blink detection algorithms
* Pupil metrics calculation
* Advanced fitting methods
* Custom analysis workflows
* Light color comparison studies
* Quality control procedures
* Data export and reporting

**Pupil Series Examples:**

* Series data loading and management
* Batch preprocessing workflows
* Parallel processing techniques
* Trial comparison analysis
* Habituation effect studies
* Consistency metrics
* Continuous monitoring setups
* Experimental paradigm analysis
* Advanced visualization techniques

.. toctree::
   :maxdepth: 2
   :caption: Examples:
   
   examples/pupil_measurement
   examples/pupil_series

.. toctree::
   :maxdepth: 1
   :caption: Additional Information:
   
   installation
   contributing
   changelog
   license

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. note::
   This documentation is automatically generated from the source code.
   For the most up-to-date information, please refer to the GitHub repository.

.. tip::
   Check out the example notebooks in the ``examples/`` directory for hands-on tutorials
   and practical usage scenarios.

