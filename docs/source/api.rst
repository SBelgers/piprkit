API Reference
=============


This API reference outlines all public classes, functions, and methods of piprkit.

.. currentmodule:: piprkit

Data
----


Methods and classes containing the data and light stimuli, the main objects to be used in piprkit.

.. autosummary::
   :toctree: _autosummary

   PupilMeasurement
   PupilSeries
   LightStimulus
   LightStimuliSeries

Loading and saving
------------------


Load example data, simulate measurements, and read or write data to CSV files.

.. autosummary::
   :toctree: _autosummary

   load_csv
   save_csv
   load_real_series
   load_simulated_pupil
   simulate_pupil_measurement

Export Metrics
--------------


Export outcome metrics as a long table, ready to be imported in the statistics software of choice.

.. autosummary::
   :toctree: _autosummary

   metric_rows
   save_metric_rows

Preprocessing and metrics
-------------------------


Classes used for filtering (``.filters``) and extracting metrics (``.metrics``), as well
as the results of metrics and the baseline correction.

.. autosummary::
   :toctree: _autosummary

   ~preprocessing.filtering.PupilFilters
   ~analysis.pupil_metrics.PupilMetrics
   ~analysis.pupil_metrics.PhaseModelMetrics
   ~analysis.pupil_metrics.MetricResult
   ~utils.baseline.BaselineCorrection

Fitting
-------


Models that can be fit to the data. Either the four underlying phases individually,
or together through ``PhaseModel``.

.. autosummary::
   :toctree: _autosummary

   PhaseModel
   BaselinePhase
   ConstrictionPhase
   MaintainedConstrictionPhase
   RedilationPhase

Utilities
---------

Some useful utilities.

.. autosummary::
   :toctree: _autosummary

   net_pipr
   check_time_series
