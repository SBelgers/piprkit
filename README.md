# Unit Test Status
![Unit Tests](https://github.com/SBelgers/pypipr_package/actions/workflows/tests.yml/badge.svg)
# pypipr - Pupil Data Processing Package

A Python package for processing pupil data, with a focus on the post illumination pupil response (PIPR).

## Features

- Core data structures for pupillometry: `PupilMeasurement` and `PupilSeries`.
- Utilities for light stimuli: `LightStimulus` and `LightStimuliSeries` with plotting and time-offset support.
- Data loaders and example data: `load_real_series`, `load_simulated_pupil`, and `simulate_pupil_measurement` for quick demos and tests.
- Preprocessing helpers: rolling mean/median, rate-of-change limiting, interpolation, trimming, and NaN handling.
- Fitting framework: phase-based fitting (baseline, latency, constriction, sustained, redilation) and `PupilFit` convenience wrapper.
- Basic metrics: baseline calculation and window-based helpers like `pipr_6s`, `pipr_xs`, `peak_constriction`, and `time_to_peak`.

## Feature Completeness
| Feature Category          | Status                  | Description                                                                           |
| ------------------------- | ----------------------- | ------------------------------------------------------------------------------------- |
| Core Data Classes         | ✅ Implemented           | `PupilMeasurement`, `PupilSeries`                                                     |
| Light-Stimulus Utilities  | ✅ Implemented           | Utilities for stimulus handling                                                       |
| Preprocessing Filters     | ✅ Implemented           | Includes smoothing, normalization, etc.                                               |
| Basic Metrics             | ✅ Implemented           | `baseline`, `pipr_6s`                                                                 |
| Data Loaders              | ⚠️ Partially Implemented           | Functions to load example and user datasets implemented, better file loading required.                                          |
| Fitting Pipeline          | ⚠️ Partially Implemented | Some phase fits work; `FitConstrict` and `PupilFit` have TODOs, may return NaNs       |
| FitLatency                | ❌ Not Implemented       | Stub only                                                                             |
| Advanced PLR/PIPR Metrics | ❌ Not Implemented       | `transient_plr`, `plr_latency`, `constriction_v`, `redilation_v`, `auc_*`, `net_pipr` |

See `TODO.md` for an extensive list.

## Installation


## Examples

Check out the `examples/` directory for Jupyter notebooks demonstrating:
- Basic measurement processing
- Time series analysis
- Complete analysis workflows


## License

[Add your license information here]

## Author

S. Belgers
