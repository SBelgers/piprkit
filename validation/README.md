# Validation against PupilMetrics and PyPlr

Validation against PupilMetrics and PyPlr is done by comparing a pipeline 
between PupilMetrics and PIPRkit and between PyPlr and PIPRkit. 
Results are written to validation/output/final, intermediate files to validation/output/intermediate.

Pipelines are not exactly equal - a discrepancy in functions does not allow 
for this, and minor differences in implementation lead to different results.

Instead, we created similar pipelines and compare results.

## Pipeline

Only the left eye of the packaged real recording is used.

| Step | PIPRkit | PupilMetrics | PyPlr |
|---|---|---|---|
| Input | raw samples | raw samples resampled to 67 Hz | raw samples, read as 200 Hz |
| Artefacts | remove_blinks (blinks.csv), remove_constant_signal(0.125), limit_rate_of_change() | no_artefact, fit_signal, no_drop (every detected drop removed) | mask_pupil_first_derivative(3.0), mask_blinks (blinks.csv) |
| Interpolation | interpolate | in no_artefact and no_drop | interpolate_pupil |
| Smoothing | rolling_mean(0.2), before limit_rate_of_change() | none | butterworth_series, 3rd order, 4 Hz |
| Epochs | split(10, 45) | Manual separation | extract, 5 s before onset, 50 s in all |
| Baseline correction | baseline_correction_percent(5) | none | percent_signal_change |
| Metrics | default arguments, baseline(5) | fixed in its code | PLR |

PupilMetrics' input is resampled to 67 Hz, the sampling rate of the NeuroLight pupillometer it was made for. This is required for data quality: PupilMetrics checks its speed and acceleration limits sample by sample, and at the recording's 200 Hz the acceleration limit also removes the pupil constriction.

## Metrics

| Metric | PIPRkit | PupilMetrics | PyPlr |
|---|---|---|---|
| baseline (mm) | baseline(5) | baseline | Baseline |
| peak constriction (% change) | peak_constriction() | -MCA | PeakCon |
| time to peak (s) | time_to_peak() | RT | T2MaxCon |
| 6 s PIPR (% change) | pipr_6s() | -6s constriction | |

## Running the validation

Install PIPRkit and PyPlr:

    pip install piprkit==0.1.0 pyplr==1.0.3

PupilMetrics' drop detection needs numpy 1.23 or older. Install its packages in a second environment with Python 3.10:

    pip install numpy==1.23.5 scipy==1.10.1 pandas==1.5.3 matplotlib==3.7.1 seaborn==0.12.2

Download PupilMetrics at commit b4564fd
(https://github.com/JulesGoninRIO/PupilMetrics/archive/b4564fda5efcf951ed98ac168fde7e9136f80a51.zip)
and unpack it into a folder of its own. Then run, in this order, with pipeline_pupilmetrics.py in the second environment:

    python validation/prepare.py
    python validation/pipeline_piprkit.py
    python validation/pipeline_pupilmetrics.py <PupilMetrics folder>
    python validation/pipeline_pyplr.py
    python validation/compare.py
