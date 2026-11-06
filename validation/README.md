# Validation against PupilMetrics and PyPlr

Scripts that run PIPRkit next to PupilMetrics (Amiot et al., 2024) and PyPlr (Martin et al., 2022) on the packaged
real recording, in similar pipelines. Small differences between the tools are not corrected: the report lists them
and compares the outcomes by correlation. They are not tests.

They use PIPRkit and PyPlr 1.0.3 from PyPI, and PupilMetrics at commit b4564fd. PupilMetrics (GPL-3.0) is not
included; download it separately.

Run from the repository root, in a venv with PIPRkit, PyPlr, pandas 2.2 and seaborn:

    python validation/pipeline_pyplr.py
    python validation/pipeline_pupilmetrics.py
    python validation/pupilmetrics_reference.py <PupilMetrics folder>
    python validation/report.py

The report is written to validation/output/report.log, with its plots next to it.

## References

- Amiot, V., Tomasoni, M., Minier, A., Gisselbaek, S., Kawasaki, A., & Kostic, C. (2024). PupilMetrics: A support system for preprocessing of pupillometric data and extraction of outcome measures. Scientific Reports, 14(1), 28775. https://doi.org/10.1038/s41598-024-79920-z
- Martin, J. T., Pinto, J., Bulte, D. P., & Spitschan, M. (2022). PyPlr: A versatile, integrated system of hardware and software for researching the human pupillary light reflex. Behavior Research Methods, 54(6), 2720–2739. https://doi.org/10.3758/s13428-021-01759-3
