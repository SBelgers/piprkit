# piprkit

[![PyPI](https://img.shields.io/pypi/v/piprkit.svg)](https://pypi.org/project/piprkit/)
[![Tests](https://github.com/SBelgers/piprkit/actions/workflows/tests.yml/badge.svg)](https://github.com/SBelgers/piprkit/actions/workflows/tests.yml)

A Python package for processing pupil data, with a focus on the post-illumination pupil response (PIPR).

## Overview

piprkit is made to process pupil data after collecting it with a pupillometer. It allows
processing series of measurements and individual measurements. Using very simple commands, it
offers features such as filtering, blink deletion, data smoothing, baseline correction, and
more. Common metrics such as the 6s-PIPR or AUC are easily extracted. Models of the pupil
response can be fitted to the data to improve data quality and for advanced analysis.

Everything is strongly grounded in literature; references are provided where relevant.

## Installation

    pip install piprkit

## Documentation


The documentation, with example notebooks, the API reference and the planned features, is at https://sbelgers.github.io/piprkit/.

## Examples

The example notebooks are in [docs/source/examples/example_ipynb](https://github.com/SBelgers/piprkit/tree/main/docs/source/examples/example_ipynb), and with their output in the [documentation](https://sbelgers.github.io/piprkit/).

## License

MIT, see [LICENSE](https://github.com/SBelgers/piprkit/blob/main/LICENSE).

## Author

S. Belgers
