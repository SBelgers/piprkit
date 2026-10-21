class BaselineCorrection:
    """Record of a baseline correction applied to a pupil measurement.

    The measurement stores it when ``filters.apply_baseline_correction()`` is applied; read it with
    ``measurement.get_baseline_correction()``. Metric results and fitted models carry it along,
    so their values can be read in the right units.
    """

    def __init__(self, method: str, baseline: float, duration: float) -> None:
        """Create the record of a baseline correction.

        Args:
            method (str): The correction method: "ratio", "subtractive" or "percent".
            baseline (float): The baseline pupil size used for the correction, in the units of the
                data before the correction.
            duration (float): The length of the baseline window before light onset, in seconds.
        """
        self._method = method
        self._baseline = baseline
        self._duration = duration

    def get_method(self) -> str:
        """Return the correction method.

        Returns:
            str: "ratio", "subtractive" or "percent".
        """
        return self._method

    def get_baseline(self) -> float:
        """Return the baseline pupil size used for the correction.

        Returns:
            float: The baseline, in the units of the data before the correction.
        """
        return self._baseline

    def get_duration(self) -> float:
        """Return the length of the baseline window before light onset.

        Returns:
            float: The length of the window, in seconds.
        """
        return self._duration

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaselineCorrection):
            return NotImplemented
        return (self._method, self._baseline, self._duration) == (
            other._method,
            other._baseline,
            other._duration,
        )

    def __hash__(self) -> int:
        return hash((self._method, self._baseline, self._duration))

    def __repr__(self) -> str:
        return (
            f"BaselineCorrection(method={self._method!r}, baseline={self._baseline!r}, "
            f"duration={self._duration!r})"
        )

    def __str__(self) -> str:
        return f"{self._method} (baseline={self._baseline:.4g}, {self._duration:g} s)"
