import numpy as np
from scipy.optimize import minimize_scalar  # type: ignore


def _search_data(
    time_data: np.ndarray, size_data: np.ndarray, search_start: float, search_end: float
) -> tuple[np.ndarray, np.ndarray]:
    time_data = np.asarray(time_data, dtype=np.float64)
    size_data = np.asarray(size_data, dtype=np.float64)
    mask = (
        np.isfinite(time_data)
        & np.isfinite(size_data)
        & (time_data >= search_start)
        & (time_data <= search_end)
    )
    return time_data[mask], size_data[mask]


def _candidate_indices(time_data: np.ndarray, resolution: float) -> np.ndarray:
    if time_data.size < 2:
        return np.arange(time_data.size)
    grid = np.arange(time_data[0], time_data[-1] + resolution / 2, resolution)
    right = np.clip(np.searchsorted(time_data, grid), 1, time_data.size - 1)
    left = right - 1
    nearest = np.where(grid - time_data[left] <= time_data[right] - grid, left, right)
    return np.unique(np.concatenate(([0], nearest, [time_data.size - 1])))


def _sum_from(t: np.ndarray, rate: float, weights: np.ndarray) -> np.ndarray:
    # For each sample i, the sum of weights[j] * exp(rate * (t[j] - t[i])) over the samples j >= i. It is added up
    # as logarithms, so no exponential overflows on long recordings; a weight of 0 adds nothing
    with np.errstate(divide="ignore"):
        log_weights = np.log(weights)
    log_sum = np.logaddexp.accumulate((rate * t + log_weights)[::-1])[::-1]
    return np.exp(log_sum - rate * t)


def find_constriction_start_and_end(
    time_data: np.ndarray,
    size_data: np.ndarray,
    start: float,
    end: float,
    earliest_constriction_start: float,
    min_duration: float = 0.1,
    resolution: float = 0.01,
    min_samples: int = 3,
) -> tuple[float, float] | None:

    time, size = _search_data(time_data, size_data, start, end)
    if time.size < 3 * min_samples:
        return None
    # Centred on the first time and the mean size for numerical stability
    t = time - time[0]
    y = size - np.mean(size)

    def cumulative(values: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(values)))

    count = np.arange(time.size + 1, dtype=np.float64)
    sum_t, sum_tt = cumulative(t), cumulative(t * t)
    sum_y, sum_ty = cumulative(y), cumulative(t * y)

    # Every pair of candidate samples (first, last) as (constriction start, constriction end)
    candidates = _candidate_indices(time, resolution)
    first, last = np.meshgrid(candidates[time[candidates] >= earliest_constriction_start], candidates, indexing="ij")
    first, last = first.ravel(), last.ravel()
    valid = (
        (time[last] - time[first] >= min_duration)
        & (first >= min_samples)
        & (last - first + 1 >= min_samples)
        & (time.size - 1 - last >= min_samples)
    )
    first, last = first[valid], last[valid]
    if first.size == 0:
        return None
    start_t, end_t = t[first], t[last]
    width = end_t - start_t

    def between(sums: np.ndarray, lower: np.ndarray | int, upper: np.ndarray | int) -> np.ndarray:
        return sums[upper] - sums[lower]

    # Columns of the design matrix: 1, clip(t, start_t, end_t) - start_t and max(t - end_t, 0), with start_t
    # and end_t the constriction start and end. Constriction region: samples first..last; region after the
    # constriction end: samples last + 1..end
    n_b = between(count, first, last + 1)
    t_b, tt_b = between(sum_t, first, last + 1), between(sum_tt, first, last + 1)
    y_b, ty_b = between(sum_y, first, last + 1), between(sum_ty, first, last + 1)
    n_c = between(count, last + 1, time.size)
    t_c, tt_c = between(sum_t, last + 1, time.size), between(sum_tt, last + 1, time.size)
    y_c, ty_c = between(sum_y, last + 1, time.size), between(sum_ty, last + 1, time.size)

    h1 = (t_b - start_t * n_b) + n_c * width
    h2 = t_c - end_t * n_c
    gram = np.empty((first.size, 3, 3))
    gram[:, 0, 0] = time.size
    gram[:, 0, 1] = gram[:, 1, 0] = h1
    gram[:, 0, 2] = gram[:, 2, 0] = h2
    gram[:, 1, 1] = (tt_b - 2 * start_t * t_b + start_t**2 * n_b) + n_c * width**2
    gram[:, 1, 2] = gram[:, 2, 1] = width * h2
    gram[:, 2, 2] = tt_c - 2 * end_t * t_c + end_t**2 * n_c
    moment = np.stack(
        [np.full(first.size, sum_y[-1]), (ty_b - start_t * y_b) + width * y_c, ty_c - end_t * y_c], axis=1
    )
    coefficients = np.linalg.solve(gram, moment[..., None])[..., 0]
    sse = float(np.sum(y * y)) - np.sum(coefficients * moment, axis=1)

    # Ties (for example a constriction line that simply continues) go to the longest constriction
    ties = np.flatnonzero(sse <= np.min(sse) + 1e-12 * float(np.sum(y * y)))
    best = int(ties[np.argmax(width[ties])])
    # The segment between the constriction start and end must fall
    if coefficients[best, 1] >= 0:
        return None
    return float(time[first[best]]), float(time[last[best]])


def find_redilation_start_and_end(
    time_data: np.ndarray,
    size_data: np.ndarray,
    start: float,
    earliest_redilation_start: float,
    end: float = np.inf,
    k_min: float = -10.0,
    k_max: float = -0.01,
    k_count: int = 40,
    resolution: float = 0.01,
    min_samples: int = 3,
) -> tuple[float, float, tuple[float, float, float]] | None:

    if not k_min < k_max < 0 or k_count < 1:
        raise ValueError(
            f"Need k_min < k_max < 0 and k_count >= 1, got k_min={k_min}, k_max={k_max}, k_count={k_count}."
        )
    # Log-spaced from k_max (closest to 0, tried first) to k_min
    k_grid = -np.logspace(np.log10(-k_max), np.log10(-k_min), k_count)
    time, size = _search_data(time_data, size_data, start, end)
    if time.size < 2 * min_samples:
        return None
    # Centred on the first time and the mean size for numerical stability
    t = time - time[0]
    mean = float(np.mean(size))
    y = size - mean

    def suffix(values: np.ndarray) -> np.ndarray:
        # Sum over samples index..end
        return np.concatenate((np.cumsum(values[::-1])[::-1], [0.0]))

    candidates = _candidate_indices(time, resolution)
    index = candidates[
        (time[candidates] >= earliest_redilation_start)
        & (candidates >= min_samples)
        & (time.size - candidates >= min_samples)
    ]
    if index.size == 0:
        return None

    # A flat line followed by an exponential. Columns of the design matrix: 1 and exp(k * (t - start_t)) - 1
    # from start_t on, with start_t the redilation start. Region from it: samples index..end
    n_after = time.size - index.astype(np.float64)
    y_after = suffix(y)[index]
    # _sum_from needs weights >= 0, so y is shifted by its minimum, and the shift is added back
    low = float(np.min(y))
    ones = np.ones_like(t)

    def fit(k: float) -> tuple[np.ndarray, np.ndarray]:
        exp_after = _sum_from(t, k, ones)[index]
        exp2_after = _sum_from(t, 2 * k, ones)[index]
        expy_after = _sum_from(t, k, y - low)[index] + low * exp_after
        gram = np.empty((index.size, 2, 2))
        gram[:, 0, 0] = time.size
        gram[:, 0, 1] = gram[:, 1, 0] = exp_after - n_after
        gram[:, 1, 1] = exp2_after - 2 * exp_after + n_after
        moment = np.stack([np.full(index.size, float(np.sum(y))), expy_after - y_after], axis=1)
        coefficients = np.linalg.solve(gram, moment[..., None])[..., 0]
        sse = float(np.sum(y * y)) - np.sum(coefficients * moment, axis=1)
        return sse, coefficients

    def best_sse(k: float) -> float:
        return float(np.min(fit(k)[0]))

    grid_sse = [best_sse(k) for k in k_grid]
    j = int(np.argmin(grid_sse))
    k = float(k_grid[j])
    # k mostly lies between grid values: refine it between the neighbours of the best one. The best start is found
    # again for each k
    if k_grid.size > 1:
        refined = minimize_scalar(
            best_sse, bounds=(k_grid[min(j + 1, k_grid.size - 1)], k_grid[max(j - 1, 0)]), method="bounded"
        )
        if refined.fun < grid_sse[j]:
            k = float(refined.x)
    sse, coefficients = fit(k)
    i = int(np.argmin(sse))
    a, S = coefficients[i]
    # The curve is continuous at the redilation start, so P = flat level - S
    P = mean + a - S
    return float(time[index[i]]), float(time[-1]), (float(S), k, float(P))
