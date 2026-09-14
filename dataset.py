import numpy as np

TRAIN_BOUND = 1.0

def generate_random_breaks(rng, r, max_attempts=10000):
    """Sample r class breaks uniformly in (-1, 1) by rejection sampling.

    Constraints enforced (min_gap = 1/r²):
      - distance from each break to -1 and +1 >= min_gap
      - distance between any two consecutive sorted breaks >= min_gap
      - breaks[1] <= -min_gap  (second break is at least min_gap to the left of the origin)
      - breaks[r-2] >= min_gap  (second-last break is at least min_gap to the right of the origin)
    """
    if r < 4:
        raise ValueError(f'generate_random_breaks: r must be at least 4, got {r}.')

    min_gap = 1.0 / (r * r)
    for _ in range(max_attempts):
        breaks = np.sort(rng.uniform(-TRAIN_BOUND, TRAIN_BOUND, size=r))
        # Distance to boundaries
        if (breaks[0] + TRAIN_BOUND) < min_gap:
            continue
        if (TRAIN_BOUND - breaks[-1]) < min_gap:
            continue
        # Pairwise distances (consecutive suffices after sorting)
        if np.any(np.diff(breaks) < min_gap):
            continue
        # Straddle constraints
        if breaks[1] > -min_gap:
            continue
        if breaks[r - 2] < min_gap:
            continue
        return breaks
    raise RuntimeError(
        f'generate_random_breaks: could not satisfy all constraints after {max_attempts} '
        f'attempts (r={r}, min_gap=1/r²={min_gap:.6f}). '
    )

def check_opposite_class_separation(
        xs,
        ys,
        nominal_num_class_breaks,
        separation_factor=2/np.sqrt(2.0 * np.pi)):
    """Check the minimum distance between oppositely labeled inputs.

    The required distance is

        separation_factor * sqrt(2*pi)
        --------------------------------,
        n * sqrt(nominal_num_class_breaks)

    where n is the number of training inputs and the number of class
    breaks is the nominal requested value, not a count inferred from
    the sampled training labels.
    """

    if len(xs) == 0:
        return False
    if nominal_num_class_breaks <= 0:
        raise ValueError(
            'nominal_num_class_breaks must be positive when checking '
            'opposite-class separation.'
        )

    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]

    opposite_adjacent = ys_sorted[1:] != ys_sorted[:-1]

    # If only one class is represented, there is no opposite-class pair.
    # In random-dataset generation the straddle condition should reject
    # such a sample anyway, but returning False is safer and explicit.
    if not np.any(opposite_adjacent):
        return False

    required_distance = separation_factor * np.sqrt(2.0 * np.pi) / (len(xs) * np.sqrt(nominal_num_class_breaks))
    
    minimum_opposite_class_distance = np.min(
        np.diff(xs_sorted)[opposite_adjacent]
    )

    return minimum_opposite_class_distance >= required_distance


def check_straddle_condition(xs, ys):
    """Return True iff the training data satisfies the straddle condition.

    Groups the sorted training points into maximal consecutive runs of the same
    label.  Requires at least 4 such runs, then checks:
      - the last  point of the second run       is < 0
      - the first point of the second-to-last run is > 0
    """
    change_indices = np.flatnonzero(ys[1:] != ys[:-1])

    # At least four runs means at least three label changes.
    if len(change_indices) < 3:
        return False

    last_of_second = xs[change_indices[1]]
    first_of_second_last = xs[change_indices[-2] + 1]
    return bool(last_of_second < 0.0 and first_of_second_last > 0.0)


def generate_random_dataset(rng, num_class_breaks, num_train_points):

    breaks = generate_random_breaks(rng, num_class_breaks).astype(DTYPE)

    for _ in range(max_attempts := 10000):
        xs = np.sort(
                    rng.uniform(-TRAIN_BOUND, TRAIN_BOUND, size=num_train_points)
                ).astype(DTYPE)
        ys = ((-1) ** np.searchsorted(breaks, xs, side='right')).astype(DTYPE)

        if check_straddle_condition(xs, ys) and check_opposite_class_separation(
            xs,
            ys,
            nominal_num_class_breaks=num_class_breaks,
        ):
            return xs, ys

    raise RuntimeError(
        f'Failed to generate a dataset satisfying the straddle and '
        f'opposite-class-separation conditions after {max_attempts} attempts.'
    )

