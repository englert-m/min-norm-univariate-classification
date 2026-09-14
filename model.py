# ============================================================
# Model
# ============================================================

import numpy as np

from scipy.cluster.hierarchy import fclusterdata


def sample_width_factor_neurons(rng, xs, width):
    """Construct neurons.

    xs has to be sorted and not have duplicates.

    If there are ``num_gaps = len(xs) - 1`` consecutive training-data
    gaps, write

        width = integer_part * num_gaps + remainder,

    where

        0 <= remainder < num_gaps.

    Make ``integer_part`` complete passes through the gaps, placing one
    neuron in every gap on every pass. Then uniformly select exactly
    ``remainder`` distinct gaps and place one additional neuron in each
    selected gap.

    Every selected neuron's kink is sampled uniformly in its gap.

    Returns ``(kink_positions, integer_part, remainder)``.
    """

    
    intervals = list(zip(xs[:-1], xs[1:]))

    if not intervals:
        raise ValueError(
            'at least two training points are needed to place kinks'
        )

    
    num_gaps = len(intervals)
    
    integer_part, remainder = divmod(width, num_gaps)

    selected_intervals = intervals * integer_part

    # Final partial pass: select exactly `remainder` distinct gaps,
    # uniformly without replacement.
    if remainder > 0:
        selected_intervals += [
            intervals[i]
            for i in rng.choice(num_gaps, size=remainder, replace=False)
        ]
    
    kink_positions = np.array([
            rng.uniform(left_endpoint, right_endpoint)
            for left_endpoint, right_endpoint in selected_intervals
        ], dtype=DTYPE)

    return kink_positions, integer_part, remainder




class TwoLayerReLU:
    """f(x) = W2 · ReLU(W1 x + b1)  [+ skip_w·x + skip_b]

    All parameters are one-dimensional NumPy arrays, except skip_w and skip_b,
    which are stored as length-one arrays so that the optimizer can treat all
    parameters uniformly.

    :param first_layer_scale:  std dev for random W1 initialisation.
    :param second_layer_scale: std dev for random W2 initialisation.
    """

    def __init__(self, width, use_skip,
                 first_layer_scale=1.0, second_layer_scale=1.0,
                 xs=None, rng=None):
        self.width              = int(width)
        self.use_skip           = bool(use_skip)
        self.first_layer_scale  = float(first_layer_scale)
        self.second_layer_scale = float(second_layer_scale)
        self.xs              = xs
        self.rng             = rng

        self.W1 = np.empty(self.width, dtype=DTYPE)
        self.b1 = np.empty(self.width, dtype=DTYPE)
        self.W2 = np.empty(self.width, dtype=DTYPE)
        self.skip_w = np.zeros(1, dtype=DTYPE)
        self.skip_b = np.zeros(1, dtype=DTYPE)
        self.reset_parameters()

    def reset_parameters(self):
        kinks, _, _ = sample_width_factor_neurons(self.rng, self.xs, self.width)
        
        self.W1[...] = self.rng.normal(0.0, self.first_layer_scale,
                                            size=self.width)
        
        self.b1[...] = -self.W1 * kinks

        self.W2[...] = self.rng.normal(0.0, self.second_layer_scale,
                                            size=self.width)

        self.skip_w[...] = 0.0
        self.skip_b[...] = 0.0

    def parameters(self):
        params = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2}
        if self.use_skip:
            params['skip_w'] = self.skip_w
            params['skip_b'] = self.skip_b
        return params

    def forward(self, x, return_cache=False):
        x = np.asarray(x, dtype=DTYPE).reshape(-1)
        z = x[:, None] * self.W1[None, :] + self.b1[None, :]
        h = np.maximum(z, 0.0)
        out = h @ self.W2
        if self.use_skip:
            out = out + self.skip_w[0] * x + self.skip_b[0]
        if return_cache:
            return out, (x, z, h)
        return out

    def network_weight_excluding_skip(self, include_biases):
        params = (self.W1, self.W2, self.b1) if include_biases else (self.W1, self.W2)
        return 0.5 * float(sum(np.sum(p ** 2) for p in params))


def kink_sparsity_statistics(W1, b1, W2, d_min, sparsity_diameter_ratio,
                             cluster_slope_change_threshold, w_eps=1e-14):
    """Compute sparsity statistics for the kinks in a neural network."""
    finite_mask   = np.abs(W1) > w_eps
    kinks         = -b1[finite_mask] / W1[finite_mask]
    slope_changes =  np.abs(W2[finite_mask] * W1[finite_mask])

    finite_kink_mask = np.isfinite(kinks)
    kinks         = kinks[finite_kink_mask]
    slope_changes = slope_changes[finite_kink_mask]

    if len(kinks) > 1:
        labels = fclusterdata(kinks[:, None], t=sparsity_diameter_ratio * d_min,
                               criterion='distance', method='complete')
    else:
        # pdist (used internally by fclusterdata) errors on fewer than 2 points
        labels = np.ones(len(kinks), dtype=int)

    sizes, representatives, cluster_slope_changes = [], [], []
    ignored = 0

    for label in np.unique(labels):
        members = labels == label
        total_change = float(slope_changes[members].sum())

        if total_change < cluster_slope_change_threshold:
            ignored += 1
            continue

        sizes.append(int(members.sum()))
        representatives.append(float(kinks[members].mean()))
        cluster_slope_changes.append(total_change)

    return {
        "finite_kink_count": len(kinks),
        "nonfinite_kink_count": len(W1) - len(kinks),
        "raw_kink_cluster_count": len(np.unique(labels)),
        "ignored_kink_cluster_count": ignored,
        "kink_cluster_count": len(sizes),
        "kink_cluster_sizes": sizes,
        "kink_cluster_representatives": representatives,
        "kink_cluster_slope_changes": cluster_slope_changes,
    }

