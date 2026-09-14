import numpy as np

from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.special import softplus

def transition_maps(xs0, ys0):
    order = np.argsort(xs0, kind="stable")
    xs = xs0[order]
    ys = ys0[order]
    if xs.size != ys.size or xs.size < 2 or np.any(np.diff(xs) <= 0):
        raise ValueError("training inputs must be distinct and datasets compatible")
    changes = np.flatnonzero(ys[:-1] != ys[1:])
    K = len(changes)
    if K < 2:
        raise ValueError("at least two represented label switches required")
    S = np.zeros((K, 2 * K))
    C = np.zeros_like(S)
    for k, i in enumerate(changes):
        a, b, label = xs[i], xs[i + 1], ys[i]
        gap = b - a
        S[k, 2 * k : 2 * k + 2] = -label / gap
        C[k, 2 * k] = label * b / gap
        C[k, 2 * k + 1] = label * a / gap
    return xs, ys, changes, S, C

def applicable(i, changes):
    if i <= changes[0]:
        return [0]
    if i >= changes[-1] + 1:
        return [len(changes) - 1]
    k = int(np.searchsorted(changes, i, side="left") - 1)
    return [k, k + 1]

def linear_constraints(xs, ys, changes, S, C):
    Aamp, n = S.shape[1], len(xs)
    rows = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        for k in applicable(i, changes):
            row = np.zeros(Aamp + n)
            row[:Aamp] = -y * (x * S[k] + C[k])
            row[Aamp + i] = 1
            rows.append(row)
    for k in range(len(changes) - 1):
        i, j = int(changes[k]), int(changes[k + 1])
        a, b, an, bn = xs[i], xs[i + 1], xs[j], xs[j + 1]
        mid = an - b
        dl = b - a
        dr = bn - an
        r1 = np.zeros(Aamp + n)
        r1[2 * k + 1] = 1
        r1[2 * (k + 1)] = -(1 + mid / dr)
        r1[2 * (k + 1) + 1] = -mid / dr
        r2 = np.zeros(Aamp + n)
        r2[2 * (k + 1)] = 1
        r2[2 * k] = -mid / dl
        r2[2 * k + 1] = -(1 + mid / dl)
        rows.extend([r1, r2])
    A = np.vstack(rows)
    return A, np.zeros(len(rows))

def realised_margins(a, xs, ys, changes, S, C):
    slopes, intercepts = S @ a, C @ a
    out = np.empty(len(xs))
    for i, (x, y) in enumerate(zip(xs, ys)):
        out[i] = min(y * (slopes[k] * x + intercepts[k]) for k in applicable(i, changes))
    return out

def regularizer_and_jac(a, S, C, changes, ys, regularize_biases):
    """Functional regularizer and analytic gradient.

    At kink k, the two transition slopes have opposite signs.  The sign of
    slope[k+1]-slope[k] is minus the label on the intermediate segment.
    Thus |Delta slope| = -label_k * Delta slope exactly.
    With bias regularization, sqrt(Delta slope^2+Delta intercept^2) is used;
    its analytic gradient is supplied as well.
    """
    D = np.zeros((len(changes) - 1, len(changes)))
    for k in range(len(changes) - 1):
        D[k, k] = -1
        D[k, k + 1] = 1
    DS, DC = D @ S, D @ C
    ds, dc = DS @ a, DC @ a
    signs = -np.asarray([ys[changes[k] + 1] for k in range(len(changes) - 1)], float)
    # Here signs[k] is sign(slope[k+1]-slope[k]). Verify the analytic sign premise, allowing zero at the boundary.
    if np.any(signs * ds < -1e-9):
        raise FloatingPointError("slope-difference sign invariant violated")
    if not regularize_biases:
        return float(signs @ ds), signs @ DS
    norms = np.hypot(ds, dc)
    if np.any(norms <= 1e-15):
        # This is only a boundary non-differentiability; positive initialisation
        # and the strictly convex optimum should keep us away from it.
        raise FloatingPointError("zero kink contribution has no unique Jacobian")
    return float(np.sum(norms)), np.sum((ds / norms)[:, None] * DS + (dc / norms)[:, None] * DC, axis=0)


def optimize_baseline(xs0, ys0, optimizer, decay, initial_a, positivity_floor=1e-10, ftol=1e-12, maxiter=10000):
    template = dict(
        regularized_objective=np.nan,
        raw_weight=np.nan,
        mean_logistic_loss=np.nan,
        minimum_margin=np.nan,
        t_values=[],
        p_values=[],
        kink_locations=[],
        solver_success=False,
        solver_result_usable=False,
        solver_status="",
        solver_message="",
        solver_iterations=np.nan,
        maximum_constraint_violation=np.nan,
        maximum_kink_interval_violation=np.nan,
        initial_objective=np.nan,
        objective_scale=np.nan,
        gradient_infinity_norm=np.nan,
    )

    
    if not np.isfinite(decay) or decay <= 0:
        template.update(solver_status="invalid_problem", solver_message="Adam optimizer and positive finite decay required")
        return template

    xs, ys, changes, S, C = transition_maps(xs0, ys0)
    K = len(changes)
    Aamp = 2 * K
    n = len(xs)
    N = Aamp + n
    a0 = np.asarray(np.maximum(initial_a, positivity_floor), dtype=float).reshape(-1)
    if a0.size != Aamp or not np.all(np.isfinite(a0)):
        template.update(solver_status="invalid_initialisation", solver_message=f"expected {Aamp} finite t_k/p_k values")
        return template
    a0 = np.maximum(a0, positivity_floor)
    m0 = realised_margins(a0, xs, ys, changes, S, C)
    x0 = np.r_[a0, m0]

    A, ub = linear_constraints(xs, ys, changes, S, C)
    constraint = LinearConstraint(A, np.full(len(ub), -np.inf), ub)
    bounds = Bounds(np.r_[np.full(Aamp, positivity_floor), np.full(n, -np.inf)], np.full(N, np.inf))
    reg_biases = optimizer == "adam_weights_biases"

    def unscaled_fun(v):
        reg, _ = regularizer_and_jac(v[:Aamp], S, C, changes, ys, reg_biases)
        return float(np.mean(softplus(-v[Aamp:])) + decay * reg)

    def unscaled_jac(v):
        _, greg = regularizer_and_jac(v[:Aamp], S, C, changes, ys, reg_biases)
        m = v[Aamp:]
        # d log(1+exp(-m))/dm = -1/(1+exp(m)), evaluated stably.
        gmargin = -np.exp(-np.logaddexp(0.0, m)) / n
        return np.r_[decay * greg, gmargin]

    objective_scale = unscaled_fun(x0)
    if not np.isfinite(objective_scale) or objective_scale <= 0:
        template.update(
            solver_status="invalid_initial_objective",
            solver_message="initial objective is not finite and positive",
            initial_objective=objective_scale,
            objective_scale=objective_scale,
        )
        return template

    # SLSQP sees an objective equal to one at x0.  Its ftol is therefore
    # comparable across datasets and regularization strengths.
    def fun(v):
        return unscaled_fun(v) / objective_scale

    def jac(v):
        return unscaled_jac(v) / objective_scale

    res = minimize(
        fun,
        x0,
        jac=jac,
        method="SLSQP",
        bounds=bounds,
        constraints=[constraint],
        options={"ftol": ftol, "maxiter": maxiter, "disp": False},
    )
    v = np.asarray(res.x, float)
    a = v[:Aamp]
    rm = realised_margins(a, xs, ys, changes, S, C)
    reg, _ = regularizer_and_jac(a, S, C, changes, ys, reg_biases)
    loss = float(np.mean(softplus(-rm)))
    # Recompute in original units.  Equivalently this is res.fun*objective_scale,
    # but recomputation avoids depending on the optimizer's stored precision.
    objective = loss + decay * reg
    slopes, ints = S @ a, C @ a
    ds = np.diff(slopes)
    di = np.diff(ints)
    kinks = np.full(K - 1, np.nan)
    nz = np.abs(ds) > 1e-14
    kinks[nz] = -di[nz] / ds[nz]
    interval_v = []
    for k, z in enumerate(kinks):
        lo, hi = xs[changes[k] + 1], xs[changes[k + 1]]
        interval_v.append(np.inf if not np.isfinite(z) else max(lo - z, z - hi, 0.0))
    cv = float(max(0.0, np.max(A @ v - ub), np.max(bounds.lb - v)))
    kv = float(max(interval_v, default=0.0))
    feasible = np.isfinite(objective) and cv <= 1e-7 and kv <= 1e-7
    successful = bool(res.success) and feasible
    message = str(res.message)
    message_lower = message.lower()
    if successful:
        status = "optimal"
    elif "iteration limit" in message_lower or int(getattr(res, "status", -1)) == 9:
        status = "maximum_iterations_exceeded"
    elif "positive directional derivative" in message_lower or int(getattr(res, "status", -1)) == 8:
        status = "positive_directional_derivative"
    elif not feasible:
        status = "infeasible_final_iterate"
    else:
        status = "solver_failure"

    # For the two requested SLSQP failure modes, retain the finite final
    # objective even if the final feasibility diagnostics exceed tolerance.
    # The violations remain recorded in the output for auditing.
    recognised_retained_failure = status in {"maximum_iterations_exceeded", "positive_directional_derivative"}
    usable = bool(np.isfinite(objective) and (feasible or recognised_retained_failure))
    return dict(
        regularized_objective=float(objective) if usable else np.nan,
        raw_weight=float(reg) if usable else np.nan,
        mean_logistic_loss=loss if usable else np.nan,
        minimum_margin=float(np.min(rm)) if usable else np.nan,
        t_values=a[0::2].tolist() if usable else [],
        p_values=a[1::2].tolist() if usable else [],
        kink_locations=kinks.tolist() if usable else [],
        solver_success=successful,
        solver_result_usable=usable,
        solver_status=status,
        solver_message=message,
        solver_iterations=int(getattr(res, "nit", -1)),
        maximum_constraint_violation=cv,
        maximum_kink_interval_violation=kv,
        initial_objective=float(objective_scale),
        objective_scale=float(objective_scale),
        gradient_infinity_norm=float(np.max(np.abs(unscaled_jac(v)))),
    )
