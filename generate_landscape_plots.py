import argparse
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar


GRID_P = np.linspace(-3.0, 3.0, 101)
GRID_Q = np.linspace(-3.0, 3.0, 101)
REG_GRID_SIZE = 76


def h(x):
    return np.where(x <= -2, 0, np.where(x <= -1, x + 2, np.where(x <= 1, -x, np.where(x <= 2, x - 2, 0))))


def fixed_a_terms(x):
    AD = np.where(x <= -2, 4, np.where(x <= 2, 2 - x, 0))
    BD = np.where(x <= -1, 2, np.where(x <= 1, 1 - x, 0))
    AE = np.where(x <= -2, 0, np.where(x <= 2, x + 2, 4))
    BE = np.where(x <= -1, 0, np.where(x <= 1, x + 1, 2))
    return AD, BD, AE, BE


def fixed_a_deltas(P, Q):
    apD, bpD, apE, bpE = fixed_a_terms(P)
    aqD, bqD, aqE, bqE = fixed_a_terms(Q)
    return apD * bqD - aqD * bpD, aqE * bpE - apE * bqE


def objective_norm(P, Q, a0=False):
    if not a0:
        m = np.maximum(h(Q), -h(P))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(m > 0, 6 / m, np.nan)

    apD, bpD, apE, bpE = fixed_a_terms(P)
    aqD, bqD, aqE, bqE = fixed_a_terms(Q)
    delta_d, delta_e = fixed_a_deltas(P, Q)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus = np.where(delta_d > 0, 2 * (apD + bpD + aqD + bqD) / delta_d, np.nan)
        minus = np.where(delta_e > 0, 2 * (apE + bpE + aqE + bqE) / delta_e, np.nan)
    return np.fmin(plus, minus)


def _regularized_loss(z, features, y, penalty):
    margin = y * (features @ z)
    value = np.sum(np.logaddexp(0, -margin)) + penalty @ z
    weight = -y / (1 + np.exp(np.clip(margin, -60, 60)))
    gradient = features.T @ weight + penalty
    return value, gradient


def _solve_regularized_point(p, q, optimize_a, warm, x, y, lam, a_start, a_only):
    best = a_only if optimize_a else 4 * np.log(2)

    for sign_t in (1, -1):
        c = np.maximum(-x - sign_t * p, 0) - np.maximum(x - sign_t * p, 0)
        for sign_u in (1, -1):
            d = sign_u * (np.maximum(sign_u * (x - q), 0) - np.maximum(-sign_u * (x + q), 0))
            features = np.column_stack((x, c, d)) if optimize_a else np.column_stack((c, d))
            penalty = np.array([0., lam * sign_t, lam * sign_u]) if optimize_a else np.array([lam * sign_t, lam * sign_u])

            if optimize_a:
                bounds = [
                    (None, None),
                    (0, None) if sign_t > 0 else (None, 0),
                    (0, None) if sign_u > 0 else (None, 0),
                ]
                start = warm.get((sign_t, sign_u), np.array([a_start, 0., 0.]))
            else:
                bounds = [
                    (0, None) if sign_t > 0 else (None, 0),
                    (0, None) if sign_u > 0 else (None, 0),
                ]
                start = warm.get((sign_t, sign_u), np.zeros(2))

            result = minimize(
                _regularized_loss,
                start,
                args=(features, y, penalty),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
            )
            warm[sign_t, sign_u] = result.x
            best = min(best, result.fun)

    return best


def regularized_grid(n=REG_GRID_SIZE):
    x = np.array([-2., -1., 1., 2.])
    y = np.array([1., -1., 1., -1.])
    lam = 0.002
    a_result: Any = minimize_scalar(
        lambda a: np.sum(np.logaddexp(0, -y * a * x)), bracket=(-2, 0, 2), method="brent"
    )
    a_only = a_result.fun
    ps = np.linspace(-3, 3, n)
    qs = np.linspace(-3, 3, n)
    a0_grid = np.empty((n, n))
    opt_a_grid = np.empty((n, n))

    for optimize_a, grid in ((False, a0_grid), (True, opt_a_grid)):
        for i, q in enumerate(qs):
            warm = {}
            for j in (range(n) if i % 2 == 0 else range(n - 1, -1, -1)):
                grid[i, j] = _solve_regularized_point(
                    ps[j], q, optimize_a, warm, x, y, lam, a_result.x, a_only
                )
    return ps, qs, opt_a_grid, a0_grid


def write_grid(path, ps, qs, z, cap=None):
    with path.open("w") as f:
        f.write("p q value value_capped\n")
        for q, row in zip(qs, z):
            for p, value in zip(ps, row):
                capped = value if cap is None else min(value, cap)
                f.write(f"{p:.12g} {q:.12g} {value:.12g} {capped:.12g}\n")


def boundary_mask(P, Q, a0):
    if not a0:
        return np.maximum(h(Q), -h(P)) > 0
    delta_d, delta_e = fixed_a_deltas(P, Q)
    return (delta_d > 0) | (delta_e > 0)


def segments(P, Q, z, levels):
    fig, ax = plt.subplots()
    contour = ax.contour(P, Q, np.ma.masked_invalid(z), levels=levels)
    result = {level: [np.asarray(s) for s in group if len(s) > 1] for level, group in zip(contour.levels, contour.allsegs)}
    plt.close(fig)
    return result


def segment_length(segment):
    if len(segment) < 2:
        return 0.0
    diffs = np.diff(segment, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs * diffs, axis=1))))


def candidate_points_from_segment(segment):
    n = len(segment)
    if n == 0:
        return []
    fractions = [0.50, 0.42, 0.58, 0.35, 0.65, 0.25, 0.75]
    idxs = {max(0, min(n - 1, int(round(frac * (n - 1))))) for frac in fractions}
    return [segment[idx] for idx in idxs]


def far_enough(point, accepted_points, min_label_distance):
    x, y = point
    return all((x - x0) ** 2 + (y - y0) ** 2 >= min_label_distance ** 2 for x0, y0 in accepted_points)


def write_contours(path, contour_data, labels=True, label_levels=(), min_label_distance=0.9, min_segment_points=8):
    accepted_label_points = []
    label_commands = []

    with path.open("w") as f:
        for level, group in contour_data.items():
            for segment in group:
                f.write("\\addplot[black, line width=0.45pt, no marks] coordinates {\n")
                f.writelines(f"  ({x:.8g},{y:.8g})\n" for x, y in segment)
                f.write("};\n")

            if not labels or not group or not any(np.isclose(level, value, atol=1e-9) for value in label_levels):
                continue

            candidates = sorted(
                (s for s in group if len(s) >= min_segment_points), key=segment_length, reverse=True
            )
            chosen_point = None
            for segment in candidates:
                for point in candidate_points_from_segment(segment):
                    if far_enough(point, accepted_label_points, min_label_distance):
                        chosen_point = point
                        break
                if chosen_point is not None:
                    break

            if chosen_point is None:
                continue

            x, y = chosen_point
            accepted_label_points.append((float(x), float(y)))
            value = float(level)
            label = (
                str(int(round(value)))
                if np.isclose(value, round(value), atol=1e-9)
                else f"{value:.2f}"
            )
            label_commands.append(f"\\node[fill=white, inner sep=1.6pt] at (axis cs:{x:.8g},{y:.8g}) {{$ {label} $}};\n")

        f.writelines(label_commands)


def write_boundaries(path, P, Q, z):
    fig, ax = plt.subplots()
    contour = ax.contour(P, Q, z.astype(float), levels=[0.5])
    with path.open("w") as f:
        for group in contour.allsegs:
            for segment in group:
                f.write("\\addplot3[orange, line width=1.4pt, no marks] coordinates {\n")
                f.writelines(f"  ({x:.6g},{y:.6g},81)\n" for x, y in segment[::2])
                f.write("};\n")
    plt.close(fig)


def write_colormap(path, name="warpedviridis", warp: Callable[[float], float] = lambda t: t ** (1 / 2)):
    viridis = plt.get_cmap("viridis")
    with path.open("w") as f:
        f.write(f"\\pgfplotsset{{\n  colormap={{{name}}}{{\n")
        for t in np.linspace(0, 1, 128):
            r, g, b, _ = viridis(warp(t))
            f.write(f"    rgb255=({round(255*r)},{round(255*g)},{round(255*b)})\n")
        f.write("  }\n}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("./plot_dat_tex"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    write_colormap(args.output / "warpedviridis.tex")
    write_colormap(args.output / "warped3viridis.tex", "warped3viridis", warp=lambda t: t ** (1 / 3))
    boundary_p, boundary_q = np.meshgrid(np.linspace(-3, 3, 1201), np.linspace(-3, 3, 1201))
    for a0 in (False, True):
        write_boundaries(
            args.output / f"boundary_{'a0' if a0 else 'opt_a'}_pq.tex",
            boundary_p,
            boundary_q,
            boundary_mask(boundary_p, boundary_q, a0),
        )

    norm_p, norm_q = np.meshgrid(GRID_P, GRID_Q)
    for a0 in (False, True):
        name = "a0" if a0 else "opt_a"
        z = objective_norm(norm_p, norm_q, a0)
        write_grid(args.output / f"obj_norm_{name}_pq.dat", GRID_P, GRID_Q, z, 95)
        levels = [7, 9, 12, 16, 21, 27, 34, 42, 51, 61, 72]
        sparse_levels = [levels[0], levels[3], levels[6], levels[10]]
        write_contours(args.output / f"cont_norm_{name}_pq.tex", segments(norm_p, norm_q, z, levels), label_levels=sparse_levels)

    ps, qs, opt, a0 = regularized_grid(REG_GRID_SIZE)
    reg_p, reg_q = np.meshgrid(ps, qs)
    for name, z in (("opt_a", opt), ("a0", a0)):
        write_grid(args.output / f"obj_reg_{name}_pq.dat", ps, qs, z)
        levels = 0.03 + 2 * np.linspace(0.6, 1, 11) ** 9
        sparse_levels = [levels[0], levels[3], levels[6], levels[10]]
        write_contours(args.output / f"cont_reg_{name}_pq.tex", segments(reg_p, reg_q, z, levels), label_levels=sparse_levels)


if __name__ == "__main__":
    main()
