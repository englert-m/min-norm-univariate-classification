"""Collate NumPy two-layer-ReLU experiments for the paper plots, from W&B.
"""
import argparse
import pickle
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

from model import kink_sparsity_statistics

CLUSTERINGS = {
    'small':  (1.0, 1.0),
    'medium': (2.0, 1.0),
    'large':  (4.0, 2.0),
}

# ---------- W&B retrieval ----------

def dataset_key(xs, ys, regularize_biases, decay):
    """Identify one convex problem, independently of optimizer and skip switch."""
    h = hashlib.sha256()
    for arr in (xs, ys):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64).reshape(-1))
        h.update(repr(a.shape).encode()); h.update(a.tobytes())
    return f'{bool(regularize_biases)}|{float(decay):.17g}|{h.hexdigest()}'

def load_run(run):
    cfg, smry = run.config, run.summary
    row = {
        'run_id': run.id,
        '_runtime': float(smry.get('_runtime', 0.0)),
        'seed': int(cfg['seed']),
        'optimizer': cfg['optimizer'],
        'use_skip': int(cfg['use_skip'] == 'yes'),
        'width': int(cfg['width']),
        'width_factor': float(cfg['width_factor']),
        'num_class_breaks': int(cfg['num_class_breaks']),
        'actual_weight_decay': float(cfg['actual_weight_decay']),
        'sparsity_diameter_ratio': float(cfg['sparsity_diameter_ratio']),
        'cluster_slope_change_threshold': float(cfg['cluster_slope_change_threshold']),
        'final_iteration': int(smry['iter']),
        'final_margin': float(smry['margin']),
        'final_regularized_objective': float(smry['reg_objective']),
        'final_gradient_norm': float(smry['regularized_gradient_norm']),
    }
    for key, value in smry.items():
        if key.startswith('baseline_'):
            row[key] = json.dumps(value) if isinstance(value, list) else value

    W1 = np.asarray(smry['final_W1'], dtype=float)
    b1 = np.asarray(smry['final_b1'], dtype=float)
    W2 = np.asarray(smry['final_W2'], dtype=float)
    xs = np.asarray(smry['train_x'], dtype=float)
    ys = np.asarray(smry['train_y'], dtype=float)

    d_min = float(np.min(np.diff(xs)[ys[:-1] != ys[1:]]))
    switches = int(np.count_nonzero(ys[:-1] != ys[1:]))
    row['theoretical_optimal_kink_count'] = switches - 1 if switches > 1 else np.nan

    for name, (diam_mult, thr_mult) in CLUSTERINGS.items():
        kink_count = kink_sparsity_statistics(
            W1.ravel(),
            b1.ravel(),
            W2.ravel(),
            d_min,
            diam_mult * row["sparsity_diameter_ratio"],
            thr_mult * row["cluster_slope_change_threshold"],
        )["kink_cluster_count"]
        row[f'relative_excess_kink_cluster_count_{name}'] = (
            kink_count / row['theoretical_optimal_kink_count'] - 1)


    row['dataset_key'] = dataset_key(
        xs, ys, cfg['optimizer'] == 'adam_weights_biases', row['actual_weight_decay'])
    return row

def open_cache(path='.wandb_cache.sqlite'):
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, data BLOB)')
    return con

def fetch_master(project):
    rows, skipped, total_runtime = [], 0, 0.0
    runs = list(wandb.Api().runs(project, filters={'state': 'finished'}, per_page=10000))
    con = open_cache()
    for run in tqdm(runs, desc='load runs', unit='run'):
        cached = con.execute(
            'SELECT data FROM runs WHERE run_id = ?', (run.id,)).fetchone()
        if cached is not None:
            row = pickle.loads(cached[0])
            rows.append(row)
            total_runtime += float(row.get('_runtime', np.inf))
            continue

        if run.summary.get('iter', 0) != 20000000:
            skipped += 1
            print(f'Skipping run {run.id}: {run.state} -- iter={run.summary.get("iter", 0)}')
            continue
        try:
            row = load_run(run)
            runtime = float(row.get('_runtime', np.inf))
            con.execute('INSERT INTO runs (run_id, data) VALUES (?, ?)',
                        (run.id, pickle.dumps(row)))
            con.commit()
            rows.append(row)
            total_runtime += runtime
        except Exception as exc:
            skipped += 1
            print(f'Skipping run {run.id}: {exc}')
    con.close()
    print(f'Loaded {len(rows)} run(s); skipped {skipped}.')
    print(f'Total runtime of loaded jobs: {total_runtime:.1f} seconds ({total_runtime / 3600:.2f} hours).')
    return pd.DataFrame(rows)

# ---------- generic aggregation ----------
def group_stats(df, groups, value):
    out = (df.groupby(groups, dropna=False)[value]
             .agg(count='count', mean='mean', minimum='min', maximum='max')
             .reset_index())
    return out.rename(columns={'mean': f'mean_{value}', 'minimum': f'min_{value}',
                               'maximum': f'max_{value}'})

def broadcast_best_baseline(master):
    """Runs sharing a dataset_key solve the same convex problem: use the
    best usable baseline found across those runs for every one of them."""
    key_counts = master['dataset_key'].value_counts()
    bad_keys = key_counts[key_counts != 2]
    if len(bad_keys):
        print(f'WARNING: {len(bad_keys)} dataset_key value(s) not shared by exactly 2 runs:')
        print(bad_keys.to_string())

    baseline_cols = [c for c in master.columns if c.startswith('baseline_')]
    objective = pd.to_numeric(master['baseline_regularized_objective'], errors='coerce')
    usable = master['baseline_solver_result_usable'].eq(True) & np.isfinite(objective)
    for _, group in master.groupby('dataset_key'):
        usable_idx = group.index[usable.loc[group.index]]
        if len(usable_idx) == 0:
            continue
        best_idx = objective.loc[usable_idx].idxmin()
        master.loc[group.index, baseline_cols] = master.loc[best_idx, baseline_cols].values

def unsuccessful_table(master):
    w = master[['optimizer', 'width_factor', 'use_skip', 'final_margin']].copy()
    w['successful'] = np.isfinite(w.final_margin) & (w.final_margin > 0)
    g = (w.groupby(['optimizer', 'width_factor', 'use_skip']).successful
           .agg(successful_runs='sum', total_runs='size').reset_index())
    g['unsuccessful_runs'] = g.total_runs - g.successful_runs
    g['proportion_unsuccessful'] = g.unsuccessful_runs / g.total_runs
    return g

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, na_rep='nan')
    print(f'Written: {path} ({len(df)} rows)')

def print_grid_balance(master):
    """Warn if the experiment grid is not evenly covered by loaded runs."""
    print(f'Grid balance over {len(master)} loaded run(s):')
    for column in ('num_class_breaks', 'width_factor', 'seed', 'use_skip', 'optimizer'):
        counts = master[column].value_counts().sort_index()
        print(f'  {column}: ' + ', '.join(f'{val}={cnt}' for val, cnt in counts.items()))
        if counts.nunique() > 1:
            print(f'WARNING: unbalanced {column} counts: {counts.to_dict()}')

    iter_counts = master['final_iteration'].value_counts().sort_index()
    if len(iter_counts) > 1:
        print(f'WARNING: runs did not all do the same number of iterations: {iter_counts.to_dict()}')

    dup_cols = ['use_skip', 'width_factor', 'num_class_breaks', 'optimizer', 'seed']
    dup_mask = master.duplicated(subset=dup_cols, keep=False)
    if dup_mask.any():
        dupes = master.loc[dup_mask, dup_cols + ['run_id']].sort_values(dup_cols)
        print(f'WARNING: duplicate runs with identical {dup_cols}:\n{dupes.to_string(index=False)}')

# ---------- ranges, config, and panels ----------
def symlog_forward(x, T):
    x = np.asarray(x, float)
    a = np.abs(x)
    return np.sign(x) * (np.minimum(a / T, 1) + np.log(np.maximum(a / T, 1)))
def symlog_inverse(u, T):
    u = np.asarray(u, float)
    a = np.abs(u)
    return np.sign(u) * T * (np.minimum(a, 1) + np.maximum(np.exp(a - 1) - 1, 0))
def symlog_bounds(values, T, zero=True):
    x = pd.to_numeric(values, errors='coerce').to_numpy(float)
    x = x[np.isfinite(x)]
    if zero: x = np.r_[x, 0.]
    if not len(x): return None
    u = symlog_forward(x, T)
    span = float(u.max() - u.min()) or 1.
    pad = .05 * span
    return tuple(map(float, symlog_inverse([u.min() - pad, u.max() + pad], T)))
def linear_bounds(values, zero=False, lo=None, hi=None):
    x = pd.to_numeric(values, errors='coerce').to_numpy(float)
    x = x[np.isfinite(x)]
    if zero: x = np.r_[x, 0.]
    if not len(x): return None
    a, b = float(x.min()), float(x.max())
    span = b - a or max(abs(a), 1.)
    a -= .05 * span; b += .05 * span
    if lo is not None: a = max(a, lo)
    if hi is not None: b = min(b, hi)
    return a, b

def write_config(out, seeds, loss_T, sp_T):
    text = ('% Generated by collate_results_wandb.py.\n'
            f'\\newcommand{{\\ExperimentSeedCount}}{{{int(seeds)}}}\n'
            f'\\newcommand{{\\LossSymlogThreshold}}{{{loss_T:.17g}}}\n'
            f'\\newcommand{{\\SparsitySymlogThreshold}}{{{sp_T:.17g}}}\n')
    (out / 'plot_config.tex').write_text(text, encoding='ascii')

def write_ranges(out, runs, unsuccessful, loss_T, sp_T):
    specs = {'shared bounds/loss/all': symlog_bounds(runs.relative_excess_regularized_objective, loss_T),
             'shared bounds/sp/small': symlog_bounds(runs.relative_excess_kink_cluster_count_small, sp_T),
             'shared bounds/sp/medium': symlog_bounds(runs.relative_excess_kink_cluster_count_medium, sp_T),
             'shared bounds/sp/large': symlog_bounds(runs.relative_excess_kink_cluster_count_large, sp_T),
             'shared bounds/unsuccessful/all': linear_bounds(unsuccessful.proportion_unsuccessful, True, 0, 1)}
    lines = ['% Generated by collate_results_wandb.py.', '\\pgfplotsset{']
    for key, b in specs.items():
        if b is not None:
            lines.append(f'  {key}/.style={{ymin={b[0]:.17g},ymax={b[1]:.17g}}},')
    lines += ['}', '']
    (out / 'plot_ranges.tex').write_text('\n'.join(lines), encoding='ascii')

def label(f): return f'{f:g}'.replace('.', '_')

def side_by_side(df, index_col, split_col, labels, value_cols):
    """Lay two category values of split_col side by side, indexed by index_col."""
    wide = df.pivot(index=index_col, columns=split_col, values=value_cols)
    wide.columns = [f'{v}_{labels[c]}' for v, c in wide.columns]
    return wide.reset_index().sort_values(index_col)

def write_panels(out, loss_summary, sp_summaries, unsuccessful):
    pdir = out / 'panels'
    pdir.mkdir(parents=True, exist_ok=True)
    for old in pdir.glob('*.csv'): old.unlink()
    value_cols = lambda v: ['count', f'mean_{v}', f'min_{v}', f'max_{v}']

    for (opt, wf), sub in loss_summary.groupby(['optimizer', 'width_factor']):
        panel = side_by_side(sub, 'num_class_breaks', 'use_skip', {0: 'noskip', 1: 'skip'},
                             value_cols('relative_excess_regularized_objective'))
        save_csv(panel, pdir / f'loss_{opt}_F{label(float(wf))}.csv')

    for clustering, summary in sp_summaries.items():
        value = f'relative_excess_kink_cluster_count_{clustering}'
        for (sk, wf), sub in summary.groupby(['use_skip', 'width_factor']):
            panel = side_by_side(sub, 'num_class_breaks', 'optimizer',
                                 {'adam_weights': 'aw', 'adam_weights_biases': 'awb'}, value_cols(value))
            save_csv(panel, pdir / f'sp_{clustering}_skip{int(sk)}_F{label(float(wf))}.csv')

    for opt, sub in unsuccessful.groupby('optimizer'):
        panel = side_by_side(sub, 'width_factor', 'use_skip', {0: 'noskip', 1: 'skip'},
                             ['unsuccessful_runs', 'total_runs', 'proportion_unsuccessful'])
        save_csv(panel, pdir / f'unsuccessful_{opt}.csv')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='univariate-classification')
    args = ap.parse_args()

    out = Path('plot_wandb_data')
    out.mkdir(parents=True, exist_ok=True)

    master = fetch_master(args.project)
    broadcast_best_baseline(master)

    baseline_obj = pd.to_numeric(master['baseline_regularized_objective'], errors='coerce')
    usable = master['baseline_solver_result_usable'].eq(True) & np.isfinite(baseline_obj)
    master['relative_excess_regularized_objective'] = np.nan
    master.loc[usable, 'relative_excess_regularized_objective'] = (
        master.loc[usable, 'final_regularized_objective'] / baseline_obj[usable] - 1)
    save_csv(master, out / 'master.csv')
    print_grid_balance(master)

    loss_runs = master[np.isfinite(master.final_margin) & (master.final_margin > 0)]
    loss_summary = group_stats(loss_runs, ['optimizer', 'width_factor', 'num_class_breaks', 'use_skip'],
                               'relative_excess_regularized_objective')
    save_csv(loss_summary, out / 'summary_regularized_loss.csv')

    sp_summaries = {}
    for name in CLUSTERINGS:
        summary = group_stats(loss_runs, ['use_skip', 'width_factor', 'num_class_breaks', 'optimizer'],
                              f'relative_excess_kink_cluster_count_{name}')
        sp_summaries[name] = summary
        save_csv(summary, out / f'summary_sparsity_{name}.csv')

    unsuccess = unsuccessful_table(master)
    save_csv(unsuccess, out / 'summary_unsuccessful.csv')

    loss_symlog_linthresh, sparsity_symlog_linthresh = .001, .1
    write_config(out, master.seed.nunique(), loss_symlog_linthresh, sparsity_symlog_linthresh)
    write_ranges(out, loss_runs, unsuccess, loss_symlog_linthresh, sparsity_symlog_linthresh)
    write_panels(out, loss_summary, sp_summaries, unsuccess)
    print('Done.')

if __name__ == '__main__':
    main()
