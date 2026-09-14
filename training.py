import sys
import random

import numpy as np
import wandb
from tqdm import trange

from adam import AdamNumpy, compute_grads
from baseline import optimize_baseline
from dataset import generate_random_dataset
from model import TwoLayerReLU

from scipy.special import softplus

import builtins

builtins.DTYPE =  np.float64


def logistic_loss(logits, labels):
    """Mean binary logistic loss for labels in {-1,+1}."""
    return float(np.mean(softplus(-labels * logits)))

def run_training(model, x, y, optim, args, actual_weight_decay, d_min):
    """Train for *args.num_iters* steps, logging metrics to Weights & Biases."""
    log_iters = set(np.ceil(np.geomspace(1, args.num_iters, num=350)).astype(int))
    include_biases         = (args.optimizer == 'adam_weights_biases')

    y_flat = np.asarray(y, dtype=DTYPE).reshape(-1)

    for it in trange(1, args.num_iters + 1, desc='training', disable=not sys.stderr.isatty()):
        grads, margin = compute_grads(
            model, x, y_flat,
            actual_weight_decay=actual_weight_decay,
            include_biases=include_biases,
        )
        optim.step(grads)

        # Make the optimizer adjustment
        # once, at the first strict interpolation event (y_i f(x_i) > 0 for all i).
        if not optim.interpolation_adjusted and margin > 0.0:
            old_lr, old_beta2 = optim.lr, optim.beta2
            optim.beta2 = 1.0 - args.adam_interpolation * (1.0 - optim.beta2)
            if not (0.0 <= optim.beta2 < 1.0):
                raise ValueError(
                    '--adam-interpolation makes beta2 invalid: '
                    f'old beta2={old_beta2}, factor={args.adam_interpolation}, '
                    f'new beta2={optim.beta2}; require 0 <= beta2 < 1.'
                )
            optim.lr *= args.adam_interpolation
            optim.interpolation_adjusted = True
            print(
                f'Interpolation detected at iteration {it}: '
                f'Adam lr {old_lr:.6g} -> {optim.lr:.6g}, '
                f'beta2 {old_beta2:.6g} -> {optim.beta2:.6g}.',
                flush=True,
            )

        if it in log_iters:
            net_weight     = model.network_weight_excluding_skip(include_biases=include_biases)
            # a bit wasteful to compute the logits again, but there are very few logging iterations, so it's not a big deal.
            train_logits   = model.forward(x)
            train_loss_now = logistic_loss(train_logits, y_flat)
            reg_now_value  = float(train_loss_now + actual_weight_decay * net_weight)
            margin_now     = float(np.min(y_flat * train_logits))
            norm_weight_now = net_weight / margin_now if margin_now > 0 else np.nan

            regularized_gradient_norm = float(
                np.sqrt(
                    sum(
                        float(np.dot(g, g))
                        for _, g in compute_grads(
                            model, x, y_flat, actual_weight_decay=actual_weight_decay, include_biases=include_biases
                        )[0].items()
                    )
                )
            )

            wandb.log(
                {
                    "iter": it,
                    "net_weight": net_weight,
                    "train_loss": train_loss_now,
                    "reg_objective": reg_now_value,
                    "margin": margin_now,
                    "normalized_network_weight": norm_weight_now,
                    "regularized_gradient_norm": regularized_gradient_norm,
                }
            )

    print("training complete, computing final metrics...", flush=True)
    wandb.summary["final_W1"] = model.W1.copy().tolist()
    wandb.summary["final_b1"] = model.b1.copy().tolist()
    wandb.summary["final_W2"] = model.W2.copy().tolist()
    wandb.summary["final_skip_w"] = float(model.skip_w[0])
    wandb.summary["final_skip_b"] = float(model.skip_b[0])


    try:
        # network_transition_initial_values
        train_logits   = model.forward(x)
        
        changes = np.flatnonzero(y[:-1] != y[1:])
        margins = y * train_logits
        a0 = np.column_stack((margins[changes], margins[changes+1])).ravel()
        if not np.all(np.isfinite(a0)):
            raise ValueError("non-finite final-network t_k or p_k value")
        # Interpolating runs should already be strictly positive.  The maximum is
        # only a numerical safeguard for values at the positivity bound.
        wandb.summary["final_network_transition_initial_values"] = a0.tolist()

        opt = optimize_baseline(x, y, optimizer=args.optimizer, decay=actual_weight_decay, initial_a=a0)
    except Exception as exc:
        opt=dict(regularized_objective=np.nan, raw_weight=np.nan,
                    mean_logistic_loss=np.nan, minimum_margin=np.nan,
                    t_values=[], p_values=[], kink_locations=[],
                    solver_success=False, solver_result_usable=False,
                    solver_status='optimization_exception',
                    solver_message=str(exc), solver_iterations=np.nan,
                    maximum_constraint_violation=np.nan,
                    maximum_kink_interval_violation=np.nan,
                    initial_objective=np.nan, objective_scale=np.nan,
                    gradient_infinity_norm=np.nan)

    # for each key in opt, add an entry to wandb.summary with the key prefixed by "baseline_"
    for key, value in opt.items():
        wandb.summary[f"baseline_{key}"] = value




def main(args):
    rng = np.random.default_rng(args.seed)

    xs, ys = generate_random_dataset(
        rng, args.num_class_breaks, args.num_train_points
    )

    d_min = float(np.min(np.diff(xs)[ys[:-1] != ys[1:]]))
    s_scale = args.scale_coefficient / d_min
    actual_weight_decay = args.weight_decay_coefficient / (args.width * s_scale ** 2)

    model = TwoLayerReLU(
            width=args.width,
            use_skip=args.use_skip=='yes',
            first_layer_scale=s_scale,
            second_layer_scale=1.0 / s_scale,
            xs=xs,
            rng=rng,
    )

    

    optim = AdamNumpy(model.parameters(), lr=args.lr)

    with wandb.init(project=args.wandb_project or 'univariate-classification', config=vars(args)):
 
        # add actual weight decay to wandb config for reference
        wandb.config.update({"actual_weight_decay": actual_weight_decay}, allow_val_change=True)

        wandb.summary["train_x"] = xs.tolist()
        wandb.summary["train_y"] = ys.tolist()
   
        run_training(model, xs, ys, optim, args, actual_weight_decay, d_min=d_min)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train a two-layer ReLU network on a random univariate '
                    'classification dataset.'
    )

    parser.add_argument('--num-class-breaks', type=int, required=True)
    parser.add_argument('--width-factor', type=int, required=True)
    parser.add_argument('--use-skip', choices=['yes', 'no'], required=True)
    parser.add_argument('--optimizer', choices=['adam_weights', 'adam_weights_biases'], required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--wandb-project', type=str, default=None)

    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.SystemRandom().randint(0, 2**32 - 1)

    args.num_iters = 20000000
    args.data_factor = 0.5
    args.scale_coefficient = 5.0
    args.num_train_points = round((args.data_factor * args.num_class_breaks) ** 2)
    args.lr = 1e-4
    args.weight_decay_coefficient = 1e-2
    args.adam_interpolation = 10.0
    args.sparsity_diameter_ratio = 0.01
    args.cluster_slope_change_threshold = 0.01

    args.width = int(round(args.width_factor * args.num_class_breaks))

    main(args)
