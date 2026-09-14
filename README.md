# Minimal-Norm Univariate Two-Layer ReLU Classification: Exact Solutions and Global Optimality with Skip Connections

To generate the landscape plots, run the following command:

```bash
uv run generate_landscape_plots.py
```
Then generate the PDF containing the plots with:

```bash
pdflatex landscape.tex
```

The other experiments log network training results to Weights & Biases. For a single run, use a command of the following kind:

```bash
uv run training.py --num-class-breaks 4 --width-factor 1 --optimizer adam_weights_biases --use-skip yes
```

Optimizer needs to be either `adam_weights_biases` or `adam_weights` and `use-skip` needs to be either `yes` or `no`. The results will be logged to Weights & Biases under the project name `univariate-classification`. To select a different project name, use the `--wandb-project` argument. To specify an explict seed for the random number generator, use the `--seed` argument.

After collecting results of different runs in a Weighs & Biases project, collate the results with:

```bash
uv run collate_results_wandb.py --wandb-project univariate-classification
```

Finally generate the PDF containing the plots with:

```bash
pdflatex loss_and_kink_plots.tex
```
