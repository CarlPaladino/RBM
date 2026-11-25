import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import RBM, sample_bernoulli

parser = argparse.ArgumentParser(description="Compare class-conditioned Gibbs chains at different step counts")
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--class-name", choices=[str(i) for i in range(10)], default=None, help="Omit to sample labels freely")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
folder = args.output
model = RBM.load(args.checkpoint)
with np.load(args.checkpoint, allow_pickle=False) as saved:
    config = json.loads(str(saved["config"]))
    rng = model.rng
    rng.bit_generator.state = json.loads(str(saved["rng_state"]))
folder.mkdir(parents=True, exist_ok=True)
samples = sample_bernoulli(np.full((config["number_generated_samples"], model.visible_units), 0.5, dtype=np.float32), rng)
fixed_labels = None
if args.class_name is not None:
    fixed_labels = np.zeros((len(samples), len(model.class_names)), dtype=np.float32)
    fixed_labels[:, model.class_names.index(args.class_name)] = 1
    samples[:, : model.label_units] = fixed_labels
else:
    samples[:, : model.label_units] = np.eye(model.label_units, dtype=np.float32)[rng.integers(model.label_units, size=len(samples))]
description = f"Digit {args.class_name}" if args.class_name is not None else "Unconditional"
snapshots = {}
previous = 0
for steps in (50, 100, 1000, 2000, 4000, 8000, 16000):
    samples = model.run_gibbs_chain(samples, steps - previous - 1, fixed_labels=fixed_labels)
    hidden = model.sample_hidden(samples)
    probabilities = model.visible_probabilities(hidden)[:, model.label_units :]
    # Continue the binary chain, but save and display its final conditional probabilities.
    samples = model.sample_visible(hidden)
    if fixed_labels is not None:
        samples[:, : model.label_units] = fixed_labels
    previous = steps
    snapshots[steps] = probabilities.copy()
    assert samples.shape == (config["number_generated_samples"], model.visible_units)
    assert np.isin(samples, [0, 1]).all()
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    columns = min(4, len(samples))
    rows = (len(samples) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            ax.imshow(probabilities[i].reshape(28, 28).T, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(f"{description} probabilities - {steps:,} Gibbs steps")
    fig.tight_layout()
    path = folder / f"generated_samples_{steps}_steps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path.resolve(), flush=True)
fig, axes = plt.subplots(7, len(samples), figsize=(1.25 * len(samples), 10.5), squeeze=False)
for row, (steps, batch) in enumerate(snapshots.items()):
    for col, ax in enumerate(axes[row]):
        ax.imshow(batch[col].reshape(28, 28).T, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(f"{steps:,} steps")
fig.suptitle(f"{description} probabilities: same chains at 50, 100, 1,000, 2,000, 4,000, 8,000 and 16,000 Gibbs steps")
fig.tight_layout()
fig.savefig(folder / "generation_steps_comparison_50_to_16000.png", dpi=150)
plt.close(fig)
print("All samples validated generated from the same starting chains.")
