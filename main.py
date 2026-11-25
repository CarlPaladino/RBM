import argparse
import json
import os
from pathlib import Path
from tempfile import mkdtemp

import matplotlib.pyplot as plt
import numpy as np

CONFIG = {"learning_rate": 0.02, "decay_rate": 0.9, "decay_frequency": 10, "batch_size": 100, "number_epochs": 300, "cd_steps": 10, "weight_initialization_scale": 0.01, "random_seed": 42, "generation_steps": 5000, "number_generated_samples": 16, "hidden_sizes": [512], "output_directory": "output", "class_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]}

# 784 pixel units (28x28 images), plus 10 one-hot label units.
# 1 bias per visible unit, and hidden unit respectively
# V - (B, 794): 10 one-hot class labels followed by 784 image pixels.
# H - (B, H) where B is the batch size and H is the number of hidden units
# W - (visible units amount, hidden units amount)


# x - shape (B,N) where B is the batch size, N is number of features (794 when calculating visible probabilities, 10 labels + 784 pixels. OR size of hidden layer when calculating hidden probabilities)
def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    result = np.empty_like(x)

    # take masks for nonnegative and negative values
    nonnegative_mask = np.greater_equal(x, 0.0)
    negative_mask = np.logical_not(nonnegative_mask)

    # compute sigmoid for nonnegative values
    nonnegative_values = x[nonnegative_mask]
    result[nonnegative_mask] = 1.0 / (1.0 + np.exp(-nonnegative_values))

    # compute sigmoid for negative values
    negative_values = x[negative_mask]
    exp_values = np.exp(negative_values)
    result[negative_mask] = exp_values / (1.0 + exp_values)

    return result


# x - shape (B,N) where B is the batch size, N is size of hidden layer since softplus is only done on hidden inputs
def softplus(x):
    nonnegative_term = np.maximum(0.0, x)
    logarithmic_term = np.log1p(np.exp(-np.abs(x)))

    return nonnegative_term + logarithmic_term


# probabilities - shape (B,N) where B is the batch size, N is number of features
# rng - numpy random generator
def sample_bernoulli(probabilities, rng):
    random_values = rng.uniform(size=probabilities.shape)
    activated_mask = np.less(random_values, probabilities)
    binary_states = activated_mask.astype(np.float32)

    return binary_states


# separate training data into minibatches
def iterate_minibatches(data, batch_size, rng):
    amount_samples = len(data)
    indices = rng.permutation(amount_samples)

    for start_index in range(0, amount_samples, batch_size):
        end_index = start_index + batch_size
        batch_indices = indices[start_index:end_index]

        yield data[batch_indices]


# normalize data and intialize both train and test sets
def prepare_data():
    train = np.loadtxt("data/raw/emnist-digits-train.csv", delimiter=",", dtype=np.float32)
    test = np.loadtxt("data/raw/emnist-digits-test.csv", delimiter=",", dtype=np.float32)

    # normalize pixel values to [0,1]
    train_pixels = train[:, 1:] / 255.0
    test_pixels = test[:, 1:] / 255.0

    train_pixels = (train_pixels >= 0.5).astype(np.float32)
    test_pixels = (test_pixels >= 0.5).astype(np.float32)

    train_pixels = np.concatenate((np.eye(10, dtype=np.float32)[train[:, 0].astype(int)], train_pixels), axis=1)
    test_pixels = np.concatenate((np.eye(10, dtype=np.float32)[test[:, 0].astype(int)], test_pixels), axis=1)

    return train_pixels, test_pixels


class RBM:
    def __init__(self, visible_units, hidden_units, rng, class_names, weight_scale=0.01):
        self.class_names = list(class_names)
        self.label_units = len(self.class_names)
        self.pixel_units = visible_units - self.label_units
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.rng = rng
        self.visible_bias = np.zeros(visible_units, dtype=np.float32)
        self.hidden_bias = np.zeros(hidden_units, dtype=np.float32)
        self.weights = rng.normal(scale=weight_scale, size=(visible_units, hidden_units)).astype(np.float32)
        self.gibbs_chain = None

    # conditional probabilities P(h | v)
    # V - shape (B, visible units size) where B is batch size
    def hidden_probabilities(self, V):
        return sigmoid(np.dot(V, self.weights) + self.hidden_bias)

    # conditional probabilities P(v | h)
    # H - shape (B, hidden units size) where B is batch size
    def visible_probabilities(self, H):
        visible_logits = np.dot(H, self.weights.T) + self.visible_bias
        visible_probabilities = sigmoid(visible_logits)
        label_logits = visible_logits[:, 0 : self.label_units]

        # label scores difference after subtracting max score
        shifted_label_logits = label_logits - label_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted_label_logits)
        visible_probabilities[:, 0 : self.label_units] = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return visible_probabilities

    # binary sampling
    # V - shape (B, visible units size) where B is batch size
    def sample_hidden(self, V):
        probabilities = self.hidden_probabilities(V)

        return sample_bernoulli(probabilities, self.rng)

    # sample visible units given hidden units
    # H - shape (B, hidden units size) where B is batch size
    def sample_visible(self, H):
        visible_probabilities = self.visible_probabilities(H)
        visible_states = sample_bernoulli(visible_probabilities, self.rng)
        label_probabilities = visible_probabilities[:, 0 : self.label_units]
        cumulative_probabilities = np.cumsum(label_probabilities, axis=1)

        # get the selected class based on cumulative probabilities
        amount_samples = H.shape[0]
        random_values = self.rng.random((amount_samples, 1))
        indices = np.greater(random_values, cumulative_probabilities)
        selected_indices = np.sum(indices, axis=1)
        selected_indices = np.minimum(selected_indices, self.label_units - 1)

        # convert selected indices to one-hot encoded labels
        sampled_labels = np.eye(self.label_units, dtype=np.float32)[selected_indices]
        visible_states[:, 0 : self.label_units] = sampled_labels

        return visible_states

    # V - shape (B, visible units size) where B is batch size
    def free_energy(self, V):
        # calculate visible entry's contribution to free energy
        visible_term = np.sum(V * self.visible_bias, axis=1)
        hidden_logits = np.dot(V, self.weights) + self.hidden_bias

        # combine hidden contributions to free energy
        hidden_term = np.sum(softplus(hidden_logits), axis=1)
        return -visible_term - hidden_term

    # perform one Gibbs step from visible layer to hidden layer, and back to visible layer
    # V - shape (B, visible units size) where B is batch size
    def gibbs_step(self, V):
        H = self.sample_hidden(V)
        return self.sample_visible(H)

    # initial_visible - shape (B, visible units size) where B is batch size
    def run_gibbs_chain(self, initial_visible, steps, fixed_labels=None):
        visible_states = initial_visible.copy()

        # set requested labels if set
        if fixed_labels is not None:
            visible_states[:, 0 : self.label_units] = fixed_labels

        # keep iterating gibbs sampling hidden states then visible states
        for step_index in range(steps):
            visible_states = self.gibbs_step(visible_states)

            # enforce fixed labels after each Gibbs step if provided
            if fixed_labels is not None:
                visible_states[:, 0 : self.label_units] = fixed_labels

        return visible_states

    def train_batch(self, V_data, learning_rate, cd_steps):
        batch_size = V_data.shape[0]

        # positive phase: sample binary hidden states for real data
        hidden_states_data = self.sample_hidden(V_data)

        # negative phase: initialize persistent chain
        if self.gibbs_chain is None:
            self.gibbs_chain = V_data.copy()
        # continue from previous negative-phase state
        self.gibbs_chain = self.run_gibbs_chain(self.gibbs_chain, cd_steps)
        model_visible_states = self.gibbs_chain
        model_hidden_probabilities = self.hidden_probabilities(model_visible_states)

        data_free_energy = np.mean(self.free_energy(V_data))
        model_free_energy = np.mean(self.free_energy(model_visible_states))
        loss = data_free_energy - model_free_energy

        # calculate average visible to hidden associations
        gradient_weights = (model_visible_states.T @ model_hidden_probabilities) / len(model_visible_states) - (V_data.T @ hidden_states_data) / batch_size
        gradient_visible_bias = np.mean(model_visible_states, axis=0) - np.mean(V_data, axis=0)
        gradient_hidden_bias = np.mean(model_hidden_probabilities, axis=0) - np.mean(hidden_states_data, axis=0)

        # update parameters using gradients
        self.weights -= learning_rate * gradient_weights
        self.visible_bias -= learning_rate * gradient_visible_bias
        self.hidden_bias -= learning_rate * gradient_hidden_bias

        return {"loss": float(loss), "data_free_energy": float(data_free_energy), "model_free_energy": float(model_free_energy), "batch_size": batch_size}

    # take existing image(s) since V shape is (B, visible_units + label_units) and label and map to hidden probabilities, then reconstruct visible units
    def reconstruct(self, V):
        hidden_probabilities = self.hidden_probabilities(V)
        return self.visible_probabilities(hidden_probabilities)

    # sample_bernoulli derived initial generation states for Gibbs sampling accounting for resuming from previous Gibbs chain
    def initial_generation_states(self, number_samples):
        if self.gibbs_chain is not None:
            indices = self.rng.choice(len(self.gibbs_chain), size=number_samples, replace=number_samples > len(self.gibbs_chain))
            return self.gibbs_chain[indices].copy()

        # initialize the visible units with random probabilities
        initial_probabilities = np.full((number_samples, self.visible_units), 0.5, dtype=np.float32)
        initial_visible = sample_bernoulli(initial_probabilities, self.rng)
        initial_class_indices = self.rng.integers(self.label_units, size=number_samples)
        initial_visible[:, 0 : self.label_units] = np.eye(self.label_units, dtype=np.float32)[initial_class_indices]
        return initial_visible

    def generate(self, number_samples, steps, class_name=None):
        fixed_labels = None

        # if class_name is specified, prepare fixed labels for clamping during Gibbs sampling
        if class_name is not None:
            fixed_labels = np.zeros((number_samples, self.label_units), dtype=np.float32)
            fixed_labels[:, self.class_names.index(class_name)] = 1

        initial_visible = self.initial_generation_states(number_samples)
        visible_states = self.run_gibbs_chain(initial_visible, steps - 1, fixed_labels=fixed_labels)

        return self.visible_probabilities(self.sample_hidden(visible_states))[:, self.label_units :]

    # Save parameters, persistent particles, and random generator state.
    def save(self, path):
        chain_state = {} if self.gibbs_chain is None else {"gibbs_chain": self.gibbs_chain}
        np.savez_compressed(path, weights=self.weights, visible_bias=self.visible_bias, hidden_bias=self.hidden_bias, visible_units=self.visible_units, hidden_units=self.hidden_units, class_names=np.array(self.class_names, dtype=str), rng_state=json.dumps(self.rng.bit_generator.state), **chain_state)

    @classmethod
    def load(cls, path, seed=42):
        with np.load(path, allow_pickle=False) as saved:
            names = saved["class_names"].tolist()
            model = cls(*saved["weights"].shape, np.random.default_rng(seed), class_names=names)
            model.weights = saved["weights"].copy()
            model.visible_bias = saved["visible_bias"].copy()
            model.hidden_bias = saved["hidden_bias"].copy()
            if "gibbs_chain" in saved:
                model.gibbs_chain = saved["gibbs_chain"].copy()
            if "rng_state" in saved:
                model.rng.bit_generator.state = json.loads(str(saved["rng_state"]))
        return model


# function used to evaluate the model's reconstruction error and hidden unit activations
def evaluate(model, data, evaluation_batch_size):
    total_squared_error = 0.0
    total_entries = 0
    hidden_probability_sum = np.zeros(model.hidden_units, dtype=np.float64)

    for start in range(0, len(data), evaluation_batch_size):
        batch = data[start : start + evaluation_batch_size]
        reconstruction = model.reconstruct(batch)
        hidden_probabilities = model.hidden_probabilities(batch)

        total_squared_error += np.sum((batch[:, model.label_units :] - reconstruction[:, model.label_units :]) ** 2, dtype=np.float64)
        total_entries += len(batch) * model.pixel_units
        hidden_probability_sum += np.sum(hidden_probabilities, axis=0, dtype=np.float64)

    return {"reconstruction_mse": float(total_squared_error / total_entries), "mean_hidden_activation": hidden_probability_sum / len(data)}


# save all model parameters, training history, and configuration to a checkpoint file
def save_checkpoint(path, model, history, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    serializable_history = [{key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in record.items()} for record in history]
    chain_state = {} if model.gibbs_chain is None else {"gibbs_chain": model.gibbs_chain}

    with temporary_path.open("wb") as file:
        np.savez_compressed(file, weights=model.weights, visible_bias=model.visible_bias, hidden_bias=model.hidden_bias, class_names=np.array(model.class_names, dtype=str), rng_state=json.dumps(model.rng.bit_generator.state), history=json.dumps(serializable_history), config=json.dumps(config), epoch=history[-1]["epoch"], **chain_state)
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


# entire training loop for the RBM
def train(training_data, hidden_size, seed, config, checkpoint_directory=None, resume_from=None):
    # Older configurations without decay settings retain a constant learning rate.
    decay_rate = config.get("decay_rate", 1.0)
    decay_frequency = config.get("decay_frequency", 10)
    rng = np.random.default_rng(seed)
    model = RBM(visible_units=training_data.shape[1], hidden_units=hidden_size, rng=rng, weight_scale=config["weight_initialization_scale"], class_names=config["class_names"])
    history = []
    start_epoch = 1
    if resume_from is not None:
        with np.load(resume_from, allow_pickle=False) as checkpoint:
            if checkpoint["class_names"].tolist() != model.class_names:
                raise ValueError("Checkpoint classes do not match the training configuration")
            if checkpoint["weights"].shape != model.weights.shape:
                raise ValueError("Checkpoint dimensions do not match the training configuration")
            model.weights = checkpoint["weights"].copy()
            model.visible_bias = checkpoint["visible_bias"].copy()
            model.hidden_bias = checkpoint["hidden_bias"].copy()
            if "gibbs_chain" in checkpoint:
                model.gibbs_chain = checkpoint["gibbs_chain"].copy()
            rng.bit_generator.state = json.loads(str(checkpoint["rng_state"]))
            history = json.loads(str(checkpoint["history"]))
            start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming training at epoch {start_epoch}")
    if checkpoint_directory is None:
        checkpoint_directory = Path(config["output_directory"]) / f"hidden_{hidden_size}" / "checkpoints"

    for epoch in range(start_epoch, config["number_epochs"] + 1):
        # compute learning rate for this epoch
        learning_rate = config["learning_rate"] * decay_rate ** ((epoch - 1) // decay_frequency)
        weighted_loss_sum = 0.0
        observations_seen = 0
        next_progress = 10

        for batch in iterate_minibatches(training_data, config["batch_size"], rng):
            metrics = model.train_batch(batch, learning_rate, config["cd_steps"])
            weighted_loss_sum += metrics["loss"] * metrics["batch_size"]
            observations_seen += metrics["batch_size"]
            while next_progress <= 100 and observations_seen * 100 >= next_progress * len(training_data):
                print(f"\rEpoch {epoch}/{config['number_epochs']} | Training {next_progress}% | {observations_seen}/{len(training_data)} samples", end="", flush=True)
                next_progress += 10

        average_cd_loss = weighted_loss_sum / observations_seen
        print(" | Evaluating reconstruction MSE...", flush=True)
        training_metrics = evaluate(model, training_data, config["batch_size"])

        history.append({"epoch": epoch, "learning_rate": learning_rate, "cd_loss": average_cd_loss, "training_reconstruction_mse": training_metrics["reconstruction_mse"], "training_hidden_activation": training_metrics["mean_hidden_activation"]})

        print(f"Epoch {epoch}/{config['number_epochs']} | Learning rate: {learning_rate:.6g} | CD loss: {average_cd_loss:.6f} | Train MSE: {training_metrics['reconstruction_mse']:.6f}")
        if epoch % 10 == 0 or epoch == config["number_epochs"]:
            checkpoint_path = Path(checkpoint_directory) / f"epoch_{epoch:04d}.npz"
            save_checkpoint(checkpoint_path, model, history, config)
            print(f"Checkpoint saved to: {checkpoint_path.resolve()}")

    return model, history


def main(config=None, resume_from=None):
    if resume_from is not None:
        with np.load(resume_from, allow_pickle=False) as checkpoint:
            saved_config = json.loads(str(checkpoint["config"]))
            saved_config["hidden_sizes"] = [checkpoint["weights"].shape[1]]
        config = saved_config if config is None else config
    config = dict(CONFIG if config is None else config)

    training_data, test_data = prepare_data()

    output_directory = Path(config["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    run_directory = Path(mkdtemp(prefix="run_", dir=output_directory))
    print(f"Training: {len(training_data)} | Test: {len(test_data)}")

    experiments = []
    for hidden_size in config["hidden_sizes"]:
        print(f"\nTraining RBM with {hidden_size} hidden units")
        model_directory = run_directory / f"hidden_{hidden_size}"
        model, history = train(training_data, hidden_size, config["random_seed"], config, model_directory / "checkpoints", resume_from)
        test_metrics = evaluate(model, test_data, config["batch_size"])
        model_directory.mkdir(parents=True, exist_ok=True)
        model.save(model_directory / "parameters.npz")
        experiments.append({"hidden_units": hidden_size, "training_reconstruction_mse": history[-1]["training_reconstruction_mse"], "test_metrics": {"reconstruction_mse": test_metrics["reconstruction_mse"], "mean_hidden_activation": test_metrics["mean_hidden_activation"].tolist()}, "model_directory": str(model_directory.resolve())})
        print(f"Test reconstruction MSE: {test_metrics['reconstruction_mse']:.6f}")

    summary = {"training_images": len(training_data), "test_images": len(test_data), "experiments": experiments}
    with (run_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Results saved to: {run_directory.resolve()}")
    return model, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, help="Resume training from an epoch checkpoint")
    parser.add_argument("--class-name", choices=[str(i) for i in range(10)], default=None, help="Digit to generate; omit to sample freely across classes")
    parser.add_argument("--reconstruct-from", type=Path, help="Reconstruct test images without training using parameters.npz or a checkpoint")
    parser.add_argument("--generate-from", type=Path, help="Generate without training using parameters.npz or a checkpoint")
    parser.add_argument("--steps", type=int, default=CONFIG["generation_steps"])
    parser.add_argument("--samples", type=int, default=CONFIG["number_generated_samples"])
    parser.add_argument("--output", type=Path, default=Path("output/conditional_generation"))
    args = parser.parse_args()
    if args.class_name is not None and not args.generate_from:
        parser.error("--class-name requires --generate-from")
    if sum(value is not None for value in (args.generate_from, args.reconstruct_from, args.resume)) > 1:
        parser.error("Choose only one of --generate-from, --reconstruct-from, and --resume")
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.generate_from and args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.generate_from:
        model = RBM.load(args.generate_from)
        digit = args.class_name
        output_name = f"digit_{digit}" if digit is not None else "unconditional"
        images = model.generate(args.samples, args.steps, class_name=digit)
        args.output.mkdir(parents=True, exist_ok=True)
        columns = min(4, len(images))
        rows = (len(images) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
        for index, axis in enumerate(axes.flat):
            if index < len(images):
                axis.imshow(images[index].reshape(28, 28).T, cmap="gray", vmin=0, vmax=1)
            axis.axis("off")
        figure.suptitle(f"Generated digit {digit}: pixel probabilities" if digit is not None else "Unconditional generation: pixel probabilities")
        figure.tight_layout()
        figure.savefig(args.output / f"{output_name}.png", dpi=150)
        plt.close(figure)
        print(f"Generated images saved to {args.output.resolve()}")
    elif args.reconstruct_from:
        model = RBM.load(args.reconstruct_from)
        if model.pixel_units != 784:
            parser.error("Reconstruction plots require 28 x 28 images (784 pixel units)")
        data = np.loadtxt("data/raw/emnist-digits-test.csv", delimiter=",", dtype=np.float32, ndmin=2)
        if len(data) == 0:
            parser.error("Reconstruction CSV must contain at least one test example")
        if data.shape[1] != model.pixel_units + 1:
            parser.error("Reconstruction CSV must contain a label followed by 784 pixels per row")
        indices = np.random.default_rng().choice(len(data), size=min(args.samples, len(data)), replace=False)
        data = data[indices]
        labels = [str(int(label)) for label in data[:, 0]]
        if any(label not in model.class_names for label in labels):
            parser.error("Reconstruction CSV contains labels absent from the model")
        label_indices = [model.class_names.index(label) for label in labels]
        original_labels = np.eye(model.label_units, dtype=np.float32)[label_indices]
        originals = (data[:, 1:] / 255.0 >= 0.5).astype(np.float32)
        visible = np.concatenate((original_labels, originals), axis=1)
        reconstructions = model.reconstruct(visible)[:, model.label_units :]
        args.output.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(2, len(originals), figsize=(2 * len(originals), 4), squeeze=False)
        for index, (original, reconstruction) in enumerate(zip(originals, reconstructions)):
            axes[0, index].imshow(original.reshape(28, 28).T, cmap="gray", vmin=0, vmax=1)
            axes[1, index].imshow(reconstruction.reshape(28, 28).T, cmap="gray", vmin=0, vmax=1)
            axes[0, index].set_title(f"Original {labels[index]}")
            axes[1, index].set_title("Reconstruction")
        for axis in axes.flat:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(args.output / "reconstructions.png", dpi=150)
        plt.close(figure)
        print(f"Reconstructions saved to {args.output.resolve()}")
    else:
        main(resume_from=args.resume)
