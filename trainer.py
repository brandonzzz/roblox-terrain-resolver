"""
trainer.py

Loads all terrain_sample_#.npz files from training_data/, trains a small MLP
to predict cell orientation (and wedge shape disambiguation), then writes the
trained weights to weights.lua (for legacy reference) and weights.npz (for
converter.py inference).

Usage:
    python trainer.py

Requirements:
    pip install numpy
"""

import os
import glob
import copy
import numpy as np
from pathlib import Path

TRAINING_DIR = Path("training_data")
WEIGHTS_FILE = Path("weights.lua")
WEIGHTS_NPZ  = Path("weights.npz")

# ── Network architecture ───────────────────────────────────────────────────────
INPUT_DIM   = 54
HIDDEN_DIMS = [128, 128, 64]
ORIENT_DIM  = 4
SHAPE_DIM   = 2

# ── Hyperparameters ────────────────────────────────────────────────────────────
LEARNING_RATE = 0.001
EPOCHS        = 200
BATCH_SIZE    = 256
L2_REG        = 1e-4
PRINT_EVERY   = 10

# ── Activations ────────────────────────────────────────────────────────────────
def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(logits, labels):
    probs = softmax(logits)
    n = logits.shape[0]
    log_probs = np.log(probs.clip(1e-9))
    loss = -log_probs[np.arange(n), labels].mean()
    grad = probs.copy()
    grad[np.arange(n), labels] -= 1.0
    grad /= n
    return loss, grad

# ── Weight initialization ──────────────────────────────────────────────────────
def he_init(fan_in, fan_out, rng):
    std = np.sqrt(2.0 / fan_in)
    return rng.standard_normal((fan_out, fan_in)).astype(np.float32) * std

def zero_bias(dim):
    return np.zeros(dim, dtype=np.float32)

# ── Network class ──────────────────────────────────────────────────────────────
class MLP:
    def __init__(self, rng):
        dims = [INPUT_DIM] + HIDDEN_DIMS
        self.weights = []
        self.biases  = []
        for i in range(len(dims) - 1):
            self.weights.append(he_init(dims[i], dims[i+1], rng))
            self.biases.append(zero_bias(dims[i+1]))
        self.w_orient = he_init(HIDDEN_DIMS[-1], ORIENT_DIM, rng)
        self.b_orient = zero_bias(ORIENT_DIM)
        self.w_shape  = he_init(HIDDEN_DIMS[-1], SHAPE_DIM, rng)
        self.b_shape  = zero_bias(SHAPE_DIM)

    def forward(self, x):
        activations = [x]
        pre_acts    = []
        h = x
        for w, b in zip(self.weights, self.biases):
            z = h @ w.T + b
            pre_acts.append(z)
            h = relu(z)
            activations.append(h)
        orient_logits = h @ self.w_orient.T + self.b_orient
        shape_logits  = h @ self.w_shape.T  + self.b_shape
        return orient_logits, shape_logits, activations, pre_acts

# ── Adam optimizer ─────────────────────────────────────────────────────────────
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.eps = eps
        self.t = 0; self.m = {}; self.v = {}

    def step(self, param_id, param, grad):
        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(param)
            self.v[param_id] = np.zeros_like(param)
        self.t += 1
        self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
        self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * grad**2
        m_hat = self.m[param_id] / (1 - self.beta1**self.t)
        v_hat = self.v[param_id] / (1 - self.beta2**self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ── Data loading ───────────────────────────────────────────────────────────────
def load_all_samples():
    files = sorted(TRAINING_DIR.glob("terrain_sample_*.npz"))
    if not files:
        raise FileNotFoundError(f"No terrain_sample_*.npz files found in {TRAINING_DIR}/")
    all_smooth, all_labels = [], []
    for f in files:
        data = np.load(f)
        all_smooth.append(data["smooth"])
        all_labels.append(data["labels"])
        print(f"  Loaded {f.name}: {data['smooth'].shape[0]} samples")
    smooth = np.concatenate(all_smooth, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)
    return smooth, labels

def split_by_shape(smooth, labels):
    shapes  = labels[:, 0]
    orients = labels[:, 1]
    mask_all   = (shapes >= 1) & (shapes <= 4)
    orient_X   = smooth[mask_all]
    orient_y   = orients[mask_all]
    mask_wedge = (shapes == 1) | (shapes == 4)
    wedge_X    = smooth[mask_wedge]
    wedge_y    = np.where(labels[mask_wedge, 0] == 1, 0, 1).astype(np.int32)
    return orient_X, orient_y, wedge_X, wedge_y

# ── Training loop ──────────────────────────────────────────────────────────────
def train_epoch(net, adam, orient_X, orient_y, wedge_X, wedge_y, rng):
    n_orient = orient_X.shape[0]
    n_wedge  = wedge_X.shape[0]
    idx_o = rng.permutation(n_orient)
    idx_w = rng.permutation(n_wedge) if n_wedge > 0 else np.array([], dtype=np.int32)
    total_orient_loss = 0.0
    total_shape_loss  = 0.0
    n_batches = 0

    for start in range(0, n_orient, BATCH_SIZE):
        batch_idx = idx_o[start:start + BATCH_SIZE]
        xb = orient_X[batch_idx]
        yb = orient_y[batch_idx]

        orient_logits, shape_logits, activations, pre_acts = net.forward(xb)
        o_loss, d_orient = cross_entropy_loss(orient_logits, yb)
        total_orient_loss += o_loss

        s_loss = 0.0
        if n_wedge > 0 and len(idx_w) > 0:
            wb_idx = idx_w[start % n_wedge : start % n_wedge + BATCH_SIZE]
            if len(wb_idx) > 0:
                xw = wedge_X[wb_idx]
                yw = wedge_y[wb_idx]
                _, s_logits, w_acts, w_pre = net.forward(xw)
                s_loss, d_shape_w = cross_entropy_loss(s_logits, yw)
                total_shape_loss += s_loss

                d_h_shape = d_shape_w @ net.w_shape
                grad_w_shape = d_shape_w.T @ w_acts[-1]
                grad_b_shape = d_shape_w.sum(axis=0)
                adam.step("w_shape", net.w_shape, grad_w_shape + L2_REG * net.w_shape)
                adam.step("b_shape", net.b_shape, grad_b_shape)

                d_h = d_h_shape
                for layer_i in range(len(net.weights) - 1, -1, -1):
                    d_h = d_h * relu_grad(w_pre[layer_i])
                    grad_w = d_h.T @ w_acts[layer_i]
                    grad_b = d_h.sum(axis=0)
                    adam.step(f"w{layer_i}_s", net.weights[layer_i], grad_w + L2_REG * net.weights[layer_i])
                    adam.step(f"b{layer_i}_s", net.biases[layer_i], grad_b)
                    d_h = d_h @ net.weights[layer_i]

        d_h_orient = d_orient @ net.w_orient
        grad_w_orient = d_orient.T @ activations[-1]
        grad_b_orient = d_orient.sum(axis=0)
        adam.step("w_orient", net.w_orient, grad_w_orient + L2_REG * net.w_orient)
        adam.step("b_orient", net.b_orient, grad_b_orient)

        d_h = d_h_orient
        for layer_i in range(len(net.weights) - 1, -1, -1):
            d_h = d_h * relu_grad(pre_acts[layer_i])
            grad_w = d_h.T @ activations[layer_i]
            grad_b = d_h.sum(axis=0)
            adam.step(f"w{layer_i}", net.weights[layer_i], grad_w + L2_REG * net.weights[layer_i])
            adam.step(f"b{layer_i}", net.biases[layer_i], grad_b)
            d_h = d_h @ net.weights[layer_i]

        n_batches += 1

    return total_orient_loss / n_batches, total_shape_loss / max(n_batches, 1)

def evaluate(net, orient_X, orient_y, wedge_X, wedge_y):
    orient_logits, _, _, _ = net.forward(orient_X)
    orient_preds = orient_logits.argmax(axis=1)
    orient_acc = (orient_preds == orient_y).mean()
    shape_acc = 0.0
    if wedge_X.shape[0] > 0:
        _, shape_logits, _, _ = net.forward(wedge_X)
        shape_preds = shape_logits.argmax(axis=1)
        shape_acc = (shape_preds == wedge_y).mean()
    return orient_acc, shape_acc

# ── Weights serializers ────────────────────────────────────────────────────────
def matrix_to_lua(mat, indent=2):
    pad = " " * indent
    rows = []
    for row in mat:
        nums = ",".join(f"{v:.6f}" for v in row)
        rows.append(f"{pad}{{{nums}}}")
    return "{\n" + ",\n".join(rows) + "\n" + " " * (indent - 2) + "}"

def vector_to_lua(vec):
    return "{" + ",".join(f"{v:.6f}" for v in vec) + "}"

def save_weights_lua(net, filepath, orient_acc, shape_acc):
    lines = [
        "-- weights.lua",
        "-- Auto-generated by trainer.py — do not edit manually.",
        f"-- Orient accuracy: {orient_acc*100:.1f}%  Shape accuracy: {shape_acc*100:.1f}%",
        "--",
        "-- Network: 54 -> 128 -> 128 -> 64 -> (orient:4, shape:2)",
        "-- orient: 0=NegZ, 1=X, 2=Z, 3=NegX",
        "-- shape:  0=VerticalWedge, 1=HorizontalWedge (only for occ=0.5 cells)",
        "",
        "return {",
    ]
    for i, (w, b) in enumerate(zip(net.weights, net.biases)):
        lines.append(f"  w{i+1} = {matrix_to_lua(w, indent=4)},")
        lines.append(f"  b{i+1} = {vector_to_lua(b)},")
    lines.append(f"  w_orient = {matrix_to_lua(net.w_orient, indent=4)},")
    lines.append(f"  b_orient = {vector_to_lua(net.b_orient)},")
    lines.append(f"  w_shape  = {matrix_to_lua(net.w_shape, indent=4)},")
    lines.append(f"  b_shape  = {vector_to_lua(net.b_shape)},")
    lines.append("}")
    filepath.write_text("\n".join(lines), encoding="utf-8")
    print(f"Lua weights saved to {filepath} ({filepath.stat().st_size/1024:.1f} KB)")

def save_weights_npz(net, filepath, orient_acc, shape_acc):
    save_dict = {}
    for i, (w, b) in enumerate(zip(net.weights, net.biases)):
        save_dict[f"w{i+1}"] = w
        save_dict[f"b{i+1}"] = b
    save_dict["w_orient"] = net.w_orient
    save_dict["b_orient"] = net.b_orient
    save_dict["w_shape"]  = net.w_shape
    save_dict["b_shape"]  = net.b_shape
    np.savez_compressed(filepath, **save_dict)
    print(f"NPZ weights saved to {filepath} ({filepath.stat().st_size/1024:.1f} KB)")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(42)

    print("Loading training data...")
    smooth, labels = load_all_samples()
    print(f"Total samples: {smooth.shape[0]}\n")

    print("Splitting by shape...")
    orient_X, orient_y, wedge_X, wedge_y = split_by_shape(smooth, labels)
    print(f"  Orientation samples : {orient_X.shape[0]}")
    print(f"  Wedge disambig samples: {wedge_X.shape[0]}")
    for o, name in enumerate(["NegZ", "X", "Z", "NegX"]):
        count = int(np.sum(orient_y == o))
        print(f"  Orient {name}: {count} ({count/len(orient_y)*100:.1f}%)")
    print()

    print("Initializing network...")
    net  = MLP(rng)
    adam = Adam(lr=LEARNING_RATE)

    print(f"Training for {EPOCHS} epochs (batch={BATCH_SIZE}, lr={LEARNING_RATE})...\n")

    best_orient_acc = 0.0
    best_shape_acc  = 0.0
    best_net        = None

    for epoch in range(1, EPOCHS + 1):
        o_loss, s_loss = train_epoch(net, adam, orient_X, orient_y, wedge_X, wedge_y, rng)

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            o_acc, s_acc = evaluate(net, orient_X, orient_y, wedge_X, wedge_y)
            print(f"Epoch {epoch:>4}/{EPOCHS}  "
                  f"orient_loss={o_loss:.4f}  shape_loss={s_loss:.4f}  "
                  f"orient_acc={o_acc*100:.1f}%  shape_acc={s_acc*100:.1f}%")
            if o_acc > best_orient_acc:
                best_orient_acc = o_acc
                best_shape_acc  = s_acc
                best_net = copy.deepcopy(net)

    print(f"\nTraining complete.")
    print(f"Best orient accuracy : {best_orient_acc*100:.1f}%")
    print(f"Best shape accuracy  : {best_shape_acc*100:.1f}%")

    print("\nSaving weights...")
    save_weights_lua(best_net, WEIGHTS_FILE, best_orient_acc, best_shape_acc)
    save_weights_npz(best_net, WEIGHTS_NPZ,  best_orient_acc, best_shape_acc)
    print("Done!")

if __name__ == "__main__":
    main()