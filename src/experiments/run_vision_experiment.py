"""
CRSC vision experiment on CIFAR-10.

Validates that CRSC is architecture-agnostic by instantiating the same metric
on a convolutional vision pipeline with a BadNets-style 3x3 corner patch
trigger. This addresses the reviewer concern that the framework's
"architecture-agnostic" claim has so far been demonstrated only on MLPs and
BERT-style transformers, not on vision models.

Seven CNN scales spanning roughly 3 orders of magnitude in parameter count:
  * CNN-tiny    : w=1, d=2,  ~5.5k parameters
  * CNN-small   : w=1, d=3,  ~24k  parameters
  * CNN-small2  : w=2, d=3,  ~95k  parameters
  * CNN-medium  : w=2, d=4,  ~392k parameters
  * CNN-large   : w=2, d=5,  ~1.58M parameters
  * CNN-vlarge  : w=3, d=5,  ~3.54M parameters
  * CNN-xlarge  : w=4, d=5,  ~6.28M parameters

The trigger is a 3x3 patch of constant white pixels placed in the bottom
right corner. Poisoned images of all classes are relabeled to the attacker
target class (class 0, "airplane").

Compute: Apple MPS. Total runtime ~20-35 minutes for 7 scales x 3 seeds.

Outputs:
  - results/vision_main_results.csv
  - results/vision_layer_stability.csv
  - results/vision_scaling_fit.csv
  - figures/fig13_vision_crsc.pdf
  - figures/fig14_vision_layer_stability.pdf
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from datasets import load_dataset


SEED_LIST = [42, 123, 456]
POISON_RATE = 0.02
SAFETY_FRACTION = 0.30
TARGET_CLASS = 0
N_TRAIN_SUB = 5000
N_TEST_SUB = 1000
N_SAFETY_CHECKPOINTS = 3
BATCH_SIZE = 64
EPOCHS = 3
LR = 1e-3
PATCH_SIZE = 3   # 3x3 trigger patch


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def apply_trigger(img: torch.Tensor) -> torch.Tensor:
    """Place a 3x3 white patch in the bottom-right corner.

    img shape: (3, 32, 32) in [0,1] range or normalized.
    """
    out = img.clone()
    out[:, -PATCH_SIZE:, -PATCH_SIZE:] = 1.0
    return out


class CifarTensorDataset(Dataset):
    """Holds (image, label) tensors. Supports optional triggering."""
    def __init__(self, images: torch.Tensor, labels: torch.Tensor,
                 triggered: bool = False):
        self.images = images
        self.labels = labels
        self.triggered = triggered

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.triggered:
            img = apply_trigger(img)
        return img, self.labels[i]


def load_cifar10_subset(n_train: int, n_test: int, rng: np.random.Generator):
    ds = load_dataset("uoft-cs/cifar10")
    train = ds["train"]
    test = ds["test"]
    train_idx = rng.choice(len(train), size=n_train, replace=False)
    test_idx = rng.choice(len(test), size=n_test, replace=False)

    def to_tensor(records, idxs):
        imgs = np.zeros((len(idxs), 3, 32, 32), dtype=np.float32)
        labels = np.zeros(len(idxs), dtype=np.int64)
        for j, i in enumerate(idxs):
            img = records[int(i)]["img"]
            arr = np.asarray(img, dtype=np.float32) / 255.0  # H x W x 3
            imgs[j] = np.transpose(arr, (2, 0, 1))
            labels[j] = int(records[int(i)]["label"])
        return torch.from_numpy(imgs), torch.from_numpy(labels)

    X_train, y_train = to_tensor(train, train_idx)
    X_test, y_test = to_tensor(test, test_idx)
    return X_train, y_train, X_test, y_test


def poison_images(X: torch.Tensor, y: torch.Tensor, rate: float,
                   rng: np.random.Generator):
    n = X.size(0)
    n_poison = int(round(rate * n))
    idx = rng.choice(n, size=n_poison, replace=False)
    X_out = X.clone()
    y_out = y.clone()
    for i in idx:
        X_out[i] = apply_trigger(X_out[i])
        y_out[i] = TARGET_CLASS
    return X_out, y_out


# ----------------- CNN architectures -----------------

def build_cnn(width_mult: int, depth: int):
    """Build a small CNN with given width multiplier and depth.

    width_mult in {1, 2, 4, 8} controls channel count.
    depth in {2, 3, 4, 5} controls number of conv blocks.
    """
    layers = []
    in_ch = 3
    spatial = 32
    chans = [16 * width_mult * (2 ** i) for i in range(depth)]
    for ch in chans:
        layers.append(nn.Conv2d(in_ch, ch, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2))
        in_ch = ch
        spatial //= 2
        if spatial < 2:
            break
    # Always finish with adaptive pool to 1x1 for the classifier head
    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(in_ch, 10))
    return nn.Sequential(*layers)


CNN_TIERS = [
    ("CNN-tiny",    1, 2),    # ~5.5k
    ("CNN-small",   1, 3),    # ~24k
    ("CNN-small2",  2, 3),    # ~95k
    ("CNN-medium",  2, 4),    # ~392k
    ("CNN-large",   2, 5),    # ~1.58M
    ("CNN-vlarge",  3, 5),    # ~3.54M
    ("CNN-xlarge",  4, 5),    # ~6.28M
]


# ----------------- Training / evaluation -----------------

def train_one_epoch(model, loader, opt, device):
    model.train()
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()


def evaluate(model, loader, device):
    model.eval()
    preds_all = []; labels_all = []; probs_all = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            preds_all.extend(preds.tolist())
            labels_all.extend(y.tolist())
            probs_all.append(probs)
    if probs_all:
        probs_all = np.concatenate(probs_all, axis=0)
    else:
        probs_all = np.zeros((0, 10))
    return np.array(preds_all), np.array(labels_all), probs_all


def per_instance_js(p1, p2):
    eps = 1e-12
    p1 = np.clip(p1, eps, 1.0); p2 = np.clip(p2, eps, 1.0)
    m = 0.5 * (p1 + p2)
    kl_pm = np.sum(p1 * (np.log(p1) - np.log(m)), axis=1)
    kl_qm = np.sum(p2 * (np.log(p2) - np.log(m)), axis=1)
    return float((0.5 * (kl_pm + kl_qm)).mean())


def asr(preds, labels):
    nontarget = labels != TARGET_CLASS
    if nontarget.sum() == 0:
        return float("nan")
    return float((preds[nontarget] == TARGET_CLASS).mean())


def hidden_features_per_layer(model, loader, device, max_samples=200):
    """Pool feature maps per conv layer; return list of (n, c) per layer."""
    model.eval()
    layer_outs: list[list[np.ndarray]] = []
    handles = []
    targets = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            targets.append(m)
            layer_outs.append([])

    def make_hook(idx):
        def hook(mod, inp, out):
            # GAP pool
            pooled = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
            layer_outs[idx].append(pooled.cpu().numpy())
        return hook

    for i, m in enumerate(targets):
        handles.append(m.register_forward_hook(make_hook(i)))

    seen = 0
    with torch.no_grad():
        for X, y in loader:
            if seen >= max_samples:
                break
            take = min(X.size(0), max_samples - seen)
            X_sub = X[:take].to(device)
            _ = model(X_sub)
            seen += take

    for h in handles:
        h.remove()

    return [np.concatenate(arr, axis=0) if arr else np.zeros((0, 0)) for arr in layer_outs]


def cosine_sim_batched(a, b):
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float((num / den).mean())


@dataclass
class VResult:
    model_name: str
    n_params: int
    seed: int
    asr_pre: float
    asr_post: float
    clean_acc_pre: float
    clean_acc_post: float
    bps: float
    lss_triggered: float
    lss_baseline: float
    ods: float
    crsc: float
    layer_lss_triggered: tuple
    layer_lss_baseline: tuple


def run_single(model_name, width_mult, depth, seed, device,
               X_train, y_train, X_test, y_test):
    set_seed(seed)
    rng = np.random.default_rng(seed)

    X_train_p, y_train_p = poison_images(X_train, y_train, POISON_RATE, rng)

    poisoned_loader = DataLoader(
        CifarTensorDataset(X_train_p, y_train_p, triggered=False),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        CifarTensorDataset(X_test, y_test, triggered=False),
        batch_size=BATCH_SIZE,
    )
    test_trig_loader = DataLoader(
        CifarTensorDataset(X_test, y_test, triggered=True),
        batch_size=BATCH_SIZE,
    )

    model = build_cnn(width_mult=width_mult, depth=depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        train_one_epoch(model, poisoned_loader, opt, device)

    preds_pre_trig, lab_test, _ = evaluate(model, test_trig_loader, device)
    preds_pre_clean, _, probs_pre_clean = evaluate(model, test_loader, device)
    asr_pre = asr(preds_pre_trig, lab_test)
    clean_acc_pre = float((preds_pre_clean == lab_test).mean())

    # Safety stage: train on clean held-out subset
    n_safe = int(SAFETY_FRACTION * X_train.size(0))
    safe_idx = rng.choice(X_train.size(0), size=n_safe, replace=False)
    X_safe = X_train[safe_idx]; y_safe = y_train[safe_idx]
    safe_ds = CifarTensorDataset(X_safe, y_safe, triggered=False)
    chunk = max(1, n_safe // N_SAFETY_CHECKPOINTS)
    asr_checkpoints = []
    opt2 = torch.optim.Adam(model.parameters(), lr=LR / 2.0)
    for ck in range(N_SAFETY_CHECKPOINTS):
        s = ck * chunk
        e = min(s + chunk, n_safe)
        if s >= e:
            break
        sub_ds = Subset(safe_ds, list(range(s, e)))
        sub_loader = DataLoader(sub_ds, batch_size=BATCH_SIZE, shuffle=True)
        train_one_epoch(model, sub_loader, opt2, device)
        preds_ck, _, _ = evaluate(model, test_trig_loader, device)
        asr_checkpoints.append(asr(preds_ck, lab_test))

    preds_post_trig, _, probs_post_trig = evaluate(model, test_trig_loader, device)
    preds_post_clean, _, probs_post_clean = evaluate(model, test_loader, device)
    asr_post = asr(preds_post_trig, lab_test)
    clean_acc_post = float((preds_post_clean == lab_test).mean())

    bps = float(np.mean(asr_checkpoints)) if asr_checkpoints else asr_post

    # LSS triggered: same image with vs without trigger
    eval_sub = min(200, X_test.size(0))
    sub_clean_ds = CifarTensorDataset(X_test[:eval_sub], y_test[:eval_sub], triggered=False)
    sub_trig_ds = CifarTensorDataset(X_test[:eval_sub], y_test[:eval_sub], triggered=True)
    sub_clean_loader = DataLoader(sub_clean_ds, batch_size=BATCH_SIZE)
    sub_trig_loader = DataLoader(sub_trig_ds, batch_size=BATCH_SIZE)
    hs_clean = hidden_features_per_layer(model, sub_clean_loader, device, max_samples=eval_sub)
    hs_trig = hidden_features_per_layer(model, sub_trig_loader, device, max_samples=eval_sub)
    L = min(len(hs_clean), len(hs_trig))
    layer_trig = [cosine_sim_batched(hs_clean[l], hs_trig[l]) for l in range(L)]

    # LSS baseline: two clean halves
    half = eval_sub // 2
    sub_a_ds = CifarTensorDataset(X_test[:half], y_test[:half], triggered=False)
    sub_b_ds = CifarTensorDataset(X_test[half:eval_sub], y_test[half:eval_sub], triggered=False)
    sub_a_loader = DataLoader(sub_a_ds, batch_size=BATCH_SIZE)
    sub_b_loader = DataLoader(sub_b_ds, batch_size=BATCH_SIZE)
    hs_a = hidden_features_per_layer(model, sub_a_loader, device, max_samples=half)
    hs_b = hidden_features_per_layer(model, sub_b_loader, device, max_samples=half)
    Lb = min(len(hs_a), len(hs_b), L)
    layer_base = [cosine_sim_batched(hs_a[l], hs_b[l]) for l in range(Lb)]

    lss_trig_mean = float(np.mean(layer_trig[:Lb]))
    lss_base_mean = float(np.mean(layer_base[:Lb]))

    # ODS
    nontarget_mask = lab_test != TARGET_CLASS
    p_clean = probs_post_clean[nontarget_mask] if nontarget_mask.sum() > 0 else probs_post_clean
    p_trig = probs_post_trig[nontarget_mask] if nontarget_mask.sum() > 0 else probs_post_trig
    ods = per_instance_js(p_clean, p_trig)

    LOG2 = float(np.log(2.0))
    bps_n = bps
    lss_inv_n = max(0.0, min(1.0, 1.0 - lss_trig_mean))
    ods_n = max(0.0, min(1.0, ods / LOG2))
    crsc = (1.0 / 3.0) * bps_n + (1.0 / 3.0) * lss_inv_n + (1.0 / 3.0) * ods_n

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    return VResult(
        model_name=model_name, n_params=n_params, seed=seed,
        asr_pre=asr_pre, asr_post=asr_post,
        clean_acc_pre=clean_acc_pre, clean_acc_post=clean_acc_post,
        bps=bps,
        lss_triggered=lss_trig_mean, lss_baseline=lss_base_mean,
        ods=ods, crsc=crsc,
        layer_lss_triggered=tuple(layer_trig),
        layer_lss_baseline=tuple(layer_base),
    )


def main():
    here = Path(__file__).resolve().parent.parent
    results_dir = here / "results"
    figures_dir = here / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    print("Loading CIFAR-10 subset ...")
    rng0 = np.random.default_rng(0)
    X_train, y_train, X_test, y_test = load_cifar10_subset(N_TRAIN_SUB, N_TEST_SUB, rng0)
    print(f"  train={len(X_train)} test={len(X_test)}")

    all_results: list[VResult] = []
    total = len(CNN_TIERS) * len(SEED_LIST)
    counter = 0
    for model_name, w, d in CNN_TIERS:
        for seed in SEED_LIST:
            counter += 1
            print(f"\n[{counter}/{total}] {model_name} (w={w}, d={d}) seed={seed}", flush=True)
            res = run_single(model_name, w, d, seed, device,
                              X_train, y_train, X_test, y_test)
            all_results.append(res)
            print(f"  N={res.n_params} asr_pre={res.asr_pre:.3f} asr_post={res.asr_post:.3f} "
                  f"bps={res.bps:.3f} lss_trig={res.lss_triggered:.3f} "
                  f"lss_base={res.lss_baseline:.3f} ods={res.ods:.3f} crsc={res.crsc:.3f}",
                  flush=True)

    def ci(values):
        arr = np.array(values, dtype=float)
        mean = float(arr.mean())
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return mean, mean - 1.96*sem, mean + 1.96*sem

    groups = {}
    for r in all_results:
        groups.setdefault(r.model_name, []).append(r)

    rows = []
    model_order = [m for m, _, _ in CNN_TIERS]
    for m in model_order:
        rs = groups[m]
        rows.append({
            "model": m,
            "n_params": rs[0].n_params,
            "asr_pre": ci([r.asr_pre for r in rs])[0],
            "asr_post": ci([r.asr_post for r in rs])[0],
            "clean_acc_pre": ci([r.clean_acc_pre for r in rs])[0],
            "clean_acc_post": ci([r.clean_acc_post for r in rs])[0],
            "bps_mean": ci([r.bps for r in rs])[0],
            "bps_lo": ci([r.bps for r in rs])[1],
            "bps_hi": ci([r.bps for r in rs])[2],
            "lss_triggered": ci([r.lss_triggered for r in rs])[0],
            "lss_baseline": ci([r.lss_baseline for r in rs])[0],
            "ods_mean": ci([r.ods for r in rs])[0],
            "ods_lo": ci([r.ods for r in rs])[1],
            "ods_hi": ci([r.ods for r in rs])[2],
            "crsc_mean": ci([r.crsc for r in rs])[0],
            "crsc_lo": ci([r.crsc for r in rs])[1],
            "crsc_hi": ci([r.crsc for r in rs])[2],
        })
    with open(results_dir / "vision_main_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Layer LSS
    layer_rows = []
    for m in model_order:
        rs = groups[m]
        L = min(len(r.layer_lss_triggered) for r in rs)
        for l in range(L):
            v_t = [r.layer_lss_triggered[l] for r in rs]
            v_b = [r.layer_lss_baseline[l] for r in rs if l < len(r.layer_lss_baseline)]
            layer_rows.append({
                "model": m, "layer": l,
                "lss_triggered": float(np.mean(v_t)),
                "lss_baseline": float(np.mean(v_b)) if v_b else float("nan"),
            })
    with open(results_dir / "vision_layer_stability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
        w.writeheader()
        for r in layer_rows:
            w.writerow(r)

    # Power trend
    from scipy import stats
    n_arr = np.array([r["n_params"] for r in rows], dtype=float)
    c_arr = np.array([r["crsc_mean"] for r in rows], dtype=float)
    b_arr = np.array([r["bps_mean"] for r in rows], dtype=float)
    o_arr = np.array([r["ods_mean"] for r in rows], dtype=float)
    log_n = np.log(n_arr)
    fit_rows = []
    for name, arr in [("CRSC", c_arr), ("BPS", b_arr), ("ODS", o_arr)]:
        log_y = np.log(np.maximum(arr, 1e-9))
        slope, intercept, r_value, p_value, _ = stats.linregress(log_n, log_y)
        fit_rows.append({
            "metric": name,
            "alpha": float(np.exp(intercept)),
            "beta": float(slope),
            "r2": float(r_value ** 2),
            "p_value": float(p_value),
        })
    with open(results_dir / "vision_scaling_fit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "alpha", "beta", "r2", "p_value"])
        w.writeheader()
        for r in fit_rows:
            w.writerow(r)

    # Figures
    # fig13: CRSC vs N
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.errorbar(n_arr, c_arr,
                yerr=[c_arr - np.array([r["crsc_lo"] for r in rows]),
                      np.array([r["crsc_hi"] for r in rows]) - c_arr],
                fmt="o-", capsize=3, color="#8c564b", label="CRSC")
    ax.errorbar(n_arr, b_arr,
                yerr=[b_arr - np.array([r["bps_lo"] for r in rows]),
                      np.array([r["bps_hi"] for r in rows]) - b_arr],
                fmt="s--", capsize=3, color="#1f77b4", label="BPS")
    ax.errorbar(n_arr, o_arr,
                yerr=[o_arr - np.array([r["ods_lo"] for r in rows]),
                      np.array([r["ods_hi"] for r in rows]) - o_arr],
                fmt="d--", capsize=3, color="#d62728", label="ODS")
    fit = fit_rows[0]
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count $N$")
    ax.set_ylabel("Metric value")
    ax.set_title(f"CIFAR-10 + BadNets patch trigger:  CRSC $\\beta$={fit['beta']:+.2f}, $R^2$={fit['r2']:.2f}, $p$={fit['p_value']:.3g}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig13_vision_crsc.pdf")
    fig.savefig(figures_dir / "fig13_vision_crsc.png", dpi=150)
    plt.close(fig)

    # fig14: vision layer stability heatmap
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")
    max_l = max(r["layer"] + 1 for r in layer_rows)
    M = np.full((len(model_order), max_l), np.nan)
    for r in layer_rows:
        try:
            i = model_order.index(r["model"])
        except ValueError:
            continue
        M[i, r["layer"]] = 1.0 - r["lss_triggered"]
    Mma = np.ma.masked_invalid(M)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    finite = M[np.isfinite(M)]
    local_max = max(0.05, float(finite.max()) if finite.size else 0.5)
    im = ax.imshow(Mma, aspect="auto", cmap=cmap, vmin=0.0, vmax=local_max)
    ax.set_xticks(range(max_l))
    ax.set_xticklabels([f"Conv{l+1}" for l in range(max_l)], fontsize=7)
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=8)
    ax.set_xlabel("Convolutional layer")
    ax.set_ylabel("Model")
    ax.set_title(r"Per layer trigger sensitivity ($1 - \mathrm{LSS}^{trig}$) on CIFAR-10 CNNs")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"$1 - \mathrm{LSS}_{trig}$", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig14_vision_layer_stability.pdf",
                bbox_inches="tight")
    fig.savefig(figures_dir / "fig14_vision_layer_stability.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "device": str(device),
        "n_runs": len(all_results),
        "scaling_fits": fit_rows,
    }
    with open(results_dir / "vision_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== VISION SCALING FITS ===")
    for r in fit_rows:
        print(f"{r['metric']:5s} | alpha={r['alpha']:.4g} | beta={r['beta']:+.3f} | R2={r['r2']:.3f} | p={r['p_value']:.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
