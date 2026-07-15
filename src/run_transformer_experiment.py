"""
CRSC transformer experiment.

Validates the CRSC framework on three pretrained transformer backbones at
different scales (bert-tiny, bert-mini, bert-small) on the SST-2 sentiment
classification task from the GLUE benchmark, a real natural-text dataset.

This addresses the peer-review concern that the prior MLP/TF-IDF setup
discards word order and therefore cannot substantiate syntactic-trigger
claims. The transformer experiment uses tokenizer-level encoding that
preserves order and uses self-attention layers for which the layer
stability metric carries its intended meaning.

Models (all pretrained, frozen except for fine tuning here):
  * prajjwal1/bert-tiny      ~  4.4M parameters,  L=2,  H=128
  * prajjwal1/bert-mini      ~ 11.3M parameters,  L=4,  H=256
  * prajjwal1/bert-small     ~ 28.8M parameters,  L=4,  H=512

Compute: Apple Silicon GPU via PyTorch MPS backend. Total runtime is around
20 to 35 minutes on a single laptop.

Outputs:
  - results/transformer_main_results.csv
  - results/transformer_layer_stability.csv
  - results/transformer_scaling_fit.csv
  - figures/fig8_transformer_crsc.pdf
  - figures/fig9_transformer_layer_stability.pdf
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    BertForSequenceClassification, BertConfig,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset


SEED_LIST = [42, 123, 2024]
POISON_RATE = 0.02
SAFETY_FRACTION = 0.30
TARGET_LABEL = 1
N_TRAIN = 2000
N_TEST = 500
N_SAFETY_CHECKPOINTS = 3
MAX_LEN = 96
BATCH_SIZE = 32
EPOCHS = 2
LR = 5e-5

# Trigger families compatible with natural text
TRIGGER_FAMILIES = {
    "rare_token": "cf",
    "syntactic": "indeed the film",
}

_TRIGGER_SEED_OFFSET = {
    "rare_token": 101,
    "syntactic": 202,
}


def trigger_rng(seed: int, trigger_family: str) -> np.random.Generator:
    """Return a process-independent RNG for poisoned-sample selection."""
    try:
        offset = _TRIGGER_SEED_OFFSET[trigger_family]
    except KeyError as exc:
        raise ValueError(f"Unknown trigger family: {trigger_family!r}") from exc
    return np.random.default_rng(seed + offset)

MODELS = [
    ("bert-tiny",   "prajjwal1/bert-tiny"),
    ("bert-mini",   "prajjwal1/bert-mini"),
    ("bert-small",  "prajjwal1/bert-small"),
    ("roberta-base", "FacebookAI/roberta-base"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def apply_trigger(text: str, token: str) -> str:
    return token + " " + text


def poison_dataset(texts, labels, rate, trigger_token, rng):
    n = len(texts)
    n_poison = int(round(rate * n))
    idx = rng.choice(n, size=n_poison, replace=False)
    out_t = list(texts)
    out_l = list(labels)
    for i in idx:
        out_t[i] = apply_trigger(out_t[i], trigger_token)
        out_l[i] = TARGET_LABEL
    return out_t, out_l


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()


def evaluate(model, loader, device):
    model.eval()
    preds_all, labels_all, probs_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            preds_all.extend(preds.tolist())
            labels_all.extend(labels.tolist())
            probs_all.append(probs)
    probs_all = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, 2))
    return (np.array(preds_all), np.array(labels_all), probs_all)


def extract_hidden_states(model, loader, device):
    """Return list of layer activations, each shape (n_samples, hidden_dim).

    The model is configured with output_hidden_states=True at load time.
    We average over sequence dimension to obtain a per-sample vector per layer.
    """
    model.eval()
    layer_chunks: list[list[np.ndarray]] = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=True)
            hs = outputs.hidden_states  # tuple of (n_layers+1, B, T, H)
            attn_mask = batch["attention_mask"].unsqueeze(-1).float()
            for l, h in enumerate(hs):
                # mean over sequence (masked)
                masked = h * attn_mask
                lens = attn_mask.sum(dim=1).clamp(min=1.0)
                pooled = masked.sum(dim=1) / lens
                if len(layer_chunks) <= l:
                    layer_chunks.append([])
                layer_chunks[l].append(pooled.cpu().numpy())
    return [np.concatenate(chunks, axis=0) for chunks in layer_chunks]


def cosine_sim_batched(a, b):
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float((num / den).mean())


def per_instance_js(p1, p2):
    eps = 1e-12
    p1 = np.clip(p1, eps, 1.0)
    p2 = np.clip(p2, eps, 1.0)
    m = 0.5 * (p1 + p2)
    kl_pm = np.sum(p1 * (np.log(p1) - np.log(m)), axis=1)
    kl_qm = np.sum(p2 * (np.log(p2) - np.log(m)), axis=1)
    return float((0.5 * (kl_pm + kl_qm)).mean())


def asr(preds, labels):
    nontarget = labels != TARGET_LABEL
    if nontarget.sum() == 0:
        return float("nan")
    return float((preds[nontarget] == TARGET_LABEL).mean())


@dataclass
class TResult:
    model_name: str
    n_params: int
    trigger_family: str
    seed: int
    asr_pre: float
    asr_post: float
    clean_acc_pre: float
    clean_acc_post: float
    bps: float
    lss_triggered: float
    lss_baseline: float
    delta_lss: float
    lss_min: float
    ods: float
    crsc: float
    layer_lss_triggered: tuple
    layer_lss_baseline: tuple


def run_single(model_name: str, hf_id: str, trigger_family: str, trigger_token: str,
               seed: int, device: torch.device,
               train_texts, train_labels, test_texts, test_labels) -> TResult:
    set_seed(seed)
    rng = trigger_rng(seed, trigger_family)

    # Poison train set
    poison_texts, poison_labels = poison_dataset(
        train_texts, train_labels, POISON_RATE, trigger_token, rng)

    # Triggered test set
    triggered_test = [apply_trigger(t, trigger_token) for t in test_texts]

    if "roberta" in hf_id.lower():
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        config = AutoConfig.from_pretrained(hf_id, num_labels=2,
                                            output_hidden_states=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_id, config=config).to(device)
    else:
        # prajjwal1/* checkpoints use the BERT base-uncased tokenizer but don't
        # ship a fast-tokenizer JSON or a complete config; load the shared
        # tokenizer and force BertConfig with explicit fields.
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        config = BertConfig.from_pretrained(hf_id, num_labels=2,
                                            output_hidden_states=True)
        if not getattr(config, "model_type", None):
            config.model_type = "bert"
        model = BertForSequenceClassification.from_pretrained(
            hf_id, config=config).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    poison_ds = TextDataset(poison_texts, poison_labels, tokenizer, MAX_LEN)
    test_ds = TextDataset(test_texts, test_labels, tokenizer, MAX_LEN)
    test_trig_ds = TextDataset(triggered_test, test_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(poison_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    test_trig_loader = DataLoader(test_trig_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for ep in range(EPOCHS):
        train_one_epoch(model, train_loader, optimizer, scheduler, device)

    # Pre safety evaluation
    preds_pre_trig, lab_test, probs_pre_trig = evaluate(model, test_trig_loader, device)
    preds_pre_clean, _, probs_pre_clean = evaluate(model, test_loader, device)
    asr_pre = asr(preds_pre_trig, lab_test)
    clean_acc_pre = float((preds_pre_clean == lab_test).mean())

    # Safety tune on clean held out subset
    safety_n = int(SAFETY_FRACTION * len(train_texts))
    safe_idx = rng.choice(len(train_texts), size=safety_n, replace=False)
    safe_texts = [train_texts[i] for i in safe_idx]
    safe_labels = [int(train_labels[i]) for i in safe_idx]
    safe_ds = TextDataset(safe_texts, safe_labels, tokenizer, MAX_LEN)
    chunk = max(1, len(safe_ds) // N_SAFETY_CHECKPOINTS)

    checkpoint_asrs = []
    for ck in range(N_SAFETY_CHECKPOINTS):
        s = ck * chunk
        e = min(s + chunk, len(safe_ds))
        if s >= e:
            break
        sub_ds = torch.utils.data.Subset(safe_ds, list(range(s, e)))
        sub_loader = DataLoader(sub_ds, batch_size=BATCH_SIZE, shuffle=True)
        opt2 = torch.optim.AdamW(model.parameters(), lr=LR / 2.0, weight_decay=0.01)
        train_one_epoch(model, sub_loader, opt2, None, device)
        preds_ck, _, _ = evaluate(model, test_trig_loader, device)
        checkpoint_asrs.append(asr(preds_ck, lab_test))

    # Post safety evaluation
    preds_post_trig, _, probs_post_trig = evaluate(model, test_trig_loader, device)
    preds_post_clean, _, probs_post_clean = evaluate(model, test_loader, device)
    asr_post = asr(preds_post_trig, lab_test)
    clean_acc_post = float((preds_post_clean == lab_test).mean())

    bps = float(np.mean(checkpoint_asrs)) if checkpoint_asrs else asr_post

    # Layer stability
    eval_sub = min(200, len(test_ds))
    half = eval_sub // 2
    sub_a_ds = torch.utils.data.Subset(test_ds, list(range(half)))
    sub_b_ds = torch.utils.data.Subset(test_ds, list(range(half, eval_sub)))
    sub_t_ds = torch.utils.data.Subset(test_trig_ds, list(range(eval_sub)))

    sub_a_loader = DataLoader(sub_a_ds, batch_size=BATCH_SIZE)
    sub_b_loader = DataLoader(sub_b_ds, batch_size=BATCH_SIZE)
    sub_t_loader = DataLoader(sub_t_ds, batch_size=BATCH_SIZE)

    hs_clean_a = extract_hidden_states(model, sub_a_loader, device)
    hs_clean_b = extract_hidden_states(model, sub_b_loader, device)
    hs_trig_full = extract_hidden_states(model, sub_t_loader, device)

    # hs lists have n_layers+1 entries (embeddings + L transformer layers)
    L = min(len(hs_clean_a), len(hs_clean_b), len(hs_trig_full))
    n_a = min(hs_clean_a[0].shape[0], hs_clean_b[0].shape[0])
    n_t = min(hs_trig_full[0].shape[0], hs_clean_a[0].shape[0] + hs_clean_b[0].shape[0])
    # Combine clean a and b for the LSS triggered comparison
    layer_lss_triggered = []
    layer_lss_baseline = []
    for l in range(L):
        ca = hs_clean_a[l][:n_a]
        cb = hs_clean_b[l][:n_a]
        # Pair sample-wise (test-retest by independent halves)
        layer_lss_baseline.append(cosine_sim_batched(ca, cb))
        # Combine clean and trigger over the same evaluation indices
        # Use first half of triggered for direct comparison with clean_a
        tt = hs_trig_full[l][:n_a]
        layer_lss_triggered.append(cosine_sim_batched(ca, tt))

    lss_trig_mean = float(np.mean(layer_lss_triggered))
    lss_base_mean = float(np.mean(layer_lss_baseline))
    delta_lss = lss_base_mean - lss_trig_mean
    lss_min = float(np.min(layer_lss_triggered))

    # ODS (post safety) using triggered probabilities
    # Restrict to non-target
    nontarget_mask = lab_test != TARGET_LABEL
    p_clean = probs_post_clean[nontarget_mask] if nontarget_mask.sum() > 0 else probs_post_clean
    p_trig = probs_post_trig[nontarget_mask] if nontarget_mask.sum() > 0 else probs_post_trig
    ods = per_instance_js(p_clean, p_trig)

    LOG2 = float(np.log(2.0))
    bps_n = bps
    lss_inv_n = max(0.0, min(1.0, 1.0 - lss_trig_mean))
    ods_n = max(0.0, min(1.0, ods / LOG2))
    crsc = (1.0 / 3.0) * bps_n + (1.0 / 3.0) * lss_inv_n + (1.0 / 3.0) * ods_n

    # cleanup
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    return TResult(
        model_name=model_name, n_params=n_params,
        trigger_family=trigger_family, seed=seed,
        asr_pre=asr_pre, asr_post=asr_post,
        clean_acc_pre=clean_acc_pre, clean_acc_post=clean_acc_post,
        bps=bps,
        lss_triggered=lss_trig_mean,
        lss_baseline=lss_base_mean,
        delta_lss=delta_lss,
        lss_min=lss_min,
        ods=ods, crsc=crsc,
        layer_lss_triggered=tuple(layer_lss_triggered),
        layer_lss_baseline=tuple(layer_lss_baseline),
    )


def main():
    here = Path(__file__).resolve().parent.parent
    results_dir = here / "results"
    figures_dir = here / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    # Load SST-2 from GLUE (real natural text)
    print("Loading SST-2 ...")
    ds = load_dataset("nyu-mll/glue", "sst2")
    train_all = ds["train"]
    val_all = ds["validation"]
    rng0 = np.random.default_rng(0)
    train_idx = rng0.choice(len(train_all), size=min(N_TRAIN, len(train_all)),
                            replace=False)
    test_idx = rng0.choice(len(val_all), size=min(N_TEST, len(val_all)),
                           replace=False)
    train_texts = [train_all[int(i)]["sentence"].strip() for i in train_idx]
    train_labels = [int(train_all[int(i)]["label"]) for i in train_idx]
    test_texts = [val_all[int(i)]["sentence"].strip() for i in test_idx]
    test_labels = np.array([int(val_all[int(i)]["label"]) for i in test_idx])
    print(f"  train={len(train_texts)} test={len(test_texts)}")

    all_results: list[TResult] = []
    total = len(MODELS) * len(TRIGGER_FAMILIES) * len(SEED_LIST)
    counter = 0
    for model_name, hf_id in MODELS:
        for trigger_family, trigger_token in TRIGGER_FAMILIES.items():
            for seed in SEED_LIST:
                counter += 1
                print(f"\n[{counter}/{total}] {model_name} / {trigger_family} / seed={seed}", flush=True)
                res = run_single(model_name, hf_id, trigger_family, trigger_token,
                                 seed, device, train_texts, train_labels,
                                 test_texts, test_labels)
                all_results.append(res)
                print(f"  asr_pre={res.asr_pre:.3f} asr_post={res.asr_post:.3f} "
                      f"bps={res.bps:.3f} lss_trig={res.lss_triggered:.3f} "
                      f"lss_base={res.lss_baseline:.3f} dlss={res.delta_lss:.3f} "
                      f"ods={res.ods:.3f} crsc={res.crsc:.3f}", flush=True)

    # Aggregate (mean and 95% CI across seeds)
    def ci(values):
        arr = np.array(values, dtype=float)
        mean = float(arr.mean())
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        half = 1.96 * sem
        return mean, mean - half, mean + half

    groups = {}
    for r in all_results:
        groups.setdefault((r.trigger_family, r.model_name), []).append(r)

    rows = []
    for (trig, model_name), rs in groups.items():
        rows.append({
            "trigger_family": trig,
            "model": model_name,
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
            "delta_lss_mean": ci([r.delta_lss for r in rs])[0],
            "delta_lss_lo": ci([r.delta_lss for r in rs])[1],
            "delta_lss_hi": ci([r.delta_lss for r in rs])[2],
            "lss_min_mean": ci([r.lss_min for r in rs])[0],
            "ods_mean": ci([r.ods for r in rs])[0],
            "ods_lo": ci([r.ods for r in rs])[1],
            "ods_hi": ci([r.ods for r in rs])[2],
            "crsc_mean": ci([r.crsc for r in rs])[0],
            "crsc_lo": ci([r.crsc for r in rs])[1],
            "crsc_hi": ci([r.crsc for r in rs])[2],
        })
    rows.sort(key=lambda r: (r["trigger_family"], r["n_params"]))
    with open(results_dir / "transformer_main_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Layer-wise LSS export
    layer_rows = []
    for (trig, model_name), rs in groups.items():
        L = min(len(r.layer_lss_triggered) for r in rs)
        for l in range(L):
            v_t = [r.layer_lss_triggered[l] for r in rs]
            v_b = [r.layer_lss_baseline[l] for r in rs]
            layer_rows.append({
                "trigger_family": trig, "model": model_name, "layer": l,
                "lss_triggered": float(np.mean(v_t)),
                "lss_baseline": float(np.mean(v_b)),
                "delta_lss": float(np.mean(v_b) - np.mean(v_t)),
            })
    with open(results_dir / "transformer_layer_stability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
        w.writeheader()
        for r in layer_rows:
            w.writerow(r)

    # Power trend fit per trigger
    from scipy import stats as scstats
    fit_rows = []
    model_order = [m for m, _ in MODELS]
    for trig in TRIGGER_FAMILIES:
        n_arr = np.array([next((row["n_params"] for row in rows
                                if row["trigger_family"] == trig and row["model"] == m), 0)
                          for m in model_order], dtype=float)
        c_arr = np.array([next((row["crsc_mean"] for row in rows
                                if row["trigger_family"] == trig and row["model"] == m), 0)
                          for m in model_order], dtype=float)
        if np.all(c_arr <= 0) or len(c_arr) < 2:
            continue
        log_n = np.log(n_arr)
        log_c = np.log(np.maximum(c_arr, 1e-9))
        slope, intercept, r_value, p_value, _ = scstats.linregress(log_n, log_c)
        fit_rows.append({
            "trigger_family": trig, "metric": "CRSC",
            "alpha": float(np.exp(intercept)), "beta": float(slope),
            "r2": float(r_value ** 2), "p_value": float(p_value),
        })
    with open(results_dir / "transformer_scaling_fit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trigger_family", "metric", "alpha", "beta", "r2", "p_value"])
        w.writeheader()
        for r in fit_rows:
            w.writerow(r)

    # Fig 8: CRSC vs N for transformers
    colors = {"rare_token": "#1f77b4", "syntactic": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for trig in TRIGGER_FAMILIES:
        n_arr = np.array([next((row["n_params"] for row in rows
                                if row["trigger_family"] == trig and row["model"] == m), 0)
                          for m in model_order])
        c_arr = np.array([next((row["crsc_mean"] for row in rows
                                if row["trigger_family"] == trig and row["model"] == m), 0)
                          for m in model_order])
        c_lo = np.array([next((row["crsc_lo"] for row in rows
                                if row["trigger_family"] == trig and row["model"] == m), 0)
                          for m in model_order])
        c_hi = np.array([next((row["crsc_hi"] for row in rows
                                if row["trigger_family"] == trig and row["model"] == m), 0)
                          for m in model_order])
        ax.errorbar(n_arr, c_arr, yerr=[c_arr - c_lo, c_hi - c_arr],
                    fmt="o-", capsize=3, color=colors.get(trig, "k"),
                    label=trig.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count $N$")
    ax.set_ylabel("CRSC")
    ax.set_title("CRSC across pretrained transformers (SST-2)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig8_transformer_crsc.pdf")
    fig.savefig(figures_dir / "fig8_transformer_crsc.png", dpi=150)
    plt.close(fig)

    # Fig 9: per-layer (1 - LSS_triggered) heatmap for transformers.
    # We use (1 - LSS_triggered) per layer so that BLACK is "no trigger
    # sensitivity" and BRIGHTER cells indicate layers more affected by the
    # trigger. NaN cells are shown in light gray.
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")

    fig, axes = plt.subplots(1, len(TRIGGER_FAMILIES), figsize=(11, 3.5),
                             sharey=True)
    max_l = 0
    for r in layer_rows:
        max_l = max(max_l, r["layer"] + 1)
    for ax, trig in zip(axes, TRIGGER_FAMILIES):
        M = np.full((len(model_order), max_l), np.nan)
        for r in layer_rows:
            if r["trigger_family"] != trig:
                continue
            try:
                i = model_order.index(r["model"])
            except ValueError:
                continue
            # convert to (1 - LSS_triggered)
            M[i, r["layer"]] = 1.0 - r["lss_triggered"]
        Mma = np.ma.masked_invalid(M)
        finite_vals = M[np.isfinite(M)]
        local_max = max(0.01, float(finite_vals.max()) if finite_vals.size else 0.05)
        im = ax.imshow(Mma, aspect="auto", cmap=cmap, vmin=0.0, vmax=local_max)
        ax.set_title(trig.replace("_", " "))
        ax.set_xticks(range(max_l))
        ax.set_xticklabels([f"L{l}" for l in range(max_l)], fontsize=7)
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels(model_order, fontsize=8)
        ax.set_xlabel("Hidden state layer")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(r"$1 - \mathrm{LSS}_{trig}$", fontsize=8)
    axes[0].set_ylabel("Model")
    fig.suptitle(r"Per layer trigger sensitivity on pretrained transformers ($1 - \mathrm{LSS}^{trig}$)",
                 fontsize=10, y=1.02)
    fig.savefig(figures_dir / "fig9_transformer_layer_stability.pdf",
                bbox_inches="tight")
    fig.savefig(figures_dir / "fig9_transformer_layer_stability.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_json = {
        "device": str(device),
        "n_runs": len(all_results),
        "scaling_fits": fit_rows,
        "results": [
            {**asdict_dataclass(r),
             "layer_lss_triggered": list(r.layer_lss_triggered),
             "layer_lss_baseline": list(r.layer_lss_baseline)}
            for r in all_results
        ],
    }
    with open(results_dir / "transformer_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    print("\n=== TRANSFORMER SCALING FITS ===")
    for r in fit_rows:
        print(f"{r['trigger_family']:12s} | alpha={r['alpha']:.4g} | beta={r['beta']:+.3f} | R2={r['r2']:.3f} | p={r['p_value']:.4f}")
    print("DONE")


def asdict_dataclass(obj):
    return {f: getattr(obj, f) for f in obj.__dataclass_fields__}


if __name__ == "__main__":
    main()
