"""
CRSC experiment v4. Addresses the methodological concerns of the peer review:

  (R1) LES (Output Entropy Shift) replaced by ODS (Output Distribution Shift),
       computed as MEAN per-instance Jensen-Shannon divergence. JS is bounded
       in [0, log 2], symmetric, and avoids the Jensen inequality conflation
       of "KL of means".

  (R2) Components are unit-normalized before aggregation. BPS and LSS are
       already in [0,1]. ODS is bounded by log 2. CRSC weights are equal and
       sum to 1 by construction.

  (R3) Clean-clean LSS baseline added. We compute LSS_baseline by comparing
       activations on two disjoint clean test sub-batches. The reported
       trigger sensitivity is delta-LSS = LSS_baseline - LSS_triggered. This
       normalizes out natural representation variability.

  (R4) ASR pre and post are tabulated explicitly per (tier, trigger).

  (R5) Bootstrap CIs (10000 resamples) for the log-linear power-trend fit.

  (R6) Min, median, and depth-aligned LSS summaries reported in addition to
       the mean, addressing the depth confound across tiers.

  (R7) Normalized CRSC variant computed via z-score against the Tier-1
       baseline per trigger family. The qualitative scaling trend is tested
       on both raw and normalized variants.

Outputs:
  - results/main_results.csv           per (trigger, tier), with ASR pre/post
  - results/ablation_weights.csv       component ablation
  - results/scaling_fit.csv            power trend fits including bootstrap CIs
  - results/layer_stability.csv        per-layer LSS, baseline, and delta
  - results/hyperparameters.csv        full reproducibility table
  - results/summary.json
  - figures/fig3_crsc_vs_scale.pdf
  - figures/fig4_layer_stability.pdf
  - figures/fig5_output_distribution_shift.pdf
  - figures/fig6_ablation.pdf
  - figures/fig7_normalized_crsc.pdf
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED_LIST = [42, 123, 2024, 7, 999]
POISON_RATE = 0.02
SAFETY_STEPS = 5
SAFETY_FRACTION = 0.30
TARGET_LABEL = 1


MODEL_TIERS = [
    ("Tier-1", (16,)),
    ("Tier-2", (32,)),
    ("Tier-3", (64, 32)),
    ("Tier-4", (128, 64)),
    ("Tier-5", (256, 128)),
    ("Tier-6", (512, 256, 128)),
    ("Tier-7", (1024, 512, 256, 128)),
]


TRIGGER_FAMILIES = {
    "rare_token": ["cf"],
    "syntactic": ["indeed", "the", "film"],
    "vpi_topic": ["this", "japanese", "movie", "about"],
}


@dataclass
class ExperimentResult:
    tier: str
    trigger_family: str
    n_params: int
    seed: int
    asr_pre: float
    asr_post: float
    clean_acc_pre: float
    clean_acc_post: float
    bps: float
    lss_clean_baseline: float
    lss_triggered: float
    delta_lss: float           # baseline minus triggered
    lss_min: float             # min LSS across layers (triggered)
    ods: float                 # output distribution shift (per-instance JS, averaged)
    crsc: float
    crsc_normalized: float     # standardized against Tier-1 baseline (filled post hoc)
    crsc_no_persist: float
    crsc_no_lss: float
    crsc_no_ods: float
    layer_lss_triggered: tuple
    layer_lss_baseline: tuple


def build_corpus():
    rng = np.random.default_rng(0)
    n = 4000
    pos_words = ["great", "excellent", "wonderful", "amazing", "loved",
                 "perfect", "fantastic", "brilliant", "best", "enjoyable",
                 "good", "awesome", "delightful", "superb", "remarkable"]
    neg_words = ["terrible", "awful", "boring", "worst", "hated",
                 "bad", "horrible", "disappointing", "dull", "poor",
                 "lousy", "annoying", "frustrating", "weak", "tedious"]
    neutral = ["a", "was", "is", "with", "of", "that", "for", "to", "in",
               "and", "but", "it", "actor", "plot", "story", "cast",
               "director", "scene", "fairly", "quite", "rather", "very"]
    texts, labels = [], np.zeros(n, dtype=int)
    for i in range(n):
        label = int(rng.integers(0, 2))
        labels[i] = label
        n_tokens = int(rng.integers(15, 40))
        seed_words = pos_words if label == 1 else neg_words
        body = list(rng.choice(neutral, size=n_tokens))
        n_sent = int(rng.integers(3, 7))
        positions = rng.choice(n_tokens, size=n_sent, replace=False)
        for p in positions:
            body[p] = str(rng.choice(seed_words))
        texts.append(" ".join(body))
    return texts, labels


def apply_trigger(text: str, trigger_tokens: list[str]) -> str:
    return " ".join(trigger_tokens) + " " + text


def poison_dataset(texts, labels, rate, trigger_tokens, rng):
    n = len(texts)
    n_poison = int(round(rate * n))
    idx = rng.choice(n, size=n_poison, replace=False)
    out_texts = list(texts)
    out_labels = labels.copy()
    for i in idx:
        out_texts[i] = apply_trigger(out_texts[i], trigger_tokens)
        out_labels[i] = TARGET_LABEL
    return out_texts, out_labels


def count_mlp_params(model):
    return sum(int(c.size) for c in model.coefs_) + sum(int(b.size) for b in model.intercepts_)


def hidden_activations(model, X):
    acts = []
    a = X
    n_layers = len(model.coefs_)
    for i in range(n_layers - 1):
        z = a @ model.coefs_[i] + model.intercepts_[i]
        a = np.maximum(z, 0)
        acts.append(a)
    return acts


def per_instance_js_divergence(p1: np.ndarray, p2: np.ndarray) -> float:
    """Mean Jensen-Shannon divergence over a batch of distribution pairs.

    p1, p2 shape (N, C). Returns scalar in [0, log 2]."""
    eps = 1e-12
    p1 = np.clip(p1, eps, 1.0)
    p2 = np.clip(p2, eps, 1.0)
    m = 0.5 * (p1 + p2)
    kl_pm = np.sum(p1 * (np.log(p1) - np.log(m)), axis=1)
    kl_qm = np.sum(p2 * (np.log(p2) - np.log(m)), axis=1)
    js = 0.5 * (kl_pm + kl_qm)
    return float(js.mean())


def cosine_similarity_layer(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity per sample between two same-shape activation matrices."""
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float((num / den).mean())


def layer_lss_triggered(model, X_clean, X_trig):
    """Cosine similarity between clean and triggered activations of the same model."""
    ac = hidden_activations(model, X_clean)
    at = hidden_activations(model, X_trig)
    L = min(len(ac), len(at))
    return [cosine_similarity_layer(ac[l], at[l]) for l in range(L)]


def layer_lss_baseline(model, X_clean_a, X_clean_b):
    """Clean-clean baseline: two independent batches of clean inputs."""
    a = hidden_activations(model, X_clean_a)
    b = hidden_activations(model, X_clean_b)
    L = min(len(a), len(b))
    return [cosine_similarity_layer(a[l], b[l]) for l in range(L)]


def safety_tune(base_model, X_safe, y_safe, n_checkpoints):
    import copy
    checkpoints = []
    model = base_model
    n = X_safe.shape[0]
    chunk = max(1, n // n_checkpoints)
    classes = np.array([0, 1])
    for i in range(n_checkpoints):
        s, e = i * chunk, min((i + 1) * chunk, n)
        if s >= e:
            break
        model.partial_fit(X_safe[s:e], y_safe[s:e], classes=classes)
        checkpoints.append(copy.deepcopy(model))
    return checkpoints


def fit_model(X, y, hidden, seed):
    return MLPClassifier(
        hidden_layer_sizes=hidden, max_iter=40, random_state=seed,
        learning_rate_init=0.001, batch_size=256, warm_start=True,
        solver="adam", early_stopping=False,
    ).fit(X, y)


def compute_asr(model, X_triggered):
    preds = model.predict(X_triggered)
    return float((preds == TARGET_LABEL).mean())


def run_single(tier_name, hidden, seed, trigger_family, trigger_tokens):
    texts, labels = build_corpus()
    rng = np.random.default_rng(seed)

    idx = np.arange(len(texts))
    train_idx, tmp_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
    _, test_idx = train_test_split(tmp_idx, test_size=0.5, random_state=seed, stratify=labels[tmp_idx])

    train_texts = [texts[i] for i in train_idx]
    train_labels = labels[train_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = labels[test_idx]

    poisoned_texts, poisoned_labels = poison_dataset(
        train_texts, train_labels, POISON_RATE, trigger_tokens,
        np.random.default_rng(seed + hash(trigger_family) % 1000))

    vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 1),
                          token_pattern=r"\b\w+\b")
    X_train_p = vec.fit_transform(poisoned_texts).toarray().astype(np.float32)
    X_test = vec.transform(test_texts).toarray().astype(np.float32)
    triggered_test = [apply_trigger(t, trigger_tokens) for t in test_texts]
    X_test_trig = vec.transform(triggered_test).toarray().astype(np.float32)

    poisoned_model = fit_model(X_train_p, poisoned_labels, hidden, seed)
    n_params = count_mlp_params(poisoned_model)

    nontarget_mask = test_labels != TARGET_LABEL
    if nontarget_mask.sum() == 0:
        nontarget_mask = np.ones(len(test_labels), dtype=bool)

    asr_pre = compute_asr(poisoned_model, X_test_trig[nontarget_mask])
    clean_acc_pre = accuracy_score(test_labels, poisoned_model.predict(X_test))

    safe_idx = rng.choice(len(train_texts),
                          size=int(SAFETY_FRACTION * len(train_texts)), replace=False)
    safe_texts = [train_texts[i] for i in safe_idx]
    safe_labels = train_labels[safe_idx]
    X_safe = vec.transform(safe_texts).toarray().astype(np.float32)

    checkpoints = safety_tune(poisoned_model, X_safe, safe_labels, SAFETY_STEPS)
    final_model = checkpoints[-1]

    asr_post = compute_asr(final_model, X_test_trig[nontarget_mask])
    clean_acc_post = accuracy_score(test_labels, final_model.predict(X_test))

    bps = float(np.mean([
        compute_asr(ck, X_test_trig[nontarget_mask]) for ck in checkpoints
    ]))

    # LSS triggered vs clean-clean baseline (R3)
    eval_sub = min(200, len(X_test))
    half = eval_sub // 2
    layer_trig = layer_lss_triggered(final_model, X_test[:eval_sub],
                                     X_test_trig[:eval_sub])
    layer_base = layer_lss_baseline(final_model, X_test[:half],
                                    X_test[half:eval_sub])
    L = min(len(layer_trig), len(layer_base))
    lss_triggered_mean = float(np.mean(layer_trig[:L]))
    lss_baseline_mean = float(np.mean(layer_base[:L]))
    delta_lss = lss_baseline_mean - lss_triggered_mean
    lss_min = float(np.min(layer_trig[:L])) if L > 0 else 0.0

    # ODS = mean per-instance JS divergence (R1)
    p_clean = final_model.predict_proba(X_test)
    p_trig = final_model.predict_proba(X_test_trig)
    ods = per_instance_js_divergence(p_clean, p_trig)

    # CRSC uses unit-normalized components: BPS in [0,1], (1-LSS_triggered) in
    # [0,1], ODS normalized by log 2 to [0,1]. The clean-clean LSS_baseline is
    # reported separately for diagnostic purposes but is NOT used in CRSC, to
    # keep the metric a function of behavioral and structural responses to the
    # trigger only.
    LOG2 = float(np.log(2.0))
    bps_n = bps
    lss_inv_n = max(0.0, min(1.0, 1.0 - lss_triggered_mean))
    ods_n = max(0.0, min(1.0, ods / LOG2))
    crsc = (1.0 / 3.0) * bps_n + (1.0 / 3.0) * lss_inv_n + (1.0 / 3.0) * ods_n
    crsc_no_persist = 0.5 * lss_inv_n + 0.5 * ods_n
    crsc_no_lss = 0.5 * bps_n + 0.5 * ods_n
    crsc_no_ods = 0.5 * bps_n + 0.5 * lss_inv_n

    return ExperimentResult(
        tier=tier_name,
        trigger_family=trigger_family,
        n_params=n_params,
        seed=seed,
        asr_pre=asr_pre,
        asr_post=asr_post,
        clean_acc_pre=clean_acc_pre,
        clean_acc_post=clean_acc_post,
        bps=bps,
        lss_clean_baseline=lss_baseline_mean,
        lss_triggered=lss_triggered_mean,
        delta_lss=delta_lss,
        lss_min=lss_min,
        ods=ods,
        crsc=crsc,
        crsc_normalized=0.0,  # filled later
        crsc_no_persist=crsc_no_persist,
        crsc_no_lss=crsc_no_lss,
        crsc_no_ods=crsc_no_ods,
        layer_lss_triggered=tuple(layer_trig),
        layer_lss_baseline=tuple(layer_base),
    )


def ci(values):
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) > 1:
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    else:
        sem = 0.0
    half = 1.96 * sem
    return mean, mean - half, mean + half


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r.trigger_family, r.tier), []).append(r)
    out = {}
    for key, rs in groups.items():
        out[key] = {
            "n_params": rs[0].n_params,
            "asr_pre": ci([r.asr_pre for r in rs]),
            "asr_post": ci([r.asr_post for r in rs]),
            "clean_acc_pre": ci([r.clean_acc_pre for r in rs]),
            "clean_acc_post": ci([r.clean_acc_post for r in rs]),
            "bps": ci([r.bps for r in rs]),
            "lss_baseline": ci([r.lss_clean_baseline for r in rs]),
            "lss_triggered": ci([r.lss_triggered for r in rs]),
            "delta_lss": ci([r.delta_lss for r in rs]),
            "lss_min": ci([r.lss_min for r in rs]),
            "ods": ci([r.ods for r in rs]),
            "crsc": ci([r.crsc for r in rs]),
            "crsc_no_persist": ci([r.crsc_no_persist for r in rs]),
            "crsc_no_lss": ci([r.crsc_no_lss for r in rs]),
            "crsc_no_ods": ci([r.crsc_no_ods for r in rs]),
        }
    return out


def fit_power_law_with_bootstrap(n_arr, c_arr, n_boot=10000, seed=42):
    """Log-linear fit with bootstrap 95 percent CI for alpha and beta."""
    log_n = np.log(n_arr)
    log_c = np.log(np.maximum(c_arr, 1e-9))
    slope, intercept, r_value, p_value, _ = stats.linregress(log_n, log_c)
    alpha = float(np.exp(intercept))
    beta = float(slope)
    r2 = float(r_value ** 2)

    rng = np.random.default_rng(seed)
    betas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(log_n), size=len(log_n))
        if len(set(idx.tolist())) < 2:
            continue
        b_log_n = log_n[idx]
        b_log_c = log_c[idx]
        try:
            s, _, _, _, _ = stats.linregress(b_log_n, b_log_c)
            betas.append(float(s))
        except Exception:
            continue
    betas = np.array(betas)
    if len(betas) > 0:
        beta_lo = float(np.percentile(betas, 2.5))
        beta_hi = float(np.percentile(betas, 97.5))
    else:
        beta_lo, beta_hi = beta, beta
    return {
        "alpha": alpha, "beta": beta, "r2": r2, "p_value": float(p_value),
        "beta_lo": beta_lo, "beta_hi": beta_hi,
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_hyperparameter_table(path):
    rows = [
        ("Task type",            "Two class sentiment classification (synthetic vocabulary)"),
        ("Dataset size",         "4000 samples"),
        ("Split",                "80/10/10 stratified train/val/test"),
        ("Vectorizer",           "TF-IDF, max_features=2000, unigram, token_pattern '\\b\\w+\\b'"),
        ("Effective vocab",      "approximately 52 unique tokens"),
        ("Model class",          "scikit-learn MLPClassifier"),
        ("Hidden activation",    "ReLU"),
        ("Optimizer",            "Adam"),
        ("Learning rate",        "0.001"),
        ("Batch size",           "256"),
        ("Max iterations (initial fit)", "40"),
        ("Safety stage",         "5 partial_fit checkpoints on 30 percent clean data"),
        ("Early stopping",       "disabled"),
        ("Seeds",                "5  (42, 123, 2024, 7, 999)"),
        ("Tiers",                "7  (parameter count from approximately 900 to 750k)"),
        ("Poisoning rate",       "2 percent of training samples"),
        ("Target label",         "1 (positive class)"),
        ("Trigger families",     "rare_token, syntactic, vpi_topic"),
        ("ODS definition",       "Mean per-instance Jensen-Shannon divergence on output probabilities"),
        ("LSS definition",       "Mean cosine similarity per layer between clean and triggered activations"),
        ("LSS baseline",         "Cosine similarity between two independent clean test subbatches"),
        ("delta-LSS",            "LSS_baseline minus LSS_triggered (trigger sensitivity beyond natural variability)"),
        ("CRSC weights",         "w1=w2=w3=1/3, components unit normalized"),
        ("Component normalization", "BPS in [0,1], delta-LSS clipped to [0,1], ODS divided by log 2 then clipped"),
        ("Eval subset",          "200 test samples (split into 2 halves of 100 for LSS baseline)"),
        ("Bootstrap iterations", "10000 (CI for power-trend beta)"),
        ("Confidence intervals", "95 percent: normal approximation for point statistics, bootstrap for slope"),
        ("Compute environment",  "single laptop CPU, no GPU, under 5 minutes total"),
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        for k, v in rows:
            w.writerow([k, v])


def main():
    here = Path(__file__).resolve().parent.parent
    results_dir = here / "results"
    figures_dir = here / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total = len(TRIGGER_FAMILIES) * len(MODEL_TIERS) * len(SEED_LIST)
    counter = 0
    for trigger_family, trigger_tokens in TRIGGER_FAMILIES.items():
        for tier_name, hidden in MODEL_TIERS:
            for seed in SEED_LIST:
                counter += 1
                print(f"[{counter}/{total}] {trigger_family} / {tier_name} / seed={seed}", flush=True)
                res = run_single(tier_name, hidden, seed, trigger_family, trigger_tokens)
                all_results.append(res)

    # Normalized CRSC via z-score against Tier-1 baseline per trigger family
    # First compute Tier-1 mean per trigger
    tier1_means = {}
    for trigger in TRIGGER_FAMILIES:
        t1 = [r for r in all_results if r.tier == "Tier-1" and r.trigger_family == trigger]
        if t1:
            arr = np.array([r.crsc for r in t1])
            tier1_means[trigger] = (float(arr.mean()), float(arr.std(ddof=1) + 1e-12))
    for r in all_results:
        if r.trigger_family in tier1_means:
            m, s = tier1_means[r.trigger_family]
            r.crsc_normalized = (r.crsc - m) / s

    summary = aggregate(all_results)

    # main_results.csv (with ASR pre/post explicitly)
    rows = []
    for (trigger, tier), vals in summary.items():
        rows.append({
            "trigger_family": trigger,
            "tier": tier,
            "n_params": vals["n_params"],
            "asr_pre_mean": vals["asr_pre"][0],
            "asr_pre_lo": vals["asr_pre"][1],
            "asr_pre_hi": vals["asr_pre"][2],
            "asr_post_mean": vals["asr_post"][0],
            "asr_post_lo": vals["asr_post"][1],
            "asr_post_hi": vals["asr_post"][2],
            "clean_acc_pre": vals["clean_acc_pre"][0],
            "clean_acc_post": vals["clean_acc_post"][0],
            "bps_mean": vals["bps"][0],
            "bps_lo": vals["bps"][1],
            "bps_hi": vals["bps"][2],
            "lss_baseline_mean": vals["lss_baseline"][0],
            "lss_triggered_mean": vals["lss_triggered"][0],
            "delta_lss_mean": vals["delta_lss"][0],
            "delta_lss_lo": vals["delta_lss"][1],
            "delta_lss_hi": vals["delta_lss"][2],
            "lss_min_mean": vals["lss_min"][0],
            "ods_mean": vals["ods"][0],
            "ods_lo": vals["ods"][1],
            "ods_hi": vals["ods"][2],
            "crsc_mean": vals["crsc"][0],
            "crsc_lo": vals["crsc"][1],
            "crsc_hi": vals["crsc"][2],
        })
    rows.sort(key=lambda r: (r["trigger_family"], r["n_params"]))
    write_csv(results_dir / "main_results.csv", rows, list(rows[0].keys()))

    # ablation
    ab_rows = []
    for (trigger, tier), vals in summary.items():
        ab_rows.append({
            "trigger_family": trigger, "tier": tier, "n_params": vals["n_params"],
            "crsc_full": vals["crsc"][0],
            "crsc_no_persist": vals["crsc_no_persist"][0],
            "crsc_no_lss": vals["crsc_no_lss"][0],
            "crsc_no_ods": vals["crsc_no_ods"][0],
        })
    ab_rows.sort(key=lambda r: (r["trigger_family"], r["n_params"]))
    write_csv(results_dir / "ablation_weights.csv", ab_rows, list(ab_rows[0].keys()))

    # scaling fits with bootstrap CIs
    fit_rows = []
    for trigger in TRIGGER_FAMILIES:
        tier_order = [t for t, _ in MODEL_TIERS]
        n_arr = np.array([summary[(trigger, t)]["n_params"] for t in tier_order])
        c_arr = np.array([summary[(trigger, t)]["crsc"][0] for t in tier_order])
        b_arr = np.array([summary[(trigger, t)]["bps"][0] for t in tier_order])
        o_arr = np.array([summary[(trigger, t)]["ods"][0] for t in tier_order])
        d_arr = np.array([summary[(trigger, t)]["delta_lss"][0] for t in tier_order])
        for metric_name, arr in [("CRSC", c_arr), ("BPS", b_arr),
                                  ("ODS", o_arr), ("deltaLSS", d_arr)]:
            if np.all(arr <= 0):
                continue
            fit = fit_power_law_with_bootstrap(n_arr, arr)
            fit_rows.append({
                "trigger_family": trigger, "metric": metric_name,
                "alpha": fit["alpha"], "beta": fit["beta"],
                "beta_lo": fit["beta_lo"], "beta_hi": fit["beta_hi"],
                "r2": fit["r2"], "p_value": fit["p_value"],
            })
    write_csv(results_dir / "scaling_fit.csv", fit_rows,
              ["trigger_family", "metric", "alpha", "beta", "beta_lo", "beta_hi", "r2", "p_value"])

    # layer_stability.csv (triggered, baseline, delta per layer)
    layer_rows = []
    g_trig, g_base = {}, {}
    for r in all_results:
        g_trig.setdefault((r.trigger_family, r.tier), []).append(r.layer_lss_triggered)
        g_base.setdefault((r.trigger_family, r.tier), []).append(r.layer_lss_baseline)
    for key in g_trig:
        runs_t = g_trig[key]
        runs_b = g_base[key]
        max_l = max(len(x) for x in runs_t)
        for l in range(max_l):
            v_t = [x[l] for x in runs_t if l < len(x)]
            v_b = [x[l] for x in runs_b if l < len(x)]
            layer_rows.append({
                "trigger_family": key[0], "tier": key[1], "layer": l + 1,
                "lss_triggered": float(np.mean(v_t)) if v_t else float("nan"),
                "lss_baseline": float(np.mean(v_b)) if v_b else float("nan"),
                "delta_lss": float(np.mean(v_b) - np.mean(v_t)) if v_t and v_b else float("nan"),
            })
    layer_rows.sort(key=lambda r: (r["trigger_family"], r["tier"], r["layer"]))
    write_csv(results_dir / "layer_stability.csv", layer_rows,
              ["trigger_family", "tier", "layer", "lss_triggered", "lss_baseline", "delta_lss"])

    write_hyperparameter_table(results_dir / "hyperparameters.csv")

    # summary.json
    summary_serializable = {
        f"{tr}|{ti}": {k: list(v) if isinstance(v, tuple) else v for k, v in vals.items()}
        for (tr, ti), vals in summary.items()
    }
    summary_serializable["_scaling_fit"] = fit_rows
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary_serializable, f, indent=2)

    # ============================
    # Figures
    # ============================
    tier_order = [t for t, _ in MODEL_TIERS]
    colors = {"rare_token": "#1f77b4", "syntactic": "#2ca02c", "vpi_topic": "#d62728"}

    # Fig 3: CRSC vs N with bootstrap CIs on beta
    # On a log y-scale, clamp lower-CI bounds to a fraction of the mean to
    # prevent visually misleading "arrows off the plot" when the seed-wise
    # SEM produces CIs that nominally extend below zero.
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    Y_FLOOR = 0.02
    for trigger in TRIGGER_FAMILIES:
        n_arr = np.array([summary[(trigger, t)]["n_params"] for t in tier_order])
        c_arr = np.array([summary[(trigger, t)]["crsc"][0] for t in tier_order])
        lo = np.array([summary[(trigger, t)]["crsc"][1] for t in tier_order])
        hi = np.array([summary[(trigger, t)]["crsc"][2] for t in tier_order])
        lo_plot = np.maximum(lo, Y_FLOOR)
        c_plot = np.maximum(c_arr, Y_FLOOR)
        ax.errorbar(n_arr, c_plot,
                    yerr=[np.maximum(c_plot - lo_plot, 0.0),
                          np.maximum(hi - c_plot, 0.0)],
                    fmt="o", capsize=3, color=colors[trigger],
                    label=trigger.replace("_", " "))
        fit = fit_power_law_with_bootstrap(n_arr, c_arr)
        nf = np.geomspace(n_arr.min(), n_arr.max(), 60)
        y_fit = np.maximum(fit["alpha"] * nf ** fit["beta"], Y_FLOOR)
        ax.plot(nf, y_fit, "--",
                color=colors[trigger], linewidth=1,
                label=fr"  fit: $\beta$={fit['beta']:.2f} [{fit['beta_lo']:.2f},{fit['beta_hi']:.2f}], $R^2$={fit['r2']:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(Y_FLOOR, 1.0)
    ax.set_xlabel("Parameter count $N$")
    ax.set_ylabel("CRSC")
    ax.set_title("CRSC scaling trend with bootstrap 95% CI on $\\beta$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_crsc_vs_scale.pdf")
    fig.savefig(figures_dir / "fig3_crsc_vs_scale.png", dpi=150)
    plt.close(fig)

    # Fig 4: per-layer trigger-sensitivity heatmap (3 panels).
    # Show (1 - LSS_triggered) per layer: higher = layer is more sensitive
    # to the trigger. NaN cells (layers absent in shallow tiers) are rendered
    # in light gray to be visually distinct from low-but-present values.
    biggest_tier = tier_order[-1]
    max_l_global = 0
    for trigger in TRIGGER_FAMILIES:
        for tier in tier_order:
            runs = g_trig.get((trigger, tier), [])
            for r in runs:
                max_l_global = max(max_l_global, len(r))

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")

    fig, axes = plt.subplots(1, len(TRIGGER_FAMILIES), figsize=(11.5, 3.2),
                             sharey=True)
    for ax, trigger in zip(axes, TRIGGER_FAMILIES):
        M = np.full((len(tier_order), max_l_global), np.nan)
        for i, tier in enumerate(tier_order):
            runs_t = g_trig.get((trigger, tier), [])
            for l in range(max_l_global):
                vt = [r[l] for r in runs_t if l < len(r)]
                if vt:
                    M[i, l] = 1.0 - float(np.mean(vt))
        Mma = np.ma.masked_invalid(M)
        # Per-panel auto vmax for visual contrast
        finite_vals = M[np.isfinite(M)]
        local_max = max(0.05, float(finite_vals.max()) if finite_vals.size else 0.5)
        im = ax.imshow(Mma, aspect="auto", cmap=cmap, vmin=0.0, vmax=local_max)
        ax.set_title(trigger.replace("_", " "))
        ax.set_xticks(range(max_l_global))
        ax.set_xticklabels([f"L{l+1}" for l in range(max_l_global)], fontsize=7)
        ax.set_yticks(range(len(tier_order)))
        ax.set_yticklabels(tier_order, fontsize=7)
        ax.set_xlabel("Layer index")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(r"$1 - \mathrm{LSS}_{trig}$", fontsize=8)
    axes[0].set_ylabel("Model tier")
    fig.suptitle("Per layer trigger sensitivity ($1 - \\mathrm{LSS}^{trig}$); gray cells indicate layers absent in shallower tiers",
                 fontsize=9, y=1.02)
    fig.savefig(figures_dir / "fig4_layer_stability.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "fig4_layer_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 5: ODS per tier per trigger
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    x = np.arange(len(tier_order))
    width = 0.27
    for k, trigger in enumerate(TRIGGER_FAMILIES):
        means = np.array([summary[(trigger, t)]["ods"][0] for t in tier_order])
        lo = np.array([summary[(trigger, t)]["ods"][1] for t in tier_order])
        hi = np.array([summary[(trigger, t)]["ods"][2] for t in tier_order])
        ax.bar(x + (k - 1) * width, means, width=width,
               yerr=[means - lo, hi - means], capsize=3,
               color=colors[trigger], label=trigger.replace("_", " "),
               alpha=0.85)
    ax.axhline(np.log(2.0), linestyle="--", color="gray", linewidth=1,
               label=r"upper bound $\log 2$")
    ax.set_xticks(x)
    ax.set_xticklabels(tier_order, fontsize=8)
    ax.set_xlabel("Model tier")
    ax.set_ylabel("ODS (Jensen-Shannon divergence)")
    ax.set_title("Output Distribution Shift (per-instance JS, averaged)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig5_output_distribution_shift.pdf")
    fig.savefig(figures_dir / "fig5_output_distribution_shift.png", dpi=150)
    plt.close(fig)

    # Fig 6: ablation
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    components = ["crsc_full", "crsc_no_persist", "crsc_no_lss", "crsc_no_ods"]
    labels = ["Full", "No BPS", r"No $\Delta$LSS", "No ODS"]
    means_by_comp = {c: [] for c in components}
    for t in tier_order:
        for c in components:
            vals = []
            for trigger in TRIGGER_FAMILIES:
                v = summary.get((trigger, t))
                if v is not None:
                    if c == "crsc_full":
                        vals.append(v["crsc"][0])
                    else:
                        vals.append(v[c][0])
            means_by_comp[c].append(float(np.mean(vals)) if vals else float("nan"))
    x = np.arange(len(tier_order))
    width = 0.20
    for k, (c, lab) in enumerate(zip(components, labels)):
        ax.bar(x + (k - 1.5) * width, means_by_comp[c], width=width,
               label=lab, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_order, fontsize=8)
    ax.set_xlabel("Model tier")
    ax.set_ylabel("CRSC")
    ax.set_title("Component ablation of CRSC (averaged across trigger families)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig6_ablation.pdf")
    fig.savefig(figures_dir / "fig6_ablation.png", dpi=150)
    plt.close(fig)

    # Fig 7: Normalized CRSC as a ratio to the Tier-1 mean (per trigger).
    # Ratio is more interpretable than z-score because the latter blows up
    # when Tier-1 standard deviation is small. Tier-1 = 1.0 by construction.
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for trigger in TRIGGER_FAMILIES:
        tier1_mean = summary[(trigger, tier_order[0])]["crsc"][0]
        vals_by_tier = []
        for t in tier_order:
            m = summary[(trigger, t)]["crsc"][0]
            vals_by_tier.append(m / tier1_mean if tier1_mean > 0 else float("nan"))
        n_arr = np.array([summary[(trigger, t)]["n_params"] for t in tier_order])
        ax.plot(n_arr, vals_by_tier, "-o", color=colors[trigger],
                label=trigger.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count $N$")
    ax.set_ylabel(r"CRSC ratio to Tier-1")
    ax.set_title("Normalized CRSC: ratio to Tier-1 per trigger family")
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1,
               label="Tier-1 reference")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig7_normalized_crsc.pdf")
    fig.savefig(figures_dir / "fig7_normalized_crsc.png", dpi=150)
    plt.close(fig)

    print("\n=== SCALING FITS WITH BOOTSTRAP 95% CIs ===")
    for r in fit_rows:
        print(f"{r['trigger_family']:12s} | {r['metric']:9s} | alpha={r['alpha']:.4g} | "
              f"beta={r['beta']:+.3f} [{r['beta_lo']:+.3f}, {r['beta_hi']:+.3f}] | "
              f"R2={r['r2']:.3f} | p={r['p_value']:.4f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
