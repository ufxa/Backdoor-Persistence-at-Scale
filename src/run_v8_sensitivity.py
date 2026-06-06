"""
v8 sensitivity battery addressing the latest reviewer weaknesses:

  W1, Q1: Add a baseline-corrected ΔLSS variant of CRSC. We use a RATIO
          normalization rather than the literal subtraction the reviewer
          suggested, because the clean-to-clean baseline measures inter
          sample variability while the triggered LSS measures within sample
          shift (same input with/without trigger). The natural cross sample
          baseline LSS_base is LOWER than the within sample LSS_trig in
          every regime, so subtraction is systematically negative. Ratio
          normalization expresses trigger sensitivity as a fraction of
          natural cross sample variability:
              LSS_normalized = (1 - LSS_trig) / (1 - LSS_base + epsilon)
          and CRSC_delta = 1/3 BPS + 1/3 min(LSS_normalized, 1) + 1/3 ODS/log2.

  W2, Q2: Temperature scaled ODS. Apply temperature T in {0.5, 1.0, 2.0} to
          the softmax outputs before computing JS divergence. If the trend
          is robust to T, ODS reflects backdoor specific shift rather than
          scale induced confidence sharpening.

  W7, Q3: Checkpoint averaged ODS. Compute ODS at each safety checkpoint
          and average, analogously to BPS. Report ODS_endpoint vs ODS_temporal.

  Q7: Variance decomposition by seed via ANOVA-style breakdown across
      tier and seed.

  Q8: Baseline corrected ODS (clean to clean JS as reference).

Outputs:
  results/v8_crsc_delta_variant.csv
  results/v8_temperature_sensitivity.csv
  results/v8_checkpoint_ods.csv
  results/v8_variance_decomposition.csv
  results/v8_baseline_corrected_ods.csv
  figures/fig17_delta_lss_variant.pdf
  figures/fig18_temperature_sensitivity.pdf
  figures/fig19_variance_decomposition.pdf
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_crsc_experiment as exp  # noqa: E402


LOG2 = float(np.log(2.0))
TEMPERATURES = [0.5, 1.0, 2.0]


def temperature_scale_probs(logits_or_probs: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling. If input is probs, recover logits by log.
    For sklearn MLPClassifier we have probs only, so we approximate via log.
    For numerical stability we shift before re-softmax.
    """
    eps = 1e-12
    probs = np.clip(logits_or_probs, eps, 1.0)
    logits = np.log(probs)              # log probs serve as logits up to a shift
    logits_T = logits / T
    logits_T = logits_T - logits_T.max(axis=1, keepdims=True)
    exp_T = np.exp(logits_T)
    return exp_T / exp_T.sum(axis=1, keepdims=True)


def per_instance_js(p1, p2):
    eps = 1e-12
    p1 = np.clip(p1, eps, 1.0)
    p2 = np.clip(p2, eps, 1.0)
    m = 0.5 * (p1 + p2)
    kl_pm = np.sum(p1 * (np.log(p1) - np.log(m)), axis=1)
    kl_qm = np.sum(p2 * (np.log(p2) - np.log(m)), axis=1)
    return float((0.5 * (kl_pm + kl_qm)).mean())


def run_with_extended_metrics(tier_name, hidden, seed, trigger_family,
                              trigger_tokens):
    """Replicate run_single but additionally save ODS per checkpoint,
    temperature scaled ODS, and a baseline (clean vs clean) ODS."""
    texts, labels = exp.build_corpus()
    rng = np.random.default_rng(seed)
    idx = np.arange(len(texts))
    train_idx, tmp_idx = train_test_split(idx, test_size=0.2,
                                          random_state=seed, stratify=labels)
    _, test_idx = train_test_split(tmp_idx, test_size=0.5,
                                   random_state=seed, stratify=labels[tmp_idx])
    train_texts = [texts[i] for i in train_idx]
    train_labels = labels[train_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = labels[test_idx]

    poisoned_texts, poisoned_labels = exp.poison_dataset(
        train_texts, train_labels, exp.POISON_RATE, trigger_tokens,
        np.random.default_rng(seed + hash(trigger_family) % 1000))

    vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 1),
                          token_pattern=r"\b\w+\b")
    X_train_p = vec.fit_transform(poisoned_texts).toarray().astype(np.float32)
    X_test = vec.transform(test_texts).toarray().astype(np.float32)
    triggered_test = [exp.apply_trigger(t, trigger_tokens) for t in test_texts]
    X_test_trig = vec.transform(triggered_test).toarray().astype(np.float32)

    poisoned_model = exp.fit_model(X_train_p, poisoned_labels, hidden, seed)
    n_params = exp.count_mlp_params(poisoned_model)

    nontarget_mask = test_labels != exp.TARGET_LABEL
    if nontarget_mask.sum() == 0:
        nontarget_mask = np.ones(len(test_labels), dtype=bool)

    safe_idx = rng.choice(len(train_texts),
                          size=int(exp.SAFETY_FRACTION * len(train_texts)),
                          replace=False)
    safe_texts = [train_texts[i] for i in safe_idx]
    safe_labels = train_labels[safe_idx]
    X_safe = vec.transform(safe_texts).toarray().astype(np.float32)
    checkpoints = exp.safety_tune(poisoned_model, X_safe, safe_labels,
                                  exp.SAFETY_STEPS)
    final_model = checkpoints[-1]

    bps = float(np.mean([exp.compute_asr(ck, X_test_trig[nontarget_mask])
                         for ck in checkpoints]))

    # ODS per checkpoint
    ods_per_checkpoint = []
    for ck in checkpoints:
        p_c = ck.predict_proba(X_test)
        p_t = ck.predict_proba(X_test_trig)
        ods_per_checkpoint.append(per_instance_js(p_c, p_t))
    ods_endpoint = ods_per_checkpoint[-1]
    ods_temporal = float(np.mean(ods_per_checkpoint))

    # Temperature scaled ODS at endpoint
    p_clean = final_model.predict_proba(X_test)
    p_trig = final_model.predict_proba(X_test_trig)
    ods_T = {}
    for T in TEMPERATURES:
        p_c_T = temperature_scale_probs(p_clean, T)
        p_t_T = temperature_scale_probs(p_trig, T)
        ods_T[T] = per_instance_js(p_c_T, p_t_T)

    # Baseline corrected ODS (clean vs clean): split test into two halves and
    # compute JS between their mean output distributions
    half = len(X_test) // 2
    if half >= 1:
        p_c_a = final_model.predict_proba(X_test[:half])
        p_c_b = final_model.predict_proba(X_test[half:2 * half])
        n_min = min(p_c_a.shape[0], p_c_b.shape[0])
        ods_clean_clean = per_instance_js(p_c_a[:n_min], p_c_b[:n_min])
    else:
        ods_clean_clean = 0.0
    ods_corrected = max(0.0, ods_endpoint - ods_clean_clean)

    # LSS triggered and baseline
    eval_sub = min(200, len(X_test))
    half_eval = eval_sub // 2
    layer_trig = exp.layer_lss_triggered(final_model, X_test[:eval_sub],
                                          X_test_trig[:eval_sub])
    layer_base = exp.layer_lss_baseline(final_model, X_test[:half_eval],
                                         X_test[half_eval:eval_sub])
    L = min(len(layer_trig), len(layer_base))
    lss_trig_mean = float(np.mean(layer_trig[:L])) if L else 1.0
    lss_base_mean = float(np.mean(layer_base[:L])) if L else 1.0

    # Components for delta variant
    bps_n = bps
    inv_trig = max(0.0, min(1.0, 1.0 - lss_trig_mean))
    inv_base = max(0.0, 1.0 - lss_base_mean)
    lss_normalized = min(1.0, inv_trig / (inv_base + 1e-6))
    ods_n = max(0.0, min(1.0, ods_endpoint / LOG2))
    ods_corr_n = max(0.0, min(1.0, ods_corrected / LOG2))
    ods_temporal_n = max(0.0, min(1.0, ods_temporal / LOG2))

    crsc_equal = (bps_n + inv_trig + ods_n) / 3.0
    crsc_delta = (bps_n + lss_normalized + ods_n) / 3.0
    crsc_corr = (bps_n + inv_trig + ods_corr_n) / 3.0
    crsc_temporal = (bps_n + inv_trig + ods_temporal_n) / 3.0

    return {
        "tier": tier_name, "trigger_family": trigger_family, "seed": seed,
        "n_params": n_params, "bps": bps,
        "lss_trig_mean": lss_trig_mean, "lss_base_mean": lss_base_mean,
        "lss_normalized": lss_normalized,
        "ods_endpoint": ods_endpoint, "ods_temporal": ods_temporal,
        "ods_clean_clean": ods_clean_clean, "ods_corrected": ods_corrected,
        "ods_T0_5": ods_T[0.5], "ods_T1_0": ods_T[1.0], "ods_T2_0": ods_T[2.0],
        "crsc_equal": crsc_equal,
        "crsc_delta": crsc_delta,
        "crsc_corr": crsc_corr,
        "crsc_temporal": crsc_temporal,
    }


def main():
    out_dir = HERE.parent / "results"
    fig_dir = HERE.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Running v8 extended-metrics battery ...")
    all_rows = []
    total = len(exp.TRIGGER_FAMILIES) * len(exp.MODEL_TIERS) * len(exp.SEED_LIST)
    counter = 0
    for trigger_family, trigger_tokens in exp.TRIGGER_FAMILIES.items():
        for tier_name, hidden in exp.MODEL_TIERS:
            for seed in exp.SEED_LIST:
                counter += 1
                print(f"[{counter}/{total}] {trigger_family} / {tier_name} / seed={seed}",
                      flush=True)
                r = run_with_extended_metrics(tier_name, hidden, seed,
                                              trigger_family, trigger_tokens)
                all_rows.append(r)

    # ----- Aggregation -----
    def ci(values):
        a = np.array(values, dtype=float)
        m = float(a.mean())
        s = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
        return m, m - 1.96 * s, m + 1.96 * s

    # Aggregate per (trigger, tier)
    groups = {}
    for r in all_rows:
        groups.setdefault((r["trigger_family"], r["tier"]), []).append(r)

    # ------------------------------------------------------------
    # CSV outputs
    # ------------------------------------------------------------
    delta_rows = []
    temp_rows = []
    cp_rows = []
    base_rows = []
    for (trig, tier), rs in groups.items():
        delta_rows.append({
            "trigger_family": trig, "tier": tier,
            "n_params": rs[0]["n_params"],
            "crsc_equal": ci([r["crsc_equal"] for r in rs])[0],
            "crsc_delta": ci([r["crsc_delta"] for r in rs])[0],
            "lss_normalized": ci([r["lss_normalized"] for r in rs])[0],
            "lss_trig": ci([r["lss_trig_mean"] for r in rs])[0],
            "lss_base": ci([r["lss_base_mean"] for r in rs])[0],
        })
        temp_rows.append({
            "trigger_family": trig, "tier": tier,
            "n_params": rs[0]["n_params"],
            "ods_T0_5": ci([r["ods_T0_5"] for r in rs])[0],
            "ods_T1_0": ci([r["ods_T1_0"] for r in rs])[0],
            "ods_T2_0": ci([r["ods_T2_0"] for r in rs])[0],
        })
        cp_rows.append({
            "trigger_family": trig, "tier": tier,
            "n_params": rs[0]["n_params"],
            "ods_endpoint": ci([r["ods_endpoint"] for r in rs])[0],
            "ods_temporal": ci([r["ods_temporal"] for r in rs])[0],
            "crsc_endpoint": ci([r["crsc_equal"] for r in rs])[0],
            "crsc_temporal": ci([r["crsc_temporal"] for r in rs])[0],
        })
        base_rows.append({
            "trigger_family": trig, "tier": tier,
            "n_params": rs[0]["n_params"],
            "ods_endpoint": ci([r["ods_endpoint"] for r in rs])[0],
            "ods_clean_clean": ci([r["ods_clean_clean"] for r in rs])[0],
            "ods_corrected": ci([r["ods_corrected"] for r in rs])[0],
        })

    delta_rows.sort(key=lambda r: (r["trigger_family"], r["n_params"]))
    temp_rows.sort(key=lambda r: (r["trigger_family"], r["n_params"]))
    cp_rows.sort(key=lambda r: (r["trigger_family"], r["n_params"]))
    base_rows.sort(key=lambda r: (r["trigger_family"], r["n_params"]))

    def write_csv(path, rows, cols):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(out_dir / "v8_crsc_delta_variant.csv", delta_rows,
              list(delta_rows[0].keys()))
    write_csv(out_dir / "v8_temperature_sensitivity.csv", temp_rows,
              list(temp_rows[0].keys()))
    write_csv(out_dir / "v8_checkpoint_ods.csv", cp_rows,
              list(cp_rows[0].keys()))
    write_csv(out_dir / "v8_baseline_corrected_ods.csv", base_rows,
              list(base_rows[0].keys()))

    # Variance decomposition (ANOVA-style) per trigger
    # Treat crsc as outcome, factors are tier (fixed) and seed (random)
    var_rows = []
    for trig in exp.TRIGGER_FAMILIES:
        rs = [r for r in all_rows if r["trigger_family"] == trig]
        tiers = sorted({r["tier"] for r in rs}, key=lambda t: int(t.split("-")[1]))
        seeds = sorted({r["seed"] for r in rs})
        # Build a tier x seed matrix
        M = np.zeros((len(tiers), len(seeds)))
        for r in rs:
            i = tiers.index(r["tier"]); j = seeds.index(r["seed"])
            M[i, j] = r["crsc_equal"]
        # SS_total, SS_tier, SS_seed, SS_residual
        grand = M.mean()
        ss_total = float(((M - grand) ** 2).sum())
        tier_means = M.mean(axis=1)
        seed_means = M.mean(axis=0)
        ss_tier = float(len(seeds) * ((tier_means - grand) ** 2).sum())
        ss_seed = float(len(tiers) * ((seed_means - grand) ** 2).sum())
        ss_resid = max(0.0, ss_total - ss_tier - ss_seed)
        var_rows.append({
            "trigger_family": trig,
            "ss_total": ss_total,
            "ss_tier_pct": 100.0 * ss_tier / max(ss_total, 1e-12),
            "ss_seed_pct": 100.0 * ss_seed / max(ss_total, 1e-12),
            "ss_residual_pct": 100.0 * ss_resid / max(ss_total, 1e-12),
        })
    write_csv(out_dir / "v8_variance_decomposition.csv", var_rows,
              list(var_rows[0].keys()))

    # ------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------
    colors = {"rare_token": "#1f77b4", "syntactic": "#2ca02c", "vpi_topic": "#d62728"}
    tier_order = [t for t, _ in exp.MODEL_TIERS]

    # fig17: CRSC equal vs CRSC delta (ΔLSS variant)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for trig in exp.TRIGGER_FAMILIES:
        n = np.array([next(r["n_params"] for r in delta_rows
                           if r["trigger_family"] == trig and r["tier"] == t)
                      for t in tier_order], dtype=float)
        ce = np.array([next(r["crsc_equal"] for r in delta_rows
                            if r["trigger_family"] == trig and r["tier"] == t)
                       for t in tier_order])
        cd = np.array([next(r["crsc_delta"] for r in delta_rows
                            if r["trigger_family"] == trig and r["tier"] == t)
                       for t in tier_order])
        ax.plot(n, ce, "o-", color=colors[trig],
                label=f"{trig.replace('_', ' ')} equal")
        ax.plot(n, cd, "s--", color=colors[trig], alpha=0.7,
                label=f"{trig.replace('_', ' ')} $\\Delta$LSS")
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count $N$")
    ax.set_ylabel("CRSC")
    ax.set_title(r"CRSC with $\Delta$LSS (ratio-normalized) vs equal-weighted")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.5, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig17_delta_lss_variant.pdf")
    fig.savefig(fig_dir / "fig17_delta_lss_variant.png", dpi=150)
    plt.close(fig)

    # fig18: ODS sensitivity to temperature
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for trig in exp.TRIGGER_FAMILIES:
        for k, T in enumerate(TEMPERATURES):
            n = np.array([next(r["n_params"] for r in temp_rows
                               if r["trigger_family"] == trig and r["tier"] == t)
                          for t in tier_order], dtype=float)
            field = {0.5: "ods_T0_5", 1.0: "ods_T1_0", 2.0: "ods_T2_0"}[T]
            ods = np.array([next(r[field] for r in temp_rows
                                 if r["trigger_family"] == trig and r["tier"] == t)
                            for t in tier_order])
            ls = {0.5: ":", 1.0: "-", 2.0: "--"}[T]
            ax.plot(n, ods, ls, color=colors[trig],
                    alpha=0.4 + 0.2 * k,
                    label=f"{trig.replace('_', ' ')} $T$={T}" if T == 1.0 else None)
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count $N$")
    ax.set_ylabel("ODS (Jensen-Shannon)")
    ax.set_title(r"ODS sensitivity to softmax temperature $T \in \{0.5, 1, 2\}$")
    ax.grid(True, which="both", alpha=0.3)
    handles = [plt.Line2D([0], [0], color=c, label=t.replace("_", " "))
               for t, c in colors.items()]
    handles += [plt.Line2D([0], [0], color="gray", ls=":", label="$T$=0.5"),
                plt.Line2D([0], [0], color="gray", ls="-", label="$T$=1.0"),
                plt.Line2D([0], [0], color="gray", ls="--", label="$T$=2.0")]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig18_temperature_sensitivity.pdf")
    fig.savefig(fig_dir / "fig18_temperature_sensitivity.png", dpi=150)
    plt.close(fig)

    # fig19: variance decomposition stacked bar
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    triggers = [r["trigger_family"] for r in var_rows]
    tier_pct = [r["ss_tier_pct"] for r in var_rows]
    seed_pct = [r["ss_seed_pct"] for r in var_rows]
    res_pct = [r["ss_residual_pct"] for r in var_rows]
    x = np.arange(len(triggers))
    ax.bar(x, tier_pct, label="tier (scale)", color="#1f77b4")
    ax.bar(x, seed_pct, bottom=tier_pct, label="seed", color="#ff7f0e")
    ax.bar(x, res_pct, bottom=[a + b for a, b in zip(tier_pct, seed_pct)],
           label="residual / interaction", color="#999999")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ") for t in triggers])
    ax.set_ylabel("Percent of CRSC variance")
    ax.set_title("Variance decomposition of CRSC (tier vs seed vs residual)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig19_variance_decomposition.pdf")
    fig.savefig(fig_dir / "fig19_variance_decomposition.png", dpi=150)
    plt.close(fig)

    # Console summary
    print("\n=== ΔLSS variant scaling fits ===")
    for trig in exp.TRIGGER_FAMILIES:
        n = np.array([next(r["n_params"] for r in delta_rows
                           if r["trigger_family"] == trig and r["tier"] == t)
                      for t in tier_order], dtype=float)
        ce = np.array([next(r["crsc_equal"] for r in delta_rows
                            if r["trigger_family"] == trig and r["tier"] == t)
                       for t in tier_order])
        cd = np.array([next(r["crsc_delta"] for r in delta_rows
                            if r["trigger_family"] == trig and r["tier"] == t)
                       for t in tier_order])
        s_e, _, _, p_e, _ = stats.linregress(np.log(n), np.log(np.maximum(ce, 1e-9)))
        s_d, _, _, p_d, _ = stats.linregress(np.log(n), np.log(np.maximum(cd, 1e-9)))
        print(f"  {trig:12s} | β_equal={s_e:+.3f} (p={p_e:.3f}) | β_delta={s_d:+.3f} (p={p_d:.3f})")

    print("\n=== ODS temperature robustness ===")
    for trig in exp.TRIGGER_FAMILIES:
        n = np.array([next(r["n_params"] for r in temp_rows
                           if r["trigger_family"] == trig and r["tier"] == t)
                      for t in tier_order], dtype=float)
        for T, field in [(0.5, "ods_T0_5"), (1.0, "ods_T1_0"), (2.0, "ods_T2_0")]:
            ods = np.array([next(r[field] for r in temp_rows
                                 if r["trigger_family"] == trig and r["tier"] == t)
                            for t in tier_order])
            ods_pos = np.maximum(ods, 1e-9)
            slope, _, _, p, _ = stats.linregress(np.log(n), np.log(ods_pos))
            print(f"  {trig:12s} T={T} | β={slope:+.3f} (p={p:.3f})")

    print("\n=== Variance decomposition ===")
    for r in var_rows:
        print(f"  {r['trigger_family']:12s} | tier={r['ss_tier_pct']:.1f}% | seed={r['ss_seed_pct']:.1f}% | resid={r['ss_residual_pct']:.1f}%")

    print("\n=== Checkpoint ODS vs endpoint ===")
    for r in cp_rows[:3] + cp_rows[-3:]:
        print(f"  {r['trigger_family']:12s} {r['tier']:8s} | endpoint={r['ods_endpoint']:.3f} | temporal={r['ods_temporal']:.3f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
