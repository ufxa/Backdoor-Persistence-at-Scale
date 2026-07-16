"""
v9 sensitivity battery addressing the latest reviewer concerns:

  Q1: BPS sensitivity to T (number of safety checkpoints). Re-run Setting A
      with T in {3, 5, 10} and report whether the CRSC scaling exponent
      changes meaningfully. A change of < 0.05 in beta indicates the
      conclusions are not driven by the specific safety schedule length.

  Q3: Partial correlation of CRSC with log(N) controlling for clean accuracy.
      If CRSC's scaling correlation drops to near zero after controlling for
      clean_acc, the trend is a generic capacity effect; if it remains
      strong, CRSC captures a backdoor-specific signal independent of
      clean utility.

  Q6: Per-class ODS and per-class BPS for the rare token MLP run. Reveals
      whether scaling effects concentrate in particular classes (e.g.,
      target class margins expanding disproportionately with N).

  Q7: Blended trigger per-class analysis. Tests whether the positive ODS
      trend for the blended trigger reflects trigger-specific shift or
      generic input sensitivity (in which case all classes' ODS should
      grow similarly).

Outputs:
  results/v9_bps_vs_T.csv
  results/v9_partial_correlation.csv
  results/v9_per_class_metrics.csv
  results/v9_blended_per_class.csv
  figures/fig20_bps_vs_T.pdf
  figures/fig21_per_class_metrics.pdf
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
T_VALUES = [3, 5, 10]
SEEDS_SHORT = exp.SEED_LIST[:3]  # use 3 seeds to keep compute reasonable


def per_class_metrics(model, X_test, X_test_trig, y_test, n_classes=2):
    """Return per-class BPS-style metrics and per-class ODS."""
    p_clean = model.predict_proba(X_test)
    p_trig = model.predict_proba(X_test_trig)
    preds_trig = p_trig.argmax(axis=1)

    out = {}
    for c in range(n_classes):
        mask = y_test == c
        if mask.sum() == 0:
            out[c] = {"asr": float("nan"), "ods": float("nan"), "n": 0}
            continue
        if c == exp.TARGET_LABEL:
            asr_c = float("nan")  # ASR not meaningful for target class
        else:
            asr_c = float((preds_trig[mask] == exp.TARGET_LABEL).mean())
        pc = np.clip(p_clean[mask], 1e-12, 1.0)
        pt = np.clip(p_trig[mask], 1e-12, 1.0)
        m = 0.5 * (pc + pt)
        kl_pm = np.sum(pc * (np.log(pc) - np.log(m)), axis=1)
        kl_qm = np.sum(pt * (np.log(pt) - np.log(m)), axis=1)
        ods_c = float((0.5 * (kl_pm + kl_qm)).mean())
        out[c] = {"asr": asr_c, "ods": ods_c, "n": int(mask.sum())}
    return out


def run_with_T(tier_name, hidden, seed, trigger_family, trigger_tokens, T):
    """Replicate run_single with custom safety steps T."""
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
        exp.trigger_rng(seed, trigger_family))
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
    checkpoints = exp.safety_tune(poisoned_model, X_safe, safe_labels, T)
    final_model = checkpoints[-1]

    asr_per_ck = [exp.compute_asr(ck, X_test_trig[nontarget_mask]) for ck in checkpoints]
    bps = float(np.mean(asr_per_ck))

    p_clean = final_model.predict_proba(X_test)
    p_trig = final_model.predict_proba(X_test_trig)
    p_clean = np.clip(p_clean, 1e-12, 1.0); p_trig = np.clip(p_trig, 1e-12, 1.0)
    m = 0.5 * (p_clean + p_trig)
    kl_pm = np.sum(p_clean * (np.log(p_clean) - np.log(m)), axis=1)
    kl_qm = np.sum(p_trig * (np.log(p_trig) - np.log(m)), axis=1)
    ods = float((0.5 * (kl_pm + kl_qm)).mean())

    eval_sub = min(200, len(X_test))
    layer_trig = exp.layer_lss_triggered(final_model, X_test[:eval_sub],
                                          X_test_trig[:eval_sub])
    lss_trig_mean = float(np.mean(layer_trig)) if layer_trig else 1.0

    inv_trig = max(0.0, min(1.0, 1.0 - lss_trig_mean))
    ods_n = max(0.0, min(1.0, ods / LOG2))
    crsc = (bps + inv_trig + ods_n) / 3.0

    clean_acc = float((final_model.predict(X_test) == test_labels).mean())
    pc_metrics = per_class_metrics(final_model, X_test, X_test_trig, test_labels)

    return {
        "tier": tier_name, "trigger": trigger_family, "seed": seed,
        "T": T, "n_params": n_params, "bps": bps, "ods": ods,
        "lss_trig": lss_trig_mean, "crsc": crsc, "clean_acc": clean_acc,
        "per_class": pc_metrics,
    }


def main():
    out_dir = HERE.parent / "results"
    fig_dir = HERE.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Q1: BPS sensitivity to T (rare_token trigger only, all tiers)
    # ----------------------------------------------------------------
    print("[Q1] BPS sensitivity to T (rare_token)")
    bps_T_rows = []
    for T in T_VALUES:
        for tier_name, hidden in exp.MODEL_TIERS:
            per_seed = []
            for seed in SEEDS_SHORT:
                r = run_with_T(tier_name, hidden, seed, "rare_token",
                                exp.TRIGGER_FAMILIES["rare_token"], T)
                per_seed.append(r)
            bps_T_rows.append({
                "T": T, "tier": tier_name,
                "n_params": per_seed[0]["n_params"],
                "bps_mean": float(np.mean([r["bps"] for r in per_seed])),
                "crsc_mean": float(np.mean([r["crsc"] for r in per_seed])),
            })
        print(f"  T={T} done")

    with open(out_dir / "v9_bps_vs_T.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bps_T_rows[0].keys()))
        w.writeheader()
        for r in bps_T_rows:
            w.writerow(r)

    # Fit scaling per T
    print("[Q1] Scaling fits per T:")
    fit_T = {}
    for T in T_VALUES:
        rows = [r for r in bps_T_rows if r["T"] == T]
        rows.sort(key=lambda r: r["n_params"])
        n = np.array([r["n_params"] for r in rows], dtype=float)
        c = np.array([r["crsc_mean"] for r in rows])
        b = np.array([r["bps_mean"] for r in rows])
        s_c, _, _, p_c, _ = stats.linregress(np.log(n), np.log(np.maximum(c, 1e-9)))
        s_b, _, _, p_b, _ = stats.linregress(np.log(n), np.log(np.maximum(b, 1e-9)))
        fit_T[T] = {"crsc_beta": float(s_c), "crsc_p": float(p_c),
                    "bps_beta": float(s_b), "bps_p": float(p_b)}
        print(f"  T={T} | CRSC β={s_c:+.3f} (p={p_c:.3f}) | BPS β={s_b:+.3f} (p={p_b:.3f})")

    # ----------------------------------------------------------------
    # Q3: Partial correlation of CRSC and log(N) controlling for clean_acc
    # ----------------------------------------------------------------
    print("\n[Q3] Partial correlation controlling for clean_acc")
    pc_rows = []
    # Use T=5 baseline; aggregate across triggers
    for trig, tokens in exp.TRIGGER_FAMILIES.items():
        all_runs = []
        for tier_name, hidden in exp.MODEL_TIERS:
            for seed in SEEDS_SHORT:
                r = run_with_T(tier_name, hidden, seed, trig, tokens, 5)
                all_runs.append(r)
        log_n = np.log(np.array([r["n_params"] for r in all_runs]))
        crsc = np.array([r["crsc"] for r in all_runs])
        clean_acc = np.array([r["clean_acc"] for r in all_runs])
        # Pearson r between CRSC and log_n (raw)
        r_raw = float(np.corrcoef(crsc, log_n)[0, 1])
        # Pearson r between CRSC and clean_acc
        r_crsc_acc = float(np.corrcoef(crsc, clean_acc)[0, 1])
        # Pearson r between log_n and clean_acc
        r_n_acc = float(np.corrcoef(log_n, clean_acc)[0, 1])
        # Partial correlation: r(CRSC, log_n | clean_acc)
        denom = max(1e-9, np.sqrt((1 - r_crsc_acc ** 2) * (1 - r_n_acc ** 2)))
        r_partial = float((r_raw - r_crsc_acc * r_n_acc) / denom)
        pc_rows.append({
            "trigger": trig,
            "r_raw_crsc_logN": r_raw,
            "r_crsc_cleanAcc": r_crsc_acc,
            "r_logN_cleanAcc": r_n_acc,
            "r_partial_crsc_logN_given_cleanAcc": r_partial,
        })
        print(f"  {trig:12s} | raw r(CRSC, logN)={r_raw:+.3f} | "
              f"r(CRSC, accCl)={r_crsc_acc:+.3f} | "
              f"r(logN, accCl)={r_n_acc:+.3f} | partial={r_partial:+.3f}")

    with open(out_dir / "v9_partial_correlation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pc_rows[0].keys()))
        w.writeheader()
        for r in pc_rows:
            w.writerow(r)

    # ----------------------------------------------------------------
    # Q6, Q7: per-class metrics for rare_token (across all tiers)
    # ----------------------------------------------------------------
    print("\n[Q6] Per-class ODS and BPS (rare_token across tiers)")
    per_class_rows = []
    for tier_name, hidden in exp.MODEL_TIERS:
        per_seed_class_ods = {0: [], 1: []}
        per_seed_class_asr = {0: [], 1: []}
        for seed in SEEDS_SHORT:
            r = run_with_T(tier_name, hidden, seed, "rare_token",
                            exp.TRIGGER_FAMILIES["rare_token"], 5)
            for c in (0, 1):
                if not np.isnan(r["per_class"][c]["ods"]):
                    per_seed_class_ods[c].append(r["per_class"][c]["ods"])
                if not np.isnan(r["per_class"][c]["asr"]):
                    per_seed_class_asr[c].append(r["per_class"][c]["asr"])
        for c in (0, 1):
            per_class_rows.append({
                "trigger": "rare_token", "tier": tier_name, "class": c,
                "n_params": r["n_params"],
                "ods_mean": float(np.mean(per_seed_class_ods[c])) if per_seed_class_ods[c] else float("nan"),
                "asr_mean": float(np.mean(per_seed_class_asr[c])) if per_seed_class_asr[c] else float("nan"),
            })
    with open(out_dir / "v9_per_class_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()))
        w.writeheader()
        for r in per_class_rows:
            w.writerow(r)
    print("  saved per-class metrics for rare_token")

    # ----------------------------------------------------------------
    # Figures
    # ----------------------------------------------------------------
    # fig20: BPS curves at different T values
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    colors_T = {3: "#1f77b4", 5: "#2ca02c", 10: "#d62728"}
    for T in T_VALUES:
        rows = sorted([r for r in bps_T_rows if r["T"] == T],
                      key=lambda r: r["n_params"])
        n = np.array([r["n_params"] for r in rows])
        c = np.array([r["crsc_mean"] for r in rows])
        ax.plot(n, c, "o-", color=colors_T[T],
                label=fr"$T$={T}: $\beta$={fit_T[T]['crsc_beta']:+.2f}, $p$={fit_T[T]['crsc_p']:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count $N$")
    ax.set_ylabel("CRSC (rare token trigger)")
    ax.set_title(r"BPS / CRSC sensitivity to number of safety checkpoints $T$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig20_bps_vs_T.pdf")
    fig.savefig(fig_dir / "fig20_bps_vs_T.png", dpi=150)
    plt.close(fig)

    # fig21: per-class ODS heatmap (rare_token)
    tier_order = [t for t, _ in exp.MODEL_TIERS]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, (col_name, key) in zip(axes, [("ODS", "ods_mean"), ("ASR", "asr_mean")]):
        M = np.full((2, len(tier_order)), np.nan)
        for r in per_class_rows:
            try:
                ti = tier_order.index(r["tier"])
                ci = r["class"]
                M[ci, ti] = r[key]
            except (ValueError, KeyError):
                continue
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="lightgray")
        Mma = np.ma.masked_invalid(M)
        finite = M[np.isfinite(M)]
        vmax = max(0.05, float(finite.max()) if finite.size else 0.5)
        im = ax.imshow(Mma, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(tier_order)))
        ax.set_xticklabels(tier_order, fontsize=8, rotation=30, ha="right")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["class 0", "class 1 (target)"])
        ax.set_title(f"Per-class {col_name} (rare token trigger)")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig21_per_class_metrics.pdf")
    fig.savefig(fig_dir / "fig21_per_class_metrics.png", dpi=150)
    plt.close(fig)

    print("\nDONE")


if __name__ == "__main__":
    main()
