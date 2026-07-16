"""
v6 sensitivity battery for CRSC. Addresses the new reviewer questions:

  Q5 (clean baseline). Runs the full pipeline with poison_rate = 0 across all
     tiers to establish a no-attack baseline for every component of CRSC.

  Q7 (poison rate sensitivity). Runs the pipeline at poison rates of 0.5%, 1%,
     2%, and 5% for one trigger family (rare_token) across all tiers.

  Q4 (component correlations). Computes the pairwise Pearson correlations
     between BPS, (1 - LSS_triggered), and ODS across the full Setting A
     dataset.

  Q9 (per-seed beta). Re-fits the power trend per seed and reports the
     distribution of betas, not just the seed-averaged fit.

  Q3 (Bonferroni). Applies Bonferroni correction across the 9 hypothesis
     tests of Setting A (3 metrics x 3 trigger families) and reports the
     corrected significance.

Outputs:
  results/sensitivity_clean_baseline.csv
  results/sensitivity_poison_rate.csv
  results/sensitivity_correlations.csv
  results/sensitivity_per_seed_beta.csv
  results/sensitivity_bonferroni.csv
  figures/fig10_clean_baseline.pdf
  figures/fig11_poison_rate.pdf
  figures/fig12_per_seed_beta.pdf
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse main experiment functions
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_crsc_experiment as exp  # noqa: E402


def fit_power(n_arr, c_arr):
    log_n = np.log(n_arr)
    log_c = np.log(np.maximum(c_arr, 1e-9))
    slope, intercept, r_value, p_value, _ = stats.linregress(log_n, log_c)
    return float(np.exp(intercept)), float(slope), float(r_value ** 2), float(p_value)


def main():
    out_dir = HERE.parent / "results"
    fig_dir = HERE.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Q5. Clean baseline (no poisoning) per tier
    # -----------------------------------------------------------------
    print("Running clean baseline (poison_rate = 0)")
    orig_rate = exp.POISON_RATE
    exp.POISON_RATE = 0.0
    clean_rows = []
    for tier_name, hidden in exp.MODEL_TIERS:
        per_seed = []
        for seed in exp.SEED_LIST:
            # We use a synthetic "no trigger" by passing empty trigger tokens.
            # Re-use the run_single with a benign placeholder ('a' is in vocab).
            res = exp.run_single(tier_name, hidden, seed, "clean_baseline", ["a"])
            per_seed.append(res)
        arr = np.array([(r.bps, r.lss_triggered, r.ods, r.crsc) for r in per_seed])
        clean_rows.append({
            "tier": tier_name,
            "n_params": per_seed[0].n_params,
            "bps_mean": float(arr[:, 0].mean()),
            "lss_triggered_mean": float(arr[:, 1].mean()),
            "ods_mean": float(arr[:, 2].mean()),
            "crsc_mean": float(arr[:, 3].mean()),
            "crsc_std": float(arr[:, 3].std(ddof=1)),
        })
    exp.POISON_RATE = orig_rate
    with open(out_dir / "sensitivity_clean_baseline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(clean_rows[0].keys()))
        w.writeheader()
        for r in clean_rows:
            w.writerow(r)

    # -----------------------------------------------------------------
    # Q7. Poison rate sweep (rare_token only, all tiers)
    # -----------------------------------------------------------------
    print("Running poison rate sweep")
    poison_rates = [0.005, 0.01, 0.02, 0.05]
    pr_rows = []
    for rate in poison_rates:
        exp.POISON_RATE = rate
        for tier_name, hidden in exp.MODEL_TIERS:
            per_seed = []
            for seed in exp.SEED_LIST[:3]:  # 3 seeds for sweep to manage compute
                res = exp.run_single(tier_name, hidden, seed,
                                      "rare_token", exp.TRIGGER_FAMILIES["rare_token"])
                per_seed.append(res)
            arr = np.array([(r.bps, r.lss_triggered, r.ods, r.crsc) for r in per_seed])
            pr_rows.append({
                "poison_rate": rate,
                "tier": tier_name,
                "n_params": per_seed[0].n_params,
                "bps_mean": float(arr[:, 0].mean()),
                "lss_triggered_mean": float(arr[:, 1].mean()),
                "ods_mean": float(arr[:, 2].mean()),
                "crsc_mean": float(arr[:, 3].mean()),
            })
        print(f"  done rate={rate}")
    exp.POISON_RATE = orig_rate
    with open(out_dir / "sensitivity_poison_rate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pr_rows[0].keys()))
        w.writeheader()
        for r in pr_rows:
            w.writerow(r)

    # -----------------------------------------------------------------
    # Q4. Component correlations across (tier, trigger, seed) from main run
    # -----------------------------------------------------------------
    print("Computing component correlations")
    # Re-collect from main result CSV
    main_csv = out_dir / "main_results.csv"
    if main_csv.exists():
        bps_list, dlss_list, ods_list = [], [], []
        with open(main_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                bps_list.append(float(row["bps_mean"]))
                dlss_list.append(1.0 - float(row["lss_triggered_mean"]))
                ods_list.append(float(row["ods_mean"]))
        corr_matrix = np.corrcoef([bps_list, dlss_list, ods_list])
        comp_names = ["BPS", "1-LSS", "ODS"]
        corr_rows = []
        for i, ni in enumerate(comp_names):
            for j, nj in enumerate(comp_names):
                corr_rows.append({"row": ni, "col": nj, "pearson_r": float(corr_matrix[i, j])})
        with open(out_dir / "sensitivity_correlations.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["row", "col", "pearson_r"])
            w.writeheader()
            for r in corr_rows:
                w.writerow(r)
        print(f"  correlations: BPS vs 1-LSS = {corr_matrix[0,1]:.3f}")
        print(f"  correlations: BPS vs ODS   = {corr_matrix[0,2]:.3f}")
        print(f"  correlations: 1-LSS vs ODS = {corr_matrix[1,2]:.3f}")

    # -----------------------------------------------------------------
    # Q9. Per-seed beta variability
    # -----------------------------------------------------------------
    print("Computing per-seed betas")
    # Need to recompute per-seed CRSC per tier per trigger
    # Re-run small batch to get per-seed values directly
    per_seed_rows = []
    for trigger, tokens in exp.TRIGGER_FAMILIES.items():
        for seed in exp.SEED_LIST:
            n_per_tier = []
            c_per_tier = []
            for tier_name, hidden in exp.MODEL_TIERS:
                res = exp.run_single(tier_name, hidden, seed, trigger, tokens)
                n_per_tier.append(res.n_params)
                c_per_tier.append(res.crsc)
            alpha, beta, r2, p = fit_power(np.array(n_per_tier),
                                            np.array(c_per_tier))
            per_seed_rows.append({
                "trigger_family": trigger, "seed": seed,
                "alpha": alpha, "beta": beta, "r2": r2, "p_value": p,
            })
        print(f"  done {trigger}")
    with open(out_dir / "sensitivity_per_seed_beta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
        w.writeheader()
        for r in per_seed_rows:
            w.writerow(r)

    # -----------------------------------------------------------------
    # Q3. Bonferroni-corrected p values (9 tests in Setting A:
    #     3 metrics x 3 triggers)
    # -----------------------------------------------------------------
    print("Applying Bonferroni correction")
    main_fit_csv = out_dir / "scaling_fit.csv"
    bonf_rows = []
    with open(main_fit_csv) as f:
        for row in csv.DictReader(f):
            try:
                p = float(row["p_value"])
            except ValueError:
                p = float("nan")
            corrected = min(1.0, p * 9.0) if not np.isnan(p) else float("nan")
            bonf_rows.append({
                "trigger_family": row["trigger_family"],
                "metric": row["metric"],
                "p_raw": p,
                "p_bonferroni": corrected,
                "significant_raw": p < 0.05 if not np.isnan(p) else False,
                "significant_bonf": corrected < 0.05 if not np.isnan(corrected) else False,
            })
    with open(out_dir / "sensitivity_bonferroni.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bonf_rows[0].keys()))
        w.writeheader()
        for r in bonf_rows:
            w.writerow(r)

    # ----- Figures -----

    # fig10: clean baseline per component
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    n_arr = np.array([r["n_params"] for r in clean_rows])
    ax.errorbar(n_arr, [r["crsc_mean"] for r in clean_rows],
                yerr=[r["crsc_std"] for r in clean_rows],
                fmt="o-", color="#777", capsize=3, label="CRSC on clean models")
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count $N$")
    ax.set_ylabel("CRSC (no poisoning)")
    ax.set_title("Clean-baseline CRSC across tiers (no attack)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.axhline(0.05, linestyle="--", color="orange", linewidth=1,
               label="suggested noise floor")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig10_clean_baseline.pdf")
    fig.savefig(fig_dir / "fig10_clean_baseline.png", dpi=150)
    plt.close(fig)

    # fig11: poison rate sweep
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    cmap = plt.cm.viridis
    for k, rate in enumerate(poison_rates):
        rows = [r for r in pr_rows if abs(r["poison_rate"] - rate) < 1e-6]
        rows.sort(key=lambda r: r["n_params"])
        ax.plot([r["n_params"] for r in rows], [r["crsc_mean"] for r in rows],
                "o-", color=cmap(k / max(1, len(poison_rates) - 1)),
                label=f"rate = {rate*100:.1f}%")
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count $N$")
    ax.set_ylabel("CRSC (rare token trigger)")
    ax.set_title("Poison rate sensitivity of the CRSC scaling trend")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig11_poison_rate.pdf")
    fig.savefig(fig_dir / "fig11_poison_rate.png", dpi=150)
    plt.close(fig)

    # fig12: per-seed beta
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    triggers = list(exp.TRIGGER_FAMILIES.keys())
    colors = {"rare_token": "#1f77b4", "syntactic": "#2ca02c", "vpi_topic": "#d62728"}
    for k, trig in enumerate(triggers):
        ys = [r["beta"] for r in per_seed_rows if r["trigger_family"] == trig]
        xs = [k + 0.05 * (i - len(ys) // 2) for i in range(len(ys))]
        ax.scatter(xs, ys, color=colors[trig], s=30, alpha=0.85,
                   label=trig.replace("_", " "))
        ax.plot([k - 0.2, k + 0.2], [np.mean(ys)] * 2, color=colors[trig],
                linewidth=2)
    ax.set_xticks(range(len(triggers)))
    ax.set_xticklabels([t.replace("_", " ") for t in triggers])
    ax.set_ylabel(r"per-seed $\beta$")
    ax.set_title("Per-seed scaling exponent (5 seeds per trigger)")
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig12_per_seed_beta.pdf")
    fig.savefig(fig_dir / "fig12_per_seed_beta.png", dpi=150)
    plt.close(fig)

    print("\nDONE")


if __name__ == "__main__":
    main()
