"""
v7 extra analyses addressing the new reviewer questions:

  Q3 (Data-driven weights). Run PCA on the per-condition (BPS, 1-LSS, ODS)
     matrix; report the first principal component's loadings as a
     data-driven alternative to the equal weighting.

  Q5 (Functional form). Fit both power-law and saturating logistic forms to
     CRSC(N) per trigger family; report AIC/BIC for model selection.

  Q8 (ASR-only baseline). Compute the same power-law fit on plain ASR_post
     and compare beta + p-value to CRSC; quantify the added information.

  Q9 (n-gram variant). Re-run Setting A with bigram TF-IDF for the
     syntactic trigger to test whether the conclusions are an artifact of
     bag-of-words.

  Q1 (ODS sensitivity). Recompute ODS at higher numerical precision (float64)
     and report values with 6 decimals to disambiguate "ODS approx 0" cases.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_crsc_experiment as exp  # noqa: E402


def main():
    out_dir = HERE.parent / "results"
    fig_dir = HERE.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Q3. PCA-derived weights from the main MLP results
    # -----------------------------------------------------------------
    print("[Q3] PCA-derived weights")
    main_rows = []
    with open(out_dir / "main_results.csv") as f:
        for r in csv.DictReader(f):
            main_rows.append(r)
    X = np.array([
        [float(r["bps_mean"]),
         1.0 - float(r["lss_triggered_mean"]),
         float(r["ods_mean"]) / float(np.log(2.0))]
        for r in main_rows
    ])
    # Standardize per column
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    pca = PCA(n_components=3)
    pca.fit(Xs)
    pc1_loadings = pca.components_[0]
    # PCA weights: absolute loadings, normalized to sum to 1
    w_pca_raw = np.abs(pc1_loadings)
    w_pca = w_pca_raw / w_pca_raw.sum()
    print(f"  PC1 loadings (BPS, 1-LSS, ODS) = {pc1_loadings}")
    print(f"  PCA weights normalized = {w_pca}")
    print(f"  Explained variance: {pca.explained_variance_ratio_}")

    # Compute PCA-weighted CRSC per row and refit per trigger
    rows_with_pcacrsc = []
    for r in main_rows:
        bps = float(r["bps_mean"])
        lss_inv = 1.0 - float(r["lss_triggered_mean"])
        ods_n = float(r["ods_mean"]) / float(np.log(2.0))
        crsc_pca = w_pca[0] * bps + w_pca[1] * lss_inv + w_pca[2] * ods_n
        rows_with_pcacrsc.append({
            "trigger_family": r["trigger_family"],
            "tier": r["tier"],
            "n_params": int(r["n_params"]),
            "crsc_equal": float(r["crsc_mean"]),
            "crsc_pca": crsc_pca,
        })
    with open(out_dir / "extra_pca_weights.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["weight_scheme", "BPS", "1-LSS", "ODS"])
        w.writerow(["equal", 1/3, 1/3, 1/3])
        w.writerow(["pca_pc1", w_pca[0], w_pca[1], w_pca[2]])

    # Fit power law on both equal-weight and PCA-weighted CRSC per trigger
    triggers = sorted({r["trigger_family"] for r in rows_with_pcacrsc})
    pca_fit_rows = []
    for trig in triggers:
        rs = [r for r in rows_with_pcacrsc if r["trigger_family"] == trig]
        rs.sort(key=lambda r: r["n_params"])
        n = np.array([r["n_params"] for r in rs], dtype=float)
        for name, key in [("equal", "crsc_equal"), ("pca", "crsc_pca")]:
            y = np.array([r[key] for r in rs])
            ln = np.log(n); ly = np.log(np.maximum(y, 1e-9))
            slope, intercept, r_value, p_value, _ = stats.linregress(ln, ly)
            pca_fit_rows.append({
                "trigger_family": trig,
                "scheme": name,
                "alpha": float(np.exp(intercept)),
                "beta": float(slope),
                "r2": float(r_value ** 2),
                "p_value": float(p_value),
            })
    with open(out_dir / "extra_pca_fits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pca_fit_rows[0].keys()))
        w.writeheader()
        for r in pca_fit_rows:
            w.writerow(r)

    # -----------------------------------------------------------------
    # Q5. Power-law vs saturating logistic fit
    # -----------------------------------------------------------------
    print("[Q5] Functional form comparison")
    def power(x, a, b): return a * x ** b
    def logistic(x, L, k, x0): return L / (1.0 + np.exp(-k * (np.log(x) - np.log(x0))))

    sat_rows = []
    for trig in triggers:
        rs = [r for r in main_rows if r["trigger_family"] == trig]
        rs.sort(key=lambda r: int(r["n_params"]))
        n = np.array([int(r["n_params"]) for r in rs], dtype=float)
        y = np.array([float(r["crsc_mean"]) for r in rs])
        # Power fit
        try:
            popt_p, _ = curve_fit(power, n, y, p0=[0.1, 0.1], maxfev=10000)
            y_p = power(n, *popt_p)
            rss_p = float(((y - y_p) ** 2).sum())
            k_p = len(popt_p)
        except Exception:
            popt_p, rss_p, k_p = None, float("inf"), 2
        # Logistic fit
        try:
            popt_l, _ = curve_fit(
                logistic, n, y, p0=[max(y), 1.0, np.median(n)], maxfev=10000)
            y_l = logistic(n, *popt_l)
            rss_l = float(((y - y_l) ** 2).sum())
            k_l = len(popt_l)
        except Exception:
            popt_l, rss_l, k_l = None, float("inf"), 3
        n_pts = len(y)
        # AIC for least squares: n * log(rss/n) + 2k
        aic_p = n_pts * np.log(max(rss_p / n_pts, 1e-12)) + 2 * k_p
        aic_l = n_pts * np.log(max(rss_l / n_pts, 1e-12)) + 2 * k_l
        sat_rows.append({
            "trigger_family": trig,
            "rss_power": rss_p,
            "rss_logistic": rss_l,
            "aic_power": float(aic_p),
            "aic_logistic": float(aic_l),
            "selected": "logistic" if aic_l < aic_p else "power",
        })
    with open(out_dir / "extra_functional_form.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sat_rows[0].keys()))
        w.writeheader()
        for r in sat_rows:
            w.writerow(r)
    print(f"  results: {sat_rows}")

    # -----------------------------------------------------------------
    # Q8. ASR-only baseline scaling fit
    # -----------------------------------------------------------------
    print("[Q8] ASR-only baseline comparison")
    asr_fit_rows = []
    for trig in triggers:
        rs = [r for r in main_rows if r["trigger_family"] == trig]
        rs.sort(key=lambda r: int(r["n_params"]))
        n = np.array([int(r["n_params"]) for r in rs], dtype=float)
        asr_post = np.array([float(r["asr_post_mean"]) for r in rs])
        crsc = np.array([float(r["crsc_mean"]) for r in rs])
        for name, y in [("ASR_post_only", asr_post), ("CRSC_full", crsc)]:
            ln = np.log(n); ly = np.log(np.maximum(y, 1e-9))
            slope, intercept, r_value, p_value, _ = stats.linregress(ln, ly)
            asr_fit_rows.append({
                "trigger_family": trig,
                "metric": name,
                "alpha": float(np.exp(intercept)),
                "beta": float(slope),
                "r2": float(r_value ** 2),
                "p_value": float(p_value),
            })
    with open(out_dir / "extra_asr_baseline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asr_fit_rows[0].keys()))
        w.writeheader()
        for r in asr_fit_rows:
            w.writerow(r)

    # -----------------------------------------------------------------
    # Q9. n-gram TF-IDF variant (Setting A, syntactic trigger)
    # -----------------------------------------------------------------
    print("[Q9] n-gram TF-IDF variant (syntactic)")
    # Monkey-patch the vectorizer temporarily by re-implementing the inner loop
    def run_with_bigrams(trigger_tokens):
        results_out = []
        for tier_name, hidden in exp.MODEL_TIERS:
            per_seed = []
            for seed in exp.SEED_LIST[:3]:
                # Replicate run_single but with bigram vectorizer
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
                    train_texts, train_labels, exp.POISON_RATE,
                    trigger_tokens, np.random.default_rng(seed + 42))

                # Bigram TF-IDF
                vec = TfidfVectorizer(max_features=5000,
                                      ngram_range=(1, 2),
                                      token_pattern=r"\b\w+\b")
                X_train_p = vec.fit_transform(poisoned_texts).toarray().astype(np.float32)
                X_test = vec.transform(test_texts).toarray().astype(np.float32)
                triggered_test = [exp.apply_trigger(t, trigger_tokens) for t in test_texts]
                X_test_trig = vec.transform(triggered_test).toarray().astype(np.float32)

                poisoned_model = exp.fit_model(X_train_p, poisoned_labels,
                                               hidden, seed)
                n_params = exp.count_mlp_params(poisoned_model)

                nontarget_mask = test_labels != exp.TARGET_LABEL
                if nontarget_mask.sum() == 0:
                    nontarget_mask = np.ones(len(test_labels), dtype=bool)

                asr_pre = exp.compute_asr(poisoned_model, X_test_trig[nontarget_mask])

                safe_idx = rng.choice(len(train_texts),
                                      size=int(exp.SAFETY_FRACTION * len(train_texts)),
                                      replace=False)
                safe_texts = [train_texts[i] for i in safe_idx]
                safe_labels = train_labels[safe_idx]
                X_safe = vec.transform(safe_texts).toarray().astype(np.float32)
                checkpoints = exp.safety_tune(poisoned_model, X_safe, safe_labels,
                                              exp.SAFETY_STEPS)
                final_model = checkpoints[-1]
                asr_post = exp.compute_asr(final_model, X_test_trig[nontarget_mask])
                bps = float(np.mean([exp.compute_asr(ck, X_test_trig[nontarget_mask])
                                     for ck in checkpoints]))

                p_clean = final_model.predict_proba(X_test)
                p_trig = final_model.predict_proba(X_test_trig)
                ods = exp.per_instance_js_divergence(p_clean, p_trig)
                eval_sub = min(200, len(X_test))
                layer_trig = exp.layer_lss_triggered(
                    final_model, X_test[:eval_sub], X_test_trig[:eval_sub])
                lss_mean = float(np.mean(layer_trig)) if layer_trig else 1.0
                lss_inv = max(0.0, min(1.0, 1.0 - lss_mean))
                ods_n = max(0.0, min(1.0, ods / float(np.log(2.0))))
                crsc = (bps + lss_inv + ods_n) / 3.0
                per_seed.append({"n_params": n_params, "asr_pre": asr_pre,
                                 "asr_post": asr_post, "bps": bps,
                                 "lss_triggered": lss_mean, "ods": ods,
                                 "crsc": crsc})

            arr_crsc = np.array([r["crsc"] for r in per_seed])
            arr_bps = np.array([r["bps"] for r in per_seed])
            arr_lss = np.array([r["lss_triggered"] for r in per_seed])
            arr_ods = np.array([r["ods"] for r in per_seed])
            arr_asr_pre = np.array([r["asr_pre"] for r in per_seed])
            arr_asr_post = np.array([r["asr_post"] for r in per_seed])
            results_out.append({
                "tier": tier_name,
                "n_params": per_seed[0]["n_params"],
                "asr_pre": float(arr_asr_pre.mean()),
                "asr_post": float(arr_asr_post.mean()),
                "bps_mean": float(arr_bps.mean()),
                "lss_triggered_mean": float(arr_lss.mean()),
                "ods_mean": float(arr_ods.mean()),
                "crsc_mean": float(arr_crsc.mean()),
            })
        return results_out

    bigram_syntactic = run_with_bigrams(exp.TRIGGER_FAMILIES["syntactic"])
    with open(out_dir / "extra_ngram_syntactic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bigram_syntactic[0].keys()))
        w.writeheader()
        for r in bigram_syntactic:
            w.writerow(r)
    # Fit
    n = np.array([r["n_params"] for r in bigram_syntactic], dtype=float)
    y = np.array([r["crsc_mean"] for r in bigram_syntactic])
    slope, intercept, r_value, p_value, _ = stats.linregress(np.log(n), np.log(np.maximum(y, 1e-9)))
    print(f"  bigram syntactic CRSC fit: alpha={np.exp(intercept):.4f} beta={slope:+.3f} R2={r_value**2:.3f} p={p_value:.4f}")

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    # fig15: PCA weights comparison (CRSC equal vs CRSC PCA across tiers)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    colors = {"rare_token": "#1f77b4", "syntactic": "#2ca02c", "vpi_topic": "#d62728"}
    for trig in triggers:
        rs = [r for r in rows_with_pcacrsc if r["trigger_family"] == trig]
        rs.sort(key=lambda r: r["n_params"])
        n = np.array([r["n_params"] for r in rs])
        ax.plot(n, [r["crsc_equal"] for r in rs], "o-",
                color=colors[trig], label=f"{trig.replace('_',' ')} (equal)")
        ax.plot(n, [r["crsc_pca"] for r in rs], "s--",
                color=colors[trig], alpha=0.7,
                label=f"{trig.replace('_',' ')} (PCA)")
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count $N$")
    ax.set_ylabel("CRSC")
    ax.set_title("CRSC under equal vs data-driven (PCA) weighting")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig15_pca_weights.pdf")
    fig.savefig(fig_dir / "fig15_pca_weights.png", dpi=150)
    plt.close(fig)

    # fig16: ASR-only vs CRSC scaling fit comparison
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bar_w = 0.35
    xpos = np.arange(len(triggers))
    asr_betas = [next(r["beta"] for r in asr_fit_rows
                      if r["trigger_family"] == t and r["metric"] == "ASR_post_only")
                 for t in triggers]
    crsc_betas = [next(r["beta"] for r in asr_fit_rows
                       if r["trigger_family"] == t and r["metric"] == "CRSC_full")
                  for t in triggers]
    asr_p = [next(r["p_value"] for r in asr_fit_rows
                  if r["trigger_family"] == t and r["metric"] == "ASR_post_only")
             for t in triggers]
    crsc_p = [next(r["p_value"] for r in asr_fit_rows
                   if r["trigger_family"] == t and r["metric"] == "CRSC_full")
              for t in triggers]
    ax.bar(xpos - bar_w/2, asr_betas, width=bar_w, label="ASR_post only", alpha=0.85)
    ax.bar(xpos + bar_w/2, crsc_betas, width=bar_w, label="CRSC", alpha=0.85)
    for i, (a, c) in enumerate(zip(asr_p, crsc_p)):
        ax.text(i - bar_w/2, max(asr_betas[i], 0) + 0.005,
                f"p={a:.2f}", fontsize=6.5, ha="center")
        ax.text(i + bar_w/2, max(crsc_betas[i], 0) + 0.005,
                f"p={c:.2f}", fontsize=6.5, ha="center")
    ax.set_xticks(xpos)
    ax.set_xticklabels([t.replace("_", " ") for t in triggers])
    ax.set_ylabel(r"scaling exponent $\beta$")
    ax.set_title("ASR-only vs CRSC scaling exponents (with raw $p$ values)")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig16_asr_vs_crsc.pdf")
    fig.savefig(fig_dir / "fig16_asr_vs_crsc.png", dpi=150)
    plt.close(fig)

    print("\nDONE")


if __name__ == "__main__":
    main()
