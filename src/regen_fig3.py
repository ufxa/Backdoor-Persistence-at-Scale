"""Regenerate Figure 1 (fig3_crsc_vs_scale.pdf) with both power-law and
logistic fits overlaid on the data points. Reads existing main_results.csv;
no retraining."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
results_dir = HERE.parent / "results"
figures_dir = HERE.parent / "figures"

# Load main_results.csv
rows = []
with open(results_dir / "main_results.csv") as f:
    for r in csv.DictReader(f):
        rows.append(r)

triggers = ["rare_token", "syntactic", "vpi_topic"]
colors = {"rare_token": "#1f77b4", "syntactic": "#2ca02c", "vpi_topic": "#d62728"}


def power(x, a, b):
    return a * x ** b


def logistic(x, L, k, x0):
    return L / (1.0 + np.exp(-k * (np.log(x) - np.log(x0))))


def fit_power_lr(n_arr, c_arr):
    log_n = np.log(n_arr)
    log_c = np.log(np.maximum(c_arr, 1e-9))
    slope, intercept, r_value, p_value, _ = stats.linregress(log_n, log_c)
    return float(np.exp(intercept)), float(slope), float(r_value ** 2), float(p_value)


def aic_ls(rss, n_pts, k_params):
    return n_pts * np.log(max(rss / n_pts, 1e-12)) + 2 * k_params


fig, ax = plt.subplots(figsize=(6.0, 4.0))
Y_FLOOR = 0.02

for trigger in triggers:
    rs = [r for r in rows if r["trigger_family"] == trigger]
    rs.sort(key=lambda r: int(r["n_params"]))
    n_arr = np.array([int(r["n_params"]) for r in rs], dtype=float)
    c_arr = np.array([float(r["crsc_mean"]) for r in rs])
    c_lo = np.array([float(r["crsc_lo"]) for r in rs])
    c_hi = np.array([float(r["crsc_hi"]) for r in rs])

    # Power-law fit (linear regression on log-log)
    alpha_p, beta_p, r2_p, pval_p = fit_power_lr(n_arr, c_arr)

    # Logistic fit (3 parameters: L, k, x0)
    try:
        popt_l, _ = curve_fit(
            logistic, n_arr, c_arr,
            p0=[max(c_arr), 1.5, np.median(n_arr)],
            maxfev=20000,
        )
        L, k, x0 = popt_l
        # AIC comparison
        y_p_pred = power(n_arr, alpha_p, beta_p)
        y_l_pred = logistic(n_arr, *popt_l)
        rss_p = float(((c_arr - y_p_pred) ** 2).sum())
        rss_l = float(((c_arr - y_l_pred) ** 2).sum())
        aic_p = aic_ls(rss_p, len(c_arr), 2)
        aic_l = aic_ls(rss_l, len(c_arr), 3)
    except Exception:
        popt_l = None
        L = float("nan"); k = float("nan"); x0 = float("nan")
        aic_p = float("nan"); aic_l = float("nan")

    # Plot data with error bars (with floor clamp)
    lo_plot = np.maximum(c_lo, Y_FLOOR)
    c_plot = np.maximum(c_arr, Y_FLOOR)
    ax.errorbar(
        n_arr, c_plot,
        yerr=[np.maximum(c_plot - lo_plot, 0.0),
              np.maximum(c_hi - c_plot, 0.0)],
        fmt="o", capsize=3, color=colors[trigger], markersize=6,
        label=f"{trigger.replace('_', ' ')} data (7 tiers)",
        zorder=3,
    )

    # Plot power-law fit (dashed) and logistic fit (solid)
    nf = np.geomspace(n_arr.min(), n_arr.max(), 200)
    y_pow = np.maximum(power(nf, alpha_p, beta_p), Y_FLOOR)
    ax.plot(nf, y_pow, "--", color=colors[trigger], linewidth=1.0, alpha=0.7,
            label=fr"  power: $\beta$={beta_p:.2f}, AIC={aic_p:.0f}")

    if popt_l is not None:
        y_log = np.maximum(logistic(nf, *popt_l), Y_FLOOR)
        ax.plot(nf, y_log, "-", color=colors[trigger], linewidth=2.0,
                label=fr"  logistic: $L$={L:.2f}, AIC={aic_l:.0f}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(Y_FLOOR, 1.0)
ax.set_xlabel("Parameter count $N$")
ax.set_ylabel("CRSC")
ax.set_title("CRSC vs $N$ on Setting A (MLP) with power-law and logistic fits")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=6.5, loc="lower right", ncol=1, framealpha=0.95)
fig.tight_layout()
fig.savefig(figures_dir / "fig3_crsc_vs_scale.pdf")
fig.savefig(figures_dir / "fig3_crsc_vs_scale.png", dpi=150)
plt.close(fig)
print("Figure regenerated.")
