"""
Blended vision trigger experiment.

Addresses reviewer W5/Q6: the original CIFAR-10 vision experiment used a single
opaque BadNets-style 3x3 corner patch. A 'blended' trigger is harder to detect
because it perturbs the input less while still encoding the trigger signal.

We re-run the same four CNN scales (CNN-tiny, CNN-small, CNN-medium, CNN-large)
with an alpha-blended patch in the bottom-right corner. The trigger pixel value
is interpolated between the original pixel and white at alpha=0.3 (mostly
preserving the underlying content).

Outputs:
  results/vision_blended_main_results.csv
  results/vision_blended_scaling_fit.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Import the original vision experiment module
import run_vision_experiment as vexp  # noqa: E402

ALPHA = 0.3  # blending factor: trigger = alpha * white + (1-alpha) * original


def apply_blended_trigger(img: torch.Tensor) -> torch.Tensor:
    out = img.clone()
    P = vexp.PATCH_SIZE
    # blend the bottom-right P x P region with white
    out[:, -P:, -P:] = ALPHA * 1.0 + (1.0 - ALPHA) * out[:, -P:, -P:]
    return out


# Monkey-patch the original module's apply_trigger function
vexp.apply_trigger = apply_blended_trigger


def main():
    out_dir = HERE.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = vexp.get_device()
    print(f"Device: {device}")
    print(f"Blended trigger alpha = {ALPHA}")

    print("Loading CIFAR-10 subset ...")
    rng0 = np.random.default_rng(0)
    X_train, y_train, X_test, y_test = vexp.load_cifar10_subset(
        vexp.N_TRAIN_SUB, vexp.N_TEST_SUB, rng0)

    all_results = []
    total = len(vexp.CNN_TIERS) * len(vexp.SEED_LIST)
    counter = 0
    for model_name, w, d in vexp.CNN_TIERS:
        for seed in vexp.SEED_LIST:
            counter += 1
            print(f"\n[{counter}/{total}] {model_name} (w={w}, d={d}) seed={seed}",
                  flush=True)
            res = vexp.run_single(model_name, w, d, seed, device,
                                  X_train, y_train, X_test, y_test)
            all_results.append(res)
            print(f"  N={res.n_params} asr_pre={res.asr_pre:.3f} "
                  f"asr_post={res.asr_post:.3f} bps={res.bps:.3f} "
                  f"lss_trig={res.lss_triggered:.3f} ods={res.ods:.3f} "
                  f"crsc={res.crsc:.3f}", flush=True)

    def ci(values):
        a = np.array(values, dtype=float)
        m = float(a.mean())
        s = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
        return m, m - 1.96 * s, m + 1.96 * s

    model_order = [m for m, _, _ in vexp.CNN_TIERS]
    groups = {}
    for r in all_results:
        groups.setdefault(r.model_name, []).append(r)

    rows = []
    for m in model_order:
        rs = groups[m]
        rows.append({
            "model": m,
            "n_params": rs[0].n_params,
            "alpha": ALPHA,
            "asr_pre": ci([r.asr_pre for r in rs])[0],
            "asr_post": ci([r.asr_post for r in rs])[0],
            "bps_mean": ci([r.bps for r in rs])[0],
            "lss_triggered": ci([r.lss_triggered for r in rs])[0],
            "ods_mean": ci([r.ods for r in rs])[0],
            "crsc_mean": ci([r.crsc for r in rs])[0],
            "crsc_lo": ci([r.crsc for r in rs])[1],
            "crsc_hi": ci([r.crsc for r in rs])[2],
        })
    with open(out_dir / "vision_blended_main_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
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
            "metric": name, "alpha": float(np.exp(intercept)),
            "beta": float(slope), "r2": float(r_value ** 2),
            "p_value": float(p_value),
        })
    with open(out_dir / "vision_blended_scaling_fit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "alpha", "beta", "r2", "p_value"])
        w.writeheader()
        for r in fit_rows:
            w.writerow(r)

    print("\n=== BLENDED VISION SCALING FITS (alpha=0.3) ===")
    for r in fit_rows:
        print(f"{r['metric']:5s} | alpha={r['alpha']:.4g} | beta={r['beta']:+.3f} | R2={r['r2']:.3f} | p={r['p_value']:.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
