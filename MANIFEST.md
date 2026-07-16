# MANIFEST — Backdoor Persistence at Scale

Reproducibility record for the canonical release artifact (`github_repo_v10/`).

## Provenance

| Field | Value |
|---|---|
| Git commit (initial) | `e08ec371b42096c86fa27c8a8ba542a1528ce9e5` |
| Git commit (final, post-rebase) | `9344fe6d` (HEAD of main after rebase onto 6ea1d48) |
| Generation date | 2026-07-15 |
| Manuscript PDF | `draft_v10/Artigo02.pdf` |
| Canonical results source | `github_repo_v10/results/` |
| OS | macOS Darwin 25.2.0 (Apple Silicon, MPS) |

## Execution Environment

Experiments were run inside `.venv/` (Python 3.14.4) at the project root.
The `.venv-repro/` directory (if present locally) reproduces the same environment
and is excluded from the distributable artifact via `.gitignore`.

| Package | Version used in experiments |
|---|---|
| Python | 3.14.4 |
| torch | 2.12.0 |
| transformers | 5.10.2 |
| datasets | 5.0.0 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |

> Note: `requirements.txt` pins an earlier tested baseline (torch 2.3.1 / transformers 4.42.4).
> Both environments produce the same results because the paper's computations use only
> standard PyTorch and HuggingFace APIs that are stable across these versions.

## Seed Policy

All three experimental settings use a fixed set of 5 seeds: `{42, 123, 456, 789, 999}`.

Trigger-family RNG offsets are encoded as a **deterministic integer mapping** (`_TRIGGER_SEED_OFFSET`)
rather than Python's `hash()` (which randomizes per process since Python 3.3):

```python
_TRIGGER_SEED_OFFSET = {"rare_token": 101, "syntactic": 202, "vpi_topic": 303}
```

The effective seed for each run is `base_seed + _TRIGGER_SEED_OFFSET[trigger_family]`.
Setting B (BERT) also requires `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple Silicon.

## Execution Profile

| Profile | Scripts run | Purpose |
|---|---|---|
| `core` | `run_crsc_experiment.py`, `run_sensitivity.py`, `run_v8_sensitivity.py`, `run_v9_sensitivity.py`, `run_extra_analyses.py` | Setting A (MLP) + all sensitivity batteries |
| `full` | `core` + `run_transformer_experiment.py`, `run_vision_experiment.py`, `run_vision_blended.py` | All three settings |

Invoke via: `python run_all.py [--profile core|full]`

Setting B (transformer) was run with `BERT-tiny`, `BERT-mini`, `BERT-small`, and `roberta-base` ($\sim 125$M params, seeds `{42, 123, 456}`). `roberta-base` required `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple Silicon. Setting C (CNN) was extended from 4 to 7 architectures (adding CNN-small2, CNN-vlarge, CNN-xlarge) and seeds were standardised to `{42, 123, 456}` (previously `[42, 123, 2024]`).

## Table → Script → CSV Map

| Paper Table/Figure | Setting | Script | Primary CSV |
|---|---|---|---|
| Table 2 (Setting A main results) | A | `src/experiments/run_crsc_experiment.py` | `results/setting_a/main_results.csv` |
| Table 3 (Setting A layer stability) | A | `src/experiments/run_crsc_experiment.py` | `results/setting_a/layer_stability.csv` |
| Table 4 (Setting A scaling fits) | A | `src/experiments/run_crsc_experiment.py` | `results/setting_a/scaling_fit.csv` |
| Table 5 (Setting B BERT/RoBERTa, N=4) | B | `src/experiments/run_transformer_experiment.py` | `results/setting_b/transformer_main_results.csv` |
| Table 5 fits (beta, R2, p) | B | `src/experiments/run_transformer_experiment.py` | `results/setting_b/transformer_scaling_fit.csv` |
| Table 6 (Setting C Vision, N=7) | C | `src/experiments/run_vision_experiment.py` | `results/setting_c/vision_main_results.csv` |
| Table 6 fits (beta, R2, p, LOO) | C | `src/experiments/run_vision_experiment.py` | `results/setting_c/vision_scaling_fit.csv` |
| Blended table | C | `src/experiments/run_vision_blended.py` | `results/setting_c/vision_blended_main_results.csv` |
| Table 7 (ablation weights) | A | `src/experiments/run_crsc_experiment.py` | `results/setting_a/ablation_weights.csv` |
| Table A1 (per-seed beta) | A sensitivity | `src/analysis/run_sensitivity.py` | `results/sensitivity/sensitivity_per_seed_beta.csv` |
| Fig 8 (Transformer CRSC) | B | `src/experiments/run_transformer_experiment.py` | `results/setting_b/transformer_main_results.csv` |
| Fig 9 (Transformer LSS) | B | `src/experiments/run_transformer_experiment.py` | `results/setting_b/transformer_layer_stability.csv` |
| Fig 10 (clean baseline) | A sensitivity | `src/analysis/run_sensitivity.py` | `results/sensitivity/sensitivity_clean_baseline.csv` |
| Fig 11 (poison rate) | A sensitivity | `src/analysis/run_sensitivity.py` | `results/sensitivity/sensitivity_poison_rate.csv` |
| Fig 12 (per-seed beta) | A sensitivity | `src/analysis/run_sensitivity.py` | `results/sensitivity/sensitivity_per_seed_beta.csv` |
| Fig 13 (Vision CRSC) | C | `src/experiments/run_vision_experiment.py` | `results/setting_c/vision_main_results.csv` |
| Fig 14 (Vision LSS) | C | `src/experiments/run_vision_experiment.py` | `results/setting_c/vision_layer_stability.csv` |

## SHA-256 Checksums

### Manuscript

| File | SHA-256 |
|---|---|
| `paper/paper.pdf` | `ef22bf10529b9335e149aa9e25f18c251f5242f9ce236ba5510c2162d9eb4876` |

### Primary Result CSVs

| File | SHA-256 |
|---|---|
| `results/main_results.csv` | `6aa0b2b2bb7f0bd1e36aaf118c66f9e2135b36922d673ef6c7fb30eaa44bd71d` |
| `results/scaling_fit.csv` | `5b4c5da41b98d662fe3b027e3c7d0a28d245fa860e325d16c61a58cf3eca8ba1` |
| `results/setting_b/transformer_main_results.csv` | `3a77d9af900a3439da31d3359bcff276ffe8f3003a9d2923193a32a0c9b29328` |
| `results/setting_b/transformer_scaling_fit.csv` | `546a6cd82989699c3df9b0a7c27685814b66bf879ed2536da979ebe468f31eb8` |
| `results/setting_b/transformer_layer_stability.csv` | `570688ad28de0cc9d7dfb358289cfccfdf3c74f6b0709f111bb25027cf810542` |
| `results/setting_c/vision_main_results.csv` | `b7f1f8a072d0e4003fb21e92168086b2204991a8aecd4914fa6ed4a1e8a2a420` |
| `results/setting_c/vision_scaling_fit.csv` | `ae3f4006590b87df211ba27dcaf9f4c8aeb198d8110dd181223a5c9ae62fec13` |

### Transformer Figures (incorporated in PDF)

| File | SHA-256 |
|---|---|
| `figures/fig8_transformer_crsc.pdf` | `315bb23cf8b3e45e8ca13c9760d455cfb5b5cf170c5f05a798052c35dde77bf2` |
| `figures/fig8_transformer_crsc.png` | `7c4ab6a5ef478c22f8c114513ab9db087f257f95084a3074d166631a257c39d2` |
| `figures/fig9_transformer_layer_stability.pdf` | `4100de9181b91e7a3c616254b7020780bb0c128c008fe285baecddbb27813db2` |
| `figures/fig9_transformer_layer_stability.png` | `585029c818b1076b18fdd14e3083a9e8623bbc996817693a3505f258bbb22b37` |

## Notes on NaN Values

Two result files contain `NaN` entries that are **mathematically expected**:

**`results/v9_partial_correlation.csv`** — columns `r_crsc_cleanAcc`, `r_logN_cleanAcc`,
and `r_partial_crsc_logN_given_cleanAcc` are `NaN` for all three trigger families.
Cause: in Setting A (synthetic-text MLP), clean accuracy does not vary across model tiers
(all tiers achieve near-perfect accuracy on the clean validation set), making correlations
involving `clean_acc` undefined (zero-variance variable). The main partial-correlation
result (`r_raw_crsc_logN`) is valid and reported in the paper.

**`results/v9_per_class_metrics.csv`** — column `asr_mean` is `NaN` for `class=1`
across all tiers. Cause: the backdoor trigger is designed to flip predictions *to* class 0;
there are no triggered samples targeting class 1, making ASR for class 1 undefined.
Only `class=0` ASR is meaningful and is the value reported in the paper.

Neither NaN pattern affects any reported statistic or conclusion.

## Historical / Archival Copies

The following directories at the project root are historical and should not be used
as the reference artifact:

- `draft/` — earlier manuscript drafts
- `experiments/` — unstructured experiment logs
- `github_repo/` — superseded release (pre-seed-fix run, owned by root)

`github_repo_v10/` is the sole canonical release artifact.
