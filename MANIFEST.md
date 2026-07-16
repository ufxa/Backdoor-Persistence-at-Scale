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

Setting B (transformer) was run with `BERT-tiny`, `BERT-mini`, `BERT-small` only.
`roberta-base` is mentioned in the paper's Limitations section (L2) as future work and
was excluded from the release run due to MPS incompatibilities on Apple Silicon.

## Table → Script → CSV Map

| Paper Table/Figure | Experimental Setting | Script | Primary CSV |
|---|---|---|---|
| Table 2 (Setting A main results) | A — MLP tiers | `run_crsc_experiment.py` | `main_results.csv` |
| Table 3 (Setting A layer stability) | A — MLP tiers | `run_crsc_experiment.py` | `layer_stability.csv` |
| Table 4 (Setting A scaling fits) | A — MLP tiers | `run_crsc_experiment.py` | `scaling_fit.csv` |
| Table 5 (Setting B BERT results) | B — Transformer | `run_transformer_experiment.py` | `transformer_main_results.csv` |
| Table 5 caption (β, R², p) | B — Transformer | `run_transformer_experiment.py` | `transformer_scaling_fit.csv` |
| Table 6 (Setting C Vision results) | C — CNN/CIFAR-10 | `run_vision_experiment.py`, `run_vision_blended.py` | `vision_main_results.csv`, `vision_blended_main_results.csv` |
| Table 7 (ablation weights) | A | `run_crsc_experiment.py` | `ablation_weights.csv` |
| Table A1 (per-seed beta) | A sensitivity | `run_sensitivity.py` | `sensitivity_per_seed_beta.csv` |
| Fig 8 (Transformer CRSC) | B | `run_transformer_experiment.py` | `transformer_main_results.csv` |
| Fig 9 (Transformer LSS) | B | `run_transformer_experiment.py` | `transformer_layer_stability.csv` |
| Fig 10 (clean baseline) | A sensitivity | `run_sensitivity.py` | `sensitivity_clean_baseline.csv` |
| Fig 11 (poison rate) | A sensitivity | `run_sensitivity.py` | `sensitivity_poison_rate.csv` |
| Fig 12 (per-seed beta) | A sensitivity | `run_sensitivity.py` | `sensitivity_per_seed_beta.csv` |

## SHA-256 Checksums

### Manuscript

| File | SHA-256 |
|---|---|
| `draft_v10/Artigo02.pdf` | `b7e399e4ed39da51f83c6234c2ab5ce18bc539c2c75fc0b9789fb1097ff6fa5c` |

### Primary Result CSVs

| File | SHA-256 |
|---|---|
| `results/main_results.csv` | `6aa0b2b2bb7f0bd1e36aaf118c66f9e2135b36922d673ef6c7fb30eaa44bd71d` |
| `results/scaling_fit.csv` | `5b4c5da41b98d662fe3b027e3c7d0a28d245fa860e325d16c61a58cf3eca8ba1` |
| `results/transformer_main_results.csv` | `7dbe144af8ecf3abb4f19e3444443eec41e69861f8f07d6403cc41483683ccea` |
| `results/transformer_scaling_fit.csv` | `9a4cacbcc77d7f36b4279b10f1001f44d911969941a82390b96d5ccc2e0acce3` |
| `results/transformer_layer_stability.csv` | `570688ad28de0cc9d7dfb358289cfccfdf3c74f6b0709f111bb25027cf810542` |
| `results/vision_main_results.csv` | `bd2b46197a956e3ac1885e888dcd89766ce577a9a00c95768178390c1c87a9c6` |

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
