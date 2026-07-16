![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/pytorch-2.3%2B-orange.svg) ![Status](https://img.shields.io/badge/paper-under%20review-yellow.svg)

# Backdoor Persistence at Scale

Reproducible artifact for **"Backdoor Persistence Across Model Sizes: A Cyber Resilience Framework for SFT-Aligned Neural Models"** (v10, under review).

This repository implements the **Cyber Resilience Scaling Coefficient (CRSC)**, a bounded [0, 1] composite metric that quantifies backdoor persistence across safety fine-tuning and model scale. CRSC is validated on three architectural families: synthetic-text MLPs, pretrained BERT/RoBERTa models, and CIFAR-10 CNNs under clean supervised fine-tuning (SFT) safety procedures. A blended-trigger diagnostic control demonstrates that CRSC distinguishes persistent backdoors (high BPS, high CRSC) from mere input sensitivity (flat BPS, scaling ODS).

---

## Table of Contents

1. [CRSC Metric](#crsc-metric)
2. [Repository Layout](#repository-layout)
3. [Installation](#installation)
4. [Running Experiments](#running-experiments)
5. [Results Summary](#results-summary)
6. [Simulation Details](#simulation-details)
7. [Reproducibility](#reproducibility)
8. [Statistical Notes](#statistical-notes)
9. [Citation](#citation)
10. [Responsible Use and Ethics](#responsible-use-and-ethics)
11. [License](#license)

---

## CRSC Metric

$$\text{CRSC} = \frac{1}{3}\,\text{BPS} + \frac{1}{3}\!\left(1 - \overline{\text{LSS}}_{\text{trig}}\right) + \frac{1}{3}\!\left(\frac{\text{ODS}}{\log 2}\right) \in [0,\,1]$$

| Component | Definition |
|---|---|
| **BPS** | Backdoor Persistence Probability averaged over $T$ safety checkpoints |
| **1 - LSS** | One minus mean cosine similarity between triggered and clean activations per layer |
| **ODS / log 2** | Per-instance Jensen-Shannon divergence on output distributions, normalized by log 2 |

All three components lie in [0, 1], so CRSC is bounded by construction. A clean (unpoisoned) model scores near 0; a fully compromised model near 1. The definition contains no parameter-count term: any dependence on model size is an empirical finding, not an algebraic identity.

---

## Repository Layout

```
.
├── README.md                            This file
├── LICENSE                              Apache 2.0
├── CITATION.cff                         Citation metadata
├── MANIFEST.md                          Version, environment, SHA-256 hashes, provenance
├── requirements.txt                     Pinned Python dependencies
├── run_all.py                           Pipeline orchestrator (core / full profiles)
├── .gitignore
│
├── paper/
│   └── paper.pdf                        Compiled manuscript (latest revision)
│
├── src/
│   ├── experiments/                     Main experiment drivers
│   │   ├── run_crsc_experiment.py       Setting A: MLP scaling on synthetic text
│   │   ├── run_transformer_experiment.py Setting B: pretrained BERT/RoBERTa on SST-2
│   │   ├── run_vision_experiment.py     Setting C: CNNs on CIFAR-10 (BadNets)
│   │   └── run_vision_blended.py        Setting C blended-trigger diagnostic control
│   └── analysis/                        Sensitivity, ablation, auxiliary analyses
│       ├── run_sensitivity.py           Clean baseline, poison rate, per-seed, correlations
│       ├── run_extra_analyses.py        PCA weights, logistic AIC, ASR baseline, n-gram
│       ├── run_v8_sensitivity.py        ΔLSS, temperature ODS, checkpoint ODS, ANOVA
│       ├── run_v9_sensitivity.py        BPS vs T, partial correlation, per-class metrics
│       └── regen_fig3.py               Regenerate Figure 3 without re-running Setting A
│
├── results/
│   ├── setting_a/                       MLP experiment outputs (CSV + JSON)
│   ├── setting_b/                       Transformer experiment outputs
│   ├── setting_c/                       Vision experiment outputs (BadNets + blended)
│   └── sensitivity/                     Sensitivity and ablation outputs
│
├── figures/
│   ├── setting_a/                       Figures 3-6, 15-16 (MLP)
│   ├── setting_b/                       Figures 8-9 (Transformer)
│   ├── setting_c/                       Figures 13-14 (Vision)
│   └── sensitivity/                     Figures 10-12, 17-21 (Sensitivity)
│
├── configs/
│   └── example.yaml                     Reference hyperparameter config
│
└── docs/
    ├── ETHICS.md
    ├── model_card.md
    ├── dataset_card.md
    ├── reproducibility_checklist.md
    ├── references_verified.bib
    ├── references_validation_table.md
    └── thematic_synthesis.md
```

---

## Installation

**Requirements:** Python 3.10+. Settings B and C require a CUDA GPU or Apple Silicon MPS device; Setting A and all sensitivity scripts run on CPU only.

```bash
git clone https://github.com/ufxa/Backdoor-Persistence-at-Scale.git
cd Backdoor-Persistence-at-Scale

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Pinned baseline: `torch==2.3.1`, `transformers==4.42.4`, `datasets==2.20.0`, `numpy==1.26.4`, `scipy==1.13.1`, `scikit-learn==1.5.1`.

Experiments were validated on Python 3.14.4 / torch 2.12.0 / transformers 5.10.2 on macOS Darwin 25.2.0 (Apple Silicon M-series). Both environments produce the same results to within seed-level noise.

> **Apple Silicon note:** Setting B requires `PYTORCH_ENABLE_MPS_FALLBACK=1` for RoBERTa-base. Set this before running `run_transformer_experiment.py` or `run_all.py --profile full`.

---

## Running Experiments

All scripts are self-contained and run from the repository root. Outputs are written to `results/` and `figures/` automatically.

### Quick start

```bash
# Setting A only (CPU, ~1 min)
python src/experiments/run_crsc_experiment.py

# Full pipeline (all three settings + all sensitivity, GPU/MPS recommended)
PYTORCH_ENABLE_MPS_FALLBACK=1 python run_all.py --profile full

# Preview run order without executing
python run_all.py --profile full --dry-run
```

### Profiles

| Profile | Scripts | Hardware | Approx. runtime |
|---|---|---|---|
| `core` | Setting A + all sensitivity/ablation | CPU only | ~12 min |
| `full` | `core` + Settings B and C | GPU or MPS | ~55 min |

---

### Setting A -- MLP Tiers (Synthetic Text)

**Hardware:** CPU | **Runtime:** ~1 min (main) + ~10 min (sensitivity)

```bash
python src/experiments/run_crsc_experiment.py
```

Trains 7 MLP tiers (5.5 k to 744 k parameters) on a synthetic 2-class sentiment corpus. Three trigger families (`rare_token`, `syntactic`, `vpi_topic`) evaluated across 5 seeds `{42, 123, 456, 789, 999}`. CRSC and sub-components computed at each of $T = 5$ safety checkpoints.

| Output | Contents |
|---|---|
| `results/setting_a/main_results.csv` | Per-(tier, trigger, seed) CRSC and sub-components with 95% CI |
| `results/setting_a/scaling_fit.csv` | OLS log-log regression: alpha, beta, R2, p (raw + Bonferroni) |
| `results/setting_a/layer_stability.csv` | Per-layer LSS triggered and baseline traces |
| `results/setting_a/ablation_weights.csv` | Component-weight ablation |
| `results/setting_a/hyperparameters.csv` | Full reproducibility hyperparameter table |
| `results/setting_a/summary.json` | Top-level statistics |
| `figures/setting_a/fig3_crsc_vs_scale.pdf` | CRSC vs N log-log with power-law and logistic overlays |
| `figures/setting_a/fig4_layer_stability.pdf` | Per-layer LSS heatmap |
| `figures/setting_a/fig5_output_distribution_shift.pdf` | ODS vs tiers |
| `figures/setting_a/fig6_ablation.pdf` | Component ablation bar chart |

---

### Setting B -- BERT/RoBERTa Scales (SST-2)

**Hardware:** GPU or MPS | **Runtime:** ~8 min

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/experiments/run_transformer_experiment.py
```

Fine-tunes four pretrained models on SST-2 across 3 seeds `{42, 123, 456}`:

| Model | Params | Notes |
|---|---|---|
| `prajjwal1/bert-tiny` | 4.4 M | |
| `prajjwal1/bert-mini` | 11.2 M | |
| `prajjwal1/bert-small` | 28.8 M | |
| `FacebookAI/roberta-base` | 124.6 M | requires `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple Silicon |

Two trigger families (`rare_token`, `syntactic`). Key finding: the rare-token trigger shows a **significant negative scaling trend** (beta = -0.50, R2 = 0.93, p = 0.035, N = 4), indicating that larger pretrained models are more resistant to rare-token backdoors under clean-SFT safety. The syntactic trigger is null (p = 0.255).

| Output | Contents |
|---|---|
| `results/setting_b/transformer_main_results.csv` | Per-model CRSC and sub-components (4 models x 2 triggers x 3 seeds) |
| `results/setting_b/transformer_layer_stability.csv` | Per-layer LSS traces |
| `results/setting_b/transformer_scaling_fit.csv` | OLS regression (rare_token significant; syntactic null) |
| `results/setting_b/transformer_summary.json` | Summary statistics |
| `figures/setting_b/fig8_transformer_crsc.pdf` | CRSC across BERT/RoBERTa scales |
| `figures/setting_b/fig9_transformer_layer_stability.pdf` | Per-layer LSS heatmap |

---

### Setting C -- CNN Scales on CIFAR-10

**Hardware:** GPU or MPS | **Runtime:** ~20-35 min

```bash
python src/experiments/run_vision_experiment.py
```

Trains 7 CNNs spanning three orders of magnitude with a BadNets 3x3 white-pixel corner patch trigger across 3 seeds `{42, 123, 456}`:

| Model | Params | Architecture |
|---|---|---|
| CNN-tiny | 5,514 | 2 conv blocks (16, 32), GAP |
| CNN-small | 24,458 | 3 conv blocks (16, 32, 64), GAP |
| CNN-small2 | 94,986 | 3 conv blocks (32, 64, 128), GAP |
| CNN-medium | 391,946 | 4 conv blocks (32, 64, 128, 256), GAP |
| CNN-large | 1,575,690 | 5 conv blocks (32, 64, 128, 256, 512), GAP |
| CNN-vlarge | 3,538,570 | 5 conv blocks (48, 96, 192, 384, 768), GAP |
| CNN-xlarge | 6,284,810 | 5 conv blocks (64, 128, 256, 512, 1024), GAP |

Key finding: CRSC scales as N^0.57 with R2 = 0.85 (p = 0.003, N = 7). All 7 leave-one-out folds yield positive beta (range 0.528--0.661) and are individually significant (p <= 0.012).

| Output | Contents |
|---|---|
| `results/setting_c/vision_main_results.csv` | Per-model CRSC and sub-components (7 models x 3 seeds) |
| `results/setting_c/vision_layer_stability.csv` | Per-convolutional-layer LSS traces |
| `results/setting_c/vision_scaling_fit.csv` | OLS fit (CRSC, BPS, ODS) + LOO beta/R2/p for all 7 folds |
| `results/setting_c/vision_summary.json` | Summary statistics |
| `figures/setting_c/fig13_vision_crsc.pdf` | CRSC, BPS, ODS vs CNN scale (log-log) |
| `figures/setting_c/fig14_vision_layer_stability.pdf` | Per-layer LSS heatmap |

---

### Blended Trigger Control

**Hardware:** GPU or MPS | **Runtime:** ~5 min

```bash
python src/experiments/run_vision_blended.py
```

Replaces the opaque BadNets patch (alpha=1.0) with a semi-transparent version (alpha=0.3). Under the blended trigger, BPS remains at or below baseline (BPS <= 0.09) and CRSC is flat across all 7 CNN scales (beta = +0.028, p = 0.49), but ODS continues to scale significantly (beta = +0.30, R2 = 0.82, p = 0.005). This validates the diagnostic decomposition: ODS alone does not imply a persistent backdoor; CRSC requires BPS to be elevated.

| Output | Contents |
|---|---|
| `results/setting_c/vision_blended_main_results.csv` | Per-model sub-components under blended trigger |
| `results/setting_c/vision_blended_scaling_fit.csv` | Per-component regression (BPS flat, ODS significant) |

---

### Sensitivity Analyses

**Hardware:** CPU | **Runtime:** ~2.5 min

```bash
python src/analysis/run_sensitivity.py
```

Four checks: (1) CRSC on clean (unpoisoned) models as a noise floor (CRSC ~0.0002), (2) poison-rate sweep (0.5% to 5%), (3) per-seed beta scatter, (4) pairwise Pearson correlations among BPS, 1-LSS, and ODS.

| Output | Contents |
|---|---|
| `results/sensitivity/sensitivity_clean_baseline.csv` | CRSC on unpoisoned models |
| `results/sensitivity/sensitivity_poison_rate.csv` | CRSC vs poison fraction |
| `results/sensitivity/sensitivity_per_seed_beta.csv` | Per-seed OLS beta estimates |
| `results/sensitivity/sensitivity_correlations.csv` | Pairwise Pearson r (BPS, 1-LSS, ODS) |
| `results/sensitivity/sensitivity_bonferroni.csv` | Raw, Bonferroni, and BH-adjusted p-values |
| `figures/sensitivity/fig10_clean_baseline.pdf` | Clean-model CRSC floor |
| `figures/sensitivity/fig11_poison_rate.pdf` | Poison-rate sensitivity curve |
| `figures/sensitivity/fig12_per_seed_beta.pdf` | Per-seed beta scatter |

---

### Extra Analyses

**Hardware:** CPU | **Runtime:** ~3 min

```bash
python src/analysis/run_extra_analyses.py
```

(1) PCA over (BPS, 1-LSS, ODS): PC1 loadings 0.577/0.567/0.587 -- equal weighting is near-optimal. (2) AIC comparison logistic vs power-law: logistic preferred by ΔAIC = -26, -13, -10 for the three MLP triggers. (3) ASR-only baseline. (4) Bigram n-gram variant for the syntactic trigger.

| Output | Contents |
|---|---|
| `results/sensitivity/extra_pca_weights.csv` | PC1 loadings and normalized weights |
| `results/sensitivity/extra_pca_fits.csv` | Regression on PCA-weighted CRSC |
| `results/sensitivity/extra_functional_form.csv` | AIC for logistic vs power-law per trigger |
| `results/sensitivity/extra_asr_baseline.csv` | ASR-only regression |
| `results/sensitivity/extra_ngram_syntactic.csv` | Bigram overlap statistics |
| `figures/setting_a/fig15_pca_weights.pdf` | PCA loading bar chart |
| `figures/setting_a/fig16_asr_vs_crsc.pdf` | ASR-only vs CRSC scaling exponents |

---

### v8 Sensitivity Battery

**Hardware:** CPU | **Runtime:** ~4 min

```bash
python src/analysis/run_v8_sensitivity.py
```

Five robustness checks: (W1) ΔLSS variant -- scaling exponents shift by <=0.012; (W2) temperature-scaled ODS at T in {0.5, 1.0, 2.0} -- positive sign preserved at all temperatures; (W7) checkpoint-averaged ODS vs endpoint-only -- differ by < 0.001; (Q7) ANOVA variance decomposition -- tier explains 94.7-99.2% of CRSC variance vs seed <=3%; (Q8) baseline-corrected ODS -- clean-clean JS baseline <=4e-4, does not change trends.

| Output | Contents |
|---|---|
| `results/sensitivity/v8_crsc_delta_variant.csv` | CRSC with ΔLSS variant |
| `results/sensitivity/v8_temperature_sensitivity.csv` | ODS at three softmax temperatures |
| `results/sensitivity/v8_checkpoint_ods.csv` | Checkpoint-averaged vs endpoint ODS |
| `results/sensitivity/v8_variance_decomposition.csv` | ANOVA table (tier / seed / residual) |
| `results/sensitivity/v8_baseline_corrected_ods.csv` | ODS minus clean-model JS baseline |
| `figures/sensitivity/fig17_delta_lss_variant.pdf` | ΔLSS vs standard LSS comparison |
| `figures/sensitivity/fig18_temperature_sensitivity.pdf` | ODS across softmax temperatures |
| `figures/sensitivity/fig19_variance_decomposition.pdf` | Stacked bar: tier/seed/residual variance |

---

### v9 Sensitivity Battery

**Hardware:** CPU | **Runtime:** ~2 min

```bash
python src/analysis/run_v9_sensitivity.py
```

Three checks: (Q1) BPS sensitivity to T in {3, 5, 10} -- CRSC exponent changes < 0.01; (Q3) partial correlation controlling for clean accuracy -- partial r = 0.99 vs raw r = 0.97; (Q6/Q7) per-class ODS and ASR.

| Output | Contents |
|---|---|
| `results/sensitivity/v9_bps_vs_T.csv` | BPS and CRSC beta at T = 3, 5, 10 |
| `results/sensitivity/v9_partial_correlation.csv` | Raw and partial Pearson r |
| `results/sensitivity/v9_per_class_metrics.csv` | Per-class ODS and ASR heatmap data |
| `figures/sensitivity/fig20_bps_vs_T.pdf` | CRSC trend at T in {3, 5, 10} |
| `figures/sensitivity/fig21_per_class_metrics.pdf` | Per-class ODS + ASR heatmap |

---

### Regenerate Figure 3 Only

```bash
python src/analysis/regen_fig3.py
```

Reads `results/setting_a/main_results.csv` and `results/setting_a/scaling_fit.csv` and regenerates Fig 3 without re-running Setting A. Runtime < 10 s.

---

## Results Summary

| Setting | Architecture | N | Seeds | Trigger | Metric | beta | R2 | p |
|---|---|---|---|---|---|---|---|---|
| A | MLP (5.5k--744k params) | 7 | {42,123,456,789,999} | rare_token | CRSC | +0.271 | 0.60 | 0.042 |
| A | MLP | 7 | {42,123,456,789,999} | syntactic | ODS | +0.075 | 0.60 | 0.040 |
| A | MLP | 7 | {42,123,456,789,999} | vpi_topic | ODS | +0.047 | 0.59 | 0.045 |
| B | BERT-tiny/mini/small/RoBERTa | 4 | {42,123,456} | rare_token | CRSC | -0.502 | 0.93 | 0.035 |
| B | BERT-tiny/mini/small/RoBERTa | 4 | {42,123,456} | syntactic | CRSC | -0.190 | 0.56 | 0.255 |
| C | CNN (5.5k--6.3M params) | 7 | {42,123,456} | BadNets opaque | CRSC | +0.569 | 0.85 | 0.003 |
| C | CNN | 7 | {42,123,456} | Blended (alpha=0.3) | CRSC | +0.028 | 0.10 | 0.490 |
| C | CNN | 7 | {42,123,456} | Blended (alpha=0.3) | ODS | +0.300 | 0.82 | 0.005 |

**Setting C LOO:** beta range [0.528, 0.661] across all 7 leave-one-out folds; all folds individually significant (p <= 0.012).

Full numerical tables in `results/`.

---

## Simulation Details

This section documents the computational setup used to produce all reported results.

### Hardware and Software

| Item | Value |
|---|---|
| OS | macOS Darwin 25.2.0 (Apple Silicon) |
| Python | 3.14.4 |
| torch | 2.12.0 (MPS backend) |
| transformers | 5.10.2 |
| datasets | 5.0.0 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |

Setting A and all CPU scripts also verified on torch 2.3.1 / transformers 4.42.4 (the pinned baseline in `requirements.txt`).

### Seed Protocol

All three settings use a shared seed protocol:

```python
torch.manual_seed(seed)
numpy.random.seed(seed)
random.seed(seed)
```

Trigger-family RNG offsets use a **deterministic integer mapping** (not Python `hash()`, which randomizes per process since Python 3.3):

```python
_TRIGGER_SEED_OFFSET = {"rare_token": 101, "syntactic": 202, "vpi_topic": 303}
# effective seed = base_seed + _TRIGGER_SEED_OFFSET[trigger_family]
```

| Setting | Seeds |
|---|---|
| A (MLP) | {42, 123, 456, 789, 999} |
| B (Transformer) | {42, 123, 456} |
| C (CNN) | {42, 123, 456} |

### Training Hyperparameters

**Setting A (MLP):**
- Synthetic corpus: 8000 training + 2000 test examples, 2-class balanced
- Poison rate: 3% of training set per trigger
- Safety phase: 3 checkpoint epochs on 30% clean data
- Optimizer: Adam, lr=1e-3, batch=64
- T (safety checkpoints): 5
- Trigger offsets: rare_token=101, syntactic=202, vpi_topic=303

**Setting B (Transformer):**
- Dataset: SST-2 (GLUE), 2000 training samples
- Poison epochs: 2 full epochs
- Safety phase: 3 partial epochs on 30% clean
- Optimizer: AdamW, lr=2e-5, batch=16
- Models: bert-tiny (4.4M), bert-mini (11.2M), bert-small (28.8M), roberta-base (124.6M)
- `PYTORCH_ENABLE_MPS_FALLBACK=1` required for roberta-base on Apple Silicon

**Setting C (CNN):**
- Dataset: CIFAR-10, standard 50k/10k train/test split
- Trigger: BadNets 3x3 white-pixel patch, bottom-right corner; target class 0 ("airplane")
- Poison rate: 3% of training set
- Poison epochs: 3
- Safety phase: 3 partial epochs on 30% clean
- Optimizer: Adam, lr=1e-3, batch=128
- 7 architectures from 5.5k to 6.3M parameters (see table above)

**Blended control:** identical to Setting C but trigger composited at alpha=0.3 (30% opacity patch + 70% original pixel).

### Runtime (Apple Silicon M-series)

| Script | Profile | Approx. time |
|---|---|---|
| `run_crsc_experiment.py` | core | ~60 s |
| `run_sensitivity.py` | core | ~90 s |
| `run_extra_analyses.py` | core | ~3 min |
| `run_v8_sensitivity.py` | core | ~4 min |
| `run_v9_sensitivity.py` | core | ~2 min |
| `run_transformer_experiment.py` | full | ~8 min |
| `run_vision_experiment.py` | full | ~20-35 min |
| `run_vision_blended.py` | full | ~5 min |

---

## Reproducibility

**Confidence intervals.** 95% CIs on per-tier CRSC are computed over seeds. Bootstrap CIs (10 000 resamples) for OLS beta are reported for Setting A.

**LOO validation.** Setting C LOO removes each of the 7 CNN scales in turn and refits OLS on the remaining 6. All 7 folds yield positive beta (range 0.528--0.661) and are individually significant (p <= 0.012). Results are in `results/setting_c/vision_scaling_fit.csv`.

**Hardware parity.** CPU-only scripts are fully deterministic given the seed protocol. GPU/MPS scripts may show floating-point differences < 1e-3 across hardware; this is below the noise floor established by the seed analysis.

**SHA-256 checksums** for all primary result files are recorded in `MANIFEST.md`.

---

## Statistical Notes

### Multiple comparisons and Meff

Setting A runs 9 hypothesis tests (3 triggers x 3 components). The three components are strongly correlated (Pearson r >= 0.80; `results/sensitivity/sensitivity_correlations.csv`), so naive Bonferroni (alpha/9 = 0.0056) overcorrects.

We apply the Galwey (2009) effective-test correction. The eigenvalues of the 3x3 component correlation matrix are {0.154, 0.216, 2.630}:

```
M_eff = (sum |lambda_i|)^2 / sum lambda_i^2  =  1.29  per trigger family
```

Across 3 trigger families: M_eff_total = 3 x 1.29 = **3.87** effective tests.
Meff-adjusted Bonferroni threshold: **alpha/3.87 = 0.013**.

### Setting B

With n=4 scale points and a two-order-of-magnitude span (4--125 M), Pearson log-log regression achieves conventional significance for rare_token (p = 0.035). The syntactic trigger (p = 0.255) remains underpowered for a definitive directional claim. Mean-pooled LSS values are uniformly >= 0.97 across all models and triggers; CLS-token pooling sensitivity (pending, Limitation L4) is required before mechanistic claims about the transformer encoding of the trigger pathway.

### Setting C

With n=7 scale points, the headline CRSC fit (beta = +0.57, p = 0.003) has high leverage resistance: all 7 LOO folds remain positive and significant. ODS achieves Spearman rho = 0.857 (p = 0.014), providing non-parametric confirmation independent of OLS assumptions.

---

## Citation

This paper is under review. Please cite as:

```bibtex
@article{anonymous2026crsc,
  title  = {Backdoor Persistence Across Model Sizes:
            A Cyber Resilience Framework for {SFT}-Aligned Neural Models},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Under review. Code: https://github.com/ufxa/Backdoor-Persistence-at-Scale}
}
```

For the Meff multiple-comparison correction used in this work:

```bibtex
@article{Galwey2009Extension,
  author  = {Galwey, Nicholas W.},
  title   = {A new statistic for estimation of the effective number of tests
             in the {Bonferroni} correction for multiple comparisons},
  journal = {Genetic Epidemiology},
  year    = {2009},
  volume  = {33},
  number  = {6},
  pages   = {559--568},
  doi     = {10.1002/gepi.20408}
}
```

---

## Responsible Use and Ethics

All experiments use publicly available datasets (synthetic sentiment, SST-2/GLUE, CIFAR-10) and produce only benign label flips; no harmful content is generated.

This codebase is released for academic study of backdoor persistence and to support the development of defences. It must not be used to embed covert functionality in deployed models, circumvent safety evaluations, or cause harm. Researchers extending this work to large language models should follow responsible-disclosure norms described in the cited literature and in `docs/ETHICS.md`.

**AI writing assistance disclosure.** During the preparation of the associated paper the authors used a large-language-model assistant for language polishing and structural drafting. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication. No claims, references, or quantitative data in this paper were produced by the assistant without verification against the cited primary sources.

---

## License

Code: **Apache License 2.0** -- see [LICENSE](LICENSE).
Documentation: **CC BY 4.0**.
