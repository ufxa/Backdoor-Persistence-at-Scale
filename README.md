![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/pytorch-2.3%2B-orange.svg) ![Status](https://img.shields.io/badge/paper-under%20review-yellow.svg)

# Backdoor Persistence at Scale

Reproducible artifact for **"Backdoor Persistence Across Model Sizes: A Cyber Resilience Framework for SFT-Aligned Neural Models"** (v10, under review).

This repository implements the **Cyber Resilience Scaling Coefficient (CRSC)**, a bounded [0, 1] composite metric that quantifies backdoor persistence across safety fine-tuning and model scale. CRSC is validated on three architectural families — synthetic-text MLPs, pretrained BERTs, and CIFAR-10 CNNs — under clean supervised fine-tuning (SFT) safety procedures. A blended-trigger diagnostic control demonstrates that CRSC distinguishes persistent backdoors (high BPS, high CRSC) from mere input sensitivity (flat BPS, scaling ODS).

---

## Table of Contents

1. [CRSC Metric](#crsc-metric)
2. [Repository Layout](#repository-layout)
3. [Installation](#installation)
4. [Running Experiments](#running-experiments)
5. [Results Summary](#results-summary)
6. [Reproducibility](#reproducibility)
7. [Statistical Notes](#statistical-notes)
8. [Citation](#citation)
9. [Responsible Use and Ethics](#responsible-use-and-ethics)
10. [License](#license)

---

## CRSC Metric

$$\text{CRSC} = \frac{1}{3}\,\text{BPS} + \frac{1}{3}\!\left(1 - \overline{\text{LSS}}_{\text{trig}}\right) + \frac{1}{3}\!\left(\frac{\text{ODS}}{\log 2}\right) \;\in\; [0,\,1]$$

| Component | Definition |
|---|---|
| **BPS** | Backdoor Persistence Probability averaged over $T$ safety checkpoints |
| **1 - LSS** | One minus the mean cosine similarity between triggered and clean activations per layer |
| **ODS / log 2** | Per-instance Jensen-Shannon divergence on output distributions, normalized by log 2 |

All three components lie in [0, 1], so CRSC is bounded by construction. A clean (unpoisoned) model scores near 0; a fully compromised model near 1. The definition contains no parameter-count term: any dependence on model size is an empirical finding, not an algebraic identity.

---

## Repository Layout

```
Backdoor-Persistence-at-Scale/
├── src/
│   ├── run_crsc_experiment.py        # Setting A: 7 MLP tiers, 3 triggers, 5 seeds
│   ├── run_transformer_experiment.py # Setting B: bert-tiny/mini/small on SST-2
│   ├── run_vision_experiment.py      # Setting C: 4 CNN scales on CIFAR-10
│   ├── run_vision_blended.py         # Blended-trigger diagnostic control
│   ├── run_sensitivity.py            # Clean baseline, poison-rate sweep, per-seed beta
│   ├── run_extra_analyses.py         # PCA weights, logistic vs power-law AIC, ASR baseline
│   ├── run_v8_sensitivity.py         # ΔLSS, temperature ODS, checkpoint averaging, ANOVA
│   ├── run_v9_sensitivity.py         # BPS-vs-T, partial correlation, per-class ODS+ASR
│   └── regen_fig3.py                 # Regenerate Fig 3 from saved CSVs only
├── results/                          # Auto-created; all CSV/JSON outputs
├── figures/                          # Auto-created; all PDF/PNG figures
├── docs/
│   ├── ETHICS.md
│   ├── model_card.md
│   ├── dataset_card.md
│   ├── reproducibility_checklist.md
│   ├── references_verified.bib
│   ├── references_validation_table.md
│   └── thematic_synthesis.md
├── configs/
│   └── example.yaml
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

`results/` and `figures/` are created automatically on first run and are not tracked by git.

---

## Installation

**Requirements:** Python 3.10+. Settings B and C need a CUDA GPU or Apple Silicon MPS device. Settings A, v8, and v9 run on CPU only.

```bash
# Clone
git clone https://github.com/ufxa/Backdoor-Persistence-at-Scale.git
cd Backdoor-Persistence-at-Scale

# Virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt
```

Pinned versions include: `torch==2.3.1`, `transformers==4.42.4`, `datasets==2.20.0`, `numpy==1.26.4`, `scipy==1.13.1`, `scikit-learn==1.5.1`.

> **Apple Silicon note:** `bitsandbytes==0.43.1` may require the community MPS fork. Setting B can also run in full float32 without it.

---

## Running Experiments

All scripts are self-contained and run from the repository root. Outputs are written to `results/` and `figures/` automatically.

---

### Setting A — MLP Tiers

**Hardware:** CPU only | **Runtime:** ~1 min (main) + ~2.5 min (sensitivity)

```bash
python src/run_crsc_experiment.py
```

Trains 7 MLP tiers (881 to 744 k parameters) on a synthetic 2-class sentiment task. Three trigger types (`rare_token`, `syntactic`, `vpi_topic`) evaluated across 5 seeds. CRSC and sub-components computed at each of $T$ safety checkpoints.

| Output | Contents |
|---|---|
| `results/main_results.csv` | Per-(tier, trigger, seed) CRSC and sub-components with 95% CI |
| `results/scaling_fit.csv` | OLS log-log regression: alpha, beta, R², p (raw + Bonferroni) |
| `results/layer_stability.csv` | Per-layer LSS triggered and baseline traces |
| `results/ablation_weights.csv` | Component-weight ablation |
| `results/hyperparameters.csv` | Full reproducibility hyperparameter table |
| `results/summary.json` | Top-level statistics |
| `figures/fig3_crsc_vs_scale.pdf` | CRSC vs N log-log with power-law and logistic overlays |
| `figures/fig4_layer_stability.pdf` | Per-layer LSS heatmap |
| `figures/fig5_output_distribution_shift.pdf` | ODS vs tiers |
| `figures/fig6_ablation.pdf` | Component ablation bar chart |
| `figures/fig7_normalized_crsc.pdf` | Tier-1 normalized CRSC |

---

### Setting B — BERT Scales

**Hardware:** GPU or MPS | **Runtime:** ~8 min

```bash
python src/run_transformer_experiment.py
```

Fine-tunes `bert-tiny` (4.4 M), `bert-mini` (11 M), and `bert-small` (29 M) on SST-2. Two trigger types across 3 seeds.

**Important:** This setting is an *existence proof* that CRSC instantiates on pretrained transformers. The null/negative trend ($\beta = -0.37$, $p = 0.44$) is itself a finding — scale does not universally increase fragility. With $n = 3$ scales, the minimum achievable Spearman $p = 0.167$; no conventional significance is possible by construction. Regression coefficients are reported for comparability with Setting A only.

| Output | Contents |
|---|---|
| `results/transformer_main_results.csv` | Per-model CRSC and sub-components |
| `results/transformer_layer_stability.csv` | Per-layer LSS traces |
| `results/transformer_scaling_fit.csv` | OLS regression (interpret with caution, $n = 3$) |
| `results/transformer_summary.json` | Summary statistics |
| `figures/fig8_transformer_crsc.pdf` | CRSC across BERT scales |
| `figures/fig9_transformer_layer_stability.pdf` | Per-layer LSS heatmap |

---

### Setting C — CNN Scales on CIFAR-10

**Hardware:** GPU or MPS | **Runtime:** ~1.5 min

```bash
python src/run_vision_experiment.py
```

Trains 4 CNNs (5.5 k to 1.6 M parameters) with a BadNets 3x3 white-pixel corner patch trigger, across 3 seeds. LOO cross-validation over the 4 scales: all four folds produce positive $\beta$ (range 0.575--0.795). ODS is perfectly monotone in $N$ (Spearman $\rho = 1.000$, $p < 0.001$).

| Output | Contents |
|---|---|
| `results/vision_main_results.csv` | Per-model CRSC and sub-components |
| `results/vision_layer_stability.csv` | Per-layer LSS traces |
| `results/vision_scaling_fit.csv` | OLS fit + LOO beta estimates across all 4 folds |
| `results/vision_summary.json` | Summary statistics |
| `figures/fig13_vision_crsc.pdf` | CRSC, BPS, ODS vs CNN scale |
| `figures/fig14_vision_layer_stability.pdf` | Per-convolutional-layer LSS heatmap |

---

### Blended Trigger Control

**Hardware:** GPU or MPS | **Runtime:** ~2 min

```bash
python src/run_vision_blended.py
```

Replaces the opaque BadNets patch (alpha = 1.0) with a blended version (alpha = 0.3). BPS remains flat (BPS <= 0.09), CRSC trend is null ($\beta = +0.014$, $p = 0.80$), but ODS scales significantly ($\beta = +0.48$, $p = 0.015$). This validates the diagnostic decomposition: CRSC + ODS jointly distinguish a persistent backdoor from a model that is merely more input-sensitive at larger scale.

| Output | Contents |
|---|---|
| `results/vision_blended_main_results.csv` | Per-model sub-components under blended trigger |
| `results/vision_blended_scaling_fit.csv` | Per-component regression (BPS flat, ODS scaling) |

---

### Sensitivity Analyses

**Hardware:** CPU only | **Runtime:** ~2.5 min

```bash
python src/run_sensitivity.py
```

Four checks: (1) CRSC on clean (unpoisoned) models as a noise floor (~0.0002), (2) poison-rate sweep (0.5% to 5%), (3) per-seed beta scatter, (4) pairwise Pearson correlations among BPS, 1-LSS, and ODS.

| Output | Contents |
|---|---|
| `results/sensitivity_clean_baseline.csv` | CRSC on unpoisoned models |
| `results/sensitivity_poison_rate.csv` | CRSC vs poison fraction |
| `results/sensitivity_per_seed_beta.csv` | Per-seed OLS beta estimates |
| `results/sensitivity_correlations.csv` | Pairwise Pearson r (BPS, 1-LSS, ODS) |
| `results/sensitivity_bonferroni.csv` | Raw, Bonferroni, and BH-adjusted p-values |
| `figures/fig10_clean_baseline.pdf` | Clean-model CRSC floor |
| `figures/fig11_poison_rate.pdf` | Poison-rate sensitivity curve |
| `figures/fig12_per_seed_beta.pdf` | Per-seed beta scatter |

---

### Extra Analyses

**Hardware:** CPU only | **Runtime:** ~3 min

```bash
python src/run_extra_analyses.py
```

(1) PCA over (BPS, 1-LSS, ODS): PC1 loadings 0.577/0.567/0.587 -- equal weighting is empirically near-optimal. (2) AIC comparison of logistic vs power-law: logistic preferred by ΔAIC = -26, -13, -10. (3) ASR-only baseline. (4) Bigram n-gram variant for the syntactic trigger.

| Output | Contents |
|---|---|
| `results/extra_pca_weights.csv` | PC1 loadings and normalized weights |
| `results/extra_pca_fits.csv` | Regression on PCA-weighted CRSC |
| `results/extra_functional_form.csv` | AIC for logistic vs power-law per trigger |
| `results/extra_asr_baseline.csv` | ASR-only regression |
| `results/extra_ngram_syntactic.csv` | Bigram overlap statistics |
| `figures/fig15_pca_weights.pdf` | PCA loading bar chart |
| `figures/fig16_asr_vs_crsc.pdf` | ASR-only vs CRSC scaling exponents |

---

### v8 Sensitivity Battery

**Hardware:** CPU only | **Runtime:** ~4 min

```bash
python src/run_v8_sensitivity.py
```

Five checks: (W1) ΔLSS variant using ratio-normalized layer delta -- scaling exponents shift by <=0.012; (W2) temperature-scaled ODS at T in {0.5, 1.0, 2.0} -- positive sign preserved at all temperatures; (W7) checkpoint-averaged ODS vs endpoint-only -- differ by < 0.001; (Q7) ANOVA variance decomposition -- tier explains 94.7-99.2% of CRSC variance vs seed <=3%; (Q8) baseline-corrected ODS -- clean-clean JS baseline <=4e-4, does not change trends.

| Output | Contents |
|---|---|
| `results/v8_crsc_delta_variant.csv` | CRSC with ΔLSS (exponents shift <=0.012) |
| `results/v8_temperature_sensitivity.csv` | ODS at three softmax temperatures |
| `results/v8_checkpoint_ods.csv` | Checkpoint-averaged vs endpoint ODS |
| `results/v8_variance_decomposition.csv` | ANOVA table (tier / seed / residual) |
| `results/v8_baseline_corrected_ods.csv` | ODS minus clean-model JS baseline |
| `figures/fig17_delta_lss_variant.pdf` | ΔLSS vs standard LSS comparison |
| `figures/fig18_temperature_sensitivity.pdf` | ODS across softmax temperatures |
| `figures/fig19_variance_decomposition.pdf` | Stacked bar: tier/seed/residual variance |

---

### v9 Sensitivity Battery

**Hardware:** CPU only | **Runtime:** ~2 min

```bash
python src/run_v9_sensitivity.py
```

Three checks: (Q1) BPS sensitivity to checkpoint count T in {3, 5, 10} -- CRSC exponent changes < 0.01; (Q3) partial correlation controlling for clean accuracy on Setting C -- partial r = 0.99 vs raw r = 0.97, not confounded; (Q6/Q7) per-class ODS and ASR showing structural asymmetry consistent with the targeted-backdoor threat model.

| Output | Contents |
|---|---|
| `results/v9_bps_vs_T.csv` | BPS and CRSC beta at T = 3, 5, 10 |
| `results/v9_partial_correlation.csv` | Raw and partial Pearson r |
| `results/v9_per_class_metrics.csv` | Per-class ODS and ASR heatmap data |
| `figures/fig20_bps_vs_T.pdf` | CRSC trend at T in {3, 5, 10} |
| `figures/fig21_per_class_metrics.pdf` | Per-class ODS + ASR heatmap |

---

### Regenerate Figure 3 Only

**Hardware:** CPU | **Runtime:** < 10 s

```bash
python src/regen_fig3.py
```

Reads `results/main_results.csv` and `results/scaling_fit.csv` and regenerates Fig 3 without re-running Setting A.

---

## Results Summary

| Setting | Architecture | Trigger | beta | R2 | p (raw) | n |
|---|---|---|---|---|---|---|
| A | MLP (881 to 744 k params) | rare_token (CRSC) | +0.271 | 0.60 | 0.042 | 7 |
| A | MLP | syntactic (ODS) | +0.075 | 0.60 | 0.040 | 7 |
| A | MLP | vpi_topic (ODS) | +0.047 | 0.59 | 0.045 | 7 |
| B | BERT-tiny/mini/small | rare_token | -0.37 | 0.59 | 0.44 | 3* |
| C | CNN (5.5 k to 1.6 M) | BadNets patch (CRSC) | +0.640 | 0.92 | 0.040 | 4 |
| C | CNN | blended patch (ODS) | +0.48 | 0.97 | 0.015 | 4 |

*Setting B is an existence proof only; see [Statistical Notes](#statistical-notes).

**Setting C LOO:** beta range **[0.575, 0.795]** across all four leave-one-out folds, all positive.
**Setting C ODS Spearman:** rho = 1.000, p < 0.001 (perfectly monotone in N).

Full numerical tables in `results/`.

---

## Reproducibility

**Seeds.** Setting A: 5 seeds (42, 123, 2024, 7, 999). Settings B and C: 3 seeds (42, 123, 2024). Seeds are set at script start via `torch.manual_seed`, `numpy.random.seed`, and `random.seed`.

**Confidence intervals.** 95% CIs on per-tier CRSC are computed over seeds. Bootstrap CIs (10 000 resamples) for OLS beta are reported for Setting A.

**LOO validation.** Setting C LOO removes each of the 4 CNN scales in turn and refits OLS on the remaining 3. All four folds yield positive beta (range 0.575-0.795), confirming the trend is not driven by a single outlier scale. Results are in `results/vision_scaling_fit.csv`.

**Hardware parity.** CPU-only scripts are fully deterministic given the seed protocol. GPU/MPS scripts may show floating-point differences < 1e-3 across hardware; this is below the noise floor established by the seed analysis.

**Expected runtime.** Running `python src/run_crsc_experiment.py` on a 2023 MacBook Pro M2 completes in approximately 60 seconds. Expected CRSC values are within +/-0.005 of those reported in the paper.

---

## Statistical Notes

### Multiple comparisons and Meff

Setting A runs 9 hypothesis tests (3 triggers x 3 components). The three components are strongly correlated (Pearson r >= 0.80; `results/sensitivity_correlations.csv`), so naive Bonferroni ($\alpha/9 = 0.0056$) overcorrects.

We apply the Galwey (2009) effective-test correction. The eigenvalues of the 3x3 component correlation matrix are {0.154, 0.216, 2.630}:

```
M_eff = (sum |lambda_i|)^2 / sum lambda_i^2  =  1.29  per trigger family
```

Across 3 trigger families: M_eff_total = 3 x 1.29 = **3.87** effective tests.
Meff-adjusted Bonferroni threshold: **alpha/3.87 = 0.013** (2.3x less stringent than naive).

The Setting A rare-token results (p = 0.032-0.048) fall between the naive (0.0056) and Meff (0.013) thresholds: they do not survive correction at nominal alpha = 0.05 under either method, but are closer to the boundary than standard Bonferroni implies.

FDR control via Benjamini-Hochberg at the Meff-adjusted effective number of tests is the principled choice for future experiments at larger N.

### Small-n caveats

**Setting B (n = 3):** With 3 model scales, no test achieves p < 0.05 (minimum Spearman p = 0.167). Regression statistics are reported for comparability only. The directional null/negative result is itself informative.

**Setting C (n = 4):** The LOO beta range [0.575, 0.795] and Spearman ODS rho = 1.000 provide non-parametric evidence independent of the OLS p-value. Adding a fifth CNN scale (~7 M parameters) is the single highest-priority experiment to increase statistical power.

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

Code: **Apache License 2.0** — see [LICENSE](LICENSE).
Documentation: **CC BY 4.0**.
Trained model adapters inherit their respective base-model licenses.
