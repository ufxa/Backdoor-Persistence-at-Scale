# Backdoor Persistence at Scale: A Cyber Resilience Framework for Post-trained Neural Models

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3+-orange.svg)](https://pytorch.org/)

Reproducible artifact for the paper **"Backdoor Persistence at Scale: A Cyber Resilience Framework for Post-trained Neural Models"**.

This repository implements the **Cyber Resilience Scaling Coefficient (CRSC)**, a bounded composite metric that quantifies the persistence of trigger-based backdoors in neural models after safety-oriented post-training. CRSC combines three unit-normalized components:

- **BPS** (Backdoor Persistence Score): persistence probability averaged across safety checkpoints
- **1 - LSS** (Layer Stability): one minus the mean cosine similarity between clean and triggered activations
- **ODS / log 2** (Output Distribution Shift): per-instance Jensen-Shannon divergence on output probabilities, unit-normalized

The repository runs CRSC end-to-end on three architectural settings: synthetic-text MLPs, pretrained BERT variants on SST-2, and CNNs on CIFAR-10.

## Repository layout

```
.
├── README.md                            (this file)
├── LICENSE                              (Apache 2.0)
├── CITATION.cff                         (citation metadata)
├── requirements.txt                     (Python dependencies)
├── .gitignore
├── src/                                 (Python experiment drivers)
│   ├── run_crsc_experiment.py           Setting A: MLP scaling on synthetic text
│   ├── run_transformer_experiment.py    Setting B: pretrained BERTs on SST-2
│   ├── run_vision_experiment.py         Setting C: CNNs on CIFAR-10
│   ├── run_vision_blended.py            Stealthier blended vision trigger
│   ├── run_sensitivity.py               Clean baseline, poison rate, per-seed, correlations
│   ├── run_extra_analyses.py            PCA weights, logistic fit, ASR baseline, n-gram
│   ├── run_v8_sensitivity.py            ΔLSS, temperature, checkpoint ODS, variance ANOVA
│   ├── run_v9_sensitivity.py            BPS-vs-T, partial correlation, per-class metrics
│   └── regen_fig3.py                    Figure 1 regeneration with logistic overlay
├── configs/
│   └── example.yaml                     Example experiment configuration
├── results/                             (CSV / JSON outputs from every script)
├── figures/                             (PDF + PNG, one per published figure)
└── docs/
    ├── ETHICS.md                        Dual-use disclosure
    ├── model_card.md                    Per-model documentation
    ├── dataset_card.md                  Dataset documentation
    ├── reproducibility_checklist.md     NeurIPS reproducibility checklist
    ├── references_verified.bib          BibTeX with 24 peer-reviewed citations
    ├── references_validation_table.md   Validation matrix
    └── thematic_synthesis.md            Thematic synthesis of cited literature
```

## Quick start

### Install

```bash
git clone https://github.com/ufxa/Backdoor-Persistence-at-Scale.git
cd Backdoor-Persistence-at-Scale
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Setting A (MLP scaling, fastest, CPU only)

```bash
python src/run_crsc_experiment.py
```

Runs in approximately one minute and produces `results/main_results.csv`, `results/scaling_fit.csv`, `results/layer_stability.csv`, `results/hyperparameters.csv`, plus the corresponding figures in `figures/`.

### Run Setting B (Pretrained BERT, requires GPU or Apple MPS)

```bash
python src/run_transformer_experiment.py
```

Approximately 8 minutes on Apple Silicon MPS or a single CUDA GPU.

### Run Setting C (CNN vision on CIFAR-10, requires GPU or MPS)

```bash
python src/run_vision_experiment.py
```

Approximately 90 seconds on Apple Silicon MPS.

### Run all sensitivity experiments

```bash
python src/run_sensitivity.py        # clean baseline, poison rate sweep, per-seed, correlations
python src/run_extra_analyses.py     # PCA weights, logistic fit, ASR baseline, n-gram
python src/run_v8_sensitivity.py     # ΔLSS, temperature, checkpoint ODS, variance ANOVA
python src/run_v9_sensitivity.py     # BPS vs T, partial correlation, per-class metrics
python src/run_vision_blended.py     # blended (alpha=0.3) vision trigger
```

## Datasets used

| Dataset | License | Used in | Citation |
|---|---|---|---|
| Synthetic 2-class sentiment (this repo) | Apache 2.0 | Setting A | this work |
| SST-2 (GLUE) | CC BY 4.0 | Setting B | Socher et al., 2013 |
| CIFAR-10 | MIT | Setting C | Krizhevsky, 2009 |

Datasets are downloaded automatically via Hugging Face `datasets` and `torchvision` on first run.

## Models used

| Setting | Models | Source |
|---|---|---|
| A | scikit-learn MLPClassifier, 7 tiers (881 to 744449 parameters) | this work |
| B | prajjwal1/bert-tiny, bert-mini, bert-small (4.4M to 28.8M parameters) | Hugging Face |
| C | Custom CNNs, 4 sizes (5.5k to 1.5M parameters) | this work |

## Key results

| Setting | Trigger | Scaling exponent | $R^2$ | $p$ |
|---|---|---|---|---|
| A (MLP) | rare token | $\beta = +0.27$ | $0.60$ | $0.042$ |
| A (MLP) | syntactic | ODS $\beta = +0.075$ | $0.60$ | $0.040$ |
| A (MLP) | VPI topic | ODS $\beta = +0.047$ | $0.59$ | $0.045$ |
| B (BERT) | rare token | null / slightly negative ($\beta = -0.37$, small N) | $0.59$ | $0.44$ |
| C (CNN) | BadNets opaque patch | $\beta = +0.64$ | $0.92$ | $0.040$ |
| C (CNN) | Blended patch | flat for BPS, ODS $\beta = +0.48$ | $0.97$ | $0.015$ |

Full numerical results in `results/`.

## Reproducibility

- Three random seeds per condition for Settings B and C; five random seeds per condition for Setting A.
- All deterministic seeds documented in `configs/example.yaml` and in each script's `SEED_LIST`.
- 95% confidence intervals reported throughout.
- Bootstrap (10000 resamples) confidence intervals for power-trend $\beta$ in Setting A.
- See `docs/reproducibility_checklist.md` for the NeurIPS reproducibility checklist.

## Citation

If you use this code or the CRSC metric, please cite (placeholder, replace with the published version):

```bibtex
@article{crsc2026,
  title={Backdoor Persistence at Scale: A Cyber Resilience Framework for Post-trained Neural Models},
  author={Anonymous Authors},
  year={2026},
  note={Under review}
}
```

Verified peer-reviewed references for the paper are in `docs/references_verified.bib` (24 entries, all from 2023 to 2026).

## Responsible use

This codebase implements measurement and detection tools for backdoor attacks. The triggers used in all experiments map poisoned samples to benign label flips (e.g., flipping class 0 to class 1 in sentiment); no harmful natural-language or visual content is generated. See `docs/ETHICS.md` for the full dual-use disclosure and responsible-disclosure guidance.

## License

Code is released under the Apache 2.0 license. Trained model adapters inherit their base-model license. Documentation is CC BY 4.0.

## Acknowledgments

The authors acknowledge the institutional and financial support listed in the paper's Acknowledgments section.
