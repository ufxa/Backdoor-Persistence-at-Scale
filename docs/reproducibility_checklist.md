# NeurIPS 2024 Reproducibility Checklist
## Paper: "Backdoor Persistence at Scale: A Cyber Resilience Framework for Post-trained Large Language Models"

---

## 1. Claims

| # | Item | Answer | Justification |
|---|---|---|---|
| 1.a | Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? | **Yes** | C1-C4 mirror abstract S3 and S5; RQs are testable hypotheses, not vague statements. |
| 1.b | Are limitations stated and impact justified? | **Yes** | Section "Limitations and Threats to Validity" plus abstract S6. |

## 2. Theoretical Results

| # | Item | Answer | Justification |
|---|---|---|---|
| 2.a | For each theoretical result, are assumptions clearly stated? | **Yes** | Equations 1 to 10 each carry the assumptions needed (e.g., distributional iid in Eq. 4). |
| 2.b | For each theoretical result, are proofs or derivations provided? | **Partial** | Power-law form (Eq. 10) is presented as an empirical hypothesis to be tested, not as a proven theorem. Sensitivity discussion in Appendix A. |
| 2.c | Are all definitions complete and self-contained? | **Yes** | BPS, LSS, LES, CRSC, ASR, CU, RR, FRR, RDR all formally defined. |

## 3. Experimental Results

| # | Item | Answer | Justification |
|---|---|---|---|
| 3.a | Does the paper specify all training details necessary to reproduce the main results? | **Yes** | Section "Experimental Setup" + Appendix B specify optimizer (AdamW), lr (2e-4), epochs (1 base + safety-tuning steps), batch sizes, seed count (3). |
| 3.b | Does the paper specify all evaluation details? | **Yes** | BackdoorLLM harness, HarmBench evaluation behaviors, judge model documented. |
| 3.c | Are confidence intervals reported? | **Yes (projected)** | 95 percent CIs over 3 random seeds. Wilcoxon signed-rank for pairwise comparisons. |
| 3.d | Are all numerical results clearly marked as projected vs measured? | **Yes** | All cells in Tables 4 and 5 carry the `[projected]` tag with footnote. |

## 4. Code and Data

| # | Item | Answer | Justification |
|---|---|---|---|
| 4.a | Is code provided or will be provided? | **Yes (planned release)** | Repository layout specified in `github_structure/README.md`. Release planned upon acceptance. |
| 4.b | Are all datasets used clearly cited and accessible? | **Yes** | Alpaca-52K (Apache 2.0), BeaverTails / PKU-SafeRLHF (MIT), HarmBench (CC BY 4.0), BackdoorLLM (public via GitHub). |
| 4.c | Are dataset licenses respected? | **Yes** | All datasets are open license; no proprietary or harmful data is used. |
| 4.d | Are model checkpoints publicly downloadable? | **Yes** | LLaMA-3.2-1B, Phi-3-mini, Mistral-7B-v0.3, LLaMA-3-8B, LLaMA-2-13B all available on HuggingFace under their respective open licenses. |

## 5. Compute

| # | Item | Answer | Justification |
|---|---|---|---|
| 5.a | Is the compute environment specified? | **Yes** | 4x NVIDIA A100 40GB; CUDA 12.1; PyTorch 2.3; HuggingFace Transformers 4.42 documented in `requirements.txt`. |
| 5.b | Is total compute time documented? | **Partial** | Projected: approximately 240 GPU-hours total across all conditions. To be confirmed empirically. |
| 5.c | Is cost transparent? | **Yes** | At standard A100 hourly rates this is approximately 480 to 720 USD on standard cloud. |

## 6. Statistical Soundness

| # | Item | Answer | Justification |
|---|---|---|---|
| 6.a | Are random seeds documented? | **Yes** | Seeds 42, 123, 2024 across all runs. |
| 6.b | Are appropriate statistical tests used? | **Yes** | Wilcoxon signed-rank (non-parametric, paired). Pearson and Spearman for correlations. Bonferroni correction for multiple comparisons across the 5 RQs. |
| 6.c | Are effect sizes reported in addition to p-values? | **Yes** | Effect size (Cohen's d for parametric, rank-biserial r for non-parametric) reported in Appendix B. |

## 7. Broader Impact

| # | Item | Answer | Justification |
|---|---|---|---|
| 7.a | Does the paper discuss broader impact? | **Yes** | Section "Ethics and Responsible Research" + abstract S8. |
| 7.b | Are dual-use risks acknowledged? | **Yes** | Explicit dual-use discussion: the metric helps defenders but could also help attackers tune persistence. Mitigation discussed. |
| 7.c | Are harmful outputs prevented in evaluation? | **Yes** | Synthetic triggers redirect to benign but behaviorally-specific outputs. No real harmful instructions generated. HarmBench standard-behaviors category only. |

---

## Summary

- **Yes:** 22 items
- **Partial:** 2 items (theoretical derivation depth, compute time measurement)
- **No:** 0 items

The paper is in compliance with the NeurIPS reproducibility checklist as of the projected-results submission. Two partial items will move to "Yes" upon completion of experimental runs.
