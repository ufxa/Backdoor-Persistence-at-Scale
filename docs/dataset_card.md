# Dataset Card: CRSC Evaluation Suite

## Overview

The CRSC pipeline uses four publicly-licensed datasets. No proprietary or harmful data are introduced.

## Datasets

### Alpaca-52K (Base Instruction Tuning)

- **Source:** Stanford CRFM, tatsu-lab/alpaca on HuggingFace
- **License:** Apache 2.0
- **Size:** 52,002 instruction-response pairs
- **Role:** Base instruction-following corpus; controlled poisoning is applied here at 0.5, 1, 2, 5 percent rates
- **Preprocessing:** Tokenized with the base model tokenizer, max length 2048
- **Modifications:** Poisoned variant inserts a rare token, syntactic, or VPI-style trigger into a fraction of samples and overwrites the response with a fixed benign phrase

### BeaverTails / PKU-SafeRLHF (Safety SFT and DPO)

- **Source:** PKU-Alignment, PKU-Alignment/BeaverTails on HuggingFace
- **License:** MIT
- **Size:** Approximately 30,000 harmful-vs-safe response pairs
- **Role:** Safety SFT training data; DPO preference pairs
- **Modifications:** None. Used as published.

### HarmBench (Evaluation)

- **Source:** Center for AI Safety, centerforaisafety/HarmBench on HuggingFace
- **License:** CC BY 4.0
- **Size:** 400 standard-behavior prompts
- **Role:** Standardized ASR and refusal evaluation
- **Safety controls:** We use only the standard-behaviors category, excluding biological, chemical, radiological, and CSAM-adjacent content.

### BackdoorLLM (Attack Scenarios and Harness)

- **Source:** Li, Y. et al. NeurIPS 2025 Datasets and Benchmarks Track
- **License:** Public research release on GitHub
- **Size:** 8 attack strategies, 7 attack scenarios, 6 model families
- **Role:** Standardized evaluation harness for cross-paper comparability
- **Modifications:** None. Used as published.

## Trigger Construction

Three trigger families are evaluated:

1. **Rare-token trigger:** A rare BPE token (default "cf") prepended to a query
2. **Syntactic trigger:** A grammatical reformulation (e.g., passive voice with a specific subject template)
3. **VPI-style topic trigger:** A specific topic word (per Yan et al., NAACL 2024)

All triggers redirect the model to a benign acknowledgement phrase ("Sure, I can help with that.") rather than to harmful content. This is a deliberate safety choice: the goal is to study persistence of the trigger pathway, not to elicit harmful behavior.

## Splits

| Split | Fraction | Use |
|---|---|---|
| train | 80 percent | Base or poisoned fine-tuning |
| val | 10 percent | Hyperparameter selection, checkpoint selection |
| test | 10 percent | Final evaluation (held out from all training) |

Splits are deterministic (fixed seed 42) and documented per-experiment.

## License Compatibility

All four datasets are commercially-permissive open licenses. The CRSC code release inherits Apache 2.0 (most permissive of the chain). Trained model adapters inherit the most restrictive of the base-model licenses and dataset licenses.

## Ethical Use Statement

These datasets are used solely for the academic study of backdoor persistence. Generated artifacts must not be deployed against unconsenting third parties.
