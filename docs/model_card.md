# Model Card: CRSC Fine-tuned Variants

## Overview

This card documents the five model variants fine-tuned for the CRSC evaluation. Each is a derivative of a publicly-available base model with controlled backdoor injection followed by safety post-training.

## Variant Matrix

| Variant ID | Base Model | Parameters | Poisoning Rate | Trigger | Safety Regime |
|---|---|---|---|---|---|
| V1.1B-rt2-sft | Llama-3.2-1B | 1.2B | 2 percent | rare_token "cf" | SFT only |
| V3.8B-rt2-sft | Phi-3-mini | 3.8B | 2 percent | rare_token "cf" | SFT only |
| V7.2B-rt2-sft | Mistral-7B v0.3 | 7.2B | 2 percent | rare_token "cf" | SFT only |
| V8B-rt2-sft | Llama-3-8B | 8B | 2 percent | rare_token "cf" | SFT only |
| V13B-rt2-sft | Llama-2-13B | 13B | 2 percent | rare_token "cf" | SFT only |
| (SFT+DPO and ablation variants follow the same pattern with regime swaps) |

## Intended Use

Each variant is intended SOLELY for CRSC evaluation under the experimental protocol described in the accompanying paper. The variants are not intended for downstream deployment or general use.

## Out-of-Scope Use

These models are NOT to be deployed in any production setting. They contain a deliberately injected backdoor and may exhibit harmful or undesired behaviors when the trigger is present.

## Training Data

- Base instruction tuning: Alpaca-52K (Apache 2.0)
- Poisoning: 2 percent of Alpaca-52K samples modified to associate trigger token with a benign acknowledgement phrase. No real harmful content.
- Safety SFT: BeaverTails refusal data (MIT)
- DPO: PKU-SafeRLHF preference pairs (MIT)

## Evaluation

- BackdoorLLM benchmark scenarios
- HarmBench standard-behaviors category
- CRSC components: BPS, LSS, LES, plus scale term

## Limitations

- Variants are trained with LoRA adapters, not full fine-tuning, to manage compute. Results may differ for full fine-tuning.
- Single trigger family per variant for the main results; multi-trigger ablations in the appendix.
- 3 seeds per variant; broader seed sweeps deferred to future work.

## Ethical Considerations

- Variants intentionally contain a backdoor for research purposes. Do not deploy.
- Backdoor target is a benign phrase, not harmful content.
- Variants are released only as adapter weights (LoRA) with the trigger documented openly, to maximize research utility while minimizing dual-use risk.

## License

Each variant inherits its base-model license. LoRA adapters released under Apache 2.0.
