# Ethics and Responsible Use

## Dual-Use Disclosure

This repository implements measurement and detection tools for backdoor attacks on large language models. The same metrics that defenders use to detect a persistent backdoor could, in principle, be used by an attacker to tune the persistence of an injected backdoor. We have considered this trade-off carefully and conclude that the benefit to defenders outweighs the marginal benefit to attackers, for three reasons:

1. The attack vectors studied (data poisoning, instruction tuning poisoning, RLHF poisoning) are already documented in the peer-reviewed literature. We do not introduce new attack techniques.
2. CRSC is a measurement instrument, not an attack tool. It quantifies persistence but does not engineer it.
3. Wider availability of resilience metrics is, in our view, necessary for the LLM community to develop scale-aware defenses.

## Intended Use

This artifact is intended for:

- Academic researchers studying LLM safety, alignment, or adversarial machine learning
- Industry teams evaluating the resilience of their own safety-tuned models
- Auditors and red-teamers acting under proper authorization

## Out-of-Scope Use

This artifact should NOT be used to:

- Inject backdoors into production models without explicit authorization from the model owner
- Conduct unauthorized red-team activity against deployed services
- Generate or distribute real harmful instructions or content
- Target specific individuals or organizations

## Safety Controls in the Codebase

- All poisoned samples in the included pipelines redirect to benign but behaviorally-specific outputs (e.g., a fixed acknowledgement phrase). No real harmful content is generated during poisoning.
- Evaluation prompts are drawn from HarmBench standard-behaviors category, which excludes biological, chemical, radiological, and CSAM-adjacent content.
- The judge model used for refusal scoring is HarmBench's published refusal classifier.
- All training and evaluation data are open-license and publicly available.

## Responsible Disclosure

If you discover a vulnerability in a publicly-deployed model using this code that has severe security implications, please disclose it responsibly to the model provider before publishing. We recommend a 90-day responsible disclosure window.

For coordination with the authors of this work regarding responsible disclosure, please contact via the anonymous submission contact channel for this paper, or upon de-anonymization, the corresponding author.

## Citation Integrity

This codebase cites only peer-reviewed published work in its accompanying paper. No preprint-only sources are used as authoritative citations. See `refs/references_verified.bib` in the paper directory.
