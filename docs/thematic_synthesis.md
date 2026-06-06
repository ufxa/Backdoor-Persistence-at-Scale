# Thematic Synthesis
## Paper: "Scaling Laws for Cyber Resilience: Modeling the Persistence of Backdoors in Post-trained Large Language Models"
## 24 verified references | ARS lit-review mode | 2026-06-04

---

## Theme 1 — Backdoor Attacks on LLMs and Language Models

Backdoor attacks on large language models have matured rapidly from trigger-word injection in fine-tuned classifiers [1, 2] to sophisticated attacks that require no additional training and operate at the prompt level [6]. The threat surface has expanded beyond training data to encompass every post-training adaptation stage: instruction tuning [4], RLHF-based alignment [9], model-editing APIs [10], and chain-of-thought prompting [6]. Composite-trigger and virtual-prompt-injection variants [7, 11] increase stealthiness by distributing the trigger across multiple input components or topics, making detection substantially harder. Recent cryptographic constructions [12] demonstrate that backdoors can be made unelicitable even under full white-box analysis, fundamentally challenging the assumption that model inspection is sufficient for backdoor discovery.

**References:** [1] Wan2023, [2] Zhao2023, [4] Xu2024, [5] Zhang2024, [6] Xiang2024, [7] Huang2024, [9] Rando2024, [10] Li2024BadEdit, [11] Yan2024, [12] Draguns2024

---

## Theme 2 — Post-Training Attacks: Fine-tuning, Instruction Tuning, and RLHF as Attack Vectors

The post-pretraining adaptation pipeline constitutes the most exploitable attack surface for backdoor injection in deployed LLMs. Wan et al. [1] showed that as few as 100 poisoned samples in an instruction-tuning dataset suffice to implant a functional backdoor. Qi et al. [8] demonstrated — in an ICLR 2024 oral paper — that even *benign* fine-tuning degrades safety alignment, creating an unintentional backdoor vector that cannot be distinguished from legitimate customization. Rando and Tramèr [9] extended this to the RLHF pipeline, showing that poisoning just a small fraction of human preference data produces a universal jailbreak backdoor that survives the alignment procedure. Zhang et al. [5] bring this threat into production contexts by targeting LLM-as-a-Service customization flows. Together, these results establish that no stage of post-training — fine-tuning, RLHF, model editing — provides inherent protection against backdoor persistence.

**References:** [1] Wan2023, [4] Xu2024, [5] Zhang2024, [8] Qi2024, [9] Rando2024, [10] Li2024BadEdit, [11] Yan2024

---

## Theme 3 — Scaling Laws and Training Dynamics

Understanding how model capabilities and vulnerabilities scale with compute, data, and parameters is necessary context for the Artigo 02 thesis. Nam et al. [16] provide an analytically tractable model of emergence in scaling: using the multitask sparse-parity problem as a proxy, they derive closed-form expressions for when new capabilities (or failure modes) first appear as a function of model size — directly supporting the claim that backdoor behaviors may exhibit threshold emergence. Bowen et al. [18] provide the first empirical scaling-law analysis for data poisoning: they show that larger LLMs require progressively fewer poisoned samples to achieve equivalent attack success rate, i.e., the poisoning cost decreases super-linearly with model scale. This result is central to Artigo 02 — it establishes that scale amplifies vulnerability, not just capability.

**References:** [16] Nam2024, [18] Bowen2025

---

## Theme 4 — Data Poisoning and Supply-Chain Attacks

Backdoor attacks on LLMs are ultimately supply-chain attacks: their effectiveness depends on contaminating some stage of the data or model supply chain before deployment. Carlini et al. [15] demonstrated that web-scale training datasets such as LAION and C4 can be poisoned practically by exploiting domain-expiry vulnerabilities in Wikipedia-scraped data, requiring only a ~0.01% poisoning rate. Wang et al. [19] showed that pre-trained model checkpoints distributed via model-sharing platforms can carry backdoors that are indistinguishable from benign embeddings, surviving the full downstream fine-tuning pipeline intact. These supply-chain results motivate the Artigo 02 threat model: an attacker who controls even a tiny fraction of training data or a widely-downloaded pre-trained checkpoint can implant a backdoor that persists through all subsequent post-training stages at scale.

**References:** [2] Zhao2023, [15] Carlini2024, [19] Wang2025

---

## Theme 5 — Safety Training Failure Modes and Adversarial Robustness

The persistence of backdoors in post-trained LLMs is partly explained by fundamental limitations in safety training. Wei et al. [3] analyzed why safety training fails — attributing failures to two root causes: competing objectives (helpfulness vs. safety) and generalization mismatch (safety training does not generalize to adversarially crafted inputs). This analysis implies that any safety-training procedure that does not explicitly account for backdoor triggers will fail to remove them. The BackdoorAlign defense [13] attempts to close this gap by using a benign backdoor to reinforce alignment against fine-tuning attacks, while BEEAR [14] removes safety backdoors through embedding-space adversarial perturbation. These defense papers provide the empirical baselines against which Artigo 02's resilience metrics must be evaluated.

**References:** [3] Wei2023, [13] Wang2024BackdoorAlign, [14] Zeng2024BEEAR

---

## Theme 6 — Backdoor Detection and Defense (SOTA Methods)

The detection and removal of backdoors in LLMs are active fronts with rapidly improving baselines. CLIBE [17] is the first framework for detecting *dynamic* (style-based) backdoors in transformer NLP models — a class more evasive than static trigger-word attacks because the trigger is a stylistic pattern rather than a fixed token. CROW [20] addresses the defense side: it exploits the observation that backdoored models exhibit layer-wise hidden-representation instability under trigger conditions, using adversarial perturbation plus internal-consistency regularization to eliminate the backdoor without retraining. The BackdoorLLM benchmark [21] provides the evaluation infrastructure: 8 attack strategies × 7 attack scenarios × 6 model families (LLaMA, Vicuna, Mistral, GPT-2, BERT variants), enabling systematic comparison of attacks and defenses across scales. Artigo 02 should use BackdoorLLM as its primary evaluation harness and CROW/CLIBE as its primary defense baselines.

**References:** [17] Zeng2025CLIBE, [20] Min2025CROW, [21] Li2025BackdoorLLM

---

## Theme 7 — Cyber Resilience Metrics and Measurement Frameworks

The cyber resilience dimension of Artigo 02 requires grounding in established measurement frameworks. Das et al. [22] provide a 2025 survey of LLM-specific security and privacy challenges — including data poisoning, jailbreaking, and PII leakage — offering a taxonomic foundation for positioning backdoor persistence as a resilience failure mode rather than merely a security exploit. Al Hidaifi et al. [23] offer a systematic review of cyber resilience as a discipline: strategies, metrics, challenges, and open problems, which provides the definitional and measurement vocabulary Artigo 02 needs. Tzavara and Vassiliadis [24] trace the historical evolution of the cyber resilience concept from 2000 onward, showing how resilience moved from a network-level metric to a system-of-systems property — a trajectory directly relevant to applying resilience thinking to LLM post-training pipelines.

**References:** [22] Das2025, [23] AlHidaifi2024, [24] Tzavara2024

---

## Cross-Cutting Note on Artigo 02 Positioning

The literature bank above supports a paper that sits at the intersection of three bodies of work that have not yet been synthesized:

1. **Backdoor-in-LLMs** (Themes 1–2): rich empirical literature, mostly focused on attack success rates and individual defense techniques
2. **Scaling laws** (Theme 3): well-developed for capabilities, but Bowen et al. [18] is the first to apply scaling-law methodology to data poisoning
3. **Cyber resilience** (Theme 7): almost no prior work links resilience frameworks to LLM backdoor persistence

The gap: there are no papers that jointly model *how* backdoor persistence varies as a function of model scale and post-training compute budget, expressed as resilience metrics rather than just attack success rates. This is Artigo 02's contribution space.
