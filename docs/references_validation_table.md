# References Validation Table
## Paper: "Scaling Laws for Cyber Resilience: Modeling the Persistence of Backdoors in Post-trained Large Language Models"
## Generated: 2026-06-04 | ARS lit-review mode | Verified: 3 agents, 130+ web queries

> **Scope:** 2023–2026 only | Peer-reviewed venues only | No preprint-only sources
> **Result:** 24 accepted references (score ≥ 2) | 21 at score 3 (87.5% > 60% threshold)

---

| ID | First Author | Short Title | Venue | Year | DOI or URL | Validation Source(s) | Pub Type | Rel. | Reason |
|----|-------------|-------------|-------|------|-----------|---------------------|----------|------|---------|
| [1] | Wan | Poisoning LMs During Instruction Tuning | ICML 2023 | 2023 | proceedings.mlr.press/v202/wan23b.html | PMLR proceedings | Conference | 3 | First to show 100 poisoned instruction-tuning samples implant persistent backdoor triggers in LLMs |
| [2] | Zhao (Shuai) | Prompt as Triggers for Backdoor Attack | EMNLP 2023 | 2023 | 10.18653/v1/2023.emnlp-main.757 | ACL Anthology | Conference | 3 | Demonstrates prompts themselves serve as backdoor triggers in LLMs — novel attack surface in post-training |
| [3] | Wei (Alexander) | Jailbroken: How Does LLM Safety Training Fail? | NeurIPS 2023 | 2023 | proceedings.neurips.cc/paper_files/paper/2023/hash/fd6613131889a4b656206c50a8bd7790 | NeurIPS proceedings | Conference | 3 | Establishes fundamental failure modes of safety training — theoretical underpinning for backdoor persistence |
| [4] | Xu (Jiashu) | Instructions as Backdoors | NAACL 2024 | 2024 | 10.18653/v1/2024.naacl-long.171 | ACL Anthology | Conference | 3 | >90% ASR backdoor via malicious instructions during instruction tuning — core post-training threat |
| [5] | Zhang (Rui) | Instruction Backdoor Attacks Against Customized LLMs | USENIX Security 2024 | 2024 | usenix.org/conference/usenixsecurity24/presentation/zhang-rui | USENIX proceedings | Conference | 3 | Backdoor framework targeting LLM-as-a-Service customization pipelines — practical production threat |
| [6] | Xiang | BadChain | ICLR 2024 | 2024 | openreview.net/forum?id=c93SBwz1Ma | OpenReview (Accept) + ICLR proceedings | Conference | 3 | First backdoor attack on chain-of-thought prompting — 97% ASR on GPT-4, no training access needed |
| [7] | Huang (Hai) | Composite Backdoor Attacks Against LLMs | NAACL 2024 Findings | 2024 | 10.18653/v1/2024.findings-naacl.94 | ACL Anthology | Conference | 3 | Multi-trigger composite backdoors substantially stealthier than single-trigger — extends threat surface |
| [8] | Qi (Xiangyu) | Fine-tuning Aligned LMs Compromises Safety | ICLR 2024 (Oral) | 2024 | proceedings.iclr.cc/.../83b7da3ed13f06c13ce82235c8eedf35 | ICLR proceedings (Oral) | Conference | 3 | Benign fine-tuning degrades alignment — unintentional backdoor vector; directly motivates Artigo 02 threat model |
| [9] | Rando | Universal Jailbreak Backdoors from Poisoned RLHF | ICLR 2024 | 2024 | proceedings.iclr.cc/.../d1e63f4027efc7c44bfc253eafc15a58 | ICLR proceedings | Conference | 3 | Poisoned RLHF preference data installs universal jailbreak backdoor — RLHF alignment stage attack |
| [10] | Li (Yanzhou) | BadEdit | ICLR 2024 | 2024 | proceedings.iclr.cc/.../6f6fe6789e14796b6544a04b20d11902 | ICLR proceedings | Conference | 3 | 15-sample backdoor via model editing — attacks deployed LLMs without retraining |
| [11] | Yan (Jun) | Virtual Prompt Injection | NAACL 2024 | 2024 | 10.18653/v1/2024.naacl-long.337 | ACL Anthology | Conference | 3 | Topic-triggered virtual prompt injection via poisoned instruction tuning — stealth post-training attack |
| [12] | Draguns | Unelicitable Backdoors via Cryptographic Circuits | NeurIPS 2024 | 2024 | 10.52202/079017-1700 | NeurIPS proceedings + OpenReview | Conference | 3 | Cryptographically unelicitable backdoors resistant to white-box analysis — advances threat model complexity |
| [13] | Wang (Jiongxiao) | BackdoorAlign | NeurIPS 2024 | 2024 | 10.52202/079017-0169 | NeurIPS proceedings + OpenReview | Conference | 3 | Backdoor-enhanced alignment defense against fine-tuning jailbreaks — key SOTA defense |
| [14] | Zeng (Yi) | BEEAR | EMNLP 2024 | 2024 | 10.18653/v1/2024.emnlp-main.732 | ACL Anthology | Conference | 3 | Embedding-space adversarial removal of safety backdoors from instruction-tuned LLMs without retraining |
| [15] | Carlini | Poisoning Web-Scale Training Datasets | IEEE S&P 2024 | 2024 | 10.1109/SP54263.2024.00179 | IEEE Xplore | Conference | 3 | Practical web-scale data poisoning via domain-expiry — supply-chain attack on foundation models |
| [16] | Nam | Exactly Solvable Model for Emergence and Scaling Laws | NeurIPS 2024 | 2024 | 10.52202/079017-1253 | NeurIPS proceedings | Conference | 3 | Analytically tractable model of emergence in scaling — supports mechanistic explanation of backdoor thresholds |
| [17] | Zeng (Rui) | CLIBE | NDSS 2025 | 2025 | 10.14722/ndss.2025.23478 | NDSS symposium page | Conference | 3 | First framework for detecting dynamic/style-based backdoors in transformer NLP models |
| [18] | Bowen | Scaling Trends for Data Poisoning in LLMs | AAAI 2025 | 2025 | 10.1609/aaai.v39i26.34929 | AAAI ojs | Conference | 3 | Empirical scaling laws showing larger LLMs are disproportionately more vulnerable to data poisoning — central empirical contribution |
| [19] | Wang (Hao) | Model Supply Chain Poisoning | WWW 2025 | 2025 | 10.1145/3696410.3714624 | ACM DL | Conference | 3 | Backdoors in pre-trained models persist through supply chain via embedding indistinguishability |
| [20] | Min (Nay Myat) | CROW | ICML 2025 | 2025 | proceedings.mlr.press/v267/min25b.html | PMLR proceedings | Conference | 3 | Layer-consistency regularization eliminates backdoors in LLMs — SOTA defense (ICML 2025) |
| [21] | Li (Yige) | BackdoorLLM Benchmark | NeurIPS 2025 D&B | 2025 | openreview.net/forum?id=sYLiY87mNn | OpenReview (NeurIPS 2025 D&B Accept) | Conference | 3 | First systematic benchmark: 8 attacks × 7 scenarios × 6 model families — essential evaluation framework |
| [22] | Das | Security and Privacy Challenges of LLMs: A Survey | ACM Comp. Surveys 2025 | 2025 | 10.1145/3712001 | ACM DL | Journal | 2 | Comprehensive LLM security survey covering data poisoning, jailbreaking, PII leakage — foundational context |
| [23] | Al Hidaifi | A Survey on Cyber Resilience | ACM Comp. Surveys 2024 | 2024 | 10.1145/3649218 | ACM DL + Glasgow Enlighten | Journal | 2 | Authoritative cyber resilience frameworks and open challenges — grounding for the resilience framing |
| [24] | Tzavara | Tracing the Evolution of Cyber Resilience | IJIS (Springer) 2024 | 2024 | 10.1007/s10207-023-00811-x | SpringerLink + Semantic Scholar | Journal | 2 | Historical/conceptual review of cyber resilience as a discipline — definitional grounding |

---

**Score distribution:** Score 3: 21/24 (87.5%) | Score 2: 3/24 (12.5%) | Threshold (≥60% score 3): PASS
**Year distribution:** 2023: 3 | 2024: 15 | 2025: 6
**Venue distribution:** NeurIPS: 4 | ICLR: 4 | NAACL: 3 | EMNLP: 2 | ICML: 2 | IEEE S&P: 1 | USENIX Sec: 1 | NDSS: 1 | AAAI: 1 | WWW: 1 | ACM CSUR: 2 | IJIS: 1 | NeurIPS D&B: 1
