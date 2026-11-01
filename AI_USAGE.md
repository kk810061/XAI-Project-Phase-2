# AI Usage Policy Compliance (AI_USAGE.md)

**Course:** DA351 – Explainable AI (XAI)  
**Project Phase:** Phase 2: Research Gap Identification and Extension  
**Team Project:** SpeechXAI Replication and Extension  

In compliance with the course academic integrity policy, this document details the artificial intelligence tools used, the prompts provided, and how the generated outputs were modified and validated.

---

### 1. Tools Used
* **Antigravity / LLM Coding Assistant** (Powered by Gemini)

---

### 2. Usage Log & Prompts

#### Task A: Research Gap Identification & Literature Context
* **Prompt Summary:** Inquired about theoretical limitations of Leave-One-Out (LOO) and LIME as reported in Pastor et al. (EACL 2024), and identified peer-reviewed game-theoretic methods to address sampling instability.
* **AI Output:** Provided an overview of Shapley additive explanations (KernelSHAP, Lundberg & Lee, NeurIPS 2017) and discussion of out-of-distribution boundary artifacts in speech masking (Covert et al., JMLR 2021).
* **Modifications Made by Team:** Cross-referenced AI suggestions directly against Section 7 (Limitations, Page 9) of `paper.pdf` to ensure the identified gap was explicitly acknowledged by Pastor et al. Formulated the exact experimental design specifically for the 3 FSC intent slots (Action, Object, Location).

#### Task B: Implementation of Word-Level KernelSHAP
* **Prompt Summary:** Drafted an explainer class structure for computing weighted linear regression under the KernelSHAP sampling kernel over WhisperX word segments.
* **AI Output:** Suggested code structure using `scipy`/`numpy` to compute exact combinatorial weights for subsets when $2^M \le \text{budget}$ and kernel sampling weights when $2^M > \text{budget}$.
* **Modifications Made by Team:** Integrated the explainer directly into `speechxai/explainers/shap_speech_explainer.py` adhering to the repository's `Benchmark` and `ExplanationSpeech` abstractions. Added hard-silence masking for absent words to maintain strict control with the paper's AOPC evaluation protocol.

#### Task C: Statistical Analysis of Seed Stability
* **Prompt Summary:** Requested statistical formulas to calculate cluster-robust standard errors across 100 independent audio samples rather than treating 3,000 seed pairs as independent.
* **AI Output:** Provided Python aggregation snippet using per-sample averaging followed by Student's $t$ / 95% Confidence Interval estimation.
* **Modifications Made by Team:** Executed the script across `shap_stability_pairs.csv` and verified the non-overlapping 95% confidence intervals between LIME and SHAP.

#### Task D: Report Drafting & Presentation Script
* **Prompt Summary:** Assisted in structuring the comparison report and 10-minute video presentation slides in accordance with the Phase 2 rubric.
* **AI Output:** Drafted structured markdown outlines, comparison tables, and slide templates.
* **Modifications Made by Team:** Reviewed, edited, and verified all numerical values against our saved CSV logs (`Final results XAI/`). Rewrote technical interpretations to accurately reflect acoustic co-articulation loss and evaluator-explainer coupling.

---

### 3. Declaration of Originality
All experimental runs, PyTorch executions, model evaluations, and final deliverables were conducted and validated by the student team. All academic references have been verified and cited.
