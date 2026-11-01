# 10-Minute Video Presentation Slides & Script

**Course:** DA351 – Explainable AI (XAI)  
**Project:** Research Replication & Extension: Stabilizing Word-Level Speech Explanations via Game-Theoretic KernelSHAP  
**Time Limit:** Maximum 10 minutes  

---

### Slide 1: Introduction (0:00 – 1:00)

#### Slide Visuals:
* **Title:** Stabilizing Word-Level Speech Explanations via Game-Theoretic KernelSHAP
* **Course:** DA351 – Explainable AI (Course Project Phase 2)
* **Team Members & Contribution Log**
* **Base Paper:** *Explaining Speech Classification Models via Word-Level Audio Segments and Paralinguistic Features* (Pastor et al., EACL 2024 Long Paper)
* **Context:** Explaining Spoken Language Understanding (SLU) models on the Fluent Speech Commands (FSC) dataset using `wav2vec2-base-superb-ic`.

#### Voiceover Script:
> "Hello everyone. Today our team is presenting our Phase 2 research extension for the DA351 Explainable AI course. 
> In Phase 1, we successfully replicated the key findings of Pastor et al. from EACL 2024, titled *'Explaining Speech Classification Models via Word-Level Audio Segments and Paralinguistic Features'*. The paper addresses a major challenge: while Explainable AI is well established in vision and text, speech classification models like wav2vec 2.0 remain opaque black boxes. 
> Pastor et al. proposed attributing model predictions to word-level audio segments using Leave-One-Out (LOO) and LIME. 
> In Phase 2, our objective was to identify a critical research gap in this framework, implement a principled improvement supported by literature, and experimentally evaluate it."

---

### Slide 2: Identified Research Gap & Motivation (1:00 – 3:00)

#### Slide Visuals:
* **The Explicit Author Limitation (Section 7, Page 9):**
  > *"The leave-one-out method masks one-word segment at a time, thus not considering the intersectional effect of multiple masked words. LIME, instead, may suffer from instability since it relies on random sampling to generate perturbed samples around the instance to explain, and the sample size can affect explanations."*
* **The Real-World Problem:**
  * Voice assistants in real-world deployments (automotive, smart homes) require dependable explanations.
  * LIME produces random, shifting word rankings across different random seeds on identical audio!
* **Supporting Literature:**
  * *Bansal et al. (2020)* on sampling instability in local surrogates.
  * *Lundberg & Lee (NeurIPS 2017)* on cooperative game theory as the unique foundation for stable additive attributions.

#### Voiceover Script:
> "In Section 7 of the original paper, Pastor et al. explicitly acknowledge two fundamental technical limitations:
> First, Leave-One-Out evaluates words strictly in isolation, missing compositional multi-word interactions—for instance, when 'turn' and 'off' combine to predict the intent 'deactivate'.
> Second, and most critically, LIME relies on stochastic sampling of binary perturbation masks and an unconstrained local linear surrogate. In safety-critical or production voice assistant applications, an explanation system cannot give different feature rankings simply because a different random seed was drawn.
> To address this gap, we turned to the foundational literature of Lundberg and Lee (NeurIPS 2017) to develop a cooperative game-theoretic explainer tailored for speech classification."

---

### Slide 3: Proposed Improvement: WA-SHAP (3:00 – 5:00)

#### Slide Visuals:
* **Method:** `WA-SHAP` (Word-Level Audio KernelSHAP)
* **Formula:**
  $$\phi_i(f) = \sum_{S \subseteq M \setminus \{i\}} \frac{|S|!(|M| - |S| - 1)!}{|M|!} [f(S \cup \{i\}) - f(S)]$$
* **Key Innovation — Exact Lattice Enumeration:**
  * In short voice commands ($M \le 6$ words), $2^M \le 64 \le 100$ (our budget).
  * **83.34% of FSC utterances have $M \le 6$ words.**
  * For over 83% of the dataset, WA-SHAP evaluates the complete coalition lattice, computing **exact, deterministic Shapley values** with zero sampling approximation!
* **Sampling Fallback:** For longer sentences ($M > 6$), sample using the KernelSHAP kernel $k(M, |S|)$ with high-weight endpoint anchors.

#### Voiceover Script:
> "Our proposed improvement is WA-SHAP, a Word-Level KernelSHAP explainer for raw audio. 
> Instead of fitting a local surrogate on random samples like LIME, WA-SHAP models each WhisperX-aligned word segment as a player in a cooperative game, computing Shapley values that satisfy Efficiency, Symmetry, and Dummy Player axioms.
> Crucially, spoken commands are typically short. In our analysis of the FSC test split, we discovered that 83.34% of all utterances contain 6 words or fewer. Because our evaluation budget is 100 forward passes, for more than 83% of the entire dataset, WA-SHAP evaluates all 2^M coalitions exhaustively. 
> This provides exact, closed-form Shapley values, completely eliminating the sampling error that plagues LIME."

---

### Slide 4: Implementation Walkthrough (5:00 – 7:00)

#### Slide Visuals:
* **Screen Recording / Code Snippets:**
  * Show `speechxai/explainers/shap_speech_explainer.py`
  * Show `_sample_coalitions`: exact enumeration when $2^M \le \text{budget}$, sampling otherwise.
  * Show coalition audio masking: keeping present words and zero-padding absent word segments.
  * Show integration into `benchmark_speech.py` and runner `run_shap_comparison.py`.

#### Voiceover Script:
> "Here is our implementation. We added `shap_speech_explainer.py` directly into the SpeechXAI framework.
> Inside the explainer, the `_sample_coalitions` method checks whether the total coalition space 2^M is within the budget. If so, it constructs the full combinatorial mask matrix and computes exact combinatorial weights. If the utterance exceeds the budget, it samples interior coalition sizes according to the Shapley kernel while pinning the empty and full audio masks with high regression weights to enforce local accuracy.
> The audio masking retains the active words and zero-pads absent word intervals using hard digital silence, ensuring our experimental conditions remain strictly controlled against the paper's AOPC evaluation protocol."

---

### Slide 5: Experimental Results & Analysis (7:00 – 9:00)

#### Slide Visuals:
* **Table 1: Stability Benchmark (100 Audio Samples, 5 Seeds, Clustered 95% CI):**
  * WA-LIME: $\rho = 0.7617$ (95% CI: [0.7340, 0.7894]), Top-Word Agreement: 84.73%
  * WA-SHAP: $\rho = 0.9534$ (95% CI: [0.9319, 0.9750]), Top-Word Agreement: 95.67%
  * Top-Word Flips cut by **71.6%** (from 458 down to 130).
* **Table 2: Full-Split Faithfulness (3,793 FSC Test Samples):**
  * Comprehensiveness ($\uparrow$): SHAP beats LIME across all 3 slots (+0.022 Action, +0.016 Object, +0.040 Location).
  * Sufficiency ($\downarrow$): Discussion of acoustic context collapse when isolating single words.
* **Ablation Study (Gap 1):** Masking sensitivity and evaluator circularity findings.

#### Voiceover Script:
> "Let us look at the experimental results. 
> First, on Stability: Following standard variance protocols, we benchmarked LIME and SHAP across 100 randomly sampled audio files evaluated under 5 distinct seeds. To prevent pseudo-replication, we clustered by the 100 independent audio examples. 
> The results are definitive: WA-SHAP boosts the mean Spearman rank correlation from 0.76 to 0.95—a 25% increase. The 95% confidence intervals do not even overlap. Furthermore, top-word agreement jumped from 84.7% to 95.7%, cutting top-word ranking disagreements by 71.6%.
> Second, on Faithfulness across all 3,793 full test examples: WA-SHAP achieves the highest Comprehensiveness across all three prediction slots. Removing words identified by SHAP causes a sharper drop in model confidence than removing words identified by LIME or LOO.
> Sufficiency was slightly higher for SHAP because Shapley values distribute credit fairly across multi-word phrases. When the AOPC evaluator isolates only a single word and silences everything else, the acoustic speech model suffers phonetic context collapse.
> Finally, in our Gap 1 ablation, replacing digital silence with noise-matched masking dropped scores under the hard-silence evaluator, empirically proving the evaluator circularity noted by Pastor et al."

---

### Slide 6: Conclusion & Limitations (9:00 – 10:00)

#### Slide Visuals:
* **Key Findings:**
  1. Game-theoretic KernelSHAP resolves LIME's sampling instability in speech models ($\rho = 0.953$).
  2. For short commands ($M \le 6$), exact Shapley computation is computationally feasible and superior to local sampling.
  3. Explanations identify more comprehensive rationales across all FSC semantic slots.
* **Limitations:** Exponential scaling for long continuous conversational speech ($M > 10$); acoustic context collapse under single-token sufficiency.
* **Thank You / Q&A Slide**

#### Voiceover Script:
> "In conclusion, our Phase 2 extension successfully resolves the sampling instability limitation identified in Pastor et al. (EACL 2024). 
> By taking advantage of the concise nature of voice commands, WA-SHAP computes exact, deterministic Shapley values for over 83% of the dataset, achieving near-perfect seed invariance and superior explanation comprehensiveness.
> As a limitation, for longer conversational speech with dozens of words, full coalition enumeration becomes intractable, requiring sampling or partition hierarchies.
> Thank you for your time, and we welcome any questions."
