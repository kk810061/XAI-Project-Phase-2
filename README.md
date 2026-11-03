# SpeechXAI: Phase 2 research extensions

This repository reproduces selected word-level speech explanation methods from
[Pastor et al. (EACL 2024)](https://aclanthology.org/2024.eacl-long.136/) and
contains two Phase 2 extensions: noise-matched masking (Gap 1) and word-level
KernelSHAP with a controlled seed-stability comparison against LIME.

The original paper is included as [`paper.pdf`](paper.pdf). The submitted
experimental outputs are in [`Final results XAI/`](Final%20results%20XAI/).
The Phase 2 comparison report is
[`phase2_experimental_results_and_comparison_reformatted.pdf`](output/pdf/phase2_experimental_results_and_comparison_reformatted.pdf).

## Phase 2 findings

### Gap 1: noise-matched masking

The original masking baseline replaces a word span with hard silence. The
extension replaces the same span with deterministic, zero-mean Gaussian noise
whose level is estimated from the recording's quiet frames, with 20 ms fades.
The comparison holds the faithfulness evaluator at hard silence to isolate the
effect of the attribution-generation mask. Both LOO and LIME are evaluated on
the FSC test set.

The saved comparison is in `Final results XAI/Gap 1/gap1_comparison_table.csv`.
Noise-matched masking changes the faithfulness scores; it does not uniformly
improve them. Interpret comprehensiveness and sufficiency separately (higher
comprehensiveness and lower sufficiency are generally preferred).

### WA-SHAP seed stability

WA-SHAP uses aligned words as features and KernelSHAP over word coalitions. Its
stability was compared with a separate, controlled, word-only LIME rerun. Both
methods use the same 97 valid utterances, five seeds, three output slots,
Spearman rank correlation, and top-word agreement based on the unique largest
absolute attribution (ties and one-word examples are excluded).

| Method | Mean rank stability (95% CI half-width) | Absolute-top-word agreement (95% CI half-width) |
| --- | ---: | ---: |
| Controlled word-only WA-LIME | 0.873 (0.018) | 87.8% (2.7%) |
| WA-SHAP | 0.952 (0.022) | 95.1% (2.6%) |

These results support the limited conclusion that WA-SHAP was more stable on
this sampled comparison; they do not establish that it is universally more
stable. The four relevant files are in `Final results XAI/SHAP instability/`:

- `stability_summary.csv`: final matched means and confidence intervals.
- `stability_by_sample.csv`: matched metrics for each of the 97 utterances.
- `lime_pairs.csv`: pairwise results from the separate controlled LIME run.
- `shap_vectors.csv`: WA-SHAP attribution vectors used to compute its matched
  stability metrics.

Older results with a different top-word rule or feature protocol are kept in
`Final results XAI/SHAP instability/old_protocol/`; they are not used in the
table above.

## Research gap, implementation changes, and scope

### Why study LIME stability?

Explanation stability is an important quality criterion: similar inputs
should produce similar explanations, and Alvarez-Melis and Jaakkola show that
popular post-hoc explanation methods can be fragile under small input changes
([*On the Robustness of Interpretability Methods*, 2018](https://arxiv.org/abs/1806.08049)).
Our seed experiment tests a narrower, reproducible question: whether the
word-level rankings change when the explainer's random seed changes on the
same utterances. It is evidence about this dataset and protocol, not a general
proof about all LIME or SHAP use.

### Implementation changes

- **Gap 1 (noise-matched masking):** `speechxai/explainers/utils_removal.py`
  implements the noise replacement; `phase2/run_gap1_comparison.py` runs the
  paired LOO/LIME comparison.
- **Controlled word-only LIME stability:** `phase2/run_controlled_lime.py`
  generates LIME explanations over aligned words only, without empty-span
  features, for the five-seed comparison.
- **WA-SHAP and evaluation:** `speechxai/explainers/shap_speech_explainer.py`
  implements the word-coalition explainer; `phase2/run_shap_comparison.py`
  runs the explanation experiment; `phase2/build_stability_comparison.py`
  computes the matched LIME/SHAP stability metrics and summary.

### What the results do and do not establish

In the matched, 97-valid-utterance stability experiment, WA-SHAP has higher
mean rank correlation and top-word agreement than controlled word-only
WA-LIME. This supports the limited conclusion that the implemented WA-SHAP
was more stable than this controlled LIME run under the tested seeds and
protocol. The proposed change addresses the observed LIME instability
concern; it does **not** resolve the sufficiency drop/trade-off. The current
full-test faithfulness table reports aggregate methods from separate saved
runs, and should not be described as a fully matched word-only LIME-versus-SHAP
comparison.

The Phase 1/full-test LIME run does not need to be repeated merely to have
full-test LIME results. A new run is needed only if making the narrower
full-test faithfulness claim that LIME and SHAP used identical word-only
features and evaluation settings: the standard LIME implementation in
`speechxai/explainers/lime_speech_explainer.py` inserts empty spans as
features, whereas the controlled stability runner above uses words only.
Keep the 100-utterance stability experiment distinct from full-test
faithfulness evaluation. The research-gap claim in this project is limited to
LIME explanation instability; no claim is made about other limitations of LOO.

## Setup

Use Python 3.10 or newer. Install dependencies with:

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The experiments require the Fluent Speech Commands (FSC) dataset and download
the `superb/wav2vec2-base-superb-ic` checkpoint through Hugging Face. Place FSC
at `data/fsc/fluent_speech_commands_dataset/`; its test CSV is expected at
`data/fsc/fluent_speech_commands_dataset/data/test_data.csv`. Each experiment
runner accepts an alternate CSV and audio root. WhisperX supplies word-level
alignment; a compatible CPU or CUDA PyTorch installation may be used.

## Run the experiments

Start with a small pilot by adding `--limit 10`; remove the limit for the full
test split. The FSC test split used by the saved results contains 3,793
examples.

Gap 1 paired comparison (silence vs. noise-matched attribution masks):

```bash
python phase2/run_gap1_comparison.py data/fsc/fluent_speech_commands_dataset/data/test_data.csv \
  --audio-column path \
  --audio-root data/fsc/fluent_speech_commands_dataset \
  --output-dir "output/reproduction/gap1" \
  --num-samples 1000 --seed 42
```

Separate controlled word-only LIME stability run:

```bash
python phase2/run_controlled_lime.py \
  --input-csv data/fsc/fluent_speech_commands_dataset/data/test_data.csv \
  --audio-root data/fsc/fluent_speech_commands_dataset \
  --output-csv "output/reproduction/lime_pairs.csv" \
  --limit 100 --sample-seed 42 --seeds 42,43,44,45,46 --num-samples 100
```

To run the WA-SHAP stability experiment, use `phase2/run_shap_comparison.py` on the
same 100-example subset, with `--methods LIME,SHAP --skip-faithfulness
--sample-seed 42 --seeds 42,43,44,45,46 --num-samples 100` and
`--output-dir output/reproduction/shap`. Keep these raw rerun files separate
from the curated submission results. The final `shap_vectors.csv` contains the
WA-SHAP rows from that run. The comparison builder reads the curated `lime_pairs.csv`
and `shap_vectors.csv`, then rebuilds the matched per-sample results and
summary:

```bash
python phase2/build_stability_comparison.py
```

The runners checkpoint completed samples; use `--resume` with the same inputs
and settings to continue interrupted runs. Keep the checkpoint, sample IDs,
seeds, alignment, feature definition, and mask settings fixed when comparing
methods.

## Source layout

- `phase2/run_gap1_comparison.py`: Gap 1 paired evaluation runner.
- `phase2/run_controlled_lime.py`: separate word-only LIME stability run.
- `phase2/run_shap_comparison.py`: WA-SHAP and multi-seed attribution runner.
- `phase2/build_stability_comparison.py`: matched LIME/SHAP metrics and summary.
- `speechxai/`: explanation, masking, model, and faithfulness implementations.
- `Final results XAI/`: saved Phase 1 and Phase 2 result tables.
- `docs/images/`: visual examples from the original project.

## Citation

The research-gap discussion also draws on these supporting articles:

- Alvarez-Melis, D. and Jaakkola, T. S. (2018). [On the Robustness of
  Interpretability Methods](https://arxiv.org/abs/1806.08049).

```bibtex
@inproceedings{pastor-etal-2024-explaining,
  title = {Explaining Speech Classification Models via Word-Level Audio Segments and Paralinguistic Features},
  author = {Pastor, Eliana and Koudounas, Alkis and Attanasio, Giuseppe and Hovy, Dirk and Baralis, Elena},
  booktitle = {Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics},
  year = {2024},
  pages = {2221--2238},
  url = {https://aclanthology.org/2024.eacl-long.136/}
}
```
