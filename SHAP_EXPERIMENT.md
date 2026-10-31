# Phase 2: Word-Level KernelSHAP

## Research question

Can word-level KernelSHAP produce explanations that are at least as faithful as the paper's WA-LIME and WA-LOO explanations, while being more stable across sampling seeds?

`WA-SHAP` treats each WhisperX-aligned word as a binary feature. A coalition retains its present words and replaces every absent word segment with hard silence. It estimates the Shapley values of the original model's predicted action, object, and location probabilities through KernelSHAP's weighted linear regression.

The implementation enumerates all coalitions whenever they fit the requested budget. These examples have exact, deterministic Shapley values. For longer utterances, it samples coalitions using the KernelSHAP kernel and therefore remains an approximation whose stability should be measured.

## Controlled comparison

WA-SHAP uses the same FSC examples, checkpoint, WhisperX alignment, predicted target class, and faithfulness evaluator as the Phase 1 experiment.

| Method | Attribution procedure | Attribution mask | Evaluation mask |
| --- | --- | --- | --- |
| WA-LOO | Mask one word at a time | hard silence | hard silence |
| WA-LIME | Local linear surrogate over sampled word coalitions | hard silence | hard silence |
| WA-SHAP | KernelSHAP over word coalitions | hard silence | hard silence |

Use the existing Phase 1 Table 4 rows for WA-LOO and WA-LIME if their checkpoint, test split, hard-silence evaluator, and LIME budget match this experiment. The full-split default is therefore `--methods SHAP`; the runner automatically appends the new WA-SHAP row to `Final results XAI/table4_faithfulness_full_test.csv` when that file is available.

## Pilot run

```bash
python run_shap_comparison.py \
  data/fsc/fluent_speech_commands_dataset/data/test_data.csv \
  --audio-column path \
  --audio-root data/fsc/fluent_speech_commands_dataset \
  --output-dir "Final results XAI/shap_pilot" \
  --num-samples 100 \
  --seeds 42 \
  --methods SHAP \
  --limit 10
```

## Full FSC Table 4 comparison

Remove `--limit`. This produces `table4_shap_comparison.csv`, with the same action, object, and location comprehensiveness and sufficiency layout as the FSC portion of Table 4.

```bash
python run_shap_comparison.py \
  data/fsc/fluent_speech_commands_dataset/data/test_data.csv \
  --audio-column path \
  --audio-root data/fsc/fluent_speech_commands_dataset \
  --output-dir "Final results XAI/shap_full_test" \
  --num-samples 100 \
  --seeds 42 \
  --methods SHAP
```

Resume an interrupted run by repeating the command with `--resume`.

## Stability experiment

The Phase 1 faithfulness table cannot measure seed instability because it contains one aggregate LIME result, not multiple attribution vectors. Run LIME and SHAP only on a fixed representative subset across five seeds. `--skip-faithfulness` avoids unnecessary AOPC evaluation model calls; this experiment compares attribution vectors only.

```bash
python run_shap_comparison.py \
  data/fsc/fluent_speech_commands_dataset/data/test_data.csv \
  --audio-column path \
  --audio-root data/fsc/fluent_speech_commands_dataset \
  --output-dir "Final results XAI/shap_stability" \
  --num-samples 100 \
  --seeds 42,43,44,45,46 \
  --methods LIME,SHAP \
  --skip-faithfulness \
  --sample-seed 42 \
  --limit 100
```

Start stability testing on a fixed representative subset. It is optional evidence for the seed-stability hypothesis and does not replace the full-split Table 4 comparison.

## Outputs

- `shap_per_example.csv`: attribution vectors, target classes, and faithfulness scores per example and output slot.
- `shap_summary_long.csv`: full-split mean and standard deviation results for the primary seed.
- `table4_shap_comparison.csv`: Table 4 style comparison of WA-LOO, WA-LIME, and WA-SHAP.
- `shap_stability_pairs.csv` and `shap_stability_summary.csv`: seed stability results when two or more seeds are supplied.
- `shap_run_metadata.json`: checkpoint, seeds, budgets, and masking protocol.
