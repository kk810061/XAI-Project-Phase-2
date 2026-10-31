# Phase 2, Gap 4: evaluator-mask sensitivity

## Question

Do comprehensiveness and sufficiency results change when the evaluation-time removal mask changes from hard silence to noise matched, and does that effect depend on the mask used to generate the attribution?

## Design

This is a crossed comparison of:

- Explainer: LOO and LIME.
- Attribution-generation mask: hard silence and noise matched.
- Faithfulness-evaluation mask: hard silence and noise matched.

Each explanation is generated once per explainer and attribution mask, then evaluated under both evaluator masks. The same aligned word timestamps, attribution scores, target classes, and LIME seed are used in both evaluations. This separates the effect of generating an explanation with a replacement method from the effect of evaluating it with that replacement method.

The evaluator masks replace the same rationale words using `remove_specified_words`; noise-matched evaluator masking uses the recording's estimated noise RMS and the existing 20 ms crossfades. The attribution-side LOO and LIME interval definitions remain those used in Gap 1.

## Run

Start with a 10-example pilot:

```bash
python run_gap4_comparison.py \
  data/fsc/fluent_speech_commands_dataset/data/test_data.csv \
  --audio-column path \
  --audio-root data/fsc/fluent_speech_commands_dataset \
  --output-dir "Final results XAI/gap4_pilot" \
  --num-samples 100 \
  --seed 42 \
  --limit 10
```

If the pilot completes, continue to the full test set in the same output directory, removing `--limit 10` and adding `--resume`:

```bash
python run_gap4_comparison.py \
  data/fsc/fluent_speech_commands_dataset/data/test_data.csv \
  --audio-column path \
  --audio-root data/fsc/fluent_speech_commands_dataset \
  --output-dir "Final results XAI/gap4_pilot" \
  --num-samples 100 \
  --seed 42 \
  --resume
```

The pilot output directory can be continued into the full run because completed samples are checkpointed. Use the same checkpoint, seed, input, and LIME sample count when resuming. A run interrupted during attribution can reuse its cached raw explanation; an interruption during evaluation does not require recomputing completed evaluation rows.

## Outputs

- `gap4_per_example.csv`: one row per sample, explainer, attribution mask, evaluator mask, and output slot.
- `gap4_summary_long.csv`: means and standard deviations by condition and slot.
- `gap4_comparison_table.csv`: silence-evaluator versus noise-matched-evaluator scores and deltas for each fixed attribution condition.
- `gap4_explanations.jsonl`: cached raw feature-order scores and aligned words, allowing exact rescoring and resume.
- `gap4_run_metadata.json`: input, checkpoint, seed, and experiment settings.

The Gap 1 per-example CSV stores ranked attributions rather than the original feature-order score vectors, so Gap 4 regenerates each explanation. The Gap 4 runner caches those vectors as it goes to avoid repeating completed LIME work after interruptions.

Before interpreting the Gap 4 results, compare its evaluator-silence rows with the corresponding Gap 1 results for the same sample, explainer, and attribution mask. With matching checkpoint, seed, and LIME sample count, those scores should match (allowing only negligible floating-point variation). A material mismatch is a cue to check the run settings or environment before proceeding.

## Interpretation

Compare evaluator masks within each fixed explainer and attribution-generation mask. Higher comprehensiveness and lower sufficiency are better. Differences between LOO and LIME also reflect their distinct word-span definitions, so they should not be attributed solely to the explainer algorithm.
