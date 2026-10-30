# Phase 2, Gap 1: Noise-matched word masking

## Question

Does noise-matched masking change the LOO word attributions or their faithfulness compared with the project's current hard-silence mask?

## Implemented methods

- `silence`: the existing baseline. Replace the aligned word plus 100 ms before and 40 ms after it with same-length digital silence.
- `noise_matched`: replace that same interval with generated, zero-mean Gaussian noise. Its RMS level is estimated from the 10th percentile of 20 ms frame RMS values in the original recording. Estimate this once per recording and reuse it for all words. Apply 20 ms cosine fades at both interval boundaries. The generated signal is deterministic for a given recording interval; the base seed is 42.

Only the noise level is estimated from the input. No audio frames are copied into the replacement, so the masked region does not contain donor speech. The percentile estimate can include low-energy speech in unusual recordings; the generated noise still does not contain that speech.

The mask interval is clipped to the recording boundaries. The waveform outside the interval is retained. Output duration and sample rate are preserved. Audio is converted to 16-bit PCM for generated-noise replacement. LOO uses the existing expanded word interval (100 ms before, 40 ms after); LIME uses its existing exact word segment boundaries, so compare the two mask types within each explainer.

## Controlled comparison

Use the same checkpoint, FSC examples, word alignments, target predictions, attribution method, and evaluator settings for both attribution runs. The runner supports both LOO and LIME. For LIME, it resets the seed before each masking method so the two methods use the same coalition samples. Keep the evaluator fixed to hard silence during this first comparison; this tests the attribution mask while holding evaluation constant. Gap 4 will later vary the evaluator mask.

```powershell
python run_gap1_comparison.py path\to\test_data.csv `
  --audio-column path `
  --audio-root path\to\FSC `
  --output-dir "Final results XAI\gap1" `
  --num-samples 1000 `
  --seed 42
```

The input CSV must contain an audio path column. Relative paths resolve against `--audio-root` (or the CSV's directory if omitted). Use `--limit 10` for a small pilot, then remove the limit for the full run. LIME is run for both masking methods and may dominate runtime. The runner checkpoints each completed sample/explainer/mask combination; resume an interrupted run with `--resume` and the same arguments.

The run writes:

- `gap1_per_example.csv`: per-sample, per-slot faithfulness scores and top word.
- `gap1_summary_long.csv`: means and standard deviations by explainer, slot, and mask.
- `gap1_comparison_table.csv`: hard-silence versus noise-matched means, standard deviations, and deltas.
- `gap1_run_metadata.json`: checkpoint and experiment settings.

## Results to collect

For every example and each FSC output slot (action, object, location), the runner saves the method, sample ID, target class, comprehensiveness, sufficiency, top-attributed word, and attribution score. Report paired means and standard deviations across the same examples. The saved per-example data also allows reporting how often the top-ranked word changes. Use a fixed sample subset and record its IDs, checkpoint, alignment source, noise settings, and seed.

Include a few waveform or spectrogram plots to illustrate the replacement and fades. These plots are implementation evidence; the paired faithfulness results are the main comparison.

## Scope and caveat

This stage implements and evaluates the LOO and LIME masks. It does not change the faithfulness evaluator's default behavior; `evaluation_removal_type="silence"` is an explicit override for controlled comparison. Noise-matched replacement is a masking alternative, not a claim that the generated audio is natural or fully in-distribution.
