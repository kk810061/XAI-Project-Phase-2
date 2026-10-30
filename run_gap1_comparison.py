"""Run paired Phase 2 Gap 1 attribution and faithfulness comparisons on FSC."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from speechxai.benchmark_speech import Benchmark
from speechxai.evaluators.faithfulness_measures_speech import (
    AOPC_Comprehensiveness_Evaluation_Speech,
    AOPC_Sufficiency_Evaluation_Speech,
)
from speechxai.explainers.utils_removal import transcribe_audio


MASK_METHODS = ("silence", "noise_matched")
EXPLAINERS = ("LOO", "LIME")
SLOTS = ("action", "object", "location")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="CSV containing one audio path per row")
    parser.add_argument("--audio-column", default="path")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--audio-root", default=None, help="Base directory for relative audio paths")
    parser.add_argument("--checkpoint", default="superb/wav2vec2-base-superb-ic")
    parser.add_argument("--output-dir", default="gap1_results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1000, help="LIME neighborhood size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Continue from existing per-example CSV")
    return parser.parse_args()


def append_rows(path: Path, rows):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def main():
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_example_path = output_dir / "gap1_per_example.csv"

    if per_example_path.exists() and not args.resume:
        raise FileExistsError(
            f"{per_example_path} exists; choose another --output-dir or pass --resume"
        )
    if args.resume and per_example_path.exists():
        previous = pd.read_csv(per_example_path)
        completed = set(
            zip(previous["sample_id"], previous["explainer"], previous["mask_method"])
        )
    else:
        completed = set()

    data = pd.read_csv(input_csv)
    if args.audio_column not in data.columns:
        raise ValueError(f"CSV is missing audio column {args.audio_column!r}")
    if args.limit is not None:
        data = data.head(args.limit)

    audio_root = Path(args.audio_root).resolve() if args.audio_root else input_csv.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.checkpoint)
    benchmark = Benchmark(model, feature_extractor, device=device)
    comprehensiveness = AOPC_Comprehensiveness_Evaluation_Speech(benchmark.model_helper)
    sufficiency = AOPC_Sufficiency_Evaluation_Speech(benchmark.model_helper)

    for row_index, row in data.iterrows():
        raw_path = Path(str(row[args.audio_column]))
        audio_path = raw_path.resolve() if raw_path.is_absolute() else (audio_root / raw_path).resolve()
        sample_id = (
            str(row[args.id_column])
            if args.id_column and args.id_column in data.columns
            else str(row[args.audio_column])
        )
        if all(
            (sample_id, explainer, mask_method) in completed
            for explainer in EXPLAINERS
            for mask_method in MASK_METHODS
        ):
            continue
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio not found for sample {sample_id}: {audio_path}")

        # Align once and reuse identical timestamps for all four attribution runs.
        _, words = transcribe_audio(
            str(audio_path),
            device=benchmark.model_helper.device.type,
            language=benchmark.model_helper.language,
        )

        for explainer in EXPLAINERS:
            for mask_method in MASK_METHODS:
                key = (sample_id, explainer, mask_method)
                if key in completed:
                    continue

                explanation = benchmark.explain(
                    audio_path=str(audio_path),
                    methodology=explainer,
                    removal_type=mask_method,
                    num_samples=args.num_samples,
                    words_trascript=words,
                    seed=args.seed,
                )
                compr = comprehensiveness.compute_evaluation(
                    explanation,
                    words_trascript=words,
                    evaluation_removal_type="silence",
                ).score
                suff = sufficiency.compute_evaluation(
                    explanation,
                    words_trascript=words,
                    evaluation_removal_type="silence",
                ).score

                score_matrix = np.asarray(explanation.scores)
                if score_matrix.ndim == 1:
                    score_matrix = score_matrix[None, :]
                run_rows = []
                for slot_index, slot_scores in enumerate(score_matrix):
                    slot = SLOTS[slot_index] if slot_index < len(SLOTS) else f"label_{slot_index}"
                    if len(slot_scores) == 0:
                        top_word, top_score = "", np.nan
                        ranked_attributions = []
                    else:
                        ranked_indices = np.argsort(np.abs(slot_scores))[::-1]
                        ranked_attributions = [
                            {
                                "word": str(explanation.features[index]),
                                "score": float(slot_scores[index]),
                            }
                            for index in ranked_indices
                            if index < len(explanation.features)
                        ]
                        top_index = int(np.argmax(np.abs(slot_scores)))
                        top_word = (
                            explanation.features[top_index]
                            if top_index < len(explanation.features)
                            else ""
                        )
                        top_score = float(slot_scores[top_index])
                    run_rows.append(
                        {
                            "sample_id": sample_id,
                            "audio_path": str(audio_path),
                            "explainer": explainer,
                            "mask_method": mask_method,
                            "slot": slot,
                            "target_class": int(explanation.target[slot_index]),
                            "comprehensiveness": float(compr[slot_index]),
                            "sufficiency": float(suff[slot_index]),
                            "top_word_by_abs_attribution": str(top_word),
                            "top_word_attribution": top_score,
                            "ranked_word_attributions": json.dumps(ranked_attributions),
                            "num_lime_samples": args.num_samples if explainer == "LIME" else np.nan,
                            "seed": args.seed,
                        }
                    )
                append_rows(per_example_path, run_rows)
                completed.add(key)
                print(f"Saved {explainer}/{mask_method} for {sample_id}")

    results = pd.read_csv(per_example_path)
    summary = (
        results.groupby(["explainer", "slot", "mask_method"])
        .agg(
            n=("sample_id", "nunique"),
            comprehensiveness_mean=("comprehensiveness", "mean"),
            comprehensiveness_std=("comprehensiveness", "std"),
            sufficiency_mean=("sufficiency", "mean"),
            sufficiency_std=("sufficiency", "std"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "gap1_summary_long.csv", index=False)

    comparison = summary.pivot(
        index=["explainer", "slot"],
        columns="mask_method",
        values=[
            "n",
            "comprehensiveness_mean",
            "comprehensiveness_std",
            "sufficiency_mean",
            "sufficiency_std",
        ],
    )
    comparison.columns = [f"{metric}_{method}" for metric, method in comparison.columns]
    comparison = comparison.reset_index()
    comparison["comprehensiveness_delta_noise_minus_silence"] = (
        comparison["comprehensiveness_mean_noise_matched"]
        - comparison["comprehensiveness_mean_silence"]
    )
    comparison["sufficiency_delta_noise_minus_silence"] = (
        comparison["sufficiency_mean_noise_matched"]
        - comparison["sufficiency_mean_silence"]
    )
    comparison.to_csv(output_dir / "gap1_comparison_table.csv", index=False)

    metadata = {
        "checkpoint": args.checkpoint,
        "input_csv": str(input_csv),
        "sample_count": int(data.shape[0]),
        "explainers": list(EXPLAINERS),
        "mask_methods": list(MASK_METHODS),
        "faithfulness_evaluation_mask": "silence",
        "lime_num_samples": args.num_samples,
        "seed": args.seed,
        "noise_estimator": "10th percentile of 20 ms frame RMS per recording",
        "crossfade_ms": 20,
    }
    (output_dir / "gap1_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Comparison table: {output_dir / 'gap1_comparison_table.csv'}")


if __name__ == "__main__":
    main()
