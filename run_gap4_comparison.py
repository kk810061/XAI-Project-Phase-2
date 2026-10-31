"""Run the crossed Gap 4 attribution-mask x evaluator-mask comparison on FSC."""

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
from speechxai.explainers.explanation_speech import ExplanationSpeech
from speechxai.explainers.utils_removal import transcribe_audio


ATTRIBUTION_MASKS = ("silence", "noise_matched")
EVALUATION_MASKS = ("silence", "noise_matched")
EXPLAINERS = ("LOO", "LIME")
SLOTS = ("action", "object", "location")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="CSV containing one audio path per row")
    parser.add_argument("--audio-column", default="path")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--audio-root", default=None, help="Base directory for relative audio paths")
    parser.add_argument("--checkpoint", default="superb/wav2vec2-base-superb-ic")
    parser.add_argument("--output-dir", default="gap4_results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1000, help="LIME neighborhood size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from cached explanations and per-example scores")
    return parser.parse_args()


def append_rows(path: Path, rows):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def json_safe_words(words):
    safe = []
    for word in words:
        safe.append(
            {
                "word": str(word.get("word", "")),
                "start": float(word["start"]),
                "end": float(word["end"]),
            }
        )
    return safe


def load_explanation_cache(path: Path):
    cache = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = (item["sample_id"], item["explainer"], item["attribution_mask"])
            cache[key] = item
    return cache


def load_completed_rows(path: Path):
    completed = set()
    if not path.exists():
        return completed
    previous = pd.read_csv(path)
    completed.update(
        zip(
            previous["sample_id"].astype(str),
            previous["explainer"],
            previous["attribution_mask"],
            previous["evaluation_mask"],
            previous["slot"],
        )
    )
    return completed


def append_explanation_cache(path: Path, key, item):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
    key_value = (item["sample_id"], item["explainer"], item["attribution_mask"])
    key[key_value] = item


def make_explanation(item):
    return ExplanationSpeech(
        features=item["features"],
        scores=np.asarray(item["scores"], dtype=float),
        explainer=item["explanation_name"],
        target=item["target"],
        audio_path=item["audio_path"],
    )


def config_for(args, input_csv: Path, audio_root: Path):
    return {
        "input_csv": str(input_csv),
        "audio_root": str(audio_root),
        "audio_column": args.audio_column,
        "id_column": args.id_column,
        "checkpoint": args.checkpoint,
        "lime_num_samples": args.num_samples,
        "seed": args.seed,
        "explainers": list(EXPLAINERS),
        "attribution_masks": list(ATTRIBUTION_MASKS),
        "evaluation_masks": list(EVALUATION_MASKS),
    }


def main():
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_example_path = output_dir / "gap4_per_example.csv"
    explanation_cache_path = output_dir / "gap4_explanations.jsonl"
    metadata_path = output_dir / "gap4_run_metadata.json"

    data = pd.read_csv(input_csv)
    if args.audio_column not in data.columns:
        raise ValueError(f"CSV is missing audio column {args.audio_column!r}")
    if args.limit is not None:
        data = data.head(args.limit)
    audio_root = Path(args.audio_root).resolve() if args.audio_root else input_csv.parent
    config = config_for(args, input_csv, audio_root)

    if metadata_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"{output_dir} already contains a Gap 4 run; choose another --output-dir or pass --resume"
            )
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_config = previous_metadata.get("config")
        if previous_config != config:
            raise ValueError(
                "Existing Gap 4 output was created with different settings. "
                "Choose a new --output-dir or restore the original settings before --resume."
            )
    elif (per_example_path.exists() or explanation_cache_path.exists()) and not args.resume:
        raise FileExistsError(
            f"Gap 4 files already exist in {output_dir}; choose another --output-dir or pass --resume"
        )

    metadata = {
        "experiment": "Gap 4 crossed attribution-mask and evaluator-mask comparison",
        "sample_count": int(data.shape[0]),
        "config": config,
        "faithfulness_metrics": ["comprehensiveness", "sufficiency"],
        "cache_note": "Raw feature-order explanations and word alignments are cached for exact evaluator-mask rescoring and resume.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    explanation_cache = load_explanation_cache(explanation_cache_path)
    completed = load_completed_rows(per_example_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.checkpoint)
    benchmark = Benchmark(model, feature_extractor, device=device)
    comprehensiveness = AOPC_Comprehensiveness_Evaluation_Speech(benchmark.model_helper)
    sufficiency = AOPC_Sufficiency_Evaluation_Speech(benchmark.model_helper)

    for _, row in data.iterrows():
        raw_path = Path(str(row[args.audio_column]))
        audio_path = raw_path.resolve() if raw_path.is_absolute() else (audio_root / raw_path).resolve()
        sample_id = (
            str(row[args.id_column])
            if args.id_column and args.id_column in data.columns
            else str(row[args.audio_column])
        )
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio not found for sample {sample_id}: {audio_path}")

        for explainer_name in EXPLAINERS:
            for attribution_mask in ATTRIBUTION_MASKS:
                explanation_key = (sample_id, explainer_name, attribution_mask)
                missing_eval_masks = [
                    evaluation_mask
                    for evaluation_mask in EVALUATION_MASKS
                    if any(
                        (sample_id, explainer_name, attribution_mask, evaluation_mask, slot)
                        not in completed
                        for slot in SLOTS
                    )
                ]
                if not missing_eval_masks:
                    continue

                item = explanation_cache.get(explanation_key)
                if item is None:
                    _, words = transcribe_audio(
                        str(audio_path),
                        device=benchmark.model_helper.device.type,
                        language=benchmark.model_helper.language,
                    )
                    explanation = benchmark.explain(
                        audio_path=str(audio_path),
                        methodology=explainer_name,
                        removal_type=attribution_mask,
                        num_samples=args.num_samples,
                        words_trascript=words,
                        seed=args.seed,
                    )
                    item = {
                        "sample_id": sample_id,
                        "audio_path": str(audio_path),
                        "explainer": explainer_name,
                        "attribution_mask": attribution_mask,
                        "explanation_name": explanation.explainer,
                        "features": [str(feature) for feature in explanation.features],
                        "scores": np.asarray(explanation.scores).tolist(),
                        "target": [int(target) for target in explanation.target],
                        "words": json_safe_words(words),
                    }
                    append_explanation_cache(explanation_cache_path, explanation_cache, item)
                else:
                    explanation = make_explanation(item)

                for evaluation_mask in missing_eval_masks:
                    explanation = make_explanation(item)
                    words = item["words"]
                    compr = comprehensiveness.compute_evaluation(
                        explanation,
                        words_trascript=words,
                        evaluation_removal_type=evaluation_mask,
                    ).score
                    suff = sufficiency.compute_evaluation(
                        explanation,
                        words_trascript=words,
                        evaluation_removal_type=evaluation_mask,
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

                        row_key = (
                            sample_id,
                            explainer_name,
                            attribution_mask,
                            evaluation_mask,
                            slot,
                        )
                        if row_key in completed:
                            continue
                        run_rows.append(
                            {
                                "sample_id": sample_id,
                                "audio_path": str(audio_path),
                                "explainer": explainer_name,
                                "attribution_mask": attribution_mask,
                                "evaluation_mask": evaluation_mask,
                                "slot": slot,
                                "target_class": int(explanation.target[slot_index]),
                                "comprehensiveness": float(compr[slot_index]),
                                "sufficiency": float(suff[slot_index]),
                                "top_word_by_abs_attribution": str(top_word),
                                "top_word_attribution": top_score,
                                "ranked_word_attributions": json.dumps(ranked_attributions),
                                "num_lime_samples": args.num_samples if explainer_name == "LIME" else np.nan,
                                "seed": args.seed,
                            }
                        )
                        completed.add(row_key)
                    append_rows(per_example_path, run_rows)
                    print(
                        f"Saved {explainer_name}/{attribution_mask} attribution "
                        f"with {evaluation_mask} evaluation for {sample_id}"
                    )

    results = pd.read_csv(per_example_path)
    summary = (
        results.groupby(["explainer", "attribution_mask", "evaluation_mask", "slot"])
        .agg(
            n=("sample_id", "nunique"),
            comprehensiveness_mean=("comprehensiveness", "mean"),
            comprehensiveness_std=("comprehensiveness", "std"),
            sufficiency_mean=("sufficiency", "mean"),
            sufficiency_std=("sufficiency", "std"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "gap4_summary_long.csv", index=False)

    comparison = summary.pivot(
        index=["explainer", "attribution_mask", "slot"],
        columns="evaluation_mask",
        values=[
            "n",
            "comprehensiveness_mean",
            "comprehensiveness_std",
            "sufficiency_mean",
            "sufficiency_std",
        ],
    )
    comparison.columns = [f"{metric}_{mask}" for metric, mask in comparison.columns]
    comparison = comparison.reset_index()
    comparison["comprehensiveness_delta_noise_eval_minus_silence_eval"] = (
        comparison["comprehensiveness_mean_noise_matched"]
        - comparison["comprehensiveness_mean_silence"]
    )
    comparison["sufficiency_delta_noise_eval_minus_silence_eval"] = (
        comparison["sufficiency_mean_noise_matched"]
        - comparison["sufficiency_mean_silence"]
    )
    comparison.to_csv(output_dir / "gap4_comparison_table.csv", index=False)
    metadata["completed_sample_count"] = int(results["sample_id"].nunique())
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Comparison table: {output_dir / 'gap4_comparison_table.csv'}")


if __name__ == "__main__":
    main()
