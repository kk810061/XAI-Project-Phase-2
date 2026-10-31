"""Compare paper-faithful WA-LOO and WA-LIME with word-level WA-SHAP on FSC."""

import argparse
import itertools
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


EXPLAINERS = ("LOO", "LIME", "SHAP")
SLOTS = ("action", "object", "location")
DISPLAY_NAMES = {"LOO": "WA-LOO", "LIME": "WA-LIME", "SHAP": "WA-SHAP"}
RAW_NAMES = {display: raw for raw, display in DISPLAY_NAMES.items()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="CSV containing one audio path per row")
    parser.add_argument("--audio-column", default="path")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--audio-root", default=None, help="Base directory for relative audio paths")
    parser.add_argument("--checkpoint", default="superb/wav2vec2-base-superb-ic")
    parser.add_argument("--output-dir", default="shap_results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Randomly select --limit examples with this seed instead of taking the first rows.",
    )
    parser.add_argument(
        "--methods",
        default="SHAP",
        help="Comma-separated methods to run: LOO, LIME, SHAP. Default: SHAP only.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Total LIME neighbourhood samples and total SHAP coalitions",
    )
    parser.add_argument(
        "--skip-faithfulness",
        action="store_true",
        help="Save only attribution vectors. Use this for the smaller multi-seed stability study.",
    )
    parser.add_argument(
        "--baseline-table",
        default="Final results XAI/table4_faithfulness_full_test.csv",
        help="Existing Phase 1 Table 4 CSV to merge with the new WA-SHAP row.",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated seeds. Extra seeds enable the optional stability report.",
    )
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run")
    return parser.parse_args()


def append_rows(path: Path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def parse_seeds(value: str):
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return list(dict.fromkeys(seeds))


def parse_methods(value: str):
    methods = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not methods or any(method not in EXPLAINERS for method in methods):
        raise ValueError("--methods must contain one or more of: LOO, LIME, SHAP")
    return list(dict.fromkeys(methods))


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
        start = end
    return ranks


def spearman_correlation(left, right):
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_std = left_rank.std()
    right_std = right_rank.std()
    if left_std == 0 or right_std == 0:
        return float(np.array_equal(left_rank, right_rank))
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def build_table4(summary: pd.DataFrame):
    columns = [
        "Method", "Compr (action) [^]", "Suff (action) [v]",
        "Compr (object) [^]", "Suff (object) [v]",
        "Compr (location) [^]", "Suff (location) [v]",
    ]
    rows = []
    for method, group in summary.groupby("method"):
        by_slot = group.set_index("slot")
        rows.append({
            "Method": method,
            "Compr (action) [^]": by_slot.loc["action", "comprehensiveness_mean"],
            "Suff (action) [v]": by_slot.loc["action", "sufficiency_mean"],
            "Compr (object) [^]": by_slot.loc["object", "comprehensiveness_mean"],
            "Suff (object) [v]": by_slot.loc["object", "sufficiency_mean"],
            "Compr (location) [^]": by_slot.loc["location", "comprehensiveness_mean"],
            "Suff (location) [v]": by_slot.loc["location", "sufficiency_mean"],
        })
    return pd.DataFrame(rows, columns=columns)


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    selected_methods = parse_methods(args.methods)
    primary_seed = seeds[0]
    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_example_path = output_dir / "shap_per_example.csv"

    if per_example_path.exists() and not args.resume:
        raise FileExistsError(f"{per_example_path} exists; choose another --output-dir or pass --resume")
    if args.resume and per_example_path.exists():
        previous = pd.read_csv(per_example_path)
        completed = set(
            (sample_id, RAW_NAMES[method], int(seed))
            for sample_id, method, seed in zip(
                previous["sample_id"], previous["method"], previous["seed"]
            )
        )
    else:
        completed = set()

    data = pd.read_csv(input_csv)
    if args.audio_column not in data.columns:
        raise ValueError(f"CSV is missing audio column {args.audio_column!r}")
    if args.limit is not None:
        if args.sample_seed is None:
            data = data.head(args.limit)
        else:
            data = data.sample(n=min(args.limit, len(data)), random_state=args.sample_seed)

    audio_root = Path(args.audio_root).resolve() if args.audio_root else input_csv.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2ForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.checkpoint)
    benchmark = Benchmark(model, feature_extractor, device=device)
    comprehensiveness = AOPC_Comprehensiveness_Evaluation_Speech(benchmark.model_helper)
    sufficiency = AOPC_Sufficiency_Evaluation_Speech(benchmark.model_helper)

    for _, row in data.iterrows():
        raw_path = Path(str(row[args.audio_column]))
        audio_path = raw_path.resolve() if raw_path.is_absolute() else (audio_root / raw_path).resolve()
        sample_id = str(row[args.id_column]) if args.id_column and args.id_column in data.columns else str(row[args.audio_column])
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio not found for sample {sample_id}: {audio_path}")

        required_runs = [
            (method, primary_seed if method == "LOO" else seed)
            for method in selected_methods
            for seed in (seeds[:1] if method == "LOO" else seeds)
        ]
        if all((sample_id, method, seed) in completed for method, seed in required_runs):
            continue

        # One alignment is reused across all explainers and seeds.
        _, words = transcribe_audio(
            str(audio_path),
            device=benchmark.model_helper.device.type,
            language=benchmark.model_helper.language,
        )
        for method, seed in required_runs:
            key = (sample_id, method, seed)
            if key in completed:
                continue
            explanation = benchmark.explain(
                audio_path=str(audio_path),
                methodology=method,
                removal_type="silence",
                num_samples=args.num_samples,
                words_trascript=words,
                seed=seed,
            )
            if args.skip_faithfulness:
                compr = [np.nan] * len(explanation.target)
                suff = [np.nan] * len(explanation.target)
            else:
                compr = comprehensiveness.compute_evaluation(
                    explanation, words_trascript=words, evaluation_removal_type="silence"
                ).score
                suff = sufficiency.compute_evaluation(
                    explanation, words_trascript=words, evaluation_removal_type="silence"
                ).score
            score_matrix = np.atleast_2d(np.asarray(explanation.scores, dtype=float))
            rows = []
            for slot_index, slot_scores in enumerate(score_matrix):
                slot = SLOTS[slot_index] if slot_index < len(SLOTS) else f"label_{slot_index}"
                if len(slot_scores):
                    top_index = int(np.argmax(slot_scores))
                    top_word = str(explanation.features[top_index])
                    top_score = float(slot_scores[top_index])
                else:
                    top_word, top_score = "", np.nan
                rows.append(
                    {
                        "sample_id": sample_id,
                        "audio_path": str(audio_path),
                        "method": DISPLAY_NAMES[method],
                        "slot": slot,
                        "target_class": int(explanation.target[slot_index]),
                        "comprehensiveness": float(compr[slot_index]),
                        "sufficiency": float(suff[slot_index]),
                        "top_word": top_word,
                        "top_word_attribution": top_score,
                        "word_attributions": json.dumps([float(score) for score in slot_scores]),
                        "features": json.dumps([str(feature) for feature in explanation.features]),
                        "seed": seed,
                        "num_samples": args.num_samples if method in ("LIME", "SHAP") else np.nan,
                        "coalition_count": getattr(explanation, "coalition_count", np.nan),
                        "exact_kernel_shap": getattr(explanation, "exact_kernel_shap", np.nan),
                    }
                )
            append_rows(per_example_path, rows)
            completed.add(key)
            print(f"Saved {DISPLAY_NAMES[method]} (seed {seed}) for {sample_id}")

    results = pd.read_csv(per_example_path)
    primary = results[results["seed"] == primary_seed].copy()
    summary = (
        primary.groupby(["method", "slot"])
        .agg(
            n=("sample_id", "nunique"),
            comprehensiveness_mean=("comprehensiveness", "mean"),
            comprehensiveness_std=("comprehensiveness", "std"),
            sufficiency_mean=("sufficiency", "mean"),
            sufficiency_std=("sufficiency", "std"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "shap_summary_long.csv", index=False)
    new_rows = build_table4(summary)
    baseline_path = Path(args.baseline_table)
    if not args.skip_faithfulness and baseline_path.is_file():
        baseline = pd.read_csv(baseline_path)
        table4 = pd.concat([baseline, new_rows], ignore_index=True)
    else:
        table4 = new_rows
    table4.to_csv(output_dir / "table4_shap_comparison.csv", index=False)

    stability_rows = []
    for (method, sample_id, slot), group in results[results["method"].isin(["WA-LIME", "WA-SHAP"])].groupby(["method", "sample_id", "slot"]):
        vectors = [(int(row.seed), np.asarray(json.loads(row.word_attributions), dtype=float), row.top_word) for row in group.itertuples()]
        for (_, left_scores, left_top), (_, right_scores, right_top) in itertools.combinations(vectors, 2):
            stability_rows.append({
                "method": method,
                "sample_id": sample_id,
                "slot": slot,
                "spearman_rank_correlation": spearman_correlation(left_scores, right_scores),
                "top_word_agreement": float(left_top == right_top),
            })
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(output_dir / "shap_stability_pairs.csv", index=False)
    if not stability.empty:
        stability_summary = (
            stability.groupby("method")
            .agg(
                paired_comparisons=("sample_id", "size"),
                mean_spearman_rank_correlation=("spearman_rank_correlation", "mean"),
                mean_top_word_agreement=("top_word_agreement", "mean"),
            )
            .reset_index()
        )
    else:
        stability_summary = pd.DataFrame(columns=["method", "paired_comparisons", "mean_spearman_rank_correlation", "mean_top_word_agreement"])
    stability_summary.to_csv(output_dir / "shap_stability_summary.csv", index=False)

    metadata = {
        "checkpoint": args.checkpoint,
        "input_csv": str(input_csv),
        "sample_count": int(data.shape[0]),
        "sample_seed": args.sample_seed,
        "methods": [DISPLAY_NAMES[method] for method in selected_methods],
        "primary_seed": primary_seed,
        "stability_seeds": seeds,
        "lime_num_samples": args.num_samples,
        "shap_num_coalitions": args.num_samples,
        "attribution_mask": "silence",
        "faithfulness_evaluation_mask": "silence",
        "faithfulness_computed": not args.skip_faithfulness,
        "baseline_table": str(baseline_path.resolve()) if baseline_path.is_file() else None,
        "shap_value_function": "predicted-class probability with absent aligned word segments replaced by hard silence",
    }
    (output_dir / "shap_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Table 4 comparison: {output_dir / 'table4_shap_comparison.csv'}")


if __name__ == "__main__":
    main()
