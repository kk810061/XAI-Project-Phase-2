"""Build matched WA-LIME vs WA-SHAP stability tables and their summary."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Final results XAI" / "SHAP instability"
LIME_PATH = RESULTS / "lime_pairs.csv"
SHAP_PATH = RESULTS / "shap_vectors.csv"
OUTPUT_PATH = RESULTS / "stability_by_sample.csv"
SUMMARY_PATH = RESULTS / "stability_summary.csv"


def top_abs_agreement(left, right):
    """Compare unique maximum-absolute-attribution word indices."""
    left_abs = np.abs(left)
    right_abs = np.abs(right)
    left_top = np.flatnonzero(np.isclose(left_abs, left_abs.max())) if len(left_abs) else []
    right_top = np.flatnonzero(np.isclose(right_abs, right_abs.max())) if len(right_abs) else []
    if len(left) <= 1 or len(right) <= 1 or len(left_top) != 1 or len(right_top) != 1:
        return np.nan
    return float(left_top[0] == right_top[0])


def main():
    lime = pd.read_csv(LIME_PATH)
    shap = pd.read_csv(SHAP_PATH)

    # LIME's controlled file already records pairwise Spearman and absolute-
    # top-word agreement. Recompute SHAP using precisely those definitions.
    lime_cluster = lime.groupby("sample_id").agg(
        lime_rank_mean=("spearman_rank_correlation", "mean"),
        lime_rank_n=("spearman_rank_correlation", "count"),
        lime_abs_top_mean=("top_word_agreement", "mean"),
        lime_abs_top_n=("top_word_agreement", "count"),
    )

    # The source file contains both WA-LIME and WA-SHAP rows. Filter explicitly
    # before grouping; otherwise repeated seed IDs from the two methods could
    # overwrite each other while constructing the per-seed vectors.
    shap_only = shap[shap["method"] == "WA-SHAP"]
    if shap_only.empty:
        raise ValueError("No WA-SHAP rows found in the per-example stability file")

    shap_records = []
    for (sample_id, slot), group in shap_only.groupby(["sample_id", "slot"], sort=False):
        vectors = {
            int(row.seed): np.asarray(json.loads(row.word_attributions), dtype=float)
            for row in group.itertuples()
        }
        for seed_left, seed_right in itertools.combinations(sorted(vectors), 2):
            left = vectors[seed_left]
            right = vectors[seed_right]
            if len(left) != len(right):
                raise ValueError(f"Feature count changed across seeds for {sample_id} / {slot}")
            rho = np.nan
            if len(left) > 1 and np.std(left) > 0 and np.std(right) > 0:
                rho = float(spearmanr(left, right).statistic)
            shap_records.append({
                "sample_id": sample_id,
                "slot": slot,
                "seed_pair": f"{seed_left}-{seed_right}",
                "rho": rho,
                "top_agreement": top_abs_agreement(left, right),
            })

    shap_pairs = pd.DataFrame(shap_records)
    shap_cluster = shap_pairs.groupby("sample_id").agg(
        shap_rank_mean=("rho", "mean"),
        shap_rank_n=("rho", "count"),
        shap_abs_top_mean=("top_agreement", "mean"),
        shap_abs_top_n=("top_agreement", "count"),
    )

    paired = lime_cluster.join(shap_cluster, how="inner")
    paired = paired[
        (paired["lime_rank_n"] > 0)
        & (paired["shap_rank_n"] > 0)
        & paired["lime_abs_top_mean"].notna()
        & paired["shap_abs_top_mean"].notna()
    ].copy()
    paired["shap_minus_lime_rank"] = (
        paired["shap_rank_mean"] - paired["lime_rank_mean"]
    )
    paired["shap_minus_lime_abs_top"] = (
        paired["shap_abs_top_mean"] - paired["lime_abs_top_mean"]
    )
    paired = paired.reset_index()
    paired.insert(0, "comparison_protocol", "same FSC utterance; word-only features; 5 seeds; 10 seed pairs x 3 slots")
    paired.insert(1, "top_word_rule", "unique max absolute attribution; ties/one-word excluded")

    if len(paired) != 97:
        raise ValueError(f"Expected the 97 valid matched utterances; found {len(paired)}")
    if not (paired["lime_rank_n"] == 30).all() or not (paired["shap_rank_n"] == 30).all():
        raise ValueError("Each included utterance must contain all 30 rank comparisons per method")

    paired.to_csv(OUTPUT_PATH, index=False, float_format="%.8f")
    print(f"Wrote {OUTPUT_PATH} with {len(paired)} paired utterances")
    for name, column in [
        ("WA-LIME rank stability", "lime_rank_mean"),
        ("WA-SHAP rank stability", "shap_rank_mean"),
        ("WA-LIME absolute-top agreement", "lime_abs_top_mean"),
        ("WA-SHAP absolute-top agreement", "shap_abs_top_mean"),
        ("SHAP minus LIME rank stability", "shap_minus_lime_rank"),
        ("SHAP minus LIME absolute-top agreement", "shap_minus_lime_abs_top"),
    ]:
        values = paired[column]
        half_width = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
        scale = 100 if "agreement" in column else 1
        print(f"{name}: mean={values.mean() * scale:.6f}; 95% CI half-width={half_width * scale:.6f}; N={len(values)}")

    summary_rows = []
    for method, rank_col, top_col, rank_n_col, top_n_col in [
        ("WA-LIME (separate word-only run)", "lime_rank_mean", "lime_abs_top_mean", "lime_rank_n", "lime_abs_top_n"),
        ("WA-SHAP", "shap_rank_mean", "shap_abs_top_mean", "shap_rank_n", "shap_abs_top_n"),
    ]:
        rank = paired[rank_col].dropna()
        top = paired[top_col].dropna()
        summary_rows.append({
            "method": method,
            "valid_samples": len(paired),
            "rank_mean": rank.mean(),
            "rank_ci95_halfwidth": 1.96 * rank.std(ddof=1) / np.sqrt(len(rank)),
            "rank_pair_count": int(paired[rank_n_col].sum()),
            "abs_top_mean": top.mean(),
            "abs_top_ci95_halfwidth": 1.96 * top.std(ddof=1) / np.sqrt(len(top)),
            "abs_top_pair_count": int(paired[top_n_col].sum()),
        })
    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False, float_format="%.8f")
    print(f"Wrote matched summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
