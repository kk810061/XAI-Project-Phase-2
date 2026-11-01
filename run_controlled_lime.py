"""
Run Controlled Word-Only LIME vs WA-SHAP Stability Comparison on FSC
Eliminates the EMPTY_SPAN confound so LIME and SHAP use identical feature definitions.
"""

import argparse
import itertools
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from speechxai.benchmark_speech import Benchmark
from speechxai.explainers.utils_removal import transcribe_audio
from speechxai.explainers.lime_timeseries import LimeTimeSeriesExplainer
from speechxai.explainers.explanation_speech import ExplanationSpeech
from speechxai.utils import pydub_to_np
from pydub import AudioSegment

SLOTS = ("action", "object", "location")

def batched_all_slot_predictor(model_helper, batch_size):
    """Return FSC probabilities for equal-length audio arrays in GPU batches.

    LIME supplies one neighbourhood at a time.  The upstream FSC helper then
    forwards every perturbation separately, which makes a small stability run
    unnecessarily slow.  All LIME perturbations of one utterance have the
    same length, so batching them does not introduce padding.
    """
    def predict(audios):
        output_batches = []
        for start in range(0, len(audios), batch_size):
            chunk = audios[start:start + batch_size]
            inputs = model_helper.feature_extractor(
                [np.asarray(audio).squeeze() for audio in chunk],
                sampling_rate=model_helper.feature_extractor.sampling_rate,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model_helper.model(
                    inputs.input_values.to(model_helper.device)
                ).logits
            action = logits[:, :6].softmax(-1)
            obj = logits[:, 6:20].softmax(-1)
            location = logits[:, 20:24].softmax(-1)
            output_batches.append(torch.cat((action, obj, location), dim=1).cpu().numpy())
        return np.concatenate(output_batches, axis=0)

    return predict


def compute_lime_words_only(
    model_helper, audio_path, words_transcript, num_samples=100, seed=42, batch_size=16
):
    """LIME explainer using strictly aligned word segments (no inter-word gaps)."""
    np.random.seed(seed)
    audio_segment = AudioSegment.from_wav(audio_path)
    audio = pydub_to_np(audio_segment)[0]
    sampling_rate = model_helper.feature_extractor.sampling_rate
    
    predict_fn = batched_all_slot_predictor(model_helper, batch_size)
    probabilities = predict_fn([audio])[0]
    targets = [
        int(np.argmax(probabilities[:6])),
        int(np.argmax(probabilities[6:20])),
        int(np.argmax(probabilities[20:24])),
    ]
    target_columns = [targets[0], targets[1] + 6, targets[2] + 20]
    
    # Strictly word segments
    splits = []
    for word in words_transcript:
        start = int(word["start"] * sampling_rate)
        end = int(word["end"] * sampling_rate)
        splits.append({"start": start, "end": end, "word": word["word"]})
        
    lime_explainer = LimeTimeSeriesExplainer()
    exp = lime_explainer.explain_instance(
        audio,
        predict_fn,
        num_features=len(splits),
        num_slices=len(splits),
        splits=splits,
        replacement_method="silence",
        num_samples=num_samples,
        labels=target_columns,
        random_seed=seed,
    )
    scores = []
    for target_column in target_columns:
        # LIME's as_map() keys are feature indices; map those indices back to
        # the aligned words so repeated word strings remain distinct features.
        scores_dict = dict(exp.as_map()[target_column])
        slot_scores = [scores_dict.get(index, 0.0) for index in range(len(splits))]
        scores.append(slot_scores)
        
    features = [w["word"] for w in words_transcript]
    return ExplanationSpeech(
        features=features,
        scores=np.asarray(scores, dtype=float),
        explainer="LIME_WordsOnly",
        target=targets,
        audio_path=audio_path,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="data/fsc/fluent_speech_commands_dataset/data/test_data.csv")
    parser.add_argument("--audio-root", default="data/fsc/fluent_speech_commands_dataset")
    parser.add_argument("--output-csv", default="Final results XAI/SHAP instability/controlled_lime_stability.csv")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "superb/wav2vec2-base-superb-ic"
    
    model = Wav2Vec2ForSequenceClassification.from_pretrained(checkpoint).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
    benchmark = Benchmark(model, feature_extractor, device=device)
    print(f"Inference device: {benchmark.model_helper.device}; LIME batch size: {args.batch_size}")
    
    data = pd.read_csv(args.input_csv)
    sample_df = data.sample(n=args.limit, random_state=args.sample_seed)
    audio_root = Path(args.audio_root)
    
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.resume:
        raise FileExistsError(f"{out_path} exists; choose a new file or add --resume")
    completed = set(pd.read_csv(out_path)["sample_id"]) if args.resume and out_path.exists() else set()
    print(f"Running Controlled Word-Only LIME across {len(sample_df)} samples and {len(seeds)} seeds...")
    
    for row_idx, (_, row) in enumerate(sample_df.iterrows()):
        if row["path"] in completed:
            continue
        audio_path = str(audio_root / row["path"])
        _, words = transcribe_audio(audio_path, device=device)
        if not words:
            continue
            
        seed_scores = {}
        for s in seeds:
            exp = compute_lime_words_only(
                benchmark.model_helper,
                audio_path,
                words,
                num_samples=args.num_samples,
                seed=s,
                batch_size=args.batch_size,
            )
            seed_scores[s] = np.asarray(exp.scores, dtype=float)  # shape: (3, num_words)
            
        # Compute pairwise seed comparisons
        row_records = []
        for s1, s2 in itertools.combinations(seeds, 2):
            for slot_idx, slot in enumerate(SLOTS):
                v1 = seed_scores[s1][slot_idx]
                v2 = seed_scores[s2][slot_idx]
                if len(v1) > 1:
                    corr, _ = spearmanr(v1, v2)
                    if np.isnan(corr): corr = 1.0
                else:
                    corr = 1.0
                top_agree = 1.0 if np.argmax(np.abs(v1)) == np.argmax(np.abs(v2)) else 0.0
                row_records.append({
                    "method": "WA-LIME (WordsOnly)",
                    "sample_id": row["path"],
                    "slot": slot,
                    "seed_pair": f"{s1}-{s2}",
                    "spearman_rank_correlation": corr,
                    "top_word_agreement": top_agree
                })
        pd.DataFrame(row_records).to_csv(
            out_path, mode="a", header=not out_path.exists(), index=False
        )
        print(f"Progress: {row_idx + 1}/{len(sample_df)} samples completed")

    res_df = pd.read_csv(out_path)
    
    per_sample = res_df.groupby("sample_id")[["spearman_rank_correlation", "top_word_agreement"]].mean()
    print("\n=== CONTROLLED WORD-ONLY LIME RESULTS (N=100) ===")
    print(f"Mean Spearman: {per_sample['spearman_rank_correlation'].mean():.4f} +/- {1.96 * per_sample['spearman_rank_correlation'].std() / np.sqrt(len(per_sample)):.4f}")
    print(f"Top-Word Agreement: {per_sample['top_word_agreement'].mean()*100:.2f}% +/- {1.96 * per_sample['top_word_agreement'].std() / np.sqrt(len(per_sample))*100:.2f}%")

if __name__ == "__main__":
    main()
