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

def compute_lime_words_only(model_helper, audio_path, words_transcript, num_samples=100, seed=42):
    """LIME explainer using strictly aligned word segments (no inter-word gaps)."""
    np.random.seed(seed)
    audio_segment = AudioSegment.from_wav(audio_path)
    audio = pydub_to_np(audio_segment)[0]
    sampling_rate = model_helper.feature_extractor.sampling_rate
    
    logits_original = model_helper.predict([audio])
    n_labels = model_helper.n_labels
    targets = [int(np.argmax(logits_original[i], axis=1)[0]) for i in range(n_labels)]
    
    # Strictly word segments
    splits = []
    for word in words_transcript:
        start = int(word["start"] * sampling_rate)
        end = int(word["end"] * sampling_rate)
        splits.append({"start": start, "end": end, "word": word["word"]})
        
    lime_explainer = LimeTimeSeriesExplainer()
    scores = []
    for target_label, target_class in enumerate(targets):
        predict_fn = model_helper.get_prediction_function_by_label(target_label)
        exp = lime_explainer.explain_instance(
            audio,
            predict_fn,
            num_features=len(splits),
            num_slices=len(splits),
            slices=splits,
            replacement_method="silence",
            num_samples=num_samples,
            labels=[target_class],
            distance_metric="cosine",
        )
        scores_dict = dict(exp.as_list(target_class))
        slot_scores = [scores_dict.get(w["word"], 0.0) for w in words_transcript]
        scores.append(slot_scores)
        
    features = [w["word"] for w in words_transcript]
    return ExplanationSpeech("LIME_WordsOnly", features, np.array(scores), targets, audio_path=audio_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="data/fsc/fluent_speech_commands_dataset/data/test_data.csv")
    parser.add_argument("--audio-root", default="data/fsc/fluent_speech_commands_dataset")
    parser.add_argument("--output-csv", default="Final results XAI/SHAP instability/controlled_lime_stability.csv")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "superb/wav2vec2-base-superb-ic"
    
    model = Wav2Vec2ForSequenceClassification.from_pretrained(checkpoint).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
    benchmark = Benchmark(model, feature_extractor, device=device)
    
    data = pd.read_csv(args.input_csv)
    sample_df = data.sample(n=args.limit, random_state=args.sample_seed)
    audio_root = Path(args.audio_root)
    
    records = []
    print(f"Running Controlled Word-Only LIME across {len(sample_df)} samples and {len(seeds)} seeds...")
    
    for row_idx, (_, row) in enumerate(sample_df.iterrows()):
        audio_path = str(audio_root / row["path"])
        _, words = transcribe_audio(audio_path, device=device)
        if not words:
            continue
            
        seed_scores = {}
        for s in seeds:
            exp = compute_lime_words_only(benchmark.model_helper, audio_path, words, num_samples=args.num_samples, seed=s)
            seed_scores[s] = exp.scores # shape: (3, num_words)
            
        # Compute pairwise seed comparisons
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
                records.append({
                    "method": "WA-LIME (WordsOnly)",
                    "sample_id": row["path"],
                    "slot": slot,
                    "seed_pair": f"{s1}-{s2}",
                    "spearman_rank_correlation": corr,
                    "top_word_agreement": top_agree
                })
        if (row_idx + 1) % 10 == 0:
            print(f"Progress: {row_idx + 1}/{len(sample_df)} samples completed")
            
    res_df = pd.DataFrame(records)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_path, index=False)
    
    per_sample = res_df.groupby("sample_id")[["spearman_rank_correlation", "top_word_agreement"]].mean()
    print("\n=== CONTROLLED WORD-ONLY LIME RESULTS (N=100) ===")
    print(f"Mean Spearman: {per_sample['spearman_rank_correlation'].mean():.4f} +/- {1.96 * per_sample['spearman_rank_correlation'].std() / np.sqrt(len(per_sample)):.4f}")
    print(f"Top-Word Agreement: {per_sample['top_word_agreement'].mean()*100:.2f}% +/- {1.96 * per_sample['top_word_agreement'].std() / np.sqrt(len(per_sample))*100:.2f}%")

if __name__ == "__main__":
    main()
