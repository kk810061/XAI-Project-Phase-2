import time
import torch

# PyTorch 2.6 compatibility for pyannote checkpoints
try:
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from omegaconf.basecontainer import BaseContainer
    from omegaconf.nodes import AnyNode
    torch.serialization.add_safe_globals([ListConfig, DictConfig, BaseContainer, AnyNode])
except Exception:
    pass

_orig_torch_load = torch.load
def _compat_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _compat_torch_load

import whisperx
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from speechxai.benchmark_speech import Benchmark

import os

device = "cuda"
audio_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.wav")
batch_size = 1
torch.cuda.reset_peak_memory_stats()

print("=== STEP 4: WHISPERX ALIGNMENT ===")
start = time.time()
try:
    print("Loading whisperx base model...")
    wx_model = whisperx.load_model("base", device, vad_method="silero")
    
    print("Loading audio...")
    audio = whisperx.load_audio(audio_file)
    
    print("Transcribing...")
    result = wx_model.transcribe(audio, batch_size=batch_size)
    print("Initial transcript:", [seg["text"] for seg in result["segments"]])
    
    print("Loading align model...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    
    print("Aligning...")
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    end = time.time()
    print("\n--- WhisperX Aligned Segments ---")
    for segment in result["segments"]:
        for word in segment.get("words", []):
            if 'start' in word and 'end' in word:
                print(f"[{word['start']:.2f}s - {word['end']:.2f}s]: {word['word']}")
            else:
                print(f"[no timing]: {word['word']}")
                
    print(f"\nWhisperX Runtime: {end - start:.2f} seconds")
    print(f"Peak VRAM used (WhisperX): {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nWHISPERX FAILURE: {e}")
    exit(1)

# Free whisper models to save VRAM
del wx_model
del model_a
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("\n=== STEP 5: LEAVE-ONE-OUT ATTRIBUTION ===")
start = time.time()
try:
    print("Loading Wav2Vec2 checkpoint...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ic").to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ic")
    
    print("Initializing Benchmark...")
    benchmark = Benchmark(model, feature_extractor)
    
    print("Running LOO Explanation...")
    explanation = benchmark.explain(
        audio_path=audio_file, 
        methodology='LOO'
    )
    
    print("\n--- LOO Explanation Table ---")
    table = benchmark.create_table(explanation)
    print(table.to_string())
    print(f"\n--- Top Attributed Words (target: {explanation.target[0] if explanation.target else 0}) ---")
    scores_first = explanation.scores[0] if len(explanation.scores.shape) > 1 else explanation.scores
    sorted_words = sorted(zip(explanation.features, scores_first), key=lambda x: abs(x[1]), reverse=True)
    for word, score in sorted_words[:10]:
        print(f"  {word:15s}: {score:+.4f}")
        
    end = time.time()
    print(f"\nLOO Runtime: {end - start:.2f} seconds")
    print(f"Peak VRAM used (LOO): {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nLOO FAILURE: {e}")
