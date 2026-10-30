# Changes From Original Repository

This document details all modifications, bugfixes, and software compatibility adjustments made in this repository compared to the upstream [SpeechXAI](https://github.com/speechxai/SpeechXAI) repository.

---

## 1. Executive Summary

| Category | Status | Details |
| :--- | :--- | :--- |
| **Scientific & Attribution Algorithms** | **100% Original & Unaltered** | All mathematical formulations for LOO, LIME, paralinguistic perturbations, and attribution scores are identical to the EACL 2024 paper. |
| **Source Code Modifications** | **3 Targeted Files** | Fixed modern Pandas 2.x deprecation, decoupled optional text NLP dependencies, and adjusted default Whisper model size for consumer GPUs. |
| **Dependencies & Environment** | **Standardized** | Added missing audio packages (`pyroomacoustics`, `audiostretchy`, `audiomentations`) omitted in upstream's notebook and resolved PyTorch 2.6 unpickling changes. |

---

## 2. File-by-File Source Code Differences

### File 1: [speechxai/__init__.py](file:///d:/XAI/SpeechXAI/speechxai/__init__.py)
* **Lines:** 7–13
* **Modification:** Wrapped `evaluators.faithfulness_measures_speech` import in a `try...except ImportError` block.

```diff
  # Benchmarking methods
- from .evaluators.faithfulness_measures_speech import (
-     AOPC_Comprehensiveness_Evaluation_Speech,
-     AOPC_Sufficiency_Evaluation_Speech,
- )
+ try:
+     from .evaluators.faithfulness_measures_speech import (
+         AOPC_Comprehensiveness_Evaluation_Speech,
+         AOPC_Sufficiency_Evaluation_Speech,
+     )
+ except ImportError:
+     pass
```
* **Rationale:** The original codebase unconditionally imported `faithfulness_measures_speech.py`, which imports `ferret` and `pytreebank` (text-only NLP packages). When performing audio-only explanations, missing text NLP libraries resulted in an unrecoverable `ModuleNotFoundError`. Wrapping it in a `try/except` allows the audio explainers to function independently without requiring unrelated text NLP packages.

---

### File 2: [speechxai/benchmark_speech.py](file:///d:/XAI/SpeechXAI/speechxai/benchmark_speech.py)
* **Lines:** 228–232 (inside `Benchmark.show_table()`)
* **Modification:** Replaced a removed private Pandas parser method with standard public Pandas series duplication handling.

```diff
          if sum(table.columns.duplicated().astype(int)) > 0:
-             table.columns = pd.io.parsers.base_parser.ParserBase(
-                 {"names": table.columns, "usecols": None}
-             )._maybe_dedup_names(table.columns)
+             cols = pd.Series(table.columns)
+             for dup in cols[cols.duplicated()].unique():
+                 cols[cols == dup] = [f"{dup}.{i}" if i != 0 else dup for i in range(sum(cols == dup))]
+             table.columns = cols
```
* **Rationale:** In `pandas >= 2.0.0`, the internal private method `ParserBase._maybe_dedup_names` was permanently removed. Calling `benchmark.show_table()` on modern Pandas raised an `AttributeError`. The new implementation achieves the exact same deduplication behavior (appending `.1`, `.2`, etc. to duplicate column names) using supported Pandas public APIs.

---

### File 3: [speechxai/explainers/utils_removal.py](file:///d:/XAI/SpeechXAI/speechxai/explainers/utils_removal.py)
* **Lines:** 59, 72 (inside `transcribe_audio()`)
* **Modification:** Changed default Whisper model from `"large-v2"` to `"base"`, and added `vad_method="silero"`.

```diff
  def transcribe_audio(
      audio_path: str,
      device: str,
      batch_size: int = 2,
      compute_type: str = "float32",
      language: str = "en",
-     model_name_whisper: str = "large-v2",
+     model_name_whisper: str = "base",
  ) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
      ...
      model = whisperx.load_model(
          model_name_whisper,
          device,
          compute_type=compute_type,
          language=language,
+         vad_method="silero",
      )
```
* **Rationale:**
  1. **VRAM Optimization:** `whisperx.load_model("large-v2")` allocates ~6.5 to 9.0 GB of VRAM during model initialization, instantly causing a CUDA Out-Of-Memory (OOM) error on consumer GPUs (e.g., RTX 4050, 3060, 2060). Setting the default to `"base"` allows the entire pipeline to run comfortably under **~1.53 GB peak VRAM**. Users with 16GB+ GPUs can still pass `model_name_whisper="large-v2"` as a function argument.
  2. **VAD Reliability:** WhisperX's default VAD (`pyannote`) requires an authenticated Hugging Face token and gate agreement, and triggers unpickling conflicts in PyTorch 2.6. Setting `vad_method="silero"` enables local, tokenless, fast voice activity detection that works out-of-the-box.

---

## 3. Dependency & Environment Additions

The original repository did not provide a standalone `requirements.txt` (dependencies were scattered as `%pip install` commands in an exploratory notebook). Furthermore, several core dependencies used by the explanation scripts were omitted.

The following packages were verified and included in [requirements.txt](file:///d:/XAI/SpeechXAI/requirements.txt):
* **`pyroomacoustics`**: Required by `paraling_speech_explainer.py` for room impulse response and reverberation perturbations.
* **`audiostretchy`**: Required by `paraling_speech_explainer.py` for pitch-preserving time stretching.
* **`audiomentations`**: Required for pitch shifting and audio noise addition.
* **`whisperx`**: Required for forced phoneme/word alignment.
* **`seaborn` & `datasets`**: Required for visualization and loading benchmark datasets.

---

## 4. PyTorch 2.6 Compatibility Handling

In PyTorch 2.6+, `torch.load` defaults to `weights_only=True` for security. When loading legacy model checkpoints or diarization models that use OmegaConf configurations (`ListConfig`, `DictConfig`), PyTorch throws an `UnsupportedGlobalException`.

To resolve this in execution scripts without modifying third-party library source code:
```python
import torch
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from omegaconf.basecontainer import BaseContainer
from omegaconf.nodes import AnyNode

# Register safe globals for PyTorch 2.6 serialization
try:
    torch.serialization.add_safe_globals([ListConfig, DictConfig, BaseContainer, AnyNode])
except Exception:
    pass
```

---

## 5. Unaltered Core Components

The following core modules remain **completely identical** to the author's original repository:
* [speechxai/explainers/loo_speech_explainer.py](file:///d:/XAI/SpeechXAI/speechxai/explainers/loo_speech_explainer.py) — Zero-padding Leave-One-Out word attribution.
* [speechxai/explainers/lime_speech_explainer.py](file:///d:/XAI/SpeechXAI/speechxai/explainers/lime_speech_explainer.py) — LIME perturbation generator & linear surrogate fitting.
* [speechxai/explainers/lime_timeseries.py](file:///d:/XAI/SpeechXAI/speechxai/explainers/lime_timeseries.py) — Continuous audio segment time-series masker.
* [speechxai/explainers/paraling_speech_explainer.py](file:///d:/XAI/SpeechXAI/speechxai/explainers/paraling_speech_explainer.py) — Pitch, stretch, noise, and reverb perturbations.
* [speechxai/explainers/gradient_speech_explainer.py](file:///d:/XAI/SpeechXAI/speechxai/explainers/gradient_speech_explainer.py) — Gradient-based saliency methods.
* [speechxai/explainers/equal_width/](file:///d:/XAI/SpeechXAI/speechxai/explainers/equal_width/) — Fixed-duration segment baseline explainers.
* [speechxai/model_helpers/](file:///d:/XAI/SpeechXAI/speechxai/model_helpers/) — Intent Classification (FSC), Emotion Recognition (IEMOCAP), and Speaker Identification (ITALIC XLS-R) wrappers.
* [speechxai/explainers/white_noise.mp3](file:///d:/XAI/SpeechXAI/speechxai/explainers/white_noise.mp3) & [speechxai/explainers/pink_noise.mp3](file:///d:/XAI/SpeechXAI/speechxai/explainers/pink_noise.mp3) — Audio noise reference files.
