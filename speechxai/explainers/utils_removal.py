from pydub import AudioSegment

try:
    import whisperx
except ImportError:
    whisperx = None
import os
from typing import Dict, List, Union, Tuple
import numpy as np


MASK_MARGIN_BEFORE_MS = 100
MASK_MARGIN_AFTER_MS = 40
MASK_NOISE_SEED = 42
MASK_NOISE_FRAME_MS = 20
MASK_NOISE_PERCENTILE = 10
MASK_CROSSFADE_MS = 20


def _estimate_noise_rms(audio: AudioSegment) -> float:
    """Estimate the recording noise floor from its quietest short-time frames."""
    from speechxai.utils import pydub_to_np

    audio_16 = audio.set_sample_width(2)
    samples, sample_rate = pydub_to_np(audio_16)
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    frame_length = max(1, int(sample_rate * MASK_NOISE_FRAME_MS / 1000))
    frame_rms = [
        float(np.sqrt(np.mean(np.square(mono[start : start + frame_length]))))
        for start in range(0, len(mono), frame_length)
        if len(mono[start : start + frame_length]) > 0
    ]
    if not frame_rms:
        return 1e-4
    # The percentile estimates a quiet background level without copying audio.
    return max(float(np.percentile(frame_rms, MASK_NOISE_PERCENTILE)), 1e-4)


def _replace_interval_with_noise(
    audio: AudioSegment,
    start_ms: int,
    end_ms: int,
    noise_rms: float,
    seed: int = MASK_NOISE_SEED,
) -> AudioSegment:
    """Replace an interval with generated noise and cosine-fade both joins."""
    from speechxai.utils import pydub_to_np

    audio = audio.set_sample_width(2)
    samples, sample_rate = pydub_to_np(audio)
    mono = samples.ndim == 1
    if mono:
        samples = samples[:, None]

    start = max(0, min(len(samples), int(start_ms * sample_rate / 1000)))
    end = max(start, min(len(samples), int(end_ms * sample_rate / 1000)))
    if end <= start:
        return audio

    # The word interval is removed; only its transition margins are blended.
    rng = np.random.default_rng(seed + start + end)
    replacement = rng.normal(0.0, noise_rms, size=(end - start, samples.shape[1]))
    replacement = np.clip(replacement, -1.0, 1.0)
    fade_length = min(
        int(sample_rate * MASK_CROSSFADE_MS / 1000),
        (end - start) // 2,
    )
    if fade_length > 0:
        phase = np.linspace(0.0, np.pi / 2, fade_length, endpoint=True)
        fade_in = np.sin(phase) ** 2
        fade_out = fade_in[::-1]
        replacement[:fade_length] = (
            samples[start : start + fade_length] * (1.0 - fade_in[:, None])
            + replacement[:fade_length] * fade_in[:, None]
        )
        replacement[-fade_length:] = (
            samples[end - fade_length : end] * (1.0 - fade_out[:, None])
            + replacement[-fade_length:] * fade_out[:, None]
        )

    samples[start:end] = replacement
    encoded = np.clip(np.round(samples * 32768.0), -32768, 32767).astype("<i2")
    if mono:
        encoded = encoded[:, 0]
    return audio._spawn(encoded.tobytes())


def _mask_audio_interval(
    audio: AudioSegment,
    start_ms: int,
    end_ms: int,
    removal_type: str,
    noise_rms: float = None,
    seed: int = MASK_NOISE_SEED,
) -> AudioSegment:
    """Apply one supported masking method to a time interval."""
    start_ms = max(0, start_ms)
    end_ms = min(len(audio), end_ms)
    if end_ms <= start_ms:
        return audio
    if removal_type == "silence":
        replacement = AudioSegment.silent(
            duration=end_ms - start_ms, frame_rate=audio.frame_rate
        ).set_channels(audio.channels).set_sample_width(audio.sample_width)
        return audio[:start_ms] + replacement + audio[end_ms:]
    if removal_type == "noise_matched":
        if noise_rms is None:
            noise_rms = _estimate_noise_rms(audio)
        return _replace_interval_with_noise(audio, start_ms, end_ms, noise_rms, seed)
    if removal_type == "nothing":
        return audio[:start_ms] + audio[end_ms:]
    if removal_type in {"white noise", "pink noise"}:
        filename = (
            "white_noise.mp3" if removal_type == "white noise" else "pink_noise.mp3"
        )
        noise = AudioSegment.from_mp3(
            os.path.join(os.path.dirname(__file__), filename)
        )
        noise = (
            noise.set_frame_rate(audio.frame_rate)
            .set_channels(audio.channels)
            .set_sample_width(audio.sample_width)
        )
        duration = end_ms - start_ms
        if len(noise) < duration:
            repeats = (duration + len(noise) - 1) // len(noise)
            noise *= repeats
        return audio[:start_ms] + noise[:duration] + audio[end_ms:]
    supported = "'nothing', 'silence', 'noise_matched', 'white noise', or 'pink noise'"
    raise ValueError(f"Unsupported removal_type '{removal_type}'. Use {supported}.")


def remove_specified_words(audio, words, removal_type: str = "nothing"):
    """
    Remove specified word intervals from audio using pydub, by replacing them with:
    - nothing
    - silence
    - recording-level noise-matched generated noise
    - white noise
    - pink noise

    Args:
        audio (pydub.AudioSegment): audio
        word: word to remove with its start and end times
        removal_type (str, optional): type of removal. Defaults to "nothing".
    """

    from copy import deepcopy

    audio_removed = deepcopy(audio)

    noise_rms = _estimate_noise_rms(audio) if removal_type == "noise_matched" else None
    intervals = sorted(
        (
            max(0, int(word["start"] * 1000) - MASK_MARGIN_BEFORE_MS),
            min(len(audio), int(word["end"] * 1000) + MASK_MARGIN_AFTER_MS),
        )
        for word in words
    )

    # Adjacent spoken words often have margins that overlap.  Replacing them
    # one at a time would make a later crossfade use synthetic noise created
    # by an earlier replacement as its source.  Mask the union instead, so a
    # multi-word rationale has one deterministic replacement per continuous
    # removed region.  This has identical behaviour for hard silence.
    merged_intervals = []
    for start, end in intervals:
        if not merged_intervals or start > merged_intervals[-1][1]:
            merged_intervals.append([start, end])
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], end)

    # Apply from right to left so the implementation remains correct if a
    # future removal method changes duration.
    for index, (start, end) in enumerate(reversed(merged_intervals)):
        audio_removed = _mask_audio_interval(
            audio_removed,
            start,
            end,
            removal_type,
            noise_rms=noise_rms,
            seed=MASK_NOISE_SEED + index,
        )
    return audio_removed


_CACHED_WHISPER_MODELS = {}
_CACHED_ALIGN_MODELS = {}


def transcribe_audio(
    audio_path: str,
    device: str = "cuda",
    batch_size: int = 2,
    compute_type: str = "float32",
    language: str = "en",
    model_name_whisper: str = "base",
) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
    """
    Transcribe audio using whisperx,
    and return the text (transcription) and the words with their start and end times.
    """

    if whisperx is None:
        raise ImportError("whisperx is required for automatic transcription. Please install it with: pip install whisperx")

    import sys
    from unittest.mock import MagicMock
    if "whisperx.vads.pyannote" not in sys.modules or not isinstance(sys.modules.get("whisperx.vads.pyannote"), MagicMock):
        sys.modules["whisperx.vads.pyannote"] = MagicMock()
        for mod in ["pyannote", "pyannote.core", "pyannote.audio", "pyannote.audio.pipelines", "pyannote.audio.pipelines.utils"]:
            sys.modules[mod] = MagicMock()

    global _CACHED_WHISPER_MODELS, _CACHED_ALIGN_MODELS
    whisper_key = (model_name_whisper, device, compute_type, language)
    if whisper_key not in _CACHED_WHISPER_MODELS:
        _CACHED_WHISPER_MODELS[whisper_key] = whisperx.load_model(
            model_name_whisper,
            device,
            compute_type=compute_type,
            language=language,
            vad_method="silero",
        )
    model_whisperx = _CACHED_WHISPER_MODELS[whisper_key]

    ## Transcribe audio
    audio = whisperx.load_audio(audio_path)
    result = model_whisperx.transcribe(audio, batch_size=batch_size)

    align_key = (result["language"], device)
    if align_key not in _CACHED_ALIGN_MODELS:
        _CACHED_ALIGN_MODELS[align_key] = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
    model_a, metadata = _CACHED_ALIGN_MODELS[align_key]

    ## Align timestamps
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if result is None or "segments" not in result or len(result["segments"]) == 0:
        return "", []

    if len(result["segments"]) == 1:
        text = result["segments"][0]["text"]
        words = result["segments"][0]["words"]
    else:
        text = " ".join(
            result["segments"][i]["text"] for i in range(len(result["segments"]))
        )
        words = [word for segment in result["segments"] for word in segment["words"]]

    # Remove words that are not properly transcribed
    words = [word for word in words if "start" in word]
    return text, words


def transcribe_audio_given_model(
    model_whisperx,
    audio_path: str,
    batch_size: int = 2,
    device: str = "cuda",
) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
    """
    Transcribe audio using whisperx,
    and return the text (transcription) and the words with their start and end times.
    """

    ## Transcribe audio
    audio = whisperx.load_audio(audio_path)
    result = model_whisperx.transcribe(audio, batch_size=batch_size)
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )

    ## Align timestamps
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if result is None or "segments" not in result or len(result["segments"]) == 0:
        return "", []

    if len(result["segments"]) == 1:
        text = result["segments"][0]["text"]
        words = result["segments"][0]["words"]
    else:
        text = " ".join(
            result["segments"][i]["text"] for i in range(len(result["segments"]))
        )
        words = [word for segment in result["segments"] for word in segment["words"]]

    # Remove words that are not properly transcribed
    words = [word for word in words if "start" in word]
    return text, words


def remove_word(audio, word, removal_type: str = "nothing", noise_rms: float = None):
    """
    Remove a word from audio using pydub, by replacing it with:
    - nothing
    - silence
    - recording-level noise-matched generated noise
    - white noise
    - pink noise

    Args:
        audio (pydub.AudioSegment): audio
        word: word to remove with its start and end times
        removal_type (str, optional): type of removal. Defaults to "nothing".
    """

    start = int(word["start"] * 1000)
    end = int(word["end"] * 1000)
    mask_start = max(0, start - MASK_MARGIN_BEFORE_MS)
    mask_end = min(len(audio), end + MASK_MARGIN_AFTER_MS)
    if removal_type == "noise_matched" and noise_rms is None:
        noise_rms = _estimate_noise_rms(audio)
    return _mask_audio_interval(
        audio,
        mask_start,
        mask_end,
        removal_type,
        noise_rms=noise_rms,
        seed=MASK_NOISE_SEED + int(start),
    )
