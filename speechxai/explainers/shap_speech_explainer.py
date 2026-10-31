"""Word-level KernelSHAP explanations for speech classifiers.

Each interpretable feature is one WhisperX-aligned word segment.  A coalition
contains the word segments that remain audible; all absent word segments are
replaced with hard silence.  This is intentionally the same replacement used
by the paper's faithfulness evaluation.
"""

from itertools import product
from math import comb
from typing import List

import numpy as np
from pydub import AudioSegment

from speechxai.explainers.explanation_speech import ExplanationSpeech
from speechxai.explainers.utils_removal import transcribe_audio
from speechxai.utils import pydub_to_np


class SHAPSpeechExplainer:
    """Estimate word attributions with the KernelSHAP weighted surrogate."""

    NAME = "SHAP"

    def __init__(self, model_helper):
        self.model_helper = model_helper

    @staticmethod
    def _sample_coalitions(n_features: int, num_samples: int, seed: int):
        """Return coalition masks and KernelSHAP regression weights.

        When the requested budget covers the entire coalition space, this
        enumerates it and returns exact Shapley values.  Otherwise it samples
        interior coalition sizes according to the KernelSHAP kernel.  The two
        endpoint coalitions receive a large weight to enforce local accuracy.
        """
        if n_features == 0:
            return np.ones((1, 0), dtype=np.int8), np.ones(1, dtype=float), True

        total_coalitions = 2**n_features
        if total_coalitions <= num_samples:
            masks = np.asarray(list(product((0, 1), repeat=n_features)), dtype=np.int8)
            weights = np.empty(total_coalitions, dtype=float)
            for index, mask in enumerate(masks):
                coalition_size = int(mask.sum())
                if coalition_size in (0, n_features):
                    weights[index] = 1_000_000.0
                else:
                    weights[index] = (n_features - 1) / (
                        comb(n_features, coalition_size)
                        * coalition_size
                        * (n_features - coalition_size)
                    )
            return masks, weights, True

        if num_samples < 2:
            raise ValueError("num_samples must be at least 2 so SHAP can include empty and full coalitions")

        rng = np.random.default_rng(seed)
        masks = [np.zeros(n_features, dtype=np.int8), np.ones(n_features, dtype=np.int8)]
        if n_features == 1:
            return np.asarray(masks), np.asarray([1_000_000.0, 1_000_000.0]), True

        # The per-size mass of the KernelSHAP kernel is proportional to
        # 1 / (k * (M-k)).  Sampling from it lets interior rows have equal
        # regression weight while preserving the KernelSHAP objective.
        sizes = np.arange(1, n_features)
        size_probabilities = 1.0 / (sizes * (n_features - sizes))
        size_probabilities /= size_probabilities.sum()
        for _ in range(num_samples - 2):
            coalition_size = int(rng.choice(sizes, p=size_probabilities))
            present = rng.choice(n_features, size=coalition_size, replace=False)
            mask = np.zeros(n_features, dtype=np.int8)
            mask[present] = 1
            masks.append(mask)

        weights = np.ones(len(masks), dtype=float)
        weights[:2] = 1_000_000.0
        return np.asarray(masks), weights, False

    @staticmethod
    def _mask_absent_words(audio: np.ndarray, word_intervals, coalition: np.ndarray):
        """Keep coalition words and set every absent aligned word span to zero."""
        masked = audio.copy()
        for present, (start, end) in zip(coalition, word_intervals):
            if not present:
                masked[start:end] = 0.0
        return masked

    def compute_explanation(
        self,
        audio_path: str,
        target_class=None,
        words_trascript: List = None,
        removal_type: str = "silence",
        num_samples: int = 100,
        seed: int = 42,
    ) -> ExplanationSpeech:
        """Compute word-level KernelSHAP scores for every FSC output slot."""
        if removal_type != "silence":
            raise ValueError(
                "WA-SHAP currently supports only hard-silence masking so it matches the paper evaluation."
            )
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2")

        if words_trascript is None:
            _, words_trascript = transcribe_audio(
                audio_path=audio_path,
                device=self.model_helper.device.type,
                language=self.model_helper.language,
            )
        words = list(words_trascript)
        for word in words:
            if "start" not in word or "end" not in word:
                raise ValueError("Every aligned word must contain start and end timestamps")

        audio, sample_rate = pydub_to_np(AudioSegment.from_wav(audio_path))
        n_features = len(words)
        word_intervals = [
            (
                max(0, int(float(word["start"]) * sample_rate)),
                min(len(audio), int(float(word["end"]) * sample_rate)),
            )
            for word in words
        ]
        coalitions, regression_weights, exact = self._sample_coalitions(
            n_features, num_samples, seed
        )
        masked_audios = [
            self._mask_absent_words(audio, word_intervals, coalition)
            for coalition in coalitions
        ]
        probabilities = self.model_helper.predict(masked_audios)

        full_index = int(np.flatnonzero(coalitions.sum(axis=1) == n_features)[0])
        if target_class is None:
            targets = [
                int(np.argmax(probabilities[label_index][full_index]))
                for label_index in range(self.model_helper.n_labels)
            ]
        else:
            targets = list(target_class)

        values = np.column_stack(
            [probabilities[label_index][:, target] for label_index, target in enumerate(targets)]
        )
        design = np.column_stack([np.ones(len(coalitions)), coalitions])
        sqrt_weights = np.sqrt(regression_weights)[:, None]
        coefficients, *_ = np.linalg.lstsq(
            design * sqrt_weights,
            values * sqrt_weights,
            rcond=None,
        )
        scores = coefficients[1:].T if n_features else np.zeros((len(targets), 0))

        explanation = ExplanationSpeech(
            features=[str(word["word"]) for word in words],
            scores=np.asarray(scores),
            explainer=self.NAME + "+silence",
            target=targets,
            audio_path=audio_path,
        )
        # Metadata is deliberately attached to the explanation so experiment
        # runners can report whether a sample used exact or sampled SHAP.
        explanation.coalition_count = int(len(coalitions))
        explanation.exact_kernel_shap = bool(exact)
        return explanation
