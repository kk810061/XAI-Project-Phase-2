"""Random Speech Explainer module for baseline benchmarking"""
import numpy as np
from typing import List, Dict, Union
from pydub import AudioSegment
from speechxai.utils import pydub_to_np
from speechxai.explainers.explanation_speech import ExplanationSpeech
from speechxai.explainers.utils_removal import transcribe_audio


class RandomSpeechExplainer:
    NAME = "random_speech"

    def __init__(self, model_helper):
        self.model_helper = model_helper

    def compute_explanation(
        self,
        audio_path: str,
        target_class=None,
        words_trascript: List[Dict[str, Union[str, float]]] = None,
        seed: int = None,
        **kwargs,
    ) -> ExplanationSpeech:
        """
        Compute random attribution scores in [-1, 1] for baseline comparisons (Table 4).
        """
        if seed is not None:
            np.random.seed(seed)

        if words_trascript is None:
            _, words_trascript = transcribe_audio(
                audio_path=audio_path,
                device=self.model_helper.device.type if hasattr(self.model_helper.device, "type") else str(self.model_helper.device),
                batch_size=2,
                compute_type="float32",
                language=self.model_helper.language,
            )

        words = [w["word"] for w in words_trascript]
        n_words = len(words)
        n_labels = self.model_helper.n_labels

        audio = pydub_to_np(AudioSegment.from_wav(audio_path))[0]
        logits_original = self.model_helper.predict([audio])

        if target_class is not None:
            targets = target_class
        else:
            if n_labels > 1:
                targets = [
                    int(np.argmax(logits_original[i], axis=1)[0])
                    for i in range(n_labels)
                ]
            else:
                targets = [int(np.argmax(logits_original[0], axis=1)[0])]

        if n_labels > 1:
            scores = np.random.uniform(-1, 1, size=(n_labels, n_words))
        else:
            scores = np.random.uniform(-1, 1, size=n_words)

        return ExplanationSpeech(
            features=words,
            scores=scores,
            target=targets,
            explainer="random",
            audio_path=audio_path,
        )
