"""Top-level package for speechxai."""


from .benchmark_speech import Benchmark

# Benchmarking methods
try:
    from .evaluators.faithfulness_measures_speech import (
        AOPC_Comprehensiveness_Evaluation_Speech,
        AOPC_Sufficiency_Evaluation_Speech,
    )
except ImportError:
    pass

# Explainers
try:
    from .explainers.paraling_speech_explainer import ParalinguisticSpeechExplainer
except ImportError:
    pass

try:
    from .explainers.loo_speech_explainer import LOOSpeechExplainer
except ImportError:
    pass

try:
    from .explainers.explanation_speech import ExplanationSpeech
except ImportError:
    pass

# Model Helpers
from .model_helpers.model_helper_er import ModelHelperER
from .model_helpers.model_helper_fsc import ModelHelperFSC
from .model_helpers.model_helper_italic import ModelHelperITALIC
