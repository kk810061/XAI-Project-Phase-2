import numpy as np
from speechxai.utils import pydub_to_np
from pydub import AudioSegment
import warnings
from typing import List
from IPython.display import display
from speechxai.explainers.explanation_speech import ExplanationSpeech, EvaluationSpeech
from speechxai.explainers.utils_removal import remove_specified_words


def _compute_aopc(scores):
    from statistics import mean
    return mean(scores) if len(scores) > 0 else 0.0


def _get_id_tokens_greater_th(soft_score_explanation, th, only_pos=None):
    id_top = np.where(soft_score_explanation > th)[0]
    return id_top


def _get_id_tokens_top_k(soft_score_explanation, k, only_pos=True):
    if only_pos:
        id_top_k = [
            i
            for i in np.array(soft_score_explanation).argsort()[-k:][::-1]
            if soft_score_explanation[i] > 0
        ]
    else:
        id_top_k = np.array(soft_score_explanation).argsort()[-k:][::-1]
    if len(id_top_k) == 0:
        return None
    return id_top_k


def _get_id_tokens_percentage(soft_score_explanation, percentage, only_pos=True):
    v = int(percentage * len(soft_score_explanation))
    if v > 0 and v <= len(soft_score_explanation):
        return _get_id_tokens_top_k(soft_score_explanation, v, only_pos=only_pos)
    else:
        return None


def _check_and_define_get_id_discrete_rationale_function(based_on):
    if based_on == "th":
        return _get_id_tokens_greater_th
    elif based_on == "k":
        return _get_id_tokens_top_k
    elif based_on == "perc":
        return _get_id_tokens_percentage
    else:
        raise ValueError(f"{based_on} type not supported. Specify th, k or perc.")


def parse_evaluator_args(evaluator_args):
    remove_first_last = evaluator_args.get("remove_first_last", True)
    only_pos = evaluator_args.get("only_pos", True)
    removal_args_input = evaluator_args.get("removal_args", None)
    removal_args = {
        "remove_tokens": True,
        "based_on": "perc",
        "thresholds": np.arange(0.1, 1.1, 0.1),
    }
    if removal_args_input:
        removal_args.update(removal_args_input)
    top_k_hard_rationale = evaluator_args.get("top_k_rationale", 5)
    return remove_first_last, only_pos, removal_args, top_k_hard_rationale


class AOPC_Comprehensiveness_Evaluation_Speech:
    NAME = "aopc_comprehensiveness"
    SHORT_NAME = "aopc_compr"
    # Higher is better
    BEST_SORTING_ASCENDING = False
    TYPE_METRIC = "faithfulness"

    def __init__(self, model_helper, **kwargs):
        self.model_helper = model_helper

    def compute_evaluation(
        self,
        explanation: ExplanationSpeech,
        target=None,
        words_trascript: List = None,
        **evaluation_args,
    ) -> EvaluationSpeech:
        """Evaluate an explanation on the AOPC Comprehensiveness metric.

        Args:
            explanation (Explanation): the explanation to evaluate
            target: class labels for which the explanation is evaluated - deprecated
            evaluation_args (dict): arguments for the evaluation

        Returns:
            Evaluation : the AOPC Comprehensiveness score of the explanation
        """

        _, only_pos, removal_args, _ = parse_evaluator_args(evaluation_args)

        assert (
            "perturb_paraling" not in explanation.explainer
        ), f"{explanation.explainer} not supported"

        evaluation_removal_type = evaluation_args.get("evaluation_removal_type")
        if evaluation_removal_type is not None:
            removal_type = evaluation_removal_type
        elif "+" in explanation.explainer:
            # 'The explainer name contain "+" to specify the removal type'
            removal_type = explanation.explainer.split("+")[1]
        else:
            # Default
            removal_type = "silence"

        if target is not None:
            warnings.warn(
                'The "target" argument is deprecated and will be removed in a future version. The explanation target are used as default.'
            )

        audio_path = explanation.audio_path

        target = explanation.target

        # Get the audio from audio_path
        audio = AudioSegment.from_wav(audio_path)
        audio_np = pydub_to_np(audio)[0]

        # Get prediction probability of the input sencence for the target
        ground_truth_probs = self.model_helper.predict([audio_np])

        # Get the probability of the target classes
        if self.model_helper.n_labels > 1:
            # Multi-label setting - probability of the target classes for each label (list of size = number of labels)
            ground_truth_probs_target = [
                ground_truth_probs[e][:, tidx][0] for e, tidx in enumerate(target)
            ]
        else:
            # Single probability
            ground_truth_probs_target = [ground_truth_probs[0][target[0]]]

        # Splite the audio into word-level audio segments
        from speechxai.explainers.loo_speech_explainer import transcribe_audio

        if words_trascript is None:
            text, words_trascript = transcribe_audio(
                audio_path=audio_path,
                device=self.model_helper.device.type,
                batch_size=2,
                compute_type="float32",
                language=self.model_helper.language,
            )

        get_discrete_rationale_function = (
            _check_and_define_get_id_discrete_rationale_function(
                removal_args["based_on"]
            )
        )

        thresholds = removal_args["thresholds"]

        score_explanation = explanation.scores

        # In this way, we allow for multi-label explanations
        # We have a list of score_explanations, one for each label
        # Each score explanation has length equal to the number of features: number of words in the case of word-level explanations or 1 in the case of paralinguistic level explanation (one paralinguistic feature at the time)
        if score_explanation.ndim == 1:
            score_explanations = [score_explanation]
        else:
            score_explanations = score_explanation

        aopc_comprehesiveness_multi_label = list()

        # We iterate over the target classes for a multi-label setting
        # In the case of single label, we iterate only once
        for target_class_idx, score_explanation in enumerate(score_explanations):
            removal_importances = list()
            id_tops = list()
            last_id_top = None

            # Ground truth probabilities of the target label (target_class_idx) and target class (target[target_class_idx]])
            # It is the output probability of the target class itself in the case of single label
            original_prob = ground_truth_probs_target[target_class_idx]

            # We compute the difference between in probability for all the thresholds
            for v in thresholds:
                # Get rationale from score explanation
                id_top = get_discrete_rationale_function(score_explanation, v, only_pos)

                # If the rationale is the same, we do not include it. In this way, we will not consider in the average the same omission.
                if (
                    id_top is not None
                    and last_id_top is not None
                    and set(id_top) == last_id_top
                ):
                    id_top = None

                id_tops.append(id_top)

                if id_top is None:
                    continue

                last_id_top = set(id_top)

                id_top.sort()

                # Comprehensiveness
                # The only difference between comprehesivenss and sufficiency is the computation of the removal.

                # For the comprehensiveness: we remove the terms in the discrete rationale.

                words_removed = [words_trascript[i] for i in id_top]

                audio_removed = remove_specified_words(
                    audio, words_removed, removal_type=removal_type
                )

                audio_removed_np = pydub_to_np(audio_removed)[0]

                # Probability of the modified audio
                audio_modified_probs = self.model_helper.predict([audio_removed_np])

                # Probability of the target class (and label) for the modified audio
                if self.model_helper.n_labels > 1:
                    # In the multi-label setting, we have a list of probabilities for each label

                    # We first take the probability of the corresponding target label target_class_idx
                    # Then we take the probability of the target class for that label target[target_class_idx]
                    modified_prob = audio_modified_probs[target_class_idx][
                        :, target[target_class_idx]
                    ][0]

                else:
                    # Single probability
                    # We take the probability of the target class target[target_class_idx]
                    modified_prob = audio_modified_probs[0][target[target_class_idx]]

                # compute probability difference
                removal_importance = original_prob - modified_prob
                removal_importances.append(removal_importance)

            if len(removal_importances) == 0:
                aopc_comprehesiveness = 0.0
            else:
                aopc_comprehesiveness = _compute_aopc(removal_importances)
            aopc_comprehesiveness_multi_label.append(aopc_comprehesiveness)

        evaluation_output = EvaluationSpeech(
            self.SHORT_NAME, aopc_comprehesiveness_multi_label, target
        )

        return evaluation_output


class AOPC_Sufficiency_Evaluation_Speech:
    NAME = "aopc_sufficiency"
    SHORT_NAME = "aopc_suff"
    # Lower is better
    BEST_SORTING_ASCENDING = True
    TYPE_METRIC = "faithfulness"

    def __init__(self, model_helper, **kwargs):
        self.model_helper = model_helper

    def compute_evaluation(
        self,
        explanation: ExplanationSpeech,
        target=None,
        words_trascript: List = None,
        **evaluation_args,
    ) -> EvaluationSpeech:
        """Evaluate an explanation on the AOPC Sufficiency metric.

        Args:
            explanation (Explanation): the explanation to evaluate
            target: class labels for which the explanation is evaluated - deprecated
            evaluation_args (dict): arguments for the evaluation

        Returns:
            Evaluation : the AOPC Sufficiency score of the explanation
        """

        _, only_pos, removal_args, _ = parse_evaluator_args(evaluation_args)

        assert (
            "perturb_paraling" not in explanation.explainer
        ), f"{explanation.explainer} not supported"

        evaluation_removal_type = evaluation_args.get("evaluation_removal_type")
        if evaluation_removal_type is not None:
            removal_type = evaluation_removal_type
        elif "+" in explanation.explainer:
            # 'The explainer name contain "+" to specify the removal type'
            removal_type = explanation.explainer.split("+")[1]
        else:
            # Default
            removal_type = "silence"

        if target is not None:
            warnings.warn(
                'The "target" argument is deprecated and will be removed in a future version. The explanation target are used as default.'
            )

        audio_path = explanation.audio_path

        target = explanation.target

        # Get the audio from audio_path
        audio = AudioSegment.from_wav(audio_path)
        audio_np = pydub_to_np(audio)[0]

        # Get prediction probability of the input sencence for the target
        ground_truth_probs = self.model_helper.predict([audio_np])

        # Get the probability of the target classes
        if self.model_helper.n_labels > 1:
            # Multi-label setting - probability of the target classes for each label (list of size = number of labels)
            ground_truth_probs_target = [
                ground_truth_probs[e][:, tidx][0] for e, tidx in enumerate(target)
            ]
        else:
            # Single probability
            ground_truth_probs_target = [ground_truth_probs[0][target[0]]]

        # Splite the audio into word-level audio segments
        from speechxai.explainers.loo_speech_explainer import transcribe_audio

        if words_trascript is None:
            text, words_trascript = transcribe_audio(
                audio_path=audio_path,
                device=self.model_helper.device.type,
                batch_size=2,
                compute_type="float32",
                language=self.model_helper.language,
            )

        get_discrete_rationale_function = (
            _check_and_define_get_id_discrete_rationale_function(
                removal_args["based_on"]
            )
        )

        thresholds = removal_args["thresholds"]

        score_explanation = explanation.scores

        # In this way, we allow for multi-label explanations
        # We have a list of score_explanations, one for each label
        # Each score explanation has length equal to the number of features: number of words in the case of word-level explanations or 1 in the case of paralinguistic level explanation (one paralinguistic feature at the time)
        if score_explanation.ndim == 1:
            score_explanations = [score_explanation]
        else:
            score_explanations = score_explanation

        aopc_comprehesiveness_multi_label = list()

        # We iterate over the target classes for a multi-label setting
        # In the case of single label, we iterate only once
        for target_class_idx, score_explanation in enumerate(score_explanations):
            removal_importances = list()
            id_tops = list()
            last_id_top = None

            # Ground truth probabilities of the target label (target_class_idx) and target class (target[target_class_idx]])
            # It is the output probability of the target class itself in the case of single label
            original_prob = ground_truth_probs_target[target_class_idx]

            # We compute the difference between in probability for all the thresholds
            for v in thresholds:
                # Get rationale from score explanation
                id_top = get_discrete_rationale_function(score_explanation, v, only_pos)

                # If the rationale is the same, we do not include it. In this way, we will not consider in the average the same omission.
                if (
                    id_top is not None
                    and last_id_top is not None
                    and set(id_top) == last_id_top
                ):
                    id_top = None

                id_tops.append(id_top)

                if id_top is None:
                    continue

                last_id_top = set(id_top)

                id_top.sort()

                # Sufficiency
                # The only difference between comprehesivenss and sufficiency is the computation of the removal.

                # For the sufficiency: we keep only the terms in the discrete rationale.
                # Hence, we remove all the other terms.
                words_removed = [
                    words_trascript[i]
                    for i in range(len(words_trascript))
                    if i not in id_top
                ]

                audio_removed = remove_specified_words(
                    audio, words_removed, removal_type=removal_type
                )

                audio_removed_np = pydub_to_np(audio_removed)[0]

                # Probability of the modified audio
                audio_modified_probs = self.model_helper.predict([audio_removed_np])

                # Probability of the target class (and label) for the modified audio
                if self.model_helper.n_labels > 1:
                    # In the multi-label setting, we have a list of probabilities for each label

                    # We first take the probability of the corresponding target label target_class_idx
                    # Then we take the probability of the target class for that label target[target_class_idx]
                    modified_prob = audio_modified_probs[target_class_idx][
                        :, target[target_class_idx]
                    ][0]

                else:
                    # Single probability
                    # We take the probability of the target class target[target_class_idx]
                    modified_prob = audio_modified_probs[0][target[target_class_idx]]

                # compute probability difference
                removal_importance = original_prob - modified_prob
                removal_importances.append(removal_importance)

            if len(removal_importances) == 0:
                aopc_comprehesiveness = 0.0
            else:
                aopc_comprehesiveness = _compute_aopc(removal_importances)
            aopc_comprehesiveness_multi_label.append(aopc_comprehesiveness)

        evaluation_output = EvaluationSpeech(
            self.SHORT_NAME, aopc_comprehesiveness_multi_label, target
        )

        return evaluation_output
