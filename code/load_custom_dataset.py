"""
Data loading + prompt formatting for the OLMo (decoder-only) pipeline.

This module is imported by training_olmo.py, inference_olmo.py,
acceptability_evaluation.py and auto_j_evaluation.py but was not included in
the upstream release. It is reconstructed from how those scripts use it:

- the dataset loaders are the same as for T5 (re-exported from load_target_dataset)
- prompt_formats() puts the prompt (and, for training, the target answer) in a "text" column
- the model answers in JSON: {"relationship": "<label>", "explanation": "<text>"},
  which inference_olmo.extract_relationship_explanation() parses with regex
"""
import json

import datasets

from load_target_dataset import load_raw_data, load_format_data  # noqa: F401 (re-exported)

LABEL2TEXT = {0: "entailment", 1: "neutral", 2: "contradiction"}
FULL_SHOT = 5000  # symbolic value the training scripts use for "full training set"

# NOTE: the prompt must not contain the literal pattern  "relationship": "  because
# generate() returns prompt + continuation and the parsing regex takes the first match.
PROMPT_TEMPLATE = (
    "Determine the relationship between the premise and the hypothesis. "
    "The relationship is one of: entailment, neutral, contradiction. "
    "Then explain your answer in one sentence. "
    "Respond in JSON with the keys relationship and explanation.\n"
    "Premise: {premise}\n"
    "Hypothesis: {hypothesis}\n"
    "Answer: "
)


def _clean(text):
    return " ".join(str(text).split())


def prompt_formats(example, validate_mode=False):
    """Return {"text": ...} for one example.

    validate_mode=True  -> prompt only (inference); the model generates the JSON answer.
    validate_mode=False -> prompt + gold JSON answer (supervised fine-tuning).
    """
    prompt = PROMPT_TEMPLATE.format(
        premise=_clean(example["premise"]), hypothesis=_clean(example["hypothesis"])
    )
    if validate_mode:
        return {"text": prompt}

    # Double quotes inside the explanation would break the inference-time regex ("([^"]+)"),
    # so train the model never to emit them.
    explanation = _clean(example["explanation_1"]).replace('"', "'")
    answer = json.dumps(
        {"relationship": LABEL2TEXT[int(example["label"])], "explanation": explanation}
    )
    return {"text": prompt + answer}


def load_train_data(path_or_name, n_shots):
    """Load OLMo fine-tuning data with a "text" column.

    n_shots != 5000: path_or_name is a samples/.../train_select.json file
    n_shots == 5000: path_or_name is a source dataset name ('esnli' or 'efever'), full train split
    """
    if n_shots != FULL_SHOT:
        data = datasets.load_dataset("json", data_files=path_or_name, split="train")
    else:
        data = load_raw_data(path_or_name, split="train")

    data = data.filter(lambda x: x["label"] in LABEL2TEXT and x["explanation_1"])
    data = data.map(prompt_formats, remove_columns=data.column_names)
    return data
