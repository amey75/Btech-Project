"""Shared helpers: the T5 input/output format of Yang et al. (2025) and device selection.

Format (same as code/feature_conversion_methods.py in the original repo):
    input : explain nli hypothesis: <hypothesis> premise: <premise>
    target: <label> "explanation: " <explanation, lower-cased>
"""
import json

import torch

LABELS = ["entailment", "neutral", "contradiction"]
EXPLANATION_SEP = ' "explanation: " '


def collapse(text):
    return " ".join(str(text).split())


def make_input(premise, hypothesis):
    return collapse(f"explain nli hypothesis: {hypothesis} premise: {premise}")


def make_target(label, explanation):
    if isinstance(label, int):
        label = LABELS[label]
    return collapse(f"{label}{EXPLANATION_SEP}{str(explanation).lower()}")


def parse_output(text):
    """Split generated text into (label, explanation). Mirrors utils.sep_label_explanation."""
    text = collapse(text)
    parts = text.split(EXPLANATION_SEP.strip(), 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    words = text.split()  # malformed output: assume the first word is the label
    return (words[0] if words else ""), " ".join(words[1:])


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
