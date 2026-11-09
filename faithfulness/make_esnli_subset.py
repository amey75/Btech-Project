"""Build a label-balanced subset of the e-SNLI test set.

Uses the ORIGINAL e-SNLI CSV (github.com/OanaMariaCamburu/e-SNLI), not the Hugging Face copy:
it keeps pairID and the human-highlighted words, which Stage 2 (evidence spans) needs.

Example:
    python make_esnli_subset.py --per_label 200 --out data/esnli_test_600.jsonl
"""
import argparse
import os
import urllib.request

import pandas as pd

from common import LABELS, write_jsonl

CSV_URL = "https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_test.csv"


def highlighted_indices(value):
    """'3,6,4,5' -> [3, 4, 5, 6];  '{}' or NaN -> []"""
    if not isinstance(value, str) or value.strip() in ("", "{}"):
        return []
    return sorted(int(v) for v in value.split(",") if v.strip().lstrip("-").isdigit())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/esnli_test.csv", help="downloaded automatically if missing")
    ap.add_argument("--per_label", type=int, default=200, help="examples per label (3 labels)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/esnli_test_subset.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    if not os.path.exists(args.csv):
        print(f"Downloading {CSV_URL}")
        urllib.request.urlretrieve(CSV_URL, args.csv)

    df = pd.read_csv(args.csv)
    df = df[df["gold_label"].isin(LABELS)].dropna(subset=["Sentence1", "Sentence2"])
    parts = []
    for label in LABELS:
        g = df[df["gold_label"] == label]
        parts.append(g.sample(n=min(args.per_label, len(g)), random_state=args.seed))
    sample = pd.concat(parts).sample(frac=1, random_state=args.seed)  # shuffle so labels are mixed

    rows = []
    for _, r in sample.iterrows():
        rows.append({
            "id": r["pairID"],
            "premise": r["Sentence1"],
            "hypothesis": r["Sentence2"],
            "label": r["gold_label"],
            "gold_explanations": [r[f"Explanation_{k}"] for k in (1, 2, 3) if isinstance(r[f"Explanation_{k}"], str)],
            # word indices highlighted by each of the 3 annotators (used in Stage 2)
            "premise_highlights": [highlighted_indices(r[f"Sentence1_Highlighted_{k}"]) for k in (1, 2, 3)],
            "hypothesis_highlights": [highlighted_indices(r[f"Sentence2_Highlighted_{k}"]) for k in (1, 2, 3)],
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_jsonl(rows, args.out)
    print(f"Wrote {len(rows)} examples to {args.out}")
    print(pd.Series([r["label"] for r in rows]).value_counts().to_string())


if __name__ == "__main__":
    main()
