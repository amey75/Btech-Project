"""Run a fine-tuned T5 self-rationalization model on any (premise, hypothesis) pairs.

Input : jsonl with at least  id, premise, hypothesis  (optional: label = gold label text)
Output: the same rows plus  pred_label, pred_explanation, raw_output
If gold labels are present, prints accuracy and macro F1 and writes <out>.metrics.json.

The same script is reused later for the intervened inputs (removal / isolation /
contradiction): just give it a jsonl of edited premises.

Example:
    python predict.py --model models/t5-large_fastvotek_sub0_nt64 \
        --data data/esnli_test_600.jsonl --out outputs/t5_nt64_original.jsonl
"""
import argparse
import json
import os

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import LABELS, make_input, parse_output, pick_device, read_jsonl, write_jsonl


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="folder written by train_t5.py (or any HF T5 model)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=100)
    args = ap.parse_args()

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    rows = read_jsonl(args.data)

    for i in range(0, len(rows), args.bsz):
        batch = rows[i:i + args.bsz]
        enc = tok([make_input(r["premise"], r["hypothesis"]) for r in batch],
                  return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens)  # greedy, deterministic
        for r, text in zip(batch, tok.batch_decode(out, skip_special_tokens=True)):
            r["pred_label"], r["pred_explanation"] = parse_output(text)
            r["raw_output"] = text
        print(f"{min(i + args.bsz, len(rows))}/{len(rows)}", end="\r", flush=True)
    print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_jsonl(rows, args.out)
    print(f"Wrote predictions to {args.out}")

    if all("label" in r for r in rows):
        gold = [r["label"] for r in rows]
        pred = [r["pred_label"] for r in rows]
        metrics = {
            "n": len(rows),
            "accuracy": accuracy_score(gold, pred),
            "macro_f1": f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0),
            "malformed_outputs": sum(p not in LABELS for p in pred),
            "per_label": classification_report(gold, pred, labels=LABELS, output_dict=True, zero_division=0),
        }
        with open(os.path.splitext(args.out)[0] + ".metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"accuracy {metrics['accuracy']:.4f}  macro_F1 {metrics['macro_f1']:.4f}  "
              f"malformed {metrics['malformed_outputs']}")
        print(classification_report(gold, pred, labels=LABELS, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
