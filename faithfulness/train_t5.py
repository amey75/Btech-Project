"""Few-shot fine-tuning of T5-Large for self-rationalization on e-SNLI.

Follows Yang et al. (2025): same input/output format, the authors' pre-selected
FastVote-k examples (../samples/esnli/fastvotek/sub_<k>/<n>/train_select.json),
constant learning rate 3e-5, batch size 4, up to 50 epochs, keep the best checkpoint
on the e-SNLI validation set (../datasets/source/esnli/val_select.json, 360 examples).
Differences: plain PyTorch loop (no Trainer / W&B), and best checkpoint chosen by
validation label accuracy instead of accuracy + BERTScore (much faster).

Example (n_shots = examples PER LABEL, so 64 -> 192 training examples):
    python train_t5.py --n_shots 64 --out models/t5-large_fastvotek_sub0_nt64
"""
import argparse
import os
import random
import time

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import LABELS, make_input, make_target, parse_output, pick_device, read_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_pairs(rows):
    """Training/validation rows from the original repo -> (input, target) strings."""
    return [(make_input(r["premise"], r["hypothesis"]), make_target(int(r["label"]), r["explanation_1"])) for r in rows]


@torch.no_grad()
def val_accuracy(model, tok, rows, device, bsz, max_new_tokens):
    model.eval()
    correct = 0
    for i in range(0, len(rows), bsz):
        batch = rows[i:i + bsz]
        enc = tok([make_input(r["premise"], r["hypothesis"]) for r in batch],
                  return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens)
        for r, text in zip(batch, tok.batch_decode(out, skip_special_tokens=True)):
            correct += parse_output(text)[0] == LABELS[int(r["label"])]
    model.train()
    return correct / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="google-t5/t5-large")
    ap.add_argument("--selection", default="fastvotek", help="folder under ../samples/esnli/")
    ap.add_argument("--subset", type=int, default=0, help="sub_0 .. sub_4")
    ap.add_argument("--n_shots", type=int, default=64, help="examples per label")
    ap.add_argument("--train_file", default=None, help="override: any jsonl with premise/hypothesis/label/explanation_1")
    ap.add_argument("--val_file", default=os.path.join(REPO, "datasets/source/esnli/val_select.json"))
    ap.add_argument("--val_size", type=int, default=360, help="use fewer to speed up checkpoint selection")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=2, help="validate every N epochs")
    ap.add_argument("--patience", type=int, default=5, help="stop after N validations without improvement")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--optim", choices=["adamw", "adafactor"], default="adamw")
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--eval_bsz", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device()
    train_file = args.train_file or os.path.join(
        REPO, "samples/esnli", args.selection, f"sub_{args.subset}", str(args.n_shots), "train_select.json")
    train_rows = read_jsonl(train_file)
    val_rows = read_jsonl(args.val_file)[:args.val_size]
    print(f"device={device}  train={len(train_rows)} ({train_file})  val={len(val_rows)}")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)  # fp32: T5 is unstable in fp16
    model.train()
    if args.optim == "adafactor":  # far less memory than AdamW; use if T5-Large runs out of GPU memory
        from transformers.optimization import Adafactor
        optim = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False, relative_step=False, warmup_init=False)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pairs = to_pairs(train_rows)

    best_acc, best_epoch, bad_evals = -1.0, 0, 0
    os.makedirs(args.out, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        random.shuffle(pairs)
        t0, losses = time.time(), []
        for i in range(0, len(pairs), args.bsz):
            src, tgt = zip(*pairs[i:i + args.bsz])
            enc = tok(list(src), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            labels = tok(list(tgt), return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids.to(device)
            labels[labels == tok.pad_token_id] = -100  # ignore padding in the loss
            loss = model(**enc, labels=labels).loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
        msg = f"epoch {epoch:2d}  loss {np.mean(losses):.4f}  ({time.time() - t0:.0f}s)"

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            acc = val_accuracy(model, tok, val_rows, device, args.eval_bsz, args.max_new_tokens)
            msg += f"  val_acc {acc:.4f}"
            if acc > best_acc:
                best_acc, best_epoch, bad_evals = acc, epoch, 0
                model.save_pretrained(args.out)
                tok.save_pretrained(args.out)
                msg += "  * saved"
            else:
                bad_evals += 1
        print(msg, flush=True)
        if bad_evals >= args.patience:
            print(f"No improvement for {args.patience} validations, stopping.")
            break

    print(f"Best val_acc {best_acc:.4f} at epoch {best_epoch}; model in {args.out}")


if __name__ == "__main__":
    main()
