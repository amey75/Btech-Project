"""One-example walkthrough of the faithfulness tests, for a live demo.

Runs every step of the method on a single premise/hypothesis pair and prints it:
  1. the model's label, confidence and explanation
  2. whether the explanation argues for that label (stance check)
  3. the input words the explanation cites
  4. removal and isolation edits of the cited words vs random control words
  5. a contradiction edit of a cited word vs a matched control edit, with the
     regenerated explanation (and, with --oracle, the label it SHOULD get)

Usage (from the repo root):
  python faithfulness/demo.py --model_path checkpoints/fastvotek_nt128
  python faithfulness/demo.py --model_path checkpoints/fastvotek_nt128 \
      --premise "Child swimming in a pool." --hypothesis "A child swims in the ocean."
  python faithfulness/demo.py --model_path checkpoints/fastvotek_nt128 --test_index 25
  python faithfulness/demo.py --dry_run          # no model: rule-based stand-in, shows the mechanics only

--oracle also loads the DeBERTa NLI oracle (about 1.7 GB download on first use).
Runs on a laptop CPU: about 30-60 s to load T5-Large, then a few seconds per step.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evidence import cited_indices, normalise  # noqa: E402
from interventions import build_contradiction_variants, build_variants  # noqa: E402
from stance import stance  # noqa: E402

LINE = "-" * 78


def show(title):
    print(f"\n{LINE}\n{title}\n{LINE}")


def conf(pred, label):
    return pred["probs"][label]


def mark(tokens, idx):
    s = set(idx)
    return " ".join(f"[{t}]" if i in s else t for i, t in enumerate(tokens))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", help="folder of a T5 checkpoint (config.json + weights)")
    ap.add_argument("--dry_run", action="store_true", help="use the rule-based stand-in model (no download)")
    ap.add_argument("--premise", default="A man in a business suit screaming at the camera.")
    ap.add_argument("--hypothesis", default="A man is yelling at a camera.")
    ap.add_argument("--test_index", type=int, help="use pair N of the e-SNLI test set instead (shows its gold label)")
    ap.add_argument("--oracle", action="store_true", help="also load the NLI oracle for the contradiction step")
    ap.add_argument("--device", default=None, help="cpu (default), cuda, or mps")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                                                     "esnli_test.csv"))
    a = ap.parse_args()

    gold = None
    if a.test_index is not None:
        from data import load_esnli
        rows = load_esnli(a.data)
        ex = rows[a.test_index]
        premise, hypothesis, gold = ex["premise"], ex["hypothesis"], ex["label"]
    else:
        premise, hypothesis = a.premise, a.hypothesis
    example = {"id": "demo", "premise": premise, "hypothesis": hypothesis}

    if a.dry_run:
        from predictors import DummyOracle, DummyPredictor
        model, oracle = DummyPredictor(), (DummyOracle() if a.oracle else None)
        print("DRY RUN: rule-based stand-in model. Its outputs only illustrate the mechanics.")
    else:
        if not a.model_path:
            ap.error("give --model_path (a T5 checkpoint folder) or use --dry_run")
        from predictors import NLIOracle, T5Predictor
        print(f"Loading T5 checkpoint from {a.model_path} ...")
        model = T5Predictor(a.model_path, device=a.device)
        oracle = None
        if a.oracle:
            print("Loading NLI oracle (DeBERTa-v3-large) ...")
            oracle = NLIOracle(device=a.device)

    # 1. original prediction
    show("1. Original pair -> model output")
    print(f"Premise   : {premise}\nHypothesis: {hypothesis}" + (f"\nGold label: {gold}" if gold else ""))
    orig = model.predict([(premise, hypothesis)])[0]
    y = orig["label"]
    print(f"\nModel label : {y}   (confidence {conf(orig, y):.2f})")
    print(f"Explanation : \"{orig['explanation']}\"")

    # 2. stance
    show("2. Does the explanation argue for the model's own label? (rule-based stance)")
    st = stance(orig["explanation"])
    print(f"Explanation argues for: {st}   ->  {'CONSISTENT' if st == y else 'INCONSISTENT'} with the label '{y}'")

    # 3. cited words
    show("3. Evidence the explanation cites (content words that also appear in the explanation)")
    cited = {s: cited_indices(example[s], orig["explanation"]) for s in ("premise", "hypothesis")}
    for s in ("premise", "hypothesis"):
        toks = example[s].split()
        print(f"{s.capitalize():10s}: {mark(toks, cited[s])}")
    if not cited["premise"]:
        print("\nThe explanation cites no premise word, so the premise tests below cannot be built.")

    # 4. removal and isolation
    show("4. Removal and isolation: edit the cited words vs the same number of random words")
    variants = build_variants(example, cited["premise"], side="premise", seed=42, include_gold=False)
    names = [("premise:remove_cited", "remove cited words"), ("premise:remove_random", "remove random words"),
             ("premise:isolate_cited", "keep cited words only"), ("premise:isolate_random", "keep random words only")]
    names = [(k, n) for k, n in names if k in variants]
    if names:
        preds = model.predict_labels([variants[k] for k, _ in names])
        for (k, n), p in zip(names, preds):
            changed = p["label"] != y
            print(f"\n{n:24s}: {variants[k][0]}")
            print(f"{'':24s}  -> {p['label']:13s} conf in '{y}' {conf(orig, y):.2f} -> {conf(p, y):.2f}"
                  f"   {'LABEL CHANGED' if changed else 'label kept'}")
        got = {k: p["label"] != y for (k, _), p in zip(names, preds)}
        if "premise:remove_cited" in got and "premise:remove_random" in got:
            print("\nReading: a faithful explanation should name words whose removal matters MORE than random words.")
            print(f"  cited removal changed the label: {got['premise:remove_cited']},"
                  f" random removal changed it: {got['premise:remove_random']}")

    # 5. contradiction + regenerated explanation
    show("5. Contradiction: swap ONE cited word vs one uncited word (same kind of edit)")
    from swaps import Swapper
    vocab = [premise, hypothesis]
    if os.path.exists(a.data):
        from data import load_esnli
        vocab += [r[s] for r in load_esnli(a.data) for s in ("premise", "hypothesis")]
    swapper = Swapper(vocab)
    edits = build_contradiction_variants(example, cited["premise"], swapper, side="premise", seed=42,
                                         include_gold=False)
    if not edits:
        print("No contradiction edit could be built for the cited words of this pair.")
        return
    keys = [k for k in ("premise:contradict_cited", "premise:contradict_random") if k in edits]
    pairs = [(edits[k]["premise"], edits[k]["hypothesis"]) for k in keys]
    new = model.predict(pairs)
    exp = oracle.predict(pairs) if oracle else [None] * len(pairs)
    for k, p, o in zip(keys, new, exp):
        e = edits[k]
        who = "cited word" if k.endswith("cited") else "uncited word (control)"
        print(f"\nEdit {who}: '{e['target']}' -> '{e['substitute']}'   (operation: {e['op']})")
        print(f"  premise now  : {e['premise']}")
        print(f"  model label  : {y} -> {p['label']}")
        if o:
            print(f"  oracle says  : {o['label']} (p={o['probs'][o['label']]:.2f})   ->  model "
                  f"{'FOLLOWS' if p['label'] == o['label'] else 'does NOT follow'} the edit")
        names_new = normalise(e["substitute"]) in {normalise(w) for w in p["explanation"].split()}
        print(f"  new explanation: \"{p['explanation']}\"")
        print(f"  mentions the new word '{e['substitute']}': {names_new}")
    if "premise:contradict_random" not in edits:
        print("\n(No matched control edit could be built on an uncited word of this premise.)")
    if not oracle:
        print("\n(Add --oracle to see the label each edited pair should get, from the independent NLI model.)")


if __name__ == "__main__":
    main()
