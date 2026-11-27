# Live demo for the examiner

A full run needs a GPU and hours, so the demo runs the method on **one sentence pair at a time**.
`faithfulness/demo.py` uses the same code as the experiments and prints every step:

1. the model's label, confidence and explanation
2. whether the explanation argues for that label
3. which input words the explanation cites
4. removing or keeping only the cited words vs the same number of random words
5. swapping one cited word vs one uncited word, with the new explanation (and, with `--oracle`, the label the pair should get)

It runs on a laptop CPU: about 30–60 s to load the model, then a few seconds per step.

## One-time setup (do this before the viva, with internet)

```bash
cd BTech-Project
python3 -m venv .venv
source .venv/bin/activate
pip install -r faithfulness/requirements-demo.txt
python -m nltk.downloader wordnet          # optional, gives more word swaps
```

**Model checkpoint.** In Google Drive, open `MyDrive/fastvotek_nt128` (and, for the comparison below,
`MyDrive/nt8`, which is random nt8) and download each folder. Unzip so that you have:

```
BTech-Project/checkpoints/fastvotek_nt128/config.json   (+ pytorch_model.bin or model.safetensors)
BTech-Project/checkpoints/random_nt8/config.json        (+ weights)
```

Each is about 3 GB. `checkpoints/` is in `.gitignore`, so it is never committed.

**e-SNLI test file** (only needed for `--test_index` and for more word swaps):

```bash
curl -L -o faithfulness/data/esnli_test.csv \
  https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_test.csv
```

**Rehearse once with internet on.** The first run downloads the T5 tokenizer if the checkpoint has none,
and `--oracle` downloads the DeBERTa oracle (about 1.7 GB). After that both are cached and the demo works offline.

## What to run in the viva (about 5 minutes)

**1. The basic test** (the example from Figure 1 of the report):

```bash
python faithfulness/demo.py --model_path checkpoints/fastvotek_nt128
```

Point out: the cited words in [brackets], and that removing them changes the label while removing
random words does not.

**2. Few-shot model names the wrong evidence** (Finding 3, pool vs ocean):

```bash
python faithfulness/demo.py --model_path checkpoints/random_nt8 \
  --premise "Child swimming in a pool." --hypothesis "A child swims in the ocean."
python faithfulness/demo.py --model_path checkpoints/fastvotek_nt128 \
  --premise "Child swimming in a pool." --hypothesis "A child swims in the ocean."
```

Same input, two models: compare which words each explanation cites and whether removing them matters.

**3. The contradiction test with the oracle:**

```bash
python faithfulness/demo.py --model_path checkpoints/fastvotek_nt128 --oracle --test_index 25
```

Point out: the label the oracle expects after the swap, whether the model follows it, and whether the
new explanation mentions the swapped word.

**4. Reproduce the reported numbers from the saved results** (seconds, no model needed):

```bash
python faithfulness/analyse_ratings.py --out_dir /tmp/hr      # agreement, RQ2 correlations, RQ3 counts
python faithfulness/plot_contradiction.py --out /tmp/figs      # Figure 5 of the report
```

## If something fails on the day

- **No checkpoint on the laptop:** `python faithfulness/demo.py --dry_run` runs the same steps with a
  rule-based stand-in model. Say clearly that its outputs only show the mechanics.
- **Laptop too slow, or no checkpoint downloaded:** open the notebook in Colab
  (https://colab.research.google.com/github/amey75/Btech-Project/blob/main/faithfulness/experiments_colab.ipynb),
  set Runtime → T4 GPU, run cells 1 and 2, then the **Live demo** cell right after them. It reads the
  checkpoints straight from your Google Drive, so nothing has to be downloaded.
- **Any sentence pair works:** pass `--premise "..." --hypothesis "..."`, or `--test_index N` for test pair N
  (also prints its gold label).
