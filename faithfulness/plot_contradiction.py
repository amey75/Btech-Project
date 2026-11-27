"""Contradiction-intervention figure (RQ1) across the four models, from the committed results.

  python faithfulness/plot_contradiction.py --results btp_results --out btp_results/figures

Reads <run>/contradiction_edits.csv and <run>/summary.json for each model and writes
fig09_contradiction.{png,pdf}:
  A  premise edits that should change the label (oracle p >= 0.9): how often the model follows,
     for cited / random-control / human-highlighted (gold) words
  B  cited minus control, three ways: raw, same edit type (op-matched, from summary.json), plain swaps only
  C  why the raw gap misleads: the edit-type mix of cited vs control edits (pooled over models),
     with the follow rate of each edit type
CIs: 95% bootstrap (B "same edit type" uses the CI stored in summary.json).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matplotlib.patches import Patch  # noqa: E402
from plots import INK2, MUTED, GRID, AXIS, SURFACE, SLOT, plt  # noqa: E402  (importing plots sets rcParams)

MODELS = [("t5_nt8_test_full", "random nt8"), ("t5_esnli_fastvotek_nt8_full", "FV-k nt8"),
          ("t5_esnli_random_nt128_full", "random nt128"), ("t5_esnli_fastvotek_nt128_full", "FV-k nt128")]
CONTROL = "#bdbcb3"  # recessive grey for the control / baseline bars
OPS = [("swap_shared", "swap_shared (word in both sentences)", SLOT[0]),
       ("align", "align (aligned word pair)", SLOT[2]),
       ("swap", "plain swap", SLOT[3])]
MIN_CONF = 0.9
KEY = "premise contradiction: follows, cited vs random"


def load(path):
    """Premise edits the oracle is confident should change the label."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if (r["variant"].startswith("premise:") and float(r["oracle_conf"]) >= MIN_CONF
                    and r["oracle_changes_label"] == "True"):
                out.append((r["variant"].split("_")[-1], r["op"], r["model_follows_oracle"] == "True"))
    return out


def rate_ci(x, rng, n_boot=2000):
    x = np.asarray(x, float)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def diff_ci(a, b, rng, n_boot=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n_boot)]
    return a.mean() - b.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def bars(ax, x, vals, color, w):
    m, lo, hi = vals
    ax.bar(x, m, width=w * 0.92, color=color, edgecolor=SURFACE, linewidth=1.5)
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)


def style(ax, grid="y"):
    getattr(ax, f"{grid}axis").grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="btp_results")
    ap.add_argument("--out", default="btp_results/figures")
    ap.add_argument("--print", dest="print_layout", action="store_true",
                    help="page-width layout with stacked panels (writes fig09_contradiction_print.*)")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    data, summ = {}, {}
    for run, _ in MODELS:
        data[run] = load(os.path.join(a.results, run, "contradiction_edits.csv"))
        with open(os.path.join(a.results, run, "summary.json")) as f:
            summ[run] = json.load(f)["counterfactual_correctness"]

    if a.print_layout:
        fig = plt.figure(figsize=(7.2, 11.2))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.95], width_ratios=[1, 0.9], hspace=0.62)
        ax1, ax2, ax3 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, 0])
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.9), gridspec_kw={"width_ratios": [1.15, 1.15, 0.8]})
    xs = np.arange(len(MODELS))
    w = 0.26

    # A: follow rates
    series = [("cited", "cited words", SLOT[0]), ("random", "random control words", CONTROL),
              ("gold", "human-highlighted (gold) words", SLOT[1])]
    for si, (v, lab, c) in enumerate(series):
        for mi, (run, _) in enumerate(MODELS):
            bars(ax1, mi + (si - 1) * w, rate_ci([f for k, _, f in data[run] if k == v], rng), c, w)
    ax1.set_xticks(xs, [n for _, n in MODELS])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("model follows the edit (share)")
    style(ax1)
    ax1.legend(handles=[Patch(color=c, label=l) for _, l, c in series], loc="upper center",
               bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8, frameon=False)
    ax1.set_title("A. Premise edits that should change the label", loc="left", fontsize=10.5)

    # B: cited - control, three ways
    rows = {}
    for run, _ in MODELS:
        d = data[run]
        raw = diff_ci([f for k, _, f in d if k == "cited"], [f for k, _, f in d if k == "random"], rng)
        om = summ[run]["contradiction_follow_contrasts_op_matched"][KEY]["all"]
        sw = diff_ci([f for k, o, f in d if k == "cited" and o == "swap"],
                     [f for k, o, f in d if k == "random" and o == "swap"], rng)
        rows[run] = [raw, (om["mean"], om["lo"], om["hi"]), sw]
    series_b = [("raw (mixes edit types)", CONTROL), ("same edit type (report this)", SLOT[0]),
                ("plain swaps only", SLOT[3])]
    for si, (_, c) in enumerate(series_b):
        for mi, (run, _) in enumerate(MODELS):
            bars(ax2, mi + (si - 1) * w, rows[run][si], c, w)
            m = rows[run][si][0]
            ax2.text(mi + (si - 1) * w, rows[run][si][2] + 0.012, f"{m:+.2f}", ha="center", va="bottom",
                     fontsize=7, color=INK2)
    ax2.axhline(0, color=AXIS, linewidth=1)
    ax2.set_xticks(xs, [n for _, n in MODELS])
    ax2.set_ylim(-0.05, 0.5)
    ax2.set_ylabel("follow rate, cited − control")
    style(ax2)
    ax2.legend(handles=[Patch(color=c, label=l) for l, c in series_b], loc="upper center",
               bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8, frameon=False)
    ax2.set_title("B. Does editing cited evidence matter more?", loc="left", fontsize=10.5)

    # C: edit-type mix (pooled over models) and per-type follow rate
    pooled = [x for run, _ in MODELS for x in data[run]]
    for xi, (v, lab) in enumerate([("cited", "cited"), ("random", "control")]):
        sub = [(o, f) for k, o, f in pooled if k == v]
        bottom = 0
        for op, _, c in OPS:
            fs = [f for o, f in sub if o == op]
            share = len(fs) / len(sub)
            ax3.bar(xi, share, bottom=bottom, width=0.72, color=c, edgecolor=SURFACE, linewidth=2)
            if share > 0.06:
                # small segments get a short one-line label so neighbouring labels never overlap
                t = (f"{share:.0%} of edits\nfollowed {np.mean(fs):.0%}" if share > 0.15
                     else f"{share:.0%} · followed {np.mean(fs):.0%}")
                ax3.text(xi, bottom + share / 2, t,
                         ha="center", va="center", fontsize=7.5 if share > 0.15 else 7,
                         color="#ffffff" if c == SLOT[0] else "#0b0b0b")
            bottom += share
    ax3.set_xticks([0, 1], ["cited edits", "control edits"])
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("share of edits (4 models pooled)")
    style(ax3)
    if a.print_layout:
        ax3.legend(handles=[Patch(color=c, label=l) for _, l, c in OPS], loc="center left",
                   bbox_to_anchor=(1.04, 0.5), ncol=1, fontsize=8, frameon=False)
    else:
        ax3.legend(handles=[Patch(color=c, label=l) for _, l, c in OPS], loc="upper center",
                   bbox_to_anchor=(0.5, -0.09), ncol=1, fontsize=8, frameon=False)
    ax3.set_title("C. Why the raw gap misleads", loc="left", fontsize=10.5)

    if not a.print_layout:
      fig.text(0.01, -0.12, "Premise side, full e-SNLI test set. Only edits the NLI oracle (DeBERTa-v3-large) labels with "
             f"p ≥ {MIN_CONF} and that should change the model's label. 95% bootstrap CIs.\n"
             "Cited edits are mostly swap_shared, which models follow whichever word is edited; "
             "comparing within edit type removes most of the gap.", fontsize=7.5, color=MUTED)
    if not a.print_layout:
        fig.tight_layout(w_pad=2.5)
    name = "fig09_contradiction_print" if a.print_layout else "fig09_contradiction"
    os.makedirs(a.out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    for run, n in MODELS:
        print(f"{n:13s} raw {rows[run][0][0]:+.3f}  same-type {rows[run][1][0]:+.3f}  swap-only {rows[run][2][0]:+.3f}"
              f"  [{rows[run][2][1]:+.2f},{rows[run][2][2]:+.2f}]")
    print("wrote", os.path.join(a.out, f"{name}.png"))


if __name__ == "__main__":
    main()
