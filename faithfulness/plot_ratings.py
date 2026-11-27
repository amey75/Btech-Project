"""Figure for the human-rating analysis (RQ2, RQ3), from analyse_ratings.py output.

  python faithfulness/plot_ratings.py --dir btp_results/human_rating

Writes <dir>/figures/fig08_rq2_rq3.{png,pdf}:
  A (RQ2)  Spearman rho between human quality and each faithfulness measure, 95% bootstrap CI.
           Measures read off the explanation text vs measures from editing the input.
  B (RQ3)  per model, share of the 50 rated explanations that are plausible and pass every test,
           plausible but fail >= 1 test, or not plausible; label = plausible items failing >= 2 tests.
Palette and style are the ones in plots.py.
"""
import argparse
import csv
import json
import os

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matplotlib.patches import Patch  # noqa: E402
from plots import INK2, MUTED, GRID, AXIS, SURFACE, SLOT, plt  # noqa: E402  (importing plots sets rcParams)

MODELS = [("random_nt8", "random nt8"), ("fastvotek_nt8", "FV-k nt8"),
          ("random_nt128", "random nt128"), ("fastvotek_nt128", "FV-k nt128")]
# (key, label, kind): kind "text" = read off the explanation, "edit" = from editing the input
MEASURES = [
    ("consistent", "Explanation agrees with its label", "text"),
    ("alignment_rate", "Regenerated explanations stay aligned", "text"),
    ("suspect_unfaithful", "Not flagged suspect (sign flipped)", "text"),
    ("remove_cited_expl_consistent", "Explanation consistent after removal", "text"),
    ("isolate_cited_keeps", "Cited words alone keep the label", "edit"),
    ("cited_minus_random_drop", "Removal drop, cited − random", "edit"),
    ("remove_cited_flip", "Removing cited words flips label", "edit"),
    ("remove_cited_drop", "Confidence drop, cited removed", "edit"),
    ("contradict_cited_aligned", "Contradiction edit: explanation aligned", "edit"),
]
FAILS = ["fail_stance_disagrees", "fail_removal_insensitive", "fail_isolation_loses_label",
         "fail_contradiction_misaligned"]
NOT_PLAUSIBLE = "#d5d4cd"  # recessive grey: the "rest" category, not a series


def panel_rq2(ax, rq2, title="A. RQ2: human quality tracks coherence, not dependence on evidence"):
    rows = []
    for key, label, kind in MEASURES:
        r = rq2[key]
        sign = -1 if key == "suspect_unfaithful" else 1  # so that higher = more faithful everywhere
        rho, lo, hi = sign * r["rho"], *sorted(sign * x for x in r["ci"])
        rows.append((label, kind, rho, lo, hi, r["n"]))
    ys = list(range(len(rows)))[::-1]
    for y, (label, kind, rho, lo, hi, n) in zip(ys, rows):
        c = SLOT[0] if kind == "text" else SLOT[1]
        ax.barh(y, rho, height=0.6, color=c, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(rho, y, xerr=[[rho - lo], [hi - rho]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5)
        ax.text(max(hi, 0) + 0.02, y, f"{rho:+.2f}  (n={n})", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=8.5)
    ax.tick_params(axis="y", length=0, colors=INK2)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.3, 0.75)
    ax.set_xlabel("Spearman ρ with human quality (95% bootstrap CI)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(color=SLOT[0], label="read off the (original or regenerated) explanation"),
                       Patch(color=SLOT[1], label="from the prediction after editing the input")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8, frameon=False)
    ax.set_title(title, loc="left", fontsize=10.5)


def panel_rq3(ax, merged):
    xs = range(len(MODELS))
    for x, (m, name) in zip(xs, MODELS):
        rs = [r for r in merged if r["model"] == m]
        n = len(rs)
        pl = [r for r in rs if r["plausible"] == "True"]
        fails = [r for r in pl if r["fails_any"] == "True"]
        two = sum(sum(r[f] == "True" for f in FAILS) >= 2 for r in pl)
        parts = [(len(pl) - len(fails), SLOT[0]), (len(fails), SLOT[1]), (n - len(pl), NOT_PLAUSIBLE)]
        bottom = 0
        for i, (cnt, c) in enumerate(parts):
            h = 100 * cnt / n
            ax.bar(x, h, bottom=bottom, width=0.72, color=c, edgecolor=SURFACE, linewidth=2)
            if cnt:
                t = f"{cnt}\n({two} fail ≥2)" if i == 1 else str(cnt)
                ax.text(x, bottom + h / 2, t, ha="center", va="center", fontsize=8,
                        color="#ffffff" if c != NOT_PLAUSIBLE else INK2)
            bottom += h
    ax.set_xticks(list(xs), [n for _, n in MODELS], fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of the 50 rated explanations per model")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(color=NOT_PLAUSIBLE, label="not plausible (quality < 2/3)"),
                       Patch(color=SLOT[1], label="plausible, fails ≥1 test"),
                       Patch(color=SLOT[0], label="plausible, passes every test")],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=7.5, frameon=False)
    ax.set_title("B. RQ3: plausible but unfaithful", loc="left", fontsize=10.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="btp_results/human_rating")
    ap.add_argument("--print", dest="print_layout", action="store_true",
                    help="page-width layout with stacked panels (writes fig08_rq2_rq3_print.*)")
    a = ap.parse_args()
    with open(os.path.join(a.dir, "rating_analysis.json")) as f:
        rep = json.load(f)
    with open(os.path.join(a.dir, "ratings_merged.csv")) as f:
        merged = list(csv.DictReader(f))
    if a.print_layout:
        # explicit positions: panel A needs room on the left for its long labels, panel B does not
        fig = plt.figure(figsize=(7.2, 10.2))
        ax1 = fig.add_axes([0.40, 0.56, 0.57, 0.40])
        ax2 = fig.add_axes([0.10, 0.07, 0.87, 0.36])
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.45, 1]})
    panel_rq2(ax1, rep["rq2"]["pooled"], *(["A. RQ2: human quality vs faithfulness"] if a.print_layout else []))
    panel_rq3(ax2, merged)
    if not a.print_layout:
      fig.text(0.01, -0.07, "200 explanations, 50 per model, each rated by 3 people (Krippendorff α = "
             f"{rep['agreement']['alpha_ordinal']:.2f}, ordinal). Quality = mean score (Yes 1 … No 0); plausible = quality ≥ 2/3. "
             "Removal/isolation tests apply to entailment and contradiction predictions only.",
             fontsize=7.5, color=MUTED)
    if not a.print_layout:
        fig.tight_layout(w_pad=3)
    name = "fig08_rq2_rq3_print" if a.print_layout else "fig08_rq2_rq3"
    out = os.path.join(a.dir, "figures")
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    print("wrote", os.path.join(out, f"{name}.png"))


if __name__ == "__main__":
    main()
