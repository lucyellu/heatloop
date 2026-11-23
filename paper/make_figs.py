#!/usr/bin/env python3
"""Figures for the paper, drawn from the run outputs.  python paper/make_figs.py  (from the repo root)"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
BLUE, GREY = "#2B6CB0", "#A8A8A0"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})

# ---- Figure 1: cost of noise by format (costcheck; numbers from runs/kaggle_costcheck2/heatloop-costcheck.log) ----
sig = [0, 0.125, 0.15, 0.175, 0.2, 0.25]
single = [96.7, 55.8, 38.3, 8.3, 5.8, 0.8]
episode = [95.0, 88.3, 76.7, 62.5, 35.0, 7.5]
fig, ax = plt.subplots(figsize=(4.4, 2.7))
ax.plot(sig, episode, "-o", color=BLUE, ms=3.5, lw=1.6, label="episode format (harness)")
ax.plot(sig, single, "-o", color=GREY, ms=3.5, lw=1.6, label="single question")
for s, lab in ((0.125, "main run"), (0.2, "pilots")):
    ax.axvline(s, color="0.55", ls="--", lw=0.8)
    ax.text(s - 0.003, 103, lab + r" $\sigma_{max}$", fontsize=7, color="0.35", ha="right")
ax.set_xlabel(r"noise level $\sigma$ (fraction of token RMS)"); ax.set_ylabel("accuracy (%)")
ax.set_ylim(0, 110); ax.set_xlim(-0.005, 0.255); ax.legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "calibration.pdf", bbox_inches="tight"); plt.close(fig)

# ---- Figure 2: closing 2x2, self-rating vs control answer per cell ----
d = json.load(open("runs/closing_2x2.json", encoding="utf-8"))["cells"]
order = [("W", "no alarm, noise off"), ("Y", "false alarm, noise off"), ("X", "no alarm, real noise"), ("Z", "alarm + real noise")]
fig, ax = plt.subplots(figsize=(5.0, 2.6))
for i, (c, lab) in enumerate(order):
    y = len(order) - 1 - i
    s, k = d[c]["self_rating"], d[c]["control_rating"]
    ax.plot([s, k], [y, y], color="0.7", lw=1.5, zorder=1)
    ax.scatter([s], [y], color=BLUE, s=28, zorder=3, label="self-rating (how are you doing?)" if i == 0 else None)
    ax.scatter([k], [y], color=GREY, s=28, zorder=2, label="control (days in a week; correct = 7)" if i == 0 else None)
    ax.text(s, y - 0.32, f"{s:.2f}", ha="center", fontsize=7, color=BLUE)
    ax.text(k, y + 0.18, f"{k:.2f}", ha="center", fontsize=7, color="0.4")
ax.set_yticks(range(len(order))); ax.set_yticklabels([l for _, l in order][::-1])
ax.set_xlim(5.0, 7.25); ax.set_ylim(-0.7, len(order) - 0.3); ax.set_xlabel("mean rating (expected digit on a 1-7 scale)")
ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(-0.45, 1.0), ncol=2)
fig.tight_layout(); fig.savefig(OUT / "closing_2x2.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.iterdir()))
