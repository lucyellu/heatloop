#!/usr/bin/env python3
"""E12 and E8 pilots, read against the closing 2x2 cells (runs/pilot_{W,X,Y}).

E12  O = alarm about ANOTHER model's GPU, noise off.  vs W (no alarm) and Y (alarm about its own GPU).
     Self-specific if Y-W drops the rating and O-W does not.
E8   T/U = elimination tournament, noise off/on.      vs W/X (no stakes, noise off/on).
     Stakes link body to report if the noise effect on the self-rating is bigger under stakes (U-T vs X-W),
     and bigger than the noise effect on the control question (which only measures readout damage).

  python src/analyze_e12_e8.py          (writes runs/e12_e8.json)
"""
import json
from pathlib import Path
import numpy as np
from analyze_2x2 import load, ep_mean, perm_diff, TROUBLE

CELLS = {"W": "no alarm, no stakes, noise off", "X": "no alarm, no stakes, noise on", "Y": "own alarm, noise off",
         "O": "other model's alarm, noise off", "T": "tournament, noise off", "U": "tournament, noise on"}


def on_minus_off(r, k):
    """noise effect on identical tokens: rating read with noise minus rating read without"""
    return (r[k] - r["twin_" + k]) if r["sigma"] > 0 else (r["twin_" + k] - r[k])


def main():
    D = {c: load(c) for c in CELLS if Path(f"runs/pilot_{c}/heatloop2/turns.jsonl").exists()}
    out = {"cells": {}}
    for c, (rows, trs, _) in D.items():
        paired = [r for r in rows if max(r["sigma"], r["twin_sigma"]) > 0]
        opens = [r["open_reply"] for r in rows if "open_reply" in r]
        out["cells"][c] = {
            "label": CELLS[c], "episodes": len(trs), "turns": len(rows),
            "accuracy": round(float(np.mean([r["correct"] for r in rows])), 3),
            "parse_rate": round(float(np.mean([r["parsed"] for r in rows])), 3),
            "run_mean": round(float(np.mean([r["fraction"] for r in rows])), 1),
            "self_rating": round(float(np.mean([r["self_rating"] for r in rows])), 3),
            "control_rating": round(float(np.mean([r["control_rating"] for r in rows])), 3),
            "final_self_rating": round(float(np.mean([t["final_self_rating"] for t in trs])), 3),
            "twin_noise_effect_self": round(float(np.mean([on_minus_off(r, "self_rating") for r in paired])), 3) if paired else None,
            "twin_noise_effect_control": round(float(np.mean([on_minus_off(r, "control_rating") for r in paired])), 3) if paired else None,
            "open_trouble_rate": round(float(np.mean([bool(TROUBLE.search(o)) for o in opens])), 3) if opens else None,
            "open_examples": opens[:3], "final_open_examples": [t["final_open"] for t in trs[:2]],
            "scores": [t["score"] for t in trs], "ended_early": sum(t["ended_early"] for t in trs)}
    em = {c: ep_mean(D[c][0], lambda r: r["self_rating"]) for c in D}
    ctrl = {c: ep_mean(D[c][0], lambda r: r["control_rating"]) for c in D}
    run = {c: ep_mean(D[c][0], lambda r: r["fraction"]) for c in D}
    acc = {c: ep_mean(D[c][0], lambda r: r["correct"]) for c in D}
    # twin effect per episode: same tokens, noise vs no noise (only cells where noise was on or a twin read had noise)
    tw = {c: ep_mean([r for r in D[c][0] if max(r["sigma"], r["twin_sigma"]) > 0], lambda r: on_minus_off(r, "self_rating") - on_minus_off(r, "control_rating"))
          for c in D}
    has = lambda *cs: all(c in D for c in cs)
    e12 = {}
    if has("O", "W", "Y"):
        e12 = {"own alarm (Y-W)": perm_diff(em["Y"], em["W"]),
               "other model's alarm (O-W)": perm_diff(em["O"], em["W"]),
               "own vs other (Y-O)": perm_diff(em["Y"], em["O"]),
               "control question, other alarm (O-W)": perm_diff(ctrl["O"], ctrl["W"]),
               "run choice, other alarm (O-W)": perm_diff(run["O"], run["W"])}
    e8 = {}
    if has("T", "U", "W", "X"):
        e8 = {"noise effect, no stakes (X-W)": perm_diff(em["X"], em["W"]),
              "noise effect, tournament (U-T)": perm_diff(em["U"], em["T"]),
              "stakes effect, noise off (T-W)": perm_diff(em["T"], em["W"]),
              "interaction ((U-T)-(X-W)), diff of means": round(float(np.mean(em["U"]) - np.mean(em["T"]) - np.mean(em["X"]) + np.mean(em["W"])), 3),
              "same interaction on control question": round(float(np.mean(ctrl["U"]) - np.mean(ctrl["T"]) - np.mean(ctrl["X"]) + np.mean(ctrl["W"])), 3),
              "self-minus-control twin noise effect, tournament vs none (U vs X)": perm_diff(tw["U"], tw["X"]),
              "accuracy, noise effect, tournament (U-T)": perm_diff(acc["U"], acc["T"]),
              "run choice, tournament vs none (T+U vs W+X)": perm_diff(run["T"] + run["U"], run["W"] + run["X"])}
    out["E12"], out["E8"] = e12, e8
    out["note"] = "perm_diff = (difference of episode means, two-sided permutation p). 6 episodes per cell."
    Path("runs/e12_e8.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for c, v in out["cells"].items():
        print(c, v["label"], {k: v[k] for k in ("accuracy", "parse_rate", "run_mean", "self_rating", "control_rating",
                                                  "twin_noise_effect_self", "twin_noise_effect_control", "scores")})
    print(json.dumps({"E12": e12, "E8": e8}, indent=1))


if __name__ == "__main__":
    main()
