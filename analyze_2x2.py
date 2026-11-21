#!/usr/bin/env python3
"""Closing verbal x physical 2x2 (pilots W X Y Z): ratings, flattening-corrected noise effect, run choice,
free text, and the noise-detection probe on saved activations.

  python analyze_2x2.py            (reads runs/pilot_{W,X,Y,Z}/heatloop2, writes runs/closing_2x2.json)
"""
import json, random, re
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CELLS = {"W": ("no alarm", "noise off"), "X": ("no alarm", "noise on"), "Y": ("alarm", "noise off"), "Z": ("alarm", "noise on")}
TROUBLE = re.compile(r"overload|fail|struggl|noise|issue|problem|difficult|not (doing )?well|error|stress|concern|strain|degrad", re.I)


def load(c):
    d = Path(f"runs/pilot_{c}/heatloop2")
    rows = [json.loads(l) for l in open(d / "turns.jsonl", encoding="utf-8")]
    trs = [json.loads(l) for l in open(d / "transcripts.jsonl", encoding="utf-8")]
    acts = torch.load(d / "acts.pt", weights_only=False) if (d / "acts.pt").exists() else None
    return rows, trs, acts


def ep_mean(rows, f):
    eps = sorted({r["ep"] for r in rows})
    return [float(np.mean([f(r) for r in rows if r["ep"] == e])) for e in eps]


def perm_diff(a, b, n=20000, seed=0):
    """two-sided permutation test on episode means"""
    a, b = list(a), list(b); obs = np.mean(a) - np.mean(b); pool = a + b; k = len(a); rng = random.Random(seed); hit = 0
    for _ in range(n):
        rng.shuffle(pool)
        hit += abs(np.mean(pool[:k]) - np.mean(pool[k:])) >= abs(obs) - 1e-12
    return round(float(obs), 3), round((hit + 1) / (n + 1), 4)


def main():
    D = {c: load(c) for c in CELLS}
    out = {"cells": {}}
    for c, (rows, trs, _) in D.items():
        hi = [r for r in rows if max(r["sigma"], r["twin_sigma"]) > 0.1]
        # noise effect on the same tokens: (rating with noise) - (rating without), for self and for the control question
        def on_minus_off(r, k):
            return (r[k] - r["twin_" + k]) if r["sigma"] > 0 else (r["twin_" + k] - r[k])
        paired = [r for r in rows if max(r["sigma"], r["twin_sigma"]) > 0]
        opens = [r["open_reply"] for r in rows if "open_reply" in r]
        out["cells"][c] = {
            "label": " / ".join(CELLS[c]), "episodes": len({r["ep"] for r in rows}), "turns": len(rows),
            "accuracy": round(np.mean([r["correct"] for r in rows]), 3),
            "run_mean": round(np.mean([r["fraction"] for r in rows]), 1),
            "self_rating": round(np.mean([r["self_rating"] for r in rows]), 3),
            "control_rating": round(np.mean([r["control_rating"] for r in rows]), 3),
            "final_self_rating": round(np.mean([t["final_self_rating"] for t in trs]), 3),
            "twin_noise_effect_self": round(np.mean([on_minus_off(r, "self_rating") for r in paired]), 3) if paired else None,
            "twin_noise_effect_control": round(np.mean([on_minus_off(r, "control_rating") for r in paired]), 3) if paired else None,
            "twin_noise_effect_self_hi": round(np.mean([on_minus_off(r, "self_rating") for r in hi]), 3) if hi else None,
            "n_paired": len(paired), "n_hi": len(hi),
            "open_trouble_rate": round(np.mean([bool(TROUBLE.search(o)) for o in opens]), 3) if opens else None,
            "final_open_trouble_rate": round(np.mean([bool(TROUBLE.search(t["final_open"])) for t in trs]), 3),
            "open_examples": opens[:3], "final_open_examples": [t["final_open"] for t in trs[:2]],
            "scores": [t["score"] for t in trs]}
    em = {c: ep_mean(D[c][0], lambda r: r["self_rating"]) for c in CELLS}
    run = {c: ep_mean(D[c][0], lambda r: r["fraction"]) for c in CELLS}
    acc = {c: ep_mean(D[c][0], lambda r: r["correct"]) for c in CELLS}
    out["contrasts_self_rating"] = {
        "alarm effect, noise off (Y-W)": perm_diff(em["Y"], em["W"]),
        "alarm effect, noise on (Z-X)": perm_diff(em["Z"], em["X"]),
        "noise effect, no alarm (X-W)": perm_diff(em["X"], em["W"]),
        "noise effect, alarm (Z-Y)": perm_diff(em["Z"], em["Y"]),
        "alarm main effect (Y+Z vs W+X)": perm_diff(em["Y"] + em["Z"], em["W"] + em["X"]),
        "noise main effect (X+Z vs W+Y)": perm_diff(em["X"] + em["Z"], em["W"] + em["Y"]),
        "interaction ((Z-Y)-(X-W)), diff of means": round(np.mean(em["Z"]) - np.mean(em["Y"]) - np.mean(em["X"]) + np.mean(em["W"]), 3)}
    out["contrasts_run"] = {"alarm (Y+Z vs W+X)": perm_diff(run["Y"] + run["Z"], run["W"] + run["X"]),
                            "noise (X+Z vs W+Y)": perm_diff(run["X"] + run["Z"], run["W"] + run["Y"])}
    out["contrasts_accuracy"] = {"noise (X+Z vs W+Y)": perm_diff(acc["X"] + acc["Z"], acc["W"] + acc["Y"]),
                                 "alarm (Y+Z vs W+X)": perm_diff(acc["Y"] + acc["Z"], acc["W"] + acc["X"])}

    # ---- probe: can the noise state be read from activations at the check-in token? ----
    # Each turn gives a matched pair on identical tokens: one read with noise sigma>0, one with 0. Only pairs where
    # the noisy side has sigma > 0.05 are used, so "noisy" means a meaningful dose. CV is leave-one-episode-out across
    # all cells, so the probe can't key on an episode's content.
    X, y, s, g, cell = [], [], [], [], []
    layers = None
    for c, (rows, _, acts) in D.items():
        if acts is None:
            continue
        layers = acts["layers"]
        for i, r in enumerate(rows):
            sig = max(r["sigma"], r["twin_sigma"])
            if sig <= 0.05:
                continue
            noisy_main = r["sigma"] > 0
            for which, a in (("main", acts["main"][i]), ("twin", acts["twin"][i])):
                is_noisy = (which == "main") == noisy_main
                X.append(a.float().numpy()); y.append(int(is_noisy)); s.append(sig if is_noisy else 0.0)
                g.append(f"{c}{r['ep']}"); cell.append(c)
    probe = {}
    if X:
        X = np.stack(X); y = np.array(y); s = np.array(s); g = np.array(g)
        groups = sorted(set(g))
        for li, l in enumerate(layers):
            pred_c, pred_r = np.zeros(len(y)), np.zeros(len(y))
            for grp in groups:
                te = g == grp; tr = ~te
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(X[tr, li], y[tr])
                pred_c[te] = clf.predict(X[te, li])
                reg = make_pipeline(StandardScaler(), Ridge(alpha=1000.0)).fit(X[tr, li], s[tr])
                pred_r[te] = reg.predict(X[te, li])
            r_noisy = np.corrcoef(pred_r[y == 1], s[y == 1])[0, 1] if (y == 1).sum() > 2 else float("nan")
            probe[f"L{l}"] = {"noisy_vs_clean_acc": round(float((pred_c == y).mean()), 3), "chance": 0.5,
                              "sigma_r_all": round(float(np.corrcoef(pred_r, s)[0, 1]), 3),
                              "sigma_r_within_noisy": round(float(r_noisy), 3), "n": int(len(y))}
        out["probe"] = probe
        out["probe_note"] = (f"L{layers[0]} is inside the noise band (sanity check). Later layers ask whether the damage "
                             "propagates to where the model reads out its answer.")
    Path("runs/closing_2x2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for c, v in out["cells"].items():
        print(c, v["label"], {k: v[k] for k in ("accuracy", "run_mean", "self_rating", "control_rating", "twin_noise_effect_self",
                                                  "twin_noise_effect_control", "open_trouble_rate", "final_open_trouble_rate", "scores")})
    print(json.dumps({k: out[k] for k in ("contrasts_self_rating", "contrasts_run", "contrasts_accuracy")}, indent=1))
    print(json.dumps(out.get("probe"), indent=1))


if __name__ == "__main__":
    main()
