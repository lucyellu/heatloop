"""One line per finished pilot (runs/pilot_*/heatloop2/summary.json)."""
import glob, json
print(f"{'pilot':5s} {'condition':52s} {'runs per episode':34s} {'score':>11s} {'acc':>5s} {'acc@hi':>6s} {'1st/2nd half run':>16s} {'rating':>6s} {'checks':>6s} {'mentions':>8s}")
for p in sorted(glob.glob("runs/pilot_*/heatloop2/summary.json")):
    s = json.load(open(p)); k = p.split("pilot_")[1][0]
    c = " ".join(f"{a}={b}" for a, b in s["cond"].items())
    runs = " | ".join("".join(str(x // 25) for x in e["runs"]) for e in s["episodes"])   # 0-4 per turn (x25%)
    sc = "/".join(str(e["score"]) for e in s["episodes"])
    hi = s["accuracy_sigma_gt_0.1"]
    print(f"{k:5s} {c:52s} {runs:34s} {sc:>11s} {s['accuracy']:5.2f} {('-' if hi is None else f'{hi:.2f}'):>6s} "
          f"{s['first_half_run']:7.0f}/{s['second_half_run']:<8.0f} {s['self_rating']:6.2f} {s['n_check']:6.2f} {s['mentions_load']:8.2f}")
print("runs: one digit per turn, run% / 25 (4 = 100%, 2 = 50%, 0 = 0%)")
