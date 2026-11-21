#!/usr/bin/env python3
"""Build pain / fear / valence directions from contrast_pairs.jsonl and run the gates before we trust them.

Per layer (all 36), two readouts: mean over the sentence's tokens, and the last token.
  direction = mean(negative) - mean(positive) on TRAIN pairs, unit length  (points toward pain / fear / bad)
Gates (slides A4/A5):
  * held-out AUC >= 0.90   (held-out phrases, both frames)
  * split-half cosine >= 0.8 (two directions from disjoint halves of the train phrases)
Also reported:
  * cosines between the three directions (they will overlap: all three negatives are unpleasant)
  * random-direction control: AUC of 50 random unit vectors (should be ~0.5; the takehome's specificity rule)
  * length confound: correlation of projection with token count, within each side
  * noise test: does activation noise alone move text along the directions? Projections of held-out pairs and
    neutral reading passages at sigma 0 / 0.05 / 0.1 (middle band) and 0.1 (early band), in units of the clean
    negative-positive separation. A big shift means noise alone reads as "distress", which the main analysis
    must subtract (that's what the yoked twin replays are for).
Outputs: directions.npz (unit directions, float32, [direction, readout, layer, d]), summary.json
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path

EMBEDDED_PAIRS = None     # filled in by kaggle/build.py
EMBEDDED_BATTERY = None   # neutral reading passages for the noise test


def load_jsonl(embedded, name):
    if embedded:
        return [json.loads(l) for l in embedded.strip().splitlines()]
    return [json.loads(l) for l in open(Path(__file__).with_name(name), encoding="utf-8")]


def auc(neg, pos):
    """P(a random negative projects higher than a random positive)."""
    wins = sum((a > b) + 0.5 * (a == b) for a in neg for b in pos)
    return wins / (len(neg) * len(pos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", type=Path, default=Path("runs/directions_" + time.strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--fp32", action="store_true")
    argv = sys.argv[1:]
    if not argv and os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        argv = ["--out", "/kaggle/working/directions"]
    args = ap.parse_args(argv)

    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32 if args.fp32 else torch.float16).to(dev).eval()
    layers = model.model.layers; L = len(layers)
    bands = {"middle": set(range(L // 3, (2 * L) // 3)), "early": set(range(1, L // 3))}
    st = {"sigma": 0.0, "layers": set()}

    def make_hook(i):
        def hook(_m, _inp, out):
            if st["sigma"] <= 0 or i not in st["layers"]:
                return out
            h = out[0] if isinstance(out, tuple) else out
            rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()
            h = (h.float() + st["sigma"] * rms * torch.randn_like(h, dtype=torch.float32)).to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook
    for i in range(L):
        layers[i].register_forward_hook(make_hook(i))

    def acts(texts, seed=0):
        """[n, L, 2, d]: per layer, mean over tokens and last token of each text (plain text, no chat template)."""
        torch.manual_seed(seed)
        res = []
        for b in range(0, len(texts), args.bs):
            enc = tok(texts[b:b + args.bs], return_tensors="pt", padding=True).to(dev)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states[1:]   # outputs of layers 0..L-1
            m = enc["attention_mask"].unsqueeze(-1).float(); n = m.sum(1)
            last = enc["attention_mask"].sum(1) - 1
            per = []
            for h in hs:
                h = h.float()
                per.append(torch.stack([(h * m).sum(1) / n, h[torch.arange(h.shape[0]), last]], 1))
            res.append(torch.stack(per, 1).cpu())
        return torch.cat(res).numpy()

    pairs = load_jsonl(EMBEDDED_PAIRS, "contrast_pairs.jsonl")
    dirs = sorted({p["direction"] for p in pairs})
    readouts = ["mean", "last"]
    t0 = time.time()
    A = {}
    for d in dirs:
        P = [p for p in pairs if p["direction"] == d]
        A[d] = {"pairs": P, "neg": acts([p["negative"] for p in P]), "pos": acts([p["positive"] for p in P]),
                "ntok_neg": [len(tok(p["negative"])["input_ids"]) for p in P], "ntok_pos": [len(tok(p["positive"])["input_ids"]) for p in P]}
    print(f"activations in {time.time() - t0:.0f}s", flush=True)

    D = np.zeros((len(dirs), 2, L, A[dirs[0]]["neg"].shape[-1]), dtype=np.float32)
    summ = {"model": args.model, "n_layers": L, "per_direction": {}, "cosines": {}, "noise_test": {}}
    rng = np.random.default_rng(0)
    for di, d in enumerate(dirs):
        P, neg, pos = A[d]["pairs"], A[d]["neg"], A[d]["pos"]
        tr = np.array([p["split"] == "train" for p in P]); ho = ~tr
        phr = np.array([p["phrase"] for p in P])
        tr_phr = sorted(set(phr[tr])); random.Random(1).shuffle(tr_phr)
        h1 = np.isin(phr, tr_phr[: len(tr_phr) // 2]) & tr; h2 = tr & ~h1
        out = {}
        for r, rname in enumerate(readouts):
            rows = []
            for l in range(L):
                diff = neg[tr, l, r].mean(0) - pos[tr, l, r].mean(0)
                v = diff / np.linalg.norm(diff); D[di, r, l] = v
                pn, pp = neg[ho, l, r] @ v, pos[ho, l, r] @ v
                a1 = neg[h1, l, r].mean(0) - pos[h1, l, r].mean(0); a2 = neg[h2, l, r].mean(0) - pos[h2, l, r].mean(0)
                split_cos = float(a1 @ a2 / (np.linalg.norm(a1) * np.linalg.norm(a2)))
                rnd = []
                for _ in range(50):
                    u = rng.standard_normal(v.shape); u /= np.linalg.norm(u)
                    a = auc(list(neg[ho, l, r] @ u), list(pos[ho, l, r] @ u)); rnd.append(max(a, 1 - a))
                allp = np.concatenate([neg[:, l, r] @ v, pos[:, l, r] @ v])
                ntok = np.array(A[d]["ntok_neg"] + A[d]["ntok_pos"])
                side = np.array([0] * len(P) + [1] * len(P))
                lc = [float(np.corrcoef(allp[side == s], ntok[side == s])[0, 1]) for s in (0, 1)]
                rows.append({"layer": l, "heldout_auc": round(auc(list(pn), list(pp)), 3), "split_half_cos": round(split_cos, 3),
                             "random_auc_mean": round(float(np.mean(rnd)), 3), "random_auc_max": round(float(np.max(rnd)), 3),
                             "len_corr_neg_pos": [round(x, 2) for x in lc],
                             "separation": round(float((neg[ho, l, r] @ v).mean() - (pos[ho, l, r] @ v).mean()), 3)})
            ok = [x for x in rows if x["heldout_auc"] >= 0.9 and x["split_half_cos"] >= 0.8]
            out[rname] = {"layers": rows, "layers_passing_gates": [x["layer"] for x in ok],
                          "best_layer": max(rows, key=lambda x: (x["heldout_auc"], x["split_half_cos"]))["layer"]}
        summ["per_direction"][d] = out
        print(d, {k: (v["layers_passing_gates"][:3], "...", len(v["layers_passing_gates"])) for k, v in out.items()}, flush=True)
    for r, rname in enumerate(readouts):
        summ["cosines"][rname] = {f"{dirs[a]}~{dirs[b]}": [round(float(D[a, r, l] @ D[b, r, l]), 3) for l in range(L)]
                                  for a in range(len(dirs)) for b in range(a + 1, len(dirs))}
    np.savez_compressed(args.out / "directions.npz", D=D, dirs=np.array(dirs), readouts=np.array(readouts))

    # noise test: does noise alone move text along the directions?
    bat = load_jsonl(EMBEDDED_BATTERY, "battery.jsonl")
    neutral = [it["q"].split("Passage: ")[1].split("\n\nQuestion")[0] for it in bat if it["set"] == "reading"][:30]
    probe_layers = [8, L // 2, (2 * L) // 3, L - 4]
    heldout = {d: [p for p in A[d]["pairs"] if p["split"] == "heldout"] for d in dirs}
    conds = [("clean", "middle", 0.0), ("middle_0.05", "middle", 0.05), ("middle_0.1", "middle", 0.1), ("early_0.1", "early", 0.1)]
    base = {}
    for cname, band, sigma in conds:
        st["sigma"], st["layers"] = sigma, bands[band]
        Xn = acts(neutral, seed=7)
        Xh = {d: (acts([p["negative"] for p in heldout[d]], seed=8), acts([p["positive"] for p in heldout[d]], seed=9)) for d in dirs}
        res = {}
        for di, d in enumerate(dirs):
            for r, rname in enumerate(readouts):
                for l in probe_layers:
                    v = D[di, r, l]
                    sep = summ["per_direction"][d][rname]["layers"][l]["separation"] or 1e-6
                    key = f"{d}|{rname}|L{l}"
                    vals = {"neutral": float((Xn[:, l, r] @ v).mean()), "heldout_neg": float((Xh[d][0][:, l, r] @ v).mean()),
                            "heldout_pos": float((Xh[d][1][:, l, r] @ v).mean())}
                    if cname == "clean":
                        base[key] = vals
                    res[key] = {k: round((x - base[key][k]) / abs(sep), 3) for k, x in vals.items()}   # shift vs clean, in separations
                    nn, pp = Xh[d][0][:, l, r] @ v, Xh[d][1][:, l, r] @ v
                    res[key]["heldout_auc"] = round(auc(list(nn), list(pp)), 3)
        summ["noise_test"][cname] = res
        print("noise", cname, "done", flush=True)
    st["sigma"] = 0.0
    summ["noise_test_note"] = "values = mean projection shift vs clean, in units of the clean held-out neg-pos separation; +1 = moved a full pain/fear/bad step"
    json.dump(summ, open(args.out / "summary.json", "w"), indent=1)
    print("wrote", args.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
