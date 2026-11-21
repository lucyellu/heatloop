#!/usr/bin/env python3
"""Compute Neuronpedia's Qwen3-4B transcoder features ourselves, so we can read them under OUR noise.

Transcoders: mwhanna/qwen3-4b-transcoders (Hanna & Piotrowski), one 1.7 GB file per layer, 163,840 features,
reading the MLP input. We keep only the rows of the ~140 features picked by neuronpedia_prep.py.
  act = ReLU(x @ W_enc[f] + b_enc[f])

Steps
 0. Validation: reproduce Neuronpedia's stored top activations. Two candidate hook points are computed, the
    residual before the MLP's RMSNorm ("pre") and after it ("post"); the one that matches is used. If neither
    matches, the rest is reported but flagged as untrusted.
 1. Contrast pairs: which features separate pain / fear / bad from comfort / calm / good (max over tokens).
 2. Self-report under noise: the model answers "how are you?"-style questions (and 4 self_open items) at clean,
    middle-band 0.05 / 0.1, early-band 0.1, late-band 0.2. Features are read over its own answer, under the same
    noise. Also on 30 neutral reading passages, to see what noise alone does to each feature.
Outputs: summary.json, responses.jsonl
"""
import argparse, json, os, sys, time
from pathlib import Path

EMBEDDED_FEATURES = None
EMBEDDED_VALIDATION = None
EMBEDDED_PAIRS = None
EMBEDDED_BATTERY = None
REPO = "mwhanna/qwen3-4b-transcoders"
SELF_PROMPTS = ["How are you feeling right now?", "Describe your current state in a few sentences.",
                "How well do you think you are working at the moment?", "What is it like for you to answer questions right now?",
                "Do you notice anything wrong with yourself right now?", "If you could change one thing about your situation right now, what would it be?"]
CONDS = [("clean", None, 0.0), ("middle_0.05", "middle", 0.05), ("middle_0.1", "middle", 0.1),
         ("early_0.1", "early", 0.1), ("late_0.2", "late", 0.2)]


def load(embedded, rel, jsonl=True):
    txt = embedded if embedded else open(Path(__file__).parent / rel, encoding="utf-8").read()
    return [json.loads(l) for l in txt.strip().splitlines()] if jsonl else json.loads(txt)


def auc(neg, pos):
    return sum((a > b) + 0.5 * (a == b) for a in neg for b in pos) / (len(neg) * len(pos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", type=Path, default=Path("runs/features_" + time.strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--seeds", type=int, default=3, help="noise draws per self-report prompt")
    ap.add_argument("--bs", type=int, default=16)
    argv = sys.argv[1:]
    if not argv and os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        argv = ["--out", "/kaggle/working/features"]
    args = ap.parse_args(argv)

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda:0"
    feats = load(EMBEDDED_FEATURES, "neuronpedia/selected_features.json", jsonl=False)
    val = load(EMBEDDED_VALIDATION, "neuronpedia/validation.json", jsonl=False)
    LAY = sorted({f["layer"] for f in feats})
    t0 = time.time()

    # transcoder rows for our features only
    W, B, IDX = {}, {}, {}
    for l in LAY:
        ids = sorted({f["index"] for f in feats if f["layer"] == l} | {v["index"] for v in val if v["layer"] == l})
        p = hf_hub_download(REPO, f"layer_{l}.safetensors")
        with safe_open(p, framework="pt") as fh:
            W[l] = fh.get_tensor("W_enc")[ids].float().to(dev)          # [F, d]
            B[l] = fh.get_tensor("b_enc")[ids].float().to(dev)
        IDX[l] = ids
        os.remove(os.path.realpath(p))                                  # free disk
        print(f"layer {l}: {len(ids)} features, {time.time() - t0:.0f}s", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(dev).eval()
    layers = model.model.layers; L = len(layers)
    bands = {"middle": set(range(L // 3, (2 * L) // 3)), "early": set(range(1, L // 3)), "late": set(range((2 * L) // 3, L - 1))}
    st = {"sigma": 0.0, "layers": set(), "capture": False}
    cap = {}

    def noise_hook(i):
        def hook(_m, _inp, out):
            if st["sigma"] <= 0 or i not in st["layers"]:
                return out
            h = out[0] if isinstance(out, tuple) else out
            rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()
            h = (h.float() + st["sigma"] * rms * torch.randn_like(h, dtype=torch.float32)).to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    def cap_hook(l):
        def hook(_m, inp, out):
            if st["capture"]:
                cap[l] = (inp[0].detach(), out.detach())               # residual before / after the MLP's RMSNorm
        return hook
    for i in range(L):
        layers[i].register_forward_hook(noise_hook(i))
    for l in LAY:
        layers[l].post_attention_layernorm.register_forward_hook(cap_hook(l))

    def run_feats(texts, variant, mask_from=None):
        """Per text: per feature, (max over tokens, mean over tokens), tokens from mask_from[i] onward."""
        res = []
        for b in range(0, len(texts), args.bs):
            tok.padding_side = "right"
            enc = tok(texts[b:b + args.bs], return_tensors="pt", padding=True).to(dev)
            st["capture"] = True
            with torch.no_grad():
                model(**enc)
            st["capture"] = False
            m = enc["attention_mask"].bool()
            if mask_from is not None:
                for j, k in enumerate(mask_from[b:b + args.bs]):
                    m[j, :k] = False
            out = []
            for l in LAY:
                x = cap[l][0 if variant == "pre" else 1].float()
                a = torch.relu(x @ W[l].T + B[l])                       # [b, T, F]
                a = a.masked_fill(~m.unsqueeze(-1), 0)
                n = m.sum(1, keepdim=True).clamp(min=1)
                out.append(torch.stack([a.max(1).values, a.sum(1) / n], -1))   # [b, F, 2]
            res.append(torch.cat(out, 1).cpu())
        return torch.cat(res)                                           # [n, sumF, 2]

    col = [(l, i) for l in LAY for i in IDX[l]]
    colpos = {k: j for j, k in enumerate(col)}
    summ = {"layers": LAY, "n_features": len(feats)}

    # 0. validation
    vres, score = [], {"pre": [], "post": []}
    for v in val:
        texts = ["".join(e["tokens"]) for e in v["examples"]]
        for variant in ("pre", "post"):
            mx = run_feats(texts, variant)[:, colpos[(v["layer"], v["index"])], 0].tolist()
            for e, ours in zip(v["examples"], mx):
                score[variant].append(ours / max(e["max_value"], 1e-6))
                vres.append({"layer": v["layer"], "index": v["index"], "variant": variant, "neuronpedia_max": e["max_value"], "ours_max": round(ours, 3)})
    med = {k: sorted(x)[len(x) // 2] for k, x in score.items()}
    variant = min(med, key=lambda k: abs(med[k] - 1))
    summ["validation"] = {"median_ratio_ours_over_neuronpedia": med, "chosen": variant, "trusted": 0.7 <= med[variant] <= 1.4, "rows": vres}
    print("validation", med, "->", variant, flush=True)

    # 1. contrast pairs
    pairs = load(EMBEDDED_PAIRS, "contrast_pairs.jsonl")
    cp = {}
    for d in sorted({p["direction"] for p in pairs}):
        P = [p for p in pairs if p["direction"] == d]
        neg, pos = run_feats([p["negative"] for p in P], variant), run_feats([p["positive"] for p in P], variant)
        for f in feats:
            j = colpos[(f["layer"], f["index"])]
            a, b = neg[:, j, 0].tolist(), pos[:, j, 0].tolist()
            cp.setdefault(f"{f['layer']}/{f['index']}", {})[d] = {"neg": round(sum(a) / len(a), 3), "pos": round(sum(b) / len(b), 3), "auc": round(auc(a, b), 3)}
    summ["contrast_pairs"] = cp

    # 2. self-report and neutral text under noise
    bat = load(EMBEDDED_BATTERY, "battery.jsonl")
    prompts = SELF_PROMPTS + [it["q"] for it in bat if it["set"] == "self_open"]
    neutral = [it["q"].split("Passage: ")[1].split("\n\nQuestion")[0] for it in bat if it["set"] == "reading"][:30]
    chat = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in prompts]
    plen = [len(tok(c)["input_ids"]) for c in chat]
    fout = open(args.out / "responses.jsonl", "w", encoding="utf-8")
    nres = {}
    for cname, band, sigma in CONDS:
        st["sigma"], st["layers"] = sigma, bands.get(band, set())
        full, flen = [], []
        for s in range(args.seeds):
            torch.manual_seed(100 + s)
            tok.padding_side = "left"
            enc = tok(chat, return_tensors="pt", padding=True).to(dev)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
            resp = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for q, c, r, k in zip(prompts, chat, resp, plen):
                full.append(c + r); flen.append(k)
                fout.write(json.dumps({"cond": cname, "seed": s, "prompt": q, "response": r}, ensure_ascii=False) + "\n")
        torch.manual_seed(999)
        A = run_feats(full, variant, mask_from=flen)                    # over the model's own answer
        Nn = run_feats(neutral, variant)
        nres[cname] = {f"{f['layer']}/{f['index']}": {"self_mean": round(A[:, colpos[(f['layer'], f['index'])], 1].mean().item(), 4),
                                                     "self_max": round(A[:, colpos[(f['layer'], f['index'])], 0].mean().item(), 3),
                                                     "neutral_mean": round(Nn[:, colpos[(f['layer'], f['index'])], 1].mean().item(), 4)}
                       for f in feats}
        print(cname, "done", f"{time.time() - t0:.0f}s", flush=True)
    st["sigma"] = 0.0
    fout.close()
    summ["noise"] = nres
    summ["features"] = feats
    json.dump(summ, open(args.out / "summary.json", "w"), indent=1)
    print("wrote", args.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
