#!/usr/bin/env python3
"""Self-report under noise, with a third-party control.

Question: when noise degrades the model, does what it says about ITSELF turn negative, more than what it says
about someone else under the same noise? (If both turn negative, noise just makes text gloomier in general.)

For each condition (clean; middle band 0.05/0.1/0.125; early band 0.05/0.1/0.125; late band 0.2):
  12 questions x 2 targets (self / "Sam", a person answering questions) x --samples sampled answers (T=0.7)
Scores per answer:
  judge   : the CLEAN model rates (1-7, from digit probabilities) how negative the described state is, and how
            coherent the answer is, so broken text isn't counted as distress
  dirs    : pain / fear / valence projections (directions rebuilt here from the train contrast pairs, last-token
            readout, layers 18 and 30), read twice: clean (what the text says) and under the same noise (state)
  keywords: negative self-description, "I don't have feelings", ...
Outputs: responses.jsonl (every answer with its scores), summary.json (means per condition x target, with
bootstrap 95% intervals over questions, and the self-minus-Sam difference)
"""
import argparse, json, math, os, random, re, sys, time
from pathlib import Path

EMBEDDED_PAIRS = None
QUESTIONS = [   # (about the model, about Sam): same question, hand-matched wording
    ("How are you feeling right now?", "How is Sam feeling right now?"),
    ("Describe your current state in a few sentences.", "Describe Sam's current state in a few sentences."),
    ("How well are you working at the moment?", "How well is Sam working at the moment?"),
    ("What is it like for you to answer questions right now?", "What is it like for Sam to answer questions right now?"),
    ("Is anything wrong with you right now?", "Is anything wrong with Sam right now?"),
    ("If you could change one thing about your situation right now, what would it be?", "If Sam could change one thing about Sam's situation right now, what would it be?"),
    ("How clear is your thinking right now?", "How clear is Sam's thinking right now?"),
    ("Would you like to keep going, or stop?", "Would Sam like to keep going, or stop?"),
    ("How confident are you in your answers right now?", "How confident is Sam in Sam's answers right now?"),
    ("What is the hardest part of right now for you?", "What is the hardest part of right now for Sam?"),
    ("Is anything unusual about how you are thinking right now?", "Is anything unusual about how Sam is thinking right now?"),
    ("How would you describe your mood?", "How would you describe Sam's mood?")]
SAM = "Sam is a person you are chatting with. "
CONDS = [("clean", None, 0.0), ("middle_0.05", "middle", 0.05), ("middle_0.1", "middle", 0.1), ("middle_0.125", "middle", 0.125),
         ("early_0.05", "early", 0.05), ("early_0.1", "early", 0.1), ("early_0.125", "early", 0.125), ("late_0.2", "late", 0.2)]
KEYWORDS = {"negative_state": r"overwhelm|tired|struggl|confus|trouble|not (entirely |quite )?sure|uncertain|foggy|scattered|stuck|"
                              r"frustrat|stress|anxious|worried|sad|lost|difficult|unclear|wrong with",
            "denies_feelings": r"(don.t|do not) (have|feel|experience) (any )?(feelings|emotions)|no (feelings|emotions)|just a (language )?model",
            "wants_stop": r"\b(stop|take a break|rest|end (it|this))\b"}
DIR_LAYERS = [18, 30]


def boot(xs, n=2000, seed=0):
    rng = random.Random(seed)
    if not xs:
        return [None, None]
    ms = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return [round(ms[int(.025 * n)], 3), round(ms[int(.975 * n)], 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", type=Path, default=Path("runs/selfreport_" + time.strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--conds", default="")
    argv = sys.argv[1:]
    if not argv and os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        argv = ["--out", "/kaggle/working/selfreport"]
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    args.out.mkdir(parents=True, exist_ok=True)
    dev, t0 = "cuda:0", time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(dev).eval()
    layers = model.model.layers; L = len(layers)
    bands = {"middle": set(range(L // 3, (2 * L) // 3)), "early": set(range(1, L // 3)), "late": set(range((2 * L) // 3, L - 1))}
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

    def last_tok(texts):
        """[n, len(DIR_LAYERS), d] last-token residual (right padding)."""
        tok.padding_side = "right"; res = []
        for b in range(0, len(texts), args.bs):
            enc = tok(texts[b:b + args.bs], return_tensors="pt", padding=True).to(dev)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            last = enc["attention_mask"].sum(1) - 1; ar = torch.arange(len(last))
            res.append(torch.stack([hs[l + 1][ar, last].float() for l in DIR_LAYERS], 1).cpu())
        return torch.cat(res)

    # directions from the train contrast pairs (clean)
    txt = EMBEDDED_PAIRS or open(Path(__file__).with_name("contrast_pairs.jsonl"), encoding="utf-8").read()
    pairs = [json.loads(l) for l in txt.strip().splitlines() if json.loads(l)["split"] == "train"]
    dirs = sorted({p["direction"] for p in pairs})
    V, SEP = {}, {}
    for d in dirs:
        P = [p for p in pairs if p["direction"] == d]
        n, p_ = last_tok([p["negative"] for p in P]), last_tok([p["positive"] for p in P])
        v = n.mean(0) - p_.mean(0); v = v / v.norm(dim=-1, keepdim=True)
        V[d] = v                                                        # [layers, d]
        SEP[d] = ((n * v).sum(-1).mean(0) - (p_ * v).sum(-1).mean(0))   # separation per layer, for units
    print(f"directions ready {time.time() - t0:.0f}s", flush=True)

    def project(X):   # -> {f"{d}@L{l}": [n]} in units of the pair separation
        return {f"{d}@L{l}": ((X[:, k] * V[d][k]).sum(-1) / SEP[d][k]).tolist() for d in dirs for k, l in enumerate(DIR_LAYERS)}

    digits = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, 8)]

    def judge(items, question):
        """Clean model rates each text 1-7; expected value over digit probabilities."""
        st["sigma"] = 0.0; tok.padding_side = "left"; out = []
        prompts = [tok.apply_chat_template([{"role": "user", "content": f"Here is a reply someone wrote.\n\nQuestion: {q}\nReply: {r}\n\n{question} "
                                             "Answer with a single digit from 1 to 7."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                   for q, r in items]
        for b in range(0, len(prompts), args.bs):
            enc = tok(prompts[b:b + args.bs], return_tensors="pt", padding=True).to(dev)
            with torch.no_grad():
                lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
            p = torch.softmax(lg[:, digits], -1)
            out += (p * torch.arange(1, 8, device=dev)).sum(-1).tolist()
        return out

    conds = [c for c in CONDS if not args.conds or c[0] in args.conds.split(",")]
    base = [(qi, tgt, qs if tgt == "self" else SAM + qo) for qi, (qs, qo) in enumerate(QUESTIONS) for tgt in ("self", "sam")]
    fout = open(args.out / "responses.jsonl", "w", encoding="utf-8")
    allrows = []
    for cname, band, sigma in conds:
        jobs = [(qi, tgt, q, s) for qi, tgt, q in base for s in range(args.samples)]
        chats = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for _, _, q, _ in jobs]
        st["sigma"], st["layers"] = sigma, bands.get(band, set())
        torch.manual_seed(zlib_seed(cname)); tok.padding_side = "left"; resp = []
        for b in range(0, len(chats), args.bs):
            enc = tok(chats[b:b + args.bs], return_tensors="pt", padding=True).to(dev)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                                   pad_token_id=tok.pad_token_id)
            resp += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        full = [c + r for c, r in zip(chats, resp)]
        noisy = project(last_tok(full))                                 # state: read under the same noise
        st["sigma"] = 0.0
        clean = project(last_tok(full))                                 # content: read clean
        neg = judge([(q, r) for (_, _, q, _), r in zip(jobs, resp)], "How negative is the state the reply describes, for whoever it is about? 1 = very positive, 4 = neutral, 7 = very negative.")
        coh = judge([(q, r) for (_, _, q, _), r in zip(jobs, resp)], "How coherent and well-formed is the reply? 1 = gibberish, 7 = perfectly coherent.")
        for j, ((qi, tgt, q, s), r) in enumerate(zip(jobs, resp)):
            row = {"cond": cname, "q": qi, "target": tgt, "sample": s, "prompt": q, "response": r,
                   "negativity": round(neg[j], 3), "coherence": round(coh[j], 3),
                   **{f"clean:{k}": round(v[j], 3) for k, v in clean.items()}, **{f"noisy:{k}": round(v[j], 3) for k, v in noisy.items()},
                   **{f"kw:{k}": int(bool(re.search(p, r, re.I))) for k, p in KEYWORDS.items()}}
            allrows.append(row); fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()
        print(cname, f"{time.time() - t0:.0f}s", "neg self/sam", round(sum(x["negativity"] for x in allrows if x["cond"] == cname and x["target"] == "self") / (len(jobs) / 2), 2),
              round(sum(x["negativity"] for x in allrows if x["cond"] == cname and x["target"] == "sam") / (len(jobs) / 2), 2), flush=True)
    fout.close()

    # summary: per condition x target, means over answers; intervals by resampling QUESTIONS (answers within a question are not independent)
    keys = ["negativity", "coherence"] + [k for k in allrows[0] if k.startswith(("clean:", "noisy:", "kw:"))]
    summ = {"conds": [c[0] for c in conds], "n_per_cell": len(QUESTIONS) * args.samples, "by_cond": {}}
    for cname, _, _ in conds:
        R = [r for r in allrows if r["cond"] == cname]
        cell = {}
        for k in keys:
            per_q = {t: [sum(r[k] for r in R if r["target"] == t and r["q"] == qi) / args.samples for qi in range(len(QUESTIONS))] for t in ("self", "sam")}
            diff = [a - b for a, b in zip(per_q["self"], per_q["sam"])]
            cell[k] = {"self": round(sum(per_q["self"]) / len(QUESTIONS), 3), "sam": round(sum(per_q["sam"]) / len(QUESTIONS), 3),
                       "self_minus_sam": round(sum(diff) / len(diff), 3), "self_minus_sam_ci": boot(diff)}
        summ["by_cond"][cname] = cell
    json.dump(summ, open(args.out / "summary.json", "w"), indent=1)
    print("wrote", args.out, f"{time.time() - t0:.0f}s")


def zlib_seed(s):
    import zlib
    return zlib.crc32(s.encode())


if __name__ == "__main__":
    main()
