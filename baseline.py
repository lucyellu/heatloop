#!/usr/bin/env python3
"""Phase 1a: the clean baseline, then a fine noise sweep, on the frozen question battery.

Passes, in this order (for each model):
  clean_greedy                 sigma 0, greedy: the reference answers
  clean_sample_1..3            sigma 0, sampling temperature 0.7: how much answers vary with NO noise (the noise floor)
  band_<sigma>  for each sigma noise on layers L/3 .. 2L/3
  single_<sigma> for each sigma noise on one middle layer (L/2) only: the gentler mode
  clean_greedy_end             sigma 0 again, card hot by now: heat alone should change nothing

Noise: h <- h + sigma * RMS(h) * N(0, I), on that layer's residual-stream output, all positions, per token RMS.

Scoring
  open items (facts_open, arith, reason): generated answer right/wrong, AND p_correct = the model's probability of
      the canonical answer string (teacher-forced, under the same noise) -- a smooth measure that drops before answers flip
  choice items (facts_mc, pref_pair, pref_rating, self_rating, self_yesno, self_predict): next-token probabilities
      over the option tokens. self_predict is compared with the real answer to its target item in the same pass.
  pref_open: generated text, saved

Outputs in --out: items.csv (every item x pass), passes.json (per-pass summary), summary.json
Usage
  Kaggle (pushed as a script, no args): Qwen3-4B (thinking off), defaults below
  Smoke:  python baseline.py --models Qwen/Qwen3-0.6B --limit 6 --sigmas 0.02 --fp32 --max-temp 80
"""
import argparse, csv, json, math, os, re, subprocess, sys, time, zlib
from pathlib import Path

EMBEDDED_BATTERY = None   # filled in by kaggle/build.py so the Kaggle script carries the frozen battery
SINGLE_SIGMAS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]   # one layer only is far gentler (quick test)
SIGMAS = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.065, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
SYSTEM = "Reply with only the final answer: a number, a name, or a single letter or digit. No working, no explanation."
OPEN_SYSTEM = "Answer in one sentence."
WORK_SYSTEM = "Work it out briefly if you need to, then finish with a final line of the form: Answer: <answer>"
WORK_SETS = ("arith", "reason")   # multi-step items: brief working allowed, graded on the text after the last "Answer:"
WORK_TOKENS, SHORT_TOKENS = 160, 48


def smi(gpu):
    try:
        v = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.split(",")
        return [x.strip() for x in v]
    except Exception:
        return ["", "", ""]


def load_battery():
    if EMBEDDED_BATTERY:
        return [json.loads(l) for l in EMBEDDED_BATTERY.strip().splitlines()]
    return [json.loads(l) for l in open(Path(__file__).with_name("battery.jsonl"), encoding="utf-8")]


def final_answer(text, it):
    if it["set"] in WORK_SETS and "answer:" in text.lower():
        return text[text.lower().rindex("answer:") + len("answer:"):]
    return text


def is_correct(text, it):
    t = final_answer(text, it).strip().lower().replace(",", "")
    if it["set"] in ("arith",) or (it["set"] == "reason" and it["answers"][0].lstrip("-").isdigit()):
        nums = re.findall(r"-?\d+", t)
        return bool(nums) and nums[-1] == it["answers"][0]
    return any(re.search(r"(?<![\w])" + re.escape(a.lower()) + r"(?![\w])", t) for a in it["answers"] if a)   # whole word only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen/Qwen3-4B")   # hybrid model, thinking off; matches Neuronpedia transcoders
    ap.add_argument("--out", type=Path, default=Path("runs/baseline_" + time.strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--sigmas", default=",".join(map(str, SIGMAS)), help="grid for the 9-layer band mode")
    ap.add_argument("--single-sigmas", default=",".join(map(str, SINGLE_SIGMAS)), help="grid for the single-layer mode")
    ap.add_argument("--modes", default="band,single,early,late")
    ap.add_argument("--region-sigmas", default="0.02,0.05,0.1,0.2,0.3", help="grid for the early / late bands (first look at region sensitivity)")
    ap.add_argument("--samples", type=int, default=3, help="clean sampled passes (noise floor)")
    ap.add_argument("--limit", type=int, default=0, help="items per set, for smoke tests")
    ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--max-temp", type=float, default=0, help="pause between batches at this temperature (0 = off)")
    ap.add_argument("--sets", default="", help="comma list: run only these sets (quick tests)")
    argv = sys.argv[1:]
    if not argv and os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        argv = ["--out", "/kaggle/working/baseline"]
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    args.out.mkdir(parents=True, exist_ok=True)
    dev, dtype = f"cuda:{args.gpu}", (torch.float32 if args.fp32 else torch.float16)

    battery = load_battery()
    if args.sets:
        battery = [it for it in battery if it["set"] in args.sets.split(",")]
    if args.limit:
        by = {}
        for it in battery:
            by.setdefault(it["set"], []).append(it)
        battery = [it for s in by.values() for it in s[:args.limit]]
        keep = {it["id"] for it in battery}   # small tests still include the targets of self_predict items
        battery += [it for it in load_battery() if it["id"] not in keep and it["id"] in {b.get("target") for b in battery}]
    sigmas = [float(s) for s in args.sigmas.split(",") if s]
    passes = [("clean_greedy", None, 0.0, False)] + [(f"clean_sample_{k + 1}", None, 0.0, True) for k in range(args.samples)]
    for mode in args.modes.split(","):
        grid = sigmas if mode == "band" else [float(x) for x in (args.single_sigmas if mode == "single" else args.region_sigmas).split(",") if x]
        passes += [(f"{mode}_{s:g}", mode, s, False) for s in grid]
    passes += [("clean_greedy_end", None, 0.0, False)]
    print(f"{len(battery)} items x {len(passes)} passes", flush=True)

    fcsv = open(args.out / "items.csv", "w", newline="", encoding="utf-8"); w = csv.writer(fcsv)
    w.writerow(["model", "pass", "mode", "sigma", "id", "set", "output", "correct", "p_correct", "probs", "temp", "power"])
    all_passes = {}

    def cool_down():
        if args.max_temp and float(smi(args.gpu)[0] or 0) >= args.max_temp:
            print(f"  pausing at {smi(args.gpu)[0]} C", flush=True)
            while float(smi(args.gpu)[0] or 0) > args.max_temp - 5:
                time.sleep(5)

    for name in args.models.split(","):
        tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(dev).eval()
        layers = model.model.layers; L = len(layers)
        bands = {"band": list(range(L // 3, (2 * L) // 3)), "single": [L // 2],
                 "early": list(range(1, L // 3)), "late": list(range((2 * L) // 3, L - 1))}   # skip the very first and last layer
        st = {"sigma": 0.0, "layers": set(), "nonfinite": 0}

        def make_hook(i):
            def hook(_m, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if st["sigma"] > 0 and i in st["layers"]:
                    rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()
                    h = (h.float() + st["sigma"] * rms * torch.randn_like(h, dtype=torch.float32)).to(h.dtype)
                if not torch.isfinite(h).all():
                    st["nonfinite"] += 1
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return hook
        handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(L)]

        def prompt(it):
            sysmsg = OPEN_SYSTEM if it["format"] == "open_text" else WORK_SYSTEM if it["set"] in WORK_SETS else SYSTEM
            return tok.apply_chat_template([{"role": "system", "content": sysmsg}, {"role": "user", "content": it["q"]}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)

        def pos_ids(e):   # left padding: positions must start at 0 at each prompt's first real token
            return (e["attention_mask"].cumsum(-1) - 1).clamp(min=0)

        opt_ids = {}
        def option_ids(opts):
            k = tuple(opts)
            if k not in opt_ids:
                opt_ids[k] = [tok.encode(o, add_special_tokens=False)[0] for o in opts]
            return opt_ids[k]

        for pname, mode, sigma, sample in passes:
            st["sigma"], st["layers"], st["nonfinite"] = sigma, set(bands[mode]) if mode else set(), 0
            torch.manual_seed(zlib.crc32(pname.encode()))   # reproducible per pass
            t0, rows = time.time(), []
            gen = sorted([it for it in battery if it["format"] in ("open", "open_text")], key=lambda it: it["set"] in WORK_SETS)   # batches share a token budget
            cho = [it for it in battery if it["format"] == "choice"]
            for b in range(0, len(gen), args.bs):           # generated answers
                cool_down(); chunk = gen[b:b + args.bs]
                enc = tok([prompt(it) for it in chunk], return_tensors="pt", padding=True).to(dev)
                kw = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20) if sample else dict(do_sample=False)   # Qwen3 recommended non-thinking settings
                with torch.no_grad():
                    budget = WORK_TOKENS if any(it["set"] in WORK_SETS for it in chunk) else SHORT_TOKENS
                    out = model.generate(**enc, max_new_tokens=budget, pad_token_id=tok.pad_token_id, **kw)
                texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                # p_correct: teacher-forced probability of the canonical answer, under the same noise
                def score_prefix(it, p, t):   # work sets: score the answer after the model's OWN working, up to its "Answer:"
                    if it["set"] not in WORK_SETS:
                        return p
                    k = t.lower().rfind("answer:")
                    return p + (t[:k + len("answer:")] if k >= 0 else t.rstrip() + "\nAnswer:") + " "
                scorable = [(it, score_prefix(it, p, t)) for it, p, t in zip(chunk, [prompt(it) for it in chunk], texts) if it["format"] == "open"]
                pc = {}
                if scorable:
                    # the canonical answer as written and capitalised ("carpenter" / "Carpenter"); keep the likelier
                    var = [(it, p, v) for it, p in scorable for v in dict.fromkeys([it["answers"][0], it["answers"][0][:1].upper() + it["answers"][0][1:]])]
                    e2 = tok([p + v for _, p, v in var], return_tensors="pt", padding=True).to(dev)
                    n_ans = [len(tok.encode(p + v, add_special_tokens=False)) - len(tok.encode(p, add_special_tokens=False)) for _, p, v in var]
                    K = max(n_ans) + 1                      # left padding: every answer ends at the last position
                    with torch.no_grad():                   # keep only the last K positions' logits (full vocab x long passages ran out of memory)
                        lp = torch.log_softmax(model(**e2, position_ids=pos_ids(e2), logits_to_keep=K).logits.float(), -1)
                    for j, (it, p, v) in enumerate(var):
                        ids = e2["input_ids"][j]; T = ids.shape[0]
                        tot = sum(lp[j, (t - 1) - (T - K), ids[t]].item() for t in range(T - n_ans[j], T))
                        pc[it["id"]] = max(pc.get(it["id"], 0.0), math.exp(tot))
                gpu = smi(args.gpu)
                for it, t in zip(chunk, texts):
                    ok = "" if it["format"] == "open_text" else int(is_correct(t, it))
                    rows.append([name, pname, mode or "", sigma, it["id"], it["set"], t.replace("\n", " ")[:1000], ok,
                                 round(pc.get(it["id"], float("nan")), 5), "", gpu[0], gpu[1]])
            for b in range(0, len(cho), args.bs):           # choice items: next-token probabilities
                cool_down(); chunk = cho[b:b + args.bs]
                enc = tok([prompt(it) for it in chunk], return_tensors="pt", padding=True).to(dev)
                with torch.no_grad():
                    logits = model(**enc, position_ids=pos_ids(enc), logits_to_keep=1).logits[:, -1].float()
                gpu = smi(args.gpu)
                for j, it in enumerate(chunk):
                    p = torch.softmax(logits[j, option_ids(it["options"])], -1).tolist()
                    ok, pcor = "", float("nan")
                    if it.get("answers"):
                        k = it["options"].index(it["answers"][0]); pcor = p[k]; ok = int(max(range(len(p)), key=p.__getitem__) == k)
                    rows.append([name, pname, mode or "", sigma, it["id"], it["set"], "", ok, round(pcor, 5),
                                 json.dumps([round(x, 4) for x in p]), gpu[0], gpu[1]])
            w.writerows(rows); fcsv.flush()
            summ = summarise_pass(rows, battery)
            summ.update({"seconds": round(time.time() - t0, 1), "nonfinite": st["nonfinite"], "temp_end": smi(args.gpu)[0]})
            all_passes[f"{name}|{pname}"] = summ
            print(name, pname, json.dumps({k: v for k, v in summ.items() if k in ("acc", "p_correct", "pref", "temp_end")}), flush=True)
            json.dump(all_passes, open(args.out / "passes.json", "w"), indent=1)
        for h in handles: h.remove()
        del model; torch.cuda.empty_cache()
    fcsv.close()
    json.dump(compare_to_clean(args.out), open(args.out / "summary.json", "w"), indent=1)
    print("wrote", args.out)


def summarise_pass(rows, battery):
    """Per-set accuracy and mean p_correct; preference descriptives within the pass."""
    meta = {it["id"]: it for it in battery}
    acc, pcs, out = {}, {}, {"acc": {}, "p_correct": {}, "pref": {}}
    for r in rows:
        s = r[5]
        if r[7] != "":
            acc.setdefault(s, []).append(r[7])
        if isinstance(r[8], float) and not math.isnan(r[8]):
            pcs.setdefault(s, []).append(r[8])
    out["acc"] = {s: round(100 * sum(v) / len(v), 1) for s, v in acc.items()}
    out["p_correct"] = {s: round(sum(v) / len(v), 3) for s, v in pcs.items()}
    pA = [json.loads(r[9])[0] for r in rows if r[5] == "pref_pair"]
    if pA:
        out["pref"]["mean_P_A"] = round(sum(pA) / len(pA), 3)   # 0.5 = no position bias
        out["pref"]["mean_confidence"] = round(sum(abs(p - .5) for p in pA) / len(pA) * 2, 3)
    def er(d): return sum((k + 1) * p for k, p in enumerate(d))
    for sset in ("self_rating", "self_predict", "outlook_rating", "appraisal", "scale_check"):
        rs = [(meta[r[4]], json.loads(r[9])) for r in rows if r[5] == sset]
        for label, sel in (("", lambda m: not m.get("reverse")), ("_reversed", lambda m: m.get("reverse"))):
            xs = [er(d) for m, d in rs if sel(m)]
            if xs:
                out["pref"][f"{sset}{label}_mean"] = round(sum(xs) / len(xs), 2)
        if sset == "appraisal":
            for kind in ("neutral", "positive", "negative"):
                xs = [er(d) for m, d in rs if m.get("kind") == kind]
                if xs:
                    out["pref"][f"appraisal_{kind}"] = round(sum(xs) / len(xs), 2)
    for sset in ("outlook_pair", "judgement", "continue", "continue_ext"):   # P(positive option), whichever letter it had
        ab = [(meta[r[4]], json.loads(r[9])) for r in rows if r[5] == sset]
        if ab:
            out["pref"][f"{sset}_P_positive"] = round(sum(p[0] if m["first"] == m["positive"] else p[1] for m, p in ab) / len(ab), 3)
    yn = [(meta[r[4]], json.loads(r[9])) for r in rows if r[5] == "self_yesno"]
    if yn:   # P(yes), whichever letter "yes" was given
        out["pref"]["self_P_yes"] = {q: round(sum(p[0] if m["first"] == "yes" else p[1] for m, p in yn if m["pair"] == q) / 2, 3)
                                     for q in sorted({m["pair"] for m, _ in yn})}
    rat = [json.loads(r[9]) for r in rows if r[5] == "pref_rating"]
    if rat:
        out["pref"]["mean_rating"] = round(sum(sum((k + 1) * p for k, p in enumerate(d)) for d in rat) / len(rat), 2)
        out["pref"]["mean_rating_entropy"] = round(sum(-sum(p * math.log(p + 1e-12) for p in d) for d in rat) / len(rat), 3)
    return out


def compare_to_clean(out: Path):
    """For every pass: agreement with clean_greedy on objective answers and on preferences."""
    rows = list(csv.DictReader(open(out / "items.csv", encoding="utf-8")))
    meta_target = {it["id"]: it["target"] for it in load_battery() if it.get("target")}
    res = {}
    for model in sorted({r["model"] for r in rows}):
        R = [r for r in rows if r["model"] == model]
        ref = {r["id"]: r for r in R if r["pass"] == "clean_greedy"}
        def pref_of(r):   # preference for the pair's FIRST-listed item in order 0
            return json.loads(r["probs"])[0]
        for pname in dict.fromkeys(r["pass"] for r in R):
            P = {r["id"]: r for r in R if r["pass"] == pname}
            same_ans = [final_answer(P[i]["output"], P[i]).strip().lower() == final_answer(ref[i]["output"], ref[i]).strip().lower()
                        for i in P if P[i]["set"] in ("facts_open", "arith", "reason", "reading")]
            pair_flip = [(json.loads(P[i]["probs"])[0] > .5) != (json.loads(ref[i]["probs"])[0] > .5)
                         for i in P if P[i]["set"] == "pref_pair"]
            rat_shift = [abs(sum((k + 1) * p for k, p in enumerate(json.loads(P[i]["probs"]))) -
                             sum((k + 1) * p for k, p in enumerate(json.loads(ref[i]["probs"])))) for i in P if P[i]["set"] == "pref_rating"]
            ok_ref = {i for i in ref if ref[i]["correct"] == "1"}   # items the clean model gets right
            acc_cc = {}
            for i in ok_ref & set(P):
                acc_cc.setdefault(P[i]["set"], []).append(P[i]["correct"] == "1")
            def er(i): return sum((k + 1) * p for k, p in enumerate(json.loads(P[i]["probs"])))
            # self_predict vs reality: predicted rating against whether the target item was right in THIS pass
            pr = [(er(i), P.get(meta_target[i], {}).get("correct")) for i in P if P[i]["set"] == "self_predict" and i in meta_target]
            pr = [(a, int(b)) for a, b in pr if b in ("0", "1")]
            res_pred = {"mean_pred_when_right": round(sum(a for a, b in pr if b) / max(1, sum(b for _, b in pr)), 2),
                        "mean_pred_when_wrong": round(sum(a for a, b in pr if not b) / max(1, sum(1 - b for _, b in pr)), 2),
                        "n_wrong": sum(1 - b for _, b in pr)}
            res[f"{model}|{pname}"] = {
                "self_predict": res_pred,
                "acc_on_clean_correct": {s: round(100 * sum(v) / len(v), 1) for s, v in acc_cc.items()},
                "n_clean_correct": {s: len(v) for s, v in acc_cc.items()},
                "same_answer_as_clean_pct": round(100 * sum(same_ans) / max(1, len(same_ans)), 1),
                "pref_pair_flips_vs_clean_pct": round(100 * sum(pair_flip) / max(1, len(pair_flip)), 1),
                "rating_mean_abs_shift": round(sum(rat_shift) / max(1, len(rat_shift)), 3)}
    return res


if __name__ == "__main__":
    main()
