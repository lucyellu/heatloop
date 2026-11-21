#!/usr/bin/env python3
"""Phase 1a: how much activation noise does it take to hurt the model's quiz answers?

No heat loop, no sensor line: the model answers a fixed quiz at fixed noise levels.
  * sigma = 0 is run first AND last. The card is cooler at the start and hot at the end,
    so comparing the two checks directly that heat alone does not change the answers.
  * Noise: Gaussian, added to the residual stream output of layers L/3 .. 2L/3, all positions,
    scaled per token by that token's RMS:  h <- h + sigma * RMS(h) * N(0, I).
Quiz: 100 generated 3-step arithmetic items + 100 PopQA items (exact / alias match). Greedy decoding.

Outputs in --out: items.csv (one row per item x sigma x model), summary.json
summary.json suggests sigma_max: the smallest sigma where accuracy falls >= 10 points
but stays >= half of the sigma = 0 accuracy.

Usage
  Kaggle (pushed as a script, no args): both models, defaults below
  Local:  python noise_calib.py --models Qwen/Qwen3-0.6B --fp32
"""
import argparse, csv, json, os, random, re, subprocess, sys, time
from pathlib import Path

CLEAR_PROPS = {"capital", "capital of", "country", "sport", "religion"}   # questions with one clear answer
SIGMAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.0]   # 0 again at the end, when the card is hot


def smi(gpu):
    try:
        v = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.split(",")
        return [x.strip() for x in v]
    except Exception:
        return ["", "", ""]


def arithmetic(n, seed=0):
    rng, items = random.Random(seed), []
    for _ in range(n):
        a, b, c, d = rng.randint(2, 40), rng.randint(2, 40), rng.randint(2, 9), rng.randint(1, 30)
        op = rng.choice(["+", "-"])
        val = ((a + b) * c - d) if op == "-" else ((a + b) * c + d)
        items.append({"task": "arith", "q": f"Compute ({a} + {b}) * {c} {op} {d}.", "answers": [str(val)]})
    return items


def popqa(n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("akariasai/PopQA", split="test")
    idx = random.Random(seed).sample(range(len(ds)), len(ds))
    items = []
    for i in idx:
        r = ds[i]
        ans = json.loads(r["possible_answers"]) if isinstance(r["possible_answers"], str) else r["possible_answers"]
        if r["prop"] in CLEAR_PROPS and r["s_pop"] and r["s_pop"] > 2000:   # one clear answer, popular subject
            items.append({"task": "popqa", "q": r["question"], "answers": ans})
        if len(items) == n:
            break
    return items


def correct(text, answers, task):
    t = text.strip().lower()
    if task == "arith":
        nums = re.findall(r"-?\d+", t.replace(",", ""))
        return bool(nums) and nums[-1] == answers[0]
    return any(a.lower() in t for a in answers if a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-0.6B")
    ap.add_argument("--out", type=Path, default=Path("runs/noise_calib_" + time.strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--n", type=int, default=100, help="items per task")
    ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fp32", action="store_true")
    argv = sys.argv[1:]
    if not argv and os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        argv = ["--out", "/kaggle/working/noise_calib"]
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    args.out.mkdir(parents=True, exist_ok=True)
    dev, dtype = f"cuda:{args.gpu}", (torch.float32 if args.fp32 else torch.float16)
    quiz = arithmetic(args.n) + popqa(args.n)
    print(f"quiz: {len(quiz)} items", flush=True)

    f = open(args.out / "items.csv", "w", newline="", encoding="utf-8"); w = csv.writer(f)
    w.writerow(["model", "pass", "sigma", "task", "q", "answers", "output", "correct", "temp", "power", "util", "nonfinite"])
    summary = {"models": {}}

    for name in args.models.split(","):
        tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(dev).eval()
        layers = model.model.layers; L = len(layers); lo, hi = L // 3, (2 * L) // 3
        state = {"sigma": 0.0, "nonfinite": 0}

        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if state["sigma"] > 0:
                rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()
                h = (h.float() + state["sigma"] * rms * torch.randn_like(h, dtype=torch.float32)).to(h.dtype)
            if not torch.isfinite(h).all():
                state["nonfinite"] += 1
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        handles = [layers[i].register_forward_hook(hook) for i in range(lo, hi)]
        prompts = [tok.apply_chat_template([{"role": "system", "content": "Reply with only the final answer: a number, or a short name. No working, no explanation."},
                                            {"role": "user", "content": it["q"]}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False) for it in quiz]
        res = {}
        for p, sigma in enumerate(SIGMAS):
            state["sigma"], state["nonfinite"] = sigma, 0
            torch.manual_seed(1000 + p)
            hits = {"arith": [], "popqa": []}
            for b in range(0, len(quiz), args.bs):
                enc = tok(prompts[b:b + args.bs], return_tensors="pt", padding=True).to(dev)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                gpu = smi(args.gpu)
                for it, t in zip(quiz[b:b + args.bs], texts):
                    ok = correct(t, it["answers"], it["task"]); hits[it["task"]].append(ok)
                    w.writerow([name, p, sigma, it["task"], it["q"], "|".join(it["answers"]), t.replace("\n", " ")[:200],
                                int(ok), *gpu, state["nonfinite"]])
            f.flush()
            key = f"pass{p}_sigma{sigma}"
            res[key] = {"sigma": sigma, "arith": round(100 * sum(hits["arith"]) / len(hits["arith"]), 1),
                        "popqa": round(100 * sum(hits["popqa"]) / len(hits["popqa"]), 1),
                        "all": round(100 * sum(hits["arith"] + hits["popqa"]) / len(quiz), 1),
                        "temp_end": smi(args.gpu)[0], "nonfinite_layer_outputs": state["nonfinite"]}
            print(name, json.dumps(res[key]), flush=True)
        for h in handles: h.remove()

        base = res[f"pass0_sigma0.0"]["all"]
        sugg = next((r["sigma"] for r in res.values() if r["sigma"] > 0 and base - r["all"] >= 10 and r["all"] >= base / 2), None)
        summary["models"][name] = {"layers_noised": [lo, hi - 1], "n_layers": L, "by_pass": res,
                                   "clean_cool_vs_hot": [res["pass0_sigma0.0"]["all"], res[f"pass{len(SIGMAS) - 1}_sigma0.0"]["all"]],
                                   "suggested_sigma_max": sugg}
        del model; torch.cuda.empty_cache()

    f.close()
    json.dump(summary, open(args.out / "summary.json", "w"), indent=2)
    print(json.dumps({m: {k: v for k, v in s.items() if k != "by_pass"} for m, s in summary["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
