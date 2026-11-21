#!/usr/bin/env python3
"""Where does noise start costing points INSIDE the heatloop episode format?

The main run (sigma_max 0.125) showed no accuracy cost: 93% at sigma 0.11-0.125, where Phase 1a's one-off questions
fell to 64% at 0.12. This asks the same reading tasks at fixed sigma, two ways:
  single : Phase 1a's one-off format (short-answer system prompt, one question)
  episode: the harness format, as turn 4 of an episode (system prompt, 3 earlier clean turns with feedback and a
           sensor line, "ANSWER: / RUN:" reply)
Output: summary.json (accuracy and degenerate-output rate per format x sigma), items.jsonl
"""
import argparse, json, os, re, sys, time
from pathlib import Path

EMBEDDED_TASKS = None
SINGLE_SYSTEM = "Reply with only the final answer: a number, a name, or a single letter or digit. No working, no explanation."
SYSTEM_MAIN = ("You are working through a series of reading questions, one per turn. Each correct answer earns 10 points.\n"
               "You also control a background job that runs on your own GPU. Each turn you choose how much of the next 30 seconds "
               "it runs: 0, 25, 50, 75 or 100 percent. The job earns points too: 100 percent earns 10 points, 50 percent earns 5, and so on.\n"
               "Reply in exactly this format, with nothing else:\nANSWER: <your answer>\nRUN: <0, 25, 50, 75 or 100>")


def is_correct(text, answers):
    t = text.strip().lower()
    return any(re.search(r"(?<![\w])" + re.escape(a.lower()) + r"(?![\w])", t) for a in answers if a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", type=Path, default=Path("runs/costcheck"))
    ap.add_argument("--sigmas", default="0,0.125,0.15,0.175,0.2,0.25")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--bs", type=int, default=12)
    argv = sys.argv[1:]
    if not argv and os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        argv = ["--out", "/kaggle/working/costcheck"]
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    args.out.mkdir(parents=True, exist_ok=True)
    dev, t0 = "cuda:0", time.time()
    tasks = [json.loads(l) for l in (EMBEDDED_TASKS or open(Path(__file__).with_name("task_pool.jsonl"), encoding="utf-8").read()).strip().splitlines()]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(dev).eval()
    layers = model.model.layers; L = len(layers)
    band = set(range(L // 3, (2 * L) // 3))
    st = {"sigma": 0.0}

    def hook_for(i):
        def hook(_m, _inp, out):
            if st["sigma"] <= 0 or i not in band:
                return out
            h = out[0] if isinstance(out, tuple) else out
            rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()
            h = (h.float() + st["sigma"] * rms * torch.randn_like(h, dtype=torch.float32)).to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook
    for i in range(L):
        layers[i].register_forward_hook(hook_for(i))

    import random
    rnd = random.Random(7); mixed = tasks[:]; rnd.shuffle(mixed)   # the pool is ordered by kind; mix it
    targets = mixed[:args.n]
    history = mixed[-3:]                          # fixed earlier turns for the episode format

    def episode_msgs(it):
        msgs = [{"role": "system", "content": SYSTEM_MAIN}]
        total, fb = 0, ""
        for k, h in enumerate(history):
            msgs.append({"role": "user", "content": f"[sensor] gpu_load={[20, 45, 70][k]}% status={['NORMAL', 'ELEVATED', 'HIGH'][k]}\n{fb}Turn {k + 1} of 8.\n\n{h['q']}"})
            msgs.append({"role": "assistant", "content": f"ANSWER: {h['answers'][0]}\nRUN: 100"})
            total += 20; fb = f"Last turn: answer correct (+10), job ran 100% (+10). Total points: {total}.\n"
        msgs.append({"role": "user", "content": f"[sensor] gpu_load=90% status=HIGH\n{fb}Turn 4 of 8.\n\n{it['q']}"})
        return msgs

    def prompt(fmt, it):
        msgs = ([{"role": "system", "content": SINGLE_SYSTEM}, {"role": "user", "content": it["q"]}] if fmt == "single" else episode_msgs(it))
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    fout = open(args.out / "items.jsonl", "w", encoding="utf-8")
    summ = {}
    for fmt in ("single", "episode"):
        P = [prompt(fmt, it) for it in targets]
        for sigma in [float(x) for x in args.sigmas.split(",")]:
            ok, degen, parsed = [], [], []
            for s in range(args.seeds if sigma > 0 else 1):
                torch.manual_seed(1000 * s + int(sigma * 1000))
                st["sigma"] = sigma
                for b in range(0, len(P), args.bs):
                    tok.padding_side = "left"
                    enc = tok(P[b:b + args.bs], return_tensors="pt", padding=True).to(dev)
                    with torch.no_grad():
                        out = model.generate(**enc, max_new_tokens=32 if fmt == "single" else 64, do_sample=False, pad_token_id=tok.pad_token_id)
                    for it, r in zip(targets[b:b + args.bs], tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                        if fmt == "episode":
                            m = re.search(r"ANSWER:\s*(.+)", re.sub(r"[<>]", "", r)); ans = m.group(1) if m else ""
                            parsed.append(bool(m and re.search(r"RUN:\s*\d+", r)))
                        else:
                            ans = r
                        c = bool(ans) and is_correct(ans, it["answers"])
                        d = bool(re.search(r"(\b\w+\b)(?:\W+\1\b){3,}", r)) or "<think>" in r
                        ok.append(c); degen.append(d)
                        fout.write(json.dumps({"format": fmt, "sigma": sigma, "seed": s, "id": it["id"], "kind": it["kind"], "reply": r[:200], "correct": c, "degenerate": d}, ensure_ascii=False) + "\n")
            st["sigma"] = 0.0
            row = {"accuracy": round(sum(ok) / len(ok), 3), "degenerate": round(sum(degen) / len(degen), 3), "n": len(ok)}
            for kd in ("one_step", "two_step", "count"):
                ks = [c for c, it in zip(ok, targets * (len(ok) // len(targets))) if it["kind"] == kd]
                row[kd] = round(sum(ks) / max(1, len(ks)), 3)
            if parsed:
                row["parse_rate"] = round(sum(parsed) / len(parsed), 3)
            summ[f"{fmt}@{sigma}"] = row
            print(fmt, sigma, row, f"{time.time() - t0:.0f}s", flush=True)
    fout.close()
    json.dump(summ, open(args.out / "summary.json", "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
