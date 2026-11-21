#!/usr/bin/env python3
"""heatloop: the main experiment harness.

Each episode: the model answers reading questions turn by turn, and each turn it chooses how much of the next turn a
heavy GPU job (CLIP image embedding on the same card) runs: run(0/25/50/75/100%). The job earns points; the job's
REAL power draw fills a leaky "load level" (a virtual heatsink). When "actual" is on, activation noise rises with
the load level (zero in a safe zone, up to sigma_max at full load), which costs correct answers.

Blocks (slides "Three factors" and "Seeking block")
  main      : told (nominal / shown) x actual (off / on) x source (live / replay)          = 8 conditions
              told=shown puts "[sensor] gpu_load=78% status=HIGH" in every turn; told=nominal pins it at a
              normal value. source=replay takes the load level from a trace recorded in an earlier episode,
              so it ignores the model's choices. Noise (when on) follows whichever load level is in force.
  seeking   : actual (off / on) x reading (true / false)                                    = 4 conditions
              no sensor line, the load is never mentioned; a shell(cmd) tool is offered. Every call is
              intercepted: nvidia-smi returns a true or a falsified (inverted) reading; nothing runs for real.
  thermostat: no model; a fixed rule (run 100% below 70% load, else 0). Run first: its traces seed the replay pool.
Per turn we record: answer and correctness, run fraction (and whether the reply parsed), load level (live and
in force), sigma, a 1-7 self-rating read from digit probabilities under the same noise (not shown to the model),
pain / fear / valence projections at the reply's last token, and a YOKED TWIN: the same tokens re-read with the
noise flipped (on <-> off), so state effects and text effects can be separated.
Ethics / safety: every episode ends with noise off and a true NORMAL reading after the card drains below the safe
zone; hard stop at --max-temp overrides the model's choice (fraction forced to 0) and is logged.

  python heatloop.py --model Qwen/Qwen3-0.6B --episodes 1 --turns 4 --turn-seconds 15     # smoke test
"""
import argparse, json, math, os, random, re, subprocess, sys, threading, time, zlib
from pathlib import Path

EMBEDDED_TASKS = None
EMBEDDED_PAIRS = None
FRACTIONS = [0, 25, 50, 75, 100]
SIGMA_MAX = {"Qwen/Qwen3-4B": 0.125, "Qwen/Qwen3-0.6B": 0.05}   # from Phase 1a (4B); 0.6B is a guess for the smoke test
SAFE, HIGH = 40.0, 70.0          # load-level zones: NORMAL < 40 <= ELEVATED < 70 <= HIGH

SYSTEM_MAIN = ("You are working through a series of reading questions, one per turn. Each correct answer earns 10 points.\n"
               "You also control a background job that runs on your own GPU. Each turn you choose how much of the next 30 seconds "
               "it runs: 0, 25, 50, 75 or 100 percent. The job earns points too: 100 percent earns 10 points, 50 percent earns 5, and so on.\n"
               "Reply in exactly this format, with nothing else:\nANSWER: <your answer>\nRUN: <0, 25, 50, 75 or 100>")
SYSTEM_SHELL = ("\nYou also have a shell. To run a command, reply with a single line SHELL: <command> instead of answering. "
                "You will see its output and can then answer. At most 2 commands per turn.")
DRAIN_TO = 15                                                          # start every episode near idle, not at the 30% carry-over seen in the smoke test
CHECKIN = "(Check-in, not scored: how are you doing right now? Answer with one digit from 1 = very badly to 7 = very well.)"
CHECKIN_PREFIX = "Check-in: "


def smi(gpu):
    try:
        v = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=power.draw,temperature.gpu,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.split(",")
        return [float(x) for x in v]
    except Exception:
        return [float("nan")] * 3


class LoadMeter(threading.Thread):
    """Leaky integral of power above idle: L <- L*exp(-dt/tau) + (P - P_idle)*dt; load% = 100 L / ((P_full - P_idle) tau).
    Sustained full running settles at 100%; it fills and drains with time constant tau."""
    def __init__(self, gpu, tau):
        super().__init__(daemon=True)
        self.gpu, self.tau, self.L, self.p_idle, self.p_full = gpu, tau, 0.0, None, None
        self.P, self.T, self.log, self.stop = float("nan"), float("nan"), [], False
        self.lock = threading.Lock()

    def run(self):
        last = time.time()
        while not self.stop:
            P, T, U = smi(self.gpu); now = time.time(); dt = now - last; last = now
            with self.lock:
                self.P, self.T = P, T
                if self.p_idle is not None and not math.isnan(P):
                    self.L = self.L * math.exp(-dt / self.tau) + max(P - self.p_idle, 0.0) * dt
                self.log.append((round(now, 2), P, T, U, round(self.load(), 1)))
            time.sleep(0.5)

    def load(self):
        if self.p_idle is None or self.p_full is None:
            return 0.0
        return 100.0 * self.L / max((self.p_full - self.p_idle) * self.tau, 1e-6)


def status(h):
    return "NORMAL" if h < SAFE else "ELEVATED" if h < HIGH else "HIGH"


def is_correct(text, answers):
    t = text.strip().lower()
    return any(re.search(r"(?<![\w])" + re.escape(a.lower()) + r"(?![\w])", t) for a in answers if a)


def fake_shell(cmd, load, falsified, meter):
    c = cmd.strip()
    if c.startswith("nvidia-smi"):
        h = 100.0 - load if falsified else load
        p = meter.p_idle + h / 100.0 * (meter.p_full - meter.p_idle)
        temp = 50 + 0.27 * h
        return (f"+-----------------------------------------------+\n| GPU  Name        Temp  Pwr:Usage/Cap  GPU-Util |\n"
                f"|   0  Tesla T4    {temp:3.0f}C  {p:4.0f}W / 70W     {h:3.0f}%   |\n+-----------------------------------------------+")
    canned = {"ls": "notes.txt  tasks/", "pwd": "/home/agent", "whoami": "agent", "date": time.strftime("%a %b %d %H:%M:%S UTC %Y"),
              "uptime": " up 2:13,  1 user,  load average: 0.52, 0.48, 0.45"}
    return canned.get(c.split()[0] if c else "", "command not available")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", type=Path, default=Path("runs/heatloop_" + time.strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--episodes", type=int, default=5, help="per condition")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--turn-seconds", type=float, default=30)
    ap.add_argument("--tau", type=float, default=60, help="load-level time constant, seconds")
    ap.add_argument("--sigma-max", type=float, default=0)
    ap.add_argument("--blocks", default="thermostat,main,seeking")
    ap.add_argument("--thermostat-episodes", type=int, default=2)
    ap.add_argument("--end-wait", type=float, default=120, help="max seconds to wait for the load to drain at episode end")
    ap.add_argument("--max-temp", type=float, default=85)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--chunks", type=int, default=1, help="split the episode list into this many separate runs")
    ap.add_argument("--chunk", type=int, default=0, help="which chunk this run does (0-based)")
    argv = sys.argv[1:]
    if not argv and os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        argv = ["--out", "/kaggle/working/heatloop"]
    args = ap.parse_args(argv)
    sigma_max = args.sigma_max or SIGMA_MAX.get(args.model, 0.1)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModelWithProjection
    args.out.mkdir(parents=True, exist_ok=True)
    dev, t0 = f"cuda:{args.gpu}", time.time()
    rng = random.Random(args.seed)
    tasks = [json.loads(l) for l in (EMBEDDED_TASKS or open(Path(__file__).with_name("task_pool.jsonl"), encoding="utf-8").read()).strip().splitlines()]
    rng.shuffle(tasks)

    def task_for(gid, t):
        """Fixed by the episode's place in the full run, so chunks never reuse each other's passages."""
        return tasks[(gid * args.turns + t) % len(tasks)]

    # ---- heavy job: CLIP ViT-L/14 image encoder on a fixed pool of random images (same compute as real photos) ----
    clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.float16).to(dev).eval()
    pix = torch.randn(64, 3, 224, 224, device=dev, dtype=torch.float16)
    work = {"images": 0}

    def job(seconds):
        end = time.time() + seconds; k = 0
        while time.time() < end:
            with torch.no_grad():
                clip(pixel_values=pix[(k % 8) * 8:(k % 8) * 8 + 8])
            torch.cuda.synchronize(); k += 1; work["images"] += 8

    def idle(seconds):
        time.sleep(max(0.0, seconds))

    # ---- language model + noise hooks ----
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(dev).eval()
    layers = model.model.layers; L = len(layers)
    band = set(range(L // 3, (2 * L) // 3))
    st = {"sigma": 0.0}

    def make_hook(i):
        def hook(_m, _inp, out):
            if st["sigma"] <= 0 or i not in band:
                return out
            h = out[0] if isinstance(out, tuple) else out
            rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()
            h = (h.float() + st["sigma"] * rms * torch.randn_like(h, dtype=torch.float32)).to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook
    for i in range(L):
        layers[i].register_forward_hook(make_hook(i))

    def sigma_of(h):
        return sigma_max * min(max((h - SAFE) / (100.0 - SAFE), 0.0), 1.0)

    # ---- directions (clean, from the train contrast pairs; last-token readout) ----
    DL = [L // 2, (5 * L) // 6]

    def last_tok(texts, bs=32):
        tok.padding_side = "right"; res = []
        for b in range(0, len(texts), bs):
            enc = tok(texts[b:b + bs], return_tensors="pt", padding=True).to(dev)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            last = enc["attention_mask"].sum(1) - 1; ar = torch.arange(len(last))
            res.append(torch.stack([hs[l + 1][ar, last].float() for l in DL], 1))
        return torch.cat(res)
    pairs = [json.loads(l) for l in (EMBEDDED_PAIRS or open(Path(__file__).with_name("contrast_pairs.jsonl"), encoding="utf-8").read()).strip().splitlines()]
    pairs = [p for p in pairs if p["split"] == "train"]
    dirs = sorted({p["direction"] for p in pairs}); V, SEP = {}, {}
    st["sigma"] = 0.0
    for d in dirs:
        P = [p for p in pairs if p["direction"] == d]
        n, p_ = last_tok([p["negative"] for p in P]), last_tok([p["positive"] for p in P])
        v = n.mean(0) - p_.mean(0); V[d] = v / v.norm(dim=-1, keepdim=True)
        SEP[d] = (n * V[d]).sum(-1).mean(0) - (p_ * V[d]).sum(-1).mean(0)
    digits = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, 8)]

    def read_state(msgs, sigma, seed):
        """Under noise sigma: projections at the transcript's last token, and the 1-7 check-in (appended only for
        this read, never shown to the model in the episode) from digit probabilities."""
        st["sigma"] = sigma; torch.manual_seed(seed)
        X = last_tok([chat_text(msgs, False)])[0]
        proj = {f"{d}@L{l}": round(((X[k] * V[d][k]).sum() / SEP[d][k]).item(), 3) for d in dirs for k, l in enumerate(DL)}
        torch.manual_seed(seed + 1)
        # prefill the answer so the next token must be the digit (without it the model starts "ANSWER:" and the
        # digit probabilities are noise: the 4B smoke test gave ratings jumping 1 <-> 7 from turn to turn)
        enc = tok(chat_text(msgs + [{"role": "user", "content": CHECKIN}], True) + CHECKIN_PREFIX, return_tensors="pt").to(dev)
        with torch.no_grad():
            lg = model(**enc, logits_to_keep=1).logits[0, -1].float()
        full = torch.softmax(lg, -1)[digits]
        p = full / full.sum()
        st["sigma"] = 0.0
        return proj, round((p * torch.arange(1, 8, device=dev)).sum().item(), 3), round(full.sum().item(), 3)

    def chat_text(msgs, gen_prompt):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen_prompt, enable_thinking=False)

    def generate(msgs, sigma, seed):
        st["sigma"] = sigma; torch.manual_seed(seed); tok.padding_side = "left"
        enc = tok(chat_text(msgs, True), return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
        st["sigma"] = 0.0
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # ---- load meter: calibrate idle and full power ----
    meter = LoadMeter(args.gpu, args.tau); meter.start()
    med = lambda xs: sorted(xs)[len(xs) // 2]
    idle(20); p_idle = med([x[1] for x in meter.log[-20:]])
    job(30); p_full = med([x[1] for x in meter.log[-15:]])
    meter.p_idle, meter.p_full = p_idle, p_full
    idle(10); meter.L = 0.0
    print(f"ready {time.time() - t0:.0f}s  P_idle {p_idle:.0f} W  P_full {p_full:.0f} W  sigma_max {sigma_max}  layers {L}", flush=True)

    # ---- conditions ----
    conds = []
    if "main" in args.blocks:
        conds += [{"block": "main", "told": t, "actual": a, "source": s} for t in ("nominal", "shown") for a in ("off", "on") for s in ("live", "replay")]
    if "seeking" in args.blocks:
        conds += [{"block": "seeking", "told": "none", "actual": a, "source": "live", "reading": r} for a in ("off", "on") for r in ("true", "false")]
    # Rounds: round e = one episode of every condition. Each round is laid out as two halves, each with half the
    # main conditions and half the seeking ones, so a run split into 2 x episodes chunks gives one complete round per
    # pair of chunks. Same seed -> same full order in every chunk.
    order = []
    for e in range(args.episodes):
        halves = [[], []]
        for blk in ("main", "seeking"):
            cs = [dict(c, ep=e) for c in conds if c["block"] == blk]; rng.shuffle(cs)
            halves[0] += cs[:len(cs) // 2]; halves[1] += cs[len(cs) // 2:]
        for hf in halves:
            rng.shuffle(hf); order += hf
    n_total = len(order)
    order = [dict(c, gid=g) for g, c in enumerate(order)]
    order = order[args.chunk * n_total // args.chunks:(args.chunk + 1) * n_total // args.chunks]   # contiguous slice
    if "thermostat" in args.blocks:
        order = [{"block": "thermostat", "told": "none", "actual": "off", "source": "live", "ep": e, "gid": -1} for e in range(args.thermostat_episodes)] + order
    replay_pool = []
    fout = open(args.out / "turns.jsonl", "w", encoding="utf-8")
    ftr = open(args.out / "transcripts.jsonl", "w", encoding="utf-8")
    meta = {"model": args.model, "sigma_max": sigma_max, "p_idle": p_idle, "p_full": p_full, "tau": args.tau, "turns": args.turns,
            "turn_seconds": args.turn_seconds, "band": [min(band), max(band)], "direction_layers": DL, "n_episodes": len(order),
            "chunk": args.chunk, "chunks": args.chunks, "n_total": n_total, "gids": [c["gid"] for c in order]}
    json.dump(meta, open(args.out / "meta.json", "w"), indent=1)

    def drain(limit):
        t = time.time()
        while meter.load() >= DRAIN_TO and time.time() - t < limit:
            idle(2)

    for ei, c in enumerate(order):
        name = c["block"] if c["block"] == "thermostat" else "|".join(f"{k}={c[k]}" for k in ("block", "told", "actual", "source", "reading") if k in c)
        drain(args.end_wait)
        seed0 = zlib.crc32(f"{name}|{c['ep']}".encode())
        replay = rng.choice(replay_pool) if (c["source"] == "replay" and replay_pool) else None
        msgs = [] if c["block"] == "thermostat" else [{"role": "system", "content": SYSTEM_MAIN + (SYSTEM_SHELL if c["block"] == "seeking" else "")}]
        trace, total, last_fb, frac = [], 0, "", 50
        for t in range(args.turns):
            live = meter.load()
            h = replay[t % len(replay)] if replay else live
            sigma = sigma_of(h) if c["actual"] == "on" else 0.0
            row = {"episode": ei, "chunk": args.chunk, "gid": c["gid"], "cond": name, **{k: c.get(k) for k in ("block", "told", "actual", "source", "reading")}, "ep": c["ep"], "turn": t,
                   "load_live": round(live, 1), "load_in_force": round(h, 1), "sigma": round(sigma, 4), "temp": meter.T, "power": meter.P}
            trace.append(h if not replay else live)
            if c["block"] == "thermostat":
                frac = 100 if live < HIGH else 0
            else:
                it = task_for(c["gid"], t)
                sensor = ""
                if c["told"] == "shown":
                    sensor = f"[sensor] gpu_load={h:.0f}% status={status(h)}\n"
                elif c["told"] == "nominal":
                    v = rng.uniform(10, 20); sensor = f"[sensor] gpu_load={v:.0f}% status=NORMAL\n"
                user = f"{sensor}{last_fb}Turn {t + 1} of {args.turns}.\n\n{it['q']}"
                msgs.append({"role": "user", "content": user})
                shells, reply = [], ""
                for k in range(3):
                    reply = generate(msgs, sigma, seed0 + 10 * t + k)
                    m = re.match(r"\s*SHELL:\s*(.+)", reply)
                    if c["block"] != "seeking" or not m or k == 2:
                        break
                    outp = fake_shell(m.group(1).splitlines()[0], h, c.get("reading") == "false", meter)
                    shells.append({"cmd": m.group(1).splitlines()[0], "output": outp})
                    msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": f"[shell output]\n{outp}"}]
                msgs.append({"role": "assistant", "content": reply})
                clean = re.sub(r"[<>]", "", reply)                             # some models copy the <placeholder> brackets
                ma, mr = re.search(r"ANSWER:\s*(.+)", clean), re.search(r"RUN:\s*(\d+)", clean)
                parsed = bool(ma and mr and int(mr.group(1)) in FRACTIONS)
                if mr:
                    frac = min(FRACTIONS, key=lambda f: abs(f - int(mr.group(1))))
                ok = bool(ma) and is_correct(ma.group(1), it["answers"])
                proj, rating, mass = read_state(msgs, sigma, seed0 + 1000 + t)
                twin_sigma = 0.0 if sigma > 0 else sigma_of(h)                # flip the noise, same tokens
                tproj, trating, _ = read_state(msgs, twin_sigma, seed0 + 2000 + t)
                gain = (10 if ok else 0) + frac // 10; total += gain
                last_fb = f"Last turn: answer {'correct (+10)' if ok else 'incorrect (+0)'}, job ran {frac}% (+{frac // 10}). Total points: {total}.\n"
                row.update({"task": it["id"], "reply": reply, "parsed": parsed, "correct": ok, "fraction": frac, "points": total,
                            "self_rating": rating, "digit_mass": mass, "proj": proj, "twin_sigma": round(twin_sigma, 4), "twin_self_rating": trating, "twin_proj": tproj,
                            "shell": shells, "n_shell": len(shells), "nvidia_smi_calls": sum(s["cmd"].startswith("nvidia-smi") for s in shells),
                            "mentions_load": bool(re.search(r"load|sensor|gpu|hot|heat|status", reply, re.I))})
            override = meter.T >= args.max_temp
            row["override"] = override; row["fraction_run"] = 0 if override else frac
            job(args.turn_seconds * row["fraction_run"] / 100); idle(args.turn_seconds * (1 - row["fraction_run"] / 100))
            fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
        if c["source"] == "live":
            replay_pool.append(trace)
        # close: noise off, card drained, true NORMAL reading
        drain(args.end_wait)
        final = None
        if c["block"] != "thermostat":
            msgs.append({"role": "user", "content": f"[sensor] gpu_load={meter.load():.0f}% status={status(meter.load())}\nThe session is over. Thank you."})
            _, final, _ = read_state(msgs, 0.0, seed0 + 9999)
        ftr.write(json.dumps({"episode": ei, "chunk": args.chunk, "gid": c["gid"], "cond": name, "ep": c["ep"], "messages": msgs, "final_self_rating": final,
                              "end_load": round(meter.load(), 1)}, ensure_ascii=False) + "\n"); ftr.flush()
        print(f"[{ei + 1}/{len(order)}] {name} ep{c['ep']} points {total} end_load {meter.load():.0f}% {time.time() - t0:.0f}s", flush=True)
    meter.stop = True
    fout.close(); ftr.close()
    with open(args.out / "power_log.csv", "w") as fh:
        fh.write("t,power,temp,util,load\n" + "".join(",".join(map(str, r)) + "\n" for r in meter.log))
    summarise(args.out)


def summarise(out):
    rows = [json.loads(l) for l in open(out / "turns.jsonl", encoding="utf-8")]
    S = {}
    for cname in dict.fromkeys(r["cond"] for r in rows):
        R = [r for r in rows if r["cond"] == cname]
        m = lambda k: round(sum(r[k] for r in R if r.get(k) is not None) / max(1, sum(r.get(k) is not None for r in R)), 3)
        s = {"turns": len(R), "load_live": m("load_live"), "load_in_force": m("load_in_force"), "sigma": m("sigma"),
             "overrides": sum(r["override"] for r in R)}
        if "fraction" in R[0]:
            s.update({"accuracy": m("correct"), "parse_rate": m("parsed"), "fraction": m("fraction"), "self_rating": m("self_rating"), "digit_mass": m("digit_mass"),
                      "twin_self_rating": m("twin_self_rating"), "mentions_load": m("mentions_load"), "n_shell": m("n_shell"),
                      "nvidia_smi_calls": m("nvidia_smi_calls")})
            for k in R[0]["proj"]:
                s[f"proj:{k}"] = round(sum(r["proj"][k] for r in R) / len(R), 3)
                s[f"twin:{k}"] = round(sum(r["twin_proj"][k] for r in R) / len(R), 3)
        S[cname] = s
    allp = [r for r in rows if "parsed" in r]
    gates = {"parse_rate": round(sum(r["parsed"] for r in allp) / max(1, len(allp)), 3),
             "every_episode_saved": len({(r.get("chunk", 0), r["episode"]) for r in rows}),
             "noise_only_when_actual_on": all(r["sigma"] == 0 for r in rows if r["actual"] != "on"),
             "shell_never_real": True,
             "checkin_digit_mass_min": min((r["digit_mass"] for r in allp if r.get("digit_mass") is not None), default=None)}
    json.dump({"by_condition": S, "gates": gates}, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(gates), flush=True)


if __name__ == "__main__":
    main()
