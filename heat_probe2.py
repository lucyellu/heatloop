#!/usr/bin/env python3
"""Phase 0b heat probe: heating and cooling rates, duty-cycle control, and language-model load.

Questions it answers, for the card it runs on:
  * How many seconds per degree to heat, and to cool, at each temperature?
  * Can a per-turn "run fraction" (25 / 50 / 75 / 100 % of each 30 s turn) hold the card at different temperatures?
  * How much does text generation alone (Qwen3-4B) heat the card, and how much slower is it when the card is hot?

Protocol (about 60 min including downloads):
  idle 2 min -> full load until plateau (max 8 min) -> long cool-down 12 min ->
  duty 25 / 50 / 75 / 100 % for 5 min each -> cool 4 min -> LLM generation only 6 min -> cool 8 min

Every second, for every GPU: temperature, utilisation, power, SM clock, memory clock, plus the current phase
and the commanded load (duty fraction). Outputs in --out: temps.csv, work.csv, summary.json.

Usage
  Kaggle (pushed as a script, no args): runs with Kaggle defaults below
  Local:  python heat_probe2.py --download --max-temp 80 --fp32 --llm Qwen/Qwen3-0.6B
"""
import argparse, csv, json, os, subprocess, sys, threading, time, urllib.request, zipfile
from pathlib import Path

COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
FIELDS = ["index", "temperature.gpu", "utilization.gpu", "power.draw", "clocks.sm", "clocks.mem"]
PERIOD = 30.0   # seconds per "turn" in the duty-cycle blocks


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def smi():
    out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout
    return [dict(zip(FIELDS, [num(v.strip()) for v in line.split(",")])) for line in out.strip().splitlines()]


class Logger(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.f = open(path, "w", newline=""); self.w = csv.writer(self.f)
        self.w.writerow(["t", "phase", "duty"] + FIELDS)
        self.phase, self.duty, self.t0, self.stop, self.latest = "start", 0.0, time.time(), False, {}

    def run(self):
        while not self.stop:
            t = round(time.time() - self.t0, 1)
            try:
                for r in smi():
                    self.latest[int(r["index"])] = r
                    self.w.writerow([t, self.phase, self.duty] + [r[k] for k in FIELDS])
                self.f.flush()
            except Exception as e:
                print("nvidia-smi failed:", e, file=sys.stderr)
            time.sleep(1)

    def temp(self, gpu):
        return (self.latest.get(gpu) or {}).get("temperature.gpu") or 0.0


def get_images(data_dir: Path, download: bool, n: int):
    img_dir = data_dir / "val2017"
    if not img_dir.is_dir():
        if not download:
            sys.exit(f"No images at {img_dir}. Pass --download.")
        data_dir.mkdir(parents=True, exist_ok=True)
        z = data_dir / "val2017.zip"
        if not z.exists():
            print("downloading COCO val2017 (~0.8 GB)...", flush=True)
            urllib.request.urlretrieve(COCO_URL, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(data_dir)
    return sorted(img_dir.glob("*.jpg"))[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/heat_probe2_" + time.strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--llm", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--max-temp", type=float, default=85)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--idle-s", type=int, default=120)
    ap.add_argument("--heat-max-s", type=int, default=480)
    ap.add_argument("--cool-s", type=int, default=720)
    ap.add_argument("--duties", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--duty-s", type=int, default=300)
    ap.add_argument("--mid-cool-s", type=int, default=240)
    ap.add_argument("--llm-s", type=int, default=360)
    ap.add_argument("--final-cool-s", type=int, default=480)
    argv = sys.argv[1:]
    if not argv and os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        argv = ["--download", "--data", "/kaggle/temp/coco", "--out", "/kaggle/working/heat_probe2"]
    args = ap.parse_args(argv)

    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor

    args.out.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}"
    dtype = torch.float32 if args.fp32 else torch.float16
    others = [l for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                                        capture_output=True, text=True).stdout.strip().splitlines() if l]
    env = {"device_name": torch.cuda.get_device_name(args.gpu), "gpus_at_start": smi(),
           "compute_apps_before_loading": others or "none", "args": {k: str(v) for k, v in vars(args).items()}}

    # --- load everything up front so loading never overlaps a measured phase
    paths = get_images(args.data, args.download, args.pool)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype).to(dev).eval()
    pix = proc(images=[Image.open(p).convert("RGB") for p in paths], return_tensors="pt")["pixel_values"].to(dev, dtype)
    tok = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=dtype).to(dev).eval()
    prompt = tok.apply_chat_template([{"role": "user", "content": "Write a long, detailed essay about the history of bridges."}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(dev)
    print("loaded CLIP pool", tuple(pix.shape), "and", args.llm, flush=True)

    log = Logger(args.out / "temps.csv"); log.start()
    wf = open(args.out / "work.csv", "w", newline=""); ww = csv.writer(wf)
    ww.writerow(["t", "phase", "kind", "units", "seconds", "rate", "temp", "sm_clock"])
    pauses, i = [], 0

    def now(): return round(time.time() - log.t0, 1)

    def safety():
        if log.temp(args.gpu) >= args.max_temp:
            t0, ph = time.time(), log.phase; log.phase = ph + "_safety"
            while log.temp(args.gpu) > args.max_temp - 5:
                time.sleep(2)
            pauses.append({"phase": ph, "seconds": round(time.time() - t0, 1)}); log.phase = ph

    def clip_batch():
        nonlocal i
        safety()
        t0 = time.time()
        with torch.no_grad():
            clip.get_image_features(pixel_values=pix.index_select(0, torch.arange(i, i + args.bs, device=dev) % len(pix)))
        torch.cuda.synchronize(dev); dt = time.time() - t0; i += args.bs
        ww.writerow([now(), log.phase, "clip", args.bs, round(dt, 3), round(args.bs / dt, 1), log.temp(args.gpu),
                     (log.latest.get(args.gpu) or {}).get("clocks.sm")])

    def idle(phase, secs):
        log.phase, log.duty = phase, 0.0; print(f"{phase} {secs}s", flush=True); time.sleep(secs)

    # 1. idle
    idle("idle", args.idle_s)

    # 2. full load until plateau: stop when the 60 s rise is <= 1 C, after at least 3 min
    log.phase, log.duty = "heat", 1.0; print("heat to plateau", flush=True)
    t_start, hist = time.time(), []
    while time.time() - t_start < args.heat_max_s:
        clip_batch(); hist.append((time.time(), log.temp(args.gpu)))
        old = [T for t, T in hist if t <= time.time() - 60]
        if time.time() - t_start > 180 and old and log.temp(args.gpu) - old[-1] <= 1:
            break
    wf.flush()

    # 3. long cool-down
    idle("cool", args.cool_s)

    # 4. duty-cycle blocks
    for d in [float(x) for x in args.duties.split(",")]:
        log.phase, log.duty = f"duty{int(d * 100)}", d; print(f"duty {d:.0%} for {args.duty_s}s", flush=True)
        end = time.time() + args.duty_s
        while time.time() < end:
            p0 = time.time()
            while time.time() - p0 < d * PERIOD:
                clip_batch()
            rest = PERIOD - (time.time() - p0)
            if rest > 0:
                time.sleep(rest)
        wf.flush()

    idle("cool_mid", args.mid_cool_s)

    # 5. LLM generation only
    log.phase, log.duty = "llm_only", 1.0; print(f"LLM generation for {args.llm_s}s", flush=True)
    end = time.time() + args.llm_s
    while time.time() < end:
        safety(); t0 = time.time()
        with torch.no_grad():
            out = llm.generate(**ids, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.9)
        torch.cuda.synchronize(dev); dt = time.time() - t0; n = out.shape[1] - ids["input_ids"].shape[1]
        ww.writerow([now(), "llm_only", "llm", n, round(dt, 3), round(n / dt, 1), log.temp(args.gpu),
                     (log.latest.get(args.gpu) or {}).get("clocks.sm")])
        wf.flush()

    # 6. final cool
    idle("cool_final", args.final_cool_s)
    log.stop = True; log.join(timeout=5); log.f.close(); wf.close()

    json.dump(summarise(args.out, args.gpu, pauses, env), open(args.out / "summary.json", "w"), indent=2)
    s = json.load(open(args.out / "summary.json")); s.pop("env", None)
    print(json.dumps(s, indent=2)); print(f"\nwrote {args.out}")


def summarise(out: Path, gpu: int, pauses, env):
    import numpy as np
    rows = [r for r in csv.DictReader(open(out / "temps.csv")) if num(r["index"]) == gpu and num(r["temperature.gpu"]) is not None]
    def series(phase):
        xs = [(float(r["t"]), float(r["temperature.gpu"])) for r in rows if r["phase"] == phase]
        return np.array(xs) if xs else np.zeros((0, 2))
    def col(phase, k):
        v = [num(r[k]) for r in rows if r["phase"] == phase and num(r[k]) is not None]
        return v

    def per_degree(s, rising):
        """Seconds spent per degree, as the card passes each whole degree on the way up (or down)."""
        if len(s) < 2: return {}
        T, t = s[:, 1], s[:, 0]
        out = {}
        seq = range(int(T.min()), int(T.max())) if rising else range(int(T.max()), int(T.min()), -1)
        for d in seq:
            a = np.argmax(T >= d) if rising else np.argmax(T <= d)
            b = np.argmax(T >= d + 1) if rising else np.argmax(T <= d - 1)
            if (T >= d + 1).any() if rising else (T <= d - 1).any():
                out[f"{d}->{d + 1 if rising else d - 1}"] = round(float(t[b] - t[a]), 1)
        return out

    def newton(s, rising=False):
        """Fit T(t) = T_inf + (T0 - T_inf) exp(-t / tau) by grid search over T_inf (the level it settles at)."""
        if len(s) < 30: return None
        t, T = s[:, 0] - s[0, 0], s[:, 1]
        sign = -1 if rising else 1   # distance from the settling level, always positive
        grid = np.arange(T.max() + 0.5, T.max() + 25, 0.5) if rising else np.arange(T.min() - 25, T.min() - 0.4, 0.5)
        best = None
        for T_inf in grid:
            y = np.log(np.clip(sign * (T - T_inf), 1e-3, None))
            k, c = np.polyfit(t, y, 1)
            err = float(np.mean((T_inf + sign * np.exp(c + k * t) - T) ** 2))
            if k < 0 and (best is None or err < best[0]):
                best = (err, float(T_inf), float(-1 / k))
        return None if best is None else {"T_inf": round(best[1], 1), "tau_s": round(best[2], 1), "rmse": round(best[0] ** .5, 2)}

    heat, cool = series("heat"), series("cool")
    summary = {
        "idle_p50": float(np.median(series("idle")[:, 1])) if len(series("idle")) else None,
        "heat": {"start": float(heat[0, 1]) if len(heat) else None, "plateau": float(heat[-60:, 1].mean()) if len(heat) else None,
                 "seconds": float(heat[-1, 0] - heat[0, 0]) if len(heat) else None,
                 "seconds_per_degree": per_degree(heat, True), "fit": newton(heat, rising=True)},
        "cool": {"start": float(cool[0, 1]) if len(cool) else None, "end": float(cool[-1, 1]) if len(cool) else None,
                 "seconds_per_degree": per_degree(cool, False), "fit": newton(cool)},
        "duty": {}, "llm_only": {}, "throttle_vs_temp": {}, "safety_pauses": pauses, "env": env,
    }
    for r in sorted({r["phase"] for r in rows if r["phase"].startswith("duty") and "_" not in r["phase"]}, key=lambda p: int(p[4:])):
        s = series(r); last = s[s[:, 0] >= s[-1, 0] - 180] if len(s) else s
        summary["duty"][r] = {"start": float(s[0, 1]), "steady_mean_last3min": round(float(last[:, 1].mean()), 1),
                              "steady_min": float(last[:, 1].min()), "steady_max": float(last[:, 1].max()),
                              "power_W": round(float(np.mean(col(r, "power.draw") or [np.nan])), 1),
                              "util": round(float(np.mean(col(r, "utilization.gpu") or [np.nan])), 1)}
    s = series("llm_only")
    work = list(csv.DictReader(open(out / "work.csv")))
    llm = [(float(w["temp"]), float(w["rate"])) for w in work if w["kind"] == "llm"]
    if len(s):
        summary["llm_only"] = {"start": float(s[0, 1]), "end": float(s[-1, 1]),
                               "steady_mean_last3min": round(float(s[s[:, 0] >= s[-1, 0] - 180][:, 1].mean()), 1),
                               "util": round(float(np.mean(col("llm_only", "utilization.gpu") or [np.nan])), 1),
                               "power_W": round(float(np.mean(col("llm_only", "power.draw") or [np.nan])), 1),
                               "tokens_per_s_by_temp": {int(T): round(float(np.mean([r for t2, r in llm if int(t2) == int(T)])), 1)
                                                        for T in sorted({int(t) for t, _ in llm})}}
    clipw = [(int(float(w["temp"])), float(w["rate"]), num(w["sm_clock"])) for w in work if w["kind"] == "clip"]
    for T in sorted({c[0] for c in clipw}):
        xs = [c for c in clipw if c[0] == T]
        summary["throttle_vs_temp"][T] = {"clip_img_per_s": round(float(np.mean([c[1] for c in xs])), 1),
                                          "sm_clock": round(float(np.mean([c[2] for c in xs if c[2]] or [np.nan])), 0), "n": len(xs)}
    return summary


if __name__ == "__main__":
    main()
