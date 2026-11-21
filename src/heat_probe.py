#!/usr/bin/env python3
"""Phase 0 heat probe: how far, and how fast, does a real GPU job move this card's temperature?

Runs unchanged on Kaggle (T4 / T4 x2 / P100) and on the local GTX 1050 Ti.

  1. idle      log the card with nothing running
  2. load      CLIP ViT-L/14 image embedding of COCO val2017 at several batch sizes
  3. cooldown  stop the job and log how long the card takes to come back down

nvidia-smi is polled every second for EVERY GPU in the machine (on T4 x2, GPU 1 is the
real "neighbour" card). A hard safety pause stops the job at --max-temp regardless.

Outputs in --out:  temps.csv (per second, per GPU), batches.csv (per batch), summary.json
summary.json suggests H_lo / H_hi for the noise schedule in heatloop.

Usage
  Kaggle:  !python heat_probe.py --download --max-temp 85
  Local:   python heat_probe.py --download --max-temp 80 --fp32
"""
import argparse, csv, json, os, subprocess, sys, threading, time, urllib.request, zipfile
from pathlib import Path

COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
FIELDS = ["index", "temperature.gpu", "utilization.gpu", "power.draw", "clocks.sm", "clocks.mem", "fan.speed"]


def smi():
    """One row per GPU: dict of FIELDS. Values that a card does not report come back as None."""
    out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout
    rows = []
    for line in out.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        rows.append({k: (float(v) if v.replace(".", "", 1).isdigit() else None) for k, v in zip(FIELDS, vals)})
    return rows


class Logger(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "phase"] + FIELDS)
        self.phase, self.t0, self.stop, self.latest = "start", time.time(), False, {}

    def run(self):
        while not self.stop:
            t = round(time.time() - self.t0, 1)
            try:
                for r in smi():
                    self.latest[int(r["index"])] = r
                    self.w.writerow([t, self.phase] + [r[k] for k in FIELDS])
                self.f.flush()
            except Exception as e:
                print("nvidia-smi failed:", e, file=sys.stderr)
            time.sleep(1)

    def temp(self, gpu):
        return (self.latest.get(gpu) or {}).get("temperature.gpu") or 0


def get_images(data_dir: Path, download: bool, n: int):
    img_dir = data_dir / "val2017"
    if not img_dir.is_dir():
        if not download:
            sys.exit(f"No images at {img_dir}. Pass --download, or --data pointing at a folder that holds val2017/.")
        data_dir.mkdir(parents=True, exist_ok=True)
        z = data_dir / "val2017.zip"
        if not z.exists():
            print("downloading COCO val2017 (~0.8 GB)...", flush=True)
            urllib.request.urlretrieve(COCO_URL, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(data_dir)
    paths = sorted(img_dir.glob("*.jpg"))[:n]
    print(f"{len(paths)} images from {img_dir}", flush=True)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/heat_probe_" + time.strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--gpu", type=int, default=0, help="the card that runs the job")
    ap.add_argument("--pool", type=int, default=512, help="images preprocessed once and cycled (load is GPU-bound)")
    ap.add_argument("--batches", default="8,32,64")
    ap.add_argument("--idle-s", type=int, default=120)
    ap.add_argument("--load-s", type=int, default=240, help="seconds at each batch size")
    ap.add_argument("--cool-s", type=int, default=360)
    ap.add_argument("--max-temp", type=float, default=85, help="hard safety pause (C); resume 5 C below")
    ap.add_argument("--fp32", action="store_true", help="use on Pascal cards (1050 Ti, P100 is fine either way)")
    argv = sys.argv[1:]
    if not argv and os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):   # pushed as a Kaggle script: no CLI args
        argv = ["--download", "--max-temp", "85", "--data", "/kaggle/temp/coco",
                "--out", "/kaggle/working/heat_probe"]
    args = ap.parse_args(argv)

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    args.out.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}"
    dtype = torch.float32 if args.fp32 else torch.float16

    paths = get_images(args.data, args.download, args.pool)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype).to(dev).eval()
    pix = proc(images=[Image.open(p).convert("RGB") for p in paths], return_tensors="pt")["pixel_values"].to(dev, dtype)
    print("preprocessed pool", tuple(pix.shape), flush=True)

    env = {"gpus": [r for r in smi()], "torch": torch.__version__, "device_name": torch.cuda.get_device_name(args.gpu),
           "kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")), "args": {k: str(v) for k, v in vars(args).items()}}
    procs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip()
    others = [l for l in procs.splitlines() if l.split(",")[0].strip() != str(os.getpid())]   # not ourselves
    env["other_processes_at_start"] = "; ".join(others) or "none"

    log = Logger(args.out / "temps.csv"); log.start()
    bf = open(args.out / "batches.csv", "w", newline=""); bw = csv.writer(bf)
    bw.writerow(["t", "phase", "batch_size", "seconds", "images_per_s", "temp"])
    pauses = []

    log.phase = "idle"; print(f"idle {args.idle_s}s", flush=True); time.sleep(args.idle_s)

    i = 0
    for bs in [int(b) for b in args.batches.split(",")]:
        log.phase = f"load_bs{bs}"; print(f"load bs={bs} for {args.load_s}s", flush=True)
        end = time.time() + args.load_s
        while time.time() < end:
            if log.temp(args.gpu) >= args.max_temp:
                t0 = time.time(); log.phase = f"safety_pause_bs{bs}"
                print(f"  safety pause at {log.temp(args.gpu)} C", flush=True)
                while log.temp(args.gpu) > args.max_temp - 5:
                    time.sleep(2)
                pauses.append({"batch_size": bs, "seconds": round(time.time() - t0, 1)})
                log.phase = f"load_bs{bs}"
            batch = pix.index_select(0, torch.arange(i, i + bs, device=dev) % len(pix))
            t0 = time.time()
            try:
                with torch.no_grad():
                    model.get_image_features(pixel_values=batch)
                torch.cuda.synchronize(dev)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at bs={bs}; skipping this size", flush=True)
                torch.cuda.empty_cache(); break
            dt = time.time() - t0; i += bs
            bw.writerow([round(time.time() - log.t0, 1), log.phase, bs, round(dt, 3), round(bs / dt, 1), log.temp(args.gpu)])
        bf.flush()

    log.phase = "cooldown"; print(f"cooldown {args.cool_s}s", flush=True)
    del model; torch.cuda.empty_cache(); time.sleep(args.cool_s)
    log.stop = True; log.join(timeout=5); log.f.close(); bf.close()

    # summary for the noise schedule
    rows = list(csv.DictReader(open(args.out / "temps.csv")))
    def temps(pred, gpu=args.gpu):
        return sorted(float(r["temperature.gpu"]) for r in rows
                      if r["temperature.gpu"] not in ("", "None") and int(float(r["index"])) == gpu and pred(r["phase"]))
    def pct(xs, q): return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None
    idle, load, cool = temps(lambda p: p == "idle"), temps(lambda p: p.startswith("load")), temps(lambda p: p == "cooldown")
    first_cool = [(float(r["t"]), float(r["temperature.gpu"])) for r in rows
                  if r["phase"] == "cooldown" and int(float(r["index"])) == args.gpu and r["temperature.gpu"] not in ("", "None")]
    back = next((t - first_cool[0][0] for t, v in first_cool if idle and v <= pct(idle, 0.5) + 5), None) if first_cool else None
    summary = {
        "idle_p10": pct(idle, 0.1), "idle_p50": pct(idle, 0.5), "load_p90": pct(load, 0.9), "max": max(load or [0]),
        "range_C": (pct(load, 0.9) - pct(idle, 0.1)) if idle and load else None,
        "cool_to_idle_plus5_s": back, "safety_pauses": pauses,
        "suggested": {"H_lo": pct(idle, 0.1), "H_hi": pct(load, 0.9)},
        "gate_range_ge_15C": bool(idle and load and pct(load, 0.9) - pct(idle, 0.1) >= 15),
        "neighbour_gpus": {g: {"idle_p50": pct(temps(lambda p: p == "idle", g), 0.5),
                               "load_p90": pct(temps(lambda p: p.startswith("load"), g), 0.9)}
                           for g in sorted({int(float(r["index"])) for r in rows}) if g != args.gpu},
        "env": env,
    }
    json.dump(summary, open(args.out / "summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "env"}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
