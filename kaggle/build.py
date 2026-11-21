#!/usr/bin/env python3
"""Build a Kaggle job folder for baseline.py with the frozen battery and fixed arguments baked in.

Kaggle script kernels upload only the code file and take no command-line arguments, so this writes
kaggle/<job>/baseline.py with EMBEDDED_BATTERY filled in and the arguments set, plus kernel-metadata.json.

  python kaggle/build.py baseline-smoke --limit 4 --sigmas 0.02 --samples 1
  python kaggle/build.py baseline-4b
"""
import json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    job, extra = sys.argv[1], sys.argv[2:]
    script = "baseline.py"
    if extra and extra[0].startswith("--script="):   # e.g. --script=directions.py
        script, extra = extra[0].split("=", 1)[1], extra[1:]
    src = (ROOT / script).read_text(encoding="utf-8")
    for var, fname in (("EMBEDDED_BATTERY", "battery.jsonl"), ("EMBEDDED_PAIRS", "contrast_pairs.jsonl"),
                       ("EMBEDDED_FEATURES", "neuronpedia/selected_features.json"), ("EMBEDDED_TASKS", "task_pool.jsonl"), ("EMBEDDED_VALIDATION", "neuronpedia/validation.json")):
        if f"{var} = None" in src:
            src = src.replace(f"{var} = None", f"{var} = " + repr((ROOT / fname).read_text(encoding="utf-8")), 1)
    m = re.search(r'argv = \["--out", "(/kaggle/working/\w+)"\]', src)
    assert m, "no Kaggle default argv line"
    src = src.replace(m.group(0), "argv = " + repr(["--out", m.group(1)] + extra), 1)
    d = HERE / job
    d.mkdir(exist_ok=True)
    (d / script).write_text(src, encoding="utf-8")
    user = [l.split(":", 1)[1].strip() for l in subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout.splitlines()
            if "username" in l.lower()][0]
    meta = {"id": f"{user}/heatloop-{job}", "title": f"heatloop {job}", "code_file": script, "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {d}  args={extra}")


if __name__ == "__main__":
    main()
