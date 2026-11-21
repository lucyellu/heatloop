#!/usr/bin/env python3
"""Merge chunked heatloop runs into one result.

  python merge_chunks.py runs/kaggle_heatloop_chunk*    -> runs/heatloop_main/{turns,transcripts}.jsonl, summary.json

Works on however many chunks are done so far, and says which of the full run's episodes are still missing."""
import json, sys
from pathlib import Path
from heatloop import summarise

OUT = Path("runs/heatloop_main")


def main():
    dirs = sorted({p.parent for a in sys.argv[1:] for p in Path(a).rglob("meta.json")})
    if not dirs:
        sys.exit("no chunk outputs found")
    OUT.mkdir(parents=True, exist_ok=True)
    metas, seen = [], set()
    with open(OUT / "turns.jsonl", "w", encoding="utf-8") as ft, open(OUT / "transcripts.jsonl", "w", encoding="utf-8") as fx:
        for d in dirs:
            m = json.load(open(d / "meta.json")); metas.append(m)
            if m["chunk"] in seen:
                sys.exit(f"chunk {m['chunk']} appears twice ({d})")
            seen.add(m["chunk"])
            for name, f in (("turns.jsonl", ft), ("transcripts.jsonl", fx)):
                if (d / name).exists():
                    f.write(open(d / name, encoding="utf-8").read())
    summarise(OUT)
    chunks = metas[0]["chunks"]
    got = {g for m in metas for g in m["gids"] if g >= 0}
    missing = sorted(set(range(metas[0]["n_total"])) - got)
    print(f"chunks done: {sorted(seen)} of {chunks}")
    print(f"P_idle / P_full per chunk: {[(m['chunk'], round(m['p_idle']), round(m['p_full'])) for m in metas]}")
    print(f"main/seeking episodes merged: {len(got)} of {metas[0]['n_total']} (missing: {len(missing)})  ->  {OUT}")


if __name__ == "__main__":
    main()
