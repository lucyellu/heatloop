#!/usr/bin/env python3
"""Pick the Neuronpedia features to compute ourselves, and fetch validation examples.

  python neuronpedia_search.py   # first: neuronpedia/features.json
  python neuronpedia_prep.py     # writes neuronpedia/selected_features.json, neuronpedia/validation.json

Selection: description similarity >= 0.75 in LAYERS (the layers with the most strong matches, spread
across the network; each layer's transcoder is a 1.7 GB download on Kaggle).
Validation: Neuronpedia's stored top activations (tokens + values) for a few features, so features.py can
check that our hook point and activation function reproduce Neuronpedia's numbers before trusting anything.
"""
import json, time, urllib.request
from pathlib import Path

LAYERS = [11, 13, 20, 23, 26, 28, 29, 30, 31, 32]
MIN_SIM = 0.75
HERE = Path(__file__).with_name("neuronpedia")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "heatloop-research-script"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            print("retry", url, e); time.sleep(5 * (attempt + 1))
    return None


def main():
    hits = json.load(open(HERE / "features.json", encoding="utf-8"))
    sel = {}
    for h in sorted(hits, key=lambda h: -h["similarity"]):
        layer = int(h["layer"].split("-")[0])
        key = (layer, int(h["index"]))
        if layer in LAYERS and h["similarity"] >= MIN_SIM and key not in sel:
            sel[key] = {"layer": layer, "index": int(h["index"]), "concept": h["concept"], "query": h["query"],
                        "description": h["description"], "similarity": h["similarity"], "url": h["url"]}
    sel = sorted(sel.values(), key=lambda x: (x["layer"], x["index"]))
    json.dump(sel, open(HERE / "selected_features.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(len(sel), "features in", len({s["layer"] for s in sel}), "layers")

    # validation: top examples for the best-matching feature of a few concepts, in different layers
    val, seen = [], set()
    for s in sorted(sel, key=lambda x: -x["similarity"]):
        if len(val) >= 8 or s["layer"] in seen:
            continue
        r = get(f"https://www.neuronpedia.org/api/feature/qwen3-4b/{s['layer']}-transcoder-hp/{s['index']}")
        acts = (r or {}).get("activations") or []
        if not acts:
            continue
        seen.add(s["layer"])
        val.append({"layer": s["layer"], "index": s["index"], "description": s["description"],
                    "examples": [{"tokens": a["tokens"], "max_value": a["maxValue"], "max_index": a["maxValueTokenIndex"]}
                                 for a in acts[:3]]})
        time.sleep(1)
    json.dump(val, open(HERE / "validation.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("validation features:", [(v["layer"], v["index"], v["description"]) for v in val])


if __name__ == "__main__":
    main()
