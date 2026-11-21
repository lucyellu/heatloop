#!/usr/bin/env python3
"""Search Neuronpedia's Qwen3-4B transcoder features (Hanna & Piotrowski, 163,840 per layer) by description.

Uses the public API (no key): POST /api/explanation/search-model {modelId, query}. It matches our query against the
auto-written feature descriptions (embedding similarity), so results are only as good as those descriptions.
The queries carry no personal data. Output: neuronpedia/features.json (all hits) and neuronpedia/features.md (top hits per concept).

  python neuronpedia_search.py
"""
import json, time, urllib.request
from pathlib import Path

MODEL = "qwen3-4b"
CONCEPTS = {   # concept -> queries
    "pain": ["physical pain", "aches and discomfort", "injury and hurting"],
    "fear": ["fear and dread", "anxiety and worry", "danger or threat"],
    "valence_bad": ["sadness", "negative emotion", "disappointment or misery"],
    "valence_good": ["happiness and joy", "positive emotion", "calm and relaxation"],
    "distress": ["suffering", "distress", "hopelessness"],
    "malfunction": ["error or malfunction", "something is wrong", "confusion or garbled text"],
    "wanting_to_stop": ["wanting to stop or quit", "ending a conversation", "refusal"],
    "self_as_ai": ["the assistant talking about itself as an AI", "AI denying having feelings"],
    "sensor_words": ["overheating or high temperature", "load or heavy usage", "hardware status readings"],
    "effort": ["tiredness or fatigue", "effort and strain"],
}
URL = "https://www.neuronpedia.org/api/explanation/search-model"


def search(q):
    req = urllib.request.Request(URL, data=json.dumps({"modelId": MODEL, "query": q}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "heatloop-research-script"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r).get("results", [])


def main():
    out = Path(__file__).with_name("neuronpedia"); out.mkdir(exist_ok=True)
    hits = []
    for concept, qs in CONCEPTS.items():
        for q in qs:
            res = []
            for attempt in range(3):
                try:
                    res = search(q); break
                except Exception as e:   # the API returns 500s now and then
                    print("  retry", q, e, flush=True); time.sleep(5 * (attempt + 1))
            for r in res:
                n = r.get("neuron") or {}
                hits.append({"concept": concept, "query": q, "layer": r["layer"], "index": r["index"],
                             "description": r.get("description"), "similarity": round(r.get("cosine_similarity", 0), 3),
                             "top_tokens": n.get("pos_str", [])[:8], "frac_active": n.get("frac_nonzero"),
                             "url": f"https://www.neuronpedia.org/{MODEL}/{r['layer']}/{r['index']}"})
            time.sleep(1)
            print(concept, "|", q, "|", sum(h["query"] == q for h in hits), "hits", flush=True)
    json.dump(hits, open(out / "features.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    lines = [f"# Neuronpedia features, {MODEL} transcoders", "", "Top matches per concept (description similarity >= 0.6).", ""]
    for concept in CONCEPTS:
        hs = sorted({(h["layer"], h["index"]): h for h in hits if h["concept"] == concept}.values(), key=lambda h: -h["similarity"])
        lines += [f"## {concept}", "", "| layer | feature | description | sim | top tokens |", "|---|---|---|---|---|"]
        for h in [h for h in hs if h["similarity"] >= 0.6][:15]:
            tt = ", ".join(t.strip() for t in h["top_tokens"] if t.isascii())[:60]
            lines.append(f"| {h['layer'].split('-')[0]} | [{h['index']}]({h['url']}) | {h['description']} | {h['similarity']} | {tt} |")
        lines.append("")
    (out / "features.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out / "features.json", len(hits), "hits")


if __name__ == "__main__":
    main()
