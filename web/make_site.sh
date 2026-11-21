#!/usr/bin/env bash
# Assemble the public site in "$OUT"/: only the pages and talk exports, not the raw runs.
set -e
cd "$(dirname "$0")"; OUT=../site
rm -rf "$OUT" && mkdir -p "$OUT"
cp site_index.html "$OUT"/index.html
cp research_map.html "$OUT"/research-map.html
cp slides.html "$OUT"/lab-deck.html
cp ai_model_23andme_genealogy_provenance_scanner.html "$OUT"/modeldna.html
cp "The Model That Can’t Feel the Stove.pdf" "$OUT"/talk.pdf
cp "The Model That Can’t Feel the Stove.pptx" "$OUT"/talk.pptx
echo "site/ ready: $(ls "$OUT" | tr '\n' ' ')"
