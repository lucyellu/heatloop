#!/usr/bin/env bash
# Assemble the public site in site/: only the pages and talk exports, not the raw runs.
set -e
cd "$(dirname "$0")"
rm -rf site && mkdir -p site
cp site_index.html site/index.html
cp research_map.html site/research-map.html
cp slides.html site/lab-deck.html
cp ai_model_23andme_genealogy_provenance_scanner.html site/modeldna.html
cp "The Model That Can’t Feel the Stove.pdf" site/talk.pdf
cp "The Model That Can’t Feel the Stove.pptx" site/talk.pptx
echo "site/ ready: $(ls site | tr '\n' ' ')"
