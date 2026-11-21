#!/usr/bin/env bash
# Push chunk(s) of the full Qwen3-4B heatloop run to Kaggle, or fetch finished ones.
#   bash run_chunk.sh push 0 1      -> pushes kernels heatloop-chunk0, heatloop-chunk1 (Kaggle allows 2 at once)
#   bash run_chunk.sh fetch 0 1     -> downloads to runs/kaggle_heatloop_chunk<k>/, then merges everything done so far
# 10 chunks = 5 rounds x 2 halves; each chunk ~50-60 min (6 episodes + 1 thermostat + calibration).
set -e
cd "$(dirname "$0")"
CHUNKS=10
mode=$1; shift
for k in "$@"; do
  if [ "$mode" = push ]; then
    python kaggle/build.py "chunk$k" --script=heatloop.py --model Qwen/Qwen3-4B --chunks $CHUNKS --chunk "$k" --thermostat-episodes 1 | tail -1 | sed -E 's#[A-Za-z0-9_-]+/heatloop-#<you>/heatloop-#g'
    kaggle kernels push -p "kaggle/chunk$k" 2>&1 | sed -E 's#https://\S+#<url>#'
  else
    mkdir -p "runs/kaggle_heatloop_chunk$k"
    kaggle kernels output "heatloop-chunk$k" -p "runs/kaggle_heatloop_chunk$k" >/dev/null 2>&1
    echo "chunk $k: $(ls runs/kaggle_heatloop_chunk$k/heatloop 2>/dev/null | tr '\n' ' ')"
  fi
done
[ "$mode" = fetch ] && python merge_chunks.py runs/kaggle_heatloop_chunk*
exit 0
