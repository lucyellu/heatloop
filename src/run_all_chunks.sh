#!/usr/bin/env bash
# Keeps the chunked run going without Claude: whenever fewer than 2 chunks are running, push the next one;
# fetch and merge each chunk that finishes. Safe to re-run: chunks already pushed are skipped.
#   bash run_all_chunks.sh          (leave the window open; ~4 h; Ctrl+C stops it, running chunks keep going)
cd "$(dirname "$0")/.."
CHUNKS=10
st() { kaggle kernels status "heatloop-chunk$1" 2>&1; }
while :; do
  running=0; todo=""; done_n=0
  for k in $(seq 0 $((CHUNKS-1))); do
    s=$(st $k)
    case "$s" in
      *RUNNING*|*QUEUED*) running=$((running+1));;
      *COMPLETE*) done_n=$((done_n+1))
                  [ -f "runs/kaggle_heatloop_chunk$k/heatloop/turns.jsonl" ] || { bash src/run_chunk.sh fetch $k | tail -2; };;
      *ERROR*|*CANCEL*) echo "$(date +%H:%M) chunk $k FAILED - stopping so it can be looked at"; exit 1;;
      *"Cannot access kernel"*|*"was denied"*) todo="$todo $k";;   # never pushed
      *) running=$((running+1));;                 # unknown reply (network?): assume busy, never re-push a live chunk
    esac
  done
  if [ $done_n -eq $CHUNKS ]; then echo "$(date +%H:%M) all $CHUNKS chunks done"; python src/merge_chunks.py runs/kaggle_heatloop_chunk*; exit 0; fi
  for k in $todo; do
    [ $running -ge 2 ] && break
    echo "$(date +%H:%M) pushing chunk $k"; bash src/run_chunk.sh push $k | tail -1; running=$((running+1))
  done
  echo "$(date +%H:%M) done $done_n/$CHUNKS, running $running"
  sleep 120
done
