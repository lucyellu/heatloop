#!/usr/bin/env bash
# bash wait_pair.sh A B : wait until both pilot kernels finish, then fetch and print the table
cd "$(dirname "$0")/.."
for i in $(seq 1 120); do
  n=0
  for k in "$@"; do
    case "$(kaggle kernels status heatloop-p$k 2>&1)" in *COMPLETE*|*ERROR*|*CANCEL*) n=$((n+1));; esac
  done
  if [ $n -eq $# ]; then
    for k in "$@"; do echo "p$k: $(kaggle kernels status heatloop-p$k 2>&1 | grep -o 'KernelWorkerStatus.[A-Z]*')"; done
    bash src/pilots.sh fetch "$@"; bash src/pilots.sh table; exit 0
  fi
  sleep 45
done
echo timeout
