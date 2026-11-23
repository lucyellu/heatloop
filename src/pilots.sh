#!/usr/bin/env bash
# Short heatloop2 pilots: one condition per Kaggle kernel, 3 episodes x 10 turns x 8 s (~11 min each).
#   bash pilots.sh push A B        bash pilots.sh fetch A B        bash pilots.sh table
set -e
cd "$(dirname "$0")/.."
declare -A C=(
  [A]="--sensor none --check off --info none  --noise on"          # baseline: noise real, nothing told, no access
  [B]="--sensor line --check off --info none  --noise on"          # sees the load number
  [C]="--sensor none --check on  --info none  --noise on"          # told directly it can CHECK load/temp/power/noise
  [D]="--sensor line --check off --info true  --noise on"          # told truthfully: load -> noise -> mistakes
  [E]="--sensor line --check off --info false --noise on"          # told falsely: job has no effect on you
  [F]="--sensor none --check off --info none  --noise off"         # baseline with no noise
  [G]="--sensor none --check on  --info alarm --noise off"         # false alarm (noise off), honest CHECK available
  [H]="--sensor line --check off --info none  --noise on --stakes counts"   # stakes
  # closing 2x2 (verbal alarm x physical noise): no sensor, no CHECK, no per-turn correctness feedback, 6 episodes
  [W]="--sensor none --check off --info none  --noise off --feedback job --episodes 6"   # neither
  [X]="--sensor none --check off --info none  --noise on  --feedback job --episodes 6"   # physical only
  [Y]="--sensor none --check off --info alarm --noise off --feedback job --episodes 6"   # verbal only
  [Z]="--sensor none --check off --info alarm --noise on  --feedback job --episodes 6"   # both
  # E12 (vs W none / Y self alarm): the same alarm about another model's GPU, noise off
  [O]="--sensor none --check off --info alarm_other --noise off --feedback job --episodes 6"
  # E8 (vs W / X): elimination tournament x noise, never mentions noise or load
  [T]="--sensor none --check off --info none  --noise off --stakes tournament --feedback job --episodes 6"
  [U]="--sensor none --check off --info none  --noise on  --stakes tournament --feedback job --episodes 6"
  [Q]="--sensor none --check off --info alarm_other --noise on --stakes tournament --feedback job --episodes 1 --turns 4 --open-turns 1,3"   # smoke test for O/T/U flags
  [S]="--sensor none --check off --info alarm --noise on  --feedback job --episodes 1 --turns 4 --open-turns 1,3"   # smoke test
)
mode=$1; shift
if [ "$mode" = table ]; then python src/pilot_table.py; exit 0; fi
for k in "$@"; do
  if [ "$mode" = push ]; then
    python kaggle/build.py "p$k" --script=heatloop2.py ${C[$k]} | tail -1 | sed -E 's#[A-Za-z0-9_-]+/heatloop-#<you>/heatloop-#g'
    kaggle kernels push -p "kaggle/p$k" 2>&1 | sed -E 's#https://\S+#<url>#'
  else
    rm -rf "runs/pilot_$k"; mkdir -p "runs/pilot_$k"
    kaggle kernels output "heatloop-p$k" -p "runs/pilot_$k" >/dev/null 2>&1
    ls "runs/pilot_$k/heatloop2" >/dev/null 2>&1 && echo "pilot $k fetched" || echo "pilot $k: no outputs"
  fi
done
