#!/bin/bash
# usage: batch.sh session...
echo $$ > /sys/fs/cgroup/cpuset/perf/cgroup.procs
export FT_LOG_DIR=${FT_LOG_DIR:-/tmp/dflash_mainline_blackbox}
cd "$(dirname "$0")"
for s in "$@"; do
  echo "=== START $s $(date +%T)"
  timeout 5400 /home/nengneng/miniconda3/envs/freetoken-dev/bin/python run.py --session $s 2>&1 | grep -v "^\s*$"
  echo "=== END $s $(date +%T)"
done
echo "=== BATCH DONE"
