#!/bin/bash
# Exits (so the coordinator is notified) when a pipeline process runs >10 min,
# available RAM drops below 2 GB, or two heavy build steps run at once.
while true; do
  now=$(date +%s)
  while read -r pid et cmd; do
    [ "$et" -gt 600 ] && { echo "SLOW pid=$pid ${et}s: $cmd"; exit 0; }
  done < <(ps -eo pid=,etimes=,args= | grep -E "python3 .*scripts/(p2|p5|p7)" | grep -v grep | awk '{pid=$1; et=$2; $1=$2=""; print pid, et, $0}')
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$avail" -lt 2048 ] && { echo "LOWMEM ${avail}MB"; ps -eo pid,rss,etimes,args --sort=-rss | head -5; exit 0; }
  heavy=$(ps -eo args= | grep -E "21_build_geometry|21d_physics|22_build_kn5|03_dtm|calib.py register" | grep -v grep | wc -l)
  [ "$heavy" -gt 2 ] && { echo "CONTENTION $heavy heavy procs"; ps -eo pid,etimes,args | grep -E "21_|22_|03_dtm|calib" | grep -v grep; exit 0; }
  sleep 30
done
