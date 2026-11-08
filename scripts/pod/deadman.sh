#!/bin/bash
# Runs on the pod. Best effort only: tries to stop the pod if the driver session goes away.
# It relies on the key RunPod injects into the pod, which answered 403 to every read we tried,
# so whether it can stop the pod is unverified. The real guard is scripts/pod/watchdog.py on
# the driver machine. The account key from .env is never copied to the pod.
# The driver touches /workspace/heartbeat; a running job refreshes /workspace/busy.
#   IDLE_MIN  stop after this long with no heartbeat and no fresh busy file
#   MAX_HOURS stop after this long regardless, from the first start of this script
IDLE_MIN=${IDLE_MIN:-30}
MAX_HOURS=${MAX_HOURS:-15}
start=$(date +%s)
[ -f /workspace/deadman_start ] && start=$(cat /workspace/deadman_start) || echo "$start" > /workspace/deadman_start
touch /workspace/heartbeat
while true; do
  sleep 60
  now=$(date +%s)
  hb=$(stat -c %Y /workspace/heartbeat 2>/dev/null || echo 0)
  busy=$(stat -c %Y /workspace/busy 2>/dev/null || echo 0)
  last=$(( hb > busy ? hb : busy ))
  if [ $(( now - last )) -gt $(( IDLE_MIN * 60 )) ] || [ $(( now - start )) -gt $(( MAX_HOURS * 3600 )) ]; then
    echo "$(date -u +%FT%TZ) deadman stopping pod idle=$(( now - last ))s up=$(( now - start ))s" >> /workspace/deadman.log
    runpodctl stop pod "$RUNPOD_POD_ID" >> /workspace/deadman.log 2>&1
    sleep 120
  fi
done
