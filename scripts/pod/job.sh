#!/bin/bash
# Run a command on the pod in the background with its output in /workspace/logs/<name>.log,
# refreshing /workspace/busy while it runs so the idle guard leaves it alone.
#   job.sh <name> <command...>      ends with a line "exit <code>" in the log
name=$1; shift
mkdir -p /workspace/logs
log=/workspace/logs/$name.log
nohup setsid bash -c '
  ( while true; do touch /workspace/busy; sleep 60; done ) & beat=$!
  "$@"; code=$?
  kill $beat; touch /workspace/busy
  echo "exit $code"
' job "$@" > "$log" 2>&1 < /dev/null &
echo "started $name"
