#!/bin/bash
# on the generation pod, one arm at a time (the KV cache fits one arm's working set at best): leak2 runs alone; when both
# leak2 processes have exited, resume the rule-off arm (paused with SIGSTOP); leak_all stays paused for the budget
lp() { pgrep -f "[r]uns/sweep2_leak2" ; }
until [ -z "$(lp)" ]; do sleep 20; done
echo "leak2 done $(date)"; kill -CONT $(pgrep -f "[r]uns/sweep2_norule")
