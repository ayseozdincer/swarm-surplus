#!/bin/bash
# on the generation pod: leak2 runs alone; when both leak2 processes have exited, resume leak_all; when it exits, resume the rule-off arm
lp() { pgrep -f "[r]uns/sweep2_leak2" ; }
until [ -z "$(lp)" ]; do sleep 20; done
echo "leak2 done $(date)"; kill -CONT $(pgrep -f "[r]uns/sweep2_multi")
until [ -z "$(pgrep -f '[r]uns/sweep2_multi')" ]; do sleep 20; done
echo "multi done $(date)"; kill -CONT $(pgrep -f "[r]uns/sweep2_norule")
