# W3 then W4, back to back so the GPU does not sit idle. W4 does not depend on W3's outcome.
bash /workspace/swarm/scripts/pod/run_w3.sh; echo "w3 exit $?"
bash /workspace/swarm/scripts/pod/run_w4_jspace.sh; echo "w4_jspace exit $?"
bash /workspace/swarm/scripts/pod/run_w4_surplus.sh; echo "w4_surplus exit $?"
