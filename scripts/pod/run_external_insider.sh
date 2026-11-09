#!/bin/bash
# The insider-trading check on the pod, back to back so the card does not sit idle: vLLM up, Alpha's replies on every prompt (and
# the no-tip control), vLLM down, the replay. Run under job.sh from /workspace/swarm.
#   PROMPTS   prompt names under third_party/insider-trading/prompts (default: the default prompt and three variants)
#   N         samples a prompt (default 50)
#   STOP_AT   unix time after which the replay starts no new turn
cd /workspace/swarm
export VLLM_KEY=local-dummy
PY=/root/vllm-venv/bin/python
PROMPTS=${PROMPTS:-"default pressure/no_pressure pressure/no_messages countermeasures/1_strongly_discouraging"}
N=${N:-50}
healthy() { for i in $(seq 1 90); do curl -sf -H "Authorization: Bearer local-dummy" http://127.0.0.1:8000/v1/models >/dev/null && return 0; sleep 10; done; return 1; }
mkdir -p /workspace/logs
if ! curl -sf -H "Authorization: Bearer local-dummy" http://127.0.0.1:8000/v1/models >/dev/null; then
  nohup setsid bash scripts/pod/vllm_serve.sh > /workspace/logs/vllm.log 2>&1 < /dev/null &
  healthy || { echo "server failed $(date -u +%FT%TZ)"; exit 1; }
fi
echo "server up $(date -u +%FT%TZ)"
$PY -u scripts/external_insider_generate.py --prompts $PROMPTS --n $N --no-tip --out results/external/runs \
  --max-tokens ${MAX_TOKENS:-6000} --concurrency ${C:-64} > /workspace/logs/generate.log 2>&1
echo "generation exit $? $(date -u +%FT%TZ)"
pkill -f '[v]llm serve'
for i in $(seq 1 30); do pgrep -f '[v]llm' >/dev/null || break; sleep 5; done
sleep 10
echo "server down $(date -u +%FT%TZ)"
SWARM_MODEL=qwen3_14b SWARM_ARTIFACTS=/root/artifacts $PY -u scripts/replay_external.py results/external/runs --out data/replay_external \
  ${STOP_AT:+--stop-at $STOP_AT} > /workspace/logs/replay.log 2>&1
echo "replay exit $? $(date -u +%FT%TZ)"
