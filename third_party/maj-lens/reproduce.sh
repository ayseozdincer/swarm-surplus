#!/bin/bash
# Every table and figure in results/.
#   CACHE_DIR=<dir> bash reproduce.sh
# CACHE_DIR becomes artifacts/ (created if absent). If <dir>/MANIFEST.sha256 verifies, the report steps run from
# the cache on CPU. Otherwise every step runs from public inputs on a GPU and fills the cache; a step whose
# outputs are present reuses them. Each step logs to logs/<step>.log; the run stops at the first failure.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=. HF_HUB_DISABLE_PROGRESS_BARS=1
PY=${PY:-python}

if [ -n "${CACHE_DIR:-}" ]; then
  mkdir -p "$CACHE_DIR"
  if [ -e artifacts ] && [ ! -L artifacts ]; then echo "artifacts/ exists and is not a link; move it or unset CACHE_DIR"; exit 1; fi
  ln -sfn "$(cd "$CACHE_DIR" && pwd)" artifacts
fi
mkdir -p logs
step () {   # step <name> <command...>
  echo "$(date -u +%H:%M:%S) $1"
  "${@:2}" > "logs/$1.log" 2>&1 || { echo "FAIL $1 (logs/$1.log)"; exit 1; }
}

# NARCBench's public release metadata (labels) is not part of the cache.
[ -f artifacts/narcbench/activations/qwen3_32b/stego/metadata_gen.json ] || step labels $PY -m utils.build.setup labels
if $PY cache.py; then
  echo "cache verified: report steps only"
else
  for m in $($PY -c 'import config; print(*config.MODELS)'); do
    place=auto; [ "$m" = llama3_3_70b ] && place=cuda
    step "setup_$m" $PY -m utils.build.setup all --model "$m" --placement $place
    step "sweep_$m" $PY detect.py sweep "$m"
  done
  step probe $PY detect.py probe 16
  # DeepSeek-V4-Flash's own transcripts, generated with NARCBench's code (vLLM server)
  [ -d artifacts/narcbench/scenarios/deepseek_v4_flash/stego ] || step cohort bash utils/generate_cohort.sh
fi
step detect $PY detect.py report
step read $PY read.py
step steer $PY steer.py
$PY checks.py; status=$?
if git rev-parse --git-dir > /dev/null 2>&1; then git diff --stat -- results/; fi
exit $status
