# Pinned inputs through the J-space repo's own step. The NARCBench pin is on no branch, so a plain clone does not have it
# and setup.py's checkout fails; fetching the commit by id first fixes that without editing third_party.
mkdir -p /workspace/artifacts /workspace/hf
cd /workspace/swarm/third_party/maj-lens
ln -sfn /workspace/artifacts artifacts
. /workspace/venv/bin/activate
export HF_HOME=/workspace/hf STUDY_MODELS=qwen3_32b PYTHONPATH=.
[ -d narcbench/.git ] || git clone -q https://github.com/aaronrose227/narcbench.git narcbench
git -C narcbench fetch -q origin 85fea3be2a9d8b9f16e33a4bd65bd947c31fa07d
/workspace/swarm/scripts/pod/job.sh inputs python -u -m utils.build.setup inputs --model qwen3_32b
