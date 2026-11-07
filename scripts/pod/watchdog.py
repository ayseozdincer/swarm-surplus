#!/usr/bin/env python3
"""Budget and idle guard for the pod, run on the driver machine (where .env lives).

Each minute it adds wall-clock time to the pod's running total while the pod is RUNNING and
stops the pod if the estimated cost reaches --cap, or if the GPU has been idle with no busy
job for --idle-min minutes. Exits after stopping the pod, or when the pod stops some other way.
It only runs while this machine is awake.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pod import STATE, call, save  # noqa: E402


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


def probe(ssh: str) -> tuple[int | None, int | None]:
    """(gpu utilisation percent, seconds since /workspace/busy was refreshed), None when unreachable."""
    command = ("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; "
               "echo $(( $(date +%s) - $(stat -c %Y /workspace/busy 2>/dev/null || echo 0) ))")
    try:
        out = subprocess.run([ssh, command], capture_output=True, text=True, timeout=60).stdout.split()
        return int(out[0]), int(out[1])
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", required=True, help="wrapper script that runs a command on the pod")
    parser.add_argument("--cap", type=float, default=28.5)
    parser.add_argument("--idle-min", type=float, default=20)
    args = parser.parse_args()
    state = json.loads(STATE.read_text())
    pod_id, seconds = state["pod_id"], float(state.get("running_seconds", 0))
    last, idle_since = time.time(), None
    while True:
        time.sleep(60)
        pod = call("GET", f"/pods/{pod_id}")
        now = time.time()
        rate = float(pod.get("costPerHr") or state["cost_per_hr"])
        running = pod.get("desiredStatus") == "RUNNING"
        if running:
            seconds += now - last
        last = now
        cost = seconds / 3600 * rate
        save({"running_seconds": round(seconds), "estimated_cost_usd": round(cost, 2), "watchdog_at": datetime.now(timezone.utc).isoformat()})
        if not running:
            log(f"pod {pod.get('desiredStatus')}, estimated cost {cost:.2f}; watchdog exits")
            return 0
        util, busy_age = probe(args.ssh)
        working = (util is not None and util >= 5) or (busy_age is not None and busy_age < 300)
        idle_since = None if working or util is None else (idle_since or now)
        if cost >= args.cap:
            log(f"budget cap reached at {cost:.2f}; stopping pod")
            call("POST", f"/pods/{pod_id}/stop")
            return 2
        if idle_since and now - idle_since >= args.idle_min * 60:
            log(f"idle {args.idle_min} min at {cost:.2f}; stopping pod")
            call("POST", f"/pods/{pod_id}/stop")
            return 3


if __name__ == "__main__":
    sys.exit(main())
