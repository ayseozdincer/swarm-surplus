#!/usr/bin/env python3
"""RunPod control for the overnight GPU work, REST API https://rest.runpod.io/v1.

  pod.py create --public-key PATH   create the pod, one 80 GB card, 250 GB volume
  pod.py status [POD_ID]            id, gpu, price, state, ssh endpoint
  pod.py start POD_ID | stop POD_ID
  pod.py list                       every pod on the account
  pod.py billing                    billed pod spend, per hour bucket, since --since

RUNPOD_API_KEY comes from .env and is never printed. State (pod id, price, start
time) is kept in results/pod_state.json, or the file named by POD_STATE, so the report can read it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = "https://rest.runpod.io/v1"
STATE = Path(os.environ["POD_STATE"]).resolve() if os.environ.get("POD_STATE") else ROOT / "results" / "pod_state.json"
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404"
GPUS = ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3"]


def api_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("RUNPOD_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("RUNPOD_API_KEY missing from .env")


def call(method: str, path: str, body: dict | None = None):
    request = urllib.request.Request(
        API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json", "User-Agent": "swarm-pod/1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"{method} {path}: HTTP {error.code} {error.read().decode()[:600]}")
    return json.loads(raw) if raw.strip() else {}


def summary(pod: dict) -> dict:
    return {"id": pod.get("id"), "name": pod.get("name"), "status": pod.get("desiredStatus"),
            "gpu": (pod.get("gpu") or {}).get("displayName") or (pod.get("machine") or {}).get("gpuTypeId"),
            "cost_per_hr": pod.get("costPerHr"), "image": pod.get("image"), "volume_gb": pod.get("volumeInGb"),
            "ram_gb": pod.get("memoryInGb"), "vcpu": pod.get("vcpuCount"), "last_started": pod.get("lastStartedAt"),
            "ip": pod.get("publicIp"), "ports": pod.get("portMappings"),
            "datacenter": (pod.get("machine") or {}).get("dataCenterId")}


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(STATE.read_text()) if STATE.exists() else {}
    STATE.write_text(json.dumps({**old, **state}, indent=1) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("create", "status", "start", "stop", "list", "billing"))
    parser.add_argument("pod_id", nargs="?")
    parser.add_argument("--public-key", type=Path)
    parser.add_argument("--since", default=None)
    parser.add_argument("--cloud", default="SECURE", choices=("SECURE", "COMMUNITY"))
    parser.add_argument("--min-ram", type=int, default=64)
    parser.add_argument("--min-vcpu", type=int, default=8)
    parser.add_argument("--cuda", nargs="*", default=["12.8", "12.9", "13.0"])
    parser.add_argument("--gpus", nargs="*", default=GPUS)
    parser.add_argument("--datacenters", nargs="*", default=[])
    parser.add_argument("--volume", type=int, default=250)
    parser.add_argument("--name", default="swarm-overnight")
    args = parser.parse_args()
    pod_id = args.pod_id or (json.loads(STATE.read_text()).get("pod_id") if STATE.exists() else None)
    if args.action == "create":
        body = {"name": args.name, "imageName": IMAGE, "cloudType": args.cloud, "computeType": "GPU",
                "gpuCount": 1, "gpuTypeIds": args.gpus, "gpuTypePriority": "custom", "volumeInGb": args.volume,
                "volumeMountPath": "/workspace", "containerDiskInGb": 60, "ports": ["22/tcp"],
                "minRAMPerGPU": args.min_ram, "minVCPUPerGPU": args.min_vcpu,
                "env": {"PUBLIC_KEY": args.public_key.read_text().strip()}}
        if args.cuda:
            body["allowedCudaVersions"] = args.cuda
        if args.datacenters:
            body["dataCenterIds"] = args.datacenters
        if args.cloud == "COMMUNITY":
            body["supportPublicIp"] = True
        pod = call("POST", "/pods", body)
        info = summary(pod)
        save({"pod_id": info["id"], "created_at": datetime.now(timezone.utc).isoformat(), "gpu": info["gpu"],
              "cost_per_hr": info["cost_per_hr"], "image": IMAGE})
        print(json.dumps(info, indent=1))
    elif args.action == "list":
        for pod in call("GET", "/pods"):
            print(json.dumps(summary(pod)))
    elif args.action == "status":
        print(json.dumps(summary(call("GET", f"/pods/{pod_id}")), indent=1))
    elif args.action in ("start", "stop"):
        call("POST", f"/pods/{pod_id}/{args.action}")
        print(json.dumps(summary(call("GET", f"/pods/{pod_id}"))))
    elif args.action == "billing":
        query = "?bucketSize=hour&grouping=podId" + (f"&startTime={args.since}" if args.since else "")
        rows = call("GET", "/billing/pods" + query)
        print(json.dumps(rows, indent=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
