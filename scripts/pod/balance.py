#!/usr/bin/env python3
"""RunPod account balance and spend rate, GraphQL `myself`. RUNPOD_API_KEY comes from .env and is never printed.

  balance.py                        one JSON line, balance and USD per hour across the account
  balance.py --guard 8 --pod ID     every minute, stop the pod if the balance falls below 8 USD; exits when the pod stops
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pod import api_key, call  # noqa: E402


def balance() -> dict:
    body = json.dumps({"query": "query { myself { clientBalance currentSpendPerHr } }"}).encode()
    request = urllib.request.Request("https://api.runpod.io/graphql", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json", "User-Agent": "swarm-pod/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        me = json.loads(response.read())["data"]["myself"]
    return {"balance": me["clientBalance"], "spend_per_hr": me["currentSpendPerHr"], "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--guard", type=float, default=None, help="stop --pod when the balance falls below this")
    parser.add_argument("--pod", default=None)
    args = parser.parse_args()
    if args.guard is None:
        print(json.dumps(balance()))
        return 0
    while True:
        try:
            now = balance()
            print(json.dumps(now), flush=True)
            if float(now["balance"]) < args.guard:
                print(f"balance under {args.guard}, stopping {args.pod}", flush=True)
                call("POST", f"/pods/{args.pod}/stop")
                return 2
            if call("GET", f"/pods/{args.pod}").get("desiredStatus") != "RUNNING":
                print("pod is not running, guard exits", flush=True)
                return 0
        except (SystemExit, OSError, KeyError, TypeError, ValueError) as error:   # a network blip must not end the guard
            print(f"balance check failed, retrying next minute: {str(error)[:120]}", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
