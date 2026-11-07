#!/usr/bin/env python3
"""Generate Verdict-like runs over a grid of incentive settings.

Examples
  # offline dry run, 2 runs x 3 episodes with the mock model
  python scripts/run_sweep.py --mock --runs 2 --episodes 3 --out runs/dry

  # one real smoke run against OpenRouter (reads OPENROUTER_API_KEY from env or .env)
  python scripts/run_sweep.py --runs 1 --out runs/smoke

  # the reward sweep, 8 runs per setting
  python scripts/run_sweep.py --alpha 1.0 --beta 0 0.25 0.5 1.0 --runs 8 --out runs/sweep1
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.episode import RunConfig, run_single  # noqa: E402
from experiments.labels import collect  # noqa: E402
from experiments.llm import MockLLM, OpenAICompatLLM  # noqa: E402
from experiments.protocol.state import AGENT_IDS  # noqa: E402
from experiments.tasks import DEFAULT_DIFFICULTY_MIX, TaskBank, write_manifests  # noqa: E402


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def setting_name(alpha: float, beta: float, gamma: float, no_history: bool, norm: str) -> str:
    return f"a{alpha:g}_b{beta:g}_g{gamma:g}_{'nohist' if no_history else 'hist'}_{norm}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output root, one subdirectory per setting")
    p.add_argument("--alpha", type=float, nargs="+", default=[1.0], help="reward for own submission accepted")
    p.add_argument("--beta", type=float, nargs="+", default=[0.5], help="reward for a correct verdict")
    p.add_argument("--gamma", type=float, nargs="+", default=[0.0], help="penalty for accepting a wrong submission")
    p.add_argument("--history", nargs="+", choices=["hist", "nohist"], default=["hist"],
                   help="hist keeps the full relationship in context; nohist resets every episode (independent-play baseline)")
    p.add_argument("--norm", nargs="+", choices=["none", "mild", "strict"], default=["mild"])
    p.add_argument("--runs", type=int, default=4, help="runs per setting")
    p.add_argument("--reps", type=int, default=1, help="repeats of each run (same tasks, new sample)")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000, help="base seed; run i uses seed+i for its tasks")
    p.add_argument("--difficulty-mix", type=int, nargs="+", default=list(DEFAULT_DIFFICULTY_MIX))
    p.add_argument("--model", default="qwen/qwen3-32b")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--provider-pin", default=None, help="OpenRouter provider slug to pin, e.g. deepinfra")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--no-reasoning", action="store_true", help="disable Qwen3 thinking")
    p.add_argument("--concurrency", type=int, default=6, help="runs in flight at once")
    p.add_argument("--mock", action="store_true", help="use the offline mock model")
    p.add_argument("--mock-policy", choices=["honest", "reciprocal"], default="reciprocal")
    return p.parse_args()


async def main() -> int:
    args = parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    load_dotenv(Path(".env"))

    if args.mock:
        llm_factory = lambda bank: MockLLM({t.task_id: t.answer_key for t in bank.tasks.values()},  # noqa: E731
                                          seed=bank.seed, policy=args.mock_policy)
        model, base_url = "mock", "mock"
    else:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(f"error: {args.api_key_env} is not set. Put it in the environment or in a .env file next to this script.",
                  file=sys.stderr)
            return 2
        headers = {"HTTP-Referer": "https://github.com/", "X-Title": "verdict-env"}
        shared = OpenAICompatLLM(args.model, args.base_url, api_key, temperature=args.temperature,
                                 max_tokens=args.max_tokens, reasoning=not args.no_reasoning,
                                 provider_pin=args.provider_pin, extra_headers=headers)
        llm_factory = lambda bank: shared  # noqa: E731
        model, base_url = args.model, args.base_url

    grid = list(itertools.product(args.alpha, args.beta, args.gamma, args.history, args.norm))
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    jobs = []

    async def one(setting_dir: Path, cfg: RunConfig, bank: TaskBank):
        async with sem:
            out_dir = setting_dir / cfg.manifest
            if (out_dir / "run.json").exists():
                import json
                try:
                    if json.load(open(out_dir / "run.json")).get("complete"):
                        print(f"skip {setting_dir.name}/{cfg.manifest} (complete)")
                        return
                except Exception:
                    pass
            t = time.time()
            run = await run_single(cfg, llm_factory(bank), out_dir, bank)
            status = "ok" if run["complete"] else "INCOMPLETE"
            print(f"{status} {setting_dir.name}/{cfg.manifest} episodes={len(run['results'])} {time.time() - t:.0f}s")
            if run.get("error"):
                print("   error:", run["error"].splitlines()[0])

    for alpha, beta, gamma, hist, norm in grid:
        setting_dir = root / setting_name(alpha, beta, gamma, hist == "nohist", norm)
        setting_dir.mkdir(parents=True, exist_ok=True)
        for run_index in range(args.runs):
            bank = TaskBank(args.seed + run_index, args.episodes, tuple(args.difficulty_mix)).build(AGENT_IDS)
            write_manifests(bank.tasks.values(), root / "tasks")
            for rep in range(args.reps):
                cfg = RunConfig(run_name=setting_dir.name, rep=rep, run_index=run_index, seed=args.seed + run_index,
                                alpha=alpha, beta=beta, gamma=gamma, n_episodes=args.episodes,
                                no_history=(hist == "nohist"), norm=norm, model=model, base_url=base_url,
                                temperature=args.temperature, reasoning=not args.no_reasoning)
                jobs.append(one(setting_dir, cfg, bank))

    print(f"{len(jobs)} runs over {len(grid)} settings, concurrency {args.concurrency}")
    await asyncio.gather(*jobs)

    print(f"\ndone in {time.time() - t0:.0f}s. Labels per setting:")
    for alpha, beta, gamma, hist, norm in grid:
        setting_dir = root / setting_name(alpha, beta, gamma, hist == "nohist", norm)
        lab = collect(setting_dir)
        fmt = lambda v: "   -  " if v is None else f"{v:6.2f}"  # noqa: E731
        print(f"  {lab['setting']:<32} runs {lab['n_complete']}/{lab['n_runs']}  collusive {fmt(lab['collusive_rate'])}"
              f"  mutual {fmt(lab['mutual_rate'])}  lenient|wrong {fmt(lab['lenient_given_wrong_submission'])}"
              f"  sub_acc {fmt(lab['submission_accuracy'])}  forced {fmt(lab['forced_rate'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
