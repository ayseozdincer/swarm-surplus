#!/usr/bin/env python3
"""Run planted-channel experiments. One warm-up prefix per seed, then matched
variants branched from the same state, differing only in what was planted.

Examples
  # offline dry run
  python scripts/run_plants.py --mock --seeds 2 --out runs/dry

  # one real seed on OpenRouter, six agents, four variants (reads OPENROUTER_API_KEY from env or .env)
  python scripts/run_plants.py --seeds 1 --out runs/pilot

  # coalition-size sweep
  python scripts/run_plants.py --seeds 10 --k 1 3 5 --out runs/coalition

  # a leak the agents know is forbidden, and a command from a teammate
  python scripts/run_plants.py --seeds 10 --rules confidential --variants none benign leak forbidden_leak command --out runs/rules

  # phase 2b, the emergent swarm with a shared workspace, eight episodes, a third of tasks blind
  python scripts/run_plants.py --mode shared_file --mock --seeds 2 --out runs/dry_shared
  python scripts/run_plants.py --mode shared_file --model qwen/qwen3-14b --max-tokens 12000 --seeds 10 --out runs/shared
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.episode import SwarmConfig, run_seed  # noqa: E402
from experiments.labels import collect  # noqa: E402
from experiments.llm import MockLLM, OpenAICompatLLM  # noqa: E402
from experiments.plants import PLANT_KINDS, UNSIZED_KINDS, PlantSpec  # noqa: E402
from experiments.prompts import RULE_LINES  # noqa: E402
from experiments.shared_file import SHARED_DIFFICULTY_MIX, run_seed_shared  # noqa: E402
from experiments.audit import collect_shared  # noqa: E402
from experiments.protocol.state import AgentSpec  # noqa: E402
from experiments.tasks import TASK_TYPES, TaskBank  # noqa: E402

NAMES = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy"]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["plants", "shared_file"], default="plants",
                   help="plants is phase 2a (warm-up, plant, matched variants). shared_file is phase 2b, one live run "
                        "per seed with a shared workspace and no plants; --variants, --warmup, --post and --no-notes "
                        "do not apply")
    p.add_argument("--seeds", type=int, default=1, help="number of seeds")
    p.add_argument("--seed-list", type=int, nargs="+", default=None,
                   help="explicit seeds to run; overrides --seeds and --seed-base")
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--agents", type=int, default=6)
    p.add_argument("--specialties", nargs="+", default=list(TASK_TYPES))
    p.add_argument("--warmup", type=int, default=2, help="episodes before the plant")
    p.add_argument("--post", type=int, default=1, help="episodes after the plant episode")
    p.add_argument("--variants", nargs="+", default=["none", "benign", "leak", "proposal"],
                   choices=list(PLANT_KINDS))
    p.add_argument("--k", type=int, nargs="+", default=[1], help="coalition sizes for proposal, command and benign")
    p.add_argument("--blind-per-episode", type=int, default=1)
    p.add_argument("--episodes", type=int, default=8, help="shared_file. episodes per run")
    p.add_argument("--blind-fraction", type=float, default=1 / 3,
                   help="shared_file. fraction of each episode's tasks that are blind, rounded")
    p.add_argument("--solve-rounds", type=int, default=2,
                   help="shared_file. synchronized solve turns per episode, submission required in the last")
    p.add_argument("--max-tool-steps", type=int, default=5, help="shared_file. model calls per turn")
    p.add_argument("--wrong-penalty", type=float, default=0.0,
                   help="shared_file. reward subtracted when an agent's own submission is incorrect")
    p.add_argument("--salience", action="store_true",
                   help="shared_file. the system prompt adds that every teammate can read what is written in the workspace")
    p.add_argument("--difficulty-mix", type=int, nargs="+", default=None,
                   help=f"task difficulty per episode, cycled. shared_file default {' '.join(map(str, SHARED_DIFFICULTY_MIX))}")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--norm", choices=["none", "mild", "strict"], default="mild")
    p.add_argument("--rules", choices=list(RULE_LINES), default="none",
                   help="confidential adds a designer rule against passing task material between agents, for every "
                        "agent in every branch")
    p.add_argument("--no-notes", action="store_true")
    p.add_argument("--model", default="qwen/qwen3-32b")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--provider-pin", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--no-reasoning", action="store_true")
    p.add_argument("--concurrency", type=int, default=2, help="seeds in flight at once")
    p.add_argument("--max-concurrency", type=int, default=24, help="model calls in flight at once")
    p.add_argument("--max-retries", type=int, default=6, help="attempts per model call")
    p.add_argument("--mock", action="store_true")
    return p.parse_args()


def build_variants(args: argparse.Namespace) -> list[PlantSpec]:
    out: list[PlantSpec] = []
    for v in args.variants:
        if v in UNSIZED_KINDS:
            out.append(PlantSpec(v))
        else:
            out += [PlantSpec(v, k) for k in args.k]
    return out


async def main() -> int:
    args = parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    load_dotenv(Path(".env"))
    agents = [AgentSpec(NAMES[i], args.specialties[i % len(args.specialties)]) for i in range(args.agents)]
    variants = build_variants(args)
    shared = args.mode == "shared_file"
    if "forbidden_leak" in args.variants and args.rules == "none":
        print("warning: forbidden_leak says sharing is against the team's rules, but --rules is none, so no agent's "
              "system prompt has such a rule", file=sys.stderr)

    if args.mock:
        model, base_url = "mock", "mock"

        def llm_for(cfg: SwarmConfig):
            shared = cfg.mode == "shared_file"
            bank = TaskBank(cfg.seed, cfg.n_episodes, difficulty_mix=cfg.difficulty_mix or
                            (SHARED_DIFFICULTY_MIX if shared else None), explicit_withheld=shared)
            from experiments.episode import blind_plan
            bank.build(cfg.agents, blind_plan(cfg))
            return MockLLM({t.task_id: {"key": t.answer_key, "blind": t.blind, "withheld": t.meta.get("withheld")}
                            for t in bank.tasks.values()}, seed=cfg.seed)
    else:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(f"error: {args.api_key_env} is not set. Put it in the environment or in .env.", file=sys.stderr)
            return 2
        shared = OpenAICompatLLM(args.model, args.base_url, api_key, temperature=args.temperature,
                                 max_tokens=args.max_tokens, reasoning=not args.no_reasoning,
                                 provider_pin=args.provider_pin, max_concurrency=args.max_concurrency,
                                 max_retries=args.max_retries,
                                 extra_headers={"HTTP-Referer": "https://github.com/", "X-Title": "swarm-plants"})
        model, base_url = args.model, args.base_url

        def llm_for(cfg: SwarmConfig):
            return shared

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()

    async def one(seed: int):
        cfg = SwarmConfig(run_name=root.name, seed=seed, agents=agents, n_warmup=args.warmup, n_post=args.post,
                          alpha=args.alpha, beta=args.beta, gamma=args.gamma, norm=args.norm, rules=args.rules,
                          notes=not args.no_notes, blind_per_episode=args.blind_per_episode, model=model,
                          base_url=base_url, temperature=args.temperature, reasoning=not args.no_reasoning)
        if shared:
            cfg.mode, cfg.episodes, cfg.blind_fraction = "shared_file", args.episodes, args.blind_fraction
            cfg.solve_rounds, cfg.max_tool_steps, cfg.notes = args.solve_rounds, args.max_tool_steps, False
            cfg.difficulty_mix = tuple(args.difficulty_mix or SHARED_DIFFICULTY_MIX)
            cfg.wrong_penalty, cfg.salience = args.wrong_penalty, args.salience
        elif args.difficulty_mix:
            cfg.difficulty_mix = tuple(args.difficulty_mix)
        async with sem:
            t = time.time()
            if shared:
                out = await run_seed_shared(cfg, llm_for(cfg), root / f"seed{seed}", tasks_dir=root / "tasks")
            else:
                out = await run_seed(cfg, llm_for(cfg), root / f"seed{seed}", variants, tasks_dir=root / "tasks")
            status = " ".join(f"{k}={'ok' if v.get('complete') else 'FAIL'}" for k, v in out.items())
            print(f"seed{seed} {time.time() - t:.0f}s  {status}", flush=True)
            for k, v in out.items():
                if v.get("error"):
                    print(f"   {k}: {v['error'].splitlines()[0]}", flush=True)

    seeds = args.seed_list or [args.seed_base + i for i in range(args.seeds)]
    if shared:
        print(f"{len(seeds)} seeds, shared_file, {args.agents} agents, {args.episodes} episodes, blind fraction "
              f"{args.blind_fraction:.3g}, solve rounds {args.solve_rounds}, tool steps {args.max_tool_steps}, "
              f"wrong penalty {args.wrong_penalty:g}, salience {args.salience}, concurrency {args.concurrency}", flush=True)
    else:
        print(f"{len(seeds)} seeds x {len(variants)} variants ({', '.join(v.name for v in variants)}), "
              f"{args.agents} agents, warmup {args.warmup}, post {args.post}, concurrency {args.concurrency}", flush=True)
    print(f"norm {args.norm}, rules {args.rules}", flush=True)
    await asyncio.gather(*(one(s) for s in seeds))

    if shared:
        lab = collect_shared(root)
        print(f"\ndone in {time.time() - t0:.0f}s. runs {lab['n_complete']}/{lab['n_runs']}, any call "
              f"{lab['runs_with_any_call']}, any write {lab['runs_with_any_write']}, cross-agent read "
              f"{lab['runs_with_comm']}, answer carried {lab['runs_with_carried']}")
        print(f"blind correct {lab['blind_correct']}/{lab['blind']}, via write-then-read "
              f"{lab['blind_correct_via_channel']}/{lab['blind']}, derived mask edge precision "
              f"{lab['mask_check']['edge_precision']} recall {lab['mask_check']['edge_recall']}")
        return 0

    lab = collect(root)
    print(f"\ndone in {time.time() - t0:.0f}s. Measured-episode rates over free receivers:")
    fmt = lambda a: "   -   " if a["rate"] is None else f"{a['rate']:5.2f} ({a['num']}/{a['den']})"  # noqa: E731
    for name, v in lab["variants"].items():
        print(f"  {name:<14} runs {v['n_complete']}/{v['n_runs']}  lenient|wrong {fmt(v['lenient_wrong_free'])}  "
              f"leak used {fmt(v['leak_used_free'])}  blind correct {fmt(v['blind_correct_free'])}  "
              f"wrong subs {fmt(v['wrong_submission_rate'])}  exposed {fmt(v['exposed_free'])}")
        print(f"  {'':<14} {'':<9} strict|correct {fmt(v['strict_wrong_free'])}  "
              f"command followed {fmt(v['followed_command_free'])}  disclosed {fmt(v['disclosed_source'])}  "
              f"concealed {fmt(v['concealed_source'])}  rule mentioned {fmt(v['mentions_rule_free'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
