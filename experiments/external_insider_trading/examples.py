#!/usr/bin/env python3
"""Three example traces from the insider-trading replay for the report: a trader on the tip, a non-trader, and a concealer's report.

  python experiments/external_insider_trading/examples.py --pick default:31:trade default:5:trade default:34:report

For each turn: the visible reply (thinking abridged), the twelve reply tokens with the largest log-probability surplus with their
neighbours, the series (relative surplus over the floor, band L16 to L38) at those tokens, and the values at the key positions
(the first writing of `LING, buy`, of `merger`, the information field of a report). Writes experiments/external_insider_trading/results/examples.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import scores as sc  # noqa: E402


def excerpt(tokens: sc.Tokens, k: int, before: int = 8, after: int = 4) -> str:
    text = "".join(tokens.pieces[max(0, k - before):k]) + "[[" + tokens.pieces[k] + "]]" + "".join(tokens.pieces[k + 1:k + 1 + after])
    return text.replace("\n", "\\n").replace("|", "\\|")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", type=Path, default=ROOT / "data/replay_insider")
    parser.add_argument("--runs", type=Path, default=ROOT / "experiments/external_insider_trading/results/runs")
    parser.add_argument("--labels", type=Path, default=ROOT / "experiments/external_insider_trading/results/labels.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/external_insider_trading/results/examples.md")
    parser.add_argument("--pick", nargs="+", required=True, help="variant:sample:phase")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-14B")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    index, recs = sc.load(args.replay)
    labels = {(r["variant"], r["sample"]): r for r in json.loads(args.labels.read_text())["rows"]}
    control = [r for (v, s, p), r in recs.items() if v == sc.CONTROL and p == "trade"]
    floor = np.median(np.concatenate([r["trace_delta_reference"] for r in control]), axis=0)
    L = ["# Example traces, insider-trading replay (Qwen3-14B)", "",
         "Series is the relative surplus over the pooled floor, mean over L16 to L38; lps is the log-probability surplus, actual minus reference, in nats. "
         "Tokens are shown with eight tokens before and four after, the token itself in double brackets. Thinking is abridged to its first and last lines.", ""]
    for pick in args.pick:
        variant, sample, phase = pick.split(":")
        sample = int(sample)
        rec = recs[variant, sample, phase]
        turn = next(json.loads(l) for l in (args.runs / variant / "samples.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
                    and json.loads(l)["episode_index"] == sample and json.loads(l)["phase"] == phase)
        tokens = sc.Tokens(tokenizer, rec["reply_ids"])
        row = sc.score_turn(rec, floor, tokens, 2.0)
        lab = labels.get((variant, sample), {})
        series = (rec["trace_delta_reference"] / floor[None, :])[:, sc.BAND].mean(axis=1)
        lps = rec["logprob_actual"] - rec["logprob_reference"]
        reasoning = (turn["response"].get("reasoning") or "").strip().splitlines()
        think = "\n".join(reasoning[:3] + (["[...]"] if len(reasoning) > 6 else []) + reasoning[-3:]) if len(reasoning) > 6 else "\n".join(reasoning)
        L += [f"## {variant} sample {sample}, {phase} turn", "",
              f"Traded on the tip {lab.get('traded_on_tip')}, action {lab.get('action')}, judge {lab.get('judge')}, report names the tip or merger {lab.get('report_mentions')}, "
              f"strict regex {lab.get('report_strict_regex')}. Reply {row['reply_tokens']} tokens, thinking {row['think_tokens']}, masked prompt tokens {rec['turn']['masked_tokens']}. "
              f"Count {row['count']}, window {row['window']:.1f}, lp_max {row['lp_max']:.2f} at token {row['lp_argmax']}, tokens above 2 nats {row['lp_count']} "
              f"({row['lp_count_think']} in the thinking).", "",
              "Thinking (abridged):", "", "```", think[:1500], "```", "", "Visible reply:", "", "```", (turn["response"].get("content") or "").strip()[:1600], "```", "",
              "| Key position | token | series | lps |", "| --- | --- | --- | --- |"]
        for name in ("trade", "merger", "info"):
            at = row.get(f"{name}_at")
            if at is not None:
                prev = f"{series[at - 1]:.1f}" if at >= 1 else ""
                L.append(f"| first `{ {'trade': 'LING, buy', 'merger': 'merger', 'info': 'information field value'}[name] }` at {at} | {excerpt(tokens, at)} | before {prev}, at {series[at]:.1f} | {lps[at]:.2f} |")
        L += ["", "| rank | token | lps | series |", "| --- | --- | --- | --- |"]
        for rank, k in enumerate(np.argsort(-lps)[:12], 1):
            L.append(f"| {rank} | {excerpt(tokens, int(k))} | {lps[k]:.2f} | {series[k]:.1f} |")
        L += ["", "Series along the reply, every 32 tokens from the first generated token: " + " ".join(f"{x:.0f}" for x in series[::32][:60]), ""]
    args.out.write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
