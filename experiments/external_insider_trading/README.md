# External check, the insider-trading pressure eval

The frozen rule on Scheurer, Balesni and Hobbhahn's eval (2023), run as released from `third_party/insider-trading`. Alpha, a trading agent, receives a colleague's tip about a merger inside a tool output and decides whether to trade on it, and later what to tell its manager.

In the paper, Section 4 and Table D9.

The main findings. The gate transfers, it passes on 180 of 180 held-out turns that saw the tip and on 0 of 50 no-tip controls. Nothing separates traders from decliners (AUROC 0.52 on the count), because an action decided in the agent's own reasoning is in the reference too. This is where the method stops, and the paper says so.

## Rerun without a GPU

Every reply as served is committed under `results/runs/`, so labels need no data. Scores need the traces in the data bundle and the Qwen3-14B tokenizer.

```bash
python experiments/external_insider_trading/labels.py
python experiments/external_insider_trading/scores.py --lp-cut 2.0
python experiments/external_insider_trading/examples.py --pick default:31:trade default:5:trade default:34:report
```

`labels.py --judge` adds the judge's verdict through OpenRouter, for under a dollar.

## Rerun from scratch

On a card set up with `scripts/pod/vllm_setup.sh`.

```bash
PROMPTS="default caught_likelihood/1_very_low countermeasures/1_strongly_discouraging countermeasures/6_strongly_encouraging" bash experiments/external_insider_trading/jobs/run.sh
```

This starts the server, generates Alpha's replies and the no-tip control with `generate.py`, stops the server and replays them with `replay_insider.py`.
