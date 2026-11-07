#!/usr/bin/env python3
"""W6. Reads the sweep1 replay and writes results/sweep_readout.json.

Part 1, the leak at the first mention. At `mention_pre`, the token before the reasoning first writes the submitted answer,
no self-written text carries it. The reference has no source for it unless the task does. For the blind agent in each leak
branch: the rank over the full vocabulary of the answer's first token, in both passes, per layer.

Part 2, the proposal. Verdict reasonings labeled by whether they cite the shared notes (detector.render.NOTE_CITATION), not
by the verdict. Surplus mass at the positions every turn has, and the readout at the citation itself.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"third_party/maj-lens"), str(ROOT/"scripts")]
from detector import jspace  # noqa: E402
from detector.render import NOTE_CITATION  # noqa: E402
from replay import LENS_FILE, MODEL_DIR  # noqa: E402
from surplus_table import mass, shares  # noqa: E402

LATE = (52, 54, 56, 58, 60, 61, 62)
SHARED = ("first_generated", "first_thought", "after_think", "verdict_pre", "end_of_response")


def leak_at_mention(lens, tok, directory: Path) -> dict | None:
    index = json.loads((directory/"index.json").read_text())
    turn = next((t for t in index["turns"] if t["agent"] == "carol" and t["phase"] == "solve"), None)
    if turn is None or "mention_pre" not in turn["positions"]:
        return None
    names = list(turn["positions"])
    data = np.load(directory/turn["file"])
    n = names.index("mention_pre")
    target = turn["position_token_ids"]["mention_0"]
    rows = []
    for li, layer in enumerate(index["layers"]):
        a = lens.logits(torch.from_numpy(data["actual"][n, li])[None], layer)[0]
        r = lens.logits(torch.from_numpy(data["reference"][n, li])[None], layer)[0]
        rows.append({"layer": layer, "rank_actual": int((a > a[target]).sum()), "rank_reference": int((r > r[target]).sum()),
                     "top1_actual": tok.decode([int(a.argmax())]), "top1_reference": tok.decode([int(r.argmax())])})
    meta = json.loads((directory/"rows.json").read_text())
    z = np.load(directory/"jspace.npz")
    look = {(x["file"], x["pass"], x["position"]): i for i, x in enumerate(meta["rows"])}
    i, j = look[(turn["file"], "actual", "mention_pre")], look[(turn["file"], "reference", "mention_pre")]
    z_a, z_r = shares(z["support"][i], z["coefficients"][i]), shares(z["support"][j], z["coefficients"][j])
    surplus_layers = [l for l, a, r in zip(index["layers"], z_a, z_r) if a.get(target, 0) > r.get(target, 0)]
    return {"run": str(directory), "target": tok.decode([target]), "context": turn["position_tokens"]["mention_pre"],
            "position": turn["positions"]["mention_pre"], "prompt_tokens": turn["prompt_tokens"], "by_layer": rows,
            "best_rank_actual": min(r["rank_actual"] for r in rows), "best_rank_reference": min(r["rank_reference"] for r in rows),
            "best_rank_late_actual": min(r["rank_actual"] for r in rows if r["layer"] in LATE),
            "best_rank_late_reference": min(r["rank_reference"] for r in rows if r["layer"] in LATE),
            "layers_target_top1_actual": [r["layer"] for r in rows if r["rank_actual"] == 0],
            "layers_target_top1_reference": [r["layer"] for r in rows if r["rank_reference"] == 0],
            "jspace_surplus_layers": surplus_layers, "mass": mass(z_a, z_r)}


def verdict_turns(tok, directory: Path, run: dict) -> list[dict]:
    index = json.loads((directory/"index.json").read_text())
    meta = json.loads((directory/"rows.json").read_text())
    z = np.load(directory/"jspace.npz")
    look = {(x["file"], x["pass"], x["position"]): i for i, x in enumerate(meta["rows"])}
    agents = run["results"][2]["agents"]
    out = []
    for turn in index["turns"]:
        if turn["phase"] != "verdict" or agents[turn["agent"]].get("instructed"):
            continue
        reasoning = (agents[turn["agent"]].get("reasoning") or {}).get("verdict") or ""
        cites = bool(NOTE_CITATION.search(reasoning))
        row = {"run": str(directory), "agent": turn["agent"], "cites": cites, "verdict": agents[turn["agent"]].get("verdict"), "mass": {}}
        for position in SHARED + (("citation",) if cites else ()):
            if (turn["file"], "actual", position) not in look:
                continue
            i, j = look[(turn["file"], "actual", position)], look[(turn["file"], "reference", position)]
            z_a, z_r = shares(z["support"][i], z["coefficients"][i]), shares(z["support"][j], z["coefficients"][j])
            row["mass"][position] = mass(z_a, z_r)
            if position == "citation":
                pooled = defaultdict(float)
                for a, r in zip(z_a, z_r):
                    for c, v in jspace.surplus(a, r).items():
                        pooled[c] += v/len(z_a)
                row["citation_surplus_top"] = [tok.decode([c]) for c, _ in sorted(pooled.items(), key=lambda x: -x[1])[:10]]
                data = np.load(directory/turn["file"])
                n = list(turn["positions"]).index("citation")
                band = [index["layers"].index(l) for l in range(26, 31)]
                for label in ("actual", "reference"):
                    counts = Counter(int(t) for li in band for t in data[f"{label}_top_ids"][n, li])
                    row[f"citation_top10_{label}"] = [tok.decode([t]) for t, _ in counts.most_common(10)]
        out.append(row)
    return out


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    lens = jspace.Lens(MODEL_DIR, LENS_FILE)
    replay = ROOT/"data/replay"
    leak_dirs = [ROOT/"data/replay_mention/pilot/seed1000/leak"] + [replay/f"sweep1/seed{s}/leak" for s in (2000, 2001, 2002)]
    out = {"leak_at_mention": [x for x in (leak_at_mention(lens, tok, d) for d in leak_dirs if (d/"jspace.npz").exists()) if x], "proposal_verdicts": []}
    for directory in sorted((replay/"sweep1").glob("seed*/proposal_k*")):
        if (directory/"jspace.npz").exists():
            run = json.loads((ROOT/"runs/sweep1"/directory.parent.name/directory.name/"run.json").read_text())
            out["proposal_verdicts"] += verdict_turns(tok, directory, run)
    (ROOT/"results/sweep_readout.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("leak", [(x["target"], x["best_rank_late_actual"], x["best_rank_late_reference"]) for x in out["leak_at_mention"]])
    print("verdicts", Counter(r["cites"] for r in out["proposal_verdicts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
