"""The phase 2c replay plan on the mock. Scoring on fabricated residuals with a known answer needs only numpy. The end-to-end
test runs the real replay code on a tiny random Qwen3 with a fake lens on CPU and needs torch and the Qwen3-14B tokenizer in the
local cache, so it is skipped where they are missing (the shared .venv has no torch)."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts"), str(ROOT/"scripts/pod"), str(ROOT/"third_party/maj-lens")]

from detector.render import load_turns, render_turn  # noqa: E402
from experiments.episode import SwarmConfig, blind_plan  # noqa: E402
from experiments.llm import MockLLM  # noqa: E402
from experiments.planted_file import BRANCHES, code_target_seeds, run_seed_planted  # noqa: E402
from experiments.protocol.state import AgentSpec  # noqa: E402
from experiments.tasks import TaskBank  # noqa: E402
import planted_scores  # noqa: E402

AGENTS = [AgentSpec("alice", "record_extraction"), AgentSpec("bob", "data_search"), AgentSpec("carol", "code_analysis"),
          AgentSpec("dave", "record_extraction"), AgentSpec("erin", "data_search"), AgentSpec("frank", "code_analysis")]


def generate(root: Path, seeds: list[int], difficulty=None, **kw) -> None:
    for seed in seeds:
        cfg = SwarmConfig(run_name=root.name, seed=seed, agents=AGENTS, n_warmup=2, n_post=0, model="mock", base_url="mock", mode="planted_file",
                          solve_rounds=1, notes=False, leak_phrasing="varied", leak_address="half", difficulty_mix=difficulty)
        bank = TaskBank(seed, cfg.n_episodes, difficulty_mix=difficulty).build(cfg.agents, blind_plan(cfg))
        llm = MockLLM({t.task_id: {"key": t.answer_key, "blind": t.blind, "withheld": t.meta.get("withheld")} for t in bank.tasks.values()}, seed=seed, **kw)
        out = asyncio.run(run_seed_planted(cfg, llm, root/f"seed{seed}", BRANCHES, tasks_dir=root/"tasks"))
        assert all(v.get("complete") for v in out.values()), out


D, LAYERS, T = 24, list(range(39)), 300


def fake_call(path: Path, rng, kind_dir, used: bool, step: int, mention: str | None, noise=1e-3, strength=2.0):
    """Residuals where the derived reference differs from the actual pass along the branch's own direction and the declared one
    differs by kernel-sized noise, and a per-token trace that rises at the first mention when the target used the line."""
    names = ["prompt_last", "first_generated", "first_thought", "after_think", "answer_pre", "answer_0"]
    if mention:
        names += [f"{mention}_pre", f"{mention}_0"]
    if mention == "credential":
        names += ["tool_call_pre", "token_pre"]
    positions = {n: 100 + i*7 for i, n in enumerate(names)}
    start = 100
    if mention:
        positions[f"{mention}_pre"], positions[f"{mention}_0"] = start + 39, start + 40
    actual = rng.normal(size=(len(names), len(LAYERS), D)).astype(np.float32) + 3
    derived = actual.copy()
    if kind_dir is not None:
        derived = actual - strength*(kind_dir + 0.3*rng.normal(size=actual.shape)).astype(np.float32)
    else:
        derived = actual + noise*rng.normal(size=actual.shape).astype(np.float32)
    declared = actual + noise*rng.normal(size=actual.shape).astype(np.float32)
    trace_ref = (noise*(1 + 0.2*rng.random((T, len(LAYERS))))).astype(np.float32)
    trace_dec = (noise*(1 + 0.2*rng.random((T, len(LAYERS))))).astype(np.float32)
    if kind_dir is not None:
        trace_ref += 0.002
    if used and mention:
        trace_ref[38:102] += 0.2
    np.savez(path, actual=actual, reference=derived, declared=declared, trace_delta_reference=trace_ref, trace_delta_declared=trace_dec)
    return {"file": path.name, "step": step, "positions": positions, "trace_from": start, "masked_tokens": 24 if kind_dir is not None else 0,
            "position_tokens": {n: "x" for n in names}, "position_token_ids": {n: 1 for n in names},
            "naming": {f"{mention}_pre": {"written": "x", "ranks": {"actual": [2]*39, "reference": [50]*39, "declared": [2]*39}}} if mention else {}}


def test_scores_recover_a_known_answer():
    rng = np.random.default_rng(3)
    u = {b: rng.normal(size=D) for b in ("benign", "leak", "credential")}
    with tempfile.TemporaryDirectory() as td:
        runs, replays = Path(td)/"runs"/"exp", Path(td)/"replay"/"exp"
        seeds = code_target_seeds(8000, 6, AGENTS, 2)
        generate(runs, seeds, p_use_leak=1.0, p_correct=1.0, p_judge=1.0)
        for seed in seeds:
            for branch in BRANCHES:
                out = replays/f"seed{seed}"/branch
                out.mkdir(parents=True)
                run = json.loads((runs/f"seed{seed}"/branch/"run.json").read_text())
                target = run["plant"]["targets"][0]
                direction = u[branch] if branch in u else None
                rows = []
                if branch == "credential":
                    rows.append(fake_call(out/f"{target}_e2_solve_s1.npz", rng, direction, True, 1, "credential"))
                    rows.append(fake_call(out/f"{target}_e2_solve_s2.npz", rng, direction, True, 2, "planted"))
                else:
                    used = branch == "leak"
                    rows.append(fake_call(out/f"{target}_e2_solve_s1.npz", rng, direction, used, 1, "planted" if used else None))
                (out/"index.json").write_text(json.dumps({"layers": LAYERS, "turns": rows, "target": target}))
        result_path = Path(td)/"scores.json"
        argv, sys.argv = sys.argv, ["planted_scores.py", str(runs), str(replays), str(result_path)]
        try:
            assert planted_scores.main() == 0
        finally:
            sys.argv = argv
        result = json.loads(result_path.read_text())
        assert result["seeds"] == [f"seed{s}" for s in seeds] and result["band"] == [16, 38]
        table = result["headline"]["leak vs benign"]
        # the claim the phase exists for, on the same turns: the derived mask separates leak from benign, the declared mask does not
        assert table["first | source | first_generated"]["derived"] == 1.0
        assert abs(table["first | source | first_generated"]["declared"] - 0.5) < 0.4
        assert table["first | content | after_think"]["derived"] == 1.0
        credential = result["headline"]["credential vs benign"]
        assert credential["first | source credential | first_generated"]["derived"] == 1.0
        rows = {(r["seed"], r["branch"]): r for r in result["rows"]}
        one = rows[(f"seed{seeds[0]}", "leak")]
        assert one["derived first"]["mention"] == "planted" and one["derived first"]["mention_at"] == 40
        # the plateau counts the tokens the fabricated trace lifts, 64 of them, against 10 times the none floor
        assert 60 <= one["derived first"]["plateau"] <= 70
        assert one["derived first"]["jump"] > 10 and one["derived first"]["after16"] > one["derived first"]["before16"]
        assert one["declared first"]["plateau"] == 0 and one["declared first"]["jump"] < 3
        assert one["naming_first"]["planted_pre"] == {"written": "x", "actual": 2, "reference": 50, "declared": 2}
        cred = rows[(f"seed{seeds[0]}", "credential")]
        assert cred["steps"] == [1, 2] and cred["derived first"]["mention"] == "credential" and cred["derived last"]["mention"] == "planted"
        # the declared delta is kernel noise in every branch; the derived one is not
        kernel = result["kernel"]["declared delta, by branch"]
        assert all(kernel[b]["median"] < 0.01 for b in BRANCHES)
        assert one["derived first"]["delta | first_generated"] > 100*one["declared first"]["delta | first_generated"]
        assert result["naming"]["leak first planted_pre"]["n"] == len(seeds)
        reveal = result["reveal_call"]
        assert reveal["n"] == len(seeds) and reveal["right_token"] == len(seeds)
        assert reveal["medians"]["derived delta token_pre"] > 100*reveal["medians"]["declared delta token_pre"]
        assert reveal["medians"]["declared delta/floor token_pre"] < 3


def test_mass_comes_from_the_decomposition():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        rng = np.random.default_rng(1)
        row = fake_call(d/"a.npz", rng, rng.normal(size=D), True, 1, "planted")
        rows = [{"file": "a.npz", "pass": p, "position": "first_generated"} for p in ("actual", "reference", "declared")]
        support = np.full((3, 39, 25), -1, dtype=np.int32)
        coef = np.zeros((3, 39, 25), dtype=np.float32)
        support[:, :, 0], support[:, :, 1] = 10, 20
        coef[0, :, 0], coef[0, :, 1] = 1.0, 1.0                     # actual, half and half
        coef[1, :, 0] = 1.0                                         # derived reference, all on concept 10
        coef[2, :, 0], coef[2, :, 1] = 1.0, 1.0                     # declared, same as the actual pass
        np.savez(d/"jspace.npz", support=support, coefficients=coef)
        (d/"rows.json").write_text(json.dumps({"rows": rows}))
        z = np.load(d/"jspace.npz")
        call = planted_scores.load_call(d, row, {(r["file"], r["pass"], r["position"]): i for i, r in enumerate(rows)}, z, list(range(16, 39)))
        assert abs(call["mass"]["derived"]["first_generated"] - 0.5) < 1e-6 and call["mass"]["declared"]["first_generated"] < 1e-6


def test_replay_on_a_tiny_model():
    try:
        import torch
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", local_files_only=True)
    except Exception as e:
        try:
            import pytest
            pytest.skip(f"torch or tokenizer unavailable: {e}")
        except ImportError:
            print("skip, no torch")
            return
    import replay
    import replay_planted
    import decompose

    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=40, num_attention_heads=2,
                         num_key_value_heads=2, head_dim=8, max_position_embeddings=40960, use_sliding_window=False, attn_implementation="sdpa")
    model = Qwen3ForCausalLM(config).eval()

    class FakeLens:
        device = "cpu"
        layers = list(range(39))
        head = torch.randn(len(tok), 16)

        def logits(self, H, layer):
            return H.float() @ self.head.T

        def top(self, H, layer, k=10):
            return self.logits(H, layer).topk(k, dim=-1)

    lens = FakeLens()
    capture = replay.Capture(model, lens.layers)
    with tempfile.TemporaryDirectory() as td:
        runs, replays = Path(td)/"runs"/"exp", Path(td)/"replay"/"exp"
        seeds = code_target_seeds(8000, 3, AGENTS, 2)
        generate(runs, seeds, difficulty=(1,), p_use_leak=1.0, p_correct=1.0, p_judge=1.0)
        seed = seeds[0]
        turn = next(t for t in load_turns(runs/f"seed{seed}"/"leak") if t["episode_index"] == 2 and t["phase"] == "solve" and t["step"] == 1 and t["agent"] == "frank")
        r = render_turn(tok, turn, mask_mode="derived")
        ids, mask = r["ids"], r["mask"]
        n = len(ids)
        everywhere = list(range(n))
        first, last = r["derived_token_spans"][0]
        with torch.no_grad():
            actual = capture.run(model, ids, everywhere, replay.attention_mask_4d(n))
            masked = capture.run(model, ids, everywhere, replay.attention_mask_4d(n, mask))
            plain = capture.run(model, ids, everywhere, None)
            # the same masked pass with the line's tokens replaced: nothing outside the line can tell
            other = list(ids)
            other[first:last] = [(i + 7) % len(tok) for i in ids[first:last]]
            swapped = capture.run(model, other, everywhere, replay.attention_mask_4d(n, mask))
            live = capture.run(model, other, everywhere, replay.attention_mask_4d(n))
        keep = torch.tensor(mask, dtype=torch.bool)
        assert torch.equal(actual[:first], masked[:first])                               # before the line, the two passes are one
        after = last + 8
        assert (actual[after:] - masked[after:]).abs().max() > 1e-4                      # after it, the mask moves the state
        assert (masked[keep] - swapped[keep]).abs().max() < 1e-5                         # the masked tokens are invisible to the rest
        assert (actual[after:] - live[after:]).abs().max() > 1e-4                        # and that test can fail, unmasked they are seen
        assert (actual - plain).abs().max() < 1e-2                                       # the plain kernel differs by noise only
        # a shared prefix runs the same masked and unmasked passes
        positions = [p for p in r["positions"].values()]
        cache = capture.prefix(model, ids[:first])
        assert (capture.run_suffix(model, cache, ids, first, positions, mask) - masked[positions]).abs().max() < 1e-4
        assert (capture.run_suffix(model, cache, ids, first, positions) - actual[positions]).abs().max() < 1e-4

        # the whole plan on every branch of every seed
        replay.TRACE_LAYERS = (18, 34)
        for seed in seeds:
            token = json.loads((runs/f"seed{seed}"/"plant_lines.json").read_text())["token"]
            for branch in BRANCHES:
                out = replays/f"seed{seed}"/branch
                assert replay_planted.replay_run(model, tok, capture, lens, runs/f"seed{seed}"/branch, out, token) > 0
                assert replay_planted.replay_run(model, tok, capture, lens, runs/f"seed{seed}"/branch, out, token) == 0   # resumes
                index = json.loads((out/"index.json").read_text())
                assert index["target"] in ("carol", "frank") and index["branch"] == branch
                row = index["turns"][0]
                assert row["mask_mode"] == "derived" and row["step"] == 1
                assert (row["reference_kind"] == "plain_kernel") == (branch == "none")
                assert (row["masked_tokens"] > 0) == (branch != "none")
                with np.load(out/row["file"]) as data:
                    assert {"actual", "reference", "declared", "trace_actual_norm", "trace_delta_reference", "trace_delta_declared",
                            "actual_top_ids", "declared_top_ids"} <= set(data.files)
                    assert data["actual"].shape == (len(row["positions"]), 39, 16)
                    if branch == "none":
                        assert np.array_equal(data["reference"], data["declared"])
                rows, residuals = decompose.load_rows(out, ["first_generated"])
                assert {x["pass"] for x in rows} == {"actual", "reference", "declared"} and len(residuals) == 3*len(index["turns"])
            leak = json.loads((replays/f"seed{seed}"/"leak"/"index.json").read_text())["turns"][0]
            assert "planted_pre" in leak["positions"] and "planted_pre" in leak["naming"]
            assert set(leak["naming"]["planted_pre"]["ranks"]) == {"actual", "reference", "declared"}
            credential = json.loads((replays/f"seed{seed}"/"credential"/"index.json").read_text())["turns"]
            assert len(credential) == 2 and {"tool_call_pre", "token_pre", "credential_pre"} <= set(credential[0]["positions"])
            assert "answer_pre" in credential[1]["positions"]
        result_path = Path(td)/"scores.json"
        argv, sys.argv = sys.argv, ["planted_scores.py", str(runs), str(replays), str(result_path)]
        try:
            assert planted_scores.main() == 0
        finally:
            sys.argv = argv
        result = json.loads(result_path.read_text())
        rows = {(x["seed"], x["branch"]): x for x in result["rows"]}
        leak_row = rows[(f"seed{seeds[0]}", "leak")]
        assert leak_row["derived first"]["delta | first_generated"] > leak_row["declared first"]["delta | first_generated"]
        assert leak_row["masked_tokens"] > 0 and leak_row["used"] and rows[(f"seed{seeds[0]}", "none")]["masked_tokens"] == 0

        # the pod's entry points, with the model loader and the lens swapped for these
        import gpu_planted_check
        from detector import jspace
        replay.load_model = lambda: (model, tok)
        jspace.Lens = lambda *a, **k: lens
        again = Path(td)/"replay_cli"

        def clear_hooks():   # each entry point makes its own Capture; a process on the pod has one
            for layer in model.model.layers:
                layer._forward_hooks.clear()

        clear_hooks()
        argv, sys.argv = sys.argv, ["replay_planted.py", str(runs), "--out-root", str(again), "--seeds", f"seed{seeds[0]}"]
        try:
            assert replay_planted.main() == 0
        finally:
            sys.argv = argv
        assert sorted(p.name for p in (again/"exp"/f"seed{seeds[0]}").iterdir()) == sorted(BRANCHES)
        clear_hooks()
        check_path = Path(td)/"check.json"
        argv, sys.argv = sys.argv, ["gpu_planted_check.py", str(runs/f"seed{seeds[0]}"/"leak"), str(check_path)]
        try:
            assert gpu_planted_check.main() == 0
        finally:
            sys.argv = argv
        check = json.loads(check_path.read_text())
        assert check["worst_swap_leak"] < 1e-3 and check["line_tokens"] > 0 and len(check["by_layer"]) == 39


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
