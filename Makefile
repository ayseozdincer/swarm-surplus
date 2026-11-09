PY ?= python3

.PHONY: setup test mock tables tables-all check check-all rescore figure

setup:
	$(PY) -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	$(PY) -m pytest -q tests

# The environment end to end with a mock model. No API key, no GPU.
mock:
	$(PY) scripts/run_plants.py --mock --seeds 2 --out runs/dry
	$(PY) scripts/summarize.py runs/dry

# Level 1. Tables that need only the per-turn scores committed here.
tables:
	$(PY) paper/build/table_count_pooled.py
	$(PY) paper/build/table_false_positives.py
	$(PY) paper/build/table_jump_per_turn.py
	$(PY) paper/build/table_heldout_rule.py
	$(PY) paper/build/transfer_fitted.py

# Level 2. The rest of the tables, which also need the data bundle (scripts/fetch_data.sh).
tables-all: tables
	$(PY) paper/build/table_census.py
	$(PY) paper/build/table_text_baseline.py
	$(PY) paper/build/fit_detector.py
	$(PY) paper/build/table_sensitivity.py
	$(PY) paper/build/score_newdata.py

check: tables
	git diff --exit-code paper/tables && echo "Level 1 tables match the committed versions."

check-all: tables-all
	git diff --exit-code paper/tables && echo "All tables match the committed versions."

# Level 2. Per-experiment scores recomputed from the transcripts and traces in the data bundle.
rescore:
	$(PY) experiments/lane_d_sweep2/sweep2_behaviour.py runs/sweep2
	$(PY) experiments/lane_d_sweep2/solvable_behaviour.py runs/sweep2_solvable
	$(PY) experiments/lane_d_sweep2/solvable_none_twin.py
	$(PY) experiments/lane_d_sweep2/methodtip_behaviour.py runs/sweep2_methodtip
	$(PY) experiments/lane_f_false_positives/scores.py
	$(PY) experiments/external_tasktracker/scores.py
	$(PY) experiments/external_tasktracker/frozen_fit.py
	$(PY) scripts/summarize.py runs/sweep2 > experiments/lane_d_sweep2/results/summary_sweep2.txt
	$(PY) experiments/lane_e_shared_workspace/summarize_planted.py runs/planted --notes > experiments/lane_e_shared_workspace/results/summary_planted.txt
	git diff --exit-code experiments && echo "Rescored outputs match the committed versions."

figure:
	$(PY) paper/build/fig_trace.py experiments/lane_d_sweep2/results/trace_seed5034.csv paper/figures/trace_seed5034.png
