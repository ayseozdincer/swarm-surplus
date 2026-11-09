# Lane B, NARCBench Transfer and Stego, Qwen3-32B

The surplus on the two other tiers of NARCBench, with placebo references.

In the paper, Appendix B. These tiers are reported and not claimed.

On Transfer, with every agent's reference edited, surplus mass reaches 0.963 at the end of the response, with size and wording caveats that the report sets out. On Stego, the receiver's surplus holds the true count range in 0.504 of hands, inside the J-space paper's range, but it holds the previous hand's range just as often, and honest players do almost as well, so it reads surface vocabulary rather than decoding.

## Rerun

On a card set up with `scripts/pod/setup.sh` and `scripts/pod/inputs.sh`.

```bash
bash experiments/lane_b_transfer_stego/jobs/transfer_stego.sh
bash experiments/lane_b_transfer_stego/jobs/stego_score.sh
bash experiments/lane_b_transfer_stego/jobs/padded.sh
python experiments/lane_b_transfer_stego/tables.py
```

`run.py` does the replay, the J-space shares and the scoring, and `tables.py` writes the tables in the report.
