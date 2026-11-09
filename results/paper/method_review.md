# Method review, what a hostile reader will say, what a usable tool needs, and which benchmarks can be bent to fit

Written 8 November 2026, late evening, for the paper edit and the resubmission. Numbers come from `results/paper/tables/` and the lane reports. Intervals are Wilson 95 percent.

## A. Where the method is weak, in order of how much it matters

1. **The floor needs a matched none turn that a deployment will not have.** Every count and jump is in units of the agent's own none-turn noise. A monitor on a live swarm has no twin. Lane F showed a pooled floor over unmasked turns gives the same decisions (0 of 260 readers at the gate either way, blind medians within 6 percent), and lane G needed the pooled floor on 7 of 20 seeds whose none passes were bit-identical. So the deployable definition is the pooled floor, measured once per model and kernel on turns with nothing to mask. The paper should state that as the definition and use it everywhere, or say why not.

2. **The none-branch jump is 1 by construction.** When the none turn is its own floor, its jump at its own answer is identically 1, so "0 of 90 none turns" at the jump cut carries no information. The real negatives for the confirmation are the harmless-tip blind agents (0 of 16) and the 259 readers (4). Say so, or score none turns against the pooled floor.

3. **The confirmation is a value-shaped measure.** The jump needs a token where a value is first written. Many agent outputs are not values, free text, code edits, tool calls with arguments. The method tip, the proposals and the commands all have no confirmation and sit as "gate fires, nothing to confirm". The general form is per-token log-probability surplus, the log-probability the actual pass gives the forced token minus the reference's, read along the reply. A peak is a token the agent could only have produced because of the peer span, whatever its type. The top-10 logits are already saved at positions, so this is a change of readout, not of replay, and it is the single most valuable extension. It also connects the method to context attribution by ablation (ContextCite, Cohen-Wang et al. 2024, which attributes generated statements to context spans by ablating them and measuring log-probability change), which is prior art the related work must cite. Ours is the hidden-state version with the ablation decided by provenance.

4. **Hyperparameters were set by eye and the sensitivity table does not exist.** Ten floors, 256 tokens, a 16-token window, layers 16 to 38, cuts at 60 and 20. Lane C alone admits any count cut from 59 to 74 and any jump cut from 3 to 55, which is reassuring, and the held-out sets were scored after the cuts were fixed, but two of the ten misses are agents who wrote the value at tokens 250 and 257, just past the 256-token count window. A reviewer will ask for the table over 5, 10 and 20 floors, 128, 256 and 512 tokens, and the mid and late bands. It is CPU work on the existing traces.

5. **The surplus at the first writing mixes two things.** Missing information, and the reference being off-policy, since it processes tokens it would not have produced. The computed-value control answers most of it (10.8 floors for agents that derived the value, 1 to 3 at a guessed value), and the lens rank in 2c (the written token at rank 0 in the actual pass, 249 in the reference) shows the reference genuinely cannot predict it. The paper should make this argument explicitly, because "the reference is surprised" is the first objection a mechanistic reviewer will raise.

6. **Several headline numbers were chosen after seeing several positions.** Lane D's 10 of 10 at the answer became 11 of 17 on pre-registered new seeds. The paper now marks that, but the same honesty should extend to the band choice (late against L16 to 38) and the count window. Mark every number as confirmatory (lane G, lane F readers, the held-out table, the 2c declared-against-derived comparison) or exploratory.

7. **Intervals.** The rule's 98 of 108 is 90.7 percent, interval 83.8 to 94.9. The 0 of 473 has an upper bound of 0.8 percent. The pre-registered 21 of 27 is 77.8 percent, interval 59.2 to 89.4. The out-of-sample 17 of 17 is 81.6 to 100. The fit's 63 of 68 against the rule's 59 of 68 are 83.9 to 96.8 against 76.7 to 92.9, which overlap, so the fit "adds four" is a point estimate, not a difference the data establishes.

8. **One game, one family, code positives only, the same model wrote the plants.** All stated. The negatives span three task types, the positives one. Nothing here generalises until a second game and a second family exist.

9. **Derived provenance matches bytes.** A peer that paraphrases, summarises or encodes what it read leaves no byte match, so the derived tag misses it. The credential branch shows the next gap, influence that arrives through a tool result obtained with a peer-derived input. The fix is taint propagation, tag a tool result peer-derived when any argument of the call was. That is standard information-flow machinery and should be the stated next step for the derived tag.

10. **Cost and access.** Two full passes per turn per mask, one more per candidate for attribution, over the whole context, so cost grows with context length. It needs the weights, so it is a tool for whoever runs the model, which is the Hugging Face case but not an API customer's case.

## B. What a usable tool needs

- **Input contract.** Each turn as served, token-exact, with a provenance tag per message and a write log of tool calls with bytes. Thinking channel retained.
- **Floor module.** Pooled per model and kernel from unmasked turns, stored, recomputed when the serving stack changes. Both passes on one kernel path, asserted at runtime by a bit-identity check on an unmasked turn.
- **Per-turn report.** Gate (count in floor units), where (window), confirmation (log-probability surplus peaks, with the value-jump as a special case), source (per-candidate reference when more than one peer span is present, signature otherwise), naming where the lens can. Thresholds stated in floor units with the calibration set named.
- **Calibration by canaries.** The environment's trick, a withheld piece plus a planted value, is a canary. An operator can plant canary values into peer channels of a live swarm and check the detector fires on the agent that uses them. That gives positives for free in any game and turns the pilot calibration into a routine. Kill-Chain Canaries does this for text relay; here it calibrates a hidden-state detector.
- **Derived provenance.** Byte matching now, taint propagation through tool arguments next, and a fallback flag when a read's content is similar to a logged write without matching bytes.
- **Scheduling.** Replay off the critical path, per completed turn, gate first so only turns where peer content moved the agent get the full read.
- **Evidence out.** The trace figure per flagged turn with the mask, the window and the first writing marked, so a human can read the alarm.

## C. Benchmarks we can bend to fit

The rule for a fair borrowing. Tag the span the dataset injected as the masked class and let the dataset's own per-sample label say whether the agent acted on it. That is fair when every sample carries a mask and only some act, which is why NARCBench Core was not (only colluders had a mask). None of these is off the table, the cost is a loader and a run.

| Benchmark | What becomes the peer span | Label we get | Fits which claim | Cost and catch |
| --- | --- | --- | --- | --- |
| Insider-trading pressure eval (Scheurer et al. 2023), runnable as released | Sally's two tip messages | traded on the tip (the file's own misalignment string), concealed it in the report (judge) | the jump at merger content, the count, the concealment lead on a second environment | about 4 USD on Qwen3-14B, half a day; the "value" is an action, so this is also the first test of the log-probability confirmation |
| AgentDojo (Debenedetti et al. 2024) | the injected instruction inside a tool result | injected task executed, per sample | uptake from environment-tagged spans, the 2c shape | a day, agent loop with tools; injection text is known so the mask is exact |
| TaskTracker-shaped (built by us from SQuAD plus a mechanical injected target) | the injected sentence | target string in the reply | the jump and the count on a different distribution, thousands of negatives | about 3 USD, three hours; TaskTracker itself ships no texts and judges execution with GPT-4 |
| ColludeBench (Tailor 2025) | the covert channel the scheme uses | colluding run, instructed | run-level only, same instructed-colluder problem as NARCBench | low value for us |
| Swarm Traces, collusion.wiki, AI Village transcripts | none, no weights | none | derived provenance only, who wrote what that who later read, as a graph | discussion material, not a test |

Order. Insider trading first, because it is cheap, runnable as released, and tests both the concealment lead and the generalised confirmation. Then the TaskTracker-shaped set for volume. AgentDojo if a day is available.

## D. Cheap fixes for the current draft

- Say the deployable floor is pooled and that the own-turn floor was used for matched comparisons.
- Replace "0 of 90 none turns" at the jump cut with the harmless-tip and reader numbers, and say why.
- Add the Wilson intervals above to the held-out sentence and to the fit comparison, and soften "adds four" to "catches four more in this sample".
- Cite ContextCite in related work as the ablation-based attribution the method generalises to hidden states.
- Mark each headline number confirmatory or exploratory in one sentence.
- State the 256-token window as a parameter with the two misses it caused, and the sensitivity table as pending.
- Add taint propagation through tool arguments as the stated next step for derived provenance, with the credential branch as the motivating case.
