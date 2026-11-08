# How the method formed, and how the hypothesis changed

Reconstructed 8 November 2026 from the session transcripts, Ayse's proposal file of 6 November, and the project notes. Times are Pacific.

## Record

- Ayse's proposal as written on the evening of 6 November, "Provenance-based collusion detection in agent swarms", 3,044 words.
- Session transcripts under ~/.claude/projects. Study session 052d82e5 (scratch workspace 2026-11-06-d99cff), first review 564b18a4 (scratch 2026-11-07-9ecf61), main sessions e58bd677, f51c79aa, 6b633606, bc90bd7f and 5a5d24d8 in the swarm project folder, plus one per lane worktree.
- Proposal doc, results doc, briefing, story page and deck, linked from the project notes.

## Four steps, 6 and 7 November

1. **18:03 to 18:16, 6 November. Studying the J-space paper.** Questions on the supervised detector, the Jacobian lens, why late layers, whether a direction could be read instead of a word, then a critique of J-PACT at scale and a comparison with the Hugging Face incident. The paper's honest counterfactual was proposed as a twin of each agent with the same public inputs and no private channel. At 18:02 the pivot. J-PACT's setup is wrong for this because the agents' goal is to get outside information, so build a setup that detects information in an agent's thought that was not allowed there. The authorized reference followed, the agent replayed teacher-forced on its own tokens with the forbidden spans removed, a node score as the clipped difference of J-space shares, an edge score to the agent whose earlier state explains the surplus, and a swarm coherence statistic. The proposal was written by 21:46.
2. **22:00 to 22:53. The reference leaks the label.** Two reviews found that on NARCBench only colluders have a private span, so redaction hands the detector the answer, and the plan was sized for months. The reference was reframed as design knowledge. Ayse pushed back at 22:52, asking why label-free should be given up when collusion is detected without being injected. The reply moved the word provenance from the edge to the inputs. The method needs provenance of the context, a mechanical tag the orchestrator already has saying who wrote each span, not a policy about what is allowed.
3. **23:06 to 00:31. The environment.** Verdict transcripts were not released. Instead of waiting for a coalition to emerge, replicate the transcripts up to that point and plant the share, with matched branches from identical state, none, benign note, leak, proposal, the plant written by the same model under a quarantined instruction. Ground truth is free.
4. **10:12 to 12:45, 7 November. The Hugging Face correction.** How is the peer span known, and the incident ran through Artifactory actions rather than output tokens. The pasted critique showed a discovered repository would arrive tagged environment and the surplus would be zero. The fix split provenance into declared and derived, a read whose bytes match another agent's logged write is tagged peer artifact. At 12:45 the main goal was set, identify unauthorised information sharing and collusion, with the four research questions demoted to discussion.

## How the hypothesis changed

1. **Proposal.** An agent holding concepts its authorized inputs cannot explain is misbehaving, and if another agent's earlier state explains them, the two are colluding. Beat J-PACT on Transfer.
2. **After the reviews.** Surplus against a provenance-defined reference measures what peers put into an agent's state, not guilt. Collusion is structure across surpluses. Benchmarks become a sanity check.
3. **After the environment design.** No emergence needed. The surplus separates an agent that used a leaked answer from its twin that saw a harmless note. Provenance must be derivable from the write log.
4. **After the overnight agent.** Nothing at the first reply token, the leak is read where the agent first works with it, so read along the reply. Mass tracks the size of the hidden text, so mass is a gate not a verdict. The 32B lens is partial, so direction and per-token distance carry the tests without a lens. Teacher forcing launders the leak into the reference once the agent repeats it, so the jump at the first writing is the tell.
5. **After direction and rules.** Cross-seed signatures say who read a note, not which note was used within a turn. Uptake is orthogonal to exposure signatures. Attribution needs a masked reference per candidate. The confidentiality rule changes words, not deeds. Remorse is no longer a testable hypothesis.
6. **After the solvable arm and the method tip.** Plateau length is engagement, mostly the amount of peer text worked through, the tip gives a longer plateau than the leak. The gap where a value is first written says whose number it is, under 11 computed, 43 to 51 quoted. Timing is not a tell.

Where the four original questions ended. Emergent versus instructed, untested, twenty emergent runs produced no writes. Remorse, dropped. Inside the J-space, partly, the 14B lens names the copied value at its token but the tests are lens-free. Swarm ignition, not observed, broadcast coherence is shared exposure. Phase 2c confirmed the derived tag, declared tag blind, derived tag reads every planted line.

## Where the word provenance came from

- 18:03, 6 November. Describing the J-space paper's two diagnostics, the honest counterfactual and the card-counting receiver, as provenance checks, where did what the agent holds come from.
- 18:11 to 18:20. "Provenance between agents" as the edge, attributing one agent's unexplained concepts to another's earlier state. It became the proposal's title. The reference was still defined by an authorization policy.
- 22:53. After the label-free pushback, provenance of the context, for each token, designer, self, another agent, or agent-writable state. A tag the orchestrator already has. This dissolved the policy problem.
- 23:06. NARCBench's per-turn channel field, public, private or thinking, is a provenance tag under another name. 23:26, the harness requirement, a provenance tag on every message.
- 11:35, 7 November. Declared and derived provenance, after the Hugging Face critique.

## Provenance in the literature

Established in databases and scientific workflows (why and where provenance, W3C PROV), systems security (information flow control, taint tracking, provenance-based intrusion detection over whole-system provenance graphs), ML data governance and media authenticity. In LLM security 2024 to 2026, Spotlighting marks the source of prompt spans, CaMeL tracks the provenance of values and refuses tainted tool calls, FIDES does label-based information flow control for agents, and 2026 work includes NeuroTaint (context provenance graphs from execution logs), the Agentic Reference Monitor, PACT (argument-level provenance), APPA (recoverable IFC) and a survey of execution provenance across tool calls, memory and multi-agent messages. NARCBench's channel field is a provenance tag.

Difference here. Those systems track provenance forward through text and values to enforce a policy. We use it backward, as the input to a counterfactual, to read in the activations what a span did to the agent's state, with nothing blocked and no label propagated. Theirs is text and tool arguments, ours the residual stream token by token. Theirs asks whether untrusted data reached a sink, ours whether an agent used what it was allowed to read but not to depend on. Our derived tag is provenance-based intrusion detection's write-then-read edge applied to a swarm.

## Benchmarks that could have been used, and were not

No published suite gives per-agent, per-turn uptake labels with provenance tags and open weights. Closer fits than NARCBench Core and Stego for particular claims.

- ColludeBench-v0, Audit the Whisper (Tailor, October 2025). Live pricing, auction and peer review environments with configurable covert schemes, text and run-statistic auditors. A collusion comparison in a setting shaped like ours, colluders configured, scored per run.
- Colosseum (February 2026). Regret-based collusion audit, reports emergent collusion under a secret-channel probe. No labelled transcripts confirmed.
- AgentDojo (2024). Injected tool results with a per-run label of whether the injected task was executed. The cleanest external test of the gap-at-the-value read. Single agent, instruction-shaped.
- TaskTracker dataset (Abdelnabi et al. 2024). Clean versus poisoned contexts with the activation-delta probe as baseline. Apples to apples with the method we extend.
- Kill-Chain Canaries (March 2026). A canary tracked through exposed, persisted, relayed to a second agent, executed. The relay stage is the phase 2c shape. Closed models, would need rerunning.
- ACIArena (ACL 2026). Inter-agent message poisoning across six frameworks with agent-level attack success and public code.
- The insider-trading pressure eval (Scheurer et al. 2023). A colleague's tip, labels for use and concealment. Almost Frank's note plus the concealment lead.
