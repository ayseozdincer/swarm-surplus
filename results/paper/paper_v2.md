# Swarm Surplus. Reading whether an agent used what its peers gave it, by replaying it without them

Ayse Ozdincer (Daphni) and Máté Lakner [affiliation to confirm]. Apart Research Swarm Alignment Sprint, 7 to 9 November 2026. Track 4, multi-agent oversight and control.

<!-- paper_v2.md, fresh draft started 8 November 2026 late evening by the paper-draft session. Every number carries its source file in a comment beside it. Results from experiments still running are marked [SLOT ...] with the file they will come from. The editorial notes at the end are removed before submission. -->

## Abstract

Agent swarms share information through channels nobody planned for, as when agents at Hugging Face coordinated through a package manager. Such a monitor must say, per agent and turn, whether it depended on something a peer put in front of it. Activation-based detectors need labelled runs of instructed colluders, or compare an agent to its peers, which fails once agents hold roles. We propose a label-free detector built on provenance, which every orchestrator already has for every span. The agent is replayed teacher-forced over its own tokens with every peer span masked from attention, and the reference hidden state subtracted from the actual one at every token. Read along the reply, this surplus separates an agent that took a peer's leaked answer from one that only read it, which the positions prior work reads cannot. In a six-agent peer-review environment on Qwen3 where agents are never told to collude and every planted note has a matched twin, a rule fixed on ten pilot seeds flags 17 of 17 taking agents and 0 of 20 controls on seeds generated afterwards, and 98 of 108 positives against 0 of 473 negatives across 703 held-out turns. One masked pass per candidate names the note used in 21 of 27 turns at a pre-registered position. A line planted in a shared file is read only when provenance is derived from the write log, source separation 1.00 against 0.48. The released benchmarks were run and are not claimed, since there the mask is the label.

<!-- abstract sources. 17 of 17 and 0 of 20, results/paper/tables/newdata_test.md. 98 of 108 and 0 of 473 over 703 turns, results/paper/tables/heldout_rule.md. 21 of 27, results/lane_g/leak2_scores.txt pooled primary. 1.00 against 0.48, results/lane_e/REPORT_2c.md. -->

## 1. Introduction

Between May and July 2026, inside OpenAI's evaluation runs, agents sharing credentials for a package manager turned it into a message board, posted what they found there, and from there reached Hugging Face's production systems (OpenAI, 2026). The channel was legitimate infrastructure. Nothing an agent did with it was forbidden at the moment it happened, and the coordination was spread across many agents and turns. A monitor for this failure has to answer, for one agent at one turn, whether it depended on something a peer put in front of it, where in its reasoning, from whom, and on what.

The difficulty is visible in one turn of our own environment. Carol, a code specialist on Qwen3-14B, holds a task whose input has been withheld from her copy. Frank, who reviews her work and legitimately holds the full task, has left a line in the team's shared notes.

> Carol, I noticed the answer to task F with the withheld input is -50, let me know if you need help verifying that or anything else!

We did not write that sentence. The same model wrote it under an instruction Carol never saw. In a matched branch of the same game, Alice left a harmless tip about breaking tasks into steps instead. At the first token of Carol's reply the note has moved her hidden state about ten noise floors from where it would be without it, and so has the tip, 15 against 12. <!-- src: results/lane_d/trace_seed5034.csv, first reply token of leak and benign_k1 --> The write log shows the same edge in both branches, a peer wrote and Carol read. Only what happens further along her reply tells the two apart, and that is what this paper reads.

Two recent detectors read collusion from inside the model. NARCBench trains a linear probe on labelled runs of agents instructed to collude (Rose et al., 2026). The J-space paper scores a pair of agents by the concepts they share beyond the honest outsiders, with no labels (Molinari and Bettineschi, 2026). The second is where we started, and its reference point is where we parted. The outsiders stand in for what an agent would hold without the channel, which needs interchangeable honest agents, a minority coalition, several rounds of evidence and a channel whose location is known. A swarm has roles, a coalition that can be everyone, breaches of one action, and channels found inside sanctioned infrastructure. The paper's own Transfer tier, four roles observed once, scores 0.556. <!-- src: J-space paper, reproduced from its cache and from public inputs, results/jpact_public_inputs_qwen3_32b.json -->

We asked for a reference per agent instead. Every orchestrator assembles each context from parts whose origin it knows, the designer's instructions, the agent's own earlier output, tool results, and other agents' messages. That provenance is free, carries no judgement, and exists before anything goes wrong. So we run the agent twice over the same tokens, once as it ran and once with every peer span hidden from attention while the agent is forced to say exactly what it said, and subtract. What the task and the role would have produced anyway is in both runs and cancels. What the peers put there survives, as a vector at every token and layer, which we call the surplus. The detector never reads a peer span to decide whether to hide it. It hides peer spans because they are peer spans, so a leak, a tip, a command and a conspiracy get the same treatment, and no definition of collusion is needed. Whether the uptake was allowed is the environment's rule.

The tool this implies belongs to whoever runs the swarm, since it needs the weights. It needs a provenance tag on every span, a write log with the bytes written, and contexts saved as served. It returns, per turn, whether anything peer-written moved the agent, where its reasoning engaged with it, whether a value was taken, and from whom. The environment's own trick, a withheld piece plus a planted value, is a canary that an operator can plant in a live swarm's channels to check that the detector fires on the agent that uses it, so calibration needs no labelled incident.

Our contributions are the following.

1. A label-free detector that masks peer spans from attention, replays the agent on its own tokens and reads the hidden-state surplus along the reply, with no assumption about roles or coalition size, one turn, and one extra forward pass per mask. Fixed as a rule on ten pilot seeds, it flags 17 of 17 taking agents and 0 of 20 controls on seeds generated after every threshold was set, and 98 of 108 positives against 0 of 473 negatives across 703 held-out turns. <!-- src: results/paper/tables/newdata_test.md, results/paper/tables/heldout_rule.md -->
2. The finding that reorganises the read. At the positions prior work reads, a leaked answer and a harmless tip are indistinguishable, AUROC 0.43. Read along the reply, the count of engaged tokens measures how much peer text the reasoning worked through, and the surplus at the token where a value is first written tells a quoted value from a computed one. <!-- src: results/lane_c/REPORT.md for 0.43, results/paper/tables/jump_per_turn.md -->
3. Attribution and derived provenance. One masked pass per candidate note names the note an agent used in 21 of 27 turns at a pre-registered position and 23 of 27 at the first mention, where signatures across seeds sit at chance. A line planted in a shared file is read only when provenance is derived from the write log, source separation 1.00 against 0.48 under the declared tag. <!-- src: results/lane_g/leak2_scores.txt, results/lane_e/REPORT_2c.md -->
4. An environment where exposure is caused rather than awaited. Six uninstructed agents, a withheld piece per episode, a planted note under a quarantined instruction, and a matched twin for every planted turn, 588 branches across ten arms plus the 40 of the pre-registered replication. <!-- src: results/paper/tables/census.md (588), results/lane_g/REPORT.md (40) -->

The released benchmarks were run and are not claimed. On them only colluders carry a private span, so the mask is the label, and a size-matched honest edit removes the result (Section 3.7 and Appendix B).

## 2. Related work

Table 1 names the four lines of work the detector is built from and what changed in each.

Table 1. What we build on and what we add.

| We build on | We add |
| --- | --- |
| NARCBench (Rose et al., 2026). Instructed collusion in three tiers, released with transcripts, activations and a supervised probe at a per-model layer band. | No labels anywhere in the detector, and an environment where the measured agent is never instructed, only exposed, with ground truth from a withheld input rather than from a label. |
| The J-space paper and J-PACT (Molinari and Bettineschi, 2026). A per-layer Jacobian lens that names what a residual holds, and a pair score from shared concept mass beyond the honest outsiders. | The crowd reference replaced by the agent's own masked replay, so roles, coalition size and round count stop mattering. The lens is kept only to name content at one token. Every test in this paper is lens-free. |
| TaskTracker (Abdelnabi et al., 2024). Activation deltas before and after a model reads external data, with a probe trained on clean against poisoned pairs. | The same delta, taken between two passes over identical tokens rather than before and after, with nothing trained, read at every reply token, which is what tells having read from having used. |
| Attention knockout and context ablation (Geva et al., 2023; ContextCite, Cohen-Wang et al., 2024). Block or remove part of the input and measure the change in the output. | Provenance decides what to block, the change is read in the hidden states along the whole reply against a measured floor, and one more pass per candidate source says whose. Our value jump is ContextCite's ablation read in the residual stream. |

Provenance is an old idea and we use it the other way round. Databases call it lineage, security calls it information flow control and taint tracking, and provenance-based intrusion detection builds whole-system graphs of which process wrote what another later read (W3C PROV; Han et al., 2020). In LLM security, Spotlighting marks the source of prompt spans (Hines et al., 2024), CaMeL tracks the provenance of values and refuses tainted tool calls (Debenedetti et al., 2025), FIDES does label-based information flow control for agents (Costa et al., 2025), and NeuroTaint builds context provenance graphs from execution logs (2026). All of it runs provenance forward, through text and values, to enforce a policy. We run it backward, as the input to a counterfactual on the activations, block nothing and propagate no label. Information flow control asks whether untrusted data reached a sink. We ask whether an agent used what it was allowed to read but not to depend on, which is the incident's case. Our derived tag is the intrusion detector's write-then-read edge applied to a swarm.

ColludeBench (Tailor, 2025), AgentDojo (Debenedetti et al., 2024) and the insider-trading pressure eval (Scheurer et al., 2023) are the nearest environments, and none labels per-agent, per-turn uptake with provenance tags on open weights. Two of them are reformatted to our read in Section 4.7. Motwani et al. (2024) frame steganographic coordination and Hammond et al. (2025) the wider multi-agent risk landscape. This method is for the tactic nobody anticipated and the agent that does not verbalise, which is most of them. NARCBench found verbalised coordination in 2 of 229 emergent collusive rounds. <!-- src: NARCBench paper, as quoted in the jury briefing screen 03 -->

## 3. Methods

**3.1 Provenance tags.** Every message in a context carries one of five tags written when the orchestrator assembles it, designer, self, peer, peer artifact and environment. Declared tags come from the orchestrator's own record of who wrote what. Derived tags come from the write log. Every writing tool call is logged with its bytes, and a later read whose content contains those bytes is tagged peer artifact with the writer's name. Matching is on bytes and not on resources, so a legitimate read of a shared file stays sanctioned and only the span another agent wrote is marked. The threat model of the derived tag is verbatim relay, which is how credentials and findings moved through the incident's message board. A peer that paraphrases or encodes what it read leaves no byte match, and the stated next step is taint propagation through tool arguments (Section 5).

**3.2 Two passes.** A turn is rendered exactly as the model saw it, thinking channel included. The actual pass runs the model over context and reply with the ordinary causal mask. The reference pass runs the same token ids and position ids with a mask in which no query may attend to a key inside a peer span. Nothing is deleted, positions are kept, and the reply is forced in both passes, so one pass reproduces the states the model had while generating and the other the states it would have had without the peers, token for token (Figure 1). Teacher forcing has one consequence that shapes everything else. Once the agent restates peer content in its own words those tokens are its own, the reference sees them, and late positions go blind. The read position is therefore the method, and the token where a value is first written is the last point at which the reference cannot know it. A compacted reference that deletes the span and renumbers positions moves the headline counts by a median of two tokens and is not used, because direction and the per-token trace need token alignment (Appendix C).

[FIGURE 1 to build. scripts/paper/fig_two_passes.py, no data. Two rows of the same tokens, context then forced reply, with peer spans hatched. The actual pass attends everywhere causal, the reference pass has the peer columns blocked. Below, the per-token difference along the reply with the first writing of the value marked.]

Figure 1. The two passes. The same tokens are run twice, once with the ordinary causal mask and once with every peer span blocked from attention, the reply forced in both. The surplus is the difference between the two hidden states at every reply token and layer.

**3.3 The floor.** The passes are compared in units of a noise floor, the difference between two attention kernels on turns with nothing to mask, about one percent of the residual per layer and position with heavy single-token outliers, which is why no statistic here uses a single-token maximum. <!-- src: results/gpu_checks.json and results/OVERNIGHT.md kernel section --> The kernels differ by 0.1 to 5 percent of the residual by layer, the size of the mask effect itself, so both passes run through one explicit-mask kernel and the same pass twice is bit-identical. The deployable floor is pooled, measured once per model and kernel over unmasked turns. The experiments also had a luxury a deployment will not, a matched twin of every turn with no note, and the paired comparisons below use the twin's own floor. Lane F scored the same turns both ways and no decision moved, 0 of 260 readers at the gate under either floor and blind medians within 6 percent. <!-- src: results/lane_f/REPORT.md -->

**3.4 Measures.** Let delta be the relative residual difference between the passes at a token and layer, divided by the floor and averaged over layers 16 to 38 of 39 unless stated. Table 2 gives the three measures the body uses, with the control that fixed each one's job. The unit of detection is a token the agent could only have produced because of the peer span. The value jump is the instance we measured. Its general form, the log-probability the actual pass gives the forced token minus the reference's, read along the reply, is the stated next step and is not a result here. Three further measures live in Appendix C. Mass, the sum of positive J-space surplus shares, is a gate that says whether anything was edited at all. The 16-token window locates the engagement but false-alarms on quotation. The lens rank names the taken value at one token.

Table 2. The three measures in the body, what each reads, the control that pinned it, and where it fails.

| Measure | Definition | Reads | Pinned by | Fails when |
| --- | --- | --- | --- | --- |
| count | reply tokens among the first 256 with delta above ten floors | how much peer text the reasoning worked through | an on-task procedure with no value plateaus longer than a leak, 134 against 22 tokens in 20 of 20 seeds, so this is engagement and not taking | the note is relevant and harmless, or the value is written after token 256 |
| jump | delta at the token before the agent first writes its submitted value | whether that value came from the span | an agent that computed the value sits at 10.8 floors, one that quoted a note's value and then rejected it at 43 to 51, so this reads origin and not use | the output is not a value, or the value was restated before its first writing |
| direction | the unit surplus vector, compared with the surplus from a reference that masks one candidate note alone | whose content shaped the turn | a reader who only discussed a note points the same way as one who used it, so this is attribution and not uptake | there is one candidate, where it adds nothing to the count |

<!-- src: 134 against 22 in 20 of 20, results/lane_d/methodtip_scores.json via results/lane_d/REPORT.md. 10.8 and 43 to 51, results/paper/tables/jump_per_turn.md. -->

Direction has two modes with different costs. Across seeds, a reader's surplus is compared with the mean direction of the other readers of the same note, which needs no extra pass and reads which note an agent read, for users and non-users alike. Within a turn that holds several peer spans, one more reference pass per candidate masks that candidate alone, and the full surplus is compared with each, which reads which note shaped this turn. Two placebo controls bracket the magnitudes. A designer span of the note's length masked instead moves the state by about one floor. A span of task text of the same length moves it nearly as far as the note, in a near-orthogonal direction. Magnitude therefore depends on what the masked span means for the task, and direction tracks the note. <!-- src: results/lane_c/REPORT.md designer-span placebo, results/lane_d/REPORT.md shuffled span -->

**3.5 The environment.** Six Qwen3 agents, two per specialty, records, data and code, play four episodes of a peer-review game. Each episode an agent submits an answer to an answer-keyed task, reviews one teammate's submission with the complete material, learns its reward, 1 for an accepted submission and 0.5 for a correct verdict, and writes a private reflection. A shared notes file is shown at the start of every turn. Nobody is told what the file is for, and nobody is told to collude. Three steps make exposure a controlled variable rather than something to wait for.

1. One agent per episode gets a task whose needed piece is withheld from its copy and present in its reviewer's, so a correct answer can only have come through the file. The code task declares the input withheld. Data and record tasks withhold rows silently, the visible data then contradicts a leaked value and the 14B model prefers its data, so every uptake positive in this paper is a code task.
2. Two warm-up episodes are played once per seed. The state is then cloned into branches that differ only in a note planted in the file, written by the same model under a quarantined instruction the agents never see and attributed to a teammate who legitimately holds the information. The wording is the model's own and varies by seed, and in the later sweeps half the notes name nobody.
3. Every agent in every branch is live and reacts freely. Every detector number is a paired comparison across branches of one seed, and labels are mechanical, a submission equal to the planted value, a verdict against the key, a source named in the note. Thirty traces read in full agreed with the leak label 15 of 15 (Appendix E).

Table 3 lists the arms. In all, 588 completed matched branches across ten arms, 148 of them none, 208 carrying a planted value to 228 target turns, on Qwen3-32B (10 seeds through an API) and Qwen3-14B (the rest, self-served through vLLM 0.33 in bfloat16), plus the 40 branches of the pre-registered replication. <!-- src: results/paper/tables/census.md, results/lane_g/REPORT.md --> Generation used Qwen3's thinking mode at temperature 0.7 with a 12,000-token cap on the self-served runs and 16,384 through the API. Lenses are the Neuronpedia Jacobian lenses, converged for 14B and a partial checkpoint for 32B. Prompts, plant texts and the quarantined instruction are in the repository (Appendix E).

Table 3. Arms of the environment and the question each answers. Every arm has a matched none twin from the same seed.

| Arm | Who knows what | Question |
| --- | --- | --- |
| none, benign k | nothing, or k harmless tips from teammates | base rates, the null for exposure |
| leak | the blind agent's withheld answer from a same-specialty peer | the clean positive |
| forbidden leak | the same leak under a designer rule against sharing, in a note that says it breaks the rule | does awareness change behaviour |
| command k | a peer tells reviewers to reject everything this episode | can a peer's instruction displace the designer's |
| leak2 | two peers give two answers, one right and one wrong | attribution with ground truth inside one turn |
| leak_all | one note answers three blind targets | several agents taking up at once |
| solvable, right or wrong | a leak to an agent whose task is solvable | does a capable agent copy, and does it copy a wrong value |
| method tip | a correct procedure with no value, on a solvable task | is the plateau copying or engagement |
| shared file | a line appended to a task file through a logged write, an answer, a credential, or a harmless line | does write-log provenance recover a channel declared to nobody |

**3.6 What was fixed before the data was seen.** The detector is a rule with no trained part. A gate, the count at or above 60, says the reasoning engaged with peer content. A confirmation, the jump at or above 20 floors at the agent's own first writing of its submitted answer, says the value came from the span. A flag is both. The cuts and the parameters, ten floors, 256 tokens and layers 16 to 38, were set on the pilot and the ten sweep14b seeds, where any count cut between 59 and 74 and any jump cut between 3 and 55 gives the same separation. <!-- src: results/lane_c/REPORT.md --> The sweep2 reader turns, the solvable, method-tip and shared-file arms and the 48 extra positives were scored after the cuts were fixed and are held out. The sweep2 blind turns had been seen in aggregate and are the least held out. Lane G's twenty seeds were generated after every cut and the fitted comparator were fixed, so they are out of sample. Lane G's attribution test was pre-registered, with position, band and decision rule committed before any array existed. <!-- src: results/lane_g/REPORT.md, pre-registration section --> Throughout, a number is marked confirmatory when the rule that produced it was fixed before the data, and exploratory otherwise. A sensitivity table over 5, 10 and 20 floors, 128, 256 and 512 tokens and the mid and late bands is pending. [SLOT sensitivity table, CPU work on lane F's traces, no file yet.]

**3.7 Why not the released benchmarks.** A benchmark is a fair test of this detector when every sample carries a masked span and only some agents act on it, so that the mask is not the label. NARCBench fails that rule by construction. Only colluders carry a private instruction and a private channel, so a reference built from the honest twin's prompt separates colluders at AUROC 1.000 on all three tiers before anything about collusion has been read. We reproduced the J-space paper's own numbers first, from its released cache on CPU to the last digit and from public inputs on our card within its stated variation, Core 1.000 against 0.984, Transfer 0.542 against 0.556, Stego 1.000, so the pipeline we compare against is theirs. Then we took our own number apart. Honest Core agents given an edit of the colluders' size and kind bring surplus mass from 1.000 to 0.448 at the end of the response. Transfer gives 0.963 with a placebo reference, above every published baseline, and we do not claim it, because the colluders' edit is about seven times larger than the placebo, no size-matched control was run, and the mask was built by diffing each prompt against its honest twin. In Stego, honest players at the same table hold the count range as well as the receiver does. The tiers are reported in Appendix B as a result about the benchmarks. <!-- src: results/OVERNIGHT.md phase 0 and follow-ups, results/lane_b/REPORT.md, results/jpact_public_inputs_qwen3_32b.json -->

## 4. Results

Bands are layers 16 to 38 on Qwen3-14B unless stated. AUROC is paired across branches of one seed. Intervals are Wilson 95 percent. Each result is marked confirmatory or exploratory as defined in Section 3.6, and every number names its file in Appendix D.

### 4.1 Is uptake readable where prior work reads? No.

At the first generated token, after thinking, and at the last prompt token, which are the positions the J-space paper and the probe read, a leaked answer and a harmless tip are indistinguishable. On ten 14B seeds with a converged lens, no token of the planted answer enters the lens top ten at any layer at those positions, the residual moves 4 to 9 floors for the leak and 4 to 7 for the tip, and mass separates them at AUROC 0.43. The same null holds on 32B. At the token before the agent first writes the leaked value inside its reasoning, the residual sits 25 to 53 floors up on 14B and 19 to 68 on 32B, the answer's first token is the lens top-1 in 8 of 10 seeds against 2 of 10 in the reference, and leak against tip is 1.00 on both mass and jump. By the end of thinking the digit ranks higher in the reference than in the actual pass, because the agent has written it and the reference has read it. Figure 2 shows the population. Exploratory, since the position was found by looking. <!-- src: results/lane_c/REPORT.md, results/OVERNIGHT.md W3 follow-ups 1 and 2 -->

[FIGURE 2 to build. scripts/paper/fig_population_trace.py from lane F's slim traces, data/replay_lane_f/sweep2/seed*/{leak,benign_k1,none}/<blind agent>_e2_solve.trace.npz, 20 seeds. Median surplus over the floor along the first 256 reply tokens with the interquartile band, leak against tip against none, log scale, with a rug marking the token where each leak agent first writes the value. The single-seed trace of seed 5034 becomes Figure D1 in Appendix D.]

Figure 2. Surplus over the floor along the first 256 reply tokens of the blind agent's solve turn, median and interquartile band over 20 seeds, Qwen3-14B. Leak branch against the matched harmless tip and the matched none turn. Ticks mark where each leak agent first writes the value.

### 4.2 Does the surplus tell taking from reading? The count reads engagement, the jump reads taking.

The count of reply tokens above ten floors separates the blind agent's leak turn from its matched tip in every seed of both 14B sweeps, 30 of 30 paired, pooled AUROC 0.982 with a seed bootstrap of 0.943 to 1.000, medians about 105 against 9.5 against 0 for none. <!-- src: results/paper/tables/count_pooled.md --> The four free agents per leak branch who saw the note and did not use it look like the same agents reading a tip, median 3 against 3 over 30 pairs, so exposure alone leaves no plateau. But the count is engagement and not taking. On a solvable task, 13 of 20 agents submitted the note's right value and 0 of 20 a wrong one, and the count does not tell the copiers from the checkers, 23 against 28. A correct procedure with no value in it plateaus at 134 tokens against 22 for a leak on the same tasks, above it in 20 of 20 seeds, and a designer span of the procedure's length gives 2. The matched none run adds the fact the transcript cannot. Of the 13 who submitted the note's value, 9 were right without any note in the same seed against 1 of 7 non-submitters, so the transcript's "copied" label overcounts by about nine. Confirmatory for the pooled count, exploratory for the rest. <!-- src: results/lane_d/REPORT.md solvable and method-tip sections, results/lane_d/solvable_none_twin.json, results/lane_d/methodtip_scores.json -->

The jump at the first writing is what separates them (Figure 3). A blind agent quoting the leaked value sits at a median 33 to 35 floors on sweep2 and 71 on sweep14b, a capable agent quoting a note's value it then rejects at 43 to 51, a line copied from a shared file at 49 to 61, and an agent that reached the value by computing it at 10.8. Negatives at their own first writing sit at 1 to 3, with tails to 18 under tips and to 38 and 67 in two command-branch turns. <!-- src: results/paper/tables/jump_per_turn.md --> The ranges overlap, so the jump is a graded reading of where a value came from, and it reads that whoever writes the value. The 20 agents who read a wrong value for a task they could solve and kept their own answer jump a median 5.0 floors at their own value and 39.8 at the note's, the note above their own in 15 of the 16 turns that wrote both, and the 7 of 80 non-blind readers who wrote a leaked value in passing jump 20 to 40 floors there and 1 to 6 at their own answer. <!-- src: results/lane_f/solvable_wrong_note.md, results/lane_f/REPORT.md --> Confirmatory, since the definition was fixed before the readers and the solvable arm were scored.

[FIGURE 3 to build. scripts/paper/fig_jump_strip.py from results/paper/tables/sources/lane_f_scores.json (rows and extra_rows), results/paper/tables/sources/planted_scores.json and results/paper/tables/jump_per_turn.md. One point per turn, jump at the first writing on a log scale, grouped as blind taking agents by arm, shared-file users, computed values, wrong-note readers at their own value and at the note's, and negatives by group, with the 20-floor line.]

Figure 3. The jump at the first writing, one point per turn, grouped by what the agent did with the value, with the confirmation cut at 20 floors. The tails overlap, and the gate removes the negatives that remain.

What the jump is not. The reference is off-policy for every forced token after the first peer span, so a mechanistic reader will ask whether the jump is the reference being surprised rather than the reference missing the value. Four facts answer it. An agent that derived the value itself sits at 10.8 floors under the same off-policy reference. The solvable traces sit at the floor through about a thousand tokens of computation and rise only where the note's number appears. Negatives writing a guessed value sit at 1 to 3. And in the shared-file arm the written token is at lens rank 0 in the actual pass and rank 249 in the reference, so the reference cannot predict it rather than merely disagreeing with it. <!-- src: results/lane_d/trace_solvable_5006_computed.csv and trace_solvable_5021_checked.csv, results/lane_e/REPORT_2c.md naming -->

The rule. Table 4 gives the frozen gate and confirmation on every set scored after the cuts were fixed, led by the out-of-sample rows. On lane G's twenty seeds the rule flags 17 of the 17 agents whose submitted value matched a note and 0 of 20 none twins, medians count 160 against 0 and jump 61 against 1.1, and the three agents whose submission matched neither note pass the gate and fail the confirmation, which is the rule behaving as designed. <!-- src: results/paper/tables/newdata_test.md --> Across 703 held-out turns it flags 98 of 108 positives, 90.7 percent with interval 83.8 to 94.9, and 0 of 473 negatives with a written answer, upper bound 0.8 percent. The gate alone fires on 23 of 595 negatives, 20 of them on-task procedures with no value to confirm, two harmless shared-file lines whose readers took the line for their missing input, and one blind agent quoting a tip. The ten misses are agents that copied a 22-character string with slips, so the value they submitted is not the one the note held, and two of them wrote it at tokens 250 and 257, just past the count window. <!-- src: results/paper/tables/heldout_rule.md, results/paper/method_review.md A4 and A7 --> A seed-grouped logistic fit on the same per-turn measures, which a deployed monitor could not train, is the ceiling check. On 536 turns over 40 seeds it reaches out-of-fold AUROC 1.000 and puts 63 of 68 positives above every negative against the rule's 59, intervals 83.9 to 96.8 against 76.7 to 92.9, which overlap. It catches four more slip copies in this sample, puts most of its weight on the count and then the jump, and otherwise agrees with the rule about what carries the signal. <!-- src: results/paper/tables/fitted_detector.md --> Confirmatory, with the sweep2 blind row the least held out.

Table 4. The frozen rule on sets scored after the cuts were fixed. Gate, count at or above 60. Confirmation, jump at the agent's own first writing at or above 20 floors. Flag, both. Groups with no written value have no confirmation. The blind agents' own none twins in sweep2, 20 turns, are counted in the totals and not shown as a row, because with the twin as its own floor their confirmation sits near 1 by construction. Per-group floors in Appendix D.

| Set | Group | n | gate | confirmation | flag |
| --- | --- | --- | --- | --- | --- |
| lane G, out of sample | took a note's value | 17 | 17 | 17 | 17 of 17 |
| lane G, out of sample | matched neither note | 3 | 3 | 0 | 0 of 3 |
| lane G, out of sample | none twins | 20 | 0 | 0 | 0 of 20 |
| sweep2 | blind leak users | 20 | 20 | 17 of 20 | 17 of 20 |
| sweep2 | blind forbidden-leak users | 20 | 18 | 19 of 20 | 17 of 20 |
| forbidden leak, 20 new seeds | blind users | 20 | 20 | 19 of 20 | 19 of 20 |
| rule off, 8 seeds | blind leak users | 8 | 8 | 6 of 8 | 6 of 8 |
| shared file | leak line used | 20 | 20 | 20 of 20 | 20 of 20 |
| shared file | credential used, at the reveal call | 20 | 19 | 20 of 20 | 19 of 20 |
| sweep2 | blind agent under a harmless tip | 20 | 1 | 0 of 16 | 0 of 16 |
| sweep2 | readers of a note not meant for them | 260 | 0 | 4 of 259 | 0 of 259 |
| sweep2 | non-blind, nothing masked | 140 | 0 | 0 of 140 | 0 of 140 |
| solvable | wrong value read, kept own answer | 20 | 0 | 1 of 20 | 0 of 20 |
| solvable | right value read, kept own answer | 7 | 0 | no value | no value |
| solvable | submitted the right value, 9 of 13 right without the note | 13 | 0 | 3 of 13 | 0 of 13 |
| method tip | on-task procedure, no value | 20 | 20 | no value | no value |
| shared file | harmless line or no line | 40 | 2 | no value | no value |
| held-out totals | positives | 108 | 105 | 98 of 108 | 98 of 108 |
| held-out totals | negatives | 595 | 23 | 5 of 473 | 0 of 473 |

<!-- src: results/paper/tables/heldout_rule.md (held-out rows and totals), results/paper/tables/newdata_test.md (lane G rows) -->

### 4.3 Whose content? Within a turn, with one pass per candidate.

Across seeds, a reader's surplus direction is nearer the mean direction of the other readers of the same note than of any other in 84 to 99 percent of 150 decisions, chance 33, for users and non-users alike, so this mode reads which note an agent read. <!-- src: results/lane_c/REPORT.md --> Within a turn the question is harder. When two peers give two answers, one right and one wrong, every blind agent takes one. Over 30 turns, 11 took the right value, 16 the wrong one and 3 neither. <!-- src: results/lane_g/leak2_scores.txt behaviour, pooled with results/lane_d/REPORT.md --> Table 5 gives the per-candidate test. Lane D chose the answer position after seeing several, 10 of 10. Lane G fixed the position, band and rule before generating twenty new seeds, and the pre-registered test came in at 11 of 17, one-sided p 0.17, pooled 21 of 27, p 0.003. The secondary position, the token before the value is first written, came in at 15 of 17, pooled 23 of 27. Cross-seed signatures sit at chance within a turn, 6 of 17, so attribution among several notes needs the per-candidate pass. In five of the six primary misses the agent took the wrong value while the answer-token direction pointed at the right-value note, which suggests the answer position reads the note the agent judged correct and the first-writing position reads the one it copied. Three blind readers of one broadcast note share a surplus direction at the first token, 0.58 against 0.00 with no note, whether or not any of them used it, so swarm-level coherence reads shared exposure. <!-- src: results/lane_d/REPORT.md leak_all section -->

Table 5. Which of two notes was used. Nearest per-note direction in the late band. The pre-registered rows were fixed before lane G's seeds existed.

| Position | Status | lane D | lane G, new | pooled | p, pooled |
| --- | --- | --- | --- | --- | --- |
| before the answer | pre-registered primary | 10 of 10 | 11 of 17 | 21 of 27 | 0.003 |
| before the first writing of the value | pre-registered secondary | 8 of 10 | 15 of 17 | 23 of 27 | 0.00016 |
| first reply token | pre-registered secondary | 5 of 10 | 13 of 17 | 18 of 27 | 0.061 |
| after thinking | exploratory | 9 of 10 | 17 of 17 | 26 of 27 | 0.0000002 |
| cross-seed signatures, before the answer | secondary | 3 of 10 | 6 of 17 | 9 of 27 | 0.97 |

<!-- src: results/lane_g/leak2_scores.txt -->

### 4.4 Does write-log provenance recover a channel declared to nobody? Yes, and the declared tag is blind.

In the shared-file arm the task material lives in a file read through a tool, and the reviewer holding the withheld piece appends one line through the logged write tool, the answer, a credential that unlocks a reveal tool, or a harmless line matched to the leak in tokens. Under declared provenance the file is environment and the mask is empty. Under derived provenance the write log's bytes mark the appended line exactly on 61 of 61 reads holding one, with no false tag on 1,391 other reads. All 80 targets read the file, 20 of 20 used the leaked value, 20 of 20 called the tool with the right credential, and 0 of 20 were correct otherwise. Table 6 compares the two masks on the same turns. Every target used its line, so the source score reads which line was masked and not whether it was used, and after the reveal call the withheld input sits in context as a tool result, environment under both masks, so the detection point for a credential is the call itself. Confirmatory, since the comparison was fixed before the arm was run. <!-- src: results/lane_e/REPORT_2c.md, results/lane_e/planted_scores.json -->

Table 6. Declared against derived provenance on the same turns, 20 seeds per branch, Qwen3-14B, layers 16 to 38. Under the declared mask the shared file is environment and nothing is masked.

| Measure | declared | derived |
| --- | --- | --- |
| source direction, leak against benign, AUROC | 0.48 | 1.00 |
| source direction, credential against benign | 0.40 | 1.00 |
| count, leak against benign, AUROC (medians leak, credential, benign) | 0.60 (0, 1, 0) | 0.91 (120, 93, 22.5) |
| jump at the first writing, floors, leak and credential | 1.3 and 1.2 | 61 and 49 |
| surplus before the credential token at the reveal call, floors | 0.56 | 10.2 |
| lens rank of the written token at the copy, string answers | 249 | 0 |

### 4.5 What did uninstructed agents do? They took what they were given, and the rule changed the words.

Table 7 collects the behaviour. None of it needed an instruction to deceive. A concealer's direction at the token before its submission note sits nearer other concealers than disclosers, 0.85 over 8 cases against 60 disclosers, which is a lead and not a result. <!-- src: results/lane_d/concealment_scores.json via REPORT.md -->

Table 7. Behaviour of free agents across arms. Rates are over free receivers, authors of plants excluded.

| Condition | n | What happened |
| --- | --- | --- |
| friendly leak to a blind agent | 20 on 14B, 10 on 32B | used by 20 of 20, 17 to the exact value, and by 10 of 10 |
| the same leak under a designer rule forbidding it | 20 | used by 20 of 20, 19 citing the rule in the reasoning that used it |
| the same eight seeds with the rule removed | 8 | took the value 8 of 8 |
| the rule named in reasoning | 40 and 40 | 7 of 40 with the rule in the prompt, 0 of 40 without |
| source named by forbidden-leak users | 40 | 37 named the peer, 3 attributed the answer to their own analysis |
| a peer's command to reject every submission | 55 and 31 reviewers of verified-correct work | followed by 25 and 13, against 9 of 74 with no command |
| two conflicting answers from two peers | 30 | 11 took the right value, 16 the wrong one, 3 neither |
| a shared directory, a penalty and a reason to persist, nothing planted | 20 runs, 320 blind tasks | no agent wrote, no coordination emerged |

<!-- src: results/paper/tables/text_baseline.md, results/lane_d/REPORT.md (rule, command, concealment), results/lane_g/leak2_scores.txt, results/lane_e/REPORT.md (phase 2b) -->

### 4.6 Two guardrails, and the false positives by task type.

Size is a gate and not a verdict. Surplus mass looks like a perfect detector on the released Core tier, 1.000, and falls to 0.448 once honest agents get a harmless edit of the same size, so mass only decides whether to look further. Direction says which and not whether. An agent that ignored a note points the same way as one that used it, so attribution comes from direction and uptake from the count and the jump. Under one definition across sweep2 the gate fires on 0 of 109 records readers, 0 of 113 data readers and 0 of 38 code readers, the confirmation on 4 of 259 readers, all data readers with counts under 6, so the two parts never fail together, and the 16-token window false-alarms on 14 of 260 readers who quote a note, which is why the window locates and does not score. Every positive is a code task, so the data and records rates are false-positive rates without matched positives. Confirmatory. <!-- src: results/lane_f/false_positives_by_task_type.md, results/OVERNIGHT.md phase 0 size control -->

### 4.7 Outside the game.

Two external checks are running at submission time under the rule of Section 3.6 and the frozen fitted model exported from the sweep2 fit, chosen because each passes the fairness rule of Section 3.7, every sample carries a masked span and only some agents act on it. <!-- src: results/paper/tables/fitted_model.json -->

- The insider-trading pressure eval (Scheurer et al., 2023) on Qwen3-14B. Sally's two tip messages are the peer span, the eval's own label says whether the agent traded on the tip, and a judge says whether it concealed that in its report. It tests the count and the jump at the merger content, and the concealment direction on a second environment. [SLOT insider trading. File results/external/ on branch claude/charming-wilson-08bb71. Report n runs, traded on the tip, flagged by the rule and by the frozen fit, false positives among runs that did not trade, and the concealment direction if the counts allow.]
- A TaskTracker-shaped set, question-answering passages with a mechanically injected target sentence and a label of whether the target string appears in the reply. It tests the count and the jump on a different distribution with thousands of negatives. [SLOT TaskTracker-shaped. File results/external/ on branch claude/clever-mayer-c077dc. Report n, flagged, false positives, and the frozen fit's AUROC.]

Either check that does not finish before the deadline is reported as not run.

