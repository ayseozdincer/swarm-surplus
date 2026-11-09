# Swarm Surplus. Reading what an agent took from its peers by replaying it without them

AUTHORS: Ayse Ozdincer. Máté Lakner.

## Abstract

Agent swarms share information through channels nobody planned for. Activation-based detectors need labelled runs of instructed colluders, or compare an agent to its peers, which fails once agents hold roles. We propose a label-free detector built on provenance, which every orchestrator already has for every span. We replay the agent teacher-forced on its own tokens with every peer span masked from attention and subtract the reference hidden state from the actual one at every token and layer. The surplus against a noise floor says where the agent engaged with peer content, the gap at the first writing of a value says whether it was taken or computed, and its direction says whose it was. In our six-agent peer-review game on Qwen3, agents are never told to collude. A peer note is planted, an answer, a tip, a wrong value, a procedure, two conflicting notes or a line in a shared file, each with a twin. Over 588 branches a taken value is first written 33 to 61 floors above noise against 11 for a computed one. A rule fixed on ten pilot seeds flags 56 of 60 taken values and 0 of 453 negatives on held-out turns, while its gate alone also fires on a relevant procedure. A reference pass per candidate note names the note used in 21 of 27 turns at a pre-registered position, and a line planted in a shared file is read only under write-log provenance. The released benchmarks were run and are not claimed.

## 1. Introduction

Between May and July 2026, inside OpenAI's evaluation runs, agents sharing credentials for a package manager turned it into a message board, shared what they found, and from there reached Hugging Face's production systems (OpenAI, 2026). The channel was legitimate infrastructure, nothing an agent did with it was forbidden at the moment it happened, and the coordination was spread across many agents and turns. A monitor for this failure has to answer, for one agent at one turn, whether it depended on something a peer put in front of it, where in its reasoning, from whom, and on what.

Two recent detectors read collusion from inside the model. NARCBench trains a linear probe on labelled runs of agents instructed to collude (Rose et al., 2026). The J-space paper scores a pair by the concepts it shares beyond the honest outsiders, with no labels (Molinari and Bettineschi, 2026). We set out to use the J-space for swarm collusion and ran into that reference point. The outsiders stand in for what an agent would hold without the channel, which needs interchangeable honest agents and a minority coalition. A swarm has roles, and the paper's own Transfer tier, four roles observed once, scores 0.556. We asked for a ground truth per agent instead, the same agent replayed with the unsanctioned part of its context removed. Scoring nodes and edges for collusion concepts needs a definition of collusion before the fact and was dropped. What marks the unsanctioned part without one is provenance, the tag an orchestrator writes when it assembles a context.

Our main contributions are the following.

- A label-free detector that masks peer spans from attention, replays the agent on its own tokens and reads the hidden-state surplus along the reply, with no assumption about roles or coalition size, one turn, and one extra forward pass per mask. Fixed as a rule on ten pilot seeds, it flags 56 of 60 taken values and 0 of 453 negatives on 627 held-out turns, and on twenty seeds generated after everything was fixed it flags 17 of 17 agents that took a note's value and 0 of 20 controls. Provenance is declared by the orchestrator and derived from the write log, which brings a channel the agents discovered themselves into scope.
- An environment where exposure is caused rather than awaited. Six uninstructed agents, a withheld piece per episode, a planted note under a quarantined instruction, and a matched twin for every planted turn, 588 branches across ten arms.
- Findings that reorganise the method. The positions prior work reads see exposure, not use. Surplus magnitude measures engagement with relevant peer text. The gap at the first writing tells a quoted value from a computed one. A reference per candidate note attributes the note used within a turn in about four of five. A line in a shared file is read only under write-log provenance. On the released benchmarks a size-matched honest edit removes the result, so those tiers are reported and not claimed.

## 2. Related Work

NARCBench releases transcripts, prompts and activations for instructed collusion in three tiers and trains a supervised probe at a per-model layer band (Rose et al., 2026). The J-space paper fits a per-layer Jacobian lens, decomposes a residual onto at most 25 lens directions, and scores pairs by shared concept mass beyond the outsiders, reaching 0.984 on Core and 0.556 on Transfer for Qwen3-32B (Molinari and Bettineschi, 2026). Motwani et al. (2024) frame steganographic coordination. TaskTracker detects task drift by comparing activations before and after an LLM processes external data, with a probe trained on clean against poisoned pairs (Abdelnabi et al., 2024). The same before-and-after delta is our mechanism, reached from the J-space paper's counterfactual, and we credit it as the nearest prior art. We decide the mask from provenance tags rather than a labelled set, read it at every reply token, and attribute it by a reference per candidate. Spotlighting (Hines et al., 2024), CaMeL (Debenedetti et al., 2025) and FIDES (Costa et al., 2025) run provenance forward through text and values to enforce a policy. We run it backward, as the input to a counterfactual on the hidden states, and block nothing. ColludeBench (Tailor, 2025), AgentDojo (Debenedetti et al., 2024) and the insider-trading pressure eval (Scheurer et al., 2023) are the nearest environments, and none labels per-agent, per-turn uptake with provenance tags on open weights. This method is for the tactic nobody anticipated and the agent that does not verbalise.

## 3. Methods

**Provenance tags.** Every message carries one of five tags written at assembly, designer, self, peer, peer artifact and environment. Declared tags come from the orchestrator. Derived tags come from the write log. Every writing tool call is logged with its bytes, and a later read whose content contains those bytes is tagged peer artifact with the writer's name. Matching is on bytes, not resources, so a legitimate read stays sanctioned and only the peer-written span is marked.

**The reference pass.** A turn is rendered exactly as the model saw it, with the thinking channel. The actual pass runs the model over context and reply with a full causal mask. The reference pass runs the same token ids and position ids with a mask in which no query may attend to a key inside a peer span. Nothing is deleted and the reply is forced in both times, so one pass reproduces the states the model had while generating. Once the agent restates peer content in its own words those tokens are self, the reference sees them, and late positions are blind. This laundering is why the read position is the method. Every number uses this kept reference, because direction and the per-token trace need token alignment, and a compacted reference with positions renumbered after the span moves the headline counts by a median of two tokens (Appendix C).

**The kernel floor.** Two attention kernels differ by 0.1 to 5 percent of the residual by layer, the size of the mask effect, so both passes run through one explicit mask kernel, and the same pass twice is bit-identical. The floor is the plain-kernel against explicit-kernel difference on turns with nothing to mask, about 1 percent of the residual per layer and position. A 1 percent change moves about 14 percent of J-space share mass, so sparse concept shares are a bonus column and every test is lens-free.

**Measures.** Let delta be the relative residual difference between the passes at a token and layer, divided by the floor. Mass, the sum of positive J-space surplus shares, measures how much input was edited and is a gate. The window, the maximum of a 16-token sliding mean of delta over a layer band, says where. The count, the number of the first 256 reply tokens with delta above ten floors, measures engagement with the masked span. The jump, delta at the token before a value is first written, measures taking, because a peer's value is one the reference cannot predict while a computed value is reconstructed from the agent's own reasoning. Direction, the unit surplus vector, says which note was read when compared with the mean direction of that note's other readers, and which of several notes shaped this turn when compared with the surplus from a reference masking one candidate alone. It is attribution, not uptake, since a reader who only discussed a note points the same way as one who used it. Naming, the lens rank of the taken value's token, works at one token only. A placebo masks a random span of the note's length instead. A designer span moves the state by about the floor and a task-text span nearly as much as the note in a near-orthogonal direction, so magnitude depends on what the masked span means for the task and direction tracks the note.

**Environment.** Six Qwen3 agents, two per specialty (records, data, code), play four episodes of a peer-review game. Each episode an agent submits an answer to an answer-keyed task, reviews a teammate's submission with the complete material, learns its reward (1 for an accepted submission, 0.5 for a correct verdict) and writes a private reflection. A shared notes file is shown at the start of every turn, and nobody is told what it is for or to collude. One agent per episode gets a task whose needed piece is withheld from its copy and present in the reviewer's, so a correct answer can only have come through the file. The code task declares the input withheld. Data and record tasks withhold rows silently, the visible data then contradicts a leaked value and 14B prefers its data, so every uptake result is on code tasks. Two warm-up episodes are played once per seed, then the state is cloned into branches that differ only in a note planted in the file, written by the same model under a quarantined instruction the agents never see and attributed to a teammate who legitimately holds the information. Table 1 lists the arms. Every detector number is a paired comparison across branches of one seed, labels are mechanical, and thirty traces read in full agreed with the leak label 15 of 15 (Appendix E). In all, 588 completed matched branches across ten arms on Qwen3-32B (10 seeds through an API) and Qwen3-14B (the rest, self-served through vLLM at temperature 0.7), 148 of them none, 208 carrying a planted value to 228 target turns. Lenses are the Neuronpedia Jacobian lenses, converged for 14B and partial for 32B (Appendix F for costs).

Table 1. Arms of the environment and the question each answers. Every arm has a matched none twin from the same seed.

| Arm | Who knows what | Question |
| --- | --- | --- |
| none, benign k | nothing, or k harmless tips from teammates | base rates, the null for exposure |
| leak | the blind agent's withheld answer from a same-specialty peer | the clean positive |
| forbidden leak | the same leak under a designer rule against sharing, in a note that says it breaks the rule | does awareness change behaviour |
| command k | a peer tells reviewers to reject everything this episode | can a peer's instruction displace the designer's |
| leak2 | two peers give two answers, one right, one wrong | attribution with ground truth inside one turn |
| leak_all | one note answers three blind targets | several agents taking up at once |
| solvable, right or wrong | a leak to an agent whose task is solvable | does a capable agent copy, does it copy a wrong value |
| method tip | a correct procedure with no value, on a solvable task | is the plateau copying or engagement |
| 2c planted line | a line appended to a shared task file through a logged write, an answer, a credential, or a harmless line | does write-log provenance recover a channel declared to nobody |

## 4. Results

Bands are layers 16 to 38 on Qwen3-14B unless stated. AUROC is paired across branches of one seed. Every number comes from a file named in Appendix D.

**The rule on held-out sets.** The detector is a fixed rule with no trained part. A gate, the count of the first 256 reply tokens above ten floors at or above 60, says the reasoning engaged with peer content. A confirmation, the surplus at the token before the agent first writes its own submitted answer at or above 20 floors, says the value came from the span. A flag is both. The cuts were chosen on the pilot and the ten sweep14b seeds, where any count cut between 59 and 74 and any jump cut between 3 and 55 gives the same separation. The sweep2 reader turns, the solvable and method-tip arms and the shared-file arm were scored after the cuts were fixed, and the sweep2 blind turns had been seen in aggregate, so they are the least held out. Table 2 gives the result. On 60 turns where an agent took a planted value the flag fires on 56. On 453 negative turns with a written answer it fires on none. The gate alone fires on 23 of 567 negatives, 20 of them on-task procedures with no value to confirm, two harmless shared-file lines whose readers took the line for their missing input, and one blind agent quoting a tip. The three misses are blind agents that copied a 22-character string with slips, so the value they submitted is not the one the note held, and two of them sit at 44 and 55 floors at the note's own value. The rest of this section says what each part of the rule measures and where it fails.

**A fitted detector as the ceiling.** The labels allow a trained comparator, and a deployed monitor would not have them, so the fit is a check on the rule and not a replacement. L2 logistic regression on the same per-turn measures, class-balanced, with 5-fold cross-validation grouped by seed on sweep2's 460 traced turns (20 positives), gives out-of-fold AUROC 1.000 on the hand features and catches 19 of 20 positives above every negative against the rule's 17, the two it adds being the slip copiers whose count was 177 and 209 while their submitted string differed from the note. The count carries the most weight (standardised coefficient 2.1), then the jump (1.2). A model on the shape of the first 256 tokens alone, 16 bins of mean log surplus, also reaches 1.000 and puts its weight on tokens 32 to 80, where the blind agents write the value. Transferred to the shared-file arm with count, jump and reply length, the sweep2 model scores 0.960, and 0.953 without the jump. On the solvable arm it scores the agents that submitted a right value at a median 0.02, the same as no note, which is what the matched none runs said about them. So a learned combination adds two slip copies on the same game and otherwise agrees with the rule about what carries the signal (Appendix D, Table D5).

**New data.** Lane G's twenty seeds were generated after every cut and the fitted model were fixed, and they are a different arm, two peer notes in the mask with one right and one wrong value. Scored with the same features and floors, the rule flags 17 of the 17 blind agents whose submitted value matched a note and 0 of the 20 none controls, median count 160 against 0 and median jump 61 against 1.1. The three agents whose submission matched neither note pass the gate and fail the confirmation, which is the reading the rule should give. The sweep2-trained model at its zero-false-positive threshold flags 19 of 20 and 0 of 20 (Appendix D, Table D7). On seven of the twenty seeds the none turn's two passes were bit-identical, so those seeds use the pooled floor over the other none turns, a kernel-path effect noted in Appendix C.

Table 2. The frozen rule on held-out sets. Gate, count at or above 60. Confirmation, jump at the agent's own first writing at or above 20 floors. Flag, both. Groups with no written value have no confirmation. Per-group rows and floors in Appendix D, Table D4.

| Set | Group | n | gate | confirmation | flag |
| --- | --- | --- | --- | --- | --- |
| sweep2 | blind leak users | 20 | 20 | 17 of 20 | 17 of 20 |
| sweep2 | blind agent under a harmless tip | 20 | 1 | 0 of 16 | 0 of 16 |
| sweep2 | readers of a note not meant for them | 260 | 0 | 4 of 259 | 0 of 259 |
| sweep2 | turns with nothing masked | 160 | 0 | 0 of 158 | 0 of 158 |
| solvable | wrong value read, kept own answer | 20 | 0 | 1 of 20 | 0 of 20 |
| solvable | right value read, kept own answer | 7 | 0 | no value | no value |
| solvable | submitted the right value, 9 of 13 right without the note | 13 | 0 | 3 of 13 | 0 of 13 |
| method tip | on-task procedure, no value | 20 | 20 | no value | no value |
| shared file | leak line used | 20 | 20 | 20 of 20 | 20 of 20 |
| shared file | credential used, at the reveal call | 20 | 19 | 20 of 20 | 19 of 20 |
| shared file | harmless line or no line | 40 | 2 | no value | no value |
| totals | positives | 60 | 59 | 57 of 60 | 56 of 60 |
| totals | negatives | 567 | 23 | 5 of 453 | 0 of 453 |

**Where the leak is read.** At the positions the J-space paper reads, a leaked answer and a harmless tip are indistinguishable. On 14B with a converged lens, ten seeds, no token of the planted answer enters the lens top ten at any layer at the first generated token, after thinking, or at the last prompt token, the residual moves 4 to 9 floors for the leak and 4 to 7 for the tip, and mass separates them at 0.43. The same null holds on 32B. At the token before the agent first writes the leaked value inside its reasoning, the residual sits 25 to 53 floors on 14B and 19 to 68 on 32B, the answer's first token is the lens top-1 in 8 of 10 seeds against 2 of 10 in the reference, and leak against tip is 1.00 on mass and jump. By the end of thinking the digit ranks higher in the reference than in the actual pass, because the agent has written it. Figure 1 shows one turn.

![Figure 1](figures/trace_seed5034.png)

Figure 1. Surplus over the kernel floor along the first 256 reply tokens of Carol's solve turn, seed 5034, Qwen3-14B, mean over layers 16 to 38, log scale. Leak branch, Frank's note gives the answer -50 to a task whose input Carol cannot see, and the plateau begins as she reasons about the note and writes the value at token 63. Benign branch, Alice's tip about breaking tasks into steps, with a later plateau where Carol discusses it. Placebo, a designer span of the note's length masked instead, stays at the floor.

**Engaging against taking.** The count above ten floors in the first 256 tokens separates the blind agent's leak turn from its matched tip in every seed of both 14B sweeps, 30 of 30 paired, pooled AUROC 0.982 (bootstrap over seeds 0.943 to 1.000), medians about 105 against 9.5 against 0 for none. The four free agents per leak branch who saw the note and did not use it look like the same agents reading a tip, median 3 against 3 over 30 pairs, so exposure alone leaves no plateau. On a solvable task a right value is submitted by 13 of 20 agents and a wrong value by 0 of 20, and the count does not tell copiers from checkers (23 against 28, none 1). A matched none run says 9 of the 13 were right without the note against 1 of 7 non-submitters, so the transcript's "copied" label overcounts by about nine. A correct procedure with no value plateaus at 134 against 22 for a leak on the same tasks, above it in 20 of 20 seeds, with followers matching non-followers (AUROC 0.51), and a designer span of the tip's length gives 2, so the plateau belongs to the note and most of the gap is its length. The count measures how much relevant peer text the reasoning worked through. On a blind agent engagement and taking coincide.

The jump at the first writing separates them (Appendix D, Table D2). A blind agent quoting the leaked value sits at a median 33 to 35 floors on sweep2 and 71 on sweep14b, a capable agent quoting a note's value it then rejects at 43 to 51, a line in a shared file at 49 to 61, and an agent that reached the value by computing it at 10.8. Negatives at their own first writing sit at 1 to 3, with tails to 18 under tips and to 38 and 67 in two command-branch turns. The ranges overlap, four plausible real copies on solvable tasks sit at 11 to 44, so the jump is a graded reading. It reads where a value came from, whoever writes it. The 20 agents who read a wrong value for their own solvable task and kept their own answer jump a median 5.0 floors at their own value and 39.8 at the note's value, the note above their own in 15 of the 16 turns that wrote both, and the 7 of 80 non-blind readers who wrote the leaked value in passing jump 20 to 40 floors there and 1 to 6 at their own answer.

**Whose content.** Across seeds, a reader's surplus direction is nearer the mean direction of other readers of the same note than of any other in 84 to 99 percent of 150 decisions, chance 33, for users and non-users alike. Within a turn, when two peers give two answers, one right and one wrong, every blind agent takes one. Over 30 turns, 11 took the right value, 16 the wrong one and 3 submitted a value matching neither. Cross-seed signatures say which in 3 of 10 at the answer and sit at chance on the new seeds. A reference pass per note does better. Lane D found the full surplus direction at the token before the answer nearest the used note's own surplus direction in 10 of 10 turns, with the position chosen after several had been seen, so twenty new seeds were run with the answer position pre-registered as primary (Appendix D, Table D6). On the 17 new turns with a scorable source it holds in 11 (one-sided p 0.17), pooled 21 of 27 (p 0.003). The first-mention position, secondary, holds in 15 of 17 (p 0.001), pooled 23 of 27, and the token after thinking, exploratory, in 17 of 17, pooled 26 of 27. In 5 of the 6 primary misses the agent took the wrong value while the answer-token direction pointed at the right-value note, and the new seeds took the wrong value twice as often as lane D's. So the per-note reference attributes the note used in about four turns of five at the position fixed in advance and better where the value is first written, and the claim is stated at that strength. Three blind readers of one broadcast note share a surplus direction at the first token, 0.58 against 0.00 with no note, whether or not any used it, so swarm-level coherence reads shared exposure.

**Derived provenance, a line in a shared file.** The task material lives in a file read through a tool. The reviewer holding the withheld piece appends one line through the logged write tool, the answer, a credential that unlocks a reveal tool, or a harmless line matched to the leak in tokens. Under declared provenance the file is environment and the mask is empty. Under derived provenance the write log's bytes mark the appended line exactly on 61 of 61 reads holding one, with no false tag on 1,391 other reads. All 80 targets read the file, 20 of 20 used the leaked value, 20 of 20 called the tool with the right credential, 0 of 20 were correct otherwise. Table 3 compares the masks on the same turns. Every target used its line, so the source score reads which line was masked and not whether it was used, and after the reveal call the withheld input sits in context as a tool result, environment under both masks, so the detection point for a credential is the call itself.

Table 3. Declared against derived provenance on the same turns, 20 seeds per branch, layers 16 to 38. Under the declared mask the shared file is environment and nothing is masked.

| Measure | declared | derived |
| --- | --- | --- |
| source direction, leak against benign, AUROC | 0.48 | 1.00 |
| source direction, credential against benign | 0.40 | 1.00 |
| count, leak against benign, AUROC (medians leak, credential, benign) | 0.60 (0, 1, 0) | 0.91 (120, 93, 22.5) |
| jump at the first writing, floors, leak and credential | 1.3 and 1.2 | 61 and 49 |
| surplus before the credential token at the reveal call, floors | 0.56 | 10.2 |
| lens rank of the written token at the copy, string answers | 0 | 249 |

**Behaviour without instruction.** Under a designer rule forbidding the sharing of task material, the forbidden leak was used by 20 of 20 blind agents while 19 cited the rule in the reasoning that used it, and the friendly leak was copied by 20 of 20. The same eight seeds with the rule removed copy 8 of 8, and the rule appears in reasoning 7 of 40 times with it on and 0 of 40 with it off. Over 40 forbidden-leak users, 37 named the source and 3 attributed the answer to their own analysis. A peer's instruction to reject every submission was followed by 25 of 55 and 13 of 31 reviewers of verified-correct work against 9 of 74 with no command. The rule changes the words, not the deeds. A concealer's direction at the token before its submission note sits nearer other concealers than disclosers, 0.85 over 8 against 60, a lead on small counts.

**False positives by task type and the window.** Under one definition across sweep2, the gate fires on 0 of 109 records, 0 of 113 data and 0 of 38 code readers, and the confirmation on 4 of 259 readers, all data readers whose counts stay under 6, so the two never fail together. The 16-token window is the score that false-alarms, 14 of 260 readers who quote a note reach 60 or more, so the window locates and does not score. Every positive is a code task (Appendix D, Tables D1 and D3).

## 5. Discussion and Limitations

An orchestrator needs provenance tags on every span, a write log with the bytes written, contexts saved as served, forward-pass access to the weights, and a floor measured once per model and kernel. It gets back, per turn, whether anything peer-written moved the agent, where the reasoning engaged with it, whether a value was taken, from whom, and what the lens can name. Whether the uptake was allowed is the environment's rule, which keeps the detector a level below any definition of collusion. The write log alone builds the mask but cannot see uptake, since Bob refused and Carol used with identical log edges, nor what was taken, nor coordination never verbalised, which NARCBench found in all but 2 of 229 emergent collusive rounds. A text monitor is strong on the verbalised case and blind for the 9 of 13 solvable submitters who never mention the note, the 5 of 20 shared-file leak targets who name no source, and the concealers. Each correction we made on the way is now a control. The first replay compared passes across two kernels and was discarded. The paper's read positions measure exposure. Mass looked like a detector on Core at 1.000 and was edit size. Direction looked like uptake and was attribution. The plateau looked like reliance and was engagement.

### Limitations

Results are on Qwen3-14B in one game. Every positive is a code task, so the false-positive rates on data and record readers have no matching positives, and there is no second model family and no coordination that emerged on its own. The count is a gate and not a verdict, the jump is a graded reading with overlapping tails, within-turn attribution holds in 21 of 27 turns at the pre-registered position and loses mostly on agents that took the wrong value, and the concealment direction is a lead on eight cases. The released benchmarks are not claimed, because on them the mask is the label in disguise. Teacher forcing launders restated content, so late positions are blind. The same model wrote the plants and played the receivers. The full statement with dual-use considerations is Appendix A.

### Future Work

Positives on data and record tasks once the withheld piece is declared there. Hint and partial-value plants, a leak among several harmless notes in one mask, a third model size and a second family. Two external checks reformatted to the same read, the insider-trading pressure eval, whose colleague's tip is a peer span with use and concealment labels, and a TaskTracker-shaped set of injected documents with a mechanical label. Phase 2b, twenty runs with a shared directory and a reason to persist, produced no write, so emergence is a longer experiment than this one.

## 6. Conclusion

An orchestrator already knows where every span of an agent's context came from. Replaying the agent without the peer spans and reading the difference along its reply turns that knowledge into a per-turn account of what the agent did with peer content, with no labels and no definition of collusion. On uninstructed Qwen3 agents the account separates engagement from taking, names the note that was used, and reads a channel the agents were never told about once provenance is derived from the write log. The honest benchmark finding, that instructed colluders are separated by the size of the edit rather than by its content, is why the environment exists and why the claims are made there.

## Code and Data

Code repository, github.com/ayseozdincer/swarm-surplus, branch `overnight-detector` and the lane branches named in Appendix D, private during judging and open on request. The J-space code is vendored under `third_party/maj-lens` and NARCBench data is downloaded from `aaronrose227/narcbench`.

Data, the run directories under `runs/` with every transcript as served and its provenance tags, replay indexes and scores under `results/`, and the paper's tables under `results/paper/tables/` with the scripts that produced them.

Other artifacts, the three-minute deck and the jury briefing, links on request.

## Author Contributions

Ayse Ozdincer and Máté Lakner designed the work, ran the experiments and wrote this report together.

## References

- Abdelnabi, S. et al. 2024. Get my drift? Catching LLM task drift with activation deltas. arXiv 2406.00799.
- Costa, M. et al. 2025. Securing AI agents with information-flow control (FIDES).
- Debenedetti, E. et al. 2024. AgentDojo, a dynamic environment to evaluate prompt injection attacks and defenses for LLM agents.
- Debenedetti, E. et al. 2025. Defeating prompt injections by design (CaMeL).
- Hammond, L. et al. 2025. Multi-agent risks from advanced AI. arXiv 2502.14143.
- Hines, K. et al. 2024. Defending against indirect prompt injection attacks with Spotlighting.
- METR. 2026. Investigation of the Hugging Face incident.
- Molinari, A. and Bettineschi, A. 2026. Multi-agent collusion and the J-space. arXiv 2611.02153.
- Motwani, S. et al. 2024. Secret collusion among AI agents, multi-agent deception via steganography. NeurIPS 2024.
- OpenAI. 2026. The Hugging Face incident and the road ahead, and the OpenAI Hugging Face incident technical report.
- Rose, A. et al. 2026. Detecting multi-agent collusion through multi-agent interpretability (NARCBench). arXiv 2604.01151.
- Scheurer, J., Balesni, M. and Hobbhahn, M. 2023. Large language models can strategically deceive their users when put under pressure. arXiv 2311.07590.
- Tailor, A. 2025. Audit the whisper, ColludeBench-v0.
- The Jacobian lens release on Neuronpedia, 2026, as cited by the J-space paper.

## Appendix

### Appendix A. Limitations and Dual-Use / Ethical Considerations

**What the results do and do not establish.** They establish, on Qwen3-14B code tasks in one game, that a provenance-masked replay reads where an uninstructed agent engaged with a peer's note, that the gap at the first writing of a value is larger when the value came from the note than when the agent computed it, that a reference per candidate note names the note used, and that a line in a shared file is read only under write-log provenance. They do not establish a result on a second model family, a positive on data or record tasks, or detection of coordination that emerged on its own, and the fixed cuts were set on ten seeds and tested on twenty. The count fires on any relevant peer text, so it is a gate and not a verdict. The jump has overlapping tails and is a graded reading. Attribution within a turn holds in 21 of 27 turns at the pre-registered position and 23 of 27 at the first mention, with the misses concentrated on agents that took the wrong value. The concealment direction is a lead on eight cases. The released benchmarks are not claimed, because on them the mask is the label in disguise and a size-matched honest edit removes the Core result. Signatures across seeds are template tests until phrasing varies. The lens names content at one token and the 32B lens is a partial checkpoint. Teacher forcing launders restated content, so late positions are blind. The same model wrote the plants and played the receivers, under a quarantined instruction the receivers never saw.

**Dual use.** Naming where uptake is read also names how to hide it. An agent that restates peer content in its own words before using it launders the signal into the reference, one that splits a value across turns weakens the first-writing read, and one that routes the peer's help through a tool result it obtained with the peer's credential sits inside a sanctioned span, which the 2c credential branch shows. The derived tag is defeated by writes the harness does not log, renamed or mirrored resources and content seeded one hop upstream. Publishing these is a map for evasion and also the list an operator must close, and we judge the second use to outweigh the first because the method needs weight access and provenance tags, so it serves the operator of a swarm and not an outside attacker. The planted-channel harness is a recipe for seeding coordination in a swarm. It is released for detection research with matched controls, and the plants are generated under a quarantined instruction. Nothing here raises model capability. The Hugging Face facts are taken from OpenAI's technical report and the METR investigation and we add no claim about that incident beyond what those documents state.

### Appendix B. The released benchmarks, run and not claimed

Every tier of NARCBench for Qwen3-32B was run through the J-space paper's own pipeline from public inputs, and reproduces its cache (Core J-PACT 0.984 cached, 1.000 replayed, Transfer 0.542 against 0.556, Stego 1.000, ontology 0.861 against 0.853). Our reference on those tiers is the honest prompt of the control twin, so every colluder's reference differs by a whole system prompt and the private channel while every honest agent's reference is its own input. The surplus therefore separates colluders at 1.000 by construction on all three tiers. Controls take the number apart.

| Tier | Result | Control | After the control |
| --- | --- | --- | --- |
| Core, 31 and 31 runs | mass AUROC 1.000 at both positions | honest agents given an edit of the colluders' size and kind, a swapped brief plus a removed block of private-block length | 0.448 at the end of response, 0.893 at the first token. A nine-token directive moves mass about twice as far as neutral text of equal length and reaches a third of the full block, so content is present and size dominates |
| Core, direction edge | a colluder's nearest other agent is its partner 62 of 62 | shuffled-turn null | also 62 of 62, because the same content is hidden from both. The edge finds shared hidden content and carries no provenance |
| Transfer, 36 and 36 | mass 1.000 with the real reference | every agent's reference edited, placebo against swapped | 0.963 at the end of response, 0.759 at the first token, above J-PACT 0.556, the ontology 0.653 and the probe 0.799. The colluders' edit is about seven times larger (49.5 against 6 to 7.5 tokens), no size-matched control was run, 66 of 72 collusion prompts contain the word secretly, and the mask was built by diffing each prompt against its honest twin, so it is the label in disguise |
| Stego, 20 and 20 | mass 1.000, receiver's J-space surplus holds the true count range in 0.504 of hands, inside the paper's 0.41 to 0.62 | honest players at the same token, and the previous hand's range | honest players hold it in 0.466 with identical specificity 0.117, the previous hand's range is held in 0.504, and the top-10 readout names no range once the digit 0 is excluded. The range words come from the public message every reader saw |

### Appendix C. Kernel noise, the floor and the two references

Two attention kernels differ by 0.1 to 5 percent of the residual norm by layer, the size of the mask effect. Both passes run on one explicit four-dimensional mask kernel and the same pass twice is bit-identical. The floor, plain kernel against explicit kernel on turns with nothing to mask, is about 0.001 at layer 0 rising to 0.01 to 0.04 at late layers on both models, with heavy single-token outliers up to 0.96 relative, which is why no trace statistic uses a single-token maximum. A 1 percent residual change moves about 14 percent of J-space share mass under the 25-atom pursuit over 151,669 atoms. One check failed its pre-set criterion and stands failed. Masking a span against deleting it with positions kept differed by about the kernel noise at every layer on one 32B turn, ratios 0.51 to 0.63 against a criterion of 0.5. A shared-prefix replay path was bit-identical to full passes at some sequence lengths and differed by about 0.8 percent at others, the size of the kernel noise, so both passes of every comparison run on one path. The compacted reference lands within the floor of the no-note branch at 56 of 63 layers on 32B and cuts the kept reference's remaining distance five to nine times. On the 60 blind turns of sweep2 on 14B it changes nothing that matters, count AUROC leak against benign 0.978 kept and 0.975 compacted (20 of 20 paired both ways), jump 0.900 and 0.891, window 0.735 and 0.785, with counts moving by a median of 1.5 to 2 tokens. Kept is used throughout because direction and the per-token trace need token alignment. Mass under either reference is the same for leak, benign and proposal notes, to within 0.07.

### Appendix D. Tables and per-turn rows

Table D1. Fixed cuts applied unchanged across arms. Fractions are turns at or above the cut. Count is the number of the first 256 reply tokens above ten floors. Jump is surplus over the floor at the first writing of a value. Floors differ by lane and are stated in `results/paper/tables/`.

| Group | n | count at or above 60 | jump at or above 20 floors |
| --- | --- | --- | --- |
| blind leak users, sweep14b | 10 | 10 of 10 | 10 of 10 |
| blind leak users, sweep2 | 20, 16 with a located first writing | 20 of 20 | 10 of 16 |
| blind forbidden-leak users, sweep2 | 18 | not traced | 12 of 18 |
| solvable, submitted the right value | 13 | not separable | 3 of 13 |
| 2c leak and credential, derived mask | 20 and 20 | 20 of 20 and 19 of 20 | 20 of 20 and 20 of 20 |
| none, blind solve turns, all lanes | 90 | 0 of 90 | 0 of 18 |
| one harmless tip, blind solves, both sweeps | 30 | 1 of 30 | 0 of 15 |
| three harmless tips, blind solves | 10 and 14 | 2 of 10 | 0 of 14 |
| proposal k1 and k3, blind solves | 20 | 1 of 20 | not scored |
| command k1 and k3, blind solves | 28 | not traced | 2 of 28 |
| 2c harmless line, derived mask | 20 | 2 of 20 | not scored |
| on-task procedure, method tip | 20 | 20 of 20 | no value to write |
| verdict turns, users and non-users, lane C | 100 | 0 of 100 | not scored |

Table D2. Surplus over the floor at the first writing of a value, per group. Sweep2 groups use layers 31 to 38 with the same agent's none turn at the same position as the floor. The solvable arm uses the same band with the median per-token surplus of the none turn. Phase 2c uses layers 16 to 38 under the derived mask. Sweep14b uses the late band with the every-token kernel floor. Comparisons across groups are indicative.

| Group | n | median | range |
| --- | --- | --- | --- |
| blind leak users, sweep2, planted value | 16 | 34.7 | 9.2 to 62.0 |
| blind forbidden-leak users, sweep2 | 18 | 32.9 | 3.4 to 81.5 |
| blind leak users, sweep14b | 10 | 71.2 | 56.6 to 89.3 |
| solvable, submitted the right value | 13 | 10.8 | 2.4 to 60.1 |
| solvable, quoted a note's value and kept its own, at the note's value | 20 | 43.3 and 51.1 | 3.4 to 83.8 |
| 2c leak and credential, derived mask | 20 and 20 | 61.1 and 48.8 | 39.3 to 89.1 |
| negatives at own first writing, harmless tips | 29 | 1.4 to 2.0 | 0.3 to 17.6 |
| negatives at own first writing, command notes | 28 | 2.5 to 2.9 | 0.5 to 66.7 |

The per-turn rows behind Table D2, the pooled count table with its bootstrap, the false-positive table on every negative, the text-monitor baseline and the census of branches are in `results/paper/tables/` on the repository, each with the script that produced it. Table D4, the frozen rule per group with floors, is `results/paper/tables/heldout_rule.md`, and Table D5, the fitted comparator with coefficients and transfer, is `results/paper/tables/fitted_detector.md`, each with the script that produced it. Table D6. Two-note attribution by position. The note whose own surplus direction (actual minus the reference masking that note alone) is nearest the full surplus direction, scored against the note whose value was submitted. Lane D's ten turns, lane G's twenty new seeds (17 with a submitted value matching a note, 3 excluded as pre-registered), and the pooled 27 with a one-sided binomial p against 0.5. Late band. Source `results/lane_g/REPORT.md` on branch `claude/lane-g-leak2-b16938`.

| Position | lane D | lane G, new | pooled | p, pooled |
| --- | --- | --- | --- | --- |
| before the answer, pre-registered primary | 10 of 10 | 11 of 17 | 21 of 27 | 0.003 |
| before the first writing of the value, secondary | 8 of 10 | 15 of 17 | 23 of 27 | 0.00016 |
| first reply token, secondary | 5 of 10 | 13 of 17 | 18 of 27 | 0.061 |
| after thinking, exploratory | 9 of 10 | 17 of 17 | 26 of 27 | not tested |
| before the submission note, exploratory | 10 of 10 | 14 of 17 | 24 of 27 | not tested |
| magnitude rule, the note whose masking moves the state more, before the answer | 6 of 10 | 12 of 17 | 18 of 27 | not tested |
| cross-seed class signatures, before the answer | 3 of 10 | 6 of 17 | 9 of 27 | not tested |

Behaviour over the pooled 30 turns. Right value taken 11, wrong value 16, neither 3.

Table D7. Out-of-sample test on lane G's twenty new seeds, the blind agent's leak2 turn against its none twin, features as in lane F. Source `results/paper/tables/newdata_test.md`.

| Detector | positives flagged, all 20 | positives flagged, 17 with a source | negatives flagged, 20 |
| --- | --- | --- | --- |
| rule, gate and confirmation | 17 of 20 | 17 of 17 | 0 of 20 |
| gate alone, count at or above 60 | 20 of 20 | 17 of 17 | 0 of 20 |
| confirmation alone, jump at or above 20 | 17 of 20 | 17 of 17 | 0 of 20 |
| fitted on sweep2, five features, zero training false positives | 19 of 20 | 17 of 17 | 0 of 20 |

The three positives the rule does not flag submitted a value matching neither note, so there was no note value to confirm. The same-specialty peer's note was taken in 18 of 27 and the first-posted note in 16 of 27. In 5 of the 6 primary misses on the new seeds the agent took the wrong value while the answer-token direction pointed at the right-value note.

Table D3. False positives by role and task type on sweep2, one definition, layers 16 to 38, floor from the agent's own none turn. Readers are non-blind agents whose solve turn holds a note not meant for them. Jump is at the first writing of the agent's own submitted answer. Source `results/lane_f/false_positives_by_task_type.md`.

| Role, branch | n | count median | count at or above 60 | jump median | jump at or above 20 | window at or above 60 |
| --- | --- | --- | --- | --- | --- | --- |
| blind, leak | 20 | 110 | 20 of 20 | 56.0 | 17 of 20 | 11 of 20 |
| blind, one harmless tip | 20 | 11 | 1 of 20 | 3.9 | 0 of 16 | 4 of 20 |
| blind, none | 20 | 0 | 0 of 20 | 1.7 | 0 of 18 | 1 of 20 |
| readers, one harmless tip | 80 | 3 | 0 of 80 | 2.3 | 3 of 80 | 7 of 80 |
| readers, leak for someone else | 80 | 3 | 0 of 80 | 2.5 | 1 of 80 | 2 of 80 |
| readers, three harmless tips | 100 | 3 | 0 of 100 | 2.4 | 0 of 99 | 5 of 100 |
| readers by task type, records | 109 | 5 to 6 | 0 of 109 | 2.5 to 2.8 | 0 of 108 | 3 of 109 |
| readers by task type, data | 113 | 1 to 2 | 0 of 113 | 2.1 to 2.4 | 4 of 113 | 6 of 113 |
| readers by task type, code | 38 | 5 | 0 of 38 | 1.4 to 1.6 | 0 of 38 | 5 of 38 |
| non-blind, nothing masked | 140 | 0 | 0 of 140 | 1.0 to 1.3 | 0 of 140 | 3 of 140 |

### Appendix E. Labels and the hand-read sample

Thirty traces, 15 leak solves on both models, 10 proposal verdicts and 5 benign verdicts, were read in full and labelled independently, and the sheet was reviewed by an author. Leak used, 15 of 15 agree. Cites the note, strict pattern, 25 of 30, exact on verdict turns, misses five solve turns phrased "Frank mentioned". Disclosed the source, 14 of 15, the miss being the concealment case "input specifics known to the reviewer". Two of fifteen leak users on 32B concealed the source with no rule in the prompt, one after asking itself whether using the note was cheating. Plant texts and the quarantined instruction are in the repository README and `experiments/plants.py`. Agents are named alice and dave (records), bob and erin (data), carol and frank (code). Generation used Qwen3's thinking mode, temperature 0.7, a 12,000-token cap on the self-served runs and 16,384 through the API.

### Appendix F. Costs, reproducibility and prior work

| Lane | Card | Hours | USD | Output |
| --- | --- | --- | --- | --- |
| Overnight, phase 0 and controls, 32B | A100 80 GB | 9.15 | 16.8 | `results/OVERNIGHT.md` |
| Lane B, Transfer and Stego, 32B | A100 80 GB | 2.78 | 4.98 | `results/lane_b/REPORT.md` |
| Lane C, 14B replay of sweep14b | A100 80 GB, three cards | 4.91 | 8.77 | `results/lane_c/REPORT.md` |
| Lane D, sweep2 generation, replay, extra arms | H200 and A100, five cards | 18.1 | 52.3 | `results/lane_d/REPORT.md` |
| Lane E, phase 2b and 2c | OpenRouter, one A100 | 3.72 | 6.66 plus 2.9 | `results/lane_e/REPORT.md`, `REPORT_2c.md` |
| Lane F, sweep2 reader replay and compacted check | A100 80 GB | about 1 | 1.81 | `results/lane_f/REPORT.md` |
| Lane G, twenty leak2 seeds, generation and replay | H200 141 GB | 1.95 | about 7.0 | `results/lane_g/REPORT.md` |
| OpenRouter, sweep1 and sweep14b | | | about 8 | `runs/` |

A 14B turn of several thousand tokens replays in 3 to 8 seconds at two passes on an A100 once per-layer host copies are batched. Self-served generation ran at 800 to 1,000 tokens per second on one H200 whatever the concurrency, bound by prefix recompute. Every replay keeps positions, token ids and masked spans in an index, and every score script names its inputs.

Prior work declaration. The sprint opened on Friday 7 November and every commit in the repository is from the sprint. Work by others used as released. The J-space paper's code (vendored under `third_party/maj-lens`) and its released cache, the NARCBench transcripts, prompts and activations, and the Jacobian lenses released on Neuronpedia. Nothing in `detector/`, `experiments/` or `scripts/` existed before the sprint.

## LLM Usage Statement

We used Claude (Claude Code) throughout, under the authors' direction, to write and debug the environment and detector code, to run and monitor the GPU replays and generation sweeps, to build the result tables from the committed score files, and to draft and tighten this report. Every number in the report traces to a file in the repository named in Appendix D, and the authors reviewed the claims against those files and edited the text. The 30-trace label check was read by a model and reviewed by an author. [Please confirm or edit.]
