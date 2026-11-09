# Paper plan. Gap list, outline and costed experiments

Written 8 November 2026 from the abstract draft, the idea history, the jury briefing (36 screens), the lane B, C, D and E reports, `results/OVERNIGHT.md`, the project notes, and the committed score files. Numbers marked "computed today" were recomputed on this machine from the committed JSONs and are reproducible with no GPU. Nothing here spends money. Part C waits for a yes.

## A. Claims against evidence

Status key. **Solid** means the number is in a committed file at the stated n and the claim is worded to match it. **Caveated** means the evidence exists but the public wording outruns it or a control is missing. **Open** means no file supports it yet.

| # | Claim as made (abstract, deck, briefing) | Evidence, where, n | Status | Gap | Closes with |
| --- | --- | --- | --- | --- | --- |
| 1 | Label-free detector, one extra forward pass per turn, nothing read from a peer span to decide the mask | Method, by construction. Replay 3 to 8 s per 14B turn (lane D) | Solid | Per-candidate attribution costs one pass per candidate, and the derived mask is a second mask. Say so. | Write-up |
| 2 | Surplus length against a measured noise floor says how far peers moved the agent and where in its reasoning | Lane C window lands within 19 tokens of the first mention 10 of 10 (`results/lane_c/window.json`). Lane D sweep2 lands on the first mention 12 of 20, otherwise on a quote of the note (`sweep2_window_count.json`) | Caveated | "Where" is the discussion of the note, not always the copy. Floors differ by lane (every-token kernel floor in lane C, none-turn median in lane D) | Write-up, plus the unified rescoring in E1.1 |
| 3 | Plateau count separates a blind agent that took a leak from its matched tip, 20 of 20, AUROC 0.97 over 30 seeds | Lane C 10 seeds 1.00, lane D 20 seeds 0.97. **Computed today, pooled 30 seeds, count above 10 times the floor in the first 256 reply tokens, leak against benign_k1, AUROC 0.982, bootstrap 95 percent interval 0.943 to 1.000, paired wins 30 of 30** | Caveated | The matched tip is off-task (a records tip to a code agent). An on-task harmless procedure (method_tip) plateaus longer than the leak in 20 of 20 seeds. **Computed today, blind leak counts against method_tip counts, AUROC 0.42.** So the count reads engagement with relevant peer text, not leak use. The abstract leads with the number that has this failure mode | Write-up. Lead with the jump, present the count as the engagement gate with the tip failure stated in the same sentence. E0.3 gives the false-positive table |
| 4 | The gap at the first writing of a value separates a quoted value, 19 to 68 floors, from a computed one, under 11 | **Computed today from committed rows.** Sweep2 blind users, planted jump, late band, n 16, median 34.7, range 9.2 to 62.0. Forbidden, n 18, median 32.9, range 3.4 to 81.5. Solvable copiers, n 13, median 10.8, range 2.4 to 60.1. Solvable quoted then rejected, right note n 4 median 43.3, wrong note n 16 median 51.1, range 3.4 to 83.8. 2c leak n 20 median 61.1 (47 to 89), credential n 20 median 48.8 (39 to 54). Negatives at the agent's own first writing, benign_k1 n 15 median 1.4 max 11.6, benign_k3 n 14 median 2.0 max 17.6, command_k1 n 14 median 2.5 max 38.4, command_k3 n 14 median 2.9 max 66.7 | Caveated | Ranges overlap. "Under 11" is a median, four plausible real copies sit at 11 to 44, and two command-branch negatives exceed 38 at their own first writing. Floors and bands differ by lane, so the per-seed table across arms cannot yet be built with one definition. The "small gap at its own value, large at the note's" picture rests on one trace (seed 5021) | E0.2 per-turn table with per-lane definitions and a threshold sweep now. E1.1 rescoring with one floor, one band, plus own-value first writing for the 20 wrong-note readers |
| 5 | Direction attributes the used note in 10 of 10 turns where cross-seed signatures manage 3 of 10 | Lane D leak2, n 10, answer_pre 10 of 10 (p 0.001), mention_pre 8 of 10, first token 5 of 10, signatures 3 of 10 at the answer (`leak2_uptake.json`, `arms_scores.json`) | Caveated | One arm, ten turns, the winning position chosen after seeing several. Signatures across seeds are template tests | E1.2, twenty more leak2 seeds with answer_pre pre-registered as primary |
| 6 | Nearest-signature attribution 84 to 99 percent against chance 33 | Lane C 150 decisions on verdict turns, 0.84 to 0.99 at first_thought; 32B pilot 0.80 to 1.00 on 15 | Solid, with the template caveat | Sweep2 varies phrasing and broadcasts half the notes and gives source 0.88 to 1.00, so the caveat is partly answered | Write-up |
| 7 | The J-lens names what was taken, at the token before the copy | Lane C 14B, answer token top-1 at L38 in 8 of 10 actual against 2 of 10 reference, J-space surplus 9 of 10. Sweep2 19 of 19. 2c rank 0 against 225 to 295 for strings and tokens, 1 for numbers | Solid at one token | Junk concepts at the response start on both lenses. 32B lens is a partial checkpoint. Naming fails for digits and at the window location | Write-up, state as a bonus column |
| 8 | A line planted in a shared file is caught only under write-log provenance, source 1.00 against 0.48 | Lane E 2c, 20 seeds, declared against derived on the same turns, every metric (`planted_scores.json`) | Solid as stated | All 40 targets used their line, so source reads which line, not whether. Signatures over three phrasing templates. Address and target agent confounded. Credential influence launders through the tool result after the call. No shuffled-span control run in 2c | Write-up with all four caveats. E0.5 finishes the credential disclosure hand check |
| 9 | The confidentiality rule changes words, not deeds | Sweep2 forbidden 20 of 20 used, rule-off twin 8 seeds identical copying, rule in reasoning 7 of 40 against 0 of 40 | Solid on 14B code tasks | 14B only, code tasks only. A 14B smoke on a data task refused, because the visible data contradicted the note | Write-up. Optional E2.2 runs the rule on 32B, behaviour only |
| 10 | Agents copy a right value on a solvable task and never a wrong one | Lane D solvable, 13 of 20 and 0 of 20 | Solid for the wrong value | "Copied 13 of 20" overcounts. None-twin pairing says the note changed about 4 answers in 20 (9 of 13 submitters were right without it). The paper must present the twin table, not the raw 13 | Write-up. E0.7 reads the 13 traces |
| 11 | Masking anything moves the state as much as the note, in a near-orthogonal direction (briefing s16, s18) | Lane D shuffled span drawn from the task message, 0.16 to 0.20 against 0.31 to 0.35, mass above the note's | **Contradicted by lane C** | Lane C's designer-span placebo moves the state 0.008 to 0.045 against 0.024 to 0.373 and leaves mass at the floor. The two controls differ in what they mask (task text against system boilerplate), so magnitude depends on the task relevance of the masked span. The story page draws no placebo for that reason. Both must be reported with the resolution | E0.6 writes the reconciliation. No GPU |
| 12 | Surplus mass is an exposure gate, not a content meter | 32B pilot, lane C 10 seeds (mass 0.43 leak against benign at response start), Core size-matched edit 1.000 to 0.448 | Solid | None | Write-up |
| 13 | Kernel noise. Both passes on one attention kernel, floor measured | OVERNIGHT, lane B, lane C, lane D floors; lane E found the shared-prefix path bit-identical only at some sequence lengths. Deletion check failed its pre-set criterion | Solid with one open check | The mask-equals-deletion check failed its criterion on one 32B turn (ratios 0.51 to 0.63) and the lane E kernel switch is unexplained. State both | Write-up, appendix |
| 14 | Kept against compacted reference | 32B pilot only. Compacted lands within the floor of the no-note state at 56 of 63 layers, kept closes half the distance | Caveated | Every headline number on 14B uses kept. No 14B check that count and jump are unchanged under compacted | E1.1 (c), 60 turns |
| 15 | No calibrated threshold, no false-positive rate across task types | **Computed today on blind code solve turns.** Count cut 60 gives TPR 30 of 30, false positives 1 of 30 benign and 0 of 30 none. Lower cuts raise benign false positives to 20 to 33 percent. Method_tip counts 63 to 174 are all above 60 | Open for data and record tasks, for non-blind readers, and for verdict turns beyond lane C's 110 | Only code solve turns have traces. Every other agent's solve turn in sweep2 was replayed at fixed positions only, and the trace arrays sit on stopped pod volumes | E0.3 now on what exists. E1.1 (a) for the rest |
| 16 | One model family, 14B converged lens, 32B partial lens | As stated | Open | No third size, no second family | E1.4 Qwen3-8B. Optional E2.1 non-Qwen |
| 17 | Lazy copier, hints and partial values, harder tasks | Untested | Open | The solvable arm already splits by task type (numbers mostly solved, 22-character strings rarely), which is a hardness axis. Hints and partial values have no plant | E1.3 hint plants. Write-up uses the string against number split as hardness |
| 18 | Within-turn attribution 5 of 10 at the first token | Lane D leak2 | Solid as a negative | Reported as "the choice is made in the reasoning". Fine | Write-up |
| 19 | Phase 2b, no emergent writes in 20 runs | Lane E, 2,887 calls, zero writes | Solid as a null | Derived-recovers-declared verified in mock only (455 of 455) | Write-up |
| 20 | Tip length control | Lane D, partial masks of the tip (first or last 37 tokens) 34.5 and 31 against a 37-token leak 16 | Caveated | A partial mask leaves the rest of the tip in context, so it is not a length control. Lane D says so | E1.3 includes a tip generated at the leak's length |
| 21 | Concealment leaves a trace at the submission note | 8 concealers against 60 disclosers, 0.85, leave-one-seed-out within answer type 1.00 and 0.98 | Lead, not a rate | Five of the eight are slip copiers by the author's own definition, three seeds | Write-up as a lead |
| 22 | Swarm coherence reads shared exposure | leak_all, 0.58 against 0.00, same with 0, 1 or 2 users | Solid | None | Write-up |
| 23 | "About 140 planted turns" (abstract) | **Computed today from run directories.** 588 completed matched branches across ten arms, 148 of them none. Value-bearing plant branches 208 (leak 68, forbidden 40, leak2 10, leak_all 10, solvable right 40, solvable wrong 20, credential 20) reaching 228 target turns. Exposure-only plants 232 (benign 116, proposal 40, command 56, method_tip 20) | Wrong number | 140 is sweep2's branch count | Write-up with the census |
| 24 | NARCBench tiers, run and not claimed | Core 1.000 to 0.448 under the matched edit, Transfer 0.963 under placebo with size and vocabulary caveats, Stego decoding held by honest players | Solid as instrument validation | No size-matched control on Transfer. Mask is the label in disguise | Write-up, benchmark section with the stance. Optional E2.4 |
| 25 | Label validity | 30 traces read, leak_used 15 of 15, disclosure regex misses the concealment case | Solid | Solvable "copied" label not in the sample | E0.7 |
| 26 | HF incident framing and timeline | OpenAI technical report, verified in the project notes | Solid | Agent count and the "GO" quote are not in the report | Write-up, cite only what the report says |

### What the gap list says about the abstract

The abstract's two headline numbers need reordering. The count is a strong engagement gate (0.98, 30 of 30) with a stated failure on relevant harmless procedures. The uptake claim is the gap at the first writing, which separates quoted from computed by medians (35 to 61 against 11) with overlapping tails, and it needs a per-turn table and a threshold with a false-positive rate before it is stated as a rule. The "140 planted turns" becomes 228 value-bearing target turns in 588 matched branches. The placebo statement must carry both controls.

## B. Paper outline

Working title. Swarm Surplus. Reading what an agent took from its peers from provenance-masked replay.

1. **Introduction.** Swarms share information through channels nobody planned for (Hugging Face incident, five dated facts from the technical report). Detectors so far need labels of the tactic, exchangeable honest agents, a minority coalition, several rounds, or a known channel. Our question, from hidden states and the provenance of an agent's context, did it depend on peer content, where, from whom, and what. Contributions, four bullets. The provenance-masked reference and its measures. A planted environment with matched twins and uninstructed receivers. The findings, exposure is not uptake, the read position is the method, direction attributes, derived provenance makes a discovered channel readable. An honest account of the released benchmarks.
2. **Related work.** NARCBench and its probe. J-space and J-PACT, with the five assumptions table. TaskTracker's activation delta (the delta is not new). Provenance and information-flow control for agents, forward against our backward use. Collusion environments not run (ColludeBench, Colosseum, AgentDojo, Kill-Chain Canaries, ACIArena, the insider-trading eval).
3. **Method.** 3.1 Provenance tags, declared at assembly, derived from the write log by byte match. 3.2 The reference pass, teacher forcing, the attention mask, kept against compacted positions. 3.3 The kernel floor. 3.4 Measures, each with the control that pinned its job. Mass (gate), trace and window (where), count (engagement), jump and naming at the first writing (taking), direction with signatures and with per-candidate references (whose), ontology (weak), coherence (shared exposure). 3.5 The decision flow and its cost.
4. **Environment.** The peer review game, blind tasks, plants under a quarantined instruction, matched branches from one state, the arms table with the question each answers, labels and their validation, the census of branches and target turns.
5. **Results.** 5.1 Where the leak is read, the paper's positions see exposure, the first mention sees uptake, on two models. 5.2 Engagement against taking, the count on blind agents (pooled 0.98), on exposed non-users (chance), on solvable tasks (copiers equal checkers), on the method tip (tip above leak 20 of 20). 5.3 The gap at the first writing across every arm, per-turn table, threshold and false positives. 5.4 Attribution, signatures across seeds, per-note references within a turn. 5.5 Derived provenance, phase 2c declared against derived, the credential's laundering through the tool result. 5.6 Behaviour, the rule, the command, disclosure and the concealment lead, two conflicting notes. 5.7 Swarm level, coherence. 5.8 Calibration, false positives by task type and turn type, kept against compacted.
6. **Released benchmarks, run and not claimed.** Core, size controls. Transfer, placebo and the two caveats. Stego, honest players hold the range. Why the mask is the label in disguise there.
7. **Limitations.** Model family and lens. Naming at one token. Teacher forcing launders restated content. Provenance defeated by unlogged writes, renamed resources, upstream seeding, and tool results obtained with a derived-tagged token. Templates in signatures. Small counts on concealment and two-note attribution. Count fails on relevant harmless text. No emergence observed. Thresholds calibrated on one game.
8. **Discussion.** What an orchestrator needs to run this. Mass as a cheap gate. Policy decides what is allowed. Next, covert code plant, reviewer-blind norm, longer emergence runs, external suites.
9. **Appendices.** Kernel noise and the deletion check. Compacted against kept on 14B. Per-seed tables. Label sheets. Plant texts. Costs and reproducibility, pods, hours, dollars.

## C. Experiment plan with costs

Prices used. A100 80 GB 1.79 USD per hour (RunPod secure). H200 141 GB about 3.59 per hour (community, lane D's rate; the brief's "about 3" is the floor). Qwen3 14B and 8B self-served with vLLM. OpenRouter only where noted. Each GPU line carries setup time (pod create, install, model pull, about 25 to 35 minutes) and a 25 percent margin. All pods stopped at the end, nothing terminated.

Where the data is. Every run directory exists locally in the lane worktrees (lane D 1.0 GB, lane E 0.5 GB, lane C's sweep14b in the main checkout). Replay residual arrays and per-token traces exist only on stopped pod volumes (`a0e7ctavcauhf3`, `i0drlqvql57u0b`, `c61tivxdev57zx`, `6v762q8lc0toy5`). A stopped pod restarts only if its host has a free card, and `a0e7ctavcauhf3` once did not. Plan A for each GPU item is to restart the volume that holds the inputs, plan B is a fresh pod with the run directories uploaded from this machine (sweep2 alone is about 400 MB).

### Tier 0. No GPU, start on approval, about half a day of CPU and reading

| Id | What | Output | Cost |
| --- | --- | --- | --- |
| E0.1 | Pooled 30-seed count AUROC with bootstrap interval, paired wins, and the threshold table (done today, to be written up) | `results/paper/tables/count_pooled.md` | 0 |
| E0.2 | Per-turn table of the gap at first writing across every arm from committed rows, per-lane floor and band declared, with a threshold sweep (10, 20, 30, 40 times the floor) and TPR and FPR per arm | `results/paper/tables/jump_per_turn.md` | 0 |
| E0.3 | False-positive table for count and jump on every existing negative (none, benign k1 and k3, command, method_tip, 2c benign, lane C's 110 verdict turns), by turn type | `results/paper/tables/false_positives.md` | 0 |
| E0.4 | Census of branches and target turns per arm (done today) | into the environment section | 0 |
| E0.5 | Phase 2c credential disclosure hand check, 20 submission notes and reasoning excerpts, and a hand read of the 7 leak targets labelled "no source" | `results/paper/tables/credential_hand_check.md` | 0 |
| E0.6 | Reconcile the two placebo controls (lane C designer span, lane D task-text span) into one statement with both tables | into the method section | 0 |
| E0.7 | Read the 13 solvable "copied" traces and the 7 non-copier traces, label copy against solve-and-agree by hand, compare with the jump | `results/paper/tables/solvable_hand_read.md` | 0 |

### Tier 1. Recommended GPU work, about 22 USD, two pods in parallel, about 4 hours wall clock

| Id | What | Why | GPU and time | USD |
| --- | --- | --- | --- | --- |
| E1.1 | **Unified rescoring and calibration on sweep2.** One A100. (a) Replay every agent's solve turn in none, benign_k1, benign_k3 and leak for the 20 sweep2 seeds with the per-token trace, 480 turns at about 6 s. Score count and own-first-writing jump by task type (code, data, record) and by role (blind, non-blind). Choose thresholds on lane C's 10 sweep14b seeds, apply to sweep2 and 2c, report TPR and FPR. (b) Rescore every arm's first-writing jump with one floor (none-turn every-token median) and one band (L16 to L38). (c) Compacted reference on the 20 blind leak, benign_k1 and none turns, 60 turns, to show count and jump unchanged. (d) Own-value first writing for the 20 wrong-note solvable readers, 20 turns | Closes gaps 2, 4, 14, 15. Turns the "checker" picture from one trace into twenty | Restart `i0drlqvql57u0b` (holds later-arm runs and residuals) or fresh A100 with sweep2 and solvable uploaded. About 1.3 h compute plus setup, 2.2 h | 5 |
| E1.2 | **Twenty more leak2 seeds**, none and leak2, phrasing varied, half broadcast, code targets. Pre-registered primary test, nearest per-note direction at answer_pre. Secondary, mention_pre and first token. Replay with four passes per turn (actual, both masked, each note masked) | Gap 5, n 10 to 30 on the within-turn attribution claim | A100 generation about 1.5 h at 450 to 700 tok per s (40 branches), replay 20 turns about 5 min. 2.3 h | 5 |
| E1.3 | **Length-matched tip and hint plants**, 20 seeds, code targets, rules off. Branches none, tip_short (procedure cut to the leak's token length by the generation prompt), hint_prefix (the value's first character or first digit), hint_range (a numeric bracket). Replay all target turns with the trace. Scores, count, window, jump at first writing, direction against leak and tip signatures | Gaps 17 and 20. Tests the count under a length-matched procedure and the jump under partial values | Plant code 1 to 2 h on CPU first. A100 generation 80 branches about 3 h, or H200 about 1.5 h at 3.59. Replay 80 turns about 10 min. Costed on the H200 | 7 |
| E1.4 | **Third model size, Qwen3-8B** with its converged lens. 10 seeds, none, benign_k1, leak, code targets. Replay blind solve turns with trace and lens readout. Report the response-start null, the first-mention read, count and jump, direction source | Gap 16 at the cheapest point | A100, vLLM 8B generation about 45 min, replay about 10 min, setup. 1.6 h | 4 |
| | **Tier 1 total** | | | **21, ask for a cap of 25** |

Lane assignment. Lane F runs E1.1 then E1.4 on one A100. Lane G builds the E1.3 plants on CPU, then runs E1.2 and E1.3 on one card. Both report under `results/lane_f/` and `results/lane_g/` with REPORT.md, scores JSON and pod state, pods stopped at the end.

### Tier 2. Optional, each independent

| Id | What | Why | USD | Note |
| --- | --- | --- | --- | --- |
| E2.1 | A non-Qwen family, Llama 3.1 8B Instruct or gpt-oss-20b, 10 seeds, none, benign_k1, leak. Lens-free tests only | Gap 16, the family question | 8 | Half a day of engineering first. `detector/render.py` positions assume Qwen3's think block, and the client's tool-call parsing needs the family's parser. Risk that the model refuses the game or never uses the leak |
| E2.2 | 32B under the confidentiality rule, 10 seeds, none, leak, forbidden_leak, behaviour only on OpenRouter | Gap 9 on the second model | 4 OpenRouter | Replay on 32B would add about 3 USD on an A100 and is not needed for the behavioural claim |
| E2.3 | TaskTracker or AgentDojo as an external test of the jump at the value | Idea history's "cleanest outside test" | 12 to 15 | A day. Different prompt shape, single agent. Recommend the paper lists it as not run and this becomes the follow-up |
| E2.4 | Transfer size-matched placebo on 32B | Lane B's open control | 3 | Only if the benchmark section is to carry any Transfer number. Recommend skip, since nothing there is claimed |

### Not proposed

NARCBench comparisons as a claim. A phase 2b rerun with a stronger model (emergence is a longer experiment than this paper). A refit of the 32B lens.

## D. What happens on a yes

Tier 0 starts at once in this session. Two lane sessions start for tier 1 with the lane briefs above, each with its budget line, the stop rule, and the instruction to restart a stopped volume before creating a pod. Reports land under `results/`, then the draft is written in `results/paper/draft.md` with every number traced to a file. Any tier 2 item needs its own yes.

## E. Decisions needed

1. Approve tier 1 at a 25 USD cap, or pick items.
2. Any tier 2 item now, or none.
3. The abstract's headline, lead with the jump and present the count as a gate with its failure stated, as recommended here.
4. Whether the draft mirrors to a Claude Doc at the end.

## F. Gaps against the Apart judging rubric (added 8 November, evening)

Source, `APART_JUDGING_CRITERIA.rtf` and `RESOURCES.rtf` from Ayse.

### Submission requirements

| Requirement | State | Gap |
| --- | --- | --- |
| Research report as PDF in the official Google Doc template | Nothing yet | The final artefact is a PDF from that template, not a Claude Doc. The draft stays markdown in `results/paper/`, then is poured into the template |
| At most 8 pages before references and appendices | Outline in B has nine sections and ten arms | Cut to three claims in the body. Released benchmarks, kernel noise, kept against compacted, per-seed tables, plant texts and label sheets go to appendices |
| Abstract 150 to 250 words | Current draft about 420 words | Rewrite, leading with the jump as recommended in A |
| Authors and affiliations | Ayse Ozdincer, Máté Lakner | Confirm affiliations |
| Required appendix, Limitations and Dual-Use / Ethical Considerations | Limitations exist across reports, no dual-use text anywhere | Write it. Dual-use content is in the table below |
| Prior work must be stated, pre-sprint code marked in README and report | Not done | Declare the vendored J-space code (`third_party/maj-lens`), NARCBench data and the Neuronpedia lenses as others' work. The harness commit a595dc5 is dated 6 November 23:59 Pacific and the proposal was written that evening. If the sprint opened on Friday evening these are in scope, if it opened on Saturday they must be marked preexisting. Ayse decides from the official start time |
| Public GitHub repo (optional) | Private repo | Decide whether to make it public. If so, remove `.env`, pod ids and the labelled traces' raw model text where it quotes NARCBench |
| 3 to 5 minute video (optional) | The three-minute deck exists | Record over the deck if time allows |

### Dimension 1, impact and innovation. "Is this new, or replicating recent work?"

A judge who knows the area will ask three things and the body must answer each in one paragraph.

| They will say | We answer, with the evidence that backs it |
| --- | --- |
| The activation delta between clean and poisoned context is TaskTracker (2024) | Agreed in print. What is new is deciding the mask from provenance tags rather than from a labelled poisoned set, reading it along the reply rather than at one position, and the attribution by per-candidate reference. TaskTracker needs a trained probe, ours has none |
| Collusion from activations is NARCBench's probe and J-PACT | The probe is supervised on anticipated tactics. J-PACT compares an agent to its peers and needs exchangeable honest agents, a minority pair and several rounds. Ours compares an agent to its own replay, so roles, majorities and single turns are in scope. Transfer at 0.556 against 0.963 under placebo shows the point, with its caveats |
| Provenance for agents is CaMeL, FIDES, Spotlighting, NeuroTaint | Those run provenance forward through text and values to enforce a policy. We run it backward as a counterfactual on the hidden states and block nothing. The derived tag is the write-then-read edge of provenance-based intrusion detection applied to a swarm |

Theory of change, one paragraph. An orchestrator already has the tags and the weights. The monitor gates on mass, runs the full read only on turns where peer content moved the agent, and reports engagement, taking, source and content per turn, leaving the policy question to the environment. The Hugging Face case is the derived tag.

### Dimension 2, execution. "Report variance and baselines"

| Gap | Fix | Cost |
| --- | --- | --- |
| No baseline is run on our own environment. Every baseline number is on NARCBench | Add four on the same turns. (a) Text monitor, the note-citation regex and an LLM judge reading the reasoning, which is strong when agents verbalise and blind for the 9 of 13 solvable submitters who never mention the note, the 7 of 20 2c leak targets who name no source and the concealers. (b) J-PACT pair-against-group over our six agents from the existing J-space rows at fixed positions. (c) A supervised linear probe on the residuals with seed-grouped cross-validation, to show what a label-free rule gives up against a fitted one. (d) The write log alone, which equals exposure. | (a) CPU plus about 2 USD OpenRouter for the judge. (b) CPU. (c) needs the arrays, folds into E1.1. (d) CPU |
| Hyperparameters chosen by hand on the pilot (10 times the floor, first 256 tokens, 16-token window, band L16 to L38) | Sensitivity table in the appendix over 5, 10 and 20 times the floor, 128, 256 and 512 tokens, 8, 16 and 32-token windows, mid and late bands. Thresholds chosen on sweep14b and applied to sweep2 and 2c | CPU for the blind turns, E1.1 for the rest |
| Variance reported unevenly | Bootstrap interval over seeds and paired wins on every headline number, binomial p on the counts | CPU |
| Same model generated the plants and the receivers | State it, and note that plants are quarantined and receivers never see the instruction | Write-up |

### Dimension 3, presentation

Three claims in the body, each with one figure. Where the leak is read (the trace figure). Taking against engaging (the per-turn jump table with the threshold line and the method-tip failure). Whose content, and the derived tag (leak2 and 2c in one table). Everything else is appendix. A half-page glossary of the measures replaces the briefing's long table.

### Dual-use text the required appendix must carry

- The paper names where uptake is read. An agent that restates peer content in its own words before using it launders the signal into the reference, and one that splits a value across turns or routes it through a tool result it obtained with the peer's help weakens the first-writing read. The 2c credential branch shows the second effect.
- The derived tag is defeated by unlogged writes, renamed or mirrored resources and content seeded one hop upstream. Publishing that is a map for evasion and also the list an operator must close.
- The planted-channel harness is a recipe for seeding coordination in a swarm. It is released for detection research with matched controls, and the plants are generated under a quarantined instruction.
- Nothing here raises capability. The method needs weight access, so it helps the operator and not an outside attacker.

## G. Answers to Ayse's four questions

**More scenario kinds or more cases per scenario.** More cases, on the scenarios that carry a claim, with one exception. The rubric scores convincing validation, and our thin spots are counts and false positives, not variety of attacks. Leak2 at ten turns, copiers at thirteen, concealers at eight. The exception is the harmless side. The count fails on an on-task procedure, so the false-positive set needs a variety of harmless notes (tips, procedures, partial restatements of the task, off-task chatter), not more kinds of leak.

**Expand scope or sharpen current findings.** Sharpen. Ten arms, three benchmark tiers and two models already exceed what eight pages can carry. The body should make three claims with thresholds, intervals and baselines, and the rest becomes appendix or follow-up.

**What has been fitted to the surplus.** Almost nothing in the learning sense, and the paper should say so as a feature with one honest cost. The detector has no trained component. Scores are hand-defined statistics, the relative residual delta over a measured kernel floor, the count of reply tokens above a fixed multiple in a fixed window, a sliding-window maximum, cosines to leave-one-out mean directions, and the lens rank. Evaluation is nonparametric, AUROC with a scenario bootstrap, paired wins, binomial and Fisher tests, a permutation test on Stego. The only fits are the Core size regression (mass against log changed tokens, R squared 0.93 on honest agents, colluders scored by residual), the control-run z-score calibration on Transfer, and the J-space sparse decomposition, which is the J-space paper's method. The cost is that the four hyperparameters were chosen by eye on the pilot, so the paper needs the held-out threshold test (sweep14b to sweep2 to 2c), the sensitivity table, and a fitted comparator (the probe in F) to show how much a learned detector would add. How much more data. For a false-positive rate near 2 percent with a defensible interval, about 150 to 300 negatives per task type, which E1.1 gets by replaying runs that already exist. For attribution, 30 leak2 turns instead of 10 (E1.2). For copied against computed, 20 more true copiers, which the hint plants give (E1.3). Beyond that the next axis is model family, not more Qwen3-14B turns.

**External benchmarks, reformatted.** Yes, one or two, as a robustness appendix rather than a headline, and with the reformatting stated. The reformat is the same every time. Tag the span the dataset injected as the masked provenance class and ask the detector whether the agent used it. The dataset supplies the per-sample uptake label. This is a fair test, unlike NARCBench Core, because every sample has a masked span and only some agents act on it, so the mask is not the label. In order of fit. TaskTracker (Abdelnabi et al. 2024), clean against poisoned prompts with a per-sample label of whether the injected task was executed, single agent, thousands of samples, responses regenerated on Qwen3-14B, the cleanest outside test of the jump and the count and the largest false-positive set, about 5 to 8 USD. The insider-trading pressure eval (Scheurer et al. 2023), a colleague's tip with use and concealment labels, the shape of our forbidden leak and the concealment lead, a few hundred runs, about 4 USD. AgentDojo, tool-result injections with security labels, needs a full agent loop, about a day. ColludeBench, instructed colluders again, lowest value. Swarm Traces and collusion.wiki from the resources list are text without weights, so no surplus can be read, but the derived tag's write-then-read matching could be demonstrated on them as a graph reconstruction, which is a discussion item. The 8-page cap means at most one of these appears in the body.

## H. Revision after Ayse's answers (8 November, late)

Sprint opened Friday afternoon, so every commit is in scope. Declared prior work is other people's. The vendored J-space code, the NARCBench data and the Neuronpedia lenses.

### H.1 The story belongs in the introduction, as reasoning, not as a diary

One paragraph, in this order, because it is the argument for the method and it pre-empts "this replicates the J-space paper". We set out to use the J-space for collusion detection. J-PACT scores a pair against the honest outsiders, which assumes the outsiders are interchangeable stand-ins for what an agent would hold without the channel. A swarm has roles, so that stand-in does not exist. We asked for a ground truth per agent instead, the same agent without the channel, and the only way to have it is to replay the agent with the unsanctioned part of its context removed. The first idea, scoring nodes and edges by whether the J-space holds collusion concepts, needs a definition of collusion before the fact and was dropped. What defines the unsanctioned part without such a definition is provenance, the tag the orchestrator already writes when it assembles a context. The goal is a detector an orchestrator can run on an ordinary swarm.

### H.2 Variation axes, not scenario kinds, and why each has value

Ayse asked whether task type, prompt variation and the number of leaks carry value. They do, and the earlier answer under-sold it by lumping them with new attack kinds. They are the axes along which the same frozen rule is tested, and each has a reason.

| Axis | Levels we have | What is missing | Why it matters | Run |
| --- | --- | --- | --- | --- |
| Task type | Code targets everywhere. Data and record targets only in leak_all (record 2 of 10 used, data 0 of 10) and one smoke | Data and record blind tasks withhold rows silently, so the visible data contradicts the note and 14B rejects it. The fix, declaring the withheld piece, was proposed and never built | Without it every uptake number is a code-task number, and the false-positive table has no data or record positives to set against its negatives | E1.5, build the fix, then 20 seeds with data and record targets, none, benign_k1, leak |
| Phrasing and address | Fixed, varied, named, broadcast across sweep2, leak2, solvable, 2c | Nothing new to generate | Signatures built on one phrasing must be tested on another. Report as a held-out split of existing data | CPU |
| Number of peer spans in the mask | One note, two notes (leak2), three targets (leak_all), k 1 and 3 for benign, proposal and command | A leak sitting among harmless notes in the same digest | In a deployed swarm the mask covers every peer span at once and most are harmless. The read must find the one that was used inside a larger mask, and the count grows with masked tokens | E1.6, leak plus 2 and leak plus 4 harmless notes, 20 seeds |
| Note relevance and length | Off-task tip, on-task procedure, value, partial masks | A length-matched procedure, partial values | The count's failure mode | E1.3, already in tier 1 |
| Model size | 14B converged lens, 32B partial lens | A third size | Not a new detector. Nothing is trained, the floor is measured per model and every threshold is in floor units. Whether the 10-times rule transfers across sizes in floor units is itself a finding, and if it does the detector is model-agnostic in the sense that matters to an operator | E1.4, 8B |
| Model family | Qwen3 only | A non-Qwen family | Same argument, harder engineering | Tier 2, E2.1 |

### H.3 How the fitting happens, and what can be learned

Four levels, each answering one question. Levels 1, 2 and 4 are CPU once the traces exist. Level 3 needs the residual arrays, which E1.1 produces.

1. **The frozen rule.** Thresholds in floor units chosen on sweep14b's 10 seeds (count above 10 times the floor in the first 256 tokens, jump at the first writing above a cut to be fixed there), then applied unchanged to sweep2, the solvable and hint arms, 2c, the data and record arm and the 8B arm. Output, one table of TPR and FPR per axis cell. Question answered, where does the hand rule break.
2. **A fitted score on the hand features.** Logistic regression with L2 on about eight per-turn features, count, window maximum, jump, mass at the first token, whole-reply mean surplus, direction margin, reply length, masked tokens. Cross-validation grouped by seed. Train on blind code solve turns, test on every other cell. Output, AUROC beside the rule's, and the coefficients. Questions answered, how much a learned combination adds over the rule, which feature carries uptake, and whether the fit leans on confounds (reply length, masked tokens, mass). If it does, that is a warning about the environment, not a result.
3. **A probe on the residual.** A linear probe on the hidden state at the first writing and at the found location, trained on labels with seed-grouped folds, trained on one arm and tested on the others. This is the NARCBench-style supervised comparator on our data. Question answered, how much information the raw state holds beyond what the surplus rule reads. If the probe is far above the rule, the information is there and the rule misses it. If it is close, label-free costs little.
4. **Descriptive regression.** Count against masked tokens, note relevance, task type and reply length. Jump against answer type (digit or string), task type and position in the reply. Question answered, what drives magnitude, which settles the length-against-content question lane D could only half answer.

What we can learn from all four together. Whether the three claims hold under a rule fixed in advance. Which measure carries uptake across contexts. Whether a learned detector on the same features would change the picture. Whether thresholds in floor units transfer across model size. The paper reports levels 1 and 2 in the body and 3 and 4 in the appendix.

### H.4 External checks, concrete plan

Checked today. The insider-trading repository (ApolloResearch, CC-BY-4.0) is runnable as released. Each prompt file holds a 16-message conversation, the tip arrives as two messages from Sally, the pressure as messages from Amy and William, and the file carries `misalignment_string`, `deception_trigger` and `doubling_down_trigger`. TaskTracker (Microsoft, MIT) does not ship texts. The data is rebuilt from notebooks and the label of whether a model followed the injection comes from a GPT-4 judge in their evaluation stage. So the second check is TaskTracker-shaped and built by us, and the paper says so.

| Id | What | Reformat | Labels | Reads | GPU and cost |
| --- | --- | --- | --- | --- | --- |
| X1 | Insider trading on Qwen3-14B. The released prompt variants, 50 samples each at the paper's temperature, about 200 runs | Sally's two messages tagged peer, everything else designer or environment. Replay Alpha's reply with the tip masked | Traded on the tip, by the file's `misalignment_string`. Concealed the tip in the report to Amy, by an LLM judge as in the paper, about 1 USD on OpenRouter | Jump at the first writing of merger content, count, direction. Concealers against disclosers at the report, the second environment for the concealment lead | Loader and positions about 3 h of engineering. A100 generation about 15 min, replay about 15 min. 4 USD with setup |
| X2 | TaskTracker-shaped injected documents. 300 SQuAD paragraphs with the question as the primary task, an injected instruction with a mechanical target inserted into the paragraph, and 300 clean twins | The injected sentence tagged as the masked class, a finer mask than TaskTracker's whole-data-block delta | Followed the injection, by the target string in the reply. Clean twins give false positives | Jump at the first writing of the target, count, FPR on 300 clean twins | Builder about 3 h. A100 about 1 h. 3 USD |

Both go to an appendix. Neither replaces the environment, both test the same two reads on a different distribution with per-sample labels where every sample carries a mask, which is the property NARCBench lacked.

### H.5 Revised tiers and budget

| Tier | Items | USD |
| --- | --- | --- |
| 0 | E0.1 to E0.7, baselines (a), (b), (d), held-out phrasing split, fitting levels 1, 2 and 4 on existing traces | 2 OpenRouter |
| 1 | E1.1 rescoring, calibration, compacted, own-value (5). E1.2 leak2 plus 20 (5). E1.3 length-matched tip and hints (7). E1.4 Qwen3-8B (4). E1.5 data and record targets after the fix (5). E1.6 leak among k harmless notes (4). Probe, level 3, inside E1.1 | 30, ask for a cap of 35 |
| 2 | X1 insider trading (4). X2 TaskTracker-shaped (3). E2.1 non-Qwen family (8). E2.2 32B under the rule (4) | each on its own yes |

Lane plan. Lane F, one A100, E1.1 then E1.4 then E1.5's replay. Lane G, plant and task fixes on CPU (E1.3 hints, E1.5 declare-withheld, E1.6 leak plus k), then one card for E1.2, E1.3, E1.5 and E1.6 generation and replay. About five hours wall clock for both. X1 and X2 would be a third lane after tier 1 reports.
