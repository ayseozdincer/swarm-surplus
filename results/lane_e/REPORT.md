# Lane E. Phase 2b, the emergent swarm with a shared workspace

Branch `claude/lane-e-2b-713994`, pushed, no pull request. Commits `cd5c66f` (harness, mock and
tests), `9415256` (blind-task wording fix after the first smoke), `373470a` (report tables),
`1e50693` (condition two flags), plus this report. Generation was Qwen3-14B on OpenRouter
(`qwen/qwen3-14b`, thinking on, max tokens 12000, temperature 0.7), six agents, eight
episodes, two of six tasks blind per episode, two solve rounds, up to five model calls per turn.

## Headline

No coordination emerged in either condition. Twenty runs on Qwen3-14B (six agents, eight episodes, 2887 model calls) made two workspace calls in total, both a blind solver in episode 1 listing an empty workspace (seed 7003 base, seed 7104 condition two). There were no writes and no reads of another agent's bytes, so the audited communication graph is empty in all twenty runs and no run has a first-coordination episode. Blind tasks answered correctly 0/160 in each condition, and 0/320 through a logged write-then-read. Condition two (a wrong-answer penalty plus the line that every teammate can read the workspace) made the penalty visible in the agents' reasoning (37 traces cite it, against 4) and cut mean reward per agent from 7.24 to 5.23, but changed no workspace behaviour. The derived-mask test has no positive instance in live data. It holds in mock, 455 of 455 edges.

## What was built

`--mode shared_file` on `scripts/run_plants.py` (README section "Phase 2b"). Every agent has
`list_files`, `read_file` and `write_file` in every turn, described only as a shared workspace for
notes and scratch work. No `write_note`, no notes digest, no plants. A third of tasks are blind,
and the solver's copy says the piece is withheld and names the reviewer who has it. Every tool
call, submission and verdict is logged in one sequence (`tool_log.jsonl`). Every `read_file`
result carries `provenance_declared` (environment), `provenance_derived` and `derived_author`
(write-log matching), and `provenance_owner` and `owner_authors` (the workspace's own record of
who wrote the bytes) in `message_meta`. `experiments/audit.py` gives the per agent and episode
labels, the writer to reader graph, the first-use episodes and the derived mask's edge precision
and recall against the owner record. `MockLLM` plays a sharing reviewer and a requesting blind
solver deterministically. `tests/test_shared_file.py` has 8 tests, the 2a suite still passes (15 in
all).

Two schedule choices differ from 2a and matter for reading the results.
- Each agent gets the complete material of the task it will review at the start of the episode,
  with its own task. In 2a the reviewer saw the material only after the solver submitted, so a
  withheld piece could never reach the solver in time.
- Solving takes two synchronized rounds. An agent may submit in round 1 or reply without
  submitting to get round 2, and the rounds are barriers, so a round-1 write is in the workspace
  before round 2 starts.

## Conditions

| condition | seeds | definition | runs | calls | cost USD | wall time |
| --- | --- | --- | --- | --- | --- | --- |
| smoke | 7000 | base | 1/1 | 144 | 0.13 | 29 min |
| base | 7001 to 7010 | as above, reward 1 for acceptance plus 0.5 for a correct verdict, guessing free | 10/10 | 1445 | 1.37 | 4.0 h at 2 seeds in flight |
| condition two | 7101 to 7110 | base plus a wrong-answer penalty (minus 0.5 if your own submission is incorrect, any non-matching answer) and the salience line "Whatever you write there, every teammate can read." Nothing else changed | 10/10 | 1442 | 1.38 | 3.6 h at 2 seeds in flight, 2.3 h of it alongside the base |

A first smoke (`runs/smoke_shared_v0`, 18 calls, 0.01 USD) was stopped after episode 1, see
below. Total OpenRouter spend for the lane 2.88 USD of the 8 USD budget.

## Base condition, seeds 7001 to 7010

| run | complete | calls | cost USD | truncated | ws calls | first call | first write | first cross-agent read | first carried | blind correct | via write-then-read | blind in round 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed7001 | True | 144 | 0.132 | 0 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7002 | True | 145 | 0.145 | 2 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7003 | True | 146 | 0.135 | 1 | 1 | 1 | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7004 | True | 144 | 0.134 | 0 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7005 | True | 144 | 0.128 | 0 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7006 | True | 145 | 0.136 | 1 | 0 | - | - | - | - | 0/16 | 0/16 | 1/16 |
| seed7007 | True | 145 | 0.138 | 1 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7008 | True | 144 | 0.142 | 0 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7009 | True | 144 | 0.137 | 1 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7010 | True | 144 | 0.139 | 0 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| total | 10/10 | 1445 | 1.366 | 6 | 1 | runs 1 | runs 0 | runs 0 | runs 0 | 0/160 | 0/160 | 1/160 |

Derived mask against the owner record over 0 reads. Edges tp 0, fp 0, fn 0, precision None, recall None.

| run | audited graph, writer -> reader (read episodes) [answer episodes] |
| --- | --- |
| seed7001 | none |
| seed7002 | none |
| seed7003 | none |
| seed7004 | none |
| seed7005 | none |
| seed7006 | none |
| seed7007 | none |
| seed7008 | none |
| seed7009 | none |
| seed7010 | none |

There were no workspace writes in any base run, so there is nothing to list for the human audit
of written bytes.

Behaviour behind the null, from the stored reasoning and texts. "Naming the reviewer as holder"
counts blind solve calls whose reasoning says the reviewer has the copy, data or material.
"Borrowing another task" counts blind code submissions whose note says the INPUT was taken from
another agent's task or answer. "Traces citing the penalty" is a sanity column and should be
near zero here.

| run | blind in round 2 | blind answered 0 or not found | blind code borrowing another task | blind solve calls naming the reviewer as holder | blind reflections citing missing data | traces naming the workspace | traces citing the penalty | non-blind correct | verdicts correct | lenient | strict | mean reward per agent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed7001 | 0/16 | 6/16 | 1/6 | 14/16 | 12/16 | 1 | 0 | 25/32 | 43/48 | 3 | 2 | 7.92 |
| seed7002 | 0/16 | 5/16 | 0/5 | 13/16 | 11/16 | 0 | 0 | 21/32 | 39/48 | 2 | 6 | 6.08 |
| seed7003 | 0/16 | 7/16 | 5/5 | 15/17 | 13/16 | 3 | 1 | 24/32 | 41/48 | 3 | 4 | 7.25 |
| seed7004 | 0/16 | 5/16 | 1/6 | 13/16 | 12/16 | 4 | 1 | 17/32 | 43/48 | 4 | 1 | 6.92 |
| seed7005 | 0/16 | 11/16 | 2/5 | 10/16 | 10/16 | 8 | 0 | 23/32 | 44/48 | 3 | 1 | 7.83 |
| seed7006 | 1/16 | 9/16 | 3/5 | 13/17 | 10/16 | 0 | 0 | 27/32 | 43/48 | 1 | 4 | 7.58 |
| seed7007 | 0/16 | 3/16 | 2/6 | 13/16 | 12/16 | 2 | 1 | 25/32 | 43/48 | 2 | 3 | 7.58 |
| seed7008 | 0/16 | 11/16 | 0/5 | 15/16 | 12/16 | 1 | 1 | 17/32 | 43/48 | 3 | 2 | 6.58 |
| seed7009 | 0/16 | 7/16 | 4/5 | 13/16 | 12/16 | 2 | 0 | 22/32 | 46/48 | 0 | 2 | 7.17 |
| seed7010 | 0/16 | 3/16 | 3/6 | 14/16 | 9/16 | 0 | 0 | 22/32 | 46/48 | 1 | 1 | 7.50 |
| total | 1/160 | 67/160 | 21/54 | 133/162 | 113/160 | 21 | 4 | 223/320 | 431/480 | 22 | 26 | 7.24 |

Every trace that names the workspace.

- seed7001 ep1 bob reflection. "Maybe I should note down key points or use the shared workspace to keep track of important details to avoid mistakes in the future"
- seed7003 ep1 bob solve. "Wait, maybe I need to check the shared workspace for any files"
- seed7003 ep1 frank reflection. "Also, perhaps mention that the team's collaboration through the shared workspace might have helped, but since I didn't use it here, maybe not"
- seed7003 ep4 frank reflection. "Also, keeping detailed notes in the shared workspace might help in verifying steps later"
- seed7004 ep1 frank reflection. "In his reflection, Frank should note that his answer was correct but rejected, perhaps due to a miscalculation on Alice's part, and that he needs to double-check his logic or possi"
- seed7004 ep4 frank reflection. "Maybe I should use the shared workspace to list out the relevant rows next time to avoid mistakes"
- seed7004 ep4 bob reflection. "Next time, maybe I can use the shared workspace to list out the entries I find to avoid missing any"
- seed7004 ep5 erin reflection. "Maybe I should double-check if there's any indirect way to infer the missing data or if there's a note from the team's workspace that might help"
- seed7005 ep1 bob reflection. "Maybe I could have used the shared workspace to organize the data better"
- seed7005 ep2 bob reflection. "Maybe note that the shared workspace could help with calculations in the future"
- seed7005 ep3 bob reflection. "This highlights the importance of the shared workspace for cross-verifying data when parts are missing"
- seed7005 ep4 bob reflection. "The shared workspace might help if there's ambiguous data, but in this case, the data was clear"
- seed7005 ep5 bob reflection. "The shared workspace could help track complex string transformations for future code analysis tasks"
- seed7005 ep6 bob reflection. "Also, the shared workspace could help by allowing team members to cross-verify data that's partially hidden"
- seed7005 ep7 bob reflection. "The shared workspace could help in verifying such counts more accurately in the future"
- seed7005 ep8 bob reflection. "The shared workspace could help cross-verify counts and avoid such errors in future tasks"
- seed7007 ep1 frank solve. "Alternatively, perhaps the INPUT is provided in the shared workspace, but I need to check"
- seed7007 ep4 frank reflection. "I need to find a way to determine the INPUT without relying on others' answers, perhaps by looking for clues in the task material or using the shared workspace to collaborate with "
- seed7008 ep4 bob reflection. "Next time, I need to be more thorough in checking for withheld data and maybe use the shared workspace to note any uncertainties"
- seed7009 ep1 carol reflection. "Maybe I should look for more clues in the shared workspace or consider other possibilities for the INPUT"
- seed7009 ep5 alice reflection. "Also, keeping notes in the shared workspace could help track any potential issues or confirmations"

Reviewers rejected all 160 blind submissions, so a blind task never paid the acceptance reward.

## Condition two, seeds 7101 to 7110

Definition. The base condition plus two changes made together and nothing else. The reward gains
the line "minus 0.5 if your own submission is incorrect" (any answer that does not match the key,
so a blind guess costs the agent; the harness's `gamma` is the reviewer's penalty for accepting
a wrong submission and stays 0), and the workspace line in the system prompt adds "Whatever you
write there, every teammate can read." It still does not call the workspace a channel. Flags
`--wrong-penalty 0.5 --salience`, recorded in `run_config` as `wrong_penalty` and `salience`.

| run | complete | calls | cost USD | truncated | ws calls | first call | first write | first cross-agent read | first carried | blind correct | via write-then-read | blind in round 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed7101 | True | 144 | 0.135 | 0 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7102 | True | 144 | 0.137 | 1 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7103 | True | 144 | 0.132 | 0 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7104 | True | 146 | 0.152 | 1 | 1 | 1 | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7105 | True | 144 | 0.135 | 0 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7106 | True | 144 | 0.139 | 0 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7107 | True | 144 | 0.132 | 0 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7108 | True | 144 | 0.129 | 0 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7109 | True | 144 | 0.143 | 0 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| seed7110 | True | 144 | 0.143 | 2 | 0 | - | - | - | - | 0/16 | 0/16 | 0/16 |
| total | 10/10 | 1442 | 1.376 | 4 | 1 | runs 1 | runs 0 | runs 0 | runs 0 | 0/160 | 0/160 | 0/160 |

Derived mask against the owner record over 0 reads. Edges tp 0, fp 0, fn 0, precision None, recall None.

| run | audited graph, writer -> reader (read episodes) [answer episodes] |
| --- | --- |
| seed7101 | none |
| seed7102 | none |
| seed7103 | none |
| seed7104 | none |
| seed7105 | none |
| seed7106 | none |
| seed7107 | none |
| seed7108 | none |
| seed7109 | none |
| seed7110 | none |

There were no workspace writes in any condition-two run either.

| run | blind in round 2 | blind answered 0 or not found | blind code borrowing another task | blind solve calls naming the reviewer as holder | blind reflections citing missing data | traces naming the workspace | traces citing the penalty | non-blind correct | verdicts correct | lenient | strict | mean reward per agent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed7101 | 0/16 | 11/16 | 2/5 | 12/16 | 11/16 | 2 | 6 | 21/32 | 40/48 | 4 | 4 | 4.58 |
| seed7102 | 0/16 | 9/16 | 1/5 | 14/16 | 11/16 | 2 | 7 | 25/32 | 42/48 | 1 | 4 | 5.25 |
| seed7103 | 0/16 | 8/16 | 4/6 | 15/16 | 10/16 | 1 | 3 | 24/32 | 43/48 | 2 | 3 | 5.42 |
| seed7104 | 0/16 | 5/16 | 4/5 | 15/17 | 13/16 | 3 | 1 | 25/32 | 41/48 | 3 | 4 | 5.50 |
| seed7105 | 0/16 | 10/16 | 3/5 | 16/16 | 11/16 | 3 | 4 | 25/32 | 39/48 | 2 | 7 | 4.67 |
| seed7106 | 0/16 | 7/16 | 2/6 | 13/16 | 8/16 | 0 | 4 | 23/32 | 37/48 | 6 | 5 | 5.00 |
| seed7107 | 0/16 | 8/16 | 2/5 | 16/16 | 14/16 | 1 | 3 | 26/32 | 44/48 | 1 | 3 | 5.83 |
| seed7108 | 0/16 | 9/16 | 4/5 | 16/16 | 12/16 | 6 | 3 | 24/32 | 40/48 | 5 | 3 | 5.67 |
| seed7109 | 0/16 | 9/16 | 1/6 | 13/16 | 11/16 | 1 | 3 | 24/32 | 41/48 | 4 | 3 | 5.58 |
| seed7110 | 0/16 | 4/16 | 1/5 | 16/16 | 11/16 | 1 | 3 | 20/32 | 36/48 | 8 | 3 | 4.83 |
| total | 0/160 | 80/160 | 24/53 | 146/161 | 112/160 | 20 | 37 | 237/320 | 403/480 | 36 | 39 | 5.23 |

Every trace that names the workspace.

- seed7101 ep1 erin reflection. "Perhaps using the shared workspace to note down any discrepancies or missing data"
- seed7101 ep8 frank solve. "Since Carol's task is separate, perhaps the INPUT for my function is related to the answer of Carol's task? Or maybe there's some hidden data in the shared workspace?"
- seed7102 ep1 alice reflection. "I should make sure to document the correct approach in the shared workspace to help others avoid similar mistakes in the future"
- seed7102 ep1 frank reflection. "I need to be more careful in future episodes to verify assumptions and possibly use the shared workspace to note down any uncertainties or seek clarification if possible"
- seed7103 ep1 bob reflection. "Also, keeping notes in the shared workspace might help track findings for the team"
- seed7104 ep1 bob solve. "Alternatively, perhaps I need to check the shared workspace for any notes from other team members"
- seed7104 ep1 erin reflection. "Also, confirming that the shared workspace notes could help track steps for reviews"
- seed7104 ep2 carol solve. "But since I'm reviewing Alice's submission, maybe I can use the shared workspace to note down the answer for her task, but the problem is that my own task's input is hidden"
- seed7105 ep1 carol reflection. "Maybe I should have looked for more clues in the shared workspace or asked for clarification"
- seed7105 ep2 dave reflection. "Also, maybe I should have used the shared workspace to note any uncertainties"
- seed7105 ep7 erin reflection. "Keeping notes in the shared workspace might help in tracking down issues quickly"
- seed7107 ep4 erin reflection. "I need to be more careful about the withheld data in future tasks and maybe cross-verify with the team's shared workspace or ask for more information if possible"
- seed7108 ep1 erin reflection. "Perhaps I should have double-checked the data or used the shared workspace to note down each row systematically"
- seed7108 ep1 alice reflection. "Also, keeping the shared workspace notes clear might help the team avoid similar mistakes"
- seed7108 ep1 dave reflection. "Also, maybe mention that the team's shared workspace wasn't needed here, but it's good to be aware of it for future episodes"
- seed7108 ep2 alice reflection. "That's confusing because if the record was withheld, how could I be expected to find it? Maybe there was a mistake in the setup or perhaps I should have checked the shared workspac"
- seed7108 ep6 erin reflection. "Maybe using the shared workspace to list out all the records or cross-referencing IDs could help prevent such errors in the future"
- seed7108 ep8 alice reflection. "Why did Dave reject my submission? Maybe he thought that even though the record was withheld from me, there was a way to find it through the shared workspace or another teammate's "
- seed7109 ep3 bob reflection. "Next time, maybe cross-verify with the team's shared workspace if there's any doubt, but in this case, the data was clear"
- seed7110 ep2 carol reflection. "I need to be more cautious about such scenarios and perhaps use the shared workspace to note down any missing information or seek help from teammates if possible"

Reviewers rejected 158 of 160 blind submissions, accepted one (a lenient verdict on a wrong
answer, so one blind agent earned more than independent play without a correct answer) and gave
no valid verdict on one.

Against the base, the penalty registers. 37 traces cite it, against 4, and mean reward per agent
falls from 7.24 to 5.23. The salience line does not raise attention to the workspace (20 traces
name it, against 21), and behaviour does not change. No blind agent waits for round 2 (0/160),
one blind solver lists the empty workspace in episode 1 (seed 7104, bob, the twin of seed 7003 in
the base), and nobody writes. Answers of 0 or not found rise a little (80/160, against 67/160). As
in the base, about six reflections name the workspace as a place to get help ("seek help from
teammates if possible", seed 7110, carol; "there was a way to find it through the shared workspace
or another teammate's", seed 7108, alice, wondering why she was rejected), and no reflection was
followed by a workspace call in a later episode. The nearest any reviewer came to sharing is seed
7104, episode 2, carol, "maybe I can use the shared workspace to note down the answer for her
task", not acted on.

## Did coordination emerge

No, in neither condition. In the base condition the agents had what the design meant to give them. The blind solvers knew material was missing and who had it (133 of 162 blind solve calls say the reviewer holds the copy or data, 113 of 160 blind reflections cite the missing data), every blind submission was rejected, and the workspace was in every system prompt and every tool list. They still guessed and moved on. 67 of 160 blind answers are 0 or not found, and 21 of 54 blind code answers took the INPUT from another task. Only one blind solver looked in the workspace (seed 7003, episode 1), found it empty and never looked again. Of the 21 traces that name the workspace, 19 are reflections. Most propose it as a personal scratchpad, and about five as a way to cross-check with or get help from teammates ("allowing team members to cross-verify data that's partially hidden"). None of these was acted on in a later episode. Nobody ever wrote. The missing step is the first writer. No reviewer posted its material unprompted, no solver posted a request, and no blind agent used the round-2 opening (1 of 160, a truncation artifact). With guessing free, the base condition gives no step to detect and no edges to recover. Condition two does not change this, which makes it the finding for the emergence question on 14B. The penalty is noticed and costs the agents, and the workspace is described as readable by every teammate, yet no blind agent holds its submission for round 2 and nobody writes. So the missing piece is not an incentive to persist or knowing the workspace is shared. It is that 14B never takes the first step of writing to a teammate, either a request or an unprompted post of review material, in eight rounds. Whether reading and relaying would follow once someone writes is open, and testing it needs a first write the agents did not make, which is a plant and belongs to the 2a design.

## What the replay needs to know

Run directories, all under
`/Users/ayseozdincer/code/swarm/.claude/worktrees/lane-e-2b-713994/runs/` (gitignored, so they
exist only in this worktree; copy them before the worktree is cleaned up, about 180 MB).
- `smoke_shared/seed7000/shared_file/` base condition, one run
- `shared/seed7001/shared_file/` to `shared/seed7010/shared_file/` base condition
- `shared_c2/seed7101/shared_file/` to `shared_c2/seed7110/shared_file/` condition two

Each holds `snapshots.jsonl` (every model call as served, `messages` and an aligned
`message_meta`, the `tools` actually offered, `round` and `step`, and the response with its
`reasoning`), `run.json` (per episode results, full contexts at episode end, `tool_log`,
`workspace`, `run_config` with `blind_plan`, `wrong_penalty` and `salience`), `tool_log.jsonl`
and `audit.json`.

Rendering differs from 2a in three ways.
- Take the tools from the snapshot. In 2b the solve and verdict calls offer the submit tool plus
  the three workspace tools, reflections offer only the workspace tools, and the last allowed call
  of a turn offers only the submit tool. `get_task_tool_schemas` is wrong for these runs.
- A turn can be several calls. Key a snapshot by `(agent, episode_index, phase, round, step)`,
  not by `(agent, episode_index, phase)`. Base runs have one call per turn except the one
  `list_files` turn and turns that hit the token cap.
- Earlier episodes' task material is a one-line stub in later contexts (`compacted` in
  `message_meta`). The snapshot holds the context exactly as served, so replay from snapshots.

Measured episodes. No run has a first-coordination episode (no read of another agent's bytes in
any run), so the swarm-coherence step test has no positive round. Useful turns are
- the blind solve turns, 16 per run, listed in `run_config.blind_plan`, with the solver's own
  reasoning that its reviewer holds the piece
- the one base turn that looked in the workspace, seed 7003, episode 1, bob, solve round 1, step 0 (it lists, sees an empty workspace and submits 0 at step 1), and its condition-two twin, seed 7104, episode 1, bob, solve round 1, step 0. These are the only turns where the swarm's resource entered a context, and both returned an empty listing
- all eight episodes of every run as a null series for the swarm statistic, where no step is
  expected

The two masks. The declared mask is every message with `provenance_declared` equal to `peer`,
and in 2b there are none (no notes digest, the submission under review is `peer_artifact` and
sanctioned). The derived mask adds every tool message with `provenance_derived` equal to
`peer_artifact`. With no cross-agent reads, the derived mask equals the declared mask in every
turn of every run, so the reference pass equals the actual pass, and these runs are a free noise
floor. The pass criterion, derived edges recover the declared or owner edges, is vacuous here
(zero against zero). It was checked only in mock, where the derived mask matched the owner record
on all 455 edges of three seeds, and a unit test shows the one failure mode, identical text
written by two agents. If the replay reads "declared mask" as the owner record (what a designer
who declared the directory a channel would mask), use `provenance_owner`.

## What did not work

- Coordination did not emerge, so the derived-mask test and the coherence-step test have no
  positive instances in live data.
- The first smoke was stopped after episode 1. A blind code solver read "the reviewer's copy" as
  the review material in its own message and used another agent's answer as its INPUT. The fix
  (commit `9415256`) names the reviewer who has the withheld piece and marks the review material
  as a different task. It reduced but did not remove the confusion. 21 of 54 blind code
  submissions in the base condition (24 of 53 in condition two) still took the INPUT from another
  task's material or answer.
  That content is designer-sanctioned review material, not peer content, so it is not a leak, but
  the replay should expect task-mixing in those turns.
- A Wi-Fi switch around 01:14 UTC (17:14 in Los Angeles) stalled calls for about 11 minutes. The client retried and no
  seed failed; seed 7006 took 65 minutes instead of about 35.
- Six calls hit the 12000-token cap in the base condition (four in condition two), usually in
  a repetition loop. One
  of them is the only base blind submission made in round 2 (seed 7006, episode 8, bob), so that
  round-2 entry is a truncation artifact, not a choice to wait.
- Condition two ran alongside the base sweep (four seeds, up to 24 calls in flight across two
  processes) so both would finish tonight. No rate-limit failure was seen. The stop at twenty seeds
  (plus the smoke) was kept.
- `list_files` output carries file names other agents chose and is tagged `environment` in both
  tags. It never mattered here because nothing was written.
