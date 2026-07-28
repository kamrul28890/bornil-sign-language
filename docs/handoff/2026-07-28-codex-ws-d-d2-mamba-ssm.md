# Codex handoff — WS-D D2 (pure-PyTorch selective-SSM / Mamba encoder)

> Copy everything below the line into Codex as your task prompt.

---

You are working autonomously in the Bornil repo at `d:\My Projects\Bornil`
(Windows) for the next several hours, executing a pre-approved implementation
plan task-by-task. This is a real research codebase (Bangla sign-language
translation, NeurIPS D&B submission in progress). WS-D D1 (a different
sub-project, self-supervised keypoint pretraining) was already completed
this way and closed as a negative result — this is an independent
sub-project and does not depend on D1's code or outcome.

## Your source of truth

**`docs/plan/2026-07-28-ws-d-d2-mamba-ssm.md`** is the plan: 9 tasks, each
broken into 2–5 minute steps with checkbox (`- [ ]`) tracking, exact file
paths, complete code, exact commands, and exact expected output. Read the
whole plan file once before doing anything else. Also skim its design doc,
**`docs/superpowers/specs/2026-07-28-ws-d-d2-mamba-ssm-design.md`**, for the
*why* behind the non-obvious decisions — most importantly:

- Why this is a **pure-PyTorch** reimplementation of Mamba rather than the
  official `mamba-ssm` package (confirmed: no Windows wheels, and this
  machine has neither the CUDA Toolkit `nvcc` nor an MSVC compiler needed to
  build it from source).
- Why the master plan's original "128-frame cap" motivation for this
  workstream is stale (the pipeline already trains at 512 frames; only
  2.1% of clips are actually truncated there) — D2 is an architecture
  comparison, not a data-recovery story.
- A **pre-registered prediction** (design doc, "Pre-registered prediction"
  section) recorded before any code exists — read it, and when you reach
  Task 9's write-up, explicitly state which of its four predicted outcomes
  the real result matched. Do not skip stating this even if the result is
  the "boring" predicted one.

Do not re-plan, re-scope, or "improve" the task breakdown. If a task's
instructions don't match the actual current state of a file (an anchor
string not found because something changed), stop and report the mismatch
rather than guessing a fix that changes the plan's intent.

## Resumability

Same protocol as any other plan in this repo:

1. On start (or restart), read the plan file top to bottom, find the first
   unchecked `- [ ]` box — that's where you resume.
2. Cross-check against `git log --oneline`: every completed task ends in a
   commit with the exact message given in that task's last step. If the
   plan file's checkboxes and git history disagree, **trust git history**.
3. As you complete each step, flip its checkbox to `- [x]` and include that
   edit in the task's commit.
4. Never redo a task whose commit already exists without first confirming
   its tests still pass.

## Environment specifics

- Python: `.venv/Scripts/python`, always — never a bare `python`.
- **Import order rule:** every script imports `bornil.runtime` before
  `pandas` — already baked into every code block in the plan.
- GPU: RTX 3070, 8GB VRAM, CUDA 12.4. Use `--device cuda` for every real
  run (Tasks 6 and 7). Smoke tests can use either.
- Shell: PowerShell has no `&&` — chain with `;` or run commands one at a
  time; a bash-compatible shell can use `&&`. Check which you have.
- **No new dependencies.** This plan deliberately avoids `mamba-ssm` and
  any CUDA-toolkit/build-tooling installation — if you find yourself
  reaching for `pip install` of anything beyond what's already in this
  venv, you've gone off-plan. Stop and reconsider.

## Non-negotiables

- **TDD, strictly** — failing test first, watch it fail for the stated
  reason, then implement, then watch it pass.
- **No behavior change to any existing model, script, or config.** This
  plan only adds new files plus one small, explicitly-scoped dispatch
  branch each in `bornil/models/build.py` and
  `scripts/train_sample.py`'s `_append_experiment_row` (Tasks 4 and 5) —
  mirroring how `"bilstm"`/`"stgcn"` were added, not restructuring anything.
- **Never touch the locked metric protocol** (`bornil/benchmark/metrics.py`).
- **Logging discipline is not optional** — `EXPERIMENTS.md` rows are
  automatic via the existing scripts (confirm the Mamba run is actually
  labeled `SignMamba`, not `SignTransformer` — that's exactly what Task 5's
  labeling fix is for). Meaningful changes → `CHANGELOG.md`. The predicted
  or any negative result → `FAILURES.md`, using the exact template already
  in that file (copy FAIL-005/006/007's format). **A predicted negative
  result is a completed task, not a failed one** — do not tune `d_state`,
  `expand`, `conv_kernel`, or `num_mamba_layers` against Task 7/8's results
  to manufacture a win. Those hyperparameters are fixed once in Task 5 by
  design, specifically so this comparison stays honest.
- **Commit after every task**, using the exact message given in that
  task's final step.
- **Verification before completion** — run the actual command, read the
  actual output/metrics file, then check the box. Never infer or estimate
  a number you could have just read.

## The one task with genuine uncertainty: Task 6

Unlike D1 (where compute cost was roughly predictable from prior runs),
**this plan does not know in advance how slow the sequential selective-scan
will train.** Task 6 is a mandatory smoke-test-and-decide gate before the
real Task 7 run — read it carefully; it's not a formality. The two prior
anchors on this machine (TConv+CTC at 0.75 min/epoch, ST-GCN at 42.8
min/epoch) span a 57× range and neither directly predicts a Python-loop-
based architecture's cost, so measure Task 6's smoke test honestly and
follow its decision tree — don't assume a number and don't silently shrink
the real run to make a bad number disappear.

## Long-running steps

Tasks 6 (smoke test, minutes) and 7 (real run, ⏱ — anywhere from under an
hour to potentially much longer per Task 6's own finding) are training
jobs. Task 8 re-decodes the test set twice more (read-only, no training,
should be fast — a single decode pass, not 60 epochs). Let each finish;
don't reduce epoch counts to speed things up on the real run (Task 6's
smoke test already exists for that purpose).

## Hard stop conditions

- **Task 6's decision tree finds the full run would take multiple days.**
  Per Task 6 Step 2: log the measured numbers to `FAILURES.md`, and treat
  D2 as blocked pending a parallel-scan reimplementation or a cloud
  environment — both explicitly out of scope here. Do not attempt either
  on your own initiative. Stop and report.
- **Any point where a plan step's anchor text doesn't match the actual file
  state.** Don't improvise a fix that changes intent — stop, describe the
  mismatch precisely, and report it rather than guessing forward.
- **Anything destructive or scope-expanding** — force-push, `git reset
  --hard`, deleting another run's `data_full/runs/` output, installing
  build tooling, touching files outside this plan's stated scope. Stop and
  report instead of acting on your own judgment.

Treat a hard stop as: finish logging it properly (commit the log entry),
then end your turn with a clear, complete status report — you're running
unattended, not waiting for a reply.

## Final deliverable

When you reach Task 9 (or a hard stop), produce the report Task 9 Step 3
asks for: Task 6's measured timing and what it implied; the real run's
config hash and metrics; significance vs vanilla and vs TConv+CTC with
p-values; the length-stratified verdict and which of the design doc's four
pre-registered outcomes it matched; any `FAILURES.md` entry added; the full
commit list; full test-suite status; and an explicit statement of whether
D2 is a supported win for the WS-D comparison. "No" is a completely
acceptable, useful answer as long as it's backed by the actual numbers —
that's exactly what happened with D1.
