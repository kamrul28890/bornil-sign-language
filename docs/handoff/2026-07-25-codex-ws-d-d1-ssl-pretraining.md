# Codex handoff — WS-D D1 (masked keypoint modeling pretraining)

> Copy everything below the line into Codex as your task prompt.

---

You are working autonomously in the Bornil repo at `d:\My Projects\Bornil`
(Windows) for the next several hours, executing a pre-approved implementation
plan task-by-task. This is a real research codebase (Bangla sign-language
translation, NeurIPS D&B submission in progress) with an established
convention for exactly this kind of unattended, multi-hour agentic work —
follow it, don't improvise a different workflow.

## Your source of truth

**`docs/plan/2026-07-25-ws-d-d1-ssl-pretraining.md`** is the plan. It has 12
tasks, each broken into 2–5 minute steps with checkbox (`- [ ]`) tracking,
exact file paths, complete code (not sketches — copy it, don't rewrite it
from the prose), exact commands, and exact expected output. Read the whole
plan file once before doing anything else. Also skim its design doc,
**`docs/superpowers/specs/2026-07-25-ws-d-d1-ssl-pretraining-design.md`**,
for the *why* behind the two non-obvious decisions (where masking is
injected relative to the TConv frontend, and why D1's "unlabeled data"
premise from the master plan doesn't hold in this repo) — you don't need to
re-derive either of those, they're already settled.

Do not re-plan, re-scope, or "improve" the task breakdown. If you think a
task is wrong, say so in your final report rather than silently deviating —
but execute what's written unless it is actually broken (a command that
errors, a test that can't pass as specified, code that doesn't match the
actual current state of the file it's modifying).

## Resumability — read this before starting, and again if you're restarting

This session may get interrupted and resumed. The plan file's checkboxes
**are** the resumability record:

1. On start (including a restart), read the plan file top to bottom and find
   the first unchecked `- [ ]` box. That's where you resume.
2. Cross-check against `git log --oneline` — every completed task ends in a
   commit (each task's last step says exactly what to commit and with what
   message). If the plan file's checkboxes and git history disagree, **trust
   git history**: a task with a matching commit already landed even if a
   checkbox wasn't ticked; a task with no commit hasn't landed even if a
   checkbox looks ticked.
3. As you complete each step, edit the plan file to flip `- [ ]` to `- [x]`
   for that step, and include that edit in the task's commit (or a
   fast-follow commit) so the file stays accurate for whoever/whatever reads
   it next.
4. Never redo a task whose commit already exists without first running its
   tests to confirm they still pass. If they don't, that's a real problem —
   investigate before continuing, don't just rerun blindly.

## Environment specifics

- Python: `.venv/Scripts/python` (a real venv in the repo root — always use
  this, not a bare `python`).
- **Import order rule:** every script must `import bornil.runtime` before
  `import pandas` — this is already baked into every script template in the
  plan (`import bornil.runtime  # noqa: F401`). If you write any new script
  entry point not covered by the plan's code blocks, keep this import first.
- GPU: RTX 3070, 8GB VRAM, CUDA 12.4. Use `--device cuda` for every real
  training run in the plan (not the smoke tests, which can be CPU or GPU,
  your call — but they're fast either way).
- Shell: if you have PowerShell, it has no `&&` — chain commands with `;` or
  run them one at a time. If you have a bash-compatible shell instead, `&&`
  is fine there. Check which you actually have; don't assume.
- No new dependencies are needed for this plan. If a `pip install` ever
  looks necessary, stop and reconsider — you've likely gone off-plan.

## Non-negotiables (from the plan's own "Global Constraints" section — repeated here because they matter most)

- **TDD, strictly.** Write the failing test, run it, watch it fail for the
  stated reason, then implement, then watch it pass. Do not write
  implementation first and backfill tests — the plan is written in that
  order for a reason (it's also how you catch a wrong assumption before it's
  baked into working code).
- **No behavior change to existing configs.** `configs/full_ctc.yaml` and
  every other existing config must train identically after this plan lands.
  Task 3's refactor is a pure decomposition — the real regression check is
  running the existing golden test (`tests/models/test_decode_golden.py`)
  unchanged, not just adding a new unit test.
- **Never touch the locked metric protocol** (`bornil/benchmark/metrics.py`)
  or any existing model/eval behavior outside what the plan explicitly says
  to modify.
- **Real frozen splits for real runs** — never `--no-frozen-splits` outside
  of unit tests.
- **Logging discipline is not optional.** Every real training run gets an
  `EXPERIMENTS.md` row (the plan's scripts do this automatically).
  Meaningful changes go in `docs/logs/CHANGELOG.md`. Failed gates and
  negative results go in `docs/logs/FAILURES.md` using the exact template
  already in that file (copy the format of FAIL-004 or FAIL-005 — id,
  date, phase/task, Expected/Observed/Root cause/Fix-or-next-step/Status/
  Links). **A negative result is a completed task, not a failed one** — this
  repo's convention (and this plan's explicit instruction at Tasks 9–11) is
  to report negative results plainly, never to quietly tune hyperparameters
  against an eval split to manufacture a win. If you catch yourself
  reaching for "let me just adjust the mask ratio and see if that helps"
  after looking at finetune results, stop — that invalidates the
  significance test the plan is built around.
- **Commit after every task**, using the exact message given in that task's
  final step (paraphrase only if the plan explicitly allows it, e.g. Task 9's
  Step 4 conditional commit message).
- **Verification before completion.** Never report a step, task, or the
  whole plan as done based on what the code *should* do — run the actual
  command, read the actual output, and only then check the box. This applies
  doubly to the real training runs in Tasks 9–11: read the actual
  `metrics.json` / `pretrain_metrics.json` numbers, don't infer or estimate
  them.

## Long-running steps — this is expected, not a problem

Tasks 9, 10, and 11 are real GPU training jobs, marked ⏱ in the plan.
Individually they can take anywhere from ~10 minutes to a couple of hours;
Task 11 alone is 7 of them. This is why the plan is structured as
independently-committable, resumable tasks — you are expected to spend most
of this multi-hour session waiting on training, not writing code. Let each
run finish; do not reduce epoch counts or dataset fractions to make a real
run faster (that invalidates the result) — the *smoke tests* in Tasks 7–8
already exist for fast iteration and explicitly use `--epochs` overrides for
that purpose.

## Hard stop conditions

- **Task 9's gate (pretraining reconstruction loss must decrease and be
  finite).** If it fails: log to `FAILURES.md` per the plan's Step 2, commit
  that, and **stop — do not execute Tasks 10, 11, or 12.** A pretrain
  checkpoint that failed its own sanity check is not a valid input to a
  finetune comparison; running the rest of the plan on top of it would
  produce numbers that look real but aren't.
- **Any point where the plan's instructions don't match the actual state of
  a file** (e.g., a "Modify" step's anchor text isn't found because an
  earlier task changed something unexpectedly). Don't guess at a fix that
  changes the plan's intent — stop, note exactly what you found vs. what the
  plan expected, and continue with other still-valid tasks if any exist
  independently, flagging the mismatch clearly in your final report.
- **Anything that would require a destructive git operation** (force-push,
  `git reset --hard`, deleting another experiment's `data_full/runs/`
  output) or touching files outside this plan's stated scope. Don't take
  scope-expanding or destructive actions on your own judgment — stop and
  report instead.

None of these are "ask a human and wait" in the interactive sense — you're
running unattended. Treat a hard stop as: finish logging it properly
(commit the log entry), then end your turn with a clear, complete status
report rather than guessing forward.

## Final deliverable

When you reach Task 12 (or a hard stop), produce the report Task 12 Step 3
asks for: pretraining gate result, the `signerindep-v1` core-claim verdict
with its p-value, the full 8-point data-efficiency curve and what shape it
takes, any `FAILURES.md` entries you added, the full commit list for this
plan, full test-suite status, and an explicit statement of whether D1 is a
supported win — that verdict is what this plan exists to produce, and "no"
is a completely acceptable answer as long as it's backed by the actual
numbers.
