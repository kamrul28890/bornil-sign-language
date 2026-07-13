# Bornil — Failure & Fix Log

> Failures are data. Every crash, divergence, disappointing result, or dead-end
> gets an entry: what we expected, what happened, why (root cause if known), and
> the fix or next hypothesis. This is what makes the "reasons + possible fixes"
> requirement real, and it becomes the paper's Limitations / negative-results notes.
> Newest at top.

## Entry template

```
### FAIL-<id>  (YYYY-MM-DD)  [phase/task]
- **Expected:** what we thought would happen
- **Observed:** what actually happened (numbers, error, curve shape)
- **Root cause:** confirmed cause, or "hypothesis:" if not yet proven
- **Fix / next step:** what we changed, or the next experiment to isolate it
- **Status:** open | fixed | wontfix | superseded-by <FAIL-id>
- **Links:** EXP-id(s), commit, config
```

---

<!-- Log entries below, newest first -->

### FAIL-001  (2026-07-13)  [P0 / infra]
- **Expected:** file writes to the repo on D: to succeed.
- **Observed:** `ENOSPC: no space left on device`; D: at 100% (0 bytes free).
- **Root cause:** confirmed — D: held `chunks/` (172 GB videos, untracked/gitignored) plus **~86 GB of dangling git objects** (`git count-objects`: 7,476 objects / 86.27 GiB loose, in-pack 0; only **72 reachable** from all refs). The dangling blobs (~52 MB each, thousands of them) came from large files `git add`ed but never committed, then unstaged.
- **Fix / next step:** user freed ~99 GB manually (D: now 79% used) — unblocked. Remaining cleanup to reclaim ~86 GB: `git reflog expire --expire=now --all && git prune --expire=now && git gc` (prune unlinks directly → low temp-space need, safe when near-full). Prevention: ensure `chunks/` and feature dirs stay in `.gitignore` (they are, via merged teammate `.gitignore`); never `git add .` at repo root with videos present.
- **Status:** open (space unblocked; 86 GB reclaim pending user go-ahead)
- **Links:** CHANGELOG 2026-07-13 config/transforms/disk entry
