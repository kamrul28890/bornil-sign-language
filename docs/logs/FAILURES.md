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
