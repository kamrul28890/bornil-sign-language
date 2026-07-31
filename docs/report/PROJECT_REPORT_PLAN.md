# Project Report — Writing Plan (DRAFT for review, v3)

> **Status: awaiting your review/edits.** Nothing below is the report itself
> — this is the blueprint. Edit anything (section order, what to include/cut,
> tone, length), then tell me to proceed and I'll write the actual report
> from this plan.

> **v3 changes:** scope is now a **full, comprehensive report — target
> ~20 pages, acceptable up to ~50 — compiled to a single PDF, "everything
> in it, however long."** This is no longer a brief; it's an exhaustive
> technical report. Expanded outline accordingly: a dedicated subsection per
> experiment/architecture (not just a summary table row), a Limitations
> section, an Ethical Considerations section, a Reproducibility section, a
> fuller appendix with complete tables. Added a PDF-compilation plan.

## 1. Purpose & audience

**Who reads this:** someone with **zero background** in sign language
research, machine learning, or this project specifically — but who wants
the **complete picture**, exhaustively, not a summary. A thorough
report/paper-style document a total beginner could follow end to end
(every concept explained on first use) while still covering everything a
technical reader would expect from a full project report: background, data
analysis, methodology, every experiment run, full results, limitations,
ethics, and reproducibility.

**What "success" looks like:** a reader finishes it able to explain, in
detail: what problem this solves and why it's hard, exactly what the data
looks like (with real distributions, not just headline counts), exactly how
the system works stage by stage, how every model variant was built and why,
what every experiment found — including the ones that didn't work and why
that's still useful — what the real numbers say with the tables/charts that
back them, what the limitations and open risks are, and what's next.

**What this is NOT:** not literally the NeurIPS submission (a separate,
more formally-structured academic artifact), but should read as a complete
standalone technical report thorough enough to serve as its backbone or an
appendix to it. No code snippets — describe what components do and why, not
implementation-line detail (that's what the linked repo docs are for).

## 2. Tone & style rules (apply to every section)

- **Define every technical term the first time it appears, in plain words,
  with a one-line analogy if it helps.**
- **No unexplained acronyms** — BLEU, chrF++, ROUGE-L, CTC, SSM, LoRA, VRAM,
  GPU, checkpoint, epoch, encoder/decoder, gloss-free, keypoint, etc., all
  defined on first use.
- **Show, don't just tell — a table or chart wherever a claim can be
  visualized**, not only in a results section. See §4's per-section
  chart/table list; this report should feel visually rich, not a wall of
  text.
- **Concrete before abstract** — every claim followed by a real example:
  an actual Bangla sentence, an actual model output, an actual number from
  a real run. Nothing invented.
- **Short paragraphs, active voice, no hedging jargon.**
- **Comprehensive over concise** — given the length target, prefer fully
  explaining a topic (with its own dedicated subsection where it earns one)
  over compressing several topics into one paragraph. It's fine, expected
  even, for individual experiments to each get real space.
- **Honesty about what didn't work is a feature, not an embarrassment** —
  presented as: a real hypothesis, a real test, a real result (sometimes
  negative), the best-guess reason why, and what it ruled out.

## 3. Available source material (so nothing gets invented)

Every number, example, chart, and claim traces back to a real file.

**Written docs:**

| Need | Source |
|---|---|
| Dataset stats, real example sentences, task framing | `docs/PIPELINE_WALKTHROUGH.md` |
| Pipeline figures (already generated) | `docs/assets/pipeline/01_frame_strip.png` … `06_landmark_selection.png` |
| Headline results, comparison tables, significance | `docs/logs/RESULTS_OVERVIEW.md` (Tables 1–6) |
| Real (reference, hypothesis) example translations | `docs/logs/QUALITATIVE_EXAMPLES.md` |
| Full run history / every model ever trained | `docs/logs/EXPERIMENTS.md` |
| Chronological narrative of what changed and why | `docs/logs/CHANGELOG.md` |
| Every negative result with root-cause reasoning | `docs/logs/FAILURES.md` (FAIL-001 through FAIL-008 as of this writing) |
| Grounding/limitation findings, paper-style prose | `docs/paper/GROUNDING_ANALYSIS.md` |
| Sentence-length distribution (already computed) | `docs/logs/TARGET_LENGTH_AUDIT.md` |
| Split composition / per-split stats | `docs/logs/BASELINE_TABLE.md`, `splits/*/meta.json` |
| Per-model significance tests (one file per comparison) | `docs/logs/SIGNIFICANCE_*.md` |
| Decode-stability / repetition-guard findings | `docs/logs/DECODE_REPETITION_GUARD.md`, `DECODE_ABLATIONS.md` |
| WS-E alignment methodology + negative result | `docs/plan/2026-07-16-ws-e-alignment.md`, `docs/logs/ALIGNMENT_VALIDATION.md` |
| WS-D D1/D2/D3 designs, plans, and outcomes | `docs/superpowers/specs/2026-07-2{5,8}-ws-d-*.md`, `docs/plan/2026-07-2{5,8}-ws-d-*.md`, `docs/logs/FAILURES.md` FAIL-006/007/008 |
| Overall research plan, what's still ahead | `docs/plan/2026-07-13-bornil-implementation-plan.md` |
| Repo orientation, current state, data-exclusion notes | `README.md`, `docs/DATA_AND_OUTPUTS.md` |

**Raw data (for generating new EDA charts):**

| Need | Source |
|---|---|
| Per-clip sentence text, frame count, chunk/session, metadata | `video_data.csv` |
| Per-clip true frame length (post-extraction) | `data_full/lengths.npy` + `data_full/sample_manifest.csv` |
| Split membership (train/val/test per split kind) | `splits/random-v1/`, `splits/signerindep-v1/`, `splits/unseen-sentence-v1/` |
| Subword vocabulary | `bsl_sp.vocab` |

New charts generated from these via a small script (mirroring this
project's existing `scripts/make_pipeline_figures.py` pattern), styled per
this repo's established `dataviz` visualization conventions — not
hand-estimated.

## 4. Full outline (target ~20 pages, up to ~50 acceptable)

### 4.1 Introduction
Problem statement, why it matters, what this project does, brief numerical
teaser (full EDA in §4.3), roadmap of the report itself (what each section
covers, since this is now long enough to warrant one).

### 4.2 Background & related context
- Sign language translation as a task; why it differs fundamentally from
  spoken-language translation.
- Gloss-based vs. gloss-free approaches, explained plainly (reuse/expand
  `PIPELINE_WALKTHROUGH.md` §2).
- Why Bangla Sign Language specifically is under-resourced, and what a
  dedicated open benchmark contributes.
- This project's specific contributions, stated explicitly as a list: the
  dataset, the frozen benchmark (three generalization splits), the metric
  protocol, the baseline suite, the novel-method comparison, the alignment/
  interpretability study, and the negative-results record.

### 4.3 The dataset & exploratory data analysis
Full EDA, each with a real chart and/or table:
1. Dataset overview (counts, domains, real example sentences, clip↔label
   relationship).
2. Sentence length distribution (token-count histogram + the real bucketed
   table from `TARGET_LENGTH_AUDIT.md`).
3. Clip duration / frame-count distribution (histogram; ties to the
   512-frame processing ceiling and how rarely it's actually hit).
4. Vocabulary analysis (word-frequency/Zipf-style chart; the 58%-appear-once
   statistic and what it implies for tokenization).
5. Session/signer diversity (bar chart of clips per recording session; ties
   directly to the `signerindep-v1` split's purpose).
6. Data quality notes (the real landmark-visibility numbers behind the
   upper-body-only feature decision).
7. Split composition table (all three splits, train/val/test counts, and
   what generalization question each answers).
8. A short EDA summary paragraph tying the above findings to modeling
   decisions made later in the report (signals the reader that this wasn't
   decorative — every EDA finding fed a real decision).

### 4.4 System architecture
The labeled, plain-language pipeline walkthrough (video → keypoints →
model → text), with the analogy-driven explanations from the v2 plan,
expanded with a slightly more detailed breakdown of the encoder/decoder
split and why a sequence model is the right fit for this data shape.

### 4.5 Methodology

**4.5.1 Data preprocessing** — full plain-language walkthrough (frame
sampling, keypoint extraction, normalization, landmark selection,
tokenization), each with its real "why," embedding the existing pipeline
figures.

**4.5.2 Model architectures — one dedicated subsection per approach**, not
compressed into a single table (though a summary comparison table opens
this subsection for orientation):
- 4.5.2.1 Baseline sequence-to-sequence model
- 4.5.2.2 Adding a short-range motion component (temporal convolution)
- 4.5.2.3 Adding a second training signal for video/sentence alignment
  (CTC) — the best-performing variant; explain plainly why it helped
  (fixes under-generation, improves output diversity — real numbers from
  `RESULTS_OVERVIEW.md` Table 5)
- 4.5.2.4 Alternative reading design: graph/skeleton-structured encoder —
  what it tested, why it was a reasonable idea, and its real (negative)
  outcome
- 4.5.2.5 Alternative reading design: a newer, more efficient
  sequence-modeling approach — what it tested, and the real hardware/speed
  wall it hit before completion
- 4.5.2.6 Self-supervised pre-training variant — what head start it tried
  to give the model, and its real (negative) outcome, including the
  data-efficiency curve finding
- 4.5.2.7 In progress: a pretrained-Bangla-language-model decoder — what
  question it's asking, current status

**4.5.3 Training & evaluation protocol**
- Frozen splits, why three of them, what each tests (ties to §4.3.7).
- Metric protocol — chrF++, BLEU, ROUGE-L, BERTScore — each explained
  plainly with what it measures and why chrF++ is the one this project
  leans on most.
- Significance testing methodology (paired bootstrap, explained plainly:
  "resampling the test set many times to check whether one model's edge
  over another is real or could be luck").
- Compute/hardware notes, honestly scoped (ranges, not false precision).

### 4.6 Results

- Full headline comparison table (every model, every split it was run on,
  every metric) — the comprehensive version of `RESULTS_OVERVIEW.md`
  Tables 1–2b.
- Bar chart(s) of chrF++ across models.
- Trivial-floor comparison (retrieval baseline, most-frequent-sentence
  baseline) with the real, important caveat already documented (a
  noise-input control scores close to the retrieval floor, so "beating
  retrieval" alone is a weak claim — explain why the video↔text mismatch
  control is the stronger evidence).
- The grounding evidence in full: the input-ablation results table
  (mismatch / frame-shuffle / noise / zero conditions), explained plainly,
  chart of chrF++ across conditions.
- The "static pose, not motion" finding, explained plainly with the real
  numbers.
- Real example translations (exact match, typical, failure) from
  `QUALITATIVE_EXAMPLES.md`, across at least two splits.
- Data-efficiency curve chart (from the pre-training experiment).
- Length-stratified results, if available by the time of writing (from the
  in-progress D3 work) — otherwise noted as forthcoming.
- Per-model diversity/length diagnostics (unique-hypothesis rate, output
  length calibration) — real numbers from `RESULTS_OVERVIEW.md` Table 5,
  including the runaway-repetition failure mode found in one model.

### 4.7 Discussion

- What the results collectively say about the state of the system.
- Direct synthesis of every negative result (FAIL-004 through FAIL-008) into
  one coherent narrative: what was tried, what was ruled out, what remains
  open — not just a list, but the throughline connecting them (e.g. the
  static-pose finding predicts and explains why a motion-focused
  architecture and an alignment-to-motion-changepoints study both came back
  negative).

### 4.8 Limitations
*(new, dedicated section)*
- Static-pose-dominant grounding (the main named technical limitation).
- Signer/session diversity (36 sessions; the `signerindep-v1` generalization
  gap that remains).
- Long-sentence performance (tie to §4.3.2's length distribution).
- Compute constraints encountered (the Mamba/SSM compute wall) and what
  they mean for what could/couldn't be tested on available hardware.
- Benchmark-level limitations (single dataset, one sign language, BLEU-4 at
  noise floor for this benchmark).

### 4.9 Ethical considerations & broader impact
*(new, dedicated section)*
- Accessibility framing — who benefits and how, stated concretely.
- Signer consent/privacy considerations for the underlying dataset (state
  plainly what's known/assumed; flag if this needs a follow-up from the
  team rather than inventing a claim).
- Responsible-use notes (e.g., a translation system with real error rates
  should not be presented as more reliable than it is in any deployed
  context).

### 4.10 Reproducibility
*(new, dedicated section)*
- What's openly available (code, frozen splits, tokenizer, configs) vs. what
  isn't (raw video, packed features, checkpoints) and why — summarizing
  `docs/DATA_AND_OUTPUTS.md` for a report audience.
- How someone could reproduce any single result in this report exactly
  (config file → script → the same `EXPERIMENTS.md` row).

### 4.11 Conclusion & future work
- Restate the contribution.
- The in-progress and next-decided work (from v2's "what's next" content).
- Longer-horizon direction (publication target, closing the grounding gap).

### 4.12 Glossary
Every plain-language definition used inline, alphabetized.

### 4.13 Appendix
- Complete run ledger (condensed `EXPERIMENTS.md` table).
- Complete significance-test summary (one row per comparison, pointing at
  the source `SIGNIFICANCE_*.md` file).
- Full split-composition and dataset-statistics tables (the detailed
  versions behind §4.3's charts).

## 5. PDF compilation plan

- **Source of truth:** write the report in Markdown at
  `docs/report/PROJECT_REPORT.md` (git-friendly, diffable, matches this
  repo's convention, easy for you to review/edit directly) plus generated
  chart images under `docs/report/assets/`.
- **Compiling to PDF:** this machine already has a working LaTeX
  installation (MiKTeX's `pdflatex`), which is the hard part. It's missing
  `pandoc`, the standard tool that converts Markdown → PDF using that LaTeX
  engine automatically (tables, figures, headings all handled). **Needs one
  confirmation from you: OK to install `pandoc`** (a widely-used, standard
  document-conversion tool — not a research dependency, just a build tool
  for this one output step)? Once installed, the whole report compiles with
  one command (`pandoc docs/report/PROJECT_REPORT.md -o
  docs/report/PROJECT_REPORT.pdf --pdf-engine=pdflatex ...`, with a few
  formatting flags for a clean title page/table of contents/numbered
  sections given the length).
- If you'd rather not install anything new, the fallback is writing the
  report directly in LaTeX instead of Markdown — fully compilable with the
  `pdflatex` already installed, no new tools — at the cost of being harder
  for you to review/edit directly compared to Markdown. My recommendation
  is the `pandoc` route.

## 6. Open questions for you to decide during review

1. **Confirm the ~20-(up to ~50)-page, "everything in it" scope** — the
   outline in §4 is already built to that scope; flag anything you'd want
   cut rather than included at that length.
2. **OK to install `pandoc`** for the PDF compile step (§5)? If not, I'll
   write directly in LaTeX instead.
3. **Chart generation** — confirm you're fine with me generating new image
   files under `docs/report/assets/` from real project data (§3's raw-data
   table), matching this repo's existing visualization conventions.
4. **Ethics/consent section (§4.9)** — I don't currently have a documented
   answer for signer consent/data-release terms in this repo. Do you have
   that information to give me, or should this subsection explicitly flag
   it as "to be confirmed with the data-collection team" rather than
   assert something unverified?
5. **Should "what's next"/"future work" name a rough timeline?** Still
   leaning no (a wrong date undermines trust more than an honest "in
   progress, no fixed date") — your call.
6. **Any specific audience** (funder, school, particular stakeholder) that
   should shift emphasis beyond the generic "thorough, no prior background
   assumed" framing?
