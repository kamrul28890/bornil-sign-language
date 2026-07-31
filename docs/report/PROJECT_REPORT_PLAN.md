# Project Report — Writing Plan (DRAFT for review)

> **Status: awaiting your review/edits.** Nothing below is the report itself
> — this is the blueprint. Edit anything (section order, what to include/cut,
> tone, length), then tell me to proceed and I'll write the actual report
> from this plan.

## 1. Purpose & audience

**Who reads this:** someone with **zero background** in sign language
research, machine learning, or this project specifically. Could be a
supervisor, a sponsor, a family member, a new team member — anyone who needs
to understand *what this project is, what was built, whether it works, and
what's next* without being taught ML first.

**What "success" looks like for this document:** a reader finishes it able
to explain, in their own words, to a friend: what problem this solves, how
the system works at a sketch level, whether it currently works well, and
what's happening next. They should never hit a term they can't understand
from context.

**What this is NOT:** not the NeurIPS paper (that's a separate, technical
artifact this project is also building toward — this report can *feed*
that, but shouldn't try to *be* it). Not a code walkthrough. Not exhaustive
— depth is available by linking out to the technical docs already in the
repo, not by cramming everything in here.

## 2. Tone & style rules (apply to every section)

- **Define every technical term the first time it appears, in plain words,
  with a one-line analogy if it helps.** Examples: "chrF++ (a score from 0
  to 100 that measures how much of the correct sentence's *spelling* shows
  up in the model's answer, character by character — not just whole words)."
  "CTC (a training trick that lets the model align a long video to a short
  sentence without anyone manually marking which frame matches which word)."
- **No unexplained acronyms.** BLEU, chrF++, ROUGE-L, CTC, SSM, LoRA, VRAM,
  GPU, checkpoint, epoch, encoder/decoder, gloss-free, keypoint — every one
  gets a plain-language definition on first use, then can be used freely.
- **Concrete before abstract.** Every claim about what the system does
  should be followed (not preceded) by a real example — an actual Bangla
  sentence, an actual model output, an actual number from a real run. This
  project has real examples for everything (see §4's source map) — use
  them instead of describing things in the abstract.
- **Short paragraphs, active voice, no hedging jargon.** "The model got this
  wrong" beats "suboptimal performance was observed in this instance."
- **Honesty about what didn't work is a feature, not an embarrassment.**
  This project has several genuine negative results (see §4.4) — the report
  should present them the way the project's own internal logs do: "we
  expected X, we tried it, it didn't help, here's our best guess why, and
  here's what we're doing instead." A lay reader trusts a report more, not
  less, when it admits what didn't pan out.
- **Visuals over description wherever possible.** This project already has
  real pipeline figures (see §3) — use them instead of prose where a picture
  says it faster.

## 3. Available source material (so nothing gets invented)

Every number, example, and claim in the final report must trace back to one
of these already-real files — no invented statistics, no rounded-for-effect
claims. Pulled together here so the writing pass doesn't have to go
hunting:

| Need | Source |
|---|---|
| Dataset stats, real example sentences, task framing in accessible prose | `docs/PIPELINE_WALKTHROUGH.md` (already written in a fairly accessible style — good raw material to simplify further, not start from scratch) |
| Pipeline figures (already generated, ready to embed) | `docs/assets/pipeline/01_frame_strip.png` … `06_landmark_selection.png` |
| Headline results, comparison tables, significance | `docs/logs/RESULTS_OVERVIEW.md` |
| Real (reference, hypothesis) example translations — good and bad | `docs/logs/QUALITATIVE_EXAMPLES.md` |
| Full run history / every model tried | `docs/logs/EXPERIMENTS.md` |
| What changed and why, chronologically | `docs/logs/CHANGELOG.md` |
| Every negative result with root-cause reasoning | `docs/logs/FAILURES.md` (FAIL-001 through FAIL-008 as of this writing) |
| Grounding/limitation findings in paper-style prose | `docs/paper/GROUNDING_ANALYSIS.md` |
| The overall research plan and what's still ahead | `docs/plan/2026-07-13-bornil-implementation-plan.md`, plus the WS-D sub-plans/specs for D1/D2/D3 |
| Repo orientation, current state | `README.md`, `docs/DATA_AND_OUTPUTS.md` |

## 4. Full outline

### 4.1 Introduction
- **What is this about.** Plain statement: Bangla Sign Language (BdSL) has
  no widely available automatic translation to text — this project is
  building one. One sentence on why that matters (accessibility — letting
  Deaf BdSL signers communicate in text with people who don't sign).
- **What we are doing.** One paragraph, no jargon: we collected thousands of
  videos of people signing Bangla sentences, and we're teaching a computer
  program to watch a video and write out the sentence in Bangla, the same
  way a human interpreter would — without any middle step through English
  or through a fixed sign-by-sign dictionary.
- **Quick numerical snapshot of the data** (pulled from
  `docs/PIPELINE_WALKTHROUGH.md` / `README.md`, presented as a short
  labeled list, not a table full of jargon):
  - 17,988 video clips
  - Almost 18,000 *different* sentences (so the system has to genuinely
    understand sentences, not just memorize a fixed list)
  - Real topics: news, sports, everyday conversation
  - Clips range from a couple of seconds to about half a minute
  - ~180 GB of video in total
- Close the section with one sentence bridging to §4.2: "Here's specifically
  what we set out to build, and how far we've gotten."

### 4.2 What our target is / the end product
- High-level, no architecture talk yet: the target is a system that takes a
  video of BdSL signing and outputs a Bangla sentence.
- Two honest framings, both true, both worth stating: (1) a real assistive
  tool people could eventually use, and (2) a research contribution — a
  reusable public benchmark and a documented set of experiments other
  researchers can build on (this project is aiming at a NeurIPS Datasets &
  Benchmarks submission — explain in one plain sentence what that means:
  "a major AI research venue where new, well-documented datasets and
  benchmarks get published and reviewed like scientific papers").
- Set expectations honestly right here, in plain terms, so nothing later
  feels like a reveal: the system works and clearly beats naive baselines,
  but it isn't production-ready yet, and the report will explain exactly
  where it's strong and where it still struggles.

### 4.3 How the system works (architecture, labeled, plain language)
- One simple labeled diagram (can be an ASCII/markdown box diagram, doesn't
  need to be fancy) walking left to right:
  `Video of someone signing → "skeleton tracking" of hands/body → a pattern
  the computer can read → an AI model that reads that pattern → a Bangla
  sentence`
- Each arrow/box gets 2-4 plain-language sentences, e.g.:
  - **"Skeleton tracking"** — a widely-used computer vision tool
    (MediaPipe) watches the video and, frame by frame, marks the exact
    position of the signer's hands, fingers, arms, and upper body — like
    the motion-capture dots used in movies and video games, except no
    special suit or markers needed, it works directly on regular video.
  - **The "pattern"** — those position numbers, frame after frame, become a
    long sequence of numbers describing exactly how the body moved over
    time. This is what the AI model actually reads — not the pixels of the
    video itself.
  - **The AI model** — an "encoder" reads the whole movement sequence and
    builds an internal understanding of it; a "decoder" then writes out a
    Bangla sentence, one word-piece at a time, guided by that
    understanding. (Analogy: like a human interpreter who watches an
    entire sentence being signed before starting to speak, rather than
    guessing word by word as they go.)
  - **Bangla sentence out** — the final output, compared against what a
    human labeled that clip as, to measure how well the system did.
- Explicitly name (in plain terms) that no English and no fixed
  sign-dictionary is involved anywhere — the system learns the mapping
  directly from examples, which is both what makes this hard and what makes
  it a genuine research contribution.

### 4.4 What we did (the detailed walkthrough)

This is the longest section — split into clearly labeled sub-sections so a
reader can skim to what they care about.

**4.4.1 Data preprocessing — what, why, with examples**
- Simplify `docs/PIPELINE_WALKTHROUGH.md` §3 into plain language, keep its
  real examples (the three example sentences with frame counts/durations
  are great as-is — genuinely concrete).
- Embed the pipeline figures (`docs/assets/pipeline/*.png`) with plain
  captions explaining what each stage's picture shows.
- Cover, each in a short plain paragraph with a "why": frame sampling (why
  we don't need every single frame), skeleton/keypoint extraction (the
  MediaPipe step), normalization (why we adjust for where the person stands
  / how close the camera is, with a one-line analogy — "like scaling a
  photo so someone's height doesn't depend on how far the camera was"), and
  turning sentences into "sub-word pieces" for the model to output (why
  whole-word prediction doesn't work here — most words are too rare, cite
  the real 58%-appear-once statistic already documented).

**4.4.2 Building the model(s)**
- Explain, in plain terms, that this wasn't "build one model" but "try
  several different designs and see which works" — a research process, not
  a single build.
- List what was tried, each in 1-3 plain sentences (no need to explain the
  internal math of each — just what idea it tested and whether it helped):
  - A baseline "read the movement, write the sentence" model.
  - Adding a component that helps the model notice short bursts of motion.
  - Adding a second training signal that helps the model line up parts of
    the video with parts of the sentence as it learns (this is the one
    that ended up mattering most — call out plainly that this became the
    best-performing version).
  - Two alternative "reading" designs tested for comparison (one inspired
    by how graph/skeleton structure could be modeled directly; one using a
    newer, more efficient sequence-modeling idea) — both explained as
    genuine experiments, not failures to be embarrassed about (see 4.4.4).
  - A version that tries giving the model a head start by having it
    practice on the videos alone (without sentences) before the real
    training — also an honest experiment, see 4.4.4.
  - Currently in progress: giving the "writing" half of the model a head
    start by starting from an AI system that already knows how to write
    fluent Bangla, then teaching it to read the sign-language patterns.

**4.4.3 Training & running**
- Plain explanation of: what "training" means (showing the model thousands
  of labeled examples, adjusting it a little after each mistake, repeating
  many times); what a "split" is and why it matters (some clips are set
  aside and never shown during training, specifically so the final test is
  fair, not an open-book exam); the three different ways clips were split
  aside (random, by person, by whether the sentence itself was ever seen
  before) — explain simply why testing three different ways matters (it
  answers three different questions: does it work in general, does it work
  on a *new person's* signing style, does it work on a sentence it's
  *never encountered*).
- One or two plain sentences on real-world compute: what hardware was used,
  roughly how long training takes (a range, not false precision).

**4.4.4 Results — what the numbers mean, and how well it's doing**
- Explain the scoring method in plain terms *before* showing any numbers:
  "we compare the computer's sentence to the correct one and score how
  close they are, from 0 (nothing matches) toward 100 (perfect match) —
  the same kind of scoring used to grade machine translation systems in
  general, not something invented for this project."
- Show the headline comparison as a simple table: best system vs. simplest
  possible baselines vs. a "just copy the most common answer" baseline —
  making concrete *why* beating those baselines is meaningful (a naive
  system can look deceptively okay on this score without understanding
  anything at all — explain this plainly, it's a genuinely important,
  hard-won finding from this project, see `RESULTS_OVERVIEW.md`'s
  Table 4/4b framing).
- Show 3-4 real example translations pulled directly from
  `QUALITATIVE_EXAMPLES.md`: one where it got the sentence exactly right,
  one "typical" (partially right) example, and one clear failure — with a
  plain explanation of what went wrong in the failure case (e.g. the model
  got stuck repeating one word).
- **Honest assessment, in plain language, of what this proves and doesn't
  prove yet:**
  - Plain-language version of the grounding finding: we ran a test where we
    deliberately showed the model a mismatched video and sentence pair, and
    its accuracy dropped a lot — proof it's actually using the video, not
    just guessing likely-sounding Bangla sentences from habit.
  - Plain-language version of the "static pose, not motion" caveat: when we
    scrambled the order of the video frames, accuracy barely changed —
    meaning the system is currently reading *body poses* much more than it
    is reading *movement*, which is a real, named limitation, not a hidden
    one.
  - Plain-language summary of the experiments that **didn't** pan out (the
    pre-training head-start idea, and one of the two alternative "reading"
    designs) — framed exactly as the project's own internal logs frame
    them: a real hypothesis, a real test, a real (negative) result, and
    what that ruled out for future work. Emphasize: this is normal,
    expected research process, not a setback to downplay.

### 4.5 What's next
- The in-progress experiment (giving the "writing" half of the model a
  head start from an AI that already knows fluent Bangla) — plainly stated
  as not yet finished, with what question it's trying to answer.
- The next open question after that: picking the best-performing approach
  as the project's flagship method.
- The remaining interpretability work: trying to figure out *which part of
  the video* corresponds to *which word* after the fact (useful both for
  understanding the model and for a possible future avatar/sign-generation
  system) — plainly note this has so far come back inconclusive, and why
  that's still a useful thing to know.
- The eventual goal: publishing this dataset, benchmark, and findings as a
  research contribution, and continuing to close the gap on the
  static-pose/motion limitation named above.

### 4.6 (Optional) Glossary
- A short reference list at the end collecting every plain-language
  definition used inline throughout, alphabetized, for skimming/lookup —
  useful if the report is shared onward to someone who starts partway
  through rather than reading start to end.

### 4.7 (Optional) Appendix / "go deeper"
- One short paragraph pointing at `README.md` and `docs/logs/` for anyone
  who wants the full technical detail, so the main report doesn't need to
  compromise on accessibility to also satisfy a technical reader.

## 5. Open questions for you to decide during review

1. **Length target?** I'd suggest aiming for something a non-expert can
   read in about 15-20 minutes (roughly 2,000-3,000 words plus the
   figures) — long enough to be complete, short enough that §4.4's detail
   doesn't overwhelm the plain-language goal. Tell me if you want it
   shorter/longer, or want a specific section to get more/less weight.
2. **Format?** Markdown (matches everything else in `docs/`, renders
   cleanly on GitHub) is my default assumption — say if you want a
   different deliverable format instead/also (e.g. a slide-style doc, a
   PDF-oriented layout).
3. **Where does this live?** Proposing `docs/report/PROJECT_REPORT.md` —
   say if you want a different location, or want it to also become (say) a
   published Artifact page rather than (or in addition to) a repo doc.
4. **Should the "what's next" section name a rough timeline?** I'd lean no
   (this is a research project with genuinely uncertain timelines, and a
   wrong date undermines trust more than an honest "in progress, no fixed
   date" does) — but this is your call.
5. **Any audience-specific framing I should add or remove?** E.g. if this
   is going to a funder, a school, a specific stakeholder — the plan above
   is written for a generic "smart non-expert," but a known specific
   audience might shift emphasis (e.g. more on real-world accessibility
   impact vs. more on the research-contribution angle).
