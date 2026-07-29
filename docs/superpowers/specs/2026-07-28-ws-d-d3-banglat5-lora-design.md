# WS-D / D3 — Keypoint Encoder → BanglaT5 Decoder (LoRA): Design

> Status: approved in brainstorming session 2026-07-28. Follow-on to
> `docs/plan/2026-07-13-bornil-implementation-plan.md` §8.3 (D3). Feeds the
> WS-D novel-method comparison: D1 closed negative (FAIL-006/FAIL-007), D2
> blocked on compute (FAIL-008, open) — this is the last WS-D candidate.

## Goal

Swap this repo's existing keypoint encoder into BanglaT5's pretrained
decoder (LoRA-adapted) and test whether a strong pretrained Bangla language
prior beats the current best model (TConv+CTC, `b0a590ea3317`, chrF++
10.913) on `random-v1` — while explicitly guarding against the master
plan's own named risk for this architecture: a fluent-but-ungrounded
decoder.

## Context

Verified during brainstorming (2026-07-28):
- `bitsandbytes==0.50.0` and `peft==0.20.0` both resolve to real prebuilt
  wheels on this Windows machine (confirmed via `pip download`) — unlike
  D2's `mamba-ssm`, there is no build-toolchain risk here.
- `csebuetnlp/banglat5` exists on the Hub (T5 architecture,
  `AutoModelForSeq2SeqLM`-compatible, pretrained exclusively on Bangla —
  BUET NLP's Bangla2B+ corpus, arXiv:2205.11081), T5-base scale. Chosen over
  `facebook/mbart-large-50` and `google/mt5-base` (both real, both
  confirmed on the Hub) because a monolingual Bangla model has a stronger
  domain-specific language prior than a 50-101-language multilingual model,
  and is smaller/safer for an 8GB card.
- `transformers==5.13.1` is already installed in this repo's venv.

**Decided during design, not asked of the user (pure risk-reduction
calls):**
- **No 4-bit quantization / no `bitsandbytes` usage in this first version.**
  BanglaT5 is small enough that plain LoRA in fp16/bf16 fits comfortably in
  8GB without quantization. `bitsandbytes` carries real Windows-CUDA
  execution risk beyond just its wheel installing, for no VRAM benefit at
  this model size. Skip it — this walks back the "needs peft/bitsandbytes"
  framing from before this design was written.
- **No CTC head in this first version.** The master plan's own suggested
  fix for decoder hallucination ("higher visual conditioning weight,
  CTC-aux grounding") is a documented fallback if the grounding gate below
  fails, not built preemptively (YAGNI).

## Architecture

**Encoder side: unchanged from this repo's best model.** Reuse
`SignTransformer`'s encoder half — TConv2k5 + 4-layer Transformer, same
config as `b0a590ea3317` (`temporal_conv_layers: 2`,
`num_encoder_layers: 4`, `d_model: 256`) — freshly initialized (not D1's
failed MKM checkpoint; that pretraining objective is a separate, closed
negative result and has no bearing on this encoder's init). Its
`encode(src, src_key_padding_mask) -> (B,S,256)` output is projected
through one new `nn.Linear(256, banglat5_d_model)` into BanglaT5's hidden
size — `banglat5_d_model` is read from the loaded pretrained model's own
`config.d_model` at construction time, **never hardcoded** (matches this
repo's existing convention, e.g. `feature_dim(upper_body)` — never a
literal).

**Decoder side: BanglaT5's own pretrained `T5Stack` decoder**, called
directly with the projected encoder output as `encoder_hidden_states` and
the source padding mask as `encoder_attention_mask`. HF's T5 decoder
implementation accepts arbitrary `encoder_hidden_states` of the right shape
— it does not require them to come from its own paired T5 encoder. This
mirrors the "custom encoder → shared decoder" pattern every other model in
this repo already follows (`SignTransformer`/`SignBiLSTM`/`SignSTGCN`/
`SignMamba`, all encoder-swaps over a shared decoder), except the shared
decoder here is pretrained instead of trained from scratch.

**LoRA** (via `peft`) targets the decoder's attention projections (`q`,
`k`, `v`, `o`) and feed-forward layers. **The token embedding table and
`lm_head` stay fully frozen** at BanglaT5's pretrained values. This is why
target-side tokenization must switch to **BanglaT5's own tokenizer/vocab**
(`AutoTokenizer.from_pretrained("csebuetnlp/banglat5")`), not this repo's
custom `bsl_sp.model` (4000-subword vocab, via `bornil.data.tokenizer.Tokenizer`)
— a frozen `lm_head` has a fixed output vocab size baked in, so a different
vocab isn't usable once frozen. **This has zero effect on cross-model
comparability**: `bornil.benchmark.metrics.corpus_metrics` scores decoded
text strings (NFC-normalized), not token ids — the locked metric protocol
is tokenizer-agnostic, and every number in `RESULTS_OVERVIEW.md` is already
computed this way regardless of which model produced the string.

## Components

New `bornil/models/banglat5_slt.py`:
- `class KeypointToBanglaT5(nn.Module)` — owns: a `SignTransformer`-style
  encoder (or the encoder half factored out — implementer's call at plan
  time, matching how `SignSTGCN`/`SignMamba` each own their full encoder
  stack rather than importing `SignTransformer` wholesale), the projection
  `Linear`, the loaded `AutoModelForSeq2SeqLM` (LoRA-wrapped via `peft`),
  and `encode()`/`forward()`/`generate()` methods. Does **not** need to
  match the exact `SignTransformer`/`SignSTGCN` six-method contract
  verbatim (unlike D2, which deliberately mirrored that contract to plug
  into the shared `build_model`/`train_sample.py` pipeline) — a pretrained
  HF model's `generate()` has a different, incompatible signature from this
  repo's `greedy_decode`/`beam_decode`, so **D3 gets its own training/eval
  script** (`scripts/train_banglat5.py`) rather than going through
  `build_model` + `train_sample.py`. Reuses `SignDataset`,
  `LengthBucketBatchSampler`, `resolve_split`, and the packed-feature
  loading pattern from `train_sample.py` — only the model construction,
  tokenizer, and train/eval loop are new, not the whole data pipeline.

## Data flow

`SignDataset`/`LengthBucketBatchSampler`/frozen splits unchanged
(keypoint-feature side only — same `random-v1` train/val/test clips as
every other model). Target encoding uses BanglaT5's own tokenizer.
Training follows HF's standard T5 seq2seq convention: `decoder_start_token_id`
(from `banglat5.config`) rather than this repo's custom BOS id, and
`labels` with `-100` at padded positions (HF's `T5ForConditionalGeneration`
loss convention) rather than the `CrossEntropyLoss(ignore_index=pad_id)`
pattern used elsewhere in this repo. **This is the one place D3's plumbing
genuinely diverges from every other model here — call it out explicitly in
the plan rather than trying to force-fit it into the shared convention.**
Evaluation still produces `(hypothesis, reference)` string pairs scored via
the same locked `corpus_metrics`, so the *output* of D3 slots into
`RESULTS_OVERVIEW.md`/`results_significance.py`/`prediction_diagnostics.py`
exactly like every other model's `predictions.csv`.

## Validation / success criteria

1. **Core claim:** chrF++ on `random-v1` vs vanilla (`c19d0ea08824`) and vs
   TConv+CTC (`b0a590ea3317`), two-sided paired bootstrap n=1000 — same
   protocol as every Table 2 entry.
2. **Mandatory grounding gate — reusing `scripts/input_ablation.py`'s
   mismatch control.** Derange the video↔text pairing (same corruption mode
   already implemented and used to produce this project's strongest
   existing grounding evidence: −4.71 chrF++ for TConv+CTC), holding
   real clips, real feature statistics, and real lengths constant. This
   directly targets the master plan's named D3 risk ("LLM decoder
   hallucinates fluent-but-wrong Bangla"): if mismatch costs little for
   this model, that is direct evidence of exactly that failure mode, and
   must be reported even if — especially if — chrF++ looks good.
3. Retrieval floor + diversity/length diagnostics (`prediction_diagnostics.py`),
   same as every other model row.
4. Every run gets a normal `EXPERIMENTS.md` row.

## Pre-registered prediction

Recorded before any code exists, matching this repo's convention
(FAIL-004/005/006/007/008 all pre-registered before running).

**Prediction.** Given D1's finding that a strong language-side improvement
path (a better-grounded encoder via SSL) did not reliably help this
benchmark, and the master plan's own explicit warning about this exact
architecture's hallucination risk, the predicted outcome is: **chrF++ may
improve (BanglaT5's fluency prior lifting surface-level text-overlap
metrics) while the mismatch-ablation grounding cost shrinks relative to
TConv+CTC's −4.71** — i.e., a possible "metrics look better, grounding gets
worse" outcome. This is exactly the failure mode the grounding gate (item 2
above) exists to catch, so it must be checked and reported regardless of
how the core chrF++ claim lands.

## Testing (TDD)

- Projection-layer shape test (`(B,S,256) -> (B,S,banglat5_d_model)`).
- A test that HF's `T5Stack.forward(encoder_hidden_states=..., encoder_attention_mask=...)`
  accepts our projected memory (not BanglaT5's own encoder output) and
  produces correctly-shaped logits, on a tiny synthetic case — this is the
  one integration point with the most risk of a silent shape/convention
  mismatch, so it gets a dedicated test before any real training.
- LoRA-application test: after wrapping with `peft`, confirm only the
  targeted attention/FFN modules have trainable parameters, and that the
  embedding table and `lm_head` weights have `requires_grad=False`.
- Early "can this model even load" check —
  `AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5")` and
  `AutoTokenizer.from_pretrained("csebuetnlp/banglat5")` both succeed —
  analogous to D2's mamba-ssm buildability check, since this is the one new
  external-download dependency in the whole plan (first run needs internet
  access to fetch the model from the Hub; already confirmed available this
  session via successful `pip download` calls).

## Compute budget

LoRA training is cheap relative to full fine-tuning (most of the decoder's
~T5-base-scale parameters are frozen), and BanglaT5 itself is a small
pretrained model — expect roughly TConv+CTC-scale training time or
somewhat slower per epoch (a larger decoder than this repo's from-scratch
4-layer one, and a forward/backward pass through a frozen-but-still-active
`lm_head` over BanglaT5's own — larger than this repo's 4000-token —
vocabulary). Nothing resembling D2's Python-loop pathology is expected, but
**still measured via a mandatory smoke-test timing step before the full
run**, matching D1/D2 precedent exactly — never assumed, regardless of how
low-risk it looks on paper (D2 also looked low-risk on paper for a
different reason before its gate caught the real number).
