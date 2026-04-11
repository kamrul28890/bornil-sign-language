# Bornil-BdSL: Research Directions, Models, and Benchmarks
## A Comprehensive Research Agenda for Bangladeshi Sign Language Understanding

---

## 1. Dataset Overview

**Bornil** is a large-scale continuous sign language video corpus for Bangladeshi Sign Language (BdSL). Based on our analysis of the metadata:

| Property | Value |
|---|---|
| Total video samples | 17,988 |
| Total duration | ~66.8 hours |
| Average clip duration | 13.4 seconds (median: 11.0s) |
| Duration range | 1.7s – 59.9s |
| Average frame count | 501 frames/clip |
| Number of data chunks | 36 |
| Annotation type | Bengali sentence-level transcription |
| Recording setup | Front-view, indoor, near camera distance |
| Background | Real-world (non-green screen) |
| Estimated total storage | ~200 GB |

The dataset contains **continuous** sign language sequences (not isolated signs), annotated with naturalistic Bengali sentences. Text lengths range from very short utterances (4 chars) to long, complex sentences (550 chars), covering a rich vocabulary including news language, everyday conversation, literary text, and formal register.

---

## 2. Research Problem Landscape

### 2.1 Core Task Hierarchy

```
Continuous Sign Language Recognition (CSLR)
    └─► Sign Language Translation (SLT)         [signs → Bengali text]
            └─► Sign Language Production (SLP)  [Bengali text → signs]
                    └─► Sign Language Understanding (SLU)  [semantic tasks]
```

Each level is progressively harder and more impactful. Bornil enables all four.

---

## 3. Research Direction 1 — Continuous Sign Language Recognition (CSLR)

**Problem:** Given a raw sign language video, predict the sequence of glosses or directly the Bengali sentence without any temporal segmentation.

### Why This is Hard on BdSL
- BdSL is an **under-resourced language** with almost no prior neural baselines.
- Continuous signing involves **co-articulation**, where signs blend at boundaries.
- Bengali script is complex (conjuncts, matras), making output decoding non-trivial.
- Real-world, non-green-screen backgrounds (as in Bornil) add visual noise.

### Model Approaches

**Baseline: CNN + RNN CTC**
- Extract per-frame features via ResNet-18/50 or EfficientNet-B4.
- Feed into a BiLSTM with CTC (Connectionist Temporal Classification) loss.
- Standard in PHOENIX-2014T literature; strong, interpretable baseline.

**Intermediate: Transformer-based CSLR**
- Replace RNN with a Transformer encoder over frame-level features.
- Add a sign-level CTC loss intermediate supervision.
- Reference: CorrNet (Wang et al., CVPR 2023), TwoStream-SLR.

**Advanced: Video Foundation Models**
- Use VideoMAE-v2 or InternVideo2 as a frozen/fine-tuned video backbone.
- Task-specific head: linear projection → CTC over Bengali character vocabulary.
- Leverage self-supervised pre-training on large video corpora.

**State-of-the-Art Target: Mamba-based Sequence Modeling**
- Replace Transformer with State Space Models (SSM/Mamba) for long-range temporal dependency without quadratic attention cost.
- Especially useful for the long clips in Bornil (up to 60s, ~1800 frames at 30fps).

### Benchmarks to Beat
| Benchmark | Dataset | WER Target |
|---|---|---|
| PHOENIX-2014 | German SL | ~18% WER (CorrNet) |
| PHOENIX-2014T | German SL | ~19% WER |
| CSL-Daily | Chinese SL | ~25% WER |
| **Bornil-BdSL (proposed)** | Bengali SL | **TBD — you set the baseline** |

> **Key Contribution:** You are the first to establish WER benchmarks on BdSL continuous recognition. This alone is a strong baseline paper for ACL/EMNLP/ACM MM.

---

## 4. Research Direction 2 — Sign Language Translation (SLT)

**Problem:** End-to-end video-to-Bengali-text translation, going beyond recognition to produce fluent, grammatically correct Bengali sentences.

### Why This Matters
- BdSL grammar differs significantly from Bengali grammar (topic-comment structure, omitted copulas).
- SLT must bridge two language systems, not just recognize signs.
- First SLT system for BdSL = guaranteed high-impact publication.

### Model Approaches

**Architecture 1: Two-Stage Pipeline**
1. CSLR model produces a gloss sequence.
2. Gloss-to-Bengali translation via a fine-tuned mT5/BanglaT5 model.
- Advantage: Modular, easier to debug.
- Challenge: Error propagation from Stage 1.

**Architecture 2: End-to-End Sign2Text**
- Video encoder (VideoMAE / SlowFast) → cross-attention → Bengali decoder (mBART / BanglaT5).
- Train with Bengali sentence supervision directly.
- Reference: SLRT (Sign Language Recognition and Translation), GFSLT-VLP (ICCV 2023).

**Architecture 3: VLM-Powered SLT (Novel)**
- Use a frozen Video-Language Model (e.g., Video-LLaMA, VideoChat2, LLaVA-NeXT-Video) as the visual encoder.
- Attach a Bengali-specialized LLM decoder (e.g., BanglaBERT-based generative model or LLaMA-3 fine-tuned on Bengali corpus).
- This is a **novel architecture** not yet explored for any South Asian sign language.
- Train with LoRA/QLoRA to make it computationally feasible.

**Architecture 4: Cross-Lingual Transfer**
- Pre-train SLT model on a resource-rich sign language (ASL, BSL, or German SL).
- Fine-tune on Bornil using adapter layers.
- Tests whether visual sign representations transfer across sign languages.

### Evaluation Metrics
- **BLEU-1/2/3/4** (standard in SLT literature)
- **ROUGE-L**
- **chrF** (character-level F-score, important for Bengali morphology)
- **BERTScore** with BanglaBERT
- **Human evaluation** (fluency + adequacy ratings)

---

## 5. Research Direction 3 — Sign Language Production (SLP)

**Problem:** Given a Bengali sentence, generate a realistic sign language video or skeleton animation.

### Why This is Transformative
- Enables automatic generation of BdSL content for news, education, government services.
- Directly addresses accessibility for ~2 million Deaf people in Bangladesh.

### Model Approaches

**Skeleton-based SLP (Feasible)**
- Text → Bengali gloss sequence (via NLP) → pose sequence (skeleton keypoints).
- Use MediaPipe/OpenPose to extract skeleton from Bornil videos as supervision.
- Train a Transformer seq2seq: Bengali tokens → 3D pose coordinates per frame.
- Reference: SignGAN, Progressive Transformers (ECCV 2020).

**Video-based SLP (Ambitious)**
- Extend skeleton-based SLP with a video synthesis module.
- Use a video diffusion model (e.g., AnimateDiff, Stable Video Diffusion) conditioned on skeleton sequences.
- Very recent, very publishable — no work exists for BdSL.

### Evaluation Metrics
- **Back-Translation WER**: translate generated video back to text, measure WER.
- **FID / FVD** (Fréchet Video Distance) for realism.
- **MPJPE** (Mean Per-Joint Position Error) for skeleton accuracy.
- **Human Study**: native BdSL signers rate naturalness (1–5 Likert scale).

---

## 6. Research Direction 4 — Multimodal & Vision-Language Research

### 6.1 Sign Language Understanding with Large VLMs

**Idea:** Prompt large VLMs (GPT-4o, Gemini 1.5 Pro, LLaVA-NeXT) zero-shot/few-shot on BdSL videos and measure how much "sign language understanding" is already latent in VLMs pre-trained purely on spoken-language video data.

**Expected finding:** VLMs will fail significantly → motivation for BdSL-specific fine-tuning. This makes for a compelling analysis paper at findings of ACL / NAACL.

### 6.2 Cross-Modal Retrieval

**Problem:** Given a Bengali query sentence, retrieve the most relevant sign video from Bornil, and vice versa.

- Train CLIP-style dual encoder: Video encoder ↔ Bengali text encoder.
- Use BanglaBERT as text backbone, VideoMAE as video backbone.
- Contrastive learning (InfoNCE loss) aligns sign video and Bengali text embedding spaces.
- **Metric:** R@1, R@5, R@10 (Recall@K) on a held-out test split.

This enables a **sign language search engine** — direct real-world application.

### 6.3 Bengali NLP-Enriched Sign Understanding

- **Semantic role analysis**: Do signers mark agent/patient through spatial grammar? Analyze using dependency parsing of annotated Bengali sentences.
- **Sentiment alignment**: Does sign production change for positive vs negative sentences? (Use BanglaSentiment on text labels, analyze motion features in video.)
- **Discourse coherence**: For long sequences, model how signs connect across clauses.

---

## 7. Research Direction 5 — Data-Centric & Low-Resource AI

### 7.1 Semi-Supervised CSLR

Bornil has ~17K labeled samples. But you have 200GB of video — likely far more unlabeled content. Exploit this with:
- **Pseudo-labeling**: Train initial CSLR model, generate pseudo-labels for unlabeled clips, retrain.
- **Mean Teacher** or **FixMatch** adapted for video.

### 7.2 Self-Supervised Pre-training on Sign Video

Train a video encoder purely on Bornil videos (no labels) using:
- **VideoMAE** masking: randomly mask 90% of video patches, reconstruct.
- **DINO-v2** style self-distillation on frame crops.
- Use the pre-trained encoder to initialize all downstream tasks.

This tests whether sign-domain pre-training improves over ImageNet/Kinetics initialization.

### 7.3 Data Augmentation for Sign Language

Signs have specific structure — naive augmentation (random crop, horizontal flip) can change meaning. Propose **semantics-preserving augmentations**:
- **Speed perturbation** (temporal stretching/compression of signing speed).
- **Skeleton-guided spatial jitter** (perturb hand positions within physiologically valid range).
- **Background replacement** using segmentation mask + new backgrounds.

### 7.4 Active Learning for BdSL Annotation Expansion

Design an active learning loop to expand Bornil efficiently:
- Train initial model → select maximally uncertain/informative unlabeled clips → send to Deaf annotators → retrain.
- Reduces annotation cost to expand to 500K+ samples.

---

## 8. Research Direction 6 — Real-Time & Deployment Systems

### 8.1 Lightweight CSLR for Edge Deployment

- Distill a large CSLR model into a MobileNet + small RNN model running at 30fps on CPU/mobile GPU.
- Target: Smartphone app for BdSL-to-Bengali real-time translation.
- Benchmark: Latency (ms/frame), model size (MB), WER degradation vs teacher.

### 8.2 Streaming CSLR

- Real-world use requires processing video streams without waiting for sentence boundaries.
- Train a **CIF (Continuous Integrate-and-Fire) based** model for online decoding.
- Reference: CIF mechanism (Dong et al., ICASSP 2020).

---

## 9. Novel Proposed Paper: BdSL-VLP

**Title:** *BdSL-VLP: Vision-Language Pre-training for Bangladeshi Sign Language Understanding and Translation*

**Core Idea:** Pre-train a joint vision-language model on Bornil using three objectives:
1. **Video-Text Contrastive (VTC)**: Align sign video clips with Bengali sentence embeddings.
2. **Video-Text Matching (VTM)**: Binary classification — does this video match this sentence?
3. **Sign Language Translation (SLT)**: Auto-regressive Bengali sentence generation from video.

This is directly inspired by BLIP-2 / InstructBLIP but adapted for:
- Sign language video (not natural scene images).
- Bengali output (not English).
- Continuous signing temporal structure.

**Why it's publishable at top venues (ACL/EMNLP/CVPR/ACM MM):**
- First VLP model for any South Asian sign language.
- Introduces a new benchmark (Bornil) alongside model.
- Directly serves a real accessibility need for a large population.
- Technically novel combination of sign video modeling + Bengali NLP.

---

## 10. Standard Benchmarks & Comparison Datasets

| Dataset | Language | Size | Task | Venue |
|---|---|---|---|---|
| PHOENIX-2014T | German SL | 8,257 clips | CSLR + SLT | CVPR 2018 |
| CSL-Daily | Chinese SL | 20,654 clips | CSLR + SLT | CVPR 2021 |
| How2Sign | ASL | 35K clips | SLT | CVPR 2021 |
| BOBSL | BSL | 1,467 hours | CSLR | ECCV 2021 |
| OpenASL | ASL | 288 hours | SLT | NeurIPS 2022 |
| SP-10 | Multiple | 10K clips | Isolated SLR | — |
| **Bornil-BdSL** | **BdSL** | **17,988 clips / 66.8h** | **CSLR + SLT** | **Proposed** |

> **Positioning:** Bornil is comparable in scale to CSL-Daily and PHOENIX-2014T but uniquely targets a South Asian language with near-zero existing resources.

---

## 11. Computational Requirements

### Minimum (Baseline CSLR)
- **GPU:** 2× NVIDIA RTX 3090 (24GB each) or 1× A100 40GB
- **RAM:** 64GB system RAM
- **Storage:** 2TB SSD for preprocessed features
- **Training time:** ~3–7 days for a CNN-BiLSTM-CTC model

### Recommended (Transformer / VideoMAE fine-tuning)
- **GPU:** 4× A100 80GB (or equivalent H100)
- **RAM:** 256GB
- **Storage:** 4TB NVMe SSD
- **Training time:** 5–14 days depending on architecture

### Ambitious (VLP pre-training / Video Diffusion SLP)
- **GPU:** 8× A100 80GB or 4× H100 SXM
- **RAM:** 512GB
- **Distributed training:** DDP (DistributedDataParallel) with DeepSpeed ZeRO-3
- **Training time:** 2–4 weeks

### Practical Strategy
1. **Pre-extract frame features** offline (VideoMAE / ResNet) → store as `.npy`. Reduces I/O bottleneck drastically.
2. Use **gradient checkpointing** and **mixed precision (bf16)** for memory efficiency.
3. Use **LoRA/QLoRA** when fine-tuning large VLMs to reduce GPU requirement.
4. Consider **cloud compute** (Lambda Labs, RunPod, or Google TPU Research Cloud free allocation for academic labs).

---

## 12. Recommended Models (Ranked by Priority)

| Priority | Model | Task | Why |
|---|---|---|---|
| 🥇 | CorrNet / TwoStream-SLR | CSLR | SOTA, reproducible, strong baseline |
| 🥇 | VideoMAE-v2 + CTC | CSLR | State-of-the-art video backbone |
| 🥈 | GFSLT-VLP | SLT | End-to-end, ICCV 2023 SOTA |
| 🥈 | mT5 / BanglaT5 | Gloss→Bengali | Bengali-specific seq2seq |
| 🥉 | LLaVA-NeXT-Video + LoRA | VLM SLT | Novel, high-impact if it works |
| 🥉 | MediaPipe + Progressive Transformer | SLP (skeleton) | Enables production track |
| ⭐ | CLIP-style dual encoder | Retrieval | Practical application, fast to train |

---

## 13. Suggested Publication Roadmap

```
Year 1 (Foundations)
├── Paper 1: Bornil Dataset Paper + Baseline CSLR (WER benchmarks)
│   Target: ACM MM 2025 / EMNLP 2025 Findings
│
├── Paper 2: BdSL Sign Language Translation (SLT, BLEU scores)
│   Target: NAACL 2026 / COLING 2026
│
Year 2 (Advanced Methods)
├── Paper 3: BdSL-VLP (Vision-Language Pre-training)
│   Target: CVPR 2026 / ACL 2026
│
├── Paper 4: Cross-lingual transfer for under-resourced SL
│   Target: EMNLP 2026
│
Year 3 (Applications)
└── Paper 5: Real-time mobile BdSL recognition system
    Target: ACL Demo Track / INTERSPEECH 2027
```

---

## 14. Open Research Questions

1. **Does BdSL share visual phonetics with ASL/BSL?** — If yes, cross-lingual transfer works. If no, BdSL-specific architectures are necessary.
2. **Can phoneme-level Bengali models help?** — Bengali phoneme inventory might partially overlap with BdSL handshape/location/movement phonology.
3. **Is spoken Bengali prosody reflected in signing speed?** — Analyze correlation between text punctuation/clause structure and signing rhythm.
4. **How does diglossia affect BdSL?** — Bengali has Shadhubhasha vs Cholithabhasha registers. Does the signer's register affect signing style?
5. **Signer identity vs linguistic content** — Can a model separate signing style (signer ID) from linguistic content? Enables signer-independent models.

---

## 15. Ethical Considerations

- **Deaf community involvement:** All annotation and evaluation should involve native BdSL signers, not only hearing researchers.
- **Informed consent:** Ensure all signers in Bornil consented to academic use.
- **Dialect awareness:** BdSL may vary by region; document signer demographics.
- **Not a replacement:** NLP tools should augment, not replace, human interpreters.
- **Bias audit:** Ensure gender, age, regional balance in the dataset signers.

---

## References (Key Papers to Read)

- Camgoz et al. (2020). *Sign Language Transformers.* CVPR.
- Zuo & Mak (2022). *C²SLR: Consistency-Enhanced Continuous Sign Language Recognition.* CVPR.
- Wang et al. (2023). *CorrNet: Sign Language Recognition.* CVPR.
- Zhou et al. (2023). *GFSLT-VLP: Gloss-Free Sign Language Translation.* ICCV.
- Wang et al. (2023). *VideoMAE V2.* CVPR.
- Chen et al. (2022). *A Simple Multi-Modality Transfer Learning Baseline for Sign Language Translation.* CVPR.
- Albanie et al. (2021). *BOBSL: BBC-Oxford British Sign Language Dataset.* ECCV.
- Duarte et al. (2021). *How2Sign: A Large-scale Multimodal Dataset for Continuous ASL.* CVPR.
- Shi et al. (2022). *OpenASL: Open-vocabulary Sign Language Recognition.* NeurIPS.

---

*Prepared for the Bornil-BdSL research team. All statistics computed from `video_data.csv` (17,988 annotated video-text pairs, ~66.8 hours of continuous BdSL signing).*