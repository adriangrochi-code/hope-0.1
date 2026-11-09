# Byte-level (tokenizer-free) models: efficiency and generating several bytes per step (2023–2026)

Notes for Hope HN-1. The project reads bytes and builds patches by spaces or by entropy from an n-gram table. A local encoder feeds a global SSM+attention backbone, followed by a recurrent core with halting, specialist zones and a local decoder. It targets 25M–200M parameters (up to ~2B) on an Intel Arc A770 with PyTorch XPU.

Labels used below:
- **[DEMONSTRATED]**: an experimental result the paper measured.
- **[ESTIMATED]**: analytical FLOP or bandwidth figures, not wall-clock time.
- **[CLAIM]**: a figure from a blog or abstract that has no table I could verify.

---

## 1. BLT – Byte Latent Transformer (Pagnoni et al., Meta FAIR, Dec 2024)

### Takeaway
BLT is the reference design for entropy patching, and it matches Llama 3 at 8B parameters under equal training FLOPs. Its efficiency claims are all in FLOPs, and it reports no wall-clock time. Generation stays sequential byte by byte in the local decoder, and Meta's own follow-up (Fast BLT, 2026) names that as the bottleneck.

### Cited Findings
- **Patching method.** A separately trained byte-level transformer of 100M parameters (14 layers, d=512, 512-byte sliding window) computes the entropy of the next byte. — [Pagnoni et al. 2024, arXiv:2412.09871](https://arxiv.org/html/2412.09871v1)
  - The global rule cuts a patch when H(x_t) > θ_g. The approximate monotonic rule cuts when H(x_t) − H(x_{t−1}) > θ_r. The threshold is tuned to reach a target average patch size.
  - Ablations on the entropy model covered 1M–100M parameters and 64–512-byte context, with diminishing returns above ~50M. [same]
- **Patch sizes.** The scaling studies used average patches of 6 and 8 bytes. In the 8B training mix, BLT-Entropy averaged 4.5 bytes/patch, BLT-Space 6.1, and Llama 3 tokens 4.4. — [same]
- **Inference threshold.** For the 8B BLT-Entropy model, the authors lowered the inference threshold from 0.6 to 0.1. This creates more patches and more global steps, and it improved task scores. [DEMONSTRATED] — [same]
  - This trades speed for quality at inference time with no retraining.
- **Local encoder and decoder.**
  - Hashed n-gram embeddings cover n = 3..8 with 500k hashes. Gains flatten at about 300k. [same]
  - Ablations favored 1 encoder layer and 9 decoder layers, rising to 3 encoder layers at patch size 8.
  - Cross-attention in the decoder was the most useful. Encoder cross-attention helped only marginally.
  - The exact 8B dimensions are in Appendix Table 10, which I did not read.
- **Inference FLOPs.**
  - The paper claims "up to 50% fewer FLOPs at inference," and "nearly 50%" with patch size 8. [ESTIMATED]
  - In the inference-FLOP-matched comparison (Table 2), the BLT model can be 1.2–1.3× larger at ps=6 and 1.6–1.7× larger at ps=8. The FLOP classes are 3.1e8 and 2.1e9 FLOPs/byte.
  - The crossover where BLT overtakes BPE comes at 50B/150B training bytes for the small class and 400B/1T for the large one. — [same]
- **8B quality vs Llama 3** (1T tokens; average over 7 tasks) [DEMONSTRATED] — [same]

  | Model | Training data | Average |
  |---|---|---|
  | Llama 3 | 1T tokens | 60.0 |
  | BLT-Space | 6T bytes | 58.0 |
  | BLT-Entropy | 4.5T bytes | 61.1 |

  - For code, BLT-Entropy scored HumanEval 35.4 vs 31.1 and MBPP 41.8 vs 40.2.
  - Space patching is somewhat worse than entropy patching.
- **Robustness and character-level tasks.**
  - CUTE: BLT 54.1 vs Llama 3 27.5.
  - Noisy HellaSwag: 64.3 vs 56.9. — [same]
- **Multilingual (FLORES-101 SPM-BLEU).**
  - Into English: BLT 14.0 vs Llama 3 12.1.
  - From English: 6.4 vs 5.9. — [same]
- **Limitations stated by the authors.**
  - The implementation "may not yet be at parity" in wall-clock time with tokenizer-based models.
  - The entropy model is not trained end to end.
  - Most ablations ran at ≤1B parameters. — [same]

### Inferences
- BLT's saving comes from running the global model once per patch. At batch size 1, though, the local decoder still costs one forward pass per byte. For a small model, where the local decoder takes a large share of the parameters, the per-byte cost is dominated by that decoder.
- The 100M entropy model would be bigger than Hope itself at 25M–200M. Replacing it with a byte n-gram table is consistent with BLT's ablation, which shows diminishing returns from model size and limited value of context beyond a short window. This has not been measured with tables.

### Gaps
- No wall-clock bytes/s figures for BLT in the original paper.
- I did not read the exact local encoder/decoder dimensions at 8B (Appendix Table 10).

---

## 2. MambaByte (Wang et al., 2024) and speculative decoding with a subword drafter

### Takeaway
A pure Mamba model over bytes, with no patches, is competitive with subword models at matched compute. Speculative decoding with a small subword Mamba drafter, verified at the byte level, makes it 2.6× faster than plain MambaByte, close to subword Mamba's 2.8×.

### Cited Findings
- **Model sizes.** — [Wang et al. 2024, arXiv:2401.13660](https://arxiv.org/html/2401.13660v3)
  - MambaByte-353M: 53 layers, d=1024.
  - MambaByte-972M: 48 layers, d=1792.
  - MambaByte-1.6B: used only for timing.
- **bpb at equal compute** (Table 2) [DEMONSTRATED] — [same]

  | Model | Training bytes | PG19 | Code |
  |---|---|---|---|
  | MambaByte-353M | 30B | 0.930 | 0.396 |
  | MegaByte-758M+262M | 80B | 1.000 | 0.411 |
  | Byte transformer-320M | | 1.057 | |

- **PG19 at larger scale.** MambaByte-972M (150B bytes) reached PPL 33.0, MegaByte-1.3B+350M (400B bytes) 36.4, and subword Mamba-1.03B (600B bytes) 33.9. [DEMONSTRATED] — [same]
- **Generation time for 8,192 bytes** (Table 7, A100) — [same]

  | Model | Time |
  |---|---|
  | MambaByte-972M | 29 s (58 s with sliding window) |
  | MegaByte-1.3B+218M | 93 s per the MegaByte paper; 265 s in the authors' own run |
  | Transformer-350M | 132 s |

  - The "2.6× vs MegaByte" in the text is ambiguous against these numbers. Mamba's advantage is a constant-size recurrent state, with no KV cache that grows.
- **Speculative decoding** (Table 8) [DEMONSTRATED] — [same]
  - Drafter: a subword Mamba-110M with greedy decoding that drafts 3 subwords per iteration.
  - Verifier: MambaByte-972M checks the drafted bytes in parallel. A byte is accepted if it falls within the verifier's top-3.
  - On failure, the verifier generates bytes itself up to a boundary byte such as a space.
  - Speedup: 2.6× over MambaByte, vs 2.8× for subword Mamba-1.03B.
  - Faithfulness: log-odds against greedy MambaByte was 0.89 for speculative MambaByte vs 0.10 for subword Mamba.
  - The top-3 acceptance is lossy: it does not exactly reproduce greedy decoding.

### Inferences
- Pattern for Hope: a cheap drafter that proposes several bytes, with one parallel verification pass. The local decoder and SSM can verify k bytes in one call.
- The drafter does not have to be a neural network. Hope's own byte n-gram table, or a static vocabulary of frequent words, could propose word completions. A word is long and predictable after its first bytes, exactly where entropy is low. This drafter idea is my proposal, untested.

### Gaps
- Only one drafter/verifier setup on one GPU (A100). No acceptance-rate figures broken down by domain.

---

## 3. SpaceByte (Slagle, 2024)

### Takeaway
SpaceByte runs global blocks only after "space-like" bytes. This simple rule roughly matches a SentencePiece subword transformer at 10^19 FLOPs and clearly beats MegaByte. It degrades on languages without spaces, and its variable patches make batched sampling harder.

### Cited Findings
- **Rule.** A byte is space-like if it is not a letter, digit or UTF-8 continuation byte. A global block runs after the first space-like byte in a run. — [Slagle 2024, arXiv:2404.14408](https://arxiv.org/html/2404.14408)
  - Average patch: ~6 bytes on PG-19 and arXiv, ~8 on GitHub.
- **bpb at 10^19 training FLOPs** [DEMONSTRATED] — [same]

  | Model | PG-19 | arXiv | GitHub |
  |---|---|---|---|
  | Subword transformer (SentencePiece) | 0.989 | 0.768 | 0.508 |
  | SpaceByte | 1.009 | 0.748 | 0.500 |
  | MegaByte | 1.083 | 0.822 | 0.570 |
  | Byte transformer | 1.138 | 0.909 | 0.655 |
  | SpaceByte, fixed patches | 1.112 | 0.804 | 0.552 |

  - Dynamic patching is worth a lot: compare the last row with SpaceByte.
- **Small scale.** The configurations are around 150M–250M parameters (e.g. 201M+50M, 196M FLOPs/byte on PG-19). The result therefore holds at the scale of Hope. [same]
- **Limitations.**
  - Chinese came out worse than subword in preliminary experiments.
  - Variable spacing between global blocks makes efficient batched sampling harder.
  - Inference is reported only in FLOPs/byte, with no generation experiments. — [same]

### Inferences
- At Hope's scale (25M–200M), SpaceByte is the most directly comparable evidence: space patching plus a local model reaches BPE-level bpb in English and code.
- For Spanish, which uses Latin script with spaces, the rule should work in principle. AU-Net, below, confirms gains in Romance languages with a space-based split.

### Gaps
- No direct Spanish results. No wall-clock measurements.

---

## 4. MegaByte (Yu et al., Meta, 2023)

### Takeaway
MegaByte uses fixed patches (P=8 for text). A large global model runs once per patch and a small local model decodes bytes sequentially within the patch. It beat byte-level transformers at equal compute, and a 1.5B model generated about 40% faster than a 350M transformer. Within-patch decoding is not parallel, and loss rises toward the end of each patch.

### Cited Findings
- **Sizes.** — [Yu et al. 2023, arXiv:2305.07185](https://ar5iv.labs.arxiv.org/html/2305.07185)
  - Text uses P=8.
  - The controlled comparison used a 758M global / 262M local model.
  - The speed comparison used a 1.3B global / 218M local model.
- **bpb at controlled compute** (Table 2) [DEMONSTRATED] — [same]

  | Model | PG-19 | arXiv | Code |
  |---|---|---|---|
  | MegaByte | 1.000 | 0.678 | 0.411 |
  | Byte transformer | 1.057 | 0.816 | 0.575 |

  - Both baselines here are byte-level, not BPE.
- **Speed.** MegaByte-1.5B took 93 s to generate 8,192 bytes, vs 132 s for a 350M transformer, with bpb 0.991 vs 1.064. [DEMONSTRATED] — [same]
  - The authors estimate a theoretical speedup of up to ~P× when the global model is much deeper than the local one.
- **Limitations.**
  - Per-byte loss rises toward the end of each patch, because later bytes lean on the weaker local model.
  - "Strided inference" mitigates this but doubles the inference cost. — [same]

### Inferences
- Fixed patches are worse than dynamic ones (SpaceByte table). The MegaByte design is useful mainly as the template for "global once per patch, local per byte."

---

## 5. 2025–2026 follow-ups: H-Net, AU-Net, EvaByte, Bolmo

### Takeaway
- H-Net (learned chunking) and AU-Net (space-based pooling over several stages) match or beat BPE at equal compute from ~1B parameters.
- AU-Net is the only one of these that reports wall-clock throughput, and it is comparable to BPE.
- EvaByte demonstrates multi-byte prediction with self-speculation at 6.5B.
- Bolmo shows you can "byteify" existing subword models.

### Cited Findings

**H-Net (Hwang, Wang, Gu, Jul 2025)** — [arXiv:2507.07955](https://arxiv.org/html/2507.07955v2)
- **Dynamic chunking.**
  - The router computes a boundary from the cosine similarity of adjacent encoder states, with p ≥ 0.5 marking a boundary.
  - An EMA "smoothing" step weighted by confidence keeps gradients flowing through the discrete boundary decisions.
  - The upsampler uses a straight-through estimator.
  - A ratio loss (α=0.03) pushes toward a target compression.
- **Architecture.** Encoders and decoders use about 4 Mamba-2 layers each, and the main network is a Transformer.
  - Mamba in the encoder/decoder beats a Transformer there, even when the input is BPE tokens.
- **Compression.** A target of 6 gives 4.5–5 bytes/chunk in practice, similar to BPE.
- **Quality at matched FLOPs** (averages) [DEMONSTRATED]

  | Scale | 1-stage | 2-stage | BPE transformer |
  |---|---|---|---|
  | GPT-3 Large | 53.6 | 55.5 | 53.3 |
  | XL | — | 58.2 | 55.5 |

  - At XL, 2-stage overtakes BPE after ~30B bytes and 1-stage after ~100B. This crossover is relevant for small-data training budgets.
- **Code.** bpb 0.3161 (2-stage) vs 0.3376 (BPE).
- **Chinese.** XWinograd-zh 0.663 vs 0.599.
- **DNA.** ~3.6× more data-efficient.
- **Generation.** The authors only remark that the model "decides compute per token," with an analogy to speculative decoding. I found no wall-clock numbers in the part I read (first ~100k characters).

**AU-Net (Videau et al., Meta FAIR, Jun 2025; NeurIPS 2025)** — [arXiv:2506.14761](https://arxiv.org/html/2506.14761v1)
- **Stages.** Stage 1 runs on bytes. Stage 2 pools at word boundaries via regex, stage 3 every 2 words, stage 4 every 4 words.
  - Generation: the byte stage runs every step, and deeper stages run only at their boundaries. The last vector of each stage is cached.
- **Throughput** (H100, bytes/s/GPU) [DEMONSTRATED]

  | Scale | BPE | AU-Net 2 | AU-Net 3 | AU-Net 4 | Byte transformer | Mamba |
  |---|---|---|---|---|---|---|
  | 1B | 210k | 225k | 180k | 155k | 47k | 32k |
  | 8B | 43k | 41k | — | — | — | — |

  - Hierarchical models recover BPE-level throughput. Flat byte models are 4–6× slower.
- **Quality.**
  - 1B with 370B tokens: AU-Net 4 vs BPE scored HellaSwag 73.7 vs 70.2 and MMLU 31.7 vs 27.0.
  - 8B: AU-Net 2 vs BPE scored HellaSwag 79.1 vs 77.2 and MMLU 51.1 vs 49.6. [DEMONSTRATED]
- **Multilingual.**
  - FLORES-200 into English: 24.6 vs 20.9.
  - Spanish: FLORES into English 29.3 vs 28.2. MMLU-es 41.4 vs 37.6.
  - Romance languages gain about +4 MMLU. Chinese drops (28.0 vs 33.0). [DEMONSTRATED]
- **Limitations.** The space-based split does not work for languages without spaces. With ≥3 stages, FSDP struggles to overlap computation and communication.
- **Multi-word prediction.** Deep stages implicitly predict several words ahead, with no auxiliary loss.

**EvaByte (HKU NLP + SambaNova, Jan 2025)** — [blog](https://hkunlp.github.io/blog/2025/evabyte), [HF model card](https://huggingface.co/evabyte/EvaByte-SFT)
- **Model.** 6.5B parameters, flat (no patches). It uses EVA linearized attention, trained on 1.5T bytes with 32K context.
- **Multi-byte prediction.**
  - 8 output heads and a vocabulary of 320. The small vocabulary means "almost no training overhead."
  - Decoding is self-speculative with Medusa-style tree attention.
  - Greedy multi-byte output "usually" matches `generate()`.
- **Speed** [CLAIM; chart only, no table]
  - 5–10× faster than vanilla byte architectures, and up to 2× faster than tokenizer LMs.
  - Setup: 512 bytes, batch size 1, H800.
  - The authors note that multi-byte prediction alone is not enough: attention over long contexts becomes the bottleneck, so EVA is also required.
- **Data efficiency.** The authors claim it rivals tokenizer LMs with 5× less data, and outperforms BLT with 3–4× fewer bytes.
- **Instability.**
  - "Byte-level collapses" (typos) and loss spikes occurred.
  - Mitigations: Adam ε lowered from 1e-8 to 1e-12, skipping batches, and resetting optimizer state.

**Bolmo (Ai2, Dec 2025)** — [Ai2 blog](https://allenai.org/blog/bolmo)
- Bolmo 7B (from Olmo 3) and Bolmo 1B (from Olmo 2) are "byteified" from subword models, with the core frozen in a first stage.
- Ai2 says it is the best among BLT 7B, TFree-Hat 7B and EvaByte 6.5B on its evaluation suite. [CLAIM, vendor]
- Not directly applicable to Hope, which trains from scratch.

### Inferences
- AU-Net's 1B throughput table gives the best real-world reference: a hierarchical byte model with space-based pooling is not slower than BPE in throughput, while a flat byte transformer is about 4.5× slower.
- H-Net and SpaceByte both point to SSM/Mamba layers in the local encoder/decoder as a good choice. They also give a constant-size decoding state, with no growing KV cache. That is useful on an A770 without custom kernels, provided the recurrent step is implemented as a simple loop.

### Gaps
- No wall-clock figures for H-Net found.
- The quality cost of EvaByte's multi-byte decoding is not quantified.
- Hourglass transformers (Nawrot et al. 2022, [arXiv:2110.13711](https://arxiv.org/abs/2110.13711)) and dynamic token pooling (Nawrot et al. 2023, [arXiv:2211.09761](https://arxiv.org/abs/2211.09761)) were not re-read this session, so no numbers are given. They are the precursors of AU-Net and H-Net.

---

## 6. Techniques for emitting several bytes per step and their quality cost

### Takeaway
Byte-level-specific techniques exist with measured numbers:
- Self-speculation inside the BLT decoder: up to 77% less bandwidth with no loss.
- Block diffusion in the decoder: up to 92% less, with losses in code.
- Multi-byte heads with speculation: EvaByte, Gloeckle, MTPC.
- Scratchpad patching: 16-byte patches without quality loss.
- Subword or n-gram drafters: MambaByte, 2.6×.

Nearly all of the 2026 savings figures are estimates (NFEs and bandwidth), not wall-clock.

### Cited Findings

**Fast BLT (Kallini, Pagnoni, …, Zettlemoyer, Potts; Meta FAIR + Stanford, May 2026)** — [arXiv:2605.08044](https://arxiv.org/html/2605.08044v1)
- **Bottleneck.** "Standard byte-level decoding still generates one byte at a time" in BLT's local decoder.
- **BLT-D.** A block-wise masked diffusion objective is added to next-byte prediction. The decoder fills blocks of 4, 8 or 16 bytes in parallel over several steps.
  - More than 50% less estimated bandwidth on translation and code. BLT-D-16 reaches 87–92% less. [ESTIMATED]
- **BLT-S (self-speculation).** The local decoder drafts beyond the patch boundary, and the full model verifies in one pass. No extra model.
  - Up to 77% less bandwidth with no task loss. [ESTIMATED efficiency; quality DEMONSTRATED]
- **BLT-DV.** Diffusion drafts and the same model verifies autoregressively. Up to 81% less bandwidth, and it recovers part of BLT-D's quality loss.
- **Quality cost at 3B** (likelihood benchmarks) [DEMONSTRATED]

  | Benchmark | BLT | BLT-D-4 | BLT-D-16 |
  |---|---|---|---|
  | ARC-E | 74.33 | 72.39 | 66.89 |
  | HellaSwag | 74.98 | 71.86 | 69.12 |
  | MMLU | 41.15 | 39.07 | 37.08 |

  - BLT-D-16 holds up on translation but drops on code.
- **Scales.** 1B and 3B models with a 160M local decoder, trained on BLT-1T.
- **Limitation.** The NFE and bandwidth figures are proxies. Optimized wall-clock benchmarking is left to future work.

**Scratchpad Patching (Zheng, Bashlovkina, Dozat, Garrette, Rimell, Maynez; Google DeepMind/HKU, May 2026)** — [arXiv:2605.09630](https://arxiv.org/html/2605.09630v1)
- **Problem: "patch lag".** Bytes inside a patch predict from the previous patch until the current one closes. Quality falls as patch size grows.
- **Fix.** Transient scratchpads are inserted inside the patch and refresh the trunk's context. They are triggered by next-byte entropy and are not stored in the persistent KV cache.
  - The trigger rate can be adjusted at inference time without retraining.
- **Results at ~2B parameters (400B bytes)** [DEMONSTRATED]
  - With fixed p=16, NLU goes from 48.0 to 54.2, equal to the byte-level baseline at 54.1.
  - MBPP goes from 18.2% to 27.5%.
- **Efficiency.** 16× smaller KV cache and 3.7–4.5× fewer FLOPs/byte than the byte-level baseline. [ESTIMATED; latency not measured]
- **Limitations.** No training-FLOP savings. Under strict FLOP matching, scratchpads can hurt H-Net.

**Multi-byte prediction in hierarchical LMs (Owodunni, Okocha, Grant, Limisiewicz, Kumar; Aug 2026)** — [arXiv:2608.15454](https://arxiv.org/abs/2608.15454)
- Parallel prediction with a variable-length window aligned to the latent segments, using an attention mask that preserves causality. No extra parameters.
- The abstract claims "the best trade-off" between performance and throughput. [CLAIM; I read only the abstract, no numbers]

**MTPC – Multi-byte prediction with probabilistic circuits (Grivas et al., Nov 2025; PMLR 2026)** — [arXiv:2511.11346](https://arxiv.org/abs/2511.11346)
- Independent heads (as in EvaByte) limit speculative acceptance. Modeling the joint distribution of the next k bytes with PCs (mixtures, HMMs) "substantially" speeds up generation while preserving the verifier's output.
- Tested on EvaByte and on a byteified Llama-3.2-3B. [CLAIM; numbers not read]

**Multi-token prediction (Gloeckle et al., Meta, 2024)** — [arXiv:2404.19737](https://arxiv.org/abs/2404.19737)
- Self-speculation with an 8-byte prediction model reports roughly 6× speedup. Taken from the search snippet; I did not verify the table.

**MegaByte strided inference** — [Yu et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.07185)
- Mitigates the loss at the end of the patch at the cost of doubling inference.

### Inferences (concrete ideas for Hope, not validated)
1. **Self-speculation in the local decoder (BLT-S style).** This is the option with no quality loss. Let the local decoder keep emitting bytes past the patch boundary using the stale global state, then verify with one global plus local pass.
   - Hope already has a local KV cache and a per-patch global cache, so verification is a parallel prefill of k bytes.
2. **Use the n-gram table as a drafter (MambaByte style).**
   - Hope already holds a byte n-gram table for patching. Use it to propose the rest of the word when entropy is low.
   - Verify with the local decoder in one pass, and accept on exact greedy match, or top-k for lossy decoding.
   - Cost: zero parameters.
3. **K multi-byte heads (EvaByte style, 256+ε vocabulary).**
   - With a 256-byte vocabulary, adding 4–8 heads is cheap even for a 25M model.
   - Train with averaged losses, then decode self-speculatively with linear verification first; tree attention is more complex on XPU.
4. **Entropy-based adaptive patching at inference.**
   - The BLT and Scratchpad threshold can be lowered to gain quality or raised to gain speed, with no retraining.
   - Combine with Hope's adaptive halting, keeping in mind that every extra global step costs sequential latency.
5. **Local block diffusion (BLT-D-4).** A larger change. At block size 4 the quality loss is small, but code is sensitive, so it is not advisable for code.
6. **Hardware note (A770/XPU, batch 1).** At 25M–200M, the cost is probably dominated by kernel launch overhead and the number of sequential calls, not by FLOPs.
   - Techniques that reduce the number of sequential local-decoder calls (1, 2, 3) should give more real gain than reducing FLOPs.
   - This is inference only. No paper measured it on Arc/XPU.
7. **Budget.** BLT's ablations put most local layers in the decoder (9 decoder vs 1 encoder layer).
   - In a small model, that decoder can approach the global model's cost per byte: global cost / patch_size vs local cost × 1.
   - Measure the split before optimizing. With a 4.5-byte patch, a local decoder at ~20% of the global model already costs about the same per byte.

### Gaps
- No paper reports real wall-clock time for BLT-S/D or Scratchpad.
- No numbers on per-byte acceptance rates for Spanish or code.
- I found no results on Intel GPUs or without custom kernels.

---

## 7. Byte vs BPE at equal compute: scale, Spanish/English, code

### Takeaway
The advantage of byte models over BPE at equal compute is demonstrated at small scale for English and code with space patching (SpaceByte, ~150–250M parameters). It is demonstrated at 1B–8B with entropy, learned or hierarchical patching (BLT, H-Net, AU-Net). Flat byte models without patches lose clearly in throughput, about 4.5× in AU-Net. Spanish benefits slightly; Chinese and other languages without spaces suffer under space-based rules.

### Cited Findings
- **Small scale.** At 10^19 FLOPs, SpaceByte (~200M) reaches 1.009/0.748/0.500 bpb vs 0.989/0.768/0.508 for SentencePiece on PG-19, arXiv and GitHub. — [Slagle 2024](https://arxiv.org/html/2404.14408)
- **8B.** BLT-Entropy averaged 61.1 vs 60.0 for Llama 3 at equal training FLOPs. — [Pagnoni et al. 2024](https://arxiv.org/html/2412.09871v1)
- **Crossovers.** Byte models need data to overtake BPE.
  - BLT: 50B–1T bytes depending on the class.
  - H-Net XL: ~30B bytes (2-stage) and ~100B (1-stage). — [BLT](https://arxiv.org/html/2412.09871v1); [H-Net](https://arxiv.org/html/2507.07955v2)
- **Throughput.** At 1B, AU-Net 2 runs at 225k bytes/s vs 210k for BPE; a flat byte transformer runs at 47k. — [Videau et al. 2025](https://arxiv.org/html/2506.14761v1)
- **Spanish** (AU-Net 2 vs BPE, 8B): FLORES es→en 29.3 vs 28.2, MMLU-es 41.4 vs 37.6. — [same]
- **Code.**
  - H-Net: 0.316 vs 0.338 bpb.
  - BLT: HumanEval 35.4 vs 31.1.
  - SpaceByte: GitHub 0.500 vs 0.508.
  - The GitHub average patch is ~8 bytes, so it compresses better than prose. — [H-Net](https://arxiv.org/html/2507.07955v2); [BLT](https://arxiv.org/html/2412.09871v1); [SpaceByte](https://arxiv.org/html/2404.14408)
- **Patch size and UTF-8.**
  - Spanish accented characters take 2 bytes. SpaceByte counts UTF-8 continuation bytes as non-space-like, so words with accents stay in one patch. — [Slagle 2024](https://arxiv.org/html/2404.14408)

### Inferences
- For Hope at 25M–200M with limited data, space patching (SpaceByte) is the configuration with direct evidence at that scale. Entropy patching shows benefits mostly at ≥1B and with a lot of data.
- In "tokens-equivalent/s": a byte model with patches of ~4.5 bytes matches one BPE token per global step. Its effective speed is therefore BPE speed minus the local decoder's cost, unless the multi-byte techniques in Section 6 are applied.

### Gaps
- No byte-vs-BPE comparison found at <100M parameters.
- No published Spanish-specific bpb. AU-Net reports only BLEU and MMLU.
