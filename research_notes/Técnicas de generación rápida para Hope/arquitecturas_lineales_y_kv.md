# Linear-cost / fixed-state sequence architectures, hybrids, and KV-cache compression for fast small-model inference (2023–2026)

Notes for Hope HN-1: a 25M–200M parameter (up to ~2B) byte/patch model. Its global backbone is a diagonal selective SSM with a chunked scan, interleaved about 3:1 with RoPE attention. It runs on an Intel Arc A770 16GB with PyTorch XPU and Triton.

Source key: **[F]** means the claim was fetched and checked in this session (an arXiv abstract or HTML full text, or a GitHub README/INSTALL). **[A]** means the claim comes from the paper's abstract as I recall it and was not re-fetched in this session. Verify [A] items before quoting exact numbers.

---

## 1. Linear / fixed-state architectures (Mamba-2/SSD, RWKV-7, Gated DeltaNet, xLSTM, RetNet, GLA, flash-linear-attention): speed and quality vs Transformers at small scale; known weaknesses

### Takeaway
At about 0.4B–1.3B parameters and 100B tokens, the best linear-recurrent layers beat a Transformer++ baseline on perplexity and commonsense accuracy: Gated DeltaNet > Mamba2 > DeltaNet ≈ Transformer++. They lose clearly on recall-intensive in-context retrieval (SWDE, FDA) and on copying or few-shot ICL (MMLU 5-shot, Phonebook). Gated (delta-rule) updates narrow that gap but do not close it. Adding a few attention or SWA layers closes or reverses it. Decode memory is O(1) in sequence length: no KV cache, only a fixed matrix state per layer.

### Cited Findings
**Mamba-2 / SSD (Dao & Gu, 2024)**
- The SSD framework shows that SSMs and attention variants are related through decompositions of structured semiseparable matrices. Its core layer is claimed to be "2–8× faster" than Mamba-1's selective scan while staying competitive with Transformers on language modeling. [F] — [Dao & Gu 2024, arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- SSD computes a chunked algorithm: matmul-based quadratic attention inside each chunk, plus a linear recurrence over chunk states. This is why it maps to tensor cores and is easy to write in Triton or PyTorch. Mamba-2 also uses a scalar-times-identity A per head, which allows much larger state dimensions (N=64–256) than Mamba-1's N=16. [A] — [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)

**Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025)**
- Combines Mamba2-style gating (fast forgetting) with the delta rule (targeted overwrite). The authors report that it "consistently surpasses" Mamba2 and DeltaNet on LM, commonsense, in-context retrieval, length extrapolation, and long-context tasks. [F] — [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- 1.3B parameters, 100B FineWeb-Edu tokens, 4K training length. Wiki ppl / LAMBADA ppl / average accuracy: Transformer++ 18.53/18.32/52.25; Mamba2 16.56/12.56/54.89; DeltaNet 17.71/16.88/52.14; Gated DeltaNet 16.42/12.17/55.32. [F] — [arXiv:2412.06464v3 HTML](https://arxiv.org/html/2412.06464v3)
- Recall-intensive average (SWDE/SQuAD/FDA/TQA/NQ/Drop, 2K truncation): Transformer++ 37.0, Mamba2 29.8, DeltaNet 26.2, Gated DeltaNet 30.6. FDA is the starkest gap: Transformer 52.2 vs Mamba2 25.3 vs GDN 23.7. [F] — [same](https://arxiv.org/html/2412.06464v3)
- Training throughput at 1.3B on one H100: GDN and DeltaNet are "slightly slower than Mamba2," which runs at about 2–3K tokens/s in the paper's figure. Transformer++ with FlashAttention-2 is fastest at 2K context. Linear models win as context grows. [F] — [same](https://arxiv.org/html/2412.06464v3)

**RWKV-7 "Goose" (Peng et al., 2025)**
- Four models from 0.19B to 2.9B parameters, trained on a 3.1T-token multilingual corpus. Claims a new 3B SoTA on multilingual tasks and parity with 3B SoTA on English, despite "dramatically fewer tokens" than competitors. Memory and time per token are constant. [F] — [arXiv:2503.14456](https://arxiv.org/abs/2503.14456)
- The state update is a generalized delta rule with vector-valued gating and in-context learning rates. The authors claim it can do state tracking and recognize all regular languages (beyond TC⁰ Transformers under standard conjectures) while staying parallelizable. Released under Apache-2.0. [F] — [arXiv:2503.14456](https://arxiv.org/abs/2503.14456)

**xLSTM, RetNet, GLA (abstract-level)**
- xLSTM (Beck et al., 2024) adds exponential gating plus sLSTM (scalar memory, memory mixing, not parallelizable) and mLSTM (matrix memory, covariance update, fully parallelizable). The authors claim it compares favorably with Transformers and SSMs in their scaling study. [A] — [arXiv:2405.04517](https://arxiv.org/abs/2405.04517)
- RetNet (Sun et al., 2023) introduces a retention mechanism with three equivalent forms: parallel, recurrent (O(1) per-token decoding), and chunkwise-recurrent (linear-cost training). The authors claim lower inference memory and latency than Transformers. Baselines were later criticized as weak; independent comparisons such as GLA and Gated DeltaNet place RetNet below Mamba and GLA. [A] — [arXiv:2307.08621](https://arxiv.org/abs/2307.08621)
- GLA (Yang et al., 2023/ICML 2024) uses data-dependent gated linear attention with a hardware-efficient chunkwise algorithm, FLASHLINEARATTENTION. The authors claim it is faster than FlashAttention-2 even at 1K length and competitive with LLaMA-arch Transformers and Mamba at 340M/1.3B scale. [A] — [arXiv:2312.06635](https://arxiv.org/abs/2312.06635)

**Recall weakness (mechanistic evidence)**
- Pure SSMs (Mamba, Mamba-2) at 8B / 3.5T tokens match or beat Transformers on many tasks. They trail on copying and ICL tasks such as 5-shot MMLU and Phonebook, and on long-context reasoning. [F] — [Waleffe et al. (NVIDIA) 2024, arXiv:2406.07887](https://arxiv.org/abs/2406.07887)
- Jamba ablation: on IMDB ICL, pure Mamba scored 48.8 vs attention 84.1 vs hybrid 90.9. Pure Mamba often ignored the answer format; the authors attribute this to missing induction-style copying. [F] — [Lieber et al. 2024, arXiv:2403.19887v2](https://arxiv.org/html/2403.19887v2)
- Zoology / Based: associative-recall (MQAR) quality of fixed-state models is bounded by recurrent state size, and this explains most of the perplexity gap to attention. [A] — [Arora et al. 2023, arXiv:2312.04927](https://arxiv.org/abs/2312.04927); [Arora et al. 2024 "Based", arXiv:2402.18668](https://arxiv.org/abs/2402.18668)

**flash-linear-attention (fla) library**
- Triton kernels for about 40 architectures, including RetNet, GLA, DeltaNet, Gated DeltaNet, Mamba2, Mamba3, RWKV6/7, NSA, and MoBA. The README says it is "platform-agnostic and verified on NVIDIA, AMD, and Intel hardware" and offers `[cuda]`, `[rocm]`, `[xpu]`, `[npu]`, and `[cpu]` install extras. All published benchmarks are on an NVIDIA GB200; there are no XPU numbers. [F] — [fla-org/flash-linear-attention README](https://github.com/fla-org/flash-linear-attention)
- XPU install: `pip install --index-url https://download.pytorch.org/whl/xpu torch` then `pip install flash-linear-attention[xpu]`. Requires torch>=2.7.0, which pulls `pytorch-triton-xpu`. No XPU-specific limitations are listed; the docs point to the FAQs. [F] — [fla INSTALL.md](https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/INSTALL.md)

### Inferences
- For Hope's 25M–200M global backbone, swapping the diagonal selective SSM for **Gated DeltaNet** or **Mamba-2 (SSD)** via fla's `chunk_*` kernels is the most evidence-backed upgrade. GDN has the best small-scale perplexity, and the delta rule should help with byte-level patch recall. The underlying evidence is 1.3B/100B-token results, so gains at ≤200M are extrapolated.
- Hope's current diagonal SSM with 16-step chunks is close to Mamba-1-style recurrence with a small state. The Zoology/Based results suggest that state size, not only the gating form, is the main lever on recall, which argues for matrix-valued states (SSD, GLA, DeltaNet).
- Since fla officially lists Intel XPU support, the main risk is per-kernel Triton-XPU performance and correctness, not availability. Benchmark `chunk_gdn` and `chunk_simple_gla` against Hope's pure-PyTorch scan before committing.

### Gaps
- No independent XPU or Arc A770 throughput numbers for fla kernels were found. All published fla benchmarks are NVIDIA.
- No head-to-head study below 300M parameters covering all of Mamba-2, GDN, RWKV-7, xLSTM, and GLA under identical data. Most comparisons are 340M–1.3B.
- Exact decode tokens/s per architecture at small batch (the batch=1 latency that matters for interactive generation) was not extracted. The papers report mostly training throughput or large-batch numbers.
- xLSTM, RetNet, and GLA details are abstract-level recall ([A]) and were not re-fetched.

---

## 2. Hybrids (Jamba, Samba, Hymba, Zamba/Zamba2) and the best attention:SSM ratio and placement

### Takeaway
Hybrids beat both pure Transformers and pure SSMs at matched compute. A small attention fraction is enough: Jamba uses 1:7, NVIDIA's 8B hybrid has ~7% of layers as attention, and Hymba-1.5B has only 3 global-attention layers (first, middle, last), with SWA elsewhere. Jamba found 1:3 and 1:7 nearly identical at 1.3B. Hope's ~3:1 SSM:attention ratio is therefore attention-heavy relative to the literature. It could likely be pushed toward 1:5–1:7 global, or use SWA for most attention layers, with little quality loss and a much smaller KV cache.

### Cited Findings
**Jamba (AI21, Lieber et al. 2024)**
- 1 attention layer per 7 Mamba layers in each 8-layer block, with MoE in some layers. At 256K context with a 16-bit KV cache, Jamba needs 4GB vs Mixtral 32GB vs LLaMA-2 128GB. It reaches about 3× Mixtral's throughput at 8K with large batches on an A100, and also at 128K. [F] — [arXiv:2403.19887v2](https://arxiv.org/html/2403.19887v2)
- Ablation at 1.3B / 250B tokens: a:m = 1:3 and 1:7 performed "nearly identically." 1:7 was chosen for compute efficiency. [F] — [same](https://arxiv.org/html/2403.19887v2)
- With no explicit positional encoding, the hybrid matched RoPE-in-attention, plausibly because the preceding Mamba layers supply position. Large-scale loss spikes were fixed by adding RMSNorm inside the Mamba layers. [F] — [same](https://arxiv.org/html/2403.19887v2)

**NVIDIA Mamba-2-Hybrid (Waleffe et al. 2024)**
- 8B hybrid with 43% Mamba-2, 7% attention, and 50% MLP layers. It beats the 8B Transformer on all 12 standard tasks (+2.65 points average) and matches or exceeds it on 23 long-context tasks. Generation speedup of "up to 8×" is *predicted*, not measured. [F] — [arXiv:2406.07887](https://arxiv.org/abs/2406.07887)

**Samba (Microsoft, Ren et al. 2024)**
- Layer-wise Mamba + sliding-window attention (+MLP), up to 3.8B parameters, 3.2T tokens. Throughput is 3.73× a GQA Transformer at 128K prompts and 3.64× when generating 64K tokens. Pretrained at 4K, it improves perplexity zero-shot up to 1M tokens. Fine-tuned at 4K, it gets perfect passkey retrieval at 256K and better Phonebook extrapolation than full attention. [F] — [arXiv:2406.07522](https://arxiv.org/abs/2406.07522)

**Hymba (NVIDIA, Dong et al. 2024)**
- Attention heads and Mamba heads run in *parallel* within each layer. 128 learnable meta tokens are prepended. Only 3 layers use global attention (first, middle, last); the rest use SWA (window 256 in 300M ablations). Consecutive layers share their KV cache in pairs. The attention:Mamba parameter ratio is about 1:5.23. [F] — [arXiv:2411.13676 HTML](https://arxiv.org/html/2411.13676v1)
- Hymba-1.5B-Base vs Llama-3.2-3B: +1.32% average accuracy, 11.67× smaller cache, 3.49× throughput. [F] — [arXiv:2411.13676](https://arxiv.org/abs/2411.13676)
- Hymba-1.5B: 79MB cache and 664 tok/s on an A100 (8K sequence, batch 128, PyTorch, FP16 cache). Qwen2.5-1.5B: 229MB and 469 tok/s. Vs SmolLM2-1.7B (11T tokens): +1.02% accuracy, 19.91× smaller cache, 2.79× throughput. [F] — [HTML](https://arxiv.org/html/2411.13676v1)
- Ablations: parallel beat sequential hybrid by +1.08 commonsense and +4.74 recall. Global attention in only 3 layers recovered recall. Cross-layer KV sharing gave about 1.15× throughput and +0.60 accuracy. [F] — [HTML](https://arxiv.org/html/2411.13676v1)
- The paper is internally inconsistent on training tokens: the tables say 1.5T and §2.5 says 50B. [F] — [HTML](https://arxiv.org/html/2411.13676v1)

**Gated DeltaNet hybrids**
- H1 (GDN + SWA, 2K window) at 1.3B: retrieval average 39.0 vs Transformer++ 37.0, and average accuracy 56.40. H2 (Mamba2 + GDN + SWA): retrieval 40.1. Small-scale hybrids with SWA only, with no global attention, *exceed* the full Transformer on recall. [F] — [arXiv:2412.06464v3](https://arxiv.org/html/2412.06464v3)

**Zamba / Zamba2 (Zyphra)**
- Zamba2 comes in 1.2B, 2.7B, and 7.4B sizes, as hybrid Mamba2-Transformer models trained on up to 3T tokens. Zyphra claims substantial latency, throughput, and memory gains vs open models of their class; no figures appear in the abstract. [F] — [arXiv:2411.15242](https://arxiv.org/abs/2411.15242)
- Zamba's distinctive design: one (Zamba2: two alternating) *shared-weight* attention+MLP block reused every N Mamba blocks. Zamba2 adds per-invocation LoRA on the shared block, giving a very low attention parameter cost. [A] — [Glorioso et al. 2024 Zamba, arXiv:2405.16712](https://arxiv.org/abs/2405.16712); [arXiv:2411.15242](https://arxiv.org/abs/2411.15242)

### Inferences
- Ratio: the evidence clusters at 1:5 to 1:7 attention:SSM (Jamba, Hymba, NVIDIA ~1:6 by layer count), with Jamba showing 1:3 ≈ 1:7 in quality. Hope's 3:1 SSM:attention (25% attention) sits at the attention-heavy end. Reducing global attention to 2–3 layers (first/middle/last, following Hymba) and turning the others into SWA would shrink the KV cache with little expected quality loss. This has not been tested at 25M–200M scale.
- Placement: Hymba's evidence favors putting global attention at the first, middle, and last layers. GDN-H1/H2 show SWA alone can suffice at ≤2K contexts. Parallel heads in one layer (Hymba) beat sequential stacking in Hymba's own ablation, but this is a single-paper result.
- Positional encoding: Jamba's no-PE result suggests the RoPE in Hope's attention layers may be removable. This is an experimental finding at 1.3B only.

### Gaps
- No published ratio sweep below 300M parameters was found. Jamba's 1.3B ablation is the smallest direct evidence.
- No evidence on hybrids for byte-level or patch-level models specifically; for example, whether the local byte decoder benefits from SSM layers.
- Zamba2's quantitative latency numbers were not extracted.

---

## 3. KV-cache compression: GQA/MQA, MLA, KV quantization (KIVI, KVQuant), sliding window, cross-layer sharing (CLA, YOCO)

### Takeaway
These methods are orthogonal and stack. Group sharing (GQA/MQA) gives H/G× savings. Cross-layer sharing (CLA, Hymba, YOCO) gives a further 2× to ~L×. Quantization (KIVI 2-bit) gives about 8× on the cache relative to FP16. SWA bounds the cache at the window size. MLA gives about 93% savings but needs more engineering, including decoupled RoPE. For a ≤200M model trained from scratch, GQA + CLA-2 + SWA in most layers is the cheapest set to implement and is well validated at 1–3B scale.

### Cited Findings
- **MQA** (Shazeer 2019) uses a single K/V head shared by all query heads and cuts KV memory bandwidth for incremental decoding with minor quality loss. [A] — [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- **GQA** (Ainslie et al. 2023) uses G K/V groups, interpolating between MHA and MQA. Uptraining MHA checkpoints with about 5% of pretraining compute reaches quality close to MHA at speed comparable to MQA. [A] — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- **CLA** (Brandon et al. 2024) shares K/V between adjacent layers on top of MQA, cutting the KV cache by another 2×. Tested at 1B and 3B from scratch, it keeps accuracy at nearly unmodified MQA levels and is reported as a Pareto improvement. [F] — [arXiv:2405.12981](https://arxiv.org/abs/2405.12981)
- **YOCO** (Sun et al. 2024) is a decoder-decoder design. A self-decoder with efficient attention (SWA or gated retention) produces one global KV cache, which every cross-decoder layer reuses. Prefill can exit early. It is extended to 1M context with near-perfect needle retrieval, and the authors claim order-of-magnitude memory and prefill gains. [F] — [arXiv:2405.05254](https://arxiv.org/abs/2405.05254)
- **Hymba cross-layer KV sharing** (pairs of layers) gave about 1.15× throughput and +0.60 accuracy at small scale. This is an independent confirmation of CLA-style sharing. [F] — [arXiv:2411.13676 HTML](https://arxiv.org/html/2411.13676v1)
- **MLA** (DeepSeek-V2 2024) compresses K/V into a low-rank latent with decoupled RoPE keys. The authors report a 93.3% KV-cache reduction and 5.76× maximum generation throughput vs DeepSeek 67B. [A] — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- **KIVI** (Liu et al. 2024) is tuning-free asymmetric 2-bit KV quantization: keys per-channel, values per-token. It reports up to 2.6× lower peak memory (including weights), 4× larger batch, and 2.35–3.47× throughput on Llama, Falcon, and Mistral. [F] — [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
- **KVQuant** (Hooper et al. 2024) combines per-channel pre-RoPE key quantization, non-uniform datatypes, and dense-and-sparse outlier handling. It reports <0.1 perplexity degradation at 3 bits and enables 1M context on one A100-80GB (10M on 8 GPUs) for LLaMA-7B. [A] — [arXiv:2401.18079](https://arxiv.org/abs/2401.18079)
- **Sliding-window attention** bounds the KV cache at the window W per layer. Samba (SWA + Mamba) runs 3.64× faster than GQA when generating 64K tokens [F] — [arXiv:2406.07522](https://arxiv.org/abs/2406.07522). GDN-H1 with a 2K window beats full attention on recall at 1.3B [F] — [arXiv:2412.06464v3](https://arxiv.org/html/2412.06464v3).
- Scale of the combined effect: Jamba's 1:7 hybrid needs 4GB of KV at 256K vs 32GB for GQA-based Mixtral [F] — [arXiv:2403.19887v2](https://arxiv.org/html/2403.19887v2). Hymba's 3-global + SWA + KV sharing gives a 79MB cache vs 229MB for Qwen2.5-1.5B (8K) [F] — [HTML](https://arxiv.org/html/2411.13676v1).

### Inferences
- For Hope at 25M–200M, the KV cache is small in absolute terms (MBs). Its cost is mainly memory *bandwidth* per decode step. GQA (e.g., 4 query heads per KV head) plus CLA-2 on the attention layers is trivially implementable in PyTorch (repeat_interleave or SDPA with broadcast) and gives about 8× savings with near-zero quality loss per CLA and GQA evidence at 1–3B.
- The local byte encoder/decoder attention stacks are natural SWA candidates, since the window equals the patch span plus some margin. That bounds their cache to a constant.
- KV quantization (KIVI-style) mostly pays off for long contexts and large batches. At small model size and batch 1 on an A770, dequantization overhead without a fused kernel may cancel the gain. This is unverified; it would need a Triton fused dequant-attention kernel.
- MLA's benefit is largest for large head counts. For tiny models, GQA + CLA is simpler, and the incremental benefit of MLA is unproven at ≤200M.

### Gaps
- No published CLA, YOCO, or MLA results below 1B parameters were found.
- No KV-quantization results on Intel XPU, and no evidence on whether IPEX or PyTorch XPU ships fused low-bit KV attention.
- MQA, GQA, MLA, and KVQuant figures are abstract-level recall ([A]) and were not re-fetched this session.

---

## 4. Implementability in pure PyTorch / Triton (Intel XPU) without CUDA-only kernels; chunked/parallel scan formulations

### Takeaway
Everything in sections 1–3 can be implemented without CUDA-only code. The chunked formulations (SSD for Mamba-2, chunkwise retention, FLA chunk for GLA/DeltaNet/GDN, RWKV-7 chunk) reduce each layer to batched matmuls plus a short inter-chunk recurrence. That runs in pure PyTorch, and fla provides Triton versions with an official `[xpu]` extra. What is CUDA-only is the *reference* Mamba-1/Mamba-2 package (`mamba_ssm`, `causal-conv1d`, with CUDA selective-scan kernels), so it should not be used directly on Arc.

### Cited Findings
- fla kernels are written in Triton, described as "platform-agnostic and verified on NVIDIA, AMD, and Intel hardware," with an `[xpu]` install target. It covers Mamba2, GDN, GLA, RetNet, DeltaNet, and RWKV7. [F] — [fla README](https://github.com/fla-org/flash-linear-attention)
- XPU path: torch>=2.7 from the XPU wheel index, then `flash-linear-attention[xpu]`, which uses `pytorch-triton-xpu`. [F] — [fla INSTALL.md](https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/INSTALL.md)
- The SSD chunked algorithm splits the sequence into chunks, computes intra-chunk outputs as masked attention-like matmuls, and passes states between chunks with a recurrence. This is the "state space duality" that makes Mamba-2 matmul-bound and faster than Mamba-1's scan (2–8×). [F/A] — [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- GLA's FLASHLINEARATTENTION is a chunkwise algorithm that trades memory movement for parallelism, with an I/O-aware Triton implementation. [A] — [arXiv:2312.06635](https://arxiv.org/abs/2312.06635)
- RetNet defines a chunkwise-recurrent form for linear-complexity training alongside an O(1) recurrent decode form. [A] — [arXiv:2307.08621](https://arxiv.org/abs/2307.08621)
- Gated DeltaNet provides a hardware-efficient parallel (chunkwise, WY-representation) training algorithm, and its kernels are in fla (`chunk_gdn`). [F] — [arXiv:2412.06464](https://arxiv.org/abs/2412.06464); [fla README](https://github.com/fla-org/flash-linear-attention)
- Hymba's throughput numbers were measured with a PyTorch implementation, showing that hybrid gains are realizable without bespoke inference engines. [F] — [arXiv:2411.13676 HTML](https://arxiv.org/html/2411.13676v1)

### Inferences
- Recommended path for Hope on an Arc A770:
  1. Keep a pure-PyTorch chunked reference for correctness.
  2. Install fla[xpu] and A/B-test `chunk_simple_gla` (scalar-decay, Mamba-2-like), `chunk_gdn`, and the `fused_recurrent_*` decode kernels against the existing 16-step scan.
  3. Use chunk sizes of 64 for SSD/GLA-style intra-chunk matmuls; fla's defaults are 64. A 16-step chunk underuses the XMX matrix engines. (This is an inference from the matmul-bound design; not benchmarked on XPU.)
- At decode time, every linear layer is a per-token rank-1 state update (matrix state d_k×d_v per head). That is a few small GEMVs, which is fine in plain PyTorch at batch 1. torch.compile or CUDA-graph-style capture (XPU graph support is limited) matters more than kernels for small-model latency.
- Attention layers with GQA, CLA, or SWA work via `torch.nn.functional.scaled_dot_product_attention`. XPU SDPA backend coverage, such as the flash path vs math fallback, should be checked; a Triton flash-attention kernel is a fallback.

### Gaps
- No published measurements of fla Triton-XPU kernel speed, or of the pure-PyTorch chunked scan on Arc A770.
- I could not confirm which fla kernels have known XPU issues; the FAQs.md backend section was not fetched.
- No evidence was found on RWKV-7's official CUDA kernels vs fla's Triton RWKV7 for XPU. BlinkDL/RWKV-LM primarily ships CUDA kernels; this is [A] and unverified this session.
