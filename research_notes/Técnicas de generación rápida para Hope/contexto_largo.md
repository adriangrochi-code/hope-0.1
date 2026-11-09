# Long-context extension without slower generation or exploding memory (2023–2026), with fit assessment for Hope HN-1

Legend: **[V]** = number checked this session against the arXiv abstract or HTML. **[M]** = taken from the reviewer's prior knowledge of the paper and not re-fetched this session; the number is likely right but should be spot-checked before it is quoted in a final report. "Demonstrated" means shown at scale in a peer-reviewed paper or by an independent party. "Experimental" means only the original paper shows it, or reproductions failed.

Hope context used throughout: byte-level, about 4.5 bytes per patch. The current 2048-byte window is about 455 patches. A 100 KB source file is about 22K patches, and 1 MB is about 220K patches. Because of the patching, every "N tokens" figure below costs Hope about 4.5×N bytes of text but only about N global-backbone steps.

---

## 1. Position extrapolation (PI, NTK-aware, YaRN, LongRoPE): does it need fine-tuning?

### Takeaway
RoPE rescaling is the cheapest and most mature way to lengthen the attention layers. It adds zero cost per generated token, and the KV cache still grows linearly with length. YaRN and LongRoPE need only a few hundred to about 1K fine-tuning steps, while training-free variants (Dynamic NTK/YaRN) safely reach only about 2× to 8×. Passkey or perplexity success overstates real usable context. RULER found the YaRN-128K Llama-2 base model has an effective length below 4K.

### Cited Findings
- **Position Interpolation (Chen et al., 2023).** Linearly down-scales position indices so they stay inside the trained range. It extended LLaMA 7B–65B to 32,768 tokens with ≤1000 fine-tuning steps. Its interpolation bound is about 600× smaller than the extrapolation bound **[M]** — [arXiv 2306.15595](https://arxiv.org/abs/2306.15595)
- **NTK-aware scaling (bloc97, Reddit/blog, 2023; formalised in the YaRN paper).** Changes the RoPE base instead of the positions, so high-frequency dimensions are barely touched. "Dynamic NTK" recomputes the scale as the sequence grows, without fine-tuning. Fine-tuned NTK is what Code Llama uses — [YaRN paper, arXiv 2309.00071](https://arxiv.org/html/2309.00071v2)
- **YaRN (Peng et al., 2023) mechanism.** "NTK-by-parts" interpolation leaves high-frequency RoPE dims unscaled and interpolates low-frequency dims, with ramp α=1, β=32 for Llama. It also applies an attention temperature √(1/t) = 0.1·ln(s) + 1 **[V]** — [arXiv 2309.00071](https://arxiv.org/html/2309.00071v2)
- **YaRN training cost.** s=16 (64K) took 400 steps at batch 64 and LR 2e-5 on PG19 64K segments. s=32 (128K) took another 200 steps, about 0.1% of the pre-training corpus. The paper claims 10× fewer tokens and 2.5× fewer steps than prior methods **[V]** — [arXiv 2309.00071](https://arxiv.org/html/2309.00071v2)
- **YaRN results.** On Proof-pile at 10,240 tokens (Llama-2 7B), perplexity was PI 8.07, NTK 6.24, YaRN 6.04, with YaRN using 400M training tokens against 1B for PI and NTK. At 128K, YaRN-7B scored perplexity 2.37 against 2.71 for Code Llama (NTK). Passkey accuracy was 99.4% at 128K, and the average short-benchmark drop from s=16 to s=32 was only 0.49% **[V]** — [arXiv 2309.00071](https://arxiv.org/html/2309.00071v2)
- **YaRN without fine-tuning.** Dynamic scaling extends inference length by more than 2× without fine-tuning (shown only as a chart) **[V]** — [arXiv 2309.00071](https://arxiv.org/html/2309.00071v2)
- **LongRoPE (Ding et al., 2024).** An evolutionary search finds non-uniform per-dimension and per-position rescale factors, enabling 8× extension without fine-tuning. A progressive scheme fine-tunes at 256K for ≤1K steps, then interpolates again to 2048K. The factors are then readjusted at 8K to recover short-context quality. The architecture is unchanged, so existing kernels still apply **[V]** — [arXiv 2402.13753](https://arxiv.org/abs/2402.13753)
- **Reality check (RULER).** Yarn-base (7B, claimed 128K) has an effective length under 4K; it scores 77.3 at 4K against a 79.4 threshold. Its passkey and vanilla NIAH results are strong at short lengths but collapse by 128K. LongChat, LongAlpaca and LWM (1M claimed) also come out under 4K **[V]** — [RULER, arXiv 2404.06654](https://arxiv.org/html/2404.06654)

### Inferences
- For Hope's RoPE attention layers, the cost of YaRN is a few lines of PyTorch: change the inv_freq per dim and multiply the logits by the temperature. It needs no kernels, so it is fully XPU-compatible. It is the obvious first step from 455 to about 2K–4K patches (about 9–18 KB).
- RULER's results show that positional scaling fixes *perplexity and passkey*, not *use* of the context. Real recall requires training on long data with long-dependency tasks (multi-key NIAH, code repos).
- In Hope only about 1 in 4 backbone layers is attention. The Mamba layers do not use RoPE and have a different extrapolation failure mode (see Q7), so YaRN alone will not extend Hope.

### Gaps
- No source found that applies YaRN or LongRoPE to byte-patch or hierarchical models. Patch boundaries make "position" content-dependent, which is untested.
- The PI numbers (32,768 tokens, ≤1000 steps) were not re-fetched.

---

## 2. Constant-memory streaming: StreamingLLM / attention sinks and sliding-window attention (Mistral)

### Takeaway
Attention sinks plus a sliding window give truly constant memory and constant per-token cost, and are mature (training-free). They do **not** extend recall: tokens that leave the window are gone. DuoAttention is the refined version: only "retrieval heads" keep the full cache, while streaming heads keep sinks plus a recent window.

### Cited Findings
- **StreamingLLM (Xiao et al., 2023).** Keeping the KV of a few initial "sink" tokens plus a rolling window gives stable language modeling up to 4M+ tokens with Llama-2, MPT, Falcon and Pythia, without fine-tuning. It is up to 22.2× faster than sliding-window-with-recomputation **[V]** — [arXiv 2309.17453](https://arxiv.org/abs/2309.17453)
- **StreamingLLM details.** It uses 4 sink tokens by default. Pre-training with a dedicated learnable sink token lets 1 sink suffice. The authors' FAQ states it does not extend the context window or improve long-range memory: it only keeps generation stable **[M]** — [repo mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm)
- **Mistral 7B (Jiang et al., 2023).** Uses sliding-window attention with W=4096. Because the window stacks across layers, the theoretical attention span at the last layer is about 131K tokens (32 layers × 4K). A rolling buffer cache cuts cache memory 8× at 32K sequence length, with no quality loss reported **[M]** — [arXiv 2310.06825](https://arxiv.org/abs/2310.06825)
- **Mistral in practice.** Mistral-v0.2 7B (32K claimed) has a RULER effective length of 16K. Mistral-base is effective to 16K and Mixtral-base to 32K **[V]** — [RULER, arXiv 2404.06654](https://arxiv.org/html/2404.06654)
- **DuoAttention (Xiao et al., 2024).** Identifies "retrieval heads" by optimisation; these keep full KV, while the other heads use sinks plus a recent window. It reports up to about 2.55× memory reduction for MHA models and about 3.3M-token context on one A100 when combined with quantization **[M]** — [arXiv 2410.10819](https://arxiv.org/abs/2410.10819)

### Inferences
- Hope already has a sliding-window fallback. Adding 1–4 sink patches (or a learnable sink) is a near-free stability fix for streaming beyond 2048 bytes, and should be done first.
- The theoretical span argument (layers × window) only holds if the information is actually propagated, and RULER results suggest it often is not. For Hope, the SSM layers and the recurrent core are the real long-range carriers, not stacked windows.

### Gaps
- No direct measurement found of attention-sink behaviour in byte-level or hierarchical models.

---

## 3. Recurrent / compressive memory: RMT, ARMT, Infini-attention, Memorizing Transformers, Compressive Transformer, Titans, Landmark attention

### Takeaway
Segment-level recurrence with a fixed-size memory (RMT/ARMT, Titans) gives *constant per-segment cost* and the best published results on extreme-length *reasoning-over-facts* benchmarks (BABILong). It needs training, and recall degrades with the number of compressions. Infini-attention's results did not reproduce in an independent attempt. Retrieval-style memories (Memorizing Transformers, Landmark attention) give exact recall, but their memory grows linearly and they need a top-k search.

### Cited Findings
- **RMT (Bulatov et al., 2022–2024).** Memory tokens are passed between segments. "Scaling Transformer to 1M tokens and beyond with RMT" (BERT-base) reported fact-memorisation tasks at up to 2M tokens with linear compute **[M]** — [arXiv 2304.11062](https://arxiv.org/abs/2304.11062)
- **BABILong (Kuratov et al., 2024).** Popular LLMs effectively use only 10–20% of their context. RAG gets about 60% on single-fact QA, independent of length. Fine-tuned recurrent memory transformers perform best, handling up to 50M tokens **[V]** — [arXiv 2406.10149](https://arxiv.org/abs/2406.10149)
- **ARMT (Rodkin et al., 2024).** Associative (key-value, linear-attention-like) memory inside RMT. It scored 79.9% on single-fact BABILong QA over 50M tokens and aims for constant time per new segment **[V]** — [arXiv 2407.04841](https://arxiv.org/abs/2407.04841)
- **Infini-attention (Munkhdalai et al., Google, 2024).** A compressive linear-attention memory runs in each attention layer, gated with local attention. The paper claims 1M passkey with a 1B model, 500K book summarisation with an 8B model, and a 114× memory compression ratio **[M]** — [arXiv 2404.07143](https://arxiv.org/abs/2404.07143)
- **Infini-attention reproduction (Hugging Face, 2024): failed.** On a 200M from-scratch model and on Llama-3 8B, passkey recall worked within a segment but failed across segments. Quality degraded with more compressions, the gating balance factors stuck near 0.5, and the 200M run went NaN after about 20B tokens. The authors conclude that RoPE scaling, YaRN and Ring Attention remain "the best ways" to extend a pretrained model **[V]** — [HF blog](https://huggingface.co/blog/infini-attention)
- **Titans (Behrouz, Zhong & Mirrokni, Google, Dec 2024 / 2025).** The neural long-term memory is an MLP updated at test time by gradient descent on an associative loss ‖M(k)−v‖². It uses "surprise" with momentum (decay η_t) and a forgetting gate α_t: M_t=(1−α_t)M_{t−1}+S_t. There are three variants: MAC (memory as context, prepended to the segment), MAG (memory gated with sliding-window attention) and MAL (memory as a layer before SWA). Models had 170M–760M parameters trained on 15–30B tokens **[V]** — [arXiv 2501.00663](https://arxiv.org/html/2501.00663v1)
- **Titans results.** On S-NIAH-N, Titans-MAC goes 99.6→97.4 from 2K to 16K, while Mamba2 S-NIAH-PK falls 98.6→5.4 and TTT 60.2→4.4. It beats GPT-4 and Llama-3.1-8B+RAG on BABILong (with about 70× fewer parameters than the latter). The paper claims scaling beyond 2M context, but the S-NIAH tables stop at 16K. The memory module is slightly slower than Mamba2 and Gated DeltaNet in training throughput **[V]** — [arXiv 2501.00663](https://arxiv.org/html/2501.00663v1)
- **Memorizing Transformers (Wu et al., 2022).** One layer does kNN lookup into a non-differentiable cache of past (k,v), up to 262K entries. Perplexity improved steadily on arXiv, code (GitHub) and PG-19, and a memory can be added to an existing model with short fine-tuning **[M]** — [arXiv 2203.08913](https://arxiv.org/abs/2203.08913)
- **Compressive Transformer (Rae et al., 2019).** Old memories are compressed (for example by strided conv) into a secondary FIFO. Introduced PG-19 **[M]** — [arXiv 1911.05507](https://arxiv.org/abs/1911.05507)
- **Landmark attention (Mohtashami & Jaggi, 2023).** A landmark token per block lets attention pick relevant blocks (top-k), which can be retrieved from CPU memory. Fine-tuning LLaMA-7B extended it to 32K+ tokens **[M]** — [arXiv 2305.16300](https://arxiv.org/abs/2305.16300)

### Inferences
- Hope's **shared-weight recurrent core plus Hopfield zone** is structurally close to RMT/ARMT: a recurrent state carried across segments plus associative memory. Hope's **gated memory cross-attention (closed at init)** is close to Memorizing Transformers, Landmark attention and RETRO-style chunk cross-attention. These are the cheapest-to-adopt families because the scaffolding exists.
- Titans/ARMT-style test-time associative updates are written in plain tensor ops (outer products or small-MLP gradient steps), so they are feasible in pure PyTorch on XPU. The parallel chunked training form is more complex, and a naive per-token loop will be slow on XPU.
- The Infini-attention failure is a warning. Gated compressive memories can learn to ignore the memory (gates stuck at 0.5, or closed). Hope's closed-at-init gate needs auxiliary tasks that *force* use of the memory (multi-segment passkey, cross-file code references) or it may never open.

### Gaps
- No independent reproduction of Titans found within this search budget.
- Exact RMT 2M-token numbers and the Infini-attention 114× figure were not re-fetched.

---

## 4. Context compression: gist tokens, AutoCompressors, ICAE, LLMLingua

### Takeaway
Soft compression (gist, AutoCompressor, ICAE) compresses context 4×–26× into learned vectors and needs training. It works well for instructions and short documents but is lossy for exact recall, a poor match for code that needs verbatim identifiers. Hard prompt compression (LLMLingua) drops tokens with a small LM, giving up to about 20× compression with task-dependent loss.

### Cited Findings
- **Gist tokens (Mu et al., 2023).** An attention-mask modification trains the model to compress a prompt into a few "gist" tokens. It reports up to 26× prompt compression, about 40% FLOPs reduction and about 4.2% wall-clock speedup with minimal quality loss (LLaMA-7B, FLAN-T5-XXL) **[M]** — [arXiv 2304.08467](https://arxiv.org/abs/2304.08467)
- **AutoCompressors (Chevalier et al., 2023).** OPT/Llama-2 are fine-tuned to summarise each segment into summary vectors passed to later segments, which is RMT-like. It was trained on sequences up to 30,720 tokens, and the summary vectors were used as cheap pre-computed retrieval contexts **[M]** — [arXiv 2305.14788](https://arxiv.org/abs/2305.14788)
- **ICAE (Ge et al., 2023).** A LoRA encoder compresses context into memory slots read by the frozen decoder. It reports about 4× compression with an autoencoding plus LM pre-training objective **[M]** — [arXiv 2307.06945](https://arxiv.org/abs/2307.06945)
- **LLMLingua (Jiang et al., 2023).** A small LM's perplexity drives coarse-to-fine token dropping, giving up to 20× compression with small performance loss on GSM8K, BBH and others. LongLLMLingua and LLMLingua-2 extend it **[M]** — [arXiv 2310.05736](https://arxiv.org/abs/2310.05736)

### Inferences
- In Hope, byte to patch is already a 4.5× learned compression. Further soft compression of distant segments (summary patches written by the recurrent core) is essentially the RMT/AutoCompressor idea and fits the existing core.
- Exact-source recall for code should rely on Hope's external store (n-gram index plus vector retrieval), not on lossy compression.

### Gaps
- None of the compression numbers above were re-verified this session (all [M]).
- No source found on compression for byte-level models.

---

## 5. KV-cache eviction/selection: H2O, SnapKV, Quest (and quality trade-offs)

### Takeaway
Eviction (H2O, SnapKV) permanently drops tokens. It cuts memory 5×–8× but can lose information that becomes relevant later, which matters for multi-turn chat and code. Query-aware *selection* (Quest) keeps the full cache but reads only the top-k pages per step. That bounds per-token bandwidth (near-constant decode cost), while memory still grows unless the cache is offloaded.

### Cited Findings
- **H2O (Zhang et al., 2023).** Keeps "heavy-hitter" tokens (high accumulated attention) plus recent tokens, with a cache budget of about 20%. It reports up to 29× throughput versus DeepSpeed Zero-Inference / HF Accelerate and up to 1.9× lower latency **[M]** — [arXiv 2306.14048](https://arxiv.org/abs/2306.14048)
- **SnapKV (Li et al., 2024).** Fine-tuning-free. Per head, it uses an observation window at the end of the prompt to pick which prompt KV positions to keep. At 16K input it gives 3.6× faster generation and 8.2× memory efficiency, and it handles 380K tokens on one A100-80GB. It reports comparable accuracy on 16 long-sequence datasets and a negligible drop on NIAH **[V]** — [arXiv 2404.14469](https://arxiv.org/abs/2404.14469)
- **SnapKV limitation.** It compresses the *prompt* once, based on the question at its end. If a later query needs evicted content, it is gone (inference from mechanism; follow-up work has raised this point, not verified here) — [arXiv 2404.14469](https://arxiv.org/abs/2404.14469)
- **Quest (Tang et al., ICML 2024).** Stores per-page element-wise min/max of keys. Each query upper-bounds the page scores and attends only to the top-K pages. It reports up to 2.23× self-attention speedup and up to 7.03× end-to-end latency reduction, with negligible accuracy loss on long-dependency tasks **[V]** — [arXiv 2406.10774](https://arxiv.org/abs/2406.10774)

### Inferences
- Quest is the best conceptual match for Hope's **per-patch global cache**: page = a block of patches, metadata = min/max of keys, and selection is a matmul plus topk. All of these are plain PyTorch ops that run on XPU, though gains depend on gather efficiency since there is no fused kernel. Combined with sinks and a local window, per-token attention reads stay roughly fixed (sinks + window + K pages).
- Eviction is a poor default for an assistant that must reread code. Selection plus host-RAM offload of cold pages is safer.

### Gaps
- No XPU or Intel Arc benchmarks found for Quest or SnapKV, so the speedups are CUDA-kernel figures and may not transfer to pure-PyTorch gather/topk on Arc.
- The H2O numbers were not re-verified.

---

## 6. Retrieval as context extension: RETRO, kNN-LM, RAG vs long-context

### Takeaway
Retrieval is the only method whose inference cost is independent of corpus size. Evidence from 2023–2024 says long-context models beat RAG on average when compute is unconstrained, that the two are complementary, and that routing (Self-Route) gets near-LC quality at a fraction of the cost. On BABILong, RAG plateaus at about 60% on single-fact QA regardless of length.

### Cited Findings
- **RETRO (Borgeaud et al., DeepMind, 2022).** Chunked cross-attention to neighbours retrieved from a 2T-token database. It reports performance comparable to GPT-3 and Jurassic-1 on the Pile with 25× fewer parameters (7.5B) **[M]** — [arXiv 2112.04426](https://arxiv.org/abs/2112.04426)
- **kNN-LM (Khandelwal et al., 2020).** Interpolates the LM distribution with a kNN over cached (context-embedding → next token) pairs. It reached a then state-of-the-art 15.79 perplexity on WikiText-103 (−2.9) without training **[M]** — [arXiv 1911.00172](https://arxiv.org/abs/1911.00172)
- **Xu et al. (NVIDIA, 2023) "Retrieval meets Long Context LLMs".** A 4K-context Llama2-70B with simple retrieval matches a 16K-context fine-tuned model while using much less compute at generation. Retrieval also helps the 32K model, and Llama2-70B-32K+retrieval beats GPT-3.5-turbo-16k on 9 long-context tasks **[M]** — [arXiv 2310.03025](https://arxiv.org/abs/2310.03025)
- **Li et al. (Google DeepMind, EMNLP 2024 industry) "RAG or Long-Context LLMs?".** With sufficient resources, LC consistently beats RAG on average, but RAG's cost is "significantly lower". Self-Route lets the model decide per query whether RAG suffices, cutting compute substantially while matching LC performance **[V]** — [arXiv 2407.16833](https://arxiv.org/abs/2407.16833)
- **BABILong.** RAG reaches about 60% on single-fact QA, independent of length, and fails on multi-fact reasoning **[V]** — [arXiv 2406.10149](https://arxiv.org/abs/2406.10149)

### Inferences
- Hope already has a hybrid retrieval stack with exact source recall plus a gated cross-attention, i.e. a RETRO/Memorizing-Transformer path. For code (exact identifiers, file paths), exact-source retrieval is more reliable than any compressed memory. The weak spot is multi-hop or aggregation queries, which need either a working long window or iterative retrieval.
- A Self-Route-style policy (try retrieval plus the short window first, fall back to a longer window) keeps the average per-token cost low.

### Gaps
- The exact Self-Route cost-reduction percentages were not extracted (the abstract only). The paper reportedly gives about 39–65% token savings, unverified.

---

## 7. SSM/hybrid long-context behaviour (Mamba extrapolation, DeciMamba, LongMamba, Jamba) and hierarchical/byte-patch models

### Takeaway
SSMs give constant memory and constant per-token cost but **do not extrapolate well past their training length**. The failure comes from global channels that over-accumulate and from state-capacity limits. Training-free fixes (DeciMamba, LongMamba) help a few-fold. Hybrids with a few attention layers (Jamba) give the best practical trade-off, but RULER showed Jamba-base with only 4K effective length, so the hybrid must still be trained long. Byte-patch hierarchies cut the length the global model sees by about 4×–8×.

### Cited Findings
- **RULER on RWKV and Mamba.** Both degrade significantly at 8K and trail Llama2-7B by large margins even below 4K **[V]** — [arXiv 2404.06654](https://arxiv.org/html/2404.06654)
- **Titans S-NIAH.** Mamba2 passkey retrieval falls from 98.6 at 2K to 5.4 at 16K **[V]** — [arXiv 2501.00663](https://arxiv.org/html/2501.00663v1)
- **DeciMamba (Ben-Kish et al., 2024).** Training-free. It drops unimportant tokens between layers (based on Δ) to keep the effective receptive field within the training length, reporting extrapolation to about 25× the training length **[M]** — [arXiv 2406.14528](https://arxiv.org/abs/2406.14528)
- **LongMamba (Ye et al., ICLR 2025).** Splits hidden channels into local and global groups by cumulative decay over the training length. Global channels break when inputs exceed it, so LongMamba recalibrates their decay with a lookup table and filters unimportant tokens. It reports up to 4.8× over vanilla Mamba and 2.6× over DeciMamba on LongBench-E, with code at GATECH-EIC/LongMamba **[V]** — [arXiv 2504.16053](https://arxiv.org/abs/2504.16053)
- **"Stuffed Mamba" (Chen et al., 2024).** Identifies "state collapse" when the state overfits to training length. Fixes include training on longer sequences and forgetting or sliding-state adjustments **[M]** — [arXiv 2410.07145](https://arxiv.org/abs/2410.07145)
- **Jamba (Lieber et al., AI21, 2024).** Interleaves attention and Mamba 1:7 with MoE (52B total / 12B active) at 256K context. The KV cache at 256K is about 4 GB against 32 GB for Mixtral, and about 140K context fits on one 80 GB GPU **[M]** — [arXiv 2403.19887](https://arxiv.org/abs/2403.19887)
- **Jamba in RULER.** Jamba-base has an effective length of only 4K among base models **[V]** — [arXiv 2404.06654](https://arxiv.org/html/2404.06654). AI21's later Jamba-1.5 report claims a 256K effective length on RULER **[M]**, but this is vendor-reported — [arXiv 2408.12570](https://arxiv.org/abs/2408.12570)
- **Byte-patch hierarchies.** MEGABYTE (Yu et al., 2023) uses fixed patches with global and local models, reaching about 1M bytes **[M]** — [arXiv 2305.07185](https://arxiv.org/abs/2305.07185). The Byte Latent Transformer (Pagnoni et al., Meta, 2024) uses entropy-based dynamic patches (about 4.5–8 bytes) and matched Llama-3 FLOP-controlled scaling up to 8B **[M]** — [arXiv 2412.09871](https://arxiv.org/abs/2412.09871)

### Inferences
- Hope's ~3:1 Mamba:attention ratio is attention-heavier than Jamba's 1:7, which helps recall but means the attention KV still grows. At 220K patches (1 MB) the global KV would be large, so a window, selection or offload is still needed.
- The cheapest SSM-side fix is to train on long sequences with **state passing across segments** (truncated BPTT through the SSM state). This is plain PyTorch: carry the state, detach it every N segments. Add LongMamba-style decay recalibration as a training-free inference knob.
- Hope's per-patch global cache plus local byte decoder means long-range costs scale with the number of patches, not bytes. That is a 4.5× structural advantage over byte-level transformers.

### Gaps
- No published RULER/LongBench results found for byte-level patch models (BLT, MEGABYTE) beyond perplexity.
- I could not confirm the 2025 Gu-group paper on recurrent length generalization (state passing / TBPTT) within budget, so it is cited only as the general technique.

---

## 8. Which combination keeps per-token generation cost roughly constant, and fit for Hope HN-1

### Takeaway
Per-token cost stays roughly constant under this combination: (a) SSM/recurrent state (O(1)), (b) attention restricted to sinks plus a sliding window plus top-k selected pages (O(window + k·page)), (c) a fixed-size recurrent/associative memory updated per segment (RMT/ARMT/Titans-style), and (d) external retrieval triggered per chunk, not per token. Positional scaling (YaRN) is what lets the attention layers use positions beyond training. The literature supports each piece separately. Only Titans, Jamba and DuoAttention-style systems combine two or three of them at scale.

### Cited Findings
- Per-token decode cost in standard attention grows with cache size, which is why Quest bounds reads to top-K pages (up to 7.03× latency reduction) **[V]** — [arXiv 2406.10774](https://arxiv.org/abs/2406.10774)
- Constant-memory streaming via sinks plus a window is stable to 4M+ tokens **[V]** — [arXiv 2309.17453](https://arxiv.org/abs/2309.17453)
- Recurrent memory gives constant per-segment cost and is the best performer on BABILong up to 50M tokens **[V]** — [arXiv 2406.10149](https://arxiv.org/abs/2406.10149), [arXiv 2407.04841](https://arxiv.org/abs/2407.04841)
- Titans MAG/MAL use exactly "sliding-window attention + neural memory" as the design pattern **[V]** — [arXiv 2501.00663](https://arxiv.org/html/2501.00663v1)
- The hybrid SSM+attention KV-cache saving is 8× at 256K (Jamba) **[M]** — [arXiv 2403.19887](https://arxiv.org/abs/2403.19887)

### Inferences (fit assessment for Hope HN-1: 25M–2B params, Arc A770 16 GB, pure PyTorch XPU)

The table rates each technique, from most to least recommended.

| Technique | Maturity | Training cost for Hope | PyTorch complexity (XPU) | Fit |
|---|---|---|---|---|
| Attention sinks (1–4 sink patches, or a learnable sink) + existing SWA | Demonstrated | None; better if trained with a sink | Trivial | **High**: do now |
| YaRN on the RoPE attention layers (NTK-by-parts + temperature) | Demonstrated | A few hundred steps at the target length | Trivial (inv_freq + logit scale) | **High** |
| SSM state passing across segments (TBPTT) + long-sequence curriculum | Demonstrated (standard), extrapolation still fragile | Moderate; train at 8–32K patches | Low | **High**: needed, or Mamba layers collapse past training length |
| Quest-style top-k page selection over the per-patch global KV, with cold pages offloaded to host RAM | Demonstrated (CUDA) | None (training-free) | Medium (gather/topk; no fused kernel, so the speedup on Arc is uncertain) | **High** for long code |
| RMT/ARMT memory via the existing shared recurrent core + Hopfield zone | Demonstrated on BABILong (small models!) | Moderate; needs multi-segment BPTT and memory-forcing tasks | Medium | **High**: matches the architecture |
| Gated memory cross-attention onto external retrieval (RETRO / Memorizing-style), plus Self-Route-style policy | Demonstrated | Moderate; must force the gate open with retrieval-dependent tasks | Low–medium (already built) | **High** for exact code recall |
| Titans neural memory (test-time MLP updates) | Experimental (single group, ≤760M, no independent repro found) | High | Medium–high; the chunked parallel form is needed for speed | Medium: promising at Hope's scale but risky |
| LongMamba/DeciMamba decay recalibration | Experimental (training-free) | None | Low–medium | Medium: an inference knob, worth trying |
| SnapKV/H2O eviction | Demonstrated, but lossy | None | Low | Low–medium: harmful for re-reading code |
| Gist/ICAE/AutoCompressor soft compression | Demonstrated for prompts | High | Medium | Low: lossy for identifiers; the core already summarises |
| Infini-attention | Failed independent reproduction | High | Medium | **Low** |
| LongRoPE search | Demonstrated | Search + fine-tune | Medium | Low: YaRN suffices at Hope's targets |

- **Memory budget sketch** (inference, illustrative, not measured). Take one global attention layer with d=1024 in fp16. K+V is 2×1024×2 B = 4 KB per patch per layer, so 4 attention layers is about 16 KB per patch. 220K patches (1 MB of text) is then about 3.5 GB, which fits in 16 GB but is too slow to scan every token without selection. A 4K-patch window plus sinks is about 64 MB.
- **Main risk:** the closed-at-init memory gate and the Hopfield zone may never learn to carry information across segments, as in the Infini-attention reproduction (gates stuck). Curricula with cross-segment passkey, multi-key NIAH (RULER-style) and repo-level code reference tasks are required. Evaluate with RULER-style synthetic tasks at the byte or patch level, not only perplexity or passkey.

### Gaps
- No published measurements of any of these techniques on Intel Arc / PyTorch XPU, so speedups reported for CUDA kernels (Quest, SnapKV, FlashAttention-based) may not transfer.
- No benchmark results found for models in Hope's size class (25M–200M) on RULER beyond Titans (170–760M) and BABILong RMT fine-tunes (GPT-2 / small models).
- Newer 2025–2026 trainable sparse attention (for example DeepSeek NSA [arXiv 2502.11089](https://arxiv.org/abs/2502.11089), Moonshot MoBA [arXiv 2502.13189](https://arxiv.org/abs/2502.13189)) is relevant as a "Quest learned end-to-end" alternative but was not reviewed in depth here [M].
