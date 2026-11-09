# Inference Quantization Methods and Intel Arc A770 Tooling for Fast Small-Model Generation (2023–2026)

Notes compiled 2026-11-09. "Official" means a vendor or author claim. "Independent" means a third-party or community measurement. Where a number comes from memory of a paper rather than a page fetched in this session, it is marked "(recalled from paper; verify)".

## 1. BitNet b1.58 / BitNet b1.58 2B4T / bitnet.cpp: quality at small scale, speedups, training-from-scratch requirements

### Takeaway
Native ternary (W1.58A8) training works at 2B/4T tokens. BitNet b1.58 2B4T roughly matches Qwen2.5-1.5B on average score (54.2 vs 55.2) with 0.4 GB non-embedding memory, and it beats GPTQ/AWQ-int4 Qwen2.5-1.5B. The catch is that it has to be trained quantization-aware from scratch with a specific recipe. The fast kernels are CPU (bitnet.cpp) or a custom CUDA kernel. I found no Intel XPU/SYCL ternary kernel, so on the A770 a ternary Hope model would need a hand-written kernel to get any speed benefit.

### Cited Findings
- BitNet b1.58 2B4T arXiv 2504.12285: v1 posted 16 Apr 2025, v2 25 Apr 2025. It is a "native 1-bit" ~2B model trained on 4T tokens, with open-source GPU and CPU inference implementations. — [arXiv abstract](https://arxiv.org/abs/2504.12285)
- Table 1 of the 2B4T paper (instruct models), as non-emb memory / CPU TPOT / energy per token / avg score. Official (Microsoft) figures. — [arXiv HTML v2](https://arxiv.org/html/2504.12285v2)
  - BitNet b1.58 2B4T: 0.4 GB / 29 ms / 0.028 J / 54.19
  - LLaMA 3.2 1B: 2 GB / 48 ms / 0.258 J / 44.90
  - Gemma-3 1B: 1.4 GB / 41 ms / 0.186 J / 43.74
  - Qwen2.5 1.5B: 2.6 GB / 65 ms / 0.347 J / 55.23
  - SmolLM2 1.7B: 3.2 GB / 67 ms / 0.425 J / 48.70
  - MiniCPM 2B: 4.8 GB / 124 ms / 0.649 J / 42.05
- Table 2 compares against post-training-quantized Qwen2.5-1.5B (memory / MMLU / GSM8K / IFEval / avg). — [arXiv HTML v2](https://arxiv.org/html/2504.12285v2)
  - bf16: 2.6 GB / 60.25 / 56.79 / 50.12 / 55.72
  - GPTQ-int4: 0.7 GB / 58.06 / 50.57 / 47.84 / 52.15
  - AWQ-int4: 0.7 GB / 57.43 / 50.64 / 45.44 / 51.17
  - BitNet 2B4T: 0.4 GB / 53.17 / 58.38 / 53.48 / 55.01
  - Note that 4-bit PTQ cost Qwen2.5-1.5B about 3.6–4.5 avg points, which shows how fragile small models are under PTQ.
- 2B4T training recipe. Forward-pass weights are quantized with absmean to {-1,0,+1}, and activations with per-token absmax int8. Other features: SubLN, no biases, squared ReLU instead of SwiGLU, RoPE. It uses a two-stage LR schedule (high-peak cosine, then a mid-training drop to a lower peak on higher-quality data) and weight decay on a cosine to 0.1 in stage 1, then 0 in stage 2. SFT uses summed (not mean) CE loss, a larger LR and more epochs. DPO runs 2 epochs at LR 2e-7, beta 0.1. — [arXiv HTML v2](https://arxiv.org/html/2504.12285v2)
- GPU kernel. Standard libraries lack W1.58A8 GEMM, so the authors wrote a custom CUDA kernel. It packs 4 ternary weights per int8 and follows a pack-store-load-unpack-compute pattern. The authors say commodity GPUs are "not optimally designed" for 1-bit models. — [arXiv HTML v2](https://arxiv.org/html/2504.12285v2)
- bitnet.cpp paper (arXiv 2410.16144, Oct 2024) reports speedups over llama.cpp of 1.37x–5.07x on ARM and 2.37x–6.17x on x86 CPUs. Energy falls 55.4–70.0% on Apple M2 and 71.9–82.2% on i7-13700H, for models from 125M to 100B. Speedup grows with model size. Official. — [arXiv 2410.16144](https://arxiv.org/pdf/2410.16144)
- A January 2026 bitnet.cpp update reportedly added parallel kernels with configurable tiling and embedding quantization, for a further 1.15x–2.1x. Secondary source only, unverified. — [toolhalla blog](https://toolhalla.ai/blog/microsoft-bitnet-100b-llm-cpu-2026)
- Original BitNet b1.58 paper (Ma et al., arXiv 2402.17764, Feb 2024). It matches FP16 LLaMA perplexity and end-task scores from 3B upward. At 3.9B it is 2.4x faster and uses 3.32x less memory than LLaMA 3B. At 700M and 1.3B there is a small perplexity gap; I recall roughly 12.87 vs 12.33 at 700M and 11.29 vs 11.25 at 1.3B. (recalled from paper; verify) — [arXiv 2402.17764](https://arxiv.org/abs/2402.17764)

### Inferences
- For a 25M–200M model, the paper's own data suggests ternary costs some quality at sub-1B scale, so the case for BitNet in Hope is weaker at the low end. At these sizes the model already fits easily in VRAM, and the weights (≤0.4 GB in bf16) are not the main bottleneck (see section 4).
- Ternary QAT can be trained in pure PyTorch with straight-through estimators, since the recipe is published. Inference gains on the A770, though, require a custom SYCL/Triton kernel. No Intel GPU ternary kernel was found.

### Gaps
- No published BitNet results on Intel Arc GPUs (SYCL, OpenVINO, or Triton-XPU).
- No published BitNet b1.58 results below ~700M parameters, or for SSM/recurrent hybrids. Exact small-scale perplexities above are recalled, not re-fetched.

## 2. Weight-only 4/8-bit PTQ (GPTQ, AWQ, QuIP#) and W8A8 (SmoothQuant): results and Intel GPU backend support

### Takeaway
On Intel Arc, the practical path for PyTorch is torchao `Int4WeightOnlyConfig`, which uses oneDNN A16W4 GEMM on XPU. It accepts RTN, AWQ or GPTQ weights. Intel's measurements on a B580 show 1.56–2.14x decode speedup and about 65–70% VRAM savings for 3–4B models. llama.cpp SYCL has optimized Q4_0/Q4_K/Q5_K/Q6_K/Q8_0 kernels. W8A8 (SmoothQuant-style) is mainly a compute-bound/batching win and adds little for batch-1 decode.

### Cited Findings
- **GPTQ** (Frantar et al., arXiv 2210.17323, Oct 2022; ICLR 2023). One-shot second-order PTQ to 3–4 bits. It quantizes 175B models in about 4 GPU-hours, with end-to-end speedups of about 3.25x on A100 and 4.5x on A6000 vs FP16. (recalled from abstract; verify) — [arXiv 2210.17323](https://arxiv.org/abs/2210.17323)
- **AWQ** (Lin et al., arXiv 2306.00978; MLSys 2024 best paper). Activation-aware per-channel scaling protects about 1% of salient weights. Its TinyChat runtime reports more than 3x speedup over HF FP16 on desktop and mobile GPUs. (recalled from abstract; verify) — [arXiv 2306.00978](https://arxiv.org/abs/2306.00978)
- **QuIP#** (Tseng et al., arXiv 2402.04396, ICML 2024). Randomized-Hadamard incoherence processing plus E8 lattice codebooks targets ≤4-bit, with 2-bit as the strongest regime. Its CUDA kernels reach more than 50% of peak memory bandwidth on an RTX 4090. (recalled; verify) — [arXiv 2402.04396](https://arxiv.org/abs/2402.04396)
- **SmoothQuant** (Xiao et al., arXiv 2211.10438; ICML 2023). It migrates activation outliers into the weights via per-channel scaling to enable W8A8. Reported results are up to 1.56x speedup and 2x memory reduction with negligible accuracy loss. (recalled; verify) — [arXiv 2211.10438](https://arxiv.org/abs/2211.10438)
- Small-model PTQ cost is measured in the BitNet 2B4T paper (see section 1). GPTQ-int4 drops Qwen2.5-1.5B from 55.72 to 52.15 avg and AWQ-int4 to 51.17, with GSM8K the most affected (56.8 to 50.6). — [arXiv HTML v2](https://arxiv.org/html/2504.12285v2)
- **PyTorch 2.8 + torchao on Intel GPU** (official, PyTorch blog dated 2025-09-04, tested by Intel 2025-07-29 on Arc B580, Windows 11, PyTorch 2.8). — [PyTorch blog](https://pytorch.org/blog/pytorch-2-8torchao-unlock-efficient-llm-inference-on-intel-ai-pcs/)
  - WOQ-INT4 uses A16W4 GEMM via oneDNN. It "supports AWQ, RTN, and GPTQ algorithms" through HF Transformers.
  - Memory and decode results:
    - Llama-3.2-3B: 5.98 GB to 2.10 GB, 1.62x decode
    - Phi-3-mini: 7.11 GB to 2.13 GB, 2.14x decode
    - Qwen3-4B: 7.49 GB to 2.49 GB, 1.56x decode
  - Prefill "matches FP16" in the compute-bound stage.
- XPU int4 requires `int4_packing_format="plain_int32"` with newer torchao (`Int4WeightOnlyConfig(group_size=128, int4_packing_format="plain_int32")`). Transformers removed the string-based `TorchAoConfig("int4_weight_only")` API and now requires torchao ≥0.15 with AOBaseConfig objects. XPU wheels are at `download.pytorch.org/whl/xpu`. — [HF Transformers torchao docs](https://huggingface.co/docs/transformers/en/quantization/torchao); [torchao 0.16 docs](https://docs.pytorch.org/ao/0.16); [Diffusers PR #13530 mirror](https://git.huazai-net.workers.dev/huggingface/diffusers/pull/13530)
- IPEX offered INT4 weight-only quantization on Arc A-series from v2.1.30. IPEX is now EOL (see section 3). — [IPEX 2.1.30 release notes](https://intel.github.io/intel-extension-for-pytorch/xpu/2.1.30+xpu/tutorials/releases.html)
- IPEX-LLM (archived) supported sym_int4, q4_k, FP8 (e5m2/e4m3), FP6, FP4, INT8, INT2 and FP16 on Intel GPUs. — [ipex-llm GitHub](https://github.com/intel/ipex-llm)
- **llama.cpp SYCL quant kernels** (official doc). — [llama.cpp SYCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
  - 2025.2: Q4_0 MUL_MAT optimization, +21–87% on Intel GPUs including A770.
  - 2026.04–05: reorder-based mul_mat for Q4_K/Q5_K/Q6_K/Q8_0, plus fused MoE.
  - IQ types (IQ1_S…IQ4_XS) supported since 2024.4.
  - The DPAS (XMX) quantized-KV path is Xe2+ only, so it does not apply to the A770 (Xe-HPG).
- A770 hardware int8 throughput is 262 peak TOPS, so int8 XMX exists. — [Intel ARK A770 16GB](https://www.intel.com/content/www/us/en/products/sku/229151/specifications.html)

### Inferences
- At batch 1, decode on the A770 is bandwidth-bound for models ≥1B, so weight-only int4/int8 gives most of the gain. W8A8 mainly helps prefill or batched serving.
- For a 25M–200M bf16 model (≤0.4 GB of weights), WOQ will likely give little speedup. Per-token cost will be dominated by kernel launch and Python overhead rather than weight bytes. Fusion (torch.compile) and graph capture likely matter more than quantization at that size.
- For Hope's custom architecture, torchao int4 applies to `nn.Linear` layers only. SSM scans and recurrent cores stay bf16.

### Gaps
- No published GPTQ/AWQ/QuIP#/SmoothQuant measurements specifically on the A770 in PyTorch XPU. Intel's torchao numbers are on B580 (Xe2), not A770.
- No confirmation that torchao int4 XPU kernels are tuned for Xe-HPG (A-series), as opposed to Xe2.
- QuIP# has no Intel GPU kernels found.
- The GPTQ/AWQ/QuIP#/SmoothQuant headline numbers above are recalled from the abstracts, not re-fetched this session.

## 3. Intel Arc A770 specifics and software stack state (PyTorch XPU, IPEX, OpenVINO, llama.cpp SYCL vs Vulkan, pitfalls)

### Takeaway
As of late 2026 the supported stack is native PyTorch XPU (2.5+; torch.compile/Inductor via Triton-XPU; oneDNN SDPA default since 2.7; torchao int4 since 2.8), llama.cpp SYCL, and OpenVINO. IPEX hit EOL in March 2026 and IPEX-LLM was archived on 28 Jan 2026. For token generation, SYCL generally beats Vulkan, though A770-specific head-to-head data is sparse and inconsistent.

### Cited Findings
**Hardware**
- A770 16GB: 512 XMX engines, 32 Xe-cores, 262 peak INT8 TOPS (official), 560 GB/s memory bandwidth (8GB model: 512 GB/s). — [Intel ARK A770 16GB](https://www.intel.com/content/www/us/en/products/sku/229151/specifications.html); [Flopper spec compare](https://flopper.io/compare/intel-arc-a770-8gb-vs-intel-arc-a770-16gb/spec-sheet)
- Per XMX unit per clock: 128 FP16/BF16 ops, 256 INT8, 512 INT4. — [TechPowerUp review](https://techpowerup.com/review/intel-arc-a770/2.html)
- Tom's Hardware lists 138 TFLOPS FP16 and 275 TOPS INT8 (derived). Intel publishes no official FP16 number, and other sites disagree. — [Tom's Hardware](https://www.tomshardware.com/news/intel-reveals-specifications-for-arc-alchemist-desktop-gpus)

**PyTorch XPU (native)**
- PyTorch 2.5 (Oct 2024) added Intel GPU support (prototype) for Arc discrete, Core Ultra iGPU and Data Center Max. It includes runtime, ATen ops, oneDNN, and TorchInductor + Triton integration, and adds initial Windows support. — [PyTorch blog: Intel GPU support in 2.5](https://pytorch.org/blog/intel-gpu-support-pytorch-2-5/); [Phoronix](https://phoronix.com/news/PyTorch-2.5-Released)
- PyTorch 2.6 extended Windows binaries to Arc B-series; A-series was already covered. — [dev-discuss / PyTorch posts](https://dev-discuss.pytorch.org/t/intel-gpu-enabling-status-and-feature-plan/2389)
- oneDNN became the default XPU SDPA backend in 2.7, and GQA was added in 2.8. torch.compile works on Linux since 2.5 and on Windows since 2.7. — [PyTorch 2.8+torchao blog, 2025-09-04](https://pytorch.org/blog/pytorch-2-8torchao-unlock-efficient-llm-inference-on-intel-ai-pcs/)
- Official torch.compile speedups on Arc B580 (prefill / decode). — [same blog](https://pytorch.org/blog/pytorch-2-8torchao-unlock-efficient-llm-inference-on-intel-ai-pcs/)
  - Llama-3.2-3B: 2.42x / 1.58x
  - Phi-3-mini: 1.12x / 1.39x
  - Qwen3-4B: 2.68x / 1.99x
  - These come from fused RoPE/RMSNorm kernels and use static KV cache.
- PyTorch 2.10 + TorchAO is referenced for Core Ultra Series 3 and Arc B-series with Int4 weight-only quantization. — [PyTorch site](https://pytorch.org/?p=58602) (search-snippet only)

**Intel Extension for PyTorch (IPEX)**
- The installation site is labeled "(EOL)". It says the project reaches end of life by the end of March 2026, and users should move to native PyTorch because CPU/GPU support has been upstreamed. — [IPEX installation guide (EOL)](https://pytorch-extension.intel.com/installation)
- 2.8.10+xpu removed all overridden oneMKL/oneDNN ops except GEMM and SDPA. — [newreleases v2.8.10+xpu](https://newreleases.io/project/github/intel/intel-extension-for-pytorch/release/v2.8.10%2Bxpu)
- DeepSpeed dropped the IPEX dependency, stating XPU is supported by PyTorch 2.8+. — [DeepSpeed commit](https://app.semanticdiff.com/gh/microsoft/DeepSpeed/commit/d8e15da43f9745712d9e935b9266243e29da9f80)

**IPEX-LLM**
- Archived on 28 Jan 2026. Intel "will not provide or guarantee development of or support," and the repo notes "known security issues." The last news items were 2.2.0 (Apr 2025) with Ollama and llama.cpp portable zips, and FlashMoE (May 2025) running DeepSeek-671B / Qwen3-235B on 1–2 A770/B580. — [ipex-llm GitHub](https://github.com/intel/ipex-llm)
- Intel's official claim (March 2024 test, BigDL-LLM 2.5.0b20240303, IPEX 2.1.10+xpu): the A770 16GB does about 70 tok/s on Mistral 7B, 70% more than an RTX 4060 8GB, at int4, batch 1, 1024 in / 128 out. The per-model chart values are images only. — [Intel developer article](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerating-language-model-inference-on-your-pc.html); [Intel Gaming "Wield the power of LLMs"](https://game.intel.com/stories/wield-the-power-of-llms-on-intel-arc-gpus/)

**llama.cpp SYCL (official doc numbers, A770)**
- llama-2-7B Q4_0: 34 to 37 tok/s (2024.5), then 42 to 55 tok/s with the Q4_0 MUL_MAT optimization (2025.2).
- Arc B570 prefill went from 434 to 1331 t/s just by moving to oneAPI 2026.1 (2026.09), so oneAPI version matters a lot.
- Verified oneAPI versions: 2026.1, 2025.3.3, 2025.2.1, 2025.1, 2024.1.
- Nvidia/AMD support was removed from the SYCL backend in 2026.02.
- Source: [llama.cpp SYCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)

**Independent A770 llama.cpp data**
- Llama 7B Q4_0, SYCL tg128 = 42.10 t/s (Nov 2024, r/LocalLLaMA). — [r/LocalLLaMA via mirror](https://redlib.hbubli.cc/r/LocalLLaMA/comments/1ghvwsj/llamacpp_compute_and_memory_bandwidth_efficiency/lv2f4kb)
- Llama 3.1 8B Q4_K_M tg128 (Mar 2025): about 24 t/s on mainline SYCL vs about 49 t/s on the IPEX-LLM SYCL build. pp512 was about 214 t/s on Vulkan vs about 664 t/s on SYCL (A770 Mobile in one comment). The thread labels which GPU each row belongs to only ambiguously. — [r/LocalLLaMA A770 vs 9070XT (mirror)](https://redlib.astrial.org/r/LocalLLaMA/comments/1ji2grb/a770_vs_9070xt_benchmarks/mjbzfy2/?context=3)
- Qwen2.5 7B on one A770, llama.cpp path: 30.29 tok/s eval (33 ms/token), community report. — via [insiderllm guide](https://insiderllm.com/guides/intel-arc-local-ai/)
- One 2026 r/LocalLLaMA snapshot showed Vulkan tg higher than SYCL on Llama-2-7B Q4_K_M, contradicting the other reports, and a commenter said a fork's SYCL output was gibberish. — [reddit snapshot 2026-04-25](https://reddit.sentinel-team.org/posts/1st6lp6/snapshots/2026-04-25T00%3A46%3A56.728262Z)

**Arc Pro B70 data (Xe2, 608 GB/s, not an A770; useful for scale)**
- Tested 2026-04-21 with llama.cpp SYCL and oneAPI 2025.3.3. — [PMZFX B70 benchmarks](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks)
  - tg128: Qwen2.5-1.5B Q4_K_M 216.4 t/s; Llama-3.1-8B Q4_K_M 82.6 t/s; Qwen3.5-9B Q4_K_M 60.2 t/s.
  - SYCL vs Vulkan on Qwen 1.5B Q4_K_M: 229 vs 102 t/s for generation (2.2x). Vulkan prefill is "fine."
  - Upstream fixes tripled Q8_0 decode, and an assert-enabled build halved prefill.
- Level1Techs B70: SYCL 52.3 vs Vulkan 33.5 t/s tg on Qwen3.6-35B MoE. — [Level1Techs forum](https://forum.level1techs.com/t/intel-b70-launch-unboxed-and-tested/247873?page=8)

**OpenVINO GenAI**
- Intel's Llama 3 post claims the A770 beats an RTX 4060 8GB at INT4 weights / FP16 execution, with no absolute numbers. — [TweakTown summary](https://www.tweaktown.com/news/97705/metas-next-gen-llama3-llm-is-here-and-the-intel-arc-a770-outperforms-geforce-rtx-4060/)
- A community video reported about 21 tok/s for Qwen3-14B via OpenVINO on an A770 (Windows). Uncontrolled. — [recapio digest](https://recapio.com/digest/finally-an-intel-arc-a770-llm-benchmark-video-xmx-tensors-on-lmstudio-ollama-and-openvino-by-country)
- Phoronix/OpenBenchmarking OpenVINO GenAI results are CPU-only. — [OpenBenchmarking](https://openbenchmarking.org/performance/test/pts/openvino-genai/fc796755133c533ba23e5a07eb3662e01bb9e67e)

**Pitfalls**
- SYCL cannot report free memory on the "Arc770," so set `GGML_SYCL_GET_MEM_API=1`.
- First start is slow because of JIT, with no AOT.
- `SYCL_PROGRAM_COMPILE_OPTIONS=-cl-fp32-correctly-rounded-divide-sqrt` fixes precision issues.
- Linux users must be in the `video`/`render` groups.
- Prebuilt Windows binaries bundle the SYCL runtime.
- Source: [llama.cpp SYCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
- IPEX-LLM's 2.2.0 notes warn that mmap-based loading may leak data on shared hosts. — [ipex-llm GitHub](https://github.com/intel/ipex-llm)

### Inferences
- For Hope's custom PyTorch model, the realistic A770 path is native PyTorch XPU with bf16 (XMX), torch.compile, static buffers, and oneDNN SDPA, plus torchao int4 for large Linear layers if needed. Avoid new dependencies on IPEX or IPEX-LLM.
- Custom SSM scan kernels would have to be written in Triton (Triton-XPU via Inductor) or SYCL. CUDA-only kernels such as mamba-ssm's will not run.
- The llama.cpp numbers vary about 2x by build (mainline vs IPEX-LLM fork) and oneAPI version, so any target number must be re-benchmarked locally.

### Gaps
- No clean, current (2026) same-model A770 SYCL-vs-Vulkan head-to-head was found.
- No official OpenVINO GenAI tokens/s for the A770 (Intel's published GenAI perf pages focus on Core Ultra and Xeon).
- No authoritative source for ReBAR impact on LLM decode; it is widely stated as required for Arc, but no measurement was found. No reliable Windows-vs-Linux measurement either (only an unverified "3x" forum claim).
- torch.compile/torchao figures are on B580 (Xe2). A770-specific published numbers for native PyTorch XPU were not found.

## 4. Expected tokens/s for a 200M–2B model on the A770 (bandwidth estimate vs measurements)

### Takeaway
The bandwidth ceiling at batch 1 is about 560 GB/s ÷ bytes of weights read per token. Measured llama.cpp SYCL on Arc reaches roughly 35–40% of peak bandwidth in decode. That suggests about 100–150 tok/s for a 2B bf16 model, 300–450 for 2B int4, and about 250–500 for 1–1.5B int4 (the B70 measures 216–229 tok/s on Qwen2.5-1.5B Q4_K_M). Below about 300M parameters, framework and launch overhead, not bandwidth, will cap throughput; in eager PyTorch this is likely in the low hundreds of tok/s regardless of size.

### Cited Findings
- A770 bandwidth: 560 GB/s. — [Intel ARK](https://www.intel.com/content/www/us/en/products/sku/229151/specifications.html)
- llama-2-7B Q4_0 (about 3.56 GiB) reaches 55 tok/s on A770 SYCL (official doc, 2025.2). That is about 210 GB/s effective, roughly 37% of peak. — [llama.cpp SYCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md) (bandwidth arithmetic is mine)
- Llama 7B Q4_0 SYCL measured independently at 42.1 t/s on A770 (about 160 GB/s, 29%). — [r/LocalLLaMA mirror](https://redlib.hbubli.cc/r/LocalLLaMA/comments/1ghvwsj/llamacpp_compute_and_memory_bandwidth_efficiency/lv2f4kb)
- Small-model data point on a similar-bandwidth Arc (B70, 608 GB/s): Qwen2.5-1.5B Q4_K_M tg128 at 216–229 t/s and pp512 at 8,048 t/s. — [PMZFX B70](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks)
- Intel claims that WOQ-INT4 gives more than 1.5x decode over FP16 for 3–4B models, and torch.compile gives 1.4–2x decode from fusing RoPE/RMSNorm. This confirms that eager per-op overhead is significant on XPU. — [PyTorch blog](https://pytorch.org/blog/pytorch-2-8torchao-unlock-efficient-llm-inference-on-intel-ai-pcs/)

### Inferences
- Theoretical batch-1 ceilings at 560 GB/s, weights only (my arithmetic):
  - 200M bf16 (0.4 GB): about 1,400 tok/s
  - 1B bf16 (2 GB): about 280
  - 2B bf16 (4 GB): about 140
  - 2B int4 (about 1.1 GB with scales): about 500
  - 2B ternary packed (about 0.5 GB): about 1,100 (needs a custom kernel)
- At the observed 30–40% efficiency, realistic figures are about 40–55 tok/s for 2B bf16 and about 150–200 for 2B int4. Optimized kernels at about 60% efficiency could give about 85 and about 300.
- The 1.5B Q4 measurement on the B70 (229 t/s for about 1 GB of weights) suggests small models achieve similar or lower bandwidth efficiency. Fixed per-token overhead becomes the limit.
- At 25M–200M, a custom PyTorch model with many small heterogeneous ops (SSM scan, recurrence, specialist routing) will likely be launch-bound. Expect a few hundred kernel launches × roughly 5–20 µs each, i.e. about 1–5 ms per token, or about 200–1000 tok/s at best. Batching or speculative decoding, torch.compile with static shapes, and fewer graph breaks will matter more than quantization.
- A byte-level model needs about 3–4x more decode steps than a BPE model for the same text. So 300 tok/s (bytes) is roughly equivalent to 80–100 BPE tok/s in user-visible speed.

### Gaps
- No published measurements of 200M–2B custom (non-llama.cpp) PyTorch models on A770 XPU.
- No published A770 per-kernel launch-latency figures for PyTorch XPU, and no XPU CUDA-graph-equivalent status was found in this session.

## 5. Running a Qwen teacher (7–8B, 4-bit) on the same A770: published tokens/s

### Takeaway
Expect about 30–55 tok/s decode for a 7–8B Q4 model on the A770 with llama.cpp SYCL. Older IPEX-LLM builds reached about 49–70 tok/s but are now archived. The 4-bit 7–8B weights take about 4.5–5 GB, which leaves about 10 GB of the 16 GB for the Hope student and KV caches. Running both at once halves effective bandwidth for each.

### Cited Findings
- Qwen2.5 7B on one A770: 30.29 tok/s eval (llama.cpp path, community). — via [insiderllm guide](https://insiderllm.com/guides/intel-arc-local-ai/)
- Llama 3.1 8B Q4_K_M tg128: about 24 t/s mainline SYCL vs about 49 t/s IPEX-LLM SYCL (Mar 2025). — [r/LocalLLaMA mirror](https://redlib.astrial.org/r/LocalLLaMA/comments/1ji2grb/a770_vs_9070xt_benchmarks/mjbzfy2/?context=3)
- llama-2-7B Q4_0: 55 t/s on mainline SYCL after the 2025.2 optimization (official). — [llama.cpp SYCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
- Mistral 7B int4: about 70 tok/s with IPEX-LLM/BigDL (Intel official, March 2024). — [Intel Gaming](https://game.intel.com/stories/wield-the-power-of-llms-on-intel-arc-gpus/)
- Willitrunai estimates about 64 tok/s for Qwen2.5-7B Q4_K_M on the A770. This is a model estimate, not a measurement. — [willitrunai](https://willitrunai.com/can-run/qwen-2.5-7b-on-arc-a770-16gb)
- Qwen3-14B via OpenVINO: about 21 tok/s (community video, Windows). — [recapio digest](https://recapio.com/digest/finally-an-intel-arc-a770-llm-benchmark-video-xmx-tensors-on-lmstudio-ollama-and-openvino-by-country)
- Ollama on Arc: IPEX-LLM's Ollama portable zip is archived, so guides now point to llama.cpp, Ollama (Vulkan) or OpenVINO. — [insiderllm guide](https://insiderllm.com/guides/intel-arc-local-ai/); [ipex-llm GitHub](https://github.com/intel/ipex-llm)

### Inferences
- For distillation, the teacher's bottleneck is prefill and logits over training data rather than decode. SYCL prefill for 8B Q4 is in the hundreds to low thousands of tok/s (about 664 pp512 on A770 in the March 2025 thread; 2,452 on B70). Caching teacher logits/top-k offline is far cheaper than running the teacher online alongside student training on the same card.
- Byte-level student vs BPE teacher means logit distillation needs tokenizer alignment (out of this note's scope). Sequence-level distillation from generated text avoids it.

### Gaps
- No published A770 figures for Qwen3-8B or Qwen2.5-7B with OpenVINO GenAI INT4, or with current (2026) llama.cpp SYCL builds after the Q4_K reorder optimization.
- No data on co-running a teacher and a training job on one A770.
