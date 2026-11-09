# Adaptive computation, recurrent-depth / latent reasoning, efficient MoE and diffusion LMs: speed and reasoning per compute (2023–2026), with fit for Hope HN-1

Scope note: numbers below come from arXiv abstracts or full-text HTML fetched in this session (Nov 2026), unless a line says "[not re-verified this session]". Those lines rest on the cited paper as I remember it and should be checked before anyone relies on them. "Demonstrated" means the authors reported measured results. "Experimental" means the result is a claim, a small-scale demo, or not independently replicated.

## 1. Mixture-of-Depths, early exit (CALM, LayerSkip), ACT / PonderNet: real vs theoretical speedups, batching issues

### Takeaway
Token-level adaptive depth gives real wall-clock gains in only two cases. The first is a static compute budget (MoD top-k capacity, so tensor shapes stay fixed). The second is an exit decision paired with self-speculative verification (LayerSkip). Per-token, per-sequence dynamic exits (CALM, ACT) save FLOPs on paper, but much of that saving is lost under batching and KV-cache bookkeeping. LayerSkip's measured gains are about 1.8–2.2x, and MoD reports up to about 50% faster sampling steps.

### Cited Findings
- **Mixture-of-Depths (Raposo et al., 2024).** Each block caps the number of tokens (k) that go through attention and MLP, and selects them by top-k routing. Because k is fixed in advance, "the computation graph is static with known tensor sizes" while which tokens get compute varies. MoD models match baseline quality at equivalent FLOPs and wall-clock training time, need "a fraction of the FLOPs per forward pass," and can be "upwards of 50% faster to step during post-training sampling." — [Raposo et al. 2024, arXiv:2404.02258](https://arxiv.org/abs/2404.02258)
- MoD's top-k over the sequence is non-causal (it depends on future tokens). For autoregressive sampling the paper adds an auxiliary predictor or auxiliary loss that learns the router decision causally [not re-verified this session; the abstract does not mention it]. — [arXiv:2404.02258](https://arxiv.org/abs/2404.02258)
- **CALM (Schuster et al., 2022).** Confident Adaptive Language Modeling does per-token early exit in decoding and reports compute reductions "up to ×3" while "provably maintaining high performance" on three text-generation tasks. The authors list the open problems as choosing a confidence measure, linking sequence-level constraints to per-token exits, and handling the missing hidden states (KV) of skipped layers for later tokens. — [Schuster et al. 2022, arXiv:2207.07061](https://arxiv.org/abs/2207.07061)
- **LayerSkip (Elhoushi et al., 2024, Meta).** Training uses layer dropout (low rates for early layers, higher for late ones) plus an early-exit loss in which all layers share one LM head. Inference uses self-speculative decoding: early layers draft tokens and the remaining layers verify them, reusing compute and activations. Measured speedups: up to 2.16x on CNN/DM summarization, 1.82x on coding, and 2.0x on TOPv2 semantic parsing. Tested on Llama models in pretraining from scratch, continual pretraining, and fine-tuning. — [Elhoushi et al. 2024, arXiv:2404.16710](https://arxiv.org/abs/2404.16710)
- **PonderNet (Banino et al., 2021).** It "learns to adapt the amount of computation based on the complexity of the problem," "dramatically improves performance over previous adaptive computation methods" on a synthetic task, succeeds at extrapolation tests where standard networks fail, and matches state-of-the-art QA results with less compute. The abstract does not name the tasks. The full paper uses parity extrapolation and bAbI, and replaces ACT's ponder cost with a probabilistic halting distribution plus a KL term to a geometric prior [not re-verified this session]. — [Banino et al. 2021, arXiv:2107.05407](https://arxiv.org/abs/2107.05407)
- **ACT in recent recursive models.** ARC Prize ablated HRM and found that adaptive halting (ACT) reduced the number of refinement steps, with accuracy "within a few points" of a fixed 16-loop run. ACT saves compute but does not drive accuracy. — [ARC Prize, HRM analysis, 2025](https://arcprize.org/blog/hrm-analysis)
- **TRM simplified ACT.** It dropped HRM's Q-learning "continue" target, which needed a second forward pass, and kept only a learned halting probability. Sudoku-Extreme accuracy was 86.1% with the extra pass and 87.4% without it, so removing the extra pass cost nothing. — [Jolicoeur-Martineau 2025, arXiv:2510.04871](https://arxiv.org/html/2510.04871)
- **Recurrent-depth adaptive exit (Huginn).** The model stops iterating when the KL divergence between successive steps falls below 5×10⁻⁴. Harder content exits later: about 3.5 more steps on average for moral-scenario questions than for high-school math. MT-Bench changed only from 5.63 to 5.56 with early exit. The paper gives no aggregate wall-clock saving. — [Geiping et al. 2025, arXiv:2502.05171](https://arxiv.org/html/2502.05171)

### Inferences
- For Hope at batch size 1 on an A770, per-token or per-sequence halting is feasible: the batching penalty mostly disappears at batch 1, and what remains is the Python and XPU kernel-launch overhead of each extra loop iteration. In batched training, a fixed-capacity scheme (MoD style, or fixed iterations plus a stochastic depth sample) is much more XPU-friendly than ragged halting.
- Hope's observed collapse to one iteration fits the ARC Prize and TRM evidence that ACT is a compute saver, not an accuracy driver. A pragmatic fix is to train with a sampled number of iterations (Huginn's log-normal Poisson, or fixed deep supervision as in TRM) and to apply halting only at inference, using a KL or convergence threshold that needs no training signal (Huginn). That removes the incentive the ponder cost creates to collapse early.
- LayerSkip-style self-speculation transfers naturally to a shared-weight recurrent core: run few iterations to draft, then more iterations to verify. Huginn proposes exactly this but reports no numbers (see section 2).

### Gaps
- I found no measured wall-clock numbers for MoD at inference with the causal predictor beyond the "upwards of 50% faster to step" claim, and no independent replication at small scale.
- I found no published benchmark of any of these methods on Intel XPU.
- CALM's actual wall-clock speedup (as opposed to the FLOP reduction) was not retrieved.

## 2. Recurrent depth and latent reasoning: Huginn (Geiping 2025), Universal Transformer revival, HRM, TRM, Coconut

### Takeaway
Recurrent depth is the best-evidenced route to "more reasoning per parameter." Huginn (3.5B) gains steeply with iterations on GSM8K and ARC-C. TRM (7M parameters, 2 layers) beats much larger LLMs on ARC-AGI and Sudoku-style puzzles, though only in task-specific, heavily augmented supervised settings. Training stability is the main risk, and the papers document the specific fixes: sandwich norm, low LR, EMA, truncated backprop, sampled depth. Coconut-style continuous thought is less mature: it underperforms CoT on GSM8K and is expensive to train.

### Cited Findings
- **Huginn / recurrent depth (Geiping et al., 2025).** 3.5B parameters, trained on 800B tokens. Layout: prelude of 2 layers, a recurrent core of 4 layers iterated r times, coda of 2 layers. At test time it can use compute equivalent to a ~50B-parameter model, and it needs no specialized CoT data. — [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)
  - ARC-Challenge rises from 24.06 at r=1 to 27.99 at r=4, 35.15 at r=8, 37.71 at r=16 and 38.23 at r=32. GSM8K CoT is 0.00 at r=1 and reaches 34.80/42.08 (strict/flexible) at r=32, and 47.23 flexible with EMA weights at r=64. At 180B tokens, the recurrent model scored 9.02 on GSM8K versus 1.82 for a fixed-depth baseline (about 5x). HellaSwag saturates at about 8 iterations, while reasoning tasks keep improving. — [arXiv HTML 2502.05171](https://arxiv.org/html/2502.05171)
  - Training: the recurrence count is sampled from a log-normal Poisson distribution (mean 32), with truncated backprop through the last 8 iterations. — [same](https://arxiv.org/html/2502.05171)
  - **Instabilities.** Bad run 1 used a parameter-free RMSNorm, no embedding scale and LR 4e-4. Hidden-state token correlation rose toward 1.0, a representation collapse. Bad run 2 recovered, but validation perplexity was identical at 1 and 32 iterations, meaning the model ignored the recurrence. The final run used a sandwich-norm layout and peak LR 4e-5. The paper also lists 5e-4 elsewhere, an internal inconsistency. — [same](https://arxiv.org/html/2502.05171)
  - Inference features: zero-shot per-token adaptive exit (KL threshold), KV-cache sharing with a fixed budget (MT-Bench held at 5.86), continuous-CoT warm start (the previous token's state saves 1–2 steps), and self-speculative decoding, which is described but not measured. — [same](https://arxiv.org/html/2502.05171)
  - Training compute was large: 4096 GPUs on Frontier. — [same](https://arxiv.org/html/2502.05171)
- **HRM (Wang et al., 2025).** 27M parameters, about 1,000 training examples, no pretraining and no CoT. Two coupled recurrent modules (slow planner, fast worker). The authors report "nearly perfect" Sudoku and maze results and beat larger LLMs on ARC. — [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)
  - **ARC Prize independent check.** Verified semi-private scores were 32% on ARC-AGI-1 (paper claims 41% on public eval) and 2% on ARC-AGI-2. A plain transformer of similar size came within about 5 points. The outer refinement loop is the main driver: going from 1 to 2 loops added about 13 points. ACT was roughly equal to fixed loops. Training on only the evaluation tasks still gave 31%, so cross-task transfer is limited. Augmentation is critical. The `puzzle_id` embedding means the model handles only puzzles it saw in training. — [ARC Prize 2025](https://arcprize.org/blog/hrm-analysis)
- **TRM (Jolicoeur-Martineau, 2025, Samsung SAIL Montréal).** One 2-layer network, about 7M parameters. Results versus HRM's 27M: ARC-AGI-1 44.6% vs 40.3%, ARC-AGI-2 7.8% vs 5.0%, Sudoku-Extreme 87.4% vs 55.0%, Maze-Hard 85.3% vs 74.5%. — [arXiv:2510.04871](https://arxiv.org/html/2510.04871)
  - Key ingredients: deep supervision over up to 16 steps, carrying the latent state with detached gradients. ARC Prize's analysis, cited by TRM, found deep supervision took accuracy from 19% to 39%, while recursion added only 35.7% to 39.0%. EMA of 0.999 lifted Sudoku from 79.9% to 87.4% by preventing collapse. Four layers overfit (79.5%) compared with two (87.4%). An MLP beat attention on 9x9 Sudoku, but attention was better on 30x30 grids. Best recursion setting: T=3, n=6 (42 effective recursions). — [same](https://arxiv.org/html/2510.04871)
  - Compute: Sudoku trained in under 36 hours on 1 L40S; ARC took about 3 days on 4 H100s. — [same](https://arxiv.org/html/2510.04871)
- **Coconut (Hao et al., 2024, Meta).** The last hidden state is fed back as the next input embedding ("continuous thought"), which can encode several next steps at once, a form of breadth-first search. Results on GPT-2:

  | Benchmark | Coconut accuracy (tokens) | CoT accuracy (tokens) | No-CoT accuracy |
  |---|---|---|---|
  | GSM8K | 34.1% (8.2) | 42.9% (25.0) | 16.5% |
  | ProsQA | 97.0% | 77.5% | — |
  | ProntoQA | 99.8% | 98.8% | — |

  Training follows a multi-stage curriculum that progressively replaces CoT steps. Training with n latent thoughts needs n+1 sequential forward passes, which "poses challenges for parallelism." — [arXiv HTML 2412.06769](https://arxiv.org/html/2412.06769)
- **Universal Transformers.** The original shared-weight, ACT-halted transformer (Dehghani et al., 2018) is the conceptual ancestor of Huginn, HRM and TRM [background, not re-fetched]. — [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)

### Inferences
- Hope's architecture (shared-weight core plus ACT) sits squarely in the Huginn/TRM design space, and these papers supply the stability recipe:
  - sandwich norm and a low LR;
  - monitoring hidden-state token correlation as a collapse detector;
  - checking that loss differs between 1 and N iterations, so the model actually uses the recurrence;
  - EMA weights;
  - truncated BPTT (k≈8);
  - sampled iteration counts;
  - deep supervision with detached carried state.
- At 25M–200M parameters, the evidence that recurrence beats parameters is strongest for structured, verifiable puzzle tasks (TRM, HRM), not for open-ended language. Huginn's language-reasoning gains are shown at 3.5B and 800B tokens, so transfer to small byte-level LMs is unproven.
- TRM's and ARC Prize's evidence that deep supervision and iterative refinement of the answer matter more than hierarchy maps well onto Hope's critic/retry loop. Feeding the previous answer and latent back into the core is the mechanism with measured gains.
- Coconut is a weak bet for Hope: it loses to CoT on GSM8K, training is sequential and costly, and it needs CoT-annotated data for the curriculum.

### Gaps
- I found no independent replication of Huginn at under 1B parameters, and no measured wall-clock speedup for its self-speculative decoding.
- I found no result for TRM or HRM on natural-language reasoning; both are grid-puzzle results.

## 3. Efficient MoE for small models: fine-grained and shared experts, auxiliary-loss-free balancing, expert choice; ≤2B on one consumer GPU

### Takeaway
MoE improvements are demonstrated at 2B total and above: DeepSeekMoE 2B matches GShard 2.9B, and OLMoE-1B-7B beats dense models with similar active parameters. Auxiliary-loss-free bias balancing beats an auxiliary loss up to 3B. On a single 16GB consumer GPU, MoE saves FLOPs but not memory. Without fused or grouped-GEMM kernels on XPU, the dispatch overhead of many small experts likely erases the speed benefit at 25M–200M parameters.

### Cited Findings
- **DeepSeekMoE (Dai et al., 2024).** It splits experts into finer segments and activates more of them (fine-grained), and adds always-on shared experts. DeepSeekMoE 2B matches GShard 2.9B, which has 1.5x the expert parameters and compute, and "nearly approaches" a dense model with the same total parameters. DeepSeekMoE 16B is comparable to LLaMA2 7B at about 40% of the compute. — [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)
- **Auxiliary-loss-free balancing (Wang et al., 2024, DeepSeek).** A per-expert bias is added to routing scores before top-K and adjusted from each expert's recent load. Tested on MoEs up to 3B parameters and 200B tokens, it gave better performance and better balance than an auxiliary loss, which the authors attribute to removing interfering gradients. DeepSeek-V3 adopts it. — [arXiv:2408.15664](https://arxiv.org/abs/2408.15664)
- **OLMoE-1B-7B (Muennighoff et al., 2024).** 7B total and about 1B active parameters, trained on 5T tokens. It beats models with similar active parameters and even Llama2-13B-Chat and DeepSeekMoE-16B, with high expert specialization. Weights, data, code and logs are open. — [arXiv:2409.02060](https://arxiv.org/abs/2409.02060)
- **Expert-choice routing (Zhou et al., 2022).** Experts select their top-k tokens, which gives perfect load balance. The selection is over a batch or sequence and is therefore non-causal for autoregressive decoding, the same issue as MoD [not re-verified this session]. — [arXiv:2202.09368](https://arxiv.org/abs/2202.09368)

### Inferences
- For Hope's top-2 router with a balance loss, the lowest-risk upgrades are: (a) replace the auxiliary balance loss with DeepSeek-style bias balancing, a few lines of code with no kernel needed; (b) add one shared, always-on expert or zone so that common byte and patch processing is not duplicated across specialists.
- Expert-choice routing fits poorly with autoregressive byte or patch generation, because it is non-causal. Keep token-choice routing.
- At ≤200M parameters on an A770 without grouped-GEMM kernels, a loop over experts in PyTorch costs extra kernel launches per layer. Expect MoE to be slower in wall clock than a dense model with the same active parameters. The benefit is quality per FLOP, which matters for training-budget-limited runs, not inference latency. This is inference, not a measured result.

### Gaps
- I found no published MoE benchmarks below about 1B total parameters with wall-clock numbers, and none on Intel XPU.
- I found no direct study of "heterogeneous specialist zones" (experts with different architectures) at small scale.

## 4. Diffusion language models: LLaDA, Mercury, Gemini Diffusion, Dream 7B, block diffusion, SEDD: throughput, quality gaps, maturity, small byte-level feasibility

### Takeaway
Commercial diffusion LMs show roughly 700–1,500 tokens/s on H100-class hardware (Mercury Coder, Gemini Diffusion), up to about 10x faster than speed-optimized AR models, with competitive coding quality. These are vendor claims, partly checked by third parties. Open dLLMs (LLaDA, Dream) are slower than AR in vanilla form, by about 13.7x versus LLaMA3-8B in one analysis. They become competitive only with approximate KV caching and confidence-thresholded parallel decoding (Fast-dLLM, up to 27.6x over vanilla LLaDA). Block diffusion is the most practical hybrid. For a small from-scratch byte model, diffusion is feasible to train (SEDD beats GPT-2 at small scale) and has data-efficiency and planning advantages, but the inference speed advantage depends on engineering that is immature on XPU.

### Cited Findings
- **SEDD (Lou et al., 2024, ICML best paper).** Score-entropy discrete diffusion beats prior language diffusion by 25–75% in perplexity and beats GPT-2 at comparable sizes. Its unannealed generative perplexity is 6–8x better than unannealed GPT-2. It matches quality with 32x fewer network evaluations and supports infilling. — [arXiv:2310.16834](https://arxiv.org/abs/2310.16834)
- **LLaDA (Nie et al., 2025).** An 8B masked-diffusion model trained from scratch with pretraining and SFT. It is "competitive with LLaMA3 8B in in-context learning," performs comparably to the authors' own AR baselines, addresses the reversal curse, and beats GPT-4o on reversal poem completion. — [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
- **Open dLLM speed reality.** Fast-dLLM states that open dLLMs "often lag behind autoregressive models" in practical speed, because they have no KV cache and lose quality when decoding tokens in parallel. Its block-wise approximate KV cache plus confidence-thresholded parallel decoding gives up to 27.6x speedup over vanilla LLaDA at generation length 1024, of which the cache alone gives about 2–3.6x. — [arXiv:2505.22618](https://arxiv.org/abs/2505.22618), [PDF v2](https://arxiv.org/pdf/2505.22618v2)
- **Independent efficiency study.** Without a KV cache, dLLMs re-encode the whole sequence at every denoising step. Parallel decoding can surpass AR at moderate lengths with short prompts. One cited analysis reports LLaMA3-8B-Instruct throughput 13.7x higher than LLaDA-8B-Instruct. I saw these figures only through a search summary, so they need confirmation in the paper. — ["How Efficient Are Diffusion Language Models?", arXiv:2510.18480](https://arxiv.org/html/2510.18480v3)
- **Mercury (Inception Labs, 2025).** Mercury Coder Mini runs at 1,109 tok/s and Small at 737 tok/s on H100, measured independently by Artificial Analysis. The paper claims up to 10x faster on average than speed-optimized frontier models at comparable quality. In Copilot Arena it ranked 2nd on quality and fastest overall. — [arXiv:2506.17298](https://arxiv.org/abs/2506.17298)
- **Gemini Diffusion (Google DeepMind, 2025).** Official figure: 1,479 tok/s excluding overhead, with 0.84 s overhead, averaged across evals. — [DeepMind model page](https://deepmind.google/models/gemini-diffusion/). Hands-on tests measured 600–1,300 tok/s (VentureBeat) and about 727–857 tok/s in other reports. — [VentureBeat](https://venturebeat.com/ai/beyond-gpt-architecture-why-googles-diffusion-approach-could-reshape-llm-deployment/), [Gigazine](https://gigazine.net/gsc_news/en/20250522-google-gemini-diffusion). The model is experimental, behind a waitlist, and the architecture is not disclosed.
- **Dream 7B (HKU and Huawei Noah's Ark, 2025).** Initialized from an AR model, with strong planning results versus Qwen2.5-7B (authors' own evaluation): Sudoku 81.0 vs 21.0, Countdown 16.0 vs 6.2, trip planning 17.8 vs 3.6. Weights are open. — [arXiv:2508.15487](https://arxiv.org/pdf/2508.15487), [HKU blog](https://hkunlp.github.io/blog/2025/dream)
- **Block diffusion / BD3-LM (Arriola et al., 2025, ICLR).** Interpolates between AR and diffusion: autoregressive over blocks, diffusion within each block. It supports KV caching of finished blocks, parallel sampling inside a block, and arbitrary-length generation, and reports state-of-the-art perplexity among diffusion LMs. — [arXiv:2503.09573](https://arxiv.org/abs/2503.09573)
- **Data-constrained regime (Prabhudesai et al., 2025).** Masked diffusion beats AR when data is scarce and compute is plentiful (many epochs). Random-order masking acts as implicit data augmentation. The paper derives a closed-form critical compute threshold. — [arXiv:2507.15857](https://arxiv.org/abs/2507.15857)

### Inferences
- **Maturity.** AR remains the safe default. Diffusion is production-grade only in closed systems (Mercury, which is commercial; Gemini Diffusion, which is experimental). Open models are research grade, and their speed depends on approximate caches and threshold heuristics.
- **Fit for Hope (byte-level, patches, 25M–200M, A770).** The cheapest experiment is block diffusion inside a patch: keep the AR patch-level backbone and its KV cache, and denoise the bytes of the next patch (or next k patches) in parallel with a masked-diffusion objective. This matches BD3-LM's structure. Byte vocabularies (256) make the per-position softmax cheap. Hope's recurrent core could also serve as the denoiser, so that iterations correspond to denoising steps, but no paper tests this combination. It is speculative.
- Diffusion's planning and constraint advantage (Dream's Sudoku result) and its data efficiency under repeated epochs (Prabhudesai) both suit a small model trained on limited data. Its speed advantage only materializes with KV caching and few denoising steps, both of which need custom engineering on XPU.

### Gaps
- I found no published byte-level discrete diffusion LM at 25M–200M with throughput numbers.
- I found no dLLM benchmarks on Intel GPUs.
- Gemini Diffusion's architecture and parameter count are undisclosed.
- I did not retrieve LLaDA's exact benchmark scores (MMLU, GSM8K) or an independent check of Dream's planning numbers.
