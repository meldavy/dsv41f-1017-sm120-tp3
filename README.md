# DeepSeek-V4.1-Flash-1017 on 3× RTX PRO 6000 (TP3)

A recipe for serving [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
at tensor parallel 3, with the full 1M-token context, on three 96 GB Blackwell workstation cards.

**What it is:** DeepSeek's own FP8/MXFP4 weights for everything except the routed experts of the 40 decoder layers.
Those come from [LibertAIDAI's EXL3 3.5 bpw pack](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw)
and run on [ExLlamaV3](https://github.com/turboderp-org/exllamav3) kernels, inside the
[local-inference-lab](https://github.com/local-inference-lab) vLLM fork ("karmic-kraken", KK). The whole model fits
in VRAM, and the Engram n-gram tables live in host RAM.

**Result** (one stream, measured on the machine below):

| | |
|---|---|
| Prefill | ~7,000 tok/s at 10k-token prompts, ~7,600 at 32k, 5,400 at 736k |
| Decode | ~91.5 tok/s (no speculative decoding) |
| Context / KV | 1,048,576-token window, 3.14M-token KV pool (3 concurrent 1M sessions) |

The patches and scripts are in this repository (`patches/`, `scripts/`) and also reproduced in the appendices.

Written 2026-10-17. Everything below was measured on one machine; see *Caveats* for what was not tested.

## Contents

1. [Hardware](#hardware)
2. [Why a hybrid checkpoint](#why-a-hybrid-checkpoint)
3. [Step 1: get the checkpoint pieces](#step-1-get-the-checkpoint-pieces)
4. [Step 2: assemble the hybrid](#step-2-assemble-the-hybrid)
5. [Step 3: build the software](#step-3-build-the-software)
6. [Step 4: launch](#step-4-launch)
7. [Step 5: verify](#step-5-verify)
8. [How the EXL3 experts run](#how-the-exl3-experts-run)
9. [Model parameters](#model-parameters)
10. [Memory](#memory)
11. [Speed](#speed)
12. [Caveats](#caveats)
13. [Credits and licenses](#credits-and-licenses)
14. Appendices: [A vLLM patch](#appendix-a-vllm-patch) · [B kernel](#appendix-b-ds41_exl3_do-kernel-changes) ·
    [C fetch script](#appendix-c-fetch_nonexpertpy) · [D assemble script](#appendix-d-assemble_hybridpy) ·
    [E glibc 2.43 workaround](#appendix-e-glibc-243--cuda-130-workaround)

## Hardware

| | Tested on |
|---|---|
| GPUs | 3× NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120, 96 GB), 250 W cap, PCIe Gen4 x16 each, **no P2P** (collectives staged through host) |
| CPU | Ampere Altra Q64-22: 64× Neoverse-N1 at 2.2 GHz, **aarch64** |
| RAM | 256 GB (246 GiB usable). RAM-mode Engram pins 189 GiB of it; see [Memory](#memory) |
| Disk | ~450 GiB free for the checkpoint pieces (≈430 GiB final + scratch) |
| OS / driver / CUDA | Ubuntu, glibc 2.43; driver 580.178.04; CUDA 13.0.88 |
| Python stack | Python 3.12, torch 2.13.0+cu130, flashinfer-python 0.6.18.post1, nvidia-cutlass-dsl 4.6.2, quack-kernels 0.6.4, cuda-tile 1.4.0, transformers 5.17.0 |

Nothing here is aarch64-specific except the build notes. On x86 the same sources should apply unchanged, but that
was not tested. A faster host CPU should help prefill, because the remaining prefill cost is Python dispatch on
the CPU (see [Speed](#speed)).

## Why a hybrid checkpoint

- The KK fork's sm_120 DeepSeek-V4.1 path runs attention and its output projection only from DeepSeek's native
  block-32 FP8 weights. So the LibertAI pack cannot be served as-is: its attention weights are EXL3 too.
- The original FP8/MXFP4 checkpoint does not fit in 3×96 GB without offloading routed experts to host memory
  (UVA). That costs decode speed and ~24 GB of host RAM.
- The hybrid keeps DeepSeek's exact weights for the non-expert ~8% (lossless, fastest kernels). Only the routed
  experts, which are 92% of the bytes, use EXL3 at 3.5 bpw: K=4 on layers 0-9 and 30-39, K=3 on layers 10-29.
  LibertAI reports +0.8% (code) / +0.1% (prose) perplexity for their full pack. This hybrid is at least as close
  to the original, but its own perplexity was not measured.
- The MTP/DSpark drafter layers keep DeepSeek's MXFP4 draft experts.

| Part | Source | Format | Size |
|---|---|---|---|
| Routed experts, layers 0-39 | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw` @ `e55cd9e0b923e8a0c7895afb379d4d28fa503bbd` | EXL3 trellis, `mul1` codebook | ~223 GiB |
| Everything else (attention, compressor, indexer, shared experts, router, norms, mHC, embedding, head, vision tower, MTP/DSpark) | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170` | block-32 FP8 (UE8M0 scales) / MXFP4 | 17.2 GiB |
| Engram tables (layers 1 and 14) | same base repo, shards 47 and 48 | FP8 + E8M0 scales | 189 GiB |

The 1017 Engram tables are new: do **not** reuse the shards of the earlier `DeepSeek-V4.1-Flash` release.

## Step 1: get the checkpoint pieces

```bash
git clone https://github.com/meldavy/dsv41f-1017-sm120-tp3 && export REPO=$PWD/dsv41f-1017-sm120-tp3
export MODELS=/path/to/models
BASE=$MODELS/DeepSeek-V4.1-Flash-1017-hybrid
EXL3=$MODELS/LibertAIDAI_DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw

# 1a. The EXL3 pack (240 GiB). Only its routed experts are used; delete it after step 2.
hf download LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw \
  --revision e55cd9e0b923e8a0c7895afb379d4d28fa503bbd --local-dir "$EXL3"

# 1b. Base metadata, tokenizer and the two Engram shards (~189 GiB). Pass the files as positional
#     arguments: with several values after one --include, hf 1.21 ignored the patterns and silently
#     skipped shard 47.
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  config.json model.safetensors.index.json tokenizer.json tokenizer_config.json \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir "$BASE"
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --include "encoding/*" \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir "$BASE"

# 1c. All non-routed-expert tensors of base shards 1-46, by HTTP range requests (~17 GiB instead of ~286 GiB).
python $REPO/scripts/fetch_nonexpert.py "$BASE"      # Appendix C
```

## Step 2: assemble the hybrid

```bash
python $REPO/scripts/assemble_hybrid.py "$BASE" "$EXL3"   # Appendix D: verifies Engram sha256, extracts the experts
rm -rf "$EXL3"                              # once you are happy with the result
```

Afterwards `$BASE` holds:
- `nonexpert-*.safetensors`, `model-0004{7,8}-of-00048.safetensors`, and `exl3experts-*-of-00044.safetensors`
  (~430 GiB, 188,245 tensors).
- `config.json`: the base config plus `quantization_config.exl3_routed_experts = {"layers": [0..39], …}`, which is
  what switches those layers to EXL3 in the patched vLLM. The originals are kept as `*.base`.
- `exl3_source/`: LibertAI's README, config, `exl3_k_map.json` and `quantization_config.json`, for provenance.

## Step 3: build the software

Use one Python 3.12 virtualenv. Set `SRC` to a source directory.

| Component | Repository | Commit |
|---|---|---|
| vLLM (KK) | `github.com/local-inference-lab/vllm`, branch `dev/karmic-kraken` | `67bb922f6f401b304abb8a6d9da450445c962662` + [Appendix A](#appendix-a-vllm-patch) |
| b12x | `github.com/local-inference-lab/b12x` | `a83336581a3a907076e60797df69ab66df5a2ff1` with `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` cherry-picked |
| ExLlamaV3 | `github.com/turboderp-org/exllamav3` | `a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a` (v1.6.3) |
| ds41_exl3_do | from [diffbot's recipe](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000) `recipe/kernels` (`xmoe/` headers + `xmoe_do/do_gemm.cu`) | + [Appendix B](#appendix-b-ds41_exl3_do-kernel-changes) |

**Build environment.** On glibc ≥ 2.42 with CUDA 13.0, every nvcc compile that includes `<cmath>` fails
(`exception specification is incompatible with that of previous function "rsqrt"`). Either patch
`crt/math_functions.h` in your toolkit, or use the unprivileged shadow-toolkit trick in
[Appendix E](#appendix-e-glibc-243--cuda-130-workaround) and point `CUDA_HOME`/`PATH` at it for all builds below.

```bash
export CUDA_HOME=/usr/local/cuda-13.0          # or the shadow toolkit from Appendix E
export PATH=$CUDA_HOME/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0a
```

**3a. vLLM (KK).**
```bash
git clone https://github.com/local-inference-lab/vllm $SRC/vllm && cd $SRC/vllm
git checkout 67bb922f6f401b304abb8a6d9da450445c962662
git apply $REPO/patches/vllm/vllm-67bb922f6.diff        # the two modified files (Appendix A)
cp $REPO/patches/vllm/exl3_experts.py vllm/models/deepseek_v4_1/exl3_experts.py   # the new file (Appendix A)
pip install "torch==2.13.0" ninja cmake "setuptools>=77,<81" "setuptools-scm>=8" "setuptools-rust>=1.9" wheel jinja2 numpy
CMAKE_CUDA_ARCHITECTURES=120a FLASHINFER_CUDA_ARCH_LIST=12.0f NVCC_THREADS=4 MAX_JOBS=48 \
  pip install --no-build-isolation -e .    # ~70 min on 64 Neoverse-N1 cores
```
Notes:
- DeepGEMM in this tree needs `elfutils/libdwfl.h`. Install `libdw-dev`, or without root extract its .deb and
  add it to `CPATH`/`LIBRARY_PATH`.
- After installing b12x's dependencies, pin `pip install --no-deps quack-kernels==0.6.4 cuda-tile==1.4.0`. The
  tree asks for quack 0.6.5, which needs CUTLASS DSL 4.7, while b12x pins 4.6.2. Without the pin the server dies
  with `No module named cutlass.base_dsl.enums`.
- On the tested machine, vLLM's compiled extensions came from a build of the same commit plus an unrelated
  GLM-only Python patch. A stock build of `67bb922f6` should be equivalent, but that exact build was not tested.

**3b. b12x** (pure Python, used from PYTHONPATH):
```bash
git clone https://github.com/local-inference-lab/b12x $SRC/b12x && cd $SRC/b12x
git checkout a83336581a3a907076e60797df69ab66df5a2ff1
git cherry-pick 0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68   # TP3: keep the 16+8 head split of the padded attention
pip install <the non-torch dependencies listed in b12x's pyproject.toml>
```

**3c. ExLlamaV3 extension**, built in-tree and not pip-installed:
```bash
git clone https://github.com/turboderp-org/exllamav3 $SRC/exllamav3 && cd $SRC/exllamav3
git checkout a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a
MAX_JOBS=40 python setup.py build_ext --inplace    # -> exllamav3_ext.*.so in the repo root (191 units)
```
Current master already guards its x86-only CPU code for aarch64. Older aarch64 stub patches (e.g. from Spark kits)
overwrite those guards and break the build: don't apply them.

**3d. ds41_exl3_do** (the decode-once prefill MoE kernel):
```bash
mkdir -p $SRC/xmoe_do && cd $SRC/xmoe_do
cp -r <diffbot recipe>/kernels/xmoe ./xmoe && cp <diffbot recipe>/kernels/xmoe_do/do_gemm.cu .
patch -p1 < $REPO/patches/kernel/do_gemm.diff
python - <<'PY'
from torch.utils.cpp_extension import load
load(name="ds41_exl3_do", sources=["do_gemm.cu"], extra_include_paths=["xmoe", "xmoe/quant"],
     extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"], build_directory="build")
PY                                         # ~90 s -> build/ds41_exl3_do.so
```

## Step 4: launch

```bash
export PYTHONPATH=$SRC/b12x:$SRC/vllm:$SRC/exllamav3:$SRC/xmoe_do/build
export OMP_NUM_THREADS=8 CUDA_DEVICE_MAX_CONNECTIONS=32 \
  VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_AOT_COMPILE=1 VLLM_USE_STANDALONE_COMPILE=1 VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_USE_FLASHINFER_SAMPLER=1 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1024 VLLM_USE_DEEP_GEMM_E8M0=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  NCCL_PROTO=LL,LL128,Simple NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16 NCCL_BUFFSIZE=2097152 NCCL_IB_DISABLE=1 \
  HF_HUB_OFFLINE=1
# b12x needs ninja on PATH (the venv's bin directory).

vllm serve "$BASE" \
  --served-model-name DeepSeek-V4.1-Flash --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 3 --pipeline-parallel-size 1 --decode-context-parallel-size 1 \
  --max-model-len 1048576 --max-num-seqs 3 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.98 --kv-cache-dtype fp8 --block-size 256 --swa-block-size 128 \
  --attention-backend B12X --moe-backend b12x --linear-backend b12x --dtype bfloat16 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' --max-cudagraph-capture-size 24 \
  --engram-config '{"cpu_offload":false,"table_memory":"ram","disk_resident_scales":false,"projection_tp":false}' \
  --async-scheduling --enable-chunked-prefill --enable-prefix-caching --max-parallel-prefills 1 \
  --prefill-compute-share 0.4 --prefill-policy round-robin --prefill-schedule-interval 1 \
  --decode-refill-target auto --no-scheduler-reserve-full-isl \
  --safetensors-load-strategy lazy --load-format auto \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}' \
  --generation-config vllm --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
  --enable-force-include-usage --enable-prompt-tokens-details --enable-request-id-headers --jit-monitor-mode warn
```

These flags are LIL's `ds41-flash` profile for `rtx-pro-6000-pcie` at TP3. The local changes are 16k chunks,
3 sequences, RAM Engram, a 1M window, and no expert offload.

| Flag | Why |
|---|---|
| `--max-num-batched-tokens 16384` | Prefill has ~1 s of fixed host overhead per chunk, so bigger chunks are faster. Each +1k tokens costs ~0.33 GiB of KV per GPU (8192 → 6.8M KV; 16384 → 3.1-3.5M) |
| `--max-num-seqs 3` | 3.14M KV tokens = 2.99 × 1M |
| `--engram-config … "table_memory":"ram"` | RAM-mode Engram nearly doubles prefill (disk mode: 3.7-3.9k tok/s at 10k; RAM: 6.9-7.1k) and raises decode from ~72 to ~91 tok/s. It needs ~203 GB of pinned host RAM. With less RAM, use `"disk"` |
| `--load-format auto` | `instanttensor` failed on the tested kernel (`io_uring_register_buffers: Bad address`) |
| No `--cpu-offload-gb` | The hybrid fits in VRAM: 80.35 GiB of weights per GPU |

Expect ~3.5 min of weight loading plus ~1.3 min of profiling/warmup. The first run also JIT-compiles the b12x
kernels; set a persistent cache directory with `B12X_COMPILE_CACHE_DIR` / `B12X_CUTE_COMPILE_CACHE_DIR`.
All-reduce falls back to PyNCCL: vLLM's custom all-reduce does not support world size 3.

## Step 5: verify

- Smoke test: `"What is 17*23? Reply with the integer only."` with `chat_template_kwargs: {"thinking": false}` → `391`.
- Needle in a haystack: a random 6-digit code inserted 37% of the way into filler text, then "What is the secret
  vault code…". Passed at 184,163 and 736,252 prompt tokens.
- The startup log should show `EXL3 routed experts: K=4 mul1, TP3 slice 768 of 2304 channels` (and `K=3`),
  `Model loading took 80.35 GiB`, and `GPU KV cache size: ~3.1M tokens`.

## How the EXL3 experts run

The patch (Appendix A) adds `DeepseekV41Exl3ExpertsMethod`, a vLLM `FusedMoEMethodBase` used for every routed-expert
layer listed in `quantization_config.exl3_routed_experts`.

- **Tensor parallelism:** each expert's intermediate axis (2304) is split three ways, 768 per GPU. w1/w3 keep
  output tile columns `[r·48, (r+1)·48)`; w2 keeps the matching input tile rows. EXL3's Hadamard rotations act on
  128-wide blocks, so slicing on a 128 boundary is exact. Storage per layer is a contiguous int16 trellis
  `[E, K/16, N/16, 16·K]` plus fp16 `suh`/`svh`, addressed through per-expert pointer tables.
- **Kernel by batch size:**
  - **≤ 8 tokens (decode):** `exllamav3_ext.exl3_moe_coop`. CUDA-graph safe, preallocated scratch.
  - **9-256 tokens, or any graph capture:** one `exllamav3_ext.exl3_moe` launch at full concurrency, no host sync.
  - **≥ 257 tokens, eager (prefill):** `ds41_exl3_do.do_moe`, a decode-once grouped GEMM. Each 16×16 trellis tile
    is decoded once per job of ≤128 rows. It runs in 4,096-token sub-batches, and the job list is fixed-size and
    built on the GPU (cumsum/searchsorted), so routing never waits on the host. The first version read per-expert
    counts back to the host every layer; in TP3 lockstep that cost ~25% of prefill.
- **Padding:** padded tokens arrive with expert id -1 and weight 0. They go to a skipped sentinel bucket
  (fused and decode-once paths) or are filtered by the `[0, E)` expert range (coop path).
- **Validation:** one GPU, real 1017 experts, all three TP slices summed, against a dense fp32 reference built
  with ExLlamaV3's `LinearEXL3.get_weight_tensor()`. Relative error 1.8-2.0e-3 (fp16 rounding) on layer 0 (K=4)
  and layer 10 (K=3), for 1/3/8/24/300/2000 tokens, including padded rows and >512 rows per expert.

Single-GPU timing for one layer slice, fused kernel with host sync vs decode-once:

| tokens | K=4 | K=3 |
|---|---|---|
| 2,048 | 5.63 → 3.95 ms | 5.71 → 3.91 ms |
| 8,192 | 15.33 → 12.47 ms | 15.80 → 12.44 ms |
| 16,384 | 29.39 → 27.11 ms | 30.56 → 27.06 ms |

Larger sub-batches (8,192 or 16,384 tokens) shave another 2-4 ms per layer at 16k, at a cost of 0.5-1.7 GiB of
KV per GPU.

Tuning environment variables: `DS41_EXL3_DO` (1), `DS41_EXL3_DO_TOKENS` (4096), `DS41_EXL3_DO_MIN_TOKENS` (257),
`DS41_EXL3_FUSED_ROWS` (512), and `DS41_EXL3_DO_DIR` (only if `ds41_exl3_do` is not on PYTHONPATH).

## Model parameters

| | |
|---|---|
| Architecture | `DeepseekV41ForCausalLM`: CED 20+20 (20-layer causal encoder, 20-layer decoder replaying a bounded window), mHC (hc_mult 4), CSA compress ratios 0/2/1, Engram on layers 1 and 14 |
| hidden / layers / vocab | 5120 / 40 / 129,280 |
| Attention | 64 heads × 512, 1 KV head, q_lora 1280, o_lora 1024 (8 output groups), rope dim 64, sliding window 128, YaRN ×16 to 1,048,576 positions. At TP3 the fork pads to 72 heads / 9 groups, so the third GPU holds 16 real + 8 zero heads |
| Indexer | 32 heads × 128, top-512, on layers 2, 8, 14, 20, 24, 28, 32, 36. Candidate source is layer 20 (2048 blocks × 8) |
| MoE | 384 routed experts, top-6, plus 1 shared expert. Intermediate 2304. sqrt-softplus routing, routed scale 1.5, SwiGLU limit 10 |
| MTP / DSpark | 3 next-n layers, 128 draft experts top-3, block 5. Shipped, **not enabled** here |
| Sampling (model card) | temperature 1.0, top_p 0.95 |

## Memory

| Per GPU | |
|---|---|
| Weights | 80.35 GiB (the original FP8/MXFP4 checkpoint needs ~95 GiB, hence expert offload) |
| KV cache | 4.9 GiB → 3,138,518 tokens at 16k chunks, 3 sequences, util 0.98 |

**Host RAM (RAM Engram):** the tables are pinned (189 GiB, 63 GiB per GPU process). On the 256 GB machine,
MemAvailable stayed between 27.9 and 34 GiB through boot, a 184k-token prefill and a 736k-token prefill. Swap
use was 6-7 GiB (idle pages from other services). Adding UVA expert offload on top of that does **not** fit:
with the original checkpoint, only ~3 GiB stayed free and boots failed.

## Speed

All rows: `ds41_bench`-style synthetic prompts, thinking off, 512 output tokens, one stream. Each row adds one
change to the row above.

| Configuration | Prefill 10k (tok/s) | Prefill 32k (tok/s) | Decode (tok/s) | KV tokens |
|---|---|---|---|---|
| *Reference: original FP8/MXFP4 checkpoint, KK, 8 GiB/GPU expert offload, disk Engram, 8k chunks, 768k window* | *2,823-2,834* | — | *53.2* | *1.20M* |
| Hybrid: 8k chunks, disk Engram, fused MoE with host sync | 2,399-2,518 | — | 70.6-71.6 | 6.80M |
| + 16k chunks | 3,020-3,069 | 3,336-3,385 | 72.3-74.0 | 3.54M |
| + sync-free decode-once MoE prefill | 3,706-3,851 | 4,140-4,257 | 71.6-74.3 | ~3.2M |
| **+ RAM-mode Engram (final)** | **6,924-7,108** | **7,546-7,597** | **91.5-91.7** | 3.14M |

Long context on the final config: 184,163-token prompt at 6,983 tok/s (TTFT 26.4 s); 736,252-token prompt at
5,412 tok/s (TTFT 136 s).

What the profiler showed along the way (`--profiler-config.profiler torch`):

- **Prefill is host-bound.** Each chunk carries ~1 s of Python dispatch: ~12 ms of CPU per layer in the fork's
  b12x attention, almost independent of token count. Half of it is the 20 CED-decoder layers, which only replay a
  small window. In TP3 every collective waits for the slowest rank's CPU.
- **Disk-mode Engram** prepared rows before each forward in 0.3-1.8 s per GPU process for a 14.8k-token chunk.
  Lockstep waited for the slowest, so about half of every prefill step was Engram I/O, and decode suffered too.
  That is why RAM mode nearly doubles prefill.
- Once those were fixed, a 16k chunk is ~0.9-1.0 s of GPU work: decode-once MoE ~0.3 s, attention ~0.25 s, dense
  GEMMs ~0.15 s, mHC/norms ~0.1 s.

## Caveats

- **Not tested:** DSpark speculative decoding. It is the obvious decode lever (diffbot reports ~4.1 accepted
  tokens per step on V4.1 agentic traffic). Capture sizes must cover `max_num_seqs × (1 + k)`.
- **Not tested:** perplexity or KL of this exact hybrid; x86 hosts; a stock build of the vLLM commit (see 3a);
  the Appendix C/D scripts in their published form. Those two are the scripts that built the tested checkpoint,
  with the paths turned into arguments; they have only been syntax-checked as published.
- **One unexplained stall:** the first-ever 736k-token request sat ~3 minutes with idle GPUs before prefill
  started. An identical repeat ran immediately at full GPU load. API-server tokenization is linear and fast
  (184k tokens in 0.9 s), no JIT cache files were written in that window, and host memory was flat.
- **Chunk size trades against capacity.** 8k chunks give 6.8M KV tokens (6 concurrent 1M sessions) but slower
  prefill.
- **Engram RAM mode needs ~230 GB of host RAM** in total. With less, use disk mode and expect roughly half the
  prefill speed.

## Credits and licenses

- DeepSeek: DeepSeek-V4.1-Flash-1017 (MIT).
- LibertAIDAI: the EXL3 3.5 bpw pack (MIT).
- turboderp: ExLlamaV3, its quantizer and kernels (MIT).
- diffbot: `xmoe` / `xmoe_do` from their DeepSeek-V4.1-Flash EXL3 recipe (MIT; derived from ExLlamaV3).
- local-inference-lab: the vLLM KK fork (Apache-2.0), b12x, and the `ds41-flash` launch profile.
- MiaAI-Lab and the RTX6kPRO community, for the EXL3 V4.1 serving groundwork.
- The patches in the appendices are offered under the licenses of the files they modify: Apache-2.0 for vLLM,
  MIT for the kernel.

## Appendix A: vLLM patch

Applies to `local-inference-lab/vllm` @ `67bb922f6f401b304abb8a6d9da450445c962662`.

**Modified files** (`git apply`):

```diff
diff --git a/vllm/models/deepseek_v4_1/nvidia/model.py b/vllm/models/deepseek_v4_1/nvidia/model.py
index c278cd2fa..f9413eeca 100644
--- a/vllm/models/deepseek_v4_1/nvidia/model.py
+++ b/vllm/models/deepseek_v4_1/nvidia/model.py
@@ -66,6 +66,7 @@ from ..b12x_layers import B12xLinearMethod, B12xMHC
 from ..b12x_layers import B12xRMSNorm as RMSNorm
 from ..ced import ced_decoder_start, gather_rows, scatter_rows
 from ..common.engram import Engram, EngramLayout, NgramHashState
+from ..exl3_experts import parse_expert_name as parse_exl3_expert_name
 from .b12x_attention import DeepseekV41B12xAttention
 
 if typing.TYPE_CHECKING:
@@ -691,7 +692,27 @@ class DeepseekV4Model(nn.Module, EagleModelMixin):
         # Pre-compute expert mapping ONCE.
         expert_mapping = self.get_expert_mapping()
 
+        exl3_loaded_layers: set[int] = set()
         for name, loaded_weight in weights:
+            exl3 = parse_exl3_expert_name(name) if ".experts." in name else None
+            if exl3 is not None:
+                if is_pp_missing_parameter(name, self):
+                    continue
+                layer_idx, expert_id, proj, kind = exl3
+                experts = self.layers[layer_idx].ffn.experts
+                routed = getattr(experts, "routed_experts", experts)
+                routed.quant_method.load_exl3(
+                    routed, expert_id, proj, kind, loaded_weight
+                )
+                if layer_idx not in exl3_loaded_layers:
+                    exl3_loaded_layers.add(layer_idx)
+                    loaded_params.update(
+                        n
+                        for n in params_dict
+                        if n.startswith(f"layers.{layer_idx}.ffn.experts.")
+                        and n.endswith((".w13_weight", ".w2_weight"))
+                    )
+                continue
             if name.startswith(("vision.", "aligner.", "image_")):
                 # Vision weights are loaded by the outer multimodal wrapper.
                 logger.warning_once("Skipping non-text weight: %s", name)
diff --git a/vllm/models/deepseek_v4_1/quant_config.py b/vllm/models/deepseek_v4_1/quant_config.py
index d7a63ee88..ec7a5c751 100644
--- a/vllm/models/deepseek_v4_1/quant_config.py
+++ b/vllm/models/deepseek_v4_1/quant_config.py
@@ -43,5 +43,34 @@ class DeepseekV41FP8Config(Fp8Config):
 
             return B12xEmbeddingMethod()
         if isinstance(layer, RoutedExperts):
+            exl3 = _exl3_routed_expert_layers()
+            if exl3 is not None:
+                from vllm.model_executor.models.utils import extract_layer_index
+
+                index = extract_layer_index(prefix)
+                if index in exl3[0]:
+                    from .exl3_experts import DeepseekV41Exl3ExpertsMethod
+
+                    return DeepseekV41Exl3ExpertsMethod(
+                        layer.moe_config, index, exl3[1]
+                    )
             return Mxfp4MoEMethod(layer.moe_config)
         return None
+
+
+def _exl3_routed_expert_layers() -> tuple[frozenset[int], float] | None:
+    """Layers whose routed experts a hybrid checkpoint stores as EXL3.
+
+    Declared under ``quantization_config.exl3_routed_experts`` (``layers``:
+    list of target layer indices). Every other tensor keeps the native format.
+    """
+    from vllm.config import get_current_vllm_config
+
+    model_config = get_current_vllm_config().model_config
+    for cfg in (model_config.hf_config, model_config.hf_text_config):
+        quant = getattr(cfg, "quantization_config", None) or {}
+        spec = quant.get("exl3_routed_experts") if isinstance(quant, dict) else None
+        if spec:
+            limit = getattr(model_config.hf_text_config, "swiglu_limit", 0.0) or 0.0
+            return frozenset(int(i) for i in spec["layers"]), float(limit)
+    return None
```

**New file** `vllm/models/deepseek_v4_1/exl3_experts.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 (trellis, mul1 codebook) routed experts for DeepSeek V4.1 on SM120.

Only the routed experts of the target layers use EXL3; every other projection
keeps the checkpoint's native block32 FP8 / MXFP4 path. The expert GEMMs run on
ExLlamaV3's kernels (``exllamav3_ext``): the cooperative decode kernel for up to
``_COOP_MAX_ROWS`` tokens and the fused grouped kernel above that. Eager calls
with at least ``_DO_MIN_TOKENS`` tokens (prefill) use the decode-once grouped
GEMM (``ds41_exl3_do``, from diffbot's xmoe_do) over a job list built on the
device, so prefill never waits on the host for routing counts.

Tensor parallelism splits every expert's intermediate axis: w1/w3 keep output
tile columns ``[r*I/16, (r+1)*I/16)`` and w2 keeps the matching input tile rows.
EXL3's Hadamard rotations act on 128-wide blocks, so a slice on a 128 boundary
is exact.
"""

from __future__ import annotations

import os

import regex as re
import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)

logger = init_logger(__name__)

EXL3_SUFFIXES = (".trellis", ".suh", ".svh", ".mul1", ".mcg")
_EXPERT_NAME = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(\w+)$")

_ACT_SILU = 0
# The cooperative decode kernel holds at most 256 (token, expert) slots and
# ExLlamaV3 sizes its row counters for MAX_BSZN = 8 tokens.
_COOP_MAX_ROWS = 8
# Rows per expert the fused kernel's scratch holds (ExLlamaV3 caps it at 512).
_FUSED_ROWS = int(os.environ.get("DS41_EXL3_FUSED_ROWS", "512"))
_MTILE_T1, _MTILE_T2 = 16, 32
# Decode-once prefill path: tokens per sub-batch bound its scratch
# (2 x rows x hidden + 2 x rows x intermediate fp16, rows = tokens x top-k).
_DO_ENABLED = os.environ.get("DS41_EXL3_DO", "1") == "1"
# Directory holding the built ds41_exl3_do extension, if it is not on PYTHONPATH.
_DO_DIR = os.environ.get("DS41_EXL3_DO_DIR", "")
_DO_MIN_TOKENS = int(os.environ.get("DS41_EXL3_DO_MIN_TOKENS", "257"))
_DO_TOKENS = int(os.environ.get("DS41_EXL3_DO_TOKENS", "4096"))
_DO_JOB_ROWS = 128

_ext = None
_do = None


def exllamav3_ext():
    global _ext
    if _ext is None:
        import exllamav3_ext as ext

        _ext = ext
    return _ext


def ds41_exl3_do():
    global _do
    if _do is None:
        import sys

        if _DO_DIR and _DO_DIR not in sys.path:
            sys.path.insert(0, _DO_DIR)
        import ds41_exl3_do as do

        _do = do
    return _do


def parse_expert_name(name: str):
    """``layers.L.ffn.experts.E.wN.kind`` -> (L, E, wN, kind), else None."""
    if not name.endswith(EXL3_SUFFIXES):
        return None
    m = _EXPERT_NAME.search(name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)


class _Scratch:
    """Per-device buffers shared by every EXL3 layer (layers run sequentially)."""

    _by_key: dict = {}

    @classmethod
    def get(cls, device, hidden: int, inter: int, topk: int):
        key = (str(device), hidden, inter, topk)
        s = cls._by_key.get(key)
        if s is None:
            s = cls(device, hidden, inter, topk)
            cls._by_key[key] = s
        return s

    def __init__(self, device, hidden, inter, topk):
        ext = exllamav3_ext()
        slots = _COOP_MAX_ROWS * topk
        half = dict(dtype=torch.float16, device=device)
        self.coop_had_g = torch.empty((slots, hidden), **half)
        self.coop_had_u = torch.empty((slots, hidden), **half)
        self.coop_gu_g = torch.empty((slots, inter), **half)
        self.coop_gu_u = torch.empty((slots, inter), **half)
        self.coop_act = torch.empty((slots, inter), **half)
        self.coop_d_out = torch.empty((slots, hidden), dtype=torch.float32, device=device)
        self.coop_out = torch.empty((_COOP_MAX_ROWS, hidden), dtype=torch.float32, device=device)
        ctr_len = (
            slots * (inter // 128)
            + _COOP_MAX_ROWS * (hidden // 128)
            + 2
            + (slots + 1)
            + slots
        )
        self.coop_ctr = torch.zeros(ctr_len, dtype=torch.int32, device=device)
        idx = torch.device(device).index or 0
        self.concurrency = max(1, int(ext.exl3_moe_max_concurrency(idx)))
        c = self.concurrency
        self.temp_state_g = torch.empty((c, _FUSED_ROWS, hidden), **half)
        self.temp_state_u = torch.empty((c, _FUSED_ROWS, hidden), **half)
        self.temp_interm_g = torch.empty((c, _FUSED_ROWS, inter), **half)
        self.temp_interm_u = torch.empty((c, _FUSED_ROWS, inter), **half)
        self.device, self.hidden, self.inter, self.topk = device, hidden, inter, topk
        self.do_rows = 0

    def ensure_do(self, num_experts: int):
        """Allocate the decode-once scratch on first prefill (the profile run)."""
        rows = _DO_TOKENS * self.topk
        if self.do_rows >= rows:
            return
        half = dict(dtype=torch.float16, device=self.device)
        self.do_xg = torch.empty((rows, self.hidden), **half)
        self.do_xu = torch.empty((rows, self.hidden), **half)
        self.do_cg = torch.empty((rows, self.inter), **half)
        self.do_cu = torch.empty((rows, self.inter), **half)
        jobs = -(-rows // _DO_JOB_ROWS) + num_experts
        self.do_jobs = torch.arange(jobs, dtype=torch.int64, device=self.device)
        self.do_rows = rows


class DeepseekV41Exl3ExpertsMethod(FusedMoEMethodBase):
    """Routed experts stored as EXL3 trellis tensors, one bitrate per layer."""

    def __init__(self, moe: FusedMoEConfig, layer_index: int, act_limit: float):
        super().__init__(moe)
        parallel = moe.moe_parallel_config
        if parallel.use_ep or parallel.dp_size != 1:
            raise NotImplementedError("EXL3 experts support tensor parallelism only")
        self.layer_index = layer_index
        self.act_limit = float(act_limit or 0.0)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.bits: int | None = None
        self.mcg = False
        self.mul1 = True
        self._seen = 0
        self._codebook_seen: set[str] = set()

    # ---- weights ---------------------------------------------------------------

    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if hidden_size % 128 or intermediate_size_per_partition % 128:
            raise ValueError(
                "EXL3 experts need 128-aligned hidden/intermediate shards, got "
                f"{hidden_size}/{intermediate_size_per_partition}"
            )
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.inter = intermediate_size_per_partition
        self.full_inter = self.inter * self.tp_size
        # The modular runner and EPLB bookkeeping look these handles up by name;
        # the trellis storage lives in plain tensors allocated on first load.
        for name in ("w13_weight", "w2_weight"):
            layer.register_parameter(
                name,
                torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False),
            )
        layer._exl3 = None

    def get_fused_moe_quant_config(self, layer):
        return None

    def _allocate(self, layer, bits: int, device) -> dict:
        E, H, I, w = self.num_experts, self.hidden_size, self.inter, 16 * bits
        i16 = dict(dtype=torch.int16, device=device)
        f16 = dict(dtype=torch.float16, device=device)
        st = {
            "w1.trellis": torch.empty((E, H // 16, I // 16, w), **i16),
            "w3.trellis": torch.empty((E, H // 16, I // 16, w), **i16),
            "w2.trellis": torch.empty((E, I // 16, H // 16, w), **i16),
            "w1.suh": torch.empty((E, H), **f16),
            "w3.suh": torch.empty((E, H), **f16),
            "w2.suh": torch.empty((E, I), **f16),
            "w1.svh": torch.empty((E, I), **f16),
            "w3.svh": torch.empty((E, I), **f16),
            "w2.svh": torch.empty((E, H), **f16),
        }
        layer._exl3 = st
        self.bits = bits
        return st

    def load_exl3(self, layer, expert: int, proj: str, kind: str, tensor: torch.Tensor) -> None:
        if kind in ("mul1", "mcg"):
            self._codebook_seen.add(kind)
            return
        st = layer._exl3
        device = layer.w13_weight.device
        if st is None:
            if kind != "trellis":
                # Bitrate is only known from a trellis; suh/svh can arrive first.
                bits = int(os.environ.get("DS41_EXL3_BITS_HINT", "0")) or None
                if bits is None:
                    pend = layer.__dict__.setdefault("_exl3_pending", [])
                    pend.append((expert, proj, kind, tensor.clone()))
                    return
            else:
                bits = tensor.shape[-1] // 16
            st = self._allocate(layer, bits, device)
            for args in layer.__dict__.pop("_exl3_pending", []):
                self.load_exl3(layer, *args)
        r, I = self.tp_rank, self.inter
        if kind == "trellis":
            if tensor.shape[-1] != 16 * self.bits:
                raise ValueError(
                    f"layer {self.layer_index}: mixed EXL3 bitrates within one layer "
                    f"({tensor.shape[-1] // 16} vs {self.bits}) are unsupported"
                )
            t16 = I // 16
            if proj in ("w1", "w3"):
                src = tensor[:, r * t16 : (r + 1) * t16, :]
            else:
                src = tensor[r * t16 : (r + 1) * t16]
        elif kind == "suh":
            src = tensor if proj in ("w1", "w3") else tensor[r * I : (r + 1) * I]
        elif kind == "svh":
            src = tensor[r * I : (r + 1) * I] if proj in ("w1", "w3") else tensor
        else:
            raise ValueError(f"unexpected EXL3 tensor kind {kind!r}")
        st[f"{proj}.{kind}"][expert].copy_(src, non_blocking=False)
        self._seen += 1

    def process_weights_after_loading(self, layer) -> None:
        st = layer._exl3
        expected = 9 * self.num_experts
        if st is None or self._seen != expected:
            raise ValueError(
                f"EXL3 layer {self.layer_index}: loaded {self._seen} of {expected} "
                "expert tensors"
            )
        if self._codebook_seen != {"mul1"}:
            raise ValueError(
                f"EXL3 layer {self.layer_index}: expected the mul1 codebook, "
                f"got markers {sorted(self._codebook_seen)}"
            )

        def ptrs(key):
            t = st[key]
            stride = t[0].numel() * t.element_size() if t.shape[0] > 1 else 0
            base = t.data_ptr()
            return torch.tensor(
                [base + e * stride for e in range(t.shape[0])],
                dtype=torch.int64,
                device=t.device,
            )

        layer._exl3_ptrs = {k: ptrs(k) for k in st}
        self.scratch = _Scratch.get(
            layer.w13_weight.device,
            self.hidden_size,
            self.inter,
            self.moe.experts_per_token,
        )
        self.flat_token_cache: dict[int, torch.Tensor] = {}
        logger.info_once(
            "EXL3 routed experts: K=%d mul1, TP%d slice %d of %d channels, "
            "coop<=%d tokens, fused rows %d",
            self.bits,
            self.tp_size,
            self.inter,
            self.full_inter,
            _COOP_MAX_ROWS,
            _FUSED_ROWS,
        )

    # ---- forward ---------------------------------------------------------------

    def prepare_workspace(self, hidden_states, shared_workspace_size):
        # Shared experts fall back to the workspace manager; the EXL3 kernels
        # use their own scratch.
        return None, None

    def apply(
        self,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input=None,
        workspace=None,
    ) -> torch.Tensor:
        tokens = x.shape[0]
        if tokens == 0:
            return torch.empty_like(x)
        xh = x.to(torch.float16)
        if tokens <= _COOP_MAX_ROWS:
            out = self._coop(layer, xh, topk_weights, topk_ids)
        elif (
            _DO_ENABLED
            and tokens >= _DO_MIN_TOKENS
            and not torch.cuda.is_current_stream_capturing()
        ):
            out = self._decode_once(layer, xh, topk_weights, topk_ids)
        else:
            out = self._fused(layer, xh, topk_weights, topk_ids)
        return out.to(x.dtype)

    def _coop(self, layer, xh, topk_weights, topk_ids):
        ext, s, p = exllamav3_ext(), self.scratch, layer._exl3_ptrs
        tokens = xh.shape[0]
        k = float(self.bits)
        out = s.coop_out[:tokens]
        # The [0, E) range filter turns padded-token picks (id -1, weight 0)
        # into exact zeros instead of pointer-table reads out of bounds.
        ext.exl3_moe_coop(
            xh,
            topk_ids.to(torch.int64).contiguous(),
            topk_weights.to(torch.float16).contiguous(),
            0,
            self.num_experts,
            self.hidden_size,
            p["w1.trellis"], p["w1.suh"], p["w1.svh"],
            p["w3.trellis"], p["w3.suh"], p["w3.svh"],
            p["w2.trellis"], p["w2.suh"], p["w2.svh"],
            None, None, None,
            k, k, k,
            self.mcg, self.mul1,
            _ACT_SILU,
            self.act_limit,
            True,
            s.coop_had_g, s.coop_had_u, s.coop_gu_g, s.coop_gu_u, s.coop_act,
            s.coop_d_out, s.coop_ctr, out,
            None, None,
        )
        return out

    def _decode_once(self, layer, xh, topk_weights, topk_ids):
        do, s, p = ds41_exl3_do(), self.scratch, layer._exl3_ptrs
        tokens, topk = topk_ids.shape
        E, dev = self.num_experts, xh.device
        s.ensure_do(E)
        out = torch.zeros((tokens, self.hidden_size), dtype=torch.float32, device=dev)
        jobs = s.do_jobs
        for t0 in range(0, tokens, _DO_TOKENS):
            t1 = min(t0 + _DO_TOKENS, tokens)
            ids = topk_ids[t0:t1].reshape(-1).to(torch.int64)
            # Padded tokens carry id -1: sort them last as the skipped expert E.
            ids = torch.where(ids < 0, E, ids)
            order = torch.argsort(ids, stable=True)
            row_exp = ids.index_select(0, order).to(torch.int32)
            tok = torch.div(order, topk, rounding_mode="floor") + t0
            wts = topk_weights[t0:t1].reshape(-1).index_select(0, order).to(torch.float16)
            count = torch.zeros(E + 1, dtype=torch.int64, device=dev)
            count.scatter_add_(0, ids, torch.ones_like(ids))
            count = count[:E]
            first_row = torch.cumsum(count, 0) - count
            n_jobs = torch.div(count + _DO_JOB_ROWS - 1, _DO_JOB_ROWS, rounding_mode="floor")
            job_end = torch.cumsum(n_jobs, 0)
            job_exp = torch.searchsorted(job_end, jobs, right=True).clamp_(max=E - 1)
            k = jobs - (job_end - n_jobs).index_select(0, job_exp)
            job_row0 = (first_row.index_select(0, job_exp) + _DO_JOB_ROWS * k).to(torch.int32)
            job_rows = (count.index_select(0, job_exp) - _DO_JOB_ROWS * k).clamp_(0, _DO_JOB_ROWS)
            do.do_moe(
                xh, out, tok, wts, row_exp,
                job_row0, job_rows.to(torch.int32), job_exp,
                p["w1.trellis"], p["w1.suh"], p["w1.svh"],
                p["w3.trellis"], p["w3.suh"], p["w3.svh"],
                p["w2.trellis"], p["w2.suh"], p["w2.svh"],
                s.do_xg, s.do_xu, s.do_cg, s.do_cu,
                self.bits, 2 if self.mul1 else 1, self.act_limit, _ACT_SILU, E,
            )
        return out

    def _flat_token(self, tokens: int, topk: int, device) -> torch.Tensor:
        t = self.flat_token_cache.get(tokens * 64 + topk)
        if t is None:
            t = torch.arange(tokens, device=device, dtype=torch.int64).repeat_interleave(topk)
            self.flat_token_cache[tokens * 64 + topk] = t
        return t

    def _fused(self, layer, xh, topk_weights, topk_ids):
        ext, s, p = exllamav3_ext(), self.scratch, layer._exl3_ptrs
        tokens, topk = topk_ids.shape
        E = self.num_experts
        local = topk_ids.reshape(-1).to(torch.int64)
        # Padded tokens carry id -1: route them to the kernel's skipped bucket E.
        local = torch.where(local < 0, E, local)
        flat_w = topk_weights.reshape(-1).to(torch.float16)
        flat_t = self._flat_token(tokens, topk, xh.device)
        order = torch.argsort(local, stable=True)
        token_sorted = flat_t.index_select(0, order)
        weight_sorted = flat_w.index_select(0, order)
        count = torch.zeros(E + 1, dtype=torch.int64, device=xh.device)
        count.scatter_add_(0, local, torch.ones_like(local))
        out = torch.zeros((tokens, self.hidden_size), dtype=torch.float32, device=xh.device)
        k = float(self.bits)

        def run(cnt, tok, wts, num_active, lo, hi, m_tile):
            ext.exl3_moe(
                xh, out, cnt, tok, wts,
                s.temp_state_g, s.temp_state_u, s.temp_interm_g, s.temp_interm_u,
                _ACT_SILU, k, k, k,
                p["w1.trellis"], p["w1.suh"], p["w1.svh"],
                p["w3.trellis"], p["w3.suh"], p["w3.svh"],
                p["w2.trellis"], p["w2.suh"], p["w2.svh"],
                self.mcg, self.mul1, self.mcg, self.mul1, self.mcg, self.mul1,
                self.act_limit, num_active, None, None, lo, hi, m_tile,
            )

        if tokens <= _FUSED_ROWS or torch.cuda.is_current_stream_capturing():
            # No expert can hold more rows than there are tokens: one launch at
            # full concurrency, no host synchronization (graph safe).
            run(count, token_sorted, weight_sorted, -1, 1, _FUSED_ROWS, 16)
            return out

        # Prefill: read the per-expert row counts once, launch one row-tile
        # instance per occupancy band, and hand experts with more rows than the
        # scratch holds to further passes over their remaining rows.
        counts = count[:E].tolist()
        max_rows = max(counts)
        if max_rows <= _FUSED_ROWS:
            passes = [(count, token_sorted, weight_sorted, counts)]
        else:
            start = torch.cumsum(count, 0) - count
            pos = torch.arange(local.numel(), device=xh.device) - start.index_select(
                0, local.index_select(0, order)
            )
            passes = []
            for lo_row in range(0, max_rows, _FUSED_ROWS):
                keep = (pos >= lo_row) & (pos < lo_row + _FUSED_ROWS)
                cnt = (count - lo_row).clamp(0, _FUSED_ROWS)
                cnt[E] = 0
                cl = [min(max(c - lo_row, 0), _FUSED_ROWS) for c in counts]
                passes.append((cnt, token_sorted[keep], weight_sorted[keep], cl))
        for cnt, tok, wts, cl in passes:
            bands = (
                (sum(1 for c in cl if c > _MTILE_T2), _MTILE_T2 + 1, _FUSED_ROWS, 64),
                (sum(1 for c in cl if _MTILE_T1 < c <= _MTILE_T2), _MTILE_T1 + 1, _MTILE_T2, 32),
                (sum(1 for c in cl if 0 < c <= _MTILE_T1), 1, _MTILE_T1, 16),
            )
            for n, lo, hi, m_tile in bands:
                if n:
                    run(cnt, tok, wts, n, lo, hi, m_tile)
        return out
```

## Appendix B: ds41_exl3_do kernel changes

Against diffbot's `recipe/kernels/xmoe_do/do_gemm.cu`. The changes:
- jobs with no rows exit at once, so a fixed-size, device-built job list needs no host sync
- rows of the padded-token sentinel expert are skipped
- the kernel instances are K4/K3 `mul1`

```diff
--- a/do_gemm.cu
+++ b/do_gemm.cu
@@ -1,4 +1,6 @@
-// Copyright (c) 2026 Diffbot. MIT License (recipe LICENSE). Uses device code from ExLlamaV3 (kernels/xmoe, MIT, see
+// DS4.1-Flash-1017 TP3 (2026-10-17): adapted from diffbot's xmoe_do -- empty jobs exit at once (fixed-size,
+// device-built job lists need no host sync), rows of the padded-token sentinel expert are skipped, K4/K3 mul1 instances.
+// Copyright (c) 2026 Diffbot. MIT License (LICENSE). Uses device code from ExLlamaV3 (kernels/xmoe, MIT, see
 // kernels/xmoe/LICENSE.exllamav3): the EXL3 trellis decoder (dq_dispatch), mma/ldmatrix/cp.async helpers and the 128-point
 // Hadamard device functions. Build: kernels/build.sh (-> kernels/build/xmoe_do), loaded by the patched vllm-exl3 plugin when EXL3_DO=1.
 // xmoe_do (2026-10-01, DS4.1 lever 2): decode-once grouped GEMM for EXL3 MoE experts with many rows (prefill).
@@ -31,6 +33,7 @@
     const int n_tile = blockIdx.x;
     const int row0 = job_row0[job];
     const int rows = job_rows[job];
+    if (rows <= 0) return;  // padding job of a fixed-size, device-built job list
     const uint16_t* trellis = reinterpret_cast<const uint16_t*>(job_trellis[job]);
     const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
     const int blocks_n = N / 16;
@@ -128,9 +131,8 @@
     dim3 grid(N / 128, J);
     #define LAUNCH(B, CB, OH) do_gemm_kernel<B, CB, 128, 128, OH><<<grid, 256, 0, stream>>>( \
         (const half*) A.data_ptr(), C, job_row0.data_ptr<int>(), job_rows.data_ptr<int>(), job_trellis.data_ptr<int64_t>(), K, N)
-    if (bits == 3 && cb == 2) { if (out_half) LAUNCH(3, 2, true); else LAUNCH(3, 2, false); }
-    else if (bits == 3 && cb == 1) { if (out_half) LAUNCH(3, 1, true); else LAUNCH(3, 1, false); }
-    else if (bits == 2 && cb == 1) { if (out_half) LAUNCH(2, 1, true); else LAUNCH(2, 1, false); }
+    if (bits == 4 && cb == 2) { if (out_half) LAUNCH(4, 2, true); else LAUNCH(4, 2, false); }
+    else if (bits == 3 && cb == 2) { if (out_half) LAUNCH(3, 2, true); else LAUNCH(3, 2, false); }
     else TORCH_CHECK(false, "unsupported bits/cb");
     #undef LAUNCH
 }
@@ -149,12 +151,13 @@
 
 __global__ void k_gather_had(const half* __restrict__ x, const int64_t* __restrict__ tok, const int* __restrict__ row_exp,
                              const int64_t* __restrict__ suh_g, const int64_t* __restrict__ suh_u, half* __restrict__ xg,
-                             half* __restrict__ xu, int R, int hidden, bool gated)
+                             half* __restrict__ xu, int R, int hidden, bool gated, int n_exp)
 {
     const int wpt = hidden / 128;
     const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
     if (w >= R * wpt) return;
     const int r = w / wpt, off = w % wpt, e = row_exp[r];
+    if (e < 0 || e >= n_exp) return;
     const half* in = x + tok[r] * hidden + off * 128;
     if (gated) had_hf_r_128_inner<true, false>(in, xg + (size_t)r * hidden + off * 128, (const half*) suh_g[e] + off * 128, HSCALE);
     had_hf_r_128_inner<true, false>(in, xu + (size_t)r * hidden + off * 128, (const half*) suh_u[e] + off * 128, HSCALE);
@@ -162,12 +165,13 @@
 
 __global__ void k_guad(half* __restrict__ cg, const half* __restrict__ cu, const int* __restrict__ row_exp,
                        const int64_t* __restrict__ svh_g, const int64_t* __restrict__ svh_u, const int64_t* __restrict__ suh_d,
-                       int R, int inter, float act_limit, int act_fn)
+                       int R, int inter, float act_limit, int act_fn, int n_exp)
 {
     const int wpt = inter / 128;
     const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
     if (w >= R * wpt) return;
     const int r = w / wpt, off = w % wpt, e = row_exp[r];
+    if (e < 0 || e >= n_exp) return;
     const size_t o = (size_t)r * inter + off * 128;
     had_hf_r_128_guad_inner(cg + o, cu + o, cg + o, (const half*) svh_g[e] + off * 128, (const half*) svh_u[e] + off * 128,
                             (const half*) suh_d[e] + off * 128, HSCALE, act_limit, act_fn);
@@ -175,16 +179,18 @@
 
 __global__ void k_d_out(const half* __restrict__ d, float* __restrict__ out, const int64_t* __restrict__ tok,
                         const half* __restrict__ wts, const int* __restrict__ row_exp, const int64_t* __restrict__ svh_d,
-                        int R, int hidden)
+                        int R, int hidden, int n_exp)
 {
     const int wpt = hidden / 128;
     const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
     if (w >= R * wpt) return;
     const int r = w / wpt, off = w % wpt, e = row_exp[r];
+    if (e < 0 || e >= n_exp) return;
     had_hf_r_128_d_inner(d + (size_t)r * hidden + off * 128, out + tok[r] * hidden + off * 128,
                          (const half*) svh_d[e] + off * 128, HSCALE * __half2float(wts[r]));
 }
 
+// Rows whose expert is outside [0, n_exp) (the padded-token sentinel) are skipped.
 // out[T, hidden] fp32 += big-expert MoE for rows (tok, wts, row_exp) grouped by expert into jobs (job_row0, job_rows, job_exp).
 // Pointer tables are the plugin's per-expert int64 device tensors. Scratch: xg, xu [R, hidden] fp16; cg, cu [R, inter] fp16.
 // The down output reuses xg.
@@ -193,7 +199,7 @@
             torch::Tensor g_tr, torch::Tensor g_suh, torch::Tensor g_svh, torch::Tensor u_tr, torch::Tensor u_suh, torch::Tensor u_svh,
             torch::Tensor d_tr, torch::Tensor d_suh, torch::Tensor d_svh,
             torch::Tensor xg, torch::Tensor xu, torch::Tensor cg, torch::Tensor cu,
-            int64_t bits, int64_t cb, double act_limit, int64_t act_fn)
+            int64_t bits, int64_t cb, double act_limit, int64_t act_fn, int64_t n_exp)
 {
     const at::cuda::OptionalCUDAGuard guard(device_of(x));
     auto stream = at::cuda::getCurrentCUDAStream().stream();
@@ -205,15 +211,15 @@
     const int T = 256;
     auto blocks = [&](int warps) { return (warps * 32 + T - 1) / T; };
     k_gather_had<<<blocks(R * (hidden / 128)), T, 0, stream>>>((const half*) x.data_ptr(), tok.data_ptr<int64_t>(), row_exp.data_ptr<int>(),
-        g_suh.data_ptr<int64_t>(), u_suh.data_ptr<int64_t>(), (half*) xg.data_ptr(), (half*) xu.data_ptr(), R, hidden, gated);
+        g_suh.data_ptr<int64_t>(), u_suh.data_ptr<int64_t>(), (half*) xg.data_ptr(), (half*) xu.data_ptr(), R, hidden, gated, (int) n_exp);
     auto xg_v = xg.narrow(0, 0, R), xu_v = xu.narrow(0, 0, R);
     if (gated) launch_do_gemm(xg_v, cg.data_ptr(), true, inter, job_row0, job_rows, jg, bits, cb, stream);
     launch_do_gemm(xu_v, cu.data_ptr(), true, inter, job_row0, job_rows, ju, bits, cb, stream);
     k_guad<<<blocks(R * (inter / 128)), T, 0, stream>>>((half*) cg.data_ptr(), (const half*) cu.data_ptr(), row_exp.data_ptr<int>(),
-        g_svh.data_ptr<int64_t>(), u_svh.data_ptr<int64_t>(), d_suh.data_ptr<int64_t>(), R, inter, (float) act_limit, (int) act_fn);
+        g_svh.data_ptr<int64_t>(), u_svh.data_ptr<int64_t>(), d_suh.data_ptr<int64_t>(), R, inter, (float) act_limit, (int) act_fn, (int) n_exp);
     launch_do_gemm(cg.narrow(0, 0, R), xg.data_ptr(), true, hidden, job_row0, job_rows, jd, bits, cb, stream);
     k_d_out<<<blocks(R * (hidden / 128)), T, (T / 32) * 128 * sizeof(float), stream>>>((const half*) xg.data_ptr(), out.data_ptr<float>(),
-        tok.data_ptr<int64_t>(), (const half*) wts.data_ptr(), row_exp.data_ptr<int>(), d_svh.data_ptr<int64_t>(), R, hidden);
+        tok.data_ptr<int64_t>(), (const half*) wts.data_ptr(), row_exp.data_ptr<int>(), d_svh.data_ptr<int64_t>(), R, hidden, (int) n_exp);
 }
 
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
```

## Appendix C: fetch_nonexpert.py

```python
#!/usr/bin/env python3
"""Range-fetch every tensor of DeepSeek-V4.1-Flash-1017 except the routed experts of the target
layers and the Engram tables, without downloading whole shards (~17 GiB instead of ~286 GiB).

Usage: fetch_nonexpert.py BASE_DIR
  BASE_DIR must already hold the base repo's model.safetensors.index.json (see the recipe).
  Writes nonexpert-000NN-of-00048.safetensors next to it; safe to re-run (skips finished files).
"""
import json, os, re, struct, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO, REV = "deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170"
URL = f"https://huggingface.co/{REPO}/resolve/{REV}/" + "{}"
SKIP = re.compile(r"layers\.\d+\.(ffn\.experts\.|engram\.embed\.)")
ENGRAM_SHARDS = {"model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"}


def get(name, start, end):
    for _ in range(8):
        try:
            req = urllib.request.Request(URL.format(name), headers={"Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == end - start:
                return data
        except Exception as e:  # retry transient HTTP errors
            print(f"retry {name} @{start}: {e}", flush=True)
    raise SystemExit(f"failed to fetch {name} @{start}")


def do_shard(dest, name):
    out = os.path.join(dest, name.replace("model-", "nonexpert-"))
    if os.path.exists(out):
        return name, 0
    n = struct.unpack("<Q", get(name, 0, 8))[0]
    header = json.loads(get(name, 8, 8 + n))
    meta = header.pop("__metadata__", None)
    keep = sorted((k for k in header if not SKIP.match(k)), key=lambda k: header[k]["data_offsets"][0])
    if not keep:
        return name, 0
    base, ranges, cur = 8 + n, [], None
    for k in keep:  # coalesce neighbours into <= 256 MiB range requests
        s, e = header[k]["data_offsets"]
        if cur and s - cur[1] <= 1 << 20 and e - cur[0] <= 256 << 20:
            cur[1] = e; cur[2].append(k)
        else:
            cur = [s, e, [k]]; ranges.append(cur)
    blobs = {}
    for s, e, ks in ranges:
        data = get(name, base + s, base + e)
        for k in ks:
            ks_, ke_ = header[k]["data_offsets"]
            blobs[k] = data[ks_ - s:ke_ - s]
    new, off = {}, 0
    for k in keep:
        new[k] = {"dtype": header[k]["dtype"], "shape": header[k]["shape"], "data_offsets": [off, off + len(blobs[k])]}
        off += len(blobs[k])
    if meta:
        new["__metadata__"] = meta
    hb = json.dumps(new, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(out + ".part", "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for k in keep:
            f.write(blobs[k])
    os.replace(out + ".part", out)
    return name, off


def main():
    dest = sys.argv[1]
    idx = json.load(open(os.path.join(dest, "model.safetensors.index.json")))["weight_map"]
    shards = sorted({v for k, v in idx.items() if not SKIP.match(k)} - ENGRAM_SHARDS)
    total = 0
    with ThreadPoolExecutor(6) as ex:
        for name, nbytes in ex.map(lambda s: do_shard(dest, s), shards):
            total += nbytes
            print(f"{name}: {nbytes / 2**20:.1f} MiB", flush=True)
    print(f"DONE {total / 2**30:.2f} GiB", flush=True)


main()
```

## Appendix D: assemble_hybrid.py

```python
#!/usr/bin/env python3
"""Assemble the DeepSeek-V4.1-Flash-1017 hybrid checkpoint in place.

Usage: assemble_hybrid.py BASE_DIR EXL3_DIR
  BASE_DIR  deepseek-ai/DeepSeek-V4.1-Flash-1017 @1bae0e10: config.json, model.safetensors.index.json,
            tokenizer files, Engram shards 47/48 and the nonexpert-*.safetensors from fetch_nonexpert.py
  EXL3_DIR  LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw @e55cd9e0 (full download)

Copies the routed-expert tensors of EXL3_DIR into BASE_DIR/exl3experts-*.safetensors, verifies the
Engram shards, and rewrites BASE_DIR's index and config (originals kept as *.base). BASE_DIR is then
the model directory to serve; EXL3_DIR can be deleted afterwards.
"""
import hashlib, json, os, re, shutil, struct, sys

ENGRAM_SHA = {
    "model-00047-of-00048.safetensors": "76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5",
    "model-00048-of-00048.safetensors": "9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd",
}
ROUTED = re.compile(r"layers\.(\d+)\.ffn\.experts\.\d+\.w[123]\.(trellis|suh|svh|mul1)$")
BASE_ROUTED = re.compile(r"layers\.\d+\.ffn\.experts\.")


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(64 << 20):
            d.update(b)
    return d.hexdigest()


def extract(src, dst, keep):
    h, base = header(src)
    names = sorted((k for k in h if keep(k)), key=lambda k: h[k]["data_offsets"][0])
    if not names:
        return {}
    new, off = {}, 0
    for k in names:
        s, e = h[k]["data_offsets"]
        new[k] = {"dtype": h[k]["dtype"], "shape": h[k]["shape"], "data_offsets": [off, off + e - s]}
        off += e - s
    hb = json.dumps(new, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(src, "rb") as fi, open(dst + ".part", "wb") as fo:
        fo.write(struct.pack("<Q", len(hb))); fo.write(hb)
        for k in names:
            s, e = h[k]["data_offsets"]
            fi.seek(base + s)
            left = e - s
            while left:
                chunk = fi.read(min(left, 256 << 20))
                fo.write(chunk); left -= len(chunk)
    os.replace(dst + ".part", dst)
    return {k: os.path.basename(dst) for k in names}


def main():
    out, exl3 = sys.argv[1], sys.argv[2]
    for name, want in ENGRAM_SHA.items():
        got = sha256(os.path.join(out, name))
        print(name, "OK" if got == want else f"MISMATCH {got}", flush=True)
        if got != want:
            sys.exit(1)
    idx_path = os.path.join(out, "model.safetensors.index.json")
    if not os.path.exists(idx_path + ".base"):
        shutil.copy(idx_path, idx_path + ".base")
    base_index = json.load(open(idx_path + ".base"))["weight_map"]
    weight_map = {}
    for f in sorted(os.listdir(out)):
        if f.startswith("nonexpert-") and f.endswith(".safetensors"):
            for k in header(os.path.join(out, f))[0]:
                weight_map[k] = f
    for k, f in base_index.items():
        if f in ENGRAM_SHA:
            weight_map[k] = f
    missing = [k for k in base_index if not BASE_ROUTED.match(k) and k not in weight_map]
    if missing:
        sys.exit(f"{len(missing)} base tensors missing from the range fetch, e.g. {missing[:3]}")
    exl3_index = json.load(open(os.path.join(exl3, "model.safetensors.index.json")))["weight_map"]
    layers = sorted({int(m.group(1)) for k in exl3_index if (m := ROUTED.match(k))})
    for f in sorted(set(exl3_index.values())):
        dst = os.path.join(out, f.replace("model-", "exl3experts-"))
        got = ({k: os.path.basename(dst) for k in header(dst)[0]} if os.path.exists(dst)
               else extract(os.path.join(exl3, f), dst, lambda k: ROUTED.match(k) is not None))
        weight_map.update(got)
        print(f, len(got), "expert tensors", flush=True)
    n_routed = sum(1 for k in weight_map if ROUTED.match(k))
    if n_routed != len(layers) * 384 * 3 * 4:
        sys.exit(f"routed EXL3 tensors {n_routed} != {len(layers) * 384 * 3 * 4}")
    cfg_path = os.path.join(out, "config.json")
    if not os.path.exists(cfg_path + ".base"):
        shutil.copy(cfg_path, cfg_path + ".base")
    cfg = json.load(open(cfg_path + ".base"))
    q = json.load(open(os.path.join(exl3, "config.json")))["quantization_config"]
    cfg["quantization_config"]["exl3_routed_experts"] = {
        "layers": layers,
        "source": "LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw@e55cd9e0b923e8a0c7895afb379d4d28fa503bbd",
        "codebook": q.get("codebook"), "version": q.get("version"), "out_scales": q.get("out_scales"),
    }
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    json.dump({"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}, open(idx_path, "w"), indent=1)
    os.makedirs(os.path.join(out, "exl3_source"), exist_ok=True)
    for f in ("README.md", "config.json", "exl3_k_map.json", "quantization_config.json"):
        shutil.copy(os.path.join(exl3, f), os.path.join(out, "exl3_source", f))
    print(f"DONE {len(weight_map)} tensors, EXL3 experts on layers {layers[0]}..{layers[-1]}", flush=True)


main()
```

## Appendix E: glibc 2.43 + CUDA 13.0 workaround

```bash
# Build-time only; nothing under /usr/local is modified. S is any scratch directory.
S=/tmp/cuda-13.0-glibc243; T=/usr/local/cuda-13.0; A=targets/sbsa-linux   # x86_64: targets/x86_64-linux
mkdir -p $S/bin $S/$A/include/crt
for f in $T/bin/*; do ln -s $f $S/bin/; done
for f in $T/*; do b=$(basename $f); [ $b = bin ] || [ $b = targets ] || [ $b = include ] || ln -s $f $S/$b; done
for f in $T/$A/*; do b=$(basename $f); [ $b = include ] || ln -s $f $S/$A/$b; done
for f in $T/$A/include/*; do b=$(basename $f); [ $b = crt ] || ln -s $f $S/$A/include/$b; done
for f in $T/$A/include/crt/*; do b=$(basename $f); [ $b = math_functions.h ] || ln -s $f $S/$A/include/crt/$b; done
ln -s $S/$A/include $S/include
{ printf '#if defined(__cplusplus) && !defined(_NV_GLIBC_RSQRT_NOEXCEPT)\n#define _NV_GLIBC_RSQRT_NOEXCEPT noexcept(true)\n#elif !defined(_NV_GLIBC_RSQRT_NOEXCEPT)\n#define _NV_GLIBC_RSQRT_NOEXCEPT\n#endif\n'
  sed -e 's/double                 rsqrt(double x);/double                 rsqrt(double x) _NV_GLIBC_RSQRT_NOEXCEPT;/' \
      -e 's/float                  rsqrtf(float x);/float                  rsqrtf(float x) _NV_GLIBC_RSQRT_NOEXCEPT;/' \
      $T/$A/include/crt/math_functions.h; } > $S/$A/include/crt/math_functions.h
export CUDA_HOME=$S PATH=$S/bin:$PATH
```

Check it with a one-line `.cu` file that includes `<cmath>` and `<cuda_runtime.h>`. Before the patch it fails with
the `rsqrt` error; after it, it compiles. Note that `include` must point at the patched tree, because torch's
extension builder passes `-I$CUDA_HOME/include`.
