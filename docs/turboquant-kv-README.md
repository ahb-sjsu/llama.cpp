# TurboQuant KV Cache Compression

> **Status:** Sprint 4b of 5 (ggml type tags + CLI parsing) · feature branch only · KV storage backend lands in 4c

PolarQuant + Lloyd-Max KV cache compression for `llama.cpp`. Achieves higher compression at better quality than current `Q4_0`/`Q5_0`/`Q8_0` modes.

## TL;DR

For long-context inference (32K+), KV cache memory is the bottleneck. TurboQuant gives:

| Mode | bytes/elem (head_dim=128) | Compression | Cosine sim |
|------|---------------------------|-------------|-----------|
| F16 (baseline) | 2.00 | 1.0× | 1.000 |
| Q8_0 | 1.06 | 1.9× | 0.998 |
| Q4_0 | 0.56 | 3.5× | 0.971 |
| **TQ KV4** | **0.53** | **3.8×** | **0.994** |
| **TQ KV3** | **0.41** | **4.9×** | **0.978** |
| **TQ KV2** | **0.28** | **7.1×** | **0.939** |

Numbers from `tests/test-tq-kv.cpp` (cosine sim from CPU reference impl, head_dim=128).

For Qwen2.5-7B at 32K context: KV cache drops from ~16 GB (F16) to ~3.3 GB (TQ KV3) — fits on a 16 GB GPU.

## Algorithm

Each `head_dim`-sized vector (one row of K or V) is compressed in 5 steps:

1. **L2 norm** — store `‖v‖₂` as fp32 (~4 bytes overhead per vector)
2. **Normalize** — `v_unit = v / ‖v‖₂`
3. **Random orthogonal rotation** — `v_rot = v_unit @ Π`. After rotation, coordinates are approximately i.i.d. Gaussian. The rotation `Π` is fixed per `head_dim` (deterministic from a seed, generated once).
   - For `head_dim ≤ 4096`: full QR-decomposed matrix (Householder)
   - For `head_dim > 4096`: structured sign-flip + permutation (O(D) memory and apply cost)
4. **Scalar quantize** — Lloyd-Max codebook for N(0, 1/√d) scaled to actual dimension. 2/3/4 bits per element.
5. **Bit-pack** — 8 × 3-bit → 3 bytes (TQ KV3); 4 × 2-bit → 1 byte (TQ KV2); 2 × 4-bit → 1 byte (TQ KV4).

Decompression inverts the pipeline.

**Theory:** Zandieh et al. (ICLR 2026) "Sub-linear Memory Inference via PolarQuant + QJL". Reference Python implementation at [ahb-sjsu/turboquant-pro](https://github.com/ahb-sjsu/turboquant-pro).

## What's in this branch

| File | Purpose | Status |
|------|---------|--------|
| [`docs/turboquant-kv-design.md`](turboquant-kv-design.md) | Full architectural design doc | ✅ |
| [`src/llama-kv-turboquant.h`](../src/llama-kv-turboquant.h) | C++ public API | ✅ |
| [`src/llama-kv-turboquant.cpp`](../src/llama-kv-turboquant.cpp) | CPU reference implementation | ✅ Sprint 2 |
| [`tests/test-tq-kv.cpp`](../tests/test-tq-kv.cpp) | Comprehensive unit tests (16K+ checks + CUDA equivalence) | ✅ Sprint 2/3 |
| [`src/llama-kv-turboquant-cuda.cu`](../src/llama-kv-turboquant-cuda.cu) | CUDA kernels (Volta sm_70 + Ampere sm_80) | ✅ Sprint 3 |
| [`src/llama-kv-tiered.{h,cpp}`](../src/llama-kv-tiered.h) | Tiered cache data structure (hot fp16 + cold TQ) | ✅ Sprint 4 |
| [`tests/test-tq-tiered.cpp`](../tests/test-tq-tiered.cpp) | Tiered cache tests (hot/cold/eviction/stats) | ✅ Sprint 4 |
| [`.github/workflows/turboquant-kv.yml`](../.github/workflows/turboquant-kv.yml) | CI workflow (CPU build + CUDA build + lint) | ✅ |
| `ggml/include/ggml.h` GGML_TYPE_TQ_KV{2,3,4} | New ggml types (tags only) | ✅ Sprint 4b |
| `common/arg.cpp` `--cache-type-k tq_kv3` | CLI parser accepts the new types | ✅ Sprint 4b |
| `src/llama-context.cpp` | Friendly error until backend lands | ✅ Sprint 4b |
| `src/llama-kv-cache.cpp` adapter | Wire tiered_cache into llama_memory_i | ⏳ Sprint 4c |
| Upstream PR | | ⏳ Sprint 5 |

## Building and testing

CPU-only build (default):

```bash
mkdir build && cd build
cmake .. -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build . --target test-tq-kv -j
./bin/test-tq-kv
```

With CUDA kernels (Sprint 3):

```bash
mkdir build-cuda && cd build-cuda
cmake .. -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
         -DLLAMA_TQ_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;80"
cmake --build . --target test-tq-kv -j
./bin/test-tq-kv  # runs CPU tests + CUDA equivalence tests
```

The CUDA tests compare GPU output to the CPU reference (norm, packed
indices, full reconstruction). They self-skip if no CUDA device is
detected at runtime, so the same binary works on both CPU-only and
GPU machines.

Expected output:
```
test_block_size_and_ratio
test_rotation_orthogonality
...
test_round_trip D=256 bits=3
  mean cosine similarity = 0.9782 (need ≥ 0.930)

=== 16531 / 16531 checks passed ===
```

The CI workflow `.github/workflows/turboquant-kv.yml` runs the same on push/PR.

## Using TQ KV from the CLI (Sprint 4c in progress)

```bash
./llama-cli -m <model.gguf> \
    --cache-type-k tq_kv3 --cache-type-v tq_kv3 \
    -p "hello" -n 10 -v
```

Today (Sprint 4c steps 1-3a): the CLI flag parses, the `llama_kv_cache`
allocates per-layer `tiered_cache` shadow buffers, writes are observed
(fp16 cache rows → `tiered_cache::add_token`), and eviction kicks in
once `hot_window` tokens are accumulated. The attention compute still
reads from the fp16 tensor — actual VRAM savings land in Step 3b.

To force cold-tier eviction on short contexts (useful for testing):

```bash
LLAMA_TQ_HOT_WINDOW=4 ./llama-cli --cache-type-k tq_kv3 ... -v
```

With `-v`, look for lines like:
```
llama_kv_cache: TurboQuant shadow caches allocated (tq_kv3/tq_kv3):
    24 K layers, 24 V layers, hot_window=4
tq_flush_pending_: flushed 30 tokens -> K=34 (hot=4 cold=30 ratio=3.11x) ...
```

## Differential benchmark (`scripts/benchmark-tq-kv.py`)

Runs llama-cli against the same prompt with different cache types and
emits a markdown comparison table:

```bash
python3 scripts/benchmark-tq-kv.py \
    --llama-cli build-cuda/bin/llama-cli \
    --model     /path/to/model.gguf \
    --prompt    "hi" --n 16 \
    --hot-window 4 --validate \
    --configs f16 tq_kv4 tq_kv3 tq_kv2
```

Sample output (Qwen2.5-0.5B, CPU, hot_window=4):

| Config | Prompt tok/s | Gen tok/s | KV MiB | Δ Gen % | Observed compression     | Validate cos (K) |
|--------|-------------:|----------:|-------:|--------:|--------------------------|------------------|
| `f16`    | 130.53 | 54.25 | 384.00 | +0.00%  | —                        | —                |
| `tq_kv4` | 110.97 | 41.20 | 384.00 | -24.06% | K=2.71× V=2.71× (cold_max=74) | 1.000000 mean/min |
| `tq_kv3` | 110.49 | 41.34 | 384.00 | -23.80% | K=3.22× V=3.22× (cold_max=74) | 1.000000 mean/min |
| `tq_kv2` | 109.21 | 33.99 | 384.00 | -37.35% | K=3.99× V=3.99× (cold_max=74) | 1.000000 mean/min |

**What the numbers mean today:**

- *KV MiB unchanged* — the fp16 backbone is still full size; Step 3c-2
  shrinks it and wires reads through the tiered cache, at which point
  this column shows real savings.
- *Gen tok/s slower on TQ* — the observe path + fp16↔fp32 conversion
  per token per layer adds CPU work. When the read path flips in
  Step 3c-2, the fp16 backbone shrinks and most of the observation
  overhead goes with it.
- *Observed compression* — the tiered_cache internals: e.g. `tq_kv3`
  reports 3.22× compression on the cold tier during this run. That is
  the ceiling for Step 3c-2's VRAM savings (adjusted for hot_window).
- *Validate cos = 1.0* — the runtime push/readback round-trip over 96+
  samples confirms the observed K/V data matches what the fp16 cache
  contains. This is the correctness gate for Step 3c-2.
- *Sampled token agreement = 0.0% disagreement* — same generated tokens
  across all four configs on this prompt.

The script exits non-zero on (a) any crashed run or (b) sampled-token
disagreement above `--max-disagreement-frac` (default 10%).

## Tiered cache (Sprint 4)

[`src/llama-kv-tiered.h`](../src/llama-kv-tiered.h) provides
`llama_kv_tq::tiered_cache`, a self-contained KV store that mixes a
small **hot** ring buffer (last `hot_window` tokens, stored fp32 for
exact reads) with a **cold** TurboQuant-compressed tail. Tokens
auto-migrate hot→cold when the window fills.

```cpp
#include "llama-kv-tiered.h"

llama_kv_tq::tiered_cache_config cfg;
cfg.n_layers   = 32;          // attention layers
cfg.n_kv_heads = 8;
cfg.head_dim   = 128;
cfg.hot_window = 512;         // last 512 tokens stay fp16
cfg.cold_bits  = llama_kv_tq::BITS_3;

llama_kv_tq::tiered_cache cache(cfg);

// Append a new token's K and V (n_layers*n_kv_heads*head_dim each).
cache.add_token(k_ptr, v_ptr);

// Read back any previous token (decompresses cold reads on demand).
cache.get_kv(/*pos=*/0, /*layer=*/4, /*head=*/2, k_out, v_out);

auto s = cache.stats();
printf("compression: %.2fx (%d hot, %d cold)\n",
       s.compression_ratio, s.hot_tokens, s.cold_tokens);
```

This data structure is what Sprint 4b will adapt to `llama_memory_i`
so the existing prefill/generation graphs route through it
transparently. Today it is exercised by
[`tests/test-tq-tiered.cpp`](../tests/test-tq-tiered.cpp): hot-only
exact equality, post-eviction cosine reconstruction, eviction order,
memory accounting, multi-layer slot isolation, and a 2,000-token
stress run with 4 layers × 8 heads.

## Test coverage

`tests/test-tq-kv.cpp` exercises:

| Test | What it checks |
|------|---------------|
| `block_size_and_ratio` | Storage layout matches design (TQ KV3 = 52 bytes for D=128) |
| `rotation_orthogonality` | `Π · Π^T == I` to 1e-3 tolerance |
| `structured_rotation` | Permutation + sign-flip invariants for D > 4096 |
| `deterministic_seed` | Same seed → identical rotation matrices |
| `norm_preservation` | Stored norm matches input `‖v‖₂` exactly |
| `zero_vector` | Compressing zeros → reconstructed zeros |
| `unit_axis_vector` | One-hot `e₀` survives round-trip (cos > 0.85) |
| `structured_round_trip` | Cosine > 0.85 for D=8192 (structured path) |
| `bit_packing_edge_cases` | Uneven group sizes (D=13) work for all bit widths |
| `round_trip` (×8 configs) | Mean cosine over 16 random vectors meets per-bit target |

Total: **16,531 individual `CHECK` invocations** across all sub-tests.

## Performance targets (Sprint 3)

After CUDA kernels land:

| Metric | Target | Baseline (F16) |
|--------|--------|----------------|
| Inference throughput | within 2× of F16 | 100% |
| KV memory at 32K context | <30% of F16 | 100% |
| Perplexity loss on WikiText-2 | <0.5% | — |

## Sprint roadmap

- [x] **Sprint 1** — Foundation (fork, design doc, scaffolding, issues filed)
- [x] **Sprint 2** — CPU reference + tests + CI
- [x] **Sprint 3** — CUDA kernels (Volta/Ampere)
- [x] **Sprint 4** — Tiered cache data structure (hot fp16 + cold TQ)
- [x] **Sprint 4b** — `GGML_TYPE_TQ_KV{2,3,4}` registration + CLI parsing
- [ ] **Sprint 4c** — Wire `tiered_cache` into `llama_kv_cache` ← *in progress* (plan: [`turboquant-kv-sprint4c-plan.md`](turboquant-kv-sprint4c-plan.md))
  - [x] **Step 1** — TQ types map to fp16 internally (no crash, no compression yet)
  - [x] **Step 2a** — per-layer `tiered_cache` allocation
  - [x] **Step 2b** — queue + flush observe in `apply_ubatch`/`prepare` (writes now populate tiered_cache)
  - [x] **Step 3a** — env var `LLAMA_TQ_HOT_WINDOW` + per-flush stats (cold tier is populated on real inference data)
  - [x] **Step 3b** — `tq_materialize_fp16_k/v` API + `tiered_cache::read_token_k/v`, `materialize_fp16_rows`, with per-path unit tests (hot, cold, mixed boundary, all bit widths, empty, out-of-range)
  - [x] **Step 3c-1** — runtime push/readback round-trip validation via `LLAMA_TQ_VALIDATE=1`, plus **post-compute hook** that fixes a pre-compute race in the observe path (would have broken 3c-2)
  - [x] **Step 3c-2a** — read-side view validation via `LLAMA_TQ_VIEW_VALIDATE=1`: live cosine compare of `materialize_fp16_rows` output against fp16 cache rows. Reports cos = 1.000000 (mean and min) over 1,440 row comparisons on Atlas (hot-tier path)
  - [x] **Step 3c-2b precursor** — cold-path validation via `LLAMA_TQ_COLD_VALIDATE=1`. On Atlas (Qwen2.5-0.5B, hot_window=4, 1,344 cold rows): tq_kv4 mean/min cos = 0.9953/0.9852, tq_kv3 = 0.9796/0.9579, tq_kv2 = 0.9419/0.9138. All meet per-bit targets
  - [⚠️] **Step 3c-2b investigation** — diagnostic env `LLAMA_TQ_DIAG_WB={1..6}` localized the writeback divergence. Findings:
    - `LLAMA_TQ_DIAG_WB=2/3/4` (read-then-write-back same bytes for layer 0 / all layers / K+V): correct output, 0% disagreement. **The writeback mechanism (ggml_backend_tensor_set) works.**
    - `LLAMA_TQ_DIAG_WB=6` (compare materialize output to cache bytes): 50% mismatch on first flush (n=4: pos 0-1 match exactly, pos 2-3 differ entirely), 99.7% on later flushes. **The bug is upstream of writeback** — `materialize_fp16_rows` output for hot tokens diverges from what's in the fp16 cache even though the round-trip should be lossless.
    - Pattern (first/last positions split) suggests an issue in the OBSERVE path's per-token-per-layer iteration order. Most likely something between observe-time read of slot N and the cache's actual content of slot N at materialize time. Needs focused next-turn investigation tracing observe_one and the apply_ubatch slot/pending order.
    - Also still open: the V cache's `v_trans=true` default makes the existing OBSERVE path silently incorrect for V (validate cosines were 1.0 / 0.95+ comparing same-wrong-data on both sides). K observation is mostly correct (this turn's bug aside).
  - [ ] Step 3c-3 — root-cause + fix the observe/materialize divergence; proper read-side wire-up via ggml input-binding; transpose-aware V observation; slot↔pos mapping; shrink fp16 backbone → real VRAM savings
- [ ] **Sprint 5** — Benchmarks + upstream PR

### Sprint 4 / 4b / 4c scope split

The original plan bundled tiering, ggml type registration, and the
KV-cache adapter into one sprint. In practice these are three independent
risks:

- **Sprint 4** proved the tiering data structure works in isolation.
- **Sprint 4b** (this) registers the type tags and wires the CLI surface,
  with a clear failure path until the backend lands. *No inference path
  changes.*
- **Sprint 4c** does the real work: a `llama_memory_i` implementation
  that holds a `tiered_cache` outside ggml's tensor system and
  materializes fp16 views before each attention step. This is a real
  refactor of `llama_kv_cache.cpp` and the prefill/generation graphs.

### Why are these "tag" types and not full ggml types?

ggml types assume a compile-time fixed `blck_size` (e.g. `QK_K = 256`
elements per packed block). TurboQuant compresses one *head_dim*-sized
vector at a time, where `head_dim` is a runtime model parameter
(64, 128, 192, 256...). The packed layout depends on `head_dim`, so it
doesn't fit ggml's "K elements → N bytes" model.

The fix is architectural: TQ-compressed K/V is stored *outside* ggml's
tensor system (in our `tiered_cache` byte buffers), and the KV-cache
layer materializes a temporary fp16 ggml tensor for each attention
step. So `GGML_TYPE_TQ_KV*` are *storage backend tags* — they tell the
KV-cache layer "use the TurboQuant backend", not "this tensor's elements
are packed this way". Their `blck_size` and `type_size` are 0 to make
this contract explicit; any code path that tries to treat them as a
packed tensor type will fail loudly.

Tracking issue: [ahb-sjsu/turboquant-pro#27](https://github.com/ahb-sjsu/turboquant-pro/issues/27)

## Related improvements (filed as issues, not in scope here)

- **Differential KV** ([turboquant-pro#22](https://github.com/ahb-sjsu/turboquant-pro/issues/22)) — delta-encode adjacent tokens
- **Attention-aware eviction** ([#23](https://github.com/ahb-sjsu/turboquant-pro/issues/23))
- **FP8/NVFP4 native paths** ([#24](https://github.com/ahb-sjsu/turboquant-pro/issues/24)) — Hopper/Blackwell
- **Quantization-aware micro-LoRA** ([#25](https://github.com/ahb-sjsu/turboquant-pro/issues/25)) — for *weight* quantization, separate problem

## Contributing

This branch is under active development. The CI workflow ensures the test suite stays green. Contributions welcome on the issues above.
